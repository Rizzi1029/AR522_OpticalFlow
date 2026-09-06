"""
FlowNetS Smoke Test and Demonstration Script.

Evaluates pretrained FlowNetS on the same synthetic moving patch pair
used in the Day 1 RAFT baseline smoke test.
"""

import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import flow_to_image

from src.flow.flownet_s import load_flownet_s


def generate_synthetic_pair(
    height: int = 256,
    width: int = 256,
    obj_x: int = 80,
    obj_y: int = 80,
    obj_w: int = 64,
    obj_h: int = 64,
    shift_x: int = 8,
    shift_y: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two synthetic RGB frames with a textured (checkerboard) moving patch.
    Matches the Day 1 RAFT test pair generation exactly.
    
    Returns:
        uint8 tensors of shape (3, H, W) in [0, 255].
    """
    # Background: low-contrast dark pattern
    frame1 = np.full((height, width, 3), 32, dtype=np.uint8)
    frame2 = np.full((height, width, 3), 32, dtype=np.uint8)

    # High-contrast checkerboard texture
    grid_size = 8
    y_coords, x_coords = np.indices((obj_h, obj_w))
    checker = ((y_coords // grid_size) + (x_coords // grid_size)) % 2 == 0

    patch = np.zeros((obj_h, obj_w, 3), dtype=np.uint8)
    patch[checker] = [230, 230, 230]
    patch[~checker] = [20, 120, 220]

    # Place object in Frame 1
    frame1[obj_y : obj_y + obj_h, obj_x : obj_x + obj_w] = patch

    # Place object shifted in Frame 2
    frame2[
        obj_y + shift_y : obj_y + shift_y + obj_h,
        obj_x + shift_x : obj_x + shift_x + obj_w,
    ] = patch

    # Convert to (3, H, W) torch tensors
    t_frame1 = torch.from_numpy(frame1).permute(2, 0, 1).contiguous()
    t_frame2 = torch.from_numpy(frame2).permute(2, 0, 1).contiguous()

    return t_frame1, t_frame2


def preprocess_flownet_pair(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """
    Preprocess image pair according to canonical FlowNet conventions:
    1. Scale uint8 [0, 255] to float [0.0, 1.0].
    2. Subtract FlyingChairs training RGB dataset mean [0.411, 0.432, 0.45].
    3. Concatenate into a 6-channel tensor (1, 6, H, W).
    4. Pad height and width to multiples of 64 if necessary.
    
    Returns:
        input_tensor: Tensor of shape (1, 6, H_padded, W_padded).
        orig_shape: (H, W).
        pad_amounts: (pad_h, pad_w).
    """
    _, h, w = frame1.shape

    # Convert to float in [0.0, 1.0] on target device
    f1 = frame1.to(device=device, dtype=torch.float32) / 255.0
    f2 = frame2.to(device=device, dtype=torch.float32) / 255.0

    # Subtract FlyingChairs RGB mean
    rgb_mean = torch.tensor([0.411, 0.432, 0.45], device=device, dtype=torch.float32).view(3, 1, 1)
    f1 = f1 - rgb_mean
    f2 = f2 - rgb_mean

    # Concatenate along channel dimension and add batch dimension -> (1, 6, H, W)
    pair = torch.cat([f1, f2], dim=0).unsqueeze(0)

    # Pad to multiples of 64
    pad_h = (64 - (h % 64)) % 64
    pad_w = (64 - (w % 64)) % 64

    if pad_h > 0 or pad_w > 0:
        pair = F.pad(pair, (0, pad_w, 0, pad_h), mode="replicate")

    return pair, (h, w), (pad_h, pad_w)


def run_flownet_smoke_test(
    checkpoint_path: str = "checkpoints/flownets_EPE1.951.pth.tar",
    output_path: str = "outputs/flownet_flow_visualization.png",
):
    # 1. Device verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    # 2. Synthesize input pair with known ground-truth shift
    obj_x, obj_y, obj_w, obj_h = 80, 80, 64, 64
    gt_shift_x, gt_shift_y = 8, 0

    raw_frame1, raw_frame2 = generate_synthetic_pair(
        height=256,
        width=256,
        obj_x=obj_x,
        obj_y=obj_y,
        obj_w=obj_w,
        obj_h=obj_h,
        shift_x=gt_shift_x,
        shift_y=gt_shift_y,
    )

    # 3. Canonical preprocessing & padding
    input_tensor, (orig_h, orig_w), (pad_h, pad_w) = preprocess_flownet_pair(
        raw_frame1, raw_frame2, device=device
    )

    print(f"Input Tensor Shape: {list(input_tensor.shape)}")
    print(f"Input Dtype: {input_tensor.dtype}")
    print(f"Input Value Range: [{input_tensor.min().item():.2f}, {input_tensor.max().item():.2f}]")
    print(f"Spatial Padding: pad_h={pad_h}, pad_w={pad_w} (divisible by 64: {input_tensor.shape[2] % 64 == 0 and input_tensor.shape[3] % 64 == 0})")

    # 4. Load FlowNetS model
    print(f"\nLoading FlowNetS model from {checkpoint_path}...")
    model = load_flownet_s(checkpoint_path=checkpoint_path, device=device)

    # 5. Run inference with GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        torch.cuda.synchronize()
        start_event.record()
        raw_flow2 = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()

    gpu_inference_time_ms = start_event.elapsed_time(end_event)

    # 6. Post-processing: upsample to full resolution, unpad, and apply flow scale
    # FlowNetS predicts 1/4 resolution flow (flow2) scaled down by div_flow=20.0
    DIV_FLOW = 20.0
    upsampled_flow = F.interpolate(
        raw_flow2,
        size=(orig_h + pad_h, orig_w + pad_w),
        mode="bilinear",
        align_corners=False,
    )

    if pad_h > 0 or pad_w > 0:
        upsampled_flow = upsampled_flow[:, :, :orig_h, :orig_w]

    final_flow = upsampled_flow * DIV_FLOW  # Shape: (1, 2, H, W)

    print(f"Raw Flow2 Output Shape: {list(raw_flow2.shape)}")
    print(f"Final Flow Output Shape: {list(final_flow.shape)}")
    print(f"Flow Dtype: {final_flow.dtype}")
    print(f"GPU Inference Time: {gpu_inference_time_ms:.2f} ms")

    # 7. Compute mean flow within moving object ROI
    roi_flow = final_flow[0, :, obj_y : obj_y + obj_h, obj_x : obj_x + obj_w]
    mean_u = roi_flow[0].mean().item()
    mean_v = roi_flow[1].mean().item()

    print(f"\nGround-Truth Shift: dx = {gt_shift_x} px, dy = {gt_shift_y} px")
    print(f"Mean Horizontal Flow (u) in ROI: {mean_u:.3f} px")
    print(f"Mean Vertical Flow (v) in ROI:   {mean_v:.3f} px")

    # 8. Save flow visualization using torchvision's flow_to_image (identical to RAFT)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flow_img_tensor = flow_to_image(final_flow[0])  # (3, H, W) uint8 in RGB
    flow_img_bgr = flow_img_tensor.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    cv2.imwrite(output_path, flow_img_bgr)
    print(f"\nFlow visualization saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    ckpt = str(project_root / "checkpoints" / "flownets_EPE1.951.pth.tar")
    out_vis = str(project_root / "outputs" / "flownet_flow_visualization.png")
    run_flownet_smoke_test(checkpoint_path=ckpt, output_path=out_vis)
