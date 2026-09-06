import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image


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
    Returns uint8 tensors of shape (3, H, W) in [0, 255].
    """
    # Background: low-contrast dark pattern
    frame1 = np.full((height, width, 3), 32, dtype=np.uint8)
    frame2 = np.full((height, width, 3), 32, dtype=np.uint8)

    # Generate a high-contrast checkerboard texture for the moving object
    grid_size = 8
    y_coords, x_coords = np.indices((obj_h, obj_w))
    checker = ((y_coords // grid_size) + (x_coords // grid_size)) % 2 == 0

    patch = np.zeros((obj_h, obj_w, 3), dtype=np.uint8)
    # Vibrant high-contrast texture to avoid aperture problem
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


def run_raft_smoke_test(output_path: str = "outputs/raft_flow_visualization.png"):
    # 1. Verification of execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

    # 3. Load model and apply official weights transforms
    weights = Raft_Large_Weights.C_T_SKHT_V2
    transforms = weights.transforms()

    # Preprocessing maps [0, 255] uint8 -> [-1.0, 1.0] float32
    img1, img2 = transforms(raw_frame1, raw_frame2)

    # Add batch dimension and transfer to device
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    # 4. Print input properties
    print(f"Input Tensor Shape 1: {list(img1.shape)}")
    print(f"Input Tensor Shape 2: {list(img2.shape)}")
    print(f"Input Dtype: {img1.dtype}")
    print(f"Input Value Range: [{img1.min().item():.2f}, {img1.max().item():.2f}]")

    # 5. Initialize RAFT model in evaluation mode
    print("\nLoading RAFT model (raft_large with C_T_SKHT_V2)...")
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()

    # 6. Run inference with GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        # GPU execution timing
        torch.cuda.synchronize()
        start_event.record()
        flow_predictions = model(img1, img2)
        end_event.record()
        torch.cuda.synchronize()

    gpu_inference_time_ms = start_event.elapsed_time(end_event)

    # 7. Extract final flow prediction
    final_flow = flow_predictions[-1]  # shape: (1, 2, H, W)

    print(f"Flow Output Shape: {list(final_flow.shape)}")
    print(f"Flow Dtype: {final_flow.dtype}")
    print(f"GPU Inference Time: {gpu_inference_time_ms:.2f} ms")

    # 8. Compute mean flow within moving object ROI
    roi_flow = final_flow[0, :, obj_y : obj_y + obj_h, obj_x : obj_x + obj_w]
    mean_u = roi_flow[0].mean().item()
    mean_v = roi_flow[1].mean().item()

    print(f"\nGround-Truth Shift: dx = {gt_shift_x} px, dy = {gt_shift_y} px")
    print(f"Mean Horizontal Flow (u) in ROI: {mean_u:.3f} px")
    print(f"Mean Vertical Flow (v) in ROI:   {mean_v:.3f} px")

    # 9. Save flow visualization using torchvision's flow_to_image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flow_img_tensor = flow_to_image(final_flow[0])  # (3, H, W) uint8 in RGB
    flow_img_bgr = flow_img_tensor.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    cv2.imwrite(output_path, flow_img_bgr)
    print(f"\nFlow visualization saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    save_path = str(project_root / "outputs" / "raft_flow_visualization.png")
    run_raft_smoke_test(output_path=save_path)
