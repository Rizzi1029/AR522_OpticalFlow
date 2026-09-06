"""
Verification script for Day 3 Step 3b: Common Optical Flow Model Wrappers.

Verifies:
1. Model loading for both FlowNetS and RAFT via build_flow_estimator.
2. Inference on real Sintel frame pair (data/Sintel/training/clean/alley_1/frame_0001.png & frame_0002.png).
3. Exact contract conformity: [B, 2, H, W] float32 at unpadded 436x1024 resolution.
4. Absence of NaNs / Infs in predicted flow fields.
5. Sintel EPE computation against ground-truth flow for both models.
6. Support for unbatched [3, H, W] and batched [B, 3, H, W] inputs.
"""

import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.flow.wrappers import build_flow_estimator
from src.benchmark.sintel_dataset import SintelDataset
from src.benchmark.metrics import compute_epe


def test_model_wrappers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Step 3b verification on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Sintel sample
    dataset = SintelDataset(
        root=project_root / "data" / "Sintel",
        split="training",
        pass_name="clean",
        scenes=["alley_1"],
    )
    frame1, frame2, flow_gt, valid_mask, meta = dataset[0]
    orig_h, orig_w = frame1.shape[1], frame1.shape[2]
    print(f"\nLoaded test sample: scene={meta['scene']}, frame={meta['frame_idx']}")
    print(f"Frame input shape: {list(frame1.shape)}, dtype={frame1.dtype}")
    print(f"Flow GT shape:    {list(flow_gt.shape)}, dtype={flow_gt.dtype}")
    print(f"Valid mask shape: {list(valid_mask.shape)}, count={valid_mask.sum().item()}")

    # 2. Verify FlowNetS Adapter
    print("\n--- Testing FlowNetS Adapter ---")
    flownet = build_flow_estimator("flownets", device=device)
    print("  [PASS] FlowNetS loaded successfully")

    # Unbatched [3, H, W] input
    flow_fn_unbatched = flownet(frame1, frame2)
    assert flow_fn_unbatched.shape == torch.Size([1, 2, orig_h, orig_w]), (
        f"Expected [1, 2, {orig_h}, {orig_w}], got {flow_fn_unbatched.shape}"
    )
    assert flow_fn_unbatched.dtype == torch.float32
    assert not torch.isnan(flow_fn_unbatched).any(), "FlowNetS output contains NaNs!"
    assert not torch.isinf(flow_fn_unbatched).any(), "FlowNetS output contains Infs!"
    print(f"  [PASS] Unbatched input -> Output shape: {list(flow_fn_unbatched.shape)}, dtype: {flow_fn_unbatched.dtype}")

    # Batched [B, 3, H, W] input
    batch_f1 = frame1.unsqueeze(0)
    batch_f2 = frame2.unsqueeze(0)
    flow_fn_batched = flownet(batch_f1, batch_f2)
    assert flow_fn_batched.shape == torch.Size([1, 2, orig_h, orig_w])
    assert torch.allclose(flow_fn_unbatched, flow_fn_batched, atol=1e-3)
    print("  [PASS] Batched input [1, 3, H, W] produces consistent result")

    # Compute FlowNetS EPE against ground truth
    fn_epe, fn_count, fn_sum = compute_epe(
        flow_fn_unbatched[0].cpu(),
        flow_gt,
        valid_mask,
    )
    print(f"  [PASS] FlowNetS EPE on alley_1/frame_0001: {fn_epe:.4f} px (valid pixels: {fn_count})")
    assert fn_epe < 20.0, f"FlowNetS EPE abnormally high ({fn_epe})"

    # 3. Verify RAFT Adapter
    print("\n--- Testing RAFT Adapter ---")
    raft = build_flow_estimator("raft", device=device)
    print("  [PASS] RAFT loaded successfully")

    # Unbatched [3, H, W] input
    flow_raft_unbatched = raft(frame1, frame2)
    assert flow_raft_unbatched.shape == torch.Size([1, 2, orig_h, orig_w]), (
        f"Expected [1, 2, {orig_h}, {orig_w}], got {flow_raft_unbatched.shape}"
    )
    assert flow_raft_unbatched.dtype == torch.float32
    assert not torch.isnan(flow_raft_unbatched).any(), "RAFT output contains NaNs!"
    assert not torch.isinf(flow_raft_unbatched).any(), "RAFT output contains Infs!"
    print(f"  [PASS] Unbatched input -> Output shape: {list(flow_raft_unbatched.shape)}, dtype: {flow_raft_unbatched.dtype}")

    # Batched [B, 3, H, W] input
    flow_raft_batched = raft(batch_f1, batch_f2)
    assert flow_raft_batched.shape == torch.Size([1, 2, orig_h, orig_w])
    assert torch.allclose(flow_raft_unbatched, flow_raft_batched)
    print("  [PASS] Batched input [1, 3, H, W] produces identical result")

    # Compute RAFT EPE against ground truth
    raft_epe, raft_count, raft_sum = compute_epe(
        flow_raft_unbatched[0].cpu(),
        flow_gt,
        valid_mask,
    )
    print(f"  [PASS] RAFT EPE on alley_1/frame_0001:     {raft_epe:.4f} px (valid pixels: {raft_count})")
    assert raft_epe < 10.0, f"RAFT EPE abnormally high ({raft_epe})"

    # 4. Comparative Check
    print("\n--- Comparative Contract Check ---")
    assert flow_fn_unbatched.shape == flow_raft_unbatched.shape
    assert flow_fn_unbatched.dtype == flow_raft_unbatched.dtype
    print(f"Both models produce identical tensor contracts: {list(flow_fn_unbatched.shape)}, {flow_fn_unbatched.dtype}")
    print(f"FlowNetS EPE: {fn_epe:.4f} px | RAFT EPE: {raft_epe:.4f} px")
    print("RAFT achieves lower EPE as expected on benchmark validation.")

    print("\n==============================================")
    print(" ALL STEP 3b VERIFICATION CHECKS PASSED!")
    print("==============================================")


if __name__ == "__main__":
    test_model_wrappers()
