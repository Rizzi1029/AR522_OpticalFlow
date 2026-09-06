"""
Verification script for Day 3 Step 3a: Sintel Dataset and EPE Metrics.

Tests:
1. Synthetic tests for compute_epe and aggregate_epe where expected values are known analytically.
2. Direct file loading of real Sintel sample:
   - training/clean/alley_1/frame_0001.png
   - training/clean/alley_1/frame_0002.png
   - training/flow/alley_1/frame_0001.flo
   - training/invalid/alley_1/frame_0001.png
3. Verification of shapes, dtypes, spatial consistency, and valid pixel counts.
4. Dataset integration and self-consistency EPE checks.
"""

import sys
from pathlib import Path
import math
import tempfile
import torch

# Ensure project root is on sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import (
    SintelDataset,
    read_flo,
    read_image,
    read_invalid_mask,
)
from src.benchmark.metrics import (
    compute_epe,
    aggregate_epe,
)


def test_synthetic_metrics():
    print("--- Test 1: Synthetic EPE & Aggregation Metrics ---")

    # 1. 3D single-pixel test: displacement vector (3, 4) -> EPE = 5.0
    pred_3d = torch.tensor([[[3.0]], [[4.0]]], dtype=torch.float32)  # [2, 1, 1]
    gt_3d = torch.tensor([[[0.0]], [[0.0]]], dtype=torch.float32)    # [2, 1, 1]
    mask_3d = torch.tensor([[True]], dtype=torch.bool)               # [1, 1]

    mean_epe, count, epe_sum = compute_epe(pred_3d, gt_3d, mask_3d)
    assert math.isclose(mean_epe, 5.0, rel_tol=1e-6), f"Expected 5.0, got {mean_epe}"
    assert count == 1, f"Expected count 1, got {count}"
    assert math.isclose(epe_sum, 5.0, rel_tol=1e-6), f"Expected sum 5.0, got {epe_sum}"
    print("  [PASS] 3D single-pixel known vector (3, 4) -> EPE = 5.0")

    # 2. Multi-pixel 3D test with masked invalid pixel
    # Pixel (0,0): diff=(3, 4)   -> EPE=5.0   (valid=True)
    # Pixel (0,1): diff=(6, 8)   -> EPE=10.0  (valid=True)
    # Pixel (1,0): diff=(10, 0)  -> EPE=10.0  (valid=False, masked out!)
    # Pixel (1,1): diff=(0, 0)   -> EPE=0.0   (valid=True)
    pred_grid = torch.tensor([
        [[3.0, 6.0], [10.0, 0.0]],
        [[4.0, 8.0], [0.0, 0.0]],
    ], dtype=torch.float32)  # [2, 2, 2]

    gt_grid = torch.zeros((2, 2, 2), dtype=torch.float32)
    mask_grid = torch.tensor([[True, True], [False, True]], dtype=torch.bool)

    mean_epe, count, epe_sum = compute_epe(pred_grid, gt_grid, mask_grid)
    expected_sum = 5.0 + 10.0 + 0.0  # 15.0
    expected_mean = 15.0 / 3.0       # 5.0
    assert count == 3, f"Expected count 3, got {count}"
    assert math.isclose(epe_sum, expected_sum, rel_tol=1e-6), f"Expected {expected_sum}, got {epe_sum}"
    assert math.isclose(mean_epe, expected_mean, rel_tol=1e-6), f"Expected {expected_mean}, got {mean_epe}"
    print("  [PASS] Multi-pixel 3D masked EPE calculation matches exact expected values")

    # 3. Batched 4D test [B, 2, H, W]
    pred_4d = pred_grid.unsqueeze(0)  # [1, 2, 2, 2]
    gt_4d = gt_grid.unsqueeze(0)      # [1, 2, 2, 2]
    mean_epe_4d, count_4d, sum_4d = compute_epe(pred_4d, gt_4d, mask_grid)
    assert count_4d == count and math.isclose(mean_epe_4d, mean_epe, rel_tol=1e-6)
    print("  [PASS] Batched 4D tensor with 2D mask broadcast matches 3D computation")

    # 4. Empty mask edge case
    empty_mask = torch.zeros((2, 2), dtype=torch.bool)
    mean_empty, count_empty, sum_empty = compute_epe(pred_grid, gt_grid, empty_mask)
    assert count_empty == 0 and mean_empty == 0.0 and sum_empty == 0.0
    print("  [PASS] Zero valid pixels handled gracefully (count=0, mean=0.0, sum=0.0)")

    # 5. Aggregation helper: micro vs macro
    records = [
        {"mean_epe": 2.0, "valid_count": 100, "epe_sum": 200.0},
        {"mean_epe": 8.0, "valid_count": 300, "epe_sum": 2400.0},
    ]
    # Macro EPE: (2.0 + 8.0) / 2 = 5.0
    # Micro EPE: (200 + 2400) / (100 + 300) = 2600 / 400 = 6.5
    agg = aggregate_epe(records)
    assert math.isclose(agg["macro_epe"], 5.0, rel_tol=1e-6), f"Expected macro 5.0, got {agg['macro_epe']}"
    assert math.isclose(agg["micro_epe"], 6.5, rel_tol=1e-6), f"Expected micro 6.5, got {agg['micro_epe']}"
    assert agg["total_valid_pixels"] == 400
    assert agg["num_frames"] == 2
    print("  [PASS] aggregate_epe correctly separates micro (6.5) and macro (5.0) statistics")


def test_real_sintel_sample(project_root: Path):
    print("\n--- Test 2: Real Sintel Sample File I/O Verification ---")
    training_dir = project_root / "data" / "Sintel" / "training"
    f1_path = training_dir / "clean" / "alley_1" / "frame_0001.png"
    f2_path = training_dir / "clean" / "alley_1" / "frame_0002.png"
    flo_path = training_dir / "flow" / "alley_1" / "frame_0001.flo"
    inv_path = training_dir / "invalid" / "alley_1" / "frame_0001.png"

    print(f"  Frame 1:  {f1_path.relative_to(project_root)}")
    print(f"  Frame 2:  {f2_path.relative_to(project_root)}")
    print(f"  Flow GT:  {flo_path.relative_to(project_root)}")
    print(f"  Invalid:  {inv_path.relative_to(project_root)}")

    # 1. Read individual files
    frame1 = read_image(f1_path)
    frame2 = read_image(f2_path)
    h, w = frame1.shape[1], frame1.shape[2]
    flow_gt = read_flo(flo_path, expected_shape=(h, w))
    invalid_mask = read_invalid_mask(inv_path, expected_shape=(h, w))

    # 2. Shape checks
    assert frame1.shape == torch.Size([3, 436, 1024]), f"Unexpected frame1 shape: {frame1.shape}"
    assert frame2.shape == torch.Size([3, 436, 1024]), f"Unexpected frame2 shape: {frame2.shape}"
    assert flow_gt.shape == torch.Size([2, 436, 1024]), f"Unexpected flow shape: {flow_gt.shape}"
    assert invalid_mask.shape == torch.Size([436, 1024]), f"Unexpected mask shape: {invalid_mask.shape}"
    print(f"  [PASS] Shapes verified: frame1={list(frame1.shape)}, flow={list(flow_gt.shape)}, mask={list(invalid_mask.shape)}")

    # 3. Dtype checks
    assert frame1.dtype == torch.uint8, f"frame1 dtype expected torch.uint8, got {frame1.dtype}"
    assert frame2.dtype == torch.uint8, f"frame2 dtype expected torch.uint8, got {frame2.dtype}"
    assert flow_gt.dtype == torch.float32, f"flow_gt dtype expected torch.float32, got {flow_gt.dtype}"
    print(f"  [PASS] Dtypes verified: frames={frame1.dtype}, flow={flow_gt.dtype}")

    # 4. Dimension matching
    assert (frame1.shape[1], frame1.shape[2]) == (flow_gt.shape[1], flow_gt.shape[2]), "Dimensions do not match!"
    print(f"  [PASS] Image spatial dimensions ({h}x{w}) strictly match flow dimensions ({flow_gt.shape[1]}x{flow_gt.shape[2]})")

    # 5. Validity check
    valid_mask = (invalid_mask == 0) & torch.isfinite(flow_gt[0]) & torch.isfinite(flow_gt[1])
    assert valid_mask.dtype == torch.bool, f"valid_mask dtype expected torch.bool, got {valid_mask.dtype}"
    valid_count = int(valid_mask.sum().item())
    total_pixels = h * w
    print(f"  [PASS] Valid pixels count: {valid_count} / {total_pixels} ({valid_count / total_pixels * 100:.2f}%)")
    assert valid_count > 0, "No valid pixels found!"


def test_malformed_flo_handling():
    print("\n--- Test 3: Malformed .flo Error Handling ---")
    with tempfile.NamedTemporaryFile(suffix=".flo") as tmp:
        tmp.write(b"BADM\x00\x00\x00\x00")
        tmp.flush()
        try:
            read_flo(tmp.name)
            raise AssertionError("Failed to reject invalid magic header!")
        except ValueError as e:
            assert "invalid magic header" in str(e).lower()
            print("  [PASS] Correctly rejected malformed magic tag with clear ValueError")


def test_dataset_loader(project_root: Path):
    print("\n--- Test 4: SintelDataset Integration Test ---")
    dataset = SintelDataset(
        root=project_root / "data" / "Sintel",
        split="training",
        pass_name="clean",
        scenes=["alley_1"],
    )

    print(f"  Loaded alley_1 scene: {len(dataset)} frame pairs")
    assert len(dataset) == 49, f"Expected 49 pairs for alley_1, got {len(dataset)}"

    f1, f2, flow_gt, valid_mask, meta = dataset[0]
    assert f1.shape == torch.Size([3, 436, 1024])
    assert f2.shape == torch.Size([3, 436, 1024])
    assert flow_gt.shape == torch.Size([2, 436, 1024])
    assert valid_mask.shape == torch.Size([436, 1024])
    assert meta["scene"] == "alley_1"
    assert meta["frame_idx"] == 1
    assert meta["pass_name"] == "clean"

    # Self-EPE check: flow_gt compared with itself must give exactly 0.0 EPE
    self_mean_epe, self_count, self_sum = compute_epe(flow_gt, flow_gt, valid_mask)
    assert math.isclose(self_mean_epe, 0.0, abs_tol=1e-7), f"Self EPE was {self_mean_epe}, expected 0.0"
    assert self_count == int(valid_mask.sum().item())
    print(f"  [PASS] Self-EPE evaluation on ground truth: {self_mean_epe:.6f} px")

    # Shifted ground-truth test: add (+3 px, +4 px) offset everywhere
    offset_flow = flow_gt.clone()
    offset_flow[0] += 3.0
    offset_flow[1] += 4.0
    offset_mean_epe, offset_count, offset_sum = compute_epe(offset_flow, flow_gt, valid_mask)
    assert math.isclose(offset_mean_epe, 5.0, rel_tol=1e-5), f"Expected 5.0 EPE, got {offset_mean_epe}"
    print(f"  [PASS] Known (+3, +4) pixel perturbation on real flow gives exactly {offset_mean_epe:.4f} px EPE")


def main():
    project_root = Path(__file__).resolve().parent.parent
    print(f"Running Day 3 Step 3a verification from: {project_root}\n")

    test_synthetic_metrics()
    test_real_sintel_sample(project_root)
    test_malformed_flo_handling()
    test_dataset_loader(project_root)

    print("\n==============================================")
    print(" ALL STEP 3a VERIFICATION CHECKS PASSED!")
    print("==============================================")


if __name__ == "__main__":
    main()
