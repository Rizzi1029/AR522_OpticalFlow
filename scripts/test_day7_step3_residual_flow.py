"""
Verification Test Suite for Day 7 Step 3: Residual Optical Flow.

Validates:
1. Exact zero residual when observed flow == rigid flow.
2. Exact analytical residual for synthetic known difference (simulated independent motion).
3. Real consecutive Sintel frames (`alley_1` frames 1 & 2):
   - Loads ground-truth observed optical flow (.flo).
   - Computes camera-induced rigid flow via Step 2 with official .cam calibration.
   - Calculates residual flow: w_res = w_obs - w_rigid.
   - Validates finiteness, valid ratio, and computes full residual statistics (mean, median, p95, max, RMSE).
   - Saves a 6-panel composite visualization comparing RGB, observed flow, rigid flow,
     residual flow, residual magnitude heatmap, and validity mask.
"""

import sys
from pathlib import Path
import math
import cv2
import numpy as np
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import read_flo
from src.depth.intrinsics import CameraIntrinsics
from src.depth.midas import MiDaSDepthEstimator
from src.geometry.rigid_flow import compute_rigid_flow, flow_to_hsv_bgr
from src.geometry.residual_flow import (
    compute_residual_flow,
    compute_residual_magnitude,
    compute_residual_statistics,
)


def test_observed_equals_rigid_zero_residual() -> None:
    print("=== Test 1: Observed == Rigid -> Exact Zero Residual ===")
    height, width = 436, 1024

    # 1. Channels-last test [H, W, 2]
    torch.manual_seed(42)
    rigid_flow_last = torch.randn(height, width, 2, dtype=torch.float32)
    observed_flow_last = rigid_flow_last.clone()

    mask = torch.rand(height, width) > 0.1  # 90% valid pixels

    res_last, res_mask_last = compute_residual_flow(
        observed_flow=observed_flow_last,
        rigid_flow=rigid_flow_last,
        valid_mask=mask,
        zero_invalid=True,
    )

    assert res_last.shape == (height, width, 2)
    assert res_mask_last.shape == (height, width)
    assert (res_mask_last == mask).all().item()

    max_err_last = torch.max(torch.abs(res_last)).item()
    assert max_err_last == 0.0, f"Expected exact 0.0 residual, got {max_err_last}"
    print(f"  [PASS] Channels-last [H, W, 2]: Max residual error = {max_err_last:.2e} px")

    # 2. Channels-first test [2, H, W]
    rigid_flow_first = torch.randn(2, height, width, dtype=torch.float32)
    observed_flow_first = rigid_flow_first.clone()

    res_first, res_mask_first = compute_residual_flow(
        observed_flow=observed_flow_first,
        rigid_flow=rigid_flow_first,
        valid_mask=mask,
        zero_invalid=True,
    )

    assert res_first.shape == (2, height, width)
    max_err_first = torch.max(torch.abs(res_first)).item()
    assert max_err_first == 0.0, f"Expected exact 0.0 residual, got {max_err_first}"
    print(f"  [PASS] Channels-first [2, H, W]: Max residual error = {max_err_first:.2e} px")

    # 3. Cross-layout subtraction (observed [2, H, W] vs rigid [H, W, 2])
    res_cross, _ = compute_residual_flow(
        observed_flow=observed_flow_first,
        rigid_flow=rigid_flow_first.permute(1, 2, 0),
        valid_mask=mask,
        zero_invalid=True,
    )
    assert res_cross.shape == (2, height, width)
    max_err_cross = torch.max(torch.abs(res_cross)).item()
    assert max_err_cross == 0.0
    print(f"  [PASS] Cross-layout alignment: Max residual error = {max_err_cross:.2e} px")

    # 4. Batched 4D tensor verification [B=2, 2, H, W]
    obs_4d = observed_flow_first.unsqueeze(0).repeat(2, 1, 1, 1)
    rig_4d = rigid_flow_first.unsqueeze(0).repeat(2, 1, 1, 1)
    mask_4d = mask.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)

    res_4d, res_mask_4d = compute_residual_flow(
        observed_flow=obs_4d,
        rigid_flow=rig_4d,
        valid_mask=mask_4d,
        zero_invalid=True,
    )
    assert res_4d.shape == (2, 2, height, width)
    assert res_mask_4d.shape == (2, 1, height, width)
    assert torch.max(torch.abs(res_4d)).item() == 0.0
    print(f"  [PASS] Batched 4D tensor verification: Max residual error = 0.0 px")

    # 5. Summary statistics verification
    stats = compute_residual_statistics(res_last, res_mask_last)
    assert stats["mean_epe"] == 0.0
    assert stats["max_epe"] == 0.0
    assert np.isclose(stats["valid_ratio"], float(mask.float().mean().item() * 100.0), atol=1e-2)
    print(f"  [PASS] Residual statistics on zero-flow: mean_epe={stats['mean_epe']:.2f}, valid_ratio={stats['valid_ratio']:.1f}%")


def test_synthetic_known_difference_exact_residual() -> None:
    print("\n=== Test 2: Synthetic Known Difference -> Exact Residual ===")
    height, width = 436, 1024

    # Background rigid flow: constant flow representing horizontal camera pan
    rigid_flow = torch.full((height, width, 2), 5.0, dtype=torch.float32)
    rigid_flow[..., 1] = -2.0  # (u=5.0, v=-2.0)

    # Injected independent object motion in a known bounding box [120:220, 300:450]
    r0, r1 = 120, 220
    c0, c1 = 300, 450
    delta_u = 14.5
    delta_v = -8.25

    observed_flow = rigid_flow.clone()
    observed_flow[r0:r1, c0:c1, 0] += delta_u
    observed_flow[r0:r1, c0:c1, 1] += delta_v

    # Mask out a corner [0:50, 0:50] as invalid (e.g. occlusion/border)
    valid_mask = torch.ones((height, width), dtype=torch.bool)
    valid_mask[0:50, 0:50] = False

    res_flow, res_mask = compute_residual_flow(
        observed_flow=observed_flow,
        rigid_flow=rigid_flow,
        valid_mask=valid_mask,
        zero_invalid=True,
    )

    # Inside object bounding box: residual should match delta_w exactly
    obj_res = res_flow[r0:r1, c0:c1]
    expected_obj_mag = math.sqrt(delta_u ** 2 + delta_v ** 2)  # sqrt(14.5^2 + 8.25^2) = 16.68 px

    err_u = torch.max(torch.abs(obj_res[..., 0] - delta_u)).item()
    err_v = torch.max(torch.abs(obj_res[..., 1] - delta_v)).item()
    assert err_u < 1e-6, f"Object delta_u mismatch: {err_u}"
    assert err_v < 1e-6, f"Object delta_v mismatch: {err_v}"
    print(f"  [PASS] Injected object box [{r0}:{r1}, {c0}:{c1}]:")
    print(f"         Expected: (u={delta_u:.2f}, v={delta_v:.2f}) -> Recovered: (u={obj_res[0, 0, 0].item():.2f}, v={obj_res[0, 0, 1].item():.2f})")
    print(f"         Max error inside object box: u={err_u:.2e} px, v={err_v:.2e} px")

    # Outside object bounding box and inside valid region: residual should be exactly 0.0
    bg_mask = valid_mask.clone()
    bg_mask[r0:r1, c0:c1] = False
    bg_res = res_flow[bg_mask]
    max_bg_err = torch.max(torch.abs(bg_res)).item()
    assert max_bg_err == 0.0, f"Background residual non-zero: {max_bg_err}"
    print(f"  [PASS] Rigid background region: Exactly 0.0 residual (max err: {max_bg_err:.2e} px)")

    # In invalid corner [0:50, 0:50]: residual should be strictly zeroed out
    invalid_res = res_flow[0:50, 0:50]
    assert (invalid_res == 0.0).all().item()
    assert (~res_mask[0:50, 0:50]).all().item()
    print(f"  [PASS] Invalid region [0:50, 0:50]: Correctly masked and zeroed out")

    # Magnitude calculation check
    mag = compute_residual_magnitude(res_flow, valid_mask=res_mask)
    obj_mag = mag[r0:r1, c0:c1]
    mag_err = torch.max(torch.abs(obj_mag - expected_obj_mag)).item()
    assert mag_err < 1e-5, f"Magnitude calculation mismatch: {mag_err}"
    print(f"  [PASS] Residual magnitude calculation: {expected_obj_mag:.2f} px (err: {mag_err:.2e} px)")


def test_real_sintel_residual_flow() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    print("\n=== Test 3: Real Sintel Consecutive Frames (Ground Truth vs. Rigid Flow) ===")
    frame1_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0001.png"
    frame2_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0002.png"
    flow_path = project_root / "data" / "Sintel" / "training" / "flow" / "alley_1" / "frame_0001.flo"

    assert frame1_path.is_file(), f"Missing frame1 at {frame1_path}"
    assert frame2_path.is_file(), f"Missing frame2 at {frame2_path}"
    assert flow_path.is_file(), f"Missing ground-truth flow at {flow_path}"

    print(f"Loading Sintel sequence: alley_1 (frames 0001 & 0002)")
    img1_bgr = cv2.imread(str(frame1_path))
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

    # 1. Load ground truth observed optical flow [2, H, W]
    print("  Loading ground-truth observed optical flow (.flo)...")
    gt_flow = read_flo(flow_path)
    h, w = gt_flow.shape[1], gt_flow.shape[2]
    print(f"  Observed Flow shape: {tuple(gt_flow.shape)}, dtype={gt_flow.dtype}")
    assert torch.isfinite(gt_flow).all().item(), "Observed flow contains non-finite values"

    # 2. Load official .cam calibration
    cam1 = CameraIntrinsics.from_sintel_frame(frame1_path)
    cam2 = CameraIntrinsics.from_sintel_frame(frame2_path)

    # 3. Estimate MiDaS depth on Frame 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Estimating monocular depth via MiDaS Small on {device}...")
    estimator = MiDaSDepthEstimator(device=device, scale_factor=1000.0)
    depth = estimator(img1_rgb)  # [H, W]

    # 4. Compute camera-induced rigid flow via Step 2
    print("  Computing rigid camera-induced optical flow...")
    rigid_flow, rigid_mask = compute_rigid_flow(
        depth=depth,
        intrinsics_1=cam1,
        intrinsics_2=cam2,
        output_channels_first=True,  # Return [2, H, W] matching gt_flow
    )
    assert rigid_flow.shape == (2, h, w)

    # 5. Compute residual optical flow: w_res = w_obs - w_rigid
    print("  Computing residual optical flow: w_res = w_obs - w_rigid...")
    gt_flow_dev = gt_flow.to(device=device)
    residual_flow, res_mask = compute_residual_flow(
        observed_flow=gt_flow_dev,
        rigid_flow=rigid_flow,
        valid_mask=rigid_mask,
        zero_invalid=True,
    )

    # 6. Verify assertions and compute metrics
    assert residual_flow.shape == (2, h, w)
    assert res_mask.shape == (h, w)
    assert torch.isfinite(residual_flow).all().item(), "Residual flow contains NaN or Inf values"

    stats = compute_residual_statistics(residual_flow, res_mask)
    print("\n  --- Residual Flow Evaluation Summary ---")
    print(f"  Valid Pixel Ratio:   {stats['valid_ratio']:.2f}% ({res_mask.sum().item()} / {res_mask.numel()})")
    print(f"  Finite Pixel Ratio:  {stats['finite_ratio']:.2f}%")
    print(f"  Mean Residual EPE:   {stats['mean_epe']:.2f} px")
    print(f"  Median Residual EPE: {stats['median_epe']:.2f} px")
    print(f"  95th %-tile EPE:     {stats['p95_epe']:.2f} px")
    print(f"  Max Residual EPE:    {stats['max_epe']:.2f} px")
    print(f"  RMSE Residual:       {stats['rmse']:.2f} px")

    assert stats["valid_ratio"] > 95.0, f"Valid ratio too low: {stats['valid_ratio']:.2f}%"
    assert np.isclose(stats["finite_ratio"], 100.0, atol=1e-3), "Non-finite pixels detected"
    assert stats["mean_epe"] > 0.0, "Expected non-zero residual for real frame pair"

    print("\n  [PASS] Residual flow correctly computed on real Sintel frames")
    print("  [NOTE] Residual flow reflects kinematic discrepancy (independent motion + depth/scale mismatch).")
    print("         It is NOT yet dynamic object segmentation.")

    return gt_flow_dev, rigid_flow, residual_flow, res_mask, img1_bgr


def test_save_residual_visualization(
    gt_flow: torch.Tensor,
    rigid_flow: torch.Tensor,
    residual_flow: torch.Tensor,
    res_mask: torch.Tensor,
    img1_bgr: np.ndarray,
) -> None:
    print("\n=== Test 4: Save Multi-Panel Residual Flow Visualization ===")
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "day7_step3_residual_flow_alley_1.png"

    h, w = img1_bgr.shape[:2]
    half_size = (w // 2, h // 2)

    # Panel 1: Original RGB Image
    p1 = cv2.resize(img1_bgr, half_size)
    cv2.putText(p1, "1. RGB Frame 1", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: Observed Flow (Ground Truth) HSV
    obs_hsv = flow_to_hsv_bgr(gt_flow.permute(1, 2, 0), valid_mask=res_mask)
    p2 = cv2.resize(obs_hsv, half_size)
    cv2.putText(p2, "2. Observed Flow (GT)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: Rigid Camera Flow HSV
    rigid_hsv = flow_to_hsv_bgr(rigid_flow.permute(1, 2, 0), valid_mask=res_mask)
    p3 = cv2.resize(rigid_hsv, half_size)
    cv2.putText(p3, "3. Rigid Flow (Camera)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 4: Residual Flow HSV (w_obs - w_rigid)
    res_hsv = flow_to_hsv_bgr(residual_flow.permute(1, 2, 0), valid_mask=res_mask)
    p4 = cv2.resize(res_hsv, half_size)
    cv2.putText(p4, "4. Residual Flow", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 5: Residual Magnitude Heatmap
    res_mag = compute_residual_magnitude(residual_flow, valid_mask=res_mask).cpu().numpy()
    mask_np = res_mask.cpu().numpy().astype(bool)
    p99_mag = np.percentile(res_mag[mask_np], 99) if mask_np.any() else 5.0
    norm_mag = np.clip(res_mag / max(p99_mag, 1e-2) * 255.0, 0, 255).astype(np.uint8)
    norm_mag[~mask_np] = 0
    mag_colormap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_TURBO)
    mag_colormap[~mask_np] = 0
    p5 = cv2.resize(mag_colormap, half_size)
    cv2.putText(p5, f"5. Residual Mag (p99={p99_mag:.1f}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 6: Validity Mask
    valid_vis = (mask_np.astype(np.uint8) * 255)
    valid_bgr = cv2.cvtColor(valid_vis, cv2.COLOR_GRAY2BGR)
    p6 = cv2.resize(valid_bgr, half_size)
    cv2.putText(p6, "6. Validity Mask", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Assemble 2x3 composite grid:
    # Row 1: [RGB] [Observed Flow] [Rigid Flow]
    # Row 2: [Residual Flow] [Residual Mag] [Validity Mask]
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    composite = np.vstack([row1, row2])

    cv2.imwrite(str(out_path), composite)
    assert out_path.is_file(), f"Failed to save visualization at {out_path}"
    file_size_kb = out_path.stat().st_size / 1024.0
    assert file_size_kb > 30.0, f"Saved image file suspiciously small: {file_size_kb:.1f} KB"
    print(f"  [PASS] Composite 6-panel visualization saved: {out_path.relative_to(project_root)} ({file_size_kb:.1f} KB)")


def main() -> None:
    print("Running Day 7 Step 3: Residual Optical Flow Test Suite\n")
    test_observed_equals_rigid_zero_residual()
    test_synthetic_known_difference_exact_residual()
    gt_flow, rigid_flow, residual_flow, res_mask, img1_bgr = test_real_sintel_residual_flow()
    test_save_residual_visualization(gt_flow, rigid_flow, residual_flow, res_mask, img1_bgr)
    print("\n=======================================================")
    print("  ALL DAY 7 STEP 3 VERIFICATION CHECKS PASSED (4/4)")
    print("=======================================================")


if __name__ == "__main__":
    main()
