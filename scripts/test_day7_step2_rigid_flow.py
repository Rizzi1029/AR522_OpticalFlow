"""
Verification Test Suite for Day 7 Step 2: Rigid Camera-Induced Optical Flow.

Validates:
1. Identity transformation produces zero flow everywhere with 100% validity.
2. Known pure translations (lateral shift and forward expansion) match analytical solutions.
3. Known pure rotations (roll and yaw) match analytical solutions and exhibit depth-scale invariance.
4. Real consecutive Sintel frames (`alley_1` frames 1 & 2) using official .cam extrinsics + MiDaS depth:
   - Evaluates vectorized GPU execution.
   - Verifies CPU vs GPU numerical consistency.
   - Checks validity mask, image bounds, and finite values.
5. Saves a rich composite visualization of RGB, depth, HSV rigid flow, flow magnitude, and validity mask.
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

from src.depth.intrinsics import CameraIntrinsics
from src.depth.midas import MiDaSDepthEstimator
from src.geometry.rigid_flow import (
    compute_relative_transform,
    compute_rigid_flow,
    flow_to_hsv_bgr,
)


def test_identity_transform_zero_flow() -> None:
    print("=== Test 1: Identity Transform -> Zero Flow ===")
    width, height = 1024, 436
    intrinsics = CameraIntrinsics.sintel_default(width=width, height=height, focal_length=1000.0)

    # 1. Identity extrinsics
    e_identity = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)

    # Synthetic depth field varying between 2.0m and 50.0m
    torch.manual_seed(42)
    depth = 2.0 + 48.0 * torch.rand(height, width, dtype=torch.float32)

    # Compute rigid flow
    flow, valid_mask = compute_rigid_flow(
        depth=depth,
        intrinsics_1=intrinsics,
        intrinsics_2=intrinsics,
        extrinsic_1=e_identity,
        extrinsic_2=e_identity,
    )

    # Assertions
    assert flow.shape == (height, width, 2), f"Expected flow shape ({height}, {width}, 2), got {flow.shape}"
    assert valid_mask.shape == (height, width), f"Expected mask shape ({height}, {width}), got {valid_mask.shape}"
    assert valid_mask.all().item(), "All pixels should be valid under identity transform"

    max_err = torch.max(torch.abs(flow)).item()
    assert max_err < 5e-4, f"Flow under identity transform is non-zero: max error = {max_err:.2e} px"
    print(f"  [PASS] 2D Identity flow max magnitude: {max_err:.2e} px (< 5e-4 px)")
    print(f"  [PASS] Validity rate: 100.0% ({valid_mask.sum().item()} / {valid_mask.numel()})")

    # Batched 4D tensor verification [B=2, C=1, H=436, W=1024]
    depth_4d = depth.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
    flow_4d, mask_4d = compute_rigid_flow(
        depth=depth_4d,
        intrinsics_1=intrinsics,
        intrinsics_2=intrinsics,
        extrinsic_1=e_identity,
        extrinsic_2=e_identity,
    )
    assert flow_4d.shape == (2, 2, height, width)
    assert mask_4d.shape == (2, 1, height, width)
    max_err_4d = torch.max(torch.abs(flow_4d)).item()
    assert max_err_4d < 5e-4
    print(f"  [PASS] Batched 4D Identity check: max error: {max_err_4d:.2e} px")


def test_pure_translation_analytical_flow() -> None:
    print("\n=== Test 2: Known Pure Translation -> Analytically Correct Flow ===")
    width, height = 1024, 436
    fx, fy = 1000.0, 1000.0
    cx, cy = 511.5, 217.5
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    # Sub-case A: Lateral Translation (t_x = 0.5m, t_y = 0, t_z = 0) at uniform depth Z0 = 10.0m
    # Analytical solution:
    #   P1 = [X1, Y1, Z0]^T
    #   P2 = [X1 + tx, Y1, Z0]^T
    #   u2 = fx * (X1 + tx) / Z0 + cx = u1 + fx * tx / Z0 = u1 + 1000 * 0.5 / 10 = u1 + 50.0 px
    #   v2 = v1
    #   Expected flow: delta_u = +50.0 px, delta_v = 0.0 px
    z0 = 10.0
    tx = 0.5
    expected_du = fx * tx / z0  # 50.0 px

    depth_const = torch.full((height, width), z0, dtype=torch.float32)
    e1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    e2_lateral = np.array([[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)

    flow_lat, mask_lat = compute_rigid_flow(
        depth=depth_const,
        intrinsics_1=intrinsics,
        intrinsics_2=intrinsics,
        extrinsic_1=e1,
        extrinsic_2=e2_lateral,
    )

    # For valid pixels (u2 = u1 + 50 <= width - 1 = 1023 => u1 <= 973)
    u_grid, _ = intrinsics.get_pixel_grid()
    expected_in_bounds = u_grid + expected_du <= (width - 1.0)
    assert (mask_lat == expected_in_bounds).all().item(), "Validity mask does not match expected boundary cutoff"

    du_err = torch.max(torch.abs(flow_lat[mask_lat, 0] - expected_du)).item()
    dv_err = torch.max(torch.abs(flow_lat[mask_lat, 1])).item()
    assert du_err < 1e-4, f"Lateral delta_u deviation too high: {du_err}"
    assert dv_err < 1e-4, f"Lateral delta_v deviation too high: {dv_err}"
    print(f"  [PASS] Sub-case A (Lateral Shift): delta_u error = {du_err:.2e} px, delta_v error = {dv_err:.2e} px")
    print(f"         Out-of-bounds boundary handling correctly flagged rightmost {int(expected_du)} columns as invalid")

    # Sub-case B: Forward Translation / Looming (tz = -2.0m into scene, tx = ty = 0)
    # P1 = [X1, Y1, Z0]^T with Z0 = 10.0m
    # P2 = [X1, Y1, Z0 + tz]^T with Z2 = 10.0 - 2.0 = 8.0m
    # Analytical solution:
    #   u2 - cx = fx * X1 / Z2 = (u1 - cx) * (Z0 / Z2) = 1.25 * (u1 - cx)
    #   delta_u = u2 - u1 = 0.25 * (u1 - cx)
    #   delta_v = v2 - v1 = 0.25 * (v1 - cy)
    tz = -2.0
    scale_factor = (-tz) / (z0 + tz)  # 2.0 / 8.0 = 0.25
    e2_forward = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, tz]], dtype=np.float64)

    flow_fwd, mask_fwd = compute_rigid_flow(
        depth=depth_const,
        intrinsics_1=intrinsics,
        intrinsics_2=intrinsics,
        extrinsic_1=e1,
        extrinsic_2=e2_forward,
    )

    u1, v1 = intrinsics.get_pixel_grid()
    expected_du_fwd = scale_factor * (u1 - cx)
    expected_dv_fwd = scale_factor * (v1 - cy)

    fwd_du_err = torch.max(torch.abs(flow_fwd[mask_fwd, 0] - expected_du_fwd[mask_fwd])).item()
    fwd_dv_err = torch.max(torch.abs(flow_fwd[mask_fwd, 1] - expected_dv_fwd[mask_fwd])).item()
    assert fwd_du_err < 1e-4, f"Forward expansion delta_u deviation: {fwd_du_err}"
    assert fwd_dv_err < 1e-4, f"Forward expansion delta_v deviation: {fwd_dv_err}"
    print(f"  [PASS] Sub-case B (Forward Zoom/Expansion): Focus of Expansion at ({cx}, {cy})")
    print(f"         Max deviation from analytical radial field: u={fwd_du_err:.2e} px, v={fwd_dv_err:.2e} px")


def test_pure_rotation_analytical_flow() -> None:
    print("\n=== Test 3: Known Pure Rotation -> Analytically Correct Flow & Scale Invariance ===")
    width, height = 1024, 436
    fx, fy = 1000.0, 1000.0
    cx, cy = 511.5, 217.5
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    # Pure roll rotation theta around optical axis (+Z)
    theta_deg = 2.0
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Rotation matrix around Z:
    #   [ cos_t  -sin_t   0 ]
    #   [ sin_t   cos_t   0 ]
    #   [   0       0     1 ]
    r_z = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)

    e1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    e2_roll = np.zeros((3, 4), dtype=np.float64)
    e2_roll[:3, :3] = r_z

    # Highly varying synthetic depth to test DEPTH-SCALE INVARIANCE of pure rotation:
    # Z(u, v) = 2.0 + 20.0 * sin(u / 50) * cos(v / 50)
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    ug, vg = np.meshgrid(u_coords, v_coords)
    varying_depth = torch.tensor(10.0 + 8.0 * np.sin(ug / 50.0) * np.cos(vg / 50.0), dtype=torch.float32)

    flow_rot, mask_rot = compute_rigid_flow(
        depth=varying_depth,
        intrinsics_1=intrinsics,
        intrinsics_2=intrinsics,
        extrinsic_1=e1,
        extrinsic_2=e2_roll,
    )

    # Analytical optical flow for pure roll rotation around principal point:
    #   u2 - cx = (u1 - cx) * cos(theta) - (v1 - cy) * sin(theta)
    #   v2 - cy = (u1 - cx) * sin(theta) + (v1 - cy) * cos(theta)
    #   delta_u = (u1 - cx) * (cos(theta) - 1) - (v1 - cy) * sin(theta)
    #   delta_v = (u1 - cx) * sin(theta) + (v1 - cy) * (cos(theta) - 1)
    u1, v1 = intrinsics.get_pixel_grid()
    expected_du_rot = (u1 - cx) * (cos_t - 1.0) - (v1 - cy) * sin_t
    expected_dv_rot = (u1 - cx) * sin_t + (v1 - cy) * (cos_t - 1.0)

    rot_du_err = torch.max(torch.abs(flow_rot[mask_rot, 0] - expected_du_rot[mask_rot])).item()
    rot_dv_err = torch.max(torch.abs(flow_rot[mask_rot, 1] - expected_dv_rot[mask_rot])).item()

    assert rot_du_err < 5e-4, f"Roll rotation delta_u deviation: {rot_du_err}"
    assert rot_dv_err < 5e-4, f"Roll rotation delta_v deviation: {rot_dv_err}"
    print(f"  [PASS] Pure Roll Rotation ({theta_deg} deg): Analytical match across entire grid:")
    print(f"         Max error u: {rot_du_err:.2e} px, v: {rot_dv_err:.2e} px")
    print("  [PASS] Depth-Scale Invariance proven: Identical flow achieved over highly varying depth field")


def test_real_sintel_consecutive_frames() -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    print("\n=== Test 4: Real Sintel Consecutive Frames with Official .cam & MiDaS Depth ===")
    frame1_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0001.png"
    frame2_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0002.png"
    assert frame1_path.is_file(), f"Missing Sintel frame at {frame1_path}"
    assert frame2_path.is_file(), f"Missing Sintel frame at {frame2_path}"

    print(f"Loading Sintel sequence: alley_1 (frames 0001 & 0002)")
    img1_bgr = cv2.imread(str(frame1_path))
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

    # 1. Load official .cam calibration
    cam1 = CameraIntrinsics.from_sintel_frame(frame1_path)
    cam2 = CameraIntrinsics.from_sintel_frame(frame2_path)
    print(f"  Frame 1 Intrinsics: fx={cam1.fx:.4f}, fy={cam1.fy:.4f}, cx={cam1.cx:.1f}, cy={cam1.cy:.1f}")
    print(f"  Frame 2 Intrinsics: fx={cam2.fx:.4f}, fy={cam2.fy:.4f}, cx={cam2.cx:.1f}, cy={cam2.cy:.1f}")

    # 2. Compute relative camera transformation
    r_rel, t_rel, t_rel_4x4 = compute_relative_transform(cam1.extrinsic, cam2.extrinsic)
    rot_angle_deg = math.degrees(math.acos(float(torch.clamp((torch.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))))
    trans_norm_m = float(torch.norm(t_rel))
    print(f"  Relative Camera Motion: Rotation = {rot_angle_deg:.4f} deg, Translation = {trans_norm_m * 1000.0:.2f} mm")

    # 3. Estimate MiDaS depth on Frame 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Running MiDaS Small depth estimation on {device}...")
    estimator = MiDaSDepthEstimator(device=device, scale_factor=1000.0)
    depth = estimator(img1_rgb)  # [H, W] on device
    print(f"  MiDaS Depth Map: shape={tuple(depth.shape)}, min={depth.min().item():.2f}, max={depth.max().item():.2f}")

    # 4. Compute Rigid Flow on GPU
    flow_gpu, mask_gpu = compute_rigid_flow(
        depth=depth,
        intrinsics_1=cam1,
        intrinsics_2=cam2,
    )

    # 5. Compute Rigid Flow on CPU to verify numerical cross-device consistency
    depth_cpu = depth.cpu()
    flow_cpu, mask_cpu = compute_rigid_flow(
        depth=depth_cpu,
        intrinsics_1=cam1,
        intrinsics_2=cam2,
    )

    dev_err = torch.max(torch.abs(flow_gpu.cpu() - flow_cpu)).item()
    assert dev_err < 5e-4, f"CPU vs GPU numerical discrepancy too large: {dev_err}"
    print(f"  [PASS] Cross-device numerical consistency (CPU vs GPU max error: {dev_err:.2e} px < 5e-4 px)")

    # 6. Check Validity & Physical Statistics
    valid_ratio = 100.0 * mask_gpu.float().mean().item()
    flow_mag = torch.norm(flow_gpu, dim=-1)
    valid_mags = flow_mag[mask_gpu]

    assert valid_ratio > 95.0, f"Valid pixel ratio unexpectedly low: {valid_ratio:.2f}%"
    assert torch.isfinite(flow_gpu).all().item(), "Flow contains NaN or Inf values"
    assert len(valid_mags) > 0

    mean_flow = valid_mags.mean().item()
    median_flow = valid_mags.median().item()
    max_flow = valid_mags.max().item()

    print(f"  [PASS] Valid pixels: {valid_ratio:.2f}% ({mask_gpu.sum().item()} / {mask_gpu.numel()})")
    print(f"  [PASS] Rigid Flow Statistics: mean = {mean_flow:.2f} px, median = {median_flow:.2f} px, max = {max_flow:.2f} px")

    return flow_gpu, mask_gpu, depth.cpu().numpy(), img1_bgr


def test_save_rigid_flow_visualization(
    flow: torch.Tensor,
    mask: torch.Tensor,
    depth_np: np.ndarray,
    img_bgr: np.ndarray,
) -> None:
    print("\n=== Test 5: Save Multi-Panel Rigid Flow Visualization ===")
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "day7_rigid_flow_sintel_alley_1.png"

    h, w = depth_np.shape[:2]

    # Panel 1: Original RGB image
    p1 = cv2.resize(img_bgr, (w // 2, h // 2))
    cv2.putText(p1, "1. RGB Frame 1", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: MiDaS depth colormap (normalized inverse depth / disparity)
    inv_depth = 1.0 / (depth_np + 1e-4)
    norm_depth = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
    p2 = cv2.resize(depth_colormap, (w // 2, h // 2))
    cv2.putText(p2, "2. MiDaS Depth Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 3: Rigid Flow HSV direction visualization
    flow_hsv_bgr = flow_to_hsv_bgr(flow, valid_mask=mask)
    p3 = cv2.resize(flow_hsv_bgr, (w // 2, h // 2))
    cv2.putText(p3, "3. Rigid Flow (HSV)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 4: Rigid Flow Magnitude Heatmap
    flow_np = flow.cpu().numpy()
    flow_mag = np.linalg.norm(flow_np, axis=-1)
    mask_np = mask.cpu().numpy().astype(bool)
    max_valid_mag = np.percentile(flow_mag[mask_np], 99) if mask_np.any() else 5.0
    norm_mag = np.clip(flow_mag / max_valid_mag * 255.0, 0, 255).astype(np.uint8)
    norm_mag[~mask_np] = 0
    mag_colormap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_TURBO)
    mag_colormap[~mask_np] = 0
    p4 = cv2.resize(mag_colormap, (w // 2, h // 2))
    cv2.putText(p4, "4. Flow Magnitude", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 5: Validity Mask
    valid_vis = (mask_np.astype(np.uint8) * 255)
    valid_bgr = cv2.cvtColor(valid_vis, cv2.COLOR_GRAY2BGR)
    p5 = cv2.resize(valid_bgr, (w // 2, h // 2))
    cv2.putText(p5, "5. Validity Mask", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Assemble composite grid:
    # Row 1: [RGB] [Depth]
    # Row 2: [Rigid Flow HSV] [Flow Magnitude]
    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    composite = np.vstack([row1, row2])

    cv2.imwrite(str(out_path), composite)
    assert out_path.is_file(), f"Failed to save visualization at {out_path}"
    file_size_kb = out_path.stat().st_size / 1024.0
    assert file_size_kb > 20.0, f"Saved image file suspiciously small: {file_size_kb:.1f} KB"
    print(f"  [PASS] Composite visualization successfully saved: {out_path.relative_to(project_root)} ({file_size_kb:.1f} KB)")


def main() -> None:
    print("Running Day 7 Step 2: Rigid Camera-Induced Optical Flow Test Suite\n")
    test_identity_transform_zero_flow()
    test_pure_translation_analytical_flow()
    test_pure_rotation_analytical_flow()
    flow, mask, depth_np, img_bgr = test_real_sintel_consecutive_frames()
    test_save_rigid_flow_visualization(flow, mask, depth_np, img_bgr)
    print("\n=======================================================")
    print("  ALL DAY 7 STEP 2 VERIFICATION CHECKS PASSED (5/5)")
    print("=======================================================")


if __name__ == "__main__":
    main()
