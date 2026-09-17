"""
Verification script for Day 7 Step 1: Depth and Camera-Intrinsics Foundation.

Validates:
1. Camera intrinsic matrix K, inverse K^{-1}, and coordinate system conventions.
2. Official MPI Sintel binary .cam file loading (K, extrinsic pose, tag check).
3. Frame-specific calibration resolution (including dynamic focal changes in zoom scenes).
4. Exact round-trip consistency of 2D -> 3D backprojection and 3D -> 2D reprojection.
5. MiDaS Small monocular depth estimation on a real MPI Sintel frame:
   - Output shape matching original input image (436, 1024).
   - Finite values verification (no NaNs, no Infs).
   - Positive depth verification (Z > 0 for all pixels).
6. End-to-end backprojection/reprojection consistency of real estimated Sintel depth map
   using official Sintel .cam calibration.
7. Inference latency and memory profiling on NVIDIA GeForce RTX 5050 Laptop GPU.
"""

import sys
from pathlib import Path
import time
import numpy as np
import torch
import cv2

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.depth.intrinsics import CameraIntrinsics, read_sintel_cam_binary
from src.depth.midas import MiDaSDepthEstimator


def test_camera_intrinsics_math() -> None:
    print("=== Test 1: Camera Intrinsics Representation & Math Properties ===")
    intrinsics = CameraIntrinsics.sintel_default(width=1024, height=436, focal_length=1000.0)

    # 1. Shape and values of K
    K = intrinsics.K
    K_inv = intrinsics.K_inv
    assert K.shape == (3, 3), f"Expected K shape (3, 3), got {K.shape}"
    assert K_inv.shape == (3, 3), f"Expected K_inv shape (3, 3), got {K_inv.shape}"

    # 2. Check matrix inverse identity: K @ K_inv == I
    identity_approx = K @ K_inv
    identity_err = np.max(np.abs(identity_approx - np.eye(3)))
    assert identity_err < 1e-12, f"K @ K_inv error too large: {identity_err}"
    print(f"  [PASS] K @ K_inv == I (max error: {identity_err:.2e})")

    # 3. Check principal point matches exact geometric center
    assert intrinsics.cx == 511.5, f"Expected cx=511.5, got {intrinsics.cx}"
    assert intrinsics.cy == 217.5, f"Expected cy=217.5, got {intrinsics.cy}"
    print(f"  [PASS] Sintel principal point: cx={intrinsics.cx}, cy={intrinsics.cy}")

    # 4. Check PyTorch tensor generation
    K_t = intrinsics.get_k_tensor()
    K_inv_t = intrinsics.get_k_inv_tensor()
    assert K_t.shape == (3, 3) and K_inv_t.shape == (3, 3)
    t_err = torch.max(torch.abs(K_t @ K_inv_t - torch.eye(3))).item()
    assert t_err < 1e-6, f"PyTorch K @ K_inv error: {t_err}"
    print(f"  [PASS] PyTorch K tensor identity check (max error: {t_err:.2e})")


def test_sintel_cam_loader() -> None:
    print("\n=== Test 2: Official MPI Sintel .cam Binary Loader & Frame Calibration ===")
    cam_file = project_root / "data" / "Sintel" / "training" / "camdata_left" / "alley_1" / "frame_0001.cam"
    assert cam_file.is_file(), f"Expected .cam file at {cam_file}"

    # 1. Direct binary parsing
    K_raw, extrinsic_raw = read_sintel_cam_binary(cam_file)
    assert K_raw.shape == (3, 3), f"Expected K shape (3, 3), got {K_raw.shape}"
    assert extrinsic_raw.shape == (3, 4), f"Expected extrinsic shape (3, 4), got {extrinsic_raw.shape}"
    assert np.isclose(K_raw[0, 2], 511.5), f"Expected cx=511.5, got {K_raw[0, 2]}"
    assert np.isclose(K_raw[1, 2], 217.5), f"Expected cy=217.5, got {K_raw[1, 2]}"
    assert np.isclose(K_raw[0, 0], K_raw[1, 1]), "Expected square pixels fx == fy"
    print(f"  [PASS] Raw binary .cam read successful: fx={K_raw[0, 0]:.4f}, fy={K_raw[1, 1]:.4f}")

    # 2. CameraIntrinsics.from_cam_file
    calib = CameraIntrinsics.from_cam_file(cam_file)
    inv_err = np.max(np.abs(calib.K @ calib.K_inv - np.eye(3)))
    assert inv_err < 1e-12, f"K @ K_inv error too large: {inv_err}"
    assert calib.extrinsic is not None and calib.extrinsic.shape == (3, 4)
    print(f"  [PASS] CameraIntrinsics.from_cam_file verified (K @ K_inv error: {inv_err:.2e})")

    # 3. Frame-specific resolution via CameraIntrinsics.from_sintel_frame
    frame_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0001.png"
    calib_frame = CameraIntrinsics.from_sintel_frame(frame_path)
    assert np.allclose(calib_frame.K, calib.K)
    print(f"  [PASS] CameraIntrinsics.from_sintel_frame correctly resolved calibration for {frame_path.name}")

    # 4. Dynamic focal change verification on zoom scene (cave_2)
    cave_f1 = project_root / "data" / "Sintel" / "training" / "camdata_left" / "cave_2" / "frame_0001.cam"
    cave_f50 = project_root / "data" / "Sintel" / "training" / "camdata_left" / "cave_2" / "frame_0050.cam"
    if cave_f1.is_file() and cave_f50.is_file():
        calib_cave1 = CameraIntrinsics.from_cam_file(cave_f1)
        calib_cave50 = CameraIntrinsics.from_cam_file(cave_f50)
        print(f"  [PASS] Dynamic zoom verification on cave_2:")
        print(f"         frame_0001: fx={calib_cave1.fx:.2f} px")
        print(f"         frame_0050: fx={calib_cave50.fx:.2f} px")
        assert not np.isclose(calib_cave1.fx, calib_cave50.fx), "Focal length should change in cave_2 zoom shot"


def test_synthetic_roundtrip_consistency() -> None:
    print("\n=== Test 3: Synthetic Backprojection / Reprojection Consistency ===")
    intrinsics = CameraIntrinsics.sintel_default(width=1024, height=436, focal_length=1000.0)

    # Create synthetic test points across the image grid
    np.random.seed(42)
    sample_u = np.array([0.0, 511.5, 1023.0, 100.0, 900.0])
    sample_v = np.array([0.0, 217.5, 435.0, 50.0, 380.0])
    sample_z = np.array([1.0, 5.0, 25.0, 10.0, 100.0])

    # Manual backprojection
    x = (sample_u - intrinsics.cx) * sample_z / intrinsics.fx
    y = (sample_v - intrinsics.cy) * sample_z / intrinsics.fy
    pts_3d = np.stack([x, y, sample_z], axis=-1)  # [5, 3]

    # Manual reprojection
    u_reproj = intrinsics.fx * (pts_3d[:, 0] / pts_3d[:, 2]) + intrinsics.cx
    v_reproj = intrinsics.fy * (pts_3d[:, 1] / pts_3d[:, 2]) + intrinsics.cy

    u_err = np.max(np.abs(u_reproj - sample_u))
    v_err = np.max(np.abs(v_reproj - sample_v))
    assert u_err < 1e-12, f"Sample u reprojection error: {u_err}"
    assert v_err < 1e-12, f"Sample v reprojection error: {v_err}"
    print(f"  [PASS] Analytical point roundtrip error: u={u_err:.2e} px, v={v_err:.2e} px")

    # Full grid synthetic depth tensor roundtrip
    synthetic_depth = torch.linspace(1.0, 50.0, 436 * 1024).reshape(436, 1024)
    pts_3d_torch = intrinsics.backproject_torch(synthetic_depth)
    assert pts_3d_torch.shape == (436, 1024, 3)

    uv_reproj_torch, valid_mask = intrinsics.reproject_torch(pts_3d_torch)
    assert valid_mask.all()

    u_grid, v_grid = intrinsics.get_pixel_grid()
    max_reproj_err_u = torch.max(torch.abs(uv_reproj_torch[..., 0] - u_grid)).item()
    max_reproj_err_v = torch.max(torch.abs(uv_reproj_torch[..., 1] - v_grid)).item()

    assert max_reproj_err_u < 5e-4, f"Grid u reprojection error too large: {max_reproj_err_u}"
    assert max_reproj_err_v < 5e-4, f"Grid v reprojection error too large: {max_reproj_err_v}"
    print(f"  [PASS] Full-grid synthetic reprojection max error: u={max_reproj_err_u:.2e} px, v={max_reproj_err_v:.2e} px (< 5e-4 px)")


def test_depth_estimation_on_sintel_frame() -> None:
    print("\n=== Test 4: Monocular Depth & Official Calibration on Real Sintel Frame ===")
    sintel_frame_path = project_root / "data" / "Sintel" / "training" / "clean" / "alley_1" / "frame_0001.png"
    assert sintel_frame_path.exists(), f"Sintel training frame not found at {sintel_frame_path}"

    print(f"Loading real Sintel training frame: {sintel_frame_path.relative_to(project_root)}")
    img_bgr = cv2.imread(str(sintel_frame_path))
    assert img_bgr is not None, "Failed to load image"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    print(f"Input image shape: (H={h_orig}, W={w_orig}, C={img_rgb.shape[2]})")
    assert (h_orig, w_orig) == (436, 1024), f"Expected Sintel resolution (436, 1024), got ({h_orig}, {w_orig})"

    # Load official calibration corresponding to this frame
    print("Loading official Sintel camera calibration for frame...")
    intrinsics = CameraIntrinsics.from_sintel_frame(sintel_frame_path, width=w_orig, height=h_orig)

    print("\n--- Official Calibration Details ---")
    print("Actual Intrinsic Matrix K used:")
    print(intrinsics.K)
    print(f"  fx: {intrinsics.fx:.6f} px")
    print(f"  fy: {intrinsics.fy:.6f} px")
    print(f"  cx: {intrinsics.cx:.2f} px")
    print(f"  cy: {intrinsics.cy:.2f} px")
    if intrinsics.extrinsic is not None:
        print("Actual Extrinsic Matrix [R | T] (3x4):")
        print(intrinsics.extrinsic)

    # Verify K @ K_inv == I for the official calibrated K
    id_err = np.max(np.abs(intrinsics.K @ intrinsics.K_inv - np.eye(3)))
    assert id_err < 1e-12, f"Calibrated K @ K_inv error: {id_err}"
    print(f"  [PASS] Official K @ K_inv == I check: {id_err:.2e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing MiDaS Small depth estimator on device: {device}...")
    estimator = MiDaSDepthEstimator(device=device, scale_factor=1000.0)

    # 1. Warmup and latency benchmark
    print("Running GPU warmup...")
    _ = estimator(img_rgb)
    if device == "cuda":
        torch.cuda.synchronize()

    print("Benchmarking inference latency (10 iterations)...")
    latencies = []
    for _ in range(10):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth = estimator(img_rgb)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat
    print(f"  Latency: {mean_lat:.2f} +/- {std_lat:.2f} ms ({fps:.1f} FPS)")

    # 2. Verify Output Shape
    assert depth.shape == (h_orig, w_orig), f"Expected depth shape ({h_orig}, {w_orig}), got {tuple(depth.shape)}"
    print(f"  [PASS] Depth output shape: {tuple(depth.shape)} matches input (436, 1024)")

    # 3. Verify Finite Values
    is_finite = torch.isfinite(depth).all().item()
    nan_count = torch.isnan(depth).sum().item()
    inf_count = torch.isinf(depth).sum().item()
    assert is_finite, f"Depth contains non-finite values! NaNs={nan_count}, Infs={inf_count}"
    print(f"  [PASS] All depth values are strictly finite (NaNs={nan_count}, Infs={inf_count})")

    # 4. Verify Positive Depth
    is_positive = (depth > 0.0).all().item()
    min_depth = depth.min().item()
    max_depth = depth.max().item()
    mean_depth = depth.mean().item()
    median_depth = depth.median().item()
    assert is_positive, f"Depth contains non-positive values! Min depth={min_depth}"
    assert min_depth > 0.0, f"Minimum depth is not positive: {min_depth}"
    print(f"  [PASS] All depth values are strictly positive: min={min_depth:.3f}, max={max_depth:.3f}, median={median_depth:.3f}")

    # 5. Full Backprojection and Reprojection Consistency using Official Calibrated K
    print("\n=== Test 5: Backprojection & Reprojection with Official Calibrated K ===")

    # Backproject to 3D
    points_3d = intrinsics.backproject_torch(depth)
    assert points_3d.shape == (h_orig, w_orig, 3), f"Expected 3D points shape ({h_orig}, {w_orig}, 3), got {tuple(points_3d.shape)}"

    # Check that Z in 3D points matches input depth exactly
    z_diff = torch.max(torch.abs(points_3d[..., 2] - depth)).item()
    assert z_diff < 1e-6, f"Z coordinate deviation from depth: {z_diff}"
    print(f"  [PASS] 3D points Z-plane exact match (max diff: {z_diff:.2e})")

    # Reproject to 2D
    uv_reproj, valid = intrinsics.reproject_torch(points_3d)
    assert valid.all(), "Reprojected points have invalid / non-positive depths"

    u_grid, v_grid = intrinsics.get_pixel_grid(device=depth.device)
    max_reproj_err_u = torch.max(torch.abs(uv_reproj[..., 0] - u_grid)).item()
    max_reproj_err_v = torch.max(torch.abs(uv_reproj[..., 1] - v_grid)).item()

    assert max_reproj_err_u < 5e-4, f"Reprojected u error: {max_reproj_err_u}"
    assert max_reproj_err_v < 5e-4, f"Reprojected v error: {max_reproj_err_v}"
    print(f"  [PASS] Reprojection consistency verified with official K (< 5e-4 px):")
    print(f"         Max error u: {max_reproj_err_u:.2e} px")
    print(f"         Max error v: {max_reproj_err_v:.2e} px")

    # 6. Memory footprint check
    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\nGPU Memory Footprint on RTX 5050: Allocated = {mem_allocated:.1f} MB, Reserved = {mem_reserved:.1f} MB")
        assert mem_allocated < 500, f"VRAM usage suspiciously high: {mem_allocated:.1f} MB"


def main() -> None:
    print("Running Day 7 Step 1: Depth and Camera-Intrinsics Foundation Test Suite\n")
    test_camera_intrinsics_math()
    test_sintel_cam_loader()
    test_synthetic_roundtrip_consistency()
    test_depth_estimation_on_sintel_frame()
    print("\n=======================================================")
    print("  ALL DAY 7 STEP 1 VERIFICATION CHECKS PASSED (5/5)")
    print("=======================================================")


if __name__ == "__main__":
    main()
