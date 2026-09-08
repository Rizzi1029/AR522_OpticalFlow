"""
Day 4: Dense Joint-Signal Analysis across 8 Canonical Sintel Samples.

Evaluates and preserves dense per-pixel representations of all three candidate diagnostic signals:
1. Model Disagreement: D(x, y) = ||F_FlowNetS(x, y) - F_RAFT(x, y)||_2
2. Forward-Backward Consistency: R_FB(x, y) = ||f_fwd(x, y) + f_bwd(x + f_fwd(x, y))||_2
3. Photometric Residual: R_photo(x, y) = mean_c |I1_c(x, y) - I2_c(x + f_fwd(x, y))|

Key architectural rules:
- Process ONE pair at a time on GPU; immediately transfer computed dense 2D arrays to CPU.
- Explicitly delete GPU tensors after each sample without calling torch.cuda.empty_cache().
- Retain dense 2D arrays on CPU for all 8 samples on the common evaluation mask:
  M_common = M_valid_gt and M_in_bounds_FNS and M_in_bounds_RAFT and isfinite(all signals).
- Evaluate dense pairwise correlations (Pearson r and Spearman rho).
- Perform Leave-One-Scene-Out Cross-Validation (LOSO-CV) grouped by SCENE (alley_1 clean and final
  belong to the same held-out scene fold) to test for complementary predictive information.
- Generate structured JSON summary and spatial aligned multi-panel heatmaps.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator

# Canonical 8-sample evaluation suite (all frame 1 -> 2)
DEFAULT_SAMPLES = [
    {"scene": "alley_1", "pass_name": "clean", "pair_idx": 0},
    {"scene": "alley_1", "pass_name": "final", "pair_idx": 0},
    {"scene": "shaman_3", "pass_name": "clean", "pair_idx": 0},
    {"scene": "bamboo_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "market_6", "pass_name": "clean", "pair_idx": 0},
    {"scene": "cave_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "ambush_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "ambush_4", "pass_name": "final", "pair_idx": 0},
]


def verify_disagreement_math() -> None:
    """Synthetic unit verification for model disagreement computation."""
    f1 = torch.ones((2, 10, 10), dtype=torch.float32)
    f2 = torch.ones((2, 10, 10), dtype=torch.float32)
    diff = f1 - f2
    d_ident = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)
    assert torch.allclose(d_ident, torch.zeros_like(d_ident)), "Identical flow test failed"

    f_offset = f1.clone()
    f_offset[0] += 3.0
    f_offset[1] += 4.0
    diff_off = f_offset - f1
    d_offset = torch.sqrt(diff_off[0] ** 2 + diff_off[1] ** 2)
    assert torch.allclose(d_offset, torch.full_like(d_offset, 5.0)), "Known offset (3, 4) test failed"


def compute_forward_backward_residual(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scalar forward-backward residual and geometric in-bounds mask.

    Args:
        flow_fwd: [2, H, W] forward flow tensor (u, v) in Frame 1.
        flow_bwd: [2, H, W] backward flow tensor (u, v) in Frame 2.

    Returns:
        fb_residual: [H, W] float32 scalar Euclidean residual in pixel units.
        in_bounds_mask: [H, W] boolean mask where forward coordinates remain within Frame 2.
    """
    _, H, W = flow_fwd.shape
    device = flow_fwd.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x_warp = x_grid + flow_fwd[0]
    y_warp = y_grid + flow_fwd[1]

    in_bounds_mask = (
        (x_warp >= 0.0) & (x_warp <= float(W - 1)) &
        (y_warp >= 0.0) & (y_warp <= float(H - 1))
    )

    norm_x = 2.0 * x_warp / max(W - 1, 1) - 1.0
    norm_y = 2.0 * y_warp / max(H - 1, 1) - 1.0
    sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    warped_bwd = F.grid_sample(
        flow_bwd.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]

    residual_vec = flow_fwd + warped_bwd
    fb_residual = torch.sqrt(residual_vec[0] ** 2 + residual_vec[1] ** 2)

    return fb_residual, in_bounds_mask


def verify_warping_math() -> None:
    """Synthetic unit verification for forward-backward residual math."""
    zero_fwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    zero_bwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    res_zero, in_b_zero = compute_forward_backward_residual(zero_fwd, zero_bwd)
    assert torch.allclose(res_zero, torch.zeros_like(res_zero)), "Zero-flow FB residual failed"
    assert in_b_zero.all(), "Zero-flow in-bounds check failed"

    const_fwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_fwd[0] = 2.0
    const_fwd[1] = 3.0
    const_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_bwd[0] = -2.0
    const_bwd[1] = -3.0
    res_const, in_b_const = compute_forward_backward_residual(const_fwd, const_bwd)
    interior = in_b_const[0:16, 0:17]
    assert interior.all(), "Interior in-bounds check failed"
    assert torch.allclose(res_const[0:16, 0:17], torch.zeros_like(res_const[0:16, 0:17]), atol=1e-5), (
        "Constant-flow FB residual failed"
    )

    incon_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    res_incon, _ = compute_forward_backward_residual(const_fwd, incon_bwd)
    expected_norm = np.sqrt(2.0**2 + 3.0**2)
    assert torch.allclose(res_incon[0:16, 0:17], torch.full_like(res_incon[0:16, 0:17], expected_norm), atol=1e-5), (
        "Inconsistent FB residual failed"
    )


def compute_photometric_residual(
    frame1_float: torch.Tensor,
    frame2_float: torch.Tensor,
    flow_fwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Warp Frame 2 toward Frame 1 coordinates using forward flow and compute per-pixel RGB L1 residual.

    Args:
        frame1_float: [3, H, W] float32 RGB tensor in [0.0, 1.0].
        frame2_float: [3, H, W] float32 RGB tensor in [0.0, 1.0].
        flow_fwd: [2, H, W] float32 forward flow tensor (u, v) in pixel units from Frame 1 to Frame 2.

    Returns:
        photo_residual: [H, W] float32 per-pixel mean RGB absolute difference in [0.0, 1.0].
        in_bounds_mask: [H, W] boolean mask where forward coordinates remain within Frame 2.
        warped_frame2: [3, H, W] float32 warped Frame 2 in Frame 1 coordinates.
    """
    _, H, W = frame1_float.shape
    device = frame1_float.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x_warp = x_grid + flow_fwd[0]
    y_warp = y_grid + flow_fwd[1]

    in_bounds_mask = (
        (x_warp >= 0.0) & (x_warp <= float(W - 1)) &
        (y_warp >= 0.0) & (y_warp <= float(H - 1))
    )

    norm_x = 2.0 * x_warp / max(W - 1, 1) - 1.0
    norm_y = 2.0 * y_warp / max(H - 1, 1) - 1.0
    sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    warped_frame2 = F.grid_sample(
        frame2_float.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]

    photo_residual = torch.mean(torch.abs(frame1_float - warped_frame2), dim=0)

    return photo_residual, in_bounds_mask, warped_frame2


def verify_photometric_math() -> None:
    """Deterministic synthetic sanity checks for image warping and photometric residual math."""
    H, W = 32, 32
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    I1 = (0.01 * x_coords + 0.02 * y_coords).unsqueeze(0).repeat(3, 1, 1)

    res_zero, in_b_zero, _ = compute_photometric_residual(
        I1, I1, torch.zeros((2, H, W), dtype=torch.float32)
    )
    assert torch.allclose(res_zero, torch.zeros_like(res_zero), atol=1e-5), "Zero-motion photometric failed"
    assert in_b_zero.all(), "Zero-motion in-bounds failed"

    u_val, v_val = 2.0, 3.0
    I2 = (0.01 * (x_coords - u_val) + 0.02 * (y_coords - v_val)).unsqueeze(0).repeat(3, 1, 1)

    flow_correct = torch.zeros((2, H, W), dtype=torch.float32)
    flow_correct[0] = u_val
    flow_correct[1] = v_val
    res_correct, in_b_correct, _ = compute_photometric_residual(I1, I2, flow_correct)
    interior_mask = in_b_correct.clone()
    correct_mean = float(res_correct[interior_mask].mean().item())
    assert correct_mean < 1e-4, f"Correct flow photometric mean {correct_mean} >= 1e-4"

    flow_wrong = torch.zeros((2, H, W), dtype=torch.float32)
    res_wrong, _, _ = compute_photometric_residual(I1, I2, flow_wrong)
    wrong_mean = float(res_wrong[interior_mask].mean().item())
    assert wrong_mean > 100 * correct_mean, "Wrong flow contrast check failed"
    assert wrong_mean > 0.01, "Wrong flow absolute check failed"


def verify_all_math() -> None:
    """Execute all unit verification tests."""
    verify_disagreement_math()
    verify_warping_math()
    verify_photometric_math()


def fit_predict_ols(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fit ordinary least squares regression with bias intercept and predict."""
    X_tr = np.hstack([np.ones((X_train.shape[0], 1), dtype=np.float32), X_train])
    X_te = np.hstack([np.ones((X_test.shape[0], 1), dtype=np.float32), X_test])
    beta, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    return X_te @ beta


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute first-order partial Spearman correlation rho(x, y | z)."""
    r_xy, _ = scipy.stats.spearmanr(x, y)
    r_xz, _ = scipy.stats.spearmanr(x, z)
    r_yz, _ = scipy.stats.spearmanr(y, z)
    denom = np.sqrt(max((1.0 - r_xz**2) * (1.0 - r_yz**2), 1e-12))
    return float((r_xy - r_xz * r_yz) / denom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Dense joint-signal diagnostic analysis across 8 Sintel pairs."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Sintel",
        help="Path to Sintel dataset root (default: data/Sintel)",
    )
    parser.add_argument(
        "--flownet_checkpoint",
        type=str,
        default="checkpoints/flownets_EPE1.951.pth.tar",
        help="Path to FlowNetS checkpoint weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save diagnostic artifacts (default: outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=False,
        help="Save dense per-pixel arrays for all 8 pairs to compressed NPZ archive",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        default=False,
        help="Run mathematical synthetic sanity checks and exit immediately",
    )
    return parser.parse_args()


def main() -> None:
    # 0. Sanity-check all mathematical definitions before inference
    verify_all_math()

    args = parse_args()
    if args.verify_only:
        print("All synthetic mathematical verifications passed successfully.")
        return

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 DENSE JOINT-SIGNAL DIAGNOSTIC ANALYSIS (8 SAMPLES)    ")
    print("==================================================================")
    print(f"Device:           {device}")
    print(f"Samples:          {len(DEFAULT_SAMPLES)} pairs (frame 1 -> 2)")
    print(f"Output Directory: {output_dir}")
    print("==================================================================\n")

    # 1. Instantiate models once
    print("Loading FlowNetS adapter...")
    flownet_model = build_flow_estimator(
        model_name="flownets",
        checkpoint_path=args.flownet_checkpoint,
        device=device,
    )

    print("Loading RAFT adapter...")
    raft_model = build_flow_estimator(
        model_name="raft",
        device=device,
    )
    print("Models loaded successfully.\n")

    # Containers for dense per-sample arrays stored on CPU host RAM
    dense_samples: List[Dict[str, Any]] = []

    # 2. Sequential per-pair processing loop (GPU memory isolated)
    for i, spec in enumerate(DEFAULT_SAMPLES):
        scene = spec["scene"]
        pass_name = spec["pass_name"]
        pair_idx = spec["pair_idx"]
        sample_label = f"{scene} ({pass_name})"

        print(f"[{i + 1}/{len(DEFAULT_SAMPLES)}] Processing dense arrays for: {sample_label} ... ", end="", flush=True)

        dataset = SintelDataset(
            root=args.data_root,
            split="training",
            pass_name=pass_name,
            scenes=[scene],
        )
        frame1, frame2, flow_gt, valid_mask, meta = dataset[pair_idx]
        orig_h, orig_w = frame1.shape[1], frame1.shape[2]

        with torch.inference_mode():
            # FlowNetS forward + backward inference
            flow_fwd_fns = flownet_model(frame1, frame2)[0]
            flow_bwd_fns = flownet_model(frame2, frame1)[0]

            # RAFT forward + backward inference
            flow_fwd_raft = raft_model(frame1, frame2)[0]
            flow_bwd_raft = raft_model(frame2, frame1)[0]

            # Signal 1: Model Disagreement D(x, y) = ||f_FNS - f_RAFT||_2
            diff_models = flow_fwd_fns - flow_fwd_raft
            disag_t = torch.sqrt(diff_models[0] ** 2 + diff_models[1] ** 2)
            del diff_models

            # Signal 2: Forward-Backward Consistency R_FB
            res_fb_fns_t, in_b_fns_t = compute_forward_backward_residual(flow_fwd_fns, flow_bwd_fns)
            res_fb_raft_t, in_b_raft_t = compute_forward_backward_residual(flow_fwd_raft, flow_bwd_raft)

            # Signal 3: Photometric Consistency R_photo
            frame1_float = frame1.to(device=device, dtype=torch.float32) / 255.0
            frame2_float = frame2.to(device=device, dtype=torch.float32) / 255.0
            res_photo_fns_t, in_b_photo_fns_t, _ = compute_photometric_residual(frame1_float, frame2_float, flow_fwd_fns)
            res_photo_raft_t, in_b_photo_raft_t, _ = compute_photometric_residual(frame1_float, frame2_float, flow_fwd_raft)

            # Actual Ground-Truth Errors (EPE)
            flow_gt_dev = flow_gt.to(device)
            diff_fns = flow_fwd_fns - flow_gt_dev
            epe_fns_t = torch.sqrt(diff_fns[0] ** 2 + diff_fns[1] ** 2)
            del diff_fns

            diff_raft = flow_fwd_raft - flow_gt_dev
            epe_raft_t = torch.sqrt(diff_raft[0] ** 2 + diff_raft[1] ** 2)
            del diff_raft

        # Immediately transfer dense arrays to CPU host NumPy
        disag_np = disag_t.cpu().numpy()
        res_fb_fns_np = res_fb_fns_t.cpu().numpy()
        res_fb_raft_np = res_fb_raft_t.cpu().numpy()
        res_photo_fns_np = res_photo_fns_t.cpu().numpy()
        res_photo_raft_np = res_photo_raft_t.cpu().numpy()
        epe_fns_np = epe_fns_t.cpu().numpy()
        epe_raft_np = epe_raft_t.cpu().numpy()

        valid_gt_np = valid_mask.cpu().numpy()
        in_b_fns_np = in_b_fns_t.cpu().numpy()
        in_b_raft_np = in_b_raft_t.cpu().numpy()
        in_b_ph_fns_np = in_b_photo_fns_t.cpu().numpy()
        in_b_ph_raft_np = in_b_photo_raft_t.cpu().numpy()

        frame1_rgb = frame1.permute(1, 2, 0).cpu().numpy()

        # Strict GPU memory hygiene: delete all GPU tensors
        del flow_fwd_fns, flow_bwd_fns, flow_fwd_raft, flow_bwd_raft, flow_gt_dev
        del disag_t, res_fb_fns_t, res_fb_raft_t, in_b_fns_t, in_b_raft_t
        del res_photo_fns_t, res_photo_raft_t, in_b_photo_fns_t, in_b_photo_raft_t
        del epe_fns_t, epe_raft_t, frame1_float, frame2_float

        # Construct common evaluation mask for this pair
        mask_common = (
            valid_gt_np &
            in_b_fns_np & in_b_raft_np &
            in_b_ph_fns_np & in_b_ph_raft_np &
            np.isfinite(disag_np) &
            np.isfinite(res_fb_fns_np) & np.isfinite(res_fb_raft_np) &
            np.isfinite(res_photo_fns_np) & np.isfinite(res_photo_raft_np) &
            np.isfinite(epe_fns_np) & np.isfinite(epe_raft_np)
        )

        n_common = int(np.sum(mask_common))
        print(f"common valid pixels: {n_common:,} ({n_common / (orig_h * orig_w) * 100:.1f}%)")

        dense_samples.append({
            "sample_index": i,
            "scene": scene,
            "pass_name": pass_name,
            "pair_idx": pair_idx,
            "frame_idx": meta["frame_idx"],
            "dimensions": {"height": orig_h, "width": orig_w},
            "mask_common": mask_common,
            "disagreement": disag_np,
            "fb_fns": res_fb_fns_np,
            "fb_raft": res_fb_raft_np,
            "photo_fns": res_photo_fns_np,
            "photo_raft": res_photo_raft_np,
            "epe_fns": epe_fns_np,
            "epe_raft": epe_raft_np,
            "frame1_rgb": frame1_rgb,
        })

    # 3. Concatenate dense arrays across common evaluation pixels for pooled analysis
    print("\nAssembling pooled dense evaluation vectors...")
    pooled_disag_list: List[np.ndarray] = []
    pooled_fb_fns_list: List[np.ndarray] = []
    pooled_fb_raft_list: List[np.ndarray] = []
    pooled_ph_fns_list: List[np.ndarray] = []
    pooled_ph_raft_list: List[np.ndarray] = []
    pooled_epe_fns_list: List[np.ndarray] = []
    pooled_epe_raft_list: List[np.ndarray] = []
    pooled_scene_id_list: List[np.ndarray] = []

    for s in dense_samples:
        m = s["mask_common"]
        pooled_disag_list.append(s["disagreement"][m])
        pooled_fb_fns_list.append(s["fb_fns"][m])
        pooled_fb_raft_list.append(s["fb_raft"][m])
        pooled_ph_fns_list.append(s["photo_fns"][m])
        pooled_ph_raft_list.append(s["photo_raft"][m])
        pooled_epe_fns_list.append(s["epe_fns"][m])
        pooled_epe_raft_list.append(s["epe_raft"][m])
        pooled_scene_id_list.append(np.full(np.sum(m), s["scene"], dtype=object))

    vec_disag = np.concatenate(pooled_disag_list)
    vec_fb_fns = np.concatenate(pooled_fb_fns_list)
    vec_fb_raft = np.concatenate(pooled_fb_raft_list)
    vec_ph_fns = np.concatenate(pooled_ph_fns_list)
    vec_ph_raft = np.concatenate(pooled_ph_raft_list)
    vec_epe_fns = np.concatenate(pooled_epe_fns_list)
    vec_epe_raft = np.concatenate(pooled_epe_raft_list)
    vec_scene = np.concatenate(pooled_scene_id_list)

    total_eval_pixels = int(vec_disag.size)
    print(f"Total pooled common evaluation pixels: {total_eval_pixels:,}")

    # 4. Dense Pairwise Correlations on Common Mask
    def get_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        r_p, _ = scipy.stats.pearsonr(x, y)
        r_s, _ = scipy.stats.spearmanr(x, y)
        return {"pearson_r": float(r_p), "spearman_rho": float(r_s)}

    print("\nComputing dense pairwise correlations...")
    dense_correlations = {
        "flownet_s": {
            "disagreement_vs_epe": get_corr(vec_disag, vec_epe_fns),
            "fb_residual_vs_epe": get_corr(vec_fb_fns, vec_epe_fns),
            "photo_residual_vs_epe": get_corr(vec_ph_fns, vec_epe_fns),
            "disagreement_vs_fb_residual": get_corr(vec_disag, vec_fb_fns),
            "disagreement_vs_photo_residual": get_corr(vec_disag, vec_ph_fns),
            "fb_residual_vs_photo_residual": get_corr(vec_fb_fns, vec_ph_fns),
        },
        "raft": {
            "disagreement_vs_epe": get_corr(vec_disag, vec_epe_raft),
            "fb_residual_vs_epe": get_corr(vec_fb_raft, vec_epe_raft),
            "photo_residual_vs_epe": get_corr(vec_ph_raft, vec_epe_raft),
            "disagreement_vs_fb_residual": get_corr(vec_disag, vec_fb_raft),
            "disagreement_vs_photo_residual": get_corr(vec_disag, vec_ph_raft),
            "fb_residual_vs_photo_residual": get_corr(vec_fb_raft, vec_ph_raft),
        },
    }

    # Partial Spearman Rank Correlations
    partial_correlations = {
        "flownet_s": {
            "partial_rho_epe_photo_given_fb": compute_partial_spearman(vec_epe_fns, vec_ph_fns, vec_fb_fns),
            "partial_rho_epe_disag_given_fb": compute_partial_spearman(vec_epe_fns, vec_disag, vec_fb_fns),
        },
        "raft": {
            "partial_rho_epe_photo_given_fb": compute_partial_spearman(vec_epe_raft, vec_ph_raft, vec_fb_raft),
            "partial_rho_epe_disag_given_fb": compute_partial_spearman(vec_epe_raft, vec_disag, vec_fb_raft),
        },
    }

    # 5. Complementarity Diagnostic: Leave-One-Scene-Out Cross-Validation (LOSO-CV)
    # Note: Grouped by SCENE so alley_1 clean and final are evaluated together out-of-fold
    unique_scenes = sorted(list(set(vec_scene)))
    print(f"\nRunning Leave-One-Scene-Out Cross-Validation across {len(unique_scenes)} unique scenes: {unique_scenes}")

    def run_loso_cv(feature_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        model_names = list(feature_dict.keys())
        oof_predictions: Dict[str, np.ndarray] = {m: np.zeros_like(y_true) for m in model_names}

        for test_scene in unique_scenes:
            test_mask = (vec_scene == test_scene)
            train_mask = ~test_mask

            y_train = y_true[train_mask]

            for m_name, X_feat in feature_dict.items():
                X_train = X_feat[train_mask]
                X_test = X_feat[test_mask]
                oof_predictions[m_name][test_mask] = fit_predict_ols(X_train, y_train, X_test)

        cv_results: Dict[str, Dict[str, float]] = {}
        for m_name in model_names:
            y_pred = oof_predictions[m_name]
            r2_val = compute_r2(y_true, y_pred)
            r_val, _ = scipy.stats.pearsonr(y_true, y_pred)
            rho_val, _ = scipy.stats.spearmanr(y_true, y_pred)
            cv_results[m_name] = {
                "out_of_fold_r2": float(r2_val),
                "out_of_fold_pearson_r": float(r_val),
                "out_of_fold_spearman_rho": float(rho_val),
            }
        return cv_results

    # Build feature sets for FlowNetS
    feat_fns = {
        "disagreement_only": vec_disag[:, None],
        "fb_only": vec_fb_fns[:, None],
        "photo_only": vec_ph_fns[:, None],
        "disag_plus_fb": np.column_stack([vec_disag, vec_fb_fns]),
        "fb_plus_photo": np.column_stack([vec_fb_fns, vec_ph_fns]),
        "disag_plus_photo": np.column_stack([vec_disag, vec_ph_fns]),
        "all_three_signals": np.column_stack([vec_disag, vec_fb_fns, vec_ph_fns]),
    }
    cv_flownet = run_loso_cv(feat_fns, vec_epe_fns)

    # Build feature sets for RAFT
    feat_raft = {
        "disagreement_only": vec_disag[:, None],
        "fb_only": vec_fb_raft[:, None],
        "photo_only": vec_ph_raft[:, None],
        "disag_plus_fb": np.column_stack([vec_disag, vec_fb_raft]),
        "fb_plus_photo": np.column_stack([vec_fb_raft, vec_ph_raft]),
        "disag_plus_photo": np.column_stack([vec_disag, vec_ph_raft]),
        "all_three_signals": np.column_stack([vec_disag, vec_fb_raft, vec_ph_raft]),
    }
    cv_raft = run_loso_cv(feat_raft, vec_epe_raft)

    # 6. Quantile-Based Conditional Error Stratification (Monotonicity Check)
    def compute_quantile_bins(signal_vec: np.ndarray, epe_vec: np.ndarray, num_bins: int = 5) -> List[Dict[str, float]]:
        pct_edges = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(signal_vec, pct_edges)
        bins_summary: List[Dict[str, float]] = []

        for b in range(num_bins):
            if b == num_bins - 1:
                in_bin = (signal_vec >= bin_edges[b]) & (signal_vec <= bin_edges[b + 1])
            else:
                in_bin = (signal_vec >= bin_edges[b]) & (signal_vec < bin_edges[b + 1])

            if np.sum(in_bin) > 0:
                bin_epe = epe_vec[in_bin]
                bins_summary.append({
                    "bin_index": b,
                    "bin_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
                    "pixel_count": int(np.sum(in_bin)),
                    "mean_epe": float(np.mean(bin_epe)),
                    "median_epe": float(np.median(bin_epe)),
                    "iqr_epe": float(np.percentile(bin_epe, 75) - np.percentile(bin_epe, 25)),
                })
        return bins_summary

    quantile_analysis = {
        "flownet_s": {
            "disagreement_quantiles": compute_quantile_bins(vec_disag, vec_epe_fns),
            "fb_quantiles": compute_quantile_bins(vec_fb_fns, vec_epe_fns),
            "photo_quantiles": compute_quantile_bins(vec_ph_fns, vec_epe_fns),
        },
        "raft": {
            "disagreement_quantiles": compute_quantile_bins(vec_disag, vec_epe_raft),
            "fb_quantiles": compute_quantile_bins(vec_fb_raft, vec_epe_raft),
            "photo_quantiles": compute_quantile_bins(vec_ph_raft, vec_epe_raft),
        },
    }

    # 7. Generate Aligned Multi-Panel Spatial Diagnostic Visualization
    # Select 2 representative scenes: alley_1 clean (baseline, idx 0) and ambush_2 clean (high dynamic, idx 6)
    rep_indices = [0, 6]
    fig, axes = plt.subplots(len(rep_indices), 5, figsize=(24, 8))
    fig.suptitle(
        "Day 4 Dense Joint-Signal Spatial Diagnostic: Ground-Truth Error vs. Candidate Indicators",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, s_idx in enumerate(rep_indices):
        s_data = dense_samples[s_idx]
        mask = s_data["mask_common"]
        scene_name = f"{s_data['scene']} ({s_data['pass_name']})"

        # Panel 0: Input Frame 1 RGB
        axes[row_idx, 0].imshow(s_data["frame1_rgb"])
        axes[row_idx, 0].set_title(f"Input Frame 1\n{scene_name}", fontsize=11)
        axes[row_idx, 0].axis("off")

        # Panel 1: RAFT Actual EPE
        epe_vis = s_data["epe_raft"].copy()
        epe_vis[~mask] = np.nan
        im1 = axes[row_idx, 1].imshow(epe_vis, cmap="magma", vmin=0.0, vmax=max(float(np.nanpercentile(epe_vis, 99)), 1.0))
        axes[row_idx, 1].set_title(f"RAFT GT EPE (px)\nMean: {np.nanmean(epe_vis):.2f} px", fontsize=11)
        axes[row_idx, 1].axis("off")
        fig.colorbar(im1, ax=axes[row_idx, 1], orientation="horizontal", fraction=0.046, pad=0.08)

        # Panel 2: Model Disagreement D
        disag_vis = s_data["disagreement"].copy()
        disag_vis[~mask] = np.nan
        im2 = axes[row_idx, 2].imshow(disag_vis, cmap="magma", vmin=0.0, vmax=max(float(np.nanpercentile(disag_vis, 99)), 1.0))
        axes[row_idx, 2].set_title(f"Model Disagreement D (px)\nMean: {np.nanmean(disag_vis):.2f} px", fontsize=11)
        axes[row_idx, 2].axis("off")
        fig.colorbar(im2, ax=axes[row_idx, 2], orientation="horizontal", fraction=0.046, pad=0.08)

        # Panel 3: RAFT FB Residual
        fb_vis = s_data["fb_raft"].copy()
        fb_vis[~mask] = np.nan
        im3 = axes[row_idx, 3].imshow(fb_vis, cmap="magma", vmin=0.0, vmax=max(float(np.nanpercentile(fb_vis, 99)), 0.5))
        axes[row_idx, 3].set_title(f"RAFT FB Residual (px)\nMean: {np.nanmean(fb_vis):.2f} px", fontsize=11)
        axes[row_idx, 3].axis("off")
        fig.colorbar(im3, ax=axes[row_idx, 3], orientation="horizontal", fraction=0.046, pad=0.08)

        # Panel 4: RAFT Photometric Residual
        ph_vis = s_data["photo_raft"].copy()
        ph_vis[~mask] = np.nan
        im4 = axes[row_idx, 4].imshow(ph_vis, cmap="magma", vmin=0.0, vmax=max(float(np.nanpercentile(ph_vis, 99)), 0.05))
        axes[row_idx, 4].set_title(f"RAFT Photometric Residual\nMean: {np.nanmean(ph_vis):.4f}", fontsize=11)
        axes[row_idx, 4].axis("off")
        fig.colorbar(im4, ax=axes[row_idx, 4], orientation="horizontal", fraction=0.046, pad=0.08)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    vis_path = output_dir / "day4_dense_joint_analysis.png"
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic visualization saved to: {vis_path}")

    # 8. Save structured JSON summary
    summary_results = {
        "metadata": {
            "experiment": "day4_dense_joint_signal_analysis",
            "samples": DEFAULT_SAMPLES,
            "unique_scenes": unique_scenes,
            "num_samples": len(DEFAULT_SAMPLES),
            "pooled_common_pixels": total_eval_pixels,
            "device": str(device),
        },
        "synthetic_checks": "passed",
        "dense_pairwise_correlations": dense_correlations,
        "partial_spearman_correlations": partial_correlations,
        "loso_cross_validation": {
            "flownet_s": cv_flownet,
            "raft": cv_raft,
        },
        "quantile_monotonicity_analysis": quantile_analysis,
    }

    json_path = output_dir / "day4_dense_joint_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Summary JSON saved to: {json_path}")

    # Optional NPZ dense preservation
    if args.save_npz:
        npz_path = output_dir / "day4_dense_joint_analysis.npz"
        print(f"Saving compressed dense arrays for all {len(dense_samples)} pairs to {npz_path} ... ", end="", flush=True)
        np.savez_compressed(
            npz_path,
            disagreement=np.stack([s["disagreement"] for s in dense_samples]),
            fb_fns=np.stack([s["fb_fns"] for s in dense_samples]),
            fb_raft=np.stack([s["fb_raft"] for s in dense_samples]),
            photo_fns=np.stack([s["photo_fns"] for s in dense_samples]),
            photo_raft=np.stack([s["photo_raft"] for s in dense_samples]),
            epe_fns=np.stack([s["epe_fns"] for s in dense_samples]),
            epe_raft=np.stack([s["epe_raft"] for s in dense_samples]),
            mask_common=np.stack([s["mask_common"] for s in dense_samples]),
        )
        print("done.")

    print("\n==================================================================")
    print("                CROSS-VALIDATION DIAGNOSTIC SUMMARY               ")
    print("==================================================================")
    print("RAFT LOSO-CV Out-of-Fold Spearman Rank Correlations:")
    for m_name, m_res in cv_raft.items():
        print(f"  {m_name:<20}: rho = {m_res['out_of_fold_spearman_rho']:.4f} (R2 = {m_res['out_of_fold_r2']:.4f})")
    print("\nFlowNetS LOSO-CV Out-of-Fold Spearman Rank Correlations:")
    for m_name, m_res in cv_flownet.items():
        print(f"  {m_name:<20}: rho = {m_res['out_of_fold_spearman_rho']:.4f} (R2 = {m_res['out_of_fold_r2']:.4f})")
    print("==================================================================\n")
    print("Dense joint-signal analysis completed successfully.")


if __name__ == "__main__":
    main()
