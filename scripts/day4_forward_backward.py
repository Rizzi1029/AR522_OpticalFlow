"""
Day 4: Forward-Backward Optical Flow Consistency Diagnostic.

Evaluates bidirectional optical flow for FlowNetS and RAFT across MPI Sintel frame pairs:
1. Forward pass: Frame 1 -> Frame 2  (f_fwd)
2. Backward pass: Frame 2 -> Frame 1 (f_bwd)
3. Computes geometric in-bounds mask where forward-warped coordinates remain in Frame 2.
4. Samples backward flow at forward-arrival coordinates using bilinear grid_sample.
5. Computes scalar forward-backward residual:
   R_FB(x, y) = ||f_fwd(x, y) + f_bwd(x + f_fwd(x, y))||_2
6. Evaluates exploratory correlation of R_FB against each model's own ground-truth EPE:
   E_FNS(x, y)  = ||f_fwd,FNS(x, y)  - f_GT(x, y)||_2
   E_RAFT(x, y) = ||f_fwd,RAFT(x, y) - f_GT(x, y)||_2
7. Evaluates correlation between model disagreement D(x, y) and FB residuals.
8. Strictly manages memory via explicit tensor deletion (NO torch.cuda.empty_cache()).
9. Generates structured JSON summary and 6-panel diagnostic visualization.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
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


def verify_warping_math() -> None:
    """Synthetic unit sanity check for forward-backward warping and residual computation."""
    # Test 1: Zero motion -> residual identically 0.0 everywhere
    zero_fwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    zero_bwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    res_zero, in_b_zero = compute_forward_backward_residual(zero_fwd, zero_bwd)
    assert torch.allclose(res_zero, torch.zeros_like(res_zero)), "Zero-flow FB residual failed"
    assert in_b_zero.all(), "Zero-flow in-bounds check failed"

    # Test 2: Invertible constant translation (+2, +3) and (-2, -3)
    const_fwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_fwd[0] = 2.0
    const_fwd[1] = 3.0
    const_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_bwd[0] = -2.0
    const_bwd[1] = -3.0
    res_const, in_b_const = compute_forward_backward_residual(const_fwd, const_bwd)
    # Check interior coordinates that remain strictly in-bounds
    interior = in_b_const[0:16, 0:17]
    assert interior.all(), "Interior in-bounds check failed"
    assert torch.allclose(res_const[0:16, 0:17], torch.zeros_like(res_const[0:16, 0:17]), atol=1e-5), (
        "Constant-flow FB residual failed"
    )

    # Test 3: Known inconsistent flow (forward=(+2, +3), backward=(0, 0)) -> residual exactly sqrt(13) ≈ 3.606 px
    incon_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    res_incon, _ = compute_forward_backward_residual(const_fwd, incon_bwd)
    expected_norm = np.sqrt(2.0**2 + 3.0**2)
    assert torch.allclose(res_incon[0:16, 0:17], torch.full_like(res_incon[0:16, 0:17], expected_norm), atol=1e-5), (
        "Inconsistent FB residual failed"
    )


def compute_forward_backward_residual(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scalar forward-backward residual and in-bounds mask.

    Args:
        flow_fwd: [2, H, W] forward flow tensor from Frame 1 to Frame 2.
        flow_bwd: [2, H, W] backward flow tensor from Frame 2 to Frame 1.

    Returns:
        fb_residual: [H, W] float32 scalar Euclidean residual in pixel units.
        in_bounds_mask: [H, W] boolean mask where forward-warped coordinates
            remain strictly within Frame 2 dimensions [0, W - 1] x [0, H - 1].
    """
    _, H, W = flow_fwd.shape
    device = flow_fwd.device

    # 1. Base grid in Frame 1
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # 2. Forward arrival coordinates in Frame 2
    x_warp = x_grid + flow_fwd[0]
    y_warp = y_grid + flow_fwd[1]

    # 3. Geometric in-bounds mask
    in_bounds_mask = (
        (x_warp >= 0.0) & (x_warp <= float(W - 1)) &
        (y_warp >= 0.0) & (y_warp <= float(H - 1))
    )

    # 4. Normalized coordinates for F.grid_sample in [-1.0, 1.0] (align_corners=True)
    norm_x = 2.0 * x_warp / max(W - 1, 1) - 1.0
    norm_y = 2.0 * y_warp / max(H - 1, 1) - 1.0
    sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # 5. Bilinear sampling of backward flow at forward arrival locations
    warped_bwd = F.grid_sample(
        flow_bwd.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]  # [2, H, W]

    # 6. Scalar Euclidean residual: ||f_fwd(x) + f_bwd(x + f_fwd(x))||_2
    residual_vec = flow_fwd + warped_bwd
    fb_residual = torch.sqrt(residual_vec[0] ** 2 + residual_vec[1] ** 2)

    return fb_residual, in_bounds_mask


def compute_distribution_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute summary distribution statistics for a 1D numerical array."""
    flat = data.flatten()
    if flat.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "fraction_gt_1px": 0.0,
            "fraction_gt_3px": 0.0,
            "fraction_gt_5px": 0.0,
        }

    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "fraction_gt_1px": float(np.mean(flat > 1.0)),
        "fraction_gt_3px": float(np.mean(flat > 3.0)),
        "fraction_gt_5px": float(np.mean(flat > 5.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Forward-Backward Optical Flow Consistency Diagnostic."
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
    return parser.parse_args()


def main() -> None:
    # 0. Sanity-check warping math before inference
    verify_warping_math()

    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 FORWARD-BACKWARD FLOW CONSISTENCY EXPERIMENT          ")
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

    per_sample_results: List[Dict[str, Any]] = []

    # Pooled buffers for global analysis
    pooled_res_fns: List[np.ndarray] = []
    pooled_err_fns: List[np.ndarray] = []
    pooled_res_raft: List[np.ndarray] = []
    pooled_err_raft: List[np.ndarray] = []
    pooled_disag: List[np.ndarray] = []

    # Subsampled buffers for responsive visualization
    scatter_res_fns: List[np.ndarray] = []
    scatter_err_fns: List[np.ndarray] = []
    scatter_res_raft: List[np.ndarray] = []
    scatter_err_raft: List[np.ndarray] = []

    for i, spec in enumerate(DEFAULT_SAMPLES):
        scene = spec["scene"]
        pass_name = spec["pass_name"]
        pair_idx = spec["pair_idx"]
        sample_label = f"{scene} ({pass_name})"

        print(f"[{i + 1}/{len(DEFAULT_SAMPLES)}] Evaluating: {sample_label} ... ", end="", flush=True)

        # Load Sintel frame pair
        dataset = SintelDataset(
            root=args.data_root,
            split="training",
            pass_name=pass_name,
            scenes=[scene],
        )
        frame1, frame2, flow_gt, valid_mask, meta = dataset[pair_idx]
        orig_h, orig_w = frame1.shape[1], frame1.shape[2]
        total_pixels = orig_h * orig_w
        valid_gt_count = int(valid_mask.sum().item())

        with torch.inference_mode():
            # Forward inference: Frame 1 -> Frame 2
            flow_fwd_fns = flownet_model(frame1, frame2)[0]
            flow_fwd_raft = raft_model(frame1, frame2)[0]

            # Backward inference: Frame 2 -> Frame 1
            flow_bwd_fns = flownet_model(frame2, frame1)[0]
            flow_bwd_raft = raft_model(frame2, frame1)[0]

            # Forward-Backward Residual and In-Bounds Mask
            res_fns_t, in_b_fns_t = compute_forward_backward_residual(flow_fwd_fns, flow_bwd_fns)
            res_raft_t, in_b_raft_t = compute_forward_backward_residual(flow_fwd_raft, flow_bwd_raft)

            # Model Disagreement: ||f_fwd,FNS - f_fwd,RAFT||_2
            diff_models = flow_fwd_fns - flow_fwd_raft
            disag_t = torch.sqrt(diff_models[0] ** 2 + diff_models[1] ** 2)
            del diff_models

            # Actual Forward Errors against Ground Truth
            flow_gt_dev = flow_gt.to(device)
            diff_fns = flow_fwd_fns - flow_gt_dev
            err_fns_t = torch.sqrt(diff_fns[0] ** 2 + diff_fns[1] ** 2)
            del diff_fns

            diff_raft = flow_fwd_raft - flow_gt_dev
            err_raft_t = torch.sqrt(diff_raft[0] ** 2 + diff_raft[1] ** 2)
            del diff_raft

        # Move to CPU NumPy
        res_fns_np = res_fns_t.cpu().numpy()
        in_b_fns_np = in_b_fns_t.cpu().numpy()
        res_raft_np = res_raft_t.cpu().numpy()
        in_b_raft_np = in_b_raft_t.cpu().numpy()
        disag_np = disag_t.cpu().numpy()
        err_fns_np = err_fns_t.cpu().numpy()
        err_raft_np = err_raft_t.cpu().numpy()
        valid_gt_np = valid_mask.cpu().numpy()

        # Strict memory hygiene: explicit deletion of GPU tensors (NO empty_cache)
        del flow_fwd_fns, flow_bwd_fns, flow_fwd_raft, flow_bwd_raft, flow_gt_dev
        del res_fns_t, in_b_fns_t, res_raft_t, in_b_raft_t
        del disag_t, err_fns_t, err_raft_t

        # Define evaluation masks
        eval_mask_fns = valid_gt_np & in_b_fns_np & np.isfinite(res_fns_np) & np.isfinite(err_fns_np)
        eval_mask_raft = valid_gt_np & in_b_raft_np & np.isfinite(res_raft_np) & np.isfinite(err_raft_np)
        joint_eval_mask = eval_mask_fns & eval_mask_raft

        # In-bounds metrics
        in_b_fns_count = int(np.sum(in_b_fns_np))
        in_b_raft_count = int(np.sum(in_b_raft_np))
        eval_fns_count = int(np.sum(eval_mask_fns))
        eval_raft_count = int(np.sum(eval_mask_raft))

        # Distribution statistics over in-bounds valid evaluation pixels
        stats_res_fns = compute_distribution_stats(res_fns_np[eval_mask_fns])
        stats_res_raft = compute_distribution_stats(res_raft_np[eval_mask_raft])

        # Correlation between FB residual and Model's OWN Ground-Truth Error
        r_fns_own_p, _ = scipy.stats.pearsonr(res_fns_np[eval_mask_fns], err_fns_np[eval_mask_fns])
        r_fns_own_s, _ = scipy.stats.spearmanr(res_fns_np[eval_mask_fns], err_fns_np[eval_mask_fns])

        r_raft_own_p, _ = scipy.stats.pearsonr(res_raft_np[eval_mask_raft], err_raft_np[eval_mask_raft])
        r_raft_own_s, _ = scipy.stats.spearmanr(res_raft_np[eval_mask_raft], err_raft_np[eval_mask_raft])

        # Correlation between Model Disagreement and FB Residuals (jointly valid pixels)
        r_disag_fns_p, _ = scipy.stats.pearsonr(disag_np[joint_eval_mask], res_fns_np[joint_eval_mask])
        r_disag_fns_s, _ = scipy.stats.spearmanr(disag_np[joint_eval_mask], res_fns_np[joint_eval_mask])

        r_disag_raft_p, _ = scipy.stats.pearsonr(disag_np[joint_eval_mask], res_raft_np[joint_eval_mask])
        r_disag_raft_s, _ = scipy.stats.spearmanr(disag_np[joint_eval_mask], res_raft_np[joint_eval_mask])

        # Cross-model FB residual correlation
        r_cross_fb_p, _ = scipy.stats.pearsonr(res_fns_np[joint_eval_mask], res_raft_np[joint_eval_mask])
        r_cross_fb_s, _ = scipy.stats.spearmanr(res_fns_np[joint_eval_mask], res_raft_np[joint_eval_mask])

        mean_epe_fns = float(np.mean(err_fns_np[valid_gt_np]))
        mean_epe_raft = float(np.mean(err_raft_np[valid_gt_np]))

        print(
            f"FNS R_FB={stats_res_fns['mean']:.3f} px (r={r_fns_own_p:.3f}) | "
            f"RAFT R_FB={stats_res_raft['mean']:.3f} px (r={r_raft_own_p:.3f}) | "
            f"In-bounds FNS={in_b_fns_count / total_pixels * 100:.1f}%, RAFT={in_b_raft_count / total_pixels * 100:.1f}%"
        )

        record = {
            "sample_index": i,
            "scene": scene,
            "pass_name": pass_name,
            "frame_idx": meta["frame_idx"],
            "dimensions": {"height": orig_h, "width": orig_w},
            "pixel_counts": {
                "total": total_pixels,
                "valid_gt": valid_gt_count,
                "in_bounds_flownet": in_b_fns_count,
                "in_bounds_raft": in_b_raft_count,
                "eval_flownet": eval_fns_count,
                "eval_raft": eval_raft_count,
                "joint_eval": int(np.sum(joint_eval_mask)),
            },
            "flownet_s": {
                "mean_epe_valid_px": mean_epe_fns,
                "fb_residual_stats_eval": stats_res_fns,
                "fb_vs_own_error_correlation": {
                    "pearson_r": float(r_fns_own_p),
                    "spearman_rho": float(r_fns_own_s),
                },
            },
            "raft": {
                "mean_epe_valid_px": mean_epe_raft,
                "fb_residual_stats_eval": stats_res_raft,
                "fb_vs_own_error_correlation": {
                    "pearson_r": float(r_raft_own_p),
                    "spearman_rho": float(r_raft_own_s),
                },
            },
            "disagreement_vs_fb_correlations": {
                "disagreement_vs_flownet_fb": {
                    "pearson_r": float(r_disag_fns_p),
                    "spearman_rho": float(r_disag_fns_s),
                },
                "disagreement_vs_raft_fb": {
                    "pearson_r": float(r_disag_raft_p),
                    "spearman_rho": float(r_disag_raft_s),
                },
                "flownet_fb_vs_raft_fb": {
                    "pearson_r": float(r_cross_fb_p),
                    "spearman_rho": float(r_cross_fb_s),
                },
            },
        }
        per_sample_results.append(record)

        # Accumulate for pooled correlation
        pooled_res_fns.append(res_fns_np[eval_mask_fns])
        pooled_err_fns.append(err_fns_np[eval_mask_fns])
        pooled_res_raft.append(res_raft_np[eval_mask_raft])
        pooled_err_raft.append(err_raft_np[eval_mask_raft])
        pooled_disag.append(disag_np[joint_eval_mask])

        # Subsample for responsive visualization (2,500 points per scene)
        rng = np.random.default_rng(200 + i)
        s_fns = min(2500, eval_fns_count)
        idx_fns = rng.choice(eval_fns_count, size=s_fns, replace=False)
        scatter_res_fns.append(res_fns_np[eval_mask_fns][idx_fns])
        scatter_err_fns.append(err_fns_np[eval_mask_fns][idx_fns])

        s_raft = min(2500, eval_raft_count)
        idx_raft = rng.choice(eval_raft_count, size=s_raft, replace=False)
        scatter_res_raft.append(res_raft_np[eval_mask_raft][idx_raft])
        scatter_err_raft.append(err_raft_np[eval_mask_raft][idx_raft])

    # Global pooled correlation calculations
    cat_res_fns = np.concatenate(pooled_res_fns)
    cat_err_fns = np.concatenate(pooled_err_fns)
    cat_res_raft = np.concatenate(pooled_res_raft)
    cat_err_raft = np.concatenate(pooled_err_raft)

    pooled_r_fns_own_p, _ = scipy.stats.pearsonr(cat_res_fns, cat_err_fns)
    pooled_r_fns_own_s, _ = scipy.stats.spearmanr(cat_res_fns, cat_err_fns)

    pooled_r_raft_own_p, _ = scipy.stats.pearsonr(cat_res_raft, cat_err_raft)
    pooled_r_raft_own_s, _ = scipy.stats.spearmanr(cat_res_raft, cat_err_raft)

    # Macro averages across the 8 samples
    macro_res_fns_mean = float(np.mean([r["flownet_s"]["fb_residual_stats_eval"]["mean"] for r in per_sample_results]))
    macro_res_raft_mean = float(np.mean([r["raft"]["fb_residual_stats_eval"]["mean"] for r in per_sample_results]))
    macro_r_fns_p = float(np.mean([r["flownet_s"]["fb_vs_own_error_correlation"]["pearson_r"] for r in per_sample_results]))
    macro_r_fns_s = float(np.mean([r["flownet_s"]["fb_vs_own_error_correlation"]["spearman_rho"] for r in per_sample_results]))
    macro_r_raft_p = float(np.mean([r["raft"]["fb_vs_own_error_correlation"]["pearson_r"] for r in per_sample_results]))
    macro_r_raft_s = float(np.mean([r["raft"]["fb_vs_own_error_correlation"]["spearman_rho"] for r in per_sample_results]))
    macro_disag_fns_p = float(np.mean([r["disagreement_vs_fb_correlations"]["disagreement_vs_flownet_fb"]["pearson_r"] for r in per_sample_results]))
    macro_disag_raft_p = float(np.mean([r["disagreement_vs_fb_correlations"]["disagreement_vs_raft_fb"]["pearson_r"] for r in per_sample_results]))

    print("\n==================================================================")
    print("                     EXPERIMENT SUMMARY                           ")
    print("==================================================================")
    print(f"Macro FlowNetS FB Residual:     {macro_res_fns_mean:.4f} px (Macro EPE: {np.mean([r['flownet_s']['mean_epe_valid_px'] for r in per_sample_results]):.4f} px)")
    print(f"Macro RAFT FB Residual:         {macro_res_raft_mean:.4f} px (Macro EPE: {np.mean([r['raft']['mean_epe_valid_px'] for r in per_sample_results]):.4f} px)")
    print(f"Macro FlowNetS FB vs Error r:   {macro_r_fns_p:.4f} (Spearman: {macro_r_fns_s:.4f})")
    print(f"Macro RAFT FB vs Error r:       {macro_r_raft_p:.4f} (Spearman: {macro_r_raft_s:.4f})")
    print(f"Pooled FlowNetS FB vs Error r:  {pooled_r_fns_own_p:.4f} (Spearman: {pooled_r_fns_own_s:.4f})")
    print(f"Pooled RAFT FB vs Error r:      {pooled_r_raft_own_p:.4f} (Spearman: {pooled_r_raft_own_s:.4f})")
    print(f"Macro Disagreement vs FNS FB r: {macro_disag_fns_p:.4f}")
    print(f"Macro Disagreement vs RAFT FB r:{macro_disag_raft_p:.4f}")
    print("==================================================================\n")

    # Save JSON summary
    json_summary = {
        "metadata": {
            "num_samples": len(DEFAULT_SAMPLES),
            "samples": DEFAULT_SAMPLES,
            "flownet_checkpoint": args.flownet_checkpoint,
            "device": str(device),
        },
        "macro_summary": {
            "flownet_mean_fb_residual_px": macro_res_fns_mean,
            "raft_mean_fb_residual_px": macro_res_raft_mean,
            "flownet_fb_vs_own_error_pearson_r": macro_r_fns_p,
            "flownet_fb_vs_own_error_spearman_rho": macro_r_fns_s,
            "raft_fb_vs_own_error_pearson_r": macro_r_raft_p,
            "raft_fb_vs_own_error_spearman_rho": macro_r_raft_s,
            "disagreement_vs_flownet_fb_pearson_r": macro_disag_fns_p,
            "disagreement_vs_raft_fb_pearson_r": macro_disag_raft_p,
        },
        "pooled_correlations": {
            "flownet_fb_vs_own_error": {
                "pearson_r": float(pooled_r_fns_own_p),
                "spearman_rho": float(pooled_r_fns_own_s),
            },
            "raft_fb_vs_own_error": {
                "pearson_r": float(pooled_r_raft_own_p),
                "spearman_rho": float(pooled_r_raft_own_s),
            },
        },
        "per_sample_results": per_sample_results,
    }

    json_path = output_dir / "day4_forward_backward.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Generate 2x3 Diagnostic Visualization
    labels = [f"{r['scene']}\n({r['pass_name']})" for r in per_sample_results]
    x = np.arange(len(labels))
    width = 0.20

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "Day 4 Forward-Backward Optical Flow Consistency & Ground-Truth Correlation (8 Sintel Samples)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Mean FB Residual vs Model Ground-Truth EPE
    fns_epes = [r["flownet_s"]["mean_epe_valid_px"] for r in per_sample_results]
    fns_fbs = [r["flownet_s"]["fb_residual_stats_eval"]["mean"] for r in per_sample_results]
    raft_epes = [r["raft"]["mean_epe_valid_px"] for r in per_sample_results]
    raft_fbs = [r["raft"]["fb_residual_stats_eval"]["mean"] for r in per_sample_results]

    axes[0, 0].bar(x - 1.5 * width, fns_epes, width, label="FlowNetS EPE", color="#ff7f0e", alpha=0.9)
    axes[0, 0].bar(x - 0.5 * width, fns_fbs, width, label="FlowNetS R_FB", color="#d62728", alpha=0.85)
    axes[0, 0].bar(x + 0.5 * width, raft_epes, width, label="RAFT EPE", color="#2ca02c", alpha=0.9)
    axes[0, 0].bar(x + 1.5 * width, raft_fbs, width, label="RAFT R_FB", color="#17becf", alpha=0.85)
    axes[0, 0].set_title("1. Mean FB Residual vs. Model GT EPE", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_ylabel("Pixels", fontsize=10)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 2: FB Residual Correlation with Model's OWN Error
    r_fns_p_list = [r["flownet_s"]["fb_vs_own_error_correlation"]["pearson_r"] for r in per_sample_results]
    r_fns_s_list = [r["flownet_s"]["fb_vs_own_error_correlation"]["spearman_rho"] for r in per_sample_results]
    r_raft_p_list = [r["raft"]["fb_vs_own_error_correlation"]["pearson_r"] for r in per_sample_results]
    r_raft_s_list = [r["raft"]["fb_vs_own_error_correlation"]["spearman_rho"] for r in per_sample_results]

    axes[0, 1].bar(x - 1.5 * width, r_fns_p_list, width, label="FNS FB vs Error (Pearson r)", color="#ff7f0e", alpha=0.85)
    axes[0, 1].bar(x - 0.5 * width, r_fns_s_list, width, label="FNS FB vs Error (Spearman rho)", color="#d62728", alpha=0.85)
    axes[0, 1].bar(x + 0.5 * width, r_raft_p_list, width, label="RAFT FB vs Error (Pearson r)", color="#2ca02c", alpha=0.85)
    axes[0, 1].bar(x + 1.5 * width, r_raft_s_list, width, label="RAFT FB vs Error (Spearman rho)", color="#17becf", alpha=0.85)
    axes[0, 1].set_title("2. Correlation: FB Residual vs. Model's Own Error", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].set_ylabel("Correlation Coefficient", fontsize=10)
    axes[0, 1].set_ylim(-0.2, 1.05)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 3: Correlation Between Model Disagreement and FB Residuals
    r_disag_fns = [r["disagreement_vs_fb_correlations"]["disagreement_vs_flownet_fb"]["pearson_r"] for r in per_sample_results]
    r_disag_raft = [r["disagreement_vs_fb_correlations"]["disagreement_vs_raft_fb"]["pearson_r"] for r in per_sample_results]
    r_cross_fb = [r["disagreement_vs_fb_correlations"]["flownet_fb_vs_raft_fb"]["pearson_r"] for r in per_sample_results]

    w3 = 0.25
    axes[0, 2].bar(x - w3, r_disag_fns, w3, label="Disagreement vs. FNS FB", color="#9467bd", alpha=0.85)
    axes[0, 2].bar(x, r_disag_raft, w3, label="Disagreement vs. RAFT FB", color="#8c564b", alpha=0.85)
    axes[0, 2].bar(x + w3, r_cross_fb, w3, label="FNS FB vs. RAFT FB", color="#e377c2", alpha=0.85)
    axes[0, 2].set_title("3. Disagreement vs. FB Residual Correlations (Pearson)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, fontsize=9)
    axes[0, 2].set_ylabel("Pearson Correlation r", fontsize=10)
    axes[0, 2].set_ylim(-0.2, 1.05)
    axes[0, 2].legend(loc="lower left", fontsize=8)
    axes[0, 2].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 4: In-Bounds Geometric Retention
    in_b_fns_pct = [r["pixel_counts"]["in_bounds_flownet"] / r["pixel_counts"]["total"] * 100 for r in per_sample_results]
    in_b_raft_pct = [r["pixel_counts"]["in_bounds_raft"] / r["pixel_counts"]["total"] * 100 for r in per_sample_results]

    axes[1, 0].bar(x - 0.18, in_b_fns_pct, 0.36, label="FlowNetS In-Bounds %", color="#ff7f0e", alpha=0.85)
    axes[1, 0].bar(x + 0.18, in_b_raft_pct, 0.36, label="RAFT In-Bounds %", color="#2ca02c", alpha=0.85)
    axes[1, 0].set_title("4. Geometric In-Bounds Pixel Retention (%)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].set_ylabel("In-Bounds Pixels (%)", fontsize=10)
    axes[1, 0].set_ylim(80, 100.5)
    axes[1, 0].legend(loc="lower left", fontsize=9)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 5: Pooled Scatter: FlowNetS FB Residual vs FlowNetS Error
    sc_res_fns = np.concatenate(scatter_res_fns)
    sc_err_fns = np.concatenate(scatter_err_fns)
    max_fns_vis = float(np.percentile(np.maximum(sc_res_fns, sc_err_fns), 99.5))

    axes[1, 1].scatter(sc_res_fns, sc_err_fns, alpha=0.20, s=8, color="#ff7f0e", edgecolors="none")
    diag_line_fns = np.linspace(0, max_fns_vis, 100)
    axes[1, 1].plot(diag_line_fns, diag_line_fns, "k--", linewidth=1.2, label="Identity (y = x)")
    axes[1, 1].set_xlim(0, max_fns_vis)
    axes[1, 1].set_ylim(0, max_fns_vis)
    axes[1, 1].set_title(
        f"5. Pooled FlowNetS FB vs. Error ({len(sc_res_fns):,} pts)\nPearson r = {pooled_r_fns_own_p:.3f} | Spearman rho = {pooled_r_fns_own_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("FlowNetS FB Residual R_FB (px)", fontsize=10)
    axes[1, 1].set_ylabel("FlowNetS Error E_FNS (px)", fontsize=10)
    axes[1, 1].legend(loc="upper left", fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 6: Pooled Scatter: RAFT FB Residual vs RAFT Error
    sc_res_raft = np.concatenate(scatter_res_raft)
    sc_err_raft = np.concatenate(scatter_err_raft)
    max_raft_vis = float(np.percentile(np.maximum(sc_res_raft, sc_err_raft), 99.5))

    axes[1, 2].scatter(sc_res_raft, sc_err_raft, alpha=0.20, s=8, color="#2ca02c", edgecolors="none")
    diag_line_raft = np.linspace(0, max_raft_vis, 100)
    axes[1, 2].plot(diag_line_raft, diag_line_raft, "k--", linewidth=1.2, label="Identity (y = x)")
    axes[1, 2].set_xlim(0, max_raft_vis)
    axes[1, 2].set_ylim(0, max_raft_vis)
    axes[1, 2].set_title(
        f"6. Pooled RAFT FB vs. Error ({len(sc_res_raft):,} pts)\nPearson r = {pooled_r_raft_own_p:.3f} | Spearman rho = {pooled_r_raft_own_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("RAFT FB Residual R_FB (px)", fontsize=10)
    axes[1, 2].set_ylabel("RAFT Error E_RAFT (px)", fontsize=10)
    axes[1, 2].legend(loc="upper left", fontsize=9)
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = output_dir / "day4_forward_backward.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic visualization saved to: {fig_path}")


if __name__ == "__main__":
    main()
