"""
Day 4: Photometric Optical Flow Consistency Diagnostic across 8 Sintel Samples.

Evaluates photometric consistency for FlowNetS and RAFT across MPI Sintel frame pairs:
1. Loads Frame 1, Frame 2, and ground-truth optical flow via SintelDataset.
2. Infers forward optical flow using FlowNetS and RAFT adapters.
3. Warps Frame 2 toward Frame 1 coordinates using bilinear grid sampling with align_corners=True.
4. Computes per-pixel RGB L1 photometric residual:
   R_photo(x, y) = mean_c |I1_c(x, y) - I2_c(x + u(x, y), y + v(x, y))|
5. Constructs explicit geometric in-bounds masks (0 <= x' <= W - 1 and 0 <= y' <= H - 1).
6. Evaluates photometric residual distributions and exploratory correlation against
   each model's own ground-truth EPE over valid in-bounds pixels.
7. Evaluates primary cross-sample correlation using pooled evaluation pixels.
8. Manages GPU memory strictly via explicit tensor deletion without torch.cuda.empty_cache().
9. Generates structured JSON summary and 6-panel diagnostic visualization.
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
        in_bounds_mask: [H, W] boolean tensor indicating coordinates that forward-map within Frame 2.
        warped_frame2: [3, H, W] float32 warped Frame 2 in Frame 1 coordinates.
    """
    _, H, W = frame1_float.shape
    device = frame1_float.device

    # 1. Base coordinate grid in Frame 1
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # 2. Forward arrival coordinates in Frame 2: x' = x + u, y' = y + v
    x_warp = x_grid + flow_fwd[0]
    y_warp = y_grid + flow_fwd[1]

    # 3. Geometric in-bounds mask where forward-warped coordinates lie inside Frame 2
    in_bounds_mask = (
        (x_warp >= 0.0) & (x_warp <= float(W - 1)) &
        (y_warp >= 0.0) & (y_warp <= float(H - 1))
    )

    # 4. Normalized coordinates for F.grid_sample in [-1.0, 1.0] (align_corners=True)
    norm_x = 2.0 * x_warp / max(W - 1, 1) - 1.0
    norm_y = 2.0 * y_warp / max(H - 1, 1) - 1.0
    sample_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # 5. Bilinear sampling of Frame 2 at forward arrival locations
    warped_frame2 = F.grid_sample(
        frame2_float.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]  # [3, H, W]

    # 6. Per-pixel RGB L1 residual: mean over color channels
    photo_residual = torch.mean(torch.abs(frame1_float - warped_frame2), dim=0)  # [H, W]

    return photo_residual, in_bounds_mask, warped_frame2


def verify_photometric_math() -> None:
    """
    Deterministic synthetic sanity checks for image warping and photometric residual math.

    Test 1: Identical images and zero flow -> residual identically 0.0 everywhere.
    Test 2: Translated image with exact matching flow -> residual near zero (< 1e-4) on interior.
    Test 3: Deliberately wrong flow on translated image ->
            wrong_mean > 100 * correct_mean and wrong_mean > 0.01.
    """
    H, W = 32, 32
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Linear gradient test image: I1(x, y) = 0.01 * x + 0.02 * y across RGB channels
    I1 = (0.01 * x_coords + 0.02 * y_coords).unsqueeze(0).repeat(3, 1, 1)

    # Test 1: Zero motion on identical images -> R_photo identically 0.0
    res_zero, in_b_zero, _ = compute_photometric_residual(
        I1, I1, torch.zeros((2, H, W), dtype=torch.float32)
    )
    assert torch.allclose(res_zero, torch.zeros_like(res_zero), atol=1e-5), (
        f"Test 1 failed: max zero residual {res_zero.max().item()} > 1e-5"
    )
    assert in_b_zero.all(), "Test 1 failed: zero-motion in-bounds check failed"

    # Known translation: u = 2.0, v = 3.0
    # I2(x, y) = I1(x - u, y - v) so that I2(x + u, y + v) = I1(x, y)
    u_val, v_val = 2.0, 3.0
    I2 = (0.01 * (x_coords - u_val) + 0.02 * (y_coords - v_val)).unsqueeze(0).repeat(3, 1, 1)

    # Test 2: Matching flow field -> near zero on in-bounds interior
    flow_correct = torch.zeros((2, H, W), dtype=torch.float32)
    flow_correct[0] = u_val
    flow_correct[1] = v_val
    res_correct, in_b_correct, _ = compute_photometric_residual(I1, I2, flow_correct)
    interior_mask = in_b_correct.clone()
    correct_mean = float(res_correct[interior_mask].mean().item())
    assert correct_mean < 1e-4, f"Test 2 failed: correct_mean={correct_mean:.6f} >= 1e-4"

    # Test 3: Deliberately wrong flow (zero flow instead of true motion (2, 3))
    flow_wrong = torch.zeros((2, H, W), dtype=torch.float32)
    res_wrong, _, _ = compute_photometric_residual(I1, I2, flow_wrong)
    wrong_mean = float(res_wrong[interior_mask].mean().item())
    assert wrong_mean > 100 * correct_mean, (
        f"Test 3 failed: wrong_mean={wrong_mean:.6f} <= 100 * correct_mean ({100 * correct_mean:.6f})"
    )
    assert wrong_mean > 0.01, (
        f"Test 3 failed: wrong_mean={wrong_mean:.6f} <= 0.01"
    )


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
            "fraction_gt_0p05": 0.0,
            "fraction_gt_0p10": 0.0,
            "fraction_gt_0p20": 0.0,
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
        "fraction_gt_0p05": float(np.mean(flat > 0.05)),
        "fraction_gt_0p10": float(np.mean(flat > 0.10)),
        "fraction_gt_0p20": float(np.mean(flat > 0.20)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Photometric optical flow consistency diagnostic across 8 Sintel pairs."
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
    # 0. Sanity-check image warping math before inference
    verify_photometric_math()

    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 PHOTOMETRIC FLOW CONSISTENCY EXPERIMENT (8 SAMPLES)   ")
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

    # Pooled buffers for cross-sample correlation analysis (full population)
    pooled_res_fns: List[np.ndarray] = []
    pooled_epe_fns: List[np.ndarray] = []
    pooled_res_raft: List[np.ndarray] = []
    pooled_epe_raft: List[np.ndarray] = []

    # Subsampled buffers for responsive visualization (at most 2,500 points per pair)
    scatter_res_fns: List[np.ndarray] = []
    scatter_epe_fns: List[np.ndarray] = []
    scatter_res_raft: List[np.ndarray] = []
    scatter_epe_raft: List[np.ndarray] = []

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
            # Forward flow inference: Frame 1 -> Frame 2
            flow_fwd_fns = flownet_model(frame1, frame2)[0]
            flow_fwd_raft = raft_model(frame1, frame2)[0]

            # Convert frames to float32 [0.0, 1.0] on device for photometric evaluation
            frame1_float = frame1.to(device=device, dtype=torch.float32) / 255.0
            frame2_float = frame2.to(device=device, dtype=torch.float32) / 255.0

            # Photometric residual and in-bounds mask
            res_fns_t, in_b_fns_t, warped_fns_t = compute_photometric_residual(
                frame1_float, frame2_float, flow_fwd_fns
            )
            res_raft_t, in_b_raft_t, warped_raft_t = compute_photometric_residual(
                frame1_float, frame2_float, flow_fwd_raft
            )

            # Actual forward errors against ground truth (EPE)
            flow_gt_dev = flow_gt.to(device)
            diff_fns = flow_fwd_fns - flow_gt_dev
            epe_fns_t = torch.sqrt(diff_fns[0] ** 2 + diff_fns[1] ** 2)
            del diff_fns

            diff_raft = flow_fwd_raft - flow_gt_dev
            epe_raft_t = torch.sqrt(diff_raft[0] ** 2 + diff_raft[1] ** 2)
            del diff_raft

        # Move tensors to CPU NumPy arrays
        res_fns_np = res_fns_t.cpu().numpy()
        in_b_fns_np = in_b_fns_t.cpu().numpy()
        epe_fns_np = epe_fns_t.cpu().numpy()

        res_raft_np = res_raft_t.cpu().numpy()
        in_b_raft_np = in_b_raft_t.cpu().numpy()
        epe_raft_np = epe_raft_t.cpu().numpy()

        valid_gt_np = valid_mask.cpu().numpy()

        # Strict memory hygiene: explicit tensor deletion without torch.cuda.empty_cache()
        del flow_fwd_fns, flow_fwd_raft, flow_gt_dev
        del res_fns_t, in_b_fns_t, warped_fns_t
        del res_raft_t, in_b_raft_t, warped_raft_t
        del epe_fns_t, epe_raft_t, frame1_float, frame2_float

        # Define evaluation masks (valid GT & geometric in-bounds & finite values)
        eval_mask_fns = (
            valid_gt_np & in_b_fns_np & np.isfinite(res_fns_np) & np.isfinite(epe_fns_np)
        )
        eval_mask_raft = (
            valid_gt_np & in_b_raft_np & np.isfinite(res_raft_np) & np.isfinite(epe_raft_np)
        )

        in_b_fns_count = int(np.sum(in_b_fns_np))
        in_b_raft_count = int(np.sum(in_b_raft_np))
        eval_fns_count = int(np.sum(eval_mask_fns))
        eval_raft_count = int(np.sum(eval_mask_raft))

        # Distribution statistics over evaluation mask
        stats_fns = compute_distribution_stats(res_fns_np[eval_mask_fns])
        stats_raft = compute_distribution_stats(res_raft_np[eval_mask_raft])

        # Correlation between photometric residual and model's own ground-truth EPE
        corr_fns_p, _ = scipy.stats.pearsonr(res_fns_np[eval_mask_fns], epe_fns_np[eval_mask_fns])
        corr_fns_s, _ = scipy.stats.spearmanr(res_fns_np[eval_mask_fns], epe_fns_np[eval_mask_fns])

        corr_raft_p, _ = scipy.stats.pearsonr(res_raft_np[eval_mask_raft], epe_raft_np[eval_mask_raft])
        corr_raft_s, _ = scipy.stats.spearmanr(res_raft_np[eval_mask_raft], epe_raft_np[eval_mask_raft])

        mean_epe_valid_fns = float(np.mean(epe_fns_np[valid_gt_np]))
        mean_epe_eval_fns = float(np.mean(epe_fns_np[eval_mask_fns]))

        mean_epe_valid_raft = float(np.mean(epe_raft_np[valid_gt_np]))
        mean_epe_eval_raft = float(np.mean(epe_raft_np[eval_mask_raft]))

        print(
            f"FNS R_photo={stats_fns['mean']:.4f} (EPE={mean_epe_valid_fns:.2f} px, r={corr_fns_p:.3f}) | "
            f"RAFT R_photo={stats_raft['mean']:.4f} (EPE={mean_epe_valid_raft:.2f} px, r={corr_raft_p:.3f})"
        )

        record = {
            "sample_index": i,
            "scene": scene,
            "pass_name": pass_name,
            "pair_idx": pair_idx,
            "frame_idx": meta["frame_idx"],
            "frame1_path": meta["frame1_path"],
            "frame2_path": meta["frame2_path"],
            "flow_gt_path": meta["flow_path"],
            "invalid_mask_path": meta["invalid_path"],
            "dimensions": {"height": orig_h, "width": orig_w},
            "pixels": {
                "total": total_pixels,
                "valid_gt": valid_gt_count,
                "in_bounds_flownet": in_b_fns_count,
                "in_bounds_raft": in_b_raft_count,
                "eval_flownet": eval_fns_count,
                "eval_raft": eval_raft_count,
            },
            "flownet_s": {
                "mean_epe_valid_gt_px": mean_epe_valid_fns,
                "mean_epe_eval_px": mean_epe_eval_fns,
                "photometric_residual_stats": stats_fns,
                "photometric_vs_own_epe_correlation": {
                    "pearson_r": float(corr_fns_p),
                    "spearman_rho": float(corr_fns_s),
                },
            },
            "raft": {
                "mean_epe_valid_gt_px": mean_epe_valid_raft,
                "mean_epe_eval_px": mean_epe_eval_raft,
                "photometric_residual_stats": stats_raft,
                "photometric_vs_own_epe_correlation": {
                    "pearson_r": float(corr_raft_p),
                    "spearman_rho": float(corr_raft_s),
                },
            },
        }
        per_sample_results.append(record)

        # Accumulate for pooled correlation (full evaluation population)
        pooled_res_fns.append(res_fns_np[eval_mask_fns])
        pooled_epe_fns.append(epe_fns_np[eval_mask_fns])
        pooled_res_raft.append(res_raft_np[eval_mask_raft])
        pooled_epe_raft.append(epe_raft_np[eval_mask_raft])

        # Subsample for responsive visualization (at most 2,500 points per pair)
        rng = np.random.default_rng(200 + i)
        s_fns = min(2500, eval_fns_count)
        if s_fns > 0:
            idx_fns = rng.choice(eval_fns_count, size=s_fns, replace=False)
            scatter_res_fns.append(res_fns_np[eval_mask_fns][idx_fns])
            scatter_epe_fns.append(epe_fns_np[eval_mask_fns][idx_fns])

        s_raft = min(2500, eval_raft_count)
        if s_raft > 0:
            idx_raft = rng.choice(eval_raft_count, size=s_raft, replace=False)
            scatter_res_raft.append(res_raft_np[eval_mask_raft][idx_raft])
            scatter_epe_raft.append(epe_raft_np[eval_mask_raft][idx_raft])

    # Global pooled correlation calculations (from concatenated full evaluation pixels)
    cat_res_fns = np.concatenate(pooled_res_fns)
    cat_epe_fns = np.concatenate(pooled_epe_fns)
    cat_res_raft = np.concatenate(pooled_res_raft)
    cat_epe_raft = np.concatenate(pooled_epe_raft)

    pooled_r_fns_p, _ = scipy.stats.pearsonr(cat_res_fns, cat_epe_fns)
    pooled_r_fns_s, _ = scipy.stats.spearmanr(cat_res_fns, cat_epe_fns)

    pooled_r_raft_p, _ = scipy.stats.pearsonr(cat_res_raft, cat_epe_raft)
    pooled_r_raft_s, _ = scipy.stats.spearmanr(cat_res_raft, cat_epe_raft)

    # Macro averages across the 8 samples (ONLY for scalar quantities, NOT correlations)
    macro_epe_fns = float(np.mean([r["flownet_s"]["mean_epe_valid_gt_px"] for r in per_sample_results]))
    macro_epe_raft = float(np.mean([r["raft"]["mean_epe_valid_gt_px"] for r in per_sample_results]))
    macro_res_fns = float(np.mean([r["flownet_s"]["photometric_residual_stats"]["mean"] for r in per_sample_results]))
    macro_res_raft = float(np.mean([r["raft"]["photometric_residual_stats"]["mean"] for r in per_sample_results]))
    macro_in_b_fns = float(np.mean([r["pixels"]["in_bounds_flownet"] / r["pixels"]["total"] * 100 for r in per_sample_results]))
    macro_in_b_raft = float(np.mean([r["pixels"]["in_bounds_raft"] / r["pixels"]["total"] * 100 for r in per_sample_results]))

    print("\n==================================================================")
    print("                     EXPERIMENT SUMMARY                           ")
    print("==================================================================")
    print(f"Macro FlowNetS EPE:             {macro_epe_fns:.4f} px")
    print(f"Macro RAFT EPE:                 {macro_epe_raft:.4f} px")
    print(f"Macro FlowNetS R_photo:         {macro_res_fns:.4f}")
    print(f"Macro RAFT R_photo:             {macro_res_raft:.4f}")
    print(f"Macro FlowNetS In-Bounds %:     {macro_in_b_fns:.2f}%")
    print(f"Macro RAFT In-Bounds %:         {macro_in_b_raft:.2f}%")
    print(f"Pooled FlowNetS vs Error:       Pearson r={pooled_r_fns_p:.4f}, Spearman rho={pooled_r_fns_s:.4f} (N={cat_res_fns.size:,} px)")
    print(f"Pooled RAFT vs Error:           Pearson r={pooled_r_raft_p:.4f}, Spearman rho={pooled_r_raft_s:.4f} (N={cat_res_raft.size:,} px)")
    print("==================================================================\n")

    # Save JSON summary
    json_summary = {
        "metadata": {
            "experiment": "day4_photometric_consistency_suite",
            "num_samples": len(DEFAULT_SAMPLES),
            "samples": DEFAULT_SAMPLES,
            "flownet_checkpoint": args.flownet_checkpoint,
            "device": str(device),
        },
        "synthetic_sanity_checks": {
            "test1_zero_motion": "passed",
            "test2_translated_motion": "passed",
            "test3_wrong_motion": "passed",
        },
        "macro_summary": {
            "flownet_macro_mean_epe_valid_px": macro_epe_fns,
            "raft_macro_mean_epe_valid_px": macro_epe_raft,
            "flownet_macro_mean_residual_eval": macro_res_fns,
            "raft_macro_mean_residual_eval": macro_res_raft,
            "flownet_macro_in_bounds_pct": macro_in_b_fns,
            "raft_macro_in_bounds_pct": macro_in_b_raft,
        },
        "pooled_correlations": {
            "flownet_s": {
                "pooled_pixel_count": int(cat_res_fns.size),
                "pearson_r": float(pooled_r_fns_p),
                "spearman_rho": float(pooled_r_fns_s),
            },
            "raft": {
                "pooled_pixel_count": int(cat_res_raft.size),
                "pearson_r": float(pooled_r_raft_p),
                "spearman_rho": float(pooled_r_raft_s),
            },
        },
        "per_sample_results": per_sample_results,
    }

    json_path = output_dir / "day4_photometric.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Generate 2x3 Diagnostic Visualization using Matplotlib default color handling
    labels = [f"{r['scene']}\n({r['pass_name']})" for r in per_sample_results]
    x = np.arange(len(labels))
    width = 0.20

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "Day 4 Photometric Optical Flow Consistency & Error Correlation (8 Sintel Samples)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Mean Photometric Residual vs Model Ground-Truth EPE
    fns_epes = [r["flownet_s"]["mean_epe_valid_gt_px"] for r in per_sample_results]
    fns_res = [r["flownet_s"]["photometric_residual_stats"]["mean"] for r in per_sample_results]
    raft_epes = [r["raft"]["mean_epe_valid_gt_px"] for r in per_sample_results]
    raft_res = [r["raft"]["photometric_residual_stats"]["mean"] for r in per_sample_results]

    axes[0, 0].bar(x - 1.5 * width, fns_epes, width, label="FlowNetS EPE (px)", alpha=0.9)
    axes[0, 0].bar(x - 0.5 * width, fns_res, width, label="FlowNetS R_photo", alpha=0.9)
    axes[0, 0].bar(x + 0.5 * width, raft_epes, width, label="RAFT EPE (px)", alpha=0.9)
    axes[0, 0].bar(x + 1.5 * width, raft_res, width, label="RAFT R_photo", alpha=0.9)
    axes[0, 0].set_title("1. Mean Photometric Residual vs. Model GT EPE", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_ylabel("Metric Value", fontsize=10)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 2: Per-Sample Correlation: R_photo vs. Model's Own Error
    r_fns_p_list = [r["flownet_s"]["photometric_vs_own_epe_correlation"]["pearson_r"] for r in per_sample_results]
    r_fns_s_list = [r["flownet_s"]["photometric_vs_own_epe_correlation"]["spearman_rho"] for r in per_sample_results]
    r_raft_p_list = [r["raft"]["photometric_vs_own_epe_correlation"]["pearson_r"] for r in per_sample_results]
    r_raft_s_list = [r["raft"]["photometric_vs_own_epe_correlation"]["spearman_rho"] for r in per_sample_results]

    axes[0, 1].bar(x - 1.5 * width, r_fns_p_list, width, label="FNS vs Error (Pearson r)", alpha=0.9)
    axes[0, 1].bar(x - 0.5 * width, r_fns_s_list, width, label="FNS vs Error (Spearman rho)", alpha=0.9)
    axes[0, 1].bar(x + 0.5 * width, r_raft_p_list, width, label="RAFT vs Error (Pearson r)", alpha=0.9)
    axes[0, 1].bar(x + 1.5 * width, r_raft_s_list, width, label="RAFT vs Error (Spearman rho)", alpha=0.9)
    axes[0, 1].set_title("2. Per-Sample Correlation: R_photo vs. Model's Own Error", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].set_ylabel("Correlation Coefficient", fontsize=10)
    axes[0, 1].set_ylim(-0.2, 1.05)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 3: High Photometric Residual Tail Fractions
    fns_tail_05 = [r["flownet_s"]["photometric_residual_stats"]["fraction_gt_0p05"] * 100 for r in per_sample_results]
    fns_tail_10 = [r["flownet_s"]["photometric_residual_stats"]["fraction_gt_0p10"] * 100 for r in per_sample_results]
    raft_tail_05 = [r["raft"]["photometric_residual_stats"]["fraction_gt_0p05"] * 100 for r in per_sample_results]
    raft_tail_10 = [r["raft"]["photometric_residual_stats"]["fraction_gt_0p10"] * 100 for r in per_sample_results]

    axes[0, 2].bar(x - 1.5 * width, fns_tail_05, width, label="FNS % > 0.05", alpha=0.9)
    axes[0, 2].bar(x - 0.5 * width, fns_tail_10, width, label="FNS % > 0.10", alpha=0.9)
    axes[0, 2].bar(x + 0.5 * width, raft_tail_05, width, label="RAFT % > 0.05", alpha=0.9)
    axes[0, 2].bar(x + 1.5 * width, raft_tail_10, width, label="RAFT % > 0.10", alpha=0.9)
    axes[0, 2].set_title("3. High Photometric Residual Tail Fractions (%)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, fontsize=9)
    axes[0, 2].set_ylabel("Percentage of Eval Pixels (%)", fontsize=10)
    axes[0, 2].legend(loc="upper right", fontsize=8)
    axes[0, 2].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 4: Geometric In-Bounds Pixel Retention
    in_b_fns_pct = [r["pixels"]["in_bounds_flownet"] / r["pixels"]["total"] * 100 for r in per_sample_results]
    in_b_raft_pct = [r["pixels"]["in_bounds_raft"] / r["pixels"]["total"] * 100 for r in per_sample_results]

    axes[1, 0].bar(x - 0.18, in_b_fns_pct, 0.36, label="FlowNetS In-Bounds %", alpha=0.9)
    axes[1, 0].bar(x + 0.18, in_b_raft_pct, 0.36, label="RAFT In-Bounds %", alpha=0.9)
    axes[1, 0].set_title("4. Geometric In-Bounds Pixel Retention (%)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].set_ylabel("In-Bounds Pixels (%)", fontsize=10)
    axes[1, 0].set_ylim(80, 100.5)
    axes[1, 0].legend(loc="lower left", fontsize=9)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 5: Pooled Scatter: FlowNetS R_photo vs Error
    sc_res_fns = np.concatenate(scatter_res_fns)
    sc_epe_fns = np.concatenate(scatter_epe_fns)
    max_res_fns = float(np.percentile(sc_res_fns, 99.5))
    max_epe_fns = float(np.percentile(sc_epe_fns, 99.5))

    axes[1, 1].scatter(sc_res_fns, sc_epe_fns, alpha=0.20, s=8, edgecolors="none")
    axes[1, 1].set_xlim(0, max(max_res_fns, 0.1))
    axes[1, 1].set_ylim(0, max_epe_fns)
    axes[1, 1].set_title(
        f"5. Pooled FlowNetS R_photo vs. Error ({len(sc_res_fns):,} pts)\nPooled Pearson r = {pooled_r_fns_p:.3f} | Pooled Spearman rho = {pooled_r_fns_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("FlowNetS Photometric Residual R_photo [0, 1]", fontsize=10)
    axes[1, 1].set_ylabel("FlowNetS Error E_FNS (px)", fontsize=10)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 6: Pooled Scatter: RAFT R_photo vs Error
    sc_res_raft = np.concatenate(scatter_res_raft)
    sc_epe_raft = np.concatenate(scatter_epe_raft)
    max_res_raft = float(np.percentile(sc_res_raft, 99.5))
    max_epe_raft = float(np.percentile(sc_epe_raft, 99.5))

    axes[1, 2].scatter(sc_res_raft, sc_epe_raft, alpha=0.20, s=8, edgecolors="none")
    axes[1, 2].set_xlim(0, max(max_res_raft, 0.1))
    axes[1, 2].set_ylim(0, max_epe_raft)
    axes[1, 2].set_title(
        f"6. Pooled RAFT R_photo vs. Error ({len(sc_res_raft):,} pts)\nPooled Pearson r = {pooled_r_raft_p:.3f} | Pooled Spearman rho = {pooled_r_raft_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("RAFT Photometric Residual R_photo [0, 1]", fontsize=10)
    axes[1, 2].set_ylabel("RAFT Error E_RAFT (px)", fontsize=10)
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    vis_path = output_dir / "day4_photometric.png"
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic visualization saved to: {vis_path}")
    print("\nPhotometric consistency suite evaluation completed successfully.")


if __name__ == "__main__":
    main()
