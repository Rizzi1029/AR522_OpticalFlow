"""
Day 4 Prototype: Multi-Pair Dense Optical Flow Disagreement Analysis.

Extends the single-pair diagnostic across 8 diverse MPI Sintel frame pairs (frame 1 -> 2):
1. alley_1 (clean)
2. alley_1 (final)
3. shaman_3 (clean)
4. bamboo_2 (clean)
5. market_6 (clean)
6. cave_2 (clean)
7. ambush_2 (clean)
8. ambush_4 (final)

For each pair:
- Computes dense FlowNetS and RAFT flow predictions in memory.
- Computes dense model disagreement: D(x, y) = ||F_FlowNetS(x, y) - F_RAFT(x, y)||_2
- Computes ground-truth endpoint errors:
  E_FNS(x, y)  = ||F_FlowNetS(x, y) - F_GT(x, y)||_2
  E_RAFT(x, y) = ||F_RAFT(x, y)     - F_GT(x, y)||_2
- Measures flow distribution statistics, percentiles, and exploratory correlations (Pearson r, Spearman rho).
- Strictly frees GPU tensors per iteration (NO torch.cuda.empty_cache()).
- Generates compact machine-readable JSON summary and aggregate diagnostic visualization.
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

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator

# Fixed 8-sample evaluation suite (all frame 1 -> 2)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Multi-pair dense flow disagreement experiment across 8 Sintel pairs."
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 MULTI-PAIR DENSE FLOW DISAGREEMENT EXPERIMENT         ")
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

    # Buffers for pooled analysis across all samples
    all_d_valid: List[np.ndarray] = []
    all_e_fns_valid: List[np.ndarray] = []
    all_e_raft_valid: List[np.ndarray] = []

    # Subsampled buffers for responsive visualization
    scatter_d: List[np.ndarray] = []
    scatter_e_fns: List[np.ndarray] = []
    scatter_e_raft: List[np.ndarray] = []
    scatter_labels: List[str] = []

    for i, spec in enumerate(DEFAULT_SAMPLES):
        scene = spec["scene"]
        pass_name = spec["pass_name"]
        pair_idx = spec["pair_idx"]
        sample_label = f"{scene} ({pass_name})"

        print(f"[{i + 1}/{len(DEFAULT_SAMPLES)}] Evaluating: {sample_label} ... ", end="", flush=True)

        # Load pair
        dataset = SintelDataset(
            root=args.data_root,
            split="training",
            pass_name=pass_name,
            scenes=[scene],
        )
        frame1, frame2, flow_gt, valid_mask, meta = dataset[pair_idx]
        orig_h, orig_w = frame1.shape[1], frame1.shape[2]
        total_pixels = orig_h * orig_w
        valid_count = int(valid_mask.sum().item())
        invalid_count = total_pixels - valid_count

        # Run inference
        with torch.inference_mode():
            flow_fns_raw = flownet_model(frame1, frame2)
            flow_fns = flow_fns_raw[0]
            del flow_fns_raw

            flow_raft_raw = raft_model(frame1, frame2)
            flow_raft = flow_raft_raw[0]
            del flow_raft_raw

            # Compute disagreement
            diff_models = flow_fns - flow_raft
            disagreement_t = torch.sqrt(diff_models[0] ** 2 + diff_models[1] ** 2)
            del diff_models

            # Compute errors relative to Ground Truth
            flow_gt_dev = flow_gt.to(device)
            diff_fns = flow_fns - flow_gt_dev
            err_fns_t = torch.sqrt(diff_fns[0] ** 2 + diff_fns[1] ** 2)
            del diff_fns

            diff_raft = flow_raft - flow_gt_dev
            err_raft_t = torch.sqrt(diff_raft[0] ** 2 + diff_raft[1] ** 2)
            del diff_raft

            mag_fns_t = torch.sqrt(flow_fns[0] ** 2 + flow_fns[1] ** 2)
            mag_raft_t = torch.sqrt(flow_raft[0] ** 2 + flow_raft[1] ** 2)
            mag_gt_t = torch.sqrt(flow_gt_dev[0] ** 2 + flow_gt_dev[1] ** 2)

        # Move to CPU NumPy
        disagreement_np = disagreement_t.cpu().numpy()
        err_fns_np = err_fns_t.cpu().numpy()
        err_raft_np = err_raft_t.cpu().numpy()
        mag_fns_np = mag_fns_t.cpu().numpy()
        mag_raft_np = mag_raft_t.cpu().numpy()
        mag_gt_np = mag_gt_t.cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()
        invalid_mask_np = ~valid_mask_np

        # Strict memory deallocation: explicitly delete GPU tensors (NO torch.cuda.empty_cache)
        del flow_fns, flow_raft, flow_gt_dev
        del disagreement_t, err_fns_t, err_raft_t, mag_fns_t, mag_raft_t, mag_gt_t

        # Statistical calculations
        d_all = disagreement_np.flatten()
        d_valid = disagreement_np[valid_mask_np]
        d_invalid = disagreement_np[invalid_mask_np]
        e_fns_valid = err_fns_np[valid_mask_np]
        e_raft_valid = err_raft_np[valid_mask_np]

        stats_d_all = compute_distribution_stats(d_all)
        stats_d_valid = compute_distribution_stats(d_valid)
        stats_d_invalid = compute_distribution_stats(d_invalid)

        # Correlations over valid pixels
        corr_fns_p, _ = scipy.stats.pearsonr(d_valid, e_fns_valid)
        corr_fns_s, _ = scipy.stats.spearmanr(d_valid, e_fns_valid)
        corr_raft_p, _ = scipy.stats.pearsonr(d_valid, e_raft_valid)
        corr_raft_s, _ = scipy.stats.spearmanr(d_valid, e_raft_valid)
        corr_models_p, _ = scipy.stats.pearsonr(e_fns_valid, e_raft_valid)
        corr_models_s, _ = scipy.stats.spearmanr(e_fns_valid, e_raft_valid)

        mean_epe_fns = float(np.mean(e_fns_valid))
        mean_epe_raft = float(np.mean(e_raft_valid))

        print(
            f"FNS EPE={mean_epe_fns:.3f} px | RAFT EPE={mean_epe_raft:.3f} px | "
            f"Disag Mean={stats_d_valid['mean']:.3f} px | r(D, FNS)={corr_fns_p:.3f} | r(D, RAFT)={corr_raft_p:.3f}"
        )

        record = {
            "sample_index": i,
            "scene": scene,
            "pass_name": pass_name,
            "frame_idx": meta["frame_idx"],
            "dimensions": {"height": orig_h, "width": orig_w},
            "pixels": {
                "total": total_pixels,
                "valid": valid_count,
                "invalid": invalid_count,
                "valid_ratio": valid_count / total_pixels,
            },
            "flownet_s": {
                "mean_epe_valid_px": mean_epe_fns,
                "magnitude_mean_px": float(np.mean(mag_fns_np)),
                "magnitude_max_px": float(np.max(mag_fns_np)),
            },
            "raft": {
                "mean_epe_valid_px": mean_epe_raft,
                "magnitude_mean_px": float(np.mean(mag_raft_np)),
                "magnitude_max_px": float(np.max(mag_raft_np)),
            },
            "ground_truth": {
                "magnitude_mean_px": float(np.mean(mag_gt_np[valid_mask_np])),
                "magnitude_max_px": float(np.max(mag_gt_np[valid_mask_np])),
            },
            "disagreement": {
                "all_pixels": stats_d_all,
                "valid_pixels": stats_d_valid,
                "invalid_pixels": stats_d_invalid,
            },
            "correlations_valid": {
                "disagreement_vs_flownet_error": {
                    "pearson_r": float(corr_fns_p),
                    "spearman_rho": float(corr_fns_s),
                },
                "disagreement_vs_raft_error": {
                    "pearson_r": float(corr_raft_p),
                    "spearman_rho": float(corr_raft_s),
                },
                "flownet_error_vs_raft_error": {
                    "pearson_r": float(corr_models_p),
                    "spearman_rho": float(corr_models_s),
                },
            },
        }
        per_sample_results.append(record)

        # Accumulate for pooled correlation
        all_d_valid.append(d_valid)
        all_e_fns_valid.append(e_fns_valid)
        all_e_raft_valid.append(e_raft_valid)

        # Subsample for responsive visualization (2,500 points per scene -> 20,000 pooled points)
        rng = np.random.default_rng(100 + i)
        sample_size = min(2500, len(d_valid))
        sample_indices = rng.choice(len(d_valid), size=sample_size, replace=False)
        scatter_d.append(d_valid[sample_indices])
        scatter_e_fns.append(e_fns_valid[sample_indices])
        scatter_e_raft.append(e_raft_valid[sample_indices])
        scatter_labels.extend([sample_label] * sample_size)

    # Global pooled calculations across all valid pixels in all 8 samples
    cat_d_valid = np.concatenate(all_d_valid)
    cat_e_fns_valid = np.concatenate(all_e_fns_valid)
    cat_e_raft_valid = np.concatenate(all_e_raft_valid)

    pooled_corr_fns_p, _ = scipy.stats.pearsonr(cat_d_valid, cat_e_fns_valid)
    pooled_corr_fns_s, _ = scipy.stats.spearmanr(cat_d_valid, cat_e_fns_valid)
    pooled_corr_raft_p, _ = scipy.stats.pearsonr(cat_d_valid, cat_e_raft_valid)
    pooled_corr_raft_s, _ = scipy.stats.spearmanr(cat_d_valid, cat_e_raft_valid)
    pooled_corr_models_p, _ = scipy.stats.pearsonr(cat_e_fns_valid, cat_e_raft_valid)
    pooled_corr_models_s, _ = scipy.stats.spearmanr(cat_e_fns_valid, cat_e_raft_valid)

    # Macro averages
    macro_mean_epe_fns = float(np.mean([r["flownet_s"]["mean_epe_valid_px"] for r in per_sample_results]))
    macro_mean_epe_raft = float(np.mean([r["raft"]["mean_epe_valid_px"] for r in per_sample_results]))
    macro_disag_mean = float(np.mean([r["disagreement"]["valid_pixels"]["mean"] for r in per_sample_results]))
    macro_disag_med = float(np.mean([r["disagreement"]["valid_pixels"]["median"] for r in per_sample_results]))
    macro_corr_fns_p = float(np.mean([r["correlations_valid"]["disagreement_vs_flownet_error"]["pearson_r"] for r in per_sample_results]))
    macro_corr_fns_s = float(np.mean([r["correlations_valid"]["disagreement_vs_flownet_error"]["spearman_rho"] for r in per_sample_results]))
    macro_corr_raft_p = float(np.mean([r["correlations_valid"]["disagreement_vs_raft_error"]["pearson_r"] for r in per_sample_results]))
    macro_corr_raft_s = float(np.mean([r["correlations_valid"]["disagreement_vs_raft_error"]["spearman_rho"] for r in per_sample_results]))

    print("\n==================================================================")
    print("                     EXPERIMENT SUMMARY                           ")
    print("==================================================================")
    print(f"Total Evaluated Pixels (Valid): {len(cat_d_valid):,}")
    print(f"Macro FlowNetS EPE:             {macro_mean_epe_fns:.4f} px")
    print(f"Macro RAFT EPE:                 {macro_mean_epe_raft:.4f} px")
    print(f"Macro Mean Disagreement:        {macro_disag_mean:.4f} px (Median: {macro_disag_med:.4f} px)")
    print(f"Macro D vs E_FNS Pearson r:     {macro_corr_fns_p:.4f} (Spearman rho: {macro_corr_fns_s:.4f})")
    print(f"Macro D vs E_RAFT Pearson r:    {macro_corr_raft_p:.4f} (Spearman rho: {macro_corr_raft_s:.4f})")
    print(f"Pooled D vs E_FNS Pearson r:    {pooled_corr_fns_p:.4f} (Spearman rho: {pooled_corr_fns_s:.4f})")
    print(f"Pooled D vs E_RAFT Pearson r:   {pooled_corr_raft_p:.4f} (Spearman rho: {pooled_corr_raft_s:.4f})")
    print("==================================================================\n")

    # Save compact machine-readable JSON summary
    json_summary = {
        "metadata": {
            "num_samples": len(DEFAULT_SAMPLES),
            "total_valid_pixels_pooled": int(cat_d_valid.size),
            "samples": DEFAULT_SAMPLES,
            "flownet_checkpoint": args.flownet_checkpoint,
            "device": str(device),
        },
        "macro_summary": {
            "flownet_mean_epe_px": macro_mean_epe_fns,
            "raft_mean_epe_px": macro_mean_epe_raft,
            "disagreement_mean_px": macro_disag_mean,
            "disagreement_median_px": macro_disag_med,
            "disagreement_vs_flownet_pearson_r": macro_corr_fns_p,
            "disagreement_vs_flownet_spearman_rho": macro_corr_fns_s,
            "disagreement_vs_raft_pearson_r": macro_corr_raft_p,
            "disagreement_vs_raft_spearman_rho": macro_corr_raft_s,
        },
        "pooled_correlations": {
            "disagreement_vs_flownet_error": {
                "pearson_r": float(pooled_corr_fns_p),
                "spearman_rho": float(pooled_corr_fns_s),
            },
            "disagreement_vs_raft_error": {
                "pearson_r": float(pooled_corr_raft_p),
                "spearman_rho": float(pooled_corr_raft_s),
            },
            "flownet_error_vs_raft_error": {
                "pearson_r": float(pooled_corr_models_p),
                "spearman_rho": float(pooled_corr_models_s),
            },
        },
        "per_sample_results": per_sample_results,
    }

    json_path = output_dir / "day4_multipair_disagreement.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary saved to: {json_path}")

    # Generate 2x3 Diagnostic Visualization
    labels = [f"{r['scene']}\n({r['pass_name']})" for r in per_sample_results]
    x = np.arange(len(labels))
    width = 0.26

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "Day 4 Multi-Pair Optical Flow Disagreement & Error Analysis (8 Sintel Samples)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Per-Sample Model EPE vs Mean Disagreement
    fns_epes = [r["flownet_s"]["mean_epe_valid_px"] for r in per_sample_results]
    raft_epes = [r["raft"]["mean_epe_valid_px"] for r in per_sample_results]
    disag_means = [r["disagreement"]["valid_pixels"]["mean"] for r in per_sample_results]

    axes[0, 0].bar(x - width, fns_epes, width, label="FlowNetS EPE", color="#ff7f0e", alpha=0.9)
    axes[0, 0].bar(x, raft_epes, width, label="RAFT EPE", color="#2ca02c", alpha=0.9)
    axes[0, 0].bar(x + width, disag_means, width, label="Mean Disagreement D", color="#1f77b4", alpha=0.9)
    axes[0, 0].set_title("1. Model EPE vs. Mean Disagreement (Valid Pixels)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_ylabel("Error / Disagreement (px)", fontsize=10)
    axes[0, 0].legend(loc="upper left", fontsize=9)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 2: Correlation Coefficients Across Samples
    corr_fns_pearson = [r["correlations_valid"]["disagreement_vs_flownet_error"]["pearson_r"] for r in per_sample_results]
    corr_fns_spearman = [r["correlations_valid"]["disagreement_vs_flownet_error"]["spearman_rho"] for r in per_sample_results]
    corr_raft_pearson = [r["correlations_valid"]["disagreement_vs_raft_error"]["pearson_r"] for r in per_sample_results]
    corr_raft_spearman = [r["correlations_valid"]["disagreement_vs_raft_error"]["spearman_rho"] for r in per_sample_results]

    w2 = 0.20
    axes[0, 1].bar(x - 1.5 * w2, corr_fns_pearson, w2, label="D vs E_FNS (Pearson r)", color="#d62728", alpha=0.85)
    axes[0, 1].bar(x - 0.5 * w2, corr_fns_spearman, w2, label="D vs E_FNS (Spearman rho)", color="#e377c2", alpha=0.85)
    axes[0, 1].bar(x + 0.5 * w2, corr_raft_pearson, w2, label="D vs E_RAFT (Pearson r)", color="#17becf", alpha=0.85)
    axes[0, 1].bar(x + 1.5 * w2, corr_raft_spearman, w2, label="D vs E_RAFT (Spearman rho)", color="#9467bd", alpha=0.85)
    axes[0, 1].set_title("2. Correlation of Disagreement with Model Errors", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].set_ylabel("Correlation Coefficient", fontsize=10)
    axes[0, 1].set_ylim(-0.1, 1.05)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 3: Disagreement Distribution Percentiles Across Samples
    p50 = [r["disagreement"]["valid_pixels"]["median"] for r in per_sample_results]
    p75 = [r["disagreement"]["valid_pixels"]["p75"] for r in per_sample_results]
    p90 = [r["disagreement"]["valid_pixels"]["p90"] for r in per_sample_results]
    p95 = [r["disagreement"]["valid_pixels"]["p95"] for r in per_sample_results]

    axes[0, 2].plot(x, p50, marker="o", linewidth=1.8, label="50th %ile (Median)", color="#1f77b4")
    axes[0, 2].plot(x, p75, marker="s", linewidth=1.8, label="75th %ile", color="#2ca02c")
    axes[0, 2].plot(x, p90, marker="^", linewidth=1.8, label="90th %ile", color="#ff7f0e")
    axes[0, 2].plot(x, p95, marker="D", linewidth=1.8, label="95th %ile", color="#d62728")
    axes[0, 2].set_title("3. Disagreement Percentile Profiles (Valid Pixels)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, fontsize=9)
    axes[0, 2].set_ylabel("Disagreement D (px)", fontsize=10)
    axes[0, 2].legend(loc="upper left", fontsize=9)
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Disagreement in Valid vs. Invalid Pixels
    disag_valid_mean = [r["disagreement"]["valid_pixels"]["mean"] for r in per_sample_results]
    disag_invalid_mean = [r["disagreement"]["invalid_pixels"]["mean"] for r in per_sample_results]

    axes[1, 0].bar(x - 0.18, disag_valid_mean, 0.36, label="Valid Pixels", color="#2ca02c", alpha=0.85)
    axes[1, 0].bar(x + 0.18, disag_invalid_mean, 0.36, label="Invalid Pixels", color="#d62728", alpha=0.85)
    axes[1, 0].set_title("4. Mean Disagreement: Valid vs. Invalid Pixels", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].set_ylabel("Mean Disagreement D (px)", fontsize=10)
    axes[1, 0].legend(loc="upper left", fontsize=9)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 5: Pooled D vs FlowNetS Error Scatter
    plot_d = np.concatenate(scatter_d)
    plot_e_fns = np.concatenate(scatter_e_fns)
    plot_e_raft = np.concatenate(scatter_e_raft)

    # Cap display range at 99.5th percentile of pooled for clean visualization
    max_d_vis = float(np.percentile(plot_d, 99.5))
    max_e_fns_vis = float(np.percentile(plot_e_fns, 99.5))
    max_e_raft_vis = float(np.percentile(plot_e_raft, 99.5))

    axes[1, 1].scatter(plot_d, plot_e_fns, alpha=0.20, s=8, color="#ff7f0e", edgecolors="none")
    diag_line = np.linspace(0, min(max_d_vis, max_e_fns_vis), 100)
    axes[1, 1].plot(diag_line, diag_line, "k--", linewidth=1.2, label="Identity (y = x)")
    axes[1, 1].set_xlim(0, max_d_vis)
    axes[1, 1].set_ylim(0, max_e_fns_vis)
    axes[1, 1].set_title(
        f"5. Pooled D vs. FlowNetS Error ({len(plot_d):,} pts)\nPearson r = {pooled_corr_fns_p:.3f} | Spearman rho = {pooled_corr_fns_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Disagreement D(x, y) (px)", fontsize=10)
    axes[1, 1].set_ylabel("E_FlowNetS = ||F_FNS - GT||_2 (px)", fontsize=10)
    axes[1, 1].legend(loc="upper left", fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 6: Pooled D vs RAFT Error Scatter
    axes[1, 2].scatter(plot_d, plot_e_raft, alpha=0.20, s=8, color="#2ca02c", edgecolors="none")
    axes[1, 2].set_xlim(0, max_d_vis)
    axes[1, 2].set_ylim(0, max_e_raft_vis)
    axes[1, 2].set_title(
        f"6. Pooled D vs. RAFT Error ({len(plot_d):,} pts)\nPearson r = {pooled_corr_raft_p:.3f} | Spearman rho = {pooled_corr_raft_s:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("Disagreement D(x, y) (px)", fontsize=10)
    axes[1, 2].set_ylabel("E_RAFT = ||F_RAFT - GT||_2 (px)", fontsize=10)
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = output_dir / "day4_multipair_disagreement.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Aggregate visualization saved to: {fig_path}")


if __name__ == "__main__":
    main()
