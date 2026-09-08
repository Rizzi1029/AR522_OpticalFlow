"""
Day 4 Prototype: Dense Model Disagreement Analysis on a Single Sintel Frame Pair.

Evaluates FlowNetS and RAFT on a single MPI Sintel frame pair (default: scene=alley_1,
pass=clean, pair index 0) to:
1. Compute dense model disagreement: D(x, y) = ||F_FlowNetS(x, y) - F_RAFT(x, y)||_2
2. Compute actual model errors against Sintel ground truth:
   E_FNS(x, y)  = ||F_FlowNetS(x, y) - F_GT(x, y)||_2
   E_RAFT(x, y) = ||F_RAFT(x, y)     - F_GT(x, y)||_2
3. Measure flow distribution statistics and exploratory correlation summaries
   (Pearson r, Spearman rho) over valid pixels without forming confidence heuristics.
4. Render an 8-panel diagnostic visualization figure.
5. Manage memory strictly via explicit tensor deletion without torch.cuda.empty_cache().
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from torchvision.utils import flow_to_image

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator


def verify_disagreement_math() -> None:
    """Smallest synthetic unit verification for disagreement computation."""
    # Test 1: Identical flow fields -> disagreement is identically 0.0
    f1 = torch.ones((2, 10, 10), dtype=torch.float32)
    f2 = torch.ones((2, 10, 10), dtype=torch.float32)
    diff = f1 - f2
    d_ident = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)
    assert torch.allclose(d_ident, torch.zeros_like(d_ident)), "Identical flow test failed"

    # Test 2: Known orthogonal offset (dx=3, dy=4) -> Euclidean distance is exactly 5.0
    f_offset = f1.clone()
    f_offset[0] += 3.0
    f_offset[1] += 4.0
    diff_off = f_offset - f1
    d_offset = torch.sqrt(diff_off[0] ** 2 + diff_off[1] ** 2)
    assert torch.allclose(d_offset, torch.full_like(d_offset, 5.0)), "Known offset (3, 4) test failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Single-pair dense flow collection and disagreement analysis."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Sintel",
        help="Path to Sintel dataset root (default: data/Sintel)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="alley_1",
        help="Sintel scene name (default: alley_1)",
    )
    parser.add_argument(
        "--pass_name",
        type=str,
        choices=["clean", "final"],
        default="clean",
        help="Sintel rendering pass (default: clean)",
    )
    parser.add_argument(
        "--pair_idx",
        type=int,
        default=0,
        help="Consecutive frame pair index within scene (default: 0)",
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
    """Compute summary statistics for a 1D or 2D numerical array."""
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
    # 0. Sanity-check math before running
    verify_disagreement_math()

    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 PROTOTYPE: SINGLE-PAIR FLOW DISAGREEMENT ANALYSIS     ")
    print("==================================================================")
    print(f"Device:           {device}")
    print(f"Dataset Pass:     {args.pass_name}")
    print(f"Scene Filter:     {args.scene}")
    print(f"Pair Index:       {args.pair_idx}")
    print(f"Output Directory: {output_dir}")
    print("==================================================================\n")

    # 1. Load single sample pair explicitly via SintelDataset loader
    dataset = SintelDataset(
        root=args.data_root,
        split="training",
        pass_name=args.pass_name,
        scenes=[args.scene],
    )
    if args.pair_idx >= len(dataset):
        raise IndexError(f"Requested pair_idx {args.pair_idx} out of range ({len(dataset)} pairs)")

    frame1, frame2, flow_gt, valid_mask, meta = dataset[args.pair_idx]
    orig_h, orig_w = frame1.shape[1], frame1.shape[2]
    total_pixels = orig_h * orig_w
    valid_count = int(valid_mask.sum().item())
    invalid_count = total_pixels - valid_count

    print(f"Loaded Sample: Scene '{meta['scene']}', Frame {meta['frame_idx']} -> {meta['frame_idx'] + 1}")
    print(f"Spatial Dimensions: {orig_h} x {orig_w} ({total_pixels} total pixels)")
    print(f"Valid Pixels:       {valid_count} ({valid_count / total_pixels * 100:.2f}%)")
    print(f"Invalid Pixels:     {invalid_count} ({invalid_count / total_pixels * 100:.2f}%)\n")

    # 2. Build model adapters
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

    # 3. Dense Flow Inference (Single Frame Pair)
    print("\nRunning FlowNetS inference...")
    with torch.inference_mode():
        flow_fns_raw = flownet_model(frame1, frame2)  # [1, 2, H, W]
    flow_fns = flow_fns_raw[0]  # [2, H, W] on device
    del flow_fns_raw

    print("Running RAFT inference...")
    with torch.inference_mode():
        flow_raft_raw = raft_model(frame1, frame2)  # [1, 2, H, W]
    flow_raft = flow_raft_raw[0]  # [2, H, W] on device
    del flow_raft_raw

    # 4. Compute Dense Disagreement & Ground-Truth Errors
    # D(x, y) = ||F_FlowNetS(x, y) - F_RAFT(x, y)||_2
    diff_models = flow_fns - flow_raft
    disagreement_t = torch.sqrt(diff_models[0] ** 2 + diff_models[1] ** 2)  # [H, W] on device
    del diff_models

    # Actual errors against Ground Truth on device
    flow_gt_dev = flow_gt.to(device)
    diff_fns = flow_fns - flow_gt_dev
    err_fns_t = torch.sqrt(diff_fns[0] ** 2 + diff_fns[1] ** 2)  # [H, W] on device
    del diff_fns

    diff_raft = flow_raft - flow_gt_dev
    err_raft_t = torch.sqrt(diff_raft[0] ** 2 + diff_raft[1] ** 2)  # [H, W] on device
    del diff_raft

    # Magnitudes of predicted flows
    mag_fns_t = torch.sqrt(flow_fns[0] ** 2 + flow_fns[1] ** 2)
    mag_raft_t = torch.sqrt(flow_raft[0] ** 2 + flow_raft[1] ** 2)
    mag_gt_t = torch.sqrt(flow_gt_dev[0] ** 2 + flow_gt_dev[1] ** 2)

    # 5. Extract Visualizations for Color-Wheel Rendering
    vis_fns_rgb = flow_to_image(flow_fns).permute(1, 2, 0).cpu().numpy()  # [H, W, 3] uint8
    vis_raft_rgb = flow_to_image(flow_raft).permute(1, 2, 0).cpu().numpy()
    vis_gt_rgb = flow_to_image(flow_gt_dev).permute(1, 2, 0).cpu().numpy()
    frame1_rgb = frame1.permute(1, 2, 0).cpu().numpy()

    # Move dense arrays to NumPy on CPU for statistical processing
    disagreement_np = disagreement_t.cpu().numpy()
    err_fns_np = err_fns_t.cpu().numpy()
    err_raft_np = err_raft_t.cpu().numpy()
    mag_fns_np = mag_fns_t.cpu().numpy()
    mag_raft_np = mag_raft_t.cpu().numpy()
    mag_gt_np = mag_gt_t.cpu().numpy()
    u_fns_np = flow_fns[0].cpu().numpy()
    v_fns_np = flow_fns[1].cpu().numpy()
    u_raft_np = flow_raft[0].cpu().numpy()
    v_raft_np = flow_raft[1].cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()
    invalid_mask_np = ~valid_mask_np

    # Explicitly delete GPU tensors to guarantee flat memory usage (NO empty_cache)
    del flow_fns, flow_raft, flow_gt_dev
    del disagreement_t, err_fns_t, err_raft_t, mag_fns_t, mag_raft_t, mag_gt_t

    # 6. Extract Statistics & Exploratory Error Correlation
    # Disagreement subsets
    d_all = disagreement_np.flatten()
    d_valid = disagreement_np[valid_mask_np]
    d_invalid = disagreement_np[invalid_mask_np]

    e_fns_valid = err_fns_np[valid_mask_np]
    e_raft_valid = err_raft_np[valid_mask_np]

    stats_d_all = compute_distribution_stats(d_all)
    stats_d_valid = compute_distribution_stats(d_valid)
    stats_d_invalid = compute_distribution_stats(d_invalid)

    # Correlation between model disagreement and actual model errors over valid pixels
    corr_fns_p, _ = scipy.stats.pearsonr(d_valid, e_fns_valid)
    corr_fns_s, _ = scipy.stats.spearmanr(d_valid, e_fns_valid)

    corr_raft_p, _ = scipy.stats.pearsonr(d_valid, e_raft_valid)
    corr_raft_s, _ = scipy.stats.spearmanr(d_valid, e_raft_valid)

    corr_models_p, _ = scipy.stats.pearsonr(e_fns_valid, e_raft_valid)
    corr_models_s, _ = scipy.stats.spearmanr(e_fns_valid, e_raft_valid)

    mean_epe_fns = float(np.mean(e_fns_valid))
    mean_epe_raft = float(np.mean(e_raft_valid))

    print("--- Flow & Disagreement Summary ---")
    print(f"FlowNetS Mean EPE (valid):   {mean_epe_fns:.4f} px")
    print(f"RAFT Mean EPE (valid):       {mean_epe_raft:.4f} px")
    print(f"Disagreement Mean (all):     {stats_d_all['mean']:.4f} px (median: {stats_d_all['median']:.4f} px)")
    print(f"Disagreement Mean (valid):   {stats_d_valid['mean']:.4f} px (median: {stats_d_valid['median']:.4f} px)")
    print(f"Disagreement Mean (invalid): {stats_d_invalid['mean']:.4f} px (median: {stats_d_invalid['median']:.4f} px)")
    print(f"Disagreement 95th %ile:      {stats_d_valid['p95']:.4f} px")
    print(f"Disagreement > 1.0 px:       {stats_d_valid['fraction_gt_1px'] * 100:.2f}% (valid pixels)")
    print(f"Disagreement > 3.0 px:       {stats_d_valid['fraction_gt_3px'] * 100:.2f}% (valid pixels)\n")

    print("--- Exploratory Error Correlation (Valid Pixels) ---")
    print(f"D(x, y) vs E_FlowNetS: Pearson r = {corr_fns_p:.4f}, Spearman rho = {corr_fns_s:.4f}")
    print(f"D(x, y) vs E_RAFT:     Pearson r = {corr_raft_p:.4f}, Spearman rho = {corr_raft_s:.4f}")
    print(f"E_FlowNetS vs E_RAFT:  Pearson r = {corr_models_p:.4f}, Spearman rho = {corr_models_s:.4f}\n")

    # 7. Render 8-Panel Diagnostic Figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        f"Day 4 Optical Flow Disagreement Analysis: {meta['scene']} (Frame {meta['frame_idx']:04d} -> {meta['frame_idx'] + 1:04d}, {meta['pass_name']} pass)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Frame 1
    axes[0, 0].imshow(frame1_rgb)
    axes[0, 0].set_title(f"Input Frame 1 ({meta['scene']} #{meta['frame_idx']:04d})", fontsize=11)
    axes[0, 0].axis("off")

    # Panel 2: FlowNetS Flow
    axes[0, 1].imshow(vis_fns_rgb)
    axes[0, 1].set_title(f"FlowNetS Flow (EPE: {mean_epe_fns:.3f} px)", fontsize=11)
    axes[0, 1].axis("off")

    # Panel 3: RAFT Flow
    axes[0, 2].imshow(vis_raft_rgb)
    axes[0, 2].set_title(f"RAFT Flow (EPE: {mean_epe_raft:.3f} px)", fontsize=11)
    axes[0, 2].axis("off")

    # Panel 4: Ground Truth Flow
    axes[0, 3].imshow(vis_gt_rgb)
    axes[0, 3].set_title("Ground-Truth Flow (Sintel)", fontsize=11)
    axes[0, 3].axis("off")

    # Panel 5: Disagreement Heatmap with Pixel Colorbar
    vmax_disag = max(float(stats_d_valid["p99"]), 5.0)
    im5 = axes[1, 0].imshow(disagreement_np, cmap="turbo", vmin=0.0, vmax=vmax_disag)
    axes[1, 0].set_title(f"Disagreement D(x, y) = ||F_FNS - F_RAFT||_2 (px)\n(Mean: {stats_d_valid['mean']:.2f} px, Med: {stats_d_valid['median']:.2f} px)", fontsize=11)
    axes[1, 0].axis("off")
    cbar5 = fig.colorbar(im5, ax=axes[1, 0], orientation="horizontal", fraction=0.046, pad=0.08)
    cbar5.set_label("Displacement Difference (pixels)", fontsize=10)

    # Panel 6: Disagreement Distribution Histogram
    axes[1, 1].hist(d_valid, bins=80, range=(0.0, vmax_disag), color="#1f77b4", alpha=0.85, density=True)
    axes[1, 1].axvline(stats_d_valid["median"], color="black", linestyle="--", linewidth=1.5, label=f"Median: {stats_d_valid['median']:.2f} px")
    axes[1, 1].axvline(stats_d_valid["p95"], color="red", linestyle=":", linewidth=1.5, label=f"95th %ile: {stats_d_valid['p95']:.2f} px")
    axes[1, 1].set_title("Disagreement Distribution (Valid Pixels)", fontsize=11)
    axes[1, 1].set_xlabel("Disagreement (pixels)", fontsize=10)
    axes[1, 1].set_ylabel("Probability Density", fontsize=10)
    axes[1, 1].legend(loc="upper right", fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    # Subsample valid pixels for responsive scatter diagnostics
    rng = np.random.default_rng(42)
    sample_size = min(5000, len(d_valid))
    sample_idx = rng.choice(len(d_valid), size=sample_size, replace=False)
    sub_d = d_valid[sample_idx]
    sub_e_fns = e_fns_valid[sample_idx]
    sub_e_raft = e_raft_valid[sample_idx]

    # Panel 7: D(x, y) vs FlowNetS Error
    axes[1, 2].scatter(sub_d, sub_e_fns, alpha=0.35, s=12, color="#ff7f0e", edgecolors="none")
    axes[1, 2].set_title(f"D(x, y) vs. FlowNetS Error (Valid)\nPearson r = {corr_fns_p:.3f} | Spearman rho = {corr_fns_s:.3f}", fontsize=11)
    axes[1, 2].set_xlabel("Disagreement D(x, y) (px)", fontsize=10)
    axes[1, 2].set_ylabel("E_FlowNetS = ||F_FNS - GT||_2 (px)", fontsize=10)
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)

    # Panel 8: D(x, y) vs RAFT Error
    axes[1, 3].scatter(sub_d, sub_e_raft, alpha=0.35, s=12, color="#2ca02c", edgecolors="none")
    axes[1, 3].set_title(f"D(x, y) vs. RAFT Error (Valid)\nPearson r = {corr_raft_p:.3f} | Spearman rho = {corr_raft_s:.3f}", fontsize=11)
    axes[1, 3].set_xlabel("Disagreement D(x, y) (px)", fontsize=10)
    axes[1, 3].set_ylabel("E_RAFT = ||F_RAFT - GT||_2 (px)", fontsize=10)
    axes[1, 3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    vis_path = output_dir / "day4_single_pair_disagreement.png"
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic visualization saved to: {vis_path}")

    # 8. Save Metrics JSON
    json_results = {
        "metadata": {
            "scene": meta["scene"],
            "pass_name": meta["pass_name"],
            "frame_idx": meta["frame_idx"],
            "frame1_path": meta["frame1_path"],
            "frame2_path": meta["frame2_path"],
            "flow_gt_path": meta["flow_path"],
            "invalid_mask_path": meta["invalid_path"],
            "dimensions": {"height": orig_h, "width": orig_w},
            "pixels": {
                "total": total_pixels,
                "valid": valid_count,
                "invalid": invalid_count,
            },
        },
        "flow_statistics": {
            "flownet_s": {
                "mean_epe_valid_px": mean_epe_fns,
                "magnitude_mean_px": float(np.mean(mag_fns_np)),
                "magnitude_max_px": float(np.max(mag_fns_np)),
                "u_mean_px": float(np.mean(u_fns_np)),
                "v_mean_px": float(np.mean(v_fns_np)),
            },
            "raft": {
                "mean_epe_valid_px": mean_epe_raft,
                "magnitude_mean_px": float(np.mean(mag_raft_np)),
                "magnitude_max_px": float(np.max(mag_raft_np)),
                "u_mean_px": float(np.mean(u_raft_np)),
                "v_mean_px": float(np.mean(v_raft_np)),
            },
            "ground_truth": {
                "magnitude_mean_px": float(np.mean(mag_gt_np[valid_mask_np])),
                "magnitude_max_px": float(np.max(mag_gt_np[valid_mask_np])),
            },
        },
        "disagreement_statistics": {
            "all_pixels": stats_d_all,
            "valid_pixels": stats_d_valid,
            "invalid_pixels": stats_d_invalid,
        },
        "exploratory_error_correlation": {
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

    json_path = output_dir / "day4_single_pair_disagreement.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Analysis metrics saved to:         {json_path}")


if __name__ == "__main__":
    main()
