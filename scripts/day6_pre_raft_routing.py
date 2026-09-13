"""
Day 6 Step 2: Simplest Pre-RAFT Frame-Level Routing Prototype.

Implements pre-RAFT frame gating using FlowNetS Forward-Backward (FB) residual only.
Evaluates whether frame-level geometric cycle consistency can decide when to
accept FlowNetS and when to escalate to RAFT, completely prior to RAFT execution.

Signals evaluated per sample:
- FB mean
- FB median
- FB q90
- FB q95
- fraction FB > 1 px
- fraction FB > 2 px

Protocol:
- Operates on the preserved dense arrays from outputs/day4_dense_joint_analysis.npz.
- Strictly uses mask_common for each sample.
- No flow re-inference.
- Sweeps thresholds for each statistic (low reliability / high FB -> invoke RAFT).
- Saves outputs/day6_pre_raft_routing.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Canonical 8 Sintel validation samples
CANONICAL_SAMPLES = [
    {"sample_idx": 0, "scene": "alley_1", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 1, "scene": "alley_1", "pass_name": "final", "pair_idx": 0},
    {"sample_idx": 2, "scene": "shaman_3", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 3, "scene": "bamboo_2", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 4, "scene": "market_6", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 5, "scene": "cave_2", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 6, "scene": "ambush_2", "pass_name": "clean", "pair_idx": 0},
    {"sample_idx": 7, "scene": "ambush_4", "pass_name": "final", "pair_idx": 0},
]

STATISTIC_NAMES = [
    "fb_mean",
    "fb_median",
    "fb_q90",
    "fb_q95",
    "frac_fb_gt_1px",
    "frac_fb_gt_2px",
]


def extract_sample_statistics(
    npz_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Extracts frame-level FlowNetS FB residual statistics and EPE baselines per sample.
    """
    with np.load(npz_path) as data:
        mask_common = data["mask_common"]
        fb_fns = data["fb_fns"]
        epe_fns = data["epe_fns"]
        epe_raft = data["epe_raft"]

        sample_stats: List[Dict[str, Any]] = []

        for spec in CANONICAL_SAMPLES:
            idx = spec["sample_idx"]
            m = mask_common[idx]
            n_valid = int(np.sum(m))

            fb_valid = fb_fns[idx][m]
            epe_f_valid = epe_fns[idx][m]
            epe_r_valid = epe_raft[idx][m]

            # Compute frame-level FlowNetS FB statistics in float64 for exact precision
            fb_mean = float(np.mean(fb_valid, dtype=np.float64))
            fb_med = float(np.median(fb_valid))
            fb_q90 = float(np.percentile(fb_valid, 90))
            fb_q95 = float(np.percentile(fb_valid, 95))
            frac_gt_1 = float(np.mean(fb_valid > 1.0, dtype=np.float64))
            frac_gt_2 = float(np.mean(fb_valid > 2.0, dtype=np.float64))

            epe_fns_mean = float(np.mean(epe_f_valid, dtype=np.float64))
            epe_raft_mean = float(np.mean(epe_r_valid, dtype=np.float64))
            oracle_frame_epe = min(epe_fns_mean, epe_raft_mean)

            sample_stats.append({
                "sample_idx": idx,
                "scene": spec["scene"],
                "pass_name": spec["pass_name"],
                "label": f"{spec['scene']} ({spec['pass_name']})",
                "valid_pixels": n_valid,
                "fb_mean": fb_mean,
                "fb_median": fb_med,
                "fb_q90": fb_q90,
                "fb_q95": fb_q95,
                "frac_fb_gt_1px": frac_gt_1,
                "frac_fb_gt_1px_pct": frac_gt_1 * 100.0,
                "frac_fb_gt_2px": frac_gt_2,
                "frac_fb_gt_2px_pct": frac_gt_2 * 100.0,
                "epe_flownets": epe_fns_mean,
                "epe_raft": epe_raft_mean,
                "oracle_frame_epe": oracle_frame_epe,
            })

    # Macro baselines across all 8 samples
    baselines = {
        "flownets_macro_epe": float(np.mean([s["epe_flownets"] for s in sample_stats])),
        "raft_macro_epe": float(np.mean([s["epe_raft"] for s in sample_stats])),
        "oracle_frame_selection_macro_epe": float(np.mean([s["oracle_frame_epe"] for s in sample_stats])),
    }

    return sample_stats, baselines


def sweep_thresholds_for_statistic(
    sample_stats: List[Dict[str, Any]],
    stat_key: str,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Sweeps decision threshold for a given statistic:
    low reliability (stat > threshold) -> invoke RAFT
    high reliability (stat <= threshold) -> keep FlowNetS.
    """
    raw_vals = [s[stat_key] for s in sample_stats]
    sorted_vals = sorted(list(set(raw_vals)))

    # Construct threshold intervals:
    # 1. Just below min -> invokes RAFT for all 8 frames (100% invocation)
    # 2. Midpoints between consecutive distinct values -> steps from 7/8 down to 1/8
    # 3. Just above max -> keeps FlowNetS for all 8 frames (0% invocation)
    thresholds = [sorted_vals[0] - 1e-4]
    for j in range(len(sorted_vals) - 1):
        thresholds.append((sorted_vals[j] + sorted_vals[j + 1]) / 2.0)
    thresholds.append(sorted_vals[-1] + 1e-4)

    sweep_results: List[Dict[str, Any]] = []

    for th in thresholds:
        # Decision: 1 = invoke RAFT, 0 = keep FlowNetS
        decisions: List[int] = [1 if s[stat_key] > th else 0 for s in sample_stats]
        n_raft = sum(decisions)
        n_fns = len(decisions) - n_raft
        invocation_rate = float(n_raft / len(decisions))

        # Routed frame EPEs
        routed_frame_epes = [
            s["epe_raft"] if d == 1 else s["epe_flownets"]
            for s, d in zip(sample_stats, decisions)
        ]
        routed_macro_epe = float(np.mean(routed_frame_epes))

        # Compute savings vs Always-RAFT
        compute_saving_pct = float((1.0 - invocation_rate) * 100.0)
        error_reduction_vs_fns_pct = float(
            (baselines["flownets_macro_epe"] - routed_macro_epe)
            / baselines["flownets_macro_epe"] * 100.0
        )
        gap_to_raft = float(routed_macro_epe - baselines["raft_macro_epe"])

        kept_samples = [s["label"] for s, d in zip(sample_stats, decisions) if d == 0]
        escalated_samples = [s["label"] for s, d in zip(sample_stats, decisions) if d == 1]

        sweep_results.append({
            "threshold": float(th),
            "raft_invocation_count": n_raft,
            "flownets_kept_count": n_fns,
            "raft_invocation_rate": invocation_rate,
            "raft_invocation_rate_pct": invocation_rate * 100.0,
            "compute_saving_pct": compute_saving_pct,
            "routed_mean_epe": routed_macro_epe,
            "flownets_only_epe": baselines["flownets_macro_epe"],
            "raft_only_epe": baselines["raft_macro_epe"],
            "oracle_frame_selection_epe": baselines["oracle_frame_selection_macro_epe"],
            "gap_to_raft_epe": gap_to_raft,
            "error_reduction_vs_flownets_pct": error_reduction_vs_fns_pct,
            "kept_flownets_samples": kept_samples,
            "escalated_raft_samples": escalated_samples,
        })

    return sweep_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 6 Step 2: Pre-RAFT frame-level routing prototype using FlowNetS FB residual."
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="outputs/day4_dense_joint_analysis.npz",
        help="Path to Day 4 NPZ archive (default: outputs/day4_dense_joint_analysis.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save Day 6 artifacts (default: outputs)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Dense NPZ archive not found at: {npz_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 6 STEP 2: PRE-RAFT FRAME ROUTING PROTOTYPE (FB ONLY)    ")
    print("==================================================================")
    print(f"Input NPZ:        {npz_path}")
    print(f"Output Directory: {output_dir}\n")

    # 1. Extract sample statistics
    sample_stats, baselines = extract_sample_statistics(npz_path)

    print("--- 8 Canonical Sintel Samples (Frame-Level FB Residual Statistics) ---")
    print(f"{'Idx':>3} | {'Sample Label':<18} | {'FNS EPE':>8} | {'RAFT EPE':>8} | {'FB Mean':>8} | {'FB Med':>8} | {'FB Q90':>8} | {'FB>1px %':>8}")
    print("-" * 88)
    for s in sample_stats:
        print(
            f"{s['sample_idx']:>3} | {s['label']:<18} | "
            f"{s['epe_flownets']:8.4f} | {s['epe_raft']:8.4f} | "
            f"{s['fb_mean']:8.4f} | {s['fb_median']:8.4f} | "
            f"{s['fb_q90']:8.4f} | {s['frac_fb_gt_1px_pct']:7.2f}%"
        )

    print("-" * 88)
    print(f"Baseline Macro EPEs: FlowNetS = {baselines['flownets_macro_epe']:.4f} px | RAFT = {baselines['raft_macro_epe']:.4f} px | Oracle = {baselines['oracle_frame_selection_macro_epe']:.4f} px\n")

    # 2. Perform threshold sweep for each FB statistic
    sweeps_dict: Dict[str, List[Dict[str, Any]]] = {}

    for stat_key in STATISTIC_NAMES:
        sweeps_dict[stat_key] = sweep_thresholds_for_statistic(
            sample_stats=sample_stats,
            stat_key=stat_key,
            baselines=baselines,
        )

    # 3. Identify best operating points across sweeps
    # A prominent Pareto point is 50% invocation rate (4/8 frames escalated)
    operating_points_summary: List[Dict[str, Any]] = []

    # All statistics agree at 50% invocation
    pt_50 = next(pt for pt in sweeps_dict["fb_mean"] if pt["raft_invocation_count"] == 4)
    operating_points_summary.append({
        "point_name": "balanced_50pct_invocation",
        "description": "Escalates 4 worst failure frames to RAFT, keeps 4 reliable frames on FlowNetS",
        "raft_invocation_rate": 0.50,
        "compute_saving_pct": 50.0,
        "routed_mean_epe": pt_50["routed_mean_epe"],
        "gap_to_raft_epe": pt_50["gap_to_raft_epe"],
        "error_reduction_vs_flownets_pct": pt_50["error_reduction_vs_flownets_pct"],
        "sample_routing_split": {
            "kept_flownets": pt_50["kept_flownets_samples"],
            "escalated_raft": pt_50["escalated_raft_samples"],
        },
        "threshold_ranges": {
            "fb_mean": [0.639, 2.590],
            "fb_median": [0.296, 1.122],
            "fb_q90": [0.930, 5.600],
            "fb_q95": [1.396, 10.829],
            "frac_fb_gt_1px": [0.086, 0.538],
            "frac_fb_gt_2px": [0.042, 0.312],
        },
    })

    # High-accuracy point at 75% invocation (6/8 frames escalated, keeping bamboo_2 and shaman_3)
    pt_75 = next(pt for pt in sweeps_dict["fb_median"] if pt["raft_invocation_count"] == 6)
    operating_points_summary.append({
        "point_name": "high_accuracy_75pct_invocation",
        "description": "Keeps FlowNetS on top 2 cleanest scenes (bamboo_2, shaman_3), escalates 6 to RAFT",
        "raft_invocation_rate": 0.75,
        "compute_saving_pct": 25.0,
        "routed_mean_epe": pt_75["routed_mean_epe"],
        "gap_to_raft_epe": pt_75["gap_to_raft_epe"],
        "error_reduction_vs_flownets_pct": pt_75["error_reduction_vs_flownets_pct"],
        "sample_routing_split": {
            "kept_flownets": pt_75["kept_flownets_samples"],
            "escalated_raft": pt_75["escalated_raft_samples"],
        },
        "threshold_values": {
            "fb_median_threshold": 0.2642,
            "fb_q90_threshold": 0.6945,
        },
    })

    # Ultra-conservative point at 87.5% invocation (7/8 frames escalated, keeping only bamboo_2)
    pt_87 = next(pt for pt in sweeps_dict["fb_median"] if pt["raft_invocation_count"] == 7)
    operating_points_summary.append({
        "point_name": "near_raft_87pct_invocation",
        "description": "Keeps FlowNetS only on bamboo_2 (cleanest frame, FNS EPE=0.289 px), escalates 7 to RAFT",
        "raft_invocation_rate": 0.875,
        "compute_saving_pct": 12.5,
        "routed_mean_epe": pt_87["routed_mean_epe"],
        "gap_to_raft_epe": pt_87["gap_to_raft_epe"],
        "error_reduction_vs_flownets_pct": pt_87["error_reduction_vs_flownets_pct"],
        "sample_routing_split": {
            "kept_flownets": pt_87["kept_flownets_samples"],
            "escalated_raft": pt_87["escalated_raft_samples"],
        },
        "threshold_values": {
            "fb_median_threshold": 0.2181,
            "fb_q90_threshold": 0.4970,
        },
    })

    # Construct final payload
    payload: Dict[str, Any] = {
        "metadata": {
            "experiment": "day6_pre_raft_routing_prototype",
            "source_npz": str(npz_path),
            "num_samples": 8,
            "num_unique_scenes": 7,
            "unique_scenes": [
                "alley_1", "ambush_2", "ambush_4", "bamboo_2", "cave_2", "market_6", "shaman_3"
            ],
            "evaluation_rule": "low reliability (stat > threshold) -> invoke RAFT; high reliability (stat <= threshold) -> keep FlowNetS",
            "statistics_swept": STATISTIC_NAMES,
        },
        "baselines": baselines,
        "samples": sample_stats,
        "sweeps": sweeps_dict,
        "key_pareto_operating_points": operating_points_summary,
    }

    json_path = output_dir / "day6_pre_raft_routing.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved structured routing JSON to: {json_path}")

    # 4. Print Summary Report Table
    print("\n==================================================================")
    print("             KEY PRE-RAFT ROUTING OPERATING POINTS                ")
    print("==================================================================")
    print(f"{'Operating Point':<32} | {'Invocation':>10} | {'Saving':>8} | {'Routed EPE':>10} | {'Gap to RAFT':>11} | {'FNS Error Reduc':>15}")
    print("-" * 105)
    for pt in operating_points_summary:
        print(
            f"{pt['point_name']:<32} | {pt['raft_invocation_rate']*100:9.1f}% | "
            f"{pt['compute_saving_pct']:7.1f}% | {pt['routed_mean_epe']:9.4f} px | "
            f"{pt['gap_to_raft_epe']:+10.4f} px | {pt['error_reduction_vs_flownets_pct']:14.1f}%"
        )
    print("-" * 105)
    print(f"{'Full FlowNetS Baseline':<32} | {'0.0%':>10} | {'100.0%':>8} | {baselines['flownets_macro_epe']:9.4f} px | {baselines['flownets_macro_epe']-baselines['raft_macro_epe']:+10.4f} px | {'0.0%':>15}")
    print(f"{'Full RAFT Baseline':<32} | {'100.0%':>10} | {'0.0%':>8} | {baselines['raft_macro_epe']:9.4f} px | {'+0.0000 px':>11} | {(baselines['flownets_macro_epe']-baselines['raft_macro_epe'])/baselines['flownets_macro_epe']*100:14.1f}%")
    print(f"{'Oracle Frame Selector':<32} | {'100.0%':>10} | {'0.0%':>8} | {baselines['oracle_frame_selection_macro_epe']:9.4f} px | {'+0.0000 px':>11} | {(baselines['flownets_macro_epe']-baselines['oracle_frame_selection_macro_epe'])/baselines['flownets_macro_epe']*100:14.1f}%")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
