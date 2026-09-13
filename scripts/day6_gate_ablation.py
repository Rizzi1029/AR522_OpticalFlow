"""
Day 6: Full-Scale Gate-Statistic Ablation on MPI Sintel Training Benchmark.

Compares five forward-only FlowNetS diagnostic routing signals:
1. mag_median: Median of forward flow magnitude
2. mag_mean: Mean of forward flow magnitude
3. mag_q90: 90th percentile of forward flow magnitude
4. mag_q95: 95th percentile of forward flow magnitude
5. frac_mag_gt5: Fraction of valid pixels with forward flow magnitude > 5 px

Protocol:
- Evaluates across all 1041 frame pairs and 23 scenes of MPI Sintel training benchmark (clean pass).
- Evaluates operating targets: 22.01%, 23.0%, 43.0%, 43.43%, 50.0%, and 87.5%.
- Executes strict 23-scene Leave-One-Scene-Out Cross-Validation (LOSO-CV).
- Calibrates threshold tau on training fold (other 22 scenes) as the (1 - rho) quantile of the statistic.
- Zero test-set EPE leakage.
- Measures and reports: actual OOF invocation rate, routed macro/micro EPE, gap to RAFT,
  error reduction vs FlowNetS, and estimated cascade latency (based on measured 13.51 ms gate / 126.92 ms RAFT).
- Saves results to outputs/day6_gate_ablation.json.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator

STATISTIC_KEYS = [
    "mag_median",
    "mag_mean",
    "mag_q90",
    "mag_q95",
    "frac_mag_gt5",
]

TARGET_BUDGETS = [0.2201, 0.23, 0.43, 0.4343, 0.50, 0.875]

GATE_LATENCY_MS = 13.50767240524292   # Forward gate latency (74.0 FPS)
RAFT_LATENCY_MS = 126.92              # Always-RAFT forward pass (7.88 FPS)


def load_benchmark_records(
    flownet_json_path: Path,
    raft_json_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Loads precomputed benchmark EPE and valid pixel records for FlowNetS and RAFT."""
    if not flownet_json_path.is_file():
        raise FileNotFoundError(f"FlowNetS benchmark report not found: {flownet_json_path}")
    if not raft_json_path.is_file():
        raise FileNotFoundError(f"RAFT benchmark report not found: {raft_json_path}")

    with open(flownet_json_path, "r", encoding="utf-8") as f:
        fn_data = json.load(f)
    with open(raft_json_path, "r", encoding="utf-8") as f:
        rf_data = json.load(f)

    fn_records = fn_data["frame_records"]
    rf_records = rf_data["frame_records"]

    if len(fn_records) != len(rf_records):
        raise ValueError(
            f"Record count mismatch: FlowNetS has {len(fn_records)} frames, "
            f"RAFT has {len(rf_records)} frames."
        )

    records: List[Dict[str, Any]] = []
    for i, (fn_rec, rf_rec) in enumerate(zip(fn_records, rf_records)):
        if (fn_rec["scene"], fn_rec["frame_idx"]) != (rf_rec["scene"], rf_rec["frame_idx"]):
            raise ValueError(
                f"Frame record mismatch at index {i}: FlowNetS=({fn_rec['scene']}, {fn_rec['frame_idx']}) "
                f"vs RAFT=({rf_rec['scene']}, {rf_rec['frame_idx']})"
            )
        records.append({
            "scene": fn_rec["scene"],
            "frame_idx": fn_rec["frame_idx"],
            "valid_count": int(fn_rec["valid_count"]),
            "epe_flownets": float(fn_rec["mean_epe"]),
            "epe_sum_flownets": float(fn_rec["epe_sum"]),
            "epe_raft": float(rf_rec["mean_epe"]),
            "epe_sum_raft": float(rf_rec["epe_sum"]),
            "oracle_frame_epe": min(float(fn_rec["mean_epe"]), float(rf_rec["mean_epe"])),
        })

    baselines = {
        "flownets_macro_epe": float(fn_data["overall_summary"]["macro_epe"]),
        "flownets_micro_epe": float(fn_data["overall_summary"]["micro_epe"]),
        "raft_macro_epe": float(rf_data["overall_summary"]["macro_epe"]),
        "raft_micro_epe": float(rf_data["overall_summary"]["micro_epe"]),
        "oracle_macro_epe": float(np.mean([r["oracle_frame_epe"] for r in records])),
        "total_valid_pixels": int(fn_data["overall_summary"]["total_valid_pixels"]),
    }

    return records, baselines


def extract_all_forward_statistics(
    dataset: SintelDataset,
    records: List[Dict[str, Any]],
    flownet_checkpoint: str,
    device: torch.device,
) -> None:
    """
    Runs a single FlowNetS forward pass per frame to compute all five candidate statistics.
    Mutates records in-place.
    """
    flownet_model = build_flow_estimator(
        model_name="flownets",
        checkpoint_path=flownet_checkpoint,
        device=device,
    )
    flownet_model.eval()

    num_pairs = len(dataset)
    if num_pairs != len(records):
        raise ValueError(f"Dataset length ({num_pairs}) does not match record count ({len(records)})")

    print(f"Extracting 5 forward-only gate statistics across all {num_pairs} frames...")
    start_time = time.perf_counter()

    q_tensor = torch.tensor([0.50, 0.90, 0.95], device=device, dtype=torch.float32)

    with torch.inference_mode():
        for idx in range(num_pairs):
            frame1, frame2, _, valid_mask, meta = dataset[idx]
            rec = records[idx]

            if (rec["scene"], rec["frame_idx"]) != (meta["scene"], meta["frame_idx"]):
                raise ValueError(
                    f"Dataset alignment mismatch at index {idx}: record=({rec['scene']}, {rec['frame_idx']}) "
                    f"vs dataset=({meta['scene']}, {meta['frame_idx']})"
                )

            f1_dev = frame1.to(device)
            f2_dev = frame2.to(device)
            mask_dev = valid_mask.to(device)

            flow_fwd = flownet_model(f1_dev, f2_dev)[0]
            flow_mag = torch.sqrt(flow_fwd[0] ** 2 + flow_fwd[1] ** 2)

            mag_valid = flow_mag[mask_dev]

            # 1. Mean
            mag_mean = float(torch.mean(mag_valid).item())

            # 2. Quantiles: median (0.50), q90 (0.90), q95 (0.95)
            quantiles = torch.quantile(mag_valid, q_tensor)
            mag_median = float(quantiles[0].item())
            mag_q90 = float(quantiles[1].item())
            mag_q95 = float(quantiles[2].item())

            # 3. Fraction > 5 px
            frac_gt5 = float(torch.mean((mag_valid > 5.0).float()).item())

            rec["mag_median"] = mag_median
            rec["mag_mean"] = mag_mean
            rec["mag_q90"] = mag_q90
            rec["mag_q95"] = mag_q95
            rec["frac_mag_gt5"] = frac_gt5

            del flow_fwd, flow_mag, mag_valid, quantiles, f1_dev, f2_dev, mask_dev

            if (idx + 1) % 100 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - start_time
                fps = (idx + 1) / elapsed
                print(f"  [{idx + 1:4d}/{num_pairs}] frames processed ({fps:.1f} fps, elapsed: {elapsed:.1f}s)")

    total_time = time.perf_counter() - start_time
    print(f"Feature extraction complete in {total_time:.2f}s ({num_pairs / total_time:.1f} fps).\n")


def run_loso_evaluation_for_stat(
    records: List[Dict[str, Any]],
    stat_key: str,
    target_budget: float,
    unique_scenes: List[str],
    baselines: Dict[str, float],
    gate_latency_ms: float = GATE_LATENCY_MS,
    raft_latency_ms: float = RAFT_LATENCY_MS,
) -> Dict[str, Any]:
    """
    Executes 23-scene Leave-One-Scene-Out CV for a specific statistic and target budget.
    """
    pct = (1.0 - target_budget) * 100.0
    total_pairs = len(records)
    total_valid_pixels = baselines["total_valid_pixels"]

    oof_decisions: List[int] = []
    oof_routed_epes: List[float] = []
    oof_routed_epe_sums: List[float] = []
    oof_valid_counts: List[int] = []

    fold_summaries: List[Dict[str, Any]] = []

    for held_out in unique_scenes:
        train_records = [r for r in records if r["scene"] != held_out]
        test_records = [r for r in records if r["scene"] == held_out]

        train_vals = [r[stat_key] for r in train_records]
        tau_calibrated = float(np.percentile(train_vals, pct))

        fold_decs: List[int] = []
        fold_epes: List[float] = []

        for tr in test_records:
            dec = 1 if tr[stat_key] > tau_calibrated else 0
            e = tr["epe_raft"] if dec == 1 else tr["epe_flownets"]
            e_sum = tr["epe_sum_raft"] if dec == 1 else tr["epe_sum_flownets"]
            vc = tr["valid_count"]

            fold_decs.append(dec)
            fold_epes.append(e)

            oof_decisions.append(dec)
            oof_routed_epes.append(e)
            oof_routed_epe_sums.append(e_sum)
            oof_valid_counts.append(vc)

        fold_summaries.append({
            "held_out_scene": held_out,
            "num_pairs": len(test_records),
            "calibrated_threshold": tau_calibrated,
            "raft_invocations": int(sum(fold_decs)),
            "raft_invocation_rate": float(np.mean(fold_decs)),
            "routed_macro_epe": float(np.mean(fold_epes)),
        })

    oof_inv_rate = float(np.mean(oof_decisions))
    oof_macro_epe = float(np.mean(oof_routed_epes))
    oof_micro_epe = float(sum(oof_routed_epe_sums) / sum(oof_valid_counts))

    fn_macro = baselines["flownets_macro_epe"]
    fn_micro = baselines["flownets_micro_epe"]
    rf_macro = baselines["raft_macro_epe"]
    rf_micro = baselines["raft_micro_epe"]

    gap_macro = float(oof_macro_epe - rf_macro)
    gap_micro = float(oof_micro_epe - rf_micro)

    err_red_macro_pct = float((fn_macro - oof_macro_epe) / fn_macro * 100.0)
    err_red_micro_pct = float((fn_micro - oof_micro_epe) / fn_micro * 100.0)

    est_cascade_lat_ms = float(gate_latency_ms + oof_inv_rate * raft_latency_ms)
    est_fps = float(1000.0 / est_cascade_lat_ms)
    speedup = float(raft_latency_ms / est_cascade_lat_ms)

    return {
        "statistic": stat_key,
        "target_budget_pct": target_budget * 100.0,
        "target_budget": target_budget,
        "oof_raft_invocation_count": int(sum(oof_decisions)),
        "oof_raft_invocation_rate": oof_inv_rate,
        "oof_raft_invocation_pct": oof_inv_rate * 100.0,
        "oof_compute_saving_pct": (1.0 - oof_inv_rate) * 100.0,
        "routed_macro_epe": oof_macro_epe,
        "routed_micro_epe": oof_micro_epe,
        "gap_to_always_raft_macro_epe": gap_macro,
        "gap_to_always_raft_micro_epe": gap_micro,
        "error_reduction_vs_flownets_macro_pct": err_red_macro_pct,
        "error_reduction_vs_flownets_micro_pct": err_red_micro_pct,
        "estimated_cascade_latency_ms": est_cascade_lat_ms,
        "effective_fps": est_fps,
        "speedup_vs_always_raft": speedup,
        "fold_summaries": fold_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 6: Full-Scale Gate-Statistic Ablation on Sintel Clean Benchmark."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Sintel",
        help="Path to Sintel dataset root (default: data/Sintel)",
    )
    parser.add_argument(
        "--pass_name",
        type=str,
        default="clean",
        choices=["clean", "final"],
        help="Sintel pass name (default: clean)",
    )
    parser.add_argument(
        "--flownet_checkpoint",
        type=str,
        default="checkpoints/flownets_EPE1.951.pth.tar",
        help="Path to FlowNetS checkpoint weights",
    )
    parser.add_argument(
        "--flownet_benchmark_json",
        type=str,
        default="reports/sintel_benchmark_full/flownets_clean.json",
        help="Path to FlowNetS benchmark JSON",
    )
    parser.add_argument(
        "--raft_benchmark_json",
        type=str,
        default="reports/sintel_benchmark_full/raft_clean.json",
        help="Path to RAFT benchmark JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" DAY 6: FULL-SCALE GATE-STATISTIC ABLATION (SINTEL CLEAN 1041 PAIRS)")
    print("==================================================================")
    print(f"Timestamp (UTC):     {datetime.now(timezone.utc).isoformat()}")
    print(f"Device:              {device}")
    print(f"Dataset Root:        {args.data_root}")
    print(f"Pass Name:           {args.pass_name}")
    print(f"FlowNetS Checkpoint: {args.flownet_checkpoint}")
    print(f"Candidate Signals:   {STATISTIC_KEYS}")
    print(f"Target Budgets:      {[f'{b*100:.2f}%' for b in TARGET_BUDGETS]}")
    print(f"Output Directory:    {output_dir}\n")

    # 1. Load benchmark records
    records, baselines = load_benchmark_records(
        flownet_json_path=Path(args.flownet_benchmark_json),
        raft_json_path=Path(args.raft_benchmark_json),
    )
    unique_scenes = sorted(list(set(r["scene"] for r in records)))
    print(f"Loaded {len(records)} frame records across {len(unique_scenes)} unique scenes.")
    print(f"Baselines:")
    print(f"  FlowNetS Clean Macro EPE: {baselines['flownets_macro_epe']:.4f} px (Micro: {baselines['flownets_micro_epe']:.4f} px)")
    print(f"  Always-RAFT Macro EPE:    {baselines['raft_macro_epe']:.4f} px (Micro: {baselines['raft_micro_epe']:.4f} px)\n")

    # 2. Extract all 5 statistics from FlowNetS forward inference
    dataset = SintelDataset(root=args.data_root, split="training", pass_name=args.pass_name)
    extract_all_forward_statistics(
        dataset=dataset,
        records=records,
        flownet_checkpoint=args.flownet_checkpoint,
        device=device,
    )

    # 3. Execute 23-Scene LOSO CV across all statistics and budgets
    print("==================================================================")
    print(" RUNNING 23-SCENE LOSO-CV EVALUATIONS ACROSS 5 GATE STATISTICS    ")
    print("==================================================================")

    ablation_results: Dict[str, Dict[str, Any]] = {}

    for stat_key in STATISTIC_KEYS:
        ablation_results[stat_key] = {}
        print(f"\n--- Signal: {stat_key} ---")
        for budget in TARGET_BUDGETS:
            budget_str = f"{budget * 100:.2f}pct"
            res = run_loso_evaluation_for_stat(
                records=records,
                stat_key=stat_key,
                target_budget=budget,
                unique_scenes=unique_scenes,
                baselines=baselines,
                gate_latency_ms=GATE_LATENCY_MS,
                raft_latency_ms=RAFT_LATENCY_MS,
            )
            ablation_results[stat_key][budget_str] = res
            print(
                f"  Target {res['target_budget_pct']:5.2f}% -> Actual OOF: {res['oof_raft_invocation_pct']:5.2f}% "
                f"({res['oof_raft_invocation_count']:4d}/{len(records)}) | "
                f"Macro EPE: {res['routed_macro_epe']:.4f} px | "
                f"Gap: +{res['gap_to_always_raft_macro_epe']:.4f} px | "
                f"Latency: {res['estimated_cascade_latency_ms']:.1f} ms ({res['effective_fps']:.1f} FPS, {res['speedup_vs_always_raft']:.2f}x)"
            )

    # 4. Save JSON Results
    output_json_path = output_dir / "day6_gate_ablation.json"
    output_data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Day 6: Full-Scale Gate-Statistic Ablation on Sintel Clean Benchmark",
            "dataset": "MPI Sintel (training, clean pass)",
            "num_scenes": len(unique_scenes),
            "num_pairs": len(records),
            "statistics_evaluated": STATISTIC_KEYS,
            "target_budgets": TARGET_BUDGETS,
            "cross_validation": "Leave-One-Scene-Out (23 folds)",
            "timing_constants": {
                "gate_latency_ms": GATE_LATENCY_MS,
                "always_raft_latency_ms": RAFT_LATENCY_MS,
                "always_raft_fps": 1000.0 / RAFT_LATENCY_MS,
            },
        },
        "baselines": baselines,
        "ablation_results": ablation_results,
        "per_frame_records": [
            {
                "scene": r["scene"],
                "frame_idx": r["frame_idx"],
                "mag_median": r["mag_median"],
                "mag_mean": r["mag_mean"],
                "mag_q90": r["mag_q90"],
                "mag_q95": r["mag_q95"],
                "frac_mag_gt5": r["frac_mag_gt5"],
                "epe_flownets": r["epe_flownets"],
                "epe_raft": r["epe_raft"],
                "valid_count": r["valid_count"],
            }
            for r in records
        ],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved ablation results to: {output_json_path}")

    # 5. Print Comparative Summary Tables
    print("\n=========================================================================================================")
    print("                      GATE STATISTIC COMPARISON TABLE ACROSS KEY OPERATING TARGETS                       ")
    print("=========================================================================================================")

    # Table for ~23%, ~43%, 50%, 87.5%
    key_targets = [0.23, 0.43, 0.50, 0.875]

    for target in key_targets:
        target_key = f"{target * 100:.2f}pct"
        print(f"\nTarget Budget: {target * 100:.1f}% RAFT Invocation")
        print("| Statistic | Actual Invocation % | Macro EPE | Micro EPE | Gap to RAFT | Error Reduction | Latency (ms) | FPS | Speedup |")
        print("|---|---|---|---|---|---|---|---|---|")
        for stat_key in STATISTIC_KEYS:
            res = ablation_results[stat_key][target_key]
            print(
                f"| {stat_key:<13} | "
                f"{res['oof_raft_invocation_pct']:5.2f}% ({res['oof_raft_invocation_count']}/{len(records)}) | "
                f"{res['routed_macro_epe']:.4f} px | "
                f"{res['routed_micro_epe']:.4f} px | "
                f"+{res['gap_to_always_raft_macro_epe']:.4f} px | "
                f"{res['error_reduction_vs_flownets_macro_pct']:5.2f}% | "
                f"{res['estimated_cascade_latency_ms']:6.2f} ms | "
                f"{res['effective_fps']:5.2f} | "
                f"{res['speedup_vs_always_raft']:4.2f}x |"
            )
    print("=========================================================================================================\n")


if __name__ == "__main__":
    main()
