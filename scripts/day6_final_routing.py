"""
Day 6: Final-Pass Routing Validation on MPI Sintel Training Benchmark.

Evaluates the selected FlowNetS mag_q95 forward-only routing gate on the full
MPI Sintel Final training pass (all 1041 pairs across all 23 scenes).

Key Protocols:
1. Reuses precomputed FlowNetS and RAFT benchmark EPE and valid pixel counts from:
   - reports/sintel_benchmark_full/flownets_final.json
   - reports/sintel_benchmark_full/raft_final.json
2. Computes FlowNetS mag_q95 (95th percentile of forward flow magnitude) across all 1041 pairs
   of the Sintel Final pass in a single forward pass.
3. Executes strict 23-scene Leave-One-Scene-Out Cross-Validation (LOSO-CV):
   - For each held-out scene S_k, threshold tau_k is calibrated on the remaining 22 scenes of the Final pass
     as the (1 - rho) quantile of mag_q95.
   - Zero held-out ground-truth EPE leakage.
4. Evaluates operating targets:
   - ~23% (22.01% and 23.00%)
   - ~43% (43.00% and 43.43%)
   - 50.0%
   - ~87.5% (87.50%)
5. Reports:
   - Actual out-of-fold invocation rate (%) & count
   - Routed Macro and Micro EPE (px)
   - Gap to Always-RAFT (px)
   - Error reduction vs FlowNetS baseline (%)
   - Estimated cascade latency (ms) & throughput (FPS)
6. Compares directly against Clean-pass routing results from:
   - outputs/day6_gate_ablation.json
7. Saves outputs/day6_final_routing.json.
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

TARGET_BUDGETS = [0.2201, 0.23, 0.43, 0.4343, 0.50, 0.875]

GATE_LATENCY_MS = 13.50767240524292   # Forward gate latency (74.0 FPS)
RAFT_LATENCY_MS = 126.92              # Always-RAFT forward pass (7.88 FPS)


def load_benchmark_records(
    flownet_json_path: Path,
    raft_json_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Loads precomputed benchmark EPE and valid pixel records for FlowNetS and RAFT on Final pass."""
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


def extract_mag_q95_features(
    dataset: SintelDataset,
    records: List[Dict[str, Any]],
    flownet_checkpoint: str,
    device: torch.device,
) -> None:
    """
    Runs a single FlowNetS forward pass per frame on Sintel Final pass to extract mag_q95.
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

    print(f"Extracting FlowNetS mag_q95 across all {num_pairs} frames of Sintel Final pass...")
    start_time = time.perf_counter()

    q_val = torch.tensor([0.95], device=device, dtype=torch.float32)

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
            q95 = float(torch.quantile(mag_valid, q_val).item())

            rec["mag_q95"] = q95

            del flow_fwd, flow_mag, mag_valid, f1_dev, f2_dev, mask_dev

            if (idx + 1) % 100 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - start_time
                fps = (idx + 1) / elapsed
                print(f"  [{idx + 1:4d}/{num_pairs}] frames processed ({fps:.1f} fps, elapsed: {elapsed:.1f}s)")

    total_time = time.perf_counter() - start_time
    print(f"Extraction complete in {total_time:.2f}s ({num_pairs / total_time:.1f} fps).\n")


def run_23_scene_loso_evaluation(
    records: List[Dict[str, Any]],
    target_budget: float,
    unique_scenes: List[str],
    baselines: Dict[str, float],
    gate_latency_ms: float = GATE_LATENCY_MS,
    raft_latency_ms: float = RAFT_LATENCY_MS,
) -> Dict[str, Any]:
    """
    Executes 23-scene Leave-One-Scene-Out CV for mag_q95 on Sintel Final pass.
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

        train_vals = [r["mag_q95"] for r in train_records]
        tau_calibrated = float(np.percentile(train_vals, pct))

        fold_decs: List[int] = []
        fold_epes: List[float] = []

        for tr in test_records:
            dec = 1 if tr["mag_q95"] > tau_calibrated else 0
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
        description="Day 6: Final-Pass Routing Validation on Full MPI Sintel Benchmark."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Sintel",
        help="Path to Sintel dataset root",
    )
    parser.add_argument(
        "--pass_name",
        type=str,
        default="final",
        help="Sintel pass name (default: final)",
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
        default="reports/sintel_benchmark_full/flownets_final.json",
        help="Path to FlowNetS Final benchmark report",
    )
    parser.add_argument(
        "--raft_benchmark_json",
        type=str,
        default="reports/sintel_benchmark_full/raft_final.json",
        help="Path to RAFT Final benchmark report",
    )
    parser.add_argument(
        "--clean_ablation_json",
        type=str,
        default="outputs/day6_gate_ablation.json",
        help="Path to Clean-pass ablation report for comparison",
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
    print("   DAY 6: SINTEL FINAL-PASS ROUTING VALIDATION (mag_q95 LOSO-CV)  ")
    print("==================================================================")
    print(f"Timestamp (UTC):     {datetime.now(timezone.utc).isoformat()}")
    print(f"Device:              {device}")
    print(f"Dataset Root:        {args.data_root}")
    print(f"Pass Name:           {args.pass_name}")
    print(f"Routing Statistic:   mag_q95 (95th percentile of forward magnitude)")
    print(f"FlowNetS Checkpoint: {args.flownet_checkpoint}")
    print(f"FlowNetS Report:     {args.flownet_benchmark_json}")
    print(f"RAFT Report:         {args.raft_benchmark_json}")
    print(f"Output Directory:    {output_dir}\n")

    # 1. Load benchmark records
    records, baselines = load_benchmark_records(
        flownet_json_path=Path(args.flownet_benchmark_json),
        raft_json_path=Path(args.raft_benchmark_json),
    )
    unique_scenes = sorted(list(set(r["scene"] for r in records)))
    print(f"Loaded {len(records)} frame records across {len(unique_scenes)} unique scenes on Sintel Final.")
    print(f"Final Baselines:")
    print(f"  FlowNetS Final Macro EPE: {baselines['flownets_macro_epe']:.4f} px (Micro: {baselines['flownets_micro_epe']:.4f} px)")
    print(f"  Always-RAFT Final Macro:  {baselines['raft_macro_epe']:.4f} px (Micro: {baselines['raft_micro_epe']:.4f} px)")
    print(f"  Oracle Frame Macro EPE:   {baselines['oracle_macro_epe']:.4f} px\n")

    # 2. Extract mag_q95 on Final pass
    dataset = SintelDataset(root=args.data_root, split="training", pass_name=args.pass_name)
    extract_mag_q95_features(
        dataset=dataset,
        records=records,
        flownet_checkpoint=args.flownet_checkpoint,
        device=device,
    )

    # 3. Execute 23-Scene LOSO-CV across target budgets
    print("==================================================================")
    print(" EXECUTING 23-SCENE LOSO-CV EVALUATIONS ON SINTEL FINAL PASS      ")
    print("==================================================================")

    final_results: Dict[str, Any] = {}
    for budget in TARGET_BUDGETS:
        b_key = f"{budget * 100:.2f}pct"
        res = run_23_scene_loso_evaluation(
            records=records,
            target_budget=budget,
            unique_scenes=unique_scenes,
            baselines=baselines,
            gate_latency_ms=GATE_LATENCY_MS,
            raft_latency_ms=RAFT_LATENCY_MS,
        )
        final_results[b_key] = res
        print(
            f"Target {res['target_budget_pct']:5.2f}% -> Actual OOF: {res['oof_raft_invocation_pct']:5.2f}% "
            f"({res['oof_raft_invocation_count']:4d}/{len(records)}) | "
            f"Macro EPE: {res['routed_macro_epe']:.4f} px | "
            f"Gap: +{res['gap_to_always_raft_macro_epe']:.4f} px | "
            f"Error Red.: {res['error_reduction_vs_flownets_macro_pct']:.2f}% | "
            f"Latency: {res['estimated_cascade_latency_ms']:.1f} ms ({res['effective_fps']:.1f} FPS, {res['speedup_vs_always_raft']:.2f}x)"
        )

    # 4. Load Clean-pass comparison data if available
    clean_comparison: Dict[str, Any] = {}
    clean_path = Path(args.clean_ablation_json).resolve()
    if clean_path.is_file():
        with open(clean_path, "r", encoding="utf-8") as f:
            clean_data = json.load(f)
        clean_q95 = clean_data.get("ablation_results", {}).get("mag_q95", {})
        for b_key, res_final in final_results.items():
            if b_key in clean_q95:
                res_clean = clean_q95[b_key]
                clean_comparison[b_key] = {
                    "target_budget_pct": res_final["target_budget_pct"],
                    "clean_actual_invocation_pct": res_clean["oof_raft_invocation_pct"],
                    "clean_macro_epe": res_clean["routed_macro_epe"],
                    "clean_gap_to_raft": res_clean["gap_to_always_raft_macro_epe"],
                    "clean_error_reduction_pct": res_clean["error_reduction_vs_flownets_macro_pct"],
                    "clean_effective_fps": res_clean["effective_fps"],
                    "final_actual_invocation_pct": res_final["oof_raft_invocation_pct"],
                    "final_macro_epe": res_final["routed_macro_epe"],
                    "final_gap_to_raft": res_final["gap_to_always_raft_macro_epe"],
                    "final_error_reduction_pct": res_final["error_reduction_vs_flownets_macro_pct"],
                    "final_effective_fps": res_final["effective_fps"],
                }

    # 5. Save JSON report
    output_data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Day 6: Final-Pass Routing Validation on Full Sintel Benchmark",
            "dataset": "MPI Sintel (training, final pass)",
            "num_scenes": len(unique_scenes),
            "num_pairs": len(records),
            "routing_statistic": "mag_q95",
            "target_budgets": TARGET_BUDGETS,
            "cross_validation": "Leave-One-Scene-Out (23 folds)",
            "timing_constants": {
                "gate_latency_ms": GATE_LATENCY_MS,
                "always_raft_latency_ms": RAFT_LATENCY_MS,
                "always_raft_fps": 1000.0 / RAFT_LATENCY_MS,
            },
        },
        "final_baselines": baselines,
        "final_pass_routing_results": final_results,
        "clean_vs_final_comparison": clean_comparison,
        "per_frame_records": [
            {
                "scene": r["scene"],
                "frame_idx": r["frame_idx"],
                "mag_q95": r["mag_q95"],
                "epe_flownets": r["epe_flownets"],
                "epe_raft": r["epe_raft"],
                "valid_count": r["valid_count"],
            }
            for r in records
        ],
    }

    output_json_path = output_dir / "day6_final_routing.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved Final pass routing results to: {output_json_path}")

    # 6. Print Comparative Summary Table
    print("\n=========================================================================================================")
    print("                       SINTEL CLEAN VS FINAL PASS ROUTING COMPARISON (mag_q95)                           ")
    print("=========================================================================================================")
    print(f"Baselines:")
    print(f"  Clean Pass: FlowNetS = 5.1468 px | Always-RAFT = 0.6311 px (Oracle = 0.6311 px)")
    print(f"  Final Pass: FlowNetS = {baselines['flownets_macro_epe']:.4f} px | Always-RAFT = {baselines['raft_macro_epe']:.4f} px (Oracle = {baselines['oracle_macro_epe']:.4f} px)\n")

    print("| Target Budget | Clean Invoc. % | Clean Macro EPE | Clean Err. Red. | Final Invoc. % | Final Macro EPE | Final Gap to RAFT | Final Err. Red. | Final FPS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in [0.23, 0.43, 0.50, 0.875]:
        b_key = f"{b * 100:.2f}pct"
        if b_key in clean_comparison:
            c = clean_comparison[b_key]
            print(
                f"| {b * 100:5.1f}%        | "
                f"{c['clean_actual_invocation_pct']:5.2f}%         | "
                f"{c['clean_macro_epe']:6.4f} px       | "
                f"{c['clean_error_reduction_pct']:5.2f}%          | "
                f"{c['final_actual_invocation_pct']:5.2f}%         | "
                f"{c['final_macro_epe']:6.4f} px       | "
                f"+{c['final_gap_to_raft']:6.4f} px       | "
                f"{c['final_error_reduction_pct']:5.2f}%          | "
                f"{c['final_effective_fps']:5.2f} FPS |"
            )
    print("=========================================================================================================\n")


if __name__ == "__main__":
    main()
