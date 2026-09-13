"""
Day 6 Step 6: Full Sintel Training Benchmark Validation of Forward-Only FlowNetS Gating.

Validates the forward-only FlowNetS flow magnitude median (mag_median) routing gate
on the complete MPI Sintel training benchmark (all 1041 valid pairs across all 23 scenes, clean pass).

Protocol:
1. Reuses precomputed FlowNetS and RAFT per-frame EPEs and valid pixel counts from:
   - reports/sintel_benchmark_full/flownets_clean.json
   - reports/sintel_benchmark_full/raft_clean.json
2. Computes FlowNetS forward-only flow magnitude median (mag_median) for each of the 1041 pairs.
3. Performs true Leave-One-Scene-Out Cross-Validation (LOSO-CV) across all 23 scenes.
4. Evaluates target RAFT invocation budgets:
   - 25.0%
   - 50.0%
   - 75.0%
   - 87.5%
5. Zero held-out EPE leakage: routing threshold tau is calibrated strictly on the training fold
   as the (1 - rho) quantile of FlowNetS mag_median.
6. Reports out-of-fold (OOF) invocation rate, routed mean EPE (macro and micro),
   FlowNetS-only EPE, Always-RAFT EPE, gap to Always-RAFT, error reduction vs FlowNetS,
   and held-out scene/pair counts.
7. Saves outputs/day6_full_sintel_routing.json.
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

TARGET_BUDGETS = [0.25, 0.50, 0.75, 0.875]


def load_benchmark_records(
    flownet_json_path: Path,
    raft_json_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Loads precomputed benchmark EPE and valid pixel records for FlowNetS and RAFT.
    """
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
            "valid_count": fn_rec["valid_count"],
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


def extract_flownets_mag_median(
    dataset: SintelDataset,
    records: List[Dict[str, Any]],
    flownet_checkpoint: str,
    device: torch.device,
) -> None:
    """
    Runs FlowNetS forward inference on all dataset pairs to extract mag_median.
    Mutates records in-place to add 'mag_median' and 'mag_median_full'.
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

    print(f"Extracting FlowNetS mag_median across {num_pairs} frames...")
    start_time = time.perf_counter()

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
            mag_median_valid = float(torch.median(mag_valid).item())
            mag_median_full = float(torch.median(flow_mag).item())

            rec["mag_median"] = mag_median_valid
            rec["mag_median_full"] = mag_median_full

            del flow_fwd, flow_mag, mag_valid, f1_dev, f2_dev, mask_dev

            if (idx + 1) % 100 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - start_time
                fps = (idx + 1) / elapsed
                print(f"  [{idx + 1:4d}/{num_pairs}] frames processed ({fps:.1f} fps, elapsed: {elapsed:.1f}s)")

    total_time = time.perf_counter() - start_time
    print(f"Extraction complete in {total_time:.2f}s ({num_pairs / total_time:.1f} fps).\n")


def run_23_scene_loso_cv(
    records: List[Dict[str, Any]],
    target_budget: float,
    unique_scenes: List[str],
    baselines: Dict[str, float],
) -> Dict[str, Any]:
    """
    Executes 23-fold Leave-One-Scene-Out CV for a given target RAFT invocation budget.
    """
    pct = (1.0 - target_budget) * 100.0

    oof_decisions: List[int] = []
    oof_routed_epes: List[float] = []
    oof_routed_epe_sums: List[float] = []
    oof_valid_counts: List[int] = []

    fold_details: List[Dict[str, Any]] = []

    for fold_idx, held_out_scene in enumerate(unique_scenes, 1):
        train_records = [r for r in records if r["scene"] != held_out_scene]
        test_records = [r for r in records if r["scene"] == held_out_scene]

        train_mags = [r["mag_median"] for r in train_records]
        tau_calibrated = float(np.percentile(train_mags, pct))

        fold_decs: List[int] = []
        fold_epes: List[float] = []
        fold_epe_sums: List[float] = []
        fold_valid_counts: List[int] = []

        for tr in test_records:
            dec = 1 if tr["mag_median"] > tau_calibrated else 0
            e = tr["epe_raft"] if dec == 1 else tr["epe_flownets"]
            e_sum = tr["epe_sum_raft"] if dec == 1 else tr["epe_sum_flownets"]
            vc = tr["valid_count"]

            fold_decs.append(dec)
            fold_epes.append(e)
            fold_epe_sums.append(e_sum)
            fold_valid_counts.append(vc)

            oof_decisions.append(dec)
            oof_routed_epes.append(e)
            oof_routed_epe_sums.append(e_sum)
            oof_valid_counts.append(vc)

        fold_inv_rate = float(np.mean(fold_decs))
        fold_fn_macro = float(np.mean([r["epe_flownets"] for r in test_records]))
        fold_rf_macro = float(np.mean([r["epe_raft"] for r in test_records]))
        fold_routed_macro = float(np.mean(fold_epes))

        fold_details.append({
            "fold_idx": fold_idx,
            "held_out_scene": held_out_scene,
            "num_pairs": len(test_records),
            "calibrated_threshold_px": tau_calibrated,
            "raft_invocations": int(sum(fold_decs)),
            "raft_invocation_rate": fold_inv_rate,
            "flownets_macro_epe": fold_fn_macro,
            "raft_macro_epe": fold_rf_macro,
            "routed_macro_epe": fold_routed_macro,
            "error_reduction_pct": (
                (fold_fn_macro - fold_routed_macro) / fold_fn_macro * 100.0
                if fold_fn_macro > 0 else 0.0
            ),
        })

    oof_inv_rate = float(np.mean(oof_decisions))
    oof_macro_epe = float(np.mean(oof_routed_epes))
    oof_micro_epe = float(sum(oof_routed_epe_sums) / sum(oof_valid_counts))

    fn_macro = baselines["flownets_macro_epe"]
    fn_micro = baselines["flownets_micro_epe"]
    rf_macro = baselines["raft_macro_epe"]
    rf_micro = baselines["raft_micro_epe"]

    gap_macro = oof_macro_epe - rf_macro
    gap_micro = oof_micro_epe - rf_micro

    err_red_macro_pct = (fn_macro - oof_macro_epe) / fn_macro * 100.0
    err_red_micro_pct = (fn_micro - oof_micro_epe) / fn_micro * 100.0

    return {
        "target_budget": target_budget,
        "target_budget_pct": target_budget * 100.0,
        "calibration_percentile": pct,
        "total_held_out_scenes": len(unique_scenes),
        "total_held_out_pairs": len(records),
        "oof_raft_invocation_count": int(sum(oof_decisions)),
        "oof_raft_invocation_rate": oof_inv_rate,
        "oof_compute_saving_pct": (1.0 - oof_inv_rate) * 100.0,
        "oof_routed_macro_epe": oof_macro_epe,
        "oof_routed_micro_epe": oof_micro_epe,
        "flownets_only_macro_epe": fn_macro,
        "flownets_only_micro_epe": fn_micro,
        "always_raft_macro_epe": rf_macro,
        "always_raft_micro_epe": rf_micro,
        "gap_to_always_raft_macro_epe": gap_macro,
        "gap_to_always_raft_micro_epe": gap_micro,
        "error_reduction_vs_flownets_macro_pct": err_red_macro_pct,
        "error_reduction_vs_flownets_micro_pct": err_red_micro_pct,
        "folds": fold_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 6 Step 6: Full Sintel Training Benchmark Validation of Forward-Only mag_median Gate."
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
        help="Path to FlowNetS benchmark JSON (default: reports/sintel_benchmark_full/flownets_clean.json)",
    )
    parser.add_argument(
        "--raft_benchmark_json",
        type=str,
        default="reports/sintel_benchmark_full/raft_clean.json",
        help="Path to RAFT benchmark JSON (default: reports/sintel_benchmark_full/raft_clean.json)",
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
    print(" DAY 6 STEP 6: FULL SINTEL BENCHMARK ROUTING VALIDATION (LOSO-CV)")
    print("==================================================================")
    print(f"Timestamp (UTC):  {datetime.now(timezone.utc).isoformat()}")
    print(f"Device:           {device}")
    print(f"Dataset Root:     {args.data_root}")
    print(f"Pass Name:        {args.pass_name}")
    print(f"FlowNetS Checkpoint: {args.flownet_checkpoint}")
    print(f"FlowNetS Report:  {args.flownet_benchmark_json}")
    print(f"RAFT Report:      {args.raft_benchmark_json}")
    print(f"Output Directory: {output_dir}\n")

    # 1. Load benchmark records
    print("Loading precomputed benchmark records...")
    records, baselines = load_benchmark_records(
        flownet_json_path=Path(args.flownet_benchmark_json),
        raft_json_path=Path(args.raft_benchmark_json),
    )
    unique_scenes = sorted(list(set(r["scene"] for r in records)))
    print(f"Loaded {len(records)} frame records across {len(unique_scenes)} scenes.")
    print(f"Baselines:")
    print(f"  FlowNetS Clean Macro EPE: {baselines['flownets_macro_epe']:.4f} px (Micro: {baselines['flownets_micro_epe']:.4f} px)")
    print(f"  RAFT Clean Macro EPE:     {baselines['raft_macro_epe']:.4f} px (Micro: {baselines['raft_micro_epe']:.4f} px)")
    print(f"  Oracle Frame Macro EPE:   {baselines['oracle_macro_epe']:.4f} px")
    print(f"  Total Valid Pixels:       {baselines['total_valid_pixels']:,}\n")

    # 2. Extract mag_median from FlowNetS forward inference
    dataset = SintelDataset(root=args.data_root, split="training", pass_name=args.pass_name)
    extract_flownets_mag_median(
        dataset=dataset,
        records=records,
        flownet_checkpoint=args.flownet_checkpoint,
        device=device,
    )

    # 3. Perform 23-Scene LOSO Cross-Validation across target budgets
    print("==================================================================")
    print(" EXECUTING 23-SCENE LEAVE-ONE-SCENE-OUT CROSS-VALIDATION")
    print("==================================================================")
    loso_results: Dict[str, Any] = {}

    for budget in TARGET_BUDGETS:
        budget_key = f"{budget * 100:.1f}pct"
        print(f"\n--- Target Budget: {budget * 100:.1f}% RAFT Invocation ---")
        res = run_23_scene_loso_cv(
            records=records,
            target_budget=budget,
            unique_scenes=unique_scenes,
            baselines=baselines,
        )
        loso_results[budget_key] = res

        print(f"  OOF RAFT Invocation:    {res['oof_raft_invocation_rate'] * 100:.2f}% ({res['oof_raft_invocation_count']}/{len(records)} pairs)")
        print(f"  OOF Routed Macro EPE:   {res['oof_routed_macro_epe']:.4f} px")
        print(f"  OOF Routed Micro EPE:   {res['oof_routed_micro_epe']:.4f} px")
        print(f"  Gap to Always-RAFT:     +{res['gap_to_always_raft_macro_epe']:.4f} px (Macro)")
        print(f"  Error Reduction vs FNS: {res['error_reduction_vs_flownets_macro_pct']:.2f}% (Macro)")

    # 4. Canonical 8-Sample Generalization Comparison
    comparison_8_vs_full = {
        "budget_50pct": {
            "canonical_8_sample": {
                "oof_invocation_rate": 0.50,
                "oof_routed_macro_epe": 0.7057,
                "gap_to_raft_epe": 0.0768,
                "error_reduction_vs_flownets_pct": 89.26,
            },
            "full_1041_pairs": {
                "oof_invocation_rate": loso_results["50.0pct"]["oof_raft_invocation_rate"],
                "oof_routed_macro_epe": loso_results["50.0pct"]["oof_routed_macro_epe"],
                "gap_to_raft_epe": loso_results["50.0pct"]["gap_to_always_raft_macro_epe"],
                "error_reduction_vs_flownets_pct": loso_results["50.0pct"]["error_reduction_vs_flownets_macro_pct"],
            },
        },
        "budget_87_5pct": {
            "canonical_8_sample": {
                "oof_invocation_rate": 0.875,
                "oof_routed_macro_epe": 0.6409,
                "gap_to_raft_epe": 0.0120,
                "error_reduction_vs_flownets_pct": 90.25,
            },
            "full_1041_pairs": {
                "oof_invocation_rate": loso_results["87.5pct"]["oof_raft_invocation_rate"],
                "oof_routed_macro_epe": loso_results["87.5pct"]["oof_routed_macro_epe"],
                "gap_to_raft_epe": loso_results["87.5pct"]["gap_to_always_raft_macro_epe"],
                "error_reduction_vs_flownets_pct": loso_results["87.5pct"]["error_reduction_vs_flownets_macro_pct"],
            },
        },
    }

    # 5. Build full serializable JSON report
    output_data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Day 6 Step 6: Full Sintel Training Benchmark LOSO-CV Routing Validation",
            "dataset": "MPI Sintel (training)",
            "pass_name": args.pass_name,
            "total_scenes": len(unique_scenes),
            "total_pairs": len(records),
            "routing_signal": "mag_median (FlowNetS forward flow magnitude median)",
            "cross_validation_protocol": "Leave-One-Scene-Out Cross-Validation (23 folds)",
            "device": str(device),
        },
        "baselines": baselines,
        "target_budgets_summary": {
            k: {
                "target_budget_pct": v["target_budget_pct"],
                "oof_raft_invocation_rate": v["oof_raft_invocation_rate"],
                "oof_raft_invocation_count": v["oof_raft_invocation_count"],
                "oof_compute_saving_pct": v["oof_compute_saving_pct"],
                "oof_routed_macro_epe": v["oof_routed_macro_epe"],
                "oof_routed_micro_epe": v["oof_routed_micro_epe"],
                "flownets_only_macro_epe": v["flownets_only_macro_epe"],
                "always_raft_macro_epe": v["always_raft_macro_epe"],
                "gap_to_always_raft_macro_epe": v["gap_to_always_raft_macro_epe"],
                "gap_to_always_raft_micro_epe": v["gap_to_always_raft_micro_epe"],
                "error_reduction_vs_flownets_macro_pct": v["error_reduction_vs_flownets_macro_pct"],
                "error_reduction_vs_flownets_micro_pct": v["error_reduction_vs_flownets_micro_pct"],
                "total_held_out_scenes": v["total_held_out_scenes"],
                "total_held_out_pairs": v["total_held_out_pairs"],
            }
            for k, v in loso_results.items()
        },
        "generalization_8_vs_full": comparison_8_vs_full,
        "budget_evaluations": loso_results,
        "per_frame_records": [
            {
                "scene": r["scene"],
                "frame_idx": r["frame_idx"],
                "mag_median": r["mag_median"],
                "mag_median_full": r["mag_median_full"],
                "epe_flownets": r["epe_flownets"],
                "epe_raft": r["epe_raft"],
                "valid_count": r["valid_count"],
            }
            for r in records
        ],
    }

    output_path = output_dir / "day6_full_sintel_routing.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved full benchmark routing results to: {output_path}")

    # Print summary table
    print("\n==================================================================")
    print("               FINAL SUMMARY TABLE (FULL SINTEL)                  ")
    print("==================================================================")
    print("| Target Budget | OOF Invocation Rate | Routed Macro EPE | FlowNetS EPE | Always-RAFT EPE | Gap to RAFT | Error Reduction | Held-out Scenes/Pairs |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in loso_results.items():
        print(
            f"| {v['target_budget_pct']:5.1f}% | "
            f"{v['oof_raft_invocation_rate'] * 100:5.2f}% ({v['oof_raft_invocation_count']}/{len(records)}) | "
            f"{v['oof_routed_macro_epe']:.4f} px | "
            f"{baselines['flownets_macro_epe']:.4f} px | "
            f"{baselines['raft_macro_epe']:.4f} px | "
            f"+{v['gap_to_always_raft_macro_epe']:.4f} px | "
            f"{v['error_reduction_vs_flownets_macro_pct']:5.2f}% | "
            f"{v['total_held_out_scenes']} scenes / {v['total_held_out_pairs']} pairs |"
        )
    print("==================================================================\n")


if __name__ == "__main__":
    main()
