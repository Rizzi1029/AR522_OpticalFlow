"""
Day 6: Actual End-to-End Cascade Timing Validation on Full Sintel Clean Benchmark.

Measures the real physical branching cascade pipeline:
  Frame Pair (I_t, I_{t+1}) -> FlowNetS Forward -> mag_q95 Gate -> RAFT Forward (only when invoked)
across all 1041 frame pairs and 23 scenes of the MPI Sintel training benchmark (clean pass).

Key Protocols:
1. Uses the leakage-free 23-scene Leave-One-Scene-Out (LOSO) calibrated thresholds for mag_q95 from:
   - outputs/day6_gate_ablation.json
2. Benchmarks the real physical execution using CUDA synchronization and CUDA events:
   - torch.cuda.Event(enable_timing=True)
   - start_event.record() -> execute branching pipeline -> end_event.record() -> torch.cuda.synchronize()
3. Benchmarks modes:
   - Always-FlowNetS (pure FlowNetS forward pass on every frame)
   - Always-RAFT (pure RAFT forward pass on every frame)
   - Real Branching Cascade @ ~43% target budget (Primary Selected Operating Point)
   - Real Branching Cascade @ 50% target budget (Canonical Balanced Operating Point)
   - Real Branching Cascade @ ~23% target budget (Fast Kneedle Operating Point)
4. Reports:
   - Wall-clock total duration (sec)
   - Pure GPU inference latency: mean, median, p95, min, max (ms)
   - Effective throughput (FPS)
   - RAFT invocation count & rate (%)
   - Routed Macro & Micro EPE (px)
   - Error reduction vs FlowNetS (%) & gap to Always-RAFT (px)
   - Fast-path latency (FlowNetS only) vs Accurate-path latency (FlowNetS + RAFT)
5. Compares actual cascade latency against the previous component-based additive estimate:
   T_est = T_gate + rho_act * T_raft
   Quantifies absolute difference (ms) and relative percentage difference (%).
6. Saves outputs/day6_actual_cascade_timing.json.
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
from src.benchmark.metrics import compute_epe
from src.flow.wrappers import build_flow_estimator

BUDGET_KEYS_TO_EVALUATE = [
    ("43.00pct", "Cascade @ ~43% (Primary Selected Knee)"),
    ("50.00pct", "Cascade @ 50% (Canonical Balanced)"),
    ("23.00pct", "Cascade @ ~23% (Fast Kneedle Knee)"),
]


def load_loso_thresholds(ablation_json_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Loads 23-scene LOSO calibrated thresholds for mag_q95 across target budgets.
    Returns: {budget_str: {scene_name: tau_float}}
    """
    if not ablation_json_path.is_file():
        raise FileNotFoundError(f"Ablation report not found: {ablation_json_path}")

    with open(ablation_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q95_results = data["ablation_results"]["mag_q95"]
    thresholds_by_budget: Dict[str, Dict[str, float]] = {}

    for b_key, _ in BUDGET_KEYS_TO_EVALUATE:
        if b_key not in q95_results:
            raise KeyError(f"Budget key '{b_key}' not found in mag_q95 ablation results.")
        folds = q95_results[b_key]["fold_summaries"]
        scene_taus = {f["held_out_scene"]: float(f["calibrated_threshold"]) for f in folds}
        thresholds_by_budget[b_key] = scene_taus

    return thresholds_by_budget


def warmup_models(
    flownet_model: Any,
    raft_model: Any,
    dataset: SintelDataset,
    device: torch.device,
    num_warmup: int = 5,
) -> None:
    """Runs warmup iterations to prime CUDA kernels, cuDNN autotuners, and memory allocators."""
    print(f"Executing {num_warmup} warmup iterations on CUDA...")
    with torch.inference_mode():
        for i in range(num_warmup):
            f1, f2, _, _, _ = dataset[i]
            f1_dev = f1.to(device)
            f2_dev = f2.to(device)
            _ = flownet_model(f1_dev, f2_dev)
            _ = raft_model(f1_dev, f2_dev)
            torch.cuda.synchronize()
    print("Warmup complete.\n")


def benchmark_always_flownets(
    flownet_model: Any,
    dataset: SintelDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """Measures Always-FlowNetS baseline on all 1041 frames."""
    num_pairs = len(dataset)
    print(f"Benchmarking Mode: Always-FlowNetS across {num_pairs} frames...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms: List[float] = []
    frame_epes: List[float] = []
    total_valid_pixels = 0
    total_epe_sum = 0.0

    wall_clock_start = time.perf_counter()

    with torch.inference_mode():
        for idx in range(num_pairs):
            f1, f2, gt_flow, valid_mask, meta = dataset[idx]
            f1_dev = f1.to(device)
            f2_dev = f2.to(device)
            gt_dev = gt_flow.to(device)
            mask_dev = valid_mask.to(device)

            torch.cuda.synchronize()
            start_event.record()

            pred_flow = flownet_model(f1_dev, f2_dev)[0]

            end_event.record()
            torch.cuda.synchronize()

            lat = float(start_event.elapsed_time(end_event))
            latencies_ms.append(lat)

            mean_epe, vc, e_sum = compute_epe(pred_flow, gt_dev, mask_dev)
            frame_epes.append(float(mean_epe))
            total_valid_pixels += int(vc)
            total_epe_sum += float(e_sum)

            del f1_dev, f2_dev, gt_dev, mask_dev, pred_flow

            if (idx + 1) % 250 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - wall_clock_start
                print(f"  [{idx + 1:4d}/{num_pairs}] frames ({idx + 1 / elapsed:.1f} fps, elapsed: {elapsed:.1f}s)")

    wall_clock_total = time.perf_counter() - wall_clock_start
    lat_arr = np.array(latencies_ms, dtype=np.float64)

    mean_lat = float(np.mean(lat_arr))
    return {
        "mode_name": "Always-FlowNetS",
        "num_frames": num_pairs,
        "wall_clock_total_sec": wall_clock_total,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": float(np.median(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "min_latency_ms": float(np.min(lat_arr)),
        "max_latency_ms": float(np.max(lat_arr)),
        "effective_fps": float(1000.0 / mean_lat),
        "macro_epe": float(np.mean(frame_epes)),
        "micro_epe": float(total_epe_sum / total_valid_pixels),
        "total_valid_pixels": total_valid_pixels,
        "raft_invocations": 0,
        "raft_invocation_rate": 0.0,
    }


def benchmark_always_raft(
    raft_model: Any,
    dataset: SintelDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """Measures Always-RAFT baseline on all 1041 frames."""
    num_pairs = len(dataset)
    print(f"\nBenchmarking Mode: Always-RAFT across {num_pairs} frames...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms: List[float] = []
    frame_epes: List[float] = []
    total_valid_pixels = 0
    total_epe_sum = 0.0

    wall_clock_start = time.perf_counter()

    with torch.inference_mode():
        for idx in range(num_pairs):
            f1, f2, gt_flow, valid_mask, meta = dataset[idx]
            f1_dev = f1.to(device)
            f2_dev = f2.to(device)
            gt_dev = gt_flow.to(device)
            mask_dev = valid_mask.to(device)

            torch.cuda.synchronize()
            start_event.record()

            pred_flow = raft_model(f1_dev, f2_dev)[0]

            end_event.record()
            torch.cuda.synchronize()

            lat = float(start_event.elapsed_time(end_event))
            latencies_ms.append(lat)

            mean_epe, vc, e_sum = compute_epe(pred_flow, gt_dev, mask_dev)
            frame_epes.append(float(mean_epe))
            total_valid_pixels += int(vc)
            total_epe_sum += float(e_sum)

            del f1_dev, f2_dev, gt_dev, mask_dev, pred_flow

            if (idx + 1) % 250 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - wall_clock_start
                print(f"  [{idx + 1:4d}/{num_pairs}] frames ({idx + 1 / elapsed:.1f} fps, elapsed: {elapsed:.1f}s)")

    wall_clock_total = time.perf_counter() - wall_clock_start
    lat_arr = np.array(latencies_ms, dtype=np.float64)

    mean_lat = float(np.mean(lat_arr))
    return {
        "mode_name": "Always-RAFT",
        "num_frames": num_pairs,
        "wall_clock_total_sec": wall_clock_total,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": float(np.median(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "min_latency_ms": float(np.min(lat_arr)),
        "max_latency_ms": float(np.max(lat_arr)),
        "effective_fps": float(1000.0 / mean_lat),
        "macro_epe": float(np.mean(frame_epes)),
        "micro_epe": float(total_epe_sum / total_valid_pixels),
        "total_valid_pixels": total_valid_pixels,
        "raft_invocations": num_pairs,
        "raft_invocation_rate": 1.0,
    }


def benchmark_real_cascade(
    flownet_model: Any,
    raft_model: Any,
    dataset: SintelDataset,
    thresholds_by_scene: Dict[str, float],
    mode_name: str,
    target_budget_pct: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Measures the real branching cascade pipeline:
    FlowNetS forward -> mag_q95 gate -> RAFT forward ONLY when invoked.
    """
    num_pairs = len(dataset)
    print(f"\nBenchmarking Mode: {mode_name} across {num_pairs} frames...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_all_ms: List[float] = []
    latencies_fast_path_ms: List[float] = []   # non-invoked frames
    latencies_accurate_path_ms: List[float] = [] # invoked frames
    frame_epes: List[float] = []
    invocations: List[int] = []

    total_valid_pixels = 0
    total_epe_sum = 0.0

    wall_clock_start = time.perf_counter()

    with torch.inference_mode():
        for idx in range(num_pairs):
            f1, f2, gt_flow, valid_mask, meta = dataset[idx]
            scene = meta["scene"]
            tau = thresholds_by_scene[scene]

            f1_dev = f1.to(device)
            f2_dev = f2.to(device)
            gt_dev = gt_flow.to(device)
            mask_dev = valid_mask.to(device)

            torch.cuda.synchronize()
            start_event.record()

            # 1. FlowNetS forward pass
            flow_fn = flownet_model(f1_dev, f2_dev)[0]

            # 2. Gate statistic computation (mag_q95)
            flow_mag = torch.sqrt(flow_fn[0] ** 2 + flow_fn[1] ** 2)
            mag_valid = flow_mag[mask_dev]
            q95 = float(torch.quantile(mag_valid, 0.95).item())

            # 3. Branching execution
            if q95 > tau:
                final_flow = raft_model(f1_dev, f2_dev)[0]
                invoked = 1
            else:
                final_flow = flow_fn
                invoked = 0

            end_event.record()
            torch.cuda.synchronize()

            lat = float(start_event.elapsed_time(end_event))
            latencies_all_ms.append(lat)
            invocations.append(invoked)

            if invoked == 1:
                latencies_accurate_path_ms.append(lat)
            else:
                latencies_fast_path_ms.append(lat)

            mean_epe, vc, e_sum = compute_epe(final_flow, gt_dev, mask_dev)
            frame_epes.append(float(mean_epe))
            total_valid_pixels += int(vc)
            total_epe_sum += float(e_sum)

            del f1_dev, f2_dev, gt_dev, mask_dev, flow_fn, flow_mag, mag_valid, final_flow

            if (idx + 1) % 250 == 0 or (idx + 1) == num_pairs:
                elapsed = time.perf_counter() - wall_clock_start
                print(f"  [{idx + 1:4d}/{num_pairs}] frames ({idx + 1 / elapsed:.1f} fps, elapsed: {elapsed:.1f}s)")

    wall_clock_total = time.perf_counter() - wall_clock_start
    lat_arr = np.array(latencies_all_ms, dtype=np.float64)
    fast_arr = np.array(latencies_fast_path_ms, dtype=np.float64)
    acc_arr = np.array(latencies_accurate_path_ms, dtype=np.float64)

    mean_lat = float(np.mean(lat_arr))
    inv_count = int(sum(invocations))
    inv_rate = float(inv_count / num_pairs)

    return {
        "mode_name": mode_name,
        "target_budget_pct": target_budget_pct,
        "num_frames": num_pairs,
        "wall_clock_total_sec": wall_clock_total,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": float(np.median(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "min_latency_ms": float(np.min(lat_arr)),
        "max_latency_ms": float(np.max(lat_arr)),
        "effective_fps": float(1000.0 / mean_lat),
        "fast_path_mean_latency_ms": float(np.mean(fast_arr)) if len(fast_arr) > 0 else 0.0,
        "accurate_path_mean_latency_ms": float(np.mean(acc_arr)) if len(acc_arr) > 0 else 0.0,
        "macro_epe": float(np.mean(frame_epes)),
        "micro_epe": float(total_epe_sum / total_valid_pixels),
        "total_valid_pixels": total_valid_pixels,
        "raft_invocations": inv_count,
        "raft_invocation_rate": inv_rate,
        "raft_invocation_pct": inv_rate * 100.0,
        "compute_saving_pct": (1.0 - inv_rate) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 6: Actual End-to-End Cascade Timing Validation on Full Sintel Clean Benchmark."
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
        default="clean",
        choices=["clean", "final"],
        help="Sintel pass name",
    )
    parser.add_argument(
        "--flownet_checkpoint",
        type=str,
        default="checkpoints/flownets_EPE1.951.pth.tar",
        help="Path to FlowNetS checkpoint weights",
    )
    parser.add_argument(
        "--ablation_json",
        type=str,
        default="outputs/day6_gate_ablation.json",
        help="Path to Day 6 gate ablation report containing LOSO thresholds",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before timing",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_json_path = Path(args.ablation_json).resolve()

    print("==================================================================")
    print(" DAY 6: ACTUAL END-TO-END CASCADE TIMING VALIDATION (SINTEL CLEAN) ")
    print("==================================================================")
    print(f"Timestamp (UTC):    {datetime.now(timezone.utc).isoformat()}")
    print(f"Device:             {device} ({torch.cuda.get_device_name(device)})")
    print(f"Dataset Root:       {args.data_root}")
    print(f"Pass Name:          {args.pass_name}")
    print(f"Routing Statistic:  mag_q95 (95th percentile of forward magnitude)")
    print(f"Ablation Report:    {ablation_json_path}")
    print(f"Output Directory:   {output_dir}\n")

    # 1. Load LOSO thresholds
    thresholds_by_budget = load_loso_thresholds(ablation_json_path)

    # 2. Build Models
    print("Loading FlowNetS and RAFT models onto GPU...")
    flownet_model = build_flow_estimator(
        model_name="flownets",
        checkpoint_path=args.flownet_checkpoint,
        device=device,
    )
    flownet_model.eval()

    raft_model = build_flow_estimator(
        model_name="raft",
        device=device,
    )
    raft_model.eval()

    # 3. Load Sintel Dataset
    dataset = SintelDataset(root=args.data_root, split="training", pass_name=args.pass_name)
    print(f"Loaded SintelDataset: {len(dataset)} pairs across {len(set(s[4] for s in dataset.samples))} scenes.\n")

    # 4. Warmup
    warmup_models(flownet_model, raft_model, dataset, device, num_warmup=args.num_warmup)

    # 5. Measure Baselines
    timing_results: Dict[str, Any] = {}

    # Baseline 1: Always-FlowNetS
    fns_timing = benchmark_always_flownets(flownet_model, dataset, device)
    timing_results["always_flownets"] = fns_timing

    # Baseline 2: Always-RAFT
    raft_timing = benchmark_always_raft(raft_model, dataset, device)
    timing_results["always_raft"] = raft_timing

    # 6. Measure Real Cascades
    cascade_comparisons: List[Dict[str, Any]] = []

    for b_key, label in BUDGET_KEYS_TO_EVALUATE:
        target_pct = float(b_key.replace("pct", ""))
        scene_taus = thresholds_by_budget[b_key]

        casc_timing = benchmark_real_cascade(
            flownet_model=flownet_model,
            raft_model=raft_model,
            dataset=dataset,
            thresholds_by_scene=scene_taus,
            mode_name=label,
            target_budget_pct=target_pct,
            device=device,
        )
        timing_results[f"cascade_{b_key}"] = casc_timing

        # Compare actual measured latency against component-based additive model:
        # T_est = T_fast_path + rho_act * T_raft_only
        actual_lat = casc_timing["mean_latency_ms"]
        rho_act = casc_timing["raft_invocation_rate"]

        # Component-based estimate using the measured fast path and pure RAFT from this same benchmark run:
        fast_path_lat = casc_timing["fast_path_mean_latency_ms"]
        raft_only_lat = raft_timing["mean_latency_ms"]
        estimated_lat_current_run = fast_path_lat + rho_act * raft_only_lat

        # Also previous static estimate from Step 5/7 (13.51 ms gate + rho * 126.92 ms)
        static_est_lat = 13.50767 + rho_act * 126.92

        abs_diff_ms = actual_lat - estimated_lat_current_run
        rel_diff_pct = (abs_diff_ms / estimated_lat_current_run) * 100.0

        gap_to_raft = casc_timing["macro_epe"] - raft_timing["macro_epe"]
        err_red_vs_fns = (fns_timing["macro_epe"] - casc_timing["macro_epe"]) / fns_timing["macro_epe"] * 100.0
        speedup = raft_timing["mean_latency_ms"] / actual_lat

        comparison_record = {
            "mode": label,
            "target_budget_pct": target_pct,
            "actual_invocation_pct": casc_timing["raft_invocation_pct"],
            "actual_invocation_count": casc_timing["raft_invocations"],
            "macro_epe": casc_timing["macro_epe"],
            "gap_to_raft_epe": gap_to_raft,
            "error_reduction_pct": err_red_vs_fns,
            "actual_mean_latency_ms": actual_lat,
            "effective_fps": casc_timing["effective_fps"],
            "speedup_vs_always_raft": speedup,
            "fast_path_mean_ms": fast_path_lat,
            "accurate_path_mean_ms": casc_timing["accurate_path_mean_latency_ms"],
            "component_estimate_latency_ms": estimated_lat_current_run,
            "static_step5_estimate_latency_ms": static_est_lat,
            "absolute_diff_ms": abs_diff_ms,
            "relative_diff_pct": rel_diff_pct,
        }
        cascade_comparisons.append(comparison_record)

    # 7. Save JSON report
    output_data = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Day 6: Actual End-to-End Cascade Timing Validation on Full Sintel Clean Benchmark",
            "dataset": "MPI Sintel (training, clean pass)",
            "total_pairs": len(dataset),
            "total_scenes": len(set(s[4] for s in dataset.samples)),
            "routing_statistic": "mag_q95",
            "gpu_hardware": torch.cuda.get_device_name(device),
            "warmup_iterations": args.num_warmup,
        },
        "baselines": {
            "always_flownets": {
                "mean_latency_ms": fns_timing["mean_latency_ms"],
                "median_latency_ms": fns_timing["median_latency_ms"],
                "p95_latency_ms": fns_timing["p95_latency_ms"],
                "effective_fps": fns_timing["effective_fps"],
                "macro_epe": fns_timing["macro_epe"],
                "micro_epe": fns_timing["micro_epe"],
            },
            "always_raft": {
                "mean_latency_ms": raft_timing["mean_latency_ms"],
                "median_latency_ms": raft_timing["median_latency_ms"],
                "p95_latency_ms": raft_timing["p95_latency_ms"],
                "effective_fps": raft_timing["effective_fps"],
                "macro_epe": raft_timing["macro_epe"],
                "micro_epe": raft_timing["micro_epe"],
            },
        },
        "cascade_evaluations": timing_results,
        "cascade_vs_estimate_comparison": cascade_comparisons,
    }

    output_json_path = output_dir / "day6_actual_cascade_timing.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved timing validation report to: {output_json_path}")

    # 8. Print Comparative Summary Table
    print("\n=========================================================================================================")
    print("                     ACTUAL VS ESTIMATED CASCADE TIMING VALIDATION TABLE                                ")
    print("=========================================================================================================")
    print("| Mode | Invoc. % | Macro EPE | Actual Latency | Actual FPS | Speedup | Estimated Lat. | Diff (ms) | Diff (%) |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(
        f"| Always-FlowNetS    |   0.00%  | {fns_timing['macro_epe']:6.4f} px | "
        f"{fns_timing['mean_latency_ms']:6.2f} ms     | {fns_timing['effective_fps']:5.2f} FPS  |  {raft_timing['mean_latency_ms']/fns_timing['mean_latency_ms']:4.2f}x  | "
        f"     ---       |    ---    |   ---    |"
    )
    for c in cascade_comparisons:
        print(
            f"| {c['mode'][:18]:<18} | {c['actual_invocation_pct']:5.2f}%  | {c['macro_epe']:6.4f} px | "
            f"{c['actual_mean_latency_ms']:6.2f} ms     | {c['effective_fps']:5.2f} FPS  |  {c['speedup_vs_always_raft']:4.2f}x  | "
            f"{c['component_estimate_latency_ms']:6.2f} ms     | {c['absolute_diff_ms']:+5.2f} ms | {c['relative_diff_pct']:+5.2f}% |"
        )
    print(
        f"| Always-RAFT        | 100.00%  | {raft_timing['macro_epe']:6.4f} px | "
        f"{raft_timing['mean_latency_ms']:6.2f} ms     | {raft_timing['effective_fps']:5.2f} FPS  |  1.00x  | "
        f"     ---       |    ---    |   ---    |"
    )
    print("=========================================================================================================\n")


if __name__ == "__main__":
    main()
