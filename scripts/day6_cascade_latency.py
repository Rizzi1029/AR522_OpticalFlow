"""
Day 6 Step 4: Latency & Compute Cost Benchmarking for Selective Flow Cascading.

Measures wall-clock inference latency using CUDA events across the 8 canonical Sintel samples:
1. FlowNetS forward
2. FlowNetS forward + backward + FB residual computation
3. RAFT forward
4. Always-RAFT baseline (pure RAFT forward on every frame)
5. Selective cascade at 50% RAFT invocation
6. Selective cascade at 87.5% RAFT invocation

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
Protocol:
- Process samples with warmups.
- Multiple timed repetitions using torch.cuda.Event(enable_timing=True).
- For selective cascades, execute FlowNetS FB evaluation; only invoke RAFT if gate triggers.
- Reports mean, median, p95 latency (ms) and effective FPS.
- Saves outputs/day6_cascade_latency.json.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator

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

# Calibrated operating thresholds on FlowNetS fb_median from Day 6 Step 2
THRESHOLD_50PCT = 0.7093   # 50.0% invocation (escalates samples 4, 5, 6, 7; keeps 0, 1, 2, 3)
THRESHOLD_87PCT = 0.2181   # 87.5% invocation (escalates 7 samples; keeps sample 3 bamboo_2)


def compute_forward_backward_residual(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scalar FB consistency residual on GPU via grid_sample warping.
    Args:
        flow_fwd: [2, H, W] tensor on GPU.
        flow_bwd: [2, H, W] tensor on GPU.
    Returns:
        fb_residual: [H, W] float32 Euclidean residual map on GPU.
        in_bounds_mask: [H, W] bool coordinate in-bounds mask on GPU.
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


def load_dataset_samples(data_root: str) -> List[Dict[str, Any]]:
    """Loads input frame tensors for all 8 canonical Sintel pairs into host memory."""
    loaded = []
    for spec in CANONICAL_SAMPLES:
        dataset = SintelDataset(
            root=data_root,
            split="training",
            pass_name=spec["pass_name"],
            scenes=[spec["scene"]],
        )
        f1, f2, flow_gt, valid_mask, meta = dataset[spec["pair_idx"]]
        loaded.append({
            "spec": spec,
            "label": f"{spec['scene']} ({spec['pass_name']})",
            "frame1": f1,
            "frame2": f2,
            "valid_mask": valid_mask,
        })
    return loaded


def benchmark_mode(
    mode_name: str,
    samples_data: List[Dict[str, Any]],
    flownet_model: Any,
    raft_model: Any,
    num_warmup: int = 5,
    num_reps: int = 10,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Executes a timing benchmark for a specific execution mode across all 8 samples.
    """
    device = next(flownet_model.parameters()).device
    all_latencies_ms: List[float] = []
    per_sample_latencies: Dict[str, List[float]] = {s["label"]: [] for s in samples_data}
    invocation_counts: Dict[str, int] = {s["label"]: 0 for s in samples_data}

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for s in samples_data:
            f1 = s["frame1"].to(device)
            f2 = s["frame2"].to(device)
            m_valid = s["valid_mask"].to(device)

            # Warmup runs (not timed)
            for _ in range(num_warmup):
                if mode_name == "flownets_forward":
                    _ = flownet_model(f1, f2)
                elif mode_name == "flownets_fb":
                    f_fwd = flownet_model(f1, f2)[0]
                    f_bwd = flownet_model(f2, f1)[0]
                    fb_res, in_b = compute_forward_backward_residual(f_fwd, f_bwd)
                    mask = m_valid & in_b
                    _ = torch.median(fb_res[mask])
                elif mode_name in ["raft_forward", "always_raft"]:
                    _ = raft_model(f1, f2)
                elif mode_name in ["cascade_50pct", "cascade_87pct"]:
                    f_fwd = flownet_model(f1, f2)[0]
                    f_bwd = flownet_model(f2, f1)[0]
                    fb_res, in_b = compute_forward_backward_residual(f_fwd, f_bwd)
                    mask = m_valid & in_b
                    gate_val = float(torch.median(fb_res[mask]).item())
                    if gate_val > threshold:
                        _ = raft_model(f1, f2)
            torch.cuda.synchronize()

            # Timed repetitions
            for _ in range(num_reps):
                torch.cuda.synchronize()

                if mode_name == "flownets_forward":
                    start_event.record()
                    _ = flownet_model(f1, f2)
                    end_event.record()

                elif mode_name == "flownets_fb":
                    start_event.record()
                    f_fwd = flownet_model(f1, f2)[0]
                    f_bwd = flownet_model(f2, f1)[0]
                    fb_res, in_b = compute_forward_backward_residual(f_fwd, f_bwd)
                    mask = m_valid & in_b
                    _ = torch.median(fb_res[mask])
                    end_event.record()

                elif mode_name in ["raft_forward", "always_raft"]:
                    start_event.record()
                    _ = raft_model(f1, f2)
                    end_event.record()

                elif mode_name in ["cascade_50pct", "cascade_87pct"]:
                    start_event.record()
                    f_fwd = flownet_model(f1, f2)[0]
                    f_bwd = flownet_model(f2, f1)[0]
                    fb_res, in_b = compute_forward_backward_residual(f_fwd, f_bwd)
                    mask = m_valid & in_b
                    gate_val = float(torch.median(fb_res[mask]).item())

                    if gate_val > threshold:
                        _ = raft_model(f1, f2)
                        invoked = True
                    else:
                        invoked = False
                    end_event.record()

                    if invoked:
                        invocation_counts[s["label"]] += 1

                torch.cuda.synchronize()
                dur_ms = float(start_event.elapsed_time(end_event))
                all_latencies_ms.append(dur_ms)
                per_sample_latencies[s["label"]].append(dur_ms)

    lat_arr = np.array(all_latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(lat_arr))
    med_ms = float(np.median(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    min_ms = float(np.min(lat_arr))
    max_ms = float(np.max(lat_arr))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    sample_summary: Dict[str, Dict[str, float]] = {}
    for s_lbl, s_lats in per_sample_latencies.items():
        s_arr = np.array(s_lats, dtype=np.float64)
        sample_summary[s_lbl] = {
            "mean_ms": float(np.mean(s_arr)),
            "median_ms": float(np.median(s_arr)),
            "p95_ms": float(np.percentile(s_arr, 95)),
            "raft_invoked": bool(invocation_counts[s_lbl] > 0) if "cascade" in mode_name else (mode_name in ["raft_forward", "always_raft"]),
        }

    return {
        "mode_name": mode_name,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": med_ms,
        "p95_latency_ms": p95_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "effective_fps": fps,
        "num_measurements": len(all_latencies_ms),
        "per_sample": sample_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 6 Step 4: Latency & compute cost benchmark for selective flow cascades."
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
        help="Directory to save Day 6 artifacts (default: outputs)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations per sample (default: 5)",
    )
    parser.add_argument(
        "--num_reps",
        type=int,
        default=10,
        help="Number of timed repetitions per sample (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to benchmark on ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("==================================================================")
    print("      DAY 6 STEP 4: SELECTIVE CASCADE LATENCY BENCHMARK           ")
    print("==================================================================")
    print(f"Device:           {device} ({gpu_name})")
    print(f"Dataset Root:     {args.data_root}")
    print(f"Samples:          8 canonical Sintel pairs")
    print(f"Warmup / Reps:    {args.num_warmup} warmups / {args.num_reps} repetitions per sample")
    print(f"Total Trials:     {len(CANONICAL_SAMPLES) * args.num_reps} measurements per mode")
    print(f"Output Directory: {output_dir}\n")

    # 1. Load models
    print("Loading FlowNetS model...")
    flownet_model = build_flow_estimator(
        model_name="flownets",
        checkpoint_path=args.flownet_checkpoint,
        device=device,
    )

    print("Loading RAFT model...")
    raft_model = build_flow_estimator(
        model_name="raft",
        device=device,
    )
    print("Models loaded successfully.\n")

    # 2. Load dataset sample tensors
    print("Loading 8 canonical Sintel frames into memory...")
    samples_data = load_dataset_samples(args.data_root)
    print("Frames loaded successfully.\n")

    # 3. Benchmark all 6 requested modes
    modes_config = [
        {"key": "flownets_forward", "name": "FlowNetS Forward", "th": 0.0},
        {"key": "flownets_fb", "name": "FlowNetS Forward + Backward (FB Check)", "th": 0.0},
        {"key": "raft_forward", "name": "RAFT Forward", "th": 0.0},
        {"key": "always_raft", "name": "Always-RAFT Baseline", "th": 0.0},
        {"key": "cascade_50pct", "name": "Selective Cascade (50% Invocation)", "th": THRESHOLD_50PCT},
        {"key": "cascade_87pct", "name": "Selective Cascade (87.5% Invocation)", "th": THRESHOLD_87PCT},
    ]

    benchmark_results: Dict[str, Any] = {}

    for cfg in modes_config:
        m_key = cfg["key"]
        m_name = cfg["name"]
        th = cfg["th"]
        print(f"Benchmarking: {m_name} ... ", end="", flush=True)

        res = benchmark_mode(
            mode_name=m_key,
            samples_data=samples_data,
            flownet_model=flownet_model,
            raft_model=raft_model,
            num_warmup=args.num_warmup,
            num_reps=args.num_reps,
            threshold=th,
        )
        benchmark_results[m_key] = res
        print(f"done -> Mean: {res['mean_latency_ms']:.2f} ms | Median: {res['median_latency_ms']:.2f} ms | P95: {res['p95_latency_ms']:.2f} ms | FPS: {res['effective_fps']:.1f}")

    # 4. Comparative Speedup & Savings Analysis relative to Always-RAFT
    ref_raft_ms = benchmark_results["always_raft"]["mean_latency_ms"]

    comparisons: Dict[str, Any] = {}
    for m_key in benchmark_results:
        m_ms = benchmark_results[m_key]["mean_latency_ms"]
        speedup = ref_raft_ms / m_ms if m_ms > 0 else 0.0
        time_saved_pct = (ref_raft_ms - m_ms) / ref_raft_ms * 100.0 if ref_raft_ms > 0 else 0.0
        comparisons[m_key] = {
            "mean_latency_ms": m_ms,
            "speedup_vs_always_raft": speedup,
            "time_saved_pct": time_saved_pct,
            "effective_fps": benchmark_results[m_key]["effective_fps"],
        }

    # 5. Save structured JSON
    payload = {
        "metadata": {
            "experiment": "day6_cascade_latency_benchmark",
            "device": str(device),
            "gpu_name": gpu_name,
            "num_samples": len(CANONICAL_SAMPLES),
            "warmup_iterations": args.num_warmup,
            "repetitions_per_sample": args.num_reps,
            "total_measurements_per_mode": len(CANONICAL_SAMPLES) * args.num_reps,
            "threshold_50pct": THRESHOLD_50PCT,
            "threshold_87pct": THRESHOLD_87PCT,
        },
        "modes": benchmark_results,
        "comparisons_vs_always_raft": comparisons,
    }

    json_path = output_dir / "day6_cascade_latency.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved structured latency JSON to: {json_path}")

    # 6. Terminal Summary Report
    print("\n==========================================================================================")
    print("                    CASCADE LATENCY & THROUGHPUT BENCHMARK SUMMARY                        ")
    print("==========================================================================================")
    print(f"{'Execution Mode':<42} | {'Mean (ms)':>10} | {'Median (ms)':>11} | {'P95 (ms)':>9} | {'FPS':>7} | {'Speedup vs RAFT':>15}")
    print("-" * 106)
    for cfg in modes_config:
        k = cfg["key"]
        r = benchmark_results[k]
        c = comparisons[k]
        print(
            f"{cfg['name']:<42} | "
            f"{r['mean_latency_ms']:10.2f} | "
            f"{r['median_latency_ms']:11.2f} | "
            f"{r['p95_latency_ms']:9.2f} | "
            f"{r['effective_fps']:7.1f} | "
            f"{c['speedup_vs_always_raft']:14.2f}x"
        )
    print("==========================================================================================\n")


if __name__ == "__main__":
    main()
