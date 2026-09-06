"""
Command-Line Benchmark Runner for MPI Sintel Optical Flow Evaluation.

Evaluates FlowNetS and RAFT on MPI Sintel clean and final passes,
computing micro/macro EPE, tracking pure GPU inference latency and
end-to-end pipeline throughput, and generating reproducible reports.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.benchmark.evaluator import SintelEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPI Sintel optical flow benchmark for FlowNetS and RAFT."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["flownets", "raft", "both"],
        default="both",
        help="Model(s) to benchmark (default: both).",
    )
    parser.add_argument(
        "--pass_name",
        type=str,
        choices=["clean", "final", "both"],
        default="both",
        help="Sintel pass(es) to evaluate on (default: both).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Sintel",
        help="Root path to Sintel dataset (default: data/Sintel).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom FlowNetS checkpoint path (default: checkpoints/flownets_EPE1.951.pth.tar).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional list of specific scenes to evaluate (default: all scenes).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of frame pairs to evaluate per pass (for smoke tests).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/sintel_benchmark",
        help="Output directory to save benchmark results (default: reports/sintel_benchmark).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before recording timing (default: 5).",
    )
    return parser.parse_args()


def print_scene_table(results: Dict[str, Any]) -> None:
    """Print formatted markdown table of scene breakdowns."""
    print("\n| Scene | Frames | Micro EPE (px) | Macro EPE (px) | Mean Latency (ms) | Inference FPS |")
    print("|---|---|---|---|---|---|")
    for scene, data in sorted(results["scene_breakdown"].items()):
        print(
            f"| {scene} | {data['num_frames']} | {data['micro_epe']:.4f} | "
            f"{data['macro_epe']:.4f} | {data['mean_inference_latency_ms']:.2f} | "
            f"{data['inference_fps']:.1f} |"
        )
    overall = results["overall_summary"]
    timing = overall["timing"]
    print(
        f"| **OVERALL** | **{results['benchmark_metadata']['total_frame_pairs']}** | "
        f"**{overall['micro_epe']:.4f}** | **{overall['macro_epe']:.4f}** | "
        f"**{timing['mean_inference_latency_ms']:.2f}** | **{timing['inference_fps']:.1f}** |\n"
    )


def generate_comparison_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate markdown summary comparing evaluated model/pass configurations."""
    lines = [
        "# MPI Sintel Optical Flow Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Overall Comparison",
        "",
        "| Model | Pass | Total Pairs | Micro EPE (px) | Macro EPE (px) | Mean Latency (ms) | P95 Latency (ms) | Inference FPS | Throughput (FPS) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for res in all_results:
        meta = res["benchmark_metadata"]
        overall = res["overall_summary"]
        timing = overall["timing"]
        lines.append(
            f"| {meta['model_info']['model_name']} | {meta['pass_name']} | "
            f"{meta['total_frame_pairs']} | {overall['micro_epe']:.4f} | "
            f"{overall['macro_epe']:.4f} | {timing['mean_inference_latency_ms']:.2f} | "
            f"{timing['p95_inference_latency_ms']:.2f} | {timing['inference_fps']:.1f} | "
            f"{timing['end_to_end_throughput_fps']:.1f} |"
        )

    lines.append("")
    summary_path = output_dir / "summary_comparison.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison report saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["flownets", "raft"] if args.model == "both" else [args.model]
    passes = ["clean", "final"] if args.pass_name == "both" else [args.pass_name]

    print("================================================================")
    print("      AR522 OPTICAL FLOW: MPI SINTEL BENCHMARK HARNESS          ")
    print("================================================================")
    print(f"Device:       {args.device} ({torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'})")
    print(f"Models:       {models}")
    print(f"Passes:       {passes}")
    print(f"Output Dir:   {output_dir}")
    if args.scenes:
        print(f"Scene Filter: {args.scenes}")
    if args.limit:
        print(f"Limit:        {args.limit} pairs per pass")
    print("================================================================\n")

    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        for pass_name in passes:
            print(f"\n>>> Running Benchmark: Model='{model_name}' | Pass='{pass_name}' <<<")

            # 1. Instantiate dataset
            dataset = SintelDataset(
                root=args.data_root,
                split="training",
                pass_name=pass_name,
                scenes=args.scenes,
            )

            # Apply sample limit if requested
            if args.limit is not None and args.limit < len(dataset):
                dataset.samples = dataset.samples[:args.limit]

            print(f"Dataset ready: {len(dataset)} frame pairs across {len(set(s[4] for s in dataset.samples))} scene(s)")

            # 2. Instantiate evaluator
            evaluator = SintelEvaluator(
                model_name=model_name,
                checkpoint_path=args.checkpoint,
                device=args.device,
                num_warmup=args.num_warmup,
            )

            # 3. Progress callback for console output
            current_scene = None

            def on_progress(scene: str, frame_idx: int, total: int, rec: Dict[str, Any]):
                nonlocal current_scene
                if scene != current_scene:
                    current_scene = scene
                    print(f"  Evaluating Scene: {scene:15s} (frame {frame_idx:04d})...", end="", flush=True)
                elif frame_idx % 20 == 0:
                    print(f" {frame_idx:04d}", end="", flush=True)

            print(f"Warming up ({args.num_warmup} passes)...")
            results = evaluator.evaluate_dataset(dataset, progress_callback=on_progress)
            print(" Done.\n")

            # 4. Save JSON results
            out_file = output_dir / f"{model_name}_{pass_name}.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {out_file}")

            # 5. Print summary table
            print_scene_table(results)
            all_results.append(results)

    if len(all_results) > 1:
        generate_comparison_report(all_results, output_dir)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
