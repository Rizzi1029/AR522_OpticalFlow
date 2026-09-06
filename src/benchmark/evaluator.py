"""
MPI Sintel Optical Flow Evaluation Engine.

Orchestrates model warmup, per-frame inference timing via CUDA events,
EPE metric computation, and reproducible benchmark reporting.
"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import torch

from src.flow.wrappers import BaseOpticalFlowEstimator, build_flow_estimator
from src.benchmark.sintel_dataset import SintelDataset
from src.benchmark.metrics import compute_epe, aggregate_epe


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SintelEvaluator:
    """
    Evaluator for dense optical flow on MPI Sintel training split.

    Features:
    - GPU warmup before timed iterations.
    - Pure GPU inference latency tracking via CUDA events.
    - End-to-end wall-clock throughput accounting.
    - Zero large tensor retention for constant memory overhead.
    - Detailed hardware, environment, and checkpoint metadata logging.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        num_warmup: int = 5,
        **model_kwargs,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name.strip().lower()
        self.num_warmup = num_warmup

        # Build model adapter and record exact metadata for reproducibility
        self.model_metadata: Dict[str, Any] = {
            "model_name": self.model_name,
            "device": str(self.device),
        }

        if self.model_name in ("flownets", "flownet_s", "flownet"):
            project_root = Path(__file__).resolve().parent.parent.parent
            ckpt = Path(checkpoint_path).resolve() if checkpoint_path else (project_root / "checkpoints" / "flownets_EPE1.951.pth.tar")
            if not ckpt.is_file():
                raise FileNotFoundError(f"FlowNetS checkpoint not found: {ckpt}")

            self.model: BaseOpticalFlowEstimator = build_flow_estimator(
                model_name="flownets",
                checkpoint_path=ckpt,
                device=self.device,
            )
            self.model_metadata.update({
                "architecture": "FlowNetS",
                "checkpoint_path": str(ckpt),
                "checkpoint_size_bytes": ckpt.stat().st_size,
                "checkpoint_sha256": _compute_file_sha256(ckpt),
                "div_flow": 20.0,
                "spatial_pad_multiple": 64,
            })

        elif self.model_name == "raft":
            self.model = build_flow_estimator(
                model_name="raft",
                device=self.device,
                **model_kwargs,
            )
            self.model_metadata.update({
                "architecture": "RAFT (raft_large)",
                "weights_enum": "Raft_Large_Weights.C_T_SKHT_V2",
                "num_flow_updates": model_kwargs.get("num_flow_updates", 12),
                "spatial_pad_multiple": 8,
            })
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        self.model.eval()

    def _warmup(self, sample_frame1: torch.Tensor, sample_frame2: torch.Tensor) -> None:
        """Execute warmup passes to trigger CUDA kernel compilation and memory caching."""
        if self.num_warmup <= 0:
            return

        f1 = sample_frame1.clone()
        f2 = sample_frame2.clone()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(self.num_warmup):
            _ = self.model(f1, f2)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def evaluate_dataset(
        self,
        dataset: SintelDataset,
        progress_callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate optical flow across all frame pairs in the dataset.

        Args:
            dataset: Configured SintelDataset instance.
            progress_callback: Optional callback(scene, frame_idx, total_frames, frame_record).

        Returns:
            Structured dictionary with metadata, overall metrics, scene breakdown,
            and per-frame statistics.
        """
        num_pairs = len(dataset)
        if num_pairs == 0:
            raise ValueError("Dataset contains zero samples to evaluate.")

        # Warmup on the first frame pair (results discarded from timing and metrics)
        first_f1, first_f2, _, _, _ = dataset[0]
        self._warmup(first_f1, first_f2)

        is_cuda = (self.device.type == "cuda")
        per_frame_records: List[Dict[str, Any]] = []
        scene_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        latencies_ms: List[float] = []

        wall_clock_start = time.perf_counter()

        for idx in range(num_pairs):
            frame1, frame2, flow_gt, valid_mask, meta = dataset[idx]

            # Measure pure model inference latency using CUDA events
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                pred_flow = self.model(frame1, frame2)
                end_event.record()
                torch.cuda.synchronize()
                latency_ms = float(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                pred_flow = self.model(frame1, frame2)
                latency_ms = float((time.perf_counter() - t0) * 1000.0)

            latencies_ms.append(latency_ms)

            # Move ground-truth and mask to computation device for EPE
            flow_gt_dev = flow_gt.to(device=self.device)
            valid_mask_dev = valid_mask.to(device=self.device)

            # Compute End-Point Error strictly over valid pixels
            mean_epe, valid_count, epe_sum = compute_epe(
                pred_flow[0],
                flow_gt_dev,
                valid_mask_dev,
            )

            # Explicitly delete GPU tensors to guarantee flat memory usage
            del pred_flow, flow_gt_dev, valid_mask_dev

            record = {
                "scene": meta["scene"],
                "frame_idx": meta["frame_idx"],
                "mean_epe": float(mean_epe),
                "valid_count": int(valid_count),
                "epe_sum": float(epe_sum),
                "inference_latency_ms": float(latency_ms),
            }
            per_frame_records.append(record)
            scene_records[meta["scene"]].append(record)

            if progress_callback is not None:
                progress_callback(meta["scene"], meta["frame_idx"], num_pairs, record)

        wall_clock_total_sec = time.perf_counter() - wall_clock_start

        # Compute dataset-level EPE metrics using standard aggregation helper
        overall_epe_agg = aggregate_epe(per_frame_records)

        # Compute per-scene breakdown
        per_scene_summary: Dict[str, Dict[str, Any]] = {}
        for scene, s_records in scene_records.items():
            s_agg = aggregate_epe(s_records)
            s_latencies = [r["inference_latency_ms"] for r in s_records]
            s_mean_lat = float(np.mean(s_latencies)) if s_latencies else 0.0
            per_scene_summary[scene] = {
                "num_frames": len(s_records),
                "micro_epe": s_agg["micro_epe"],
                "macro_epe": s_agg["macro_epe"],
                "total_valid_pixels": s_agg["total_valid_pixels"],
                "mean_inference_latency_ms": s_mean_lat,
                "inference_fps": (1000.0 / s_mean_lat) if s_mean_lat > 0 else 0.0,
            }

        # Accurate timing statistics computed from actual per-frame measurements
        mean_latency = float(np.mean(latencies_ms))
        median_latency = float(np.median(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        inference_fps = (1000.0 / mean_latency) if mean_latency > 0 else 0.0
        pipeline_throughput_fps = num_pairs / wall_clock_total_sec if wall_clock_total_sec > 0 else 0.0

        # System and hardware metadata
        hardware_info: Dict[str, Any] = {
            "device": str(self.device),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }
        if is_cuda:
            hardware_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            })

        return {
            "benchmark_metadata": {
                "dataset": "MPI Sintel (training)",
                "split": dataset.split,
                "pass_name": dataset.pass_name,
                "num_scenes": len(scene_records),
                "total_frame_pairs": len(per_frame_records),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_info": self.model_metadata,
                "hardware_info": hardware_info,
            },
            "overall_summary": {
                "micro_epe": overall_epe_agg["micro_epe"],
                "macro_epe": overall_epe_agg["macro_epe"],
                "total_valid_pixels": overall_epe_agg["total_valid_pixels"],
                "timing": {
                    "warmup_iterations": self.num_warmup,
                    "mean_inference_latency_ms": mean_latency,
                    "median_inference_latency_ms": median_latency,
                    "p95_inference_latency_ms": p95_latency,
                    "inference_fps": inference_fps,
                    "wall_clock_total_sec": float(wall_clock_total_sec),
                    "end_to_end_throughput_fps": float(pipeline_throughput_fps),
                },
            },
            "scene_breakdown": per_scene_summary,
            "frame_records": per_frame_records,
        }
