"""
Evaluation Metrics for Optical Flow.

Provides standard End-Point Error (EPE) computation and benchmark aggregation
utilities supporting both 3D [2, H, W] and 4D [B, 2, H, W] tensor representations.
"""

from typing import Any, Dict, Optional, Sequence, Tuple
import torch


def compute_epe(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, int, float]:
    """
    Compute End-Point Error (EPE) between predicted and ground-truth optical flow.

    The Euclidean error is computed strictly over valid pixels indicated by valid_mask:
        EPE = sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)

    Supports both unbatched [2, H, W] and batched [B, 2, H, W] flow tensors.

    Args:
        pred_flow: Predicted flow tensor of shape [2, H, W] or [B, 2, H, W] in float32.
        gt_flow: Ground-truth flow tensor of shape matching pred_flow in float32.
        valid_mask: Optional boolean tensor of shape [H, W] or [B, H, W].
            True indicates valid evaluation pixels. If None, all pixels are considered valid.

    Returns:
        mean_epe: Average EPE across all valid pixels (0.0 if valid_count is 0).
        valid_count: Total number of valid pixels evaluated.
        epe_sum: Sum of EPE values across all valid pixels (useful for micro-averaging).
    """
    if pred_flow.shape != gt_flow.shape:
        raise ValueError(
            f"Shape mismatch between pred_flow {tuple(pred_flow.shape)} and gt_flow {tuple(gt_flow.shape)}"
        )

    if pred_flow.ndim == 3:
        if pred_flow.shape[0] != 2:
            raise ValueError(
                f"Expected 2 channels in channel dimension 0 for 3D tensor, got {pred_flow.shape[0]}"
            )
        diff = pred_flow - gt_flow
        epe = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)  # shape [H, W]

        if valid_mask is not None:
            if valid_mask.shape != epe.shape:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match spatial shape {tuple(epe.shape)}"
                )
            mask = valid_mask.bool()
            valid_epe = epe[mask]
        else:
            valid_epe = epe

    elif pred_flow.ndim == 4:
        if pred_flow.shape[1] != 2:
            raise ValueError(
                f"Expected 2 channels in channel dimension 1 for 4D tensor, got {pred_flow.shape[1]}"
            )
        diff = pred_flow - gt_flow
        epe = torch.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)  # shape [B, H, W]

        if valid_mask is not None:
            mask = valid_mask.bool()
            if mask.ndim == 3 and mask.shape == epe.shape:
                valid_epe = epe[mask]
            elif mask.ndim == 2 and mask.shape == epe.shape[1:]:
                # Broadcast [H, W] across batch [B, H, W]
                broadcast_mask = mask.unsqueeze(0).expand_as(epe)
                valid_epe = epe[broadcast_mask]
            else:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} cannot be broadcast to {tuple(epe.shape)}"
                )
        else:
            valid_epe = epe

    else:
        raise ValueError(
            f"Expected 3D [2, H, W] or 4D [B, 2, H, W] flow tensors, got shape {tuple(pred_flow.shape)}"
        )

    valid_count = int(valid_epe.numel())
    if valid_count > 0:
        epe_sum = float(valid_epe.sum().item())
        mean_epe = epe_sum / valid_count
    else:
        epe_sum = 0.0
        mean_epe = 0.0

    return mean_epe, valid_count, epe_sum


def aggregate_epe(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-frame evaluation records into micro and macro EPE statistics.

    - Micro EPE: Sum of all valid-pixel EPEs divided by total valid pixels across the dataset.
      Matches the canonical MPI Sintel benchmark leaderboard metric.
    - Macro EPE: Unweighted arithmetic mean of per-frame mean EPEs.

    Args:
        records: Sequence of dicts, each containing:
            - 'mean_epe': float
            - 'valid_count': int
            - 'epe_sum': float

    Returns:
        Dict containing:
            - 'micro_epe': float
            - 'macro_epe': float
            - 'total_valid_pixels': int
            - 'num_frames': int
    """
    if not records:
        return {
            "micro_epe": 0.0,
            "macro_epe": 0.0,
            "total_valid_pixels": 0,
            "num_frames": 0,
        }

    total_epe_sum = sum(float(r["epe_sum"]) for r in records)
    total_valid_pixels = sum(int(r["valid_count"]) for r in records)
    frame_means = [float(r["mean_epe"]) for r in records if int(r["valid_count"]) > 0]

    micro_epe = (total_epe_sum / total_valid_pixels) if total_valid_pixels > 0 else 0.0
    macro_epe = (sum(frame_means) / len(frame_means)) if frame_means else 0.0

    return {
        "micro_epe": micro_epe,
        "macro_epe": macro_epe,
        "total_valid_pixels": total_valid_pixels,
        "num_frames": len(records),
    }
