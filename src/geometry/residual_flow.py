"""
Residual Optical Flow Computation for Independent Motion Analysis.

Theoretical Formulation:
-------------------------
Given:
  1. Observed optical flow field w_obs = (u_obs, v_obs) (from ground truth or a flow estimator like RAFT/FlowNet).
  2. Rigid camera-induced background flow w_rigid = (u_rigid, v_rigid) (from camera ego-motion and depth).

The optical flow residual vector field w_res is defined as:
  w_res = w_obs - w_rigid

Component-wise:
  u_res = u_obs - u_rigid
  v_res = v_obs - v_rigid

The residual magnitude (Endpoint Error / kinematic deviation) is:
  ||w_res|| = sqrt(u_res^2 + v_res^2)

CRITICAL ARCHITECTURAL DISTINCTION:
-----------------------------------
RESIDUAL OPTICAL FLOW IS NOT YET DYNAMIC-OBJECT SEGMENTATION.
Residual optical flow measures the kinematic discrepancy between observed image displacement
and predicted camera-induced rigid background motion.

A non-zero residual does NOT necessarily imply the presence of an independently moving object.
Non-zero background residuals regularly arise from:
  1. Monocular depth scale and estimation error: When using monocular proxy depth (e.g. MiDaS)
     under camera translation, depth is known only up to an unknown affine scale and shift.
     Any depth error or scale mismatch scales the translational rigid flow:
       delta_w_rigid ~ t_rel * delta(1/Z)
     leading to non-zero residuals even on perfectly static rigid background!
  2. Optical flow estimation inaccuracies: Aperture problem, textureless regions, occlusions,
     specular highlights, motion blur, and network hallucination errors in w_obs.
  3. Non-rigid deformations: Foliage, clothing, flexible surfaces, and atmospheric distortion.
  4. Parallax discontinuities: Sub-pixel misalignment near depth boundaries.

True dynamic-object segmentation (Step 4) requires adaptive uncertainty gating, geometric
confidence weighting, statistical thresholding, and spatial clustering to isolate true
independent actors from depth/flow reconstruction noise.

Coordinate & Shape Conventions:
-------------------------------
Supports both:
  - Channels-first: [2, H, W] or [B, 2, H, W] (e.g. Sintel .flo ground truth, PyTorch models)
  - Channels-last:  [H, W, 2] or [B, H, W, 2] (e.g. OpenCV, numpy image arrays)
  Channel 0: horizontal displacement (delta_u, column axis)
  Channel 1: vertical displacement (delta_v, row axis)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def compute_residual_flow(
    observed_flow: Union[torch.Tensor, np.ndarray],
    rigid_flow: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    zero_invalid: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute residual optical flow: w_res = w_obs - w_rigid.

    Reuses and respects the rigid-flow validity mask, enforcing finiteness
    and zeroing out invalid/masked pixels consistently.

    Args:
        observed_flow: Observed optical flow as [2, H, W], [H, W, 2], [B, 2, H, W], or [B, H, W, 2].
        rigid_flow: Camera-induced rigid flow in compatible spatial resolution.
        valid_mask: Optional boolean validity mask ([H, W], [1, H, W], [B, H, W], or [B, 1, H, W]).
        zero_invalid: If True, sets invalid flow vectors to 0.0 (default: True).

    Returns:
        residual_flow: Tensor matching the shape, layout, and device of observed_flow.
        combined_valid_mask: Boolean validity mask tensor indicating valid residual pixels.
    """
    # Convert observed_flow to torch tensor
    if isinstance(observed_flow, np.ndarray):
        obs_t = torch.tensor(observed_flow, dtype=torch.float32)
    else:
        obs_t = observed_flow
    device = obs_t.device
    dtype = obs_t.dtype

    # Convert rigid_flow to torch tensor on same device and dtype
    if isinstance(rigid_flow, np.ndarray):
        rig_t = torch.tensor(rigid_flow, device=device, dtype=dtype)
    else:
        rig_t = rigid_flow.to(device=device, dtype=dtype)

    # Determine layout of observed_flow
    # Supported:
    # 2D inputs: [2, H, W] (channels-first) or [H, W, 2] (channels-last)
    # 4D inputs: [B, 2, H, W] (channels-first) or [B, H, W, 2] (channels-last)
    is_4d = obs_t.ndim == 4
    if is_4d:
        if obs_t.shape[1] == 2:
            channels_first = True
            b, c, h, w = obs_t.shape
        elif obs_t.shape[-1] == 2:
            channels_first = False
            b, h, w, c = obs_t.shape
        else:
            raise ValueError(f"Expected channel dimension 2 in 4D observed_flow, got {obs_t.shape}")
    elif obs_t.ndim == 3:
        if obs_t.shape[0] == 2:
            channels_first = True
            c, h, w = obs_t.shape
        elif obs_t.shape[-1] == 2:
            channels_first = False
            h, w, c = obs_t.shape
        else:
            raise ValueError(f"Expected channel dimension 2 in 3D observed_flow, got {obs_t.shape}")
    else:
        raise ValueError(f"Unsupported observed_flow tensor shape: {tuple(obs_t.shape)}")

    # Align rigid_flow to match observed_flow layout
    if is_4d:
        if channels_first and rig_t.shape[-1] == 2:
            rig_aligned = rig_t.permute(0, 3, 1, 2)
        elif not channels_first and rig_t.shape[1] == 2:
            rig_aligned = rig_t.permute(0, 2, 3, 1)
        else:
            rig_aligned = rig_t
    else:
        if channels_first and rig_t.ndim == 3 and rig_t.shape[-1] == 2:
            rig_aligned = rig_t.permute(2, 0, 1)
        elif not channels_first and rig_t.ndim == 3 and rig_t.shape[0] == 2:
            rig_aligned = rig_t.permute(1, 2, 0)
        else:
            rig_aligned = rig_t

    if obs_t.shape != rig_aligned.shape:
        raise ValueError(
            f"Aligned flow shape mismatch: observed={tuple(obs_t.shape)} vs rigid={tuple(rig_aligned.shape)}"
        )

    # Compute raw residual flow: w_res = w_obs - w_rigid
    residual = obs_t - rig_aligned

    # Build combined validity mask
    if channels_first:
        channel_dim = 1 if is_4d else 0
    else:
        channel_dim = -1

    finite_obs = torch.isfinite(obs_t).all(dim=channel_dim, keepdim=is_4d and channels_first)
    finite_rig = torch.isfinite(rig_aligned).all(dim=channel_dim, keepdim=is_4d and channels_first)
    combined_mask = finite_obs & finite_rig

    if valid_mask is not None:
        if isinstance(valid_mask, np.ndarray):
            vmask_t = torch.tensor(valid_mask, device=device, dtype=torch.bool)
        else:
            vmask_t = valid_mask.to(device=device, dtype=torch.bool)

        # Standardize mask shape
        if is_4d:
            if channels_first and vmask_t.ndim == 3:
                vmask_t = vmask_t.unsqueeze(1)
            elif not channels_first and vmask_t.ndim == 4 and vmask_t.shape[1] == 1:
                vmask_t = vmask_t.squeeze(1)
        else:
            if vmask_t.ndim == 3 and vmask_t.shape[0] == 1:
                vmask_t = vmask_t.squeeze(0)

        combined_mask = combined_mask & vmask_t

    # Apply zero_invalid masking
    if zero_invalid:
        if is_4d:
            if channels_first:
                mask_expanded = combined_mask.expand_as(residual)
                residual = torch.where(mask_expanded, residual, torch.zeros_like(residual))
            else:
                residual = torch.where(combined_mask.unsqueeze(-1), residual, torch.zeros_like(residual))
        else:
            if channels_first:
                mask_expanded = combined_mask.unsqueeze(0).expand_as(residual)
                residual = torch.where(mask_expanded, residual, torch.zeros_like(residual))
            else:
                residual = torch.where(combined_mask.unsqueeze(-1), residual, torch.zeros_like(residual))

    return residual, combined_mask


def compute_residual_magnitude(
    residual_flow: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """
    Compute Euclidean norm magnitude of the residual flow: ||w_res|| = sqrt(u^2 + v^2).

    Args:
        residual_flow: Residual flow tensor [2, H, W], [H, W, 2], [B, 2, H, W], or [B, H, W, 2].
        valid_mask: Optional boolean validity mask. If provided, invalid pixels are zeroed.

    Returns:
        magnitude: [H, W] or [B, H, W] float32 tensor representing residual displacement in pixels.
    """
    if isinstance(residual_flow, np.ndarray):
        res_t = torch.tensor(residual_flow, dtype=torch.float32)
    else:
        res_t = residual_flow

    is_4d = res_t.ndim == 4
    if is_4d:
        if res_t.shape[1] == 2:
            mag = torch.norm(res_t, dim=1)  # [B, H, W]
        else:
            mag = torch.norm(res_t, dim=-1)  # [B, H, W]
    else:
        if res_t.shape[0] == 2:
            mag = torch.norm(res_t, dim=0)  # [H, W]
        else:
            mag = torch.norm(res_t, dim=-1)  # [H, W]

    if valid_mask is not None:
        if isinstance(valid_mask, np.ndarray):
            vmask_t = torch.tensor(valid_mask, device=mag.device, dtype=torch.bool)
        else:
            vmask_t = valid_mask.to(device=mag.device, dtype=torch.bool)
        if vmask_t.ndim == 3 and vmask_t.shape[0] == 1 and not is_4d:
            vmask_t = vmask_t.squeeze(0)
        elif is_4d and vmask_t.ndim == 4 and vmask_t.shape[1] == 1:
            vmask_t = vmask_t.squeeze(1)
        mag = torch.where(vmask_t, mag, torch.zeros_like(mag))

    return mag


def compute_residual_statistics(
    residual_flow: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute summary metrics on residual optical flow over valid pixels.

    Args:
        residual_flow: Residual optical flow tensor.
        valid_mask: Optional boolean validity mask.

    Returns:
        stats: Dictionary containing:
            - 'valid_ratio': percentage of pixels marked valid in [0, 100]
            - 'mean_epe': mean residual endpoint error (pixels)
            - 'median_epe': median residual endpoint error (pixels)
            - 'p95_epe': 95th percentile residual endpoint error (pixels)
            - 'max_epe': maximum residual endpoint error (pixels)
            - 'rmse': root mean square error (pixels)
            - 'finite_ratio': percentage of pixels with finite values in [0, 100]
    """
    if isinstance(residual_flow, np.ndarray):
        res_t = torch.tensor(residual_flow, dtype=torch.float32)
    else:
        res_t = residual_flow

    if res_t.ndim == 4 and res_t.shape[1] == 2:
        ch_dim = 1
    elif res_t.ndim == 3 and res_t.shape[0] == 2:
        ch_dim = 0
    else:
        ch_dim = -1
    is_finite = torch.isfinite(res_t).all(dim=ch_dim)
    finite_ratio = float(is_finite.sum().item() / max(is_finite.numel(), 1) * 100.0)

    mag = compute_residual_magnitude(res_t, valid_mask=None)

    if valid_mask is not None:
        if isinstance(valid_mask, np.ndarray):
            mask_t = torch.tensor(valid_mask, device=mag.device, dtype=torch.bool)
        else:
            mask_t = valid_mask.to(device=mag.device, dtype=torch.bool)
        if mask_t.ndim == 4 and mask_t.shape[1] == 1:
            mask_t = mask_t.squeeze(1)
        elif mask_t.ndim == 3 and mask_t.shape[0] == 1 and mag.ndim == 2:
            mask_t = mask_t.squeeze(0)
        active_mask = mask_t & is_finite
    else:
        active_mask = is_finite

    valid_count = int(active_mask.sum().item())
    total_count = int(active_mask.numel())
    valid_ratio = float(valid_count / max(total_count, 1) * 100.0)

    if valid_count == 0:
        return {
            "valid_ratio": 0.0,
            "mean_epe": 0.0,
            "median_epe": 0.0,
            "p95_epe": 0.0,
            "max_epe": 0.0,
            "rmse": 0.0,
            "finite_ratio": finite_ratio,
        }

    valid_mags = mag[active_mask]

    mean_epe = float(valid_mags.mean().item())
    median_epe = float(valid_mags.median().item())
    p95_epe = float(torch.quantile(valid_mags, 0.95).item())
    max_epe = float(valid_mags.max().item())
    rmse = float(torch.sqrt((valid_mags ** 2).mean()).item())

    return {
        "valid_ratio": valid_ratio,
        "mean_epe": mean_epe,
        "median_epe": median_epe,
        "p95_epe": p95_epe,
        "max_epe": max_epe,
        "rmse": rmse,
        "finite_ratio": finite_ratio,
    }
