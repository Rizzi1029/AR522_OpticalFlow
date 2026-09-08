"""
Day 4: Confidence-Aware Optical Flow Fusion Prototype.

Implements an exploratory prototype matrix of 36 confidence-guided flow fusion
configurations comparing FlowNetS and RAFT across 8 canonical Sintel pairs,
benchmarked against 3 reference baselines (raw FlowNetS, raw RAFT, and 50/50 average).

Methodological & Design Principles:
1. Exploratory Matrix: 36 configurations (3 normalizations x 6 signal combinations x 2 fusion rules).
2. Epistemic Guardrail: Confidence values represent heuristic consistency weights, NOT calibrated uncertainty.
3. Signal Sets: D, FB, Photo, D+FB, D+Photo, D+FB+Photo.
4. Normalizations: raw residual, per-frame percentile/rank, median-normalized residual.
5. Confidence Mapping: c = exp(-r_norm).
6. Multi-Signal Composition: Summing normalized residuals in exponent (multiplicative consistency).
7. Fusion Rules: Soft confidence-weighted blending and Hard confidence-gated selection.
8. Non-Arbitrary Thresholds: Evaluates continuous reliability field c_mean_consistency = 0.5*(c_FNS + c_RAFT).
9. Signal Availability & Mask Decoupling:
   - Standard baseline EPE evaluated on base GT-valid mask (Sintel benchmark protocol).
   - Signal in-bounds conditions treated as availability masks, NOT global exclusion masks.
   - Each fusion configuration constructs its own valid evaluation mask from required signals.
   - Fusion coverage (%) reported per configuration.
   - Raw FNS and RAFT baselines also reported on each configuration's valid mask for matched comparison.
10. Strict Metrics: Mean/median EPE, >3px and >5px outlier rates (evaluation labels only),
    fusion penalty fraction, and confidence vs. oracle sparsification curves (AUSE).
11. Hardware Isolation: Process one pair at a time on GPU; CPU storage of dense evaluation arrays;
    no routine torch.cuda.empty_cache().
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

# Ensure repository root is on Python module search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark.sintel_dataset import SintelDataset
from src.flow.wrappers import build_flow_estimator

# Canonical 8-sample evaluation suite (all frame 1 -> 2)
DEFAULT_SAMPLES = [
    {"scene": "alley_1", "pass_name": "clean", "pair_idx": 0},
    {"scene": "alley_1", "pass_name": "final", "pair_idx": 0},
    {"scene": "shaman_3", "pass_name": "clean", "pair_idx": 0},
    {"scene": "bamboo_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "market_6", "pass_name": "clean", "pair_idx": 0},
    {"scene": "cave_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "ambush_2", "pass_name": "clean", "pair_idx": 0},
    {"scene": "ambush_4", "pass_name": "final", "pair_idx": 0},
]

# Experiment Matrix Definitions
NORMALIZATION_METHODS = ["raw", "percentile", "median_normalized"]

SIGNAL_SETS = {
    "D": ["D"],
    "FB": ["FB"],
    "Photo": ["Photo"],
    "D_FB": ["D", "FB"],
    "D_Photo": ["D", "Photo"],
    "D_FB_Photo": ["D", "FB", "Photo"],
}

FUSION_RULES = ["soft_weighted", "hard_gated"]

BASELINES = ["raw_flownets", "raw_raft", "simple_average"]


# =====================================================================
# 1. SYNTHETIC UNIT VERIFICATIONS
# =====================================================================

def verify_disagreement_math() -> None:
    """Synthetic unit verification for model disagreement computation."""
    f1 = torch.ones((2, 10, 10), dtype=torch.float32)
    f2 = torch.ones((2, 10, 10), dtype=torch.float32)
    diff = f1 - f2
    d_ident = torch.sqrt(diff[0] ** 2 + diff[1] ** 2)
    assert torch.allclose(d_ident, torch.zeros_like(d_ident)), "Identical flow test failed"

    f_offset = f1.clone()
    f_offset[0] += 3.0
    f_offset[1] += 4.0
    diff_off = f_offset - f1
    d_offset = torch.sqrt(diff_off[0] ** 2 + diff_off[1] ** 2)
    assert torch.allclose(d_offset, torch.full_like(d_offset, 5.0)), "Known offset (3, 4) test failed"


def compute_forward_backward_residual(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scalar forward-backward residual and geometric in-bounds mask.

    Args:
        flow_fwd: [2, H, W] forward flow tensor (u, v) in Frame 1.
        flow_bwd: [2, H, W] backward flow tensor (u, v) in Frame 2.

    Returns:
        fb_residual: [H, W] float32 scalar Euclidean residual in pixel units.
        in_bounds_mask: [H, W] boolean mask where forward coordinates remain within Frame 2.
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


def verify_warping_math() -> None:
    """Synthetic unit verification for forward-backward residual math."""
    zero_fwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    zero_bwd = torch.zeros((2, 16, 16), dtype=torch.float32)
    res_zero, in_b_zero = compute_forward_backward_residual(zero_fwd, zero_bwd)
    assert torch.allclose(res_zero, torch.zeros_like(res_zero)), "Zero-flow FB residual failed"
    assert in_b_zero.all(), "Zero-flow in-bounds check failed"

    const_fwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_fwd[0] = 2.0
    const_fwd[1] = 3.0
    const_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    const_bwd[0] = -2.0
    const_bwd[1] = -3.0
    res_const, in_b_const = compute_forward_backward_residual(const_fwd, const_bwd)
    interior = in_b_const[0:16, 0:17]
    assert interior.all(), "Interior in-bounds check failed"
    assert torch.allclose(res_const[0:16, 0:17], torch.zeros_like(res_const[0:16, 0:17]), atol=1e-5), (
        "Constant-flow FB residual failed"
    )

    incon_bwd = torch.zeros((2, 20, 20), dtype=torch.float32)
    res_incon, _ = compute_forward_backward_residual(const_fwd, incon_bwd)
    expected_norm = float(np.sqrt(2.0**2 + 3.0**2))
    assert torch.allclose(res_incon[0:16, 0:17], torch.full_like(res_incon[0:16, 0:17], expected_norm), atol=1e-5), (
        "Inconsistent FB residual failed"
    )


def compute_photometric_residual(
    frame1_float: torch.Tensor,
    frame2_float: torch.Tensor,
    flow_fwd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Warp Frame 2 toward Frame 1 coordinates using forward flow and compute per-pixel RGB L1 residual.

    Args:
        frame1_float: [3, H, W] float32 RGB tensor in [0.0, 1.0].
        frame2_float: [3, H, W] float32 RGB tensor in [0.0, 1.0].
        flow_fwd: [2, H, W] float32 forward flow tensor (u, v) in pixel units from Frame 1 to Frame 2.

    Returns:
        photo_residual: [H, W] float32 per-pixel mean RGB absolute difference in [0.0, 1.0].
        in_bounds_mask: [H, W] boolean mask where forward coordinates remain within Frame 2.
        warped_frame2: [3, H, W] float32 warped Frame 2 in Frame 1 coordinates.
    """
    _, H, W = frame1_float.shape
    device = frame1_float.device

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

    warped_frame2 = F.grid_sample(
        frame2_float.unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]

    photo_residual = torch.mean(torch.abs(frame1_float - warped_frame2), dim=0)

    return photo_residual, in_bounds_mask, warped_frame2


def verify_photometric_math() -> None:
    """Deterministic synthetic sanity checks for image warping and photometric residual math."""
    H, W = 32, 32
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    I1 = (0.01 * x_coords + 0.02 * y_coords).unsqueeze(0).repeat(3, 1, 1)

    res_zero, in_b_zero, _ = compute_photometric_residual(
        I1, I1, torch.zeros((2, H, W), dtype=torch.float32)
    )
    assert torch.allclose(res_zero, torch.zeros_like(res_zero), atol=1e-5), "Zero-motion photometric failed"
    assert in_b_zero.all(), "Zero-motion in-bounds failed"

    u_val, v_val = 2.0, 3.0
    I2 = (0.01 * (x_coords - u_val) + 0.02 * (y_coords - v_val)).unsqueeze(0).repeat(3, 1, 1)

    flow_correct = torch.zeros((2, H, W), dtype=torch.float32)
    flow_correct[0] = u_val
    flow_correct[1] = v_val
    res_correct, in_b_correct, _ = compute_photometric_residual(I1, I2, flow_correct)
    interior_mask = in_b_correct.clone()
    correct_mean = float(res_correct[interior_mask].mean().item())
    assert correct_mean < 1e-4, f"Correct flow photometric mean {correct_mean} >= 1e-4"

    flow_wrong = torch.zeros((2, H, W), dtype=torch.float32)
    res_wrong, _, _ = compute_photometric_residual(I1, I2, flow_wrong)
    wrong_mean = float(res_wrong[interior_mask].mean().item())
    assert wrong_mean > 100 * correct_mean, "Wrong flow contrast check failed"
    assert wrong_mean > 0.01, "Wrong flow absolute check failed"


def verify_normalization_math() -> None:
    """Synthetic unit verification for raw, percentile, and median normalizations."""
    mask = np.ones((5, 5), dtype=bool)
    r = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ], dtype=np.float32)

    # Raw
    r_raw = Normalizer.apply("raw", r, mask)
    assert np.allclose(r_raw, r), "Raw normalization altered values"

    # Median
    r_med = Normalizer.apply("median_normalized", r, mask)
    # median of [1,2,3,4,5] repeated is 3.0
    assert np.isclose(np.median(r_med[mask]), 1.0, atol=1e-5), "Median-normalized median is not 1.0"
    assert np.isclose(r_med[0, 2], 1.0, atol=1e-5), "Median element not equal to 1.0"

    # Percentile
    r_pct = Normalizer.apply("percentile", r, mask)
    assert np.all(r_pct[mask] >= 0.0) and np.all(r_pct[mask] <= 1.0), "Percentile out of bounds [0, 1]"
    assert r_pct[0, 0] < r_pct[0, -1], "Percentile ranking not strictly increasing"


def verify_confidence_math() -> None:
    """Synthetic unit verification for confidence mapping and multi-signal multiplicative property."""
    # Zero residual -> c = 1.0
    r0 = np.array([[0.0]], dtype=np.float32)
    c0 = ConfidenceMapper.map_exponential(r0)
    assert np.isclose(c0[0, 0], 1.0), "Zero residual did not map to 1.0"

    # Median residual (1.0) -> c = exp(-1) approx 0.367879
    r1 = np.array([[1.0]], dtype=np.float32)
    c1 = ConfidenceMapper.map_exponential(r1)
    assert np.isclose(c1[0, 0], np.exp(-1.0), atol=1e-5), "r_norm=1 did not map to exp(-1)"

    # Multi-signal exponent addition property: exp(-(a + b)) == exp(-a) * exp(-b)
    ra = np.array([[0.8]], dtype=np.float32)
    rb = np.array([[1.2]], dtype=np.float32)
    ca = ConfidenceMapper.map_exponential(ra)
    cb = ConfidenceMapper.map_exponential(rb)
    cab = ConfidenceMapper.map_exponential(ra + rb)
    assert np.isclose(cab[0, 0], ca[0, 0] * cb[0, 0], atol=1e-5), "Multi-signal multiplicative property failed"


def verify_fusion_math() -> None:
    """Synthetic unit verification for soft weighted blending and hard gating."""
    f_fns = np.zeros((2, 4, 4), dtype=np.float32)
    f_fns[0, :, :] = 10.0  # FNS says (10, 0)
    f_raft = np.zeros((2, 4, 4), dtype=np.float32)
    f_raft[0, :, :] = 2.0   # RAFT says (2, 0)

    # Case 1: Equal confidences -> 50/50 average
    c_fns_eq = np.ones((4, 4), dtype=np.float32) * 0.5
    c_raft_eq = np.ones((4, 4), dtype=np.float32) * 0.5
    f_soft_eq = FlowFuser.fuse_weighted(f_fns, f_raft, c_fns_eq, c_raft_eq)
    assert np.allclose(f_soft_eq[0], 6.0, atol=1e-4), "Equal confidence did not yield 50/50 blend"

    # Case 2: RAFT infinitely more confident
    c_fns_zero = np.zeros((4, 4), dtype=np.float32)
    c_raft_one = np.ones((4, 4), dtype=np.float32)
    f_soft_raft = FlowFuser.fuse_weighted(f_fns, f_raft, c_fns_zero, c_raft_one)
    assert np.allclose(f_soft_raft[0], 2.0, atol=1e-4), "Dominant RAFT confidence did not select RAFT"

    # Case 3: Hard gating
    f_hard_raft = FlowFuser.fuse_gated(f_fns, f_raft, c_fns_zero, c_raft_one)
    assert np.allclose(f_hard_raft[0], 2.0), "Hard gating failed to pick RAFT when c_raft > c_fns"

    f_hard_fns = FlowFuser.fuse_gated(f_fns, f_raft, c_raft_one, c_fns_zero)
    assert np.allclose(f_hard_fns[0], 10.0), "Hard gating failed to pick FNS when c_fns > c_raft"


def verify_sparsification_math() -> None:
    """Synthetic unit verification for sparsification curve and AUSE calculation."""
    epe_map = np.array([[10.0, 1.0], [5.0, 2.0]], dtype=np.float32)
    conf_perfect = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    curve, ause = FusionEvaluator.compute_sparsification(
        epe_map, conf_perfect, mask, retention_levels=[1.0, 0.5, 0.25]
    )
    assert np.isclose(ause, 0.0, atol=1e-5), f"Perfect ranking did not yield AUSE = 0.0 (got {ause})"

    conf_inverted = np.array([[0.9, 0.1], [0.7, 0.3]], dtype=np.float32)
    _, ause_bad = FusionEvaluator.compute_sparsification(
        epe_map, conf_inverted, mask, retention_levels=[1.0, 0.5, 0.25]
    )
    assert ause_bad > 0.0, "Inverted confidence did not yield positive AUSE"


def verify_masking_and_coverage_math() -> None:
    """Synthetic unit verification for mask generation, signal availability, and coverage calculation."""
    H, W = 10, 10
    mask_gt_base = np.ones((H, W), dtype=bool)
    mask_gt_base[0, :] = False  # 10 invalid GT pixels -> 90 base valid pixels
    n_base = int(np.sum(mask_gt_base))
    assert n_base == 90

    avail_D = np.ones((H, W), dtype=bool)
    avail_FB = np.ones((H, W), dtype=bool)
    avail_FB[:, 0] = False  # 10 out-of-bounds pixels for FB
    avail_Photo = np.ones((H, W), dtype=bool)

    avail_signals = {"D": avail_D, "FB": avail_FB, "Photo": avail_Photo}

    # Configuration D: requires only D
    mask_d_avail = compute_signal_availability_mask("D", avail_signals)
    mask_d = mask_gt_base & mask_d_avail
    assert np.sum(mask_d) == 90
    coverage_d = float(np.sum(mask_d)) / float(n_base) * 100.0
    assert np.isclose(coverage_d, 100.0)

    # Configuration FB: requires FB
    mask_fb_avail = compute_signal_availability_mask("FB", avail_signals)
    mask_fb = mask_gt_base & mask_fb_avail
    # 90 base pixels minus (1-9, 0) = 9 pixels -> 81 valid pixels
    assert np.sum(mask_fb) == 81
    coverage_fb = float(np.sum(mask_fb)) / float(n_base) * 100.0
    assert np.isclose(coverage_fb, 90.0)

    # Configuration D_FB_Photo: requires D, FB, Photo
    mask_all_avail = compute_signal_availability_mask("D_FB_Photo", avail_signals)
    mask_all = mask_gt_base & mask_all_avail
    assert np.sum(mask_all) == 81
    coverage_all = float(np.sum(mask_all)) / float(n_base) * 100.0
    assert np.isclose(coverage_all, 90.0)


def verify_all_math() -> None:
    """Execute all synthetic unit verification tests."""
    verify_disagreement_math()
    verify_warping_math()
    verify_photometric_math()
    verify_normalization_math()
    verify_confidence_math()
    verify_fusion_math()
    verify_sparsification_math()
    verify_masking_and_coverage_math()


# =====================================================================
# 2. NORMALIZATION & CONFIDENCE ENGINES
# =====================================================================

class Normalizer:
    """Implements residual normalization strategies across valid evaluation masks."""

    @staticmethod
    def normalize_raw(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raw physical residual, unscaled."""
        r_norm = np.zeros_like(r, dtype=np.float32)
        r_norm[mask] = np.maximum(r[mask], 0.0).astype(np.float32)
        return r_norm

    @staticmethod
    def normalize_percentile(r: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Per-frame empirical rank in (0.0, 1.0]."""
        r_norm = np.zeros_like(r, dtype=np.float32)
        valid_vals = r[mask]
        if len(valid_vals) == 0:
            return r_norm
        ranks = scipy.stats.rankdata(valid_vals, method="average") / float(len(valid_vals))
        r_norm[mask] = ranks.astype(np.float32)
        return r_norm

    @staticmethod
    def normalize_median(r: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Median-normalized residual r / (median(r) + eps)."""
        r_norm = np.zeros_like(r, dtype=np.float32)
        valid_vals = r[mask]
        if len(valid_vals) == 0:
            return r_norm
        med = float(np.median(valid_vals))
        r_norm[mask] = (valid_vals / max(med, eps)).astype(np.float32)
        return r_norm

    @classmethod
    def apply(cls, method: str, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if method == "raw":
            return cls.normalize_raw(r, mask)
        elif method == "percentile":
            return cls.normalize_percentile(r, mask)
        elif method == "median_normalized":
            return cls.normalize_median(r, mask)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class ConfidenceMapper:
    """Maps normalized residuals into (0, 1] heuristic consistency weights."""

    @staticmethod
    def map_exponential(r_norm: np.ndarray) -> np.ndarray:
        """
        Compute consistency weight c = exp(-r_norm).
        Since r_norm >= 0, c in (0.0, 1.0].
        Clips r_norm to [0.0, 50.0] to guard against numerical underflow.
        """
        r_clipped = np.clip(r_norm, 0.0, 50.0)
        return np.exp(-r_clipped).astype(np.float32)


def compute_signal_availability_mask(
    signal_set_name: str,
    avail_signals: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Construct boolean availability mask for a signal set.
    Only pixels where all required diagnostic signals are valid/in-bounds are True.
    """
    components = SIGNAL_SETS[signal_set_name]
    first_key = components[0]
    avail_mask = np.copy(avail_signals[first_key])
    for comp in components[1:]:
        avail_mask &= avail_signals[comp]
    return avail_mask


def compute_model_confidence(
    signals: Dict[str, np.ndarray],
    signal_set_name: str,
    norm_method: str,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes total normalized residual and resulting heuristic consistency weight for a model.

    Args:
        signals: Dict of candidate residuals for this model (e.g. "D", "FB", "Photo").
        signal_set_name: Key in SIGNAL_SETS.
        norm_method: One of NORMALIZATION_METHODS.
        mask: Boolean evaluation mask (configuration-specific valid fusion mask).

    Returns:
        r_norm_total: [H, W] float32 summed normalized residual.
        confidence: [H, W] float32 consistency weight in (0, 1] on mask, 0 outside.
    """
    components = SIGNAL_SETS[signal_set_name]
    H, W = mask.shape
    r_norm_total = np.zeros((H, W), dtype=np.float32)

    for comp in components:
        if comp not in signals:
            raise KeyError(f"Signal component '{comp}' not provided in signals dictionary")
        r_norm_comp = Normalizer.apply(norm_method, signals[comp], mask)
        r_norm_total += r_norm_comp

    c = np.zeros((H, W), dtype=np.float32)
    c[mask] = ConfidenceMapper.map_exponential(r_norm_total[mask])
    return r_norm_total, c


# =====================================================================
# 3. FLOW FUSION ENGINE
# =====================================================================

class FlowFuser:
    """Fuses FlowNetS and RAFT flow fields using soft confidence blending or hard gating."""

    @staticmethod
    def fuse_weighted(
        flow_fns: np.ndarray,
        flow_raft: np.ndarray,
        c_fns: np.ndarray,
        c_raft: np.ndarray,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Soft confidence-weighted blend:
        f_fused = (c_FNS * f_FNS + c_RAFT * f_RAFT) / (c_FNS + c_RAFT + eps)
        """
        weight_sum = c_fns + c_raft + eps
        w_fns = c_fns / weight_sum
        w_raft = c_raft / weight_sum
        fused = w_fns[None, :, :] * flow_fns + w_raft[None, :, :] * flow_raft
        return fused.astype(np.float32)

    @staticmethod
    def fuse_gated(
        flow_fns: np.ndarray,
        flow_raft: np.ndarray,
        c_fns: np.ndarray,
        c_raft: np.ndarray,
    ) -> np.ndarray:
        """
        Hard confidence-gated model selection:
        f_gated = f_RAFT if c_RAFT >= c_FNS else f_FNS
        """
        raft_wins = (c_raft >= c_fns)[None, :, :]
        fused = np.where(raft_wins, flow_raft, flow_fns)
        return fused.astype(np.float32)

    @classmethod
    def apply(
        cls,
        rule: str,
        flow_fns: np.ndarray,
        flow_raft: np.ndarray,
        c_fns: np.ndarray,
        c_raft: np.ndarray,
    ) -> np.ndarray:
        if rule == "soft_weighted":
            return cls.fuse_weighted(flow_fns, flow_raft, c_fns, c_raft)
        elif rule == "hard_gated":
            return cls.fuse_gated(flow_fns, flow_raft, c_fns, c_raft)
        else:
            raise ValueError(f"Unknown fusion rule: {rule}")


# =====================================================================
# 4. MULTI-LEVEL EVALUATION FRAMEWORK
# =====================================================================

class FusionEvaluator:
    """Computes multi-level optical flow evaluation metrics and sparsification curves."""

    @staticmethod
    def compute_epe_map(flow_est: np.ndarray, flow_gt: np.ndarray) -> np.ndarray:
        """Compute per-pixel End-Point Error map."""
        diff = flow_est - flow_gt
        return np.sqrt(diff[0] ** 2 + diff[1] ** 2).astype(np.float32)

    @staticmethod
    def compute_metrics(
        flow_est: np.ndarray,
        flow_gt: np.ndarray,
        mask: np.ndarray,
        flow_fns: Optional[np.ndarray] = None,
        flow_raft: Optional[np.ndarray] = None,
        flow_avg: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute flow accuracy metrics and fusion penalty/benefit fractions over the given mask.
        Also reports raw FNS and RAFT performance on this exact mask for matched comparison.
        """
        epe_map = FusionEvaluator.compute_epe_map(flow_est, flow_gt)
        valid_epe = epe_map[mask]
        n_valid = len(valid_epe)
        if n_valid == 0:
            return {
                "mean_epe": 0.0,
                "median_epe": 0.0,
                "outlier_3px_pct": 0.0,
                "outlier_5px_pct": 0.0,
                "fusion_penalty_pct": 0.0,
                "fusion_benefit_pct": 0.0,
                "fns_mean_epe_on_mask": 0.0,
                "raft_mean_epe_on_mask": 0.0,
                "avg_mean_epe_on_mask": 0.0,
            }

        mean_epe = float(np.mean(valid_epe))
        median_epe = float(np.median(valid_epe))
        outlier_3px_pct = float(np.mean(valid_epe > 3.0) * 100.0)
        outlier_5px_pct = float(np.mean(valid_epe > 5.0) * 100.0)

        penalty_pct = 0.0
        benefit_pct = 0.0
        fns_mean_epe = 0.0
        raft_mean_epe = 0.0
        avg_mean_epe = 0.0

        if flow_fns is not None and flow_raft is not None:
            epe_fns = FusionEvaluator.compute_epe_map(flow_fns, flow_gt)[mask]
            epe_raft = FusionEvaluator.compute_epe_map(flow_raft, flow_gt)[mask]
            fns_mean_epe = float(np.mean(epe_fns))
            raft_mean_epe = float(np.mean(epe_raft))

            if flow_avg is not None:
                epe_avg = FusionEvaluator.compute_epe_map(flow_avg, flow_gt)[mask]
                avg_mean_epe = float(np.mean(epe_avg))
            else:
                avg_mean_epe = float(np.mean(0.5 * epe_fns + 0.5 * epe_raft))

            min_model_epe = np.minimum(epe_fns, epe_raft)
            # Penalty: fused error strictly greater than min individual error (with margin 1e-4)
            penalty_pct = float(np.mean(valid_epe > (min_model_epe + 1e-4)) * 100.0)
            # Benefit: fused error strictly smaller than min individual error
            benefit_pct = float(np.mean(valid_epe < (min_model_epe - 1e-4)) * 100.0)

        return {
            "mean_epe": mean_epe,
            "median_epe": median_epe,
            "outlier_3px_pct": outlier_3px_pct,
            "outlier_5px_pct": outlier_5px_pct,
            "fusion_penalty_pct": penalty_pct,
            "fusion_benefit_pct": benefit_pct,
            "fns_mean_epe_on_mask": fns_mean_epe,
            "raft_mean_epe_on_mask": raft_mean_epe,
            "avg_mean_epe_on_mask": avg_mean_epe,
        }

    @staticmethod
    def compute_sparsification(
        epe_map: np.ndarray,
        c_diagnostic: np.ndarray,
        mask: np.ndarray,
        retention_levels: Optional[List[float]] = None,
    ) -> Tuple[List[Dict[str, float]], float]:
        """
        Computes empirical confidence sparsification curve vs. theoretical oracle curve.

        Args:
            epe_map: [H, W] float32 per-pixel error map.
            c_diagnostic: [H, W] float32 heuristic consistency score (higher indicates more consistent).
            mask: [H, W] boolean evaluation mask.
            retention_levels: Fractions of top consistent pixels retained [1.0 -> 0.1].

        Returns:
            curve_points: List of dicts with retention, actual_epe, oracle_epe, and gap.
            normalized_ause: Normalized Area Under Sparsification Error curve (mean gap in px).
        """
        if retention_levels is None:
            retention_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        valid_epe = epe_map[mask]
        valid_conf = c_diagnostic[mask]
        n_valid = len(valid_epe)
        if n_valid == 0:
            return [], 0.0

        # Confidence sort: descending by heuristic consistency weight
        sort_conf_idx = np.argsort(-valid_conf)
        sorted_by_conf = valid_epe[sort_conf_idx]

        # Oracle sort: ascending by true ground-truth error (lowest error retained first)
        sorted_by_oracle = np.sort(valid_epe)

        cumsum_conf = np.cumsum(sorted_by_conf)
        cumsum_oracle = np.cumsum(sorted_by_oracle)

        curve_points = []
        actual_epes = []
        oracle_epes = []

        for p in retention_levels:
            k = max(1, int(round(p * n_valid)))
            act_epe = float(cumsum_conf[k - 1] / k)
            orc_epe = float(cumsum_oracle[k - 1] / k)
            curve_points.append({
                "retention": float(p),
                "actual_epe": act_epe,
                "oracle_epe": orc_epe,
                "gap": act_epe - orc_epe,
            })
            actual_epes.append(act_epe)
            oracle_epes.append(orc_epe)

        # Normalized AUSE: trapezoidal integration of (actual - oracle) / retention_span
        p_arr = np.array(retention_levels, dtype=np.float32)
        diff_arr = np.array(actual_epes, dtype=np.float32) - np.array(oracle_epes, dtype=np.float32)
        asc_idx = np.argsort(p_arr)
        span = float(p_arr[asc_idx][-1] - p_arr[asc_idx][0])
        if span > 0.0:
            trap_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            normalized_ause = float(trap_func(diff_arr[asc_idx], p_arr[asc_idx]) / span)
        else:
            normalized_ause = 0.0

        return curve_points, normalized_ause


# =====================================================================
# 5. DIAGNOSTIC PLOTTING
# =====================================================================

def generate_diagnostic_plots(
    macro_summary: Dict[str, Any],
    sample_visual_data: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generates multi-panel diagnostic figure:
    - Panel A: Heatmap of Macro Mean EPE across 36 Matrix configurations (evaluated on valid masks).
    - Panel B: Heatmap of Fusion Penalty Fraction (%) and Coverage (%) across 36 Matrix configurations.
    - Panel C: Sparsification Curves (Actual Consistency Ranking vs. Oracle Retention).
    - Panel D: Visual inspection of a representative sample (flows, error maps, consistency diagnostic).
    """
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

    # Sub-axes
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A & B Grid Setup
    signal_keys = list(SIGNAL_SETS.keys())
    col_labels = []
    for norm in NORMALIZATION_METHODS:
        for rule in FUSION_RULES:
            norm_abbrev = "raw" if norm == "raw" else ("pct" if norm == "percentile" else "med")
            rule_abbrev = "soft" if rule == "soft_weighted" else "gate"
            col_labels.append(f"{norm_abbrev}_{rule_abbrev}")

    matrix_epe = np.zeros((len(signal_keys), len(col_labels)), dtype=np.float32)
    matrix_penalty = np.zeros((len(signal_keys), len(col_labels)), dtype=np.float32)
    matrix_coverage = np.zeros((len(signal_keys), len(col_labels)), dtype=np.float32)

    for i, sig in enumerate(signal_keys):
        col_idx = 0
        for norm in NORMALIZATION_METHODS:
            for rule in FUSION_RULES:
                cfg_name = f"{norm}__{sig}__{rule}"
                if cfg_name in macro_summary["configurations"]:
                    cfg_stats = macro_summary["configurations"][cfg_name]["macro"]
                    matrix_epe[i, col_idx] = cfg_stats["mean_epe"]
                    matrix_penalty[i, col_idx] = cfg_stats["fusion_penalty_pct"]
                    matrix_coverage[i, col_idx] = cfg_stats.get("coverage_pct", 100.0)
                col_idx += 1

    # Panel A: Macro Mean EPE Matrix
    im_a = ax_a.imshow(matrix_epe, cmap="viridis_r", aspect="auto")
    bl_dict = macro_summary.get("baselines_gt_valid", macro_summary.get("baselines", {}))
    bl_fns = bl_dict.get("raw_flownets", {}).get("mean_epe", 0.0)
    bl_avg = bl_dict.get("simple_average", {}).get("mean_epe", 0.0)
    bl_raft = bl_dict.get("raw_raft", {}).get("mean_epe", 0.0)

    ax_a.set_title(
        f"Macro Mean EPE Across 36 Configurations (on Valid Fusion Masks)\n"
        f"Standard GT-Valid Baselines: FlowNetS={bl_fns:.2f} px | "
        f"50/50 Avg={bl_avg:.2f} px | RAFT={bl_raft:.2f} px",
        fontsize=11, fontweight="bold"
    )
    ax_a.set_xticks(range(len(col_labels)))
    ax_a.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax_a.set_yticks(range(len(signal_keys)))
    ax_a.set_yticklabels(signal_keys, fontsize=10)
    for i in range(len(signal_keys)):
        for j in range(len(col_labels)):
            val = matrix_epe[i, j]
            ax_a.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 2.0 else "black", fontsize=8)
    fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04, label="Mean EPE (px)")

    # Panel B: Fusion Penalty Fraction & Coverage Matrix
    im_b = ax_b.imshow(matrix_penalty, cmap="YlOrRd", aspect="auto")
    ax_b.set_title(
        "Fusion Penalty Fraction (% pixels where E_fused > min(E_FNS, E_RAFT))\n"
        "Cell: Penalty% / (Coverage% on base GT)",
        fontsize=11, fontweight="bold"
    )
    ax_b.set_xticks(range(len(col_labels)))
    ax_b.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax_b.set_yticks(range(len(signal_keys)))
    ax_b.set_yticklabels(signal_keys, fontsize=10)
    for i in range(len(signal_keys)):
        for j in range(len(col_labels)):
            val = matrix_penalty[i, j]
            cov = matrix_coverage[i, j]
            ax_b.text(j, i, f"{val:.1f}%\n({cov:.0f}% cov)", ha="center", va="center", color="white" if val > 40.0 else "black", fontsize=7)
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04, label="Penalty Rate (%)")

    # Panel C: Sparsification Curves
    best_cfg_key = macro_summary.get("best_configurations", {}).get("lowest_macro_epe", "")
    if best_cfg_key and best_cfg_key in macro_summary["configurations"]:
        best_curve = macro_summary["configurations"][best_cfg_key]["macro_sparsification_curve"]
        retentions = [pt["retention"] for pt in best_curve]
        actual_epes = [pt["actual_epe"] for pt in best_curve]
        oracle_epes = [pt["oracle_epe"] for pt in best_curve]

        ax_c.plot(retentions, oracle_epes, "k--", linewidth=2.5, label="Oracle Retention (Sorted by True Error)")
        ax_c.plot(retentions, actual_epes, "b-o", linewidth=2.0, label=f"Best Config ({best_cfg_key})")

        fb_cfg_key = "median_normalized__FB__soft_weighted"
        if fb_cfg_key in macro_summary["configurations"]:
            fb_curve = macro_summary["configurations"][fb_cfg_key]["macro_sparsification_curve"]
            ax_c.plot(retentions, [pt["actual_epe"] for pt in fb_curve], "g-s", linewidth=1.5, label="Med-Norm FB (Soft)")

        d_cfg_key = "median_normalized__D__soft_weighted"
        if d_cfg_key in macro_summary["configurations"]:
            d_curve = macro_summary["configurations"][d_cfg_key]["macro_sparsification_curve"]
            ax_c.plot(retentions, [pt["actual_epe"] for pt in d_curve], "m-^", linewidth=1.5, label="Med-Norm D (Soft)")

    ax_c.set_xlabel("Pixel Retention Fraction (Top Consistent Pixels)", fontsize=11)
    ax_c.set_ylabel("Mean EPE (px)", fontsize=11)
    ax_c.set_title("Macro Sparsification Curves: Error vs. Retention\n(Monotonic decrease indicates informative consistency diagnostic)", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=9)
    ax_c.invert_xaxis()  # 1.0 -> 0.1

    # Panel D: Representative Visual Inspection
    if sample_visual_data is not None:
        ax_d.axis("off")
        sub_gs = gs[1, 1].subgridspec(2, 3, hspace=0.25, wspace=0.15)
        ax_d1 = fig.add_subplot(sub_gs[0, 0])
        ax_d2 = fig.add_subplot(sub_gs[0, 1])
        ax_d3 = fig.add_subplot(sub_gs[0, 2])
        ax_d4 = fig.add_subplot(sub_gs[1, 0])
        ax_d5 = fig.add_subplot(sub_gs[1, 1])
        ax_d6 = fig.add_subplot(sub_gs[1, 2])

        ax_d1.imshow(sample_visual_data["frame1_rgb"])
        ax_d1.set_title("Input Frame 1", fontsize=9)
        ax_d1.axis("off")

        gt_mag = np.sqrt(sample_visual_data["flow_gt"][0] ** 2 + sample_visual_data["flow_gt"][1] ** 2)
        ax_d2.imshow(gt_mag, cmap="inferno")
        ax_d2.set_title(f"GT Flow Mag\nMax={gt_mag.max():.1f} px", fontsize=9)
        ax_d2.axis("off")

        ax_d3.imshow(sample_visual_data["epe_fns"], cmap="hot", vmin=0.0, vmax=10.0)
        ax_d3.set_title("FlowNetS EPE\n(0 - 10 px)", fontsize=9)
        ax_d3.axis("off")

        ax_d4.imshow(sample_visual_data["epe_raft"], cmap="hot", vmin=0.0, vmax=5.0)
        ax_d4.set_title("RAFT EPE\n(0 - 5 px)", fontsize=9)
        ax_d4.axis("off")

        ax_d5.imshow(sample_visual_data["epe_fused"], cmap="hot", vmin=0.0, vmax=5.0)
        ax_d5.set_title("Fused Flow EPE\n(0 - 5 px)", fontsize=9)
        ax_d5.axis("off")

        im_c = ax_d6.imshow(sample_visual_data["c_mean_consistency"], cmap="plasma", vmin=0.0, vmax=1.0)
        ax_d6.set_title("Consistency Diagnostic\n(0.0 - 1.0)", fontsize=9)
        ax_d6.axis("off")
        fig.colorbar(im_c, ax=ax_d6, fraction=0.046, pad=0.04)

    plt.suptitle("Day 4 Confidence-Aware Optical Flow Fusion Prototype (36 Matrix Configurations)", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 6. CLI PARSER & MAIN WORKFLOW
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4: Confidence-aware optical flow fusion prototype across 8 Sintel pairs."
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
        help="Directory to save fusion artifacts (default: outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        default=False,
        help="Run mathematical synthetic sanity checks and exit immediately",
    )
    return parser.parse_args()


def main() -> None:
    # 0. Execute deterministic synthetic verification tests
    verify_all_math()

    args = parse_args()
    if args.verify_only:
        print("All synthetic mathematical verifications passed successfully.")
        print(f"- Disagreement math verified.")
        print(f"- Forward-backward residual and warping math verified.")
        print(f"- Photometric residual and image warping math verified.")
        print(f"- Normalization routines (raw, percentile, median) verified.")
        print(f"- Exponential confidence mapping and multiplicative composition verified.")
        print(f"- Soft confidence-weighted fusion and hard model gating verified.")
        print(f"- Sparsification curve and normalized AUSE calculation verified.")
        print(f"- Mask generation, signal availability, and coverage calculation verified.")
        return

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      DAY 4 CONFIDENCE-AWARE FLOW FUSION PROTOTYPE (36 MATRIX)    ")
    print("==================================================================")
    print(f"Device:           {device}")
    print(f"Samples:          {len(DEFAULT_SAMPLES)} pairs (frame 1 -> 2)")
    print(f"Normalizations:   {NORMALIZATION_METHODS}")
    print(f"Signal Sets:      {list(SIGNAL_SETS.keys())}")
    print(f"Fusion Rules:     {FUSION_RULES}")
    print(f"Total Matrix:     {len(NORMALIZATION_METHODS) * len(SIGNAL_SETS) * len(FUSION_RULES)} configurations + 3 baselines")
    print(f"Output Directory: {output_dir}")
    print("==================================================================\n")

    # 1. Instantiate models once
    print("Loading FlowNetS adapter...")
    flownet_model = build_flow_estimator(
        model_name="flownets",
        checkpoint_path=args.flownet_checkpoint,
        device=device,
    )

    print("Loading RAFT adapter...")
    raft_model = build_flow_estimator(
        model_name="raft",
        device=device,
    )
    print("Models loaded successfully.\n")

    # Data structures to accumulate per-sample results across the 8 pairs
    matrix_results: Dict[str, List[Dict[str, Any]]] = {}
    for norm in NORMALIZATION_METHODS:
        for sig in SIGNAL_SETS:
            for rule in FUSION_RULES:
                cfg_name = f"{norm}__{sig}__{rule}"
                matrix_results[cfg_name] = []

    # Standard baselines evaluated on base GT-valid mask
    baseline_results: Dict[str, List[Dict[str, Any]]] = {
        "raw_flownets": [],
        "raw_raft": [],
        "simple_average": [],
    }

    sample_metadata_list: List[Dict[str, Any]] = []
    visual_sample_data: Optional[Dict[str, Any]] = None

    # 2. Sequential per-pair processing loop (GPU memory isolated)
    for sample_idx, spec in enumerate(DEFAULT_SAMPLES):
        scene = spec["scene"]
        pass_name = spec["pass_name"]
        pair_idx = spec["pair_idx"]
        sample_label = f"{scene} ({pass_name})"

        print(f"[{sample_idx + 1}/{len(DEFAULT_SAMPLES)}] Processing pair: {sample_label} ... ", end="", flush=True)

        dataset = SintelDataset(
            root=args.data_root,
            split="training",
            pass_name=pass_name,
            scenes=[scene],
        )
        frame1, frame2, flow_gt, valid_mask, meta = dataset[pair_idx]

        with torch.inference_mode():
            # FlowNetS forward + backward inference
            flow_fwd_fns_t = flownet_model(frame1, frame2)[0]
            flow_bwd_fns_t = flownet_model(frame2, frame1)[0]

            # RAFT forward + backward inference
            flow_fwd_raft_t = raft_model(frame1, frame2)[0]
            flow_bwd_raft_t = raft_model(frame2, frame1)[0]

            # Signal 1: Model Disagreement D(x, y) = ||f_FNS - f_RAFT||_2
            diff_models = flow_fwd_fns_t - flow_fwd_raft_t
            disag_t = torch.sqrt(diff_models[0] ** 2 + diff_models[1] ** 2)
            del diff_models

            # Signal 2: Forward-Backward Consistency R_FB
            res_fb_fns_t, in_b_fns_t = compute_forward_backward_residual(flow_fwd_fns_t, flow_bwd_fns_t)
            res_fb_raft_t, in_b_raft_t = compute_forward_backward_residual(flow_fwd_raft_t, flow_bwd_raft_t)

            # Signal 3: Photometric Residual R_photo
            frame1_float = frame1.to(device=device, dtype=torch.float32) / 255.0
            frame2_float = frame2.to(device=device, dtype=torch.float32) / 255.0
            res_photo_fns_t, in_b_photo_fns_t, _ = compute_photometric_residual(frame1_float, frame2_float, flow_fwd_fns_t)
            res_photo_raft_t, in_b_photo_raft_t, _ = compute_photometric_residual(frame1_float, frame2_float, flow_fwd_raft_t)

            # Ground-Truth Flow
            flow_gt_dev = flow_gt.to(device)

        # Transfer arrays to CPU NumPy immediately
        flow_fns_np = flow_fwd_fns_t.cpu().numpy()
        flow_raft_np = flow_fwd_raft_t.cpu().numpy()
        flow_gt_np = flow_gt_dev.cpu().numpy()

        disag_np = disag_t.cpu().numpy()
        res_fb_fns_np = res_fb_fns_t.cpu().numpy()
        res_fb_raft_np = res_fb_raft_t.cpu().numpy()
        res_photo_fns_np = res_photo_fns_t.cpu().numpy()
        res_photo_raft_np = res_photo_raft_t.cpu().numpy()

        valid_gt_np = valid_mask.cpu().numpy()
        in_b_fns_np = in_b_fns_t.cpu().numpy()
        in_b_raft_np = in_b_raft_t.cpu().numpy()
        in_b_ph_fns_np = in_b_photo_fns_t.cpu().numpy()
        in_b_ph_raft_np = in_b_photo_raft_t.cpu().numpy()

        frame1_rgb = frame1.permute(1, 2, 0).cpu().numpy()

        # Strict GPU hygiene: delete all GPU tensors explicitly
        del flow_fwd_fns_t, flow_bwd_fns_t, flow_fwd_raft_t, flow_bwd_raft_t, flow_gt_dev
        del disag_t, res_fb_fns_t, res_fb_raft_t, in_b_fns_t, in_b_raft_t
        del res_photo_fns_t, res_photo_raft_t, in_b_photo_fns_t, in_b_photo_raft_t
        del frame1_float, frame2_float

        # 1. Base GT-valid evaluation mask (standard Sintel protocol: valid GT and finite flows)
        mask_gt_base = (
            valid_gt_np &
            np.isfinite(flow_fns_np[0]) & np.isfinite(flow_fns_np[1]) &
            np.isfinite(flow_raft_np[0]) & np.isfinite(flow_raft_np[1]) &
            np.isfinite(flow_gt_np[0]) & np.isfinite(flow_gt_np[1])
        )
        n_base = int(np.sum(mask_gt_base))

        sample_metadata_list.append({
            "sample_index": sample_idx,
            "scene": scene,
            "pass_name": pass_name,
            "pair_idx": pair_idx,
            "base_gt_pixels": n_base,
        })

        # 2. Signal availability masks (where diagnostic signals can be evaluated)
        avail_signals = {
            "D": np.isfinite(disag_np),
            "FB": in_b_fns_np & in_b_raft_np & np.isfinite(res_fb_fns_np) & np.isfinite(res_fb_raft_np),
            "Photo": in_b_ph_fns_np & in_b_ph_raft_np & np.isfinite(res_photo_fns_np) & np.isfinite(res_photo_raft_np),
        }

        # Prepackage candidate signals per model
        signals_fns = {
            "D": disag_np,
            "FB": res_fb_fns_np,
            "Photo": res_photo_fns_np,
        }
        signals_raft = {
            "D": disag_np,
            "FB": res_fb_raft_np,
            "Photo": res_photo_raft_np,
        }

        # 2A. Evaluate Standard Baselines on mask_gt_base (standard benchmark comparison)
        flow_avg_np = 0.5 * flow_fns_np + 0.5 * flow_raft_np

        bl_fns_metrics = FusionEvaluator.compute_metrics(
            flow_fns_np, flow_gt_np, mask_gt_base, flow_fns_np, flow_raft_np, flow_avg_np
        )
        baseline_results["raw_flownets"].append(bl_fns_metrics)

        bl_raft_metrics = FusionEvaluator.compute_metrics(
            flow_raft_np, flow_gt_np, mask_gt_base, flow_fns_np, flow_raft_np, flow_avg_np
        )
        baseline_results["raw_raft"].append(bl_raft_metrics)

        bl_avg_metrics = FusionEvaluator.compute_metrics(
            flow_avg_np, flow_gt_np, mask_gt_base, flow_fns_np, flow_raft_np, flow_avg_np
        )
        baseline_results["simple_average"].append(bl_avg_metrics)

        # 2B. Evaluate the 36 Matrix Configurations on their own valid fusion masks
        for sig in SIGNAL_SETS:
            # Construct configuration-specific valid fusion mask from required signals
            sig_avail = compute_signal_availability_mask(sig, avail_signals)
            mask_cfg = mask_gt_base & sig_avail
            n_cfg = int(np.sum(mask_cfg))
            coverage_pct = float(n_cfg / max(n_base, 1) * 100.0)

            for norm in NORMALIZATION_METHODS:
                # Compute consistency weights for FlowNetS and RAFT on mask_cfg
                _, c_fns = compute_model_confidence(signals_fns, sig, norm, mask_cfg)
                _, c_raft = compute_model_confidence(signals_raft, sig, norm, mask_cfg)

                # Diagnostic: mean heuristic consistency weight (not calibrated joint confidence)
                c_mean_consistency = np.zeros_like(mask_cfg, dtype=np.float32)
                c_mean_consistency[mask_cfg] = 0.5 * (c_fns[mask_cfg] + c_raft[mask_cfg])

                for rule in FUSION_RULES:
                    cfg_name = f"{norm}__{sig}__{rule}"
                    flow_fused = FlowFuser.apply(rule, flow_fns_np, flow_raft_np, c_fns, c_raft)

                    # Multi-level metrics on mask_cfg (including matched baselines on the same mask)
                    cfg_metrics = FusionEvaluator.compute_metrics(
                        flow_fused, flow_gt_np, mask_cfg, flow_fns_np, flow_raft_np, flow_avg_np
                    )

                    # Sparsification curve on mask_cfg
                    epe_fused_map = FusionEvaluator.compute_epe_map(flow_fused, flow_gt_np)
                    curve_points, ause = FusionEvaluator.compute_sparsification(
                        epe_fused_map, c_mean_consistency, mask_cfg
                    )

                    pair_result = {
                        **cfg_metrics,
                        "coverage_pct": coverage_pct,
                        "eval_pixels": n_cfg,
                        "base_gt_pixels": n_base,
                        "ause": ause,
                        "mean_consistency_score": float(np.mean(c_mean_consistency[mask_cfg])) if n_cfg > 0 else 0.0,
                        "sparsification_curve": curve_points,
                    }
                    matrix_results[cfg_name].append(pair_result)

                    # Cache representative visual inspection data (from sample 4: market_6, or sample 0)
                    if sample_idx == 4 and visual_sample_data is None and cfg_name == "median_normalized__FB__soft_weighted":
                        visual_sample_data = {
                            "frame1_rgb": frame1_rgb,
                            "flow_gt": flow_gt_np,
                            "epe_fns": FusionEvaluator.compute_epe_map(flow_fns_np, flow_gt_np),
                            "epe_raft": FusionEvaluator.compute_epe_map(flow_raft_np, flow_gt_np),
                            "epe_fused": epe_fused_map,
                            "c_mean_consistency": c_mean_consistency,
                            "mask_cfg": mask_cfg,
                        }

        # Fallback for visual data if sample 4 not reached
        if visual_sample_data is None and sample_idx == 0:
            visual_sample_data = {
                "frame1_rgb": frame1_rgb,
                "flow_gt": flow_gt_np,
                "epe_fns": FusionEvaluator.compute_epe_map(flow_fns_np, flow_gt_np),
                "epe_raft": FusionEvaluator.compute_epe_map(flow_raft_np, flow_gt_np),
                "epe_fused": FusionEvaluator.compute_epe_map(flow_avg_np, flow_gt_np),
                "c_mean_consistency": np.ones_like(mask_gt_base, dtype=np.float32) * 0.5,
                "mask_cfg": mask_gt_base,
            }

        print(f"done (N_base={n_base:,} px, RAFT EPE={bl_raft_metrics['mean_epe']:.3f} px, FNS EPE={bl_fns_metrics['mean_epe']:.3f} px)")

    print("\nAll 8 canonical pairs evaluated successfully.")

    # 3. Macro Aggregation Across All 8 Pairs
    print("Aggregating macro statistics across all 8 samples...")

    def aggregate_macro(sample_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Macro arithmetic mean across 8 samples."""
        keys = [
            "mean_epe", "median_epe", "outlier_3px_pct", "outlier_5px_pct",
            "fusion_penalty_pct", "fusion_benefit_pct"
        ]
        optional_keys = [
            "coverage_pct", "fns_mean_epe_on_mask", "raft_mean_epe_on_mask", "avg_mean_epe_on_mask",
            "ause", "mean_consistency_score"
        ]
        for k in optional_keys:
            if k in sample_list[0]:
                keys.append(k)

        macro = {}
        for k in keys:
            macro[k] = float(np.mean([item[k] for item in sample_list]))
        return macro

    macro_baselines = {}
    for bl_name, bl_list in baseline_results.items():
        macro_baselines[bl_name] = aggregate_macro(bl_list)

    macro_configurations: Dict[str, Any] = {}
    for cfg_name, cfg_list in matrix_results.items():
        norm_method, sig_name, rule_name = cfg_name.split("__")
        macro_stats = aggregate_macro(cfg_list)

        # Average sparsification curves across 8 samples
        retention_levels = [pt["retention"] for pt in cfg_list[0]["sparsification_curve"]]
        macro_curve = []
        for p_idx, p_val in enumerate(retention_levels):
            mean_act = float(np.mean([s["sparsification_curve"][p_idx]["actual_epe"] for s in cfg_list]))
            mean_orc = float(np.mean([s["sparsification_curve"][p_idx]["oracle_epe"] for s in cfg_list]))
            macro_curve.append({
                "retention": p_val,
                "actual_epe": mean_act,
                "oracle_epe": mean_orc,
                "gap": mean_act - mean_orc,
            })

        macro_configurations[cfg_name] = {
            "normalization": norm_method,
            "signal_set": sig_name,
            "fusion_rule": rule_name,
            "macro": macro_stats,
            "per_sample": cfg_list,
            "macro_sparsification_curve": macro_curve,
        }

    # Identify top-performing configurations under different criteria
    all_cfg_names = list(macro_configurations.keys())
    lowest_epe_cfg = min(all_cfg_names, key=lambda name: macro_configurations[name]["macro"]["mean_epe"])
    lowest_penalty_cfg = min(all_cfg_names, key=lambda name: macro_configurations[name]["macro"]["fusion_penalty_pct"])
    lowest_ause_cfg = min(all_cfg_names, key=lambda name: macro_configurations[name]["macro"]["ause"])
    lowest_outlier_cfg = min(all_cfg_names, key=lambda name: macro_configurations[name]["macro"]["outlier_3px_pct"])
    highest_coverage_cfg = max(all_cfg_names, key=lambda name: macro_configurations[name]["macro"]["coverage_pct"])

    best_configurations = {
        "lowest_macro_epe": lowest_epe_cfg,
        "lowest_fusion_penalty": lowest_penalty_cfg,
        "lowest_ause": lowest_ause_cfg,
        "lowest_outlier_3px": lowest_outlier_cfg,
        "highest_coverage": highest_coverage_cfg,
    }

    final_payload = {
        "title": "Day 4 Confidence-Aware Optical Flow Fusion Prototype Analysis",
        "description": "Exploratory evaluation of 36 confidence-fusion configurations against 3 baselines across 8 canonical Sintel pairs",
        "samples": sample_metadata_list,
        "baselines": macro_baselines,
        "baselines_gt_valid": macro_baselines,
        "configurations": macro_configurations,
        "best_configurations": best_configurations,
    }

    # 4. Save JSON and PNG Artifacts
    json_path = output_dir / "day4_confidence_fusion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved structured JSON artifact to: {json_path}")

    png_path = output_dir / "day4_confidence_fusion.png"
    generate_diagnostic_plots(final_payload, visual_sample_data, png_path)
    print(f"Saved diagnostic visualization to:   {png_path}")

    print("\n==================================================================")
    print("                     EXECUTION COMPLETE                           ")
    print("==================================================================")


if __name__ == "__main__":
    main()
