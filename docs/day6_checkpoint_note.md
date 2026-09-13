# Day 6 Research Checkpoint: Reliability-Aware Optical Flow Cascading

## 1. System Architecture: FlowNetS + Forward-Only `mag_median` Gate
- **Pipeline Structure**: Causal, two-stage cascade. FlowNetS forward inference runs on all frames and acts as a diagnostic probe.
- **Diagnostic Signal**: `mag_median` = $\text{median}(\|f_{\text{fwd}}\|_2)$ computed from predicted FlowNetS forward displacement.
- **Overhead Reduction**: Forward-only gate latency is **13.51 ms (74.0 FPS)** on the RTX 5050 Laptop GPU, eliminating the 25.36 ms bidirectional warping cost of Forward-Backward (FB) consistency checks (46.7% gate compute reduction).
- **Zero Cross-Model Leakage**: RAFT is only invoked if the gate triggers; RAFT features and FlowNetS-RAFT disagreement are never used during gating.

## 2. Validation Protocol: 23-Scene Leave-One-Scene-Out (LOSO) Cross-Validation
- **Dataset Scope**: Evaluated across all **1,041 frame pairs** and **23 scenes** of the MPI Sintel training benchmark (`clean` pass, 464,424,715 valid pixels).
- **Leakage-Free Calibration**: For each held-out scene $S_i$, the routing threshold $\tau$ is calibrated strictly on the other 22 scenes as the $(1 - \rho)$ quantile of training-fold `mag_median` values for target budget $\rho$.
- **Zero Test EPE Leakage**: No ground-truth flow or held-out EPE is used during threshold selection.

## 3. Full 1041-Pair Pareto Sweep & Marginal Gain Analysis
- **Sweep Resolution**: 10,001 evaluation points from 0.00% to 100.00% target invocation at 0.01% intervals under 23-scene LOSO-CV.
- **Monotonicity**: The resulting Pareto frontier is 100% monotonic across all 10,001 points.
- **Marginal Gain Analysis**: Smoothed numerical first difference $-d(\text{EPE}) / d(\text{Invocation \%})$ identifies three operational regimes:
  1. *High-Gain Regime (0% - 23%)*: Steep initial return up to 0.24-0.30 px EPE reduction per +1% invocation.
  2. *Transition Regime (23% - 44%)*: Return moderates from 0.05 to 0.01 px / %.
  3. *Diminishing Returns Plateau (>44%)*: Marginal gain flattens below 0.010 px / % (down to 0.005 px / % at 50% invocation).

## 4. Primary Selected Operating Point: ~43% Sub-1px Crossover Knee
- **Target Budget**: 43.43% | **Actual OOF Invocation**: **42.65%** (444 / 1041 pairs).
- **Routed Macro EPE**: **0.9958 px** (Micro EPE: 0.9949 px).
- **Error Reduction**: **80.65%** reduction vs. FlowNetS baseline (5.15 px).
- **End-to-End Latency**: **67.64 ms** (**14.78 FPS**, **1.88x speedup** vs. Always-RAFT).
- **Compute Saving**: 57.35% of frames bypass RAFT entirely.
- **Significance**: Lowest invocation operating point that breaks below the 1.0-pixel error barrier across all 23 unseen scenes.

## 5. Alternative Operating Points
- **~23% Ultra-Fast Kneedle Knee** (22.01% target / **22.96% actual**, 239 pairs):
  - Macro EPE: **1.5536 px** (69.81% error reduction)
  - End-to-End Latency: **42.65 ms** (**23.45 FPS**, **2.98x speedup** vs. RAFT)
  - Mathematical maximum-distance chord knee (Kneedle algorithm).
- **50% Canonical Balanced Point** (50.00% target / **49.86% actual**, 519 pairs):
  - Macro EPE: **0.9624 px** (81.30% error reduction)
  - End-to-End Latency: **76.78 ms** (**13.02 FPS**, **1.65x speedup** vs. RAFT)
  - Halves RAFT invocation while providing solid sub-pixel tracking.
- **~87.5% High-Fidelity Point** (87.50% target / **86.65% actual**, 902 pairs):
  - Macro EPE: **0.7319 px** (85.78% error reduction)
  - End-to-End Latency: **123.48 ms** (**8.10 FPS**, **1.03x speedup** vs. RAFT)
  - Within **+0.1007 px** of Always-RAFT (0.6311 px) while skipping 139 frames.

## 6. Measured Component Latencies (NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM)
- FlowNetS Forward Pass: **12.27 ms** (81.5 FPS)
- FlowNetS Forward Gate (`mag_median`): **13.51 ms** (74.0 FPS)
- FlowNetS Bidirectional FB Gate: **25.36 ms** (39.4 FPS)
- Always-RAFT (12 recurrent updates): **126.92 ms** (7.88 FPS)

## 7. Known Limitations and Caveats
1. **Global Proxy for Local Failures**: Frame-level median magnitude captures large global camera/object displacements reliably, but cannot detect localized occlusions or fine motion boundaries when overall frame displacement is small.
2. **False Alarms on Smooth Translating Motion**: Camera pans with smooth rigid motion (e.g. `sleeping_1`) produce median flow of ~4 px, triggering RAFT even though FlowNetS already achieves low EPE (0.30 px).
3. **Frame-Level Granularity**: Whole-frame escalation means entire images are routed to RAFT. Mixed-motion scenes could benefit from future patch-level or spatial hybrid cascading.
4. **Dataset Specificity**: Thresholds are calibrated on MPI Sintel synthetic dynamics; real-world deployment on autonomous systems may require re-calibration against domain-specific velocity profiles.
