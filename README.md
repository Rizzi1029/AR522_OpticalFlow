# AR522 Optical Flow Project

## Optical Flow-Guided Obstacle Motion Estimation

Comparative study of CNN-based and attention/recurrent optical flow estimation, followed by dynamic obstacle costmap generation and path avoidance.

### Project Pipeline

Frames → Optical Flow → Motion Estimation → Dynamic Costmap → Path Planning

### Models

- CNN-based optical flow baseline
- RAFT

### Dataset

MPI Sintel

### Hardware

- NVIDIA GeForce RTX 5050 Laptop GPU
- 8 GB VRAM

### Environment

- Ubuntu 26.04 LTS via WSL2
- Python 3.14
- PyTorch + CUDA

## Research Checkpoints

- **Day 6: Reliability-Aware Optical Flow Cascading**: See [docs/day6_checkpoint_note.md](docs/day6_checkpoint_note.md) for the complete benchmark validation of the forward-only `mag_median` cascade across all 1,041 Sintel training pairs (23-scene LOSO-CV, 10,001-point Pareto sweep, and ~43% operating point yielding 0.996 px EPE at 14.8 FPS).

