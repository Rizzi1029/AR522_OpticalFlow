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
