"""
Common Model Adapters for FlowNetS and RAFT Optical Flow Architectures.

Provides unified [B, 2, H, W] float32 output contract while preserving model-specific
preprocessing, normalization, spatial padding, and post-processing conventions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.flow.flownet_s import FlowNetS, load_flownet_s


class BaseOpticalFlowEstimator(nn.Module, ABC):
    """
    Abstract base interface establishing the canonical optical flow contract.

    Accepts RGB image pairs of arbitrary spatial dimensions and outputs
    dense 2D optical flow fields (u, v) matching original unpadded resolution.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

    def _normalize_inputs(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Validate and standardize input frame dimensions.

        Ensures inputs are 4D [B, 3, H, W] tensors and returns original (H, W).
        """
        if frame1.shape != frame2.shape:
            raise ValueError(
                f"Shape mismatch: frame1={tuple(frame1.shape)} vs frame2={tuple(frame2.shape)}"
            )

        if frame1.ndim == 3:
            # [3, H, W] -> [1, 3, H, W]
            if frame1.shape[0] != 3:
                raise ValueError(f"Expected 3 channels in dim 0 for 3D frame, got {frame1.shape[0]}")
            f1 = frame1.unsqueeze(0)
            f2 = frame2.unsqueeze(0)
        elif frame1.ndim == 4:
            # [B, 3, H, W]
            if frame1.shape[1] != 3:
                raise ValueError(f"Expected 3 channels in dim 1 for 4D frame, got {frame1.shape[1]}")
            f1 = frame1
            f2 = frame2
        else:
            raise ValueError(
                f"Expected 3D [3, H, W] or 4D [B, 3, H, W] frames, got {tuple(frame1.shape)}"
            )

        orig_h, orig_w = f1.shape[2], f1.shape[3]
        return f1, f2, (orig_h, orig_w)

    @abstractmethod
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow from frame1 to frame2.

        Args:
            frame1: RGB image tensor of shape [3, H, W] or [B, 3, H, W] in uint8 [0, 255].
            frame2: RGB image tensor of shape [3, H, W] or [B, 3, H, W] in uint8 [0, 255].

        Returns:
            torch.Tensor: Optical flow field of shape [B, 2, H, W] in float32,
            matching original unpadded spatial dimensions (H, W).
        """
        pass


class FlowNetSAdapter(BaseOpticalFlowEstimator):
    """
    Adapter for canonical FlowNetS architecture.

    Encapsulates:
    1. FlyingChairs RGB mean subtraction [0.411, 0.432, 0.45].
    2. Spatial replicate-padding to multiples of 64.
    3. Bilinear upsampling of scale-2 flow (1/4 resolution) to full resolution.
    4. Unpadding / cropping back to original (H, W).
    5. Rescaling by div_flow = 20.0 to recover true pixel displacements.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.model = load_flownet_s(checkpoint_path=checkpoint_path, device=self.device)
        self.model.eval()

        # FlyingChairs RGB dataset mean
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.411, 0.432, 0.45], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.div_flow = 20.0
        self.to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        f1, f2, (orig_h, orig_w) = self._normalize_inputs(frame1, frame2)

        # Preprocessing: convert uint8 [0, 255] to float32 [0.0, 1.0] on target device
        f1_float = f1.to(device=self.device, dtype=torch.float32) / 255.0
        f2_float = f2.to(device=self.device, dtype=torch.float32) / 255.0

        # Mean subtraction
        f1_norm = f1_float - self.rgb_mean
        f2_norm = f2_float - self.rgb_mean

        # Concatenate into 6-channel input [B, 6, H, W]
        pair = torch.cat([f1_norm, f2_norm], dim=1)

        # Pad spatial dimensions to multiples of 64
        pad_h = (64 - (orig_h % 64)) % 64
        pad_w = (64 - (orig_w % 64)) % 64
        if pad_h > 0 or pad_w > 0:
            pair = F.pad(pair, (0, pad_w, 0, pad_h), mode="replicate")

        # Forward pass through FlowNetS (eval returns flow2 of shape [B, 2, H_pad/4, W_pad/4])
        raw_flow2 = self.model(pair)

        # Bilinear upsampling to full padded resolution
        upsampled = F.interpolate(
            raw_flow2,
            size=(orig_h + pad_h, orig_w + pad_w),
            mode="bilinear",
            align_corners=False,
        )

        # Unpad / crop back to original resolution
        if pad_h > 0 or pad_w > 0:
            upsampled = upsampled[:, :, :orig_h, :orig_w]

        # Scale by div_flow = 20.0 to recover true pixel displacements
        flow_final = upsampled * self.div_flow

        return flow_final.contiguous()


class RAFTAdapter(BaseOpticalFlowEstimator):
    """
    Adapter for torchvision RAFT Large architecture.

    Encapsulates:
    1. Official Raft_Large_Weights normalization (maps [0, 255] uint8 -> [-1.0, 1.0] float32).
    2. Spatial replicate-padding to multiples of 8.
    3. Multi-iteration recurrent flow updates (default 12 iterations).
    4. Unpadding / cropping of final flow prediction back to original (H, W).
    """

    def __init__(
        self,
        weights: Optional[Raft_Large_Weights] = None,
        num_flow_updates: int = 12,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(device=device)
        if weights is None:
            weights = Raft_Large_Weights.C_T_SKHT_V2

        self.weights = weights
        self.num_flow_updates = num_flow_updates
        self.transforms = weights.transforms()

        self.model = raft_large(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        f1, f2, (orig_h, orig_w) = self._normalize_inputs(frame1, frame2)

        # Preprocessing: torchvision transforms map [0, 255] -> [-1.0, 1.0]
        # transforms accepts [B, 3, H, W] uint8 or float tensors
        t1, t2 = self.transforms(f1, f2)
        t1 = t1.to(device=self.device)
        t2 = t2.to(device=self.device)

        # Pad spatial dimensions to multiples of 8
        pad_h = (8 - (orig_h % 8)) % 8
        pad_w = (8 - (orig_w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            t1 = F.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
            t2 = F.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")

        # Forward pass (eval returns list of iterative predictions [flow_1, ..., flow_N])
        flow_predictions = self.model(t1, t2, num_flow_updates=self.num_flow_updates)

        # Extract final iteration prediction [B, 2, H_pad, W_pad]
        final_flow = flow_predictions[-1]

        # Unpad / crop back to original resolution
        if pad_h > 0 or pad_w > 0:
            final_flow = final_flow[:, :, :orig_h, :orig_w]

        return final_flow.contiguous()


def build_flow_estimator(
    model_name: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> BaseOpticalFlowEstimator:
    """
    Factory helper to instantiate optical flow model adapters by name.

    Args:
        model_name: Name of the model ('flownets' or 'raft').
        checkpoint_path: Path to FlowNetS checkpoint weights file (optional for RAFT).
        device: Computation device ('cpu', 'cuda', or torch.device).
        **kwargs: Additional model-specific keyword arguments.

    Returns:
        BaseOpticalFlowEstimator instance conforming to [B, 2, H, W] flow contract.
    """
    name = model_name.strip().lower()
    if name in ("flownets", "flownet_s", "flownet"):
        if checkpoint_path is None:
            # Default to standard repository checkpoint location
            project_root = Path(__file__).resolve().parent.parent.parent
            checkpoint_path = project_root / "checkpoints" / "flownets_EPE1.951.pth.tar"
        return FlowNetSAdapter(checkpoint_path=checkpoint_path, device=device)

    elif name == "raft":
        return RAFTAdapter(device=device, **kwargs)

    else:
        raise ValueError(
            f"Unsupported optical flow model: '{model_name}'. "
            "Supported options are 'flownets' and 'raft'."
        )
