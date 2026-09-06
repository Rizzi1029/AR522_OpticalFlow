"""
Canonical FlowNetS (FlowNet Simple) Optical Flow Architecture.

Reference:
    "FlowNet: Learning Optical Flow with Convolutional Networks"
    Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas,
    Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox.
    ICCV 2015 (https://arxiv.org/abs/1504.06852)

Implementation Details & Modern Adaptations:
    1. Pure PyTorch: Built entirely using standard torch.nn modules (Conv2d, ConvTranspose2d,
       LeakyReLU, Sequential). Zero custom C++/CUDA kernels are required.
    2. Architecture: Exact 10-layer convolutional encoder and multi-scale deconvolutional
       refinement decoder, predicting optical flow down to scale 2 (1/4 resolution).
    3. Modern PyTorch Compatibility: Weights are loaded using weights_only=False to support
       the legacy dictionary structure of the FlyingChairs benchmark checkpoint while safely
       extracting the state_dict.
    4. Flow Scaling Convention: The network predicts flow fields scaled down by a factor
       of 20 (div_flow = 20.0), matching the training objective. Outputs must be scaled
       by 20.0 to recover true pixel displacements.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """Standard FlowNet convolutional block: Conv2d + LeakyReLU(0.1)."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def deconv(in_planes: int, out_planes: int) -> nn.Sequential:
    """Standard FlowNet deconvolutional (transposed conv) block: ConvTranspose2d + LeakyReLU(0.1)."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes: int) -> nn.Conv2d:
    """Flow prediction head mapping feature channels to a 2D optical flow vector field (u, v)."""
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def crop_like(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Crop input spatial dimensions to match target spatial dimensions if needed."""
    if input_tensor.shape[2:] == target_tensor.shape[2:]:
        return input_tensor
    return input_tensor[:, :, : target_tensor.shape[2], : target_tensor.shape[3]]


class FlowNetS(nn.Module):
    """
    FlowNetS (FlowNet Simple) architecture.
    
    Accepts two stacked RGB images as a 6-channel input (B, 6, H, W) and outputs
    estimated dense optical flow.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder (Contractive Part)
        self.conv1 = conv(6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(256, 256)
        self.conv4 = conv(256, 512, stride=2)
        self.conv4_1 = conv(512, 512)
        self.conv5 = conv(512, 512, stride=2)
        self.conv5_1 = conv(512, 512)
        self.conv6 = conv(512, 1024, stride=2)
        self.conv6_1 = conv(1024, 1024)

        # Decoder (Expanding / Refinement Part)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        # Flow Prediction Heads at Multiple Scales
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        # Upsampled Flow Layers for Coarse-to-Fine Refinement
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for FlowNetS.
        
        Args:
            x: Tensor of shape (B, 6, H, W) containing stacked image pairs [I1, I2].
            
        Returns:
            In evaluation mode: flow2 of shape (B, 2, H/4, W/4).
            In training mode: tuple of (flow2, flow3, flow4, flow5, flow6).
        """
        # Feature extraction
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # Scale 6 -> 5 refinement
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        # Scale 5 -> 4 refinement
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), dim=1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        # Scale 4 -> 3 refinement
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), dim=1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        # Scale 3 -> 2 refinement
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), dim=1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # Scale 2 final prediction
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), dim=1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2


def load_flownet_s(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> FlowNetS:
    """
    Instantiate FlowNetS and load pretrained weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth or .pth.tar weights checkpoint.
        device: Target device to map the loaded parameters to.

    Returns:
        FlowNetS model in eval mode on the specified device.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Modern PyTorch requires weights_only=False to load legacy checkpoint dictionaries
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict: Dict[str, Any] = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    # Remove any potential DataParallel prefix
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[clean_key] = value

    model = FlowNetS()
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
