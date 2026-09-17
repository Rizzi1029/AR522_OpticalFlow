"""
Rigid Camera-Induced Optical Flow Computation.

Theoretical Foundation & Coordinate System Conventions:
-------------------------------------------------------
1. Camera Coordinate System (Standard Computer Vision / OpenCV Frame):
   - Origin: Optical center of the camera.
   - +X axis: Points horizontally RIGHT (column direction).
   - +Y axis: Points vertically DOWN (row direction).
   - +Z axis: Points FORWARD along optical axis into the scene (depth Z > 0).

2. Extrinsic Transformation Representation (World -> Camera):
   In the MPI Sintel benchmark and standard multiview geometry, the 3x4 extrinsic matrix
   E = [R | t] represents the Euclidean transformation from world coordinates to camera coordinates:
     P_cam = R @ P_world + t

   In 4x4 homogeneous form:
         [ R   t ]
     T = [ 0   1 ]

   The inverse transformation (Camera -> World) is:
            [ R^T   -R^T @ t ]
     T^{-1} = [  0          1  ]

3. Relative Inter-Frame Transformation (Frame 1 -> Frame 2):
   For two consecutive frames with world-to-camera matrices E1 = [R1 | t1] and E2 = [R2 | t2]:
     P_cam1 = R1 @ P_world + t1  =>  P_world = R1^T @ (P_cam1 - t1)
     P_cam2 = R2 @ P_world + t2 = R2 @ R1^T @ (P_cam1 - t1) + t2

   Therefore:
     T_{2<-1} = T2 @ T1^{-1}
     R_rel = R2 @ R1^T
     t_rel = t2 - R2 @ R1^T @ t1 = t2 - R_rel @ t1
     P_cam2 = R_rel @ P_cam1 + t_rel

4. Pinhole Projection & Rigid Optical Flow:
   Given 2D pixel coordinates p1 = [u1, v1]^T and positive depth Z1(u1, v1) in Frame 1:
     - 3D Backprojection: P_cam1 = Z1 * K1^{-1} * [u1, v1, 1]^T
     - Rigid Motion:      P_cam2 = R_rel @ P_cam1 + t_rel = [X2, Y2, Z2]^T
     - 2D Reprojection:   p2 = [u2, v2]^T = [fx2 * (X2 / Z2) + cx2, fy2 * (Y2 / Z2) + cy2]^T
     - Rigid Flow:        rigid_flow = p2 - p1 = [u2 - u1, v2 - v1]^T

5. Validity Mask Definition:
   A flow vector is defined as valid if and only if all of the following hold:
     (a) Depth in Frame 1 is strictly positive and finite: Z1 > min_depth and isfinite(Z1)
     (b) Transformed depth in Frame 2 is strictly positive: Z2 > min_depth and isfinite(Z2)
     (c) Reprojected 2D coordinates lie within Frame 2 sensor boundaries:
         0 <= u2 <= W2 - 1 and 0 <= v2 <= H2 - 1 and isfinite(u2, v2)

CRITICAL NOTE ON PROXY VS. METRIC DEPTH:
----------------------------------------
MiDaS Small estimates affine-invariant relative inverse depth (disparity up to an unknown
scale and shift). Inverting this disparity produces *proxy depth* rather than calibrated
metric depth in meters.
- Pure Camera Rotation (t_rel = 0): Rigid flow is SCALE-INVARIANT because depth Z cancels out
  completely: p2 = K2 @ R_rel @ K1^{-1} @ p1.
- Camera Translation (t_rel != 0): Rigid flow magnitude is inversely proportional to depth:
  flow ~ t_rel / Z. Using MiDaS proxy depth without metric scale alignment yields flow in
  proxy units. For physically accurate metric flow under camera translation, depth must be
  scaled to true metric units (meters).
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import cv2

from src.depth.intrinsics import CameraIntrinsics


def compute_relative_transform(
    extrinsic_1: Union[torch.Tensor, np.ndarray],
    extrinsic_2: Union[torch.Tensor, np.ndarray],
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute relative camera transformation T_{2<-1} from world-to-camera matrices E1 and E2.

    Mathematical formulation:
      P_cam1 = R1 @ P_world + t1
      P_cam2 = R2 @ P_world + t2
      T_{2<-1} = T2 @ T1^{-1}
      R_rel = R2 @ R1^T
      t_rel = t2 - R_rel @ t1

    Args:
        extrinsic_1: [3, 4] or [4, 4] world-to-camera matrix of frame 1 (or batched [B, 3/4, 4]).
        extrinsic_2: [3, 4] or [4, 4] world-to-camera matrix of frame 2 (or batched [B, 3/4, 4]).
        device: Target torch device.
        dtype: Target torch dtype (default float32).

    Returns:
        R_rel: Relative rotation matrix [3, 3] (or [B, 3, 3]).
        t_rel: Relative translation vector [3] (or [B, 3]).
        T_rel: Full 4x4 relative transformation matrix [4, 4] (or [B, 4, 4]).
    """
    if isinstance(extrinsic_1, np.ndarray):
        e1 = torch.tensor(extrinsic_1, device=device, dtype=dtype)
    else:
        e1 = extrinsic_1.to(device=device, dtype=dtype)

    if isinstance(extrinsic_2, np.ndarray):
        e2 = torch.tensor(extrinsic_2, device=device, dtype=dtype)
    else:
        e2 = extrinsic_2.to(device=device, dtype=dtype)

    is_batched = e1.ndim == 3

    def to_4x4(mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[-2:] == (4, 4):
            return mat
        elif mat.shape[-2:] == (3, 4):
            if mat.ndim == 2:
                row4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=mat.device, dtype=mat.dtype)
                return torch.cat([mat, row4], dim=0)
            else:
                b = mat.shape[0]
                row4 = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=mat.device, dtype=mat.dtype).expand(b, -1, -1)
                return torch.cat([mat, row4], dim=1)
        else:
            raise ValueError(f"Expected matrix of shape (3, 4) or (4, 4), got {tuple(mat.shape)}")

    t1_4x4 = to_4x4(e1)
    t2_4x4 = to_4x4(e2)

    # For SE(3) transformation, inverse of [R | t] is [R^T | -R^T @ t]
    if is_batched:
        r1 = t1_4x4[:, :3, :3]
        t1 = t1_4x4[:, :3, 3:]
        r1_t = r1.transpose(-2, -1)
        t1_inv = torch.cat([r1_t, -torch.bmm(r1_t, t1)], dim=-1)
        row4 = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=e1.device, dtype=e1.dtype).expand(e1.shape[0], -1, -1)
        t1_inv_4x4 = torch.cat([t1_inv, row4], dim=1)
        t_rel_4x4 = torch.bmm(t2_4x4, t1_inv_4x4)
        r_rel = t_rel_4x4[:, :3, :3]
        t_rel = t_rel_4x4[:, :3, 3]
    else:
        r1 = t1_4x4[:3, :3]
        t1 = t1_4x4[:3, 3:]
        r1_t = r1.T
        t1_inv = torch.cat([r1_t, -r1_t @ t1], dim=-1)
        row4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=e1.device, dtype=e1.dtype)
        t1_inv_4x4 = torch.cat([t1_inv, row4], dim=0)
        t_rel_4x4 = t2_4x4 @ t1_inv_4x4
        r_rel = t_rel_4x4[:3, :3]
        t_rel = t_rel_4x4[:3, 3]

    return r_rel, t_rel, t_rel_4x4


def compute_rigid_flow(
    depth: Union[torch.Tensor, np.ndarray],
    intrinsics_1: CameraIntrinsics,
    intrinsics_2: Optional[CameraIntrinsics] = None,
    extrinsic_1: Optional[Union[torch.Tensor, np.ndarray]] = None,
    extrinsic_2: Optional[Union[torch.Tensor, np.ndarray]] = None,
    relative_transform: Optional[Union[torch.Tensor, np.ndarray]] = None,
    min_depth: float = 1e-4,
    boundary_eps: float = 1e-3,
    zero_invalid: bool = True,
    output_channels_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dense rigid camera-induced optical flow from depth and camera pose change.

    Workflow:
      1. Backproject 2D pixel coordinates + depth Z1 to 3D point cloud P1 (in Camera 1 frame).
      2. Transform 3D points to Camera 2 frame: P2 = R_rel @ P1 + t_rel.
      3. Reproject 3D points P2 to 2D image coordinates (u2, v2) via Camera 2 intrinsics K2.
      4. Compute displacement field: rigid_flow = (u2 - u1, v2 - v1).
      5. Formulate validity mask: Z1 > min_depth, Z2 > min_depth, and (u2, v2) in frame 2 bounds.

    Args:
        depth: 2D tensor [H, W], 3D tensor [1, H, W], or 4D batch [B, 1, H, W] of positive depth.
        intrinsics_1: CameraIntrinsics for frame 1.
        intrinsics_2: Optional CameraIntrinsics for frame 2 (defaults to intrinsics_1).
        extrinsic_1: Optional world-to-camera matrix [3, 4] for frame 1 (defaults to intrinsics_1.extrinsic).
        extrinsic_2: Optional world-to-camera matrix [3, 4] for frame 2 (defaults to intrinsics_2.extrinsic).
        relative_transform: Optional precomputed relative transform [3, 4] or [4, 4] T_{2<-1}.
                            If provided, overrides extrinsic_1 and extrinsic_2.
        min_depth: Minimum positive depth threshold for valid points (default: 1e-4).
        zero_invalid: If True, sets invalid flow vectors to (0.0, 0.0) (default: True).
        output_channels_first: If True, returns flow with channels at dim 0/1:
                               [2, H, W] for 2D inputs, [B, 2, H, W] for 4D inputs.
                               If False, returns [H, W, 2] for 2D, [B, 2, H, W] for 4D.

    Returns:
        rigid_flow: Rigid optical flow tensor representing (u2 - u1, v2 - v1).
                    Shape: [H, W, 2] (or [2, H, W]) for 2D depth; [B, 2, H, W] for 4D depth.
        valid_mask: Boolean validity mask tensor.
                    Shape: [H, W] for 2D depth; [B, 1, H, W] for 4D depth.
    """
    if intrinsics_2 is None:
        intrinsics_2 = intrinsics_1

    # Resolve torch tensor for depth
    if isinstance(depth, np.ndarray):
        depth_t = torch.tensor(depth, dtype=torch.float32)
    else:
        depth_t = depth
    device = depth_t.device
    dtype = depth_t.dtype

    is_4d = depth_t.ndim == 4
    if is_4d:
        b, c, h, w = depth_t.shape
        if c != 1:
            raise ValueError(f"Expected 1 channel for 4D depth tensor, got {c}")
    elif depth_t.ndim == 3 and depth_t.shape[0] == 1:
        depth_t = depth_t.squeeze(0)
        is_4d = False
        h, w = depth_t.shape
    elif depth_t.ndim == 2:
        is_4d = False
        h, w = depth_t.shape
    else:
        raise ValueError(f"Unsupported depth tensor shape: {tuple(depth_t.shape)}")

    if (h, w) != (intrinsics_1.height, intrinsics_1.width):
        raise ValueError(
            f"Depth resolution ({h}, {w}) does not match frame 1 intrinsics ({intrinsics_1.height}, {intrinsics_1.width})"
        )

    # Resolve relative camera transform T_{2<-1}
    if relative_transform is not None:
        if isinstance(relative_transform, np.ndarray):
            t_rel_mat = torch.tensor(relative_transform, device=device, dtype=dtype)
        else:
            t_rel_mat = relative_transform.to(device=device, dtype=dtype)
        r_rel = t_rel_mat[:3, :3]
        t_rel = t_rel_mat[:3, 3]
    else:
        e1 = extrinsic_1 if extrinsic_1 is not None else intrinsics_1.extrinsic
        e2 = extrinsic_2 if extrinsic_2 is not None else intrinsics_2.extrinsic
        if e1 is None or e2 is None:
            raise ValueError(
                "Extrinsics must be provided either through CameraIntrinsics.extrinsic or extrinsic_1/extrinsic_2"
            )
        r_rel, t_rel, _ = compute_relative_transform(e1, e2, device=device, dtype=dtype)

    # 1. Backproject Frame 1 depth to 3D point cloud P1
    # points_1: [H, W, 3] or [B, 3, H, W]
    points_1 = intrinsics_1.backproject_torch(depth_t)

    # 2. Rigidly transform 3D points: P2 = R_rel @ P1 + t_rel
    if is_4d:
        # points_1: [B, 3, H, W]
        # P2 = R_rel @ P1 + t_rel
        if r_rel.ndim == 2:
            # Broadcast single R_rel across batch
            p2 = torch.einsum("ij, b j h w -> b i h w", r_rel, points_1) + t_rel.view(1, 3, 1, 1)
        else:
            p2 = torch.einsum("bij, b j h w -> b i h w", r_rel, points_1) + t_rel.view(-1, 3, 1, 1)
    else:
        # points_1: [H, W, 3]
        # (P1 @ R_rel^T) + t_rel
        p2 = torch.matmul(points_1, r_rel.T) + t_rel.view(1, 1, 3)

    # 3. Reproject transformed 3D points P2 into Frame 2 image plane
    # uv_2: [H, W, 2] or [B, 2, H, W]
    # valid_z: [H, W] or [B, 1, H, W] (points with Z2 > min_depth)
    uv_2, valid_z = intrinsics_2.reproject_torch(p2, eps=min_depth)

    # 4. Generate original pixel grid in Frame 1
    u1, v1 = intrinsics_1.get_pixel_grid(device=device)

    # 5. Compute displacement (rigid optical flow) and boundary validity
    w2, h2 = float(intrinsics_2.width), float(intrinsics_2.height)

    if is_4d:
        uv_1 = torch.stack([u1, v1], dim=0).unsqueeze(0).expand(depth_t.shape[0], -1, -1, -1)
        rigid_flow = uv_2 - uv_1  # [B, 2, H, W]

        # Validity conditions
        u2 = uv_2[:, 0:1, :, :]
        v2 = uv_2[:, 1:2, :, :]
        in_bounds = (
            (u2 >= -boundary_eps)
            & (u2 <= (w2 - 1.0) + boundary_eps)
            & (v2 >= -boundary_eps)
            & (v2 <= (h2 - 1.0) + boundary_eps)
        )
        valid_finite = torch.isfinite(rigid_flow).all(dim=1, keepdim=True)
        valid_z1 = (depth_t > min_depth) & torch.isfinite(depth_t)

        valid_mask = valid_z & in_bounds & valid_finite & valid_z1  # [B, 1, H, W]

        if zero_invalid:
            rigid_flow = torch.where(valid_mask.expand(-1, 2, -1, -1), rigid_flow, torch.zeros_like(rigid_flow))

        return rigid_flow, valid_mask

    else:
        uv_1 = torch.stack([u1, v1], dim=-1)  # [H, W, 2]
        rigid_flow = uv_2 - uv_1  # [H, W, 2]

        # Validity conditions
        u2 = uv_2[..., 0]
        v2 = uv_2[..., 1]
        in_bounds = (
            (u2 >= -boundary_eps)
            & (u2 <= (w2 - 1.0) + boundary_eps)
            & (v2 >= -boundary_eps)
            & (v2 <= (h2 - 1.0) + boundary_eps)
        )
        valid_finite = torch.isfinite(rigid_flow).all(dim=-1)
        valid_z1 = (depth_t > min_depth) & torch.isfinite(depth_t)

        valid_mask = valid_z & in_bounds & valid_finite & valid_z1  # [H, W]

        if zero_invalid:
            rigid_flow = torch.where(valid_mask.unsqueeze(-1), rigid_flow, torch.zeros_like(rigid_flow))

        if output_channels_first:
            # Permute [H, W, 2] -> [2, H, W]
            rigid_flow = rigid_flow.permute(2, 0, 1)

        return rigid_flow, valid_mask


def flow_to_hsv_bgr(
    flow: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_flow: Optional[float] = None,
) -> np.ndarray:
    """
    Render 2D optical flow field into standard HSV-colorcoded BGR visualization.

    Hue encodes flow direction / angle in [0, 2*pi).
    Value (brightness) encodes normalized flow magnitude.

    Args:
        flow: Optical flow field as [H, W, 2] or [2, H, W] (numpy array or torch tensor).
        valid_mask: Optional boolean mask [H, W]. Invalid pixels are rendered black.
        max_flow: Optional maximum flow magnitude for brightness normalization.
                  If None, normalizes dynamically to maximum valid magnitude.

    Returns:
        bgr_image: [H, W, 3] uint8 NumPy array formatted for OpenCV / cv2.imwrite.
    """
    if isinstance(flow, torch.Tensor):
        flow_np = flow.detach().cpu().numpy()
    else:
        flow_np = np.asarray(flow)

    if flow_np.ndim == 3 and flow_np.shape[0] == 2:
        flow_np = np.transpose(flow_np, (1, 2, 0))

    if flow_np.ndim != 3 or flow_np.shape[-1] != 2:
        raise ValueError(f"Expected flow shape [H, W, 2] or [2, H, W], got {flow_np.shape}")

    h, w = flow_np.shape[:2]
    u = flow_np[..., 0]
    v = flow_np[..., 1]

    mag, ang = cv2.cartToPolar(u, v)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Hue: angle / 2 in [0, 180)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    # Saturation: fully saturated
    hsv[..., 1] = 255

    # Value: magnitude scaled
    if max_flow is None:
        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                mask_np = valid_mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(valid_mask, dtype=bool)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            valid_mags = mag[mask_np]
            max_val = float(np.percentile(valid_mags, 99)) if len(valid_mags) > 0 else 1.0
        else:
            max_val = float(np.max(mag)) if np.max(mag) > 0 else 1.0
        max_flow = max(max_val, 1e-3)

    norm_mag = np.clip(mag / max_flow * 255.0, 0.0, 255.0).astype(np.uint8)
    hsv[..., 2] = norm_mag

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(valid_mask, dtype=bool)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)
        bgr[~mask_np] = 0

    return bgr
