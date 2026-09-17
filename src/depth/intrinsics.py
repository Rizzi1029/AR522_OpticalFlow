"""
Camera Intrinsics and 3D Backprojection / Reprojection Foundation.

Coordinate System Conventions (Standard Computer Vision / OpenCV / Robotics Frame):
----------------------------------------------------------------------------------
1. Camera Coordinate System:
   - Origin (0, 0, 0): Optical center of the pinhole camera.
   - +X axis: Points horizontally to the RIGHT along sensor columns.
   - +Y axis: Points vertically DOWNWARDS along sensor rows.
   - +Z axis: Points FORWARD along the optical axis into the scene (depth Z > 0).
   This forms a standard right-handed coordinate system: X x Y = Z.

2. Image / Pixel Coordinate System:
   - Origin (0, 0): Center of the top-left pixel (or top-left corner under continuous coords).
   - u in [0, W - 1]: Horizontal pixel coordinate (column index).
   - v in [0, H - 1]: Vertical pixel coordinate (row index).

3. Pinhole Projection Model:
   The 3x3 camera intrinsic matrix K maps 3D points P = [X, Y, Z]^T to homogeneous
   pixel coordinates p_h = [u, v, 1]^T:

         [ fx   0   cx ]
     K = [  0  fy   cy ]
         [  0   0    1 ]

   Forward Projection (3D -> 2D):
     [u, v, 1]^T = (1 / Z) * K * [X, Y, Z]^T
     u = fx * (X / Z) + cx
     v = fy * (Y / Z) + cy

   Backprojection (2D + Depth -> 3D):
     Given pixel (u, v) and positive depth Z = Z(u, v) > 0:
     [X, Y, Z]^T = Z * K^{-1} * [u, v, 1]^T
     X = (u - cx) * Z / fx
     Y = (v - cy) * Z / fy
     Z = Z

MPI Sintel Dataset Specifications & Official Calibration:
---------------------------------------------------------
- Resolution: W = 1024, H = 436.
- Principal Point: cx = (W - 1) / 2.0 = 511.5, cy = (H - 1) / 2.0 = 217.5.
- Pixel Skew: 0.0 (orthogonal sensor axes).
- Distortion: 0.0 (pinhole perspective projection in Blender).
- Official Calibration (.cam files):
  Ground truth camera calibration for the MPI Sintel training sequences is provided
  by the MPI Sintel Depth & Camera Motion dataset (MPI-Sintel-depth-training-20150305).
  Focal lengths are scene- and frame-dependent, derived from Blender virtual cameras
  with a hard-coded sensor pixel density of 32 px/mm:
    fx = fy = f_Blender (mm) * 32 px/mm.
  Values range from 576.0 px (18mm wide) to 3200.0 px (100mm telephoto), with
  dynamic zoom shots (e.g. cave_2 and sleeping_1).
  Official camera parameters are stored in binary .cam files under:
    `training/camdata_left/<scene_name>/frame_XXXX.cam`.
- Simplifying Defaults:
  fx = fy = 1000.0 px is an informal convenience assumption (not an official parameter).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch


# Official MPI Sintel camera file header tag ('PIEH' in ASCII, float32 202021.25)
SINTEL_CAM_TAG_FLOAT = 202021.25


def read_sintel_cam_binary(cam_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an official MPI Sintel binary .cam file containing intrinsic K and extrinsic [R|T].

    MPI Sintel .cam file binary specification (172 bytes total, little-endian):
      offset  size (B)   type               contents
      0       4          float32            Tag: 'PIEH' in ASCII == float 202021.25
      4       72         9 * float64        Intrinsic 3x3 matrix K in row-major order
      76      96         12 * float64       Extrinsic 3x4 matrix [R | T] in row-major order

    Args:
        cam_path: Filepath to the .cam calibration file.

    Returns:
        K: 3x3 camera intrinsic matrix as float64 NumPy array.
        extrinsic: 3x4 world-to-camera transformation matrix [R | T] as float64 NumPy array.

    Raises:
        FileNotFoundError: If the .cam file does not exist.
        ValueError: If file size or header tag is invalid.
    """
    path = Path(cam_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sintel .cam file not found at: {path}")

    file_size = path.stat().st_size
    if file_size < 172:
        raise ValueError(
            f"Invalid Sintel .cam file size ({file_size} bytes, expected at least 172 bytes): {path}"
        )

    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype="<f4", count=1)[0]
        if not np.isclose(tag, SINTEL_CAM_TAG_FLOAT, atol=1e-2):
            raise ValueError(
                f"Invalid Sintel .cam tag: expected {SINTEL_CAM_TAG_FLOAT} ('PIEH'), got {tag} in {path}. "
                f"Corrupt or endianness mismatch."
            )
        K = np.fromfile(f, dtype="<f8", count=9).reshape((3, 3))
        extrinsic = np.fromfile(f, dtype="<f8", count=12).reshape((3, 4))

    return K, extrinsic


@dataclass
class CameraIntrinsics:
    """
    Pinhole Camera Intrinsics encapsulation with backprojection and reprojection.

    Attributes:
        fx: Focal length along horizontal axis (pixels).
        fy: Focal length along vertical axis (pixels).
        cx: Principal point horizontal coordinate (pixels).
        cy: Principal point vertical coordinate (pixels).
        width: Image width in pixels.
        height: Image height in pixels.
        extrinsic: Optional 3x4 world-to-camera extrinsic matrix [R | T].
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    extrinsic: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be strictly positive: fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be strictly positive: {self.width}x{self.height}")
        if self.extrinsic is not None:
            ext = np.asarray(self.extrinsic, dtype=np.float64)
            if ext.shape != (3, 4):
                raise ValueError(f"Extrinsic matrix must have shape (3, 4), got {ext.shape}")
            self.extrinsic = ext

    @property
    def K(self) -> np.ndarray:
        """3x3 Intrinsic matrix as float64 NumPy array."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def K_inv(self) -> np.ndarray:
        """3x3 Inverse intrinsic matrix as float64 NumPy array."""
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_k_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return 3x3 intrinsic matrix K as PyTorch tensor."""
        return torch.tensor(self.K, device=device, dtype=dtype)

    def get_k_inv_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return 3x3 inverse intrinsic matrix K^{-1} as PyTorch tensor."""
        return torch.tensor(self.K_inv, device=device, dtype=dtype)

    @classmethod
    def from_cam_file(
        cls,
        cam_path: Union[str, Path],
        width: int = 1024,
        height: int = 436,
    ) -> "CameraIntrinsics":
        """
        Load official camera calibration from an MPI Sintel .cam file.

        Args:
            cam_path: Path to official .cam file.
            width: Image width in pixels (default 1024).
            height: Image height in pixels (default 436).

        Returns:
            CameraIntrinsics initialized with calibrated fx, fy, cx, cy and extrinsic.
        """
        K, extrinsic = read_sintel_cam_binary(cam_path)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            extrinsic=extrinsic,
        )

    @classmethod
    def from_sintel_frame(
        cls,
        frame_path: Union[str, Path],
        sintel_root: Optional[Union[str, Path]] = None,
        width: int = 1024,
        height: int = 436,
    ) -> "CameraIntrinsics":
        """
        Resolve and load official camera calibration for a given Sintel frame.

        Searches for the corresponding `.cam` file under `camdata_left/<scene>/<stem>.cam`.

        Args:
            frame_path: Path to a Sintel frame image (e.g., .../training/clean/alley_1/frame_0001.png).
            sintel_root: Optional root directory of the Sintel dataset. If None, inferred
                         by inspecting ancestor directories of frame_path.
            width: Image width in pixels (default 1024).
            height: Image height in pixels (default 436).

        Returns:
            CameraIntrinsics initialized with frame-specific calibration.

        Raises:
            FileNotFoundError: If the corresponding .cam file cannot be located.
        """
        p = Path(frame_path).resolve()
        scene_name = p.parent.name
        frame_stem = p.stem  # e.g., 'frame_0001'
        cam_filename = f"{frame_stem}.cam"

        candidate_paths = []
        if sintel_root is not None:
            root = Path(sintel_root).resolve()
            candidate_paths.extend([
                root / "training" / "camdata_left" / scene_name / cam_filename,
                root / "camdata_left" / scene_name / cam_filename,
            ])

        # Traverse directory hierarchy upwards from frame_path
        current = p.parent
        for _ in range(4):
            if current == current.parent:
                break
            candidate_paths.extend([
                current / "camdata_left" / scene_name / cam_filename,
                current / "training" / "camdata_left" / scene_name / cam_filename,
            ])
            current = current.parent

        for candidate in candidate_paths:
            if candidate.is_file():
                return cls.from_cam_file(candidate, width=width, height=height)

        raise FileNotFoundError(
            f"Could not locate official Sintel .cam calibration for frame '{p}'. "
            f"Checked candidates:\n" + "\n".join(f"  - {c}" for c in candidate_paths)
        )

    @classmethod
    def sintel_default(
        cls,
        width: int = 1024,
        height: int = 436,
        focal_length: float = 1000.0,
    ) -> "CameraIntrinsics":
        """
        Construct nominal camera intrinsics for MPI Sintel sequences.

        Principal point is placed at the exact sensor center:
        cx = (1024 - 1) / 2.0 = 511.5, cy = (436 - 1) / 2.0 = 217.5.

        NOTE: fx=fy=1000.0 px is an uncalibrated simplifying assumption.
        For official ground-truth geometry on Sintel training sequences,
        use `from_cam_file` or `from_sintel_frame`.
        """
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        return cls(
            fx=float(focal_length),
            fy=float(focal_length),
            cx=float(cx),
            cy=float(cy),
            width=width,
            height=height,
            extrinsic=None,
        )

    def get_pixel_grid(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dense coordinate grids (u_grid, v_grid) of shape [H, W].

        u_grid[v, u] = u (column index in [0, W - 1])
        v_grid[v, u] = v (row index in [0, H - 1])
        """
        u = torch.arange(self.width, dtype=torch.float32, device=device)
        v = torch.arange(self.height, dtype=torch.float32, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        return u_grid, v_grid

    def backproject_torch(
        self,
        depth: torch.Tensor,
        u_grid: Optional[torch.Tensor] = None,
        v_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Backproject 2D depth map to 3D point cloud [H, W, 3] or [B, 3, H, W].

        Args:
            depth: Tensor of shape [H, W], [1, H, W], or [B, 1, H, W] containing
                   positive metric/proxy depth Z > 0.
            u_grid: Optional precomputed [H, W] horizontal coordinate grid.
            v_grid: Optional precomputed [H, W] vertical coordinate grid.

        Returns:
            torch.Tensor:
                If input is 2D [H, W], returns [H, W, 3] with channels (X, Y, Z).
                If input is 4D [B, 1, H, W], returns [B, 3, H, W] with channels (X, Y, Z).
        """
        is_4d = depth.ndim == 4
        if is_4d:
            b, c, h, w = depth.shape
            if c != 1:
                raise ValueError(f"Expected 1 channel for 4D depth tensor, got {c}")
            z = depth.squeeze(1)  # [B, H, W]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            z = depth.squeeze(0)  # [H, W]
            is_4d = False
        elif depth.ndim == 2:
            z = depth  # [H, W]
            is_4d = False
        else:
            raise ValueError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")

        h, w = z.shape[-2], z.shape[-1]
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"Depth dimensions ({h}, {w}) do not match camera intrinsics ({self.height}, {self.width})"
            )

        if u_grid is None or v_grid is None:
            u_grid, v_grid = self.get_pixel_grid(device=depth.device)

        # Coordinate calculation:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x = (u_grid - self.cx) * z / self.fx
        y = (v_grid - self.cy) * z / self.fy

        if is_4d:
            # [B, 3, H, W]
            return torch.stack([x, y, z], dim=1)
        else:
            # [H, W, 3]
            return torch.stack([x, y, z], dim=-1)

    def reproject_torch(
        self,
        points_3d: torch.Tensor,
        eps: float = 1e-7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reproject 3D point cloud back to 2D pixel coordinates (u, v).

        Args:
            points_3d: Tensor of shape [H, W, 3] or [B, 3, H, W] with (X, Y, Z).
            eps: Epsilon to prevent division by zero for points with Z <= 0.

        Returns:
            uv_grid: Tensor of shape [H, W, 2] or [B, 2, H, W] with coordinates (u, v).
            valid_mask: Boolean tensor of shape [H, W] or [B, 1, H, W] indicating Z > eps.
        """
        if points_3d.ndim == 3 and points_3d.shape[-1] == 3:
            # [H, W, 3]
            x = points_3d[..., 0]
            y = points_3d[..., 1]
            z = points_3d[..., 2]
            valid = z > eps
            z_safe = torch.clamp(z, min=eps)

            u = self.fx * (x / z_safe) + self.cx
            v = self.fy * (y / z_safe) + self.cy
            uv = torch.stack([u, v], dim=-1)  # [H, W, 2]
            return uv, valid
        elif points_3d.ndim == 4 and points_3d.shape[1] == 3:
            # [B, 3, H, W]
            x = points_3d[:, 0, :, :]
            y = points_3d[:, 1, :, :]
            z = points_3d[:, 2, :, :]
            valid = (z > eps).unsqueeze(1)  # [B, 1, H, W]
            z_safe = torch.clamp(z, min=eps)

            u = self.fx * (x / z_safe) + self.cx
            v = self.fy * (y / z_safe) + self.cy
            uv = torch.stack([u, v], dim=1)  # [B, 2, H, W]
            return uv, valid
        else:
            raise ValueError(f"Unsupported 3D points tensor shape: {tuple(points_3d.shape)}")

    def backproject_numpy(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        NumPy implementation: Backproject 2D depth [H, W] to 3D point cloud [H, W, 3].
        """
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D array [H, W], got {depth.shape}")
        h, w = depth.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"Depth shape ({h}, {w}) does not match intrinsics ({self.height}, {self.width})"
            )

        u_coords = np.arange(w, dtype=np.float64)
        v_coords = np.arange(h, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        x = (u_grid - self.cx) * depth / self.fx
        y = (v_grid - self.cy) * depth / self.fy
        z = depth

        return np.stack([x, y, z], axis=-1)

    def reproject_numpy(
        self,
        points_3d: np.ndarray,
        eps: float = 1e-7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        NumPy implementation: Reproject 3D point cloud [H, W, 3] to pixel coordinates [H, W, 2].
        """
        if points_3d.ndim != 3 or points_3d.shape[-1] != 3:
            raise ValueError(f"Expected array [H, W, 3], got {points_3d.shape}")

        x = points_3d[..., 0]
        y = points_3d[..., 1]
        z = points_3d[..., 2]
        valid = z > eps
        z_safe = np.clip(z, a_min=eps, a_max=None)

        u = self.fx * (x / z_safe) + self.cx
        v = self.fy * (y / z_safe) + self.cy

        uv = np.stack([u, v], axis=-1)
        return uv, valid
