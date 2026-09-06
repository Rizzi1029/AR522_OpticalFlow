"""
MPI Sintel Optical Flow Dataset Loader and I/O Utilities.

Provides lightweight, dependency-free loading of MPI Sintel frame pairs,
Middlebury .flo ground-truth flow files, and Sintel invalid masks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(file_path: Union[str, Path]) -> torch.Tensor:
    """
    Read an image file as an RGB uint8 tensor.

    Args:
        file_path: Path to the image file.

    Returns:
        torch.Tensor of shape (3, H, W) in uint8 [0, 255].
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image at '{path}' (file unreadable or corrupted).")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor


def read_invalid_mask(
    file_path: Union[str, Path],
    expected_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Read Sintel invalid mask PNG file as a uint8 tensor.

    Args:
        file_path: Path to the invalid mask PNG file.
        expected_shape: Optional (height, width) tuple to validate against.

    Returns:
        torch.Tensor of shape (H, W) in uint8, where 0=valid, 255=invalid.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Invalid mask file not found: {path}")

    mask_np = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        raise ValueError(f"Failed to read invalid mask at '{path}' (file unreadable or corrupted).")

    if expected_shape is not None and mask_np.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch in mask '{path}': got {mask_np.shape}, expected {expected_shape}."
        )

    return torch.from_numpy(mask_np)


def read_flo(
    file_path: Union[str, Path],
    expected_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Read Middlebury / Sintel .flo optical flow file.

    Format specification:
        Bytes 0-3: 4-byte tag / magic number (b"PIEH" / 202021.25 in float32)
        Bytes 4-7: 4-byte int width (little-endian)
        Bytes 8-11: 4-byte int height (little-endian)
        Remaining: 2 * width * height 4-byte floats (little-endian u, v interleaved)

    Args:
        file_path: Path to the .flo file.
        expected_shape: Optional (height, width) tuple to validate against.

    Returns:
        torch.Tensor of shape (2, H, W) in torch.float32.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f".flo file not found: {path}")

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"PIEH":
            raise ValueError(
                f"Malformed .flo file '{path}': invalid magic header. "
                f"Expected b'PIEH', got {magic!r}."
            )

        w_bytes = f.read(4)
        h_bytes = f.read(4)
        if len(w_bytes) < 4 or len(h_bytes) < 4:
            raise ValueError(f"Malformed .flo file '{path}': truncated header dimensions.")

        width = np.frombuffer(w_bytes, dtype="<i4")[0].item()
        height = np.frombuffer(h_bytes, dtype="<i4")[0].item()

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Malformed .flo file '{path}': invalid spatial dimensions (width={width}, height={height})."
            )

        if expected_shape is not None:
            exp_h, exp_w = expected_shape
            if (height, width) != (exp_h, exp_w):
                raise ValueError(
                    f"Dimension mismatch in '{path}': flow file defines (height={height}, width={width}), "
                    f"but expected image shape is ({exp_h}, {exp_w})."
                )

        raw_data = f.read()
        expected_bytes = 2 * width * height * 4
        if len(raw_data) != expected_bytes:
            raise ValueError(
                f"Malformed .flo file '{path}': expected {expected_bytes} data bytes "
                f"for ({height}, {width}), but read {len(raw_data)} bytes."
            )

        data = np.frombuffer(raw_data, dtype="<f4")
        flow_np = data.reshape(height, width, 2).transpose(2, 0, 1).copy()

    return torch.from_numpy(flow_np).float().contiguous()


class SintelDataset(Dataset):
    """
    Dataset loader for MPI Sintel optical flow training split.

    Yields consecutive frame pairs [I_t, I_{t+1}], ground-truth optical flow,
    and ground-truth valid pixel masks.
    """

    def __init__(
        self,
        root: Union[str, Path] = "data/Sintel",
        split: str = "training",
        pass_name: str = "clean",
        scenes: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the Sintel dataset loader.

        Args:
            root: Root path to the Sintel dataset directory (containing 'training').
            split: Dataset split ('training').
            pass_name: Image pass name, either 'clean' or 'final'.
            scenes: Optional list of specific scene names to load. If None, loads all scenes.
        """
        super().__init__()
        self.root = Path(root).resolve()
        if split != "training":
            raise ValueError(f"Only split='training' is supported for ground-truth evaluation, got '{split}'.")
        self.split = split

        if pass_name not in ("clean", "final"):
            raise ValueError(f"pass_name must be 'clean' or 'final', got '{pass_name}'.")
        self.pass_name = pass_name

        split_root = self.root / self.split
        self.image_dir = split_root / self.pass_name
        self.flow_dir = split_root / "flow"
        self.invalid_dir = split_root / "invalid"

        for dir_path, name in [
            (self.image_dir, f"{self.split}/{self.pass_name}"),
            (self.flow_dir, f"{self.split}/flow"),
            (self.invalid_dir, f"{self.split}/invalid"),
        ]:
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Required Sintel directory not found: {dir_path} ({name})")

        all_scenes = sorted([p.name for p in self.image_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
        if scenes is not None:
            scene_set = set(scenes)
            selected_scenes = [s for s in all_scenes if s in scene_set]
            missing_scenes = scene_set - set(selected_scenes)
            if missing_scenes:
                raise ValueError(f"Requested scenes not found in {self.image_dir}: {sorted(missing_scenes)}")
        else:
            selected_scenes = all_scenes

        self.samples: List[Tuple[Path, Path, Path, Path, str, int]] = []
        for scene in selected_scenes:
            scene_img_dir = self.image_dir / scene
            scene_flow_dir = self.flow_dir / scene
            scene_inv_dir = self.invalid_dir / scene

            frames = sorted(scene_img_dir.glob("frame_*.png"))
            for i in range(len(frames) - 1):
                f1 = frames[i]
                f2 = frames[i + 1]
                stem = f1.stem  # e.g. "frame_0001"
                flow_file = scene_flow_dir / f"{stem}.flo"
                inv_file = scene_inv_dir / f"{stem}.png"

                if not flow_file.is_file():
                    raise FileNotFoundError(f"Missing ground truth flow file: {flow_file}")
                if not inv_file.is_file():
                    raise FileNotFoundError(f"Missing invalid mask file: {inv_file}")

                frame_idx = int(stem.split("_")[-1])
                self.samples.append((f1, f2, flow_file, inv_file, scene, frame_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get frame pair, ground truth flow, valid mask, and metadata.

        Returns:
            frame1: uint8 tensor [3, H, W] in [0, 255]
            frame2: uint8 tensor [3, H, W] in [0, 255]
            flow_gt: float32 tensor [2, H, W] in pixel displacement
            valid_mask: bool tensor [H, W] (True = valid pixel for EPE evaluation)
            meta: dictionary with scene, frame_idx, pass_name, and file paths
        """
        f1_path, f2_path, flow_path, inv_path, scene, frame_idx = self.samples[idx]

        frame1 = read_image(f1_path)
        frame2 = read_image(f2_path)
        h, w = frame1.shape[1], frame1.shape[2]

        if frame2.shape[1:] != (h, w):
            raise ValueError(
                f"Spatial dimension mismatch in scene '{scene}' frame {frame_idx}: "
                f"frame1={tuple(frame1.shape[1:])}, frame2={tuple(frame2.shape[1:])}"
            )

        flow_gt = read_flo(flow_path, expected_shape=(h, w))
        invalid_raw = read_invalid_mask(inv_path, expected_shape=(h, w))

        # Sintel ground-truth validity condition:
        # 1. invalid mask == 0 is the primary Sintel validity condition.
        # 2. Unmatched pixels remain INCLUDED in the valid mask (occlusions are not excluded).
        # 3. Finite-value check for numerical robustness (isfinite u and v).
        valid_mask = (invalid_raw == 0) & torch.isfinite(flow_gt[0]) & torch.isfinite(flow_gt[1])

        meta = {
            "scene": scene,
            "frame_idx": frame_idx,
            "pass_name": self.pass_name,
            "frame1_path": str(f1_path),
            "frame2_path": str(f2_path),
            "flow_path": str(flow_path),
            "invalid_path": str(inv_path),
        }

        return frame1, frame2, flow_gt, valid_mask, meta
