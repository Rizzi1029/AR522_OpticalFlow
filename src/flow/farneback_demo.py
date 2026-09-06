import os
from pathlib import Path
import cv2
import numpy as np


def flow_to_hsv_bgr(flow: np.ndarray) -> np.ndarray:
    """Convert a 2D optical flow field to an HSV-colored BGR image."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def run_farneback_demo(
    height: int = 160,
    width: int = 160,
    rect_x: int = 50,
    rect_y: int = 50,
    rect_w: int = 40,
    rect_h: int = 40,
    gt_shift_x: int = 6,
    gt_shift_y: int = 0,
    output_dir: str = "outputs",
):
    # 1. Create two synthetic grayscale frames containing a filled rectangle
    frame1 = np.zeros((height, width), dtype=np.uint8)
    frame2 = np.zeros((height, width), dtype=np.uint8)

    # Place rectangle at original position in frame 1
    cv2.rectangle(
        frame1,
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        color=255,
        thickness=-1,
    )

    # Place rectangle shifted horizontally by known pixels in frame 2
    cv2.rectangle(
        frame2,
        (rect_x + gt_shift_x, rect_y + gt_shift_y),
        (rect_x + rect_w + gt_shift_x, rect_y + rect_h + gt_shift_y),
        color=255,
        thickness=-1,
    )

    # 2. Estimate dense optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(
        prev=frame1,
        next=frame2,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    # 3. Estimate flow near the rectangle (motion region)
    roi_flow = flow[
        rect_y : rect_y + rect_h,
        rect_x : rect_x + rect_w + gt_shift_x,
    ]
    estimated_flow_x = float(np.mean(roi_flow[..., 0]))
    estimated_flow_y = float(np.mean(roi_flow[..., 1]))

    # 4. Print required information
    print(f"1. Frame Shape: {frame1.shape}")
    print(f"2. Flow Array Shape: {flow.shape}")
    print(
        f"3. Estimated Flow near Rectangle: "
        f"Horizontal (u) = {estimated_flow_x:.3f} px, Vertical (v) = {estimated_flow_y:.3f} px"
    )
    print(
        f"4. Ground-Truth Shift: "
        f"Horizontal (dx) = {gt_shift_x} px, Vertical (dy) = {gt_shift_y} px"
    )

    # 5. Save synthetic frames and flow visualization to output_dir
    os.makedirs(output_dir, exist_ok=True)
    frame1_path = os.path.join(output_dir, "frame1.png")
    frame2_path = os.path.join(output_dir, "frame2.png")
    flow_viz_path = os.path.join(output_dir, "flow_visualization.png")

    flow_bgr = flow_to_hsv_bgr(flow)

    cv2.imwrite(frame1_path, frame1)
    cv2.imwrite(frame2_path, frame2)
    cv2.imwrite(flow_viz_path, flow_bgr)

    print(f"\nSaved outputs:")
    print(f" - Frame 1: {frame1_path}")
    print(f" - Frame 2: {frame2_path}")
    print(f" - Flow Visualization: {flow_viz_path}")

    return {
        "frame_shape": frame1.shape,
        "flow_shape": flow.shape,
        "estimated_flow": (estimated_flow_x, estimated_flow_y),
        "ground_truth_shift": (gt_shift_x, gt_shift_y),
    }


if __name__ == "__main__":
    # Resolve output path relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = str(project_root / "outputs")
    run_farneback_demo(output_dir=out_dir)
