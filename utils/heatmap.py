from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def detection_centers_from_xyxy(xyxy: np.ndarray) -> list[tuple[int, int]]:
    """Return integer center points from an Nx4 xyxy array."""
    if xyxy is None or len(xyxy) == 0:
        return []

    centers: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in xyxy:
        cx = int(round((float(x1) + float(x2)) * 0.5))
        cy = int(round((float(y1) + float(y2)) * 0.5))
        centers.append((cx, cy))
    return centers


def build_spatial_heatmap_overlay(
    points: Iterable[tuple[int, int]],
    frame_shape: tuple[int, int],
    base_frame_bgr: np.ndarray | None = None,
    alpha: float = 0.45,
    point_radius_px: int = 18,
    blur_kernel_px: int = 61,
) -> np.ndarray | None:
    """Build a spatial heatmap image and optionally overlay it on a base frame."""
    points = list(points)
    if not points:
        return None

    if len(frame_shape) < 2:
        return None
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    if frame_height <= 0 or frame_width <= 0:
        return None

    alpha = float(max(0.0, min(1.0, alpha)))
    point_radius_px = max(1, int(point_radius_px))
    blur_kernel_px = max(3, int(blur_kernel_px))
    if blur_kernel_px % 2 == 0:
        blur_kernel_px += 1

    density = np.zeros((frame_height, frame_width), dtype=np.float32)

    # Accumulate local density around each center point.
    for cx, cy in points:
        if 0 <= cx < frame_width and 0 <= cy < frame_height:
            cv2.circle(density, (cx, cy), point_radius_px, 1.0, thickness=-1)

    if float(density.max()) <= 0.0:
        return None

    density = cv2.GaussianBlur(density, (blur_kernel_px, blur_kernel_px), 0)

    max_value = float(density.max())
    if max_value <= 0.0:
        return None

    normalized = np.clip(density / max_value, 0.0, 1.0)
    heat_uint8 = (normalized * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    if base_frame_bgr is None:
        return heatmap_bgr

    if base_frame_bgr.shape[:2] != (frame_height, frame_width):
        resized_base = cv2.resize(base_frame_bgr, (frame_width, frame_height))
    else:
        resized_base = base_frame_bgr

    return cv2.addWeighted(resized_base, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
