"""Conservative region detection and perspective correction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedRegion:
    corners: np.ndarray
    area: float


def _scaled_work_image(image: np.ndarray, maximum_side: int = 1800) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _order_corners(corners: np.ndarray) -> np.ndarray:
    points = corners.astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def detect_rectangular_regions(
    image: np.ndarray,
    *,
    minimum_area_ratio: float = 0.015,
    padding_ratio: float = 0.008,
) -> list[DetectedRegion]:
    """Find photo-like rectangles against the scanner-bed background."""

    work, scale = _scaled_work_image(image)
    work_8 = cv2.normalize(work, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(work_8, cv2.COLOR_RGB2LAB).astype(np.float32)

    border = max(5, min(work.shape[:2]) // 100)
    border_pixels = np.concatenate(
        [
            lab[:border].reshape(-1, 3),
            lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3),
            lab[:, -border:].reshape(-1, 3),
        ]
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = max(5, (min(work.shape[:2]) // 150) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = work.shape[0] * work.shape[1]
    regions: list[DetectedRegion] = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < image_area * minimum_area_ratio:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) < 20:
            continue
        rectangularity = contour_area / max(width * height, 1)
        if rectangularity < 0.60:
            continue
        padded_size = (width * (1 + padding_ratio), height * (1 + padding_ratio))
        padded_rect = (rect[0], padded_size, rect[2])
        corners = cv2.boxPoints(padded_rect) / scale
        corners[:, 0] = np.clip(corners[:, 0], 0, image.shape[1] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, image.shape[0] - 1)
        regions.append(DetectedRegion(_order_corners(corners), contour_area / scale**2))

    return sorted(regions, key=lambda item: (item.corners[:, 1].min(), item.corners[:, 0].min()))


def perspective_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_corners(corners)
    width = round(
        max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
    )
    height = round(
        max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
    )
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32), target
    )
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC)
