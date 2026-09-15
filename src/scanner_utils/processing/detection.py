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


def _bright_neutral_background(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a white scanner-bed component that separates edge-touching photos."""

    lightness = lab[:, :, 0]
    chroma = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)
    candidate_mask = ((lightness >= 235) & (chroma <= 12)).astype(np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )
    if component_count <= 1:
        return None

    largest_label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    component = labels == largest_label
    if np.count_nonzero(component) < component.size * 0.02:
        return None

    touches_top = bool(component[0].any())
    touches_bottom = bool(component[-1].any())
    touches_left = bool(component[:, 0].any())
    touches_right = bool(component[:, -1].any())
    if not ((touches_top and touches_bottom) or (touches_left and touches_right)):
        return None
    return np.median(lab[component], axis=0), component


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _edge_touching_gutter_contours(
    background_component: np.ndarray, minimum_area_ratio: float
) -> list[np.ndarray]:
    """Build photo cells from white gutters spanning the complete scan."""

    height, width = background_component.shape
    minimum_gutter = max(3, round(min(height, width) * 0.01))
    vertical_gutters = [
        (start, end)
        for start, end in _true_runs(background_component.mean(axis=0) >= 0.80)
        if start > 0 and end < width and end - start >= minimum_gutter
    ]
    horizontal_gutters = [
        (start, end)
        for start, end in _true_runs(background_component.mean(axis=1) >= 0.80)
        if start > 0 and end < height and end - start >= minimum_gutter
    ]
    # A single spanning strip does not define a grid: two diagonally placed
    # photos may share a vertical or horizontal gap while having independent
    # bounds on the other axis. Contour detection handles that layout better.
    if not vertical_gutters or not horizontal_gutters:
        return []

    def intervals(size: int, gutters: list[tuple[int, int]]) -> list[tuple[int, int]]:
        boundaries = [0, *(value for gutter in gutters for value in gutter), size]
        return [
            (boundaries[index], boundaries[index + 1]) for index in range(0, len(boundaries) - 1, 2)
        ]

    contours: list[np.ndarray] = []
    image_area = height * width
    for top, bottom in intervals(height, horizontal_gutters):
        for left, right in intervals(width, vertical_gutters):
            area = (right - left) * (bottom - top)
            if area < image_area * minimum_area_ratio:
                continue
            foreground_fraction = 1.0 - float(background_component[top:bottom, left:right].mean())
            touches_edge = left == 0 or right == width or top == 0 or bottom == height
            if foreground_fraction < 0.20 or not touches_edge:
                continue
            contours.append(
                np.array(
                    [
                        [[left, top]],
                        [[right - 1, top]],
                        [[right - 1, bottom - 1]],
                        [[left, bottom - 1]],
                    ],
                    dtype=np.int32,
                )
            )
    return contours if len(contours) >= 2 else []


def _boxes_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    first_x, first_y, first_width, first_height = cv2.boundingRect(first)
    second_x, second_y, second_width, second_height = cv2.boundingRect(second)
    return max(first_x, second_x) < min(first_x + first_width, second_x + second_width) and max(
        first_y, second_y
    ) < min(first_y + first_height, second_y + second_height)


def _merge_overlapping_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    """Rejoin photo fragments split by a large white area inside the print."""

    groups = [[contour] for contour in contours]
    merged = True
    while merged:
        merged = False
        for first_index in range(len(groups)):
            first_points = np.concatenate(groups[first_index])
            for second_index in range(first_index + 1, len(groups)):
                second_points = np.concatenate(groups[second_index])
                if not _boxes_overlap(first_points, second_points):
                    continue
                groups[first_index].extend(groups.pop(second_index))
                merged = True
                break
            if merged:
                break

    return [
        cv2.convexHull(np.concatenate(group)) if len(group) > 1 else group[0] for group in groups
    ]


def detect_rectangular_regions(
    image: np.ndarray,
    *,
    minimum_area_ratio: float = 0.015,
    padding_ratio: float = -0.008,
) -> list[DetectedRegion]:
    """Find photo-like rectangles against the scanner-bed background."""

    work, scale = _scaled_work_image(image)
    work_8 = cv2.normalize(work, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(work_8, cv2.COLOR_RGB2LAB).astype(np.float32)

    bright_background = _bright_neutral_background(lab)
    uses_bright_background = bright_background is not None
    background_component: np.ndarray | None = None
    if bright_background is None:
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
    else:
        background, background_component = bright_background
    distance = np.linalg.norm(lab - background, axis=2)
    distance_8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = max(5, (min(work.shape[:2]) // 150) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = work.shape[0] * work.shape[1]
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * minimum_area_ratio
    ]
    if background_component is not None:
        gutter_contours = _edge_touching_gutter_contours(background_component, minimum_area_ratio)
        contours = gutter_contours or _merge_overlapping_contours(contours)
    regions: list[DetectedRegion] = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) < 20:
            continue
        rectangularity = contour_area / max(width * height, 1)
        if rectangularity < (0.50 if uses_bright_background else 0.60):
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
