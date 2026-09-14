"""Conservative 35 mm frame-border trimming for converted negatives."""

from __future__ import annotations

import cv2
import numpy as np

_BORDER_INSET_FRACTION = 0.004


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _dark_border_inner_edge(
    profile: np.ndarray,
    *,
    side: str,
    maximum_width_fraction: float,
    edge_fraction: float = 0.08,
) -> int | None:
    """Return the inner edge of a dark film-border run near one side."""

    low, high = np.percentile(profile, (5, 80))
    if high - low <= 1e-6:
        return None

    threshold = low + (high - low) * 0.18
    runs = _true_runs(profile < threshold)
    size = len(profile)
    minimum_width = max(2, round(size * 0.003))
    maximum_width = max(minimum_width, round(size * maximum_width_fraction))

    if side == "start":
        eligible = [
            run
            for run in runs
            if run[0] < size * edge_fraction
            and minimum_width <= run[1] - run[0] <= maximum_width
        ]
        if not eligible:
            return None
        _, end = max(eligible, key=lambda run: (run[1] - run[0], -run[0]))
        return end

    eligible = [
        run
        for run in runs
        if run[1] > size * (1.0 - edge_fraction)
        and minimum_width <= run[1] - run[0] <= maximum_width
    ]
    if not eligible:
        return None
    start, _ = max(eligible, key=lambda run: (run[1] - run[0], run[1]))
    return start


def _strongest_edge(profile: np.ndarray, start: int, end: int) -> int | None:
    start = max(0, start)
    end = min(len(profile), end)
    if end <= start:
        return None
    return start + int(np.argmax(profile[start:end])) + 1


def _edge_peak_boundaries(
    profile: np.ndarray, start: int, end: int, limit: int = 16
) -> list[int]:
    peaks = [
        (float(profile[index]), index + 1)
        for index in range(max(1, start), min(len(profile) - 1, end))
        if profile[index] >= profile[index - 1]
        and profile[index] >= profile[index + 1]
    ]
    return [boundary for _, boundary in sorted(peaks, reverse=True)[:limit]]


def _weighted_median(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    ordered = sorted(values, key=lambda item: item[0])
    midpoint = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return ordered[-1][0]


def _estimate_border_skew(image: np.ndarray) -> float | None:
    """Estimate small film-frame skew from long lines near the image edges."""

    height, width = image.shape[:2]
    scale = min(1.0, 1800 / max(height, width))
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image
    work_height, work_width = work.shape[:2]
    normalized = cv2.normalize(work, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    minimum_side = min(work_height, work_width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 3600,
        threshold=max(50, minimum_side // 8),
        minLineLength=minimum_side // 4,
        maxLineGap=max(5, minimum_side // 30),
    )
    if lines is None:
        return None

    candidates: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        delta_x = int(x2) - int(x1)
        delta_y = int(y2) - int(y1)
        midpoint_x = (int(x1) + int(x2)) / 2
        midpoint_y = (int(y1) + int(y2)) / 2
        horizontal = abs(delta_x) >= abs(delta_y)
        near_relevant_edge = (
            midpoint_y < work_height * 0.15 or midpoint_y > work_height * 0.85
            if horizontal
            else midpoint_x < work_width * 0.15 or midpoint_x > work_width * 0.85
        )
        if not near_relevant_edge:
            continue
        angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
        skew = (angle + 45.0) % 90.0 - 45.0
        if abs(skew) <= 2.0:
            candidates.append((skew, float(np.hypot(delta_x, delta_y))))
    return _weighted_median(candidates)


def _deskew_35mm_frame(image: np.ndarray) -> np.ndarray:
    angle = _estimate_border_skew(image)
    if angle is None or abs(angle) < 0.03:
        return image
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _trim_35mm_frame(image: np.ndarray) -> np.ndarray:
    """Remove residual film/scanner borders from a converted landscape frame."""

    if image.ndim != 3 or image.shape[2] != 3:
        return image

    image = _deskew_35mm_frame(image)
    height, width = image.shape[:2]
    if width < 40 or height < 40 or width < height:
        return image

    working = image.astype(np.float32)
    luminance = working.mean(axis=2)
    column_level = np.median(luminance, axis=0)
    row_level = np.median(luminance, axis=1)
    vertical_edges = np.median(
        np.linalg.norm(np.diff(working, axis=1), axis=2), axis=0
    )
    horizontal_edges = np.median(
        np.linalg.norm(np.diff(working, axis=0), axis=2), axis=1
    )

    left = _dark_border_inner_edge(
        column_level, side="start", maximum_width_fraction=0.12
    )
    right = _dark_border_inner_edge(
        column_level, side="end", maximum_width_fraction=0.12
    )

    if left is None and right is None:
        return image

    if left is None:
        left = _strongest_edge(vertical_edges, 0, round(width * 0.12)) or 0
    if right is None:
        right = (
            _strongest_edge(vertical_edges, round(width * 0.88), width - 1)
            or width
        )

    top = _dark_border_inner_edge(
        row_level, side="start", maximum_width_fraction=0.04
    )
    bottom = _dark_border_inner_edge(
        row_level, side="end", maximum_width_fraction=0.04
    )
    top_from_gradient = top is None
    bottom_from_gradient = bottom is None

    if top is None:
        top = _strongest_edge(horizontal_edges, 0, round(height * 0.12)) or 0
    if bottom is None:
        bottom = (
            _strongest_edge(horizontal_edges, round(height * 0.88), height - 1)
            or height
        )

    ratio = (right - left) / max(1, bottom - top)
    if ratio < 1.47 or ratio > 1.53:
        top_choices = (
            _edge_peak_boundaries(horizontal_edges, 0, round(height * 0.20))
            if top_from_gradient
            else [top]
        )
        bottom_choices = (
            _edge_peak_boundaries(
                horizontal_edges, round(height * 0.80), height - 1
            )
            if bottom_from_gradient
            else [bottom]
        )

        reference = float(np.percentile(horizontal_edges, 95)) + 1e-6
        best: tuple[float, int, int] | None = None
        for candidate_top in top_choices:
            for candidate_bottom in bottom_choices:
                candidate_height = candidate_bottom - candidate_top
                if candidate_height < height * 0.70:
                    continue

                candidate_ratio = (right - left) / candidate_height
                ratio_error = abs(np.log(candidate_ratio / 1.5))
                top_strength = float(
                    horizontal_edges[
                        min(max(candidate_top - 1, 0), len(horizontal_edges) - 1)
                    ]
                )
                bottom_strength = float(
                    horizontal_edges[
                        min(
                            max(candidate_bottom - 1, 0),
                            len(horizontal_edges) - 1,
                        )
                    ]
                )
                score = -25.0 * ratio_error + 0.2 * (
                    np.log1p(top_strength / reference)
                    + np.log1p(bottom_strength / reference)
                )
                candidate = (float(score), candidate_top, candidate_bottom)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is not None:
            _, top, bottom = best

    crop_width = right - left
    crop_height = bottom - top
    if crop_width < width * 0.65 or crop_height < height * 0.65:
        return image

    final_ratio = crop_width / crop_height
    if not 1.40 <= final_ratio <= 1.58:
        return image

    if left <= 0 and top <= 0 and right >= width and bottom >= height:
        return image

    # Keep a sub-half-percent inset so antialiased or slightly irregular film
    # edges cannot survive the detected boundary.
    inset_x = max(1, round(crop_width * _BORDER_INSET_FRACTION))
    inset_y = max(1, round(crop_height * _BORDER_INSET_FRACTION))
    left += inset_x
    right -= inset_x
    top += inset_y
    bottom -= inset_y
    return np.ascontiguousarray(image[top:bottom, left:right])
