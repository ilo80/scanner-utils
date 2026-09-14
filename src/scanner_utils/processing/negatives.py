"""Film-strip frame detection and negative-to-positive conversion."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from scanner_utils.errors import ProcessingError
from scanner_utils.processing.io import read_image, write_image


def negative_to_positive(image: np.ndarray) -> np.ndarray:
    """Invert a color negative with per-channel orange-mask compensation.

    Percentile normalization is performed in floating point and converted back
    to the original integer depth only at the end.
    """

    if not np.issubdtype(image.dtype, np.integer):
        raise ProcessingError("Negative conversion expects integer image samples.")
    maximum = float(np.iinfo(image.dtype).max)
    working = image.astype(np.float32)
    result = np.empty_like(working)
    for channel in range(3):
        low, high = np.percentile(working[:, :, channel], (0.5, 99.5))
        if high - low < maximum * 0.005:
            low, high = 0.0, maximum
        normalized = np.clip((working[:, :, channel] - low) / (high - low), 0.0, 1.0)
        result[:, :, channel] = 1.0 - normalized

    medians = np.median(result.reshape(-1, 3), axis=0)
    target = float(np.mean(medians))
    gains = target / np.maximum(medians, 1e-4)
    gains = np.clip(gains, 0.5, 2.0)
    result *= gains
    return np.round(np.clip(result, 0.0, 1.0) * maximum).astype(image.dtype)


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _projection_frame_bounds(positive: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find bright frame runs in a positive image along its longest axis."""

    portrait = positive.shape[0] > positive.shape[1]
    working = np.rot90(positive) if portrait else positive
    gray = cv2.cvtColor(
        cv2.normalize(working, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    height, width = gray.shape
    y_margin = max(1, height // 20)
    central = gray[y_margin : height - y_margin]
    column_level = np.percentile(central, 65, axis=0)
    smooth_width = max(5, (width // 400) | 1)
    column_level = cv2.GaussianBlur(column_level.reshape(1, -1), (smooth_width, 1), 0).ravel()

    low, high = np.percentile(column_level, (10, 90))
    threshold = low + (high - low) * 0.28
    content = column_level > threshold
    close_width = max(3, width // 300)
    content = np.convolve(content.astype(np.uint8), np.ones(close_width), mode="same") > 0

    minimum_width = max(30, width // 20)
    runs = [(start, end) for start, end in _true_runs(content) if end - start >= minimum_width]
    x_padding = max(2, width // 500)
    y_padding = max(2, height // 100)
    landscape_bounds = [
        (
            max(0, start - x_padding),
            max(0, y_margin - y_padding),
            min(width, end + x_padding),
            min(height, height - y_margin + y_padding),
        )
        for start, end in runs
    ]
    if not portrait:
        return landscape_bounds
    original_width = positive.shape[1]
    return [
        (original_width - bottom, left, original_width - top, right)
        for left, top, right, bottom in landscape_bounds
    ]


def _largest_bright_run(profile: np.ndarray) -> tuple[int, int] | None:
    low, high = np.percentile(profile, (10, 90))
    if high - low <= 1e-6:
        return None
    threshold = low + (high - low) * 0.25
    runs = _true_runs(profile > threshold)
    return max(runs, key=lambda run: run[1] - run[0], default=None)


def _regular_edge_chain(edge_profile: np.ndarray, cross_size: int) -> list[int]:
    """Select periodically spaced film edges while rejecting holder and scene edges."""

    if len(edge_profile) < 3 or float(edge_profile.max()) == 0:
        return []
    candidates = [
        index
        for index in range(1, len(edge_profile) - 1)
        if edge_profile[index] >= edge_profile[index - 1]
        and edge_profile[index] >= edge_profile[index + 1]
        and edge_profile[index] >= float(edge_profile.max()) * 0.20
    ]
    minimum_period = max(10.0, cross_size * 0.8)
    maximum_period = cross_size * 2.2
    best: tuple[int, float, float, list[int]] = (0, 0.0, float("-inf"), [])
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            period = second - first
            if period < minimum_period or period > maximum_period:
                continue
            chain = [first, second]
            expected = second + period
            while expected < len(edge_profile):
                nearby = [
                    candidate
                    for candidate in candidates
                    if candidate > chain[-1] and abs(candidate - expected) <= period * 0.18
                ]
                if not nearby:
                    expected += period
                    continue
                selected = max(nearby, key=lambda candidate: edge_profile[candidate])
                chain.append(selected)
                expected = first + period * len(chain)
            strengths = float(sum(edge_profile[index] for index in chain))
            gaps = np.diff(chain)
            regularity = -float(np.std(gaps) / max(float(np.mean(gaps)), 1.0))
            score = (len(chain), strengths, regularity, chain)
            if score[:3] > best[:3]:
                best = score
    return best[3] if best[0] >= 2 else []


def _negative_edge_bounds(raw: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Use physical film-edge transitions in a raw color-negative scan."""

    portrait = raw.shape[0] > raw.shape[1]
    working = np.rot90(raw) if portrait else raw
    maximum = float(np.iinfo(working.dtype).max)
    normalized = working.astype(np.float32) / maximum
    luminance = normalized.mean(axis=2)

    cross_run = _largest_bright_run(np.median(luminance, axis=1))
    if cross_run is None:
        return []
    top, bottom = cross_run
    if bottom - top < 8:
        return []

    differences = np.linalg.norm(np.diff(normalized[top:bottom, :], axis=1), axis=2)
    edge_profile = np.median(differences, axis=0)
    edges = _regular_edge_chain(edge_profile, bottom - top)
    if len(edges) < 2:
        return []

    sequence_padding = max(1, round((bottom - top) * 0.02))
    cross_padding = max(1, round((bottom - top) * 0.03))
    landscape_bounds = [
        (
            max(0, left - sequence_padding),
            max(0, top - cross_padding),
            min(working.shape[1], right + sequence_padding),
            min(working.shape[0], bottom + cross_padding),
        )
        for left, right in pairwise(edges)
    ]
    if not portrait:
        return landscape_bounds
    original_width = raw.shape[1]
    return [
        (original_width - bottom, left, original_width - top, right)
        for left, top, right, bottom in landscape_bounds
    ]


def detect_frame_bounds(positive: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect bright positive frames in either strip orientation."""

    return _projection_frame_bounds(positive)


def process_negative_scan(raw_path: Path, output_paths: list[Path]) -> list[Path]:
    raw = read_image(raw_path)
    bounds = _negative_edge_bounds(raw)
    positive = None
    if not bounds:
        positive = negative_to_positive(raw)
        bounds = detect_frame_bounds(positive)
    if not bounds:
        raise ProcessingError(
            "No negative frame was detected. The untouched raw scan was kept for recovery."
        )
    if len(output_paths) < len(bounds):
        raise ProcessingError("Not enough output paths were reserved for detected frames.")

    written: list[Path] = []
    for (left, top, right, bottom), output_path in zip(bounds, output_paths, strict=False):
        if positive is None:
            frame = negative_to_positive(raw[top:bottom, left:right])
        else:
            frame = positive[top:bottom, left:right]
        write_image(output_path, frame)
        written.append(output_path)
    return written
