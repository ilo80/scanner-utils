"""Film-strip frame detection and densitometric negative conversion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from pathlib import Path

import cv2
import numpy as np

from scanner_utils.errors import ProcessingError
from scanner_utils.processing.io import read_image, write_image

Bounds = tuple[int, int, int, int]


@dataclass(frozen=True)
class _FilmCalibration:
    vertical: bool
    base_positions: np.ndarray
    film_bases: np.ndarray
    white_point: np.ndarray
    neutral_gains: np.ndarray


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _projection_frame_bounds(positive: np.ndarray) -> list[Bounds]:
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
    """Select consecutive 35 mm frame edges and never bridge a missing separator."""

    if len(edge_profile) < 3 or float(edge_profile.max()) == 0:
        return []
    candidates = [
        index
        for index in range(1, len(edge_profile) - 1)
        if edge_profile[index] >= edge_profile[index - 1]
        and edge_profile[index] >= edge_profile[index + 1]
        and edge_profile[index] >= float(edge_profile.max()) * 0.04
    ]
    # A 35 mm frame is about 36 x 24 mm, so its pitch is close to 1.5 times
    # the film image width. The tolerance accounts for holder borders and skew.
    minimum_period = max(10.0, cross_size * 1.15)
    maximum_period = cross_size * 1.85
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
                    break
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


def _negative_edge_bounds_small(raw: np.ndarray) -> list[Bounds]:
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


def _negative_edge_bounds(raw: np.ndarray) -> list[Bounds]:
    """Detect frames on a small view, avoiding a gigabyte-scale float copy."""

    step = max(1, ceil(max(raw.shape[:2]) / 2400))
    detection_view = np.ascontiguousarray(raw[::step, ::step, :])
    sampled_bounds = _negative_edge_bounds_small(detection_view)
    if len(sampled_bounds) > 6:
        return []
    height, width = raw.shape[:2]
    return [
        (
            min(width, left * step),
            min(height, top * step),
            min(width, right * step),
            min(height, bottom * step),
        )
        for left, top, right, bottom in sampled_bounds
    ]


def detect_frame_bounds(positive: np.ndarray) -> list[Bounds]:
    """Detect bright positive frames in either strip orientation."""

    return _projection_frame_bounds(positive)


def _is_vertical(bounds: list[Bounds], image: np.ndarray) -> bool:
    if len(bounds) >= 2:
        x_centers = [(left + right) / 2 for left, _, right, _ in bounds]
        y_centers = [(top + bottom) / 2 for _, top, _, bottom in bounds]
        return np.ptp(y_centers) >= np.ptp(x_centers)
    return image.shape[0] >= image.shape[1]


def _bright_film_sample(region: np.ndarray) -> np.ndarray:
    pixels = region.reshape(-1, 3).astype(np.float32)
    luminance = pixels.mean(axis=1)
    cutoff = np.percentile(luminance, 70)
    selected = pixels[luminance >= cutoff]
    if not selected.size:
        selected = pixels
    return np.median(selected, axis=0)


def _estimate_film_bases(
    raw: np.ndarray, bounds: list[Bounds], vertical: bool
) -> tuple[np.ndarray, np.ndarray]:
    left = min(bound[0] for bound in bounds)
    top = min(bound[1] for bound in bounds)
    right = max(bound[2] for bound in bounds)
    bottom = max(bound[3] for bound in bounds)
    cross_size = (right - left) if vertical else (bottom - top)
    cross_margin = max(1, round(cross_size * 0.08))
    separator_half_width = max(2, round(cross_size * 0.04))

    positions: list[float] = []
    bases: list[np.ndarray] = []
    for previous, following in pairwise(bounds):
        if vertical:
            position = round((previous[3] + following[1]) / 2)
            region = raw[
                max(0, position - separator_half_width) : position + separator_half_width,
                left + cross_margin : right - cross_margin,
            ]
        else:
            position = round((previous[2] + following[0]) / 2)
            region = raw[
                top + cross_margin : bottom - cross_margin,
                max(0, position - separator_half_width) : position + separator_half_width,
            ]
        if region.size:
            positions.append(float(position))
            bases.append(_bright_film_sample(region))

    if not bases:
        region = raw[top:bottom:4, left:right:4]
        base = _bright_film_sample(region)
        position = (top + bottom) / 2 if vertical else (left + right) / 2
        positions = [position]
        bases = [base]
    return np.asarray(positions, dtype=np.float32), np.asarray(bases, dtype=np.float32)


def _interpolated_base(
    coordinates: np.ndarray, positions: np.ndarray, bases: np.ndarray
) -> np.ndarray:
    if len(positions) == 1:
        return np.broadcast_to(bases[0], (len(coordinates), 3)).copy()
    result = np.empty((len(coordinates), 3), dtype=np.float32)
    for channel in range(3):
        result[:, channel] = np.interp(
            coordinates,
            positions,
            bases[:, channel],
            left=bases[0, channel],
            right=bases[-1, channel],
        )
    return result


def _inset_bounds(bounds: Bounds, ratio: float = 0.05) -> Bounds:
    left, top, right, bottom = bounds
    x_margin = max(1, round((right - left) * ratio))
    y_margin = max(1, round((bottom - top) * ratio))
    return left + x_margin, top + y_margin, right - x_margin, bottom - y_margin


def _density(
    raw: np.ndarray,
    bounds: Bounds,
    vertical: bool,
    positions: np.ndarray,
    bases: np.ndarray,
    step: int = 1,
) -> np.ndarray:
    left, top, right, bottom = bounds
    frame = raw[top:bottom:step, left:right:step].astype(np.float32)
    coordinates = (
        np.arange(top, bottom, step, dtype=np.float32)
        if vertical
        else np.arange(left, right, step, dtype=np.float32)
    )
    base = _interpolated_base(coordinates, positions, bases)
    base = base[:, None, :] if vertical else base[None, :, :]
    ratio = np.clip(frame / (base + 1e-6), 1e-4, 1.0)
    return -np.log10(ratio)


def _calibrate_film(raw: np.ndarray, bounds: list[Bounds]) -> _FilmCalibration:
    vertical = _is_vertical(bounds, raw)
    positions, bases = _estimate_film_bases(raw, bounds, vertical)
    density_samples: list[np.ndarray] = []
    for frame_bounds in bounds:
        core_bounds = _inset_bounds(frame_bounds)
        left, top, right, bottom = core_bounds
        step = max(1, ceil(max(right - left, bottom - top) / 700))
        density_samples.append(
            _density(raw, core_bounds, vertical, positions, bases, step).reshape(-1, 3)
        )
    core_density = np.concatenate(density_samples, axis=0)
    white_point = np.percentile(core_density, 99.7, axis=0).astype(np.float32)
    white_point = np.maximum(white_point, 1e-4)

    positive_samples = np.clip(core_density / white_point[None, :], 0.0, 1.0) ** 0.80
    luminance = positive_samples.mean(axis=1)
    chroma = positive_samples.max(axis=1) - positive_samples.min(axis=1)
    neutral = positive_samples[(luminance > 0.35) & (luminance < 0.90) & (chroma < 0.10)]
    if neutral.size:
        medians = np.median(neutral, axis=0)
        target = float(medians.mean())
        gains = np.clip(target / (medians + 1e-6), 0.85, 1.15).astype(np.float32)
    else:
        gains = np.ones(3, dtype=np.float32)
    return _FilmCalibration(vertical, positions, bases, white_point, gains)


def _convert_frame(
    raw: np.ndarray, bounds: Bounds, calibration: _FilmCalibration
) -> np.ndarray:
    density = _density(
        raw,
        bounds,
        calibration.vertical,
        calibration.base_positions,
        calibration.film_bases,
    )
    positive = np.clip(density / calibration.white_point[None, None, :], 0.0, 1.0)
    np.power(positive, 0.80, out=positive)
    positive *= calibration.neutral_gains[None, None, :]
    np.clip(positive, 0.0, 1.0, out=positive)
    result = np.round(positive * 65535.0).astype(np.uint16)
    if result.shape[0] > result.shape[1]:
        result = np.rot90(result, k=3)
    return result


def negative_to_positive(image: np.ndarray) -> np.ndarray:
    """Convert one negative frame using an estimated orange film base."""

    if not np.issubdtype(image.dtype, np.integer):
        raise ProcessingError("La conversion d'un négatif exige des pixels entiers.")
    bounds = [(0, 0, image.shape[1], image.shape[0])]
    calibration = _calibrate_film(image, bounds)
    return _convert_frame(image, bounds[0], calibration)


def process_negative_scan(raw_path: Path, output_paths: list[Path]) -> list[Path]:
    raw = read_image(raw_path)
    bounds = _negative_edge_bounds(raw)
    if not bounds:
        raise ProcessingError(
            "Aucun cadre de négatif n'a été détecté avec assez de certitude. "
            "Le scan brut intact a été conservé."
        )
    if len(output_paths) < len(bounds):
        raise ProcessingError("Le nombre de chemins de sortie réservés est insuffisant.")

    calibration = _calibrate_film(raw, bounds)
    written: list[Path] = []
    for frame_bounds, output_path in zip(bounds, output_paths, strict=False):
        write_image(output_path, _convert_frame(raw, frame_bounds, calibration))
        written.append(output_path)
    return written
