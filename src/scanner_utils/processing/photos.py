"""Positive photograph processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scanner_utils.errors import ProcessingError
from scanner_utils.processing.detection import detect_rectangular_regions, perspective_crop
from scanner_utils.processing.io import read_image, write_image


def _white_border_depth(profile: np.ndarray, maximum_depth: int) -> int:
    """Return the depth of a thin white edge that clears inside the photo."""

    limited = profile[:maximum_depth]
    if limited.size == 0 or float(limited[0]) < 0.10:
        return 0
    clear_rows = np.flatnonzero(limited <= 0.02)
    if clear_rows.size == 0:
        return 0
    return int(clear_rows[0])


def _trim_white_scanner_border(image: np.ndarray) -> np.ndarray:
    """Remove thin neutral-white scanner-bed wedges left after perspective crop."""

    if image.ndim != 3 or image.shape[2] != 3:
        return image
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
    else:
        maximum = max(1.0, float(np.nanmax(image)))
    normalized = image.astype(np.float32) / maximum
    near_white = (normalized.min(axis=2) >= 0.94) & (np.ptp(normalized, axis=2) <= 0.06)

    height, width = image.shape[:2]
    maximum_y = max(2, round(height * 0.025))
    maximum_x = max(2, round(width * 0.025))
    row_profile = near_white.mean(axis=1)
    top = _white_border_depth(row_profile, maximum_y)
    bottom = _white_border_depth(row_profile[::-1], maximum_y)
    vertical_core = near_white[top : height - bottom]
    column_profile = vertical_core.mean(axis=0)
    left = _white_border_depth(column_profile, maximum_x)
    right = _white_border_depth(column_profile[::-1], maximum_x)
    if top == bottom == left == right == 0:
        return image
    if top + bottom >= height or left + right >= width:
        return image
    return np.ascontiguousarray(image[top : height - bottom, left : width - right])


def process_photo_scan(raw_path: Path, output_paths: list[Path]) -> list[Path]:
    image = read_image(raw_path)
    regions = detect_rectangular_regions(image)
    if not regions:
        raise ProcessingError(
            "Aucune photo n'a été détectée. Le scan brut a été conservé ; essayez d'augmenter "
            "le contraste entre les photos et le fond du scanner."
        )
    if len(output_paths) < len(regions):
        raise ProcessingError("Le nombre de chemins de sortie réservés est insuffisant.")

    written: list[Path] = []
    for region, output_path in zip(regions, output_paths, strict=False):
        crop = perspective_crop(image, region.corners)
        write_image(output_path, _trim_white_scanner_border(crop))
        written.append(output_path)
    return written
