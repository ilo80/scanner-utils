"""Positive photograph processing."""

from __future__ import annotations

from pathlib import Path

from scanner_utils.errors import ProcessingError
from scanner_utils.processing.detection import detect_rectangular_regions, perspective_crop
from scanner_utils.processing.io import read_image, write_image


def process_photo_scan(raw_path: Path, output_paths: list[Path]) -> list[Path]:
    image = read_image(raw_path)
    regions = detect_rectangular_regions(image)
    if not regions:
        raise ProcessingError(
            "No photograph was detected. The raw scan was kept; try increasing contrast "
            "between the photos and the scanner background."
        )
    if len(output_paths) < len(regions):
        raise ProcessingError("Not enough output paths were reserved for detected photographs.")

    written: list[Path] = []
    for region, output_path in zip(regions, output_paths, strict=False):
        write_image(output_path, perspective_crop(image, region.corners))
        written.append(output_path)
    return written
