"""Lossless image I/O helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from scanner_utils.errors import ProcessingError


def read_image(path: Path) -> np.ndarray:
    try:
        image = tifffile.imread(path)
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise ProcessingError(f"Cannot read raw scan {path}: {exc}") from exc
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ProcessingError(f"Expected a color image, got shape {image.shape!r}.")
    return np.ascontiguousarray(image[:, :, :3])


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            tifffile.imwrite(path, image, photometric="rgb", compression="deflate")
        elif path.suffix.lower() == ".png":
            success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not success:
                raise OSError("OpenCV could not encode the PNG file")
        else:
            raise ProcessingError(f"Unsupported output format: {path.suffix}")
    except (OSError, ValueError) as exc:
        raise ProcessingError(f"Cannot write processed image {path}: {exc}") from exc
