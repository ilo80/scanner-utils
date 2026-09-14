from pathlib import Path

import cv2
import numpy as np
import tifffile

from scanner_utils.processing.detection import detect_rectangular_regions, perspective_crop
from scanner_utils.processing.negatives import (
    _negative_edge_bounds,
    _regular_edge_chain,
    detect_frame_bounds,
    negative_to_positive,
)
from scanner_utils.processing.photos import process_photo_scan


def test_detects_multiple_photos_and_deskews() -> None:
    image = np.full((700, 1000, 3), 1500, dtype=np.uint16)
    first = np.array([[80, 80], [420, 60], [440, 310], [100, 330]], dtype=np.int32)
    second = np.array([[560, 370], [900, 390], [880, 640], [540, 620]], dtype=np.int32)
    cv2.fillConvexPoly(image, first, (50000, 20000, 10000))
    cv2.fillConvexPoly(image, second, (10000, 35000, 55000))

    regions = detect_rectangular_regions(image)

    assert len(regions) == 2
    crops = [perspective_crop(image, region.corners) for region in regions]
    assert all(crop.shape[0] > 200 and crop.shape[1] > 300 for crop in crops)


def test_photo_pipeline_preserves_16_bit_output(tmp_path: Path) -> None:
    image = np.full((400, 600, 3), 1000, dtype=np.uint16)
    image[80:320, 100:500] = (50000, 30000, 10000)
    raw_path = tmp_path / "raw.tiff"
    output_path = tmp_path / "photo.tiff"
    tifffile.imwrite(raw_path, image, photometric="rgb")

    written = process_photo_scan(raw_path, [output_path])

    assert written == [output_path]
    assert tifffile.imread(output_path).dtype == np.uint16


def test_negative_conversion_preserves_depth_and_inverts() -> None:
    ramp = np.linspace(1000, 64000, 300, dtype=np.uint16)
    image = np.dstack([np.tile(ramp, (80, 1))] * 3)

    positive = negative_to_positive(image)

    assert positive.dtype == np.uint16
    assert positive[:, 10].mean() > positive[:, -10].mean()


def test_detects_negative_frames_from_dark_separators() -> None:
    image = np.full((240, 900, 3), 1200, dtype=np.uint16)
    for left, right, value in [(40, 280, 35000), (330, 570, 45000), (620, 860, 55000)]:
        image[20:220, left:right] = value
        image[50:190, left + 30 : right - 30] += 5000

    bounds = detect_frame_bounds(image)

    assert len(bounds) == 3


def test_detects_vertical_frames_from_raw_film_edges() -> None:
    raw = np.full((920, 240, 3), 2500, dtype=np.uint16)
    raw[:, 55:205] = (42000, 25000, 19000)
    raw[:110, 55:205] = 2000
    raw[810:, 55:205] = 65000
    for boundary in (110, 285, 460, 635, 810):
        raw[max(0, boundary - 2) : boundary + 2, 55:205] = 62000
    rng = np.random.default_rng(42)
    raw[112:808, 57:203] = np.clip(
        raw[112:808, 57:203].astype(np.int32)
        + rng.integers(-8000, 8000, size=(696, 146, 3)),
        0,
        65535,
    ).astype(np.uint16)

    bounds = _negative_edge_bounds(raw)

    assert len(bounds) == 4
    assert all(right - left > 130 for left, _, right, _ in bounds)
    assert all(bottom - top > 160 for _, top, _, bottom in bounds)


def test_weak_separator_is_kept_in_regular_frame_chain() -> None:
    profile = np.zeros(500, dtype=np.float32)
    profile[[20, 120, 220, 320, 420]] = [1.0, 0.8, 0.05, 0.7, 0.9]

    assert _regular_edge_chain(profile, cross_size=70) == [20, 120, 220, 320, 420]


def test_missing_separator_is_not_bridged() -> None:
    profile = np.zeros(500, dtype=np.float32)
    profile[[20, 120, 220, 420]] = [1.0, 0.8, 0.7, 0.9]

    assert _regular_edge_chain(profile, cross_size=70) == [20, 120, 220]
