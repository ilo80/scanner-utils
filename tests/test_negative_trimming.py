import cv2
import numpy as np

from scanner_utils.processing.negative_trimming import (
    _estimate_border_skew,
    _trim_35mm_frame,
)


def test_trims_scanner_and_film_borders_with_conservative_inset() -> None:
    image = np.full((180, 280, 3), 60000, dtype=np.uint16)
    image[8:172, 12:268] = 1200

    y, x = np.mgrid[0:146, 0:219]
    base = (15000 + (x / 218) * 25000 + (y / 145) * 8000).astype(np.uint16)
    content = np.stack(
        [
            base,
            np.clip(base.astype(np.int32) + 4000, 0, 65535).astype(np.uint16),
            np.clip(base.astype(np.int32) + 8000, 0, 65535).astype(np.uint16),
        ],
        axis=2,
    )
    image[20:166, 30:249] = content

    trimmed = _trim_35mm_frame(image)

    assert trimmed.shape == (144, 217, 3)
    assert np.array_equal(trimmed, content[1:-1, 1:-1])


def test_leaves_clean_frame_untouched_without_film_border_cue() -> None:
    y, x = np.mgrid[0:150, 0:225]
    base = (10000 + (x / 224) * 30000 + (y / 149) * 10000).astype(np.uint16)
    image = np.stack([base, base, base], axis=2)

    trimmed = _trim_35mm_frame(image)

    assert trimmed.shape == image.shape
    assert np.array_equal(trimmed, image)


def test_estimates_and_corrects_small_frame_rotation() -> None:
    image = np.full((720, 1120, 3), 60000, dtype=np.uint16)
    image[32:688, 48:1072] = 1200
    y, x = np.mgrid[0:584, 0:876]
    base = (15000 + (x / 875) * 25000 + (y / 583) * 8000).astype(np.uint16)
    image[70:654, 122:998] = np.stack([base, base + 2000, base + 4000], axis=2)
    transform = cv2.getRotationMatrix2D((560, 360), 0.6, 1.0)
    rotated = cv2.warpAffine(
        image,
        transform,
        (1120, 720),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    angle = _estimate_border_skew(rotated)
    trimmed = _trim_35mm_frame(rotated)

    assert angle is not None
    assert abs(angle + 0.6) < 0.15
    assert trimmed.shape == (580, 868, 3)
