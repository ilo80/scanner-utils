from datetime import datetime
from pathlib import Path

from scanner_utils.naming import OutputNamer


def test_namer_avoids_existing_sequence(tmp_path: Path) -> None:
    (tmp_path / "2026-09-14_negative_001_01.tiff").touch()
    (tmp_path / "2026-09-14_negative_004_02.png").touch()
    namer = OutputNamer(tmp_path, "negative", "tiff", datetime(2026, 9, 14, 12, 0))

    assert namer.next_sequence() == 5
    assert namer.frame_path(5, 2).name == "2026-09-14_negative_005_02.tiff"
