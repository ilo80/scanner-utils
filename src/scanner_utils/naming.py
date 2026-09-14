"""Collision-safe output naming."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class OutputNamer:
    directory: Path
    scan_type: str
    extension: str
    now: datetime

    def next_sequence(self) -> int:
        """Return the next scan sequence for this date and scan type."""

        prefix = f"{self.now:%Y-%m-%d}_{self.scan_type}_"
        sequences: list[int] = []
        if self.directory.exists():
            for path in self.directory.glob(f"{prefix}*.*"):
                suffix = path.stem.removeprefix(prefix)
                sequence_text = suffix.split("_", 1)[0]
                if sequence_text.isdigit():
                    sequences.append(int(sequence_text))
        return max(sequences, default=0) + 1

    def frame_path(self, sequence: int, frame: int) -> Path:
        return self.directory / (
            f"{self.now:%Y-%m-%d}_{self.scan_type}_{sequence:03d}_{frame:02d}"
            f".{self.extension}"
        )

    def raw_path(self, sequence: int) -> Path:
        return self.directory / ".raw" / (
            f"{self.now:%Y-%m-%d}_{self.scan_type}_{sequence:03d}_raw.tiff"
        )
