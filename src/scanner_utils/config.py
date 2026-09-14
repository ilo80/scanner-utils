"""Configuration models and TOML loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scanner_utils.errors import ScannerUtilsError


@dataclass(frozen=True)
class ScanProfile:
    resolution: int
    depth: int
    source: str
    color_correction: str


@dataclass(frozen=True)
class AppConfig:
    output_directory: Path
    output_format: str
    keep_raw_scans: bool
    photo: ScanProfile
    negative: ScanProfile


DEFAULT_CONFIG = AppConfig(
    output_directory=Path("~/Pictures/Scans").expanduser(),
    output_format="tiff",
    keep_raw_scans=False,
    photo=ScanProfile(
        resolution=1600,
        depth=16,
        source="Flatbed",
        color_correction="Built in CCT profile",
    ),
    negative=ScanProfile(
        resolution=1600,
        depth=16,
        source="Transparency Unit",
        color_correction="None",
    ),
)


def _profile_from_table(default: ScanProfile, table: dict[str, Any]) -> ScanProfile:
    known = {field: table[field] for field in default.__dataclass_fields__ if field in table}
    return replace(default, **known)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration, falling back to documented defaults.

    The default user configuration path is ``~/.config/scanner-utils/config.toml``.
    An explicitly supplied missing path is treated as an error to catch typos.
    """

    explicit = path is not None
    config_path = path or Path("~/.config/scanner-utils/config.toml").expanduser()
    if not config_path.exists():
        if explicit:
            raise ScannerUtilsError(f"Le fichier de configuration n'existe pas : {config_path}")
        return DEFAULT_CONFIG

    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ScannerUtilsError(
            f"Impossible de lire la configuration {config_path} : {exc}"
        ) from exc

    general = data.get("general", {})
    output_format = str(general.get("output_format", DEFAULT_CONFIG.output_format)).lower()
    if output_format not in {"tiff", "png"}:
        raise ScannerUtilsError("general.output_format doit valoir 'tiff' ou 'png'")

    return AppConfig(
        output_directory=Path(
            general.get("output_directory", str(DEFAULT_CONFIG.output_directory))
        ).expanduser(),
        output_format=output_format,
        keep_raw_scans=bool(general.get("keep_raw_scans", DEFAULT_CONFIG.keep_raw_scans)),
        photo=_profile_from_table(DEFAULT_CONFIG.photo, data.get("photo", {})),
        negative=_profile_from_table(DEFAULT_CONFIG.negative, data.get("negative", {})),
    )
