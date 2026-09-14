from pathlib import Path

from scanner_utils.config import DEFAULT_CONFIG, load_config


def test_default_resolution_and_depth_are_1600_dpi_and_16_bit() -> None:
    assert DEFAULT_CONFIG.photo.resolution == 1600
    assert DEFAULT_CONFIG.negative.resolution == 1600
    assert DEFAULT_CONFIG.photo.depth == 16
    assert DEFAULT_CONFIG.negative.depth == 16


def test_load_config_uses_profile_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[general]
output_directory = "~/custom-scans"
output_format = "png"
keep_raw_scans = true

[negative]
resolution = 2400
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.output_directory == Path("~/custom-scans").expanduser()
    assert config.output_format == "png"
    assert config.keep_raw_scans is True
    assert config.negative.resolution == 2400
    assert config.negative.depth == 16
