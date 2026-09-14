from pathlib import Path

from scanner_utils.config import DEFAULT_CONFIG
from scanner_utils.scanner import (
    ScannerCapabilities,
    ScannerDevice,
    build_scan_arguments,
    parse_devices,
)


def test_parse_devices_handles_sane_quotes_and_virtual_suffix() -> None:
    output = """device `epson2:libusb:001:009' is a Epson GT-8200 flatbed scanner
device `v4l:/dev/video0' is a Noname Camera virtual device
"""

    devices = parse_devices(output)

    assert [device.name for device in devices] == ["epson2:libusb:001:009", "v4l:/dev/video0"]
    assert devices[0].description == "Epson GT-8200 flatbed scanner"


def test_build_arguments_omits_options_not_advertised() -> None:
    capabilities = ScannerCapabilities("", frozenset({"--source", "--mode", "--resolution"}))
    arguments = build_scan_arguments(
        ScannerDevice("test:0", "Test scanner"),
        capabilities,
        DEFAULT_CONFIG.negative,
        Path("raw.tiff"),
    )

    assert arguments[:2] == ["-d", "test:0"]
    assert "--source" in arguments
    assert "Transparency Unit" in arguments
    assert "--depth" not in arguments
    assert "--format=tiff" in arguments
