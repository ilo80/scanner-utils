from scanner_utils.cli import _select_device
from scanner_utils.scanner import ScannerDevice


def test_epson_is_the_default_scanner(monkeypatch) -> None:
    devices = [
        ScannerDevice("v4l:/dev/video0", "HD Camera"),
        ScannerDevice("epson2:libusb:001:009", "Epson GT-8200 flatbed scanner"),
    ]
    monkeypatch.setattr("builtins.input", lambda _: "")

    selected = _select_device(devices, requested=None)

    assert selected == devices[1]
