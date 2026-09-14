"""Interactive command-line interface."""

from __future__ import annotations

import argparse
import sys
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from scanner_utils.config import AppConfig, ScanProfile, load_config
from scanner_utils.errors import ProcessingError, ScannerUtilsError
from scanner_utils.naming import OutputNamer
from scanner_utils.processing import process_negative_scan, process_photo_scan
from scanner_utils.scanner import (
    ScannerCapabilities,
    ScannerDevice,
    acquire_scan,
    discover_devices,
    inspect_capabilities,
)


def _choice(prompt: str, choices: set[str], default: str | None = None) -> str:
    while True:
        value = input(prompt).strip() or default
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(sorted(choices))}.")


def _select_device(devices: list[ScannerDevice], requested: str | None) -> ScannerDevice:
    if requested:
        for device in devices:
            if device.name == requested:
                return device
        raise ScannerUtilsError(f"Requested scanner is not available: {requested}")
    if len(devices) == 1:
        print(f"Using scanner: {devices[0].description}")
        return devices[0]
    print("\nSelect scanner:")
    for index, device in enumerate(devices, start=1):
        print(f"[{index}] {device.description} ({device.name})")
    selected = _choice("> ", {str(index) for index in range(1, len(devices) + 1)})
    return devices[int(selected) - 1]


def _output_format(default: str) -> str:
    print("\nOutput format:")
    print(f"[1] TIFF{' (default)' if default == 'tiff' else ''}")
    print(f"[2] PNG{' (default)' if default == 'png' else ''}")
    default_choice = "1" if default == "tiff" else "2"
    return {"1": "tiff", "2": "png"}[_choice("> ", {"1", "2"}, default_choice)]


def _preparation_message(scan_type: str) -> str:
    if scan_type == "negative":
        return (
            "Install and connect the Transparency Unit, position the negative strip in the "
            "holder, and ensure that the transparency area is unobstructed."
        )
    return (
        "Use the reflective white cover, remove the film holder if necessary, and place the "
        "photo(s) apart from one another on the clean scanner bed."
    )


def _warn_unsupported(capabilities: ScannerCapabilities, profile: ScanProfile) -> None:
    required = {"--source", "--mode", "--resolution"}
    missing = sorted(required - capabilities.options)
    if missing:
        raise ScannerUtilsError(
            "The selected backend lacks required options: " + ", ".join(missing)
        )
    if profile.depth > 8 and "--depth" not in capabilities.options:
        print("Warning: this backend does not expose bit depth; its default will be used.")


def _scan_once(
    scan_type: str,
    device: ScannerDevice,
    capabilities: ScannerCapabilities,
    config: AppConfig,
) -> None:
    profile = config.negative if scan_type == "negative" else config.photo
    _warn_unsupported(capabilities, profile)
    print(f"\n{_preparation_message(scan_type)}")
    input("Press Enter when ready (or Ctrl+C to cancel). ")
    output_format = _output_format(config.output_format)

    try:
        config.output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScannerUtilsError(
            f"Cannot create output directory {config.output_directory}: {exc}"
        ) from exc
    namer = OutputNamer(config.output_directory, scan_type, output_format, datetime.now())
    sequence = namer.next_sequence()
    raw_path = namer.raw_path(sequence)

    print(f"\nScanning at {profile.resolution} DPI / {profile.depth}-bit...")
    acquire_scan(device, capabilities, profile, raw_path)
    print("Detecting and processing images...")
    paths = [namer.frame_path(sequence, frame) for frame in range(1, 100)]
    try:
        if scan_type == "negative":
            written = process_negative_scan(raw_path, paths)
        else:
            written = process_photo_scan(raw_path, paths)
    except ProcessingError:
        print(f"Raw scan preserved at: {raw_path}")
        raise

    if config.keep_raw_scans:
        print(f"Raw scan kept at: {raw_path}")
    else:
        raw_path.unlink(missing_ok=True)
        with suppress(OSError):
            raw_path.parent.rmdir()
    print(f"\n{len(written)} image(s) detected and saved:")
    for path in written:
        print(f"  {path}")


def run(config: AppConfig, requested_device: str | None = None) -> None:
    print("Scanner Utils\n")
    devices = discover_devices()
    if not devices:
        raise ScannerUtilsError(
            "No scanner detected. Check its power, USB connection, and SANE permissions."
        )
    device = _select_device(devices, requested_device)
    print("Inspecting scanner capabilities...")
    capabilities = inspect_capabilities(device)

    while True:
        print("\nWhat do you want to do?")
        print("[1] Scan photo(s)")
        print("[2] Scan negative(s)")
        print("[3] Quit")
        action = _choice("> ", {"1", "2", "3"})
        if action == "3":
            print("Goodbye.")
            return
        try:
            _scan_once("photo" if action == "1" else "negative", device, capabilities, config)
        except ScannerUtilsError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive photo and negative scanning")
    parser.add_argument("--config", type=Path, help="path to a TOML configuration file")
    parser.add_argument("--device", help="SANE device name (skips device selection)")
    parser.add_argument("--output-dir", type=Path, help="override the output directory")
    parser.add_argument(
        "--keep-raw", action="store_true", help="keep raw TIFF scans after successful processing"
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        config = load_config(arguments.config)
        if arguments.output_dir:
            config = replace(config, output_directory=arguments.output_dir.expanduser())
        if arguments.keep_raw:
            config = replace(config, keep_raw_scans=True)
        run(config, arguments.device)
    except KeyboardInterrupt:
        print("\nCancelled.")
    except ScannerUtilsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
