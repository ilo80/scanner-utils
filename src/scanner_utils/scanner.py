"""SANE/scanimage integration."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scanner_utils.config import ScanProfile
from scanner_utils.errors import ScanFailedError, ScannerUnavailableError

DEVICE_PATTERN = re.compile(r"device [`'](?P<name>.+?)[`'] is a (?P<description>.+)$")
OPTION_PATTERN = re.compile(r"^\s+(?P<option>--[\w-]+|-[ltxy])(?:\s|\[|$)")


@dataclass(frozen=True)
class ScannerDevice:
    name: str
    description: str


@dataclass(frozen=True)
class ScannerCapabilities:
    raw_output: str
    options: frozenset[str]

    def supports(self, option: str) -> bool:
        return option in self.options


def _run_scanimage(
    arguments: list[str], timeout: float | None = 30
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("scanimage")
    if executable is None:
        raise ScannerUnavailableError(
            "scanimage is not installed. Install the SANE utilities for your distribution."
        )
    try:
        return subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScannerUnavailableError("SANE device discovery timed out.") from exc
    except OSError as exc:
        raise ScannerUnavailableError(f"Cannot run scanimage: {exc}") from exc


def parse_devices(output: str) -> list[ScannerDevice]:
    devices: list[ScannerDevice] = []
    for line in output.splitlines():
        match = DEVICE_PATTERN.match(line.strip())
        if match:
            description = match["description"].removesuffix(" virtual device")
            devices.append(ScannerDevice(match["name"], description))
    return devices


def discover_devices() -> list[ScannerDevice]:
    result = _run_scanimage(["-L"])
    if result.returncode != 0:
        detail = result.stderr.strip() or f"scanimage exited with status {result.returncode}"
        raise ScannerUnavailableError(f"Scanner discovery failed: {detail}")
    return parse_devices(result.stdout)


def inspect_capabilities(device: ScannerDevice) -> ScannerCapabilities:
    result = _run_scanimage(["-d", device.name, "--all-options"])
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise ScannerUnavailableError(
            f"Cannot inspect scanner options for {device.description}: {combined.strip()}"
        )
    options = frozenset(
        match["option"]
        for line in combined.splitlines()
        if (match := OPTION_PATTERN.match(line)) is not None
    )
    return ScannerCapabilities(raw_output=combined, options=options)


def build_scan_arguments(
    device: ScannerDevice,
    capabilities: ScannerCapabilities,
    profile: ScanProfile,
    output_path: Path,
) -> list[str]:
    """Build a scan command using only options advertised by the backend."""

    requested: list[tuple[str, str]] = [
        ("--source", profile.source),
        ("--mode", "Color"),
        ("--depth", str(profile.depth)),
        ("--resolution", str(profile.resolution)),
        ("--brightness", "0"),
        ("--sharpness", "0"),
        ("--gamma-correction", "Default"),
        ("--color-correction", profile.color_correction),
    ]
    arguments = ["-d", device.name]
    for option, value in requested:
        if capabilities.supports(option):
            arguments.extend([option, value])
    arguments.extend(["--format=tiff", f"--output-file={output_path}"])
    return arguments


def acquire_scan(
    device: ScannerDevice,
    capabilities: ScannerCapabilities,
    profile: ScanProfile,
    output_path: Path,
) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScanFailedError(
            f"Cannot create raw scan directory {output_path.parent}: {exc}"
        ) from exc
    arguments = build_scan_arguments(device, capabilities, profile, output_path)
    result = _run_scanimage(arguments, timeout=None)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"scanimage exited with status {result.returncode}"
        raise ScanFailedError(f"Scan failed: {detail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ScanFailedError("scanimage reported success but produced no image.")
    return output_path
