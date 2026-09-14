"""SANE/scanimage integration."""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scanner_utils.config import ScanProfile
from scanner_utils.errors import ScanFailedError, ScannerUnavailableError

DEVICE_PATTERN = re.compile(r"device [`'](?P<name>.+?)[`'] is a (?P<description>.+)$")
OPTION_PATTERN = re.compile(r"^\s+(?P<option>--[\w-]+|-[ltxy])(?:\s|\[|$)")
PERCENTAGE_PATTERN = re.compile(rb"(?P<percentage>\d+(?:[.,]\d+)?)\s*%")


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

    def numeric_values(self, option: str) -> tuple[int, ...]:
        """Return the discrete numeric values advertised for an option."""

        for line in self.raw_output.splitlines():
            stripped = line.strip()
            if not stripped.startswith(f"{option} "):
                continue
            specification = stripped[len(option) :].split(" [", maxsplit=1)[0]
            if ".." in specification:
                return ()
            return tuple(dict.fromkeys(int(value) for value in re.findall(r"\d+", specification)))
        return ()


@dataclass(frozen=True)
class ScanProgress:
    phase: str
    elapsed_seconds: float
    percentage: float | None


ProgressCallback = Callable[[ScanProgress], None]


def _scanimage_executable() -> str:
    executable = shutil.which("scanimage")
    if executable is None:
        raise ScannerUnavailableError(
            "scanimage n'est pas installé. Installez les utilitaires SANE de votre distribution."
        )
    return executable


def _run_scanimage(
    arguments: list[str], timeout: float | None = 30
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [_scanimage_executable(), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScannerUnavailableError("La détection des scanners SANE a expiré.") from exc
    except OSError as exc:
        raise ScannerUnavailableError(f"Impossible d'exécuter scanimage : {exc}") from exc


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
        detail = result.stderr.strip() or f"code de sortie {result.returncode}"
        raise ScannerUnavailableError(f"La détection des scanners a échoué : {detail}")
    return parse_devices(result.stdout)


def inspect_capabilities(device: ScannerDevice) -> ScannerCapabilities:
    result = _run_scanimage(["-d", device.name, "--all-options"])
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise ScannerUnavailableError(
            f"Impossible de lire les options de {device.description} : {combined.strip()}"
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
    arguments.extend(["--progress", "--format=tiff", f"--output-file={output_path}"])
    return arguments


def _progress_from_stderr(stderr: bytes) -> tuple[str, float | None]:
    percentages = list(PERCENTAGE_PATTERN.finditer(stderr))
    percentage = None
    if percentages:
        percentage = float(percentages[-1]["percentage"].replace(b",", b"."))
    lowered = stderr.lower()
    if b"warm" in lowered or b"lamp" in lowered:
        phase = "Préchauffage de la lampe"
    elif percentage is not None and percentage >= 100:
        phase = "Finalisation du fichier"
    elif percentage is not None and percentage > 0:
        phase = "Numérisation"
    else:
        phase = "Préparation / préchauffage du scanner"
    return phase, percentage


def _run_acquisition(arguments: list[str], callback: ProgressCallback | None) -> tuple[int, str]:
    try:
        process = subprocess.Popen(
            [_scanimage_executable(), *arguments],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ScannerUnavailableError(f"Impossible d'exécuter scanimage : {exc}") from exc

    started_at = time.monotonic()
    stderr_chunks = bytearray()
    phase = "Préparation / préchauffage du scanner"
    percentage: float | None = None
    assert process.stderr is not None
    descriptor = process.stderr.fileno()
    os.set_blocking(descriptor, False)
    try:
        while process.poll() is None:
            readable, _, _ = select.select([descriptor], [], [], 0.2)
            if readable:
                chunk = os.read(descriptor, 65536)
                if chunk:
                    stderr_chunks.extend(chunk)
                    phase, percentage = _progress_from_stderr(bytes(stderr_chunks[-131072:]))
            if callback:
                callback(ScanProgress(phase, time.monotonic() - started_at, percentage))
        try:
            while chunk := os.read(descriptor, 65536):
                stderr_chunks.extend(chunk)
        except BlockingIOError:
            pass
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise
    finally:
        process.stderr.close()

    if callback:
        final_percentage = 100.0 if process.returncode == 0 else percentage
        callback(
            ScanProgress(
                "Numérisation terminée", time.monotonic() - started_at, final_percentage
            )
        )
    return process.returncode, bytes(stderr_chunks).decode(errors="replace")


def acquire_scan(
    device: ScannerDevice,
    capabilities: ScannerCapabilities,
    profile: ScanProfile,
    output_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScanFailedError(
            f"Impossible de créer le dossier des scans bruts {output_path.parent} : {exc}"
        ) from exc
    arguments = build_scan_arguments(device, capabilities, profile, output_path)
    returncode, stderr = _run_acquisition(arguments, progress_callback)
    if returncode != 0:
        detail = stderr.replace("\r", "\n").strip() or f"code de sortie {returncode}"
        raise ScanFailedError(f"La numérisation a échoué : {detail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ScanFailedError("scanimage n'a produit aucune image malgré un retour sans erreur.")
    return output_path
