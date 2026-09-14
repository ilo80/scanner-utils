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
    ScanProgress,
    acquire_scan,
    discover_devices,
    inspect_capabilities,
)


def _choice(prompt: str, choices: set[str], default: str | None = None) -> str:
    while True:
        value = input(prompt).strip() or default
        if value in choices:
            return value
        print(f"Choisissez l'une des valeurs suivantes : {', '.join(sorted(choices))}.")


def _select_device(devices: list[ScannerDevice], requested: str | None) -> ScannerDevice:
    if requested:
        for device in devices:
            if device.name == requested:
                return device
        raise ScannerUtilsError(f"Le scanner demandé n'est pas disponible : {requested}")
    if len(devices) == 1:
        print(f"Scanner utilisé : {devices[0].description}")
        return devices[0]

    default_index = next(
        (
            index
            for index, device in enumerate(devices, start=1)
            if "epson" in f"{device.name} {device.description}".casefold()
        ),
        1,
    )
    print("\nSélectionnez le scanner :")
    for index, device in enumerate(devices, start=1):
        default_label = " (par défaut)" if index == default_index else ""
        print(f"[{index}] {device.description} ({device.name}){default_label}")
    choices = {str(index) for index in range(1, len(devices) + 1)}
    selected = _choice("> ", choices, str(default_index))
    return devices[int(selected) - 1]


def _numeric_setting(label: str, supported: tuple[int, ...], default: int) -> int:
    if supported:
        print(f"\n{label} prises en charge : {', '.join(map(str, supported))}")
        if default not in supported:
            default = min(supported, key=lambda value: abs(value - default))
        choices = {str(value) for value in supported}
    else:
        print(f"\n{label} : la liste des valeurs prises en charge n'est pas disponible.")
        choices = set()

    while True:
        value = input(f"{label} [{default}] : ").strip() or str(default)
        if value.isdigit() and (not choices or value in choices):
            return int(value)
        if choices:
            print(f"Valeur invalide. Choisissez parmi : {', '.join(map(str, supported))}.")
        else:
            print("Saisissez un nombre entier positif.")


def _scan_profile(
    profile: ScanProfile, capabilities: ScannerCapabilities
) -> ScanProfile:
    resolution = _numeric_setting(
        "Résolution en DPI", capabilities.numeric_values("--resolution"), profile.resolution
    )
    if capabilities.supports("--depth"):
        depth = _numeric_setting(
            "Profondeur en bits", capabilities.numeric_values("--depth"), profile.depth
        )
    else:
        depth = profile.depth
        print(
            "Le backend ne permet pas de choisir la profondeur ; "
            "sa valeur par défaut sera utilisée."
        )
    return replace(profile, resolution=resolution, depth=depth)


def _output_format(default: str) -> str:
    print("\nFormat de sortie :")
    print(f"[1] TIFF{' (par défaut)' if default == 'tiff' else ''}")
    print(f"[2] PNG{' (par défaut)' if default == 'png' else ''}")
    default_choice = "1" if default == "tiff" else "2"
    return {"1": "tiff", "2": "png"}[_choice("> ", {"1", "2"}, default_choice)]


def _preparation_message(scan_type: str) -> str:
    if scan_type == "negative":
        return (
            "Installez et branchez l'unité de transparence, placez correctement la bande de "
            "négatifs dans le porte-film et vérifiez que la zone transparente est dégagée."
        )
    return (
        "Utilisez le cache blanc réfléchissant, retirez le porte-film si nécessaire et espacez "
        "les photos sur la vitre propre du scanner."
    )


def _warn_unsupported(capabilities: ScannerCapabilities, profile: ScanProfile) -> None:
    required = {"--source", "--mode", "--resolution"}
    missing = sorted(required - capabilities.options)
    if missing:
        raise ScannerUtilsError(
            "Le backend sélectionné ne fournit pas les options indispensables : "
            + ", ".join(missing)
        )
    if profile.depth > 8 and "--depth" not in capabilities.options:
        print("Attention : la profondeur demandée n'est pas exposée par ce backend.")


def _format_duration(seconds: float) -> str:
    total = round(seconds)
    minutes, remaining_seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


class _ProgressDisplay:
    def __init__(self) -> None:
        self._last_line_length = 0

    def update(self, progress: ScanProgress) -> None:
        percentage = (
            f"{progress.percentage:5.1f} %"
            if progress.percentage is not None
            else "avancement indisponible"
        )
        line = f"⏱ {_format_duration(progress.elapsed_seconds)} | {progress.phase} | {percentage}"
        padding = " " * max(0, self._last_line_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self._last_line_length = len(line)

    def finish(self) -> None:
        if self._last_line_length:
            print()


class _FrenchArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        translations = {
            "unrecognized arguments:": "arguments inconnus :",
            "the following arguments are required:": "arguments obligatoires manquants :",
            "expected one argument": "une valeur est attendue",
        }
        for source, translation in translations.items():
            message = message.replace(source, translation)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog} : erreur : {message}\n")


def _scan_once(
    scan_type: str,
    device: ScannerDevice,
    capabilities: ScannerCapabilities,
    config: AppConfig,
) -> None:
    profile = config.negative if scan_type == "negative" else config.photo
    _warn_unsupported(capabilities, profile)
    print(f"\n{_preparation_message(scan_type)}")
    profile = _scan_profile(profile, capabilities)
    output_format = _output_format(config.output_format)
    input("\nAppuyez sur Entrée lorsque tout est prêt (Ctrl+C pour annuler). ")

    try:
        config.output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScannerUtilsError(
            f"Impossible de créer le dossier de sortie {config.output_directory} : {exc}"
        ) from exc
    namer = OutputNamer(config.output_directory, scan_type, output_format, datetime.now())
    sequence = namer.next_sequence()
    raw_path = namer.raw_path(sequence)

    print(f"\nNumérisation à {profile.resolution} DPI et {profile.depth} bits...")
    progress = _ProgressDisplay()
    try:
        acquire_scan(device, capabilities, profile, raw_path, progress.update)
    finally:
        progress.finish()

    print("Analyse et traitement des images...")
    paths = [namer.frame_path(sequence, frame) for frame in range(1, 100)]
    try:
        if scan_type == "negative":
            written = process_negative_scan(raw_path, paths)
        else:
            written = process_photo_scan(raw_path, paths)
    except ProcessingError:
        print(f"Le scan brut a été conservé ici : {raw_path}")
        raise

    if config.keep_raw_scans:
        print(f"Scan brut conservé ici : {raw_path}")
    else:
        raw_path.unlink(missing_ok=True)
        with suppress(OSError):
            raw_path.parent.rmdir()
    print(f"\n{len(written)} image(s) détectée(s) et enregistrée(s) :")
    for path in written:
        print(f"  {path}")


def run(config: AppConfig, requested_device: str | None = None) -> None:
    print("Scanner Utils\n")
    devices = discover_devices()
    if not devices:
        raise ScannerUtilsError(
            "Aucun scanner détecté. Vérifiez son alimentation, le câble USB et les droits SANE."
        )
    device = _select_device(devices, requested_device)
    print("Lecture des capacités du scanner...")
    capabilities = inspect_capabilities(device)

    while True:
        print("\nQue souhaitez-vous faire ?")
        print("[1] Numériser une ou plusieurs photos")
        print("[2] Numériser une bande de négatifs")
        print("[3] Quitter")
        action = _choice("> ", {"1", "2", "3"})
        if action == "3":
            print("À bientôt.")
            return
        try:
            _scan_once("photo" if action == "1" else "negative", device, capabilities, config)
        except ScannerUtilsError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = _FrenchArgumentParser(
        description="Numérisation interactive de photos et de négatifs",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="afficher cette aide et quitter")
    parser.add_argument("--config", type=Path, help="chemin du fichier de configuration TOML")
    parser.add_argument("--device", help="nom du périphérique SANE, sans menu de sélection")
    parser.add_argument("--output-dir", type=Path, help="remplacer le dossier de sortie")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="conserver les TIFF bruts après un traitement réussi",
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
        print("\nOpération annulée.")
    except ScannerUtilsError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
