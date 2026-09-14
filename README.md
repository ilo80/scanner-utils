# Scanner Utils

Scanner Utils is an interactive Linux CLI for repeated photo and film scanning through
[SANE](http://www.sane-project.org/) and `scanimage`. It discovers connected devices, inspects
the selected backend at runtime, acquires a high-quality raw TIFF, detects individual images,
and writes one lossless output file per photo or negative frame.

The initial hardware target is the **Epson Perfection 1650 Photo** (reported by SANE as the
Epson GT-8200), but device addresses are never hard-coded.

## Features

- Interactive, reusable scanning session with photo, negative, and quit actions.
- Runtime SANE device discovery and capability inspection.
- Separate defaults for reflective photos and the Transparency Unit.
- Multiple-photo detection with conservative cropping and deskewing.
- Automatic negative-frame detection for horizontal and vertical strips.
- High-bit-depth negative inversion with per-channel orange-mask compensation.
- 16-bit TIFF and PNG processing where supported by the scanner and encoder.
- Collision-safe names such as `2026-09-14_negative_001_02.tiff`.
- Automatic preservation of the untouched raw TIFF whenever processing fails.

## Requirements

- Linux with Python 3.11 or newer.
- SANE and its `scanimage` utility.
- Permission to access the scanner USB device.
- An Epson Transparency Unit for negative scanning.

On Fedora:

```console
sudo dnf install sane-backends sane-backends-drivers-scanners
```

On Debian or Ubuntu:

```console
sudo apt install sane-utils
```

Verify the scanner independently before installing Scanner Utils:

```console
scanimage -L
scanimage -d 'your-device-name' --all-options
```

The Epson Perfection 1650 Photo tested during project initialization exposed `Flatbed` and
`Transparency Unit`, `Color`, depths of 8/12/14/16 bits, resolutions including 600 and 1600
DPI, and the `epson2` color/gamma correction controls. A real 50 DPI Transparency Unit preview
also confirmed 16-bit RGB TIFF acquisition and the backend's automatic narrow scan area. Scanner
Utils only passes an option if the selected backend advertises it.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```console
uv tool install .
scanner-utils
```

For development:

```console
uv sync --extra dev
uv run scanner-utils
uv run pytest
uv run ruff check .
```

Alternatively, use a regular virtual environment and `pip install -e '.[dev]'`.

## Usage

Run:

```console
scanner-utils
```

Scanner Utils lists every device returned by SANE. Choose the Epson scanner rather than a
webcam if both appear. The scanner is selected once; after every acquisition the main menu is
shown again.

Useful overrides:

```console
scanner-utils --output-dir ./scans --keep-raw
scanner-utils --device 'epson2:libusb:001:009'
scanner-utils --config ~/.config/scanner-utils/config.toml
```

Do not rely on a saved `libusb:BUS:DEVICE` number: it can change after reconnecting the
scanner. Normal interactive discovery avoids this issue.

### Photos

Fit the reflective white cover, remove the film holder if needed, and leave visible scanner-bed
background between multiple photos. A full-bed image is acquired once; each detected rectangle
is conservatively padded, deskewed, and exported separately.

### Negatives

Connect and install the Transparency Unit, place the strip in its holder, and keep the active
transparency area clear. The default profile requests color at 1600 DPI and 16 bits per channel.
Physical film edges are detected first, then each frame is inverted independently in floating
point with per-channel percentile normalization. This prevents the holder from skewing color
levels. The final conversion back to integer samples happens only on export.

Automatic processing is intentionally conservative. If detection fails, the raw TIFF remains in
the output directory's `.raw` folder so that a slow scan is never lost.

## Configuration

Copy [`scanner-utils.example.toml`](scanner-utils.example.toml) to
`~/.config/scanner-utils/config.toml` and change only what you need. Command-line options take
precedence over this file. The standard defaults are:

| Setting | Photos | Negatives |
| --- | ---: | ---: |
| Source | Flatbed | Transparency Unit |
| Resolution | 600 DPI | 1600 DPI |
| Depth | 16 bit | 16 bit |
| Color correction | Built-in CCT | None |

TIFF is the default output format. Set `keep_raw_scans = true` for archival workflows.

## Troubleshooting

- **No scanner found:** confirm power and USB connection, run `scanimage -L`, and check udev
  permissions. On many distributions the logged-in user must be in the `scanner` group.
- **A webcam is listed:** this is a normal SANE/V4L device; select the Epson entry.
- **An option is rejected:** reconnect the scanner and inspect `scanimage --all-options`. The
  backend can activate options only after a source or mode changes; the raw scan is retained.
- **No images are detected:** increase contrast with the bed/holder and leave space between
  items. The raw file can be processed again after tuning detection settings.
- **Processing stops:** no generated output replaces or deletes the raw acquisition on failure.

## Architecture

- `scanner.py`: device discovery, capability inspection, safe command construction, acquisition.
- `config.py`: centralized profiles and optional TOML overrides.
- `processing/detection.py`: reusable region detection and perspective cropping.
- `processing/photos.py`: multi-photo extraction.
- `processing/negatives.py`: high-depth inversion and frame splitting.
- `cli.py`: interaction and repeated-session orchestration.

The scanner boundary and processing pipeline are separate so detection can be tested using
fixture images without requiring the physical scanner.

## Development

Use English for source code and documentation. Follow Conventional Commits and keep generated
scans out of Git. Before committing:

```console
uv run ruff check .
uv run pytest
```

This project is licensed under the GNU General Public License v3.0; see [`LICENSE`](LICENSE).
