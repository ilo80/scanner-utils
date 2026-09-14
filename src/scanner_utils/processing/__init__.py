"""High-bit-depth scan processing."""

from scanner_utils.processing.negatives import process_negative_scan
from scanner_utils.processing.photos import process_photo_scan

__all__ = ["process_negative_scan", "process_photo_scan"]
