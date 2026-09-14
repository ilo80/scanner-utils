"""Application-specific exceptions."""


class ScannerUtilsError(Exception):
    """Base exception for errors that should be shown without a traceback."""


class ScannerUnavailableError(ScannerUtilsError):
    """Raised when SANE or the selected scanner is unavailable."""


class ScanFailedError(ScannerUtilsError):
    """Raised when image acquisition fails."""


class ProcessingError(ScannerUtilsError):
    """Raised when an acquired scan cannot be processed safely."""
