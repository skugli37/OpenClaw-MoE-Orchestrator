class OpenClawError(Exception):
    """Base exception for production runtime failures."""


class ConfigurationError(OpenClawError):
    """Raised when required runtime configuration is invalid."""


class DataValidationError(OpenClawError):
    """Raised when market data is missing or malformed."""


class ExternalSignalError(OpenClawError):
    """Raised when external signal providers do not return verifiable data."""
