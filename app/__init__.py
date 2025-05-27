from .logger import setup_logger, logger  # type: ignore[attr-defined]

# Initialize logging when the package is imported
setup_logger()

__all__ = ["logger", "setup_logger"]
