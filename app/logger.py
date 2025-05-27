from pathlib import Path
from loguru import logger

__all__ = ["logger", "setup_logger"]


def setup_logger(log_dir: str = "logs") -> None:
    """Configure loguru to write logs to rotating files in ``log_dir``."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers so initialization is idempotent
    logger.remove()

    # Info messages
    logger.add(
        log_path / "info.log",
        level="INFO",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["level"].name == "INFO",
    )
    # Warning messages
    logger.add(
        log_path / "warning.log",
        level="WARNING",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["level"].name == "WARNING",
    )
    # Error messages
    logger.add(
        log_path / "error.log",
        level="ERROR",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["level"].name == "ERROR",
    )

__all__ = ["logger", "setup_logger"]
