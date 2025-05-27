from __future__ import annotations
from pathlib import Path
from datetime import datetime
from loguru import logger

__all__ = ["logger", "setup_logger", "start_cli_session", "start_streamlit_session"]


def _create_markdown_log(path: Path) -> None:
    """Add a markdown log file handler."""
    logger.add(path, format="{message}", level="INFO")


def setup_logger(log_dir: str = "logs") -> None:
    """Configure loguru to write JSON logs with rotation."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        log_path / "info.log",
        level="INFO",
        rotation="1 MB",
        serialize=True,
        filter=lambda r: r["level"].name == "INFO",
    )
    logger.add(
        log_path / "warning.log",
        level="WARNING",
        rotation="1 MB",
        serialize=True,
        filter=lambda r: r["level"].name == "WARNING",
    )
    logger.add(
        log_path / "error.log",
        level="ERROR",
        rotation="1 MB",
        serialize=True,
        filter=lambda r: r["level"].name == "ERROR",
    )


def start_cli_session(name: str, log_dir: str = "logs") -> None:
    """Create a CLI markdown log file."""
    date = datetime.now().strftime("%m_%d_%Y")
    _create_markdown_log(Path(log_dir) / f"cli_{date}_{name}.md")


def start_streamlit_session(name: str, log_dir: str = "logs") -> None:
    """Create a Streamlit markdown log file."""
    date = datetime.now().strftime("%m_%d_%Y")
    _create_markdown_log(Path(log_dir) / f"streamlit_{date}_{name}.md")
