"""Structured JSON logger utilities."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from loguru import logger

__all__ = ["logger", "setup_logger", "new_cli_log", "new_streamlit_log"]


def setup_logger(log_dir: str = "logs") -> None:
    """Configure loguru with JSON formatting and rotation."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        path / "app.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DDTHH:mm:ssZ}|{level}|{message}",
        serialize=True,
    )


def _session_markdown(prefix: str, log_dir: str) -> Path:
    date_str = datetime.utcnow().strftime("%m_%d_%Y")
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{prefix}_{date_str}.md"
    file.touch()
    return file


def new_cli_log(log_dir: str = "logs") -> Path:
    """Create a new CLI session markdown log file."""
    return _session_markdown("cli", log_dir)


def new_streamlit_log(log_dir: str = "logs") -> Path:
    """Create a new Streamlit session markdown log file."""
    return _session_markdown("streamlit", log_dir)
