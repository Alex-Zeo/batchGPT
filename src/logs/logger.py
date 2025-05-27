from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

__all__ = ["logger", "setup_logger", "create_cli_log", "create_streamlit_log"]


from typing import Any, Dict


def _json_formatter(record: Dict[str, Any]) -> str:
    record["extra"].setdefault("timestamp", datetime.utcnow().isoformat())
    payload: Dict[str, Any] = {
        "timestamp": record["extra"]["timestamp"],
        "level": record["level"].name,
        "logger_name": record["name"],
        "message": record["message"],
    }
    for key, value in record["extra"].items():
        if key != "timestamp":
            payload[key] = value
    return json.dumps(payload)


def setup_logger(log_dir: str = "logs") -> None:
    """Configure loguru logger with JSON output and rotation."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        path / "app.log",
        rotation="10 MB",
        retention="7 days",
        format=_json_formatter,  # type: ignore[arg-type]
        serialize=False,
    )


def _create_session_log(prefix: str, name: str, log_dir: str) -> Path:
    date = datetime.now().strftime("%m_%d_%Y")
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{prefix}_{date}_{name}.md"
    if not file.exists():
        file.write_text(f"# {prefix.capitalize()} Session {date}\n")
    return file


def create_cli_log(name: str, log_dir: str = "logs") -> Path:
    """Create a markdown log file for a CLI session."""
    return _create_session_log("cli", name, log_dir)


def create_streamlit_log(name: str, log_dir: str = "logs") -> Path:
    """Create a markdown log file for a Streamlit session."""
    return _create_session_log("streamlit", name, log_dir)
