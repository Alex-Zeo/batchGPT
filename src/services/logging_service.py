import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Formatter that outputs log records in JSON format."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        record_dict = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            record_dict["exception"] = self.formatException(record.exc_info)
        return json.dumps(record_dict)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure a JSON logger with rotation inside ``log_dir``.

    Parameters
    ----------
    log_dir:
        Directory where log files should be stored.
    level:
        Logging level to use for the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batchgpt")
    logger.setLevel(level)
    logger.handlers.clear()

    log_file = path / "app.log"
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)

    return logger
