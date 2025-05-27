"""Validation utilities for BatchGPT."""
from __future__ import annotations

import os
import re
from typing import Optional

from src.logs.logger import logger


def sanitize_input(input_str: str) -> str:
    """Remove HTML tags and non-printable characters from a string.

    Args:
        input_str: Raw user input.

    Returns:
        The sanitized string safe for processing.
    """
    sanitized = "".join(
        ch for ch in input_str if (ch.isprintable() or ch in "\n\r\t") and ch not in "<>"
    )
    sanitized = sanitized.replace("<", "").replace(">", "")
    logger.info(f"Input sanitized: {sanitized}")
    return sanitized


def validate_api_key() -> bool:
    """Check that the ``OPENAI_API_KEY`` environment variable looks valid.

    Returns:
        ``True`` if the key exists and matches the expected format, otherwise ``False``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return False

    if not (api_key.startswith("sk-") or api_key.startswith("org-")):
        logger.error("OPENAI_API_KEY appears to be malformed (should start with 'sk-' or 'org-')")
        return False

    if len(api_key) < 40:
        logger.error("OPENAI_API_KEY appears too short to be valid")
        return False

    logger.info("API key format validated")
    return True


def is_valid_poll_interval(interval: str) -> Optional[int]:
    """Validate and convert a poll interval string into seconds.

    The accepted format is ``<integer><unit>`` where ``unit`` can be ``s`` for
    seconds, ``m`` for minutes, or ``h`` for hours.

    Args:
        interval: Interval string such as ``"30s"`` or ``"5m"``.

    Returns:
        Number of seconds represented by the interval or ``None`` if invalid.
    """
    if not interval:
        return None

    match = re.match(r"^(\d+)([smh])$", interval.lower())
    if not match:
        return None

    value, unit = match.groups()
    seconds = int(value)
    if unit == "m":
        seconds *= 60
    elif unit == "h":
        seconds *= 3600
    return seconds


def calculate_batch_size(total_requests: int, max_batch_size: int = 5000) -> int:
    """Determine an appropriate batch size given a request count.

    Args:
        total_requests: Number of requests to be processed.
        max_batch_size: Maximum allowed batch size.

    Returns:
        Batch size that does not exceed ``max_batch_size``.
    """
    if total_requests <= max_batch_size:
        return total_requests

    for size in range(max_batch_size, 1000, -500):
        if total_requests % size == 0 or total_requests % size < 100:
            return size

    return max_batch_size
