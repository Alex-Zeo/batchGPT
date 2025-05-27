"""Validation utilities for BatchGPT."""

from __future__ import annotations

import os
import re
from typing import Optional

from app.logger import logger  # type: ignore[attr-defined]


def sanitize_input(input_str: str) -> str:
    """Remove non-printable and angle bracket characters from user input.

    Args:
        input_str: Raw input string from the user.

    Returns:
        The sanitized string safe for processing.
    """
    sanitized = "".join(
        ch
        for ch in input_str
        if (ch.isprintable() or ch in "\n\r\t") and ch not in "<>"
    )
    sanitized = sanitized.replace("<", "").replace(">", "")
    logger.info(f"Input sanitized: {sanitized}")
    return sanitized


def calculate_batch_size(total_requests: int, max_batch_size: int = 5000) -> int:
    """Determine an appropriate batch size given the total number of requests.

    Args:
        total_requests: Total requests to process.
        max_batch_size: Maximum allowed batch size.

    Returns:
        The selected batch size which will not exceed ``max_batch_size``.
    """
    if total_requests <= max_batch_size:
        return total_requests

    for size in range(max_batch_size, 1000, -500):
        if total_requests % size == 0 or total_requests % size < 100:
            return size

    return max_batch_size


def validate_api_key() -> bool:
    """Ensure the ``OPENAI_API_KEY`` environment variable looks valid.

    Returns:
        ``True`` if the key exists and follows expected patterns, ``False`` otherwise.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return False

    if not (api_key.startswith("sk-") or api_key.startswith("org-")):
        logger.error(
            "OPENAI_API_KEY appears to be malformed (should start with 'sk-' or 'org-')"
        )
        return False

    if len(api_key) < 40:
        logger.error("OPENAI_API_KEY appears too short to be valid")
        return False

    logger.info("API key format validated")
    return True


def is_valid_poll_interval(interval: str) -> Optional[int]:
    """Validate a poll interval string.

    Args:
        interval: Interval string such as ``"30s"`` or ``"5m"``.

    Returns:
        Number of seconds if valid, otherwise ``None``.
    """
    if not interval:
        return None

    pattern = r"^(\d+)([smh])$"
    match = re.match(pattern, interval.lower())

    if not match:
        return None

    value, unit = match.groups()
    value = int(value)

    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600

    return None


__all__ = [
    "sanitize_input",
    "calculate_batch_size",
    "validate_api_key",
    "is_valid_poll_interval",
]
