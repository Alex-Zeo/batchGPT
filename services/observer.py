# mypy: ignore-errors
"""Observer utilities for batch status updates."""

from abc import ABC, abstractmethod
from typing import Callable


class BatchStatusObserver(ABC):
    """Interface for objects that react to batch status changes."""

    @abstractmethod
    def update(self, batch_id: str, status: str) -> None:
        """Handle a change to ``status`` for ``batch_id``."""
        raise NotImplementedError


class CallbackObserver(BatchStatusObserver):
    """Invoke a callback whenever the status updates."""

    def __init__(self, callback: Callable[[str, str], None]) -> None:
        self.callback = callback

    def update(self, batch_id: str, status: str) -> None:
        self.callback(batch_id, status)
