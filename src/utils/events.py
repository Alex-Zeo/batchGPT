"""Simple observer pattern utilities."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, List


class EventManager:
    """Lightweight event dispatcher."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event: str, callback: Callable[[Any], None]) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def publish(self, event: str, payload: Any) -> None:
        for cb in self._listeners.get(event, []):
            cb(payload)

