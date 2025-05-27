from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List


class EventManager:
    """Simple observer pattern implementation."""

    def __init__(self) -> None:
        self._listeners: DefaultDict[str, List[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[..., None]) -> None:
        self._listeners[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., None]) -> None:
        if callback in self._listeners[event]:
            self._listeners[event].remove(callback)

    def notify(self, event: str, **data: Any) -> None:
        for callback in list(self._listeners[event]):
            callback(**data)
