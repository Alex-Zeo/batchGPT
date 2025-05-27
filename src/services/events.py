from __future__ import annotations
from typing import List, Protocol


class BatchObserver(Protocol):
    def update(self, batch_id: str, status: str) -> None:
        ...


class BatchEventEmitter:
    """Simple observer implementation for batch status."""

    def __init__(self) -> None:
        self._observers: List[BatchObserver] = []

    def register(self, observer: BatchObserver) -> None:
        self._observers.append(observer)

    def notify(self, batch_id: str, status: str) -> None:
        for obs in list(self._observers):
            obs.update(batch_id, status)
