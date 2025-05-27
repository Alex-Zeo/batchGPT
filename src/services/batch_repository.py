"""Repository layer for batch persistence."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .storage_service import StorageService


class BatchRepository:
    """Abstraction over :class:`StorageService` for batch data."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    def save_metadata(self, batch_id: str, data: Dict[str, Any]) -> None:
        self.storage.save_batch(batch_id, data)

    def load_metadata(self, batch_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.load_batch(batch_id)

    def save_results(self, batch_id: str, results: List[Dict[str, Any]]) -> None:
        self.storage.save_results(batch_id, results)

    def load_results(self, batch_id: str) -> List[Dict[str, Any]]:
        return self.storage.load_results(batch_id)
