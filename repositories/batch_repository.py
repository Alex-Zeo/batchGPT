# mypy: ignore-errors
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from services import StorageService

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from app.batch_manager import BatchJob


class BatchRepository:
    """Persist ``BatchJob`` instances using :class:`StorageService`."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage
        self.base_dir = storage.base_dir

    def save(self, batch: BatchJob) -> None:
        """Save ``batch`` to disk."""
        self.storage.save_batch(batch.id, batch.to_dict())

    def load(self, batch_id: str) -> Optional[BatchJob]:
        """Load a ``BatchJob`` by ``batch_id``."""
        from app.batch_manager import BatchJob  # avoid circular import

        data = self.storage.load_batch(batch_id)
        return BatchJob.from_dict(data) if data else None

    def delete(self, batch_id: str) -> bool:
        """Delete stored batch data."""
        path = self.base_dir / f"{batch_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_ids(self) -> List[str]:
        """Return all stored batch identifiers."""
        return sorted(p.stem for p in self.base_dir.glob("*.json"))
