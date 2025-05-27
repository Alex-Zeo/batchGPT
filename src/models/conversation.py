from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List
from pydantic.dataclasses import dataclass


@dataclass
class Conversation:
    """Conversation history associated with a batch."""

    batch_id: str
    log_entries: List[Dict[str, Any]] = field(default_factory=list)

    def add_log_entry(self, entry: Dict[str, Any]) -> None:
        self.log_entries.append(entry)
