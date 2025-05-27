from __future__ import annotations

from dataclasses import field
from typing import Any, Dict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class BatchRequest:
    """Request item for OpenAI batch API."""

    custom_id: str
    method: str
    url: str
    body: Dict[str, Any] = field(default_factory=dict)
