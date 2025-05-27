from __future__ import annotations

from dataclasses import field
from typing import Any, Dict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Result from batch evaluation."""

    custom_id: str
    status: str
    created_at: str
    model: str
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
