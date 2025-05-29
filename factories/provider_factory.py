# mypy: ignore-errors
"""Factory utilities for LLM provider clients."""

from typing import Any
import os


class LLMProviderFactory:
    """Create LLM provider clients by name."""

    @staticmethod
    def create(provider: str, **kwargs: Any) -> Any:
        """Return a client instance for ``provider``."""
        if provider == "openai":
            from app.openai_client import AsyncOpenAIClient

            kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY", "test"))
            return AsyncOpenAIClient(**kwargs)
        raise ValueError(f"Unknown provider: {provider}")
