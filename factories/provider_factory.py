# mypy: ignore-errors
"""Factory utilities for LLM provider clients."""

from typing import Any


class LLMProviderFactory:
    """Create LLM provider clients by name."""

    @staticmethod
    def create(provider: str, **kwargs: Any) -> Any:
        """Return a client instance for ``provider``."""
        if provider == "openai":
            from app.openai_client import AsyncOpenAIClient

            return AsyncOpenAIClient(**kwargs)
        raise ValueError(f"Unknown provider: {provider}")
