"""Factory helpers for providers and processors."""
from __future__ import annotations

from .storage_service import StorageService
from .batch_repository import BatchRepository
from app.openai_client import AsyncOpenAIClient
from app.orchestrator import Orchestrator
from app.tokenizer import Tokenizer
from app.prompt_store import PromptStore
from app.docreader import PDFDocReader


def create_openai_orchestrator(model: str, budget: float | None = None) -> Orchestrator:
    """Create an orchestrator wired for OpenAI."""
    client = AsyncOpenAIClient(model=model, budget=budget)
    tokenizer = Tokenizer(model)
    store = PromptStore("store.jsonl")
    return Orchestrator(client, tokenizer, store, PDFDocReader())


def create_repository(base_dir: str = "data") -> BatchRepository:
    return BatchRepository(StorageService(base_dir))

