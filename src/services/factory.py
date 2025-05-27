from __future__ import annotations
from typing import Any

from src.app.orchestrator import Orchestrator
from src.app.openai_client import AsyncOpenAIClient
from src.app.tokenizer import Tokenizer
from src.app.docreader import PDFDocReader
from src.app.prompt_store import PromptStore

from .storage_service import StorageService
from .batch_repository import BatchRepository


def create_llm_client(provider: str = "openai", **kwargs: Any) -> AsyncOpenAIClient:
    if provider != "openai":
        raise ValueError(f"Unsupported provider: {provider}")
    return AsyncOpenAIClient(model=kwargs.get("model", "gpt-3.5-turbo"), budget=kwargs.get("budget"))


def create_orchestrator(model: str = "gpt-3.5-turbo", storage_dir: str = "data") -> Orchestrator:
    client = create_llm_client(model=model)
    tokenizer = Tokenizer(model)
    storage = StorageService(storage_dir)
    repository = BatchRepository(storage)
    reader = PDFDocReader()
    store = PromptStore(storage_dir)
    return Orchestrator(client, tokenizer, store, reader)
