import asyncio
from typing import List, Dict

from .openai_client import AsyncOpenAIClient
from .tokenizer import Tokenizer
from .pdfreader.pdf_loader import chunk_pdf
from .prompt_store import PromptStore
from .postprocessor import merge_chunks
from .logger import logger


class Orchestrator:
    """Coordinate PDF loading, chunking and OpenAI requests."""

    def __init__(
        self,
        client: AsyncOpenAIClient,
        tokenizer: Tokenizer,
        store: PromptStore,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.store = store

    async def process_pdf(self, path: str, **kwargs) -> str:
        logger.info(f"Starting PDF processing for {path}")
        _text, chunks, digest = chunk_pdf(path, self.tokenizer, **kwargs)
        logger.info(f"PDF {path} split into {len(chunks)} chunks")
        responses: List[Dict] = []
        for chunk in chunks:
            resp = await self.client.chat_complete([
                {"role": "user", "content": chunk}
            ], max_tokens=kwargs.get("max_tokens", 500))
            responses.append(resp)
            self.store.save(chunk, resp)
        logger.info(f"Completed PDF processing for {path}")
        return merge_chunks(responses)


async def run_pdf(path: str, model: str = "gpt-3.5-turbo", budget: float = None, output: str = None) -> str:
    logger.info(f"run_pdf called with path={path} model={model} budget={budget}")
    tokenizer = Tokenizer(model)
    client = AsyncOpenAIClient(model=model, budget=budget)
    store = PromptStore(output or "store.jsonl")
    orchestrator = Orchestrator(client, tokenizer, store)
    result = await orchestrator.process_pdf(path)
    return result
