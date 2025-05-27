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
        """Initialize the orchestrator.

        Args:
            client: OpenAI client for sending requests.
            tokenizer: Tokenizer used for chunking.
            store: Prompt store to persist conversations.
        """
        self.client = client
        self.tokenizer = tokenizer
        self.store = store

    async def process_pdf(self, path: str, **kwargs) -> str:
        """Process a PDF file and return the merged LLM output.

        Args:
            path: Path to the PDF file.
            **kwargs: Additional arguments for chunking/tokenization.

        Returns:
            str: Combined LLM response for all chunks.

        Example:
            >>> await orchestrator.process_pdf("file.pdf")
        """
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
    """Convenience wrapper to process a PDF with minimal setup.

    Args:
        path: Path to the PDF file.
        model: Model name to use for the client.
        budget: Optional spend limit in USD.
        output: Path of the prompt store file.

    Returns:
        str: Combined LLM response from :meth:`Orchestrator.process_pdf`.

    Example:
        >>> await run_pdf("doc.pdf", model="gpt-4")
    """
    logger.info(f"run_pdf called with path={path} model={model} budget={budget}")
    tokenizer = Tokenizer(model)
    client = AsyncOpenAIClient(model=model, budget=budget)
    store = PromptStore(output or "store.jsonl")
    orchestrator = Orchestrator(client, tokenizer, store)
    result = await orchestrator.process_pdf(path)
    return result
