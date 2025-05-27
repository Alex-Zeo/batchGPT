from typing import List, Dict

from .openai_client import AsyncOpenAIClient
from .tokenizer import Tokenizer

from .docreader import DocReader, PDFDocReader
from .prompt_store import PromptStore
from .postprocessor import merge_chunks
from logs.logger import logger
from utils.events import EventManager


class Orchestrator:
    """Coordinate document loading, chunking and OpenAI requests."""

    def __init__(
        self,
        client: AsyncOpenAIClient,
        tokenizer: Tokenizer,
        store: PromptStore,
        reader: DocReader,
        events: EventManager | None = None,
    ) -> None:
        """Instantiate the orchestrator.

        Args:
            client: OpenAI client used for completion requests.
            tokenizer: Tokenizer used for chunking documents.
            store: Storage backend for prompts and responses.
            reader: Document reader implementation.
        """
        self.client = client
        self.tokenizer = tokenizer
        self.store = store
        self.reader = reader
        self.events = events or EventManager()

    async def process_document(self, path: str, **kwargs) -> str:
        """Process a document.

        Args:
            path: Path to the document to process.
            **kwargs: Additional options such as ``max_tokens`` or ``overlap``.

        Returns:
            Combined response from the LLM for the entire document.
        """
        logger.info(f"Starting processing for {path}")
        self.events.publish("batch_started", {"path": path})
        text, _ = self.reader.read(path, **kwargs)
        chunks = self.tokenizer.chunk(
            text,
            max_tokens=kwargs.get("max_tokens", 2000),
            overlap=kwargs.get("overlap", 50),
        )
        logger.info(f"{path} split into {len(chunks)} chunks")
        responses: List[Dict] = []
        for chunk in chunks:
            resp = await self.client.chat_complete([
                {"role": "user", "content": chunk}
            ], max_tokens=kwargs.get("response_tokens", 500))
            responses.append(resp)
            self.store.save(chunk, resp)
        logger.info(f"Completed processing for {path}")
        self.events.publish("batch_completed", {"path": path})
        return merge_chunks(responses)

    async def process_pdf(self, path: str, **kwargs) -> str:
        """Backward compatible PDF processing."""
        return await self.process_document(path, **kwargs)


async def run_pdf(path: str, model: str = "gpt-3.5-turbo", budget: float = None, output: str = None) -> str:
    """Helper to process a PDF without constructing an :class:`Orchestrator`.

    Args:
        path: Path to the PDF file.
        model: OpenAI model name.
        budget: Optional spending limit for API usage.
        output: Optional path to save responses.

    Returns:
        The aggregated LLM response for the PDF.
    """
    logger.info(f"run_pdf called with path={path} model={model} budget={budget}")
    tokenizer = Tokenizer(model)
    client = AsyncOpenAIClient(model=model, budget=budget)
    store = PromptStore(output or "store.jsonl")
    reader = PDFDocReader()
    orchestrator = Orchestrator(client, tokenizer, store, reader, EventManager())
    result = await orchestrator.process_document(path)
    return result
