import asyncio
from pathlib import Path
from typing import List

from pdf_loader import load_pdf_text
from tokenizer import Tokenizer
from prompt_store import load_prompts
from openai_client import OpenAIClient
from postprocessor import ChunkResult


class Orchestrator:
    def __init__(self, model: str = "gpt-4o", parallelism: int = 5):
        self.model = model
        self.parallelism = parallelism
        self.prompts = load_prompts()
        self.tokenizer = Tokenizer(model)
        self.client = OpenAIClient(model)

    async def _process_chunk(self, chunk: str, index: int) -> ChunkResult:
        messages = [
            {"role": "system", "content": self.prompts.system},
            {"role": "user", "content": f"{self.prompts.user}\n\n{chunk}"},
        ]
        content = await self.client.chat(messages)
        return ChunkResult(chunk_index=index, content=content)

    async def run(self, pdf: Path) -> List[ChunkResult]:
        text, _ = load_pdf_text(pdf)
        chunks = self.tokenizer.chunk(text)
        sem = asyncio.Semaphore(self.parallelism)

        async def sem_task(ch, i):
            async with sem:
                return await self._process_chunk(ch, i)

        tasks = [asyncio.create_task(sem_task(ch, i)) for i, ch in enumerate(chunks)]
        return await asyncio.gather(*tasks)
