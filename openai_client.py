import asyncio
from typing import List, Dict
from openai import AsyncOpenAI


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = AsyncOpenAI()

    async def chat(self, messages: List[Dict[str, str]], *, max_retries: int = 5) -> str:
        delay = 1
        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(model=self.model, messages=messages)
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20)
        raise RuntimeError("Exceeded max retries")
