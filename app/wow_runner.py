import asyncio
import os
import random
from typing import List, Dict, Any

import openai

from .utils import logger


class WowRunner:
    """Run prompts against OpenAI with concurrency control."""

    def __init__(self, model: str = "gpt-3.5-turbo", max_concurrent: int = 5,
                 max_retries: int = 3, base_delay: float = 1.0) -> None:
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def _call_openai(self, messages: List[Dict[str, str]]) -> Any:
        """Send a chat completion request."""
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

    async def run_prompt(self, prompt: str) -> Dict[str, Any]:
        """Run a single prompt with retries and backoff."""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    resp = await self._call_openai(messages)
                return {
                    "prompt": prompt,
                    "response": resp.choices[0].message.content,
                    "id": resp.id,
                    "success": True,
                    "status_code": 200,
                }
            except openai.APIStatusError as e:
                status = e.status_code or 0
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"OpenAI API error {status} on attempt {attempt + 1}: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"OpenAI request failed: {e}")
                return {
                    "prompt": prompt,
                    "error": str(e),
                    "success": False,
                    "status_code": status,
                }
            except Exception as e:  # Catch network or unexpected errors
                logger.error(f"Unexpected error for prompt '{prompt[:20]}': {e}")
                return {
                    "prompt": prompt,
                    "error": str(e),
                    "success": False,
                    "status_code": None,
                }
        logger.error(f"Max retries exceeded for prompt '{prompt[:20]}'")
        return {
            "prompt": prompt,
            "error": "Max retries exceeded",
            "success": False,
            "status_code": None,
        }

    async def run(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Run multiple prompts concurrently respecting the semaphore."""
        tasks = [asyncio.create_task(self.run_prompt(p)) for p in prompts]
        return await asyncio.gather(*tasks)
