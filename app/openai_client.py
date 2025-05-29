# mypy: ignore-errors
import os
import asyncio
import time
from typing import Any, Dict, List, Optional

import openai
from aiohttp import ClientError

from .logger import logger


class AsyncOpenAIClient:
    """Asynchronous OpenAI client with basic retry and throttling.

    Args:
        api_key: API key to use. If ``None`` the ``OPENAI_API_KEY`` environment
            variable will be read.
        model: Model name for all completions.
        max_tokens_per_minute: Throttle limit for token usage.
        request_limit_per_minute: Throttle limit for number of requests.
        budget: Optional spending cap in USD.
        retry_limit: Maximum retries for transient failures.

    Example:
        >>> client = AsyncOpenAIClient(model="gpt-4")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens_per_minute: int = 60000,
        request_limit_per_minute: int = 3000,
        budget: Optional[float] = None,
        retry_limit: int = 3,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: API key for authentication. Defaults to ``OPENAI_API_KEY``.
            model: Target model name.
            max_tokens_per_minute: Rate limit for token usage.
            request_limit_per_minute: Maximum requests per minute.
            budget: Optional budget cap in USD.
            retry_limit: Number of retry attempts for API calls.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_limit_per_minute = request_limit_per_minute
        self.budget = budget
        self.retry_limit = retry_limit
        self._lock = asyncio.Lock()
        self._reset_throttle()
        self.cost_spent = 0.0
        logger.info(
            f"Initialized AsyncOpenAIClient model={self.model} budget={self.budget}"
        )

    # simple pricing table per million tokens
    PRICING = {
        "gpt-4": {"input": 30 / 1_000_000, "output": 60 / 1_000_000},
        "gpt-4-turbo": {"input": 10 / 1_000_000, "output": 30 / 1_000_000},
        "gpt-3.5-turbo": {"input": 0.5 / 1_000_000, "output": 1.5 / 1_000_000},
    }

    def _reset_throttle(self) -> None:
        self._window_start = time.time()
        self._tokens_used = 0
        self._requests_used = 0
        logger.debug("Throttle counters reset")

    async def _throttle(self, token_count: int) -> None:
        async with self._lock:
            now = time.time()
            if now - self._window_start >= 60:
                self._reset_throttle()
            # wait if tokens or requests exceed limits
            while (
                self._tokens_used + token_count > self.max_tokens_per_minute
                or self._requests_used + 1 > self.request_limit_per_minute
            ):
                wait = 60 - (now - self._window_start)
                logger.debug(f"Throttling for {wait:.2f}s due to rate limits")
                await asyncio.sleep(max(wait, 0))
                now = time.time()
                if now - self._window_start >= 60:
                    self._reset_throttle()
            self._tokens_used += token_count
            self._requests_used += 1

    def _update_cost(self, usage: Dict[str, int]) -> None:
        model_price = None
        for key in self.PRICING:
            if key in self.model:
                model_price = self.PRICING[key]
                break
        if not model_price:
            return
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self.cost_spent += (
            input_tokens * model_price["input"] + output_tokens * model_price["output"]
        )

    async def chat_complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Create a chat completion request.

        Args:
            messages: List of chat messages following the OpenAI schema.
            **kwargs: Additional parameters forwarded to the OpenAI SDK.

        Returns:
            The response dictionary returned by the OpenAI SDK.

        Raises:
            RuntimeError: If the configured budget has been exhausted.
            openai.RateLimitError: When the API rejects the request due to rate limits.
            aiohttp.ClientError: On network related failures.

        Example:
            >>> await client.chat_complete([{"role": "user", "content": "Hi"}])
        """

        token_estimate = sum(len(m.get("content", "").split()) for m in messages)
        if self.budget and self.cost_spent >= self.budget:
            logger.error("Budget exhausted")
            raise RuntimeError("Budget exhausted")

        await self._throttle(token_estimate + kwargs.get("max_tokens", 0))

        attempts = 0
        while True:
            try:
                logger.debug(
                    f"Calling chat_complete attempt={attempts} tokens={token_estimate}"
                )
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                if hasattr(resp, "usage"):
                    self._update_cost(resp.usage.dict())
                return resp.model_dump() if hasattr(resp, "model_dump") else resp
            except (openai.RateLimitError, ClientError) as e:
                attempts += 1
                logger.warning(f"Rate limit or network error: {e}; retry {attempts}")
                if attempts > self.retry_limit:
                    raise
                await asyncio.sleep(2 ** attempts)
            except Exception as e:
                attempts += 1
                logger.error(f"Unexpected error: {e}; retry {attempts}")
                if attempts > self.retry_limit:
                    raise
                await asyncio.sleep(2 ** attempts)

