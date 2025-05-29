from typing import List

from logs.logger import logger

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


class Tokenizer:
    """Simple token counter wrapper."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.model = model
        if tiktoken:
            try:
                self.enc = tiktoken.encoding_for_model(model)
            except Exception:
                self.enc = tiktoken.get_encoding("cl100k_base")
        else:
            self.enc = None
        logger.debug(f"Tokenizer initialized for model {model}")

    def count(self, text: str) -> int:
        if self.enc:
            tokens = len(self.enc.encode(text))
        else:
            tokens = len(text.split())
        logger.debug(f"Counted {tokens} tokens")
        return tokens

    def chunk(self, text: str, max_tokens: int, overlap: int = 0) -> List[str]:
        if self.count(text) <= max_tokens:
            logger.debug("Text within max_tokens, no chunking needed")
            return [text]
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + max_tokens
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap if overlap else end
        logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks