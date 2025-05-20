from typing import List

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

    def count(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def chunk(self, text: str, max_tokens: int, overlap: int = 0) -> List[str]:
        if self.count(text) <= max_tokens:
            return [text]
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + max_tokens
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap if overlap else end
        return chunks
