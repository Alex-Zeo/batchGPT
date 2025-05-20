from typing import List
import tiktoken


class Tokenizer:
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, max_tokens: int = 6000, overlap: int = 200) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk = self.encoding.decode(tokens[start:end])
            chunks.append(chunk)
            if end == len(tokens):
                break
            start = end - overlap
        return chunks

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))
