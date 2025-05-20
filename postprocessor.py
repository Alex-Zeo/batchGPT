from typing import List, Dict

def merge_chunks(responses: List[Dict]) -> str:
    """Combine responses from chunked completions into a single string."""
    texts = []
    for r in responses:
        choice = r.get("choices", [{}])[0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        texts.append(content)
    return "\n".join(texts)

from typing import List
from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_index: int
    content: str


def merge_results(results: List[ChunkResult]) -> str:
    ordered = sorted(results, key=lambda r: r.chunk_index)
    return "\n".join(r.content for r in ordered)
