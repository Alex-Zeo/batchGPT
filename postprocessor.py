from typing import List
from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_index: int
    content: str


def merge_results(results: List[ChunkResult]) -> str:
    ordered = sorted(results, key=lambda r: r.chunk_index)
    return "\n".join(r.content for r in ordered)
