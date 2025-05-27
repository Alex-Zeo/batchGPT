"""File and content parsing helpers for BatchGPT."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Tuple

from app.logger import logger  # type: ignore[attr-defined]


def write_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write.
        file_path: Destination file path.
    """
    with open(file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")
    logger.info(f"Wrote {len(data)} items to {file_path}")


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    results: List[Dict[str, Any]] = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Read {len(results)} items from {file_path}")
    return results


def generate_hash(content: str) -> str:
    """Return an MD5 hash of ``content``."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def deduplicate_prompts(prompts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Remove duplicate prompts while tracking original indices."""
    unique_prompts: List[str] = []
    hash_to_indices: Dict[str, List[int]] = {}
    for i, prompt in enumerate(prompts):
        prompt_hash = generate_hash(prompt)
        if prompt_hash not in hash_to_indices:
            hash_to_indices[prompt_hash] = [i]
            unique_prompts.append(prompt)
        else:
            hash_to_indices[prompt_hash].append(i)
    logger.info(
        f"Deduplicated {len(prompts)} prompts to {len(unique_prompts)} unique prompts"
    )
    return unique_prompts, hash_to_indices


def expand_results(
    deduplicated_results: List[Dict[str, Any]], hash_to_indices: Dict[str, List[int]]
) -> List[Dict[str, Any]]:
    """Expand deduplicated results back to their original ordering."""
    expanded: List[Dict[str, Any]] = []
    custom_id_to_result = {r.get("custom_id", ""): r for r in deduplicated_results}
    for indices in hash_to_indices.values():
        for idx in indices:
            result = custom_id_to_result.get(str(indices[0]), {})
            copy = result.copy()
            copy["custom_id"] = str(idx)
            expanded.append(copy)
    expanded.sort(key=lambda x: int(x.get("custom_id", "0")))
    return expanded


__all__ = [
    "write_jsonl",
    "read_jsonl",
    "generate_hash",
    "deduplicate_prompts",
    "expand_results",
]
