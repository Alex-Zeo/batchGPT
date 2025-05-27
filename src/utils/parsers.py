"""Parsing helpers for BatchGPT."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple

from src.logs.logger import logger


def write_jsonl(data: List[Any], file_path: str) -> None:
    """Write a list of objects to a JSONL file.

    Args:
        data: Items to serialize. Dataclasses and pydantic models are supported.
        file_path: Destination file path.
    """
    with open(file_path, "w") as file:
        for item in data:
            if is_dataclass(item) and not isinstance(item, type):
                item = asdict(item)
            elif hasattr(item, "dict"):
                item = item.dict()
            file.write(json.dumps(item) + "\n")
    logger.info(f"Wrote {len(data)} items to {file_path}")


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load objects from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of parsed dictionaries.
    """
    results: List[Dict[str, Any]] = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Read {len(results)} items from {file_path}")
    return results


def generate_hash(content: str) -> str:
    """Return an MD5 hash for the given content.

    Args:
        content: Text to hash.

    Returns:
        MD5 hex digest string.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def deduplicate_prompts(prompts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Deduplicate prompts by content hash.

    Args:
        prompts: List of prompt strings.

    Returns:
        Tuple containing the unique prompts and a mapping of hash to original indices.
    """
    unique_prompts: List[str] = []
    hash_to_indices: Dict[str, List[int]] = {}
    for i, prompt in enumerate(prompts):
        prompt_hash = generate_hash(prompt)
        hash_to_indices.setdefault(prompt_hash, []).append(i)
        if len(hash_to_indices[prompt_hash]) == 1:
            unique_prompts.append(prompt)
    logger.info(
        f"Deduplicated {len(prompts)} prompts to {len(unique_prompts)} unique prompts"
    )
    return unique_prompts, hash_to_indices


def expand_results(
    deduplicated_results: List[Dict[str, Any]],
    hash_to_indices: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    """Expand deduplicated results back to the original order.

    Args:
        deduplicated_results: Results keyed by the first occurrence index.
        hash_to_indices: Mapping from prompt hash to list of original indices.

    Returns:
        Expanded list of results sorted by ``custom_id``.
    """
    expanded: List[Dict[str, Any]] = []
    custom_id_to_result = {r.get("custom_id", ""): r for r in deduplicated_results}
    for prompt_hash, indices in hash_to_indices.items():
        for idx in indices:
            result = custom_id_to_result.get(str(indices[0]), {}).copy()
            result["custom_id"] = str(idx)
            expanded.append(result)
    expanded.sort(key=lambda x: int(x.get("custom_id", "0")))
    return expanded
