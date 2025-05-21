import json
import os
import re
from typing import Any, Dict, List
from pydantic import BaseModel, ValidationError


class WowResponse(BaseModel):
    summary: str
    key_points: List[str]


SCHEMA_JSON = WowResponse.schema_json(indent=2)


def load_schema(path: str = None) -> Dict[str, Any]:
    """Load JSON schema from `wowsystem.md` if available."""
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "wow_r", "wowsystem.md"
        )
    try:
        with open(path, "r") as f:
            content = f.read()
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S)
            if match:
                return json.loads(match.group(1))
    except Exception:
        pass
    return json.loads(SCHEMA_JSON)


def validate_openai_response(content: str) -> WowResponse:
    """Parse and validate a JSON string returned by OpenAI."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    try:
        return WowResponse.parse_obj(data)
    except ValidationError as e:
        raise ValueError(f"Response does not match schema: {e}") from e

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
