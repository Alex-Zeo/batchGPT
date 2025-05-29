import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from pydantic import BaseModel, ValidationError


class WowResponse(BaseModel):
    summary: str
    key_points: List[str]


SCHEMA_JSON = WowResponse.schema_json(indent=2)


def load_schema(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON schema from `wowsystem.md` if available."""
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "wow_r"
            / "wowsystem.md"
        )
    try:
        file_path = path.resolve()
        with file_path.open("r") as f:
            content = f.read()
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S)
            if match:
                return cast(Dict[str, Any], json.loads(match.group(1)))
    except Exception:
        pass
    return cast(Dict[str, Any], json.loads(SCHEMA_JSON))


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


def merge_chunks(responses: List[Dict[str, Any]]) -> str:
    """Combine responses from chunked completions into a single string."""
    texts = []
    for r in responses:
        choice = r.get("choices", [{}])[0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = msg.get("content", "")
        texts.append(content)
    return "\n".join(texts)


class ChunkResult(BaseModel):
    chunk_index: int
    content: str


def merge_results(results: List[ChunkResult]) -> str:
    ordered = sorted(results, key=lambda r: r.chunk_index)
    return "\n".join(r.content for r in ordered)
