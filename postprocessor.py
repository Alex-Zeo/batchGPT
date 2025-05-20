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
