import json
from pathlib import Path
from typing import Dict, List, Any


class PromptStore:
    """Simple JSONL prompt/response storage."""

    def __init__(self, path: str = "store.jsonl") -> None:
        self.path = Path(path)

    def save(self, prompt: str, response: Dict[str, Any]) -> None:
        rec = {"prompt": prompt, "response": response}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
=======
from pathlib import Path
from dataclasses import dataclass

SYSTEM_PROMPT_PATH = Path("wowsystem.md")
USER_PROMPT_PATH = Path("wowuser.md")


@dataclass
class Prompts:
    system: str
    user: str


def load_prompts(system_path: Path = SYSTEM_PROMPT_PATH, user_path: Path = USER_PROMPT_PATH) -> Prompts:
    return Prompts(system=system_path.read_text(), user=user_path.read_text())

