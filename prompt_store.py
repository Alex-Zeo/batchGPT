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
