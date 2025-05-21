import os
import glob
from typing import List

try:
    import boto3
except ImportError:  # Optional dependency
    boto3 = None


def _load_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_prompts(path: str, glob_pattern: str = "*.txt") -> List[str]:
    """Load prompts from a local file, directory or S3 path."""
    if path.startswith("s3://"):
        if boto3 is None:
            raise ImportError("boto3 is required for S3 support")
        bucket_key = path[5:]
        bucket, _, key = bucket_key.partition('/')
        s3 = boto3.client('s3')
        if key and not key.endswith('/'):
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj['Body'].read().decode('utf-8')
            return [line.strip() for line in data.splitlines() if line.strip()]
        else:
            prefix = key
            paginator = s3.get_paginator('list_objects_v2')
            prompts: List[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get('Contents', []):
                    obj = s3.get_object(Bucket=bucket, Key=item['Key'])
                    data = obj['Body'].read().decode('utf-8')
                    prompts.extend([line.strip() for line in data.splitlines() if line.strip()])
            return prompts

    if os.path.isdir(path):
        prompts: List[str] = []
        files = glob.glob(os.path.join(path, glob_pattern))
        for file_path in files:
            prompts.extend(_load_file(file_path))
        return prompts

    if os.path.isfile(path):
        return _load_file(path)

    raise FileNotFoundError(f"Prompt source {path} not found")

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

from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "wow_r" / "wowsystem.md"
USER_PROMPT_PATH = BASE_DIR / "prompts" / "wow_r" / "wowuser.md"


@dataclass
class Prompts:
    system: str
    user: str


def load_prompts(system_path: Path = SYSTEM_PROMPT_PATH, user_path: Path = USER_PROMPT_PATH) -> Prompts:
    return Prompts(system=system_path.read_text(), user=user_path.read_text())
