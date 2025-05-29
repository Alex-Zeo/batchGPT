import glob
from typing import List
from pathlib import Path

from .logger import logger

try:
    import boto3
except ImportError:  # Optional dependency
    boto3 = None


def _load_file(path: str) -> List[str]:
    logger.debug(f"Loading prompt file {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(lines)} prompts from {path}")
    return lines


def load_prompt_files(path: str, glob_pattern: str = "*.txt") -> List[str]:
    """Load prompts from a local file, directory, or S3 path.

    Parameters
    ----------
    path:
        Location of the prompts. Can be a single file, a directory, or an
        ``s3://`` URL.
    glob_pattern:
        Pattern to match files when ``path`` is a directory. Defaults to
        ``"*.txt"``.

    Returns
    -------
    List[str]
        A list of prompt strings loaded from the provided source.
    """
    if path.startswith("s3://"):
        logger.info(f"Loading prompts from S3: {path}")
        if boto3 is None:
            raise ImportError("boto3 is required for S3 support")
        bucket_key = path[5:]
        bucket, _, key = bucket_key.partition("/")
        s3 = boto3.client("s3")
        if key and not key.endswith("/"):
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                data = obj["Body"].read().decode("utf-8")
                lines = [line.strip() for line in data.splitlines() if line.strip()]
                logger.info(f"Loaded {len(lines)} prompts from s3://{bucket}/{key}")
                return lines
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                raise
        else:
            prefix = key
            paginator = s3.get_paginator("list_objects_v2")
            prompts: List[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    try:
                        obj = s3.get_object(Bucket=bucket, Key=item["Key"])
                        data = obj["Body"].read().decode("utf-8")
                        prompts.extend(
                            [line.strip() for line in data.splitlines() if line.strip()]
                        )
                    except Exception as e:
                        logger.error(f"Error loading s3://{bucket}/{item['Key']}: {e}")
            return prompts

    path_obj = Path(path)

    if path_obj.is_dir():
        logger.info(f"Loading prompts from directory {path_obj}")
        prompts: List[str] = []
        files = path_obj.glob(glob_pattern)
        for file_path in files:
            prompts.extend(_load_file(str(file_path)))
        return prompts

    if path_obj.is_file():
        logger.info(f"Loading prompts from file {path_obj}")
        return _load_file(str(path_obj))

    raise FileNotFoundError(f"Prompt source {path} not found")


import json
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


from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "wow_r" / "wowsystem.md"
USER_PROMPT_PATH = BASE_DIR / "prompts" / "wow_r" / "wowuser.md"


@dataclass
class Prompts:
    """Container for the system and user prompt strings."""

    system: str
    user: str


def load_default_prompts(
    system_path: Path = SYSTEM_PROMPT_PATH, user_path: Path = USER_PROMPT_PATH
) -> Prompts:
    """Return the default WowRunner system and user prompts."""
    logger.info(f"Loading default prompts from {system_path} and {user_path}")
    return Prompts(system=system_path.read_text(), user=user_path.read_text())
