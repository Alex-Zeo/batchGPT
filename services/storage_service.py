import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageService:
    """Abstraction layer for reading and writing batch data."""

    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # Batch JSON
    def save_batch(self, batch_id: str, data: Dict[str, Any]) -> None:
        """Save batch metadata to a JSON file."""
        path = self.base_dir / f"{batch_id}.json"
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def load_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load batch metadata from disk."""
        path = self.base_dir / f"{batch_id}.json"
        if path.exists():
            with path.open() as f:
                return json.load(f)
        return None

    # Results JSONL
    def save_results(self, batch_id: str, results: List[Dict[str, Any]]) -> None:
        """Save batch results to a JSONL file."""
        path = self.base_dir / f"{batch_id}_results.jsonl"
        with path.open("w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

    def load_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Load batch results from disk."""
        path = self.base_dir / f"{batch_id}_results.jsonl"
        data: List[Dict[str, Any]] = []
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        return data
