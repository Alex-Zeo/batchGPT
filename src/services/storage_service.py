import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class StorageService:
    """Abstraction layer for reading and writing batch data."""

    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _today_dir(self) -> Path:
        folder = self.base_dir / datetime.now().strftime("%m_%d_%Y")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # Batch JSON
    def save_batch(self, batch_id: str, data: Dict[str, Any]) -> None:
        """Save batch metadata to a JSON file."""
        path = self._today_dir() / f"{batch_id}.json"
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def load_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load batch metadata from disk."""
        for day in self.base_dir.iterdir():
            path = day / f"{batch_id}.json"
            if path.exists():
                with path.open() as f:
                    data: Dict[str, Any] = json.load(f)
                    return data
        return None

    # Results JSONL
    def save_results(self, batch_id: str, results: List[Dict[str, Any]]) -> None:
        """Save batch results to a JSONL file."""
        path = self._today_dir() / f"{batch_id}_results.jsonl"
        with path.open("w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

    def load_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Load batch results from disk."""
        data: List[Dict[str, Any]] = []
        for day in self.base_dir.iterdir():
            path = day / f"{batch_id}_results.jsonl"
            if path.exists():
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                break
        return data
