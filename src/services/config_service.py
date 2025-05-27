import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values


class ConfigService:
    """Load configuration from environment variables or a file."""

    def __init__(self, config_file: Optional[str] = None) -> None:
        self.config_file = config_file
        self.settings: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Load settings from the environment and optional file."""
        if self.config_file:
            file_path = Path(self.config_file)
            if file_path.exists():
                self.settings.update({k: str(v) for k, v in dotenv_values(file_path).items() if v is not None})
        self.settings.update(os.environ)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a configuration value."""
        return self.settings.get(key, default)
