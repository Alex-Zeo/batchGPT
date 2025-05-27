from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from .logger import logger
from .file_processor import extract_text_from_pdf, extract_text_from_docx


class DocReader(ABC):
    """Abstract document reader."""

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load a document from ``path`` and return its bytes."""

    @abstractmethod
    def parse(self, data: bytes, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Parse ``data`` and return text and metadata."""

    def read(self, path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Convenience method to load and parse a document."""
        logger.debug(f"Reading document from {path}")
        data = self.load(path)
        return self.parse(data, **kwargs)


class PDFDocReader(DocReader):
    """Read and parse PDF files using :func:`extract_text_from_pdf`."""

    def load(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def parse(self, data: bytes, *, extract_tables: bool = False, **_: Any) -> Tuple[str, Dict[str, Any]]:
        file_obj = io.BytesIO(data)
        text, meta = extract_text_from_pdf(file_obj, extract_tables)
        return text, meta


class DOCXDocReader(DocReader):
    """Read and parse DOCX files using :func:`extract_text_from_docx`."""

    def load(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def parse(self, data: bytes, **_: Any) -> Tuple[str, Dict[str, Any]]:
        file_obj = io.BytesIO(data)
        text, meta = extract_text_from_docx(file_obj)
        return text, meta


__all__ = ["DocReader", "PDFDocReader", "DOCXDocReader"]
