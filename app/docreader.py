from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from .logger import logger
from .file_processor import extract_text_from_pdf, extract_text_from_docx


class DocReader(ABC):
    """Base class for document readers.

    Subclasses implement :meth:`load` and :meth:`parse` to return text and
    metadata from various document formats.
    """

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load a document from disk.

        Args:
            path: Path to the document.

        Returns:
            Raw document bytes.
        """

    @abstractmethod
    def parse(self, data: bytes, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Parse ``data`` and return text and metadata.

        Args:
            data: Document bytes.
            **kwargs: Reader specific options.

        Returns:
            Tuple containing extracted text and metadata.
        """

    def read(self, path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Load and parse a document in one call.

        Args:
            path: Path to the document.
            **kwargs: Options forwarded to :meth:`parse`.

        Returns:
            Tuple of extracted text and metadata.
        """
        logger.debug(f"Reading document from {path}")
        data = self.load(path)
        return self.parse(data, **kwargs)


class PDFDocReader(DocReader):
    """Reader for PDF files based on :func:`extract_text_from_pdf`."""

    def load(self, path: str) -> bytes:
        """Load PDF bytes from ``path``.

        Args:
            path: Location of the PDF file.

        Returns:
            The file contents as bytes.
        """
        with open(path, "rb") as f:
            return f.read()

    def parse(self, data: bytes, *, extract_tables: bool = False, **_: Any) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF bytes.

        Args:
            data: PDF data.
            extract_tables: Whether to parse tables in the PDF.

        Returns:
            Extracted text and metadata.
        """
        file_obj = io.BytesIO(data)
        text, meta = extract_text_from_pdf(file_obj, extract_tables)
        return text, meta


class DOCXDocReader(DocReader):
    """Reader for DOCX files based on :func:`extract_text_from_docx`."""

    def load(self, path: str) -> bytes:
        """Load DOCX bytes from ``path``.

        Args:
            path: Path to the DOCX file.

        Returns:
            The raw file contents.
        """
        with open(path, "rb") as f:
            return f.read()

    def parse(self, data: bytes, **_: Any) -> Tuple[str, Dict[str, Any]]:
        """Parse DOCX bytes.

        Args:
            data: DOCX data.

        Returns:
            Extracted text and metadata.
        """
        file_obj = io.BytesIO(data)
        text, meta = extract_text_from_docx(file_obj)
        return text, meta


__all__ = ["DocReader", "PDFDocReader", "DOCXDocReader"]
