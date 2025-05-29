from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from ..tokenizer import Tokenizer
from logs.logger import logger


def load_pdf(path: str) -> Tuple[str, str]:
    """Load PDF text and return text and hex digest hash."""
    p = Path(path)
    logger.info(f"Loading PDF {path}")
    with p.open("rb") as f:
        data = f.read()
    text = ""
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n\n"
    digest = hashlib.md5(data).hexdigest()
    logger.info(f"Loaded PDF {path} with {len(text)} characters")
    return text, digest


def chunk_pdf(path: str, tokenizer: Tokenizer, max_tokens: int = 2000, overlap: int = 50) -> Tuple[str, List[str], str]:
    """Load, chunk and return the PDF text, chunks and hash."""
    text, digest = load_pdf(path)
    chunks = tokenizer.chunk(text, max_tokens=max_tokens, overlap=overlap)
    logger.info(f"Chunked PDF into {len(chunks)} parts")
    return text, chunks, digest
