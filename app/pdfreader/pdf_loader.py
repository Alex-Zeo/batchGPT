from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from ..tokenizer import Tokenizer


def load_pdf(path: str) -> Tuple[str, str]:
    """Load PDF text and return text and hex digest hash."""
    p = Path(path)
    with p.open("rb") as f:
        data = f.read()
    text = ""
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n\n"
    digest = hashlib.md5(data).hexdigest()
    return text, digest


def chunk_pdf(path: str, tokenizer: Tokenizer, max_tokens: int = 2000, overlap: int = 50) -> Tuple[str, List[str], str]:
    """Load, chunk and return the PDF text, chunks and hash."""
    text, digest = load_pdf(path)
    chunks = tokenizer.chunk(text, max_tokens=max_tokens, overlap=overlap)
    return text, chunks, digest
