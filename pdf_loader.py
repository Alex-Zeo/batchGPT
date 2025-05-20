import hashlib
from pathlib import Path
from typing import Tuple
import pdfplumber


def load_pdf_text(path: Path) -> Tuple[str, str]:
    """Return text and sha256 hash for a PDF."""
    text = ""
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, sha256
