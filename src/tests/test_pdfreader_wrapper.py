from src.app.pdfreader import load_pdf, chunk_pdf
from src.app import tokenizer as tokenizer_module
from src.app.tokenizer import Tokenizer
import types


class DummyPDF:
    class Page:
        @staticmethod
        def extract_text() -> str:
            return "dummy"

    pages = [Page()]
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass


def test_load_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"dummy")
    monkeypatch.setattr("src.app.pdfreader_pkg.pdf_loader.pdfplumber.open", lambda p: DummyPDF())
    text, digest = load_pdf(str(pdf))
    assert text.strip() == "dummy"
    assert isinstance(digest, str)


def test_chunk_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"dummy")
    monkeypatch.setattr("src.app.pdfreader_pkg.pdf_loader.pdfplumber.open", lambda p: DummyPDF())
    tokenizer_module.tiktoken = None
    tokenizer = Tokenizer("gpt-3.5-turbo")
    text, chunks, digest = chunk_pdf(str(pdf), tokenizer)
    assert [c.strip() for c in chunks] == ["dummy"]
    assert isinstance(digest, str)
