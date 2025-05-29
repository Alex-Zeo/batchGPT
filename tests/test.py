import json
import types
import unittest

import pandas as pd
import pytest
from click.testing import CliRunner

from app import cli as cli_module
from app import logger, setup_logger, openai_batch, prompt_store
from app.batch_manager import BatchManager
from app.evaluation_engine import EvaluationEngine
from app.excelreader import ExcelReader
from app.file_processor import detect_file_type
from utils.validators import sanitize_input, calculate_batch_size
from utils.parsers import generate_hash


# Test BatchManager


def test_add_and_get_batch(tmp_path):
    manager = BatchManager(storage_dir=tmp_path.as_posix())
    manager.add_batch(
        "batch1",
        {"status": "pending", "created_at": "2024-01-01T00:00:00", "model": "gpt"},
    )
    batch = manager.get_batch("batch1")
    assert batch is not None
    assert batch["id"] == "batch1"
    assert batch["status"] == "pending"
    assert batch["model"] == "gpt"


# Test CLI


async def _fake_run_pdf(path: str, model: str = "gpt", budget=None, output=None):
    _fake_run_pdf.called = {
        "path": path,
        "model": model,
        "budget": budget,
        "output": output,
    }
    return "done"


def test_process_pdf_cli_json(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")

    monkeypatch.setattr(cli_module, "run_pdf", _fake_run_pdf)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--quiet", "process-pdf", str(pdf), "--output-format", "json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"result": "done"}
    assert _fake_run_pdf.called["path"] == str(pdf)


def test_process_pdf_with_config(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model": "cfg-model", "budget": 2.0}))

    monkeypatch.setattr(cli_module, "run_pdf", _fake_run_pdf)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--config", str(cfg), "--quiet", "process-pdf", str(pdf)],
    )

    assert result.exit_code == 0
    assert _fake_run_pdf.called["model"] == "cfg-model"
    assert _fake_run_pdf.called["budget"] == 2.0


# Test EvaluationEngine


def test_keyword_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "keyword",
        "The quick brown fox jumps over the lazy dog",
        {"keywords": ["quick", "lazy"]},
    )
    assert result.passed
    assert result.score == 1.0


def test_keyword_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "keyword",
        "A simple sentence",
        {"keywords": ["missing"]},
    )
    assert not result.passed
    assert result.score == 0.0


def test_regex_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "regex",
        "Order ID: 12345",
        {"pattern": r"Order ID: \d+"},
    )
    assert result.passed


def test_regex_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "regex",
        "No numbers here",
        {"pattern": r"\d+"},
    )
    assert not result.passed


def test_length_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "length",
        "abcd",
        {"min_length": 3},
    )
    assert result.passed
    assert result.score == 4


def test_length_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "length",
        "ab",
        {"min_length": 3},
    )
    assert not result.passed


def test_unregistered_strategy():
    engine = EvaluationEngine()
    try:
        engine.evaluate("unknown", "", {})
        assert False, "Expected ValueError"
    except ValueError:
        assert True


# Test ExcelReader


def test_excel_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def fake_read_excel(*args: object, **kwargs: object) -> pd.DataFrame:
        return df

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    reader = ExcelReader()
    records = reader.read("dummy.xlsx")
    assert records == df.to_dict(orient="records")


def test_excel_reader_invalid_extension() -> None:
    reader = ExcelReader()
    with pytest.raises(ValueError):
        reader.read("file.txt")


# Test file type detection


def test_detect_file_type() -> None:
    assert detect_file_type("document.pdf") == "pdf"
    assert detect_file_type("file.docx") == "docx"
    assert detect_file_type("script.py") == "py"
    assert detect_file_type("archive.zip") == "zip"
    assert detect_file_type("index.html") == "code"
    assert detect_file_type("notes.txt") == "txt"
    assert detect_file_type("sheet.xlsx") == "excel"
    assert detect_file_type("book.xls") == "excel"


# Test logger setup


def test_logger_creates_files(tmp_path):
    setup_logger(tmp_path.as_posix())
    logger.info("info")
    logger.warning("warn")
    logger.error("err")
    assert (tmp_path / "info.log").exists()
    assert (tmp_path / "warning.log").exists()
    assert (tmp_path / "error.log").exists()


# Test openai batch utilities


def test_refresh_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert openai_batch.refresh_api_key() is True
    assert openai_batch.openai.api_key == "sk-test"


def test_refresh_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert openai_batch.refresh_api_key() is False


# Test prompt store


def test_load_prompt_files_file(tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("a\nb\n")
    assert prompt_store.load_prompt_files(str(p)) == ["a", "b"]


def test_load_prompt_files_directory(tmp_path):
    d = tmp_path
    (d / "a.txt").write_text("1\n2\n")
    (d / "b.txt").write_text("3\n")
    result = prompt_store.load_prompt_files(d.as_posix())
    assert set(result) == {"1", "2", "3"}


def test_load_prompt_files_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        prompt_store.load_prompt_files(str(missing))


def test_load_prompt_files_s3(monkeypatch):
    data = "x\ny"

    class FakeBody:
        def read(self):
            return data.encode()

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "file.txt"
            return {"Body": FakeBody()}

    def fake_client(name):
        assert name == "s3"
        return FakeS3Client()

    fake_boto3 = types.SimpleNamespace(client=fake_client)
    monkeypatch.setattr(prompt_store, "boto3", fake_boto3)

    prompts = prompt_store.load_prompt_files("s3://bucket/file.txt")
    assert prompts == ["x", "y"]


def test_load_default_prompts(tmp_path):
    system = tmp_path / "sys.md"
    user = tmp_path / "usr.md"
    system.write_text("system")
    user.write_text("user")
    prompts = prompt_store.load_default_prompts(system, user)
    assert prompts.system == "system"
    assert prompts.user == "user"


# Test utility functions


def test_sanitize_input():
    assert sanitize_input("Hello<script>") == "Helloscript"


def test_calculate_batch_size_small():
    assert calculate_batch_size(10, max_batch_size=100) == 10


def test_calculate_batch_size_large():
    assert calculate_batch_size(2500, max_batch_size=5000) == 2500


def test_generate_hash_consistency():
    assert generate_hash("abc") == generate_hash("abc")
    assert generate_hash("abc") != generate_hash("abcd")


class TestUtils(unittest.TestCase):
    def test_sanitize_input_preserves_punctuation(self):
        text = "Hello, world: (test) 1+1=2"
        self.assertEqual(sanitize_input(text), text)

    def test_sanitize_input_removes_nonprintable(self):
        text = "Hello\x00World"
        self.assertEqual(sanitize_input(text), "HelloWorld")


if __name__ == "__main__":
    unittest.main()
