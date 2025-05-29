import json
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch

from interfaces.cli import main as cli_module


fake_called: dict[str, object] = {}


async def _fake_run_pdf(
    path: str,
    model: str = "gpt",
    budget: float | None = None,
    output: str | None = None,
) -> str:
    fake_called.update(
        {
            "path": path,
            "model": model,
            "budget": budget,
            "output": output,
        }
    )
    return "done"


def test_process_pdf_cli_json(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")

    monkeypatch.setattr(cli_module, "run_pdf", _fake_run_pdf)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["--quiet", "process-pdf", str(pdf), "--output-format", "json"]
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"result": "done"}
    assert fake_called["path"] == str(pdf)


def test_process_pdf_with_config(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_text("x")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model": "cfg-model", "budget": 2.0}))

    monkeypatch.setattr(cli_module, "run_pdf", _fake_run_pdf)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["--config", str(cfg), "--quiet", "process-pdf", str(pdf)]
    )

    assert result.exit_code == 0
    assert fake_called["model"] == "cfg-model"
    assert fake_called["budget"] == 2.0
