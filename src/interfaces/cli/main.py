"""Command-line interface for batchGPT.

This module implements a Click-based CLI following the command structure
outlined in AGENTS.md. It provides a ``process-pdf`` command to run the
OpenAI pipeline on a PDF file. Configuration can be loaded from a JSON
file and logging verbosity can be tuned with ``--verbose`` or
``--quiet``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import click

from src.logs.logger import logger, setup_logger
from src.app.orchestrator import run_pdf


def _load_config(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - unlikely in tests
        logger.error(f"Failed to load config {path}: {exc}")
        return {}


def _configure_logging(verbose: bool, quiet: bool, level: str) -> None:
    if verbose:
        level = "DEBUG"
    if quiet:
        level = "ERROR"
    setup_logger()
    logger.add(sys.stdout, level=level.upper())


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--quiet", is_flag=True, help="Suppress output except errors")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str | None, log_level: str, quiet: bool, verbose: bool) -> None:
    """Enterprise LLM Batch Processing CLI."""
    ctx.obj = {
        "config": _load_config(Path(config) if config else None),
    }
    _configure_logging(verbose, quiet, log_level)


@cli.command("process-pdf")
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default="gpt-3.5-turbo", help="Model name")
@click.option("--budget", type=float, default=None, help="Budget in dollars")
@click.option("--output", type=click.Path(), default=None, help="Output store file")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Machine readable output format",
)
@click.pass_context
def process_pdf(
    ctx: click.Context,
    pdf: str,
    model: str,
    budget: float | None,
    output: str | None,
    output_format: str,
) -> None:
    """Process a single PDF document."""
    conf = ctx.obj.get("config", {})
    model = conf.get("model", model)
    budget = conf.get("budget", budget)
    output = conf.get("output", output)

    result = asyncio.run(
        run_pdf(
            pdf,
            model=model,
            budget=budget,
            output=output,
        )
    )

    if output_format == "json":
        click.echo(json.dumps({"result": result}))
    else:
        click.echo(result)


if __name__ == "__main__":  # pragma: no cover
    cli()
