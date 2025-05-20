import asyncio
from pathlib import Path

import click

try:
    import rich_click as click  # type: ignore
except Exception:  # pragma: no cover
    import click  # type: ignore  # noqa: F401

from orchestrator import run_pdf


@click.command()
@click.argument("pdf", type=click.Path(exists=True))
@click.option("--model", default="gpt-3.5-turbo", help="Model name")
@click.option("--budget", type=float, default=None, help="Budget in dollars")
@click.option("--output", type=click.Path(), default=None, help="Output store file")
def main(pdf: str, model: str, budget: float, output: str) -> None:
    """Process a PDF through OpenAI and store results."""
    result = asyncio.run(run_pdf(Path(pdf), model=model, budget=budget, output=output))
    click.echo(result)


if __name__ == "__main__":
    main()
