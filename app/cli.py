from pathlib import Path

import click

try:
    import rich_click as click  # type: ignore
except Exception:  # pragma: no cover
    import click  # type: ignore  # noqa: F401

from .orchestrator import run_pdf


@click.command()
@click.argument("pdf", type=click.Path(exists=True))
@click.option("--model", default="gpt-3.5-turbo", help="Model name")
@click.option("--budget", type=float, default=None, help="Budget in dollars")
@click.option("--output", type=click.Path(), default=None, help="Output store file")
def main(pdf: str, model: str, budget: float, output: str) -> None:
    """Process a PDF through OpenAI and store results."""
    result = asyncio.run(run_pdf(Path(pdf), model=model, budget=budget, output=output))
    click.echo(result)

import argparse
import asyncio
from pathlib import Path
from .orchestrator import Orchestrator
from .postprocessor import merge_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WowRunner on PDFs")
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF file(s) to process")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--parallelism", type=int, default=5, help="Number of concurrent requests")
    args = parser.parse_args()

    orchestrator = Orchestrator(model=args.model, parallelism=args.parallelism)

    for pdf_path in args.pdf:
        results = asyncio.run(orchestrator.run(pdf_path))
        combined = merge_results(results)
        print(f"--- {pdf_path.name} ---")
        print(combined)


if __name__ == "__main__":
    main()
