import argparse
import asyncio
from pathlib import Path

from .orchestrator import run_pdf


def main() -> None:
    """Parse arguments and run a PDF through OpenAI."""
    parser = argparse.ArgumentParser(description="Process a PDF with OpenAI")
    parser.add_argument("pdf", type=Path, help="PDF file to process")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--budget", type=float, default=None, help="Budget in dollars")
    parser.add_argument("--output", type=Path, default=None, help="Output store file")
    args = parser.parse_args()

    result = asyncio.run(
        run_pdf(
            args.pdf,
            model=args.model,
            budget=args.budget,
            output=str(args.output) if args.output else None,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
