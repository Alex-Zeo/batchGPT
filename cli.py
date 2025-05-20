import argparse
import asyncio
from pathlib import Path
from orchestrator import Orchestrator
from postprocessor import merge_results


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
