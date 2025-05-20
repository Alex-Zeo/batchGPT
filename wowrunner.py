import asyncio
from pathlib import Path
from orchestrator import Orchestrator
from postprocessor import merge_results


class WowRunner:
    def __init__(self, model: str = "gpt-4o", parallelism: int = 5):
        self.orchestrator = Orchestrator(model=model, parallelism=parallelism)

    async def run(self, pdf_path: Path) -> str:
        results = await self.orchestrator.run(pdf_path)
        combined = merge_results(results)
        return combined


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Process PDF with WowRunner")
    parser.add_argument("pdf", type=Path, help="PDF file to process")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--parallelism", type=int, default=5, help="Concurrent requests")
    args = parser.parse_args()
    runner = WowRunner(model=args.model, parallelism=args.parallelism)
    result = asyncio.run(runner.run(args.pdf))
    print(result)


if __name__ == "__main__":
    main()
