import asyncio
from typing import List, Optional

from .openai_batch import run_batch, poll_batch_until_complete
from .utils import (
    format_batch_summary,
    format_batch_results,
    calculate_cost_estimate,
)
from . import prompt_store
from loguru import logger


class WowRunner:
    """High level wrapper for running batches using prompt sources."""

    def __init__(
        self,
        prompt_source: str,
        model: str = "gpt-3.5-turbo",
        budget: Optional[float] = None,
        redact: bool = False,
        glob_pattern: str = "*.txt",
    ) -> None:
        self.prompt_source = prompt_source
        self.model = model
        self.budget = budget
        self.redact = redact
        self.glob_pattern = glob_pattern

    async def run(self) -> None:
        prompts = prompt_store.load_prompts(self.prompt_source, self.glob_pattern)
        logger.info(f"Loaded {len(prompts)} prompts from {self.prompt_source}")
        if self.redact:
            logger.info("Redact mode enabled - prompt contents hidden")
        else:
            for p in prompts[:5]:
                logger.debug(f"Prompt: {p}")

        batch_job = await run_batch(prompts, model=self.model)
        status, results, errors = await poll_batch_until_complete(batch_job.id)
        logger.info(format_batch_summary(status.__dict__))

        if results:
            df = format_batch_results(results)
            cost = calculate_cost_estimate(df)
            logger.info(f"Estimated cost: ${cost['total']:.4f}")
            if self.budget is not None and cost['total'] > self.budget:
                logger.warning(
                    f"Budget exceeded: {cost['total']:.2f} > {self.budget:.2f}"
                )
        if errors:
            logger.error(f"{len(errors)} errors occurred")


if __name__ == "__main__":
    runner = WowRunner("prompts")
    asyncio.run(runner.run())
=======
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
