import asyncio
import rich_click as click
from wowrunner import WowRunner


@click.command()
@click.argument('prompt_source')
@click.option('--model', default='gpt-3.5-turbo', help='Model to use')
@click.option('--budget', type=float, help='Budget limit in USD')
@click.option('--redact', is_flag=True, help='Hide prompt contents in logs')
@click.option('--glob', 'glob_pattern', default='*.txt', help='Glob pattern for directories')
def main(prompt_source: str, model: str, budget: float, redact: bool, glob_pattern: str):
    """Run a batch from PROMPT_SOURCE using the WowRunner."""
    runner = WowRunner(prompt_source, model=model, budget=budget, redact=redact, glob_pattern=glob_pattern)
    asyncio.run(runner.run())


if __name__ == '__main__':
    main()

