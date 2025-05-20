# BatchGPT

A streamlined interface for OpenAI's batch processing API. This tool allows you to submit, monitor, and manage large batches of requests to OpenAI's models.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         ║
║ │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │    ▲    ║
║ │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │    │    ║
║ │ ▓ ▓▓▓ ▓ │   │ ▓ ▓▓▓ ▓ │   │ ▓ ▓▓▓ ▓ │   │ ▓ ▓▓▓ ▓ │   │ ▓ ▓▓▓ ▓ │    │    ║
║ │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │   │ ▓     ▓ │   Batch  ║
║ │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │   │ ▓▓▓▓▓▓▓ │    │    ║
║ └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘    │    ║
║      │             │             │             │             │          ▼    ║
║      └─────────────┴─────────────┴─────────────┴─────────────┘              ║
║                                    │                                         ║
║                                    ▼                                         ║
║                            ┌───────────────┐                                 ║
║                            │ ╔═══════════╗ │                                 ║
║                            │ ║  OpenAI   ║ │                                 ║
║                            │ ║  Batch    ║ │                                 ║
║                            │ ║  API      ║ │                                 ║
║                            │ ╚═══════════╝ │                                 ║
║                            └───────────────┘                                 ║
║                                    │                                         ║
║                                    ▼                                         ║
║                          ┌─────────────────────┐                             ║
║                          │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │                             ║
║                          │   ▓ Results     ▓   │                             ║
║                          │   ▓ • Processed ▓   │                             ║
║                          │   ▓ • Analyzed  ▓   │                             ║
║                          │   ▓ • Optimized ▓   │                             ║
║                          │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │                             ║
║                          └─────────────────────┘                             ║
╚══════════════════════ B A T C H   G P T   A P I ═════════════════════════════╝
```

## Features

- **Submit batch requests** to OpenAI's API with customizable parameters
- **Monitor batch status** and retrieve results in real-time
- **Process files** (PDF, DOCX, TXT) for batch input
- **Auto-detect and manage** batch jobs
- **Track costs and performance** metrics
- **Beautiful Streamlit UI** for easy interaction
- **WowRunner PDF wrapper** with async CLI for chunked chat completions

## Cost Savings with Batch API

BatchGPT leverages OpenAI's Batch API to significantly reduce costs for large-scale processing. See the pricing comparison below:

### Regular API Pricing
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|------------------------|------------------------|
| o1-pro | $150.00 | $600.00 |
| o1 | $15.00 | $60.00 |
| o3-mini | $1.10 | $4.40 |

### Batch API Pricing (50% Discount)
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|------------------------|------------------------|
| o1-pro | $75.00 | $300.00 |
| o1 | $7.50 | $30.00 |
| o3-mini | $0.55 | $2.20 |

Using the Batch API can save you **50% on token costs** for non-time-sensitive tasks, making it ideal for:
- Large-scale data processing
- Document analysis
- Content generation
- Research data analysis
- Training data preparation

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Directory Structure

- `orchestrator.py` - High level pipeline for PDF processing
- `openai_client.py` - Async OpenAI client with retry and budget control
- `tokenizer.py` - Token counting and chunking utilities
- `pdf_loader.py` - PDF loading and hashing helpers
- `prompt_store.py` - Simple JSONL prompt store
- `postprocessor.py` - Merge model responses
- `cli.py` - Rich command line interface

- `app.py` - Main Streamlit application
- `openai_batch.py` - OpenAI batch API integration
- `batch_manager.py` - Batch job tracking and management
- `file_processor.py` - File processing utilities
- `utils.py` - Helper functions and utilities
- `wowrunner.py` - High level runner using the orchestrator
- `orchestrator.py` - Async coordination of chunked requests
- `openai_client.py` - Thin async wrapper with retries
- `pdf_loader.py` - PDF text extraction helper
- `tokenizer.py` - Token-aware chunking utilities
- `prompt_store.py` - Loads system and user prompts
- `postprocessor.py` - Combines chunked responses
- `cli.py` - Command line interface for WowRunner
- `wowsystem.md` - System prompt
- `wowuser.md` - User prompt template

## WowRunner

Use `cli.py` or the `WowRunner` class to process PDFs with the prompts stored in
`wowsystem.md` and `wowuser.md`.

```bash
python cli.py myfile.pdf --model gpt-4o
```

The combined result for each PDF is printed to stdout.

## Requirements

- Python 3.8+
- OpenAI API key with batch API access
- Required Python packages (see requirements.txt)

## License

MIT License 

## Command Line Interface

A lightweight CLI is provided to process PDFs directly from the terminal.

```bash
python cli.py path/to/file.pdf --model gpt-3.5-turbo --budget 5 --output results.jsonl
```

The CLI loads the PDF, splits it into token sized chunks and streams each chunk
through the OpenAI API using an asynchronous client with basic retry and
throttling. Results are stored in the specified JSONL output file.
