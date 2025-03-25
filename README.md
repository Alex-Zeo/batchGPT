# BatchGPT

A streamlined interface for OpenAI's batch processing API. This tool allows you to submit, monitor, and manage large batches of requests to OpenAI's models.

## Features

- Submit batch requests to OpenAI's API with customizable parameters
- Monitor batch status and retrieve results
- Process files (PDF, DOCX, TXT) for batch input
- Auto-detect and manage batch jobs
- Track costs and performance metrics
- Beautiful Streamlit UI for easy interaction

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Directory Structure

- `app.py` - Main Streamlit application
- `openai_batch.py` - OpenAI batch API integration
- `batch_manager.py` - Batch job tracking and management
- `file_processor.py` - File processing utilities
- `utils.py` - Helper functions and utilities

## Requirements

- Python 3.8+
- OpenAI API key with batch API access
- Required Python packages (see requirements.txt)

## License

MIT 