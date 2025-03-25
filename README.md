# BatchGPT

An intuitive interface for using OpenAI's batch processing API with streamlined monitoring and management capabilities.

## Features

- Submit large batch requests to OpenAI's API
- Monitor batch progress and status
- Supports document processing and text chunking
- Auto-detects compatible batch models
- Clean UI with modern design
- File management for batch inputs and outputs
- Cost estimation tools

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/batchGPT.git
cd batchGPT
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your OpenAI API key.

## Usage

Start the application with:

```bash
streamlit run app.py
```

### Batch Processing

1. Enter your prompts or upload files
2. Select the model and processing parameters
3. Submit the batch
4. Monitor the progress in the batch monitoring panel

### File Processing

- Upload documents (PDF, DOCX, TXT) for processing
- Files are automatically chunked for optimal processing
- Process multiple files in parallel

## Directory Structure

- `app.py`: Main application file
- `openai_batch.py`: OpenAI batch API integration
- `batch_manager.py`: Batch job management
- `file_processor.py`: File processing utilities
- `utils.py`: General utility functions
- `batches/`: Stores batch input/output files
- `logs/`: Application logs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 