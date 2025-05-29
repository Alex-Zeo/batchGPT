# LLM Batch Processing System - Agent Guidelines

## Project Overview

This is an enterprise-grade LLM batch processing system that handles document evaluation through system and user prompt pairs. The application provides both CLI and Streamlit UI interfaces for professional and enterprise use cases.

## Architecture Principles

### Core Design Patterns
- **Repository Pattern**: Separate data access logic from business logic
- **Factory Pattern**: Use for creating different LLM providers and batch processors
- **Observer Pattern**: Implement for real-time batch status updates
- **Strategy Pattern**: Handle different evaluation strategies and prompt templates
- **Dependency Injection**: Use for testable and modular components

### Project Structure
```
src/
├── app/
│   ├── __init__.py
│   ├── batch_manager.py
│   ├── file_processor.py
│   ├── docreader.py
│   ├── excelreader.py
│   ├── pdfreader.py
│   ├── openai_batch.py
│   ├── orchestrator.py
│   ├── postprocessor.py
│   ├── prompt_store.py
│   ├── tokenizer.py
│   └── evaluation_engine.py
├── interfaces/
│   ├── cli/
│   └── streamlit/
├── logs/
│   ├── __init__.py
│   ├── logger.py
│   ├── cli_05_22_2025_argument1_argument2_argument3.md
│   └── streamlit_05_21_2025_prompt_argument1_argument2.md
├── models/
│   ├── __init__.py
│   ├── batch_request.py
│   ├── evaluation_result.py
│   └── conversation.py
├── prompts/
│   ├── wow_r/
│   ├── wow_nl/
│   └── ideabakery_bh/
├── data/
│   ├── 05_22_2025/
│   ├── 05_21_2025/
│   └── MM_DD_YYYY/
├── services/
│   ├── __init__.py
│   ├── logging_service.py
│   ├── config_service.py
│   └── storage_service.py
├── utils/
│   ├── __init__.py
│   ├── validators.py
│   ├── formatters.py
│   └── parsers.py
└── tests/
    ├── __init__.py
    └── test.py
```

## Project Details

### Core Application (`app/`)
This directory contains the main business logic and processing components:

- **`batch_manager.py`**: Orchestrates batch processing workflows, manages batch lifecycle, and coordinates between different processing stages
- **`file_processor.py`**: Handles file upload, validation, and initial processing of documents before batch creation
- **`docreader.py`**: Base document reader interface and common document processing functionality
- **`excelreader.py`**: Specialized reader for Excel files (.xlsx, .xls), extracting data from spreadsheets for evaluation
- **`pdfreader.py`**: PDF document reader with text extraction, table parsing, and metadata handling
- **`openai_batch.py`**: OpenAI API batch processing client, handles rate limiting, retry logic, and batch submission
- **`orchestrator.py`**: Main workflow coordinator that manages the end-to-end processing pipeline from input to output
- **`postprocessor.py`**: Processes LLM responses, formats results, and generates final evaluation reports
- **`prompt_store.py`**: Manages prompt templates, versioning, and retrieval for different evaluation scenarios
- **`tokenizer.py`**: Token counting and optimization for efficient LLM API usage and cost managemen
- **`evaluation_engine.py`**: Core evaluation logic that applies criteria and scoring to LLM responses

### User Interfaces (`interfaces/`)
Application entry points for different user interaction modes:

- **`cli/`**: Command-line interface implementation with Click framework, providing programmatic access and automation capabilities
- **`interfaces/streamlit/main.py`**: Streamlit application entry point providing the web UI with chat functionality, configuration panels, and real-time batch monitoring

### Logging Infrastructure (`logs/`)
Centralized logging system with structured output:

- **`logger.py`**: Main logging configuration, formatters, and handlers for both CLI and Streamlit interfaces
- **`cli_MM_DD_YYYY_*.md`**: CLI session logs with markdown formatting, containing batch execution details and results
- **`streamlit_MM_DD_YYYY_*.md`**: Streamlit session logs tracking user interactions, configuration changes, and processing history

### Data Models (`models/`)
Pydantic models and dataclasses for type-safe data handling:

- **`batch_request.py`**: Data structures for batch processing requests, including documents, prompts, and evaluation criteria
- **`evaluation_result.py`**: Result models containing LLM responses, scores, metadata, and processing statistics
- **`conversation.py`**: Chat interface models for message threading, session state, and conversation history

### Prompt Management (`prompts/`)
Organized prompt templates for different evaluation scenarios:

- **`wow_r/`**: "World of Work" recruitment-focused prompts and evaluation criteria
- **`wow_nl/`**: "World of Work" natural language processing prompts for document analysis
- **`ideabakery_bh/`**: Specialized prompts for behavioral health and assessment scenarios

### Data Storage (`data/`)
Date-organized storage for batch inputs, outputs, and processing artifacts:

- **`MM_DD_YYYY/`**: Daily folders containing batch inputs, intermediate files, and final results organized by processing date

### Business Services (`services/`)
Core application services and infrastructure:

- **`logging_service.py`**: Centralized logging service with JSON formatting, rotation, and enterprise-grade audit trails
- **`config_service.py`**: Configuration management with environment variable support, validation, and hierarchical settings
- **`storage_service.py`**: File system abstraction for data persistence, backup, and retrieval operations

### Utility Functions (`utils/`)
Reusable components and helper functions:

- **`validators.py`**: Input validation, schema checking, and data integrity verification functions
- **`formatters.py`**: Output formatting utilities for JSON, CSV, Excel, and report generation
- **`parsers.py`**: Document parsing utilities, text extraction, and content preprocessing functions

### Testing Suite (`tests/`)
Comprehensive testing framework:

- **`test.py`**: Main test suite with unit tests, integration tests, and performance benchmarks for all application components

## Code Style and Quality

### Python Standards
- Use **Python 3.10+** with type hints for all functions and classes
- Follow **PEP 8** with 88-character line limit (Black formatter)
- Use **dataclasses** for data models and **Pydantic** for validation
- Implement **async/await** patterns for I/O operations
- Use **pathlib** for all file operations, never os.path

### Import Organization
```python
# Standard library imports
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import pandas as pd
import streamlit as st
from pydantic import BaseModel, validator

# Local imports
from src.core.batch_processor import BatchProcessor
from src.models.batch_request import BatchRequest
```

### Class and Function Design
- **Single Responsibility**: Each class and function should have one clear purpose
- **Immutable Data**: Use frozen dataclasses and avoid mutable defaults
- **Descriptive Names**: Use verbose, self-documenting names
- **Type Safety**: Always use type hints and validate inputs

```python
@dataclass(frozen=True)
class BatchEvaluationRequest:
    """Represents a batch evaluation request with immutable data."""
    batch_id: str
    documents: List[Document]
    system_prompt: str
    user_prompt_template: str
    evaluation_criteria: EvaluationCriteria
    created_at: datetime
```

## Logging Standards

### Centralized Logging Configuration
- Use **structured logging** with JSON format for enterprise environments
- Implement **log rotation** with size and time-based policies
- Store all logs in `./logs/` directory relative to application root
- Create separate log files for CLI and Streamlit but with unified format

### Log Levels and Usage
```python
# ERROR: System failures, critical errors
logger.error("Failed to process batch", extra={
    "batch_id": batch_id,
    "error_type": type(e).__name__,
    "error_message": str(e),
    "stack_trace": traceback.format_exc()
})

# WARNING: Recoverable issues, degraded performance
logger.warning("LLM rate limit encountered", extra={
    "batch_id": batch_id,
    "retry_count": retry_count,
    "wait_time": wait_time
})

# INFO: Business events, batch lifecycle
logger.info("Batch processing started", extra={
    "batch_id": batch_id,
    "document_count": len(documents),
    "user_id": user_id,
    "evaluation_type": evaluation_type
})

# DEBUG: Technical details for troubleshooting
logger.debug("LLM response received", extra={
    "batch_id": batch_id,
    "response_tokens": token_count,
    "processing_time_ms": processing_time
})
```

### Required Log Fields
Every log entry must include:
```python
{
    "timestamp": "2024-01-15T10:30:00.123Z",
    "level": "INFO",
    "logger_name": "batch_processor",
    "message": "Human readable message",
    "batch_id": "uuid-string",
    "session_id": "uuid-string",
    "interface": "cli|streamlit",
    "component": "specific_component_name",
    "user_id": "user_identifier",
    "correlation_id": "request_trace_id"
}
```

## CLI Interface Standards

### Command Structure
- Use **Click** framework for robust CLI implementation
- Implement **progressive disclosure** with sensible defaults
- Provide **verbose** and **quiet** modes
- Support **configuration files** and **environment variables**

### CLI Commands
```python
@click.group()
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--quiet', is_flag=True, help='Suppress output except errors')
@click.pass_context
def cli(ctx, config, log_level, quiet):
    """Enterprise LLM Batch Processing CLI"""
    
@cli.command()
@click.option('--documents', required=True, type=click.Path(exists=True))
@click.option('--system-prompt', required=True)
@click.option('--user-prompt', required=True)
@click.option('--output-format', default='json', type=click.Choice(['json', 'csv', 'xlsx']))
@click.option('--batch-size', default=10, help='Documents per batch')
@click.option('--parallel-batches', default=3, help='Concurrent batch limit')
def process_batch(documents, system_prompt, user_prompt, output_format, batch_size, parallel_batches):
    """Process documents through LLM evaluation pipeline"""
```

### CLI Output Standards
- Use **rich** library for beautiful terminal output
- Implement **progress bars** for long-running operations
- Provide **machine-readable** output formats (JSON, CSV)
- Include **summary statistics** and **performance metrics**

## Streamlit UI Guidelines

### Application Structure
```python
# Main app entry point
def main():
    """Main Streamlit application with proper session state management"""
    initialize_session_state()
    render_sidebar_configuration()
    render_main_interface()
    
def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_batch_id' not in st.session_state:
        st.session_state.current_batch_id = None
```

### UI Component Standards
- Use **semantic HTML** and **accessibility** best practices
- Implement **responsive design** for different screen sizes
- Provide **loading states** and **error boundaries**
- Use **consistent spacing** and **typography** throughout

### Configuration Panel
```python
def render_configuration_panel():
    """Render the configuration sidebar with enterprise settings"""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # LLM Provider Settings
        st.subheader("LLM Provider")
        provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Azure OpenAI"])
        model = st.selectbox("Model", get_available_models(provider))
        
        # Batch Processing Settings
        st.subheader("Batch Settings")
        batch_size = st.slider("Batch Size", 1, 50, 10)
        max_concurrent = st.slider("Max Concurrent Batches", 1, 10, 3)
        
        # Evaluation Settings
        st.subheader("Evaluation Criteria")
        evaluation_type = st.selectbox("Type", ["Quality", "Accuracy", "Compliance"])
        custom_criteria = st.text_area("Custom Criteria", height=100)
```

### Chat Interface Implementation
- Implement **message threading** with proper conversation flow
- Use **streaming responses** for real-time feedback
- Provide **message actions** (copy, regenerate, export)
- Support **file uploads** with validation and preview

### Log Parsing for Conversation History
```python
def load_conversation_history() -> List[Conversation]:
    """Parse logs to reconstruct conversation history by batch_id"""
    log_files = get_log_files_from_directory("./logs/")
    conversations = {}
    
    for log_file in log_files:
        for log_entry in parse_log_file(log_file):
            if log_entry.get('batch_id'):
                batch_id = log_entry['batch_id']
                if batch_id not in conversations:
                    conversations[batch_id] = Conversation(batch_id=batch_id)
                conversations[batch_id].add_log_entry(log_entry)
    
    return list(conversations.values())
```

## Configuration Management

### Configuration Schema
```python
class AppConfig(BaseModel):
    """Application configuration with validation"""
    
    class LLMConfig(BaseModel):
        provider: str = "openai"
        model: str = "gpt-4"
        api_key: str
        max_tokens: int = 4000
        temperature: float = 0.7
        
    class BatchConfig(BaseModel):
        default_batch_size: int = 10
        max_concurrent_batches: int = 5
        retry_attempts: int = 3
        timeout_seconds: int = 300
        
    class LoggingConfig(BaseModel):
        level: str = "INFO"
        format: str = "json"
        rotation_size: str = "100MB"
        retention_days: int = 30
        
    llm: LLMConfig
    batch: BatchConfig
    logging: LoggingConfig
```

### Environment Variables
```bash
# Required
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=openai

# Optional with defaults
LOG_LEVEL=INFO
BATCH_SIZE=10
MAX_CONCURRENT_BATCHES=5
STREAMLIT_PORT=8501
```

## Error Handling and Resilience

### Exception Hierarchy
```python
class BatchProcessingError(Exception):
    """Base exception for batch processing errors"""
    
class LLMProviderError(BatchProcessingError):
    """LLM provider communication errors"""
    
class ValidationError(BatchProcessingError):
    """Input validation errors"""
    
class ConfigurationError(BatchProcessingError):
    """Configuration and setup errors"""
```

### Retry Logic
- Implement **exponential backoff** for LLM API calls
- Use **circuit breaker** pattern for external service failures
- Provide **graceful degradation** when possible
- Log all retry attempts with context

### Rate Limiting
```python
@rate_limit(calls=60, period=60)  # 60 calls per minute
@retry(max_attempts=3, backoff=exponential_backoff)
async def call_llm_api(prompt: str, **kwargs) -> LLMResponse:
    """Call LLM API with rate limiting and retry logic"""
```

## Testing Requirements

### Test Coverage Goals
- **Unit Tests**: 90% coverage minimum
- **Integration Tests**: All API endpoints and batch flows
- **UI Tests**: Critical user journeys in Streamlit
- **Performance Tests**: Batch processing under load

### Test Structure
```python
class TestBatchProcessor:
    """Test batch processing core functionality"""
    
    def setup_method(self):
        self.processor = BatchProcessor(config=test_config)
        self.mock_llm_client = Mock()
        
    async def test_process_batch_success(self):
        """Test successful batch processing"""
        
    async def test_process_batch_with_failures(self):
        """Test batch processing with partial failures"""
        
    def test_batch_size_validation(self):
        """Test batch size limits and validation"""
```

### Mock Data and Fixtures
- Use **factory_boy** for generating test data
- Create **realistic document samples** for testing
- Mock **LLM responses** with various scenarios
- Test **edge cases** and **error conditions**

## Security and Compliance

### Data Protection
- **Never log** API keys, user data, or sensitive information
- Use **environment variables** for all secrets
- Implement **input sanitization** for all user inputs
- Support **data encryption** at rest and in transit

### API Key Management
```python
def get_api_key(provider: str) -> str:
    """Securely retrieve API key from environment"""
    key = os.getenv(f"{provider.upper()}_API_KEY")
    if not key:
        raise ConfigurationError(f"API key not found for provider: {provider}")
    return key
```

### Audit Trail
- Log all **user actions** with timestamps
- Track **document processing** lineage
- Maintain **configuration changes** history
- Enable **compliance reporting** capabilities

## Performance and Scalability

### Batch Processing Optimization
- Use **async/await** for concurrent processing
- Implement **connection pooling** for LLM providers
- Support **batch size optimization** based on content
- Monitor and log **performance metrics**

### Memory Management
```python
def process_large_document_batch(documents: List[Document]) -> Iterator[BatchResult]:
    """Process large batches with memory-efficient streaming"""
    for batch in chunk_documents(documents, batch_size=config.batch_size):
        yield process_document_batch(batch)
        # Memory cleanup between batches
        gc.collect()
```

### Monitoring and Metrics
- Track **batch processing times**
- Monitor **LLM API response times**
- Log **error rates** and **retry counts**
- Report **throughput metrics** and **cost tracking**

## Documentation Standards

### Code Documentation
- Use **Google-style docstrings** for all public functions
- Include **type hints** and **parameter descriptions**
- Provide **usage examples** in docstrings
- Document **error conditions** and **exceptions**

### API Documentation
- Generate **OpenAPI specs** for any REST endpoints
- Include **request/response examples**
- Document **rate limits** and **error codes**
- Provide **integration guides** for enterprise customers

## Development Workflow

### Git Practices
- Use **conventional commits** for clear history
- Implement **branch protection** rules
- Require **code review** for all changes
- Run **automated tests** on all pull requests

### Code Quality Gates
- **Black** for code formatting
- **pylint** and **flake8** for linting
- **mypy** for type checking
- **pytest** for testing with coverage reports

---

This AGENTS.md file establishes enterprise-grade standards for building a robust, scalable, and maintainable LLM batch processing system. Follow these guidelines to ensure code quality, security, and operational excellence. 