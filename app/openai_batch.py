# mypy: ignore-errors
import openai
import asyncio
import aiohttp
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from .utils import (
    sanitize_input,
    write_jsonl,
    read_jsonl,
    calculate_batch_size,
    validate_api_key,
    estimate_batch_cost,
)
from .logger import logger

from .postprocessor import validate_openai_response

import uuid

BATCHES_DIR = os.path.join(os.path.dirname(__file__), "batches")

# Load environment variables
load_dotenv()

# Function to ensure API key is up to date
def refresh_api_key():
    """Refresh the OpenAI API key from environment variables"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        return True
    return False

# Initialize API key
refresh_api_key()

async def list_available_models() -> List[str]:
    """
    List models available for the current API key using direct API requests.
    Returns a list of model IDs.
    """
    refresh_api_key()  # Ensure key is up to date
    
    # Define fallback models if we can't get the real list
    fallback_models = [
        "o1-mini", "o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", 
        "gpt-3.5-turbo", "gpt-4", "gpt-4-vision-preview", "gpt-4-32k"
    ]
    
    try:
        if not validate_api_key():
            logger.error("Invalid or missing API key")
            return fallback_models
            
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {openai.api_key}"
        }
        
        # Make the API request using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error listing models: {error_text}")
                    return fallback_models
                
                models_data = await response.json()
                
                # Filter to models that are likely to work with batch API
                compatible_models = []
                for model in models_data.get('data', []):
                    model_id = model.get('id', '').lower()
                    # Filter for OpenAI models that work with batch API
                    if any(m in model_id for m in ["o1", "gpt", "text-embedding"]):
                        compatible_models.append(model.get('id'))
                
                if compatible_models:
                    logger.info(f"Found {len(compatible_models)} compatible models")
                    return compatible_models
                else:
                    logger.warning("No compatible models found, using fallback models")
                    return fallback_models
    except aiohttp.ClientError as e:
        logger.error(f"Network error listing models: {str(e)}")
        return fallback_models
    except Exception as e:
        # Check if it's an authentication error
        error_msg = str(e)
        logger.error(f"Error listing models: {error_msg}")
        return fallback_models

async def list_batch_compatible_models() -> List[str]:
    """
    List models that are specifically compatible with the batch API.
    This queries the batch API to determine which models actually work.
    Returns a list of model IDs that are confirmed compatible.
    """
    refresh_api_key()  # Ensure key is up to date
    
    # Define known working models as fallback
    fallback_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    try:
        if not validate_api_key():
            logger.error("Invalid or missing API key")
            return fallback_models
            
        # Get all available models first
        all_models = await list_available_models()
        batch_compatible = []
        
        # Prepare a small test request to check batch compatibility
        test_content = "This is a test message to check batch compatibility."
        test_file_path = os.path.join(BATCHES_DIR, f"batch_test_{uuid.uuid4().hex[:8]}.jsonl")

        os.makedirs(BATCHES_DIR, exist_ok=True)
        
        # Create a minimal test request
        test_request = {
            "custom_id": "test",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": test_content}]
            }
        }
        
        with open(test_file_path, 'w') as f:
            f.write(json.dumps(test_request))
        
        # Upload the test file
        client = openai.OpenAI(api_key=openai.api_key)
        try:
            with open(test_file_path, 'rb') as file:
                file_response = client.files.create(
                    file=file,
                    purpose='batch'
                )
            file_id = file_response.id
            
            # Test which models work with the batch API
            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            }
            
            # Try to get batch system information that might tell us supported models
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/batches/system",
                    headers=headers
                ) as response:
                    # If successful, we might get information about supported models
                    if response.status == 200:
                        system_info = await response.json()
                        if 'supported_models' in system_info:
                            logger.info(f"Batch API reported supported models: {system_info['supported_models']}")
                            return system_info['supported_models']
            
            # Clean up
            os.remove(test_file_path)
            
            # If we couldn't determine from system info, return fallback models
            logger.info(f"Using fallback batch-compatible models: {fallback_models}")
            return fallback_models
            
        except Exception as e:
            logger.error(f"Error testing batch compatibility: {str(e)}")
            # Clean up
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            return fallback_models
            
    except Exception as e:
        logger.error(f"Error determining batch-compatible models: {str(e)}")
        return fallback_models

async def upload_batch_file(file_path: str) -> str:
    """
    Uploads a batch file to OpenAI's batch endpoint.
    Returns the file ID.
    """
    refresh_api_key()  # Ensure key is up to date
    
    if not validate_api_key():
        raise ValueError("Invalid or missing API key. Please check your OpenAI API key.")
    
    logger.info(f"Uploading batch from {file_path}")
    try:
        # Use the synchronous version of files.create with a file path
        # This avoids the "FileObject can't be used in 'await' expression" error
        client = openai.OpenAI(api_key=openai.api_key)
        with open(file_path, 'rb') as file:
            file_response = client.files.create(
                file=file,
                purpose='batch'
            )
        return file_response.id
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error(f"Authentication error: {error_msg}")
            raise ValueError("Invalid API key. Please check your OpenAI API key.") from e
        else:
            logger.error(f"Error uploading batch file: {error_msg}")
            raise

async def create_batch_job(input_file_id: str, model: str, reasoning_effort: str) -> Any:
    """
    Creates a batch job with the specified parameters using direct API requests.
    Returns the batch job object.
    """
    refresh_api_key()  # Ensure key is up to date
    
    logger.info(f"Creating batch job with file ID {input_file_id} and model {model}")
    
    try:
        # Prepare request data
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare the data, but don't include model as it seems to be invalid for batch API
        data = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"reasoning_effort": reasoning_effort}
        }
        
        # Batch API doesn't appear to accept model parameter directly
        # We'll instead use it in the individual requests
        
        # Make the API request using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/batches",
                json=data,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API returned error {response.status}: {error_text}")
                
                batch_data = await response.json()
                logger.info(f"Batch job created with ID: {batch_data.get('id')}")
                
                # Create a simple object to mimic the SDK response
                class BatchResponse:
                    def __init__(self, data):
                        self.id = data.get('id')
                        self.status = data.get('status')
                        self.created_at = data.get('created_at')
                        self.model = model  # Store the model here for reference
                        self.input_file_id = data.get('input_file_id')
                        self.output_file_id = data.get('output_file_id', None)
                        self.error_file_id = data.get('error_file_id', None)
                        
                return BatchResponse(batch_data)
                
    except aiohttp.ClientError as e:
        logger.error(f"Network error creating batch job: {str(e)}")
        raise ValueError(f"Network error: {str(e)}") from e
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error(f"Authentication error: {error_msg}")
            raise ValueError("Invalid API key. Please check your OpenAI API key.") from e
        else:
            logger.error(f"Error creating batch job: {error_msg}")
            raise

async def retrieve_batch_status(batch_id: str) -> Any:
    """
    Retrieves batch job status using direct API requests.
    Returns the batch job object.
    """
    refresh_api_key()  # Ensure key is up to date
    
    logger.info(f"Retrieving batch status for {batch_id}")
    try:
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }
        
        # Make the API request using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.openai.com/v1/batches/{batch_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API returned error {response.status}: {error_text}")
                
                batch_data = await response.json()
                
                # Create a simple object to mimic the SDK response
                class BatchResponse:
                    def __init__(self, data):
                        self.id = data.get('id')
                        self.status = data.get('status')
                        self.created_at = data.get('created_at')
                        self.model = data.get('model')
                        self.input_file_id = data.get('input_file_id')
                        self.output_file_id = data.get('output_file_id', None) 
                        self.error_file_id = data.get('error_file_id', None)
                
                return BatchResponse(batch_data)
                
    except aiohttp.ClientError as e:
        logger.error(f"Network error retrieving batch status: {str(e)}")
        raise ValueError(f"Network error: {str(e)}") from e
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error(f"Authentication error: {error_msg}")
            raise ValueError("Invalid API key. Please check your OpenAI API key.") from e
        else:
            logger.error(f"Error retrieving batch status: {error_msg}")
            raise

async def retrieve_batch_results(batch_status: Any) -> Tuple[List[Dict], List[Dict]]:
    """
    Downloads and processes batch results from completed batch job using direct API requests.
    Returns a tuple of (results, errors).
    """
    refresh_api_key()  # Ensure key is up to date
    
    if batch_status.status != "completed":
        logger.warning(f"Batch {batch_status.id} not completed yet. Status: {batch_status.status}")
        return [], []
    
    results = []
    errors = []
    
    try:
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {openai.api_key}"
        }
        
        # Get successful results
        if batch_status.output_file_id:
            logger.info(f"Downloading output file {batch_status.output_file_id}")
            
            # Download output file content using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.openai.com/v1/files/{batch_status.output_file_id}/content",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error downloading output file: {error_text}")
                    else:
                        output_content = await response.text()
                        
                        # Parse JSONL content
                        for line in output_content.splitlines():
                            if line.strip():
                                item = json.loads(line)
                                if "choices" in item and item["choices"]:
                                    content = item["choices"][0].get("message", {}).get("content", "")
                                    try:
                                        validated = validate_openai_response(content)
                                        item["validated"] = validated.dict()
                                    except Exception as ve:
                                        item["validation_error"] = str(ve)
                                results.append(item)
        
        # Get error results if any
        if batch_status.error_file_id:
            logger.info(f"Downloading error file {batch_status.error_file_id}")
            
            # Download error file content using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.openai.com/v1/files/{batch_status.error_file_id}/content",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error downloading error file: {error_text}")
                    else:
                        error_content = await response.text()
                        
                        # Parse JSONL content
                        for line in error_content.splitlines():
                            if line.strip():
                                errors.append(json.loads(line))
        
        logger.info(f"Retrieved {len(results)} successful results and {len(errors)} errors")
        return results, errors
    except aiohttp.ClientError as e:
        logger.error(f"Network error retrieving batch results: {str(e)}")
        raise ValueError(f"Network error: {str(e)}") from e
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error(f"Authentication error: {error_msg}")
            raise ValueError("Invalid API key. Please check your OpenAI API key.") from e
        else:
            logger.error(f"Error retrieving batch results: {error_msg}")
            raise

async def prepare_batch_requests(
    inputs: List[str], 
    model: str, 
    reasoning_effort: str = "medium",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    response_format: str = "json_object"
) -> List[Dict]:
    """
    Prepare batch requests with the given parameters.
    """
    batch_requests = []
    
    for i, inp in enumerate(inputs):
        request = {
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": sanitize_input(inp)}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        # Add reasoning_effort only for o1 models
        if "o1" in model.lower():
            request["body"]["reasoning_effort"] = reasoning_effort
        
        # Add response_format if json_object is requested
        if response_format == "json_object":
            request["body"]["response_format"] = {"type": "json_object"}
        
        batch_requests.append(request)
    
    return batch_requests

async def run_batch(
    inputs: List[str],
    model: str,
    reasoning_effort: str = "medium",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    response_format: str = "json_object",
    max_batch_size: int = 5000,
    budget: float = None
) -> Any:
    """
    Complete batch lifecycle with dynamic batch sizing.
    If ``budget`` is provided, the run will abort if the predicted cost exceeds
    that value.  Returns the batch job or a list of batch jobs if multiple
    batches were created.
    """
    # Check API key validity before starting
    if not validate_api_key():
        raise ValueError("Invalid or missing API key. Please enter a valid OpenAI API key in the sidebar.")
    
    if not inputs:
        raise ValueError("No inputs provided for batch processing.")

    # Estimate cost and abort if it exceeds the optional budget
    try:
        estimate = estimate_batch_cost(inputs, model, max_tokens)
        logger.info(
            f"Estimated input tokens: {estimate['total_input_tokens']}, "
            f"output tokens: {estimate['total_output_tokens']}"
        )
        logger.info(
            f"Predicted cost for run: ${estimate['total']:.4f}"
        )
        if budget is not None and estimate["total"] > budget:
            raise ValueError(
                f"Estimated cost ${estimate['total']:.4f} exceeds budget ${budget:.2f}"
            )
    except Exception as e:
        # If estimation fails, log the error but continue
        logger.error(f"Cost estimation failed: {str(e)}")

        
    # Validate model - make sure it's supported by OpenAI
    try:
        compatible_models = await list_batch_compatible_models()
        if model not in compatible_models:
            logger.warning(f"Model {model} may not be compatible with batch API. Using gpt-3.5-turbo instead.")
            model = "gpt-3.5-turbo"  # Fallback to a known working model
    except Exception as e:
        logger.warning(f"Could not validate model {model}: {str(e)}")
        
    # Determine if we need to split into multiple batches
    total_requests = len(inputs)
    batch_size = calculate_batch_size(total_requests, max_batch_size)
    
    if batch_size < total_requests:
        logger.info(f"Splitting {total_requests} requests into batches of size {batch_size}")
        batch_jobs = []
        batch_errors = []
        
        # Process each batch
        for i in range(0, total_requests, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            try:
                batch_job = await run_single_batch(
                    batch_inputs, model, reasoning_effort, 
                    temperature, max_tokens, response_format
                )
                batch_jobs.append(batch_job)
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                batch_errors.append({
                    "batch_index": i//batch_size + 1,
                    "start_item": i,
                    "end_item": min(i+batch_size, total_requests),
                    "error": str(e)
                })
        
        if batch_errors:
            if len(batch_jobs) == 0:
                raise ValueError(f"All batches failed: {batch_errors[0]['error']}")
            logger.warning(f"Some batches failed: {len(batch_errors)} errors, {len(batch_jobs)} successful")
            
        return batch_jobs
    else:
        return await run_single_batch(
            inputs, model, reasoning_effort,
            temperature, max_tokens, response_format
        )

async def run_single_batch(
    inputs: List[str], 
    model: str, 
    reasoning_effort: str = "medium",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    response_format: str = "json_object"
) -> Any:
    """
    Run a single batch with the given parameters.
    """
    refresh_api_key()  # Ensure key is up to date
    
    try:
        batch_requests = await prepare_batch_requests(
            inputs, model, reasoning_effort,
            temperature, max_tokens, response_format
        )

        # Log estimated token usage and cost for this batch
        try:
            estimate = estimate_batch_cost(inputs, model, max_tokens)
            logger.info(
                f"Batch estimate - input tokens: {estimate['total_input_tokens']}, "
                f"output tokens: {estimate['total_output_tokens']}, "
                f"cost: ${estimate['total']:.4f}"
            )
        except Exception as e:
            logger.error(f"Cost estimation failed: {str(e)}")

        # Ensure batches directory exists
        os.makedirs(BATCHES_DIR, exist_ok=True)

        # Use a unique filename to avoid collisions
        temp_input_filename = f"batch_input_{uuid.uuid4().hex}.jsonl"
        input_path = os.path.join(BATCHES_DIR, temp_input_filename)
        write_jsonl(batch_requests, input_path)

        try:
            file_id = await upload_batch_file(input_path)
            logger.info(f"Uploaded batch file with ID: {file_id}")
            
            batch_job = await create_batch_job(file_id, model, reasoning_effort)
            logger.info(f"Created batch job with ID: {batch_job.id}")
            
            # Rename the input file to match the batch ID for tracking
            batch_id = batch_job.id
            new_input_path = os.path.join(BATCHES_DIR, f"{batch_id}.jsonl")
            
            try:
                if os.path.exists(input_path):
                    os.rename(input_path, new_input_path)
                    logger.info(f"Renamed batch input file to {new_input_path}")
            except Exception as rename_error:
                logger.warning(f"Could not rename batch input file: {str(rename_error)}")
            
            return batch_job
        except Exception as e:
            error_message = str(e)
            if "401" in error_message:
                raise ValueError("API authentication failed. Please check your API key.") from e
            elif "429" in error_message:
                raise ValueError("Rate limit exceeded. Please try again later or reduce batch size.") from e
            elif "400" in error_message:
                if "model" in error_message.lower():
                    raise ValueError(f"Invalid model: {model}. This model may not be available for batch processing.") from e
                else:
                    raise ValueError(f"Bad request: {error_message}") from e
            else:
                raise
    except Exception as e:
        logger.error(f"Error in run_single_batch: {str(e)}")
        raise

async def poll_batch_until_complete(
    batch_id: str, 
    poll_interval: int = 60,
    timeout_minutes: int = 1440,  # 24 hours max
    on_status_change: callable = None
) -> Tuple[Any, List[Dict], List[Dict]]:
    """
    Poll a batch job until it completes or times out.
    Returns a tuple of (final_status, results, errors).
    
    Args:
        batch_id: The batch ID to poll
        poll_interval: Time in seconds between status checks
        timeout_minutes: Maximum time in minutes to poll before timing out
        on_status_change: Optional callback function called when status changes with (batch_id, status) parameters
    """
    refresh_api_key()  # Ensure key is up to date
    
    start_time = asyncio.get_event_loop().time()
    timeout_seconds = timeout_minutes * 60
    last_status = None
    num_attempts = 0
    max_retries = 5
    retry_delay = 10
    
    # Use the synchronous client for consistency
    client = openai.OpenAI(api_key=openai.api_key)
    
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout_seconds:
            logger.warning(f"Polling timed out after {elapsed/60:.1f} minutes")
            break
        
        try:
            status = await retrieve_batch_status(batch_id)
            num_attempts = 0  # Reset counter on success
            
            # Call status change callback if status changed
            if status.status != last_status and on_status_change:
                try:
                    on_status_change(batch_id, status.status)
                except Exception as callback_error:
                    logger.error(f"Error in status change callback: {str(callback_error)}")
            
            last_status = status.status
            
            if status.status in ["completed", "failed", "expired", "cancelled"]:
                logger.info(f"Batch finished with status: {status.status}")
                results, errors = [], []
                
                if status.status == "completed":
                    try:
                        results, errors = await retrieve_batch_results(status)
                    except Exception as result_error:
                        logger.error(f"Error retrieving results for completed batch: {str(result_error)}")
                
                return status, results, errors
            
            logger.info(f"Batch {batch_id} status: {status.status}, waiting {poll_interval} seconds...")
            await asyncio.sleep(poll_interval)
            
        except Exception as e:
            num_attempts += 1
            error_msg = str(e)
            
            if num_attempts >= max_retries:
                logger.error(f"Failed to poll batch {batch_id} after {max_retries} attempts: {error_msg}")
                break
            
            logger.warning(f"Error polling batch {batch_id} (attempt {num_attempts}/{max_retries}): {error_msg}")
            
            # Use exponential backoff for retries
            backoff_delay = retry_delay * (2 ** (num_attempts - 1))
            logger.info(f"Retrying in {backoff_delay} seconds...")
            await asyncio.sleep(backoff_delay)
    
    # If we got here, we either timed out or had too many errors
    try:
        final_status = await retrieve_batch_status(batch_id)
        return final_status, [], []
    except Exception as e:
        logger.error(f"Could not get final batch status: {str(e)}")
        # Create a minimal status object with the information we have
        class MinimalStatus:
            def __init__(self, batch_id):
                self.id = batch_id
                self.status = "unknown"
                self.created_at = None
                
        return MinimalStatus(batch_id), [], []
