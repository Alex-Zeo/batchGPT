# mypy: ignore-errors
import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
import hashlib
import csv
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_input(input_str: str) -> str:
    """
    Sanitize input to prevent injection and adhere to OWASP standards.
    """
    sanitized = re.sub(r'[^\w\s,.!?@#%-]', '', input_str)
    logger.info(f"Input sanitized: {sanitized}")
    return sanitized

def write_jsonl(data: List[Dict], file_path: str):
    """
    Writes a list of dictionaries to a JSONL file.
    """
    try:
        with open(file_path, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')
        logger.info(f"Wrote {len(data)} items to {file_path}")
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {str(e)}")
        raise

def read_jsonl(file_path: str) -> List[Dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    results = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    results.append(json.loads(line))
        logger.info(f"Read {len(results)} items from {file_path}")
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {str(e)}")
        raise
    return results

def calculate_batch_size(total_requests: int, max_batch_size: int = 5000) -> int:
    """
    Calculate optimal batch size based on the total number of requests.
    Returns a batch size that is at most max_batch_size.
    """
    if total_requests <= max_batch_size:
        return total_requests
    
    # Find a divisor close to max_batch_size
    for size in range(max_batch_size, 1000, -500):
        if total_requests % size == 0 or total_requests % size < 100:
            return size
    
    return max_batch_size

def validate_api_key() -> bool:
    """
    Validate if the OpenAI API key is set and properly formatted.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API key exists
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return False
    
    # Check basic format for project keys (sk-proj-...) or standard keys (sk-...)
    if not (api_key.startswith("sk-") or api_key.startswith("org-")):
        logger.error("OPENAI_API_KEY appears to be malformed (should start with 'sk-' or 'org-')")
        return False
        
    # Check minimum length for API keys
    if len(api_key) < 40:  # OpenAI keys are typically long
        logger.error("OPENAI_API_KEY appears too short to be valid")
        return False
        
    logger.info("API key format validated")
    return True

def format_batch_results(results: List[Dict]) -> pd.DataFrame:
    """
    Format batch results into a pandas DataFrame for easier analysis.
    """
    formatted_data = []
    
    for item in results:
        entry = {
            "custom_id": item.get("custom_id", ""),
            "status": item.get("status", ""),
            "created_at": item.get("created_at", ""),
            "input_tokens": item.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": item.get("usage", {}).get("completion_tokens", 0),
            "reasoning_tokens": item.get("usage", {}).get("reasoning_tokens", 0),
            "total_tokens": item.get("usage", {}).get("total_tokens", 0),
            "model": item.get("model", ""),
            "content": ""
        }
        
        # Extract response content
        choices = item.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            if message:
                entry["content"] = message.get("content", "")
                entry["role"] = message.get("role", "")
        
        formatted_data.append(entry)
    
    return pd.DataFrame(formatted_data)

def calculate_cost_estimate(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate estimated cost based on token usage.
    These are approximations and rates may change.
    """
    model_pricing = {
        # Pro model pricing with 50% batch discount
        "o1-pro": {"input": 75 / 1_000_000, "output": 300 / 1_000_000, "reasoning": 300 / 1_000_000},
        "o1-mini": {"input": 8 / 1_000_000, "output": 12 / 1_000_000, "reasoning": 12 / 1_000_000},
        "gpt-4o": {"input": 3 / 1_000_000, "output": 15 / 1_000_000, "reasoning": 0},
        "gpt-4": {"input": 15 / 1_000_000, "output": 30 / 1_000_000, "reasoning": 0},
        "gpt-3.5-turbo": {"input": 0.5 / 1_000_000, "output": 1.5 / 1_000_000, "reasoning": 0},
        # Default pricing in case model is unknown
        "default": {"input": 10 / 1_000_000, "output": 30 / 1_000_000, "reasoning": 0}
    }
    
    cost = {"input": 0, "output": 0, "reasoning": 0, "total": 0, "by_model": {}}
    
    for _, row in df.iterrows():
        model = row["model"].lower() if "model" in row and row["model"] else "default"
        
        # Find the right pricing model
        price_model = None
        for key in model_pricing:
            if key in model:
                price_model = model_pricing[key]
                break
        
        if not price_model:
            price_model = model_pricing["default"]
        
        # Calculate costs
        input_cost = row["input_tokens"] * price_model["input"]
        output_cost = row["output_tokens"] * price_model["output"]
        reasoning_cost = row["reasoning_tokens"] * price_model["reasoning"] if "reasoning_tokens" in row else 0
        
        cost["input"] += input_cost
        cost["output"] += output_cost
        cost["reasoning"] += reasoning_cost
        
        # Track by model
        if model not in cost["by_model"]:
            cost["by_model"][model] = {"input": 0, "output": 0, "reasoning": 0, "total": 0}
        
        cost["by_model"][model]["input"] += input_cost
        cost["by_model"][model]["output"] += output_cost
        cost["by_model"][model]["reasoning"] += reasoning_cost
        cost["by_model"][model]["total"] += input_cost + output_cost + reasoning_cost
    
    cost["total"] = cost["input"] + cost["output"] + cost["reasoning"]
    return cost

def is_valid_poll_interval(interval: str) -> Optional[int]:
    """
    Validate and convert poll interval string to seconds.
    Returns None if invalid, integer seconds if valid.
    
    Accepts formats like "30s", "5m", "1h"
    """
    if not interval:
        return None
    
    pattern = r'^(\d+)([smh])$'
    match = re.match(pattern, interval.lower())
    
    if not match:
        return None
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    
    return None

def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time since start_time in a human-readable format.
    """
    elapsed = time.time() - start_time
    
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    elif elapsed < 3600:
        return f"{elapsed/60:.1f} minutes"
    else:
        return f"{elapsed/3600:.1f} hours"

def generate_unique_id(prefix: str = "batch") -> str:
    """
    Generate a unique ID for tracking purposes with a given prefix.
    Uses UUID4 to ensure uniqueness.
    """
    random_id = uuid.uuid4().hex[:12]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{random_id}"

def format_timestamp(timestamp: Optional[str], 
                    format_str: str = "%Y-%m-%d %H:%M:%S", 
                    timezone: str = "UTC") -> str:
    """
    Format an ISO timestamp to a more readable format.
    If timestamp is None or invalid, returns "N/A".
    """
    if not timestamp:
        return "N/A"
    
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timezone != "UTC":
            # Convert to the specified timezone
            tz = pytz.timezone(timezone)
            dt = dt.astimezone(tz)
        
        return dt.strftime(format_str)
    except Exception:
        return "N/A"

def calculate_completion_time(start_time: str, expected_duration_minutes: int) -> str:
    """
    Calculate the expected completion time based on start time and duration.
    Returns a formatted timestamp string.
    """
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        completion_dt = start_dt + timedelta(minutes=expected_duration_minutes)
        return completion_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "Unknown"

def export_results_to_csv(results: List[Dict], output_path: str) -> str:
    """
    Export batch results to CSV file.
    Returns the path to the created file.
    """
    try:
        df = format_batch_results(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(results)} results to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting results to CSV: {str(e)}")
        raise

def generate_hash(content: str) -> str:
    """
    Generate a hash of the content for deduplication purposes.
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def deduplicate_prompts(prompts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Deduplicate a list of prompts by content hash.
    Returns the deduplicated prompts and a mapping of hash to original indices.
    """
    unique_prompts = []
    hash_to_indices = {}
    indices_to_hash = {}
    
    for i, prompt in enumerate(prompts):
        prompt_hash = generate_hash(prompt)
        
        if prompt_hash not in hash_to_indices:
            hash_to_indices[prompt_hash] = [i]
            indices_to_hash[i] = prompt_hash
            unique_prompts.append(prompt)
        else:
            hash_to_indices[prompt_hash].append(i)
            indices_to_hash[i] = prompt_hash
    
    logger.info(f"Deduplicated {len(prompts)} prompts to {len(unique_prompts)} unique prompts")
    return unique_prompts, hash_to_indices

def expand_results(deduplicated_results: List[Dict], hash_to_indices: Dict[str, List[int]]) -> List[Dict]:
    """
    Expand deduplicated results back to the original size based on the hash mapping.
    """
    expanded_results = []
    
    # Create a mapping from custom_id to result
    custom_id_to_result = {r.get("custom_id", ""): r for r in deduplicated_results}
    
    # Expand results based on the hash mapping
    for prompt_hash, indices in hash_to_indices.items():
        for idx in indices:
            # Find the corresponding result
            result = custom_id_to_result.get(str(indices[0]), {})
            
            # Create a copy with the updated custom_id
            result_copy = result.copy()
            result_copy["custom_id"] = str(idx)
            expanded_results.append(result_copy)
    
    # Sort by custom_id to maintain original order
    expanded_results.sort(key=lambda x: int(x.get("custom_id", "0")))
    
    return expanded_results

def format_batch_summary(batch_info: Dict) -> str:
    """
    Format batch information into a user-friendly summary string.
    """
    status = batch_info.get("status", "Unknown")
    created_at = format_timestamp(batch_info.get("created_at"))
    model = batch_info.get("model", "Unknown")
    results_count = len(batch_info.get("results", []))
    errors_count = len(batch_info.get("errors", []))
    
    # Calculate success rate
    total_requests = results_count + errors_count
    success_rate = (results_count / total_requests * 100) if total_requests > 0 else 0
    
    summary = [
        f"Status: {status}",
        f"Created: {created_at}",
        f"Model: {model}",
        f"Results: {results_count}",
        f"Errors: {errors_count}",
        f"Success Rate: {success_rate:.1f}%"
    ]
    
    # Add token usage if available
    if batch_info.get("results"):
        df = format_batch_results(batch_info["results"])
        total_input = df["input_tokens"].sum()
        total_output = df["output_tokens"].sum()
        total_reasoning = df["reasoning_tokens"].sum() if "reasoning_tokens" in df.columns else 0
        total_tokens = total_input + total_output + total_reasoning
        
        summary.extend([
            f"Input Tokens: {total_input:,}",
            f"Output Tokens: {total_output:,}",
            f"Total Tokens: {total_tokens:,}"
        ])
        
        # Add cost estimate
        cost = calculate_cost_estimate(df)
        summary.append(f"Estimated Cost: ${cost['total']:.4f}")
    
    return "\n".join(summary)
