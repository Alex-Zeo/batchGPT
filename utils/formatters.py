"""Formatting helpers for BatchGPT."""
from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import pytz  # type: ignore[import-untyped]
import re

from logs.logger import logger


def format_batch_results(results: List[Any]) -> pd.DataFrame:
    """Convert batch result objects into a ``pandas`` DataFrame.

    Args:
        results: List of result objects or dictionaries.

    Returns:
        DataFrame containing normalized result information.
    """
    formatted: List[Dict[str, Any]] = []
    for item in results:
        if is_dataclass(item) and not isinstance(item, type):
            item = asdict(item)
        elif hasattr(item, "dict"):
            item = item.dict()
        entry = {
            "custom_id": item.get("custom_id", ""),
            "status": item.get("status", ""),
            "created_at": item.get("created_at", ""),
            "input_tokens": item.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": item.get("usage", {}).get("completion_tokens", 0),
            "reasoning_tokens": item.get("usage", {}).get("reasoning_tokens", 0),
            "total_tokens": item.get("usage", {}).get("total_tokens", 0),
            "model": item.get("model", ""),
            "content": "",
        }
        choices = item.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            if message:
                entry["content"] = message.get("content", "")
                entry["role"] = message.get("role", "")
        formatted.append(entry)
    return pd.DataFrame(formatted)


def calculate_cost_estimate(df: pd.DataFrame) -> Dict[str, float]:
    """Estimate token usage cost for a dataframe of batch results."""
    model_pricing: Dict[str, Dict[str, float]] = {
        "o1-pro": {"input": 75 / 1_000_000, "output": 300 / 1_000_000, "reasoning": 300 / 1_000_000},
        "o1-mini": {"input": 8 / 1_000_000, "output": 12 / 1_000_000, "reasoning": 12 / 1_000_000},
        "gpt-4o": {"input": 3 / 1_000_000, "output": 15 / 1_000_000, "reasoning": 0},
        "gpt-4": {"input": 15 / 1_000_000, "output": 30 / 1_000_000, "reasoning": 0},
        "gpt-3.5-turbo": {"input": 0.5 / 1_000_000, "output": 1.5 / 1_000_000, "reasoning": 0},
        "default": {"input": 10 / 1_000_000, "output": 30 / 1_000_000, "reasoning": 0},
    }

    cost: Dict[str, Any] = {"input": 0, "output": 0, "reasoning": 0, "total": 0, "by_model": {}}
    for _, row in df.iterrows():
        model = row.get("model", "default").lower()
        price_model = next((model_pricing[k] for k in model_pricing if k in model), model_pricing["default"])

        input_cost = row.get("input_tokens", 0) * price_model["input"]
        output_cost = row.get("output_tokens", 0) * price_model["output"]
        reasoning_cost = row.get("reasoning_tokens", 0) * price_model["reasoning"]

        cost["input"] += input_cost
        cost["output"] += output_cost
        cost["reasoning"] += reasoning_cost

        by_model = cost["by_model"].setdefault(model, {"input": 0, "output": 0, "reasoning": 0, "total": 0})
        by_model["input"] += input_cost
        by_model["output"] += output_cost
        by_model["reasoning"] += reasoning_cost
        by_model["total"] += input_cost + output_cost + reasoning_cost

    cost["total"] = cost["input"] + cost["output"] + cost["reasoning"]
    return cost


def estimate_token_count(text: str) -> int:
    """Very rough token estimator that counts words and punctuation."""
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


def estimate_batch_cost(inputs: List[str], model: str, max_tokens: int) -> Dict[str, Any]:
    """Estimate token usage and dollar cost for a set of prompts."""
    tokens_per_prompt = [estimate_token_count(p) for p in inputs]
    df = pd.DataFrame(
        {
            "model": [model] * len(inputs),
            "input_tokens": tokens_per_prompt,
            "output_tokens": [max_tokens] * len(inputs),
            "reasoning_tokens": [0] * len(inputs),
        }
    )
    cost = cast(Dict[str, Any], calculate_cost_estimate(df))
    cost.update(
        {
            "tokens_per_prompt": tokens_per_prompt,
            "total_input_tokens": int(df["input_tokens"].sum()),
            "total_output_tokens": int(df["output_tokens"].sum()),
        }
    )
    return cost


def format_time_elapsed(start_time: float) -> str:
    """Return a human friendly elapsed time string."""
    elapsed = time.time() - start_time
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    if elapsed < 3600:
        return f"{elapsed / 60:.1f} minutes"
    return f"{elapsed / 3600:.1f} hours"


def generate_unique_id(prefix: str = "batch") -> str:
    """Create a unique identifier with the given prefix."""
    random_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{random_id}"


def format_timestamp(timestamp: Optional[str], format_str: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> str:
    """Convert an ISO timestamp string to the given timezone and format."""
    if not timestamp:
        return "N/A"
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timezone != "UTC":
            tz = pytz.timezone(timezone)
            dt = dt.astimezone(tz)
        return dt.strftime(format_str)
    except Exception:
        return "N/A"


def calculate_completion_time(start_time: str, expected_duration_minutes: int) -> str:
    """Compute the expected completion time based on a start time."""
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        completion_dt = start_dt + timedelta(minutes=expected_duration_minutes)
        return completion_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "Unknown"


def export_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> str:
    """Save batch results to a CSV file and return the path."""
    df = format_batch_results(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(results)} results to {output_path}")
    return output_path


def format_batch_summary(batch_info: Dict[str, Any]) -> str:
    """Create a short textual summary for a batch."""
    status = batch_info.get("status", "Unknown")
    created_at = format_timestamp(batch_info.get("created_at"))
    model = batch_info.get("model", "Unknown")
    results_count = len(batch_info.get("results", []))
    errors_count = len(batch_info.get("errors", []))

    total_requests = results_count + errors_count
    success_rate = (results_count / total_requests * 100) if total_requests else 0

    summary = [
        f"Status: {status}",
        f"Created: {created_at}",
        f"Model: {model}",
        f"Results: {results_count}",
        f"Errors: {errors_count}",
        f"Success Rate: {success_rate:.1f}%",
    ]

    if batch_info.get("results"):
        df = format_batch_results(batch_info["results"])
        total_input = df["input_tokens"].sum()
        total_output = df["output_tokens"].sum()
        total_reasoning = df.get("reasoning_tokens", pd.Series(dtype=int)).sum()
        total_tokens = total_input + total_output + total_reasoning
        summary.extend(
            [
                f"Input Tokens: {total_input:,}",
                f"Output Tokens: {total_output:,}",
                f"Total Tokens: {total_tokens:,}",
            ]
        )
        cost = calculate_cost_estimate(df)
        summary.append(f"Estimated Cost: ${cost['total']:.4f}")

    return "\n".join(summary)
