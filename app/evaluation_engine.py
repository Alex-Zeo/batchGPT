"""Evaluation engine using strategy pattern for LLM responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable
import re


@dataclass
class EvaluationResult:
    """Simple result container for evaluations."""

    passed: bool
    score: float = 0.0
    details: str | None = None


class EvaluationStrategy:
    """Base interface for all evaluation strategies."""

    def evaluate(self, response: str, criteria: Dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError


class KeywordEvaluationStrategy(EvaluationStrategy):
    """Check if response contains required keywords."""

    def evaluate(self, response: str, criteria: Dict[str, Any]) -> EvaluationResult:
        keywords = criteria.get("keywords", [])
        if not keywords:
            return EvaluationResult(passed=True, score=1.0, details="no keywords specified")
        found = sum(1 for kw in keywords if kw.lower() in response.lower())
        score = found / len(keywords)
        passed = found == len(keywords)
        return EvaluationResult(passed=passed, score=score, details=f"found {found}/{len(keywords)} keywords")


class RegexEvaluationStrategy(EvaluationStrategy):
    """Validate response against a regular expression."""

    def evaluate(self, response: str, criteria: Dict[str, Any]) -> EvaluationResult:
        pattern = criteria.get("pattern")
        if not pattern:
            return EvaluationResult(passed=True, score=1.0, details="no pattern specified")
        match = re.search(pattern, response)
        return EvaluationResult(passed=bool(match), score=1.0 if match else 0.0, details=f"pattern={pattern}")


class LengthEvaluationStrategy(EvaluationStrategy):
    """Ensure response length falls within a range."""

    def evaluate(self, response: str, criteria: Dict[str, Any]) -> EvaluationResult:
        min_len = criteria.get("min_length", 0)
        max_len = criteria.get("max_length")
        length = len(response)
        if max_len is not None:
            passed = min_len <= length <= max_len
        else:
            passed = length >= min_len
        score = length
        return EvaluationResult(passed=passed, score=score, details=f"length={length}")


class EvaluationEngine:
    """Engine that dispatches evaluation to registered strategies."""

    def __init__(self) -> None:
        self._strategies: Dict[str, EvaluationStrategy] = {}
        self.register_strategy("keyword", KeywordEvaluationStrategy())
        self.register_strategy("regex", RegexEvaluationStrategy())
        self.register_strategy("length", LengthEvaluationStrategy())

    def register_strategy(self, name: str, strategy: EvaluationStrategy) -> None:
        self._strategies[name] = strategy

    def evaluate(self, strategy: str, response: str, criteria: Dict[str, Any]) -> EvaluationResult:
        if strategy not in self._strategies:
            raise ValueError(f"Strategy '{strategy}' not registered")
        return self._strategies[strategy].evaluate(response, criteria)


__all__ = [
    "EvaluationResult",
    "EvaluationStrategy",
    "KeywordEvaluationStrategy",
    "RegexEvaluationStrategy",
    "LengthEvaluationStrategy",
    "EvaluationEngine",
]
