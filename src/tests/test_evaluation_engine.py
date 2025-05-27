from src.app.evaluation_engine import (
    EvaluationEngine,
    KeywordEvaluationStrategy,
    RegexEvaluationStrategy,
    LengthEvaluationStrategy,
)


def test_keyword_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "keyword",
        "The quick brown fox jumps over the lazy dog",
        {"keywords": ["quick", "lazy"]},
    )
    assert result.passed
    assert result.score == 1.0


def test_keyword_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "keyword",
        "A simple sentence",
        {"keywords": ["missing"]},
    )
    assert not result.passed
    assert result.score == 0.0


def test_regex_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "regex",
        "Order ID: 12345",
        {"pattern": r"Order ID: \d+"},
    )
    assert result.passed


def test_regex_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "regex",
        "No numbers here",
        {"pattern": r"\d+"},
    )
    assert not result.passed


def test_length_strategy_success():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "length",
        "abcd",
        {"min_length": 3},
    )
    assert result.passed
    assert result.score == 4


def test_length_strategy_failure():
    engine = EvaluationEngine()
    result = engine.evaluate(
        "length",
        "ab",
        {"min_length": 3},
    )
    assert not result.passed


def test_unregistered_strategy():
    engine = EvaluationEngine()
    try:
        engine.evaluate("unknown", "", {})
        assert False, "Expected ValueError"
    except ValueError:
        assert True
