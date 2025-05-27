"""Root test aggregator."""

from importlib import import_module


def test_imports() -> None:
    assert import_module("src.app.batch_manager")
