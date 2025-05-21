import os
from app import openai_batch


def test_refresh_api_key(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    assert openai_batch.refresh_api_key() is True
    assert openai_batch.openai.api_key == 'sk-test'


def test_refresh_api_key_missing(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert openai_batch.refresh_api_key() is False
