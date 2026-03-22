"""Tests must not call the real Gemini API; clear keys unless explicitly opted in."""

from __future__ import annotations

import os

import pytest

from super_agent.app.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_settings_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("HLAGENT_LIVE_GEMINI_TESTS") == "1":
        yield
        get_settings.cache_clear()
        return
    # Empty env overrides values from `.env` so tests never use a real key.
    monkeypatch.setenv("SUPER_AGENT_GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
