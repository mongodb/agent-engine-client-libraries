from __future__ import annotations

import os

import pytest

from agent_engine_sdk_openai_agents import secure_model
from tests.support import FakeWrapper


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> FakeWrapper:
    """Run model calls as if inside an AER execution."""
    fake = FakeWrapper()
    monkeypatch.setattr(secure_model, "get_current_wrapper", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _aer_runtime_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build apps as the AER does unless a test chooses another runtime mode."""
    if "RUNNER_MODE" not in os.environ:
        monkeypatch.setenv("RUNNER_MODE", "aer")
