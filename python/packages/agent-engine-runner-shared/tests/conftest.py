"""Pytest configuration for agent-engine-runner-shared tests."""

import os

import pytest  # noqa: E402

# Explicitly load pytest-asyncio plugin to ensure async tests work
pytest_plugins = ["pytest_asyncio"]


os.environ.setdefault("RUNNER_MODE", "aer")


@pytest.fixture(autouse=True)
def _set_runner_mode(monkeypatch):
    """Default RUNNER_MODE to aer for all tests unless explicitly overridden."""
    if "RUNNER_MODE" not in os.environ:
        monkeypatch.setenv("RUNNER_MODE", "aer")
