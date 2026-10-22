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


@pytest.fixture(autouse=True)
def _reset_shared_http_clients():
    """Isolate the process-global, origin-keyed shared client cache per test.

    A client cached by one test (an HTTP client for a fake OE, or an HTTPS
    client built from temp certs) would otherwise be served to the next test
    and could mask the HTTPS fail-closed check.
    """
    from agent_engine_runner_shared.tls_client import close_shared_clients

    close_shared_clients()
    yield
    close_shared_clients()
