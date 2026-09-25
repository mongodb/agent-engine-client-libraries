import os

import pytest


@pytest.fixture(autouse=True)
def _set_runner_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default RUNNER_MODE to aer for all tests unless explicitly overridden."""
    if "RUNNER_MODE" not in os.environ:
        monkeypatch.setenv("RUNNER_MODE", "aer")
