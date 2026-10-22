import pytest

from agent_engine_sdk_adk import adk_compat


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let each test see the once-per-process compatibility warning again."""
    monkeypatch.setattr(adk_compat, "_inactive_workarounds", set())
