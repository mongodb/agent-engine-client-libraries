"""Insurance example customer-info saves when memory is turned off.

The dev insurance fixture runs with memory off, so a save must report that to
the model instead of failing the turn.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from agent_engine_sdk_memory import MemoryNotSupportedError

from insurance_agent import main


class MemoryOff:
    """Memory whose semantic save raises, as the SDK does when memory is off."""

    def save_semantic(self, **kwargs: Any) -> Any:
        raise MemoryNotSupportedError("memory is not enabled for this agent")


def test_save_customer_info_with_memory_off_reports_memory_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(get_current_user_id=lambda: "user-1", memory=MemoryOff())
    monkeypatch.setattr(main, "app", app)

    result = json.loads(main.save_customer_info("name", "Max Marcon"))

    assert result == {"status": "error", "error": "Memory not enabled"}
