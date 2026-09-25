"""TenantRuntime custom-type memory ops delegate to the HTTP MemoryClient.

Unlike the legacy log-and-degrade helpers, custom-type ops raise when memory
is disabled: silent None would break the SDK facade's typed return contract.
"""

from unittest.mock import Mock

import pytest
from agent_engine_sdk_memory.errors import MemoryNotSupportedError

from agent_engine_runner_shared.runtime import TenantRuntime


def _runtime_with_engine(engine) -> TenantRuntime:
    rt = TenantRuntime.__new__(TenantRuntime)
    rt._memory_engine = engine
    return rt


def test_save_custom_delegates() -> None:
    engine = Mock()
    rt = _runtime_with_engine(engine)
    rt.save_custom("tickets", "c", tags={"queue": "billing"})
    engine.create_custom.assert_called_once_with(
        memory_type="tickets",
        content="c",
        tags={"queue": "billing"},
        contextual_metadata=None,
    )


def test_retrieve_custom_delegates() -> None:
    engine = Mock()
    rt = _runtime_with_engine(engine)
    rt.retrieve_custom("tickets", "q", top_k=5)
    engine.retrieve_custom.assert_called_once_with(
        memory_type="tickets",
        query="q",
        tags=None,
        top_k=5,
    )


def test_custom_ops_raise_when_memory_disabled() -> None:
    rt = _runtime_with_engine(None)
    with pytest.raises(MemoryNotSupportedError, match="memory is not enabled"):
        rt.save_custom("tickets", "c")
    with pytest.raises(MemoryNotSupportedError, match="memory is not enabled"):
        rt.retrieve_custom("tickets", "q")
