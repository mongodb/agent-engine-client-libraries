"""Unit tests for ``agent_engine_sdk_adk.runtime._wrap_for_tool_pod``.

The wrapper bridges a user-written tool function and agent-engine-runner-shared's
``create_secure_tool_function``. Two properties matter for the ADK path:

1. **Signature preservation.** ADK introspects every tool via
   ``inspect.signature`` to build the ``FunctionDeclaration`` it sends to the
   LLM. A wrapper that hides the real parameter names produces a useless
   schema and the LLM calls the tool with empty arguments.

2. **Keyword binding from positional args.** ADK occasionally invokes tools
   positionally; agent-engine-runner-shared's ``SecureToolWrapper`` ships args to the OE
   as a dict. The wrapper must convert positional args to kwargs using the
   original signature so the OE always receives a well-shaped payload.

Tests focus on the in-process behavior of the wrapper itself by
monkey-patching ``agent_engine_runner_shared.context.get_current_wrapper`` to return a
``DummyWrapper`` that captures the args agent-engine-runner-shared would forward.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest

import agent_engine_runner_shared.context as runner_context
from agent_engine_sdk_adk.runtime import _wrap_for_tool_pod


class _DummyWrapper:
    """Stand-in for ``SecureToolWrapper`` capturing the args the runner SDK sees."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.tool_call_ids: list[str | None] = []
        self.redact_fields: list[list[str] | None] = []
        self.local_modes: list[bool] = []
        self.local_executors: list[Callable[[], Any] | None] = []

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
        redact_fields: list[str] | None = None,
        is_local: bool = True,
        local_executor: Callable[[], Any] | None = None,
        **_kwargs: Any,
    ) -> str:
        self.executed.append((tool_name, dict(arguments)))
        self.tool_call_ids.append(tool_call_id)
        self.redact_fields.append(redact_fields)
        self.local_modes.append(is_local)
        self.local_executors.append(local_executor)
        return "ok"


@pytest.fixture
def dummy_wrapper(monkeypatch: pytest.MonkeyPatch) -> _DummyWrapper:
    wrapper = _DummyWrapper()
    monkeypatch.setattr(runner_context, "get_current_wrapper", lambda: wrapper)
    return wrapper


def test_signature_preserved_on_wrapper() -> None:
    """The wrapper must expose original parameters via inspect.signature.

    This is the fix for the empty-tool-args bug: without __signature__,
    ADK builds an argument-less FunctionDeclaration and the model never
    sends the real arguments. An injected ``tool_context`` parameter is
    appended so ADK supplies the call context (excluded from the model
    declaration); the real parameters must still come first and unchanged.
    """

    def get_quote(vehicle: str, coverage: str, age: int) -> str:
        return f"{vehicle}/{coverage}/{age}"

    wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)

    sig = inspect.signature(wrapped)
    assert list(sig.parameters.keys()) == ["vehicle", "coverage", "age", "tool_context"]
    original_sig = inspect.signature(get_quote)
    assert sig.parameters["age"].annotation == original_sig.parameters["age"].annotation
    # The injected context param is keyword-only with a default so it is optional.
    ctx = sig.parameters["tool_context"]
    assert ctx.kind == inspect.Parameter.KEYWORD_ONLY
    assert ctx.default is None


def test_function_name_preserved_via_functools_wraps() -> None:
    """ADK keys tools by ``__name__``; the wrapper must not clobber it."""

    def lookup_policy(policy_number: str) -> str:
        return policy_number

    wrapped = _wrap_for_tool_pod(lookup_policy, "lookup_policy", allow_direct=False)
    assert wrapped.__name__ == "lookup_policy"


async def test_keyword_call_reaches_oe(dummy_wrapper: _DummyWrapper) -> None:
    """Pure-kwarg calls (the ADK function-call path) round-trip to the runner SDK."""

    def file_claim(policy_number: str, claim_type: str, claim_amount: float) -> str:
        return "filed"

    wrapped = _wrap_for_tool_pod(file_claim, "file_claim", allow_direct=False)

    result = await wrapped(
        policy_number="POL-1",
        claim_type="collision",
        claim_amount=250.0,
    )

    assert result == "ok"
    assert len(dummy_wrapper.executed) == 1
    tool_name, args = dummy_wrapper.executed[0]
    assert tool_name == "file_claim"
    assert args == {
        "policy_number": "POL-1",
        "claim_type": "collision",
        "claim_amount": 250.0,
    }


async def test_redact_fields_forwarded_to_execute_tool(
    dummy_wrapper: _DummyWrapper,
) -> None:
    def charge_customer(card_number: str) -> str:
        return "charged"

    wrapped = _wrap_for_tool_pod(
        charge_customer,
        "charge_customer",
        allow_direct=False,
        redact_fields=["card_number"],
    )

    await wrapped(card_number="4111-1111")

    assert dummy_wrapper.redact_fields == [["card_number"]]


async def test_local_execution_defaults_to_registered_tool(
    dummy_wrapper: _DummyWrapper,
) -> None:
    def get_quote(vehicle: str) -> str:
        return f"quote:{vehicle}"

    wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)

    await wrapped(vehicle="sedan")

    assert dummy_wrapper.local_modes == [True]
    local_executor = dummy_wrapper.local_executors[0]
    assert local_executor is not None
    assert local_executor() == "quote:sedan"


async def test_async_local_tool_executes_in_adk_worker_thread() -> None:
    from unittest.mock import patch

    from agent_engine_runner_shared.context import (
        clear_execution_context,
        set_execution_context,
    )
    from agent_engine_runner_shared.models import ToolExecuteResponse
    from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper

    async def get_quote(vehicle: str) -> str:
        await asyncio.sleep(0)
        return f"async-quote:{vehicle}"

    wrapper = SecureToolWrapper("http://localhost:8080", "exec-123")
    tokens = set_execution_context("exec-123", wrapper, "http://localhost:8080")
    try:
        wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)
        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=ToolExecuteResponse(proceed=True, route_to="callback"),
        ):
            with patch(
                "agent_engine_runner_shared.secure_wrapper.report_oe_result"
            ) as report:
                result = await wrapped(vehicle="sedan")
    finally:
        clear_execution_context(tokens)

    assert result == "async-quote:sedan"
    assert report.call_args.kwargs["status"] == "success"


async def test_positional_call_bound_to_kwargs(dummy_wrapper: _DummyWrapper) -> None:
    """Positional ADK calls must arrive at the OE as a fully-keyed dict."""

    def get_quote(vehicle: str, coverage: str, age: int) -> str:
        return f"{vehicle}/{coverage}/{age}"

    wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)

    await wrapped("2023 Toyota Camry", "comprehensive", 35)

    assert len(dummy_wrapper.executed) == 1
    _, args = dummy_wrapper.executed[0]
    assert args == {
        "vehicle": "2023 Toyota Camry",
        "coverage": "comprehensive",
        "age": 35,
    }


async def test_mixed_positional_and_keyword_args(dummy_wrapper: _DummyWrapper) -> None:
    """Mixing positional + keyword args (a legal Python call) must work."""

    def get_quote(vehicle: str, coverage: str, age: int) -> str:
        return "_"

    wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)

    await wrapped("2023 Honda CR-V", coverage="standard", age=42)

    _, args = dummy_wrapper.executed[0]
    assert args == {
        "vehicle": "2023 Honda CR-V",
        "coverage": "standard",
        "age": 42,
    }


async def test_function_call_id_forwarded_as_tool_call_id(
    dummy_wrapper: _DummyWrapper,
) -> None:
    """ADK supplies a ToolContext; its function_call_id becomes tool_call_id.

    The context must not leak into the tool arguments sent to the OE.
    """

    class _FakeToolContext:
        function_call_id = "fc-42"

    def file_claim(policy_number: str) -> str:
        return "filed"

    wrapped = _wrap_for_tool_pod(file_claim, "file_claim", allow_direct=False)

    await wrapped(policy_number="POL-1", tool_context=_FakeToolContext())

    tool_name, args = dummy_wrapper.executed[0]
    assert tool_name == "file_claim"
    assert args == {"policy_number": "POL-1"}
    assert dummy_wrapper.tool_call_ids == ["fc-42"]


async def test_missing_tool_context_forwards_none(dummy_wrapper: _DummyWrapper) -> None:
    """Calls without a ToolContext (no id available) forward tool_call_id=None."""

    def file_claim(policy_number: str) -> str:
        return "filed"

    wrapped = _wrap_for_tool_pod(file_claim, "file_claim", allow_direct=False)

    await wrapped(policy_number="POL-1")

    assert dummy_wrapper.tool_call_ids == [None]


async def test_tool_declaring_tool_context_is_not_duplicated(
    dummy_wrapper: _DummyWrapper,
) -> None:
    """A user tool that already declares tool_context must not crash the wrapper.

    Appending a second tool_context would make inspect.Signature.replace raise on
    the duplicate name. The wrapper reuses the existing parameter and still reads
    function_call_id from the injected context.
    """
    from google.adk.tools.tool_context import ToolContext

    def file_claim(policy_number: str, tool_context: ToolContext = None) -> str:  # type: ignore[assignment]
        return "filed"

    wrapped = _wrap_for_tool_pod(file_claim, "file_claim", allow_direct=False)

    # Exactly one tool_context parameter remains.
    names = list(inspect.signature(wrapped).parameters)
    assert names.count("tool_context") == 1

    class _FakeToolContext:
        function_call_id = "fc-77"

    await wrapped(policy_number="POL-1", tool_context=_FakeToolContext())

    tool_name, args = dummy_wrapper.executed[0]
    assert tool_name == "file_claim"
    assert args == {"policy_number": "POL-1"}
    assert dummy_wrapper.tool_call_ids == ["fc-77"]


def test_adk_declaration_excludes_tool_context() -> None:
    """ADK must treat tool_context as injected context, not a model argument.

    If the FunctionDeclaration exposed tool_context, the LLM would try to
    populate it. ADK detects the ToolContext annotation and excludes it.
    """
    from google.adk.tools.function_tool import FunctionTool

    def get_quote(vehicle: str, coverage: str) -> str:
        return f"{vehicle}/{coverage}"

    wrapped = _wrap_for_tool_pod(get_quote, "get_quote", allow_direct=False)
    declaration = FunctionTool(wrapped)._get_declaration()

    assert declaration.parameters_json_schema is not None
    properties = declaration.parameters_json_schema.get("properties", {})
    assert set(properties) == {"vehicle", "coverage"}
    assert "tool_context" not in properties


def test_wrapper_is_coroutine_function() -> None:
    """Guards against a full revert of the async-offload fix.

    google-adk's FunctionTool._invoke_callable dispatches on exactly this
    predicate: a sync callable is invoked inline on the AER's event loop,
    blocking it (and the health endpoint served on the same loop) for the
    full duration of the OE round trip. If anyone reverts ``wrapped`` to a
    plain ``def``, this test fails even though all the call-behavior tests
    above still pass.
    """

    def echo(x: str) -> str:
        return x

    wrapped = _wrap_for_tool_pod(echo, "echo", allow_direct=False)
    assert inspect.iscoroutinefunction(wrapped)


async def test_tool_call_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocking tool call must not park the AER's event loop.

    Stands in for the real sync chain (execute_tool -> request_oe_approval
    -> httpx.Client.post, which holds a blocking socket read for the tool's
    full execution time) with a ``threading.Event``-based gate. The wrapper
    blocks on the worker thread until the watcher sets the event; if the
    wrapper dispatched inline on the event loop (the bug), the watcher
    could never run and the test would time out.
    """

    import threading

    loop_parked = threading.Event()

    class _BlockingWrapper:
        def execute_tool(self, *_args: Any, **_kwargs: Any) -> str:
            loop_parked.wait()  # blocks the worker thread, not the event loop
            return "ok"

    monkeypatch.setattr(
        runner_context, "get_current_wrapper", lambda: _BlockingWrapper()
    )

    def slow_tool() -> str:
        return "ok"

    wrapped = _wrap_for_tool_pod(slow_tool, "slow_tool", allow_direct=False)

    async def watcher() -> str:
        await asyncio.sleep(0)  # yield once to prove the loop is responsive
        loop_parked.set()
        return "health-ok"

    tool_result, watcher_result = await asyncio.wait_for(
        asyncio.gather(wrapped(), watcher()), timeout=5.0
    )

    assert tool_result == "ok"
    assert watcher_result == "health-ok"
