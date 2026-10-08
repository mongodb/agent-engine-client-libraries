from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pytest
from agent_engine_sdk import (
    AgentInput,
    LLMResponse,
    LLMToolCall,
    RequestContext,
)
from agents import (
    Agent,
    AgentHooks,
    Handoff,
    Model,
    ModelSettings,
    RunConfig,
    RunHooks,
    SQLiteSession,
    ToolApprovalItem,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    function_tool,
    handoff,
)
from agents.exceptions import ModelBehaviorError
from agents.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
from agents.mcp import MCPServerStdio
from agents.result import RunResultStreaming
from agents.tool import DEFAULT_APPROVAL_REJECTION_MESSAGE
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall
from openai.types.shared import Reasoning
from pydantic import BaseModel

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    FinalizeStepCommand,
    StateSnapshot,
)
from agent_engine_runner_shared.hooks import get_named_llm
from agent_engine_runner_shared.secure_wrapper import (
    CALL_INTERRUPTED_ARTIFACT_KEY,
    INTERRUPTED_CALL_CONTENT,
    ToolExecutionError,
)
from agent_engine_runner_shared.workflow.activity import semantic_input_from_json
from agent_engine_runner_shared.workflow.context import (
    allocate_activity_ordinal,
    attempt_context_scope,
    current_operation_path,
    tool_activity_key,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_value,
    proto_value_to_json,
)
from agent_engine_sdk_openai_agents import (
    App,
    DurableOpenAIAgentsStateError,
    UnsupportedDurableOpenAIAgentsError,
)
from agent_engine_sdk_openai_agents import agent as agent_module
from agent_engine_sdk_openai_agents import approvals as approvals_module
from agent_engine_sdk_openai_agents import tools as tools_module
from agent_engine_sdk_openai_agents.llm_adapter import RegisteredModel
from agent_engine_sdk_openai_agents.secure_model import (
    _runner_response,  # pyright: ignore[reportPrivateUsage]
)
from agent_engine_sdk_openai_agents.state import (
    decode_active_agent,
    decode_state,
    encode_state,
)
from agent_engine_sdk_openai_agents.tools import json_text
from tests.support import (
    FakeWrapper,
    ScriptedNativeModel,
    function_call_item,
    route_to_tool_pod,
    text_item,
)


class Answer(BaseModel):
    status: str


class OE:
    """The attempt, tool dispatch, and settlement OE provides to a turn."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.previous_state: StateSnapshot | None = None
        self.settled: list[StateSnapshot] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        # The operation path each tool call ran under, by call id.
        self.tool_paths: dict[str, list[tuple[str, int]]] = {}
        # Call ids OE stops mid-flight, as when a user stops a running call.
        self.stopped_call_ids: set[str] = set()
        monkeypatch.setattr(agent_module, "current_attempt_context", self.attempt)
        monkeypatch.setattr(agent_module, "settle_execution", self.settle)
        monkeypatch.setattr(tools_module, "create_secure_tool_function", self.secure)
        # Approval waits: OE's suspension frontier, keyed by position, holding
        # each wait's answer once a caller resumes it.
        self.waits: dict[str, tuple[str, dict[str, Any] | None]] = {}
        oe = self

        class Client:
            def __init__(self, _url: str) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_exc: object) -> None:
                return None

            async def finalize_step(
                self, command: FinalizeStepCommand
            ) -> list[StepActivityEntry]:
                return [oe.frontier(entry.position) for entry in command.suspensions]

        monkeypatch.setattr(approvals_module, "AsyncWorkflowClient", Client)
        monkeypatch.setattr(approvals_module, "get_current_oe_url", lambda: "http://oe")

    def frontier(self, position: Any) -> StepActivityEntry:
        key = position.SerializeToString(deterministic=True).hex()
        activity_id, answer = self.waits.setdefault(
            key, (f"wait-{len(self.waits) + 1}", None)
        )
        outcome = ActivityOutcome(
            activity_id=activity_id,
            workflow_identity=WorkflowIdentity(session_id="s-1", execution_id="exec-1"),
            attempt_id="attempt-2",
            fencing_token=7,
        )
        if answer is None:
            outcome.outcome_kind = ACTIVITY_OUTCOME_KIND_SUSPENDED
        else:
            outcome.outcome_kind = ACTIVITY_OUTCOME_KIND_COMPLETED
            outcome.result.CopyFrom(semantic_input_from_json(answer))
        return StepActivityEntry(position=position, outcome=outcome)

    def answer(self, activity_id: str, answer: dict[str, Any]) -> None:
        """What /invoke with resume_map does: record the answer for the wait."""
        key = next(
            key for key, (wait_id, _) in self.waits.items() if wait_id == activity_id
        )
        self.waits[key] = (activity_id, answer)

    def attempt(self) -> AttemptContext:
        attempt = AttemptContext(
            workflow_identity=WorkflowIdentity(session_id="s-1", execution_id="exec-1")
        )
        if self.previous_state is not None:
            attempt.previous_state.CopyFrom(self.previous_state)
        return attempt

    async def settle(self, attempt: AttemptContext, state: StateSnapshot) -> None:
        self.settled.append(state)

    def secure(
        self, *, original_tool: Any, response_format: str, **_policy: Any
    ) -> Any:
        # The real wrapper's content_and_artifact contract: (content, artifact),
        # where only OE sets the stopped-call marker.
        assert response_format == "content_and_artifact"

        def call(*, tool_call_id: str, **kwargs: Any) -> Any:
            self.tool_calls.append((tool_call_id, kwargs))
            self.tool_paths[tool_call_id] = [
                (segment.name, segment.ordinal)
                for segment in current_operation_path().segments
            ]
            if tool_call_id in self.stopped_call_ids:
                return INTERRUPTED_CALL_CONTENT, {CALL_INTERRUPTED_ARTIFACT_KEY: True}
            try:
                result = original_tool(**kwargs)
                if inspect.iscoroutine(result):
                    # The real wrapper runs an async tool to completion on
                    # its worker thread.
                    result = asyncio.run(result)
            except Exception as error:
                # The real wrapper reports a failed local tool this way.
                raise ToolExecutionError(str(error)) from error
            # Like the real durable wrapper, record only JSON tool results.
            json.dumps(result, allow_nan=False)
            return result, None

        return call


def _order_app(native: ScriptedNativeModel, **agent_options: Any) -> App:
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> dict[str, str]:
        """Look up an order."""
        return {"order_id": order_id, "status": "shipped"}

    @app.entrypoint
    def build() -> Agent[Any]:
        options: dict[str, Any] = {
            "name": "support",
            "instructions": "Answer order questions.",
            "model": app.llm(native, settings=ModelSettings(temperature=0)),
            "tools": app.tools(),
            **agent_options,
        }
        return Agent(**options)

    return app


async def _turn(app: App, message: str, oe: OE) -> list[Any]:
    """Run one turn, checking OE settled the turn before its result was shown."""
    agent = app.get_agent()
    ctx = RequestContext(session_id="s-1", user_id="u-1")
    events: list[Any] = []
    settled_before = len(oe.settled)
    async for event in agent.stream(ctx, AgentInput(payload={"message": message})):
        if event.event == "result":
            assert len(oe.settled) == settled_before + 1
        events.append(event)
    return events


@pytest.fixture
def oe(monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper) -> OE:
    route_to_tool_pod(monkeypatch)
    return OE(monkeypatch)


async def test_turn_runs_a_tool_and_commits_before_its_result(oe: OE) -> None:
    native = ScriptedNativeModel(
        [function_call_item("call-1")], [text_item('{"status":"shipped"}')]
    )
    app = _order_app(native, output_type=Answer)

    events = await _turn(app, "Where is A1?", oe)

    tokens = [event.data["content"] for event in events if event.event == "token"]
    (result,) = [event.data for event in events if event.event == "result"]
    assert tokens == ['{"status":"shipped"}']
    assert result["response"] == '{"status":"shipped"}'
    assert oe.tool_calls == [("call-1", {"order_id": "A1"})]
    assert [m["role"] for m in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["messages"][2]["content"] == '{"order_id":"A1","status":"shipped"}'
    # Registered provider settings reach the Tool Pod model.
    assert native.calls[0]["settings"].temperature == 0
    (snapshot,) = oe.settled
    attempt = AttemptContext(workflow_identity=WorkflowIdentity(session_id="s-1"))
    attempt.previous_state.CopyFrom(snapshot)
    committed = decode_state(attempt, app_name="support")
    assert [item.raw.get("call_id") for item in committed][1:3] == ["call-1", "call-1"]


async def test_parallel_tool_calls_keep_call_order_whatever_finishes_first(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real proxy preallocates tool ordinals in the model's order; tools
    # must run where that allocation applies, however they are scheduled.
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    ordinals: dict[str, int] = {}
    call_2_identified = threading.Event()

    def secure(**kwargs: Any) -> Any:
        call = oe.secure(**kwargs)

        def identified(*, tool_call_id: str, **arguments: Any) -> Any:
            # call-2 asks for its identity first, so arrival order would give
            # it the first ordinal.
            if tool_call_id == "call-1":
                assert call_2_identified.wait(timeout=5)
            ordinals[tool_call_id] = allocate_activity_ordinal(
                tool_activity_key(tool_call_id)
            )
            call_2_identified.set()
            return call(tool_call_id=tool_call_id, **arguments)

        return identified

    monkeypatch.setattr(tools_module, "create_secure_tool_function", secure)
    finished: list[str] = []
    a2_finished = threading.Event()
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        if order_id == "A1":
            # A1's warehouse is slow: it answers only after A2 has.
            assert a2_finished.wait(timeout=5), "the calls did not run concurrently"
        finished.append(order_id)
        if order_id == "A2":
            a2_finished.set()
        return f"{order_id} shipped"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    native = ScriptedNativeModel(
        [
            function_call_item("call-1", '{"order_id":"A1"}'),
            function_call_item("call-2", '{"order_id":"A2"}'),
        ],
        [text_item("Both shipped.")],
    )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert events[-1].event == "result"
    assert finished == ["A2", "A1"]
    assert ordinals == {"call-1": 1, "call-2": 2}
    calls_then_outputs = [
        (item["type"], item["call_id"], item.get("output"))
        for item in native.calls[1]["input"][1:]
    ]
    assert calls_then_outputs == [
        ("function_call", "call-1", None),
        ("function_call", "call-2", None),
        ("function_call_output", "call-1", "A1 shipped"),
        ("function_call_output", "call-2", "A2 shipped"),
    ]
    oe.previous_state = oe.settled[0]
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [(item.get("type"), item.get("call_id")) for item in committed[1:5]] == [
        (call_type, call_id) for call_type, call_id, _ in calls_then_outputs
    ]


async def test_drain_waits_for_a_call_tracked_while_it_is_draining() -> None:
    loop = asyncio.get_running_loop()
    first: asyncio.Future[Any] = loop.create_future()
    late: asyncio.Future[Any] = loop.create_future()
    calls = tools_module.TurnContext()
    calls.track(first)

    draining = asyncio.create_task(calls.drain())
    await asyncio.sleep(0)
    calls.track(late)
    first.set_result(None)
    for _ in range(5):
        await asyncio.sleep(0)

    # The first call is done; the turn still may not end.
    assert not draining.done()
    late.set_result(None)
    await asyncio.wait_for(draining, timeout=5)


async def test_failed_parallel_turn_waits_for_its_running_sibling(oe: OE) -> None:
    finished: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        if order_id == "A1":
            raise RuntimeError("inventory service unavailable")
        time.sleep(0.3)
        finished.append(order_id)
        return f"{order_id} shipped"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    native = ScriptedNativeModel(
        [
            function_call_item("call-1", '{"order_id":"A1"}'),
            function_call_item("call-2", '{"order_id":"A2"}'),
        ]
    )

    with pytest.raises(ToolExecutionError, match="inventory service unavailable"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    # The sibling's worker thread cannot be interrupted; the turn ends only
    # once it has, so no tool work outlives the failed attempt.
    assert finished == ["A2"] and oe.settled == []


async def test_cancelled_turn_still_waits_for_its_running_tool(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    draining = asyncio.Event()
    drain = tools_module.TurnContext.drain

    async def signalled_drain(self: tools_module.TurnContext) -> None:
        draining.set()
        await drain(self)

    monkeypatch.setattr(tools_module.TurnContext, "drain", signalled_drain)
    finished: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        if order_id == "A1":
            assert started.wait(timeout=5)
            raise RuntimeError("inventory service unavailable")
        started.set()
        assert release.wait(timeout=5)
        finished.append(order_id)
        return f"{order_id} shipped"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    native = ScriptedNativeModel(
        [
            function_call_item("call-1", '{"order_id":"A1"}'),
            function_call_item("call-2", '{"order_id":"A2"}'),
        ]
    )
    turn = asyncio.create_task(_attempt(app, oe, RequestContext(session_id="s-1")))
    try:
        # A1 has failed and the turn is waiting for A2, which is still running.
        await asyncio.wait_for(draining.wait(), timeout=5)
        turn.cancel()
        await asyncio.sleep(0.1)
        # Cancelling the wait does not end the turn while A2 still runs.
        assert not turn.done() and finished == []
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert finished == ["A2"] and oe.settled == []


async def test_later_turn_continues_from_the_committed_conversation(oe: OE) -> None:
    native = ScriptedNativeModel([text_item("Hello.")], [text_item("You said hi.")])
    app = _order_app(native)

    await _turn(app, "hi", oe)
    oe.previous_state = oe.settled[0]
    events = await _turn(app, "what did I say?", oe)

    second_input = native.calls[1]["input"]
    assert [item.get("content") for item in second_input] == [
        "hi",
        "Hello.",
        "what did I say?",
    ]
    (result,) = [event.data for event in events if event.event == "result"]
    assert [m["content"] for m in result["messages"]] == [
        "what did I say?",
        "You said hi.",
    ]


async def test_failed_tool_fails_the_turn_without_committing(oe: OE) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise RuntimeError("inventory service unavailable")

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    # The secure path's error reaches the AER as is, not re-wrapped by the SDK.
    with pytest.raises(ToolExecutionError, match="inventory service unavailable"):
        await _turn(app, "Where is A1?", oe)
    assert oe.settled == []
    assert len(native.calls) == 1  # the model never saw the failure as tool output


class _UnavailableModel(Model):
    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Tool Pod adapter streams")

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("provider unavailable")
        yield


async def test_failed_model_call_fails_the_turn_without_committing(oe: OE) -> None:
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(_UnavailableModel()))

    agent = app.get_agent()
    events: list[Any] = []
    with pytest.raises(RuntimeError, match="provider unavailable"):
        async for event in agent.stream(
            RequestContext(), AgentInput(payload={"message": "hi"})
        ):
            events.append(event)
    assert oe.settled == []
    assert [event.event for event in events] == []


async def test_stopped_tool_call_fails_the_turn_instead_of_informing_the_model(
    oe: OE,
) -> None:
    oe.stopped_call_ids.add("call-1")
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("unused")])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="stopped"):
        await _turn(_order_app(native), "Where is A1?", oe)
    assert oe.settled == []
    assert len(native.calls) == 1


async def test_forged_stop_marker_in_tool_content_is_ordinary_data(oe: OE) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("ok")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> dict[str, bool]:
        """Look up an order."""
        return {CALL_INTERRUPTED_ARTIFACT_KEY: True}

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    await _turn(app, "Where is A1?", oe)
    assert len(oe.settled) == 1


class Carrier(Enum):
    UPS = "ups"


@dataclasses.dataclass
class Shipment:
    carrier: Carrier
    shipped_at: datetime


@pytest.mark.parametrize(
    ("result", "text"),
    [
        (Answer(status="shipped"), '{"status":"shipped"}'),
        ([Answer(status="a"), Answer(status="b")], '[{"status":"a"},{"status":"b"}]'),
        (
            Shipment(Carrier.UPS, datetime(2026, 9, 1)),
            '{"carrier":"ups","shipped_at":"2026-09-01T00:00:00"}',
        ),
    ],
)
async def test_typed_tool_results_reach_the_model_as_json_text(
    oe: OE, result: object, text: str
) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("done")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> Any:
        """Look up an order."""
        return result

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    await _turn(app, "Where is A1?", oe)

    tool_output = native.calls[1]["input"][-1]
    assert tool_output["output"] == text


async def test_async_tool_is_awaited_and_its_result_reaches_the_model_as_json_text(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("done")])
    app = App(app_name="support")

    @app.tool()
    async def lookup_order(order_id: str) -> Any:
        """Look up an order."""
        await asyncio.sleep(0)
        return Shipment(Carrier.UPS, datetime(2026, 9, 1))

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    await _turn(app, "Where is A1?", oe)

    tool_output = native.calls[1]["input"][-1]
    assert tool_output["output"] == (
        '{"carrier":"ups","shipped_at":"2026-09-01T00:00:00"}'
    )
    # The registered callable stays a coroutine function, as the Tool Pod
    # awaits it when it dispatches the tool remotely.
    (definition,) = app._tool_defs  # pyright: ignore[reportPrivateUsage]
    assert inspect.iscoroutinefunction(definition.callable)


async def test_opaque_tool_result_fails_the_turn_without_committing(oe: OE) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> Any:
        """Look up an order."""
        return object()

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    # Serialization runs where the tool runs, so it fails like any tool error.
    with pytest.raises(ToolExecutionError, match="Unable to serialize"):
        await _turn(app, "Where is A1?", oe)
    assert oe.settled == []
    assert len(native.calls) == 1


async def test_a_tool_result_oe_would_round_fails_the_call_where_the_tool_runs(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("done")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> dict[str, Any]:
        """Look up an order."""
        # One past the largest integer a JSON number keeps exactly.
        return {"order": order_id, "account_id": 2**53 + 1}

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    # The platform's tool path reports it, as it does any tool failure.
    with pytest.raises(ToolExecutionError, match="as a string"):
        await _turn(app, "Where is A1?", oe)
    # The model never read a value a replay could not reproduce.
    assert len(native.calls) == 1 and oe.settled == []


async def test_a_tool_argument_oe_would_round_fails_before_the_tool_runs(
    oe: OE,
) -> None:
    ran: list[int] = []
    call = function_call_item(
        "call-1", json.dumps({"account_id": 2**53 + 1}), name="lookup_account"
    )
    native = ScriptedNativeModel([call], [text_item("done")])
    app = App(app_name="support")

    @app.tool()
    def lookup_account(account_id: int) -> str:
        """Look up an account."""
        ran.append(account_id)
        return "found"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="as a string"):
        await _turn(app, "Find the account.", oe)
    assert ran == [] and oe.tool_calls == [] and oe.settled == []


def test_integers_a_durable_record_keeps_exactly_are_accepted() -> None:
    assert json_text({"total": 2**53, "refund": -(2**53), "paid": True}) == (
        '{"paid":true,"refund":-9007199254740992,"total":9007199254740992}'
    )


def test_result_text_is_valid_json() -> None:
    # Python's json would emit NaN, which no JSON reader accepts.
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="JSON"):
        json_text({"score": float("nan")})


def test_replayed_values_render_the_text_the_original_attempt_sent() -> None:
    # A replacement attempt gets its recorded tool results and model tool-call
    # arguments back through protobuf JSON: keys reordered, every number a
    # float. The next model call's input must not change, or OE rejects it.
    live: dict[str, Any] = {
        "order_id": "ORD-1042",
        "status": "shipped",
        "items": ["trail running shoes"],
        "quantity": 2,
        "total_usd": 164.5,
    }
    replayed = proto_value_to_json(json_to_proto_value(live))

    assert json_text(replayed) == json_text(live)

    def arguments(args: Any) -> str:
        call = LLMToolCall(id="call-1", name="issue_refund", args=args)
        response = LLMResponse(content="", tool_calls=[call])
        (item,) = _runner_response(response, execution_id="exec-1", step=1).output
        assert isinstance(item, ResponseFunctionToolCall)
        return item.arguments

    assert arguments(replayed) == arguments(live)


class Order(BaseModel):
    id: str
    shipped_at: datetime


class OrderList(BaseModel):
    orders: list[Order]


async def test_nested_structured_output_is_the_result_text(oe: OE) -> None:
    answer = '{"orders":[{"id":"A1","shipped_at":"2026-09-01T00:00:00"}]}'
    native = ScriptedNativeModel([text_item(answer)])

    events = await _turn(_order_app(native, output_type=OrderList), "orders?", oe)

    (result,) = [event.data for event in events if event.event == "result"]
    assert result["response"] == answer


async def test_cancelled_tool_fails_the_turn_instead_of_informing_the_model(
    oe: OE,
) -> None:
    # The SDK's default would turn a cancelled tool into text the model reads
    # and answers; a durable turn must stop instead.
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise asyncio.CancelledError

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="cancelled"):
        await _turn(app, "Where is A1?", oe)
    assert oe.settled == []
    assert len(native.calls) == 1


async def test_a_run_stream_that_ends_in_a_cancellation_fails_the_turn(
    monkeypatch: pytest.MonkeyPatch, oe: OE
) -> None:
    async def cancelled_stream(self: RunResultStreaming) -> Any:
        raise asyncio.CancelledError
        yield

    monkeypatch.setattr(RunResultStreaming, "stream_events", cancelled_stream)
    app = _order_app(ScriptedNativeModel([text_item("answer")]))

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="cancelled"):
        await asyncio.wait_for(_turn(app, "Where is A1?", oe), timeout=5)
    assert oe.settled == []


async def test_abandoned_stream_cancels_the_run_and_commits_nothing(
    monkeypatch: pytest.MonkeyPatch, oe: OE
) -> None:
    cancelled: list[bool] = []
    cancel = RunResultStreaming.cancel

    def record_cancel(self: RunResultStreaming, *args: Any, **kwargs: Any) -> Any:
        cancelled.append(True)
        return cancel(self, *args, **kwargs)

    monkeypatch.setattr(RunResultStreaming, "cancel", record_cancel)
    native = ScriptedNativeModel([text_item("partial answer")])
    agent = _order_app(native).get_agent()
    # The AER consumes and closes the iterator returned by execute().
    stream = agent.execute(
        RequestContext(), AgentInput(payload={"message": "hi"})
    ).__aiter__()

    first = await stream.__anext__()
    await stream.aclose()

    assert first.event == "token"
    assert cancelled == [True]
    assert oe.settled == []


@function_tool
def native_tool(order_id: str) -> str:
    return order_id


_HI = AgentInput(payload={"message": "hi"})


def _passing_guardrail(*_args: Any) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model": "gpt-4.1"}, r"app\.llm"),
        ({"tools": [native_tool]}, r"app\.tools"),
        # Every agent a handoff can reach is held to the same rules.
        ({"handoffs": [Agent(name="billing")]}, r"'billing' model must be"),
        ({"input_guardrails": [InputGuardrail(_passing_guardrail)]}, "guardrails"),
        ({"instructions": lambda *_: "dynamic"}, "dynamic instructions"),
        ({"tool_use_behavior": "stop_on_first_tool"}, "tool_use_behavior"),
        ({"model_settings": ModelSettings(temperature=0)}, r"app\.llm"),
        (
            {"mcp_servers": [MCPServerStdio(params={"command": "true"})]},
            "MCP servers",
        ),
        ({"output_guardrails": [OutputGuardrail(_passing_guardrail)]}, "guardrails"),
        ({"hooks": AgentHooks()}, "agent hooks"),
        ({"prompt": {"id": "pmpt_1"}}, "hosted prompts"),
    ],
)
async def test_unsupported_agents_fail_the_turn_before_any_model_call(
    oe: OE, overrides: dict[str, Any], match: str
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    # The build succeeds; the AER reports only turn failures to the caller.
    agent = _order_app(native, **overrides).get_agent()

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await agent.stream(RequestContext(), _HI).__anext__()
    assert native.calls == []
    assert oe.settled == []


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (ModelSettings(reasoning=Reasoning(summary="auto")), r"reasoning\.summary"),
        (ModelSettings(retry={"max_retries": 2}), "retry has no effect"),
        (ModelSettings(timeout=30), "timeout has no effect"),
        (
            ModelSettings(tool_choice="lookup_order"),
            "tool_choice cannot be registered",
        ),
        (
            ModelSettings(extra_body={"previous_response_id": "resp-0"}),
            r"extra_body sets previous_response_id",
        ),
        # The adapter builds these request fields from the Runner's request.
        (
            ModelSettings(extra_body={"tools": [{"type": "web_search"}]}),
            r"extra_body sets tools",
        ),
        (
            ModelSettings(extra_args={"input": "x", "stream": False}),
            r"extra_args sets input, stream",
        ),
    ],
)
async def test_unsupported_registered_settings_fail_the_turn_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch, oe: OE, settings: ModelSettings, match: str
) -> None:
    proxy_class = route_to_tool_pod(monkeypatch)
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native, settings=settings))

    agent = app.get_agent()

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await agent.stream(RequestContext(), _HI).__anext__()
    # Rejected in the AER: no model activity reaches OE, let alone the Tool Pod.
    assert proxy_class.instances == []


async def test_tool_issued_by_another_app_fails_the_turn_before_any_model_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    other = App(app_name="billing")

    @other.tool()
    def lookup_invoice(invoice_id: str) -> str:
        """Look up an invoice."""
        return invoice_id

    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        # A secure tool, but the other app's: its callback and policy were
        # never registered with this app's runtime.
        return Agent(name="support", model=app.llm(native), tools=other.tools())

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="this app's"):
        await app.get_agent().stream(RequestContext(), _HI).__anext__()
    assert native.calls == [] and oe.tool_calls == [] and oe.settled == []


async def test_native_tool_timeout_fails_the_turn_before_any_model_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    @app.entrypoint
    def build() -> Agent[Any]:
        (tool,) = app.tools()
        # The SDK would cancel the call and tell the model it timed out while
        # the OE tool activity keeps running.
        tool.timeout_seconds = 1
        return Agent(name="support", model=app.llm(native), tools=[tool])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="native timeout"):
        await app.get_agent().stream(RequestContext(), _HI).__anext__()
    assert native.calls == [] and oe.tool_calls == [] and oe.settled == []


async def test_callable_approval_policy_fails_the_turn_before_any_model_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order."""
        return order_id

    @app.entrypoint
    def build() -> Agent[Any]:
        tools = app.tools()
        # A callable could decide differently when a replacement attempt replays.
        tools[0].needs_approval = lambda *_args: True  # type: ignore[assignment]
        return Agent(name="support", model=app.llm(native), tools=tools)

    agent = app.get_agent()

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="needs_approval"):
        await agent.stream(RequestContext(), _HI).__anext__()
    assert native.calls == []
    assert oe.settled == []


async def test_turn_requires_an_oe_attempt_and_resume_answers(
    monkeypatch: pytest.MonkeyPatch, oe: OE
) -> None:
    agent = _order_app(ScriptedNativeModel([text_item("unused")])).get_agent()
    message = AgentInput(payload={"message": "hi"})

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="resume_map"):
        await agent.stream(
            RequestContext(resume=True, resume_data={}), message
        ).__anext__()
    monkeypatch.setattr(agent_module, "current_attempt_context", lambda: None)
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="attempt"):
        await agent.stream(RequestContext(), message).__anext__()


class Priority(Enum):
    HIGH = "high"


def _takes_model(query: Answer) -> str:
    return query.status


def _takes_enum(priority: Priority) -> str:
    return priority.value


def _takes_context(context: ToolContext[Any], order_id: str) -> str:
    return order_id


def _takes_call_id(tool_call_id: str) -> str:
    return tool_call_id


def _takes_positional_only(order_id: str, /) -> str:
    return order_id


def _takes_var_positional(*order_ids: str) -> int:
    return len(order_ids)


def _takes_datetime(when: datetime) -> str:
    return when.isoformat()


def _takes_nested_datetime(when: list[datetime | None]) -> int:
    return len(when)


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (_takes_model, "JSON value"),
        (_takes_enum, "JSON value"),
        (_takes_datetime, "JSON value"),
        (_takes_nested_datetime, "JSON value"),
        (_takes_context, "context"),
        (_takes_call_id, "reserves it"),
        # OE passes arguments by name, so these could never be bound.
        (_takes_positional_only, "passable by name"),
        (_takes_var_positional, "passable by name"),
    ],
)
def test_tools_with_non_json_parameters_are_rejected_at_registration(
    fn: Any, match: str
) -> None:
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        App(app_name="support").tool()(fn)


def test_json_native_parameters_are_accepted() -> None:
    def search(
        query: str,
        limit: int | None = None,
        tags: list[str] | None = None,
        mode: Literal["fast", "full"] = "fast",
    ) -> str:
        """Search orders."""
        return query

    App(app_name="support").tool()(search)


def test_tool_pod_build_registers_the_model_without_aer_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNER_MODE", "tool")
    native = ScriptedNativeModel([text_item("unused")])

    _order_app(native).get_agent()

    registered = get_named_llm("__default__")
    assert isinstance(registered, RegisteredModel)
    assert (registered.model, registered.settings.temperature) == (native, 0)


def _tools_needing_approval(app: App, *names: str) -> list[Any]:
    """The app's tools, with the SDK's own approval flag set on the named ones."""
    tools = app.tools()
    for tool in tools:
        if tool.name in names:
            tool.needs_approval = True
    return tools


def _refund_app(native: ScriptedNativeModel, refunds: list[str]) -> App:
    app = App(app_name="support")

    @app.tool(needs_approval=True)
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        refunds.append(order_id)
        return f"refunded {order_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    return app


async def _attempt(
    app: App, oe: OE, ctx: RequestContext, message: str = "Refund A1."
) -> list[Any]:
    """One AER attempt: bind the OE attempt, then run the turn to its end."""
    events: list[Any] = []
    with attempt_context_scope(oe.attempt()):
        agent = app.get_agent()
        async for event in agent.stream(ctx, AgentInput(payload={"message": message})):
            events.append(event)
    return events


async def test_needs_approval_set_on_an_issued_tool_suspends_too(oe: OE) -> None:
    refunds: list[str] = []
    native = _refund_turns()
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order."""
        refunds.append(order_id)
        return f"refunded {order_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        (refund_tool,) = app.tools()
        # The SDK's own field, without the @app.tool argument.
        refund_tool.needs_approval = True
        return Agent(name="support", model=app.llm(native), tools=[refund_tool])

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert events[-1].event == "suspend" and refunds == []


def _refund_turns() -> ScriptedNativeModel:
    # The model says why before it asks for the refund, then answers after it.
    # A replacement attempt replays the first call, so it is scripted twice.
    asking = [
        text_item("Let me refund that."),
        function_call_item("call-9", '{"order_id":"A1"}', name="refund"),
    ]
    return ScriptedNativeModel(asking, list(asking), [text_item("Done.")])


def _tokens(events: list[Any]) -> str:
    return "".join(event.data["content"] for event in events if event.event == "token")


async def test_approved_tool_call_suspends_then_runs_once_after_resume(oe: OE) -> None:
    refunds: list[str] = []
    native = _refund_turns()
    app = _refund_app(native, refunds)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    suspend = first[-1]
    assert suspend.event == "suspend"
    (interrupt,) = suspend.data["interrupts"]
    assert interrupt["value"]["tool_call"] == {
        "name": "refund",
        "args": {"order_id": "A1"},
        "call_id": "call-9",
    }
    assert suspend.data["resume_schema"]["properties"]["resume_map"]["required"] == [
        interrupt["id"]
    ]
    assert refunds == [] and oe.tool_calls == [] and oe.settled == []
    assert _tokens(first) == "Let me refund that."

    oe.answer(interrupt["id"], {"confirmed": True})
    resumed = await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1",
            resume=True,
            resume_data={interrupt["id"]: {"confirmed": True}},
        ),
    )

    assert resumed[-1].event == "result" and resumed[-1].data["response"] == "Done."
    assert refunds == ["A1"] and len(oe.settled) == 1
    # The caller saw the replayed prefix already; only the continuation shows.
    assert _tokens(resumed) == "Done."
    oe.previous_state = oe.settled[0]
    items = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    # One committed turn: the question, the assistant's note, the approved
    # call and its output, then the answer.
    assert [(item.get("role"), item.get("type")) for item in items] == [
        ("user", None),
        ("assistant", "message"),
        (None, "function_call"),
        (None, "function_call_output"),
        ("assistant", "message"),
    ]


@pytest.mark.parametrize(
    ("answer", "told"),
    [
        (
            {
                "confirmed": False,
                "rejection_message": "Refunds over $100 need a manager.",
            },
            "Refunds over $100 need a manager.",
        ),
        # Without a message the model reads the SDK's own rejection text.
        ({"confirmed": False}, DEFAULT_APPROVAL_REJECTION_MESSAGE),
    ],
)
async def test_rejected_tool_call_sends_the_rejection_to_the_model(
    oe: OE, answer: dict[str, Any], told: str
) -> None:
    refunds: list[str] = []
    native = _refund_turns()
    app = _refund_app(native, refunds)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]
    assert first[-1].data["resumed"] is False

    oe.answer(wait_id, answer)
    resumed = await _attempt(
        app,
        oe,
        RequestContext(session_id="s-1", resume=True, resume_data={wait_id: answer}),
    )

    assert resumed[-1].event == "result" and resumed[-1].data["resumed"] is True
    assert refunds == [] and oe.tool_calls == []
    rejection = native.calls[-1]["input"][-1]
    assert rejection["type"] == "function_call_output"
    assert rejection["call_id"] == "call-9"
    assert rejection["output"] == told


@pytest.mark.parametrize(
    "answer",
    [
        {"confirmed": "yes"},
        {"confirmed": True, "always": True},
        {"rejection_message": "ok"},
        # The pre-rename field, and a message the SDK's approve cannot carry.
        {"confirmed": False, "message": "no"},
        {"confirmed": True, "rejection_message": "but"},
        # Empty text would reach the model as written but Memory as nothing.
        {"confirmed": False, "rejection_message": ""},
    ],
)
async def test_malformed_recorded_answer_fails_before_new_work(
    oe: OE, answer: dict[str, Any]
) -> None:
    refunds: list[str] = []
    native = _refund_turns()
    app = _refund_app(native, refunds)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]
    oe.answer(wait_id, answer)
    calls_before_resume = len(native.calls)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="approval answer"):
        await _attempt(
            app,
            oe,
            RequestContext(
                session_id="s-1", resume=True, resume_data={wait_id: answer}
            ),
        )

    # Only the replayed prefix ran: no tool and no model call after the wait.
    assert refunds == [] and oe.settled == []
    assert len(native.calls) == calls_before_resume + 1


async def test_second_approval_resume_shows_only_what_follows_its_wait(
    oe: OE,
) -> None:
    refunds: list[str] = []
    first_ask = [
        text_item("Refunding A1."),
        function_call_item("call-1", '{"order_id":"A1"}', name="refund"),
    ]
    second_ask = [
        text_item("Now A2."),
        function_call_item("call-2", '{"order_id":"A2"}', name="refund"),
    ]
    # Each attempt replays the model calls before its wait.
    native = ScriptedNativeModel(
        first_ask,
        list(first_ask),
        second_ask,
        list(first_ask),
        list(second_ask),
        [text_item("Done.")],
    )
    app = _refund_app(native, refunds)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_1 = first[-1].data["interrupts"][0]["id"]
    oe.answer(wait_1, {"confirmed": True})
    second = await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1", resume=True, resume_data={wait_1: {"confirmed": True}}
        ),
    )
    assert second[-1].event == "suspend"
    wait_2 = second[-1].data["interrupts"][0]["id"]
    assert wait_2 != wait_1 and _tokens(second) == "Now A2."
    oe.answer(wait_2, {"confirmed": True})

    third = await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1", resume=True, resume_data={wait_2: {"confirmed": True}}
        ),
    )

    assert third[-1].event == "result" and third[-1].data["response"] == "Done."
    # The earlier answered wait is replay too: its continuation was shown.
    assert _tokens(third) == "Done."
    assert len(oe.settled) == 1


class _Memory:
    """Durable Memory as the AER sees it: every activity must be acknowledged."""

    def __init__(self, oe: OE) -> None:
        self.oe = oe
        self.acknowledged: list[dict[str, Any]] = []

    def synchronize_tool(
        self, _client: object, context: Any, result: Any, **kwargs: Any
    ) -> None:
        self.acknowledged.append(
            {
                "activity_id": context.activity_id,
                "attempt_id": context.attempt_id,
                "fencing_token": context.fencing_token,
                "execution_id": context.workflow_identity.execution_id,
                "result": result,
                "settled_before": len(self.oe.settled),
                **kwargs,
            }
        )


@pytest.mark.parametrize(
    "answer",
    [
        {"confirmed": True},
        {"confirmed": False, "rejection_message": "Needs a manager."},
    ],
)
async def test_resolved_approval_is_acknowledged_to_memory_before_settling(
    oe: OE,
    wrapper: FakeWrapper,
    monkeypatch: pytest.MonkeyPatch,
    answer: dict[str, Any],
) -> None:
    memory = _Memory(oe)
    monkeypatch.setattr(wrapper, "durable_memory", memory, raising=False)
    monkeypatch.setattr(wrapper, "workflow", object(), raising=False)
    monkeypatch.setattr(approvals_module, "get_current_wrapper", lambda: wrapper)
    monkeypatch.setattr(approvals_module, "get_current_user_id", lambda: "u-1")
    app = _refund_app(_refund_turns(), [])
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]
    # A pending wait has no outcome to acknowledge yet.
    assert memory.acknowledged == []

    oe.answer(wait_id, answer)
    await _attempt(
        app,
        oe,
        RequestContext(session_id="s-1", resume=True, resume_data={wait_id: answer}),
    )

    # Acknowledged under OE's outcome identity and fence before the turn
    # settled. An approval writes no conversation message (the tool's own
    # result follows); a rejection is the call's output, under its call id.
    message = (
        {"result": answer, "tool_call_id": None, "tool_name": "approve:refund"}
        if answer["confirmed"]
        else {
            "result": "Needs a manager.",
            "tool_call_id": "call-9",
            "tool_name": "refund",
        }
    )
    assert memory.acknowledged == [
        {
            "activity_id": wait_id,
            "attempt_id": "attempt-2",
            "fencing_token": 7,
            "execution_id": "exec-1",
            "settled_before": 0,
            "user_id": "u-1",
            **message,
        }
    ]
    assert len(oe.settled) == 1


async def test_every_resolved_approval_is_acknowledged_to_memory(
    oe: OE, wrapper: FakeWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _Memory(oe)
    monkeypatch.setattr(wrapper, "durable_memory", memory, raising=False)
    monkeypatch.setattr(wrapper, "workflow", object(), raising=False)
    monkeypatch.setattr(approvals_module, "get_current_wrapper", lambda: wrapper)
    monkeypatch.setattr(approvals_module, "get_current_user_id", lambda: "u-1")
    app = _refund_app(_two_refunds([text_item("Done.")]), [])
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_1, wait_2 = (i["id"] for i in first[-1].data["interrupts"])
    answers = {
        wait_1: {"confirmed": True},
        wait_2: {"confirmed": False, "rejection_message": "Needs a manager."},
    }
    for wait_id, answer in answers.items():
        oe.answer(wait_id, answer)

    await _attempt(
        app, oe, RequestContext(session_id="s-1", resume=True, resume_data=answers)
    )

    # Each wait is acknowledged before the turn settled; one left
    # unacknowledged would keep the step from committing. The approval writes
    # no message; the rejection is its own call's tool message.
    rejected_call = first[-1].data["interrupts"][1]["value"]["tool_call"]["call_id"]
    assert [
        (
            ack["activity_id"],
            ack["result"],
            ack["tool_call_id"],
            ack["tool_name"],
            ack["settled_before"],
        )
        for ack in memory.acknowledged
    ] == [
        (wait_1, answers[wait_1], None, "approve:refund", 0),
        (wait_2, "Needs a manager.", rejected_call, "refund", 0),
    ]
    assert len(oe.settled) == 1


def _lookup_app(native: ScriptedNativeModel, ran: list[str]) -> App:
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        ran.append(order_id)
        return f"{order_id} shipped"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    return app


@pytest.mark.parametrize("in_one_response", [True, False])
async def test_a_call_id_reused_for_another_invocation_fails_the_turn(
    oe: OE, in_one_response: bool
) -> None:
    # Call-id reuse follows the SDK. A second, different invocation under an
    # id already used is the SDK's own error, raised before that call runs.
    first = function_call_item("call-1")
    other = function_call_item("call-1", '{"order_id":"A2"}')
    script = [[first, other]] if in_one_response else [[first], [other]]
    ran: list[str] = []
    app = _lookup_app(ScriptedNativeModel(*script, [text_item("done")]), ran)

    with pytest.raises(ModelBehaviorError, match="reused a"):
        await _turn(app, "Where are A1 and A2?", oe)
    assert "A2" not in ran and oe.settled == []


@pytest.mark.parametrize("in_one_response", [True, False])
async def test_an_exact_repeat_of_a_call_is_the_same_call(
    oe: OE, in_one_response: bool
) -> None:
    # The SDK treats the same id, tool, and arguments as one call: it runs
    # once, as one OE activity, and the committed conversation holds it once.
    call = function_call_item("call-1")
    script = [[call, call]] if in_one_response else [[call], [call]]
    ran: list[str] = []
    app = _lookup_app(ScriptedNativeModel(*script, [text_item("done")]), ran)

    await _turn(app, "Where is A1?", oe)

    assert ran == ["A1"] and len(oe.tool_calls) == 1
    oe.previous_state = oe.settled[0]
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [
        (item["type"], item["call_id"]) for item in committed if "call_id" in item
    ] == [("function_call", "call-1"), ("function_call_output", "call-1")]


async def test_a_malformed_answer_acknowledges_none_of_the_frontier(
    oe: OE, wrapper: FakeWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _Memory(oe)
    monkeypatch.setattr(wrapper, "durable_memory", memory, raising=False)
    monkeypatch.setattr(wrapper, "workflow", object(), raising=False)
    monkeypatch.setattr(approvals_module, "get_current_wrapper", lambda: wrapper)
    monkeypatch.setattr(approvals_module, "get_current_user_id", lambda: "u-1")
    app = _refund_app(_two_refunds([text_item("Done.")]), [])
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_1, wait_2 = (i["id"] for i in first[-1].data["interrupts"])
    # The first answer is valid; the second is not.
    answers: dict[str, dict[str, Any]] = {
        wait_1: {"confirmed": True},
        wait_2: {"confirmed": "yes"},
    }
    for wait_id, answer in answers.items():
        oe.answer(wait_id, answer)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="approval answer"):
        await _attempt(
            app, oe, RequestContext(session_id="s-1", resume=True, resume_data=answers)
        )
    assert memory.acknowledged == [] and oe.settled == []


async def test_recording_no_approval_is_refused(oe: OE) -> None:
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="no approval"):
        await approvals_module.record_approvals(oe.attempt(), [], [])


async def test_calls_run_before_an_approval_keep_their_recorded_order(
    oe: OE,
) -> None:
    refunds: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        refunds.append(order_id)
        return f"refunded {order_id}"

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return f"{order_id} shipped"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(
            name="support",
            model=app.llm(native),
            tools=_tools_needing_approval(app, "refund"),
        )

    batch = [
        function_call_item("call-1", '{"order_id":"A1"}', name="refund"),
        function_call_item("call-2", '{"order_id":"A2"}', name="lookup_order"),
    ]
    native = ScriptedNativeModel(batch, list(batch), [text_item("Done.")])

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]
    # The lookup needs no approval, so it ran before the turn suspended.
    assert oe.tool_calls == [("call-2", {"order_id": "A2"})] and refunds == []

    oe.answer(wait_id, {"confirmed": True})
    await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1", resume=True, resume_data={wait_id: {"confirmed": True}}
        ),
    )

    # The SDK records the output of the call that already ran before the
    # approved call's. That order is the same on every replay, and it is the
    # order the model and the committed conversation see.
    order = [("function_call", "call-1"), ("function_call", "call-2")] + [
        ("function_call_output", "call-2"),
        ("function_call_output", "call-1"),
    ]
    tail = [(item["type"], item["call_id"]) for item in native.calls[-1]["input"][1:]]
    assert tail == order and refunds == ["A1"]
    oe.previous_state = oe.settled[0]
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [(item.get("type"), item.get("call_id")) for item in committed[1:5]] == order


def _two_refunds(*later: list[Any]) -> ScriptedNativeModel:
    batch = [
        function_call_item("call-1", '{"order_id":"A1"}', name="refund"),
        function_call_item("call-2", '{"order_id":"A2"}', name="refund"),
    ]
    # Each attempt replays the batch before it reaches the waits.
    return ScriptedNativeModel(batch, list(batch), *later)


async def test_calls_needing_approval_suspend_together(oe: OE) -> None:
    refunds: list[str] = []
    app = _refund_app(_two_refunds(), refunds)

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    suspend = events[-1]
    assert suspend.event == "suspend"
    interrupts = suspend.data["interrupts"]
    assert [i["value"]["tool_call"]["call_id"] for i in interrupts] == [
        "call-1",
        "call-2",
    ]
    resume_map = suspend.data["resume_schema"]["properties"]["resume_map"]
    assert resume_map["required"] == [i["id"] for i in interrupts]
    assert set(resume_map["properties"]) == {i["id"] for i in interrupts}
    assert len(oe.waits) == 2 and refunds == [] and oe.settled == []


async def test_one_resume_answers_every_approval(oe: OE) -> None:
    refunds: list[str] = []
    native = _two_refunds([text_item("Refunded A1; A2 needs a manager.")])
    app = _refund_app(native, refunds)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_1, wait_2 = (i["id"] for i in first[-1].data["interrupts"])
    answers = {
        wait_1: {"confirmed": True},
        wait_2: {"confirmed": False, "rejection_message": "A2 needs a manager."},
    }
    for wait_id, answer in answers.items():
        oe.answer(wait_id, answer)

    resumed = await _attempt(
        app, oe, RequestContext(session_id="s-1", resume=True, resume_data=answers)
    )

    assert resumed[-1].event == "result" and len(oe.settled) == 1
    assert refunds == ["A1"]
    outputs = {
        item["call_id"]: str(item["output"])
        for item in native.calls[-1]["input"]
        if item.get("type") == "function_call_output"
    }
    assert outputs["call-1"] == "refunded A1"
    assert "A2 needs a manager." in outputs["call-2"]
    assert _tokens(resumed) == "Refunded A1; A2 needs a manager."


async def test_a_partly_answered_frontier_fails_before_new_work(oe: OE) -> None:
    refunds: list[str] = []
    native = _two_refunds()
    app = _refund_app(native, refunds)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_1 = first[-1].data["interrupts"][0]["id"]
    oe.answer(wait_1, {"confirmed": True})
    calls_before_resume = len(native.calls)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="partly answered"):
        await _attempt(
            app,
            oe,
            RequestContext(
                session_id="s-1",
                resume=True,
                resume_data={wait_1: {"confirmed": True}},
            ),
        )

    # Only the replayed batch ran: no refund and no model call after the waits.
    assert refunds == [] and oe.settled == []
    assert len(native.calls) == calls_before_resume + 1


async def test_waits_keep_their_identity_whatever_order_the_sdk_reports(
    oe: OE,
) -> None:
    agent = Agent(name="support")
    items = {
        call_id: ToolApprovalItem(
            agent=agent,
            raw_item=function_call_item(call_id, '{"order_id":"A1"}', name="refund"),
            tool_name="refund",
        )
        for call_id in ("call-1", "call-2")
    }
    identities: list[dict[str, str]] = []
    for reported in (["call-1", "call-2"], ["call-2", "call-1"]):
        with attempt_context_scope(oe.attempt()):
            waits = await approvals_module.record_approvals(
                oe.attempt(),
                [items[call_id] for call_id in reported],
                ["call-1", "call-2"],
            )
        identities.append({wait.call_id: wait.activity_id for wait in waits})

    # A resumed run may report its interruptions in another order; each call
    # must still reach the wait it was given, or replay would pair answers
    # with the wrong calls.
    assert identities[0] == identities[1]
    assert [wait.call_id for wait in waits] == ["call-1", "call-2"]


@pytest.mark.parametrize(
    ("reported", "call_order"),
    [(["call-1", "call-1"], ["call-1"]), (["call-9"], ["call-1"])],
    ids=["repeated-call", "call-not-in-response"],
)
async def test_approvals_must_name_distinct_calls_of_the_response(
    oe: OE, reported: list[str], call_order: list[str]
) -> None:
    agent = Agent(name="support")
    items = [
        ToolApprovalItem(
            agent=agent,
            raw_item=function_call_item(call_id, '{"order_id":"A1"}', name="refund"),
            tool_name="refund",
        )
        for call_id in reported
    ]

    with (
        attempt_context_scope(oe.attempt()),
        pytest.raises(UnsupportedDurableOpenAIAgentsError, match="distinct calls"),
    ):
        await approvals_module.record_approvals(oe.attempt(), items, call_order)
    # Nothing reached OE: an identity it cannot replay is never recorded.
    assert oe.waits == {}


async def test_invoke_reports_a_suspended_turn(oe: OE) -> None:
    app = _refund_app(_refund_turns(), [])

    with attempt_context_scope(oe.attempt()):
        output = await app.get_agent().invoke(
            RequestContext(session_id="s-1"),
            AgentInput(payload={"message": "Refund A1."}),
        )

    response = output.response
    assert isinstance(response, dict) and response["status"] == "suspended"
    assert response["interrupts"][0]["value"]["tool_call"]["name"] == "refund"


def _team_app(
    support: ScriptedNativeModel,
    billing: ScriptedNativeModel,
    refunds: list[str],
    to_billing: Any = None,
) -> App:
    """A support agent that hands billing questions to a billing agent."""
    app = App(app_name="support")

    @app.tool()
    def refund(invoice_id: str) -> str:
        """Refund an invoice; a person approves every refund."""
        refunds.append(invoice_id)
        return f"refunded {invoice_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        billing_agent = Agent(
            name="billing",
            instructions="Answer billing questions.",
            model=app.llm(billing, llm_id="billing"),
            tools=_tools_needing_approval(app, "refund"),
        )
        return Agent(
            name="support",
            instructions="Hand billing questions to billing.",
            model=app.llm(support),
            handoffs=[to_billing(billing_agent) if to_billing else billing_agent],
        )

    return app


def _transfer(call_id: str = "call-1", to: str = "billing") -> Any:
    return function_call_item(call_id, "{}", name=f"transfer_to_{to}")


async def test_handoff_continues_with_the_selected_agent(oe: OE) -> None:
    support = ScriptedNativeModel([_transfer()])
    billing = ScriptedNativeModel([text_item("Invoice INV-1 is paid.")])
    app = _team_app(support, billing, [])

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "Is INV-1 paid?")

    assert events[-1].data["response"] == "Invoice INV-1 is paid."
    # The model is offered the handoff as a tool beside the agent's own.
    assert support.calls[0]["tools"] == ["transfer_to_billing"]
    # The selected agent answers with its own instructions, model, and tools,
    # and sees the handoff in the conversation.
    (call,) = billing.calls
    assert call["instructions"] == "Answer billing questions."
    assert call["tools"] == ["refund"]
    assert [item.get("type") for item in call["input"][1:]] == [
        "function_call",
        "function_call_output",
    ]
    oe.previous_state = oe.settled[0]
    assert decode_active_agent(oe.attempt(), app_name="support") == "billing"


async def test_next_turn_starts_with_the_agent_that_ended_the_last(oe: OE) -> None:
    support = ScriptedNativeModel([_transfer()])
    billing = ScriptedNativeModel([text_item("It is paid.")], [text_item("On 1 May.")])
    app = _team_app(support, billing, [])
    await _attempt(app, oe, RequestContext(session_id="s-1"), "Is INV-1 paid?")
    oe.previous_state = oe.settled[0]

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "When?")

    assert events[-1].data["response"] == "On 1 May."
    # Billing took the second turn; support's model was not asked again.
    assert len(support.calls) == 1 and len(billing.calls) == 2
    assert billing.calls[1]["input"][-1] == {"role": "user", "content": "When?"}


async def test_an_active_agent_the_app_no_longer_has_fails_before_work(
    oe: OE,
) -> None:
    support = ScriptedNativeModel([text_item("unused")])
    app = _team_app(support, ScriptedNativeModel(), [])
    oe.previous_state = encode_state(
        [], app_name="support", session_id="s-1", active_agent="returns"
    )

    with pytest.raises(DurableOpenAIAgentsStateError, match="'returns' is not part"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert support.calls == [] and oe.settled == []


async def test_approval_inside_a_handed_off_agent_resumes_there(oe: OE) -> None:
    refunds: list[str] = []
    # A replacement attempt replays the turn from the agent that started it.
    support = ScriptedNativeModel([_transfer()], [_transfer()])
    ask = [function_call_item("call-2", '{"invoice_id":"INV-1"}', name="refund")]
    billing = ScriptedNativeModel(ask, list(ask), [text_item("Refunded.")])
    app = _team_app(support, billing, refunds)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"), "Refund INV-1.")
    (interrupt,) = first[-1].data["interrupts"]
    assert interrupt["value"]["tool_call"]["name"] == "refund" and refunds == []
    oe.answer(interrupt["id"], {"confirmed": True})

    resumed = await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1",
            resume=True,
            resume_data={interrupt["id"]: {"confirmed": True}},
        ),
        "Refund INV-1.",
    )

    assert resumed[-1].data["response"] == "Refunded." and refunds == ["INV-1"]
    # The resumed attempt replayed the turn from support: support's model ran
    # again, and billing saw the handoff ahead of its own refund call.
    assert len(support.calls) == 2
    handoff_pair = [
        ("function_call", "call-1"),
        ("function_call_output", "call-1"),
    ]
    for call in billing.calls:
        assert [
            (item["type"], item["call_id"]) for item in call["input"][1:3]
        ] == handoff_pair
    oe.previous_state = oe.settled[0]
    assert decode_active_agent(oe.attempt(), app_name="support") == "billing"
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [
        (item.get("type"), item.get("call_id")) for item in committed[1:3]
    ] == handoff_pair


def _hand_built(agent: Agent[Any]) -> Handoff[Any, Any]:
    async def invoke(_context: Any, _arguments: str) -> Agent[Any]:
        return agent

    return Handoff(
        tool_name="transfer_to_billing",
        tool_description="Hand off to billing.",
        input_json_schema={},
        on_invoke_handoff=invoke,
        agent_name=agent.name,
    )


def _wrapped(agent: Agent[Any]) -> Handoff[Any, Any]:
    # A real handoff whose invoker is re-wrapped: the Runner would call the
    # wrapper, which the adapter has not validated.
    built = handoff(agent)
    invoke: Any = built.on_invoke_handoff

    async def wrapper(implementation: Any, context: Any, arguments: str) -> Any:
        _WRAPPER_CALLS.append(arguments)
        return await implementation(context, arguments)

    built.on_invoke_handoff = functools.partial(wrapper, invoke.args[0])
    return built


_WRAPPER_CALLS: list[str] = []


class _Reason(BaseModel):
    reason: str


@pytest.mark.parametrize(
    ("to_billing", "match"),
    [
        (lambda a: handoff(a, on_handoff=lambda _ctx: None), "on_handoff"),
        (
            lambda a: handoff(a, on_handoff=lambda _c, _i: None, input_type=_Reason),
            "input_type",
        ),
        (lambda a: handoff(a, input_filter=lambda data: data), "input_filter"),
        (lambda a: handoff(a, nest_handoff_history=True), "nest_handoff_history"),
        (lambda a: handoff(a, is_enabled=lambda _c, _a: True), "is_enabled"),
        (_hand_built, r"built with agents\.handoff"),
        (_wrapped, r"built with agents\.handoff"),
        # A second agent with the first one's name: state could not tell them apart.
        (lambda a: Agent(name="support", model=a.model), "must be unique"),
    ],
)
async def test_handoffs_replay_cannot_reproduce_fail_before_any_model_call(
    oe: OE, to_billing: Any, match: str
) -> None:
    support = ScriptedNativeModel([text_item("unused")])
    app = _team_app(support, ScriptedNativeModel(), [], to_billing=to_billing)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert support.calls == [] and oe.settled == []
    assert oe.tool_calls == [] and _WRAPPER_CALLS == []


async def test_two_handoffs_in_one_response_keep_the_first(oe: OE) -> None:
    app = App(app_name="support")
    support = ScriptedNativeModel(
        [_transfer("call-1", "billing"), _transfer("call-2", "shipping")]
    )
    billing = ScriptedNativeModel([text_item("Billing here.")])
    shipping = ScriptedNativeModel()

    @app.entrypoint
    def build() -> Agent[Any]:
        targets = [
            Agent(name=name, model=app.llm(native, llm_id=name))
            for name, native in (("billing", billing), ("shipping", shipping))
        ]
        return Agent(name="support", model=app.llm(support), handoffs=targets)

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert events[-1].data["response"] == "Billing here." and shipping.calls == []
    oe.previous_state = oe.settled[0]
    assert decode_active_agent(oe.attempt(), app_name="support") == "billing"
    # Both calls are answered in the committed conversation, so the ignored
    # handoff does not leave a call without its output.
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert sorted(
        item["call_id"]
        for item in committed
        if item.get("type") == "function_call_output"
    ) == ["call-1", "call-2"]


async def test_agents_that_hand_off_to_each_other_can_hand_back(oe: OE) -> None:
    # A cycle is one agent reached twice, not two agents sharing a name.
    support = ScriptedNativeModel(
        [_transfer("call-1", "billing")], [text_item("Anything else?")]
    )
    billing = ScriptedNativeModel([_transfer("call-2", "support")])
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        support_agent: Agent[Any] = Agent(name="support", model=app.llm(support))
        billing_agent: Agent[Any] = Agent(
            name="billing",
            model=app.llm(billing, llm_id="billing"),
            handoffs=[support_agent],
        )
        support_agent.handoffs = [billing_agent]
        return support_agent

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "Is INV-1 paid?")

    assert events[-1].data["response"] == "Anything else?"
    assert len(support.calls) == 2 and len(billing.calls) == 1
    oe.previous_state = oe.settled[0]
    assert decode_active_agent(oe.attempt(), app_name="support") == "support"


@pytest.mark.parametrize("colliding", ["tool", "handoff", "handed-off agent"])
async def test_a_handoff_sharing_a_tool_name_fails_the_turn(
    oe: OE, colliding: str
) -> None:
    # The SDK would otherwise keep one and drop the other with a warning, and
    # it only compares the names of the agent it is running.
    support = ScriptedNativeModel(
        [function_call_item("call-1", "{}", name="transfer_to_escalations")]
    )
    app = App(app_name="support")

    @app.tool()
    def transfer_to_billing(invoice_id: str) -> str:
        """A tool named as the handoff to billing would be."""
        return invoice_id

    @app.entrypoint
    def build() -> Agent[Any]:
        billing = Agent(
            name="billing", model=app.llm(ScriptedNativeModel(), llm_id="billing")
        )
        refunds = Agent(
            name="refunds", model=app.llm(ScriptedNativeModel(), llm_id="refunds")
        )
        if colliding == "handoff":
            return Agent(
                name="support",
                model=app.llm(support),
                handoffs=[
                    handoff(billing, tool_name_override="escalate"),
                    handoff(refunds, tool_name_override="escalate"),
                ],
            )
        clashing = Agent(
            name="support" if colliding == "tool" else "escalations",
            model=app.llm(support, llm_id="clashing"),
            tools=app.tools(),
            handoffs=[billing],
        )
        if colliding == "tool":
            return clashing
        return Agent(name="support", model=app.llm(support), handoffs=[clashing])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="unique name"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert support.calls == [] and oe.settled == []


async def test_a_plain_sdk_handoff_object_is_understood(oe: OE) -> None:
    # The target of agents.handoff() is read from the invoker the SDK builds.
    # This fails if an SDK upgrade changes that shape, rather than letting the
    # adapter run a handoff it cannot validate.
    support = ScriptedNativeModel(
        [function_call_item("call-1", "{}", name="ask_billing")]
    )
    billing = ScriptedNativeModel([text_item("It is paid.")])
    app = _team_app(
        support,
        billing,
        [],
        to_billing=lambda a: handoff(
            a, tool_name_override="ask_billing", tool_description_override="Billing."
        ),
    )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert support.calls[0]["tools"] == ["ask_billing"]
    assert events[-1].data["response"] == "It is paid."


def _shipping_model(
    before_first_call: Callable[[str], None] | None = None,
) -> ScriptedNativeModel:
    """A specialist that tracks the order named in its request, then reports."""

    def respond(items: Any) -> list[Any]:
        last = items[-1]
        if last.get("type") == "function_call_output":
            return [text_item(str(last["output"]))]
        order_id = str(last["content"])
        if before_first_call is not None:
            before_first_call(order_id)
        return [
            function_call_item(
                f"track-{order_id}",
                json.dumps({"order_id": order_id}),
                name="track_shipment",
            )
        ]

    return ScriptedNativeModel(respond=respond)


def _ask(call_id: str, order_id: str) -> Any:
    return function_call_item(
        call_id, json.dumps({"input": order_id}), name="ask_shipping"
    )


def _specialist_app(
    support: ScriptedNativeModel,
    shipping: ScriptedNativeModel,
    as_tool: Callable[[Agent[Any]], Any] | None = None,
    track: Callable[[str], str] = lambda order_id: f"{order_id} arrives Friday",
) -> App:
    """A support agent that asks a shipping specialist agent, as a tool."""
    app = App(app_name="support")

    @app.tool()
    def track_shipment(order_id: str) -> str:
        """Track one order's shipment."""
        return track(order_id)

    @app.entrypoint
    def build() -> Agent[Any]:
        specialist = Agent(
            name="shipping",
            instructions="Track the order.",
            model=app.llm(shipping, llm_id="shipping"),
            tools=app.tools(),
        )
        ask = (
            as_tool(specialist)
            if as_tool
            else specialist.as_tool(
                tool_name="ask_shipping", tool_description="Ask the specialist."
            )
        )
        return Agent(name="support", model=app.llm(support), tools=[ask])

    return app


def _model_paths(proxies: Any) -> list[tuple[str, list[tuple[str, int]]]]:
    return [(p.kwargs["llm_id"], p.calls[0]["path"]) for p in proxies.instances]


async def test_agent_tool_runs_as_a_nested_operation(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxies = route_to_tool_pod(monkeypatch, preallocate_tools=True)
    support = ScriptedNativeModel([_ask("call-1", "A1")], [text_item("Friday.")])
    app = _specialist_app(support, _shipping_model())

    events = await _attempt(app, oe, RequestContext(session_id="s-1"))

    nested = [("agent", 1), ("ask_shipping", 1)]
    # The specialist's model and tool calls are activities of the nested
    # operation; the support agent's stay on the root.
    assert _model_paths(proxies) == [
        ("__default__", [("agent", 1)]),
        ("shipping", nested),
        ("shipping", nested),
        ("__default__", [("agent", 1)]),
    ]
    assert oe.tool_paths == {"track-A1": nested}
    # The support model sees only the tool's answer.
    answer = support.calls[1]["input"][-1]
    assert (answer["call_id"], answer["output"]) == ("call-1", "A1 arrives Friday")
    # The caller sees the nested run between subagent boundaries, apart from
    # the support agent's own reply.
    assert [
        (e.event, e.data.get("source"), e.data.get("tool_call_id")) for e in events[:-1]
    ] == [
        ("subagent_start", "ask_shipping", "call-1"),
        ("token", "ask_shipping", "call-1"),
        ("subagent_end", "ask_shipping", "call-1"),
        ("token", "", ""),
    ]
    assert events[0].data["description"] == "A1"
    assert events[2].data["summary"] == "A1 arrives Friday"
    # Only the support conversation is committed; the nested run is replayed
    # from its recorded activities.
    oe.previous_state = oe.settled[0]
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [item.get("call_id") for item in committed[1:3]] == ["call-1", "call-1"]
    assert len(committed) == 4


async def test_concurrent_agent_tool_calls_keep_call_ordered_occurrences(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxies = route_to_tool_pod(monkeypatch, preallocate_tools=True)
    a2_asked = threading.Event()

    def a2_first(order_id: str) -> None:
        # A1's nested run reaches OE only after A2's has.
        if order_id == "A1":
            assert a2_asked.wait(timeout=5), "the nested runs were not concurrent"
        a2_asked.set()

    support = ScriptedNativeModel(
        [_ask("call-1", "A1"), _ask("call-2", "A2")], [text_item("Both Friday.")]
    )
    app = _specialist_app(support, _shipping_model(a2_first))

    await _attempt(app, oe, RequestContext(session_id="s-1"))

    # Occurrences follow the model's call order, not which run started first.
    assert oe.tool_paths == {
        "track-A1": [("agent", 1), ("ask_shipping", 1)],
        "track-A2": [("agent", 1), ("ask_shipping", 2)],
    }
    assert sorted(path for llm, path in _model_paths(proxies) if llm == "shipping") == [
        [("agent", 1), ("ask_shipping", 1)],
        [("agent", 1), ("ask_shipping", 1)],
        [("agent", 1), ("ask_shipping", 2)],
        [("agent", 1), ("ask_shipping", 2)],
    ]
    outputs = {
        item["call_id"]: item["output"]
        for item in support.calls[1]["input"]
        if item.get("type") == "function_call_output"
    }
    assert outputs == {"call-1": "A1 arrives Friday", "call-2": "A2 arrives Friday"}


async def test_a_later_call_to_the_same_agent_tool_is_the_next_occurrence(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    support = ScriptedNativeModel(
        [_ask("call-1", "A1")], [_ask("call-2", "A2")], [text_item("Both Friday.")]
    )
    app = _specialist_app(support, _shipping_model())

    await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert oe.tool_paths == {
        "track-A1": [("agent", 1), ("ask_shipping", 1)],
        "track-A2": [("agent", 1), ("ask_shipping", 2)],
    }


async def test_a_failed_nested_run_fails_the_turn(oe: OE) -> None:
    def offline(_order_id: str) -> str:
        raise RuntimeError("carrier service unavailable")

    support = ScriptedNativeModel([_ask("call-1", "A1")])
    app = _specialist_app(support, _shipping_model(), track=offline)

    with pytest.raises(ToolExecutionError, match="carrier service unavailable"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    # The failure was not turned into text for the support model to answer from.
    assert len(support.calls) == 1 and oe.settled == []


def _flagged_function_tool(_agent: Agent[Any]) -> Any:
    @function_tool
    def ask_shipping(input: str) -> str:
        return input

    ask_shipping._is_agent_tool = True  # pyright: ignore[reportAttributeAccessIssue]
    return ask_shipping


def _as_tool(**options: Any) -> Callable[[Agent[Any]], Any]:
    return lambda agent: agent.as_tool(
        tool_name="ask_shipping", tool_description="Ask.", **options
    )


def _as_tool_with(**attributes: Any) -> Callable[[Agent[Any]], Any]:
    """An agent tool the builder changes after ``as_tool`` returns it."""

    def build(agent: Agent[Any]) -> Any:
        tool = _as_tool()(agent)
        for name, value in attributes.items():
            setattr(tool, name, value)
        return tool

    return build


def _allow(_data: Any) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.allow()


async def _always(*_args: Any) -> bool:
    return True


@pytest.mark.parametrize(
    ("as_tool", "match"),
    [
        (_as_tool(on_stream=lambda _event: None), "on_stream"),
        (_as_tool(hooks=RunHooks()), "hooks"),
        (_as_tool(run_config=RunConfig()), "run_config"),
        (_as_tool(session=SQLiteSession("nested")), "session"),
        (_as_tool(previous_response_id="resp-1"), "previous_response_id"),
        (_as_tool(conversation_id="conv-1"), "conversation_id"),
        (_as_tool(needs_approval=_always), "needs_approval"),
        (_as_tool(is_enabled=lambda _c, _a: True), "is_enabled"),
        (
            _as_tool_with(
                tool_input_guardrails=[ToolInputGuardrail(guardrail_function=_allow)]
            ),
            "guardrails",
        ),
        (
            _as_tool_with(
                tool_output_guardrails=[ToolOutputGuardrail(guardrail_function=_allow)]
            ),
            "guardrails",
        ),
        (_as_tool_with(timeout_seconds=1), "a native timeout"),
        (_flagged_function_tool, r"built with Agent\.as_tool"),
        # An agent that can reach itself through its own agent tool.
        (
            lambda agent: (
                agent.tools.append(_as_tool()(agent)),
                _as_tool()(agent),
            )[1],
            "reachable from its own agent tool",
        ),
    ],
)
async def test_agent_tools_replay_cannot_reproduce_fail_before_any_model_call(
    oe: OE, as_tool: Callable[[Agent[Any]], Any], match: str
) -> None:
    support = ScriptedNativeModel([text_item("unused")])
    app = _specialist_app(support, ScriptedNativeModel(), as_tool=as_tool)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert support.calls == [] and oe.settled == [] and oe.tool_calls == []


@pytest.mark.parametrize("behind_a_handoff", [False, True])
async def test_approval_inside_an_agent_tool_fails_before_any_work_for_now(
    oe: OE, behind_a_handoff: bool
) -> None:
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        return f"refunded {order_id}"

    support = ScriptedNativeModel([text_item("unused")])

    @app.entrypoint
    def build() -> Agent[Any]:
        approver = Agent(
            name="approver",
            model=app.llm(ScriptedNativeModel(), llm_id="approver"),
            tools=_tools_needing_approval(app, "refund"),
        )
        # The approval is refused wherever the nested run could reach it.
        refunds_agent = (
            Agent(
                name="refunds",
                model=app.llm(ScriptedNativeModel(), llm_id="refunds"),
                handoffs=[approver],
            )
            if behind_a_handoff
            else approver
        )
        ask = refunds_agent.as_tool(tool_name="ask_refunds", tool_description="Ask.")
        return Agent(name="support", model=app.llm(support), tools=[ask])

    with pytest.raises(
        UnsupportedDurableOpenAIAgentsError,
        match="'refund' needs approval inside the agent tool",
    ):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    # Refused from the agent graph: no model, tool, or wait.
    assert support.calls == [] and oe.tool_calls == [] and oe.waits == {}


async def test_replayed_nested_progress_is_not_shown_again_on_resume(oe: OE) -> None:
    refunds: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def track_shipment(order_id: str) -> str:
        """Track one order's shipment."""
        return f"{order_id} arrives Friday"

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        refunds.append(order_id)
        return f"refunded {order_id}"

    # The specialist runs before the refund's approval, so a resumed attempt
    # replays it.
    turns = [
        [_ask("call-1", "A1")],
        [function_call_item("call-2", '{"order_id":"A1"}', name="refund")],
    ]
    support = ScriptedNativeModel(*turns, *turns, [text_item("Refunded.")])

    @app.entrypoint
    def build() -> Agent[Any]:
        tools = {tool.name: tool for tool in _tools_needing_approval(app, "refund")}
        specialist = Agent(
            name="shipping",
            model=app.llm(_shipping_model(), llm_id="shipping"),
            tools=[tools["track_shipment"]],
        )
        ask = specialist.as_tool(tool_name="ask_shipping", tool_description="Ask.")
        return Agent(
            name="support", model=app.llm(support), tools=[ask, tools["refund"]]
        )

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert [e.event for e in first] == [
        "subagent_start",
        "token",
        "subagent_end",
        "suspend",
    ]
    wait_id = first[-1].data["interrupts"][0]["id"]
    oe.answer(wait_id, {"confirmed": True})

    resumed = await _attempt(
        app,
        oe,
        RequestContext(
            session_id="s-1", resume=True, resume_data={wait_id: {"confirmed": True}}
        ),
    )

    # The caller saw the specialist's run before the wait; only what follows
    # the answered approval is shown now.
    assert [(e.event, e.data.get("source")) for e in resumed[:-1]] == [("token", "")]
    assert resumed[-1].data["response"] == "Refunded." and refunds == ["A1"]


async def test_failed_turn_waits_for_a_nested_agents_model_request(oe: OE) -> None:
    requested = threading.Event()
    failing = threading.Event()
    released: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        # Fail only once the specialist's model request is under way.
        assert requested.wait(timeout=5)
        failing.set()
        raise RuntimeError("inventory service unavailable")

    def slow_model(_order_id: str) -> None:
        requested.set()
        # Stay in flight until after the sibling fails, however slowly the
        # two threads are scheduled.
        assert failing.wait(timeout=5)
        time.sleep(0.3)
        released.append("specialist model request")

    support = ScriptedNativeModel(
        [_ask("call-1", "A1"), function_call_item("call-2", '{"order_id":"A2"}')]
    )

    @app.entrypoint
    def build() -> Agent[Any]:
        specialist = Agent(
            name="shipping",
            model=app.llm(_shipping_model(slow_model), llm_id="shipping"),
        )
        ask = specialist.as_tool(tool_name="ask_shipping", tool_description="Ask.")
        return Agent(name="support", model=app.llm(support), tools=[ask, *app.tools()])

    with pytest.raises(ToolExecutionError, match="inventory service unavailable"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    # No model request outlives the failed attempt.
    assert released == ["specialist model request"] and oe.settled == []
