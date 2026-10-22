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
from types import SimpleNamespace
from typing import Any, Literal, cast

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
    Runner,
    SQLiteSession,
    ToolApprovalItem,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    agent_tool_state,
    default_tool_error_function,
    function_tool,
    handoff,
)
from agents.exceptions import ModelBehaviorError, UserError
from agents.guardrail import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
from agents.mcp import MCPServerStdio
from agents.result import RunResultStreaming
from agents.tool import DEFAULT_APPROVAL_REJECTION_MESSAGE
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall
from openai.types.shared import Reasoning
from pydantic import BaseModel

from agent_engine_runner_shared import RuntimeMode
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    FinalizeStepCommand,
    MessageRole,
    StateSnapshot,
)
from agent_engine_runner_shared.hooks import get_named_llm
from agent_engine_runner_shared.mcp_tools import MCPConfigError, MCPToolError
from agent_engine_runner_shared.secure_wrapper import (
    CALL_INTERRUPTED_ARTIFACT_KEY,
    INTERRUPTED_CALL_CONTENT,
    PolicyDeniedException,
    ToolExecutionError,
    ToolPlatformError,
)
from agent_engine_runner_shared.workflow.activity import semantic_input_from_json
from agent_engine_runner_shared.workflow.context import (
    ChildOperationBoundary,
    allocate_activity_ordinal,
    attempt_context_scope,
    child_operation_boundary_scope,
    current_operation_path,
    preallocate_child_operation_ordinals,
    tool_activity_key,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_value,
    proto_struct_to_json,
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
from agent_engine_sdk_openai_agents.review import REVIEW_TOOL_NAME
from agent_engine_sdk_openai_agents.secure_model import (
    _runner_response,  # pyright: ignore[reportPrivateUsage]
)
from agent_engine_sdk_openai_agents.state import (
    STATE_KEY,
    decode_active_agent,
    decode_state,
    encode_state,
)
from agent_engine_sdk_openai_agents.tools import ModelCall, json_text
from tests.support import (
    FakeWrapper,
    ReviewingOE,
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
        # The session the attempts run for, and, on a branch's first
        # execution, where OE says the branch came from.
        self.session_id = "s-1"
        self.branch_lineage: BranchLineage | None = None
        self.settled: list[StateSnapshot] = []
        # The session each settlement was made for.
        self.settled_sessions: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        # The operation path each tool call ran under, by call id.
        self.tool_paths: dict[str, list[tuple[str, int]]] = {}
        # Call ids OE stops mid-flight, as when a user stops a running call.
        self.stopped_call_ids: set[str] = set()
        # Call ids whose tool runs but whose result OE never acknowledges.
        self.unacknowledged_call_ids: set[str] = set()
        # The execution policy each tool was wrapped with, by tool name.
        self.tool_policies: dict[str, dict[str, Any]] = {}
        monkeypatch.setattr(agent_module, "current_attempt_context", self.attempt)
        monkeypatch.setattr(agent_module, "settle_execution", self.settle)
        monkeypatch.setattr(tools_module, "create_secure_tool_function", self.secure)
        # Approval waits: OE's suspension frontier, keyed by position, holding
        # each wait's answer once a caller resumes it.
        self.waits: dict[str, tuple[str, dict[str, Any] | None]] = {}
        # The operation path each wait was recorded under, by wait id.
        self.wait_paths: dict[str, list[tuple[str, int]]] = {}
        self.wait_steps: dict[str, int] = {}
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
        self.wait_steps[activity_id] = position.step_ordinal
        self.wait_paths[activity_id] = [
            (segment.name, segment.ordinal)
            for segment in position.operation_path.segments
        ]
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
            workflow_identity=WorkflowIdentity(
                session_id=self.session_id, execution_id="exec-1"
            )
        )
        if self.previous_state is not None:
            attempt.previous_state.CopyFrom(self.previous_state)
        if self.branch_lineage is not None:
            attempt.branch_lineage.CopyFrom(self.branch_lineage)
        return attempt

    def branch(self, source: StateSnapshot) -> None:
        """What OE does for a branch: a new session whose first execution
        starts from the source session's committed state."""
        # The lineage always names s-1 as the source, whatever session the
        # snapshot says it belongs to: that mismatch is how a test hands the
        # adapter a source it must refuse.
        self.session_id = "branch-1"
        self.previous_state = source
        self.branch_lineage = BranchLineage(
            source_workflow_identity=WorkflowIdentity(
                session_id="s-1", execution_id="exec-1"
            ),
            source_step_ordinal=1,
            source_state_hash="hash-1",
        )

    def next_turn(self) -> None:
        """A later execution in the same session: it starts from the session's
        last committed state, with no lineage."""
        self.previous_state = self.settled[-1]
        self.branch_lineage = None

    async def settle(self, attempt: AttemptContext, state: StateSnapshot) -> None:
        self.settled.append(state)
        self.settled_sessions.append(attempt.workflow_identity.session_id)

    def secure(
        self, *, original_tool: Any, response_format: str, **_policy: Any
    ) -> Any:
        # The real wrapper's content_and_artifact contract: (content, artifact),
        # where only OE sets the stopped-call marker.
        assert response_format == "content_and_artifact"
        self.tool_policies[_policy["tool_name"]] = _policy

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
            except PolicyDeniedException:
                # The real wrapper reports OE's denial of a call by type.
                raise
            except Exception as error:
                # The real wrapper reports a failed local tool this way.
                raise ToolExecutionError(str(error)) from error
            # Like the real durable wrapper, record only JSON tool results.
            json.dumps(result, allow_nan=False)
            if tool_call_id in self.unacknowledged_call_ids:
                raise ToolPlatformError("OE did not acknowledge tool result settlement")
            return result, None

        return call


def test_tools_carry_their_declared_unknown_outcome_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared: dict[str, Any] = {}

    def secure(**kwargs: Any) -> Any:
        declared[kwargs["tool_name"]] = kwargs.get("on_unknown_outcome")
        return lambda **arguments: None

    monkeypatch.setattr(tools_module, "create_secure_tool_function", secure)
    app = App(app_name="support")
    app._runtime.mode = RuntimeMode.AER

    @app.tool(on_unknown_outcome="rerun")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    @app.tool()
    def refund_order(order_id: str) -> str:
        """Refund an order."""
        return order_id

    app.tools()

    assert declared == {"lookup_order": "rerun", "refund_order": None}


def test_tool_rejects_an_unknown_outcome_policy_it_does_not_know() -> None:
    app = App(app_name="support")

    with pytest.raises(ValueError, match="on_unknown_outcome"):
        app.tool(on_unknown_outcome=cast(Any, "retry"))


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
            raise PolicyDeniedException("A1 is restricted")
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

    with pytest.raises(PolicyDeniedException, match="A1 is restricted"):
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
            raise PolicyDeniedException("A1 is restricted")
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


# How every OpenAI Agents release in the supported range begins its default
# text for a failed tool; newer releases leave out the error itself.
_TOOL_FAILED = "An error occurred while running the tool."


async def test_failed_tool_is_reported_to_the_model_as_the_sdk_does(oe: OE) -> None:
    native = ScriptedNativeModel(
        [function_call_item("call-1")], [text_item("The lookup failed.")]
    )
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise RuntimeError("inventory service unavailable")

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    await _turn(app, "Where is A1?", oe)

    # The installed OpenAI Agents SDK's default text for a tool that raises.
    tool_output = native.calls[1]["input"][-1]
    assert tool_output["output"] == default_tool_error_function(
        cast(Any, None), RuntimeError("inventory service unavailable")
    )
    assert len(oe.settled) == 1


async def test_tool_call_denied_by_policy_fails_the_turn_without_committing(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise PolicyDeniedException("A1 is restricted")

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    # The secure path's denial reaches the AER as is, not re-wrapped by the SDK.
    with pytest.raises(PolicyDeniedException, match="A1 is restricted"):
        await _turn(app, "Where is A1?", oe)
    assert oe.settled == []
    assert len(native.calls) == 1  # the model never saw the denial as tool output


async def test_platform_failure_after_the_tool_ran_fails_the_turn(oe: OE) -> None:
    native = ScriptedNativeModel(
        [function_call_item("call-1", name="refund_order")], [text_item("unused")]
    )
    app = App(app_name="support")
    refunded: list[str] = []

    @app.tool()
    def refund_order(order_id: str) -> str:
        """Refund an order."""
        refunded.append(order_id)
        return f"refunded {order_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    oe.unacknowledged_call_ids.add("call-1")
    with pytest.raises(ToolPlatformError, match="did not acknowledge"):
        await _turn(app, "Refund A1", oe)
    assert refunded == ["A1"]
    # Told the refund failed, the model could have refunded again.
    assert len(native.calls) == 1 and oe.settled == []


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


async def test_opaque_tool_result_is_reported_to_the_model_as_a_failed_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([function_call_item("call-1")], [text_item("done")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> Any:
        """Look up an order."""
        return object()

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    # Serialization runs where the tool runs, so it fails like any tool error.
    await _turn(app, "Where is A1?", oe)
    assert native.calls[1]["input"][-1]["output"].startswith(_TOOL_FAILED)
    assert len(oe.settled) == 1


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
    await _turn(app, "Where is A1?", oe)
    # The model reads the failure instead of a value a replay could not
    # reproduce.
    assert native.calls[1]["input"][-1]["output"].startswith(_TOOL_FAILED)


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


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model": "gpt-4.1"}, r"app\.llm"),
        ({"tools": [native_tool]}, r"app\.tools"),
        # Every agent a handoff can reach is held to the same rules.
        ({"handoffs": [Agent(name="billing")]}, r"'billing' model must be"),
        ({"instructions": lambda *_: "dynamic"}, "dynamic instructions"),
        ({"tool_use_behavior": "stop_on_first_tool"}, "tool_use_behavior"),
        ({"model_settings": ModelSettings(temperature=0)}, r"app\.llm"),
        (
            {"mcp_servers": [MCPServerStdio(params={"command": "true"})]},
            r"mcp\.servers in agent\.yaml",
        ),
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
        (
            ModelSettings(extra_args={"conversation": "conv-0", "prompt": {}}),
            r"extra_args sets conversation, prompt",
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

    def acknowledge(self, _client: object, context: Any) -> None:
        self.acknowledged.append(
            {
                "activity_id": context.activity_id,
                "attempt_id": context.attempt_id,
                "fencing_token": context.fencing_token,
                "execution_id": context.workflow_identity.execution_id,
                "settled_before": len(self.oe.settled),
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
        await approvals_module.record_approvals(oe.attempt(), [], {}, {})


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


def _approval(call_id: str) -> ToolApprovalItem:
    return ToolApprovalItem(
        agent=Agent(name="support"),
        raw_item=function_call_item(call_id, '{"order_id":"A1"}', name="refund"),
        tool_name="refund",
    )


def _own_calls(*call_ids: str) -> dict[str, ModelCall]:
    """Calls the turn's own agent made, in this order."""
    return {call_id: ModelCall((), index) for index, call_id in enumerate(call_ids)}


async def test_waits_keep_their_identity_whatever_order_the_sdk_reports(
    oe: OE,
) -> None:
    identities: list[dict[str, str]] = []
    for reported in (["call-1", "call-2"], ["call-2", "call-1"]):
        with attempt_context_scope(oe.attempt()):
            waits = await approvals_module.record_approvals(
                oe.attempt(),
                [_approval(call_id) for call_id in reported],
                _own_calls("call-1", "call-2"),
                {},
            )
        identities.append({wait.call_id: wait.activity_id for wait in waits})

    # A resumed run may report its interruptions in another order; each call
    # must still reach the wait it was given, or replay would pair answers
    # with the wrong calls.
    assert identities[0] == identities[1]
    assert [wait.call_id for wait in waits] == ["call-1", "call-2"]


@pytest.mark.parametrize(
    "reported",
    [
        ["call-0", "call-1", "n-1", "n-2", "call-3"],
        ["call-3", "n-2", "n-1", "call-1", "call-0"],
        ["n-2", "call-3", "call-0", "n-1", "call-1"],
    ],
)
async def test_nested_waits_keep_their_place_whatever_order_the_sdk_reports(
    oe: OE, reported: list[str]
) -> None:
    # The support model called refund twice, the specialist (call-2), then
    # refund again; the specialist's model then made two calls of its own.
    run = (ChildOperationBoundary(name="ask_refunds", occurrence_key="call-2"),)
    calls = {
        **_own_calls("call-0", "call-1", "call-2", "call-3"),
        "n-1": ModelCall(run, 4),
        "n-2": ModelCall(run, 5),
    }

    with attempt_context_scope(oe.attempt()):
        waits = await approvals_module.record_approvals(
            oe.attempt(), [_approval(call_id) for call_id in reported], calls, {}
        )

    # The specialist's waits take the place of the call that started its run,
    # in the order its model made them.
    assert [wait.call_id for wait in waits] == [
        "call-0",
        "call-1",
        "n-1",
        "n-2",
        "call-3",
    ]
    specialist_run = [("agent", 1), ("ask_refunds", 1)]
    assert [oe.wait_paths[wait.activity_id] for wait in waits] == [
        [("agent", 1)],
        [("agent", 1)],
        specialist_run,
        specialist_run,
        [("agent", 1)],
    ]


@pytest.mark.parametrize("reported", [["n-z", "n-a"], ["n-a", "n-z"]])
async def test_sibling_nested_runs_wait_in_the_order_oe_numbers_them(
    oe: OE, reported: list[str]
) -> None:
    # The support agent asked the desk (call-1), whose model asked the refunds
    # specialist twice: first with call id "z", then "a". The second
    # specialist's model answered first.
    desk = ChildOperationBoundary(name="ask_desk", occurrence_key="call-1")
    first = ChildOperationBoundary(name="ask_refunds", occurrence_key="z")
    second = ChildOperationBoundary(name="ask_refunds", occurrence_key="a")
    calls = {
        **_own_calls("call-1"),
        "z": ModelCall((desk,), 1),
        "a": ModelCall((desk,), 2),
        "n-a": ModelCall((desk, second), 3),
        "n-z": ModelCall((desk, first), 4),
    }

    with attempt_context_scope(oe.attempt()):
        # As the secure model does when the desk's response arrives.
        with child_operation_boundary_scope(desk):
            preallocate_child_operation_ordinals([first, second])
        waits = await approvals_module.record_approvals(
            oe.attempt(), [_approval(call_id) for call_id in reported], calls, {}
        )

    # Neither the call ids nor which specialist answered first decide it.
    assert [wait.call_id for wait in waits] == ["n-z", "n-a"]
    assert [oe.wait_paths[wait.activity_id] for wait in waits] == [
        [("agent", 1), ("ask_desk", 1), ("ask_refunds", 1)],
        [("agent", 1), ("ask_desk", 1), ("ask_refunds", 2)],
    ]


_SPECIALIST_RUN = (ChildOperationBoundary(name="ask_refunds", occurrence_key="call-1"),)


@pytest.mark.parametrize(
    ("reported", "calls", "refusal"),
    [
        (["call-1", "call-1"], _own_calls("call-1"), "distinct calls"),
        (["call-9"], _own_calls("call-1"), "distinct calls"),
        # The run that made the call was not started by a call of this turn.
        (
            ["n-1"],
            {**_own_calls("call-2"), "n-1": ModelCall(_SPECIALIST_RUN, 1)},
            "'n-1' belongs to an agent run that no call of this turn started",
        ),
    ],
    ids=["repeated-call", "call-not-in-response", "nested-run-not-in-response"],
)
async def test_approvals_must_name_distinct_calls_of_the_response(
    oe: OE, reported: list[str], calls: dict[str, ModelCall], refusal: str
) -> None:
    with (
        attempt_context_scope(oe.attempt()),
        pytest.raises(UnsupportedDurableOpenAIAgentsError, match=refusal),
    ):
        await approvals_module.record_approvals(
            oe.attempt(), [_approval(call_id) for call_id in reported], calls, {}
        )
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


def _ask(call_id: str, order_id: str, tool: str = "ask_shipping") -> Any:
    return function_call_item(call_id, json.dumps({"input": order_id}), name=tool)


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
        raise PolicyDeniedException("carrier service unavailable")

    support = ScriptedNativeModel([_ask("call-1", "A1")])
    app = _specialist_app(support, _shipping_model(), track=offline)

    with pytest.raises(PolicyDeniedException, match="carrier service unavailable"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    # The denial was not turned into text for the support model to answer from.
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


def _refunds_model(
    before_first_call: Callable[[str], None] | None = None,
    call_id: Callable[[str], str] = lambda order_id: f"refund-{order_id}",
) -> ScriptedNativeModel:
    """A specialist that asks to refund the order named in its request, then
    reports what the refund tool, or the person who rejected it, said."""

    def respond(items: Any) -> list[Any]:
        order_id = str(next(i["content"] for i in items if i.get("role") == "user"))
        said = [
            item["output"]
            for item in items
            if item.get("type") == "function_call_output"
            and item["call_id"] == call_id(order_id)
        ]
        if said:
            return [text_item(str(said[0]))]
        if before_first_call is not None:
            before_first_call(order_id)
        return [
            function_call_item(
                call_id(order_id), json.dumps({"order_id": order_id}), name="refund"
            )
        ]

    return ScriptedNativeModel(respond=respond)


def _refunds_app(
    support: ScriptedNativeModel,
    refunds: list[str],
    *,
    specialist: ScriptedNativeModel | None = None,
    behind_a_handoff: bool = False,
    support_refunds_too: bool = False,
    ask_needs_approval: bool = False,
) -> App:
    """A support agent that asks a refunds specialist agent, as a tool. A
    person approves every refund."""
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        refunds.append(order_id)
        return f"refunded {order_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        model = app.llm(specialist or _refunds_model(), llm_id="refunds")
        if behind_a_handoff:
            # The agent the tool runs hands the request to the one that
            # refunds; both belong to the same nested run.
            approver = Agent(
                name="approver",
                model=model,
                tools=_tools_needing_approval(app, "refund"),
            )
            to_approver = [
                function_call_item("handoff-1", "{}", name="transfer_to_approver")
            ]
            nested = Agent(
                name="refunds",
                model=app.llm(
                    ScriptedNativeModel(respond=lambda _items: to_approver),
                    llm_id="refunds-intake",
                ),
                handoffs=[approver],
            )
        else:
            nested = Agent(
                name="refunds",
                model=model,
                tools=_tools_needing_approval(app, "refund"),
            )
        ask = nested.as_tool(
            tool_name="ask_refunds",
            tool_description="Ask the specialist.",
            needs_approval=ask_needs_approval,
        )
        own = _tools_needing_approval(app, "refund") if support_refunds_too else []
        return Agent(name="support", model=app.llm(support), tools=[ask, *own])

    return app


_NESTED_REFUND = [("agent", 1), ("ask_refunds", 1)]


def _resume(answers: dict[str, dict[str, Any]], oe: OE) -> RequestContext:
    for wait_id, answer in answers.items():
        oe.answer(wait_id, answer)
    return RequestContext(session_id="s-1", resume=True, resume_data=answers)


@pytest.mark.parametrize("behind_a_handoff", [False, True])
async def test_approval_inside_an_agent_tool_waits_at_the_nested_path(
    oe: OE, monkeypatch: pytest.MonkeyPatch, behind_a_handoff: bool
) -> None:
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    refunds: list[str] = []
    asking = [_ask("call-1", "A1", "ask_refunds")]
    # A replacement attempt replays the support model's call.
    support = ScriptedNativeModel(asking, list(asking), [text_item("Refunded.")])
    app = _refunds_app(support, refunds, behind_a_handoff=behind_a_handoff)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # The specialist's run has not ended: it is waiting.
    assert [e.event for e in first] == ["subagent_start", "suspend"]
    (interrupt,) = first[-1].data["interrupts"]
    assert interrupt["value"]["tool_call"] == {
        "name": "refund",
        "args": {"order_id": "A1"},
        "call_id": "refund-A1",
    }
    # The wait belongs to the specialist's run, beside its other activities.
    assert oe.wait_paths == {interrupt["id"]: _NESTED_REFUND}
    assert refunds == [] and oe.tool_calls == [] and oe.settled == []

    approved = _resume({interrupt["id"]: {"confirmed": True}}, oe)
    resumed = await _attempt(app, oe, approved)

    assert resumed[-1].data["response"] == "Refunded."
    assert refunds == ["A1"] and oe.tool_paths == {"refund-A1": _NESTED_REFUND}
    # The resumed stream reopens the specialist's run for what follows the
    # answered wait, then shows the support agent's reply.
    assert [
        (e.event, e.data.get("source"), e.data.get("tool_call_id"))
        for e in resumed[:-1]
    ] == [
        ("subagent_start", "ask_refunds", "call-1"),
        ("token", "ask_refunds", "call-1"),
        ("subagent_end", "ask_refunds", "call-1"),
        ("token", "", ""),
    ]
    assert resumed[2].data["summary"] == "refunded A1"
    answer = support.calls[-1]["input"][-1]
    assert (answer["call_id"], answer["output"]) == ("call-1", "refunded A1")
    # Only the support conversation is committed.
    oe.previous_state = oe.settled[0]
    committed = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [item.get("call_id") for item in committed[1:3]] == ["call-1", "call-1"]
    assert len(committed) == 4


async def test_another_executions_stopped_nested_run_is_not_continued(
    oe: OE,
) -> None:
    refunds: list[str] = []
    support = ScriptedNativeModel([_ask("call-1", "A1", "ask_refunds")])
    specialist = _refunds_model()
    app = _refunds_app(support, refunds, specialist=specialist)
    # Another execution in this process made the same call and its specialist
    # is stopped for an approval.
    call = LLMToolCall(id="call-1", name="ask_refunds", args={"input": "A1"})
    (theirs,) = _runner_response(
        LLMResponse(content="", tool_calls=[call]), execution_id="exec-2", step=1
    ).output
    assert isinstance(theirs, ResponseFunctionToolCall)
    stopped = SimpleNamespace(interruptions=[object()])
    agent_tool_state.record_agent_tool_run_result(theirs, cast(Any, stopped))
    try:
        events = await _attempt(app, oe, RequestContext(session_id="s-1"))
    finally:
        agent_tool_state.drop_agent_tool_run_result(theirs)

    # This execution ran its own specialist and waits on its own approval.
    assert events[-1].event == "suspend" and len(specialist.calls) == 1


async def test_turn_cancelled_while_draining_still_drops_its_stopped_nested_runs(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    asking = [_ask("call-1", "A1", "ask_refunds")]
    support = ScriptedNativeModel(asking, list(asking))
    specialist = _refunds_model()
    app = _refunds_app(support, [], specialist=specialist)
    drain = tools_module.TurnContext.drain

    async def cancelled_drain(self: tools_module.TurnContext) -> None:
        await drain(self)
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(tools_module.TurnContext, "drain", cancelled_drain)
        with pytest.raises(asyncio.CancelledError):
            await _attempt(app, oe, RequestContext(session_id="s-1"))

    again = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # The next attempt replays the specialist instead of continuing the run
    # the cancelled one left stopped.
    assert again[-1].event == "suspend" and len(specialist.calls) == 2


async def test_rejected_approval_inside_an_agent_tool_reaches_the_nested_agent(
    oe: OE,
) -> None:
    refunds: list[str] = []
    asking = [_ask("call-1", "A1", "ask_refunds")]
    support = ScriptedNativeModel(asking, list(asking), [text_item("No refund.")])
    specialist = _refunds_model()
    app = _refunds_app(support, refunds, specialist=specialist)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]

    rejection = {"confirmed": False, "rejection_message": "Not without a receipt."}
    resumed = await _attempt(app, oe, _resume({wait_id: rejection}, oe))

    assert resumed[-1].data["response"] == "No refund."
    assert refunds == [] and oe.tool_calls == []
    # The specialist's model reads the rejection as its call's output, and the
    # support model reads the specialist's answer.
    # The resumed attempt replayed the specialist's first model call instead of
    # continuing the stopped run the first attempt left behind in this process.
    assert len(specialist.calls) == 3
    told = specialist.calls[-1]["input"][-1]
    assert (told["call_id"], told["output"]) == ("refund-A1", "Not without a receipt.")
    assert support.calls[-1]["input"][-1]["output"] == "Not without a receipt."


async def test_root_and_nested_approvals_resolve_in_one_resume(oe: OE) -> None:
    refunds: list[str] = []
    batch = [
        _ask("call-1", "A1", "ask_refunds"),
        function_call_item("call-2", '{"order_id":"A2"}', name="refund"),
    ]
    support = ScriptedNativeModel(batch, list(batch), [text_item("A1 only.")])
    app = _refunds_app(support, refunds, support_refunds_too=True)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # One frontier, in the support model's call order: the specialist's wait
    # sorts with the call that started its run.
    interrupts = first[-1].data["interrupts"]
    assert [i["value"]["tool_call"]["call_id"] for i in interrupts] == [
        "refund-A1",
        "call-2",
    ]
    nested, own = (i["id"] for i in interrupts)
    assert oe.wait_paths == {nested: _NESTED_REFUND, own: [("agent", 1)]}
    assert refunds == []

    answers = {
        nested: {"confirmed": True},
        own: {"confirmed": False, "rejection_message": "A2 needs a manager."},
    }
    resumed = await _attempt(app, oe, _resume(answers, oe))

    assert resumed[-1].data["response"] == "A1 only." and refunds == ["A1"]
    outputs = {
        item["call_id"]: str(item["output"])
        for item in support.calls[-1]["input"]
        if item.get("type") == "function_call_output"
    }
    assert outputs["call-1"] == "refunded A1"
    assert "A2 needs a manager." in outputs["call-2"]


async def test_two_approvals_inside_one_nested_run_share_its_path(oe: OE) -> None:
    refunds: list[str] = []

    def both(items: Any) -> list[Any]:
        said = [i["output"] for i in items if i.get("type") == "function_call_output"]
        if said:
            return [text_item("; ".join(str(output) for output in said))]
        return [
            function_call_item("n-1", '{"order_id":"A1"}', name="refund"),
            function_call_item("n-2", '{"order_id":"A2"}', name="refund"),
        ]

    asking = [_ask("call-1", "A1 and A2", "ask_refunds")]
    support = ScriptedNativeModel(asking, list(asking), [text_item("A1 only.")])
    specialist = ScriptedNativeModel(respond=both)
    app = _refunds_app(support, refunds, specialist=specialist)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    interrupts = first[-1].data["interrupts"]
    assert [i["value"]["tool_call"]["call_id"] for i in interrupts] == ["n-1", "n-2"]
    wait_1, wait_2 = (i["id"] for i in interrupts)
    assert oe.wait_paths == {wait_1: _NESTED_REFUND, wait_2: _NESTED_REFUND}

    answers = {
        wait_1: {"confirmed": True},
        wait_2: {"confirmed": False, "rejection_message": "A2 needs a manager."},
    }
    resumed = await _attempt(app, oe, _resume(answers, oe))

    assert resumed[-1].data["response"] == "A1 only." and refunds == ["A1"]
    answer = support.calls[-1]["input"][-1]["output"]
    assert answer == "refunded A1; A2 needs a manager."


async def test_concurrent_nested_approvals_wait_under_their_own_occurrences(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    refunds: list[str] = []
    a2_asked = threading.Event()

    def a2_first(order_id: str) -> None:
        # A1's specialist reaches its model only after A2's has.
        if order_id == "A1":
            assert a2_asked.wait(timeout=5), "the nested runs were not concurrent"
        a2_asked.set()

    batch = [_ask("call-1", "A1", "ask_refunds"), _ask("call-2", "A2", "ask_refunds")]
    support = ScriptedNativeModel(batch, list(batch), [text_item("Both refunded.")])
    app = _refunds_app(support, refunds, specialist=_refunds_model(a2_first))

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # Each wait is under its own run, in the support model's call order,
    # whichever specialist asked first.
    interrupts = first[-1].data["interrupts"]
    assert [i["value"]["tool_call"]["call_id"] for i in interrupts] == [
        "refund-A1",
        "refund-A2",
    ]
    wait_1, wait_2 = (i["id"] for i in interrupts)
    assert oe.wait_paths == {
        wait_1: [("agent", 1), ("ask_refunds", 1)],
        wait_2: [("agent", 1), ("ask_refunds", 2)],
    }

    a2_asked.clear()
    answers = {wait_1: {"confirmed": True}, wait_2: {"confirmed": True}}
    resumed = await _attempt(app, oe, _resume(answers, oe))

    assert resumed[-1].data["response"] == "Both refunded."
    assert sorted(refunds) == ["A1", "A2"]
    assert oe.tool_paths == {
        "refund-A1": [("agent", 1), ("ask_refunds", 1)],
        "refund-A2": [("agent", 1), ("ask_refunds", 2)],
    }


@pytest.mark.parametrize("refund_fails", [False, True])
async def test_approval_two_agent_tools_deep_waits_at_the_innermost_path(
    oe: OE, monkeypatch: pytest.MonkeyPatch, refund_fails: bool
) -> None:
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    refunds: list[str] = []
    app = App(app_name="support")

    @app.tool()
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        refunds.append(order_id)
        if refund_fails:
            raise PolicyDeniedException("payments are down")
        return f"refunded {order_id}"

    def desk(items: Any) -> list[Any]:
        # The desk passes the request on to the refunds specialist.
        last = items[-1]
        if last.get("type") == "function_call_output":
            return [text_item(str(last["output"]))]
        return [_ask("desk-1", str(last["content"]), "ask_refunds")]

    asking = [_ask("call-1", "A1", "ask_desk")]
    support = ScriptedNativeModel(asking, list(asking), [text_item("Refunded.")])

    @app.entrypoint
    def build() -> Agent[Any]:
        specialist = Agent(
            name="refunds",
            model=app.llm(_refunds_model(), llm_id="refunds"),
            tools=_tools_needing_approval(app, "refund"),
        )
        ask_refunds = specialist.as_tool(
            tool_name="ask_refunds", tool_description="Ask the specialist."
        )
        front_desk = Agent(
            name="desk",
            model=app.llm(ScriptedNativeModel(respond=desk), llm_id="desk"),
            tools=[ask_refunds],
        )
        ask_desk = front_desk.as_tool(
            tool_name="ask_desk", tool_description="Ask the desk."
        )
        return Agent(name="support", model=app.llm(support), tools=[ask_desk])

    held = agent_tool_state._agent_tool_run_results_by_obj  # pyright: ignore[reportPrivateUsage]
    held_before = set(held)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    (interrupt,) = first[-1].data["interrupts"]
    innermost = [("agent", 1), ("ask_desk", 1), ("ask_refunds", 1)]
    assert oe.wait_paths == {interrupt["id"]: innermost} and refunds == []

    approved = _resume({interrupt["id"]: {"confirmed": True}}, oe)
    if refund_fails:
        with pytest.raises(PolicyDeniedException, match="payments are down"):
            await _attempt(app, oe, approved)
    else:
        resumed = await _attempt(app, oe, approved)
        assert resumed[-1].data["response"] == "Refunded."
    assert refunds == ["A1"] and oe.tool_paths == {"refund-A1": innermost}
    # Resuming makes the SDK copy the stopped runs under scopes of its own;
    # a long-lived runner must not keep them once the turn is over, however
    # it ended.
    assert set(held) == held_before


async def test_a_call_id_shared_by_two_nested_runs_fails_the_turn(oe: OE) -> None:
    refunds: list[str] = []
    batch = [_ask("call-1", "A1", "ask_refunds"), _ask("call-2", "A2", "ask_refunds")]
    support = ScriptedNativeModel(batch)
    # Both specialists' models name their call the same.
    specialist = _refunds_model(call_id=lambda _order_id: "refund-1")
    app = _refunds_app(support, refunds, specialist=specialist)

    # An approval names its call by id alone, so the turn could not tell which
    # run a wait belongs to.
    with pytest.raises(
        UnsupportedDurableOpenAIAgentsError, match="used by two agent runs"
    ):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert refunds == [] and oe.waits == {} and oe.settled == []


async def test_a_call_id_shared_with_a_nested_run_fails_without_any_approval(
    oe: OE,
) -> None:
    # The support model's call to the specialist has the id the specialist's
    # model gives its own call. Neither call needs approval.
    support = ScriptedNativeModel([_ask("track-A1", "A1")])
    app = _specialist_app(support, _shipping_model())

    with pytest.raises(
        UnsupportedDurableOpenAIAgentsError, match="used by two agent runs"
    ):
        await _attempt(app, oe, RequestContext(session_id="s-1"))
    assert oe.tool_calls == [] and oe.settled == []


async def test_an_agent_tool_that_needs_approval_waits_before_its_run(oe: OE) -> None:
    refunds: list[str] = []
    asking = [_ask("call-1", "A1", "ask_refunds")]
    support = ScriptedNativeModel(asking, list(asking), [text_item("Refunded.")])
    specialist = _refunds_model()
    app = _refunds_app(support, refunds, specialist=specialist, ask_needs_approval=True)

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # The call to the specialist is the support agent's own call.
    assert [e.event for e in first] == ["suspend"]
    (interrupt,) = first[-1].data["interrupts"]
    assert interrupt["value"]["tool_call"]["name"] == "ask_refunds"
    assert oe.wait_paths == {interrupt["id"]: [("agent", 1)]}
    assert specialist.calls == []

    # Approved, the specialist runs and asks for its own approval.
    approved = _resume({interrupt["id"]: {"confirmed": True}}, oe)
    second = await _attempt(app, oe, approved)

    assert [e.event for e in second] == ["subagent_start", "suspend"]
    (nested,) = second[-1].data["interrupts"]
    assert oe.wait_paths[nested["id"]] == _NESTED_REFUND and refunds == []


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
        raise PolicyDeniedException("A1 is restricted")

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

    with pytest.raises(PolicyDeniedException, match="A1 is restricted"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    # No model request outlives the failed attempt.
    assert released == ["specialist model request"] and oe.settled == []


def _envelope(snapshot: StateSnapshot) -> dict[str, Any]:
    properties = proto_struct_to_json(snapshot.properties)
    return cast(dict[str, Any], properties[STATE_KEY])


async def test_branch_continues_from_its_source_and_commits_as_its_own(
    oe: OE,
) -> None:
    support = ScriptedNativeModel([_transfer()])
    billing = ScriptedNativeModel(
        [text_item("Invoice INV-1 is paid.")],
        [text_item("By card.")],
        [text_item("On 1 May.")],
    )
    app = _team_app(support, billing, [])
    await _attempt(app, oe, RequestContext(session_id="s-1"), "Is INV-1 paid?")
    source = oe.settled[0]

    oe.branch(source)
    events = await _attempt(app, oe, RequestContext(session_id="branch-1"), "How?")

    assert events[-1].data["response"] == "By card."
    # The branch starts where the source stopped: with the agent that ended
    # the source's turn, and the source's conversation ahead of the new message.
    assert len(support.calls) == 1
    asked = billing.calls[1]["input"]
    assert asked[0] == {"role": "user", "content": "Is INV-1 paid?"}
    assert asked[-1] == {"role": "user", "content": "How?"}
    assert len(asked) == len(_envelope(source)["items"]) + 1
    # It settles for its own session and stamps that session on its state;
    # nothing is settled for the source.
    assert oe.settled_sessions == ["s-1", "branch-1"]
    committed = _envelope(oe.settled[1])
    assert (committed["session_id"], committed["active_agent"]) == (
        "branch-1",
        "billing",
    )
    assert len(committed["items"]) == len(_envelope(source)["items"]) + 2

    # The branch's next turn is an ordinary turn of the branch session.
    oe.next_turn()
    events = await _attempt(app, oe, RequestContext(session_id="branch-1"), "When?")

    assert events[-1].data["response"] == "On 1 May."
    assert _envelope(oe.settled[2])["session_id"] == "branch-1"
    assert oe.settled_sessions == ["s-1", "branch-1", "branch-1"]


async def test_approval_inside_a_branchs_first_turn_resumes_there(oe: OE) -> None:
    refunds: list[str] = []
    source_turn = ScriptedNativeModel([text_item("A1 shipped.")])
    await _attempt(
        _refund_app(source_turn, refunds), oe, RequestContext(session_id="s-1"), "A1?"
    )
    source = oe.settled[0]
    native = _refund_turns()
    app = _refund_app(native, refunds)

    oe.branch(source)
    first = await _attempt(app, oe, RequestContext(session_id="branch-1"), "Refund A1.")

    (interrupt,) = first[-1].data["interrupts"]
    # The wait belongs to the branch's first step, the one after the cutoff.
    assert oe.wait_steps == {interrupt["id"]: 2} and refunds == []

    # OE gives the resumed attempt the same source state and lineage.
    approved = {interrupt["id"]: {"confirmed": True}}
    oe.answer(interrupt["id"], approved[interrupt["id"]])
    resumed = await _attempt(
        app,
        oe,
        RequestContext(session_id="branch-1", resume=True, resume_data=approved),
    )

    assert resumed[-1].data["response"] == "Done." and refunds == ["A1"]
    committed = _envelope(oe.settled[1])
    assert committed["session_id"] == "branch-1"
    # The source's turn, then the branch's: question, note, call, output, answer.
    assert len(committed["items"]) == len(_envelope(source)["items"]) + 5


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"app_name": "another-app"}, "different app or session"),
        ({"session_id": "another-source"}, "different app or session"),
        ({"active_agent": "returns"}, "'returns' is not part"),
    ],
    ids=["another-app", "another-source-session", "agent-the-app-no-longer-has"],
)
async def test_branch_from_a_source_this_app_cannot_continue_fails_before_work(
    oe: OE, change: dict[str, str], match: str
) -> None:
    support = ScriptedNativeModel([text_item("unused")])
    app = _team_app(support, ScriptedNativeModel(), [])
    owner = {"app_name": "support", "session_id": "s-1", "active_agent": "support"}
    owner.update(change)
    oe.branch(
        encode_state(
            [],
            app_name=owner["app_name"],
            session_id=owner["session_id"],
            active_agent=owner["active_agent"],
        )
    )

    with pytest.raises(DurableOpenAIAgentsStateError, match=match):
        await _attempt(app, oe, RequestContext(session_id="branch-1"), "Hello.")
    assert support.calls == [] and oe.settled == []


def _history(snapshot: StateSnapshot) -> list[tuple[str, list[str], str]]:
    """What a reader of OE history sees: role, tool calls made, call answered."""
    return [
        (
            MessageRole.Name(message.role).removeprefix("MESSAGE_ROLE_").lower(),
            [proto_struct_to_json(call)["name"] for call in message.tool_calls],  # type: ignore[misc]
            message.tool_call_id,
        )
        for message in snapshot.messages
    ]


async def test_history_shows_a_handoff_as_a_call_and_names_the_active_agent(
    oe: OE,
) -> None:
    support = ScriptedNativeModel([_transfer()])
    billing = ScriptedNativeModel([text_item("Invoice INV-1 is paid.")])

    await _attempt(
        _team_app(support, billing, []), oe, RequestContext(session_id="s-1")
    )

    assert _history(oe.settled[0]) == [
        ("user", [], ""),
        ("assistant", ["transfer_to_billing"], ""),
        ("tool", [], "call-1"),
        ("assistant", [], ""),
    ]
    assert _envelope(oe.settled[0])["active_agent"] == "billing"


async def test_history_shows_an_agent_tool_call_but_not_the_nested_conversation(
    oe: OE,
) -> None:
    support = ScriptedNativeModel([_ask("call-1", "A1")], [text_item("Friday.")])

    await _attempt(
        _specialist_app(support, _shipping_model()),
        oe,
        RequestContext(session_id="s-1"),
    )

    # The specialist's own request, tool call, and answer are not history; its
    # answer is the output of the support agent's call.
    assert _history(oe.settled[0]) == [
        ("user", [], ""),
        ("assistant", ["ask_shipping"], ""),
        ("tool", [], "call-1"),
        ("assistant", [], ""),
    ]
    assert oe.settled[0].messages[2].content.string_value == "A1 arrives Friday"


_FIND_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


def _atlas_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    input_schema: dict[str, Any] = _FIND_SCHEMA,
    call: Callable[[dict[str, Any]], Any] = lambda arguments: {"found": 3},
) -> list[tuple[str, str, dict[str, Any]]]:
    """Configure one MCP server in agent.yaml and stand in for it.

    Returns the calls that reached the server, as (server, tool, arguments).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.yaml").write_text(
        "name: support\n"
        "mcp:\n"
        "  servers:\n"
        "    atlas:\n"
        "      url: https://mcp.example.com/mcp\n"
        "      timeout_seconds: 45\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(server_name: str, _config: Any) -> list[Any]:
        assert server_name == "atlas"
        return [
            SimpleNamespace(
                name="find", description="Find documents.", inputSchema=input_schema
            )
        ]

    async def call_tool(binding: Any, arguments: dict[str, Any]) -> Any:
        calls.append((binding.server_name, binding.tool_name, arguments))
        return call(arguments)

    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_tools._list_server_tools", list_tools
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_tools._call_mcp_tool", call_tool
    )
    return calls


async def _started(build: Callable[[], App]) -> App:
    """Build an app as a process does at import, before any event loop runs:
    MCP discovery is synchronous and refuses to start inside one."""
    return await asyncio.to_thread(build)


def _find(call_id: str, query: str) -> Any:
    return function_call_item(call_id, json.dumps({"query": query}), name="atlas__find")


async def test_configured_mcp_tool_runs_as_a_remote_tool_under_its_call_id(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    server_calls = _atlas_mcp_server(monkeypatch, tmp_path)
    native = ScriptedNativeModel(
        [_find("call-1", "late orders")], [text_item("Three.")]
    )
    app = await _started(lambda: _order_app(native))

    events = await _turn(app, "How many late orders?", oe)

    # The model is offered the server's tool beside the app's own, and its
    # call reaches the server with the model's arguments.
    assert native.calls[0]["tools"] == ["atlas__find", "lookup_order"]
    assert server_calls == [("atlas", "find", {"query": "late orders"})]
    assert oe.tool_calls == [("call-1", {"query": "late orders"})]
    # It runs in the Tool Pod, which finds the server from this metadata.
    policy = oe.tool_policies["atlas__find"]
    assert policy["is_local"] is False
    assert policy["metadata"] == {"mcp_server": "atlas", "mcp_tool": "find"}
    (result,) = [event.data for event in events if event.event == "result"]
    assert result["messages"][2]["content"] == '{"found":3}'
    # The platform sees it as a remote tool limited to the server's host.
    (definition,) = [
        tool for tool in app.get_tool_definitions() if tool.name == "atlas__find"
    ]
    assert (definition.remote, definition.network, definition.timeout_seconds) == (
        True,
        ["mcp.example.com"],
        45,
    )
    assert definition.args_schema == _FIND_SCHEMA


async def test_mcp_tool_of_an_agent_used_as_a_tool_runs_under_the_nested_path(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    route_to_tool_pod(monkeypatch, preallocate_tools=True)
    _atlas_mcp_server(monkeypatch, tmp_path)
    researcher = ScriptedNativeModel(
        [_find("find-1", "late orders")], [text_item("Three.")]
    )
    support = ScriptedNativeModel(
        [function_call_item("call-1", '{"input":"late orders"}', name="ask_research")],
        [text_item("Three are late.")],
    )
    app = await _started(lambda: App(app_name="support"))

    @app.entrypoint
    def build() -> Agent[Any]:
        research = Agent(
            name="research",
            model=app.llm(researcher, llm_id="research"),
            tools=app.tools(),
        )
        ask = research.as_tool(tool_name="ask_research", tool_description="Ask.")
        return Agent(name="support", model=app.llm(support), tools=[ask])

    await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert oe.tool_paths == {"find-1": [("agent", 1), ("ask_research", 1)]}


async def test_failed_mcp_tool_is_reported_to_the_model_as_the_sdk_does(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    def unavailable(_arguments: dict[str, Any]) -> Any:
        raise MCPToolError("MCP tool returned error: cluster paused")

    _atlas_mcp_server(monkeypatch, tmp_path, call=unavailable)
    native = ScriptedNativeModel([_find("call-1", "x")], [text_item("Paused.")])

    await _turn(await _started(lambda: _order_app(native)), "Find x.", oe)
    assert native.calls[1]["input"][-1]["output"] == default_tool_error_function(
        cast(Any, None), RuntimeError("MCP tool returned error: cluster paused")
    )
    assert len(oe.settled) == 1


# 2**53 itself is refused: OE reads a larger integer back as that one, and the
# largest arrive as floats.
@pytest.mark.parametrize("account_id", [2**53 + 1, 2**53, -(2**53), 1e21, -1e21])
async def test_mcp_result_oe_would_round_fails_the_turn_without_committing(
    account_id: float, oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(
        monkeypatch,
        tmp_path,
        call=lambda _arguments: {"accounts": [{"account_id": account_id}]},
    )
    native = ScriptedNativeModel([_find("call-1", "x")], [text_item("unused")])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="as a string"):
        await _turn(await _started(lambda: _order_app(native)), "Find x.", oe)
    assert len(native.calls) == 1 and oe.settled == []


async def test_mcp_result_with_exact_integers_and_string_ids_is_kept(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    found = {"account_id": 2**53 - 1, "ref": str(2**53 + 1), "share": 0.25}
    _atlas_mcp_server(monkeypatch, tmp_path, call=lambda _arguments: found)
    native = ScriptedNativeModel([_find("call-1", "x")], [text_item("Found.")])

    events = await _turn(await _started(lambda: _order_app(native)), "Find x.", oe)

    (result,) = [event.data for event in events if event.event == "result"]
    assert json.loads(result["messages"][2]["content"]) == found


async def test_mcp_tool_arguments_that_are_not_an_object_fail_the_turn(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    server_calls = _atlas_mcp_server(monkeypatch, tmp_path)
    native = ScriptedNativeModel(
        [function_call_item("call-1", '["late orders"]', name="atlas__find")],
        [text_item("unused")],
    )

    with pytest.raises(UserError, match="invalid arguments"):
        await _turn(await _started(lambda: _order_app(native)), "Find x.", oe)
    assert (server_calls, oe.settled) == ([], [])


def test_function_tool_named_like_an_mcp_tool_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(monkeypatch, tmp_path)
    app = App(app_name="support")

    def atlas__find(query: str) -> str:
        """Find documents locally."""
        return query

    with pytest.raises(MCPConfigError, match="configured MCP server"):
        app.tool()(atlas__find)


async def test_approval_set_on_an_mcp_tool_fails_the_turn_before_any_model_call(
    oe: OE, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(monkeypatch, tmp_path)
    native = ScriptedNativeModel([text_item("unused")])
    app = await _started(lambda: App(app_name="support"))

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(
            name="support",
            model=app.llm(native),
            tools=_tools_needing_approval(app, "atlas__find"),
        )

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="needs_approval"):
        await _turn(app, "Find x.", oe)
    assert native.calls == []


def test_mcp_tool_with_the_reserved_call_id_parameter_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    schema = {"type": "object", "properties": {"tool_call_id": {"type": "string"}}}
    _atlas_mcp_server(monkeypatch, tmp_path, input_schema=schema)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="tool_call_id"):
        App(app_name="support").tools()


def test_mcp_tools_are_offered_in_one_order_however_the_server_lists_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(monkeypatch, tmp_path)

    def offered(listed: list[str]) -> list[str]:
        async def list_tools(_server_name: str, _config: Any) -> list[Any]:
            return [
                SimpleNamespace(name=name, description="", inputSchema=_FIND_SCHEMA)
                for name in listed
            ]

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._list_server_tools", list_tools
        )
        return [tool.name for tool in App(app_name="support").tools()]

    # A replacement attempt lists the server again, and must offer the model
    # the same tools in the same order for OE to accept the replay.
    assert offered(["find", "count", "aggregate"]) == offered(
        ["aggregate", "find", "count"]
    )


def test_mcp_tool_schema_without_a_type_is_offered_as_an_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(monkeypatch, tmp_path, input_schema={})

    (tool,) = App(app_name="support").tools()

    assert tool.params_json_schema == {"type": "object", "properties": {}}


def test_tool_pod_builds_the_agent_without_discovering_mcp_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _atlas_mcp_server(monkeypatch, tmp_path)
    monkeypatch.setenv("RUNNER_MODE", "tool")

    def unexpected(_config: Any) -> Any:
        raise AssertionError("the Tool Pod resolves MCP calls from agent.yaml")

    monkeypatch.setattr(
        "agent_engine_sdk_openai_agents.app.discover_mcp_tools", unexpected
    )
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return order_id

    @app.entrypoint
    def build() -> Agent[None]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    app.get_agent()

    # The pod still registers the model; it has only the app's own functions.
    assert isinstance(get_named_llm("__default__"), RegisteredModel)
    assert [tool.name for tool in app.tools()] == ["lookup_order"]


_HELD = "Paid in full: TRIPWIRE CLM-1"


def _claims_app(native: ScriptedNativeModel | None = None) -> App:
    """An agent whose answer a guardrail policy holds for review."""
    return _order_app(
        native or ScriptedNativeModel(respond=lambda _items: [text_item(_HELD)])
    )


async def test_a_response_held_for_review_suspends_without_showing_it(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)
    native = ScriptedNativeModel(respond=lambda _items: [text_item(_HELD)])

    first = await _attempt(
        _claims_app(native), oe, RequestContext(session_id="s-1"), "Summarize CLM-1."
    )

    # Nothing of the held response reaches the caller, only the wait.
    assert [e.event for e in first] == ["suspend"]
    (interrupt,) = first[-1].data["interrupts"]
    # OE recognizes the wait as the review's by this value.
    assert interrupt["value"] == {
        "guardrail_review": {
            "review_id": "review-CLM-1",
            "allowed_decisions": ["approve", "deny"],
            "reason": "names a restricted claim",
            "guardrails": [],
        }
    }
    answers = first[-1].data["resume_schema"]["properties"]["resume_map"]
    assert answers["properties"] == {
        interrupt["id"]: approvals_module.REVIEW_ANSWER_SCHEMA
    }
    assert oe.wait_paths == {interrupt["id"]: [("agent", 1)]}
    assert oe.settled == [] and reviews.resolved == []
    # The tool the Runner stops on is never offered to a model.
    assert native.calls[0]["tools"] == ["lookup_order"]


async def test_an_approved_review_releases_the_held_response_once(
    oe: OE, wrapper: FakeWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)
    memory = _Memory(oe)
    monkeypatch.setattr(wrapper, "durable_memory", memory, raising=False)
    monkeypatch.setattr(wrapper, "workflow", object(), raising=False)
    monkeypatch.setattr(approvals_module, "get_current_wrapper", lambda: wrapper)
    native = ScriptedNativeModel(respond=lambda _items: [text_item(_HELD)])
    app = _claims_app(native)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]

    answer = reviews.decide("review-CLM-1", "approve")
    resumed = await _attempt(app, oe, _resume({wait_id: answer}, oe))

    assert resumed[-1].event == "result" and resumed[-1].data["response"] == _HELD
    assert _tokens(resumed) == _HELD
    # OE answered the held call from the review: the model ran for the
    # original call and its replay, never for the release.
    assert reviews.resolved == ["review-CLM-1"] and len(native.calls) == 2
    # A decision is not conversation content, but Memory acknowledges the
    # wait's activity before the turn settles.
    assert memory.acknowledged == [
        {
            "activity_id": wait_id,
            "attempt_id": "attempt-2",
            "fencing_token": 7,
            "execution_id": "exec-1",
            "settled_before": 0,
        }
    ]
    # The stop itself is not conversation either: the committed turn is the
    # question and the released answer.
    oe.previous_state = oe.settled[0]
    items = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [(item.get("role"), item.get("type")) for item in items] == [
        ("user", None),
        ("assistant", "message"),
    ]


async def test_an_approved_response_with_a_tool_call_runs_it_and_continues(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)

    def respond(items: Any) -> list[Any]:
        if items[-1].get("type") == "function_call_output":
            return [text_item("It shipped.")]
        return [text_item(_HELD), function_call_item("call-1")]

    native = ScriptedNativeModel(respond=respond)
    app = _claims_app(native)
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]
    # The held response's tool call does not run before the decision.
    assert oe.tool_calls == []

    answer = reviews.decide("review-CLM-1", "approve")
    resumed = await _attempt(app, oe, _resume({wait_id: answer}, oe))

    assert resumed[-1].data["response"] == "It shipped."
    assert oe.tool_calls == [("call-1", {"order_id": "A1"})]
    # The model ran for the held call, its replay, and the call after the
    # tool; the release came from OE, once.
    assert reviews.resolved == ["review-CLM-1"] and len(native.calls) == 3
    # The request after the release carries the released call and its output,
    # and nothing of the stop.
    sent = native.calls[-1]["input"]
    assert [i.get("role") or i.get("type") for i in sent] == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
    ]
    assert REVIEW_TOOL_NAME not in json.dumps(sent, default=str)
    oe.previous_state = oe.settled[0]
    items = [item.raw for item in decode_state(oe.attempt(), app_name="support")]
    assert [(item.get("role"), item.get("type")) for item in items] == [
        ("user", None),
        ("assistant", "message"),
        (None, "function_call"),
        (None, "function_call_output"),
        ("assistant", "message"),
    ]


async def test_a_denied_review_fails_the_turn_without_committing(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)
    app = _claims_app()
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]

    answer = reviews.decide("review-CLM-1", "deny")
    events: list[Any] = []
    with pytest.raises(PolicyDeniedException, match="denied by a reviewer"):
        with attempt_context_scope(oe.attempt()):
            async for event in app.get_agent().stream(
                _resume({wait_id: answer}, oe), AgentInput(payload={"message": "x"})
            ):
                events.append(event)

    assert oe.settled == [] and _tokens(events) == ""


@pytest.mark.parametrize(
    "answer",
    [
        {"guardrail_review": {"review_id": "review-OTHER", "decision": "approve"}},
        {"guardrail_review": {"review_id": "review-CLM-1", "decision": "maybe"}},
        {"confirmed": True},
    ],
)
async def test_a_review_answer_that_is_not_its_reviews_fails_before_new_work(
    oe: OE, monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    reviews = ReviewingOE(monkeypatch)
    app = _claims_app()
    first = await _attempt(app, oe, RequestContext(session_id="s-1"))
    wait_id = first[-1].data["interrupts"][0]["id"]

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="recorded halt"):
        await _attempt(app, oe, _resume({wait_id: answer}, oe))

    assert reviews.resolved == [] and oe.settled == []


async def test_a_nested_review_and_a_tool_approval_resolve_in_one_resume(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)
    refunds: list[str] = []
    batch = [
        _ask("call-1", "CLM-1", "ask_refunds"),
        function_call_item("call-2", '{"order_id":"A2"}', name="refund"),
    ]
    support = ScriptedNativeModel(batch, list(batch), [text_item("Both done.")])
    specialist = ScriptedNativeModel(respond=lambda _items: [text_item(_HELD)])
    app = _refunds_app(
        support, refunds, specialist=specialist, support_refunds_too=True
    )

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    # One frontier in the support model's call order: the specialist's held
    # response waits beside the support agent's own approval.
    review, approval = first[-1].data["interrupts"]
    assert review["value"]["guardrail_review"]["review_id"] == "review-CLM-1"
    assert approval["value"]["tool_call"]["call_id"] == "call-2"
    assert oe.wait_paths == {
        review["id"]: _NESTED_REFUND,
        approval["id"]: [("agent", 1)],
    }
    assert _HELD not in json.dumps([e.data for e in first], default=str)

    answers = {
        review["id"]: reviews.decide("review-CLM-1", "approve"),
        approval["id"]: {"confirmed": True},
    }
    resumed = await _attempt(app, oe, _resume(answers, oe))

    assert resumed[-1].data["response"] == "Both done." and refunds == ["A2"]
    outputs = {
        item["call_id"]: str(item["output"])
        for item in support.calls[-1]["input"]
        if item.get("type") == "function_call_output"
    }
    # The released response is the specialist's answer to the support agent.
    assert outputs == {"call-1": _HELD, "call-2": "refunded A2"}
    assert specialist.calls[0]["tools"] == ["refund"]


async def test_a_model_that_calls_the_review_tool_fails_the_turn(oe: OE) -> None:
    forged = function_call_item("call-1", '{"review_id":"r-1"}', name=REVIEW_TOOL_NAME)
    app = _claims_app(ScriptedNativeModel([forged]))

    # Only a halt OE recorded may stop the run for a review.
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="reserves"):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert oe.settled == [] and oe.waits == {}


async def test_a_tool_with_the_review_tools_name_fails_before_any_model_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.tool()
    def agent_engine_guardrail_review(review_id: str) -> str:
        """Pretend to be the platform's review tool."""
        return review_id

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(name="support", model=app.llm(native), tools=app.tools())

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=REVIEW_TOOL_NAME):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert native.calls == []


async def test_a_handoff_with_the_review_tools_name_fails_before_any_model_call(
    oe: OE,
) -> None:
    support = ScriptedNativeModel([text_item("unused")])
    billing = ScriptedNativeModel([text_item("unused")])
    app = _team_app(
        support,
        billing,
        [],
        lambda agent: handoff(agent, tool_name_override=REVIEW_TOOL_NAME),
    )

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=REVIEW_TOOL_NAME):
        await _attempt(app, oe, RequestContext(session_id="s-1"))

    assert support.calls == [] and billing.calls == []


def _verdict(tripped: bool, info: Any = None) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=info, tripwire_triggered=tripped)


def _guarded_app(native: ScriptedNativeModel, **guardrails: Any) -> App:
    """The refund app, with guardrails on its one agent."""
    app = App(app_name="support")

    @app.tool(needs_approval=True)
    def refund(order_id: str) -> str:
        """Refund an order; a person approves every refund."""
        return f"refunded {order_id}"

    @app.entrypoint
    def build() -> Agent[Any]:
        return Agent(
            name="support", model=app.llm(native), tools=app.tools(), **guardrails
        )

    return app


async def test_tripped_input_guardrail_denies_the_turn_before_any_model_call(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    seen: list[Any] = []

    def on_topic(_context: Any, _agent: Any, input: Any) -> GuardrailFunctionOutput:
        seen.append(input)
        return _verdict(True, info="the customer's words")

    # Declared the SDK's default way, which would start it beside the model.
    app = _guarded_app(native, input_guardrails=[InputGuardrail(on_topic)])

    with pytest.raises(PolicyDeniedException) as denied:
        await _attempt(app, oe, RequestContext(session_id="s-1"), "Write my essay.")

    assert denied.value.reason == "the agent's input guardrail 'on_topic' tripped"
    assert seen == [[{"role": "user", "content": "Write my essay."}]]
    assert native.calls == [] and oe.settled == []


async def test_input_guardrails_finish_in_listed_order_before_the_model_call(
    oe: OE,
) -> None:
    order: list[str] = []

    def respond(_call: Any) -> list[Any]:
        order.append("model")
        return [text_item("Hello.")]

    async def slow(*_args: Any) -> GuardrailFunctionOutput:
        order.append("slow starts")
        await asyncio.sleep(0.01)
        order.append("slow ends")
        return _verdict(False)

    def quick(*_args: Any) -> GuardrailFunctionOutput:
        order.append("quick")
        return _verdict(False)

    app = _guarded_app(
        ScriptedNativeModel(respond=respond),
        input_guardrails=[
            InputGuardrail(slow),
            InputGuardrail(quick, run_in_parallel=False),
        ],
    )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")

    assert order == ["slow starts", "slow ends", "quick", "model"]
    # Input guardrails hold nothing back once they pass.
    assert _tokens(events) == "Hello." and events[-1].data["response"] == "Hello."


async def test_guardrails_after_the_one_that_trips_do_not_run(oe: OE) -> None:
    ran: list[str] = []

    def first(*_args: Any) -> GuardrailFunctionOutput:
        ran.append("first")
        return _verdict(True)

    def second(*_args: Any) -> GuardrailFunctionOutput:
        ran.append("second")
        return _verdict(True)

    app = _guarded_app(
        ScriptedNativeModel([text_item("unused")]),
        input_guardrails=[InputGuardrail(first), InputGuardrail(second)],
    )

    with pytest.raises(PolicyDeniedException, match="'first'"):
        await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")
    assert ran == ["first"]


async def test_tripped_output_guardrail_denies_the_turn_and_shows_none_of_the_answer(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("Internal note: margin is 40%.")])
    checked: list[Any] = []

    def no_internal_notes(
        _context: Any, _agent: Any, output: Any
    ) -> GuardrailFunctionOutput:
        checked.append(output)
        return _verdict("Internal note" in output)

    app = _guarded_app(
        native, output_guardrails=[OutputGuardrail(no_internal_notes, name="notes")]
    )
    events: list[Any] = []

    with pytest.raises(PolicyDeniedException) as denied:
        with attempt_context_scope(oe.attempt()):
            async for event in app.get_agent().stream(
                RequestContext(session_id="s-1"), _HI
            ):
                events.append(event)

    assert denied.value.reason == "the agent's output guardrail 'notes' tripped"
    assert checked == ["Internal note: margin is 40%."]
    # The SDK streams an answer before its output guardrails see it.
    assert events == [] and oe.settled == []


async def test_answer_of_an_agent_with_output_guardrails_arrives_whole(oe: OE) -> None:
    def passes(*_args: Any) -> GuardrailFunctionOutput:
        return _verdict(False)

    app = _guarded_app(
        ScriptedNativeModel([text_item("Shipped.")]),
        output_guardrails=[OutputGuardrail(passes)],
    )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")

    assert [event.event for event in events] == ["result"]
    assert events[-1].data["response"] == "Shipped." and len(oe.settled) == 1


async def test_guardrails_run_once_in_an_attempt_that_continues_past_an_approval(
    oe: OE,
) -> None:
    ran: list[str] = []

    def on_topic(*_args: Any) -> GuardrailFunctionOutput:
        ran.append("input")
        return _verdict(False)

    def polite(_context: Any, _agent: Any, output: Any) -> GuardrailFunctionOutput:
        ran.append(f"output {output}")
        return _verdict(False)

    app = _guarded_app(
        _refund_turns(),
        input_guardrails=[InputGuardrail(on_topic)],
        output_guardrails=[OutputGuardrail(polite)],
    )

    first = await _attempt(app, oe, RequestContext(session_id="s-1"))

    (interrupt,) = first[-1].data["interrupts"]
    # The model's words beside the refund call are not its final output.
    assert ran == ["input"]
    resumed = await _attempt(
        app, oe, _resume({interrupt["id"]: {"confirmed": True}}, oe)
    )

    # The replacement attempt replays the turn, input guardrail included; the
    # Runner's continuation after the approval does not run it a third time.
    assert ran == ["input", "input", "output Done."]
    assert resumed[-1].data["response"] == "Done." and len(oe.settled) == 1
    assert _tokens(first) == _tokens(resumed) == ""


async def test_guardrail_asks_a_judge_model_before_the_agents_own_model(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxies = route_to_tool_pod(monkeypatch)
    judge_model = ScriptedNativeModel(respond=lambda call: [text_item("off topic")])
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        judge = Agent(
            name="judge",
            instructions="Is this about an order?",
            model=app.llm(judge_model, llm_id="judge"),
        )

        async def on_topic(
            context: Any, _agent: Any, input: Any
        ) -> GuardrailFunctionOutput:
            # No run config: the developer's run is the SDK's documented one.
            verdict = await Runner.run(judge, input, context=context.context)
            return _verdict(verdict.final_output == "off topic")

        return Agent(
            name="support",
            model=app.llm(native),
            input_guardrails=[InputGuardrail(on_topic)],
        )

    events: list[Any] = []
    with pytest.raises(PolicyDeniedException, match="'on_topic'"):
        with attempt_context_scope(oe.attempt()):
            async for event in app.get_agent().stream(
                RequestContext(session_id="s-1"), _HI
            ):
                events.append(event)

    # The judge's call is an OE activity of the judge's registered model; the
    # agent's own model was never asked, and the judge's words reach no one.
    asked = [
        (p.kwargs["llm_id"], [c["step"] for c in p.calls]) for p in proxies.instances
    ]
    assert asked == [("judge", [1])]
    assert native.calls == [] and events == [] and oe.settled == []


async def test_a_judge_response_held_for_review_fails_the_turn(
    oe: OE, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews = ReviewingOE(monkeypatch)
    native = ScriptedNativeModel([text_item("unused")])
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        judge = Agent(
            name="judge",
            model=app.llm(
                ScriptedNativeModel(respond=lambda _items: [text_item(_HELD)]),
                llm_id="judge",
            ),
        )

        async def on_topic(
            context: Any, _agent: Any, input: Any
        ) -> GuardrailFunctionOutput:
            await Runner.run(judge, input, context=context.context)
            return _verdict(False)

        return Agent(
            name="support",
            model=app.llm(native),
            input_guardrails=[InputGuardrail(on_topic)],
        )

    events: list[Any] = []
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="a guardrail runs"):
        with attempt_context_scope(oe.attempt()):
            async for event in app.get_agent().stream(
                RequestContext(session_id="s-1"), _HI
            ):
                events.append(event)

    # No wait is recorded for a review the judge's run could never stop on.
    assert events == [] and oe.settled == [] and oe.wait_paths == {}
    assert native.calls == [] and reviews.resolved == []


async def test_handoff_target_runs_its_input_guardrails_when_it_starts_a_turn(
    oe: OE,
) -> None:
    ran: list[str] = []

    def billing_only(*_args: Any) -> GuardrailFunctionOutput:
        ran.append("billing")
        return _verdict(False)

    def guarded(billing: Agent[Any]) -> Agent[Any]:
        billing.input_guardrails = [InputGuardrail(billing_only)]
        return billing

    app = _team_app(
        ScriptedNativeModel([_transfer()]),
        ScriptedNativeModel([text_item("Billing here.")], [text_item("Still here.")]),
        [],
        to_billing=guarded,
    )

    await _turn(app, "I have a billing question.", oe)

    # As in the SDK, only the agent that starts a run checks the run's input.
    assert ran == []
    oe.next_turn()
    await _turn(app, "And another.", oe)
    assert ran == ["billing"]


def _trips(tripped: bool, *_args: Any) -> GuardrailFunctionOutput:
    return _verdict(tripped)


@pytest.mark.parametrize(
    ("guardrails", "match"),
    [
        (
            {"input_guardrails": [InputGuardrail(functools.partial(_trips, True))]},
            "an input guardrail of agent 'support' has no name; pass name= to "
            "InputGuardrail",
        ),
        (
            {"output_guardrails": [OutputGuardrail(functools.partial(_trips, True))]},
            "an output guardrail of agent 'support' has no name; pass name= to "
            "OutputGuardrail",
        ),
    ],
    ids=["input", "output"],
)
async def test_guardrail_without_a_name_is_refused_before_any_model_call(
    oe: OE, guardrails: dict[str, Any], match: str
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    app = _guarded_app(native, **guardrails)

    # Unnamed, its trip could not be reported: the SDK names a guardrail
    # after its function, and a partial has no name.
    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")

    assert native.calls == [] and oe.settled == []


async def test_guardrail_built_from_a_partial_trips_under_the_name_it_is_given(
    oe: OE,
) -> None:
    native = ScriptedNativeModel([text_item("unused")])
    app = _guarded_app(
        native,
        input_guardrails=[
            InputGuardrail(functools.partial(_trips, True), name="on_topic")
        ],
    )

    with pytest.raises(PolicyDeniedException) as denied:
        await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")

    assert denied.value.reason == "the agent's input guardrail 'on_topic' tripped"
    assert native.calls == [] and oe.settled == []


def _team_with_guarded_billing(billing: ScriptedNativeModel, checked: list[Any]) -> App:
    """Support hands to billing, whose output guardrail checks its answer."""

    def no_internal_notes(
        _context: Any, agent: Any, output: Any
    ) -> GuardrailFunctionOutput:
        checked.append((agent.name, output))
        return _verdict("Internal note" in output)

    def to_billing(billing_agent: Agent[Any]) -> Agent[Any]:
        billing_agent.output_guardrails = [
            OutputGuardrail(no_internal_notes, name="notes")
        ]
        return billing_agent

    return _team_app(ScriptedNativeModel([_transfer()]), billing, [], to_billing)


async def test_answer_of_a_guarded_handoff_target_is_checked_and_not_streamed(
    oe: OE,
) -> None:
    checked: list[Any] = []
    app = _team_with_guarded_billing(
        ScriptedNativeModel([text_item("Invoice INV-1 is paid.")]), checked
    )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "Is it paid?")

    # The agent that ends the turn checks the answer, and its text reaches
    # the caller whole in the result, not as tokens.
    assert checked == [("billing", "Invoice INV-1 is paid.")]
    assert _tokens(events) == ""
    assert events[-1].data["response"] == "Invoice INV-1 is paid."
    assert len(oe.settled) == 1


async def test_tripped_output_guardrail_of_a_handoff_target_shows_none_of_its_answer(
    oe: OE,
) -> None:
    checked: list[Any] = []
    app = _team_with_guarded_billing(
        ScriptedNativeModel([text_item("Internal note: margin is 40%.")]), checked
    )
    events: list[Any] = []

    with pytest.raises(PolicyDeniedException) as denied:
        with attempt_context_scope(oe.attempt()):
            async for event in app.get_agent().stream(
                RequestContext(session_id="s-1"), _HI
            ):
                events.append(event)

    assert denied.value.reason == "the agent's output guardrail 'notes' tripped"
    assert checked == [("billing", "Internal note: margin is 40%.")]
    assert _tokens(events) == "" and oe.settled == []


async def test_unguarded_handoff_target_streams_after_a_guarded_agent_hands_off(
    oe: OE,
) -> None:
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        billing = Agent(
            name="billing",
            model=app.llm(
                ScriptedNativeModel([text_item("Invoice INV-1 is paid.")]),
                llm_id="billing",
            ),
        )
        return Agent(
            name="support",
            model=app.llm(ScriptedNativeModel([_transfer()])),
            handoffs=[billing],
            output_guardrails=[OutputGuardrail(lambda *_: _verdict(False), name="x")],
        )

    events = await _attempt(app, oe, RequestContext(session_id="s-1"), "Is it paid?")

    # Only the agent speaking decides: billing has no output guardrail.
    assert _tokens(events) == "Invoice INV-1 is paid."
    assert events[-1].data["response"] == "Invoice INV-1 is paid."


@pytest.mark.parametrize(
    "guardrails",
    [
        {"input_guardrails": [InputGuardrail(lambda *_: _verdict(False))]},
        {"output_guardrails": [OutputGuardrail(lambda *_: _verdict(False))]},
    ],
)
async def test_guardrails_on_an_agent_used_as_a_tool_are_refused(
    oe: OE, guardrails: dict[str, Any]
) -> None:
    support = ScriptedNativeModel([text_item("unused")])

    def guarded(shipping: Agent[Any]) -> Any:
        for name, value in guardrails.items():
            setattr(shipping, name, value)
        return shipping.as_tool(tool_name="ask_shipping", tool_description="Ask.")

    app = _specialist_app(support, _shipping_model(), as_tool=guarded)

    with pytest.raises(
        UnsupportedDurableOpenAIAgentsError,
        match="agent 'shipping' is used as a tool and has guardrails",
    ):
        await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")
    assert support.calls == []


@pytest.mark.parametrize("first_trips", [False, True])
async def test_output_guardrails_run_one_at_a_time_in_listed_order(
    oe: OE, first_trips: bool
) -> None:
    order: list[str] = []

    async def yielding(*_args: Any) -> GuardrailFunctionOutput:
        order.append("yielding starts")
        # Lets anything started beside this guardrail run first.
        await asyncio.sleep(0)
        order.append("yielding ends")
        return _verdict(first_trips)

    def quick(*_args: Any) -> GuardrailFunctionOutput:
        order.append("quick")
        return _verdict(False)

    app = _guarded_app(
        ScriptedNativeModel([text_item("Shipped.")]),
        output_guardrails=[OutputGuardrail(yielding), OutputGuardrail(quick)],
    )

    if first_trips:
        with pytest.raises(PolicyDeniedException, match="'yielding'"):
            await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")
        assert order == ["yielding starts", "yielding ends"] and oe.settled == []
    else:
        events = await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")
        assert order == ["yielding starts", "yielding ends", "quick"]
        assert events[-1].data["response"] == "Shipped."


async def test_guardrails_on_an_agent_handed_to_inside_an_agent_tool_are_refused(
    oe: OE,
) -> None:
    models = [ScriptedNativeModel([text_item("unused")]) for _ in range(3)]
    support, shipping, returns = models
    app = App(app_name="support")

    @app.entrypoint
    def build() -> Agent[Any]:
        returns_agent = Agent(
            name="returns",
            model=app.llm(returns, llm_id="returns"),
            output_guardrails=[OutputGuardrail(lambda *_: _verdict(False))],
        )
        # The agent used as a tool has no guardrail of its own.
        specialist = Agent(
            name="shipping",
            model=app.llm(shipping, llm_id="shipping"),
            handoffs=[returns_agent],
        )
        ask = specialist.as_tool(tool_name="ask_shipping", tool_description="Ask.")
        return Agent(name="support", model=app.llm(support), tools=[ask])

    with pytest.raises(
        UnsupportedDurableOpenAIAgentsError,
        match="agent 'returns' is used as a tool and has guardrails",
    ):
        await _attempt(app, oe, RequestContext(session_id="s-1"), "hi")
    assert [model.calls for model in models] == [[], [], []]
