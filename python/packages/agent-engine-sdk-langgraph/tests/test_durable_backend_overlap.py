"""Tool Pod backend operations that overlap in a durable Deep Agent turn.

These run the real Deep Agent graph with the path resolver a deployed one gets,
the real ``SecureToolWrapper`` and its attempt gate. Only the model, the
Orchestration Engine, the Tool Pod and the checkpoint store are fake.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.protobuf.json_format import MessageToJson
from agent_engine_sdk.models import LLMStreamChunk, ToolCallChunk
from agent_engine_runner_shared.context import current_wrapper
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_FAILED,
    ActivityCommand,
    ActivityContext,
    ActivityOutcome,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
    WorkflowError,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent_engine_sdk_langgraph.agent import _default_durable_subgraphs
from agent_engine_sdk_langgraph.backends.toolpod import AgentEngineToolPodBackend
from agent_engine_sdk_langgraph.deep_agent import create_agent_engine_deep_agent
from agent_engine_sdk_langgraph.durable_message_identity import (
    install_durable_message_identity,
)
from agent_engine_sdk_langgraph.durable_subgraphs import SendTaskRecorder
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer
import agent_engine_sdk_langgraph.secure_llm as secure_llm_module
from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

ABANDONED_MESSAGE = (
    "The outcome of this tool call is unknown: the runtime was lost while it was "
    "running, so it may or may not have taken effect. It was not run again."
)
HANDLER_RESULTS: dict[str, dict[str, Any]] = {
    "filesystem_ls": {"entries": [{"path": "/a.txt", "is_dir": False}]},
    "filesystem_read": {"content": "file body"},
    "filesystem_write": {"path": "/large_tool_results/call_big"},
    "filesystem_download": {"content_base64": ""},
    "shell_execute": {"output": "ran", "exit_code": 0},
}
LS = ToolCallChunk(id="call_ls", name="ls", args='{"path": "/"}', index=0)
READ = ToolCallChunk(
    id="call_read", name="read_file", args='{"file_path": "/a.txt"}', index=1
)
BACKEND_OPERATIONS = ("filesystem_", "shell_")


class _Oe:
    """Records activity outcomes by position and replays them, as OE does.

    It settles an activity a lost runner left started as failed instead of
    running it again, and refuses a position whose activity changed, so a wrong
    position cannot replay another tool's result without failing the test.
    """

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.refusals: list[str] = []
        # Each attempt's semantic input at a position, oldest first.
        self.semantic_inputs: dict[str, list[str]] = {}
        self._activities: dict[str, str] = {}
        self._outcomes: dict[str, ActivityOutcome] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _label(command: ActivityCommand) -> str:
        path = "/".join(
            f"{segment.name}/{segment.ordinal}"
            for segment in command.position.operation_path.segments
        )
        return f"{path}#{command.position.activity_ordinal}"

    def start_activity(
        self, command: ActivityCommand
    ) -> ActivityDispatch | ActivityReplay:
        with self._lock:
            label = self._label(command)
            previous = self.names.get(label)
            if previous not in (None, command.activity_name):
                # The backend turns this into a tool error, so tests read it here.
                self.refusals.append(label)
                raise AssertionError(
                    f"nondeterministic: {label} {previous} -> {command.activity_name}"
                )
            self.names[label] = command.activity_name
            self.semantic_inputs.setdefault(label, []).append(
                MessageToJson(command.semantic_input)
            )
            if label in self._outcomes:
                return ActivityReplay(self._outcomes[label])
            if previous is not None:
                return ActivityReplay(
                    ActivityOutcome(
                        activity_id="abandoned",
                        outcome_kind=ACTIVITY_OUTCOME_KIND_FAILED,
                        error=WorkflowError(
                            code=WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
                            message=ABANDONED_MESSAGE,
                        ),
                    )
                )
            activity_id = f"activity-{len(self._activities) + 1}"
            self._activities[activity_id] = label
            return ActivityDispatch(
                ActivityContext(
                    workflow_identity=command.workflow_identity,
                    activity_id=activity_id,
                    attempt_id=command.attempt_id,
                    fencing_token=command.fencing_token,
                )
            )

    def report_outcome(self, outcome: ActivityOutcome) -> None:
        with self._lock:
            self._outcomes[self._activities[outcome.activity_id]] = outcome

    def lose_runner_during(self, name: str) -> None:
        """Rewind to a runner lost with `name` in flight: started, no outcome."""
        labels = list(self.names)
        in_flight = next(
            i for i, label in enumerate(labels) if self.names[label] == name
        )
        self._outcomes.pop(labels[in_flight], None)
        for label in labels[in_flight + 1 :]:
            del self.names[label]
            self._outcomes.pop(label, None)

    def tool_positions(self) -> dict[str, str]:
        """Each backend operation's position."""
        return {
            label: name
            for label, name in self.names.items()
            if name.startswith(BACKEND_OPERATIONS)
        }


def _attempt(attempt_id: str = "attempt-1", *, replay: bool = False) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=2 if replay else 1,
        replay_mode=replay,
        workflow_identity=WorkflowIdentity(
            session_id="session", execution_id="execution"
        ),
    )


class _Harness:
    """A durable Deep Agent whose model makes `calls` once, then finishes."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        calls: Sequence[ToolCallChunk],
        *,
        held_until_all_arrive: Sequence[str] = (),
        tools: Sequence[Any] = (),
        agent_options: dict[str, Any] | None = None,
    ) -> None:
        self.oe = _Oe()
        self.dispatched: list[str] = []
        self.tool_messages: list[str] = []
        self.barrier = (
            threading.Barrier(len(held_until_all_arrive), timeout=10)
            if held_until_all_arrive
            else None
        )
        self.held = list(held_until_all_arrive)
        harness = self

        class _Proxy(SecureLLMProxy):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._workflow = harness.oe

            def _stream_llm(
                self, invoke_request: Any, resolved_step: int
            ) -> Iterator[LLMStreamChunk]:
                messages = invoke_request.messages
                if messages[-1].role == "tool":
                    harness.tool_messages[:] = [
                        str(message.content)
                        for message in messages
                        if message.role == "tool"
                    ]
                    yield LLMStreamChunk(content="finished")
                    return
                yield LLMStreamChunk(tool_calls=list(calls))

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy", _Proxy
        )
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.secure_llm.Metrics.record_latency", MagicMock()
        )
        self.wrapper = SecureToolWrapper("http://oe:8000", "execution")
        self.wrapper._workflow = self.oe
        monkeypatch.setattr(self.wrapper, "_execute_tool_native", self._native)
        current_wrapper.set(self.wrapper)
        self.llm = SecureWrappedLLM(
            llm=MagicMock(),
            get_wrapper=lambda: self.wrapper,
            llm_id="model",
            model_name="test-model",
        )

        def build(checkpointer: Any) -> Any:
            return create_agent_engine_deep_agent(
                self.llm,
                AgentEngineToolPodBackend(),
                tools=list(tools),
                checkpointer=checkpointer,
                **(agent_options or {}),
            )

        # The path resolver a deployed Deep Agent gets decides every position
        # here. Only the checkpoint store differs: the platform's needs a live OE.
        self.resolver = _default_durable_subgraphs(
            build(PlatformCheckpointer(native=InMemorySaver()))
        )
        self.agent = build(InMemorySaver())
        install_durable_message_identity(self.agent)

    def _native(self, *, tool_name: str, **_: Any) -> dict[str, Any]:
        self.dispatched.append(tool_name)
        # Answer only once every held call is in flight, so an operation that
        # does not truly overlap its sibling breaks the barrier.
        if self.barrier is not None and tool_name in self.held:
            self.barrier.wait()
        return HANDLER_RESULTS[tool_name]

    def run(self, attempt_id: str = "attempt-1", *, replay: bool = False) -> Any:
        attempt = _attempt(attempt_id, replay=replay)

        async def invoke() -> Any:
            with (
                attempt_context_scope(attempt),
                self.resolver.scope(SendTaskRecorder())
                if self.resolver is not None
                else nullcontext(),
            ):
                # A replacement attempt runs the whole graph again from its input.
                return await self.agent.ainvoke(
                    {"messages": [HumanMessage(content="start")]},
                    {"configurable": {"thread_id": attempt_id}},
                )

        return asyncio.run(invoke())

    def admit_in_order(
        self, monkeypatch: pytest.MonkeyPatch, first: str, then: str
    ) -> None:
        """Hold operation `then` until operation `first` was admitted and finished."""
        finished = threading.Event()
        admit = self.wrapper.__class__.execute_tool

        def ordered(tool_name: str, *args: Any, **kwargs: Any) -> Any:
            if tool_name == then:
                assert finished.wait(10)
            try:
                return admit(self.wrapper, tool_name, *args, **kwargs)
            finally:
                if tool_name == first:
                    finished.set()

        monkeypatch.setattr(self.wrapper, "execute_tool", ordered)


def test_parallel_file_tools_overlap_at_positions_of_their_own_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(
        monkeypatch,
        [LS, READ],
        held_until_all_arrive=["filesystem_ls", "filesystem_read"],
    )

    harness.run()

    assert sorted(harness.dispatched) == ["filesystem_ls", "filesystem_read"]
    assert harness.oe.tool_positions() == {
        "agent/1/tool:effects/2#1": "filesystem_ls",
        "agent/1/tool:effects/3#1": "filesystem_read",
    }
    # Each call's own result reaches the model, not one result and a placeholder.
    assert len(harness.tool_messages) == 2
    assert any("/a.txt" in message for message in harness.tool_messages)
    assert any("file body" in message for message in harness.tool_messages)


def test_a_replacement_attempt_sends_the_model_the_same_tool_message_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, [LS, READ])

    harness.run()
    harness.run("attempt-2", replay=True)

    def message_ids(semantic_input: str) -> list[str]:
        return re.findall(r'"id": "(durable-message:[^"]+)"', semantic_input)

    # The LLM call after the tools carries the messages they built.
    first, replay = next(
        inputs
        for inputs in harness.oe.semantic_inputs.values()
        if '"tool"' in inputs[0]
    )
    assert len(set(message_ids(first))) >= 2
    assert message_ids(replay) == message_ids(first)


def test_parallel_shell_commands_overlap_at_positions_of_their_own_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(
        monkeypatch,
        [
            ToolCallChunk(
                id=f"call_{index}",
                name="execute",
                args=f'{{"command": "step {index}"}}',
                index=index,
            )
            for index in range(2)
        ],
        held_until_all_arrive=["shell_execute", "shell_execute"],
    )

    harness.run()

    assert harness.dispatched == ["shell_execute", "shell_execute"]
    assert harness.oe.tool_positions() == {
        "agent/1/tool:effects/2#1": "shell_execute",
        "agent/1/tool:effects/3#1": "shell_execute",
    }
    assert [message.startswith("ran") for message in harness.tool_messages] == [
        True,
        True,
    ]


def test_a_lone_calls_operation_keeps_its_position_so_recorded_turns_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, [LS])

    harness.run()

    # The model call took #1 and reserved #2 for its only tool call.
    assert harness.oe.tool_positions() == {"agent/1#3": "filesystem_ls"}


def test_a_replacement_attempt_replays_operations_it_reaches_in_the_other_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, [LS, READ])
    harness.admit_in_order(monkeypatch, "filesystem_ls", "filesystem_read")
    harness.run()
    first_dispatch = list(harness.dispatched)
    positions = harness.oe.tool_positions()

    harness.admit_in_order(monkeypatch, "filesystem_read", "filesystem_ls")
    harness.run("attempt-2", replay=True)

    assert harness.oe.refusals == []
    assert harness.dispatched == first_dispatch
    assert harness.oe.tool_positions() == positions


def test_a_tools_own_model_call_leaves_its_siblings_position_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A custom tool may call the model itself, as with_structured_output does.
    # Whichever finishes first, that call or its sibling's read, the read keeps
    # the position the turn's own model call gave it.
    model_answered = threading.Event()
    read_finished = threading.Event()
    read_first = False

    @tool
    def verdict(ok: bool) -> str:
        """Record a verdict."""
        return str(ok)

    @tool
    def classify() -> str:
        """Classify the request with a model call of the tool's own."""
        if read_first:
            assert read_finished.wait(10)
        harness.llm.bind_tools([verdict]).invoke("classify this")
        model_answered.set()
        return "classified"

    classify_call = ToolCallChunk(
        id="call_classify", name="classify", args="{}", index=0
    )
    harness = _Harness(monkeypatch, [classify_call, READ], tools=[classify])
    proxy = secure_llm_module.SecureLLMProxy
    turn_model = proxy._stream_llm

    def model(self: Any, invoke_request: Any, resolved_step: int) -> Any:
        if invoke_request.messages[-1].content == "classify this":
            yield LLMStreamChunk(
                tool_calls=[
                    ToolCallChunk(
                        id="call_verdict", name="verdict", args='{"ok": true}', index=0
                    )
                ]
            )
            return
        yield from turn_model(self, invoke_request, resolved_step)

    monkeypatch.setattr(proxy, "_stream_llm", model)
    admit = harness.wrapper.__class__.execute_tool

    def ordered(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name != "filesystem_read":
            return admit(harness.wrapper, tool_name, *args, **kwargs)
        if not read_first:
            assert model_answered.wait(10)
        try:
            return admit(harness.wrapper, tool_name, *args, **kwargs)
        finally:
            read_finished.set()

    monkeypatch.setattr(harness.wrapper, "execute_tool", ordered)
    harness.run()
    positions = harness.oe.tool_positions()

    read_first = True
    model_answered.clear()
    harness.run("attempt-2", replay=True)

    assert positions == {"agent/1/tool:effects/3#1": "filesystem_read"}
    assert harness.oe.refusals == []
    assert harness.dispatched == ["filesystem_read"]
    assert harness.oe.tool_positions() == positions


def test_the_model_is_told_not_to_retry_an_operation_a_lost_runner_left_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, [LS, READ])
    harness.run()
    harness.oe.lose_runner_during("filesystem_read")
    harness.dispatched.clear()

    harness.run("attempt-2", replay=True)

    assert "filesystem_read" not in harness.dispatched
    assert f"[NON-RETRYABLE] {ABANDONED_MESSAGE}" in "\n".join(harness.tool_messages)


def test_an_operation_inside_a_recorded_tools_body_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_results: list[Any] = []
    harness: _Harness

    @tool
    def audited() -> str:
        """Records a tool whose body performs a backend operation."""

        def body() -> str:
            backend_results.append(AgentEngineToolPodBackend().ls("/"))
            return "audited"

        return harness.wrapper.execute_tool(
            "audited",
            {},
            tool_call_id="call_audited",
            is_local=True,
            local_executor=body,
        )

    harness = _Harness(
        monkeypatch,
        [ToolCallChunk(id="call_audited", name="audited", args="{}", index=0)],
        tools=[audited],
    )
    monkeypatch.setattr(
        harness.wrapper,
        "_execute_tool_native",
        lambda *, local_executor=None, tool_name, **_: (
            local_executor() if local_executor else HANDLER_RESULTS[tool_name]
        ),
    )

    harness.run()

    # A replayed recorded tool would never observe a position recorded here.
    assert harness.oe.tool_positions() == {}
    assert [result.error is not None for result in backend_results] == [True]


def test_operations_outside_any_tool_call_still_refuse_to_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nodes a static edge fans out to share one path and serve no tool call, so
    # only arrival order would number their operations.
    harness = _Harness(monkeypatch, [])
    backend = AgentEngineToolPodBackend()
    results: list[Any] = []

    with attempt_context_scope(_attempt()):
        sibling = copy_context()

        def native(*, tool_name: str, **_: Any) -> dict[str, Any]:
            harness.dispatched.append(tool_name)
            results.append(sibling.run(backend.read, "/a.txt"))
            return HANDLER_RESULTS[tool_name]

        monkeypatch.setattr(harness.wrapper, "_execute_tool_native", native)
        backend.ls("/")

    assert harness.dispatched == ["filesystem_ls"]
    assert [result.error is not None for result in results] == [True]


def test_a_large_tool_result_is_evicted_while_a_siblings_operation_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @tool
    def big_report() -> str:
        """Returns a result large enough for the filesystem middleware to evict."""
        return "x" * 200_000

    harness = _Harness(
        monkeypatch,
        [ToolCallChunk(id="call_big", name="big_report", args="{}", index=0), READ],
        held_until_all_arrive=["filesystem_write", "filesystem_read"],
        tools=[big_report],
    )

    harness.run()

    assert harness.oe.tool_positions() == {
        "agent/1/tool:effects/2#1": "filesystem_write",
        "agent/1/tool:effects/3#1": "filesystem_read",
    }


def test_parallel_subagents_load_their_skills_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_downloads = threading.Barrier(2, timeout=10)
    harness = _Harness(
        monkeypatch,
        [
            ToolCallChunk(
                id=f"call_{name}",
                name="task",
                args=f'{{"subagent_type": "researcher", "description": "{name}"}}',
                index=index,
            )
            for index, name in enumerate(["first", "second"])
        ],
        agent_options={
            "subagents": [
                {
                    "name": "researcher",
                    "description": "Researches",
                    "system_prompt": "Research.",
                    "skills": ["/skills"],
                }
            ],
        },
    )
    original = harness._native

    def native(*, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        if tool_name == "filesystem_ls" and harness.dispatched.count(tool_name) < 2:
            harness.dispatched.append(tool_name)
            skill_downloads.wait()
            return HANDLER_RESULTS[tool_name]
        return original(tool_name=tool_name, **kwargs)

    monkeypatch.setattr(harness.wrapper, "_execute_tool_native", native)

    harness.run()

    assert sorted(
        label
        for label, name in harness.oe.tool_positions().items()
        if name == "filesystem_ls"
    ) == [
        "agent/1/researcher/1#1",
        "agent/1/researcher/2#1",
    ]
