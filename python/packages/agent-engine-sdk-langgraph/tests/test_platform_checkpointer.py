"""Tests for the request-scoped platform checkpointer."""

from __future__ import annotations

import asyncio
import json
import operator
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Interrupt, Send
from agent_engine_sdk import RequestContext
from typing_extensions import TypedDict

from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent
from agent_engine_sdk_langgraph.execution_session import ExecutionSession
from agent_engine_sdk_langgraph.platform_checkpointer import (
    OE_STEP_ORDINAL_METADATA_KEY,
    PlatformCheckpointer,
    UnsupportedDurableGraphError,
    scratch_thread_id,
)
from agent_engine_runner_shared.context import current_oe_url
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    StateSnapshot,
    WorkflowMessage,
)
from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    attempt_context_scope,
    current_attempt_context,
    operation_path_resolver_scope,
)
from agent_engine_runner_shared.workflow.activity import build_activity_command
from agent_engine_runner_shared.workflow.context import (
    current_step_ordinal,
    interrupted_activities,
    observed_activity_positions,
    record_interrupted_activity,
)
from agent_engine_runner_shared.workflow.memory import DurableMemoryState


def _attempt(
    attempt_id: str = "attempt-1",
    fencing_token: int = 1,
    execution_id: str = "execution-1",
    branch_cutoff: int | None = None,
) -> AttemptContext:
    context = AttemptContext(
        attempt_id=attempt_id,
        fencing_token=fencing_token,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id=execution_id,
        ),
    )
    if branch_cutoff is not None:
        context.branch_lineage.CopyFrom(
            BranchLineage(
                source_workflow_identity=WorkflowIdentity(
                    session_id="source-session",
                    execution_id="source-execution",
                ),
                source_step_ordinal=branch_cutoff,
                source_state_hash="sha256:source",
            )
        )
    return context


def _config(thread_id: str) -> RunnableConfig:
    # A compiled graph always supplies checkpoint_ns alongside thread_id.
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _put(saver: PlatformCheckpointer, thread_id: str, value: str) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"value": value}
    checkpoint["channel_versions"] = {"value": "1"}
    saver.put(
        _config(thread_id),
        checkpoint,
        {"source": "input", "step": -1},
        {"value": "1"},
    )


def _get_value(saver: Any, thread_id: str) -> str | None:
    found = saver.get_tuple(_config(thread_id))
    if found is None:
        return None
    return found.checkpoint["channel_values"].get("value")


class TestScratchThreadId:
    def test_keys_by_execution_attempt_and_fence(self) -> None:
        context = _attempt(
            attempt_id="attempt-9", fencing_token=7, execution_id="exec-3"
        )
        assert scratch_thread_id(context) == "exec-3:attempt-9:7"

    def test_requires_execution_id(self) -> None:
        context = AttemptContext(attempt_id="attempt-1", fencing_token=1)
        with pytest.raises(UnsupportedDurableGraphError):
            scratch_thread_id(context)

    def test_requires_positive_fencing_token(self) -> None:
        context = AttemptContext(
            attempt_id="attempt-1",
            workflow_identity=WorkflowIdentity(execution_id="execution-1"),
        )
        with pytest.raises(UnsupportedDurableGraphError):
            scratch_thread_id(context)

    def test_requires_attempt_id(self) -> None:
        context = AttemptContext(
            fencing_token=1,
            workflow_identity=WorkflowIdentity(execution_id="execution-1"),
        )
        with pytest.raises(UnsupportedDurableGraphError):
            scratch_thread_id(context)


class TestDirectInterruptActivity:
    @staticmethod
    def _record(
        *,
        task_path: str,
        value: str,
        native_id: str,
    ) -> None:
        attempt = current_attempt_context()
        assert attempt is not None
        PlatformCheckpointer._record_direct_interrupts(
            attempt,
            _config("ignored"),
            [("__interrupt__", (Interrupt(value=value, id=native_id),))],
            task_path,
        )

    def test_parallel_arrival_order_does_not_change_positions(self) -> None:
        observed: list[dict[str, tuple[tuple[tuple[str, int], ...], int]]] = []
        interrupts = (
            ("a", "native-a", "~__pregel_push, 0000000000, 0000000000"),
            ("b", "native-b", "~__pregel_push, 0000000001, 0000000000"),
        )

        for attempt, order in (
            (_attempt(), interrupts),
            (
                _attempt(attempt_id="attempt-2", fencing_token=2),
                tuple(reversed(interrupts)),
            ),
        ):
            with attempt_context_scope(attempt):
                for value, native_id, task_path in order:
                    self._record(
                        task_path=task_path,
                        value=value,
                        native_id=native_id,
                    )
                observed.append(
                    {
                        str(activity.control_flow.args[0][0].value): (
                            tuple(
                                (segment.name, segment.ordinal)
                                for segment in activity.command.position.operation_path.segments
                            ),
                            activity.command.position.activity_ordinal,
                        )
                        for activity in interrupted_activities(1)
                    }
                )

        assert observed[0] == observed[1]
        assert observed[0] == {
            "a": ((("langgraph.task:~__pregel_push, 0000000000, 0000000000", 1),), 1),
            "b": ((("langgraph.task:~__pregel_push, 0000000001, 0000000000", 1),), 1),
        }

    def test_wrapped_tool_interrupt_is_not_duplicated(self) -> None:
        native = Interrupt(value="wrapped", id="native-wrapped")
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            command = build_activity_command(
                attempt=attempt,
                kind=ACTIVITY_KIND_TOOL,
                name="langgraph.interrupt",
                activity_ordinal=1,
                semantic_input={},
            )
            record_interrupted_activity(command, GraphInterrupt((native,)))

            self._record(
                task_path="~__pregel_pull, tools",
                value="wrapped",
                native_id="native-wrapped",
            )
            activities = interrupted_activities(1)

        assert [item.command.activity_name for item in activities] == [
            "langgraph.interrupt"
        ]

    def test_untyped_interrupt_shaped_content_is_ignored(self) -> None:
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            PlatformCheckpointer._record_direct_interrupts(
                attempt,
                _config("ignored"),
                [("__interrupt__", ({"id": "forged", "value": "wait"},))],
                "~__pregel_pull, review",
            )
            assert interrupted_activities(1) == []

    def test_nested_write_registers_before_root_interrupt_projection(self) -> None:
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            config = _config("ignored")
            config["configurable"]["checkpoint_ns"] = "child:random-task-id"  # type: ignore[index]
            writes = [("__interrupt__", (Interrupt(value="wait", id="native-1"),))]
            with operation_path_resolver_scope(
                lambda: (ChildOperationBoundary("child", "random-task-id"),)
            ):
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    config,
                    writes,
                    "~__pregel_pull, review",
                )
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    _config("ignored"),
                    writes,
                    "~__pregel_pull, child",
                )
            activities = interrupted_activities(1)
            assert len(activities) == 1
            assert [
                (segment.name, segment.ordinal)
                for segment in activities[0].command.position.operation_path.segments
            ] == [
                ("agent", 1),
                ("child", 1),
                ("langgraph.task:~__pregel_pull, review", 1),
            ]

    def test_waiting_interrupt_written_again_must_keep_its_value(self) -> None:
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            task_path = "~__pregel_pull, review"

            def write(value: str) -> None:
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    _config("ignored"),
                    [("__interrupt__", (Interrupt(value=value, id="native-1"),))],
                    task_path,
                )

            write("wait")
            write("wait")
            assert len(interrupted_activities(1)) == 1

            with pytest.raises(
                UnsupportedDurableGraphError, match="conflicting values"
            ):
                write("changed")
            assert len(interrupted_activities(1)) == 1

    def test_deepest_write_owns_interrupt_through_every_ancestor(self) -> None:
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            writes = [("__interrupt__", (Interrupt(value="wait", id="native-1"),))]

            grandchild_config = _config("ignored")
            grandchild_config["configurable"]["checkpoint_ns"] = (  # type: ignore[index]
                "child:outer-task|grandchild:inner-task"
            )
            with operation_path_resolver_scope(
                lambda: (
                    ChildOperationBoundary("child", "outer-task"),
                    ChildOperationBoundary("grandchild", "inner-task"),
                )
            ):
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    grandchild_config,
                    writes,
                    "~__pregel_pull, review",
                )

            child_config = _config("ignored")
            child_config["configurable"]["checkpoint_ns"] = "child:outer-task"  # type: ignore[index]
            with operation_path_resolver_scope(
                lambda: (ChildOperationBoundary("child", "outer-task"),)
            ):
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    child_config,
                    writes,
                    "~__pregel_pull, grandchild",
                )

            PlatformCheckpointer._record_direct_interrupts(
                attempt,
                _config("ignored"),
                writes,
                "~__pregel_pull, child",
            )

            activities = interrupted_activities(1)
            observed = observed_activity_positions(1)

        assert len(activities) == 1
        assert len(observed) == 1
        assert observed[0] == activities[0].command.position
        assert [
            (segment.name, segment.ordinal)
            for segment in activities[0].command.position.operation_path.segments
        ] == [
            ("agent", 1),
            ("child", 1),
            ("grandchild", 1),
            ("langgraph.task:~__pregel_pull, review", 1),
        ]

    def test_root_interrupt_requires_framework_task_path(self) -> None:
        with attempt_context_scope(_attempt()):
            attempt = current_attempt_context()
            assert attempt is not None
            with pytest.raises(
                UnsupportedDurableGraphError,
                match="stable LangGraph task path",
            ):
                PlatformCheckpointer._record_direct_interrupts(
                    attempt,
                    _config("ignored"),
                    [("__interrupt__", (Interrupt(value="wait", id="native-1"),))],
                    "",
                )


class TestPlatformCheckpointerRouting:
    def test_native_ops_route_to_native_saver_without_attempt_context(self) -> None:
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)

        _put(saver, "thread-1", "native-state")

        assert _get_value(native, "thread-1") == "native-state"
        assert _get_value(saver, "thread-1") == "native-state"

    def test_native_ops_without_native_saver_fail_descriptively(self) -> None:
        saver = PlatformCheckpointer(native=None)

        with pytest.raises(RuntimeError, match="native_checkpoint"):
            saver.get_tuple(_config("thread-1"))
        # Version allocation follows the same rule: a native session must
        # never silently fall back to durable scratch.
        with pytest.raises(RuntimeError, match="native_checkpoint"):
            saver.get_next_version(None)

    def test_durable_scratch_is_isolated_by_fenced_attempt(self) -> None:
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)
        stale = _attempt(fencing_token=1)
        current = _attempt(fencing_token=2)

        with attempt_context_scope(stale):
            _put(saver, "ignored-thread", "stale-state")
        with attempt_context_scope(current):
            _put(saver, "ignored-thread", "current-state")

        with attempt_context_scope(stale):
            assert _get_value(saver, "ignored-thread") == "stale-state"
        with attempt_context_scope(current):
            assert _get_value(saver, "ignored-thread") == "current-state"
        # Durable scratch never lands in the native saver, under any thread id.
        assert _get_value(native, "ignored-thread") is None
        assert _get_value(native, scratch_thread_id(current)) is None

    def test_release_scratch_discards_only_that_attempt(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        first = _attempt(attempt_id="attempt-1")
        second = _attempt(attempt_id="attempt-2")

        with attempt_context_scope(first):
            _put(saver, "t", "first-state")
        with attempt_context_scope(second):
            _put(saver, "t", "second-state")

        saver.release_scratch(first)

        with attempt_context_scope(first):
            assert _get_value(saver, "t") is None
        with attempt_context_scope(second):
            assert _get_value(saver, "t") == "second-state"

    def test_native_state_survives_scratch_release(self) -> None:
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)
        attempt = _attempt()

        _put(saver, "thread-1", "native-state")
        with attempt_context_scope(attempt):
            _put(saver, "thread-1", "durable-state")
        saver.release_scratch(attempt)

        assert _get_value(saver, "thread-1") == "native-state"
        assert _get_value(native, "thread-1") == "native-state"

    def test_durable_list_is_scoped_to_the_current_attempt(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        first = _attempt(attempt_id="attempt-1")
        second = _attempt(attempt_id="attempt-2")

        with attempt_context_scope(first):
            _put(saver, "ignored", "first")
        with attempt_context_scope(second):
            _put(saver, "ignored", "second")

        with attempt_context_scope(first):
            history = list(saver.list(_config("ignored")))
            assert [item.checkpoint["channel_values"]["value"] for item in history] == [
                "first"
            ]
            with pytest.raises(UnsupportedDurableGraphError):
                list(saver.list(None))

        with attempt_context_scope(second):
            history = list(saver.list(_config("ignored")))
            assert [item.checkpoint["channel_values"]["value"] for item in history] == [
                "second"
            ]

    @pytest.mark.anyio
    async def test_durable_alist_reads_the_current_attempt_scratch(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        attempt = _attempt()
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "current"}
        checkpoint["channel_versions"] = {"value": "1"}

        with attempt_context_scope(attempt):
            await saver.aput(
                _config("ignored"),
                checkpoint,
                {"source": "input", "step": -1},
                {"value": "1"},
            )
            history = [item async for item in saver.alist(_config("ignored"))]

        assert [item.checkpoint["channel_values"]["value"] for item in history] == [
            "current"
        ]

    def test_durable_send_write_is_rejected(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())

        with attempt_context_scope(_attempt()):
            with pytest.raises(UnsupportedDurableGraphError, match="Send"):
                saver.put_writes(
                    _config("ignored"),
                    [("__pregel_tasks", [Send("research", {"query": "alpha"})])],
                    "producer-task",
                )

    def test_durable_send_graph_is_rejected_before_child_starts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        monkeypatch.setattr(saver, "_finalize_step_sync", MagicMock())
        child_started = False

        def dispatch(state: _GraphState) -> list[Send]:
            return [Send("child", state)]

        def child(state: _GraphState) -> _GraphState:
            nonlocal child_started
            child_started = True
            return state

        graph: StateGraph[Any] = StateGraph(_GraphState)
        graph.add_node("child", child)
        graph.add_conditional_edges(START, dispatch, ["child"])
        graph.add_edge("child", END)
        compiled = graph.compile(checkpointer=saver)

        with attempt_context_scope(_attempt()):
            with pytest.raises(UnsupportedDurableGraphError, match="Send"):
                compiled.invoke(
                    {"value": "alpha"},
                    _config("ignored"),
                    durability="sync",
                )

        assert child_started is False

    @pytest.mark.anyio
    async def test_durable_async_send_write_is_rejected(
        self,
    ) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())

        with attempt_context_scope(_attempt()):
            with pytest.raises(UnsupportedDurableGraphError, match="Send"):
                await saver.aput_writes(
                    _config("ignored"),
                    [("__pregel_tasks", Send("research", {"query": "alpha"}))],
                    "producer-task",
                )

    @pytest.mark.anyio
    async def test_durable_async_send_graph_is_rejected_before_child_starts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        monkeypatch.setattr(saver, "_finalize_step_async", AsyncMock())
        child_started = False

        def dispatch(state: _GraphState) -> list[Send]:
            return [Send("child", state)]

        async def child(state: _GraphState) -> _GraphState:
            nonlocal child_started
            child_started = True
            return state

        graph: StateGraph[Any] = StateGraph(_GraphState)
        graph.add_node("child", child)
        graph.add_conditional_edges(START, dispatch, ["child"])
        graph.add_edge("child", END)
        compiled = graph.compile(checkpointer=saver)

        with attempt_context_scope(_attempt()):
            with pytest.raises(UnsupportedDurableGraphError, match="Send"):
                await compiled.ainvoke(
                    {"value": "alpha"},
                    _config("ignored"),
                    durability="sync",
                )

        assert child_started is False

    def test_durable_update_and_fork_puts_are_rejected(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())

        with attempt_context_scope(_attempt()):
            for source in ("update", "fork"):
                with pytest.raises(UnsupportedDurableGraphError):
                    saver.put(
                        _config("t"),
                        empty_checkpoint(),
                        {"source": source, "step": 1},
                        {},
                    )

    def test_native_list_and_update_remain_available(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())

        _put(saver, "thread-1", "native-state")
        saver.put(
            _config("thread-1"), empty_checkpoint(), {"source": "update", "step": 2}, {}
        )

        assert list(saver.list(_config("thread-1")))

    def test_durable_delete_thread_targets_the_fenced_scratch(self) -> None:
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)
        attempt = _attempt()

        _put(saver, "thread-1", "native-state")
        with attempt_context_scope(attempt):
            _put(saver, "thread-1", "durable-state")
            saver.delete_thread("thread-1")
            assert _get_value(saver, "thread-1") is None

        assert _get_value(native, "thread-1") == "native-state"


class _WorkflowClient:
    def __init__(self) -> None:
        self.completions: list[Any] = []
        self.commits: list[Any] = []

    async def __aenter__(self) -> _WorkflowClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    async def finalize_step(self, command: Any) -> list[Any]:
        self.commits.append(command)
        return []

    async def complete_execution(self, command: Any) -> None:
        self.completions.append(command)


class _InterruptWorkflowClient(_WorkflowClient):
    def __init__(self) -> None:
        super().__init__()
        self.resolved_values: dict[str, str] = {}
        self.activity_ids: dict[bytes, str] = {}

    async def finalize_step(self, command: Any) -> list[Any]:
        self.commits.append(command)
        if not command.suspensions:
            return []
        entries: list[StepActivityEntry] = []
        for suspension in command.suspensions:
            position_key = suspension.position.SerializeToString(deterministic=True)
            activity_id = self.activity_ids.setdefault(
                position_key, f"activity-{len(self.activity_ids) + 1}"
            )
            resolved_value = self.resolved_values.get(activity_id)
            outcome = ActivityOutcome(
                workflow_identity=command.workflow_identity,
                activity_id=activity_id,
                attempt_id=command.attempt_id,
                fencing_token=command.fencing_token,
                outcome_kind=(
                    ACTIVITY_OUTCOME_KIND_COMPLETED
                    if resolved_value is not None
                    else ACTIVITY_OUTCOME_KIND_SUSPENDED
                ),
            )
            if resolved_value is None:
                outcome.suspension.CopyFrom(suspension.suspension)
            else:
                outcome.result.string_value = resolved_value
            entries.append(
                StepActivityEntry(position=suspension.position, outcome=outcome)
            )
        return entries


class _MemoryWorkflowClient:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    def ensure_memory_written(self, command: Any) -> None:
        self.commands.append(command)


class TestDurableStateCompletion:
    @pytest.mark.anyio
    async def test_loop_checkpoint_records_its_absolute_oe_ordinal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        attempt = _attempt(branch_cutoff=4)
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"value": "branch-local"}
        checkpoint["channel_versions"] = {"value": "1"}

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                config = await saver.aput(
                    _config("ignored"),
                    checkpoint,
                    {"source": "loop", "step": 17},
                    checkpoint["channel_versions"],
                )
                stored = await saver.aget_tuple(config)
        finally:
            current_oe_url.reset(url_token)

        assert [command.step_ordinal for command in client.commits] == [5]
        assert stored is not None
        assert dict(stored.metadata).get(OE_STEP_ORDINAL_METADATA_KEY) == 5

    @pytest.mark.anyio
    async def test_nested_loop_checkpoint_stays_scratch_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"evidence": "billing timeline"}
        checkpoint["channel_versions"] = {"evidence": "1"}
        config = _config("ignored")
        config["configurable"]["checkpoint_ns"] = (  # type: ignore[index]
            "case_investigation:task-a|collect_evidence:task-b"
        )

        with attempt_context_scope(_attempt()):
            await saver.aput(
                config,
                checkpoint,
                {"source": "loop", "step": 0},
                checkpoint["channel_versions"],
            )
            assert current_step_ordinal() == 1

        assert client.commits == []

    @pytest.mark.anyio
    async def test_loop_checkpoint_is_only_cached_until_execution_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": [HumanMessage("hello"), AIMessage("hi")],
            "quote_count": 2,
            "branch:to:agent": None,
            "__start__": {"message": "hello"},
        }
        checkpoint["channel_versions"] = {
            name: "1" for name in checkpoint["channel_values"]
        }
        attempt = _attempt()
        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await saver.aput(
                    _config("ignored"),
                    checkpoint,
                    {"source": "input", "step": -1},
                    checkpoint["channel_versions"],
                )
                assert client.completions == []
                assert client.commits == []

                await saver.aput(
                    _config("ignored"),
                    checkpoint,
                    {"source": "loop", "step": 0},
                    checkpoint["channel_versions"],
                )
                assert client.completions == []
                assert len(client.commits) == 1
                assert client.commits[0].step_ordinal == 1
                await saver.complete_execution(attempt)
        finally:
            current_oe_url.reset(url_token)

        (command,) = client.completions
        assert dict(command.state.properties) == {"quote_count": 2}
        assert [message.content.string_value for message in command.state.messages] == [
            "hello",
            "hi",
        ]

    @pytest.mark.anyio
    async def test_completion_uses_the_latest_loop_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        attempt = _attempt()
        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                for step, value in enumerate(("first", "done")):
                    checkpoint = empty_checkpoint()
                    checkpoint["channel_values"] = {"value": value}
                    checkpoint["channel_versions"] = {"value": str(step + 1)}
                    await saver.aput(
                        _config("ignored"),
                        checkpoint,
                        {"source": "loop", "step": step},
                        checkpoint["channel_versions"],
                    )
                await saver.complete_execution(attempt)
        finally:
            current_oe_url.reset(url_token)

        assert [command.step_ordinal for command in client.commits] == [1, 2]
        assert dict(client.commits[-1].state.properties) == {"value": "done"}
        (completion,) = client.completions
        assert completion.attempt_id == attempt.attempt_id
        assert dict(completion.state.properties) == {"value": "done"}

    @pytest.mark.anyio
    async def test_completion_requires_a_final_checkpoint(self) -> None:
        saver = PlatformCheckpointer(native=InMemorySaver())
        attempt = _attempt()

        with attempt_context_scope(attempt):
            with pytest.raises(
                UnsupportedDurableGraphError,
                match="without final application state",
            ):
                await saver.complete_execution(attempt)


class _GraphState(TypedDict):
    value: str


def _compiled_graph(saver: PlatformCheckpointer) -> Any:
    graph: StateGraph[Any] = StateGraph(_GraphState)

    async def node(state: _GraphState) -> _GraphState:
        await asyncio.sleep(0.01)
        return {"value": state["value"] + "+ran"}

    graph.add_node("node", node)
    graph.add_edge(START, "node")
    return graph.compile(checkpointer=saver)


class TestOneCompiledGraphMixedAuthority:
    @pytest.mark.anyio
    async def test_concurrent_native_and_durable_invocations_do_not_share_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)
        compiled = _compiled_graph(saver)
        attempt = _attempt()

        async def durable_run() -> dict[str, Any]:
            url_token = current_oe_url.set("http://oe")
            try:
                with attempt_context_scope(attempt):
                    return await compiled.ainvoke(
                        {"value": "durable"}, _config("session-thread")
                    )
            finally:
                current_oe_url.reset(url_token)

        async def native_run() -> dict[str, Any]:
            return await compiled.ainvoke(
                {"value": "native"}, _config("session-thread")
            )

        durable_result, native_result = await asyncio.gather(
            durable_run(), native_run()
        )

        assert durable_result["value"] == "durable+ran"
        assert native_result["value"] == "native+ran"
        assert [command.step_ordinal for command in client.commits] == [1, 2]
        # Native checkpoints persist under the session thread; durable state
        # exists only in fenced scratch and is discarded with the attempt.
        assert _get_value(native, "session-thread") == "native+ran"
        saver.release_scratch(attempt)
        with attempt_context_scope(attempt):
            assert _get_value(saver, "session-thread") is None


class TestDelayedFinalizeStepBoundary:
    @pytest.mark.anyio
    async def test_multi_node_graph_commits_contiguous_ordinals_with_delayed_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FinalizeStep may be slow; the shared counter must still advance."""
        from agent_engine_sdk_langgraph import platform_checkpointer as module
        from agent_engine_runner_shared.workflow.context import current_step_ordinal

        class _DelayedClient(_WorkflowClient):
            def __init__(self) -> None:
                super().__init__()
                self.seen_steps: list[int] = []

            async def finalize_step(self, command: Any) -> list[Any]:
                await asyncio.sleep(0.05)
                return await super().finalize_step(command)

        client = _DelayedClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())

        graph: StateGraph[Any] = StateGraph(_GraphState)

        async def first(state: _GraphState) -> _GraphState:
            client.seen_steps.append(current_step_ordinal())
            return {"value": state["value"] + "+a"}

        async def second(state: _GraphState) -> _GraphState:
            client.seen_steps.append(current_step_ordinal())
            return {"value": state["value"] + "+b"}

        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        compiled = graph.compile(checkpointer=saver)
        attempt = _attempt()

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                result = await compiled.ainvoke(
                    {"value": "x"},
                    _config("delayed-thread"),
                    durability="sync",
                )
        finally:
            current_oe_url.reset(url_token)

        assert result["value"] == "x+a+b"
        # LangGraph may commit an empty root loop before the first node runs;
        # later nodes must still see a strictly advanced shared ordinal.
        assert len(client.seen_steps) == 2
        assert client.seen_steps[1] == client.seen_steps[0] + 1
        commit_ordinals = [command.step_ordinal for command in client.commits]
        assert commit_ordinals == list(range(1, len(commit_ordinals) + 1))
        assert len(commit_ordinals) >= 2
        assert client.seen_steps[-1] == commit_ordinals[-1]


class TestAgentDurableLifecycle:
    def test_explicit_checkpoint_targeting_is_rejected_in_durable_context(self) -> None:
        agent = LangGraphBaseAgent(MagicMock())
        ctx = RequestContext(
            user_id="user-1",
            session_id="session-1",
            metadata={"checkpoint_id": "checkpoint-42"},
        )

        with attempt_context_scope(_attempt()):
            with pytest.raises(UnsupportedDurableGraphError):
                agent._execution_session().build_config(ctx)

        config = agent._execution_session().build_config(ctx)
        assert config.get("configurable", {}).get("checkpoint_id") == "checkpoint-42"

    @pytest.mark.anyio
    async def test_durable_invoke_requests_sync_durability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage("ok")]})
        graph.aget_state = AsyncMock(return_value=MagicMock(next=()))
        graph.checkpointer = PlatformCheckpointer(native=InMemorySaver())
        agent = LangGraphBaseAgent(graph)
        monkeypatch.setattr(
            ExecutionSession, "seed_previous_session_state", AsyncMock()
        )
        monkeypatch.setattr(ExecutionSession, "complete_execution", AsyncMock())
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        with attempt_context_scope(_attempt()):
            await agent.invoke(ctx, AgentInput(payload={"message": "hi"}))

        assert graph.ainvoke.await_args.kwargs.get("durability") == "sync"

        graph.ainvoke.reset_mock()
        await agent.invoke(ctx, AgentInput(payload={"message": "hi"}))
        assert "durability" not in graph.ainvoke.await_args.kwargs

    @pytest.mark.anyio
    async def test_durable_stream_requests_sync_durability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)

        async def _empty_stream(*_args: Any, **_kwargs: Any) -> Any:
            if False:
                yield None

        graph = MagicMock()
        graph.astream = MagicMock(return_value=_empty_stream())
        graph.aget_state = AsyncMock(return_value=MagicMock(next=()))
        graph.checkpointer = PlatformCheckpointer(native=InMemorySaver())
        agent = LangGraphBaseAgent(graph)
        monkeypatch.setattr(
            ExecutionSession, "seed_previous_session_state", AsyncMock()
        )
        monkeypatch.setattr(ExecutionSession, "complete_execution", AsyncMock())
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        with attempt_context_scope(_attempt()):
            with pytest.raises(RuntimeError, match="without producing messages"):
                async for _ in agent.stream(ctx, AgentInput(payload={"message": "hi"})):
                    pass

        assert graph.astream.call_args.kwargs.get("durability") == "sync"

        graph.astream.reset_mock()
        graph.astream.return_value = _empty_stream()
        with pytest.raises(RuntimeError, match="without producing messages"):
            async for _ in agent.stream(ctx, AgentInput(payload={"message": "hi"})):
                pass
        assert "durability" not in graph.astream.call_args.kwargs

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    async def test_sibling_subgraph_message_ids_are_distinct_on_replay(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
    ) -> None:
        from langgraph.graph import MessagesState
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module
        from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)

        def child(content: str) -> Any:
            builder: StateGraph[Any] = StateGraph(MessagesState)
            builder.add_node(
                "respond",
                lambda state: {"messages": [AIMessage(content=content)]},
            )
            builder.add_edge(START, "respond")
            builder.add_edge("respond", END)
            return builder.compile()

        saver = PlatformCheckpointer(native=InMemorySaver())
        builder: StateGraph[Any] = StateGraph(MessagesState)
        builder.add_node("left", child("left"))
        builder.add_node("right", child("right"))
        builder.add_edge(START, "left")
        builder.add_edge(START, "right")
        builder.add_edge("left", END)
        builder.add_edge("right", END)
        graph = builder.compile(checkpointer=saver)
        agent = LangGraphBaseAgent(
            graph,
            durable_subgraphs=DurableSubgraphResolver(graph),
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        async def run() -> None:
            agent_input = AgentInput(payload={"message": "hi"})
            if method == "invoke":
                await agent.invoke(ctx, agent_input)
                return
            async for _ in agent.stream(ctx, agent_input):
                pass

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(_attempt()):
                await run()
            replay = _attempt(attempt_id="attempt-2", fencing_token=2)
            replay.replay_mode = True
            with attempt_context_scope(replay):
                await run()
        finally:
            current_oe_url.reset(url_token)

        def emitted(call_index: int) -> dict[str, str]:
            return {
                message.content.string_value: message.id
                for message in client.completions[call_index].state.messages
                if message.content.string_value in {"left", "right"}
            }

        first = emitted(0)
        replayed = emitted(1)
        assert set(first) == {"left", "right"}
        assert first == replayed
        assert len(set(first.values())) == 2
        assert all(
            message_id.startswith("durable-message:") for message_id in first.values()
        )

    @pytest.mark.anyio
    async def test_invoke_releases_the_attempt_scratch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from langchain_core.messages import AIMessage
        from langgraph.graph import MessagesState
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)

        saver = PlatformCheckpointer(native=InMemorySaver())
        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node("node", lambda state: {"messages": [AIMessage("ok")]})
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(graph.compile(checkpointer=saver))
        attempt = _attempt()
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                output = await agent.invoke(ctx, AgentInput(payload={"message": "hi"}))
                response = cast("dict[str, Any]", output.response)
                assert response["status"] == "completed"
                # The attempt's scratch was discarded before invoke returned.
                assert saver.get_tuple(_config("session-1")) is None
        finally:
            current_oe_url.reset(url_token)

        assert len(client.completions) == 1

    @pytest.mark.anyio
    async def test_previous_state_uses_graph_reducers_for_every_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prior and current values combine under the application's reducers."""
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module

        class TurnState(TypedDict):
            messages: Annotated[list[Any], add_messages]
            labels: Annotated[list[str], operator.add]

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)

        saver = PlatformCheckpointer(native=InMemorySaver())
        graph: StateGraph[Any] = StateGraph(TurnState)
        graph.add_node("node", lambda state: {"messages": [AIMessage("second answer")]})
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(
            graph.compile(checkpointer=saver),
            prepare_input=lambda _input, _ctx: {
                "messages": [HumanMessage("second question")],
                "labels": ["current"],
            },
        )
        attempt = _attempt()
        attempt.previous_state.CopyFrom(
            StateSnapshot(
                properties={"labels": ["previous"]},
                messages=[
                    WorkflowMessage(
                        role=MESSAGE_ROLE_USER,
                        content=Value(string_value="first question"),
                    ),
                    WorkflowMessage(
                        role=MESSAGE_ROLE_ASSISTANT,
                        content=Value(string_value="first answer"),
                    ),
                ],
            )
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent.invoke(
                    ctx,
                    AgentInput(payload={"message": "second question"}),
                )
        finally:
            current_oe_url.reset(url_token)

        (completion,) = client.completions
        assert completion.state.properties["labels"] == ["previous", "current"]
        assert [
            message.content.string_value for message in completion.state.messages
        ] == [
            "first question",
            "first answer",
            "second question",
            "second answer",
        ]

    @pytest.mark.anyio
    async def test_branch_first_commit_starts_after_source_cutoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from langgraph.graph import MessagesState
        from agent_engine_sdk import AgentInput

        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node("node", lambda state: {"messages": [AIMessage("branch answer")]})
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(graph.compile(checkpointer=saver))
        attempt = _attempt(branch_cutoff=4)
        attempt.previous_state.CopyFrom(
            StateSnapshot(
                messages=[
                    WorkflowMessage(
                        role=MESSAGE_ROLE_ASSISTANT,
                        content=Value(string_value="source answer"),
                    )
                ]
            )
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent.invoke(
                    ctx, AgentInput(payload={"message": "branch question"})
                )
        finally:
            current_oe_url.reset(url_token)

        assert client.commits
        assert client.commits[0].step_ordinal == 5
        assert [
            message.content.string_value
            for message in client.completions[0].state.messages
        ] == ["source answer", "branch question", "branch answer"]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "pause_value",
        [
            {"suspend_reason": "approve?"},
            # An application may pause with any value. One that looks like the
            # platform's guardrail review wait is still the application's own
            # pause, and its answer is still part of the conversation.
            {"guardrail_review": {"review_id": "review-1"}},
        ],
    )
    async def test_direct_interrupt_suspends_resumes_and_closes_memory_once(
        self, monkeypatch: pytest.MonkeyPatch, pause_value: dict[str, Any]
    ) -> None:
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )
        from agent_engine_runner_shared.context import current_user_id, current_wrapper

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )
        saver = PlatformCheckpointer(native=InMemorySaver())
        graph: StateGraph[Any] = StateGraph(MessagesState)

        def node(state: Any) -> Any:
            decision = interrupt(pause_value)
            return {"messages": [AIMessage(content=f"decision:{decision}")]}

        graph.add_node("node", node)
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(graph.compile(checkpointer=saver))
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        memory_workflow = _MemoryWorkflowClient()
        wrapper = MagicMock(
            durable_memory=DurableMemoryState(),
            workflow=memory_workflow,
        )

        initial_attempt = _attempt()
        initial_attempt.workflow_identity.tenant_scope.org_id = "org-1"
        initial_attempt.workflow_identity.tenant_scope.project_id = "project-1"
        initial_attempt.workflow_identity.tenant_scope.workspace_id = "workspace-1"

        url_token = current_oe_url.set("http://oe")
        user_token = current_user_id.set("user-1")
        wrapper_token = current_wrapper.set(wrapper)
        try:
            with attempt_context_scope(initial_attempt):
                suspended = await agent.invoke(
                    ctx, AgentInput(payload={"message": "hi"})
                )
            assert isinstance(suspended.response, dict)
            assert suspended.response["status"] == "suspended"
            assert suspended.response["interrupts"] == [
                {
                    "id": "activity-1",
                    "value": pause_value,
                }
            ]
            assert memory_workflow.commands == []

            client.resolved_values["activity-1"] = "approved"
            replacement = _attempt(attempt_id="attempt-2", fencing_token=2)
            replacement.workflow_identity.tenant_scope.CopyFrom(
                initial_attempt.workflow_identity.tenant_scope
            )
            replacement.replay_mode = True
            resume_ctx = ctx.model_copy(
                update={
                    "resume": True,
                    "resume_data": {"activity-1": "approved"},
                }
            )
            with attempt_context_scope(replacement):
                completed = await agent.invoke(
                    resume_ctx, AgentInput(payload={"message": ""})
                )
            assert isinstance(completed.response, dict)
            assert completed.response["status"] == "completed"
            assert completed.response["response"] == "decision:approved"
        finally:
            current_wrapper.reset(wrapper_token)
            current_user_id.reset(user_token)
            current_oe_url.reset(url_token)

        suspension_commands = [
            command for command in client.commits if command.suspensions
        ]
        assert len(suspension_commands) == 2
        assert (
            suspension_commands[0].suspensions[0].position
            == suspension_commands[1].suspensions[0].position
        )
        (memory_command,) = memory_workflow.commands
        assert memory_command.activity_id == "activity-1"
        assert memory_command.attempt_id == "attempt-2"
        assert memory_command.fencing_token == 2
        (memory_write,) = memory_command.memory_writes
        assert memory_write.id == "workflow:execution-1:activity-1:tool:0"
        assert json.loads(memory_write.payload_json) == {
            "session_id": "session-1",
            "org_id": "org-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "agent_id": "workspace-1",
            "idempotency_key": "workflow:execution-1:activity-1:tool:0",
            "content": "approved",
            "role": "tool",
            "tool_call_id": "langgraph.interrupt:activity-1",
            "tool_name": "langgraph.interrupt",
            "is_error": False,
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    @pytest.mark.parametrize("compiled_child", [False, True])
    async def test_one_task_interrupts_several_times_across_attempts(
        self, monkeypatch: pytest.MonkeyPatch, method: str, compiled_child: bool
    ) -> None:
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )
        from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )
        node_runs = 0
        received: list[list[Any]] = []

        def node(state: Any) -> Any:
            nonlocal node_runs
            node_runs += 1
            first = interrupt({"pause": 1})
            second = interrupt({"pause": 2, "first": first})
            third = interrupt({"pause": 3, "second": second})
            received.append([first, second, third])
            return {"messages": [AIMessage(content=f"{first}|{second}|{third}")]}

        inner: StateGraph[Any] = StateGraph(MessagesState)
        inner.add_node("node", node)
        inner.add_edge(START, "node")
        inner.add_edge("node", END)
        if compiled_child:
            root: StateGraph[Any] = StateGraph(MessagesState)
            root.add_node("child", inner.compile())
            root.add_edge(START, "child")
            root.add_edge("child", END)
        else:
            root = inner
        compiled = root.compile(
            checkpointer=PlatformCheckpointer(native=InMemorySaver())
        )
        agent = LangGraphBaseAgent(
            compiled,
            durable_subgraphs=(
                DurableSubgraphResolver(compiled) if compiled_child else None
            ),
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        agent_input = AgentInput(payload={"message": "hi"})

        async def run(
            attempt: AttemptContext, resume: dict[str, str] | None = None
        ) -> dict[str, Any]:
            run_ctx = ctx
            if resume is not None:
                run_ctx = ctx.model_copy(update={"resume": True, "resume_data": resume})
            with attempt_context_scope(attempt):
                if method == "invoke":
                    output = await agent.invoke(run_ctx, agent_input)
                    assert isinstance(output.response, dict)
                    return output.response
                events = [event async for event in agent.stream(run_ctx, agent_input)]
                assert len(events) == 1
                assert events[0].event in {"suspend", "result"}
                assert isinstance(events[0].data, dict)
                return events[0].data

        url_token = current_oe_url.set("http://oe")
        try:
            result = await run(_attempt())
            answered: list[str] = []
            for pause in (1, 2, 3):
                (pending,) = cast(list[dict[str, Any]], result["interrupts"])
                assert pending["value"]["pause"] == pause
                pause_id = cast(str, pending["id"])
                assert pause_id not in answered
                answer = f"answer-{pause}"
                client.resolved_values[pause_id] = answer
                answered.append(pause_id)
                # Each answer arrives on a fresh attempt: nothing survives from
                # the previous process except OE history.
                replacement = _attempt(
                    attempt_id=f"attempt-{pause + 1}", fencing_token=pause + 1
                )
                replacement.replay_mode = True
                result = await run(replacement, {pause_id: answer})
        finally:
            current_oe_url.reset(url_token)

        if method == "invoke":
            assert result["status"] == "completed"
        assert result["response"] == "answer-1|answer-2|answer-3"
        # Every completed run of the node saw each pause's own answer.
        assert received
        assert all(run == ["answer-1", "answer-2", "answer-3"] for run in received)
        # 1 run per pause on the first attempt path, plus the replays that walk
        # back to each later pause: the node body is side-effect free here.
        assert node_runs >= 4
        suspensions = [
            command.suspensions[0] for command in client.commits if command.suspensions
        ]
        by_position: dict[bytes, int] = {}
        for suspension in suspensions:
            value = json.loads(json_format.MessageToJson(suspension.suspension.context))
            key = suspension.position.SerializeToString(deterministic=True)
            assert (
                by_position.setdefault(key, value["value"]["pause"])
                == (value["value"]["pause"])
            )
        # Three pauses, three distinct durable positions in one task.
        assert sorted(by_position.values()) == [1, 2, 3]
        paths = {
            tuple(
                (segment.name, segment.ordinal)
                for segment in suspension.position.operation_path.segments
            )
            for suspension in suspensions
        }
        assert len(paths) == 1
        assert sorted(
            {suspension.position.activity_ordinal for suspension in suspensions}
        ) == [1, 2, 3]

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    async def test_direct_pause_keeps_its_position_around_an_interrupting_tool(
        self, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        from langchain_core.messages import BaseMessage
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )
        from agent_engine_sdk_langgraph.runtime import App
        from agent_engine_runner_shared import RuntimeMode, hooks
        from agent_engine_runner_shared.context import (
            clear_execution_context,
            set_execution_context,
        )
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_COMPLETED,
            ActivityContext,
            ActivityOutcome,
        )
        from agent_engine_runner_shared.models import ToolExecuteResponse
        from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper
        from agent_engine_runner_shared.workflow.client import (
            ActivityDispatch,
            ActivityReplay,
        )

        client = _InterruptWorkflowClient()

        class ToolActivities:
            """Tool activities sharing OE's activity ids with the frontier."""

            def __init__(self) -> None:
                self.positions: dict[str, bytes] = {}
                self.recorded: dict[bytes, ActivityOutcome] = {}

            def start_activity(self, command: Any) -> Any:
                position = command.position.SerializeToString(deterministic=True)
                if position in self.recorded:
                    return ActivityReplay(outcome=self.recorded[position])
                activity_id = client.activity_ids.setdefault(
                    position, f"activity-{len(client.activity_ids) + 1}"
                )
                self.positions[activity_id] = position
                return ActivityDispatch(
                    context=ActivityContext(
                        workflow_identity=command.workflow_identity,
                        activity_id=activity_id,
                        attempt_id=command.attempt_id,
                        fencing_token=command.fencing_token,
                    )
                )

            def resolve(self, activity_id: str, value: str) -> None:
                # The replacement named by the resume map re-enters the tool;
                # later attempts replay this answer without re-entering it.
                outcome = ActivityOutcome(
                    activity_id=activity_id,
                    outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
                )
                outcome.result.string_value = value
                self.recorded[self.positions[activity_id]] = outcome

            def report_outcome(self, outcome: ActivityOutcome) -> None:
                recorded = ActivityOutcome()
                recorded.CopyFrom(outcome)
                self.recorded[self.positions[outcome.activity_id]] = recorded

        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )
        app = App(app_name="durable mixed pauses")
        app._runtime.mode = RuntimeMode.AER
        app._register_hooks()
        tool_runs: list[str] = []

        @app.tool(is_local=True)
        def review_claim(claim_id: str) -> Any:
            """Pause inside a durable tool."""
            tool_runs.append(claim_id)
            return interrupt({"pause": "tool", "claim_id": claim_id})

        wrapped_tool = app.get_tools()[0]
        tool_activities = ToolActivities()
        wrapper = SecureToolWrapper("http://oe", "execution-1")
        wrapper._workflow = tool_activities
        received: list[list[Any]] = []

        def node(state: Any) -> Any:
            first = interrupt({"pause": "first"})
            result = wrapped_tool.func(claim_id="claim-1", tool_call_id="call-1")
            value = result[0] if isinstance(result, tuple) else result
            tool = value.content if isinstance(value, BaseMessage) else value
            second = interrupt({"pause": "second"})
            received.append([first, tool, second])
            return {"messages": [AIMessage(content=f"{first}|{tool}|{second}")]}

        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(
            graph.compile(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        approval = ToolExecuteResponse(
            proceed=True, route_to="callback", latest_step_number=1
        )
        execution_tokens = set_execution_context(
            execution_id="execution-1", wrapper=wrapper, oe_url="http://oe"
        )

        async def run(
            attempt: AttemptContext, resume: dict[str, str] | None
        ) -> dict[str, Any]:
            run_ctx = (
                ctx
                if resume is None
                else ctx.model_copy(update={"resume": True, "resume_data": resume})
            )
            agent_input = AgentInput(payload={"message": "hi"})
            with attempt_context_scope(attempt):
                if method == "invoke":
                    output = await agent.invoke(run_ctx, agent_input)
                    return cast(dict[str, Any], output.response)
                events = [event async for event in agent.stream(run_ctx, agent_input)]
                return cast(dict[str, Any], events[-1].data)

        positions: dict[str, bytes] = {}
        url_token = current_oe_url.set("http://oe")
        try:
            with (
                patch(
                    "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
                    return_value=approval,
                ),
                patch("agent_engine_runner_shared.secure_wrapper.report_oe_result"),
            ):
                result = await run(_attempt(), None)
                for number, pause in enumerate(("first", "tool", "second"), 1):
                    (waiting,) = cast(list[dict[str, Any]], result["interrupts"])
                    assert waiting["value"]["pause"] == pause
                    answer = f"answer-{pause}"
                    client.resolved_values[waiting["id"]] = answer
                    if pause == "tool":
                        tool_activities.resolve(waiting["id"], answer)
                    replacement = _attempt(
                        attempt_id=f"attempt-{number + 1}", fencing_token=number + 1
                    )
                    replacement.replay_mode = True
                    result = await run(replacement, {waiting["id"]: answer})
        finally:
            current_oe_url.reset(url_token)
            clear_execution_context(execution_tokens)
            hooks.reset_hooks()

        if method == "invoke":
            assert result["status"] == "completed"
        assert result["response"] == "answer-first|answer-tool|answer-second"
        assert received == [["answer-first", "answer-tool", "answer-second"]]
        for command in client.commits:
            for suspension in command.suspensions:
                value = json.loads(
                    json_format.MessageToJson(suspension.suspension.context)
                )["value"]
                key = suspension.position.SerializeToString(deterministic=True)
                # A replayed pause must come back at the position it was
                # recorded at, whether or not the tool re-entered its callback.
                assert positions.setdefault(value["pause"], key) == key
        assert len(set(positions.values())) == 3
        # Dispatched once, re-entered twice to reconstruct and answer its own
        # pause, then replayed without re-entry while the second pause resumes.
        assert tool_runs == ["claim-1", "claim-1", "claim-1"]

    @pytest.mark.anyio
    async def test_task_pauses_again_while_parallel_task_waits_alongside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )

        def twice(state: Any) -> Any:
            first = interrupt({"task": "twice", "pause": 1})
            second = interrupt({"task": "twice", "pause": 2})
            return {"messages": [AIMessage(content=f"twice:{first}|{second}")]}

        once_answers: list[str] = []

        def once(state: Any) -> Any:
            answer = interrupt({"task": "once", "pause": 1})
            once_answers.append(answer)
            return {"messages": [AIMessage(content=f"once:{answer}")]}

        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node("twice", twice)
        graph.add_node("once", once)
        graph.add_edge(START, "twice")
        graph.add_edge(START, "once")
        graph.add_edge("twice", END)
        graph.add_edge("once", END)
        agent = LangGraphBaseAgent(
            graph.compile(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        agent_input = AgentInput(payload={"message": "hi"})

        async def run(
            attempt: AttemptContext, resume: dict[str, str] | None = None
        ) -> dict[str, Any]:
            run_ctx = ctx
            if resume is not None:
                run_ctx = ctx.model_copy(update={"resume": True, "resume_data": resume})
            with attempt_context_scope(attempt):
                output = await agent.invoke(run_ctx, agent_input)
            assert isinstance(output.response, dict)
            return output.response

        def pending(result: dict[str, Any]) -> dict[tuple[str, int], str]:
            return {
                (item["value"]["task"], item["value"]["pause"]): item["id"]
                for item in cast(list[dict[str, Any]], result["interrupts"])
            }

        url_token = current_oe_url.set("http://oe")
        try:
            first = pending(await run(_attempt()))
            assert set(first) == {("twice", 1), ("once", 1)}
            answers = {
                first[("twice", 1)]: "t1",
                first[("once", 1)]: "o1",
            }
            client.resolved_values.update(answers)
            replacement = _attempt(attempt_id="attempt-2", fencing_token=2)
            replacement.replay_mode = True
            second = pending(await run(replacement, answers))
            # Only the task that pauses again is waiting; its answered pause and
            # the finished parallel task are not re-offered.
            assert set(second) == {("twice", 2)}
            assert second[("twice", 2)] not in answers
            client.resolved_values[second[("twice", 2)]] = "t2"
            final_attempt = _attempt(attempt_id="attempt-3", fencing_token=3)
            final_attempt.replay_mode = True
            completed = await run(final_attempt, {second[("twice", 2)]: "t2"})
        finally:
            current_oe_url.reset(url_token)

        assert completed["status"] == "completed"
        assert completed["response"] == "twice:t1|t2"
        # Human input plus one message from each task.
        assert completed["message_count"] == 3
        # The parallel task only ever receives its own answer.
        assert set(once_answers) == {"o1"}

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    async def test_compiled_child_can_interrupt_twice_in_one_root_step(
        self, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )
        from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )

        def first_node(state: Any) -> Any:
            answer = interrupt({"review": "first"})
            return {"messages": [AIMessage(content=f"first:{answer}")]}

        def second_node(state: Any) -> Any:
            answer = interrupt({"review": "second"})
            return {"messages": [AIMessage(content=f"second:{answer}")]}

        child: StateGraph[Any] = StateGraph(MessagesState)
        child.add_node("first", first_node)
        child.add_node("second", second_node)
        child.add_edge(START, "first")
        child.add_edge("first", "second")
        child.add_edge("second", END)

        parent: StateGraph[Any] = StateGraph(MessagesState)
        parent.add_node("review", child.compile())
        parent.add_edge(START, "review")
        parent.add_edge("review", END)
        compiled = parent.compile(
            checkpointer=PlatformCheckpointer(native=InMemorySaver())
        )
        agent = LangGraphBaseAgent(
            compiled,
            durable_subgraphs=DurableSubgraphResolver(compiled),
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        agent_input = AgentInput(payload={"message": "hi"})

        async def run(
            attempt: AttemptContext, resume_id: str | None = None
        ) -> dict[str, Any]:
            run_ctx = ctx
            if resume_id is not None:
                run_ctx = ctx.model_copy(
                    update={
                        "resume": True,
                        "resume_data": {resume_id: "approved"},
                    }
                )
            with attempt_context_scope(attempt):
                if method == "invoke":
                    output = await agent.invoke(run_ctx, agent_input)
                    assert isinstance(output.response, dict)
                    return output.response
                events = [event async for event in agent.stream(run_ctx, agent_input)]
                assert len(events) == 1
                assert events[0].event in {"suspend", "result"}
                assert isinstance(events[0].data, dict)
                return events[0].data

        url_token = current_oe_url.set("http://oe")
        try:
            first = await run(_attempt())
            first_interrupt = cast(list[dict[str, Any]], first["interrupts"])[0]
            first_id = cast(str, first_interrupt["id"])
            client.resolved_values[first_id] = "approved-first"

            replacement = _attempt(attempt_id="attempt-2", fencing_token=2)
            replacement.replay_mode = True
            second = await run(replacement, first_id)
            second_interrupt = cast(list[dict[str, Any]], second["interrupts"])[0]
            second_id = cast(str, second_interrupt["id"])
            assert second_id != first_id
            client.resolved_values[second_id] = "approved-second"

            final_attempt = _attempt(attempt_id="attempt-3", fencing_token=3)
            final_attempt.replay_mode = True
            completed = await run(final_attempt, second_id)
        finally:
            current_oe_url.reset(url_token)

        if method == "invoke":
            assert completed["status"] == "completed"
        else:
            assert "response" in completed
        suspension_commits = [
            command for command in client.commits if command.suspensions
        ]
        assert len(suspension_commits) == 5
        suspension_ordinals = {command.step_ordinal for command in suspension_commits}
        assert len(suspension_ordinals) == 1
        positions = [
            command.suspensions[0].position.SerializeToString(deterministic=True)
            for command in suspension_commits
        ]
        assert positions == [
            positions[0],
            positions[0],
            positions[2],
            positions[0],
            positions[2],
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    @pytest.mark.parametrize("compiled_child", [False, True])
    @pytest.mark.parametrize("second_interrupt", [False, True])
    async def test_local_tool_native_interrupt_stays_started_then_completes_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        compiled_child: bool,
        second_interrupt: bool,
    ) -> None:
        from langchain_core.messages import AIMessage, BaseMessage
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as interrupts_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )
        from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver
        from agent_engine_sdk_langgraph.runtime import App
        from agent_engine_runner_shared import RuntimeMode, hooks
        from agent_engine_runner_shared.context import (
            clear_execution_context,
            set_execution_context,
        )
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_COMPLETED,
            ActivityContext,
            ActivityOutcome,
        )
        from agent_engine_runner_shared.models import ToolExecuteResponse
        from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper
        from agent_engine_runner_shared.workflow.client import (
            ActivityDispatch,
            ActivityReplay,
        )
        from agent_engine_runner_shared.workflow.protojson import proto_value_to_json

        class ToolActivities:
            def __init__(self) -> None:
                self.positions: dict[str, bytes] = {}
                self.recorded: dict[bytes, ActivityOutcome] = {}
                self.outcome_kinds: list[int] = []
                self.commands: list[Any] = []

            def start_activity(self, command: Any) -> Any:
                self.commands.append(command)
                position = command.position.SerializeToString(deterministic=True)
                if position in self.recorded:
                    return ActivityReplay(outcome=self.recorded[position])
                activity_id = f"activity-{command.position.activity_ordinal}"
                self.positions[activity_id] = position
                return ActivityDispatch(
                    context=ActivityContext(
                        workflow_identity=command.workflow_identity,
                        activity_id=activity_id,
                        attempt_id=command.attempt_id,
                        fencing_token=command.fencing_token,
                    )
                )

            def resolve(self, activity_id: str, value: str) -> None:
                # ResolveActivities runs before OE dispatches the replacement.
                # StartActivity then replays the completed answer for the exact
                # activity selected by the replacement request's resume map.
                outcome = ActivityOutcome(
                    activity_id=activity_id,
                    attempt_id="attempt-2",
                    fencing_token=2,
                    outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
                )
                outcome.result.string_value = value
                self.recorded[self.positions[activity_id]] = outcome

            def report_outcome(self, outcome: ActivityOutcome) -> None:
                self.outcome_kinds.append(outcome.outcome_kind)
                position = self.positions[outcome.activity_id]
                recorded = ActivityOutcome()
                recorded.CopyFrom(outcome)
                self.recorded[position] = recorded

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(interrupts_module, "AsyncWorkflowClient", lambda _: client)

        app = App(app_name="durable local interrupt")
        app._runtime.mode = RuntimeMode.AER
        app._register_hooks()
        tool_calls: list[str] = []

        @app.tool(is_local=True)
        def review_claim(claim_id: str) -> Any:
            """Pause for a native durable decision."""
            tool_calls.append(claim_id)
            decision = interrupt({"claim_id": claim_id, "question": "approve?"})
            if second_interrupt:
                return interrupt({"claim_id": claim_id, "question": "again?"})
            return decision

        wrapped_tool = app.get_tools()[0]
        tool_activities = ToolActivities()
        wrapper = SecureToolWrapper("http://oe", "execution-1")
        wrapper._workflow = tool_activities

        def node(state: Any) -> Any:
            result = wrapped_tool.func(claim_id="claim-1", tool_call_id="call-1")
            value = result[0] if isinstance(result, tuple) else result
            content = value.content if isinstance(value, BaseMessage) else value
            return {"messages": [AIMessage(content=f"decision:{content}")]}

        child: StateGraph[Any] = StateGraph(MessagesState)
        child.add_node("node", node)
        child.add_edge(START, "node")
        graph: StateGraph[Any] = StateGraph(MessagesState)
        if compiled_child:
            graph.add_node("review", child.compile())
            graph.add_edge(START, "review")
        else:
            graph.add_node("node", node)
            graph.add_edge(START, "node")
        compiled = graph.compile(
            checkpointer=PlatformCheckpointer(native=InMemorySaver())
        )
        resolver = DurableSubgraphResolver(compiled) if compiled_child else None
        agent = LangGraphBaseAgent(
            compiled,
            durable_subgraphs=resolver,
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        approval = ToolExecuteResponse(
            proceed=True,
            route_to="callback",
            latest_step_number=1,
        )
        execution_tokens = set_execution_context(
            execution_id="execution-1", wrapper=wrapper, oe_url="http://oe"
        )

        async def run(run_ctx: RequestContext) -> tuple[str, dict[str, Any]]:
            agent_input = AgentInput(payload={"message": "hi"})
            if method == "invoke":
                output = await agent.invoke(run_ctx, agent_input)
                data = cast(dict[str, Any], output.response)
                return cast(str, data["status"]), data
            events = [event async for event in agent.stream(run_ctx, agent_input)]
            assert len(events) == 1
            return cast(str, events[0].event), cast(dict[str, Any], events[0].data)

        completed: tuple[str, dict[str, Any]] | None = None
        replayed: tuple[str, dict[str, Any]] | None = None
        url_token = current_oe_url.set("http://oe")
        try:
            with (
                patch(
                    "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
                    return_value=approval,
                ) as request_approval,
                patch(
                    "agent_engine_runner_shared.secure_wrapper.report_oe_result"
                ) as report_result,
            ):
                with attempt_context_scope(_attempt()):
                    suspended_event, suspended_response = await run(ctx)

                interrupts = cast(
                    list[dict[str, Any]], suspended_response["interrupts"]
                )
                interrupt_id = cast(str, interrupts[0]["id"])
                client.resolved_values[interrupt_id] = "approved"
                tool_activities.resolve(interrupt_id, "approved")
                replay = _attempt(attempt_id="attempt-2", fencing_token=2)
                replay.replay_mode = True
                with attempt_context_scope(replay):
                    resume_ctx = ctx.model_copy(
                        update={
                            "resume": True,
                            "resume_data": {interrupt_id: "ignored"},
                        }
                    )
                    if second_interrupt:
                        with pytest.raises(RuntimeError, match="cannot raise another"):
                            await run(resume_ctx)
                    else:
                        completed = await run(resume_ctx)

                if not second_interrupt:
                    final_replay = _attempt(attempt_id="attempt-3", fencing_token=3)
                    final_replay.replay_mode = True
                    with attempt_context_scope(final_replay):
                        replayed = await run(ctx)
        finally:
            current_oe_url.reset(url_token)
            clear_execution_context(execution_tokens)
            hooks.reset_hooks()

        assert suspended_event in {"suspended", "suspend"}
        assert tool_calls == ["claim-1", "claim-1", "claim-1"]
        if second_interrupt:
            assert len([commit for commit in client.commits if commit.suspensions]) == 2
            positions = [
                command.position.SerializeToString(deterministic=True)
                for command in tool_activities.commands
            ]
            assert positions
            assert len(set(positions)) == 1
            assert tool_activities.outcome_kinds == []
            assert request_approval.call_count == 1
            assert [call.kwargs["status"] for call in report_result.call_args_list] == [
                "interrupted"
            ]
            return

        assert completed is not None
        assert replayed is not None
        completed_event, completed_response = completed
        replayed_event, replayed_response = replayed
        assert completed_event in {"completed", "result"}
        assert replayed_event in {"completed", "result"}
        assert completed_response["response"] == "decision:approved"
        assert replayed_response["response"] == "decision:approved"
        assert completed_response["message_count"] == replayed_response["message_count"]
        assert completed_response["message_count"] >= 1
        positions = [
            command.position.SerializeToString(deterministic=True)
            for command in tool_activities.commands
        ]
        assert positions
        assert len(set(positions)) == 1
        operation_path = tool_activities.commands[0].position.operation_path.segments
        assert [(segment.name, segment.ordinal) for segment in operation_path] == (
            [("agent", 1), ("review", 1)] if compiled_child else [("agent", 1)]
        )
        # ResolveActivities already made the interrupted tool position
        # terminal; reconstruction must not report a second activity outcome.
        assert tool_activities.outcome_kinds == []
        # Reconstruction re-enters the captured local callback. Reusing the
        # tool-execute route would replay its earlier interrupted marker before
        # LangGraph can recreate interrupt().
        assert request_approval.call_count == 1
        assert [call.kwargs["status"] for call in report_result.call_args_list] == [
            "interrupted",
        ]
        assert [
            proto_value_to_json(outcome.result)
            for outcome in tool_activities.recorded.values()
        ] == ["approved"]

    @pytest.mark.anyio
    async def test_direct_interrupt_stream_suspends_durably(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from langgraph.graph import MessagesState
        from langgraph.types import interrupt
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import durable_session as durable_session_module
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )

        client = _InterruptWorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        monkeypatch.setattr(
            durable_session_module, "AsyncWorkflowClient", lambda _: client
        )
        saver = PlatformCheckpointer(native=InMemorySaver())
        graph: StateGraph[Any] = StateGraph(MessagesState)

        def node(state: Any) -> Any:
            decision = interrupt({"suspend_reason": "approve?"})
            return {"messages": [AIMessage(content=f"decision:{decision}")]}

        graph.add_node("node", node)
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(graph.compile(checkpointer=saver))
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(_attempt()):
                events = [
                    event
                    async for event in agent.stream(
                        ctx, AgentInput(payload={"message": "hi"})
                    )
                ]
            assert events[-1].event == "suspend"
            assert isinstance(events[-1].data, dict)
            assert events[-1].data["interrupts"] == [
                {
                    "id": "activity-1",
                    "value": {"suspend_reason": "approve?"},
                }
            ]
        finally:
            current_oe_url.reset(url_token)

    @pytest.mark.anyio
    @pytest.mark.parametrize("method", ["invoke", "stream"])
    async def test_static_interrupt_pause_is_rejected_durably(
        self, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        from langgraph.graph import MessagesState
        from agent_engine_sdk import AgentInput
        from agent_engine_sdk_langgraph import (
            platform_checkpointer as checkpointer_module,
        )

        client = _WorkflowClient()
        monkeypatch.setattr(
            checkpointer_module, "AsyncWorkflowClient", lambda _: client
        )
        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node(
            "node", lambda state: {"messages": [AIMessage(content="unreachable")]}
        )
        graph.add_edge(START, "node")
        agent = LangGraphBaseAgent(
            graph.compile(
                checkpointer=PlatformCheckpointer(native=InMemorySaver()),
                interrupt_before=["node"],
            )
        )
        ctx = RequestContext(user_id="user-1", session_id="session-1")

        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(_attempt()):
                with pytest.raises(
                    UnsupportedDurableGraphError,
                    match="interrupt_before or interrupt_after",
                ):
                    if method == "invoke":
                        await agent.invoke(ctx, AgentInput(payload={"message": "hi"}))
                    else:
                        async for _ in agent.stream(
                            ctx, AgentInput(payload={"message": "hi"})
                        ):
                            pass
        finally:
            current_oe_url.reset(url_token)


class TestStateSurvivingAStoppedTurn:
    """What a stopped turn leaves behind for the next one.

    A turn stopped part-way through is abandoned, not completed: the OE marks
    the run cancelled and the pod goes away without the graph ever reaching
    complete_execution. These tests pin what that leaves readable, because the
    failure mode is silent — a half-written superstep that the next turn treats
    as the conversation's real state reads as a plausible answer rather than as
    corruption.
    """

    @pytest.mark.anyio
    async def test_stopped_mid_superstep_commits_no_session_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An abandoned turn must not publish state for the next turn to read."""
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = _WorkflowClient()
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        saver = PlatformCheckpointer(native=InMemorySaver())
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": [HumanMessage("refund my order"), AIMessage("checking")],
            "__start__": {"message": "refund my order"},
        }
        checkpoint["channel_versions"] = {
            name: "1" for name in checkpoint["channel_values"]
        }
        attempt = _attempt()
        url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await saver.aput(
                    _config("ignored"),
                    checkpoint,
                    {"source": "loop", "step": 0},
                    checkpoint["channel_versions"],
                )
                # The stop lands here: the pod is killed, so complete_execution
                # is never reached for this attempt.
        finally:
            current_oe_url.reset(url_token)

        assert client.completions == [], (
            "a stopped turn must not commit session state; the next turn would "
            "otherwise start from a superstep the user cancelled"
        )
        assert len(client.commits) == 1, (
            "the step that did run is still recorded, so the trace can show how "
            "far the turn got before it was stopped"
        )

    def test_stopped_attempts_scratch_is_discarded(self) -> None:
        """Releasing a stopped attempt's scratch leaves nothing to answer from."""
        saver = PlatformCheckpointer(native=InMemorySaver())
        stopped = _attempt(attempt_id="attempt-stopped")

        with attempt_context_scope(stopped):
            _put(saver, "ignored", "half-written")
            assert _get_value(saver, "ignored") == "half-written"

        saver.release_scratch(stopped)

        with attempt_context_scope(stopped):
            assert _get_value(saver, "ignored") is None, (
                "a stopped run's partial state must not remain readable"
            )

    def test_a_later_turn_cannot_read_the_stopped_turns_partial_state(self) -> None:
        """The next turn is fenced off from the stopped one's scratch.

        This is the half that matters for a stopped run: the user sends another
        message in the same session, and that new turn must not inherit the
        superstep they cancelled.
        """
        saver = PlatformCheckpointer(native=InMemorySaver())
        stopped = _attempt(attempt_id="attempt-stopped", fencing_token=1)
        next_turn = _attempt(attempt_id="attempt-next", fencing_token=2)

        with attempt_context_scope(stopped):
            _put(saver, "ignored", "cancelled-superstep")

        with attempt_context_scope(next_turn):
            assert _get_value(saver, "ignored") is None, (
                "the next turn must not see the stopped turn's partial state"
            )
            _put(saver, "ignored", "fresh-turn")
            assert _get_value(saver, "ignored") == "fresh-turn"

    def test_native_session_state_outlives_a_stopped_durable_turn(self) -> None:
        """A stop must not damage the conversation the session already had.

        The previously committed state lives in the native saver. A stopped
        durable turn writes only to attempt-local scratch, so discarding it
        leaves the real conversation intact and the session reusable.
        """
        native = InMemorySaver()
        saver = PlatformCheckpointer(native=native)
        stopped = _attempt(attempt_id="attempt-stopped")

        _put(saver, "session-thread", "committed-conversation")
        with attempt_context_scope(stopped):
            _put(saver, "session-thread", "abandoned-turn")
        saver.release_scratch(stopped)

        assert _get_value(saver, "session-thread") == "committed-conversation"
        assert _get_value(native, "session-thread") == "committed-conversation", (
            "stopping a turn must leave the session's existing state usable"
        )


class _FakeCheckpointCol:
    """Minimal stand-in for MongoDBSaver.checkpoint_collection."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def distinct(self, key: str, flt: dict[str, Any]) -> list[Any]:
        return sorted({d[key] for d in self.docs if d["thread_id"] == flt["thread_id"]})

    def find_one(
        self, flt: dict[str, Any], sort: Any = None, projection: Any = None
    ) -> Any:
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in flt.items())]
        if not matches:
            return None
        return max(matches, key=lambda d: d["checkpoint_id"])


class _FakeWritesCol:
    """Minimal stand-in for MongoDBSaver.writes_collection."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def delete_many(self, flt: dict[str, Any]) -> Any:
        before = len(self.docs)
        self.docs[:] = [
            d for d in self.docs if not all(d.get(k) == v for k, v in flt.items())
        ]
        result = MagicMock()
        result.deleted_count = before - len(self.docs)
        return result


class _FakeMongoSaver:
    """Duck-typed MongoDBSaver: just the two collections the fence reads."""

    def __init__(
        self, ckpt_docs: list[dict[str, Any]], writes_docs: list[dict[str, Any]]
    ) -> None:
        self.checkpoint_collection = _FakeCheckpointCol(ckpt_docs)
        self.writes_collection = _FakeWritesCol(writes_docs)


class TestFenceCancelledPredecessorWrites:
    """A cancelled run's partial superstep writes must not leak into the next
    turn; everything else (older checkpoints, other threads, resumes) must be
    left untouched."""

    def _agent(
        self, ckpt_docs: list[dict[str, Any]], writes_docs: list[dict[str, Any]]
    ) -> Any:
        # Compile-shaped like production: the graph's checkpointer is the
        # PlatformCheckpointer wrapper, with the Mongo saver in `.native`.
        graph = MagicMock()
        graph.checkpointer = PlatformCheckpointer(
            native=cast(Any, _FakeMongoSaver(ckpt_docs, writes_docs))
        )
        return LangGraphBaseAgent(graph)

    @pytest.mark.anyio
    async def test_prunes_only_latest_checkpoint_writes_per_namespace(self) -> None:
        thread = "sess-1:ws-1"
        ckpts = [
            {"thread_id": thread, "checkpoint_ns": "", "checkpoint_id": "c1"},
            {"thread_id": thread, "checkpoint_ns": "", "checkpoint_id": "c2"},
            {"thread_id": thread, "checkpoint_ns": "sub:x", "checkpoint_id": "s1"},
        ]
        writes = [
            # Older checkpoint: a still-suspended predecessor's interrupt
            # writes live here — must survive.
            {
                "thread_id": thread,
                "checkpoint_ns": "",
                "checkpoint_id": "c1",
                "task_id": "old",
            },
            # Latest root and subgraph checkpoints: the killed superstep — pruned.
            {
                "thread_id": thread,
                "checkpoint_ns": "",
                "checkpoint_id": "c2",
                "task_id": "dead",
            },
            {
                "thread_id": thread,
                "checkpoint_ns": "sub:x",
                "checkpoint_id": "s1",
                "task_id": "dead-sub",
            },
            # Foreign thread — must survive.
            {
                "thread_id": "other",
                "checkpoint_ns": "",
                "checkpoint_id": "c2",
                "task_id": "foreign",
            },
        ]
        agent = self._agent(ckpts, writes)
        ctx = RequestContext(
            session_id="sess-1", workspace_id="ws-1", previous_execution_cancelled=True
        )

        await agent._execution_session().fence_cancelled_predecessor_writes(ctx)

        writes = agent._graph.checkpointer.native.writes_collection
        remaining = {d["task_id"] for d in writes.docs}
        assert remaining == {"old", "foreign"}

    @pytest.mark.anyio
    async def test_noop_without_flag_or_on_resume(self) -> None:
        thread = "sess-1:ws-1"
        ckpts = [{"thread_id": thread, "checkpoint_ns": "", "checkpoint_id": "c1"}]
        writes = [
            {
                "thread_id": thread,
                "checkpoint_ns": "",
                "checkpoint_id": "c1",
                "task_id": "w",
            }
        ]

        agent = self._agent(ckpts, list(writes))
        await agent._execution_session().fence_cancelled_predecessor_writes(
            RequestContext(session_id="sess-1", workspace_id="ws-1")
        )
        assert len(agent._graph.checkpointer.native.writes_collection.docs) == 1

        agent = self._agent(ckpts, list(writes))
        await agent._execution_session().fence_cancelled_predecessor_writes(
            RequestContext(
                session_id="sess-1",
                workspace_id="ws-1",
                previous_execution_cancelled=True,
                resume=True,
                resume_data={},
            )
        )
        assert len(agent._graph.checkpointer.native.writes_collection.docs) == 1

    @pytest.mark.anyio
    async def test_noop_when_checkpointer_is_not_mongo(self) -> None:
        graph = MagicMock()
        graph.checkpointer = None
        agent = LangGraphBaseAgent(graph)
        # Must simply return; no attribute errors on a saver-less graph.
        await agent._execution_session().fence_cancelled_predecessor_writes(
            RequestContext(
                session_id="sess-1",
                workspace_id="ws-1",
                previous_execution_cancelled=True,
            )
        )

    @pytest.mark.anyio
    async def test_noop_when_platform_checkpointer_has_no_native_saver(self) -> None:
        # Stateless local dev: PlatformCheckpointer with no Mongo leg.
        graph = MagicMock()
        graph.checkpointer = PlatformCheckpointer(native=None)
        agent = LangGraphBaseAgent(graph)
        await agent._execution_session().fence_cancelled_predecessor_writes(
            RequestContext(
                session_id="sess-1",
                workspace_id="ws-1",
                previous_execution_cancelled=True,
            )
        )

    @pytest.mark.anyio
    async def test_warns_when_saver_shape_is_unrecognized(self, caplog: Any) -> None:
        # A saver that exposes no Mongo collections must disable the fence
        # loudly, not silently — this is how a wrapper change gets noticed.
        graph = MagicMock()
        graph.checkpointer = object()
        agent = LangGraphBaseAgent(graph)
        with caplog.at_level("WARNING"):
            await agent._execution_session().fence_cancelled_predecessor_writes(
                RequestContext(
                    session_id="sess-1",
                    workspace_id="ws-1",
                    previous_execution_cancelled=True,
                )
            )
        assert any("no Mongo collections" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_prune_failure_never_fails_the_invoke(self) -> None:
        graph = MagicMock()
        saver = MagicMock()
        saver.checkpoint_collection = MagicMock()
        saver.checkpoint_collection.distinct.side_effect = RuntimeError("mongo down")
        saver.writes_collection = MagicMock()
        graph.checkpointer = saver
        agent = LangGraphBaseAgent(graph)

        await agent._execution_session().fence_cancelled_predecessor_writes(
            RequestContext(
                session_id="sess-1",
                workspace_id="ws-1",
                previous_execution_cancelled=True,
            )
        )
