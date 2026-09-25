"""Tests for LangGraphBaseAgent adapter."""

from __future__ import annotations

import asyncio
import math
import operator
from collections import OrderedDict
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Overwrite, interrupt
from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    RequestContext,
    SessionForkResponse,
)
from pydantic import BaseModel, JsonValue

from agent_engine_sdk_langgraph import LangGraphOutputParser, RawStreamItem
from agent_engine_sdk_langgraph.agent import InternalStreamError, LangGraphBaseAgent
from agent_engine_sdk_langgraph.checkpoint_branch import (
    LangGraphCheckpoint,
    copy_checkpoint,
    latest_checkpoint_id,
    normalize_checkpoint_id,
)
from agent_engine_sdk_langgraph.execution_session import ExecutionSession
from agent_engine_sdk_langgraph.messages import find_last_ai_content, normalize_content
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer
from agent_engine_sdk_langgraph.session_fork import (
    LangGraphForkPlugin,
    create_durable_branch,
)
from agent_engine_sdk_langgraph.workflow_state import state_snapshot_to_channel_values
from agent_engine_runner_shared.context import current_oe_url
from agent_engine_runner_shared.custom_events import emit_custom_event
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    StateSnapshot as WorkflowStateSnapshot,
)
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT
from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    DurableActivitySuspended,
    attempt_context_scope,
    current_operation_path,
)


class TestNormalizeContent:
    """Tests for normalize_content helper."""

    def test_string_content(self) -> None:
        assert normalize_content("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert normalize_content("") == ""

    def test_list_with_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
        assert normalize_content(content) == "hello world"

    def test_list_with_mixed_types(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": " world"},
        ]
        assert normalize_content(content) == "hello world"

    def test_list_with_strings(self) -> None:
        content = ["hello", " ", "world"]
        assert normalize_content(content) == "hello world"

    def test_none_content(self) -> None:
        assert normalize_content(None) == ""

    def test_other_type(self) -> None:
        assert normalize_content(123) == "123"


class TestNormalizeCheckpointId:
    """Tests for normalize_checkpoint_id helper."""

    def test_none_checkpoint_id(self) -> None:
        assert normalize_checkpoint_id(None) is None

    def test_string_none_checkpoint_id(self) -> None:
        assert normalize_checkpoint_id("None") is None

    def test_empty_checkpoint_id(self) -> None:
        assert normalize_checkpoint_id("  ") is None

    def test_valid_checkpoint_id(self) -> None:
        assert normalize_checkpoint_id("ckpt-123") == "ckpt-123"


class TestLangGraphCheckpointBranching:
    @staticmethod
    def _agent(
        resolve_thread_id: Any | None = None,
        checkpointer: Any | None = None,
    ) -> LangGraphBaseAgent:
        def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
            human = next(
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            )
            return {"messages": [AIMessage(content=f"reply:{human.content}")]}

        builder = StateGraph(MessagesState)
        builder.add_node("respond", respond)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", END)
        return LangGraphBaseAgent(
            builder.compile(
                checkpointer=(InMemorySaver() if checkpointer is None else checkpointer)
            ),
            resolve_thread_id=resolve_thread_id,
        )

    @staticmethod
    def _stub_workflow_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        from agent_engine_sdk_langgraph import platform_checkpointer as module

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.finalize_step = AsyncMock(return_value=[])
        monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
        return client

    @staticmethod
    def _interrupting_agent() -> LangGraphBaseAgent:
        def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
            human = next(
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            )
            if human.content == "pause":
                decision = interrupt({"question": "continue?"})
                return {"messages": [AIMessage(content=f"reply:{decision}")]}
            return {"messages": [AIMessage(content=f"reply:{human.content}")]}

        builder = StateGraph(MessagesState)
        builder.add_node("respond", respond)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", END)
        return LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))

    @pytest.mark.anyio
    async def test_first_invoke_copies_selected_checkpoint_to_new_thread(self) -> None:
        agent = self._agent()
        source_ctx = RequestContext(session_id="source", workspace_id="workspace")
        first = await agent.invoke(
            source_ctx,
            AgentInput(payload={"message": "one"}),
        )
        first_response = cast(dict[str, Any], first.response)
        source = LangGraphCheckpoint.model_validate(
            cast(dict[str, Any], first_response["metadata"])["langgraph_checkpoint"]
        )

        await agent.invoke(source_ctx, AgentInput(payload={"message": "two"}))

        branch_ctx = RequestContext(
            session_id="branch",
            workspace_id="workspace",
            metadata={"langgraph_branch_point": source.model_dump()},
        )
        branch = await agent.invoke(
            branch_ctx,
            AgentInput(payload={"message": "branch"}),
        )
        branch_response = cast(dict[str, Any], branch.response)
        branch_checkpoint = LangGraphCheckpoint.model_validate(
            cast(dict[str, Any], branch_response["metadata"])["langgraph_checkpoint"]
        )

        assert branch_checkpoint.thread_id == "branch:workspace"
        branch_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "branch:workspace"}}
        )
        assert [message.content for message in branch_state.values["messages"]] == [
            "one",
            "reply:one",
            "branch",
            "reply:branch",
        ]
        source_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "source:workspace"}}
        )
        assert [message.content for message in source_state.values["messages"]] == [
            "one",
            "reply:one",
            "two",
            "reply:two",
        ]

    @pytest.mark.anyio
    async def test_later_dest_turn_does_not_recopy(self) -> None:
        agent = self._agent()
        source_ctx = RequestContext(session_id="source", workspace_id="workspace")
        first = await agent.invoke(source_ctx, AgentInput(payload={"message": "one"}))
        source = LangGraphCheckpoint.model_validate(
            cast(
                dict[str, Any],
                cast(dict[str, Any], first.response)["metadata"],
            )["langgraph_checkpoint"]
        )
        branch_ctx = RequestContext(
            session_id="branch",
            workspace_id="workspace",
            metadata={"langgraph_branch_point": source.model_dump()},
        )
        await agent.invoke(branch_ctx, AgentInput(payload={"message": "branch"}))
        later = await agent.invoke(
            branch_ctx,
            AgentInput(payload={"message": "again"}),
        )
        assert "reply:again" in str(later.response)
        branch_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "branch:workspace"}}
        )
        assert [message.content for message in branch_state.values["messages"]] == [
            "one",
            "reply:one",
            "branch",
            "reply:branch",
            "again",
            "reply:again",
        ]

    @pytest.mark.anyio
    async def test_resume_does_not_reinitialize_branch(self) -> None:
        agent = self._interrupting_agent()

        source_result = await agent.invoke(
            RequestContext(session_id="source", workspace_id="workspace"),
            AgentInput(payload={"message": "source"}),
        )
        source = LangGraphCheckpoint.model_validate(
            cast(
                dict[str, Any],
                cast(dict[str, Any], source_result.response)["metadata"],
            )["langgraph_checkpoint"]
        )
        metadata = {"langgraph_branch_point": source.model_dump()}
        suspended = await agent.invoke(
            RequestContext(
                session_id="branch",
                workspace_id="workspace",
                metadata=metadata,
            ),
            AgentInput(payload={"message": "pause"}),
        )
        assert cast(dict[str, Any], suspended.response)["status"] == "suspended"

        resumed = await agent.invoke(
            RequestContext(
                session_id="branch",
                workspace_id="workspace",
                metadata=metadata,
                resume=True,
                resume_data="approved",
            ),
            AgentInput(payload={"message": ""}),
        )
        response = cast(dict[str, Any], resumed.response)
        assert response["status"] == "completed"
        assert response["response"] == "reply:approved"

    @pytest.mark.anyio
    async def test_copies_completed_source_with_historical_interrupt(self) -> None:
        agent = self._interrupting_agent()
        source_ctx = RequestContext(session_id="source", workspace_id="workspace")
        suspended = await agent.invoke(
            source_ctx, AgentInput(payload={"message": "pause"})
        )
        assert cast(dict[str, Any], suspended.response)["status"] == "suspended"
        completed = await agent.invoke(
            source_ctx.model_copy(update={"resume": True, "resume_data": "approved"}),
            AgentInput(payload={"message": ""}),
        )
        source = LangGraphCheckpoint.model_validate(
            cast(
                dict[str, Any],
                cast(dict[str, Any], completed.response)["metadata"],
            )["langgraph_checkpoint"]
        )

        branch = await agent.invoke(
            RequestContext(
                session_id="branch",
                workspace_id="workspace",
                metadata={"langgraph_branch_point": source.model_dump()},
            ),
            AgentInput(payload={"message": "branch"}),
        )

        assert cast(dict[str, Any], branch.response)["status"] == "completed"
        branch_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "branch:workspace"}}
        )
        assert [message.content for message in branch_state.values["messages"]] == [
            "pause",
            "reply:approved",
            "branch",
            "reply:branch",
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("task_error", "task_state", "message"),
        [
            ("failed", None, "failed history"),
            (None, {"configurable": {}}, "persistent subgraph history"),
        ],
    )
    async def test_rejects_unsupported_history_before_copy(
        self,
        task_error: str | None,
        task_state: dict[str, Any] | None,
        message: str,
    ) -> None:
        source = LangGraphCheckpoint(
            thread_id="source:workspace",
            checkpoint_id="source-checkpoint",
        )
        source_config = {
            "configurable": {
                "thread_id": source.thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": source.checkpoint_id,
            }
        }
        parent_config = {
            "configurable": {
                "thread_id": source.thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "parent-checkpoint",
            }
        }
        source_values = {"messages": ["source"]}
        selected = MagicMock(
            config=source_config,
            parent_config=parent_config,
            next=(),
            tasks=(),
            values=source_values,
        )
        task = MagicMock(
            error=task_error,
            state=task_state,
            interrupts=(),
            result=None,
            name="respond",
            id="task-1",
        )
        parent = MagicMock(
            config=parent_config,
            parent_config=None,
            next=(),
            tasks=(task,),
            values={"messages": ["parent"]},
        )
        saver = InMemorySaver()
        graph = MagicMock()
        graph.checkpointer = saver
        graph.aget_state = AsyncMock(side_effect=[selected, selected, parent])
        graph.abulk_update_state = AsyncMock()

        with pytest.raises(RuntimeError, match=message):
            await copy_checkpoint(
                graph,
                {"configurable": {"thread_id": "branch:workspace"}},
                source,
            )

        graph.abulk_update_state.assert_not_awaited()
        assert selected.values == source_values
        target_history = [
            checkpoint
            async for checkpoint in saver.alist(
                {"configurable": {"thread_id": "branch:workspace"}}
            )
        ]
        assert target_history == []

    def _config(self, agent: LangGraphBaseAgent, ctx: RequestContext) -> RunnableConfig:
        return agent._execution_session().build_config(ctx)

    @staticmethod
    def _stub_create_branch(monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )

    @pytest.mark.anyio
    async def test_copy_checkpoint_uses_selected_historical_completed_point(
        self,
    ) -> None:
        agent = self._agent()
        source_ctx = RequestContext(session_id="source", workspace_id="workspace")
        first = await agent.invoke(source_ctx, AgentInput(payload={"message": "one"}))
        first_checkpoint = LangGraphCheckpoint.model_validate(
            cast(dict[str, Any], cast(dict[str, Any], first.response)["metadata"])[
                "langgraph_checkpoint"
            ]
        )
        await agent.invoke(source_ctx, AgentInput(payload={"message": "two"}))

        await copy_checkpoint(
            agent._graph,
            {"configurable": {"thread_id": "branch:workspace"}},
            first_checkpoint,
        )
        branch_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "branch:workspace"}}
        )
        assert [message.content for message in branch_state.values["messages"]] == [
            "one",
            "reply:one",
        ]
        source_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "source:workspace"}}
        )
        assert [message.content for message in source_state.values["messages"]] == [
            "one",
            "reply:one",
            "two",
            "reply:two",
        ]

    @pytest.mark.anyio
    async def test_update_state_on_live_native_session_calls_fork_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))

        dest = await agent._graph.aupdate_state(self._config(agent, ctx), None)
        assert (dest.get("configurable") or {}).get("thread_id") == "branch:workspace"
        source_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "source:workspace"}}
        )
        dest_state = await agent._graph.aget_state(dest)
        assert [message.content for message in source_state.values["messages"]] == [
            "one",
            "reply:one",
        ]
        assert [message.content for message in dest_state.values["messages"]] == [
            "one",
            "reply:one",
        ]

    @pytest.mark.anyio
    async def test_update_state_on_empty_thread_is_not_intercepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal called
            called = True
            raise AssertionError("empty-thread update_state must not fork")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        agent = self._agent()
        dest = await agent._graph.aupdate_state(
            {"configurable": {"thread_id": "fresh:workspace"}},
            {"messages": [HumanMessage(content="seed")]},
            as_node="respond",
        )
        assert not called
        assert (dest.get("configurable") or {}).get("thread_id") == "fresh:workspace"

    @pytest.mark.anyio
    async def test_update_state_values_patch_only_destination(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))

        dest = await agent._graph.aupdate_state(
            self._config(agent, ctx),
            {"messages": [AIMessage(content="patched")]},
        )
        source_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "source:workspace"}}
        )
        dest_state = await agent._graph.aget_state(dest)
        assert [message.content for message in source_state.values["messages"]] == [
            "one",
            "reply:one",
        ]
        assert [message.content for message in dest_state.values["messages"]][-1] == (
            "patched"
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("replay_mode", [False, True])
    async def test_durable_update_state_creates_branch_from_committed_scratch(
        self, monkeypatch: pytest.MonkeyPatch, replay_mode: bool
    ) -> None:
        created: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> SessionForkResponse:
            created.append(kwargs)
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", fake_create
        )
        client = self._stub_workflow_client(monkeypatch)
        agent = self._agent(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            replay_mode=replay_mode,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="trusted-org",
                    project_id="trusted-project",
                    workspace_id="workspace",
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )
        config = self._config(agent, ctx)

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                fork_config: RunnableConfig = {
                    **config,
                    "callbacks": [BaseCallbackHandler()],
                    "metadata": {"request_id": "source-request"},
                    "tags": ["source"],
                }
                dest = await agent._graph.aupdate_state(fork_config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        assert len(created) == 1
        assert [
            call.args[0].step_ordinal for call in client.finalize_step.await_args_list
        ] == [1, 2]
        request = created[0]
        assert request["org_id"] == "trusted-org"
        assert request["project_id"] == "trusted-project"
        assert request["workspace_id"] == "workspace"
        assert request["session_id"] == "source"
        assert request["execution_id"] == "exec-1"
        assert request["step_ordinal"] == 2
        assert [
            message.content.string_value for message in request["state"].messages
        ] == [
            "one",
            "reply:one",
        ]
        assert request["state"].message_encoding_version == 1
        assert all(
            message.HasField("source_message") is replay_mode
            for message in request["state"].messages
        )
        assert dest == {
            "configurable": {
                "thread_id": "branch:workspace",
                "checkpoint_ns": "",
            }
        }

    @pytest.mark.anyio
    async def test_durable_fork_uses_raw_checkpoint_without_reducer_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> SessionForkResponse:
            created.append(kwargs)
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        class State(MessagesState, total=False):
            items: Annotated[dict[str, str], operator.or_]

        def respond(state: State) -> dict[str, list[AIMessage]]:
            human = next(
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            )
            return {"messages": [AIMessage(content=f"reply:{human.content}")]}

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", fake_create
        )
        self._stub_workflow_client(monkeypatch)
        checkpointer = PlatformCheckpointer(native=InMemorySaver())
        builder = StateGraph(State)
        builder.add_node("respond", respond)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", END)
        agent = LangGraphBaseAgent(builder.compile(checkpointer=checkpointer))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                config = self._config(agent, ctx)
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                raw = await checkpointer.aget_tuple(config)
                assert raw is not None
                assert "items" not in raw.checkpoint["channel_values"]
                materialized = await agent._graph.aget_state(config)
                assert materialized.values["items"] == {}
                await agent._graph.aupdate_state(config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        assert len(created) == 1
        branch_values = state_snapshot_to_channel_values(created[0]["state"])
        assert "items" not in branch_values

    @pytest.mark.anyio
    async def test_durable_update_state_can_select_committed_scratch_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> SessionForkResponse:
            created.append(kwargs)
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", fake_create
        )
        self._stub_workflow_client(monkeypatch)
        agent = self._agent(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        config = self._config(agent, ctx)
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                history = [
                    item async for item in agent._graph.aget_state_history(config)
                ]
                selected = next(
                    item for item in history if (item.metadata or {}).get("step") == 0
                )
                selected_config: RunnableConfig = {
                    **selected.config,
                    "configurable": {
                        **dict(selected.config.get("configurable") or {}),
                        "request_context": ctx,
                    },
                }
                dest = await agent._graph.aupdate_state(selected_config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        assert created[0]["step_ordinal"] == 1
        assert dest.get("configurable") == {
            "thread_id": "branch:workspace",
            "checkpoint_ns": "",
        }

    @pytest.mark.anyio
    async def test_durable_unknown_checkpoint_fails_before_branch_creation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = AsyncMock()
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", create
        )
        self._stub_workflow_client(monkeypatch)
        agent = self._agent(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        config = self._config(agent, ctx)
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                unknown_config: RunnableConfig = {
                    **config,
                    "configurable": {
                        **dict(config.get("configurable") or {}),
                        "checkpoint_id": "unknown",
                    },
                }
                with pytest.raises(RuntimeError, match="not a committed root step"):
                    await agent._graph.aupdate_state(unknown_config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_durable_branch_of_branch_preserves_the_source_cutoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[dict[str, Any]] = []

        async def fake_create(**kwargs: Any) -> SessionForkResponse:
            created.append(kwargs)
            return SessionForkResponse(session_id="branch-2", execution_id="exec-3")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", fake_create
        )
        self._stub_workflow_client(monkeypatch)
        agent = self._agent(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        ctx = RequestContext(
            session_id="branch-1",
            workspace_id="workspace",
            execution_id="exec-2",
        )
        config = self._config(agent, ctx)
        attempt = AttemptContext(
            attempt_id="attempt-2",
            fencing_token=2,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="branch-1",
                execution_id="exec-2",
            ),
            branch_lineage=BranchLineage(
                source_workflow_identity=WorkflowIdentity(
                    session_id="source", execution_id="exec-1"
                ),
                source_step_ordinal=4,
                source_state_hash="sha256:source",
            ),
        )

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                await agent._graph.aupdate_state(config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        assert created[0]["step_ordinal"] == 6

    @pytest.mark.anyio
    async def test_durable_update_state_patch_fails_before_branch_creation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = AsyncMock()
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", create
        )
        agent = self._agent(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        config = self._config(agent, ctx)
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        with attempt_context_scope(attempt):
            with pytest.raises(RuntimeError, match="does not support.*patch"):
                await agent._graph.aupdate_state(
                    config,
                    {"messages": [AIMessage(content="durable")]},
                )

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_durable_in_app_fork_rejects_custom_thread_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = AsyncMock(
            return_value=SessionForkResponse(session_id="branch", execution_id="exec-2")
        )
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", create
        )
        self._stub_workflow_client(monkeypatch)

        def resolve(ctx: RequestContext) -> str:
            return f"custom-{ctx.session_id}"

        agent = self._agent(
            resolve_thread_id=resolve,
            checkpointer=PlatformCheckpointer(native=InMemorySaver()),
        )
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        config = self._config(agent, ctx)
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        oe_url_token = current_oe_url.set("http://oe")
        try:
            with attempt_context_scope(attempt):
                await agent._graph.ainvoke(
                    {"messages": [HumanMessage(content="one")]}, config
                )
                with pytest.raises(RuntimeError, match="default thread_id formula"):
                    await agent._graph.aupdate_state(config, None)
        finally:
            current_oe_url.reset(oe_url_token)

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_durable_branch_request_uses_protojson_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: list[dict[str, Any]] = []

        async def fake_request(**kwargs: Any) -> SessionForkResponse:
            requests.append(kwargs)
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork._create_branch", fake_request
        )

        await create_durable_branch(
            org_id="org",
            project_id="project",
            workspace_id="workspace",
            session_id="source",
            execution_id="exec-1",
            branch_key="update-exec-1-step-2",
            step_ordinal=2,
            state=WorkflowStateSnapshot(properties={"phase": "selected"}),
        )

        assert requests[0]["body"] == {
            "branch_key": "update-exec-1-step-2",
            "step_ordinal": 2,
            "state": {"properties": {"phase": "selected"}},
        }

    @pytest.mark.anyio
    async def test_empty_agent_engine_invoke_on_forked_thread_calls_ainvoke_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        await agent._graph.aupdate_state(self._config(agent, ctx), None)
        seen: list[Any] = []
        original = agent._graph.ainvoke

        async def spy_ainvoke(value: Any, config: Any = None, **kwargs: Any) -> Any:
            seen.append(value)
            return await original(value, config, **kwargs)

        agent._graph.ainvoke = spy_ainvoke  # type: ignore[method-assign]
        branch_ctx = RequestContext(session_id="branch", workspace_id="workspace")
        await agent.invoke(branch_ctx, AgentInput(payload={"message": ""}))
        await agent.invoke(branch_ctx, AgentInput(payload={"message": ""}))
        assert seen[0] is None
        assert seen[1] is not None

    @pytest.mark.anyio
    async def test_aer_fork_plugin_uses_the_same_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(session_id="source", workspace_id="workspace")
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))

        plugin = LangGraphForkPlugin(
            get_agent=lambda: agent,
            workspace_id_resolver=lambda: "workspace",
        )
        identity = await plugin.fork_session(
            session_id="source",
            execution_id="exec-1",
            history_id=None,
        )
        assert identity.session_id == "branch"
        dest_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "branch:workspace"}}
        )
        assert [message.content for message in dest_state.values["messages"]] == [
            "one",
            "reply:one",
        ]

    @pytest.mark.anyio
    async def test_aer_fork_plugin_allows_custom_thread_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)

        def resolve(ctx: RequestContext) -> str:
            return f"custom-{ctx.session_id}"

        agent = self._agent(resolve_thread_id=resolve)
        ctx = RequestContext(session_id="source", workspace_id="workspace")
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        plugin = LangGraphForkPlugin(
            get_agent=lambda: agent,
            workspace_id_resolver=lambda: "workspace",
        )
        identity = await plugin.fork_session(
            session_id="source",
            execution_id="exec-1",
            history_id=None,
        )
        assert identity.session_id == "branch"
        dest_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "custom-branch"}}
        )
        assert [message.content for message in dest_state.values["messages"]] == [
            "one",
            "reply:one",
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("session_id", "execution_id", "workspace_id"),
        [
            ("other-session", "exec-1", "workspace"),
            ("source", "other-execution", "workspace"),
            ("source", "exec-1", "other-workspace"),
        ],
    )
    async def test_durable_aer_fork_plugin_rejects_foreign_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_id: str,
        execution_id: str,
        workspace_id: str,
    ) -> None:
        create = AsyncMock()
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", create
        )
        agent = self._agent()
        plugin = LangGraphForkPlugin(
            get_agent=lambda: agent,
            workspace_id_resolver=lambda: workspace_id,
        )
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        with attempt_context_scope(attempt):
            with pytest.raises(RuntimeError, match="does not match"):
                await plugin.fork_session(
                    session_id=session_id,
                    execution_id=execution_id,
                    history_id=None,
                )

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_durable_aer_fork_plugin_rejects_state_patch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        create = AsyncMock()
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_durable_branch", create
        )
        plugin = LangGraphForkPlugin(
            get_agent=self._agent,
            workspace_id_resolver=lambda: "workspace",
        )
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                tenant_scope=TenantScope(
                    org_id="org", project_id="project", workspace_id="workspace"
                ),
                session_id="source",
                execution_id="exec-1",
            ),
        )

        with attempt_context_scope(attempt):
            with pytest.raises(RuntimeError, match="does not support.*patch"):
                await plugin.fork_session(
                    session_id="source",
                    execution_id="exec-1",
                    history_id=None,
                    state={"messages": [AIMessage(content="patched")]},
                )

        create.assert_not_awaited()

    @pytest.mark.anyio
    async def test_sync_update_state_in_running_loop_raises_instead_of_patching_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        with pytest.raises(RuntimeError, match="running event loop"):
            agent._graph.update_state(self._config(agent, ctx), None)
        assert not created
        source_state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "source:workspace"}}
        )
        assert [message.content for message in source_state.values["messages"]] == [
            "one",
            "reply:one",
        ]

    @pytest.mark.anyio
    async def test_in_app_fork_rejects_custom_thread_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )

        def resolve(ctx: RequestContext) -> str:
            return f"custom-{ctx.session_id}"

        agent = self._agent(resolve_thread_id=resolve)
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        with pytest.raises(RuntimeError, match="default thread_id formula"):
            await agent._graph.aupdate_state(self._config(agent, ctx), None)
        assert not created

    @pytest.mark.anyio
    async def test_incomplete_history_does_not_create_a_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        agent = self._interrupting_agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        suspended = await agent.invoke(ctx, AgentInput(payload={"message": "pause"}))
        assert cast(dict[str, Any], suspended.response)["status"] == "suspended"
        with pytest.raises(RuntimeError, match="completed root checkpoint"):
            await agent._graph.aupdate_state(self._config(agent, ctx), None)
        assert not created

    @pytest.mark.anyio
    async def test_multi_node_patch_without_node_does_not_create_a_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )

        def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
            return {"messages": [AIMessage(content="reply")]}

        def summarize(state: MessagesState) -> dict[str, list[AIMessage]]:
            return {"messages": [AIMessage(content="summary")]}

        builder = StateGraph(MessagesState)
        builder.add_node("respond", respond)
        builder.add_node("summarize", summarize)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", "summarize")
        builder.add_edge("summarize", END)
        agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))

        with pytest.raises(RuntimeError, match="requires a destination node"):
            await agent._graph.aupdate_state(
                self._config(agent, ctx),
                {"messages": [AIMessage(content="patch")]},
            )
        assert not created

    @pytest.mark.anyio
    async def test_unknown_patch_node_does_not_create_a_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )

        def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
            return {"messages": [AIMessage(content="reply")]}

        def summarize(state: MessagesState) -> dict[str, list[AIMessage]]:
            return {"messages": [AIMessage(content="summary")]}

        builder = StateGraph(MessagesState)
        builder.add_node("respond", respond)
        builder.add_node("summarize", summarize)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", "summarize")
        builder.add_edge("summarize", END)
        agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))

        with pytest.raises(RuntimeError, match="is not a node in this graph"):
            await agent._graph.aupdate_state(
                self._config(agent, ctx),
                {"messages": [AIMessage(content="patch")]},
                "missing",
            )
        assert not created

    @pytest.mark.anyio
    async def test_second_fork_fails_when_destination_already_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        patch = {"messages": [AIMessage(content="patched")]}
        dest = await agent._graph.aupdate_state(self._config(agent, ctx), patch)
        assert (dest.get("configurable") or {}).get("thread_id") == "branch:workspace"
        with pytest.raises(RuntimeError, match="destination thread is not empty"):
            await agent._graph.aupdate_state(self._config(agent, ctx), patch)

    @pytest.mark.anyio
    async def test_update_state_forwards_positional_task_id_on_empty_thread(
        self,
    ) -> None:
        agent = self._agent()
        dest = await agent._graph.aupdate_state(
            {"configurable": {"thread_id": "fresh:workspace"}},
            {"messages": [HumanMessage(content="seed")]},
            "respond",
            "task-seed",
        )
        dest_state = await agent._graph.aget_state(dest)
        assert [message.content for message in dest_state.values["messages"]] == [
            "seed"
        ]

    @pytest.mark.anyio
    async def test_fork_branch_key_includes_task_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        keys: list[str] = []

        async def fake_create(**kwargs: Any) -> SessionForkResponse:
            keys.append(str(kwargs["branch_key"]))
            return SessionForkResponse(
                session_id=f"branch-{len(keys)}",
                execution_id=f"exec-{len(keys) + 1}",
            )

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        await agent._graph.aupdate_state(self._config(agent, ctx), None, None, "task-a")
        await agent._graph.aupdate_state(self._config(agent, ctx), None, None, "task-b")
        assert len(keys) == 2
        assert keys[0] != keys[1]

    @pytest.mark.anyio
    async def test_unsupported_ancestor_history_does_not_create_a_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        created = False

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            nonlocal created
            created = True
            return SessionForkResponse(session_id="branch", execution_id="exec-2")

        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("failed history")

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.checkpoint_replay_plan", boom
        )
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        with pytest.raises(RuntimeError, match="failed history"):
            await agent._graph.aupdate_state(self._config(agent, ctx), None)
        assert not created

    @pytest.mark.anyio
    async def test_nonempty_first_dest_turn_consumes_continue_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        self._stub_create_branch(monkeypatch)
        agent = self._agent()
        ctx = RequestContext(
            session_id="source",
            workspace_id="workspace",
            execution_id="exec-1",
        )
        await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
        await agent._graph.aupdate_state(self._config(agent, ctx), None)
        seen: list[Any] = []
        original = agent._graph.ainvoke

        async def spy_ainvoke(value: Any, config: Any = None, **kwargs: Any) -> Any:
            seen.append(value)
            return await original(value, config, **kwargs)

        agent._graph.ainvoke = spy_ainvoke  # type: ignore[method-assign]
        branch_ctx = RequestContext(session_id="branch", workspace_id="workspace")
        await agent.invoke(branch_ctx, AgentInput(payload={"message": "later"}))
        await agent.invoke(branch_ctx, AgentInput(payload={"message": ""}))
        assert seen[0] is not None
        assert seen[1] is not None

    @pytest.mark.anyio
    async def test_continue_marker_evicts_oldest_unused_dest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORG_ID", "org")
        monkeypatch.setenv("PROJECT_ID", "project")
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork._CONTINUE_MARK_LIMIT", 2
        )
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork._CONTINUE_WITHOUT_USER_MESSAGE",
            OrderedDict(),
        )
        dests: list[str] = []

        async def fake_create(**_kwargs: Any) -> SessionForkResponse:
            dests.append(f"branch-{len(dests) + 1}")
            return SessionForkResponse(
                session_id=dests[-1], execution_id=f"exec-{len(dests) + 1}"
            )

        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.session_fork.create_native_branch", fake_create
        )
        agent = self._agent()
        for index, session_id in enumerate(
            ("source-a", "source-b", "source-c"), start=1
        ):
            ctx = RequestContext(
                session_id=session_id,
                workspace_id="workspace",
                execution_id=f"exec-{index}",
            )
            await agent.invoke(ctx, AgentInput(payload={"message": "one"}))
            await agent._graph.aupdate_state(self._config(agent, ctx), None)

        seen: dict[str, Any] = {}
        original = agent._graph.ainvoke

        async def spy_ainvoke(value: Any, config: Any = None, **kwargs: Any) -> Any:
            thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
            if isinstance(thread_id, str):
                seen[thread_id] = value
            return await original(value, config, **kwargs)

        agent._graph.ainvoke = spy_ainvoke  # type: ignore[method-assign]
        for session_id in dests:
            await agent.invoke(
                RequestContext(session_id=session_id, workspace_id="workspace"),
                AgentInput(payload={"message": ""}),
            )
        assert seen["branch-1:workspace"] is not None
        assert seen["branch-2:workspace"] is None
        assert seen["branch-3:workspace"] is None


class TestFindLastAIContent:
    """Tests for find_last_ai_content helper."""

    def test_single_ai_message(self) -> None:
        messages = [AIMessage(content="hello")]
        assert find_last_ai_content(messages) == "hello"

    def test_multiple_messages(self) -> None:
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="first response"),
            HumanMessage(content="thanks"),
            AIMessage(content="second response"),
        ]
        assert find_last_ai_content(messages) == "second response"

    def test_empty_ai_message_skipped(self) -> None:
        messages = [
            AIMessage(content="real response"),
            AIMessage(content=""),  # Empty - should be skipped
        ]
        assert find_last_ai_content(messages) == "real response"

    def test_no_ai_messages(self) -> None:
        messages = [HumanMessage(content="hi")]
        assert find_last_ai_content(messages) == ""

    def test_empty_list(self) -> None:
        assert find_last_ai_content([]) == ""


class TestLangGraphBaseAgent:
    """Tests for LangGraphBaseAgent."""

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        """Create a mock compiled graph."""
        graph = MagicMock()
        graph.checkpointer = None
        return graph

    @pytest.fixture
    def agent_input(self) -> AgentInput:
        """Create a sample AgentInput."""
        return AgentInput(
            payload={
                "message": "Hello, world!",
                "user_id": "user-123",
                "session_id": "session-456",
                "thread_id": "thread-789",
            },
        )

    def test_init_with_callbacks(self, mock_graph: MagicMock) -> None:
        """Test that callbacks are stored."""
        callbacks = [MagicMock(), MagicMock()]
        agent = LangGraphBaseAgent(mock_graph, callbacks=callbacks)
        assert agent._callbacks == callbacks

    def test_init_without_callbacks(self, mock_graph: MagicMock) -> None:
        """Test that callbacks default to empty list."""
        agent = LangGraphBaseAgent(mock_graph)
        assert agent._callbacks == []

    def test_build_graph_input_fresh(self, mock_graph: MagicMock) -> None:
        """Test building graph input for fresh execution."""
        agent = LangGraphBaseAgent(mock_graph)
        ctx = RequestContext(user_id="user-1", session_id="session-1")
        input = AgentInput(payload={"message": "Hello"})

        result = agent._execution_session().build_graph_input(ctx, input)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "Hello"
        assert result["messages"][0].id is None
        assert result["user_id"] == "user-1"
        assert result["session_id"] == "session-1"

    def test_build_graph_input_resume_via_ctx(self, mock_graph: MagicMock) -> None:
        """Test building graph input for resume flow via ctx."""
        agent = LangGraphBaseAgent(mock_graph)
        ctx = RequestContext(resume=True, resume_data={"decision": "approved"})
        input = AgentInput(payload={"message": ""})

        result = agent._execution_session().build_graph_input(ctx, input)

        # Should be a Command object
        assert hasattr(result, "resume")
        assert result.resume == {"decision": "approved"}

    def test_durable_resume_replays_from_original_turn_input(
        self, mock_graph: MagicMock
    ) -> None:
        """A durable replay rebuilds the same original LangGraph input."""
        agent = LangGraphBaseAgent(mock_graph)
        fresh_ctx = RequestContext(session_id="session-1")
        resume_ctx = RequestContext(
            session_id="session-1",
            resume=True,
            resume_data={"human_review": {"decision": "approve"}},
        )
        input = AgentInput(payload={"message": "Review claim CLM-1"})

        with attempt_context_scope(
            AttemptContext(
                attempt_id="attempt-1",
                replay_mode=False,
                workflow_identity=WorkflowIdentity(execution_id="execution-1"),
            )
        ):
            fresh = agent._execution_session().build_graph_input(fresh_ctx, input)

        with attempt_context_scope(
            AttemptContext(
                attempt_id="attempt-2",
                replay_mode=True,
                workflow_identity=WorkflowIdentity(execution_id="execution-1"),
            )
        ):
            replay = agent._execution_session().build_graph_input(resume_ctx, input)

        assert isinstance(fresh, dict)
        assert isinstance(replay, dict)
        assert fresh["messages"][0].content == "Review claim CLM-1"
        assert replay["messages"][0].content == "Review claim CLM-1"
        assert fresh["messages"][0].id == "durable-input:execution-1"
        assert replay["messages"][0].id == fresh["messages"][0].id

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_durable_replay_prepare_hook_rebuilds_stable_input(
        self, mock_graph: MagicMock, overwrite: bool
    ) -> None:
        """Replay reconstructs the same hook input and hides resume plumbing."""
        seen: list[dict[str, object]] = []

        def prepare(input: AgentInput, ctx: RequestContext) -> dict[str, object]:
            assert isinstance(input.payload, dict)
            seen.append(
                {
                    "resume": ctx.resume,
                    "resume_data": ctx.resume_data,
                    "metadata": ctx.metadata,
                    "execution_id": ctx.execution_id,
                    "headers": ctx.request_headers,
                }
            )
            messages = [
                ("system", "Review the claim carefully"),
                HumanMessage(content="Caller-owned", id="caller-message"),
                {"role": "user", "content": str(input.payload["message"])},
            ]
            return {
                "messages": Overwrite(messages) if overwrite else messages,
                "extra": ctx.session_id,
            }

        agent = LangGraphBaseAgent(mock_graph, prepare_input=prepare)
        fresh_ctx = RequestContext(
            execution_id="execution-1",
            session_id="session-1",
            request_headers={"x-correlation-id": "request-1"},
        )
        replay_ctx = RequestContext(
            execution_id="execution-1",
            session_id="session-1",
            request_headers={"x-correlation-id": "request-1"},
            resume=True,
            resume_data={"human_review": {"decision": "approve"}},
            metadata={"checkpoint_id": "native-only"},
        )
        input = AgentInput(payload={"message": "Review claim CLM-1"})

        with attempt_context_scope(
            AttemptContext(
                attempt_id="attempt-1",
                replay_mode=False,
                workflow_identity=WorkflowIdentity(execution_id="execution-1"),
            )
        ):
            fresh = agent._execution_session().build_graph_input(fresh_ctx, input)

        with attempt_context_scope(
            AttemptContext(
                attempt_id="attempt-2",
                replay_mode=True,
                workflow_identity=WorkflowIdentity(execution_id="execution-1"),
            )
        ):
            replay = agent._execution_session().build_graph_input(replay_ctx, input)

        expected_context = {
            "resume": False,
            "resume_data": None,
            "metadata": None,
            "execution_id": "execution-1",
            "headers": {"x-correlation-id": "request-1"},
        }
        assert seen == [expected_context, expected_context]
        assert isinstance(fresh, dict)
        assert isinstance(replay, dict)
        assert fresh["extra"] == "session-1"
        assert replay["extra"] == "session-1"
        assert isinstance(fresh["messages"], Overwrite) is overwrite
        assert isinstance(replay["messages"], Overwrite) is overwrite
        fresh_messages = fresh["messages"].value if overwrite else fresh["messages"]
        replay_messages = replay["messages"].value if overwrite else replay["messages"]
        assert [message.id for message in fresh_messages] == [
            "durable-input:execution-1:0",
            "caller-message",
            "durable-input:execution-1:2",
        ]
        assert [message.id for message in replay_messages] == [
            message.id for message in fresh_messages
        ]

    def test_prepare_input_hook_overrides_fresh_default(
        self, mock_graph: MagicMock
    ) -> None:
        """A registered prepare_input hook builds the fresh graph input."""

        def prepare(input: AgentInput, ctx: RequestContext) -> dict[str, object]:
            assert isinstance(input.payload, dict)
            return {
                "messages": [HumanMessage(content=str(input.payload["message"]))],
                "extra": input.payload.get("extra"),
                "uid": ctx.user_id,
            }

        agent = LangGraphBaseAgent(mock_graph, prepare_input=prepare)
        ctx = RequestContext(user_id="user-1", session_id="s-1")
        input = AgentInput(payload={"message": "Hi", "extra": {"k": "v"}})

        result = agent._execution_session().build_graph_input(ctx, input)

        assert result["extra"] == {"k": "v"}
        assert result["uid"] == "user-1"
        assert result["messages"][0].content == "Hi"
        assert result["messages"][0].id is None

    def test_prepare_input_hook_not_called_on_resume(
        self, mock_graph: MagicMock
    ) -> None:
        """Resume stays platform-managed; the hook must not run on resume."""

        def prepare(input: AgentInput, ctx: RequestContext) -> dict[str, object]:
            raise AssertionError("prepare_input must not be called on resume")

        agent = LangGraphBaseAgent(mock_graph, prepare_input=prepare)
        ctx = RequestContext(resume=True, resume_data={"decision": "approved"})
        input = AgentInput(payload={"message": ""})

        result = agent._execution_session().build_graph_input(ctx, input)

        assert hasattr(result, "resume")
        assert result.resume == {"decision": "approved"}

    def test_build_config(self, mock_graph: MagicMock) -> None:
        """Test building LangGraph config; session_id is the checkpoint thread."""
        agent = LangGraphBaseAgent(mock_graph)

        config = agent._execution_session().build_config(
            RequestContext(session_id="thread-123")
        )

        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "thread-123"

    def test_build_config_scopes_thread_id_to_workspace(
        self, mock_graph: MagicMock
    ) -> None:
        """thread_id is scoped by workspace so two agents in the same project
        DB cannot collide on a shared session_id."""
        agent = LangGraphBaseAgent(mock_graph)

        config = agent._execution_session().build_config(
            RequestContext(session_id="sess-1", workspace_id="ws-1")
        )

        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "sess-1:ws-1"

    def test_build_config_same_session_different_workspace_differs(
        self, mock_graph: MagicMock
    ) -> None:
        agent = LangGraphBaseAgent(mock_graph)

        a = agent._execution_session().build_config(
            RequestContext(session_id="sess-1", workspace_id="ws-a")
        )
        b = agent._execution_session().build_config(
            RequestContext(session_id="sess-1", workspace_id="ws-b")
        )

        assert "configurable" in a
        assert "configurable" in b
        assert a["configurable"]["thread_id"] != b["configurable"]["thread_id"]

    def test_build_config_resolve_thread_id_hook_overrides_default(
        self, mock_graph: MagicMock
    ) -> None:
        """A registered resolve_thread_id hook owns the checkpoint key verbatim."""

        def resolve(ctx: RequestContext) -> str:
            return f"{ctx.session_id}__{ctx.user_id}"

        agent = LangGraphBaseAgent(mock_graph, resolve_thread_id=resolve)
        config = agent._execution_session().build_config(
            RequestContext(
                session_id="sess-1",
                user_id="alice",
                workspace_id="ws-1",
            )
        )

        # Hook return is used as-is; workspace scoping must not be appended.
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "sess-1__alice"

    def test_build_config_resolve_thread_id_hook_used_on_resume(
        self, mock_graph: MagicMock
    ) -> None:
        """Resume must recompute the same agent-owned thread_id from ctx."""

        def resolve(ctx: RequestContext) -> str:
            return f"{ctx.session_id}__actor"

        agent = LangGraphBaseAgent(mock_graph, resolve_thread_id=resolve)
        config = agent._execution_session().build_config(
            RequestContext(
                session_id="sess-1",
                workspace_id="ws-1",
                resume=True,
                resume_data={"decision": "approved"},
            )
        )

        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "sess-1__actor"

    def test_build_config_resolve_thread_id_rejects_empty(
        self, mock_graph: MagicMock
    ) -> None:
        agent = LangGraphBaseAgent(mock_graph, resolve_thread_id=lambda ctx: "   ")
        with pytest.raises(ValueError, match="@app.resolve_thread_id"):
            agent._execution_session().build_config(RequestContext(session_id="sess-1"))

    def test_build_config_resolve_thread_id_rejects_non_string(
        self, mock_graph: MagicMock
    ) -> None:
        agent = LangGraphBaseAgent(
            mock_graph,
            resolve_thread_id=lambda ctx: 42,  # type: ignore[arg-type,return-value]
        )
        with pytest.raises(ValueError, match="@app.resolve_thread_id"):
            agent._execution_session().build_config(RequestContext(session_id="sess-1"))

    def test_constructor_positional_order_preserves_output_parser(
        self, mock_graph: MagicMock
    ) -> None:
        """Pre-change positional args still bind output_parser / use_custom_parser."""

        class Parser:
            pass

        def prepare(input, ctx):  # noqa: ANN001
            return {"messages": []}

        agent = LangGraphBaseAgent(
            mock_graph,
            None,
            prepare,
            Parser,  # type: ignore[arg-type]
            True,
        )

        assert agent._prepare_input is prepare
        assert agent._output_parser is Parser
        assert agent._use_custom_parser is True
        assert agent._resolve_thread_id is None

    def test_build_config_with_checkpoint_in_metadata(
        self, mock_graph: MagicMock
    ) -> None:
        """Test building config with checkpoint ID from ctx.metadata."""
        agent = LangGraphBaseAgent(mock_graph)
        ctx = RequestContext(
            session_id="thread-123", metadata={"checkpoint_id": "ckpt-456"}
        )

        config = agent._execution_session().build_config(ctx)

        assert config["configurable"]["checkpoint_id"] == "ckpt-456"  # type: ignore[typeddict-item]

    def test_build_config_ignores_none_checkpoint_value(
        self, mock_graph: MagicMock
    ) -> None:
        """None checkpoint metadata should not block latest-checkpoint lookup."""
        agent = LangGraphBaseAgent(mock_graph)
        ctx = RequestContext(session_id="thread-123", metadata={"checkpoint_id": None})

        config = agent._execution_session().build_config(ctx)

        assert "checkpoint_id" not in config["configurable"]  # type: ignore[operator]

    def test_build_config_includes_callbacks(self, mock_graph: MagicMock) -> None:
        """Test that callbacks are added to config."""
        callbacks = [MagicMock()]
        agent = LangGraphBaseAgent(mock_graph, callbacks=callbacks)

        config = agent._execution_session().build_config(RequestContext())

        assert "callbacks" in config
        assert config["callbacks"] == callbacks

    def test_build_config_injects_request_context(self, mock_graph: MagicMock) -> None:
        """Test that RequestContext is injected into configurable."""
        agent = LangGraphBaseAgent(mock_graph)
        ctx = RequestContext(execution_id="inv-1", user_id="u-1")

        config = agent._execution_session().build_config(ctx)

        assert config["configurable"]["request_context"] is ctx  # type: ignore[literal-required]

    @pytest.mark.anyio
    async def test_invoke(self, mock_graph: MagicMock, agent_input: AgentInput) -> None:
        """Test invoke method."""
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ]
            }
        )
        # Mock aget_state to indicate no pending interrupts
        mock_state = MagicMock()
        mock_state.next = ()
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(mock_graph)
        result = await agent.invoke(RequestContext(), agent_input)

        assert isinstance(result, AgentOutput)
        assert isinstance(result.response, dict)
        assert result.response["response"] == "Hi there!"
        assert result.response["status"] == "completed"
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.anyio
    async def test_invoke_fails_closed_for_custom_event(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Custom events have no delivery channel during non-streaming invoke."""

        async def mock_ainvoke(*args: Any, **kwargs: Any) -> Any:
            await emit_custom_event({"event": "step", "data": "not streamable"})
            return {"messages": [AIMessage(content="done")]}

        mock_graph.ainvoke = mock_ainvoke
        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(
            RuntimeError, match="only available during AER streaming execution"
        ):
            await agent.invoke(RequestContext(), agent_input)

    @pytest.mark.anyio
    async def test_durable_invoke_binds_nested_operation_resolver(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        resolver = MagicMock(
            return_value=(ChildOperationBoundary("case_investigation", "task-a"),)
        )
        agent = LangGraphBaseAgent(mock_graph, durable_subgraphs=resolver)
        seen: list[tuple[str, int]] = []

        async def invoke_inner(
            ctx: RequestContext,
            input: AgentInput,
            _session: ExecutionSession,
        ) -> AgentOutput:
            seen.extend(
                (segment.name, segment.ordinal)
                for segment in current_operation_path().segments
            )
            return AgentOutput(response={"status": "completed"})

        agent._invoke_inner = invoke_inner  # type: ignore[method-assign]
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            await agent.invoke(RequestContext(), agent_input)
            restored = current_operation_path()

        resolver.validate.assert_called_once_with()
        assert seen == [("agent", 1), ("case_investigation", 1)]
        assert [(segment.name, segment.ordinal) for segment in restored.segments] == [
            ("agent", 1)
        ]

    @pytest.mark.anyio
    async def test_native_invoke_does_not_bind_or_validate_nested_resolver(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        resolver = MagicMock(
            return_value=(ChildOperationBoundary("case_investigation", "task-a"),)
        )
        agent = LangGraphBaseAgent(mock_graph, durable_subgraphs=resolver)
        seen: list[tuple[str, int]] = []

        async def invoke_inner(
            ctx: RequestContext,
            input: AgentInput,
            _session: ExecutionSession,
        ) -> AgentOutput:
            seen.extend(
                (segment.name, segment.ordinal)
                for segment in current_operation_path().segments
            )
            return AgentOutput(response={"status": "completed"})

        agent._invoke_inner = invoke_inner  # type: ignore[method-assign]
        await agent.invoke(RequestContext(), agent_input)

        resolver.validate.assert_not_called()
        assert seen == [("agent", 1)]

    @pytest.mark.anyio
    async def test_invoke_passes_the_outer_execution_session(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """invoke captures ExecutionSession once and reuses it in the inner run."""
        agent = LangGraphBaseAgent(mock_graph)
        constructed: list[ExecutionSession] = []
        original = agent._execution_session

        def capturing_session() -> ExecutionSession:
            session = original()
            constructed.append(session)
            return session

        agent._execution_session = capturing_session  # type: ignore[method-assign]
        seen: list[ExecutionSession] = []

        async def invoke_inner(
            ctx: RequestContext,
            input: AgentInput,
            session: ExecutionSession,
        ) -> AgentOutput:
            seen.append(session)
            return AgentOutput(response={"status": "completed"})

        agent._invoke_inner = invoke_inner  # type: ignore[method-assign]
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            await agent.invoke(RequestContext(), agent_input)

        assert constructed == seen
        assert len(seen) == 1
        assert seen[0].is_durable

    @pytest.mark.anyio
    async def test_invoke_suspended(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Test invoke returns suspended status on interrupt."""
        from langgraph.types import Interrupt

        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Checking..."),
                ]
            }
        )
        # Mock aget_state to indicate a pending interrupt
        mock_task = MagicMock()
        mock_task.interrupts = [Interrupt(value={"reason": "approval_needed"})]
        mock_state = MagicMock()
        mock_state.next = ("some_node",)
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(mock_graph)
        result = await agent.invoke(RequestContext(), agent_input)

        assert isinstance(result, AgentOutput)
        assert isinstance(result.response, dict)
        response = cast(dict[str, Any], result.response)
        assert response["status"] == "suspended"
        suspend_context = response.get("suspend_context")
        assert isinstance(suspend_context, dict)
        assert suspend_context["interrupt_values"] == [{"reason": "approval_needed"}]
        interrupts = response.get("interrupts")
        assert isinstance(interrupts, list)
        interrupt = cast(dict[str, Any], interrupts[0])
        assert interrupt["value"] == {"reason": "approval_needed"}
        resume_schema = cast(dict[str, Any], response["resume_schema"])
        assert resume_schema["required"] == ["resume_map"]

    @pytest.mark.anyio
    async def test_invoke_rejects_conflicting_projected_interrupt_values(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A parent projection cannot change a child interrupt's payload."""
        from langgraph.types import Interrupt

        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Checking...")]}
        )
        child_task = MagicMock(
            state=None,
            interrupts=(Interrupt(value={"claim": "claim-1"}, id="approval"),),
        )
        child_state = MagicMock(tasks=(child_task,))
        root_task = MagicMock(
            state=child_state,
            interrupts=(Interrupt(value={"claim": "claim-2"}, id="approval"),),
        )
        root_state = MagicMock(next=("some_node",), tasks=(root_task,))
        mock_graph.aget_state = AsyncMock(return_value=root_state)

        agent = LangGraphBaseAgent(mock_graph)
        with pytest.raises(
            RuntimeError,
            match='conflicting values for interrupt "approval"',
        ):
            await agent.invoke(RequestContext(), agent_input)

    @pytest.mark.anyio
    async def test_invoke_rejects_non_serializable_interrupt_value(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Non-streaming invocation must fail instead of returning bad suspend data."""
        from langgraph.types import Interrupt

        mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
        mock_task = MagicMock()
        mock_task.interrupts = [Interrupt(value={"not-json"}, id="approval")]
        mock_state = MagicMock()
        mock_state.next = ("some_node",)
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(mock_graph)
        with pytest.raises(
            ValueError,
            match=(
                'LangGraph interrupt "approval" has a non-JSON-serializable '
                "value of type set"
            ),
        ):
            await agent.invoke(RequestContext(), agent_input)

    @pytest.mark.anyio
    async def test_invoke_rejects_nan_interrupt_value(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """NaN passes json.dumps by default but is invalid JSON on the wire, so it must fail."""
        from langgraph.types import Interrupt

        mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
        mock_task = MagicMock()
        mock_task.interrupts = [Interrupt(value={"score": float("nan")}, id="approval")]
        mock_state = MagicMock()
        mock_state.next = ("some_node",)
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(mock_graph)
        with pytest.raises(
            ValueError,
            match=(
                'LangGraph interrupt "approval" has a non-JSON-serializable '
                "value of type dict"
            ),
        ):
            await agent.invoke(RequestContext(), agent_input)

    @pytest.mark.anyio
    async def test_invoke_uses_resolve_thread_id_for_graph_and_execution_id(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Custom resolver must reach ainvoke config and the completed execution_id."""
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="done")]}
        )
        mock_state = MagicMock()
        mock_state.next = ()
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(
            mock_graph,
            resolve_thread_id=lambda ctx: f"{ctx.session_id}__actor",
        )
        result = await agent.invoke(
            RequestContext(session_id="sess-1", workspace_id="ws-1"),
            agent_input,
        )

        invoke_config = mock_graph.ainvoke.call_args.args[1]
        assert invoke_config["configurable"]["thread_id"] == "sess-1__actor"
        assert isinstance(result.response, dict)
        assert result.response["execution_id"] == "sess-1__actor"

    @pytest.mark.anyio
    async def test_invoke_suspended_looks_up_checkpoint_with_resolve_thread_id(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """HITL suspend must query the checkpointer with the agent-owned key."""
        from langgraph.types import Interrupt

        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Checking...")]}
        )
        mock_task = MagicMock()
        mock_task.interrupts = [Interrupt(value={"reason": "approval_needed"})]
        mock_state = MagicMock()
        mock_state.next = ("some_node",)
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        mock_checkpointer = MagicMock()
        mock_checkpointer.aget_tuple = AsyncMock(
            return_value=MagicMock(checkpoint={"id": "ckpt-custom"})
        )
        mock_graph.checkpointer = mock_checkpointer

        agent = LangGraphBaseAgent(
            mock_graph,
            resolve_thread_id=lambda ctx: f"{ctx.session_id}__actor",
        )
        result = await agent.invoke(
            RequestContext(session_id="sess-1", workspace_id="ws-1"),
            agent_input,
        )

        lookup_config = mock_checkpointer.aget_tuple.call_args.args[0]
        assert lookup_config["configurable"]["thread_id"] == "sess-1__actor"
        assert isinstance(result.response, dict)
        assert result.response["execution_id"] == "sess-1__actor"
        suspend_context = result.response.get("suspend_context")
        assert isinstance(suspend_context, dict)
        assert suspend_context["checkpoint_id"] == "ckpt-custom"

    @pytest.mark.anyio
    async def test_stream_resume_recomputes_resolve_thread_id(
        self, mock_graph: MagicMock
    ) -> None:
        """Resume stream must pass the same agent-owned key into astream config."""
        resume_ctx = RequestContext(
            session_id="sess-1",
            workspace_id="ws-1",
            resume=True,
            resume_data={"decision": "approved"},
        )
        resume_input = AgentInput(payload={"message": ""})
        captured: dict[str, Any] = {}

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            captured["config"] = args[1] if len(args) > 1 else kwargs.get("config")
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="Resumed")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph,
            resolve_thread_id=lambda ctx: f"{ctx.session_id}__actor",
        )
        events = [event async for event in agent.stream(resume_ctx, resume_input)]

        assert isinstance(captured["config"], dict)
        assert captured["config"]["configurable"]["thread_id"] == "sess-1__actor"
        assert len(events) == 1
        assert events[0].event == "result"
        assert isinstance(events[0].data, dict)
        assert events[0].data.get("resumed") is True

    @pytest.mark.anyio
    async def test_resume(self, mock_graph: MagicMock, agent_input: AgentInput) -> None:
        """Test resume method delegates to invoke with resume metadata."""
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content="Approved and done"),
                ]
            }
        )
        mock_state = MagicMock()
        mock_state.next = ()
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        agent = LangGraphBaseAgent(mock_graph)
        result = await agent.resume(RequestContext(), agent_input, "approved")

        assert isinstance(result, AgentOutput)
        assert isinstance(result.response, dict)
        assert result.response["response"] == "Approved and done"
        assert result.response["status"] == "completed"
        assert result.response.get("resumed") is True

    @pytest.mark.anyio
    async def test_stream_tokens(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """``LangGraphBaseAgent.stream`` calls ``astream(...,
        subgraphs=True)`` so the yield shape is ``(namespace, stream_mode,
        payload)``. Token events include a ``source`` attribution key
        (empty for root-agent tokens).
        """

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            # Simulate messages stream mode with chunks
            yield ((), "messages", (AIMessageChunk(content="Hello"), {}))
            yield ((), "messages", (AIMessageChunk(content=" world"), {}))
            # Simulate updates with final result
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="Hello world")]}},
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 2 token events + 1 result event.
        # tool_call_id is empty for root-agent tokens — it only carries a
        # value for tokens emitted while a ``task`` subagent dispatch is in
        # flight (parallel same-name fan-out disambiguation).
        assert len(events) == 3
        assert events[0].event == "token"
        assert events[0].data == {"content": "Hello", "source": "", "tool_call_id": ""}
        assert events[1].event == "token"
        assert events[1].data == {"content": " world", "source": "", "tool_call_id": ""}
        assert events[2].event == "result"
        assert isinstance(events[2].data, dict)
        assert events[2].data["response"] == "Hello world"

    @pytest.mark.anyio
    async def test_durable_stream_binds_and_restores_nested_operation_resolver(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        resolver = MagicMock(
            return_value=(ChildOperationBoundary("case_investigation", "task-a"),)
        )
        agent = LangGraphBaseAgent(mock_graph, durable_subgraphs=resolver)
        seen: list[tuple[str, int]] = []

        async def stream_inner(
            ctx: RequestContext,
            input: AgentInput,
            _session: ExecutionSession,
        ) -> AsyncIterator[Any]:
            seen.extend(
                (segment.name, segment.ordinal)
                for segment in current_operation_path().segments
            )
            yield MagicMock()

        agent._stream_inner = stream_inner  # type: ignore[method-assign]
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            _ = [event async for event in agent.stream(RequestContext(), agent_input)]
            restored = current_operation_path()

        resolver.validate.assert_called_once_with()
        assert seen == [("agent", 1), ("case_investigation", 1)]
        assert [(segment.name, segment.ordinal) for segment in restored.segments] == [
            ("agent", 1)
        ]

    @pytest.mark.anyio
    async def test_durable_stream_can_be_drained_by_another_task(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        resolver = MagicMock(
            return_value=(ChildOperationBoundary("case_investigation", "task-a"),)
        )
        agent = LangGraphBaseAgent(mock_graph, durable_subgraphs=resolver)

        async def stream_inner(
            ctx: RequestContext,
            input: AgentInput,
            _session: ExecutionSession,
        ) -> AsyncIterator[Any]:
            assert current_operation_path().segments[-1].name == "case_investigation"
            yield MagicMock()

        agent._stream_inner = stream_inner  # type: ignore[method-assign]
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            stream = agent.stream(RequestContext(), agent_input)
            await anext(stream)
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(stream), timeout=1)

        resolver.validate.assert_called_once_with()

    @pytest.mark.anyio
    async def test_stream_finishes_after_terminal_agent_update(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_graph: MagicMock,
        agent_input: AgentInput,
    ) -> None:
        """A terminal ReAct agent update should complete the adapter stream.

        This guards against LangGraph streams that keep the async iterator open
        after emitting the final no-tool-call AIMessage.
        """
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.stream_scope.TERMINAL_AGENT_UPDATE_DRAIN_TIMEOUT_S",
            0.01,
        )

        class HangingTerminalStream:
            def __init__(self) -> None:
                self.closed = False
                self._step = 0
                self._released = asyncio.Event()

            def __aiter__(self) -> HangingTerminalStream:
                return self

            async def __anext__(self) -> tuple[Any, str, Any]:
                if self._step == 0:
                    self._step += 1
                    return ((), "messages", (AIMessageChunk(content="Done"), {}))
                if self._step == 1:
                    self._step += 1
                    return (
                        (),
                        "updates",
                        {"agent": {"messages": [AIMessage(content="Done")]}},
                    )
                await self._released.wait()
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.closed = True
                self._released.set()

        stream = HangingTerminalStream()
        mock_graph.astream = MagicMock(return_value=stream)

        agent = LangGraphBaseAgent(mock_graph)

        async def collect_events() -> list[Any]:
            return [
                event async for event in agent.stream(RequestContext(), agent_input)
            ]

        events = await asyncio.wait_for(collect_events(), timeout=1.0)

        assert len(events) == 2
        assert events[0].event == "token"
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Done"
        assert stream.closed is True

    @pytest.mark.anyio
    async def test_durable_stream_waits_for_graph_exhaustion_before_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_graph: MagicMock,
        agent_input: AgentInput,
    ) -> None:
        """A terminal-looking update cannot truncate later durable graph work."""
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.stream_scope.TERMINAL_AGENT_UPDATE_DRAIN_TIMEOUT_S",
            0.01,
        )

        async def delayed_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="First answer")]}},
            )
            await asyncio.sleep(0.03)
            yield (
                (),
                "updates",
                {"finalize": {"messages": [AIMessage(content="Final answer")]}},
            )

        mock_graph.astream = delayed_stream
        mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=(), tasks=()))
        agent = LangGraphBaseAgent(mock_graph)
        monkeypatch.setattr(
            ExecutionSession, "seed_previous_session_state", AsyncMock()
        )
        complete = AsyncMock()
        monkeypatch.setattr(ExecutionSession, "complete_execution", complete)
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=1,
            workflow_identity=WorkflowIdentity(
                session_id="session-1",
                execution_id="execution-1",
            ),
        )

        with attempt_context_scope(attempt):
            events = [
                event async for event in agent.stream(RequestContext(), agent_input)
            ]

        terminal_data = events[-1].data
        assert isinstance(terminal_data, dict)
        assert terminal_data["response"] == "Final answer"
        complete.assert_awaited_once()

    @pytest.mark.anyio
    async def test_stream_ignores_subgraph_terminal_agent_update(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Only a root-namespace terminal agent update can finish the stream."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                ("reviewer:abc",),
                "updates",
                {"agent": {"messages": [AIMessage(content="subagent done")]}},
            )
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="root done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)

        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        assert len(events) == 1
        assert events[0].event == "result"
        assert isinstance(events[0].data, dict)
        assert events[0].data["response"] == "root done"

    @pytest.mark.anyio
    async def test_stream_processes_root_updates_after_terminal_agent_candidate(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Custom graphs may run valid root nodes after an ``agent`` update."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="intermediate")]}},
            )
            yield (
                (),
                "updates",
                {"summary": {"messages": [AIMessage(content="final summary")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)

        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        assert len(events) == 1
        assert events[0].event == "result"
        assert isinstance(events[0].data, dict)
        assert events[0].data["response"] == "final summary"

    @pytest.mark.anyio
    async def test_stream_suspend_includes_checkpoint_id(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Test streaming with HITL suspend includes checkpoint_id."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Checking..."), {}))
            # Node output before the interrupt — this is what populates
            # all_messages (updates-mode node outputs), so it must appear in the
            # suspend event's pre-suspend messages.
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="Checking...")]}},
            )
            # Simulate interrupt
            yield (
                (),
                "updates",
                {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "suspend_reason": "Need approval",
                                "suspend_context": {"amount": 1000},
                            }
                        )
                    ]
                },
            )

        mock_graph.astream = mock_astream

        # Mock checkpointer to return a checkpoint ID
        mock_checkpointer = MagicMock()
        mock_checkpointer.aget_tuple = AsyncMock(
            return_value=MagicMock(checkpoint={"id": "ckpt-789"})
        )
        mock_graph.checkpointer = mock_checkpointer

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 1 token + 1 suspend
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[1].event == "suspend"
        assert isinstance(events[1].data, dict)
        assert events[1].data.get("suspend_payload") is not None
        # suspend_payload is the first interrupt's value directly (flat dict)
        payload = events[1].data["suspend_payload"]
        assert isinstance(payload, dict)
        assert payload["suspend_reason"] == "Need approval"
        # checkpoint_id is carried in the opaque metadata dict the AER
        # round-trips, not at the top level of the suspend event.
        metadata = events[1].data["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["checkpoint_id"] == "ckpt-789"
        # The suspend event carries the pre-suspend messages so the AER can
        # persist them to STM. The streamed assistant content must be present.
        assert isinstance(events[1].data, dict)
        presuspend = events[1].data.get("messages")
        assert isinstance(presuspend, list)
        assert any(
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and "Checking" in str(m.get("content") or "")
            for m in presuspend
        )

    @pytest.mark.anyio
    async def test_stream_emits_durable_activity_suspension(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_graph: MagicMock,
        agent_input: AgentInput,
    ) -> None:
        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            raise DurableActivitySuspended(
                "awaiting_human_review",
                {
                    "allowed_decisions": ["approve", "reject"],
                    "claim_id": "claim-1",
                },
            )
            yield  # pragma: no cover

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)
        monkeypatch.setattr(
            ExecutionSession, "seed_previous_session_state", AsyncMock()
        )

        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            events = [
                event async for event in agent.stream(RequestContext(), agent_input)
            ]

        assert len(events) == 1
        assert events[0].event == "suspend"
        assert events[0].data == {
            "suspend_payload": {
                "suspend_reason": "awaiting_human_review",
                "suspend_context": {
                    "allowed_decisions": ["approve", "reject"],
                    "claim_id": "claim-1",
                },
            },
            "resumed": False,
            "messages": [],
            "metadata": {},
        }

    @pytest.mark.anyio
    async def test_stream_finalizes_durable_suspension_commit_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_graph: MagicMock,
        agent_input: AgentInput,
    ) -> None:
        commit_error = RuntimeError("OE rejected the suspended outcome")
        suspension = DurableActivitySuspended("awaiting_human_review", {})

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "subagent_type": "reviewer",
                                    "description": "Review the claim",
                                },
                                "id": "task-1",
                            }
                        ],
                    ),
                    {},
                ),
            )
            raise suspension from commit_error

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )
        monkeypatch.setattr(
            ExecutionSession, "seed_previous_session_state", AsyncMock()
        )

        events = []
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            with pytest.raises(
                RuntimeError, match="rejected the suspended outcome"
            ) as raised:
                async for event in agent.stream(RequestContext(), agent_input):
                    events.append(event)

        assert raised.value is commit_error
        assert [event.event for event in events] == [
            "subagent_start",
            "subagent_end",
            "custom_event",
        ]
        assert events[-1].custom_event == {
            "kind": "error",
            "detail": "Internal server error",
        }

    @pytest.mark.anyio
    async def test_stream_resume_via_ctx(self, mock_graph: MagicMock) -> None:
        """Test resume flow via ctx."""
        resume_ctx = RequestContext(
            user_id="user-123",
            session_id="thread-789",
            resume=True,
            resume_data={"decision": "approved"},
            metadata={"checkpoint_id": "ckpt-123"},
        )
        resume_input = AgentInput(payload={"message": ""})

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Resumed"), {}))
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="Resumed and completed")]}},
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(resume_ctx, resume_input)]

        # Should have 1 token + 1 result
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Resumed and completed"
        assert events[1].data.get("resumed") is True

    @pytest.mark.anyio
    async def test_stream_empty_messages_raises(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Test that empty messages raises RuntimeError."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "updates", {"node": {}})  # No messages

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(RuntimeError, match="without producing messages"):
            _ = [event async for event in agent.stream(RequestContext(), agent_input)]

    @pytest.mark.anyio
    async def test_real_graph_parallel_interrupts_resume_by_framework_id(self) -> None:
        """Real LangGraph fan-out surfaces and routes every pending interrupt."""
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.types import interrupt

        received_answers: dict[str, str] = {}

        def fanout(state: Any) -> dict[str, Any]:
            return {}

        def payment_review(state: Any) -> dict[str, Any]:
            answer = interrupt({"node": "payment_review", "amount": 100})
            received_answers["payment_review"] = cast(str, answer)
            return {"messages": [AIMessage(content=f"payment:{answer}")]}

        def refund_review(state: Any) -> dict[str, Any]:
            answer = interrupt({"node": "refund_review", "amount": 50})
            received_answers["refund_review"] = cast(str, answer)
            return {"messages": [AIMessage(content=f"refund:{answer}")]}

        def finish(state: Any) -> dict[str, Any]:
            payment = received_answers["payment_review"]
            refund = received_answers["refund_review"]
            return {"messages": [AIMessage(content=f"completed:{payment}|{refund}")]}

        graph: StateGraph[Any] = StateGraph(MessagesState)
        graph.add_node("fanout", fanout)
        graph.add_node("payment_review", payment_review)
        graph.add_node("refund_review", refund_review)
        graph.add_node("finish", finish)
        graph.add_edge(START, "fanout")
        graph.add_edge("fanout", "payment_review")
        graph.add_edge("fanout", "refund_review")
        graph.add_edge("payment_review", "finish")
        graph.add_edge("refund_review", "finish")
        graph.add_edge("finish", END)
        agent = LangGraphBaseAgent(graph.compile(checkpointer=InMemorySaver()))
        agent_input = AgentInput(payload={"message": "review both"})
        initial_ctx = RequestContext(
            user_id="user-parallel",
            session_id="session-parallel",
            workspace_id="workspace-parallel",
        )

        initial_events = [
            event async for event in agent.stream(initial_ctx, agent_input)
        ]

        suspend_events = [event for event in initial_events if event.event == "suspend"]
        assert len(suspend_events) == 1
        assert isinstance(suspend_events[0].data, dict)
        suspend_data = suspend_events[0].data
        interrupts = cast(list[dict[str, Any]], suspend_data["interrupts"])
        assert len(interrupts) == 2
        assert {
            (interrupt["value"]["node"], interrupt["value"]["amount"])
            for interrupt in interrupts
        } == {("payment_review", 100), ("refund_review", 50)}
        interrupt_ids = [cast(str, interrupt["id"]) for interrupt in interrupts]
        assert all(interrupt_ids)
        assert len(set(interrupt_ids)) == 2

        resume_schema = cast(dict[str, Any], suspend_data["resume_schema"])
        assert resume_schema["required"] == ["resume_map"]
        resume_map_schema = resume_schema["properties"]["resume_map"]
        assert set(resume_map_schema["required"]) == set(interrupt_ids)
        assert set(resume_map_schema["properties"]) == set(interrupt_ids)

        resume_map = {
            cast(str, interrupt["id"]): (
                "approve" if interrupt["value"]["node"] == "payment_review" else "deny"
            )
            for interrupt in interrupts
        }
        resume_ctx = initial_ctx.model_copy(
            update={
                "resume": True,
                "resume_data": resume_map,
                "metadata": suspend_data["metadata"],
            }
        )
        resume_events = [
            event
            async for event in agent.stream(
                resume_ctx, AgentInput(payload={"message": ""})
            )
        ]

        result_events = [event for event in resume_events if event.event == "result"]
        assert len(result_events) == 1
        assert isinstance(result_events[0].data, dict)
        assert result_events[0].data["response"] == "completed:approve|deny"
        assert received_answers == {
            "payment_review": "approve",
            "refund_review": "deny",
        }

    @pytest.mark.anyio
    async def test_stream_reports_full_interrupt_snapshot(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Parallel suspensions retain every framework interrupt and id."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {
                    "__interrupt__": [
                        Interrupt(
                            value={"task": "approve_payment", "amount": 100},
                            id="payment",
                        ),
                        Interrupt(
                            value={"task": "approve_refund", "amount": 50},
                            id="refund",
                        ),
                        Interrupt(value={"task": "verify_identity"}, id="identity"),
                    ]
                },
            )

        mock_graph.astream = mock_astream
        mock_graph.checkpointer = None

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        assert len(events) == 1
        assert events[0].event == "suspend"
        assert isinstance(events[0].data, dict)
        data = events[0].data
        assert data["suspend_payload"] == {
            "task": "approve_payment",
            "amount": 100,
        }
        assert data["interrupts"] == [
            {
                "id": "payment",
                "value": {"task": "approve_payment", "amount": 100},
            },
            {"id": "refund", "value": {"task": "approve_refund", "amount": 50}},
            {"id": "identity", "value": {"task": "verify_identity"}},
        ]
        resume_schema = data["resume_schema"]
        assert isinstance(resume_schema, dict)
        assert resume_schema["required"] == ["resume_map"]
        properties = resume_schema["properties"]
        assert isinstance(properties, dict)
        resume_map_schema = properties["resume_map"]
        assert isinstance(resume_map_schema, dict)
        assert resume_map_schema["required"] == [
            "payment",
            "refund",
            "identity",
        ]
        assert data["metadata"] == {"checkpoint_id": None}

    @pytest.mark.anyio
    async def test_stream_reports_resume_map_schema_for_single_interrupt(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Require an id-exact resume map for a single pending interrupt."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {
                    "__interrupt__": [
                        Interrupt(
                            value={"task": "approve_payment", "amount": 100},
                            id="approval",
                        ),
                    ]
                },
            )

        mock_graph.astream = mock_astream
        mock_graph.checkpointer = None

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        assert len(events) == 1
        assert events[0].event == "suspend"
        assert isinstance(events[0].data, dict)
        data = events[0].data
        assert data["resume_schema"] == {
            "type": "object",
            "required": ["resume_map"],
            "properties": {
                "resume_map": {
                    "type": "object",
                    "required": ["approval"],
                    "properties": {"approval": {}},
                    "additionalProperties": False,
                },
            },
            "additionalProperties": True,
        }

    @pytest.mark.anyio
    async def test_stream_rejects_conflicting_values_for_same_interrupt_id(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """One framework identity cannot address two different reviews."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {
                    "__interrupt__": [
                        Interrupt(value={"claim": "claim-1"}, id="approval"),
                        Interrupt(value={"claim": "claim-2"}, id="approval"),
                    ]
                },
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(
            RuntimeError,
            match='conflicting values for interrupt "approval"',
        ):
            _ = [event async for event in agent.stream(RequestContext(), agent_input)]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("value", "value_type"),
        [
            (datetime(2026, 8, 7), "datetime"),
            ({"not-json"}, "set"),
            (object(), "object"),
        ],
    )
    async def test_stream_rejects_non_serializable_interrupt_value(
        self,
        mock_graph: MagicMock,
        agent_input: AgentInput,
        value: object,
        value_type: str,
    ) -> None:
        """A bad interrupt value errors before any suspend event is emitted."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {"__interrupt__": [Interrupt(value=value, id="approval")]},
            )

        mock_graph.astream = mock_astream
        events: list[Any] = []
        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(
            ValueError,
            match=(
                'LangGraph interrupt "approval" has a non-JSON-serializable '
                f"value of type {value_type}"
            ),
        ):
            async for event in agent.stream(RequestContext(), agent_input):
                events.append(event)

        assert events == []

    @pytest.mark.anyio
    async def test_stream_resume_without_data_raises(
        self, mock_graph: MagicMock
    ) -> None:
        """Test that resume=True without resume_data raises ValueError."""
        bad_ctx = RequestContext(
            user_id="user-123",
            session_id="thread-789",
            resume=True,
            metadata={"checkpoint_id": "ckpt-123"},
        )
        bad_input = AgentInput(payload={"message": ""})

        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(ValueError, match="resume_data"):
            _ = [event async for event in agent.stream(bad_ctx, bad_input)]

    @pytest.mark.anyio
    async def test_latest_checkpoint_id_returns_none_on_error(
        self, mock_graph: MagicMock
    ) -> None:
        """Checkpoint lookup errors must degrade to None per the docstring,
        so callers (notably ``stream()``) can still emit the suspend event."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.aget_tuple = AsyncMock(
            side_effect=RuntimeError("DB connection failed")
        )
        mock_graph.checkpointer = mock_checkpointer

        result = await latest_checkpoint_id(mock_graph, "thread-123")
        assert result is None

    @pytest.mark.anyio
    async def test_completed_execution_survives_checkpoint_lookup_failure(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="completed")]}
        )
        mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=[]))
        mock_graph.checkpointer = MagicMock(
            aget_tuple=AsyncMock(side_effect=RuntimeError("DB connection failed"))
        )

        result = await LangGraphBaseAgent(mock_graph).invoke(
            RequestContext(session_id="thread-123"), agent_input
        )

        response = cast(dict[str, Any], result.response)
        assert response["status"] == "completed"
        assert response["metadata"] is None

    @pytest.mark.anyio
    async def test_stream_emits_suspend_event_when_checkpoint_lookup_fails(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A transient checkpointer error must not abort the suspend stream;
        the consumer must still receive a suspend StreamEvent (with
        checkpoint_id=None) so HITL flows degrade gracefully."""
        from langgraph.types import Interrupt

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield (
                (),
                "updates",
                {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "suspend_reason": "Need approval",
                                "suspend_context": {"amount": 1000},
                            }
                        )
                    ]
                },
            )

        mock_graph.astream = mock_astream

        # Stub aget_tuple to raise — simulates a transient checkpointer fault.
        mock_checkpointer = MagicMock()
        mock_checkpointer.aget_tuple = AsyncMock(side_effect=RuntimeError("transient"))
        mock_graph.checkpointer = mock_checkpointer

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Before the fix this raised RuntimeError instead of yielding suspend.
        assert len(events) == 1
        assert events[0].event == "suspend"
        assert isinstance(events[0].data, dict)
        assert events[0].data.get("checkpoint_id") is None
        payload = events[0].data["suspend_payload"]
        assert isinstance(payload, dict)
        assert payload["suspend_reason"] == "Need approval"

    @pytest.mark.anyio
    async def test_stream_overwrite_wrapped_messages(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """stream() must unwrap Overwrite containers and collect inner messages."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Overwritten"), {}))
            yield (
                (),
                "updates",
                {
                    "patch_tool_calls": {
                        "messages": Overwrite([AIMessage(content="Overwritten")])
                    }
                },
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 1 token event + 1 result event
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[0].data == {
            "content": "Overwritten",
            "source": "",
            "tool_call_id": "",
        }
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Overwritten"

    @pytest.mark.anyio
    async def test_stream_command_update_with_messages(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """stream() must extract messages from Command.update dict."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Sub result"), {}))
            yield (
                (),
                "updates",
                {
                    "tools": Command(
                        update={
                            "messages": [
                                ToolMessage(content="done", tool_call_id="tc-1"),
                                AIMessage(content="Sub result"),
                            ],
                            "custom_field": "val",
                        }
                    )
                },
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 1 token event + 1 result event.
        # (No subagent_end is synthesized because no prior ``task`` tool_call
        # registered this tool_call_id in ``active_tasks``.)
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[0].data == {
            "content": "Sub result",
            "source": "",
            "tool_call_id": "",
        }
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Sub result"
        assert events[1].data["message_count"] == 2

    @pytest.mark.anyio
    async def test_stream_command_update_without_messages(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """stream() must gracefully skip Command updates without messages key."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Hello"), {}))
            yield (
                (),
                "updates",
                {
                    "summarize": Command(
                        update={"_summarization_event": {"cutoff_index": 5}}
                    )
                },
            )
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="Hello")]}},
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 1 token event + 1 result event
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[0].data == {"content": "Hello", "source": "", "tool_call_id": ""}
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Hello"

    @pytest.mark.anyio
    async def test_stream_standard_add_messages_regression(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Regression guard: standard add_messages pattern must still work."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="Normal"), {}))
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="Normal response")]}},
            )

        mock_graph.astream = mock_astream

        agent = LangGraphBaseAgent(mock_graph)
        events = [event async for event in agent.stream(RequestContext(), agent_input)]

        # Should have 1 token event + 1 result event
        assert len(events) == 2
        assert events[0].event == "token"
        assert events[0].data == {"content": "Normal", "source": "", "tool_call_id": ""}
        assert events[1].event == "result"
        assert isinstance(events[1].data, dict)
        assert events[1].data["response"] == "Normal response"


class _MessagesParser(LangGraphOutputParser):
    """Emits a custom event for each non-empty messages-mode chunk."""

    stream_modes = ("messages",)

    async def parse(
        self, item: RawStreamItem, ctx: RequestContext
    ) -> AsyncIterator[BaseModel | JsonValue]:
        if item.stream_mode != "messages":
            return
        chunk, _ = item.payload
        content = getattr(chunk, "content", "")
        if content:
            yield {"kind": "brief", "text": content}

    async def on_stream_error(
        self, ctx: RequestContext, error: BaseException
    ) -> BaseModel | JsonValue | None:
        return {"kind": "error", "detail": str(error)}


async def _two_token_stream(*args: Any, **kwargs: Any) -> Any:
    yield ((), "messages", (AIMessageChunk(content="Hello"), {}))
    yield ((), "messages", (AIMessageChunk(content=" world"), {}))
    yield ((), "updates", {"node": {"messages": [AIMessage(content="Hello world")]}})


class TestLangGraphBaseAgentOutputParser:
    """Story 2: the parser dual-runs alongside the untouched platform pipeline."""

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        graph = MagicMock()
        graph.checkpointer = None
        return graph

    @pytest.fixture
    def agent_input(self) -> AgentInput:
        return AgentInput(payload={"message": "hi"})

    @pytest.mark.anyio
    async def test_custom_events_emitted_alongside_platform_events(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Each parsed item yields a custom_event StreamEvent; platform events
        (token/result) are unchanged and ride on their own frames."""
        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        customs = [e for e in events if e.custom_event is not None]
        platform = [e for e in events if e.custom_event is None]
        assert [e.custom_event for e in customs] == [
            {"kind": "brief", "text": "Hello"},
            {"kind": "brief", "text": " world"},
        ]
        # Platform pipeline byte-identical: same 2 tokens + result as no-parser run.
        assert [e.event for e in platform] == ["token", "token", "result"]
        # A custom event carries no platform data, and is tagged CUSTOM_EVENT.
        assert all(e.event == CUSTOM_EVENT and e.data is None for e in customs)
        # In-sync: each custom precedes its item's token, and result is last.
        assert events[0].custom_event == {"kind": "brief", "text": "Hello"}
        assert events[1].event == "token"
        assert events[-1].event == "result"

    @pytest.mark.anyio
    async def test_no_custom_events_when_flag_off(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A registered-but-inert parser emits nothing; stream is byte-identical."""
        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=False
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        assert all(e.custom_event is None for e in events)
        assert [e.event for e in events] == ["token", "token", "result"]

    @pytest.mark.anyio
    async def test_stream_modes_union_reaches_astream(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """The parser's declared modes are unioned into the astream request."""

        class ValuesParser(_MessagesParser):
            stream_modes = ("values",)

        mock_graph.astream = MagicMock(side_effect=_two_token_stream)
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=ValuesParser, use_custom_parser=True
        )

        _ = [e async for e in agent.stream(RequestContext(), agent_input)]

        modes = mock_graph.astream.call_args.kwargs["stream_mode"]
        assert set(modes) == {"messages", "updates", "custom", "values"}

    @pytest.mark.anyio
    async def test_platform_subscribes_custom_only_when_use_custom_parser(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """``custom`` is subscribed only when ``use_custom_parser`` is on."""
        mock_graph.astream = MagicMock(side_effect=_two_token_stream)

        off = LangGraphBaseAgent(mock_graph)
        _ = [e async for e in off.stream(RequestContext(), agent_input)]
        off_modes = mock_graph.astream.call_args.kwargs["stream_mode"]
        assert "custom" not in off_modes
        assert set(off_modes) == {"messages", "updates"}

        on = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )
        _ = [e async for e in on.stream(RequestContext(), agent_input)]
        on_modes = mock_graph.astream.call_args.kwargs["stream_mode"]
        assert set(on_modes) == {"messages", "updates", "custom"}

    @pytest.mark.anyio
    async def test_platform_yields_custom_event_when_parser_omits_custom(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Without parser ``custom``, Atlas Agent Engine yields CUSTOM_EVENT verbatim."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", {"event": "step", "data": "Building brief"})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]
        customs = [e for e in events if e.custom_event is not None]
        assert [e.custom_event for e in customs] == [
            {"event": "step", "data": "Building brief"}
        ]
        assert all(e.event == CUSTOM_EVENT and e.data is None for e in customs)

    @pytest.mark.anyio
    async def test_parser_owns_custom_when_listed_in_stream_modes(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """When the author lists ``custom``, Atlas Agent Engine presents and does not
        also platform-yield the same payload (no double emit)."""

        class CustomModeParser(LangGraphOutputParser):
            stream_modes = ("custom",)

            async def parse(
                self, item: RawStreamItem, ctx: RequestContext
            ) -> AsyncIterator[BaseModel | JsonValue]:
                if item.stream_mode != "custom":
                    return
                # Author deliberately rewrites — Atlas Agent Engine presents, author owns.
                yield {"rewritten": True, "raw": item.payload}

            async def on_stream_error(
                self, ctx: RequestContext, error: BaseException
            ) -> BaseModel | JsonValue | None:
                return None

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", {"event": "step", "data": "Building brief"})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=CustomModeParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]
        customs = [e.custom_event for e in events if e.custom_event is not None]
        assert customs == [
            {
                "rewritten": True,
                "raw": {"event": "step", "data": "Building brief"},
            }
        ]

    @pytest.mark.anyio
    async def test_emit_fails_closed_during_stream_without_use_custom_parser(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Feature-off streams must not silently drop author emit calls."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            await emit_custom_event({"event": "step", "data": "should fail"})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)

        with pytest.raises(RuntimeError, match="requires features.use_custom_parser"):
            _ = [e async for e in agent.stream(RequestContext(), agent_input)]

    @pytest.mark.anyio
    async def test_on_stream_error_invalid_payload_preserves_original_failure(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Bad error-hook JSON must not replace the original graph failure."""

        class NanErrorParser(_MessagesParser):
            async def on_stream_error(
                self, ctx: RequestContext, error: BaseException
            ) -> BaseModel | JsonValue | None:
                # Intentionally invalid finite-JSON payload for the regression.
                return cast(JsonValue, {"value": math.nan})

        async def failing_stream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="partial"), {}))
            raise RuntimeError("graph boom")

        mock_graph.astream = failing_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=NanErrorParser, use_custom_parser=True
        )

        with pytest.raises(RuntimeError, match="graph boom"):
            _ = [e async for e in agent.stream(RequestContext(), agent_input)]

    @pytest.mark.anyio
    async def test_platform_ignores_custom_without_use_custom_parser(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Without the feature flag, Atlas Agent Engine does not surface custom traffic."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", {"foo": 1})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(mock_graph)

        events = [e async for e in agent.stream(RequestContext(), agent_input)]
        assert all(e.custom_event is None for e in events)
        assert events[-1].event == "result"

    @pytest.mark.anyio
    async def test_null_raw_custom_payload_does_not_fail_stream(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """JSON null on the shared custom channel must not abort the stream."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", None)
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        assert all(e.custom_event is None for e in events)
        assert events[-1].event == "result"
        assert isinstance(events[-1].data, dict)
        assert events[-1].data["response"] == "done"

    @pytest.mark.anyio
    async def test_invalid_raw_custom_payload_does_not_fail_stream(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Unrelated invalid LangGraph custom traffic is skipped safely."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", {"bad": object()})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        assert all(e.custom_event is None for e in events)
        assert events[-1].event == "result"
        assert isinstance(events[-1].data, dict)
        assert events[-1].data["response"] == "done"

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    @pytest.mark.anyio
    async def test_non_finite_raw_custom_payload_does_not_fail_stream(
        self, mock_graph: MagicMock, agent_input: AgentInput, value: float
    ) -> None:
        """Raw non-finite JSON values are skipped by the platform fallback."""

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "custom", {"value": value})
            yield (
                (),
                "updates",
                {"node": {"messages": [AIMessage(content="done")]}},
            )

        mock_graph.astream = mock_astream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=_MessagesParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        assert all(e.custom_event is None for e in events)
        assert events[-1].event == "result"
        assert isinstance(events[-1].data, dict)
        assert events[-1].data["response"] == "done"

    @pytest.mark.anyio
    async def test_parser_non_finite_payload_fails_stream(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """Parser-yielded non-finite values fail closed via finite-JSON check."""

        class NanParser(_MessagesParser):
            async def parse(
                self, item: RawStreamItem, ctx: RequestContext
            ) -> AsyncIterator[BaseModel | JsonValue]:
                chunk, _ = item.payload
                if chunk.content:
                    # Intentionally invalid finite-JSON payload for the regression.
                    yield cast(JsonValue, {"value": math.nan})

        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=NanParser, use_custom_parser=True
        )

        with pytest.raises(ValueError, match="non-JSON-serializable"):
            _ = [e async for e in agent.stream(RequestContext(), agent_input)]

    @pytest.mark.anyio
    async def test_no_custom_stream_regression_token_result_path(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """No parser / no emit: platform token+result path is unchanged."""
        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(mock_graph)

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        assert all(e.custom_event is None for e in events)
        assert [e.event for e in events] == ["token", "token", "result"]

    @pytest.mark.anyio
    async def test_parse_serializes_basemodel_return(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A pydantic return value is dumped to a JSON-safe custom_event."""

        class Brief(BaseModel):
            text: str
            n: int

        class ModelParser(_MessagesParser):
            async def parse(self, item, ctx):
                chunk, _ = item.payload
                if chunk.content:
                    yield Brief(text=chunk.content, n=1)

        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=ModelParser, use_custom_parser=True
        )

        events = [e async for e in agent.stream(RequestContext(), agent_input)]

        customs = [e.custom_event for e in events if e.custom_event is not None]
        assert customs == [{"text": "Hello", "n": 1}, {"text": " world", "n": 1}]

    @pytest.mark.anyio
    async def test_on_stream_error_emits_final_custom_event(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """On failure the parser gets a final custom_event before the error
        propagates as the terminal."""

        async def failing_stream(*args: Any, **kwargs: Any) -> Any:
            yield ((), "messages", (AIMessageChunk(content="partial"), {}))
            raise RuntimeError("boom")

        seen: list[BaseException] = []

        class CapturingParser(_MessagesParser):
            async def on_stream_error(
                self, ctx: RequestContext, error: BaseException
            ) -> BaseModel | JsonValue | None:
                seen.append(error)
                return await super().on_stream_error(ctx, error)

        mock_graph.astream = failing_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=CapturingParser, use_custom_parser=True
        )

        collected: list[Any] = []
        with pytest.raises(RuntimeError, match="boom"):
            async for event in agent.stream(RequestContext(), agent_input):
                collected.append(event)

        assert collected[-1].custom_event == {
            "kind": "error",
            "detail": "Internal server error",
        }
        assert len(seen) == 1
        assert isinstance(seen[0], InternalStreamError)

    @pytest.mark.anyio
    async def test_parse_failure_propagates_as_error(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A parser bug fails the execution rather than dropping shaped output;
        on_stream_error still emits a final custom_event before it propagates."""

        class BrokenParser(_MessagesParser):
            async def parse(self, item, ctx):
                raise RuntimeError("parser boom")
                yield  # pragma: no cover - unreachable; makes this an async generator

        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=BrokenParser, use_custom_parser=True
        )

        collected: list[Any] = []
        with pytest.raises(RuntimeError, match="parser boom"):
            async for event in agent.stream(RequestContext(), agent_input):
                collected.append(event)

        assert collected[-1].custom_event == {
            "kind": "error",
            "detail": "Internal server error",
        }

    @pytest.mark.anyio
    async def test_parse_timeout_propagates_as_error(
        self,
        mock_graph: MagicMock,
        agent_input: AgentInput,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hung parse() times out and fails the execution rather than
        stalling the stream forever; on_stream_error still runs."""
        monkeypatch.setattr(
            "agent_engine_sdk_langgraph.output_parser.PARSER_CALL_TIMEOUT_S", 0.01
        )

        class SlowParser(_MessagesParser):
            async def parse(self, item, ctx):
                await asyncio.sleep(10)
                yield None  # pragma: no cover - unreachable; timeout fires first

        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=SlowParser, use_custom_parser=True
        )

        collected: list[Any] = []
        with pytest.raises(TimeoutError):
            async for event in agent.stream(RequestContext(), agent_input):
                collected.append(event)

        assert collected[-1].custom_event == {
            "kind": "error",
            "detail": "Internal server error",
        }

    @pytest.mark.anyio
    async def test_instantiation_failure_propagates_as_error(
        self, mock_graph: MagicMock, agent_input: AgentInput
    ) -> None:
        """A parser that cannot be instantiated surfaces a clear, well-formed
        RuntimeError before streaming begins, rather than a bare __init__ error."""

        class UninstantiableParser(_MessagesParser):
            def __init__(self) -> None:
                raise ValueError("bad config")

        mock_graph.astream = _two_token_stream
        agent = LangGraphBaseAgent(
            mock_graph, output_parser=UninstantiableParser, use_custom_parser=True
        )

        with pytest.raises(
            RuntimeError, match="Failed to instantiate custom output parser"
        ):
            async for _ in agent.stream(RequestContext(), agent_input):
                pass
