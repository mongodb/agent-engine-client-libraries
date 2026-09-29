"""Tests for LangGraphCallbackAdapter."""

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

from deepagents import create_deep_agent
from langchain.agents.middleware import before_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt

from agent_engine_sdk_langgraph.node_logger_adapter import (
    LangGraphCallbackAdapter,
    _extract_node_name,
)


class TestExtractNodeName:
    """Tests for _extract_node_name helper."""

    def test_extracts_from_metadata_langgraph_node(self):
        """Prefers langgraph_node from metadata."""
        name = _extract_node_name(
            serialized={"name": "fallback"},
            tags=["some_tag"],
            metadata={"langgraph_node": "agent_node"},
        )
        assert name == "agent_node"

    def test_extracts_from_tags(self):
        """Falls back to tags when no metadata."""
        name = _extract_node_name(
            serialized={},
            tags=["my_node"],
            metadata={},
        )
        assert name == "my_node"

    def test_skips_seq_step_tags(self):
        """Ignores LangGraph internal seq:step:N tags."""
        name = _extract_node_name(
            serialized={},
            tags=["seq:step:1", "seq:step:2", "real_node"],
            metadata={},
        )
        assert name == "real_node"

    def test_skips_seq_step_tags_beyond_three(self):
        """Ignores seq:step:N for any N, including 4+ sequenced steps."""
        name = _extract_node_name(
            serialized={},
            tags=["seq:step:4", "seq:step:5", "real_node"],
            metadata={},
        )
        assert name == "real_node"

    def test_extracts_from_serialized_name(self):
        """Falls back to serialized['name']."""
        name = _extract_node_name(
            serialized={"name": "serialized_node"},
            tags=None,
            metadata=None,
        )
        assert name == "serialized_node"

    def test_returns_none_when_nothing_found(self):
        """Returns None when no name can be extracted."""
        name = _extract_node_name(serialized={}, tags=None, metadata=None)
        assert name is None


class TestLangGraphCallbackAdapterDelegation:
    """Tests for LangGraphCallbackAdapter delegation to BaseExecutionCallback."""

    def test_on_chain_start_delegates_with_node_name(self):
        """on_chain_start extracts node name and delegates to callback. A real
        node task run carries pregel's graph:step tag alongside the metadata."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_start(
            serialized={},
            inputs={"key": "value"},
            run_id=run_id,
            parent_run_id=None,
            tags=["graph:step:1"],
            metadata={"langgraph_node": "agent_node"},
        )

        mock_callback.on_node_start.assert_called_once_with(
            node_name="agent_node",
            inputs={"key": "value"},
            run_id="12345678-1234-5678-1234-567812345678",
            parent_run_id=None,
            metadata={"langgraph_node": "agent_node"},
        )

    def test_on_chain_start_skips_runnable_sequence(self):
        """Skips RunnableSequence nodes."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=None,
            metadata=None,
        )

        mock_callback.on_node_start.assert_not_called()

    def test_on_chain_end_delegates(self):
        """on_chain_end delegates to on_node_end."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_end(
            outputs={"result": "done"},
            run_id=run_id,
            metadata={"langgraph_node": "agent_node"},
        )

        mock_callback.on_node_end.assert_called_once()
        call_kwargs = mock_callback.on_node_end.call_args[1]
        assert call_kwargs["node_name"] == "agent_node"
        assert call_kwargs["run_id"] == "12345678-1234-5678-1234-567812345678"

    def test_on_chain_error_delegates(self):
        """on_chain_error delegates to on_node_error with str(error)."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_error(
            error=ValueError("something broke"),
            run_id=run_id,
            metadata={"langgraph_node": "agent_node"},
        )

        mock_callback.on_node_error.assert_called_once()
        call_kwargs = mock_callback.on_node_error.call_args[1]
        assert call_kwargs["node_name"] == "agent_node"
        assert call_kwargs["error"] == "something broke"

    def test_on_chain_error_routes_graph_interrupt_to_suspend(self):
        """GraphInterrupt is routed to on_node_suspend, not on_node_error."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_error(
            error=GraphInterrupt(()),
            run_id=run_id,
            metadata={"langgraph_node": "agent_node"},
        )

        mock_callback.on_node_suspend.assert_called_once()
        mock_callback.on_node_error.assert_not_called()
        call_kwargs = mock_callback.on_node_suspend.call_args[1]
        assert call_kwargs["node_name"] == "agent_node"
        assert call_kwargs["run_id"] == "12345678-1234-5678-1234-567812345678"

    def test_graph_interrupt_falls_back_to_on_node_end_without_suspend(self, caplog):
        """Falls back to on_node_end when callback lacks on_node_suspend, and warns."""
        # Simulate a legacy callback without on_node_suspend
        mock_callback = MagicMock(
            spec=["on_node_start", "on_node_end", "on_node_error"]
        )
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")
        adapter_logger = "agent_engine_sdk_langgraph.node_logger_adapter"

        with caplog.at_level(logging.WARNING, logger=adapter_logger):
            adapter.on_chain_error(
                error=GraphInterrupt(()),
                run_id=run_id,
                metadata={"langgraph_node": "agent_node"},
            )

        mock_callback.on_node_end.assert_called_once()
        mock_callback.on_node_error.assert_not_called()
        call_kwargs = mock_callback.on_node_end.call_args[1]
        assert call_kwargs["node_name"] == "agent_node"
        assert call_kwargs["outputs"] == {}
        assert "does not implement on_node_suspend" in caplog.text
        assert "agent_node" in caplog.text

    def test_converts_uuid_to_str(self):
        """Converts UUID run_id and parent_run_id to strings."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        parent_id = UUID("11111111-2222-3333-4444-555555555555")

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            parent_run_id=parent_id,
            tags=["graph:step:1"],
            metadata={"langgraph_node": "node"},
        )

        call_kwargs = mock_callback.on_node_start.call_args[1]
        assert call_kwargs["run_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert call_kwargs["parent_run_id"] == "11111111-2222-3333-4444-555555555555"

    def test_skips_when_no_node_name(self):
        """Skips callback when no node name can be extracted."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=None,
            metadata=None,
        )

        mock_callback.on_node_start.assert_not_called()


class _ToolThenTextModel(BaseChatModel):
    """Tool call on the first turn, plain text on the second."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-then-text-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"id": "call-1", "name": "get_weather", "args": {}}
                            ],
                        )
                    )
                ]
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="done"))]
        )


@tool
def _get_weather() -> str:
    """Get the weather."""
    return "sunny"


@before_model(can_jump_to=["end"], name="JumpMiddleware")
def _jumpy(state: Any, runtime: Any) -> None:
    return None


@before_model(name="NoJumpMiddleware")
def _plain(state: Any, runtime: Any) -> None:
    return None


class TestConditionalEdgeRouterRuns:
    """Parity guard for the TS adapter's ghost-row fix.

    A hook declaring can_jump_to is wired as a conditional edge. Python
    LangGraph fires no chain callbacks for edge routers, so a jump-capable
    hook must produce exactly one node event per execution — the doubling the
    TS adapter needed a graph:step guard for must never appear here.
    """

    def test_one_node_event_per_hook_execution_regardless_of_jump_targets(self):
        callback = MagicMock()
        agent = create_deep_agent(
            model=_ToolThenTextModel(),
            tools=[_get_weather],
            middleware=[_jumpy, _plain],
        )

        agent.invoke(
            {"messages": [HumanMessage(content="weather?")]},
            config={"callbacks": [LangGraphCallbackAdapter(callback)]},
        )

        # Two agent-loop passes (tool call, then final answer) run each
        # before_model hook twice; edge-router evaluations add no rows.
        starts = [c.kwargs["node_name"] for c in callback.on_node_start.call_args_list]
        assert starts.count("JumpMiddleware.before_model") == 2
        assert starts.count("NoJumpMiddleware.before_model") == 2
        # Pair terminal events by run_id: LangChain Python passes no metadata
        # to on_chain_end, so end-event names fall back to tags and are
        # unreliable. What matters for ghost rows is that every started run
        # closes exactly once and no end arrives for a run that never started.
        started_ids = [
            c.kwargs["run_id"] for c in callback.on_node_start.call_args_list
        ]
        ended_ids = [c.kwargs["run_id"] for c in callback.on_node_end.call_args_list]
        assert sorted(ended_ids) == sorted(started_ids)
        assert callback.on_node_error.call_count == 0


class TestNestedChainRuns:
    """A nested chain inside a node inherits the node's langgraph_node
    metadata on start but carries only seq:step tags — it is not a node
    execution and must never open a duplicate row for the node. The orphan
    starts read as "—" rows that never close, and a run cancel stamped each
    one as a giant Stopped bar."""

    def test_nested_chain_start_is_skipped(self):
        """metadata.langgraph_node without a graph:step tag = nested chain."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=["seq:step:1"],
            metadata={"langgraph_node": "agent"},
        )

        mock_callback.on_node_start.assert_not_called()

    def test_node_run_with_graph_step_tag_is_kept(self):
        """A real node task run carries the graph:step tag pregel stamps."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=["graph:step:1"],
            metadata={"langgraph_node": "agent"},
        )

        mock_callback.on_node_start.assert_called_once()
        assert mock_callback.on_node_start.call_args[1]["node_name"] == "agent"

    def test_skipped_nested_run_never_emits_end_or_error(self):
        """A skipped start must drop that run's terminal events too — an
        end-only row is as much a ghost as a never-closing start."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            tags=["seq:step:1"],
            metadata={"langgraph_node": "agent"},
        )
        adapter.on_chain_end({}, run_id=run_id, tags=["seq:step:1"])
        adapter.on_chain_error(ValueError("boom"), run_id=run_id, tags=["seq:step:1"])

        mock_callback.on_node_start.assert_not_called()
        mock_callback.on_node_end.assert_not_called()
        mock_callback.on_node_error.assert_not_called()

    def test_router_evaluation_with_custom_tag_is_fully_dropped(self):
        """A nested run whose only tag is author-added must not leak an
        end-only row either: the start-skip tracks the run id."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            tags=["author_tag"],
            metadata={"langgraph_node": "agent"},
        )
        adapter.on_chain_end({}, run_id=run_id, tags=["author_tag"])

        mock_callback.on_node_start.assert_not_called()
        mock_callback.on_node_end.assert_not_called()


class TestCancellationClosesNodeAsInterrupted:
    """A cancelled graph surfaces asyncio.CancelledError at on_chain_error —
    plain cancels carry an empty message, and langgraph's background executor
    cancels leftovers with a bare object() sentinel whose str() is an opaque
    address. Neither is a node failure: the node was interrupted."""

    def test_terminal_events_carry_the_start_events_node_name(self):
        """LangChain Python passes no metadata to terminal callbacks: the node
        name is recovered from the start event by run_id, never read off the
        internal graph:step tag."""
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=["graph:step:2"],
            metadata={"langgraph_node": "tools"},
        )
        adapter.on_chain_end(
            {},
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=["graph:step:2"],
        )
        assert mock_callback.on_node_end.call_args[1]["node_name"] == "tools"

        adapter.on_chain_start(
            serialized={},
            inputs={},
            run_id=UUID("22345678-1234-5678-1234-567812345678"),
            tags=["graph:step:3"],
            metadata={"langgraph_node": "agent"},
        )
        adapter.on_chain_error(
            asyncio.CancelledError(),
            run_id=UUID("22345678-1234-5678-1234-567812345678"),
            tags=["graph:step:3"],
            metadata=None,
        )

        mock_callback.on_node_error.assert_not_called()
        assert mock_callback.on_node_interrupted.call_args[1]["node_name"] == "agent"

    def test_cancelled_error_routes_to_interrupt_not_error(self):
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)
        run_id = UUID("12345678-1234-5678-1234-567812345678")

        adapter.on_chain_error(
            error=asyncio.CancelledError(),
            run_id=run_id,
            tags=["graph:step:2"],
            metadata=None,
        )

        mock_callback.on_node_error.assert_not_called()
        mock_callback.on_node_interrupted.assert_called_once()
        call_kwargs = mock_callback.on_node_interrupted.call_args[1]
        assert call_kwargs["run_id"] == "12345678-1234-5678-1234-567812345678"

    def test_cancelled_error_with_sentinel_message_carries_no_opaque_text(self):
        mock_callback = MagicMock()
        adapter = LangGraphCallbackAdapter(mock_callback)

        adapter.on_chain_error(
            error=asyncio.CancelledError(object()),
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            tags=["graph:step:2"],
            metadata=None,
        )

        mock_callback.on_node_error.assert_not_called()
        mock_callback.on_node_interrupted.assert_called_once()

    def test_cancelled_error_falls_back_to_on_node_end_without_interrupt(self, caplog):
        """Callbacks predating on_node_interrupted close the timing with
        on_node_end (same compat pattern as on_node_suspend) plus a warning."""
        mock_callback = MagicMock(
            spec=["on_node_start", "on_node_end", "on_node_error"]
        )
        adapter = LangGraphCallbackAdapter(mock_callback)
        adapter_logger = "agent_engine_sdk_langgraph.node_logger_adapter"

        with caplog.at_level(logging.WARNING, logger=adapter_logger):
            adapter.on_chain_error(
                error=asyncio.CancelledError(),
                run_id=UUID("12345678-1234-5678-1234-567812345678"),
                tags=["graph:step:2"],
                metadata=None,
            )

        mock_callback.on_node_end.assert_called_once()
        mock_callback.on_node_error.assert_not_called()
        assert "does not implement on_node_interrupted" in caplog.text
