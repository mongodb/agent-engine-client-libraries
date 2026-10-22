"""LangGraph BaseAgent adapter.

Wraps a LangGraph CompiledGraph to implement the framework-neutral BaseAgent
protocol from agent-engine-sdk. Native vs durable branching lives on
``ExecutionSession``; this module stores hooks and drives invoke/stream.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from uuid import uuid4

from agent_engine_runner_shared.custom_events import custom_event_transport
from agent_engine_runner_shared.execution import AgentExecutionResult
from agent_engine_runner_shared.workflow import (
    DurableActivitySuspended,
    current_attempt_context,
)
from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    BaseAgent,
    ExecutionResult,
    Message,
    OutputParser,
    RequestContext,
    StreamEvent,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import (
    Command,
    Interrupt,
    Overwrite,
    StreamMode,
)
from pydantic import JsonValue

from agent_engine_runner_shared.workflow import (
    CancelledHistory,
    current_cancelled_history,
)
from agent_engine_sdk_langgraph.cancelled_turn import (
    commit_cancelled_turn as commit_cancelled_turn_checkpoint,
)
from agent_engine_sdk_langgraph.checkpoint_branch import (
    branch_point_from_metadata,
    checkpoint_metadata,
    initialize_checkpoint_branch,
)
from agent_engine_sdk_langgraph.deep_agent_checkpointer import (
    DeepAgentPlatformCheckpointer,
)
from agent_engine_sdk_langgraph.durable_message_identity import (
    install_durable_message_identity,
)
from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver
from agent_engine_sdk_langgraph.execution_session import (
    ExecutionSession,
    execution_session,
)
from agent_engine_sdk_langgraph.hooks import (
    GraphInput,
    PrepareAgentInput,
    ResolveThreadId,
)
from agent_engine_sdk_langgraph.messages import (
    find_last_ai_content,
    lc_messages_to_platform,
    lc_to_workflow_message,
)
from agent_engine_sdk_langgraph.output_parser import (
    DisabledCustomEventTransport,
    OutputParserBinding,
    emit_parser_stream_error,
)
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer
from agent_engine_sdk_langgraph.session_fork import wrap_session_fork_update_state
from agent_engine_sdk_langgraph.stream_scope import (
    PostTerminalDrain,
    aclose_stream_iter,
    anext_graph_item,
    scoped_anext,
)
from agent_engine_sdk_langgraph.subagent_stream import (
    SubagentStreamTracker,
    source_from_namespace,
)
from agent_engine_sdk_langgraph.suspend import suspend_event

__all__ = [
    "GraphInput",
    "InternalStreamError",
    "LangGraphBaseAgent",
    "PrepareAgentInput",
    "ResolveThreadId",
]

_logger = logging.getLogger(__name__)

_TURN_MESSAGE_ID_METADATA_KEY = "agent_engine_turn_message_id"


def _record_turn_message_id(ctx: RequestContext, message_id: str) -> None:
    """Record the turn-opening message id where the teardown hook reads it.

    ctx is the one object that reaches both the preamble and the teardown
    hook; the hook pops the key when it consumes it.
    """
    metadata = ctx.metadata if isinstance(ctx.metadata, dict) else None
    if metadata is None:
        try:
            ctx.metadata = {}
        except Exception:
            return  # immutable ctx: teardown appends a fresh opener
        metadata = ctx.metadata
    metadata[_TURN_MESSAGE_ID_METADATA_KEY] = message_id


def _stash_turn_message_id(ctx: RequestContext, graph_input: Any) -> None:
    """Record the turn-opening message id for the cancelled-turn commit at teardown.

    Identity, not prompt text: a prepare_input hook may rewrite content, so the
    teardown hook must not re-discover the message by comparing strings. When
    the builder left the opener id-less (hooks are not required to set ids),
    assign one first — LangGraph would only mint it inside the run, too late to
    capture here. A fresh turn always opens a new user message; a resume turn
    opens none, so the teardown commit recovers the opener already persisted
    instead (``cancelled_turn.commit_cancelled_turn``).
    """
    messages = graph_input.get("messages") if isinstance(graph_input, dict) else None
    if isinstance(messages, Overwrite):
        # A hook may wrap its list in Overwrite; the opener the graph persists
        # lives inside it.
        messages = messages.value
    if not isinstance(messages, list):
        return
    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue
        message_id = getattr(message, "id", None)
        if not message_id:
            try:
                message.id = f"input:{uuid4()}"
            except Exception:
                return  # unassignable stand-in: teardown appends a fresh opener
            message_id = message.id
        _record_turn_message_id(ctx, message_id)
        return


class InternalStreamError(Exception):
    """Stand-in passed to ``on_stream_error`` in place of the real exception.

    The real exception may be a platform-internal one (LangGraph, an LLM
    client, network/auth libraries) whose message was never meant to reach
    the customer-visible ``custom_event`` stream that ``on_stream_error``'s
    return value feeds into. The real exception is still logged and re-raised
    unchanged; only what the parser sees is replaced.
    """

    def __init__(self) -> None:
        super().__init__("Internal server error")


class _InterruptResume(Exception):
    """Continue a streamed graph after its session resolves an interrupt."""

    def __init__(self, command: Command) -> None:
        super().__init__("resume LangGraph interrupt")
        self.command = command


async def _failed_stream_epilogue(
    tracker: SubagentStreamTracker,
    parser: OutputParser | None,
    ctx: RequestContext,
) -> AsyncIterator[StreamEvent]:
    """Close open subagent spans, then let the author parser emit an error frame."""
    for event in tracker.drain_open_tasks():
        yield event
    if parser is not None:
        async for event in emit_parser_stream_error(parser, ctx):
            yield event


def _default_durable_subgraphs(
    graph: CompiledStateGraph,
) -> DurableSubgraphResolver | None:
    """Return the durable identity resolver a platform-checkpointed graph needs."""
    checkpointer = getattr(graph, "checkpointer", None)
    if not isinstance(checkpointer, PlatformCheckpointer):
        return None
    # Deep Agent routes each ToolCall through an internal Send. Its recorded
    # paths predate Send task segments, so it keeps them.
    if isinstance(checkpointer, DeepAgentPlatformCheckpointer):
        resolver = DurableSubgraphResolver(graph, scope_send_tasks=False)
        return resolver if resolver.has_compiled_children else None
    return DurableSubgraphResolver(graph)


class LangGraphBaseAgent(BaseAgent):
    """LangGraph adapter implementing the BaseAgent protocol.

    Wraps a LangGraph CompiledGraph and provides the framework-neutral
    invoke/stream interface that the AER expects.

    This adapter handles:
    - Converting AgentInput to LangGraph state format (with HumanMessage)
    - Converting Command(resume=...) for HITL resume flows
    - Streaming with token-level events via StreamEvent
    - Converting final results to AgentOutput

    Resume state travels on the RequestContext, not the caller payload:
    - ctx.resume: Boolean indicating resume vs fresh execution
    - ctx.resume_data: Data for Command(resume=...)
    - ctx.metadata["checkpoint_id"]: Checkpoint to resume from

    Example:
    ```python
        from langgraph.graph import StateGraph
        from agent_engine_sdk_langgraph import LangGraphBaseAgent

        # Build your graph
        graph = StateGraph(...)
        compiled = graph.compile(checkpointer=...)

        # Wrap it with optional callbacks
        agent = LangGraphBaseAgent(compiled, callbacks=[my_callback])

        # Use via BaseAgent protocol
        output = await agent.execute(ctx, agent_input)          # non-streaming
        async for event in agent.execute(ctx, agent_input):     # streaming
            ...
    ```
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        callbacks: list[Any] | None = None,
        prepare_input: PrepareAgentInput | None = None,
        output_parser: type[OutputParser] | None = None,
        use_custom_parser: bool = False,
        resolve_thread_id: ResolveThreadId | None = None,
        durable_subgraphs: DurableSubgraphResolver | None = None,
    ) -> None:
        """Initialize the adapter with a compiled LangGraph.

        Args:
            graph: A compiled LangGraph StateGraph with checkpointer
            callbacks: Optional list of LangChain callbacks for observability
            prepare_input: Optional tenant hook (``@app.prepare_agent_input``)
                that builds the graph input for a fresh invocation. When None,
                the default message-wrapping is used.
            output_parser: Optional ``OutputParser`` subclass
                (``@app.output_parser``) used to shape custom stream output.
            use_custom_parser: When true, ``stream()`` dual-runs ``output_parser``
                and emits its output as ``custom_event``, and Atlas Agent Engine also
                subscribes to LangGraph's ``custom`` channel so
                ``emit_custom_event`` reaches the stream (including when the
                parser does not list ``custom``). Requires a registered
                parser at deploy / ``get_agent`` time. When false the parser
                is inert and Atlas Agent Engine does not subscribe to ``custom``.
            resolve_thread_id: Optional tenant hook
                (``@app.resolve_thread_id``) that builds the LangGraph
                checkpoint ``thread_id``. When None, the adapter derives
                ``session_id:workspace_id``. When set, the return value is
                used verbatim (no workspace suffix). Appended after the
                established positional surface so existing positional
                callers keep binding ``output_parser`` / ``use_custom_parser``.
            durable_subgraphs: Resolver for compiled subgraphs and Send tasks.
                It is used only while an OE durable attempt is active. When
                None, a platform-checkpointed graph gets the default resolver.
        """
        self._graph = graph
        self._callbacks = callbacks or []
        self._prepare_input = prepare_input
        self._output_parser = output_parser
        self._use_custom_parser = use_custom_parser
        self._resolve_thread_id = resolve_thread_id
        self._durable_subgraphs = (
            durable_subgraphs
            if durable_subgraphs is not None
            else _default_durable_subgraphs(graph)
        )
        install_durable_message_identity(self._graph)
        # The platform maps live update_state to a new session+thread, not LangGraph
        # same-thread time-travel. See docs/session-fork.md.
        wrap_session_fork_update_state(
            self._graph, resolve_thread_id=self._resolve_thread_id
        )

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        return self._graph

    def execute(self, ctx: RequestContext, input: AgentInput) -> ExecutionResult:
        """Run the agent. Await for a result, or iterate for streaming."""
        return AgentExecutionResult(
            invoke=lambda: self.invoke(ctx, input),
            stream=lambda: self.stream(ctx, input),
        )

    @contextmanager
    def cancelled_history_scope(
        self, ctx: RequestContext, input: AgentInput
    ) -> Iterator[CancelledHistory]:
        """Expose cancellation capture without a runner dependency on LangGraph."""
        del ctx
        attempt = current_attempt_context()
        if attempt is None:
            raise RuntimeError("cancellation history requires an OE attempt")
        payload = input.payload if isinstance(input.payload, dict) else {}
        history = CancelledHistory(
            attempt,
            lc_to_workflow_message(
                HumanMessage(
                    content=str(payload.get("message", "")),
                    id=f"durable-input:{attempt.workflow_identity.execution_id}",
                )
            ),
        )
        token = current_cancelled_history.set(history)
        try:
            yield history
        finally:
            current_cancelled_history.reset(token)
            history.clear()
            checkpointer = self._graph.checkpointer
            if isinstance(checkpointer, PlatformCheckpointer):
                checkpointer.release_scratch(attempt)

    def _execution_session(self) -> ExecutionSession:
        return execution_session(
            self._graph,
            callbacks=self._callbacks,
            prepare_input=self._prepare_input,
            resolve_thread_id=self._resolve_thread_id,
            durable_subgraphs=self._durable_subgraphs,
            use_custom_parser=self._use_custom_parser,
        )

    async def commit_cancelled_turn(
        self, ctx: RequestContext, input: AgentInput
    ) -> "dict[str, str] | None":
        """Checkpoint the cancelled turn so session history can serve it.

        Invoked by runner-shared when an execution-cancelled drain tears down
        invoke/stream. No-op for durable sessions: their history is OE-owned
        and scratch checkpoints are discarded by design. The is_durable check
        is a backstop — the caller gates on the attempt it started, because by
        teardown the attempt context may already be closing.

        Returns the committed checkpoint's coordinate (``thread_id`` +
        ``checkpoint_id``) so the runner can report it to OE — the branch
        source selector needs it to fork from a session whose only checkpoint
        is this rescued marker — or ``None`` when nothing was committed.
        """
        session = self._execution_session()
        if session.is_durable:
            return None
        config = session.build_config(ctx)
        config.get("configurable", {}).pop("checkpoint_id", None)  # tip, not a pin
        checkpointer = self._graph.checkpointer
        saver = getattr(checkpointer, "native", None) or checkpointer
        if saver is None or not hasattr(saver, "aget_tuple"):
            return None  # stateless graph (no Mongo store): nothing to commit
        payload = input.payload if isinstance(input.payload, dict) else {}
        user_message = str(payload.get("message", "") or "")
        if not user_message:
            return None
        metadata = ctx.metadata if isinstance(ctx.metadata, dict) else None
        turn_message_id = (
            metadata.pop(_TURN_MESSAGE_ID_METADATA_KEY, None)
            if metadata is not None
            else None
        )
        thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
        await commit_cancelled_turn_checkpoint(
            saver,
            thread_id=thread_id,
            user_message=user_message,
            turn_message_id=(
                turn_message_id if isinstance(turn_message_id, str) else None
            ),
            node_names=self._graph.get_graph().nodes.keys(),
            resume=bool(ctx.resume),
        )
        # The synthetic tip IS the just-written checkpoint: read its id back
        # rather than threading it through the commit's message-id return.
        tip_reader: Any = saver  # the guard above established aget_tuple
        tip = await tip_reader.aget_tuple({"configurable": {"thread_id": thread_id}})
        if tip is None:
            return None
        return {
            "thread_id": thread_id,
            "checkpoint_id": str(tip.checkpoint["id"]),
        }

    async def _prepare_run(
        self,
        ctx: RequestContext,
        input: AgentInput,
        session: ExecutionSession,
    ) -> tuple[Any, RunnableConfig]:
        """Shared invoke/stream preamble: input, config, branch, fence, seed."""
        graph_input = session.build_graph_input(ctx, input)
        if not ctx.resume:
            _stash_turn_message_id(ctx, graph_input)
        config = session.build_config(ctx)
        # HITL resume continues a thread that already received any branch copy.
        # Clients often still send langgraph_branch_point; do not copy again.
        if not ctx.resume:
            if source := branch_point_from_metadata(ctx.metadata):
                await initialize_checkpoint_branch(self._graph, config, source)
        if ctx.previous_execution_cancelled and not ctx.resume:
            await session.fence_cancelled_predecessor_writes(ctx)
        await session.seed_previous_session_state(config)
        return graph_input, config

    async def invoke(self, ctx: RequestContext, input: AgentInput) -> AgentOutput:
        """Execute the agent and return the final result."""
        session = self._execution_session()
        try:
            with custom_event_transport(DisabledCustomEventTransport()):
                with session.invoke_scopes():
                    return await self._invoke_inner(ctx, input, session)
        finally:
            if session.is_durable:
                session.release_scratch()

    async def _invoke_inner(
        self, ctx: RequestContext, input: AgentInput, session: ExecutionSession
    ) -> AgentOutput:
        graph_input, config = await self._prepare_run(ctx, input, session)
        is_resume = ctx.resume
        thread_id = config["configurable"]["thread_id"]  # type: ignore[index]

        # A durable replacement attempt first replays the graph until the
        # native interrupt reappears with its new id. The session matches it to
        # the stable OE activity position. Only an already-resolved interrupt
        # returns Command(resume=...) and invokes the graph again; a public
        # suspension returns to the caller, while no interrupt completes it.
        while True:
            result = await self._graph.ainvoke(
                graph_input, config, **session.checkpoint_durability()
            )

            messages = result.get("messages", [])
            response = find_last_ai_content(messages)
            interrupt = await session.invoke_interrupt(
                ctx=ctx,
                config=config,
                response=response,
                messages=messages,
            )
            if isinstance(interrupt, Command):
                graph_input = interrupt
                continue
            if interrupt is not None:
                return interrupt
            break

        await session.complete_execution()
        checkpoint = await session.persisted_checkpoint_coordinate(thread_id)
        return AgentOutput(
            response=cast(
                JsonValue,
                {
                    "response": response,
                    "execution_id": thread_id,
                    "status": "completed",
                    "message_count": len(messages),
                    "resumed": is_resume,
                    "metadata": checkpoint_metadata(checkpoint),
                },
            ),
        )

    async def resume(
        self, ctx: RequestContext, input: AgentInput, decision: str
    ) -> AgentOutput:
        """Resume a suspended agent with a human decision.

        Convenience method that sets up resume value fields and delegates to invoke().
        ``resume`` is the HITL-continuation bit (Command vs a new HumanMessage,
        skip branch copy). ``resume_data`` is the decision payload. Presence of
        data cannot imply the flag: a resume with a missing payload must fail
        closed, not look like a fresh turn.
        """
        resume_ctx = ctx.model_copy(update={"resume": True, "resume_data": decision})
        return await self.invoke(resume_ctx, input)

    async def stream(
        self, ctx: RequestContext, input: AgentInput
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution with token-level events.

        Handles both fresh executions and resume flows based on the context:
        - ctx.resume: Boolean indicating resume vs fresh execution
        - ctx.resume_data: Data for Command(resume=...)
        - ctx.metadata["checkpoint_id"]: Checkpoint to resume from

        Yields StreamEvent objects with event:
        - "token": Streaming LLM token (data["content"], data["source"],
              data["tool_call_id"])
        - "subagent_start": Subagent dispatch begins (data["source"],
              data["subagent_name"], data["tool_call_id"], data["description"])
        - "subagent_end": Subagent dispatch completes (data["source"],
              data["subagent_name"], data["tool_call_id"], data["summary"]
              with the subagent's final response text; empty string for
              defensive drain ends)
        - "result": Final completion (data with response + messages)
        - "suspend": HITL interrupt (data with suspend_payload + opaque metadata)

        On durable sessions the attempt's scratch is released when the
        stream ends.
        """
        session = self._execution_session()
        stream = self._stream_inner(ctx, input, session)
        try:
            while True:
                try:
                    # Bind custom-event + compiled-path ContextVars around this
                    # one step. A generator-lifetime scope can set them in one
                    # asyncio task and reset them in another.
                    event = await scoped_anext(stream, session)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            if session.is_durable:
                session.release_scratch()

    async def _stream_inner(
        self,
        ctx: RequestContext,
        input: AgentInput,
        session: ExecutionSession,
    ) -> AsyncIterator[StreamEvent]:
        """Prepare once, then replay resolved native interrupts in scratch."""
        graph_input, config = await self._prepare_run(ctx, input, session)
        all_messages: list[Any] = []
        tracker = SubagentStreamTracker()
        binding = OutputParserBinding.create(
            self._output_parser, use_custom_parser=self._use_custom_parser
        )
        while True:
            try:
                async for event in self._stream_iteration(
                    ctx,
                    session,
                    graph_input,
                    config,
                    all_messages,
                    tracker,
                    binding,
                ):
                    yield event
            except _InterruptResume as resume:
                graph_input = resume.command
                continue
            break

    async def _stream_iteration(
        self,
        ctx: RequestContext,
        session: ExecutionSession,
        graph_input: Any,
        config: RunnableConfig,
        all_messages: list[Any],
        tracker: SubagentStreamTracker,
        binding: OutputParserBinding,
    ) -> AsyncIterator[StreamEvent]:
        """Run one graph input or interrupt-resume continuation."""
        is_resume = ctx.resume
        thread_id = config["configurable"]["thread_id"]  # type: ignore[index]

        # Accumulated across all updates payloads — a subagent (subgraph)
        # interrupt that arrives mid-stream must not be overwritten by a later
        # root-namespace updates payload.
        captured_interrupts: list[Interrupt] = []

        # ``subgraphs=True`` is required for LangGraph to emit messages-mode
        # chunks from subagent (nested) graphs; otherwise the messages handler
        # filters them out and sourced tokens are silently dropped.
        graph_stream = self._graph.astream(
            graph_input,
            config,
            stream_mode=cast("list[StreamMode]", list(binding.stream_modes)),
            subgraphs=True,
            **session.checkpoint_durability(),
        )
        stream_iter = graph_stream.__aiter__()
        drain = PostTerminalDrain()
        try:
            try:
                while True:
                    item = await anext_graph_item(
                        stream_iter,
                        drain_timeout=drain.next_uses_timeout(
                            enabled=session.drain_after_terminal
                        ),
                    )
                    if item is None:
                        break
                    drain.received_item()
                    namespace = item.namespace
                    stream_mode = item.stream_mode
                    payload = item.payload

                    # CUSTOM_EVENT frames only. Runs the author parser when it
                    # listed this stream_mode; otherwise forwards LangGraph
                    # ``custom`` (emit_custom_event) when Atlas Agent Engine subscribed.
                    # Atlas Agent Engine token / updates handling is the branches below.
                    async for event in binding.iter_custom_events(
                        stream_mode=stream_mode,
                        namespace=namespace,
                        payload=payload,
                        ctx=ctx,
                    ):
                        yield event

                    if stream_mode == "messages":
                        # Payload is ``(chunk, metadata)``. ``chunk`` is an
                        # ``AIMessageChunk``: token text, and on the root it
                        # may also carry a streaming ``task`` tool_call that
                        # opens a subagent. ``metadata`` is unused; subagent
                        # ``source`` comes from ``namespace``, not the chunk.
                        chunk, _ = payload
                        source = source_from_namespace(namespace)
                        for event in tracker.process_messages_chunk(chunk, source):
                            yield event
                    elif stream_mode == "updates":
                        # Payload is ``{node_name: node_output, ...}``,
                        # optionally with ``__interrupt__: list|tuple``.
                        # ``node_output`` is usually ``{"messages": [...]}``
                        # (pregel unwraps Commands before streaming). A raw
                        # ``Command`` is only a defensive fallback. A parent
                        # ``task`` tool_call in those messages opens a
                        # subagent; a ``ToolMessage`` with that id closes it.
                        #
                        # LangGraph's contract is a dict. A non-dict is not a
                        # documented shape — skip it so ``.values()`` cannot
                        # abort the rest of the stream.
                        if not isinstance(payload, dict):
                            _logger.warning(
                                "Unexpected non-dict updates payload: %s",
                                type(payload).__name__,
                            )
                            continue
                        interrupts = payload.get("__interrupt__")
                        # LangGraph emits this as a tuple in production; older
                        # docs/examples imply a list. Accept both.
                        if isinstance(interrupts, (list, tuple)):
                            captured_interrupts.extend(interrupts)
                        for node_output in payload.values():
                            if (
                                isinstance(node_output, dict)
                                and "messages" in node_output
                            ):
                                for event in tracker.process_add_messages_update(
                                    node_output["messages"],
                                    all_messages,
                                ):
                                    yield event
                            elif isinstance(node_output, Command):
                                for event in tracker.process_command_update(
                                    node_output, all_messages
                                ):
                                    yield event
                        drain.consider(
                            tracker.is_terminal_agent_update(
                                namespace,
                                payload,
                                all_messages,
                                captured_interrupts,
                            )
                        )
            finally:
                await aclose_stream_iter(stream_iter)

        except (GeneratorExit, asyncio.CancelledError):
            # Yielding here raises RuntimeError and masks the cancellation.
            # ``CancelledError`` is a ``BaseException``, so it must be caught
            # before ``except Exception``.
            raise
        except DurableActivitySuspended as error:
            cause = error.__cause__
            if cause is None:
                # OE accepted the wait. This is a session boundary, not a
                # failed stream — durable resume starts a new fenced attempt.
                yield suspend_event(
                    {
                        "suspend_reason": error.reason,
                        "suspend_context": error.context,
                    },
                    resumed=is_resume,
                    messages=lc_messages_to_platform(all_messages),
                    metadata={},
                )
                return
            # OE rejected the wait. Close spans, then raise that rejection.
            async for event in _failed_stream_epilogue(tracker, binding.parser, ctx):
                yield event
            raise cause
        except Exception:
            async for event in _failed_stream_epilogue(tracker, binding.parser, ctx):
                yield event
            raise
        interrupt = await session.stream_interrupt(
            ctx=ctx,
            config=config,
            captured=captured_interrupts,
            messages=lc_messages_to_platform(all_messages),
        )
        if isinstance(interrupt, Command):
            raise _InterruptResume(interrupt)
        if interrupt is not None:
            yield interrupt
            return

        # Stream finished. Close leftover subagent spans unless HITL paused the
        # graph — those tasks continue above on replay or on the next attempt.
        for event in tracker.drain_open_tasks():
            yield event

        if not all_messages:
            raise RuntimeError("Graph completed without producing messages")

        response = find_last_ai_content(all_messages)
        platform_messages: list[Message] = lc_messages_to_platform(all_messages)

        await session.complete_execution()
        checkpoint = await session.persisted_checkpoint_coordinate(thread_id)
        yield StreamEvent(
            data=cast(
                JsonValue,
                {
                    "response": response,
                    "messages": [m.model_dump() for m in platform_messages],
                    "message_count": len(all_messages),
                    "resumed": is_resume,
                    "metadata": checkpoint_metadata(checkpoint),
                },
            ),
            event="result",
        )
