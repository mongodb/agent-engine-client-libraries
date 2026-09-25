"""Native vs durable execution captured once per invoke/stream.

``LangGraphBaseAgent`` constructs an ``ExecutionSession`` from
``current_attempt_context()`` at the start of a run and then asks the
session — it does not re-read the ContextVar. Native sessions own
``Command(resume=...)``, checkpoint targeting, interrupt handling, the
cancelled-run fence, and the 1s post-terminal drain. Durable sessions rebuild
the original turn input, reject targeting, force sync durability, own scratch
lifecycle, and map native interrupts onto OE activities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Durability, Interrupt
from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    Message,
    RequestContext,
    StreamEvent,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import (
    current_attempt_context,
    operation_path_resolver_scope,
)

from agent_engine_sdk_langgraph.checkpoint_branch import (
    LangGraphCheckpoint,
    langgraph_checkpoint_for_thread,
    latest_checkpoint_id,
    normalize_checkpoint_id,
    pending_interrupts_from_state,
)
from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver
from agent_engine_sdk_langgraph.hooks import PrepareAgentInput, ResolveThreadId
from agent_engine_sdk_langgraph.durable_errors import UnsupportedDurableGraphError
from agent_engine_sdk_langgraph.platform_checkpointer import (
    PlatformCheckpointer,
    prune_cancelled_predecessor_writes,
)
from agent_engine_sdk_langgraph.session_fork import take_continue_without_user_message
from agent_engine_sdk_langgraph.suspend import (
    invoke_suspend_output,
    stream_hitl_suspend_event,
)
from agent_engine_sdk_langgraph.thread_id import checkpoint_thread_id

__all__ = ["ExecutionSession", "NativeSession", "execution_session"]


class DurableCheckpointKwargs(TypedDict, total=False):
    durability: Durability


class ExecutionSession(ABC):
    """One invoke/stream attempt: native Mongo checkpoints or OE durable."""

    is_durable: bool
    drain_after_terminal: bool

    def __init__(
        self,
        graph: Any,
        *,
        callbacks: list[Any],
        prepare_input: PrepareAgentInput | None,
        resolve_thread_id: ResolveThreadId | None,
        durable_subgraphs: DurableSubgraphResolver | None,
        attempt: AttemptContext | None,
        use_custom_parser: bool = False,
    ) -> None:
        """Bind collaborators for one invoke/stream.

        graph: compiled LangGraph this run calls.
        callbacks: LangChain callbacks copied onto ``RunnableConfig``.
        prepare_input: ``@app.prepare_agent_input``; ``None`` wraps the payload
            as ``{messages: [HumanMessage]}``.
        resolve_thread_id: ``@app.resolve_thread_id``; ``None`` uses the
            workspace-scoped session id as LangGraph ``thread_id``.
        durable_subgraphs: compiled-subgraph resolver from the App factory.
            Native sessions ignore it. Durable sessions map active compiled
            children onto OE operation paths.
        attempt: bound OE ``AttemptContext``. ``execution_session()`` passes
            ``None`` for NativeSession and a real attempt for DurableSession.
        use_custom_parser: workspace opted into ``@app.output_parser`` /
            ``emit_custom_event``. Native and durable sessions both honor it.
        """
        self.graph = graph
        self.callbacks = callbacks
        self.prepare_input = prepare_input
        self._resolve_thread_id = resolve_thread_id
        self.durable_subgraphs = durable_subgraphs
        self.attempt = attempt
        self.use_custom_parser = use_custom_parser

    def resolve_checkpoint_thread_id(self, ctx: RequestContext) -> str:
        return checkpoint_thread_id(ctx, self._resolve_thread_id)

    def _base_config(self, ctx: RequestContext) -> RunnableConfig:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": self.resolve_checkpoint_thread_id(ctx),
                "request_context": ctx,
            }
        }
        if self.callbacks:
            config["callbacks"] = list(self.callbacks)
        return config

    def _fresh_graph_input(self, ctx: RequestContext, input: AgentInput) -> Any:
        if self.prepare_input is not None:
            return self.prepare_input(input, ctx)
        if not isinstance(input.payload, dict):
            raise TypeError(f"Expected dict payload, got {type(input.payload)}")
        input_payload: dict[str, Any] = input.payload
        message = str(input_payload.get("message", ""))
        # Durable default identity is unindexed ``durable-input:{execution_id}``.
        # Hook-built lists use indexed ids in ``assign_stable_execution_message_ids``.
        human_message = (
            HumanMessage(
                content=message,
                id=f"durable-input:{self.attempt.workflow_identity.execution_id}",
            )
            if self.attempt is not None
            else HumanMessage(content=message)
        )
        graph_input: dict[str, Any] = {
            "messages": [human_message],
        }
        if ctx.user_id:
            graph_input["user_id"] = ctx.user_id
        if ctx.session_id:
            graph_input["session_id"] = ctx.session_id
        return graph_input

    def _explicit_checkpoint_id(self, ctx: RequestContext) -> str | None:
        """Real LangGraph checkpoint id from request metadata, if any.

        JSON/protobuf often stringify a missing id as ``"None"``;
        ``normalize_checkpoint_id`` treats that as absent so we keep following
        the thread's latest checkpoint instead of looking up a bogus id.

        ``thread:{id}`` is an Atlas Agent Engine placeholder meaning "latest on this
        thread", not a LangGraph checkpoint id. Pinning it would miss and
        fork a new checkpoint root.
        """
        raw = normalize_checkpoint_id((ctx.metadata or {}).get("checkpoint_id"))
        if raw is None or raw.startswith("thread:"):
            return None
        return raw

    def subgraph_resolver(self) -> DurableSubgraphResolver | None:
        """Compiled-subgraph resolver for this run, or ``None``.

        Native sessions return ``None``: Mongo checkpoints do not need OE
        operation paths. Durable sessions return the factory-built resolver
        after validating any compiled subgraphs.
        """
        return None

    def checkpoint_durability(self) -> DurableCheckpointKwargs:
        """Extra kwargs for ``graph.ainvoke`` / ``astream``.

        Native sessions return ``{}`` and keep LangGraph's default async
        checkpoint writes. Durable sessions return ``{"durability": "sync"}``
        so FinalizeStep finishes before the next superstep admits activities.
        """
        return {}

    @contextmanager
    def invoke_scopes(self) -> Iterator[None]:
        resolver = self.subgraph_resolver()
        if resolver is None:
            yield
            return
        with operation_path_resolver_scope(resolver):
            yield

    def release_scratch(self) -> None:
        """Release attempt-local checkpoints; native sessions have none."""
        if self.attempt is None:
            return
        checkpointer = getattr(self.graph, "checkpointer", None)
        if isinstance(checkpointer, PlatformCheckpointer):
            checkpointer.release_scratch(self.attempt)

    async def seed_previous_session_state(self, config: RunnableConfig) -> None:
        """Restore durable state before input; native sessions use Mongo directly."""
        if self.attempt is None:
            return
        checkpointer = getattr(self.graph, "checkpointer", None)
        if not isinstance(checkpointer, PlatformCheckpointer):
            raise UnsupportedDurableGraphError(
                "durable execution requires the platform checkpointer"
            )
        await checkpointer.seed_previous_session_state(self.attempt, config)

    async def complete_execution(self) -> None:
        """Commit durable state after graph completion; native sessions need no hook."""
        if self.attempt is None:
            return
        checkpointer = getattr(self.graph, "checkpointer", None)
        if not isinstance(checkpointer, PlatformCheckpointer):
            raise UnsupportedDurableGraphError(
                "durable execution requires the platform checkpointer"
            )
        await checkpointer.complete_execution(self.attempt)

    @abstractmethod
    def build_graph_input(self, ctx: RequestContext, input: AgentInput) -> Any: ...

    @abstractmethod
    def build_config(self, ctx: RequestContext) -> RunnableConfig: ...

    @abstractmethod
    async def fence_cancelled_predecessor_writes(self, ctx: RequestContext) -> None: ...

    @abstractmethod
    async def persisted_checkpoint_coordinate(
        self, thread_id: str
    ) -> LangGraphCheckpoint | None:
        """Mongo thread coordinate advertised in native output metadata.

        Durable runs still checkpoint into attempt-local scratch. This is the
        persisted ``(thread_id, checkpoint_id)`` pair used for native
        branch/resume, not "whether a checkpoint exists."
        """
        ...

    @abstractmethod
    async def invoke_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        response: str,
        messages: list[Any],
    ) -> Command | AgentOutput | None:
        """Continue, suspend, or finish after one graph invocation."""
        ...

    @abstractmethod
    async def stream_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        captured: list[Interrupt],
        messages: list[Message],
    ) -> Command | StreamEvent | None:
        """Continue, suspend, or finish after one graph stream."""
        ...


class NativeSession(ExecutionSession):
    """Mongo-checkpointed session (no OE AttemptContext).

    A fresh turn is a state dict. Atlas Agent Engine HITL (``ctx.resume``) is
    ``Command(resume=...)`` so LangGraph continues from the interrupt instead
    of appending a new HumanMessage. Config may pin a real ``checkpoint_id``
    for branch/replay targeting. After a cancelled predecessor, leftover
    pending writes are fenced before this turn runs. ``drain_after_terminal``
    waits 1s so async checkpoint puts land before the process exits.
    """

    is_durable = False
    drain_after_terminal = True

    def build_graph_input(self, ctx: RequestContext, input: AgentInput) -> Any:
        if ctx.resume:
            if ctx.resume_data is None:
                raise ValueError(
                    "ctx.resume is True but resume_data is missing. "
                    "Provide resume_data to resume a suspended graph."
                )
            return Command(resume=ctx.resume_data)
        payload = input.payload if isinstance(input.payload, dict) else {}
        # See docs/session-fork.md (first dest invoke after in-app update_state).
        first_dest_continue = take_continue_without_user_message(
            self.resolve_checkpoint_thread_id(ctx)
        )
        if first_dest_continue and not str(payload.get("message", "")).strip():
            return None
        return self._fresh_graph_input(ctx, input)

    def build_config(self, ctx: RequestContext) -> RunnableConfig:
        config = self._base_config(ctx)
        checkpoint_id = self._explicit_checkpoint_id(ctx)
        if checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = checkpoint_id  # type: ignore[index]
        return config

    async def fence_cancelled_predecessor_writes(self, ctx: RequestContext) -> None:
        """Discard pending checkpoint writes a cancelled predecessor left.

        The cancelled run cannot clean this up itself: OE kills its AER pod
        mid-superstep, so any task writes already attached to the thread's
        latest checkpoint stay there. This next live invoke is the first
        process that can see them; without the fence LangGraph would fold
        those half-finished writes into this turn's first step.

        Scoped to the latest checkpoint per namespace so an older,
        still-suspended execution's interrupt/resume writes are untouched.
        Resumes never carry the cancelled-predecessor flag.
        """
        if not ctx.previous_execution_cancelled or ctx.resume:
            return
        saver = getattr(self.graph, "checkpointer", None)
        if saver is None:
            return
        thread_id = self.resolve_checkpoint_thread_id(ctx)
        if isinstance(saver, PlatformCheckpointer):
            await saver.fence_cancelled_predecessor_writes(thread_id)
            return
        await prune_cancelled_predecessor_writes(saver, thread_id)

    async def persisted_checkpoint_coordinate(
        self, thread_id: str
    ) -> LangGraphCheckpoint | None:
        return await langgraph_checkpoint_for_thread(self.graph, thread_id)

    async def invoke_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        response: str,
        messages: list[Any],
    ) -> Command | AgentOutput | None:
        state = await self.graph.aget_state(config)
        pending = pending_interrupts_from_state(state)
        if not pending and not state.next:
            return None
        thread_id = config["configurable"]["thread_id"]  # type: ignore[index]
        return invoke_suspend_output(
            response=response,
            thread_id=thread_id,
            checkpoint_id=await latest_checkpoint_id(self.graph, thread_id),
            interrupts=pending,
            message_count=len(messages),
            resumed=ctx.resume,
        )

    async def stream_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        captured: list[Interrupt],
        messages: list[Message],
    ) -> Command | StreamEvent | None:
        if not captured:
            return None
        thread_id = config["configurable"]["thread_id"]  # type: ignore[index]
        return stream_hitl_suspend_event(
            captured,
            checkpoint_id=await latest_checkpoint_id(self.graph, thread_id),
            resumed=ctx.resume,
            messages=messages,
        )


def execution_session(
    graph: Any,
    *,
    callbacks: list[Any],
    prepare_input: PrepareAgentInput | None,
    resolve_thread_id: ResolveThreadId | None,
    durable_subgraphs: DurableSubgraphResolver | None,
    use_custom_parser: bool = False,
) -> ExecutionSession:
    """Capture native vs durable once from the bound attempt context."""
    attempt = current_attempt_context()
    if attempt is None:
        return NativeSession(
            graph,
            callbacks=callbacks,
            prepare_input=prepare_input,
            resolve_thread_id=resolve_thread_id,
            durable_subgraphs=durable_subgraphs,
            attempt=None,
            use_custom_parser=use_custom_parser,
        )
    # Imported only for the durable branch so durable_session can subclass the
    # shared ExecutionSession without creating an import cycle.
    from agent_engine_sdk_langgraph.durable_session import DurableSession

    return DurableSession(
        graph,
        callbacks=callbacks,
        prepare_input=prepare_input,
        resolve_thread_id=resolve_thread_id,
        durable_subgraphs=durable_subgraphs,
        attempt=attempt,
        use_custom_parser=use_custom_parser,
    )
