"""Request-scoped platform checkpointer for native vs durable_workflow sessions.

One compiled graph can serve both routes concurrently: an absent attempt
context delegates to the existing MongoDB saver, while an OE-issued
``AttemptContext`` selects attempt-local in-memory scratch keyed by
execution + attempt + fence. OE activity history — never this scratch — is
the durable replay source.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt, Send

from agent_engine_runner_shared.context import get_current_oe_url
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    OperationPath,
    OperationPathSegment,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    FinalizeStepCommand,
    StateSnapshot,
)
from agent_engine_runner_shared.workflow import (
    AsyncWorkflowClient,
    WorkflowClient,
    complete_execution_command,
    current_attempt_context,
    finalize_current_step_command,
)
from agent_engine_runner_shared.workflow.activity import (
    build_activity_command,
)
from agent_engine_runner_shared.workflow.context import (
    InterruptedActivity,
    advance_step_ordinal,
    current_operation_path,
    current_step_ordinal,
    interrupted_activities,
    record_interrupted_activity,
    record_observed_activity,
)

from .durable_errors import UnsupportedDurableGraphError
from .workflow_state import (
    channel_values_to_state_snapshot,
    state_snapshot_to_channel_values,
)

__all__ = [
    "PlatformCheckpointer",
    "UnsupportedDurableGraphError",
    "scratch_thread_id",
]

_logger = logging.getLogger(__name__)

_REJECTED_PUT_SOURCES = frozenset({"update", "fork"})
# Absolute OE step ordinal recorded on committed root-loop scratch
# checkpoints so session-fork target selection never reconstructs LangGraph's
# seeded relative `step` offset. The key is persisted in the checkpoint store
# and shared with the TypeScript SDK; the string is part of the stored
# checkpoint format, so changing it invalidates forks of older checkpoints.
OE_STEP_ORDINAL_METADATA_KEY = "agent_engine_oe_step_ordinal"
_DIRECT_INTERRUPT_ACTIVITY_NAME = "langgraph.interrupt"
_PREGEL_TASKS_CHANNEL = "__pregel_tasks"


def is_direct_interrupt_activity(activity: InterruptedActivity) -> bool:
    """Return whether an activity was synthesized for a direct node interrupt.

    Root interrupts contain only the ``langgraph.task`` segment. Child
    interrupts prefix that segment with their stable compiled-child path.
    """
    segments = activity.command.position.operation_path.segments
    return (
        activity.command.activity_name == _DIRECT_INTERRUPT_ACTIVITY_NAME
        and bool(segments)
        and segments[-1].name.startswith("langgraph.task:")
    )


def scratch_thread_id(context: AttemptContext) -> str:
    """Return the scratch thread id isolating overlapping fenced attempts."""
    execution_id = context.workflow_identity.execution_id
    if not execution_id:
        raise UnsupportedDurableGraphError(
            "durable scratch requires AttemptContext.workflow_identity.execution_id"
        )
    if not context.attempt_id:
        raise UnsupportedDurableGraphError(
            "durable scratch requires AttemptContext.attempt_id"
        )
    if context.fencing_token <= 0:
        # OE-issued fences start at 1; a defaulted token would collide
        # scratch across attempts and break isolation.
        raise UnsupportedDurableGraphError(
            "durable scratch requires a positive AttemptContext.fencing_token"
        )
    return f"{execution_id}:{context.attempt_id}:{context.fencing_token}"


async def prune_cancelled_predecessor_writes(saver: Any, thread_id: str) -> None:
    """Discard pending checkpoint writes a cancelled predecessor left on ``saver``.

    A cancelled run's pod is killed mid-superstep, so writes its tasks had
    already produced stay attached to the thread's latest checkpoint and
    LangGraph would fold them into this fresh turn's first step. The run was
    deliberately stopped: its half-finished superstep must contribute nothing.
    Scoped to the latest checkpoint per namespace so an older, still-suspended
    execution's interrupt/resume writes are untouched.

    ``saver`` is the Mongo-backed native saver (or a duck-typed stand-in).
    Hygiene only: lookup or delete failures are logged and never raised.
    """
    ckpt_col = getattr(saver, "checkpoint_collection", None)
    writes_col = getattr(saver, "writes_collection", None)
    if ckpt_col is None or writes_col is None:
        # A saver exists but doesn't expose the Mongo collections — the
        # fence can't run, and silence here would hide a regression like
        # a saver/wrapper shape change disabling it.
        _logger.warning(
            "Cannot fence a cancelled run's pending writes: checkpointer "
            "%s exposes no Mongo collections",
            type(saver).__name__,
        )
        return

    def _prune() -> int:
        removed = 0
        for ns in ckpt_col.distinct("checkpoint_ns", {"thread_id": thread_id}):
            latest = ckpt_col.find_one(
                {"thread_id": thread_id, "checkpoint_ns": ns},
                sort=[("checkpoint_id", -1)],
                projection={"checkpoint_id": 1},
            )
            if not latest:
                continue
            result = writes_col.delete_many(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": latest["checkpoint_id"],
                }
            )
            removed += result.deleted_count
        return removed

    try:
        removed = await asyncio.to_thread(_prune)
    except Exception:
        # Hygiene, not correctness of this turn's input: never fail the
        # invoke over it.
        _logger.warning(
            "Failed to fence a cancelled run's pending writes (thread %s)",
            thread_id,
            exc_info=True,
        )
        return
    if removed:
        _logger.info(
            "Discarded %d pending checkpoint writes of a cancelled run (thread %s)",
            removed,
            thread_id,
        )


class PlatformCheckpointer(BaseCheckpointSaver[str]):
    """Delegating checkpointer selected per invocation from OE attempt context.

    Application code keeps calling ``app.checkpointer()`` once at compile time;
    routing happens on each get/put from the request-scoped ``contextvars``.
    """

    def __init__(
        self,
        *,
        native: BaseCheckpointSaver[Any] | None,
        scratch: InMemorySaver | None = None,
    ) -> None:
        super().__init__()
        self.native = native
        self._scratch = scratch if scratch is not None else InMemorySaver()

    def _reject_unsupported_send(self, writes: Sequence[tuple[str, Any]]) -> None:
        """Reject application-authored dynamic fan-out in durable graphs."""
        for channel, value in writes:
            if channel != _PREGEL_TASKS_CHANNEL:
                continue
            packets = value if isinstance(value, (list, tuple)) else (value,)
            if any(isinstance(packet, Send) for packet in packets):
                raise UnsupportedDurableGraphError(
                    "LangGraph Send is not supported on durable_workflow sessions; "
                    "use fixed graph edges, compiled subgraphs, or Deep Agent task "
                    "delegation"
                )

    def _reject_checkpoint_send(self, checkpoint: Checkpoint) -> None:
        """Reject Send where LangGraph synchronously commits the producer step.

        LangGraph backgrounds ``put_writes`` for this channel and can start a
        PUSH worker before that callback fails. Under sync durability, ``put``
        is the blocking boundary before the next step begins.
        """
        channel_values = checkpoint.get("channel_values") or {}
        candidates = (
            channel_values.get(_PREGEL_TASKS_CHANNEL),
            checkpoint.get("pending_sends"),
        )
        self._reject_unsupported_send(
            [
                (_PREGEL_TASKS_CHANNEL, candidate)
                for candidate in candidates
                if candidate
            ]
        )

    # ----- scratch lifecycle -----

    def release_scratch(self, context: AttemptContext) -> None:
        """Discard attempt-local scratch for one OE-issued context."""
        self._scratch.delete_thread(scratch_thread_id(context))

    async def seed_previous_session_state(
        self,
        context: AttemptContext,
        config: RunnableConfig,
    ) -> None:
        """Make OE's last committed session state this attempt's starting point.

        The state is written as an initial scratch checkpoint instead of being
        merged into the new invocation input. LangGraph therefore applies the
        application's channel reducers when it combines the previous turn with
        the current request.
        """
        if not context.HasField("previous_state"):
            return

        durable_config = self._durable_config(config, context)
        if await self._scratch.aget_tuple(durable_config) is not None:
            return

        checkpoint = empty_checkpoint()
        channel_values = state_snapshot_to_channel_values(context.previous_state)
        # Mark restored channels as initialized using the saver's native version
        # format. This is LangGraph checkpoint metadata, not an OE session revision.
        version = self._scratch.get_next_version(None, None)
        channel_versions: ChannelVersions = dict.fromkeys(channel_values, version)
        checkpoint["channel_values"] = channel_values
        checkpoint["channel_versions"] = channel_versions
        await self._scratch.aput(
            durable_config,
            checkpoint,
            {"source": "input", "step": -1, "parents": {}},
            channel_versions,
        )

    async def fence_cancelled_predecessor_writes(self, thread_id: str) -> None:
        """Discard pending writes a cancelled predecessor left on the native saver.

        Durable sessions recover from the turn base, markers, and recorded
        activity outcomes, so cancelled-attempt scratch is discarded elsewhere
        and this method is a no-op when there is no Mongo ``native`` saver
        (stateless local dev).
        """
        if self.native is None:
            # Stateless graph (no Mongo URI): nothing durable to fence.
            return
        await prune_cancelled_predecessor_writes(self.native, thread_id)

    @staticmethod
    def _oe_url() -> str:
        oe_url = get_current_oe_url()
        if not oe_url:
            raise UnsupportedDurableGraphError(
                "durable workflow state requires the OE callback URL"
            )
        return oe_url

    async def _final_state(self, context: AttemptContext) -> StateSnapshot:
        checkpoint = await self._scratch.aget_tuple(
            {"configurable": {"thread_id": scratch_thread_id(context)}}
        )
        if checkpoint is None:
            raise UnsupportedDurableGraphError(
                "durable execution completed without final application state"
            )
        return channel_values_to_state_snapshot(
            dict(checkpoint.checkpoint.get("channel_values") or {}),
            include_legacy_source=context.replay_mode,
        )

    async def complete_execution(self, context: AttemptContext) -> None:
        """Commit this attempt's final state for the next turn."""
        state = await self._final_state(context)
        async with AsyncWorkflowClient(self._oe_url()) as client:
            await client.complete_execution(complete_execution_command(context, state))

    def _prepare_step_finalization(
        self,
        context: AttemptContext,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        config: RunnableConfig,
    ) -> FinalizeStepCommand | None:
        """Build a settled FinalizeStep for a root loop checkpoint, or skip."""
        if not self._should_finalize_step(metadata, config):
            return None
        return finalize_current_step_command(
            context,
            channel_values_to_state_snapshot(
                dict(checkpoint.get("channel_values") or {}),
                include_legacy_source=context.replay_mode,
            ),
        )

    def _finalize_step_sync(
        self,
        context: AttemptContext,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        config: RunnableConfig,
    ) -> None:
        command = self._prepare_step_finalization(context, checkpoint, metadata, config)
        if command is None:
            return
        with WorkflowClient(self._oe_url()) as client:
            entries = client.finalize_step(command)
        if entries:
            raise RuntimeError("settled step finalization returned suspension entries")
        advance_step_ordinal(command.step_ordinal)

    async def _finalize_step_async(
        self,
        context: AttemptContext,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        config: RunnableConfig,
    ) -> None:
        command = self._prepare_step_finalization(context, checkpoint, metadata, config)
        if command is None:
            return
        async with AsyncWorkflowClient(self._oe_url()) as client:
            entries = await client.finalize_step(command)
        if entries:
            raise RuntimeError("settled step finalization returned suspension entries")
        advance_step_ordinal(command.step_ordinal)

    @staticmethod
    def _should_finalize_step(
        metadata: CheckpointMetadata, config: RunnableConfig
    ) -> bool:
        """Finalize only root-graph loop checkpoints (one OE step per superstep)."""
        if metadata.get("source") != "loop":
            return False
        configurable = config.get("configurable") or {}
        return (configurable.get("checkpoint_ns") or "") == ""

    @classmethod
    def _durable_metadata(
        cls, metadata: CheckpointMetadata, config: RunnableConfig
    ) -> CheckpointMetadata:
        if not cls._should_finalize_step(metadata, config):
            return metadata
        # LangGraph's relative `step` can shift when OE seeds previous state.
        # Record the absolute ordinal used by FinalizeStep on the same scratch
        # checkpoint so history selection never has to reconstruct that offset.
        return cast(
            CheckpointMetadata,
            {
                **metadata,
                OE_STEP_ORDINAL_METADATA_KEY: current_step_ordinal(),
            },
        )

    # ----- routing helpers -----

    def _require_native(self) -> BaseCheckpointSaver[Any]:
        if self.native is None:
            raise RuntimeError(
                "native_checkpoint session requires a MongoDB-backed checkpointer"
            )
        return self.native

    @staticmethod
    def _durable_config(
        config: RunnableConfig, context: AttemptContext
    ) -> RunnableConfig:
        configurable = dict(config.get("configurable") or {})
        configurable["thread_id"] = scratch_thread_id(context)
        configurable.setdefault("checkpoint_ns", "")
        return {**config, "configurable": configurable}

    @staticmethod
    def _reject_durable_put_source(metadata: CheckpointMetadata) -> None:
        source = metadata.get("source")
        if source in _REJECTED_PUT_SOURCES:
            raise UnsupportedDurableGraphError(
                f"update_state() / checkpoint {source} is not supported on "
                "durable_workflow sessions"
            )

    @staticmethod
    def _record_direct_interrupts(
        attempt: AttemptContext,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_path: str,
    ) -> None:
        """Represent typed graph-node interrupts as synthetic OE activities."""
        configurable = config.get("configurable") or {}
        # An interrupt raised by a wrapped durable activity already has an OE
        # identity, so its native LangGraph write must not create a synthetic
        # activity as well.
        owned_interrupt_ids = {
            interrupt.id
            for activity in interrupted_activities(current_step_ordinal())
            if not is_direct_interrupt_activity(activity)
            if isinstance(activity.control_flow, GraphInterrupt)
            for interrupt in activity.control_flow.args[0]
        }
        # LangGraph projects a nested interrupt through every ancestor
        # checkpoint. The deepest child write has already registered the
        # durable identity with its full operation path, so suppress each later
        # projection. A root-native interrupt has only its task segment and is
        # still eligible for the position-based de-duplication below.
        owned_interrupt_ids.update(
            interrupt.id
            for activity in interrupted_activities(current_step_ordinal())
            if is_direct_interrupt_activity(activity)
            and len(activity.command.position.operation_path.segments) > 1
            and isinstance(activity.control_flow, GraphInterrupt)
            for interrupt in activity.control_flow.args[0]
        )
        direct_interrupts = [
            (index, interrupt)
            for channel, value in writes
            if channel == "__interrupt__" and isinstance(value, (list, tuple))
            for index, interrupt in enumerate(value)
            if isinstance(interrupt, Interrupt)
            and interrupt.id not in owned_interrupt_ids
        ]
        if not direct_interrupts:
            return
        if not task_path:
            raise UnsupportedDurableGraphError(
                "durable direct interrupt requires a stable LangGraph task path"
            )

        # Interrupt.id is regenerated with the replacement task namespace, so
        # it cannot be the durable position. The task path is stable within its
        # durable operation boundary across attempts.
        resume_count = max(
            (
                len(value) if isinstance(value, (list, tuple)) else 1
                for channel, value in writes
                if channel == "__resume__"
            ),
            default=0,
        )
        if resume_count:
            raise UnsupportedDurableGraphError(
                "durable workflow does not support multiple sequential direct "
                "interrupts in one graph task"
            )
        existing_positions = {
            activity.command.position.SerializeToString(deterministic=True)
            for activity in interrupted_activities(current_step_ordinal())
        }
        for index, interrupt in direct_interrupts:
            operation_path = OperationPath()
            if configurable.get("checkpoint_ns"):
                # The namespace identifies an attempt-local child invocation,
                # so it cannot be persisted directly. The resolver converts it
                # to stable compiled-child ordinals. Appending task_path below
                # then distinguishes sequential interrupting nodes inside the
                # same child while keeping them in one root OE step.
                operation_path.CopyFrom(current_operation_path())
            operation_path.segments.append(
                OperationPathSegment(
                    name=f"langgraph.task:{task_path}",
                    ordinal=1,
                )
            )
            command = build_activity_command(
                attempt=attempt,
                kind=ACTIVITY_KIND_TOOL,
                name=_DIRECT_INTERRUPT_ACTIVITY_NAME,
                activity_ordinal=index + 1,
                operation_path=operation_path,
                semantic_input={"value": interrupt.value},
            )
            position_key = command.position.SerializeToString(deterministic=True)
            if position_key in existing_positions:
                continue
            record_observed_activity(command.position)
            record_interrupted_activity(command, GraphInterrupt((interrupt,)))
            existing_positions.add(position_key)

    # ----- BaseCheckpointSaver: sync -----

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        context = current_attempt_context()
        if context is None:
            return self._require_native().get_tuple(config)
        return self._scratch.get_tuple(self._durable_config(config, context))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        context = current_attempt_context()
        if context is None:
            return self._require_native().list(
                config, filter=filter, before=before, limit=limit
            )
        if config is None:
            raise UnsupportedDurableGraphError(
                "durable scratch history requires the current attempt config"
            )
        durable_before = (
            self._durable_config(before, context) if before is not None else None
        )
        return self._scratch.list(
            self._durable_config(config, context),
            filter=filter,
            before=durable_before,
            limit=limit,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        context = current_attempt_context()
        if context is None:
            return self._require_native().put(
                config, checkpoint, metadata, new_versions
            )
        self._reject_checkpoint_send(checkpoint)
        self._reject_durable_put_source(metadata)
        durable_config = self._durable_config(config, context)
        durable_metadata = self._durable_metadata(metadata, durable_config)
        result = self._scratch.put(
            durable_config, checkpoint, durable_metadata, new_versions
        )
        self._finalize_step_sync(context, checkpoint, durable_metadata, durable_config)
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        context = current_attempt_context()
        if context is None:
            self._require_native().put_writes(config, writes, task_id, task_path)
            return
        self._reject_unsupported_send(writes)
        self._record_direct_interrupts(context, config, writes, task_path)
        self._scratch.put_writes(
            self._durable_config(config, context), writes, task_id, task_path
        )

    def delete_thread(self, thread_id: str) -> None:
        context = current_attempt_context()
        if context is None:
            self._require_native().delete_thread(thread_id)
            return
        # Durable cleanup is keyed by the fenced attempt, not the caller's thread_id.
        self._scratch.delete_thread(scratch_thread_id(context))

    def get_next_version(self, current: Any, channel: None = None) -> str:
        context = current_attempt_context()
        if context is None:
            return self._require_native().get_next_version(current, channel)
        return self._scratch.get_next_version(current, channel)

    # ----- BaseCheckpointSaver: async -----

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        context = current_attempt_context()
        if context is None:
            return await self._require_native().aget_tuple(config)
        return await self._scratch.aget_tuple(self._durable_config(config, context))

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        context = current_attempt_context()
        if context is None:
            async for item in self._require_native().alist(
                config, filter=filter, before=before, limit=limit
            ):
                yield item
            return
        if config is None:
            raise UnsupportedDurableGraphError(
                "durable scratch history requires the current attempt config"
            )
        durable_before = (
            self._durable_config(before, context) if before is not None else None
        )
        async for item in self._scratch.alist(
            self._durable_config(config, context),
            filter=filter,
            before=durable_before,
            limit=limit,
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        context = current_attempt_context()
        if context is None:
            return await self._require_native().aput(
                config, checkpoint, metadata, new_versions
            )
        self._reject_checkpoint_send(checkpoint)
        self._reject_durable_put_source(metadata)
        durable_config = self._durable_config(config, context)
        durable_metadata = self._durable_metadata(metadata, durable_config)
        result = await self._scratch.aput(
            durable_config, checkpoint, durable_metadata, new_versions
        )
        await self._finalize_step_async(
            context, checkpoint, durable_metadata, durable_config
        )
        return result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        context = current_attempt_context()
        if context is None:
            await self._require_native().aput_writes(config, writes, task_id, task_path)
            return
        self._reject_unsupported_send(writes)
        self._record_direct_interrupts(context, config, writes, task_path)
        await self._scratch.aput_writes(
            self._durable_config(config, context), writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        context = current_attempt_context()
        if context is None:
            await self._require_native().adelete_thread(thread_id)
            return
        await self._scratch.adelete_thread(scratch_thread_id(context))
