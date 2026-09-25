"""OE-attempt-scoped graph execution helpers.

Native Mongo checkpoint sessions never enter these helpers. Scratch
lifecycle, compiled-path scopes, and execution-stable input message ids
are durable-only. Native LangGraph interrupts are translated here because
replacement attempts must replay them against OE activity positions before
continuing the graph.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt, Overwrite
from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    Message,
    RequestContext,
    StreamEvent,
)
from agent_engine_runner_shared.context import (
    get_current_oe_url,
    get_current_user_id,
    get_current_wrapper,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityContext,
    ActivitySuspension,
    StepSuspensionEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import (
    AsyncWorkflowClient,
    finalize_current_step_suspensions_command,
)
from agent_engine_runner_shared.workflow.activity import (
    unwrap_activity_outcome,
)
from agent_engine_runner_shared.workflow.context import (
    InterruptedActivity,
    current_step_ordinal,
    interrupted_activities,
    set_activity_reconstruction_ids,
)
from agent_engine_runner_shared.workflow.protojson import json_to_proto_struct

from agent_engine_sdk_langgraph.checkpoint_branch import (
    LangGraphCheckpoint,
    pending_interrupts_from_state,
)
from agent_engine_sdk_langgraph.durable_errors import UnsupportedDurableGraphError
from agent_engine_sdk_langgraph.durable_subgraphs import (
    DurableSubgraphResolver,
)
from agent_engine_sdk_langgraph.execution_session import (
    DurableCheckpointKwargs,
    ExecutionSession,
)
from agent_engine_sdk_langgraph.hooks import (
    GraphInput,
    PrepareAgentInput,
    ResolveThreadId,
)
from agent_engine_sdk_langgraph.platform_checkpointer import (
    is_direct_interrupt_activity,
)
from agent_engine_sdk_langgraph.suspend import (
    invoke_suspend_output,
    stream_hitl_suspend_event,
)

_FRAMEWORK_INTERRUPT_REASON = "agent_interrupt"
_DIRECT_INTERRUPT_ACTIVITY_NAME = "langgraph.interrupt"


async def settle_interrupts(
    *,
    attempt: AttemptContext,
    interrupts: list[Interrupt],
) -> Command | list[dict[str, Any]]:
    """Finalize interrupted activities or feed their resolved answers to LangGraph."""
    oe_url = get_current_oe_url()
    if not oe_url:
        raise RuntimeError("durable interrupt requires an orchestration engine URL")

    # Local-tool execution and the checkpointer record the exact command whose
    # native interrupt reached scratch. The regenerated native id joins the
    # checkpoint interrupt to that stable activity position on every attempt.
    activities_by_native_id: dict[str, InterruptedActivity] = {}
    for activity in interrupted_activities(current_step_ordinal()):
        control_flow = activity.control_flow
        if not isinstance(control_flow, GraphInterrupt):
            continue
        native_interrupts = control_flow.args[0]
        if len(native_interrupts) != 1:
            raise UnsupportedDurableGraphError(
                "a durable local tool must raise exactly one native interrupt"
            )
        activities_by_native_id[native_interrupts[0].id] = activity

    interrupts_by_position: dict[bytes, Interrupt] = {}
    suspensions: list[StepSuspensionEntry] = []
    for interrupt in interrupts:
        activity = activities_by_native_id.get(interrupt.id)
        if activity is None:
            raise UnsupportedDurableGraphError(
                "durable native interrupt checkpoint write did not register an activity"
            )
        command = activity.command
        interrupts_by_position[
            command.position.SerializeToString(deterministic=True)
        ] = interrupt
        suspensions.append(
            StepSuspensionEntry(
                position=command.position,
                activity_kind=command.activity_kind,
                activity_name=command.activity_name,
                semantic_input=command.semantic_input,
                suspension=ActivitySuspension(
                    reason=_FRAMEWORK_INTERRUPT_REASON,
                    context=json_to_proto_struct({"value": interrupt.value}),
                ),
            )
        )

    async with AsyncWorkflowClient(oe_url) as client:
        entries = await client.finalize_step(
            finalize_current_step_suspensions_command(attempt, suspensions)
        )

    # OE returns its canonical frontier order. Position is the durable join
    # key back to this attempt's regenerated native interrupt ids.
    settled = [
        (
            interrupts_by_position[
                entry.position.SerializeToString(deterministic=True)
            ],
            entry.outcome,
        )
        for entry in entries
    ]
    if settled[0][1].outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED:
        return [
            {"id": outcome.activity_id, "value": interrupt.value}
            for interrupt, outcome in settled
        ]

    wrapper = get_current_wrapper()
    if wrapper is not None and wrapper.durable_memory is not None:
        for interrupt, outcome in settled:
            activity = activities_by_native_id[interrupt.id]
            if not is_direct_interrupt_activity(activity):
                continue
            # Direct graph interrupts bypass SecureToolWrapper, so close their
            # resolved activity in Memory here under OE's winning fence.
            await asyncio.to_thread(
                wrapper.durable_memory.synchronize_tool,
                wrapper.workflow,
                ActivityContext(
                    workflow_identity=outcome.workflow_identity,
                    activity_id=outcome.activity_id,
                    attempt_id=outcome.attempt_id,
                    fencing_token=outcome.fencing_token,
                ),
                unwrap_activity_outcome(outcome),
                user_id=get_current_user_id(),
                tool_call_id=f"{_DIRECT_INTERRUPT_ACTIVITY_NAME}:{outcome.activity_id}",
                tool_name=_DIRECT_INTERRUPT_ACTIVITY_NAME,
            )

    return Command(
        resume={
            interrupt.id: unwrap_activity_outcome(outcome)
            for interrupt, outcome in settled
        }
    )


def assign_stable_execution_message_ids(
    graph_input: GraphInput,
    attempt: AttemptContext,
) -> GraphInput:
    """Stamp missing hook-produced message ids as ``durable-input:{execution_id}:{index}``.

    The default no-hook HumanMessage already carries unindexed
    ``durable-input:{execution_id}`` from ``_fresh_graph_input``. This helper
    must not rewrite that id: replacement attempts after upgrade would
    otherwise diverge from checkpoints stamped before the adapter split.

    LangGraph assigns random IDs when its message reducer receives messages
    without them. Durable replacement attempts rebuild the original input from
    the same OE execution, so identity must be stable across those attempts
    and keyed by ``workflow_identity.execution_id``.
    """
    if isinstance(graph_input, Command):
        return graph_input
    raw_messages = graph_input.get("messages")
    if raw_messages is None:
        return graph_input

    overwrite = isinstance(raw_messages, Overwrite)
    message_value = raw_messages.value if overwrite else raw_messages
    message_inputs = (
        message_value if isinstance(message_value, list) else [message_value]
    )
    messages = convert_to_messages(message_inputs)
    execution_id = attempt.workflow_identity.execution_id
    normalized_messages = [
        message.model_copy(update={"id": f"durable-input:{execution_id}:{index}"})
        if message.id is None
        else message
        for index, message in enumerate(messages)
    ]
    normalized_value = (
        Overwrite(normalized_messages) if overwrite else normalized_messages
    )
    return {**graph_input, "messages": normalized_value}


class DurableSession(ExecutionSession):
    """OE durable-workflow attempt.

    Replay rebuilds the original turn input; OE returns recorded activity
    results until a native interrupt is reconstructed, then this session
    immediately returns ``Command(resume=...)``. There is no Mongo checkpoint
    coordinate. Scratch checkpoints are attempt-local and discarded when the
    invocation ends. ``attempt`` is always set: ``execution_session()`` only
    constructs this class when an ``AttemptContext`` is bound.
    """

    is_durable = True
    drain_after_terminal = False
    attempt: AttemptContext

    def __init__(
        self,
        graph: Any,
        *,
        callbacks: list[Any],
        prepare_input: PrepareAgentInput | None,
        resolve_thread_id: ResolveThreadId | None,
        durable_subgraphs: DurableSubgraphResolver | None,
        attempt: AttemptContext,
        use_custom_parser: bool = False,
    ) -> None:
        super().__init__(
            graph,
            callbacks=callbacks,
            prepare_input=prepare_input,
            resolve_thread_id=resolve_thread_id,
            durable_subgraphs=durable_subgraphs,
            attempt=attempt,
            use_custom_parser=use_custom_parser,
        )
        if durable_subgraphs is not None:
            durable_subgraphs.validate()

    def build_graph_input(self, ctx: RequestContext, input: AgentInput) -> Any:
        reconstruction_ids: tuple[str, ...] = ()
        if ctx.resume:
            if not isinstance(ctx.resume_data, dict):
                raise UnsupportedDurableGraphError(
                    "durable resume requires an activity-id-keyed resume map"
                )
            reconstruction_ids = tuple(ctx.resume_data)
        set_activity_reconstruction_ids(reconstruction_ids)
        # Tenant hooks still own original-input construction, but must see the
        # same fresh-invocation context they saw on the first attempt.
        durable_replay = self.attempt.replay_mode
        if self.prepare_input is not None:
            prepare_context = (
                ctx.model_copy(
                    update={
                        "resume": False,
                        "resume_data": None,
                        "metadata": None,
                    }
                )
                if durable_replay
                else ctx
            )
            prepared_input = self.prepare_input(input, prepare_context)
            return assign_stable_execution_message_ids(prepared_input, self.attempt)
        return assign_stable_execution_message_ids(
            self._fresh_graph_input(ctx, input), self.attempt
        )

    def build_config(self, ctx: RequestContext) -> RunnableConfig:
        config = self._base_config(ctx)
        if self._explicit_checkpoint_id(ctx) is not None:
            raise UnsupportedDurableGraphError(
                "explicit checkpoint targeting is not supported on "
                "durable_workflow sessions"
            )
        return config

    def subgraph_resolver(self) -> DurableSubgraphResolver | None:
        return self.durable_subgraphs

    def checkpoint_durability(self) -> DurableCheckpointKwargs:
        # Sync so PlatformCheckpointer.FinalizeStep returns before the next
        # superstep can admit activities against a checkpoint that is still
        # flushing. Native sessions omit this and keep LangGraph's default.
        return {"durability": "sync"}

    async def fence_cancelled_predecessor_writes(self, ctx: RequestContext) -> None:
        del ctx
        # Durable recovery is the turn base, markers, and recorded activity
        # outcomes. Cancelled-attempt scratch is discarded with the attempt,
        # not fenced on the Mongo thread.

    async def persisted_checkpoint_coordinate(
        self, thread_id: str
    ) -> LangGraphCheckpoint | None:
        del thread_id
        # Scratch still holds this attempt's graph state. It is not the Mongo
        # thread coordinate native output metadata uses for branch/resume.
        return None

    async def _settle_pending_interrupts(
        self, config: RunnableConfig
    ) -> tuple[Any, Command | list[dict[str, Any]] | None]:
        state = await self.graph.aget_state(config, subgraphs=True)
        pending = pending_interrupts_from_state(state)
        if pending:
            return state, await settle_interrupts(
                attempt=self.attempt,
                interrupts=pending,
            )
        return state, None

    async def invoke_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        response: str,
        messages: list[Any],
    ) -> Command | AgentOutput | None:
        state, result = await self._settle_pending_interrupts(config)
        if isinstance(result, Command):
            return result
        if result is not None:
            thread_id = config["configurable"]["thread_id"]  # type: ignore[index]
            return invoke_suspend_output(
                response=response,
                thread_id=thread_id,
                checkpoint_id=None,
                interrupts=result,
                message_count=len(messages),
                resumed=ctx.resume,
            )
        if state.next:
            raise UnsupportedDurableGraphError(
                "durable workflow does not support LangGraph "
                "interrupt_before or interrupt_after pauses"
            )
        return None

    async def stream_interrupt(
        self,
        *,
        ctx: RequestContext,
        config: RunnableConfig,
        captured: list[Interrupt],
        messages: list[Message],
    ) -> Command | StreamEvent | None:
        state, result = await self._settle_pending_interrupts(config)
        if isinstance(result, Command):
            return result
        if result is not None:
            return stream_hitl_suspend_event(
                result,
                checkpoint_id=None,
                resumed=ctx.resume,
                messages=messages,
            )
        if state.next or captured:
            raise UnsupportedDurableGraphError(
                "durable workflow does not support LangGraph "
                "interrupt_before or interrupt_after pauses"
            )
        return None
