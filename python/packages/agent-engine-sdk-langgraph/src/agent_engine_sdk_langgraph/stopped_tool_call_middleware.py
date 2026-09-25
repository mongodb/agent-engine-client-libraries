"""Deterministic handling of OE-stopped in-flight tool calls.

Inspects the artifact-channel marker on a stopped tool call and reacts
deterministically, rather than leaving it to the model's free-form judgment.
A batch is every ``ToolMessage`` following the newest ``AIMessage`` with tool
calls. If every call in the batch was stopped, the turn ends immediately
with a fixed message and no retry. If only some were, the model reacts
normally to the mix of real results and "stopped" notes.

Registered by default in
:func:`agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent`.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from agent_engine_runner_shared.secure_wrapper import CALL_INTERRUPTED_ARTIFACT_KEY

#: Fixed, consistent message shown when every call in a step was interrupted.
#: Deliberately not model-generated — the whole point is a guaranteed, testable
#: outcome instead of relying on the model's free-form reaction.
ALL_INTERRUPTED_MESSAGE = (
    "This request was stopped before any of its actions completed."
)


def _is_call_interrupted(message: ToolMessage) -> bool:
    artifact = getattr(message, "artifact", None)
    return isinstance(artifact, dict) and bool(
        artifact.get(CALL_INTERRUPTED_ARTIFACT_KEY)
    )


def _latest_tool_batch(messages: list[Any]) -> list[ToolMessage]:
    """The run of ToolMessages following the most recent AIMessage with tool calls.

    Returns an empty list if the conversation doesn't currently end with such a
    batch (e.g. the last step was a plain assistant reply) — nothing to react to.
    """
    batch: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            batch.append(message)
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            break
        # A non-tool-calling AIMessage, or anything else, means we've walked
        # past the current step's boundary without finding tool calls.
        return []
    batch.reverse()
    return batch


class StoppedToolCallMiddleware(AgentMiddleware[AgentState, Any]):
    """Ends a turn deterministically when every in-flight call was stopped.

    See module docstring for the full batch-aware rationale.
    """

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        batch = _latest_tool_batch(messages)
        if not batch:
            return None

        interrupted_count = sum(1 for message in batch if _is_call_interrupted(message))
        if interrupted_count == 0 or interrupted_count != len(batch):
            # Partial (or none) stopped: let the model see the real
            # results/per-call notes and react normally — no short-circuit.
            return None

        # Every call in this batch was stopped: end the turn now instead of
        # letting the model freelance a response to an all-stopped batch.
        return {
            "jump_to": "end",
            "messages": [AIMessage(content=ALL_INTERRUPTED_MESSAGE)],
        }
