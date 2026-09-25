"""Durable ADK execution captured once per invoke/stream.

ADK is durable-only: there is no native session path. ``ADKBaseAgent``
constructs a ``DurableSession`` from ``current_attempt_context()`` at the
start of a run and then asks the session — it does not re-read the
ContextVar. Replay rebuilds the original user message; Atlas Agent Engine
``ctx.resume`` is not an ADK checkpoint resume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from agent_engine_sdk import AgentInput, RequestContext
from google.adk import runners
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_sdk_adk.agent_route import configured_sub_agents
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.platform_session import PlatformSessionService
from agent_engine_sdk_adk.route import DurableRouteAdapter, configured_adk_nodes

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent as ADKAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.workflow import BaseNode as ADKNode

__all__ = ["DurableSession", "execution_session"]

_TOOL_GUARD_ATTRIBUTE = "_agent_engine_durable_tool_guard_installed"


def _is_node_tool(tool: object) -> bool:
    try:
        from google.adk.tools._node_tool import NodeTool
    except ImportError:
        # NodeTool was added after the minimum supported ADK 2 release.
        return False
    return isinstance(tool, NodeTool)


def _validate_supported_tools(
    tools: list[Any], *, parallel_worker: bool = False
) -> None:
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.tools.base_toolset import BaseToolset
    from google.adk.tools.long_running_tool import LongRunningFunctionTool

    for tool in tools:
        if isinstance(tool, AgentTool):
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows do not support AgentTool children; "
                "configure sub_agents and use transfer_to_agent instead"
            )
        if _is_node_tool(tool):
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows do not support NodeTool children"
            )
        if parallel_worker and isinstance(tool, BaseToolset):
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows do not support toolsets inside "
                "parallel_worker children because their resolved tools cannot "
                "be validated before execution; configure static tools instead"
            )
        if parallel_worker and isinstance(tool, LongRunningFunctionTool):
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows do not support suspending "
                "LongRunningFunctionTool calls inside parallel_worker children; "
                "use configured parallel Workflow branches for concurrent HITL"
            )


def _guard_resolved_tools(agent: ADKAgent, *, parallel_worker: bool = False) -> None:
    """Validate context-dependent toolsets after ADK resolves them."""
    if getattr(agent, _TOOL_GUARD_ATTRIBUTE, False):
        return

    existing: Any = getattr(agent, "before_model_callback", None)
    callbacks = (
        cast(list[Any], existing.copy()) if isinstance(existing, list) else [existing]
    )

    async def validate_tools(
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        del callback_context
        _validate_supported_tools(
            list(llm_request.tools_dict.values()),
            parallel_worker=parallel_worker,
        )

    object.__setattr__(
        agent,
        "before_model_callback",
        [validate_tools, *(callback for callback in callbacks if callback is not None)],
    )
    object.__setattr__(agent, _TOOL_GUARD_ATTRIBUTE, True)


def _reject_unsupported_collaboration(root: ADKAgent | ADKNode) -> None:
    """Validate transfer lifecycles before any durable activity can start."""
    from google.adk.agents import BaseAgent as ADKAgentType

    agent_root = isinstance(root, ADKAgentType)
    nodes = (*configured_adk_nodes(root), *configured_sub_agents(root))
    parallel_worker_module = __import__(
        "google.adk.workflow._parallel_worker",
        fromlist=["_ParallelWorker"],
    )
    parallel_worker_type = parallel_worker_module._ParallelWorker
    parallel_worker_targets: set[int] = set()
    for node in nodes:
        if not isinstance(node, parallel_worker_type):
            continue
        worker_root = getattr(node, "_node")
        worker_nodes = (
            *configured_adk_nodes(worker_root),
            *configured_sub_agents(worker_root),
        )
        parallel_worker_targets.update(
            id(worker_node)
            for worker_node in worker_nodes
            if isinstance(worker_node, ADKAgentType)
        )
    for node in nodes:
        if not isinstance(node, ADKAgentType):
            continue
        configured_mode = cast(str | None, getattr(node, "mode", None))
        # ADK defaults sub-agents to chat, but agents used directly as Workflow
        # nodes to single_turn, where both layers can process the same transfer.
        effective_mode = configured_mode or (
            "chat" if node.parent_agent is not None else "single_turn"
        )
        if node.sub_agents and not agent_root and effective_mode != "chat":
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows require mode='chat' on Workflow "
                "agent nodes that configure sub_agents"
            )
        parallel_worker = id(node) in parallel_worker_targets
        _validate_supported_tools(
            list(getattr(node, "tools", ())),
            parallel_worker=parallel_worker,
        )
        if hasattr(node, "before_model_callback"):
            _guard_resolved_tools(node, parallel_worker=parallel_worker)


class DurableSession:
    """One OE-issued ADK attempt: scratch session, Runner, original turn input."""

    def __init__(
        self,
        adk_agent: ADKAgent | ADKNode,
        *,
        app_name: str,
        attempt: AttemptContext,
        user_id: str,
        resume_activity_ids: frozenset[str],
    ) -> None:
        _reject_unsupported_collaboration(adk_agent)
        self.attempt = attempt
        self.user_id = user_id
        # OE owns the answers. The adapter keeps only the current resume keys
        # so replayed output becomes visible after reaching that frontier.
        self.resume_activity_ids = resume_activity_ids
        self.app_name = app_name
        self.scratch = PlatformSessionService.from_attempt(
            attempt,
            app_name=app_name,
            user_id=user_id,
        )
        from google.adk.agents import BaseAgent as ADKAgentType

        # Runner accepts the same configuration for both root types, but ADK
        # requires the root itself under mutually exclusive keyword arguments.
        root: dict[str, Any]
        if isinstance(adk_agent, ADKAgentType):
            root = {"agent": adk_agent}
        else:
            root = {"node": adk_agent}
        self.route_adapter = DurableRouteAdapter(adk_agent)
        self.runner = runners.Runner(
            **root,
            app_name=app_name,
            session_service=self.scratch,
        )

    @property
    def session_id(self) -> str:
        return self.attempt.workflow_identity.session_id

    def original_message(self, input: AgentInput) -> types.Content:
        """Rebuild the original user turn. Durable continue never uses ctx.resume."""
        if not isinstance(input.payload, dict):
            raise TypeError(f"Expected dict payload, got {type(input.payload)}")
        payload: dict[str, Any] = input.payload
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=str(payload.get("message", "")))],
        )


def execution_session(
    adk_agent: ADKAgent | ADKNode,
    *,
    app_name: str,
    ctx: RequestContext,
) -> DurableSession:
    """Capture the OE attempt once and bind scratch plus Runner."""
    attempt = current_attempt_context()
    if attempt is None:
        raise UnsupportedDurableADKError(
            "Google ADK requires an OE-issued durable workflow attempt"
        )
    user_id = ctx.user_id
    if not user_id:
        raise UnsupportedDurableADKError("Google ADK requires RequestContext.user_id")
    return DurableSession(
        adk_agent,
        app_name=app_name,
        attempt=attempt,
        user_id=user_id,
        resume_activity_ids=(
            frozenset(ctx.resume_data)
            if ctx.resume and isinstance(ctx.resume_data, dict)
            else frozenset()
        ),
    )
