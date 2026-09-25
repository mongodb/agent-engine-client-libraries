"""Session fork. See ``docs/session-fork.md``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from agent_engine_sdk import RequestContext
from agent_engine_sdk.models import SessionForkResponse
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import StateSnapshot
from agent_engine_runner_shared.server.oe_url import resolve_oe_url
from agent_engine_runner_shared.tls_client import create_async_httpx_client_with_tls
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.protojson import encode_protojson

from agent_engine_sdk_langgraph.checkpoint_branch import (
    checkpoint_replay_plan,
    completed_root_checkpoint,
    copy_checkpoint,
)
from agent_engine_sdk_langgraph.hooks import ResolveThreadId
from agent_engine_sdk_langgraph.http_path import quote_path_segment
from agent_engine_sdk_langgraph.platform_checkpointer import (
    OE_STEP_ORDINAL_METADATA_KEY,
    PlatformCheckpointer,
)
from agent_engine_sdk_langgraph.thread_id import checkpoint_thread_id
from agent_engine_sdk_langgraph.workflow_state import channel_values_to_state_snapshot

__all__ = [
    "LangGraphForkPlugin",
    "create_durable_branch",
    "create_native_branch",
    "fork_native_session",
    "take_continue_without_user_message",
    "wrap_session_fork_update_state",
]

_BRANCH_KEY_SAFE = re.compile(r"[^A-Za-z0-9._/-]+")
_CREATE_BRANCH_TIMEOUT_SECONDS = 15.0
# Dest continue is process-local. Bound so unused dests cannot accumulate;
# an evicted mark behaves like a dest execution that starts in another process.
_CONTINUE_MARK_LIMIT = 256
_CONTINUE_WITHOUT_USER_MESSAGE: OrderedDict[str, None] = OrderedDict()


def take_continue_without_user_message(thread_id: str) -> bool:
    """True once: dest should ``ainvoke(None)``. See ``docs/session-fork.md``."""
    if thread_id not in _CONTINUE_WITHOUT_USER_MESSAGE:
        return False
    del _CONTINUE_WITHOUT_USER_MESSAGE[thread_id]
    return True


def _arm_continue_without_user_message(thread_id: str) -> None:
    marks = _CONTINUE_WITHOUT_USER_MESSAGE
    if thread_id in marks:
        marks.move_to_end(thread_id)
        return
    while len(marks) >= _CONTINUE_MARK_LIMIT:
        marks.popitem(last=False)
    marks[thread_id] = None


async def fork_native_session(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    org_id: str,
    project_id: str,
    ctx: RequestContext,
    history_id: str | None,
    resolve_thread_id: ResolveThreadId | None = None,
    state: Any = None,
    as_node: str | None = None,
    task_id: str | None = None,
) -> tuple[SessionForkResponse, RunnableConfig]:
    """Create a branch session and copy the source checkpoint onto it."""
    if current_attempt_context() is not None:
        raise RuntimeError("session fork is not supported on durable_workflow sessions")
    session_id = ctx.session_id
    workspace_id = ctx.workspace_id
    execution_id = ctx.execution_id
    if not session_id or not workspace_id or not execution_id:
        raise RuntimeError(
            "session, workspace, and execution identity are required to fork"
        )

    source = await completed_root_checkpoint(
        graph,
        thread_id=checkpoint_thread_id(ctx, resolve_thread_id),
        history_id=history_id,
    )
    plan = await checkpoint_replay_plan(graph, source)
    has_patch = state not in (None, {}, [])
    # Resolve the patch node before CreateBranch: a multi-node graph without an
    # explicit as_node is an unsupported target, and LangGraph would only
    # reject it after the branch and copy already exist.
    patch_node = _resolve_patch_node(graph, as_node) if has_patch else None
    if has_patch and patch_node is None:
        raise RuntimeError(
            "session fork state patch requires a destination node: "
            "pass as_node or fork a single-node graph"
        )
    identity = await create_native_branch(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        session_id=session_id,
        execution_id=execution_id,
        branch_key=_branch_key(source.checkpoint_id, state, as_node, task_id),
        checkpoint_id=source.checkpoint_id,
    )
    dest_thread_id = checkpoint_thread_id(
        RequestContext(session_id=identity.session_id, workspace_id=workspace_id),
        resolve_thread_id,
    )
    dest_config: RunnableConfig = {
        "configurable": {"thread_id": dest_thread_id, "checkpoint_ns": ""}
    }
    await copy_checkpoint(graph, dest_config, source, plan=plan)
    if patch_node is not None:
        dest_config = await _patch_dest(graph, dest_config, state, patch_node, task_id)
    dest_state = await graph.aget_state(dest_config)
    dest_checkpoint = (dest_state.config.get("configurable") or {}).get("checkpoint_id")
    if dest_checkpoint:
        dest_config = {
            **dest_config,
            "configurable": {
                **dict(dest_config.get("configurable") or {}),
                "checkpoint_id": dest_checkpoint,
            },
        }
    _arm_continue_without_user_message(dest_thread_id)
    return identity, dest_config


async def _fork_durable_session(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    context: AttemptContext,
    config: RunnableConfig,
    resolve_thread_id: ResolveThreadId | None = None,
) -> tuple[SessionForkResponse, RunnableConfig]:
    """Create an OE-owned branch from a selected committed scratch checkpoint."""
    identity = context.workflow_identity
    scope = identity.tenant_scope

    checkpointer = graph.checkpointer
    if not isinstance(checkpointer, PlatformCheckpointer):
        raise RuntimeError("durable session fork requires the platform checkpointer")
    source = await checkpointer.aget_tuple(config)
    if source is None:
        raise RuntimeError("durable branch point is not a committed root step")
    step_ordinal = _oe_step_ordinal(source)
    # Materialized state adds unwritten reducer defaults absent from the OE hash.
    source_state = channel_values_to_state_snapshot(
        dict(source.checkpoint.get("channel_values") or {}),
        include_legacy_source=context.replay_mode,
    )
    branch_source = f"{identity.execution_id}/step/{step_ordinal}"
    branch = await create_durable_branch(
        org_id=scope.org_id,
        project_id=scope.project_id,
        workspace_id=scope.workspace_id,
        session_id=identity.session_id,
        execution_id=identity.execution_id,
        branch_key=_branch_key(branch_source, None, None, None),
        step_ordinal=step_ordinal,
        state=source_state,
    )

    # RequestContext is request-local. Use a minimal value only to resolve the
    # destination thread; the future branch invocation receives its own context.
    dest_context = RequestContext(
        session_id=branch.session_id,
        workspace_id=scope.workspace_id,
        execution_id=branch.execution_id,
    )
    dest_config: RunnableConfig = {
        "configurable": {
            "thread_id": checkpoint_thread_id(dest_context, resolve_thread_id),
            "checkpoint_ns": "",
        },
    }
    return branch, dest_config


def _oe_step_ordinal(snapshot: Any) -> int:
    """Read the absolute OE ordinal recorded with a committed scratch step."""
    step_ordinal = (snapshot.metadata or {}).get(OE_STEP_ORDINAL_METADATA_KEY)
    if step_ordinal is None:
        raise RuntimeError("durable branch point is not a committed root step")
    return cast(int, step_ordinal)


def _require_exact_durable_branch(state: Any) -> None:
    """Reject patches until OE can record one as a branch-local transition."""
    if state not in (None, {}, []):
        raise RuntimeError(
            "durable_workflow session fork does not support an update_state patch"
        )


def _resolve_patch_node(
    graph: CompiledStateGraph[Any, Any, Any, Any], as_node: str | None
) -> str | None:
    """The node a dest patch applies through, or None when it is ambiguous."""
    names = [str(name) for name in graph.nodes if not str(name).startswith("__")]
    if as_node is not None:
        if as_node not in names:
            raise RuntimeError(
                f'session fork state patch node "{as_node}" is not a node in this graph'
            )
        return as_node
    return names[0] if len(names) == 1 else None


async def _patch_dest(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    dest_config: RunnableConfig,
    state: Any,
    node: str,
    task_id: str | None,
) -> RunnableConfig:
    return await graph.aupdate_state(dest_config, state, node, task_id)


def wrap_session_fork_update_state(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    resolve_thread_id: ResolveThreadId | None,
) -> None:
    """Route live ``update_state`` through the persistence-specific session fork.

    See ``docs/session-fork.md``.
    """
    original_aupdate = getattr(graph, "aupdate_state", None)
    original_update = getattr(graph, "update_state", None)
    if not callable(original_aupdate):
        return
    aupdate_fn = cast(Any, original_aupdate)
    update_fn = cast(Any, original_update)

    def live_source(config: RunnableConfig, snapshot: Any) -> RequestContext | None:
        if current_attempt_context() is not None:
            return None
        found = snapshot.config.get("configurable") or {}
        checkpoint_id = found.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        ctx = (config.get("configurable") or {}).get("request_context")
        if not isinstance(ctx, RequestContext):
            return None
        if not ctx.session_id or not ctx.workspace_id or not ctx.execution_id:
            return None
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if thread_id != checkpoint_thread_id(ctx, resolve_thread_id):
            return None
        return ctx

    async def aupdate_state(
        config: RunnableConfig,
        values: Any,
        as_node: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> RunnableConfig:
        attempt = current_attempt_context()
        if attempt is not None:
            if resolve_thread_id is not None:
                raise RuntimeError(
                    "session fork requires the default thread_id formula; "
                    "custom @app.resolve_thread_id is not supported on "
                    "graph.update_state / aupdate_state"
                )
            # CreateBranch hash-validates the selected committed snapshot. A
            # state patch needs a separate branch-local transition contract.
            _require_exact_durable_branch(values)
            _identity, dest_config = await _fork_durable_session(
                graph,
                context=attempt,
                config=config,
                resolve_thread_id=resolve_thread_id,
            )
            return dest_config
        snapshot = await graph.aget_state(config)
        ctx = live_source(config, snapshot)
        if ctx is not None:
            if resolve_thread_id is not None:
                raise RuntimeError(
                    "session fork requires the default thread_id formula; "
                    "custom @app.resolve_thread_id is not supported on "
                    "graph.update_state / aupdate_state"
                )
            org_id, project_id = _tenant_ids()
            pinned = (config.get("configurable") or {}).get("checkpoint_id")
            if isinstance(pinned, str) and pinned:
                history_id: str | None = pinned
            else:
                snapshot_id = (snapshot.config.get("configurable") or {}).get(
                    "checkpoint_id"
                )
                history_id = snapshot_id if isinstance(snapshot_id, str) else None
            _identity, dest_config = await fork_native_session(
                graph,
                org_id=org_id,
                project_id=project_id,
                ctx=ctx,
                history_id=history_id,
                resolve_thread_id=None,
                state=values,
                as_node=as_node,
                task_id=task_id,
            )
            return dest_config
        return await aupdate_fn(config, values, as_node, task_id, **kwargs)

    def update_state(
        config: RunnableConfig,
        values: Any,
        as_node: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> RunnableConfig:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                aupdate_state(config, values, as_node, task_id, **kwargs)
            )
        if current_attempt_context() is not None:
            raise RuntimeError(
                "graph.update_state cannot fork from a running event loop; "
                "await graph.aupdate_state(...) instead"
            )
        snapshot = graph.get_state(config)
        ctx = live_source(config, snapshot)
        if ctx is not None:
            raise RuntimeError(
                "graph.update_state cannot fork from a running event loop; "
                "await graph.aupdate_state(...) instead"
            )
        return update_fn(config, values, as_node, task_id, **kwargs)

    graph.aupdate_state = aupdate_state  # type: ignore[method-assign]
    if callable(update_fn):
        graph.update_state = update_state  # type: ignore[method-assign]


class LangGraphForkPlugin:
    """AER write plugin that forks a LangGraph session."""

    def __init__(
        self,
        get_agent: Callable[..., Any],
        workspace_id_resolver: Callable[[], str],
    ) -> None:
        self._get_agent = get_agent
        self._workspace_id_resolver = workspace_id_resolver

    async def fork_session(
        self,
        *,
        session_id: str,
        execution_id: str,
        history_id: str | None,
        state: Any = None,
    ) -> SessionForkResponse:
        agent = self._get_agent()
        workspace_id = self._workspace_id_resolver()
        attempt = current_attempt_context()
        if attempt is not None:
            identity = attempt.workflow_identity
            if (
                session_id != identity.session_id
                or execution_id != identity.execution_id
                or workspace_id != identity.tenant_scope.workspace_id
            ):
                raise RuntimeError(
                    "durable fork request identity does not match the current OE attempt"
                )
            # Durable CreateBranch accepts the exact committed snapshot. Do not
            # reinterpret an API state patch as part of that source snapshot.
            _require_exact_durable_branch(state)
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": checkpoint_thread_id(
                        RequestContext(
                            session_id=session_id,
                            workspace_id=workspace_id,
                            execution_id=execution_id,
                        ),
                        getattr(agent, "_resolve_thread_id", None),
                    ),
                    "checkpoint_ns": "",
                }
            }
            if history_id:
                config["configurable"]["checkpoint_id"] = history_id  # type: ignore[index]
            identity, _dest_config = await _fork_durable_session(
                agent.compiled_graph,
                context=attempt,
                config=config,
                resolve_thread_id=getattr(agent, "_resolve_thread_id", None),
            )
            return identity
        org_id, project_id = _tenant_ids()
        identity, _dest_config = await fork_native_session(
            agent.compiled_graph,
            org_id=org_id,
            project_id=project_id,
            ctx=RequestContext(
                session_id=session_id,
                workspace_id=workspace_id,
                execution_id=execution_id,
            ),
            history_id=history_id,
            resolve_thread_id=getattr(agent, "_resolve_thread_id", None),
            state=state,
        )
        return identity


async def create_native_branch(
    *,
    org_id: str,
    project_id: str,
    workspace_id: str,
    session_id: str,
    execution_id: str,
    branch_key: str,
    checkpoint_id: str | None,
) -> SessionForkResponse:
    """POST CreateBranch and return the new session identity."""
    body: dict[str, Any] = {"branch_key": branch_key}
    if checkpoint_id:
        body["checkpoint_id"] = checkpoint_id
    return await _create_branch(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        session_id=session_id,
        execution_id=execution_id,
        body=body,
    )


async def create_durable_branch(
    *,
    org_id: str,
    project_id: str,
    workspace_id: str,
    session_id: str,
    execution_id: str,
    branch_key: str,
    step_ordinal: int,
    state: StateSnapshot,
) -> SessionForkResponse:
    """POST an OE-owned branch reconstructed from durable scratch."""
    body = {
        "branch_key": branch_key,
        "step_ordinal": step_ordinal,
        "state": json.loads(encode_protojson(state)),
    }
    return await _create_branch(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        session_id=session_id,
        execution_id=execution_id,
        body=body,
    )


async def _create_branch(
    *,
    org_id: str,
    project_id: str,
    workspace_id: str,
    session_id: str,
    execution_id: str,
    body: dict[str, Any],
) -> SessionForkResponse:
    oe_url = resolve_oe_url("").rstrip("/")
    if not oe_url:
        raise RuntimeError("OE_URL is not configured; cannot fork a session")
    path = (
        "/workflow/orgs/"
        f"{quote_path_segment(org_id)}/projects/{quote_path_segment(project_id)}"
        f"/workspaces/{quote_path_segment(workspace_id)}"
        f"/sessions/{quote_path_segment(session_id)}"
        f"/executions/{quote_path_segment(execution_id)}/branches"
    )
    client = await create_async_httpx_client_with_tls(
        oe_url, timeout=_CREATE_BRANCH_TIMEOUT_SECONDS
    )
    try:
        response = await client.post(oe_url + path, json=body)
    finally:
        await client.aclose()
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"CreateBranch failed with HTTP {response.status_code}: {response.text[:200]}"
        )
    payload = response.json()
    new_session_id = payload.get("session_id")
    new_execution_id = payload.get("execution_id")
    if not isinstance(new_session_id, str) or not isinstance(new_execution_id, str):
        raise RuntimeError("CreateBranch response is missing session identity")
    return SessionForkResponse(session_id=new_session_id, execution_id=new_execution_id)


def _branch_key(
    history_id: str, state: Any, as_node: str | None, task_id: str | None
) -> str:
    payload = json.dumps(
        {
            "state": None if state in (None, {}, []) else state,
            "as_node": as_node or "",
            "task_id": task_id or "",
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(f"{history_id}\0{payload}".encode()).hexdigest()[:16]
    safe_history = _BRANCH_KEY_SAFE.sub("-", history_id).strip("-._/") or "latest"
    return f"update-{safe_history[:64]}-{digest}"


def _tenant_ids() -> tuple[str, str]:
    org_id = os.environ.get("ORG_ID", "").strip()
    project_id = os.environ.get("PROJECT_ID", "").strip()
    if not org_id or not project_id:
        raise RuntimeError("ORG_ID and PROJECT_ID are required to fork a session")
    return org_id, project_id
