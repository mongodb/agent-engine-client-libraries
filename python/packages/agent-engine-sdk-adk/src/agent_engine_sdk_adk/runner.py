"""Public runner for Google ADK applications on MongoDB."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_engine_sdk import RequestContext, SessionForkResponse

from agent_engine_runner_shared.context import (
    get_current_execution_id,
    get_current_oe_url,
    get_current_session_id,
    get_current_user_id,
)
from agent_engine_sdk_adk.execution_session import DurableSession, execution_session
from agent_engine_sdk_adk.rewind import RewindBranchError, create_rewind_branch

if TYPE_CHECKING:
    from google.adk.agents import BaseAgent as ADKAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.workflow import BaseNode as ADKNode

__all__ = ["DurableADKRunner"]


class DurableADKRunner:
    """Run durable ADK turns and create immutable rewind branches."""

    def __init__(self, *, app_name: str) -> None:
        self._app_name = app_name

    async def rewind_async(
        self,
        *,
        user_id: str,
        session_id: str,
        rewind_before_invocation_id: str,
        run_config: RunConfig | None = None,
    ) -> SessionForkResponse:
        """Fork a session immediately before one completed ADK invocation.

        Unlike Google's native implementation, this leaves ``session_id``
        unchanged and returns the new branch identity.
        """
        del run_config
        if not user_id:
            raise RewindBranchError("rewind requires a user id")
        if not session_id:
            raise RewindBranchError("rewind requires a session id")
        if not rewind_before_invocation_id:
            raise RewindBranchError("rewind requires an invocation id")

        execution_id = get_current_execution_id()
        current_session_id = get_current_session_id()
        current_user_id = get_current_user_id()
        oe_url = get_current_oe_url()
        if (
            execution_id is None
            or current_session_id is None
            or current_user_id is None
            or not oe_url
        ):
            raise RewindBranchError("rewind requires an active durable invocation")
        if user_id != current_user_id:
            raise RewindBranchError("rewind user must match the active invocation")
        if session_id != current_session_id:
            raise RewindBranchError("rewind session must match the active invocation")

        return await create_rewind_branch(
            execution_id=execution_id,
            oe_url=oe_url,
            rewind_before_invocation_id=rewind_before_invocation_id,
        )

    def _execution_session(
        self,
        adk_agent: ADKAgent | ADKNode,
        *,
        ctx: RequestContext,
    ) -> DurableSession:
        """Bind one private Google Runner to the current durable attempt."""
        return execution_session(adk_agent, app_name=self._app_name, ctx=ctx)
