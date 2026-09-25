"""Turn-level Google ADK rewind backed by an OE branch."""

from __future__ import annotations

from typing import Any, cast

import httpx
from agent_engine_sdk import SessionForkResponse

from agent_engine_runner_shared.server.oe_url import resolve_oe_url
from agent_engine_runner_shared.tls_client import create_async_httpx_client_with_tls

__all__ = ["RewindBranchError"]

_CREATE_BRANCH_TIMEOUT_SECONDS = 15.0


class RewindBranchError(RuntimeError):
    """OE could not create the requested rewind branch."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


async def create_rewind_branch(
    *,
    execution_id: str,
    oe_url: str,
    rewind_before_invocation_id: str,
) -> SessionForkResponse:
    """Create a new session from the turn before one ADK invocation."""
    if not execution_id:
        raise RewindBranchError("rewind requires an active execution")
    if not oe_url:
        raise RewindBranchError("rewind requires an orchestration engine callback URL")
    if not rewind_before_invocation_id:
        raise RewindBranchError("rewind requires an invocation id")

    body: dict[str, object] = {
        "execution_id": execution_id,
        "before_execution_id": rewind_before_invocation_id,
    }
    status_code, payload = await _post_branch(oe_url, body)
    if status_code != 201:
        raise RewindBranchError(
            f"rewind branch request failed with HTTP {status_code}",
            status_code=status_code,
        )
    if not isinstance(payload, dict):
        raise RewindBranchError("rewind branch response must be an object")

    response_payload = cast(dict[str, Any], payload)
    session = response_payload.get("session_id")
    execution = response_payload.get("execution_id")
    if not isinstance(session, str) or not session:
        raise RewindBranchError("rewind branch response is missing session identity")
    if not isinstance(execution, str) or not execution:
        raise RewindBranchError("rewind branch response is missing execution identity")
    return SessionForkResponse(session_id=session, execution_id=execution)


async def _post_branch(oe_url: str, body: dict[str, object]) -> tuple[int, object]:
    resolved_oe_url = resolve_oe_url(oe_url).rstrip("/")
    if not resolved_oe_url:
        raise RewindBranchError("rewind requires an orchestration engine callback URL")
    try:
        client = await create_async_httpx_client_with_tls(
            resolved_oe_url, timeout=_CREATE_BRANCH_TIMEOUT_SECONDS
        )
    except (RuntimeError, ValueError):
        raise RewindBranchError(
            "rewind could not initialize orchestration engine client"
        ) from None
    try:
        try:
            response = await client.post(
                resolved_oe_url + "/workflow/branches", json=body
            )
        except httpx.HTTPError:
            raise RewindBranchError(
                "rewind could not reach orchestration engine"
            ) from None
        if response.status_code != 201:
            return response.status_code, None
        try:
            payload: Any = response.json()
        except ValueError as error:
            raise RewindBranchError(
                "rewind branch response is not valid JSON",
                status_code=response.status_code,
            ) from error
        return response.status_code, payload
    finally:
        await client.aclose()
