"""AER surfaces OE policy-engine denials as a structured signal.

When the OE blocks a tool or LLM call, the agent stream raises
``PolicyDeniedException``. The AER must turn that into a *structured* signal —
an ERROR chunk carrying ``metadata.error_code == "policy_denied"`` plus the
reason — so streaming consumers can render a "blocked by policy" outcome
distinctly instead of string-matching a generic error message. The same
metadata is attached to the terminal callback (forward-compatible; see the
handler comment for the OE terminal-metadata caveat).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from agent_engine_runner_shared.models import ExecuteRequest, GuardrailMeta
from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException
from agent_engine_runner_shared.server.chunk_types import ERROR, POLICY_DENIED_ERROR_CODE

pytestmark = pytest.mark.asyncio


def _make_server() -> Any:
    from agent_engine_runner_shared.server.aer import AERServer

    server = AERServer(Mock())
    server._send_stream_chunk = AsyncMock()
    server._send_callback_body = AsyncMock()
    return server


def _deny_with(exc: PolicyDeniedException):
    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise exc

    return _stream


def _err_chunk(server: Any):
    calls = [
        c for c in server._send_stream_chunk.await_args_list if c.kwargs.get("chunk_type") == ERROR
    ]
    assert len(calls) == 1, f"expected exactly one ERROR chunk, got {len(calls)}"
    return calls[0]


async def test_policy_denied_emits_structured_error_chunk_and_callback() -> None:
    server = _make_server()
    reason = 'AUTHORIZED_MODELS policy denied model "gpt-5" at generation 5'
    server._execute_via_agent_stream = _deny_with(PolicyDeniedException(reason))

    request = ExecuteRequest(
        execution_id="exec-policy-denied",
        message="hi",
        platform_api_url="http://oe:8000",
    )

    # The denial still terminates the AER request (HTTPException), like any
    # other terminal error — the improvement is the structured metadata, below.
    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(request)
    assert reason in exc_info.value.detail

    # Live stream: one ERROR chunk tagged as a policy denial. The human-readable
    # `error` keeps the full exception text ("Policy denied: <reason>") for
    # backward compatibility; the bare reason rides in metadata.
    chunk = _err_chunk(server)
    assert chunk.kwargs.get("error") == f"Policy denied: {reason}"
    meta = chunk.kwargs.get("metadata") or {}
    assert meta.get("error_code") == POLICY_DENIED_ERROR_CODE
    assert meta.get("reason") == reason

    # Terminal callback carries the same metadata so it flows through once the
    # OE preserves terminal error metadata (forward-compatible — non-streaming
    # consumers can't rely on it until that lands).
    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    assert callback.get("status") == "ERROR"
    cb_meta = callback.get("metadata") or {}
    assert cb_meta.get("error_code") == POLICY_DENIED_ERROR_CODE
    assert cb_meta.get("reason") == reason


async def test_policy_denied_includes_guardrail_identity_when_present() -> None:
    server = _make_server()
    exc = PolicyDeniedException(
        "input guardrail blocked",
        guardrail_meta=GuardrailMeta(guardrail_id="gr-1", guardrail_category="pii"),
    )
    server._execute_via_agent_stream = _deny_with(exc)

    request = ExecuteRequest(
        execution_id="exec-guardrail-denied",
        message="hi",
        platform_api_url="http://oe:8000",
    )

    with pytest.raises(HTTPException):
        await server._handle_execute(request)

    meta = _err_chunk(server).kwargs.get("metadata") or {}
    assert meta.get("error_code") == POLICY_DENIED_ERROR_CODE
    assert meta.get("guardrail_id") == "gr-1"
    assert meta.get("guardrail_category") == "pii"


async def test_non_policy_error_is_not_tagged_as_policy_denied() -> None:
    """A plain exception must keep its generic ERROR surfacing — no false
    ``policy_denied`` tag (guards against the new branch over-catching)."""
    server = _make_server()

    async def _boom(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise RuntimeError("tool pod crashed")

    server._execute_via_agent_stream = _boom

    request = ExecuteRequest(
        execution_id="exec-generic-error",
        message="hi",
        platform_api_url="http://oe:8000",
    )

    with pytest.raises(HTTPException):
        await server._handle_execute(request)

    # Generic errors go through the catch-all, which reports an ERROR callback
    # with no policy metadata.
    callback = server._send_callback_body.await_args.args[1]
    assert callback.get("status") == "ERROR"
    assert (callback.get("metadata") or {}).get("error_code") != POLICY_DENIED_ERROR_CODE
