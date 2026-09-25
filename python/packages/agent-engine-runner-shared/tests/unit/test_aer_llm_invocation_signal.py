"""AER surfaces an LLM-provider failure with structured owner metadata.

A 401/404 from the tenant's LLM after OE approved the call is not a platform
crash. Consumers (OE persist, gateway invoke-owner) must classify it without
string-matching the error prose, so the AER tags the ERROR chunk and callback
with ``metadata.error_code == "llm_invocation_failed"`` and ``source == "llm"``
only when ``LLMInvocationError.source`` says the failure is provider-owned.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from agent_engine_runner_shared.models import ExecuteRequest
from agent_engine_runner_shared.secure_wrapper import ExternalAPICallError, LLMInvocationError
from agent_engine_runner_shared.server.chunk_types import (
    ERROR,
    LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    LLM_INVOCATION_ERROR_CODE,
    LLM_INVOCATION_ERROR_SOURCE,
    POLICY_DENIED_ERROR_CODE,
    TIMEOUT_ERROR_CODE,
    TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
)
from agent_engine_runner_shared.tool_api_error import ToolAPIError

pytestmark = pytest.mark.asyncio


def _make_server() -> Any:
    from agent_engine_runner_shared.server.aer import AERServer

    server = AERServer(Mock())
    server._send_stream_chunk = AsyncMock()
    server._send_callback_body = AsyncMock()
    return server


def _err_chunk(server: Any):
    calls = _err_chunks(server)
    assert len(calls) == 1, f"expected exactly one ERROR chunk, got {len(calls)}"
    return calls[0]


def _err_chunks(server: Any):
    return [
        c for c in server._send_stream_chunk.await_args_list if c.kwargs.get("chunk_type") == ERROR
    ]


async def test_llm_invocation_error_emits_structured_error_chunk_and_callback() -> None:
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise LLMInvocationError("Resource not found", source=LLM_INVOCATION_ERROR_SOURCE)

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-404",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )
    assert exc_info.value.status_code == 500
    assert "Resource not found" in exc_info.value.detail

    chunk = _err_chunk(server)
    meta = chunk.kwargs.get("metadata") or {}
    assert meta.get("error_code") == LLM_INVOCATION_ERROR_CODE
    assert meta.get("code") == LLM_INVOCATION_ERROR_CODE
    assert meta.get("source") == LLM_INVOCATION_ERROR_SOURCE

    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == LLM_INVOCATION_ERROR_CODE
    assert callback_meta.get("source") == LLM_INVOCATION_ERROR_SOURCE


async def test_llm_invocation_error_is_not_tagged_as_timeout_or_policy() -> None:
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise LLMInvocationError("missing subscription key", source=LLM_INVOCATION_ERROR_SOURCE)

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-not-other",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    meta = _err_chunk(server).kwargs.get("metadata") or {}
    assert meta.get("error_code") != TIMEOUT_ERROR_CODE
    assert meta.get("error_code") != POLICY_DENIED_ERROR_CODE


async def test_uncoded_llm_invocation_error_does_not_stamp_provider_owner() -> None:
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise LLMInvocationError("SSE stream ended without done signal (truncated response)")

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-truncated",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )
    assert exc_info.value.status_code == 500

    err_chunks = [
        c for c in server._send_stream_chunk.await_args_list if c.kwargs.get("chunk_type") == ERROR
    ]
    assert err_chunks == []

    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") != LLM_INVOCATION_ERROR_CODE
    assert callback_meta.get("source") != LLM_INVOCATION_ERROR_SOURCE


async def test_credential_coded_llm_error_replaces_code_and_drops_source() -> None:
    """A pod-stamped credential code supersedes the generic invocation code.

    ``source`` must NOT ride along: the gateway checks source first, and
    ``llm`` attribution would read as provider flakiness when the fix is the
    customer's own project secret.
    """
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise LLMInvocationError(
            "provider rejected the configured API key",
            source=LLM_INVOCATION_ERROR_SOURCE,
            error_code=LLM_CREDENTIAL_REJECTED_ERROR_CODE,
        )

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-401",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    chunk_meta = _err_chunk(server).kwargs.get("metadata") or {}
    assert chunk_meta.get("error_code") == LLM_CREDENTIAL_REJECTED_ERROR_CODE
    assert chunk_meta.get("code") == LLM_CREDENTIAL_REJECTED_ERROR_CODE
    assert "source" not in chunk_meta

    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == LLM_CREDENTIAL_REJECTED_ERROR_CODE
    assert "source" not in callback_meta


def _tool_auth_failure() -> ExternalAPICallError:
    return ExternalAPICallError(
        "OpenWeather API call failed: HTTP 401 AUTH_FAILED",
        ToolAPIError(classification="AUTH_FAILED", http_status=401),
    )


async def test_generic_catch_tags_tool_credential_rejection() -> None:
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise _tool_auth_failure()

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-tool-401",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    # The generic catch reports only through the terminal callback.
    assert _err_chunks(server) == []

    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == TOOL_CREDENTIAL_REJECTED_ERROR_CODE
    assert callback_meta.get("code") == TOOL_CREDENTIAL_REJECTED_ERROR_CODE
    assert "source" not in callback_meta


async def test_generic_catch_tags_wrapped_tool_credential_rejection() -> None:
    """Frameworks that nest the tool error under a wrapper still classify."""
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        try:
            raise _tool_auth_failure()
        except ExternalAPICallError as inner:
            raise RuntimeError("graph node failed") from inner

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-tool-401-wrapped",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == TOOL_CREDENTIAL_REJECTED_ERROR_CODE


async def test_generic_catch_tags_wrapped_credential_coded_llm_error() -> None:
    """A framework-wrapped LLMInvocationError keeps the pod's stamped code."""
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise RuntimeError("graph node failed") from LLMInvocationError(
            "provider rejected the configured API key",
            source=LLM_INVOCATION_ERROR_SOURCE,
            error_code=LLM_CREDENTIAL_REJECTED_ERROR_CODE,
        )

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-401-wrapped",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == LLM_CREDENTIAL_REJECTED_ERROR_CODE


async def test_unknown_coded_llm_error_falls_back_to_generic() -> None:
    """The AER honors only the pod's credential stamp on LLMInvocationError.

    The exception type is agent-raisable and the live relay forwards pod
    frames verbatim, so an unrecognized ``error_code`` must degrade to the
    generic ``llm_invocation_failed`` classification — matching the unary and
    replay paths, where the OE's closed-set sanitizer drops it.
    """
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise LLMInvocationError(
            "model exploded",
            source=LLM_INVOCATION_ERROR_SOURCE,
            error_code="platform_outage",
        )

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-unknown-code",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    chunk_meta = _err_chunk(server).kwargs.get("metadata") or {}
    assert chunk_meta.get("error_code") == LLM_INVOCATION_ERROR_CODE
    assert chunk_meta.get("source") == LLM_INVOCATION_ERROR_SOURCE

    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == LLM_INVOCATION_ERROR_CODE
    assert callback_meta.get("source") == LLM_INVOCATION_ERROR_SOURCE


async def test_generic_catch_ignores_unknown_wrapped_llm_code() -> None:
    """A wrapped LLMInvocationError carrying a forged code stays uncoded."""
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise RuntimeError("graph node failed") from LLMInvocationError(
            "model exploded",
            source=LLM_INVOCATION_ERROR_SOURCE,
            error_code=TIMEOUT_ERROR_CODE,
        )

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-llm-forged-code-wrapped",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    callback = server._send_callback_body.await_args.args[1]
    assert not callback.get("metadata")


async def test_generic_catch_ignores_implicit_context_of_auth_failure() -> None:
    """An unrelated error raised while handling a tool auth failure is a bug,
    not a credential rejection: the implicit ``__context__`` chain is not
    walked (TS parity), so the run stays uncoded.
    """
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        try:
            raise _tool_auth_failure()
        except ExternalAPICallError:
            raise RuntimeError("cache bug")  # no `from`

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-tool-context-bug",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    callback = server._send_callback_body.await_args.args[1]
    assert not callback.get("metadata")


async def test_generic_catch_leaves_other_failures_uncoded() -> None:
    """Fail-open: a plain crash must not be relabeled as a secret problem."""
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise RuntimeError("plain agent crash")

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-plain-crash",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    callback = server._send_callback_body.await_args.args[1]
    assert not callback.get("metadata")
