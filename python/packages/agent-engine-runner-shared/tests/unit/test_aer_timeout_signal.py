"""AER surfaces a deadline breach as a timeout, not a generic failure.

A tool that overruns is not a crash and not a policy denial: the work was
permitted and ran, it just ran too long. Consumers need to tell those apart to
offer "give it more time / retry" instead of surfacing it as a bug, so the AER
tags the ERROR chunk with ``metadata.error_code == "timeout"`` and answers 504
rather than 500. Whole-turn expiry uses a distinct code so consumers do not
confuse a run limit with a retryable call timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from agent_engine_runner_shared.models import ExecuteRequest
from agent_engine_runner_shared.secure_wrapper import ToolCallTimeoutError
from agent_engine_runner_shared.server.chunk_types import (
    ERROR,
    RUN_TIMEOUT_ERROR_CODE,
    TIMEOUT_ERROR_CODE,
)

pytestmark = pytest.mark.asyncio


def _make_server() -> Any:
    from agent_engine_runner_shared.server.aer import AERServer

    runtime = Mock()
    runtime.build_graph = AsyncMock(return_value=None)
    server = AERServer(runtime)
    server._send_stream_chunk = AsyncMock()
    server._send_callback_body = AsyncMock()
    return server


def _err_chunk(server: Any):
    calls = [
        c for c in server._send_stream_chunk.await_args_list if c.kwargs.get("chunk_type") == ERROR
    ]
    assert len(calls) == 1, f"expected exactly one ERROR chunk, got {len(calls)}"
    return calls[0]


@pytest.mark.parametrize("tool_name", ["slow_tool", "invoke_llm"])
async def test_call_timeout_emits_structured_error_chunk_and_callback(tool_name: str) -> None:
    server = _make_server()
    timeout = ToolCallTimeoutError(tool_name, 600.0, 601.4)

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise timeout

    server._execute_via_agent_stream = _stream

    request = ExecuteRequest(
        execution_id="exec-tool-timeout",
        message="hi",
        platform_api_url="http://oe:8000",
    )

    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(request)
    # 504, not 500: the caller can reasonably retry or allow more time.
    assert exc_info.value.status_code == 504
    assert tool_name in exc_info.value.detail

    chunk = _err_chunk(server)
    meta = chunk.kwargs.get("metadata") or {}
    assert meta.get("error_code") == TIMEOUT_ERROR_CODE
    assert meta.get("tool_name") == tool_name
    assert meta.get("elapsed_seconds") == pytest.approx(601.4)

    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    callback_meta = callback.get("metadata") or {}
    assert callback_meta.get("error_code") == TIMEOUT_ERROR_CODE
    assert callback_meta == meta


async def test_run_deadline_emits_distinct_error_chunk_and_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_engine_runner_shared.server import aer

    monkeypatch.setattr(aer, "EXECUTION_TIMEOUT", 0.01)
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        await asyncio.Event().wait()

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-run-timeout",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert exc_info.value.status_code == 504
    assert _err_chunk(server).kwargs["metadata"] == {"error_code": RUN_TIMEOUT_ERROR_CODE}
    assert server._send_callback_body.await_count == 1
    callback = server._send_callback_body.await_args.args[1]
    assert callback["metadata"] == {"error_code": RUN_TIMEOUT_ERROR_CODE}


async def test_tool_timeout_is_not_tagged_as_a_policy_denial() -> None:
    from agent_engine_runner_shared.server.chunk_types import POLICY_DENIED_ERROR_CODE

    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise ToolCallTimeoutError("slow_tool", 600.0, 601.0)

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-not-denied",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    meta = _err_chunk(server).kwargs.get("metadata") or {}
    assert meta.get("error_code") != POLICY_DENIED_ERROR_CODE


async def test_agent_timeout_before_run_deadline_keeps_legacy_code() -> None:
    server = _make_server()

    async def _stream(_agent, _ctx, _agent_input, _oe_url, _execution_id):
        raise asyncio.TimeoutError("private dependency detail")

    server._execute_via_agent_stream = _stream

    with pytest.raises(HTTPException) as exc_info:
        await server._handle_execute(
            ExecuteRequest(
                execution_id="exec-agent-timeout",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Execution timed out."
    assert _err_chunk(server).kwargs["metadata"] == {"error_code": TIMEOUT_ERROR_CODE}
    callback = server._send_callback_body.await_args.args[1]
    assert callback["metadata"] == {"error_code": TIMEOUT_ERROR_CODE}
