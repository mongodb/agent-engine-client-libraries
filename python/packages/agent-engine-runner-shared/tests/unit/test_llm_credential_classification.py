"""Credential-rejection classification on the invoke_llm path.

When the LLM provider rejects the configured key (401/403), the tool pod
stamps ``error_code="llm_credential_rejected"`` on the failure so downstream
consumers (OE persist, gateway owner attribution, playground) can classify it
as a customer-secret problem without string-matching provider prose. The
predicate is deliberately status-only: anything unrecognized must keep the
existing generic classification.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_engine_runner_shared.models import Message, ToolExecuteResponse
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import LLMInvocationError
from agent_engine_runner_shared.server.chunk_types import (
    LLM_CREDENTIAL_REJECTED_ERROR_CODE,
)
from agent_engine_runner_shared.utils import is_llm_credential_rejection


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class _ResponseError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.response = Mock(status_code=status_code)


class _CodeError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


class TestIsLlmCredentialRejection:
    @pytest.mark.parametrize("status", [401, 403])
    def test_status_code_attr(self, status: int) -> None:
        assert is_llm_credential_rejection(_StatusError("nope", status)) is True

    @pytest.mark.parametrize("status", [401, 403])
    def test_httpx_response_attr(self, status: int) -> None:
        assert is_llm_credential_rejection(_ResponseError("nope", status)) is True

    @pytest.mark.parametrize("status", [401, 403])
    def test_int_code_attr(self, status: int) -> None:
        assert is_llm_credential_rejection(_CodeError("nope", status)) is True

    def test_walks_cause_chain(self) -> None:
        try:
            try:
                raise _StatusError("bad key", 401)
            except _StatusError as inner:
                raise RuntimeError("adapter failed") from inner
        except RuntimeError as outer:
            assert is_llm_credential_rejection(outer) is True

    def test_implicit_context_is_not_walked(self) -> None:
        # Python sets ``__context__`` implicitly on a bare re-raise inside an
        # except block; only the explicit ``__cause__`` chain carries the
        # re-raise signal. Walking the implicit context would relabel an
        # unrelated failure raised while handling an auth error as a
        # credential rejection. Mirrors the TS twin, which walks only
        # ``cause``.
        try:
            try:
                raise _StatusError("bad key", 403)
            except _StatusError:
                raise RuntimeError("unannotated re-raise")  # no `from`
        except RuntimeError as outer:
            assert is_llm_credential_rejection(outer) is False

    def test_raise_from_none_suppresses_the_chain(self) -> None:
        # `raise X from None` is an explicit author signal that the failure
        # being handled is unrelated; the walk must not reach through it.
        try:
            try:
                raise _StatusError("bad key", 401)
            except _StatusError:
                raise RuntimeError("masked") from None
        except RuntimeError as outer:
            assert is_llm_credential_rejection(outer) is False

    @pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
    def test_other_statuses_stay_generic(self, status: int) -> None:
        assert is_llm_credential_rejection(_StatusError("nope", status)) is False

    def test_no_status_stays_generic(self) -> None:
        assert is_llm_credential_rejection(RuntimeError("plain crash")) is False

    def test_bool_status_code_is_not_a_status(self) -> None:
        class _BoolStatus(Exception):
            status_code = True

        assert is_llm_credential_rejection(_BoolStatus()) is False

    def test_string_code_is_not_matched(self) -> None:
        # Closed vocabulary: a string "code" like "UNAUTHENTICATED" or
        # "invalid_api_key" is provider prose, not a status — never matched.
        class _StringCode(Exception):
            code = "UNAUTHENTICATED"

        assert is_llm_credential_rejection(_StringCode()) is False

    def test_cyclic_cause_chain_terminates(self) -> None:
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first
        assert is_llm_credential_rejection(first) is False


def _make_tool_server() -> Any:
    from agent_engine_runner_shared.server.tool import ToolServer

    mock_runtime = Mock()
    mock_runtime._tools = {}
    mock_runtime._tool_definitions = {}
    mock_runtime._graph_builder = None
    mock_runtime.agent_config.feature_enabled = Mock(return_value=False)
    return ToolServer(mock_runtime)


def _failing_chunks(error: Exception):
    async def _chunks(_request: Any):
        raise error
        yield  # pragma: no cover — marks this an async generator

    return _chunks


@pytest.mark.asyncio
class TestToolPodErrorCodeEmission:
    async def test_nonstreaming_stamps_credential_code(self, monkeypatch) -> None:
        server = _make_tool_server()
        monkeypatch.setattr(
            server, "_stream_llm_chunks", _failing_chunks(_StatusError("bad key", 401))
        )

        response = await server._handle_invoke_llm(Mock())

        assert response.status == "error"
        assert response.error_code == LLM_CREDENTIAL_REJECTED_ERROR_CODE
        # The wire payload omits the field entirely when unset; when set it is
        # exactly the closed code, never provider text.
        assert json.loads(response.model_dump_json())["error_code"] == (
            LLM_CREDENTIAL_REJECTED_ERROR_CODE
        )

    async def test_streaming_stamps_credential_code(self, monkeypatch) -> None:
        server = _make_tool_server()
        monkeypatch.setattr(
            server, "_stream_llm_chunks", _failing_chunks(_StatusError("forbidden", 403))
        )

        events = [
            json.loads(frame.removeprefix("data: "))
            async for frame in server._handle_invoke_llm_stream(Mock())
        ]

        assert len(events) == 1
        assert events[0]["error_code"] == LLM_CREDENTIAL_REJECTED_ERROR_CODE

    @pytest.mark.parametrize("streaming", [False, True], ids=["nonstreaming", "streaming"])
    async def test_non_auth_failure_carries_no_code(self, monkeypatch, streaming: bool) -> None:
        server = _make_tool_server()
        monkeypatch.setattr(
            server, "_stream_llm_chunks", _failing_chunks(_StatusError("slow down", 429))
        )

        if streaming:
            events = [
                json.loads(frame.removeprefix("data: "))
                async for frame in server._handle_invoke_llm_stream(Mock())
            ]
            assert len(events) == 1
            assert events[0]["error"]
            assert "error_code" not in events[0]
        else:
            response = await server._handle_invoke_llm(Mock())
            assert response.status == "error"
            assert response.error_code is None


class TestSecureLLMProxyErrorCode:
    def test_unary_error_response_carries_code(self) -> None:
        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="error",
                error="provider rejected the key",
                error_code=LLM_CREDENTIAL_REJECTED_ERROR_CODE,
            )

            with pytest.raises(LLMInvocationError) as exc_info:
                list(proxy.stream(messages))

        assert exc_info.value.error_code == LLM_CREDENTIAL_REJECTED_ERROR_CODE
        assert exc_info.value.source == "llm"

    def test_stream_error_event_carries_code(self) -> None:
        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(
                    json.dumps(
                        {
                            "error": "provider rejected the key",
                            "error_code": LLM_CREDENTIAL_REJECTED_ERROR_CODE,
                        }
                    )
                )

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ),
            patch("agent_engine_runner_shared.secure_llm_proxy.sleep_oe_stream_retry"),
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(LLMInvocationError) as exc_info:
                list(proxy.stream(messages, step=7))

        assert exc_info.value.error_code == LLM_CREDENTIAL_REJECTED_ERROR_CODE
        assert exc_info.value.source == "llm"

    def test_stream_error_event_without_code_stays_uncoded(self) -> None:
        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"error": "model exploded"}))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ),
            patch("agent_engine_runner_shared.secure_llm_proxy.sleep_oe_stream_retry"),
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(LLMInvocationError) as exc_info:
                list(proxy.stream(messages, step=7))

        assert exc_info.value.error_code is None
        assert exc_info.value.source == "llm"
