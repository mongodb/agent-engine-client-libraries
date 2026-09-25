"""Tests for metadata/payload field resolution on ExecuteRequest, ExecutorCallbackRequest, and AER server."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from agent_engine_runner_shared.models import (
    ExecuteRequest,
    ExecutorCallbackRequest,
    PendingInterrupt,
    StreamingResult,
)
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT


class _AsyncPostStream:
    def __init__(self, post, url: str, payload: dict[str, Any]) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    async def __aenter__(self):
        return await self._post(self._url, json=self._payload)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _wire_stream_to_post(client) -> None:
    client.stream.side_effect = lambda method, url, *, json, follow_redirects: _AsyncPostStream(
        client.post, url, json
    )


class TestExecuteRequestMetadataPayload:
    """ExecuteRequest accepts optional metadata and payload fields."""

    def test_metadata_and_payload_accepted(self):
        req = ExecuteRequest(
            execution_id="exec-1",
            message="hello",
            platform_api_url="http://oe:8000",
            metadata={
                "langgraph_branch_point": {
                    "thread_id": "t-1",
                    "checkpoint_id": "ckpt-1",
                }
            },
            payload={"context": "from-payload"},
        )
        assert req.metadata == {
            "langgraph_branch_point": {
                "thread_id": "t-1",
                "checkpoint_id": "ckpt-1",
            }
        }
        assert req.payload == {"context": "from-payload"}

    def test_metadata_and_payload_default_to_none(self):
        req = ExecuteRequest(
            execution_id="exec-2",
            message="hello",
            platform_api_url="http://oe:8000",
        )
        assert req.metadata is None
        assert req.payload is None

    def test_round_trip_through_model_dump_validate(self):
        req = ExecuteRequest(
            execution_id="exec-3",
            message="hello",
            platform_api_url="http://oe:8000",
            metadata={"thread_id": "t-1"},
            payload={"context": "test"},
            root_session_id="root-session",
            root_execution_id="root-exec",
        )
        data = req.model_dump()
        restored = ExecuteRequest.model_validate(data)
        assert restored.metadata == req.metadata
        assert restored.payload == req.payload
        assert restored.root_session_id == "root-session"
        assert restored.root_execution_id == "root-exec"

    def test_conflicting_message_top_level_and_payload_rejected(self):
        """Message in both the top level and payload → clean 400, no payload leak."""
        from fastapi import HTTPException

        from agent_engine_runner_shared.server.aer import _reject_ambiguous_message

        request = ExecuteRequest(
            execution_id="exec-conflict",
            message="top-level",
            platform_api_url="http://oe:8000",
            payload={"message": "from-payload"},
        )
        with pytest.raises(HTTPException) as exc_info:
            _reject_ambiguous_message(request)
        assert exc_info.value.status_code == 400
        assert "Ambiguous request" in exc_info.value.detail
        # The opaque payload must never appear in the surfaced error.
        assert "from-payload" not in exc_info.value.detail

    def test_old_payload_with_removed_fields_still_parses(self):
        # checkpoint_id and thread_id were removed from the model (resume
        # state now rides in the opaque metadata dict; session_id is the only
        # session identifier). An old OE that still sends them must not break
        # parsing — Pydantic's default extra="ignore" drops unknown fields.
        data = {
            "execution_id": "exec-4",
            "message": "hello",
            "platform_api_url": "http://oe:8000",
            "thread_id": "t-old",
            "checkpoint_id": "ckpt-old",
        }
        req = ExecuteRequest.model_validate(data)
        assert req.suspend_generation is None
        assert req.metadata is None
        assert req.payload is None
        assert not hasattr(req, "thread_id")
        assert not hasattr(req, "checkpoint_id")

    def test_payload_only_message_not_promoted_at_model_level(self):
        # The model is a plain carrier; promoting payload["message"] to the
        # top-level message happens in the AER's _resolve_invocation_params,
        # not at construction. See test_reads_message_from_payload_when_top_level_absent.
        req = ExecuteRequest(
            execution_id="exec-5",
            platform_api_url="http://oe:8000",
            payload={"message": "from-payload-only"},
        )
        assert req.message == ""
        assert req.payload == {"message": "from-payload-only"}


class TestExecutorCallbackRequestMetadata:
    """ExecutorCallbackRequest accepts optional metadata field."""

    def test_metadata_accepted(self):
        req = ExecutorCallbackRequest(
            execution_id="exec-1",
            status="SUSPENDED",
            suspend_generation=3,
            metadata={"checkpoint_id": "ckpt-1", "function_call_id": "fc-1"},
        )
        assert req.suspend_generation == 3
        assert req.metadata == {"checkpoint_id": "ckpt-1", "function_call_id": "fc-1"}

    def test_metadata_defaults_to_none(self):
        req = ExecutorCallbackRequest(
            execution_id="exec-2",
            status="COMPLETED",
        )
        assert req.suspend_generation is None
        assert req.metadata is None

    def test_round_trip_through_model_dump_validate(self):
        req = ExecutorCallbackRequest(
            execution_id="exec-3",
            status="SUSPENDED",
            metadata={"checkpoint_id": "ckpt-1", "function_call_id": "fc-1"},
        )
        data = req.model_dump()
        restored = ExecutorCallbackRequest.model_validate(data)
        assert restored.metadata == req.metadata

    def test_old_payload_with_removed_checkpoint_id_still_parses(self):
        # checkpoint_id was removed from the model; an OE/AER that still sends
        # it must not break parsing (extra="ignore" drops the unknown field).
        data = {
            "execution_id": "exec-4",
            "status": "COMPLETED",
            "result": "done",
            "checkpoint_id": "ckpt-old",
        }
        req = ExecutorCallbackRequest.model_validate(data)
        assert req.metadata is None
        assert not hasattr(req, "checkpoint_id")


# ---------------------------------------------------------------------------
# AER server test helpers
# ---------------------------------------------------------------------------


def _make_aer_server():
    from agent_engine_runner_shared.server.aer import AERServer

    runtime = Mock()
    server = AERServer(runtime)
    server._send_callback_body = AsyncMock()
    return server


def _make_real_callback_server():
    from agent_engine_runner_shared.server.aer import AERServer
    from agent_engine_runner_shared.server.callback_delivery import CallbackDelivery

    server = AERServer.__new__(AERServer)
    server._client = None
    server._client_lock = asyncio.Lock()
    server._owner_callback_url = {}
    server._callback_delivery = CallbackDelivery(lambda url: server._get_client(url))
    return server


def _wire_execute_path(server, *, execution_outcome=None):
    server.runtime.get_agent = Mock(return_value=Mock())
    server.runtime._memory_writer = None
    server.runtime.org_id = "org-1"

    async def _fake_stream(*_args: Any, **_kwargs: Any) -> Any:
        return execution_outcome

    server._execute_via_agent_stream = _fake_stream


# ---------------------------------------------------------------------------
# AER _handle_execute metadata/payload resolution tests
# ---------------------------------------------------------------------------


class TestHandleExecuteMetadataResolution:
    """_handle_execute resolves session_id and reads checkpoint_id from metadata."""

    @pytest.mark.asyncio
    async def test_session_id_used_when_present(self):
        from agent_engine_sdk import AgentInput, RequestContext

        from agent_engine_runner_shared.context import get_current_trace_id

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured: dict[str, Any] = {}

        async def capture_stream(
            agent, ctx: RequestContext, agent_input: AgentInput, oe_url, execution_id
        ):
            captured["ctx"] = ctx
            captured["trace_id"] = get_current_trace_id()
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        request = ExecuteRequest(
            execution_id="exec-sess-prio",
            message="hello",
            platform_api_url="http://oe:8000",
            session_id="sess-from-oe",
            platform_trace_id="0123456789abcdef0123456789abcdef",
        )

        await server._handle_execute(request)

        assert captured["ctx"].session_id == "sess-from-oe"
        assert captured["trace_id"] == request.platform_trace_id
        assert get_current_trace_id() is None

    @pytest.mark.asyncio
    async def test_falls_back_to_execution_id_when_no_session_id(self):
        from agent_engine_sdk import AgentInput, RequestContext

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured: dict[str, Any] = {}

        async def capture_stream(
            agent, ctx: RequestContext, agent_input: AgentInput, oe_url, execution_id
        ):
            captured["ctx"] = ctx
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        request = ExecuteRequest(
            execution_id="exec-no-session",
            message="hello",
            platform_api_url="http://oe:8000",
        )

        await server._handle_execute(request)

        assert captured["ctx"].session_id == "exec-no-session"

    @pytest.mark.asyncio
    async def test_reads_message_from_payload_when_top_level_absent(self):
        from agent_engine_sdk import AgentInput

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured_input: dict[str, Any] = {}

        async def capture_stream(agent, ctx, agent_input: AgentInput, oe_url, execution_id):
            captured_input.update(agent_input.payload)
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        # No top-level message: a payload-only caller's message is promoted.
        request = ExecuteRequest(
            execution_id="exec-payload-msg",
            platform_api_url="http://oe:8000",
            payload={"message": "from-payload"},
        )

        await server._handle_execute(request)

        assert captured_input["message"] == "from-payload"

    @pytest.mark.asyncio
    async def test_falls_back_to_top_level_message_when_payload_absent(self):
        from agent_engine_sdk import AgentInput

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured_input: dict[str, Any] = {}

        async def capture_stream(agent, ctx, agent_input: AgentInput, oe_url, execution_id):
            captured_input.update(agent_input.payload)
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        request = ExecuteRequest(
            execution_id="exec-msg-fallback",
            message="top-level-msg",
            platform_api_url="http://oe:8000",
        )

        await server._handle_execute(request)

        assert captured_input["message"] == "top-level-msg"

    @pytest.mark.asyncio
    async def test_non_string_message_in_payload_falls_back_to_top_level(self):
        from agent_engine_sdk import AgentInput

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured_input: dict[str, Any] = {}

        async def capture_stream(agent, ctx, agent_input: AgentInput, oe_url, execution_id):
            captured_input.update(agent_input.payload)
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        request = ExecuteRequest(
            execution_id="exec-type-msg",
            message="top-msg",
            platform_api_url="http://oe:8000",
            payload={"message": 999},
        )

        await server._handle_execute(request)

        assert captured_input["message"] == "top-msg"

    @pytest.mark.asyncio
    async def test_reads_checkpoint_id_from_metadata_on_resume(self):
        from agent_engine_sdk import AgentInput, RequestContext

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        captured: dict[str, Any] = {}

        async def capture_stream(
            agent, ctx: RequestContext, agent_input: AgentInput, oe_url, execution_id
        ):
            captured["ctx"] = ctx
            return StreamingResult(content="ok", messages=[])

        server._execute_via_agent_stream = capture_stream

        request = ExecuteRequest(
            execution_id="exec-ckpt-meta",
            message="resume msg",
            platform_api_url="http://oe:8000",
            resume=True,
            metadata={"checkpoint_id": "meta-ckpt-1", "thread_id": "t-1"},
        )

        await server._handle_execute(request)

        # checkpoint_id is opaque framework state carried on ctx.metadata; the
        # langgraph adapter reads it from there.
        assert captured["ctx"].resume is True
        assert captured["ctx"].metadata == {"checkpoint_id": "meta-ckpt-1", "thread_id": "t-1"}


# ---------------------------------------------------------------------------
# AER _report_callback metadata tests
# ---------------------------------------------------------------------------


class TestReportCallbackMetadata:
    """_report_callback keeps HITL metadata separate from checkpoints."""

    @pytest.mark.asyncio
    async def test_suspended_callback_forwards_opaque_metadata(self):
        from unittest.mock import MagicMock

        server = _make_real_callback_server()

        captured_body: dict[str, Any] = {}
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def capture_post(url, *, json=None, **kwargs):
            captured_body.update(json)
            return mock_response

        mock_client = MagicMock()
        mock_client.post = capture_post
        _wire_stream_to_post(mock_client)

        async def _fake_get_client(url: str):
            return mock_client

        server._get_client = _fake_get_client

        # The AER forwards the adapter's opaque metadata verbatim — it does not
        # construct checkpoint_id/thread_id itself.
        await server._report_callback(
            "http://oe:8000",
            "exec-cb-1",
            status="SUSPENDED",
            suspend_reason="awaiting_human_review",
            suspend_context={"claim_id": "CLM-1"},
            interrupts=[
                PendingInterrupt(id="int-1", value={"question": "Approve?"}),
                PendingInterrupt(id="int-2", value=["a", "b"]),
            ],
            resume_schema={"type": "object", "required": ["resume_map"]},
            metadata={"checkpoint_id": "ckpt-suspend-1", "function_call_id": "fc-1"},
        )

        assert "checkpoint_id" not in captured_body
        assert "suspend_generation" not in captured_body
        assert captured_body["metadata"]["checkpoint_id"] == "ckpt-suspend-1"
        assert captured_body["metadata"]["function_call_id"] == "fc-1"
        assert captured_body["interrupts"] == [
            {"id": "int-1", "value": {"question": "Approve?"}},
            {"id": "int-2", "value": ["a", "b"]},
        ]
        assert captured_body["resume_schema"] == {
            "type": "object",
            "required": ["resume_map"],
        }

    @pytest.mark.asyncio
    async def test_callback_serialization_failure_reaches_execute_error_path(self):
        server = _make_real_callback_server()
        server._get_client = AsyncMock(return_value=Mock())

        with pytest.raises(TypeError, match="set is not JSON serializable"):
            await server._report_callback(
                "http://oe:8000",
                "exec-bad-callback",
                status="SUSPENDED",
                interrupts=[PendingInterrupt(id="approval", value={"not-json"})],
                resume_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_callback_retries_transient_transport_failures(self):
        from unittest.mock import MagicMock

        import httpx

        server = _make_real_callback_server()
        attempts = 0
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def flaky_post(url, *, json=None, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError("OE unavailable")
            return mock_response

        mock_client = MagicMock()
        mock_client.post = flaky_post
        _wire_stream_to_post(mock_client)

        async def _fake_get_client(url: str):
            return mock_client

        server._get_client = _fake_get_client

        await server._report_callback(
            "http://oe:8000", "exec-retry", status="COMPLETED", result="done"
        )

        assert attempts == 3

    @pytest.mark.asyncio
    async def test_completed_callback_passes_framework_metadata_through(self):
        from unittest.mock import MagicMock

        server = _make_real_callback_server()

        captured_body: dict[str, Any] = {}
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def capture_post(url, *, json=None, **kwargs):
            captured_body.update(json)
            return mock_response

        mock_client = MagicMock()
        mock_client.post = capture_post
        _wire_stream_to_post(mock_client)

        async def _fake_get_client(url: str):
            return mock_client

        server._get_client = _fake_get_client

        await server._report_callback(
            "http://oe:8000",
            "exec-cb-2",
            status="COMPLETED",
            result="done",
            metadata={
                "langgraph_checkpoint": {
                    "thread_id": "t-complete-1",
                    "checkpoint_id": "checkpoint-1",
                }
            },
        )

        assert captured_body["metadata"] == {
            "langgraph_checkpoint": {
                "thread_id": "t-complete-1",
                "checkpoint_id": "checkpoint-1",
            }
        }


# ---------------------------------------------------------------------------
# End-to-end backward compat
# ---------------------------------------------------------------------------


class TestCustomEventForwarding:
    """Output-parser custom_event rides on the /stream/chunk body."""

    @pytest.mark.asyncio
    async def test_send_stream_chunk_includes_custom_event(self):
        server = _make_aer_server()

        captured_body: dict[str, Any] = {}

        async def capture_post(client, url, payload, **kwargs):
            captured_body.update(payload)

        server._post_chunk_with_retries = capture_post
        server._get_client = AsyncMock(return_value=Mock())

        await server._send_stream_chunk(
            "http://oe:8000",
            "exec-custom",
            chunk_type=CUSTOM_EVENT,
            custom_event={"kind": "brief", "text": "hi"},
        )

        assert captured_body["custom_event"] == {"kind": "brief", "text": "hi"}
        # CUSTOM_EVENT is non-terminal, so the seq counter stays live.
        assert server._chunk_seq.get("exec-custom") == 1

    @pytest.mark.asyncio
    async def test_platform_chunk_carries_null_custom_event(self):
        """Platform chunks default custom_event to None so consumers can gate on it."""
        server = _make_aer_server()

        captured_body: dict[str, Any] = {}

        async def capture_post(client, url, payload, **kwargs):
            captured_body.update(payload)

        server._post_chunk_with_retries = capture_post
        server._get_client = AsyncMock(return_value=Mock())

        await server._send_stream_chunk(
            "http://oe:8000", "exec-text", chunk_type="text", content="hello"
        )

        assert captured_body["custom_event"] is None

    @pytest.mark.asyncio
    async def test_execute_stream_forwards_custom_event_and_skips_platform_handling(self):
        from agent_engine_sdk import AgentInput, RequestContext, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        events = [
            StreamEvent(data=None, event=CUSTOM_EVENT, custom_event={"kind": "brief"}),
            StreamEvent(data={"response": "done", "messages": []}, event="result"),
        ]

        class _FakeExec:
            async def __aiter__(self):
                for event in events:
                    yield event

        class _FakeAgent:
            def execute(self, ctx, agent_input):
                return _FakeExec()

        result = await server._execute_via_agent_stream(
            _FakeAgent(),
            RequestContext(),
            AgentInput(payload={"message": "hi"}),
            "http://oe:8000",
            "exec-1",
        )

        assert isinstance(result, StreamingResult)
        assert result.content == "done"
        # Exactly one chunk sent: the custom_event frame (CUSTOM_EVENT chunk_type).
        # The result event is terminal state, not a live chunk here.
        server._send_stream_chunk.assert_awaited_once_with(
            "http://oe:8000",
            "exec-1",
            chunk_type=CUSTOM_EVENT,
            custom_event={"kind": "brief"},
        )


class TestEndToEndBackwardCompat:
    """Old OE request (no metadata/payload) → AER completes and callbacks include metadata."""

    @pytest.mark.asyncio
    async def test_old_style_request_completion_callback_includes_metadata(self):
        from unittest.mock import MagicMock, patch

        from agent_engine_runner_shared.server.aer import AERServer

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        callbacks: list[dict[str, Any]] = []

        async def capture_callback(_oe_url: str, body: dict[str, Any], **_kwargs: Any) -> None:
            callbacks.append(
                {
                    key: value
                    for key, value in body.items()
                    if key != "execution_id" and value is not None
                }
            )

        server._send_callback_body = capture_callback

        _wire_execute_path(server, execution_outcome=StreamingResult(content="42", messages=[]))

        request = ExecuteRequest(
            execution_id="exec-e2e-old",
            message="hello",
            platform_api_url="http://oe:8000",
            thread_id="old-thread-1",
        )

        with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
            response = await server._handle_execute(request)

        assert response.status == "completed"
        assert callbacks == [
            {
                "status": "COMPLETED",
                "result": "42",
            },
        ]
