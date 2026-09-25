"""Tests for emit / emit_step."""

from __future__ import annotations

import http.server
import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.progress import emit, emit_step
from agent_engine_runner_shared.server.chunk_types import (
    DONE,
    ERROR,
    STEP,
    SUBAGENT_END,
    SUBAGENT_START,
    TEXT,
)


def _set_ctx(execution_id: str, oe_url: str):
    return set_execution_context(
        execution_id=execution_id,
        wrapper=None,
        oe_url=oe_url,
    )


# ---------------------------------------------------------------------------
# Happy path — emit_step (the common case)
# ---------------------------------------------------------------------------


def test_emit_step_posts_step_chunk() -> None:
    tokens = _set_ctx("exec-1", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit_step("Fetching page...")

            mock_client.post.assert_called_once_with(
                "http://oe:8000/stream/chunk",
                json={
                    "execution_id": "exec-1",
                    "chunk_type": STEP,
                    "content": "Fetching page...",
                    "metadata": {},
                },
            )
    finally:
        clear_execution_context(tokens)


def test_emit_step_strips_trailing_slash_from_oe_url() -> None:
    tokens = _set_ctx("exec-2", "http://oe:8000/")
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit_step("Parsing...")

            url = mock_client.post.call_args[0][0]
            assert url == "http://oe:8000/stream/chunk"
    finally:
        clear_execution_context(tokens)


# ---------------------------------------------------------------------------
# General emit() with a non-step event type
# ---------------------------------------------------------------------------


def test_emit_accepts_arbitrary_event_type() -> None:
    """The general emit() API lets tools choose the chunk_type."""
    tokens = _set_ctx("exec-emit", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit(event="custom_event", data="payload")

            mock_client.post.assert_called_once_with(
                "http://oe:8000/stream/chunk",
                json={
                    "execution_id": "exec-emit",
                    "chunk_type": "custom_event",
                    "content": "payload",
                    "metadata": {},
                },
            )
    finally:
        clear_execution_context(tokens)


# ---------------------------------------------------------------------------
# No-op outside execution context
# ---------------------------------------------------------------------------


def test_emit_is_noop_outside_context() -> None:
    """Called with no execution context set — must not raise or make HTTP calls."""
    with patch("agent_engine_runner_shared.progress._client") as mock_client:
        emit_step("should be ignored")
        emit(event="step", data="also ignored")
        mock_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Network failure is swallowed
# ---------------------------------------------------------------------------


def test_emit_swallows_http_error() -> None:
    tokens = _set_ctx("exec-3", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._client") as mock_client:
            _wire_stream_to_post(mock_client)
            mock_client.post.side_effect = httpx.ConnectError("connection refused")

            # Must not raise
            emit_step("Running...")
    finally:
        clear_execution_context(tokens)


# ---------------------------------------------------------------------------
# Reserved event types — tool code must not be able to close or spoof the stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reserved_event", [DONE, ERROR, TEXT, SUBAGENT_START, SUBAGENT_END])
def test_emit_rejects_reserved_event_types(reserved_event: str) -> None:
    """Reserved infrastructure event types must raise ValueError and not POST."""
    tokens = _set_ctx("exec-reserved", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._client") as mock_client:
            with pytest.raises(ValueError, match="reserved"):
                emit(event=reserved_event, data="should not be sent")
            mock_client.post.assert_not_called()
    finally:
        clear_execution_context(tokens)


def test_emit_rejects_reserved_event_before_checking_context() -> None:
    """Reserved-event check fires even outside execution context (programmer error wins)."""
    with patch("agent_engine_runner_shared.progress._client") as mock_client:
        with pytest.raises(ValueError, match="reserved"):
            emit(event=DONE, data="x")
        mock_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# httpx.InvalidURL must be swallowed (it extends Exception, not HTTPError)
# ---------------------------------------------------------------------------


def test_emit_swallows_invalid_url() -> None:
    tokens = _set_ctx("exec-invalid-url", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._client") as mock_client:
            _wire_stream_to_post(mock_client)
            mock_client.post.side_effect = httpx.InvalidURL("malformed url")

            # Must not raise
            emit_step("Running...")
    finally:
        clear_execution_context(tokens)


def test_emit_swallows_non_2xx_response() -> None:
    tokens = _set_ctx("exec-4", "http://oe:8000")
    try:
        with patch("agent_engine_runner_shared.progress._client") as mock_client:
            _wire_stream_to_post(mock_client)
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            mock_client.post.return_value = mock_response

            # Must not raise
            emit_step("Running...")
    finally:
        clear_execution_context(tokens)


# ---------------------------------------------------------------------------
# Connection reuse — module-level client is shared across calls
# ---------------------------------------------------------------------------


def test_emit_reuses_module_level_client() -> None:
    """Multiple emit calls must reuse the module-level _client, not create new ones."""
    import agent_engine_runner_shared.progress as progress_module

    # Clear module-level client state before test
    progress_module._client = None
    progress_module._client_oe_url = None

    tokens = _set_ctx("exec-reuse", "http://oe:8000")
    try:
        with patch(
            "agent_engine_runner_shared.progress.create_httpx_client_with_tls"
        ) as mock_create_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_create_client.return_value = mock_client

            emit_step("first")
            emit_step("second")
            emit_step("third")

            # create_httpx_client_with_tls should be called once (on first emit)
            # _get_client caches the returned client for subsequent calls
            mock_create_client.assert_called_once_with("http://oe:8000", 5.0)
            # The same client instance should be used for all three emits
            assert mock_client.post.call_count == 3
    finally:
        # Clean up module-level state
        progress_module._client = None
        progress_module._client_oe_url = None
        clear_execution_context(tokens)


# ---------------------------------------------------------------------------
# Integration: real HTTP server — validates wire format without mocking httpx
# ---------------------------------------------------------------------------


def test_emit_integration_wire_format() -> None:
    """emit_step sends a correctly-formed JSON body to a real TCP server."""
    received: list[dict] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: object) -> None:  # suppress server noise
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    tokens = _set_ctx("exec-wire", f"http://127.0.0.1:{port}")
    try:
        emit_step("Integration check")
    finally:
        clear_execution_context(tokens)

    thread.join(timeout=2)
    server.server_close()

    assert len(received) == 1
    assert received[0] == {
        "execution_id": "exec-wire",
        "chunk_type": STEP,
        "content": "Integration check",
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Owner-callback fallback: emit prefers the replica-specific owner
# URL and falls back to the service URL on any owner failure
# (transport error, redirect, or non-2xx response).
# ---------------------------------------------------------------------------

_SERVICE = "https://oe.ns.svc.cluster.local:8443"
_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"


def _set_ctx_owner(execution_id: str, oe_url: str, oe_owner_url: str):
    return set_execution_context(
        execution_id=execution_id,
        wrapper=None,
        oe_url=oe_url,
        oe_owner_url=oe_owner_url,
    )


class _SyncPostStream:
    def __init__(self, client: MagicMock, url: str, payload):
        self._client = client
        self._url = url
        self._payload = payload

    def __enter__(self):
        return self._client.post(self._url, json=self._payload)

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _wire_stream_to_post(mock_client: MagicMock) -> None:
    mock_client.stream.side_effect = lambda method, url, *, json, follow_redirects: _SyncPostStream(
        mock_client, url, json
    )


def test_emit_prefers_owner_url() -> None:
    tokens = _set_ctx_owner("exec-o1", _SERVICE, _OWNER)
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit_step("hello")

            mock_client.post.assert_called_once()
            assert mock_client.post.call_args.args[0] == f"{_OWNER}/stream/chunk"
    finally:
        clear_execution_context(tokens)


def test_emit_falls_back_to_service_on_owner_transport_error() -> None:
    tokens = _set_ctx_owner("exec-o2", _SERVICE, _OWNER)
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    raise httpx.ConnectError("owner refused")
                return MagicMock()

            mock_client.post.side_effect = side_effect
            emit_step("hello")

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
            # Identical body across the fallback.
            assert (
                mock_client.post.call_args_list[0].kwargs["json"]
                == mock_client.post.call_args_list[1].kwargs["json"]
            )
    finally:
        clear_execution_context(tokens)


def test_emit_falls_back_to_service_on_owner_http_error() -> None:
    tokens = _set_ctx_owner("exec-o4", _SERVICE, _OWNER)
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    return MagicMock(is_success=False, status_code=500)
                return MagicMock()

            mock_client.post.side_effect = side_effect
            emit_step("hello")

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
            # Identical body across the fallback; owner tried exactly once.
            assert (
                mock_client.post.call_args_list[0].kwargs["json"]
                == mock_client.post.call_args_list[1].kwargs["json"]
            )
    finally:
        clear_execution_context(tokens)


def test_emit_without_owner_uses_service_only() -> None:
    tokens = _set_ctx_owner("exec-o3", _SERVICE, None)
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit_step("hello")

            mock_client.post.assert_called_once()
            assert mock_client.post.call_args.args[0] == f"{_SERVICE}/stream/chunk"
    finally:
        clear_execution_context(tokens)


def test_emit_owner_failure_latches_for_later_emits() -> None:
    """A failed owner pre-attempt disables the owner for the rest of the
    execution: later emits go straight to the service instead of re-paying
    the owner pre-attempt timeout on every event."""
    tokens = _set_ctx_owner("exec-o5", _SERVICE, _OWNER)
    try:
        with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    raise httpx.ConnectError("owner refused")
                return MagicMock()

            mock_client.post.side_effect = side_effect
            emit_step("first")
            emit_step("second")

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [
                f"{_OWNER}/stream/chunk",
                f"{_SERVICE}/stream/chunk",
                f"{_SERVICE}/stream/chunk",
            ]
    finally:
        clear_execution_context(tokens)


def test_emit_owner_failure_latch_is_per_execution() -> None:
    """The latch resets with the execution context: a fresh request offers
    the owner URL again instead of inheriting a previous request's failure."""

    def run_once(execution_id: str) -> list:
        tokens = _set_ctx_owner(execution_id, _SERVICE, _OWNER)
        try:
            with patch("agent_engine_runner_shared.progress._get_client") as mock_get_client:
                mock_client = MagicMock()
                _wire_stream_to_post(mock_client)
                mock_get_client.return_value = mock_client

                def side_effect(url, json):
                    if url == f"{_OWNER}/stream/chunk":
                        raise httpx.ConnectError("owner refused")
                    return MagicMock()

                mock_client.post.side_effect = side_effect
                emit_step("hello")
                return [c.args[0] for c in mock_client.post.call_args_list]
        finally:
            clear_execution_context(tokens)

    expected = [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
    assert run_once("exec-o6a") == expected
    assert run_once("exec-o6b") == expected
