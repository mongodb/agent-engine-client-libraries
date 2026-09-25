"""Tests for emit_custom_event and the Tool Pod OE HTTP transport."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.custom_events import (
    clear_custom_event_transport,
    emit_custom_event,
    emit_custom_event_sync,
    install_tool_custom_event_transport,
    set_custom_event_transport,
)
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT


def _set_tool_ctx(
    *,
    execution_id: str = "exec-1",
    oe_url: str = "http://oe:8000",
):
    return set_execution_context(
        execution_id=execution_id,
        wrapper=None,
        oe_url=oe_url,
    )


@pytest.mark.asyncio
async def test_emit_custom_event_noop_without_transport() -> None:
    await emit_custom_event({"event": "step", "data": "hello"})


def test_emit_custom_event_sync_noop_without_transport() -> None:
    emit_custom_event_sync({"event": "step", "data": "hello"})


@pytest.mark.asyncio
async def test_emit_custom_event_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="JSON-object mapping"):
        await emit_custom_event("hello")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_emit_custom_event_rejects_non_json_values() -> None:
    with pytest.raises(TypeError, match="JSON-serializable"):
        await emit_custom_event({"bad": object()})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.asyncio
async def test_emit_custom_event_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(TypeError, match="JSON-serializable"):
        await emit_custom_event({"value": value})


@pytest.mark.asyncio
async def test_tool_transport_feature_off_fails_closed() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport(enabled=False)
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            with pytest.raises(RuntimeError, match="requires features.use_custom_parser"):
                await emit_custom_event({"event": "step", "data": "no"})
            mock_get_client.assert_not_called()
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


def test_tool_transport_feature_off_sync_fails_closed() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport(enabled=False)
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            with pytest.raises(RuntimeError, match="requires features.use_custom_parser"):
                emit_custom_event_sync({"event": "step", "data": "no"})
            mock_get_client.assert_not_called()
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_tool_transport_posts_opaque_custom_event_object() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            await emit_custom_event({"event": "step", "data": "Fetching account information"})

            mock_client.post.assert_called_once_with(
                "http://oe:8000/stream/chunk",
                json={
                    "execution_id": "exec-1",
                    "chunk_type": CUSTOM_EVENT,
                    "custom_event": {
                        "event": "step",
                        "data": "Fetching account information",
                    },
                },
            )
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


def test_tool_transport_sync_posts_opaque_custom_event_object() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            emit_custom_event_sync({"event": "step", "data": "sync tool"})

            mock_client.post.assert_called_once()
            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["custom_event"] == {"event": "step", "data": "sync tool"}
            assert "metadata" not in payload
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_tool_transport_posts_arbitrary_opaque_object() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            await emit_custom_event({"foo": 1, "nested": {"ok": True}})

            payload = mock_client.post.call_args.kwargs["json"]["custom_event"]
            assert payload == {"foo": 1, "nested": {"ok": True}}
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_tool_transport_swallows_transport_failure() -> None:
    tokens = _set_tool_ctx()
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_client.post.side_effect = RuntimeError("connection refused")
            mock_get_client.return_value = mock_client
            await emit_custom_event({"event": "step", "data": "still ok"})
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_dispatch_uses_installed_transport() -> None:
    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def emit_sync(self, payload) -> None:
            self.calls.append(dict(payload))

        async def emit(self, payload) -> None:
            self.calls.append(dict(payload))

    transport = RecordingTransport()
    token = set_custom_event_transport(transport)
    try:
        await emit_custom_event({"event": "step", "data": "payload"})
        emit_custom_event_sync({"event": "message", "data": "sync"})
        assert transport.calls == [
            {"event": "step", "data": "payload"},
            {"event": "message", "data": "sync"},
        ]
    finally:
        clear_custom_event_transport(token)


# ---------------------------------------------------------------------------
# Owner-callback fallback.
# ---------------------------------------------------------------------------

_SERVICE = "https://oe.ns.svc.cluster.local:8443"
_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"


def _set_tool_ctx_owner(*, oe_url: str, oe_owner_url):
    return set_execution_context(
        execution_id="exec-1",
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


@pytest.mark.asyncio
async def test_custom_event_prefers_owner_url() -> None:
    tokens = _set_tool_ctx_owner(oe_url=_SERVICE, oe_owner_url=_OWNER)
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client
            await emit_custom_event({"event": "step", "data": "hi"})

            mock_client.post.assert_called_once()
            assert mock_client.post.call_args.args[0] == f"{_OWNER}/stream/chunk"
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_custom_event_falls_back_on_owner_transport_error() -> None:
    import httpx

    tokens = _set_tool_ctx_owner(oe_url=_SERVICE, oe_owner_url=_OWNER)
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    raise httpx.ConnectError("owner refused")
                return MagicMock()

            mock_client.post.side_effect = side_effect
            await emit_custom_event({"event": "step", "data": "hi"})

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_custom_event_falls_back_on_owner_http_error() -> None:
    tokens = _set_tool_ctx_owner(oe_url=_SERVICE, oe_owner_url=_OWNER)
    transport_token = install_tool_custom_event_transport()
    try:
        with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_client.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    return MagicMock(is_success=False, status_code=500)
                return MagicMock()

            mock_client.post.side_effect = side_effect
            await emit_custom_event({"event": "step", "data": "hi"})

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
            # Identical payload across the fallback; owner tried exactly once.
            assert (
                mock_client.post.call_args_list[0].kwargs["json"]
                == mock_client.post.call_args_list[1].kwargs["json"]
            )
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)


@pytest.mark.asyncio
async def test_custom_event_owner_failure_latch_shared_with_emit() -> None:
    """The owner-failure latch is execution-scoped, not transport-scoped: a
    failed owner pre-attempt on the custom-event path also stops progress
    emits from re-trying the dead owner."""
    import httpx

    from agent_engine_runner_shared.progress import emit_step

    tokens = _set_tool_ctx_owner(oe_url=_SERVICE, oe_owner_url=_OWNER)
    transport_token = install_tool_custom_event_transport()
    try:
        with (
            patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_ce,
            patch("agent_engine_runner_shared.progress._get_client") as mock_get_pr,
        ):
            mock_client = MagicMock()
            _wire_stream_to_post(mock_client)
            mock_get_ce.return_value = mock_client
            mock_get_pr.return_value = mock_client

            def side_effect(url, json):
                if url == f"{_OWNER}/stream/chunk":
                    raise httpx.ConnectError("owner refused")
                return MagicMock()

            mock_client.post.side_effect = side_effect
            await emit_custom_event({"event": "step", "data": "hi"})
            emit_step("after owner failure")

            urls = [c.args[0] for c in mock_client.post.call_args_list]
            assert urls == [
                f"{_OWNER}/stream/chunk",
                f"{_SERVICE}/stream/chunk",
                f"{_SERVICE}/stream/chunk",
            ]
    finally:
        clear_custom_event_transport(transport_token)
        clear_execution_context(tokens)
