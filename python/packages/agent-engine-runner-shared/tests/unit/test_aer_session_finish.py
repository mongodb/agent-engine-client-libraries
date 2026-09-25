"""Tests for releasing a finished session's compute after /execute responds.

The SDK's ``request_session_finish()`` lets agent code mark
a session as done mid-turn. This module tests the AER-side wiring: the
``/execute`` route latches the request in ``_pending_session_finish`` during
``_handle_execute``'s COMPLETED branch, then - only via a FastAPI
``BackgroundTasks`` callback that runs after the HTTP response is sent - POSTs
``{platform_api_url}/executions/{execution_id}/finish`` to the OE.

This ordering is load-bearing: the OE may kill this pod with zero grace once
it honours the finish request, and a transport failure from a dead pod can
turn into a 502 that overwrites the finished run's result. So the finish POST
must never race the /execute response.

Scenarios:
1. No request -> no finish call at all.
2. Requested + normal completion -> exactly one finish POST, strictly after
   the completion callback (ordering asserted via a shared call log).
3. Suspended run -> no finish call (a suspended run must stay resumable).
4. Failing/errored run -> no finish call.
5. Pending memory writes drain before the finish POST fires; a writer stuck
   above zero still fires after the bounded timeout (no hang).
6. A finish POST that errors is swallowed - the /execute response is
   unaffected.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_engine_runner_shared.context import SessionFinishStatus, request_session_finish
from agent_engine_runner_shared.models import ExecuteRequest, InterruptResult, StreamingResult
from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.callback_delivery import (
    CallbackDelivery,
    _CallbackKey,
    _PendingCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> AERServer:
    """Construct an AERServer with mocks, bypassing FastAPI / httpx wiring."""
    runtime = MagicMock()
    runtime._memory_writer = None
    runtime.org_id = "org-1"
    runtime.get_agent = MagicMock(return_value=MagicMock())

    server = AERServer.__new__(AERServer)
    server.runtime = runtime  # type: ignore[attr-defined]
    server._client_lock = asyncio.Lock()  # type: ignore[attr-defined]
    server._a2a_registered = False  # type: ignore[attr-defined]
    server._chunk_seq = {}  # type: ignore[attr-defined]
    server._owner_callback_url = {}  # type: ignore[attr-defined]
    server._durability_owner_id = "owner-test"  # type: ignore[attr-defined]
    server._pending_session_finish = set()  # type: ignore[attr-defined]
    server._callback_delivery = CallbackDelivery(  # type: ignore[attr-defined]
        lambda url: server._get_client(url)  # type: ignore[attr-defined]
    )
    server._send_callback_body = AsyncMock()  # type: ignore[attr-defined]
    return server


def _execute_request(execution_id: str = "exec-1") -> ExecuteRequest:
    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url="http://oe",
        user_id="u-1",
        custom_headers={},
        resume=False,
    )


def _wire_execute_path(
    server: AERServer,
    *,
    execution_outcome: Optional[Any] = None,
    stream_raises: Optional[Exception] = None,
    call_finish: bool = False,
) -> None:
    """Stub the agent stream so ``_handle_execute`` runs end-to-end.

    When ``call_finish`` is set, the fake stream calls
    ``request_session_finish()`` while the execution context is live -
    mirroring an agent invoking the SDK helper mid-run (Task 4).
    """

    async def _fake_stream(*_args: Any, **_kwargs: Any) -> Any:
        if call_finish:
            request_session_finish()
        if stream_raises is not None:
            raise stream_raises
        return execution_outcome

    server._execute_via_agent_stream = _fake_stream  # type: ignore[assignment]
    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]


def _make_client(server: AERServer) -> TestClient:
    """Build a FastAPI app with only the AER's own routes wired up."""
    app = FastAPI()
    server.register_routes(app)
    return TestClient(app, raise_server_exceptions=True)


def _post_execute(client: TestClient, execution_id: str = "exec-1"):
    payload = _execute_request(execution_id).model_dump()
    return client.post("/execute", json=payload)


# ---------------------------------------------------------------------------
# 1. No request -> no finish call at all
# ---------------------------------------------------------------------------


def test_no_finish_request_means_no_finish_call() -> None:
    server = _make_server()
    outcome = StreamingResult(content="ok", messages=[])
    _wire_execute_path(server, execution_outcome=outcome, call_finish=False)
    server._send_callback_body = AsyncMock()  # type: ignore[assignment]
    server._release_finished_session = AsyncMock()  # type: ignore[assignment]

    client = _make_client(server)
    response = _post_execute(client)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    server._release_finished_session.assert_not_called()
    assert server._pending_session_finish == set()


# ---------------------------------------------------------------------------
# 2. Requested + normal completion -> exactly one POST, strictly after the
#    completion callback (i.e. after the response is being finalized).
# ---------------------------------------------------------------------------


def test_requested_and_completed_fires_exactly_one_finish_after_response() -> None:
    server = _make_server()
    outcome = StreamingResult(content="ok", messages=[])
    _wire_execute_path(server, execution_outcome=outcome, call_finish=True)

    call_log: list[str] = []

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        call_log.append("callback")

    server._send_callback_body = _capture_callback  # type: ignore[assignment]

    fake_client = MagicMock()

    async def _fake_post(url: str, json: dict) -> Any:
        call_log.append("finish")
        assert url == "http://oe/executions/exec-1/finish"
        response = MagicMock()
        response.raise_for_status = MagicMock(return_value=None)
        return response

    fake_client.post = _fake_post
    fake_client.stream.side_effect = lambda method, url, *, json, follow_redirects: (
        _AsyncPostStream(fake_client.post, url, json)
    )

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=fake_client)):
        client = _make_client(server)
        response = _post_execute(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"] == "ok"
    # Callback (part of the response body being assembled) happened, then -
    # and only then - the finish POST fired as a background task.
    assert call_log == ["callback", "finish"]
    assert server._pending_session_finish == set()


# ---------------------------------------------------------------------------
# 2b. The session-finish *latch* (not _pending_session_finish -
#     the ContextVar-backed holder request_session_finish() itself reads)
#     must be closed before anything in the finally block awaits. Otherwise
#     a task scheduled during the turn that happens to resume while
#     `await wrapper.close()` is suspended would see the latch still open
#     and get REQUESTED, even though it already missed the
#     is_session_finish_requested() drain and will never be acted on.
# ---------------------------------------------------------------------------


async def test_task_resuming_during_wrapper_close_await_gets_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server()
    outcome = StreamingResult(content="ok", messages=[])
    resume_during_close = asyncio.Event()
    result: dict[str, Any] = {}

    async def _fake_stream(*_args: Any, **_kwargs: Any) -> Any:
        async def _late_call() -> None:
            await resume_during_close.wait()
            result["status"] = request_session_finish()

        asyncio.ensure_future(_late_call())
        return outcome

    server._execute_via_agent_stream = _fake_stream  # type: ignore[assignment]
    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]
    server._release_finished_session = AsyncMock()  # type: ignore[assignment]

    original_close = SecureToolWrapper.close

    async def _close_and_let_late_task_run(self: SecureToolWrapper) -> None:
        # Mirrors the real race: flip the event right as close() starts,
        # then yield once so the late task gets its turn before close()
        # (and this whole request) finishes.
        resume_during_close.set()
        await asyncio.sleep(0)
        await original_close(self)

    monkeypatch.setattr(SecureToolWrapper, "close", _close_and_let_late_task_run)

    client = _make_client(server)
    response = _post_execute(client)

    assert response.status_code == 200
    assert result.get("status") is SessionFinishStatus.UNAVAILABLE, (
        "a task resuming while wrapper.close() is awaited must see the latch "
        "already closed, not get REQUESTED for a request that already missed "
        "the drain"
    )


# ---------------------------------------------------------------------------
# 3. Suspended run -> no finish call
# ---------------------------------------------------------------------------


def test_suspended_run_means_no_finish_call() -> None:
    server = _make_server()
    outcome = InterruptResult(
        suspend_payload={"suspend_reason": "hitl", "suspend_context": {}},
        messages=[],
    )
    _wire_execute_path(server, execution_outcome=outcome, call_finish=True)
    server._release_finished_session = AsyncMock()  # type: ignore[assignment]

    client = _make_client(server)
    response = _post_execute(client)

    assert response.status_code == 200
    assert response.json()["status"] == "suspended"
    server._release_finished_session.assert_not_called()
    assert server._pending_session_finish == set()


# ---------------------------------------------------------------------------
# 4. Failing/errored run -> no finish call
# ---------------------------------------------------------------------------


def test_failing_run_means_no_finish_call() -> None:
    server = _make_server()
    _wire_execute_path(server, stream_raises=RuntimeError("boom"), call_finish=True)
    server._release_finished_session = AsyncMock()  # type: ignore[assignment]

    client = _make_client(server)
    response = _post_execute(client)

    assert response.status_code == 500
    server._release_finished_session.assert_not_called()
    assert server._pending_session_finish == set()


# ---------------------------------------------------------------------------
# 5. Pending memory writes drain before the finish POST fires
# ---------------------------------------------------------------------------


class _StubWriter:
    """Reports a scripted sequence of pending-write counts, one per poll."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = counts
        self._idx = 0

    @property
    def pending_writes(self) -> int:
        value = self._counts[min(self._idx, len(self._counts) - 1)]
        self._idx += 1
        return value


async def test_memory_writes_drain_before_finish_post() -> None:
    server = _make_server()
    writer = _StubWriter([2, 1, 0])
    server.runtime._memory_writer = writer

    finish_calls: list[str] = []

    async def _fake_post_json_with_retries(client: Any, url: str, payload: dict) -> None:
        finish_calls.append(url)

    with (
        patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())),
        patch(
            "agent_engine_runner_shared.server.aer.post_json_with_retries",
            new=_fake_post_json_with_retries,
        ),
    ):
        await server._release_finished_session(_execute_request("exec-drain"))

    assert writer._idx >= 3  # polled until it saw 0
    assert finish_calls == ["http://oe/executions/exec-drain/finish"]


async def test_terminal_callback_settles_before_finish_post() -> None:
    server = _make_server()
    callback_key = _CallbackKey("exec-callback", "terminal")
    server._callback_delivery._pending[callback_key] = _PendingCallback(  # type: ignore[attr-defined]
        oe_url="http://oe",
        body={"execution_id": "exec-callback", "status": "COMPLETED"},
    )
    release_callback = asyncio.Event()

    async def _settle_callback() -> None:
        await release_callback.wait()
        server._callback_delivery._pending.pop(callback_key)  # type: ignore[attr-defined]

    callback_task = asyncio.create_task(_settle_callback())
    server._callback_delivery._tasks[callback_key] = callback_task  # type: ignore[attr-defined]
    finish_calls: list[str] = []

    async def _fake_post_json_with_retries(client: Any, url: str, payload: dict) -> None:
        finish_calls.append(url)

    with (
        patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())),
        patch(
            "agent_engine_runner_shared.server.aer.post_json_with_retries",
            new=_fake_post_json_with_retries,
        ),
    ):
        finish_task = asyncio.create_task(
            server._release_finished_session(_execute_request("exec-callback"))
        )
        await asyncio.sleep(0)
        assert finish_calls == []

        release_callback.set()
        await finish_task

    assert finish_calls == ["http://oe/executions/exec-callback/finish"]


async def test_memory_writes_stuck_above_zero_still_fires_after_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server()

    class _StuckWriter:
        @property
        def pending_writes(self) -> int:
            return 1

    server.runtime._memory_writer = _StuckWriter()

    # Collapse the drain deadline so the test doesn't actually wait 10s.
    monkeypatch.setattr(
        "agent_engine_runner_shared.server.aer.SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_S", 0.05
    )

    finish_calls: list[str] = []

    async def _fake_post_json_with_retries(client: Any, url: str, payload: dict) -> None:
        finish_calls.append(url)

    with (
        patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())),
        patch(
            "agent_engine_runner_shared.server.aer.post_json_with_retries",
            new=_fake_post_json_with_retries,
        ),
    ):
        await asyncio.wait_for(
            server._release_finished_session(_execute_request("exec-stuck")),
            timeout=2.0,
        )

    # No hang: the finish POST still fired despite pending_writes never
    # reaching zero.
    assert finish_calls == ["http://oe/executions/exec-stuck/finish"]


# ---------------------------------------------------------------------------
# 6. A finish POST that errors is swallowed
# ---------------------------------------------------------------------------


async def test_finish_post_error_is_swallowed() -> None:
    """A finish POST that errors is swallowed - never propagates to the caller.

    ``_release_finished_session`` runs as a FastAPI background task, i.e.
    strictly after the /execute response has already been sent, so there is
    no caller left to propagate to even in principle; this asserts the
    method's internal try/except is what makes that safe.
    """
    server = _make_server()

    async def _raising_post_json_with_retries(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("oe unreachable")

    with (
        patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())),
        patch(
            "agent_engine_runner_shared.server.aer.post_json_with_retries",
            new=_raising_post_json_with_retries,
        ),
    ):
        # Must not raise.
        await server._release_finished_session(_execute_request("exec-err"))


class _AsyncPostStream:
    def __init__(self, post: Any, url: str, payload: dict[str, Any]) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    async def __aenter__(self) -> Any:
        return await self._post(self._url, json=self._payload)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False
