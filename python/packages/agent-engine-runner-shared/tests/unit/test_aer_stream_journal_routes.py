"""Tests for the runner's stream-journal read routes.

The runner owns the journal; these routes serve it back to whichever engine
replica resumes the turn. The cases pin the wire contract a resume depends on:

  - the index lists every generation this runner opened, and 404s for an
    execution it never saw;
  - a generation streams as SSE: ``chunk`` frames after the cursor, then
    exactly one ``end`` (with the close reason) or ``incomplete``;
  - an unknown or evicted generation is a 404, not an empty stream;
  - the reader cap is a 429;
  - the routes sit behind the same auth middleware as every peer route, which
    refuses a tokenless caller only when a token is configured.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server.aer import (
    AERServer,
    _JournalStreamingResponse,
    _stream_journal_sse,
)
from agent_engine_runner_shared.server.stream_journal import (
    REASON_COMPLETED,
    StreamJournalRegistry,
)


def _server(tmp_path: Path, **registry_overrides: Any) -> AERServer:
    server = AERServer(MagicMock())
    server._stream_journals = StreamJournalRegistry(root=str(tmp_path), **registry_overrides)
    return server


def _seed(
    server: AERServer,
    *,
    execution_id: str = "exec-1",
    generation: int = 1,
    contents: tuple[str, ...] = ("a", "b"),
    reason: str = REASON_COMPLETED,
) -> None:
    """Write and close one journal, then let the writer settle.

    Run through ``asyncio.run``: a closed journal is fully on disk, so the
    request-time read needs no live task from the seeding loop.
    """

    async def _write() -> None:
        journal = await server.stream_journals.open_journal(
            execution_id=execution_id, session_id="s-1", generation=generation
        )
        for seq, content in enumerate(contents, start=1):
            journal.record(
                {
                    "execution_id": execution_id,
                    "chunk_type": "text",
                    "content": content,
                    "seq": seq,
                }
            )
        await journal.close(reason)

    asyncio.run(_write())


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def _wait_until(predicate: Any, *, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01) + 1):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


def test_index_lists_each_generation_with_its_close_state(tmp_path: Path) -> None:
    server = _server(tmp_path)
    _seed(server, generation=1, reason=REASON_COMPLETED)
    _seed(server, generation=2, contents=("c",), reason=REASON_COMPLETED)

    response = TestClient(server.create_app()).get("/stream/journal/exec-1")

    assert response.status_code == 200
    assert response.json() == {
        "execution_id": "exec-1",
        "generations": [
            {"generation": 1, "last_seq": 2, "reason": "completed", "retained": True},
            {"generation": 2, "last_seq": 1, "reason": "completed", "retained": True},
        ],
    }


def test_index_reports_an_open_generation_with_a_null_reason(tmp_path: Path) -> None:
    # A live generation has no close reason yet. Both runtimes must report the
    # same value for that state (null), so a consumer never decodes two shapes.
    server = _server(tmp_path)
    client = TestClient(server.create_app())

    async def _open() -> None:
        journal = await server.stream_journals.open_journal(
            execution_id="exec-open", session_id="s-1", generation=1
        )
        journal.record(
            {"execution_id": "exec-open", "chunk_type": "text", "content": "a", "seq": 1}
        )
        await _wait_until(lambda: journal.last_seq == 1)

    asyncio.run(_open())

    response = client.get("/stream/journal/exec-open")
    assert response.json()["generations"] == [
        {"generation": 1, "last_seq": 1, "reason": None, "retained": True}
    ]


def test_index_is_404_for_an_execution_the_runner_never_opened(tmp_path: Path) -> None:
    server = _server(tmp_path)
    response = TestClient(server.create_app()).get("/stream/journal/exec-unknown")
    assert response.status_code == 404


def test_generation_streams_chunks_then_one_end_frame(tmp_path: Path) -> None:
    server = _server(tmp_path)
    _seed(server, contents=("a", "b", "c"))

    response = TestClient(server.create_app()).get("/stream/journal/exec-1/1")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert [event for event, _data in events] == ["chunk", "chunk", "chunk", "end"]
    assert [data["chunk"]["content"] for event, data in events if event == "chunk"] == [
        "a",
        "b",
        "c",
    ]
    assert events[-1][1] == {"generation": 1, "last_seq": 3, "reason": "completed"}


def test_after_seq_replays_only_what_the_client_has_not_seen(tmp_path: Path) -> None:
    server = _server(tmp_path)
    _seed(server, contents=("a", "b", "c"))

    response = TestClient(server.create_app()).get("/stream/journal/exec-1/1?after_seq=2")

    events = _sse_events(response.text)
    assert [data["seq"] for event, data in events if event == "chunk"] == [3]
    assert events[-1][0] == "end"


def test_incomplete_journal_streams_an_incomplete_frame_not_an_end(tmp_path: Path) -> None:
    # A per-journal chunk bound hit makes the prefix untrustworthy while the
    # turn keeps running; the follower must be told so, never handed an `end`.
    server = _server(tmp_path, journal_max_chunks=1)
    _seed(server, contents=("a", "b"))

    events = _sse_events(TestClient(server.create_app()).get("/stream/journal/exec-1/1").text)

    assert events[-1][0] == "incomplete"
    assert events[-1][1]["reason"] == "incomplete"


def test_unknown_generation_is_404(tmp_path: Path) -> None:
    server = _server(tmp_path)
    _seed(server, generation=1)
    client = TestClient(server.create_app())

    assert client.get("/stream/journal/exec-1/2").status_code == 404
    assert client.get("/stream/journal/exec-unknown/1").status_code == 404


def test_invalid_path_and_query_parameters_are_422(tmp_path: Path) -> None:
    # The route's declared parameters reject a malformed generation or cursor
    # rather than reinterpreting it as a different replay range.
    server = _server(tmp_path)
    _seed(server)
    client = TestClient(server.create_app())

    assert client.get("/stream/journal/exec-1/not-a-number").status_code == 422
    assert client.get("/stream/journal/exec-1/1?after_seq=-1").status_code == 422
    assert client.get("/stream/journal/exec-1/1?after_seq=abc").status_code == 422
    # A parseable but unknown generation stays a 404.
    assert client.get("/stream/journal/exec-1/-1").status_code == 404


def test_a_finished_follow_releases_its_reader_slot(tmp_path: Path) -> None:
    server = _server(tmp_path, max_readers_per_journal=1)
    _seed(server)
    client = TestClient(server.create_app())

    assert client.get("/stream/journal/exec-1/1").status_code == 200
    # The cap is one: this only succeeds if the finished follow gave its slot up.
    assert client.get("/stream/journal/exec-1/1").status_code == 200


def test_a_response_that_fails_before_its_body_starts_releases_the_slot(
    tmp_path: Path,
) -> None:
    # An ASGI response can fail or be cancelled before the body's first step
    # (a send error on the response start, a replica shutting down). That path
    # never enters the generator, so the response object itself owns the
    # release; otherwise repeated abandoned follows exhaust the cap.
    server = _server(tmp_path, max_readers_per_journal=1)
    _seed(server)
    registry = server.stream_journals
    lease = registry.acquire_reader("exec-1", 1)
    response = _JournalStreamingResponse(
        _stream_journal_sse(registry, "exec-1", 1, 0, lease),
        media_type="text/event-stream",
        on_close=lambda: registry.release_reader(lease),
    )

    async def _send(_message: dict[str, Any]) -> None:
        raise RuntimeError("connection lost")

    async def _receive() -> dict[str, Any]:
        # Park: the loopback disconnect listener must never deliver anything,
        # so the send failure is what ends the response.
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream/journal/exec-1/1",
        "raw_path": b"/stream/journal/exec-1/1",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("runner", 8001),
    }
    with pytest.raises(RuntimeError):
        asyncio.run(response(scope, _receive, _send))

    replacement = registry.acquire_reader("exec-1", 1)
    assert replacement is not None
    registry.release_reader(replacement)


def test_a_follow_closed_mid_stream_releases_its_slot(tmp_path: Path) -> None:
    # A client disconnect cancels the response body, which closes the generator.
    # That path must free the slot rather than strand the cap.
    server = _server(tmp_path, max_readers_per_journal=1)
    _seed(server)
    registry = server.stream_journals

    async def _start_then_disconnect() -> None:
        lease = registry.acquire_reader("exec-1", 1)
        body = _stream_journal_sse(registry, "exec-1", 1, 0, lease)
        await body.__anext__()
        await body.aclose()

    asyncio.run(_start_then_disconnect())

    lease = registry.acquire_reader("exec-1", 1)
    assert lease is not None
    registry.release_reader(lease)


def test_evicted_generation_is_404(tmp_path: Path) -> None:
    # The index entry survives eviction, but the body it describes is gone, so
    # a follow must fail closed instead of serving an empty stream as complete.
    server = _server(tmp_path)
    _seed(server)
    journal = server.stream_journals._journals["exec-1"][1]
    asyncio.run(journal.evict())

    client = TestClient(server.create_app())
    assert client.get("/stream/journal/exec-1").json()["generations"][0]["retained"] is False
    assert client.get("/stream/journal/exec-1/1").status_code == 404


def test_reader_limit_is_429_while_a_follow_holds_the_slot(tmp_path: Path) -> None:
    server = _server(tmp_path, max_readers_per_journal=1)
    _seed(server)
    lease = server.stream_journals.acquire_reader("exec-1", 1)
    try:
        response = TestClient(server.create_app()).get("/stream/journal/exec-1/1")
        assert response.status_code == 429
    finally:
        server.stream_journals.release_reader(lease)

    assert TestClient(server.create_app()).get("/stream/journal/exec-1/1").status_code == 200


def test_routes_refuse_a_tokenless_caller_only_when_a_token_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control on a deployed runner is the executor network policy, which
    # admits only the per-project OE; an unset token must not refuse service.
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    server = _server(tmp_path)
    _seed(server)
    client = TestClient(server.create_app())

    assert client.get("/stream/journal/exec-1").status_code == 401
    assert (
        client.get("/stream/journal/exec-1", headers={"Authorization": "Bearer s3cret"}).status_code
        == 200
    )

    monkeypatch.delenv("RUNNER_AUTH_TOKEN")
    # A fresh app on the same server: the registry, and so the journal, is the
    # same one the tokenless app must serve.
    assert TestClient(server.create_app()).get("/stream/journal/exec-1").status_code == 200


def _drive_follow_until_first_frame_then_disconnect(app: Any, path: str) -> list[dict[str, Any]]:
    """Run a real request through the app and disconnect on the first frame.

    The ASGI `receive` parks until the test has seen a body frame and then
    answers `http.disconnect`, which is what a dropped peer looks like to
    Starlette. Returns every message the app sent.
    """

    async def _drive() -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []
        first_frame = asyncio.Event()
        disconnected = asyncio.Event()

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            if message.get("body"):
                first_frame.set()

        async def receive() -> dict[str, Any]:
            await disconnected.wait()
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("runner", 8001),
        }
        request = asyncio.ensure_future(app(scope, receive, send))
        await asyncio.wait_for(first_frame.wait(), timeout=5)
        disconnected.set()
        await asyncio.wait_for(request, timeout=5)
        return sent

    return asyncio.run(_drive())


def test_a_live_route_follow_that_disconnects_returns_both_reader_budgets(
    tmp_path: Path,
) -> None:
    # The registered route with a live journal: it emits a frame, the peer
    # disconnects, and a replacement follow must still be admitted while that
    # journal is open — for both the per-journal and the per-runner budget.
    server = _server(tmp_path, max_readers_per_journal=1, max_readers_per_runner=1)
    app = server.create_app()

    async def _open() -> None:
        journal = await server.stream_journals.open_journal(
            execution_id="exec-live", session_id="s-1", generation=1
        )
        journal.record(
            {"execution_id": "exec-live", "chunk_type": "text", "content": "a", "seq": 1}
        )
        await _wait_until(lambda: journal.last_seq == 1)

    asyncio.run(_open())
    registry = server.stream_journals
    path = "/stream/journal/exec-live/1"

    for _ in range(2):
        sent = _drive_follow_until_first_frame_then_disconnect(app, path)
        assert any(b"event: chunk" in (message.get("body") or b"") for message in sent)
        assert registry._runner_readers == 0
        assert registry._journal_readers == {}
        # The journal is still open across both follows.
        assert registry.has_generation("exec-live", 1) is True


def test_a_registered_route_that_fails_before_its_body_starts_returns_the_budget(
    tmp_path: Path,
) -> None:
    # A send failure on the response start never enters the body generator. The
    # registered route must still return the capacity, or abandoned follows
    # exhaust it.
    server = _server(tmp_path, max_readers_per_journal=1, max_readers_per_runner=1)
    _seed(server)
    registry = server.stream_journals
    app = server.create_app()

    async def _send(_message: dict[str, Any]) -> None:
        raise RuntimeError("connection lost")

    async def _receive() -> dict[str, Any]:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream/journal/exec-1/1",
        "raw_path": b"/stream/journal/exec-1/1",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("runner", 8001),
    }
    with pytest.raises(RuntimeError):
        asyncio.run(app(scope, _receive, _send))

    assert registry._runner_readers == 0
    assert registry._journal_readers == {}
    lease = registry.acquire_reader("exec-1", 1)
    assert lease is not None
    registry.release_reader(lease)
