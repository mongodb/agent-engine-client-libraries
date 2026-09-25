"""Best-effort graph warm-up lifecycle."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server import graph_warm
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.graph_warm import GraphWarmer


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def _install_fake_clock(monkeypatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(graph_warm, "monotonic", clock.monotonic)
    monkeypatch.setattr(graph_warm, "sleep", clock.sleep)
    return clock


def _server(warm_up_agent) -> AERServer:
    runtime = MagicMock()
    runtime._graph_builder = object()
    runtime.warm_up_agent = warm_up_agent
    server = AERServer(runtime)
    server._ensure_oe_registrations = AsyncMock()  # type: ignore[method-assign]
    return server


@pytest.mark.asyncio
async def test_aer_startup_does_not_start_graph_warm(monkeypatch):
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.setenv("APP_ID", "ws-test")
    warm_up = MagicMock(return_value=True)
    server = _server(warm_up)
    await asyncio.wait_for(server.on_startup(), timeout=0.1)

    warm_up.assert_not_called()
    await server.on_shutdown()


@pytest.mark.asyncio
async def test_failed_warm_up_retries_until_success(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    warm_up = MagicMock(side_effect=[False, False, True])
    warmer = GraphWarmer(warm_up)
    await warmer.run()
    await warmer.run()

    assert warm_up.call_count == 3
    assert clock.sleeps == [5.0, 10.0]


@pytest.mark.asyncio
async def test_failed_warm_up_stops_after_bounded_attempts(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    warm_up = MagicMock(return_value=False)
    warmer = GraphWarmer(warm_up)
    await warmer.run()
    await warmer.run()

    assert warm_up.call_count == 4
    assert clock.sleeps == [5.0, 10.0, 10.0]


@pytest.mark.asyncio
async def test_slow_failures_do_not_shift_retries_past_their_slots(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    attempt_starts: list[float] = []

    def slow_failure() -> bool:
        attempt_starts.append(clock.now)
        clock.now += 10.0
        return False

    warmer = GraphWarmer(slow_failure)
    await warmer.run()

    assert attempt_starts == [0.0, 15.0, 25.0]
    assert clock.sleeps == [5.0]


@pytest.mark.asyncio
async def test_delayed_sleep_does_not_start_warm_up_after_latest_start(monkeypatch):
    clock = _install_fake_clock(monkeypatch)

    async def delayed_sleep(delay: float) -> None:
        clock.sleeps.append(delay)
        clock.now += delay + 25.0

    monkeypatch.setattr(graph_warm, "sleep", delayed_sleep)
    warm_up = MagicMock(return_value=False)

    await GraphWarmer(warm_up).run()

    assert warm_up.call_count == 1
    assert clock.sleeps == [5.0]


@pytest.mark.asyncio
async def test_run_blocks_until_the_warm_up_attempt_finishes():
    started = threading.Event()
    release = threading.Event()

    def slow_warm() -> bool:
        started.set()
        release.wait(timeout=1)
        return True

    warmer = GraphWarmer(slow_warm)
    wait_task = asyncio.create_task(warmer.run())
    assert await asyncio.to_thread(started.wait, 0.5)

    assert not wait_task.done()

    release.set()
    await asyncio.wait_for(wait_task, timeout=0.5)


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_graph_build_thread():
    started = threading.Event()
    release = threading.Event()

    def slow_warm() -> bool:
        started.set()
        release.wait(timeout=1)
        return True

    warm_up = MagicMock(side_effect=slow_warm)
    warmer = GraphWarmer(warm_up)
    first_caller = asyncio.create_task(warmer.run())
    assert await asyncio.to_thread(started.wait, 0.5)

    first_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_caller

    release.set()
    await asyncio.wait_for(warmer.run(), timeout=0.5)
    assert warm_up.call_count == 1


def test_warm_up_endpoint_returns_after_an_unsuccessful_bounded_sequence(monkeypatch):
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.setenv("APP_ID", "ws-test")
    _install_fake_clock(monkeypatch)
    warm_up = MagicMock(return_value=False)
    server = _server(warm_up)

    with TestClient(server.create_app()) as client:
        response = client.post("/warm-up")

    assert response.status_code == 204
    assert warm_up.call_count == 4
