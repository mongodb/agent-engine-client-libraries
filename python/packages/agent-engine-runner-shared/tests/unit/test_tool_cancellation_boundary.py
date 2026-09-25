"""What cancelling a tool call does and does not reach.

These are characterization tests, not aspirations. Stopping a run aborts the
platform's own outbound request to the tool pod; it does not reach work the
handler started. Each test below pins the boundary as it actually is, so the
gap stays visible in the suite instead of being rediscovered from a customer
report, and so a future cooperative-cancellation change has something concrete
to flip.

The dispatch under test is the tool server's own: these drive
``ToolServer._handle_execute``, so the offload decision that matters — an async
handler awaited directly, a sync handler handed to ``asyncio.to_thread`` — is
the production one rather than a restatement of it in the test. If that dispatch
ever shields the handler or threads a cancellation signal into it, these tests
change behaviour, which is the point.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from agent_engine_runner_shared.models import ToolPodExecuteRequest
from agent_engine_runner_shared.server.tool import ToolServer


def _server(tool) -> ToolServer:
    """A tool server whose single registered tool is `tool`."""
    mock_runtime = Mock()
    mock_runtime._tools = {"tool": tool}
    mock_runtime._tool_definitions = {"tool": {}}
    return ToolServer(mock_runtime)


def _request() -> ToolPodExecuteRequest:
    return ToolPodExecuteRequest(
        execution_id="exec-1",
        tool_name="tool",
        arguments={},
        session_id="session-1",
    )


class TestExecuteDispatchOutlivesCancellation:
    """Cancelling the platform's call does not stop the handler behind it.

    Driven through ``_handle_execute``, so what is asserted is the tool server's
    real behaviour for a registered tool, not a property of ``asyncio``.
    """

    @pytest.mark.anyio
    async def test_sync_tool_body_completes_after_the_execute_is_cancelled(
        self,
    ) -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finished = threading_event()

        def blocking_tool() -> str:
            # asyncio.Event is not thread-safe and this runs on the worker
            # thread the server off-loaded to, so marshal the set onto the loop.
            loop.call_soon_threadsafe(started.set)
            # Stands in for external work: a job submission, a payment, a write.
            finished.wait_for_release()
            return "external work completed"

        task = asyncio.create_task(_server(blocking_tool)._handle_execute(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The platform has stopped waiting. The body has not stopped running.
        assert not finished.was_released, "sanity: the body is still mid-flight"
        finished.release()
        assert finished.wait_until_done(timeout=2), (
            "the blocking tool body runs to completion after the execute was "
            "cancelled — the server off-loads it to a thread, and cancelling "
            "abandons the wait rather than the thread"
        )

    @pytest.mark.anyio
    async def test_async_tool_work_started_before_the_await_is_not_undone(
        self,
    ) -> None:
        launched: list[str] = []
        started = asyncio.Event()

        async def async_tool() -> str:
            launched.append("job-1 submitted")
            started.set()
            await asyncio.sleep(3600)  # awaiting the external job
            return "job-1 result"

        task = asyncio.create_task(_server(async_tool)._handle_execute(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert launched == ["job-1 submitted"], (
            "the external job was already submitted; cancelling the execute does "
            "not cancel it, and nothing in the platform will"
        )

    @pytest.mark.anyio
    async def test_a_cooperating_async_tool_can_clean_up(self) -> None:
        """The seam a cooperative-cancellation change would build on.

        An async tool that catches CancelledError today can cancel its own
        external job, because the server awaits it directly. Nothing asks it to
        and nothing bounds how long it takes — a starting point, not a rewrite.
        """
        cleaned: list[str] = []
        started = asyncio.Event()

        async def cooperative_tool() -> str:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cleaned.append("job-1 cancelled with the provider")
                raise
            return "unreachable"

        task = asyncio.create_task(_server(cooperative_tool)._handle_execute(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert cleaned == ["job-1 cancelled with the provider"], (
            "an async tool does receive the cancellation through the server's "
            "await; the platform neither requires this nor bounds it"
        )

    @pytest.mark.anyio
    async def test_a_sync_tool_is_given_no_way_to_cooperate(self) -> None:
        """The asymmetry, stated outright.

        The same cooperative intent is unavailable to a sync tool: the server
        hands it to a thread, so no exception is ever raised inside it and no
        signal is passed to it. This is the gap cooperative cancellation has to
        close, and it is why the async case above is not sufficient on its own.
        """
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        gate = threading_event()
        cancelled_inside: list[str] = []

        def blocking_tool() -> str:
            loop.call_soon_threadsafe(started.set)
            try:
                gate.wait_for_release()
            except BaseException as exc:  # pragma: no cover - documents absence
                cancelled_inside.append(type(exc).__name__)
                raise
            return "done"

        task = asyncio.create_task(_server(blocking_tool)._handle_execute(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.release()
        assert gate.wait_until_done(timeout=2)

        assert cancelled_inside == [], (
            "nothing is raised inside a sync tool body when the execute is "
            "cancelled, so it has no opportunity to clean up"
        )


def threading_event() -> _ReleasableEvent:
    """A two-phase event so a test can prove a thread body is still running."""
    return _ReleasableEvent()


class _ReleasableEvent:
    """Lets a test hold a thread body open, release it, and confirm it finished.

    A bare ``threading.Event`` cannot distinguish "not yet released" from
    "released and finished", which is exactly the distinction these tests turn
    on.
    """

    def __init__(self) -> None:
        import threading

        self._release = threading.Event()
        self._done = threading.Event()
        self.was_released = False

    def wait_for_release(self) -> None:
        self._release.wait(timeout=5)
        self._done.set()

    def release(self) -> None:
        self.was_released = True
        self._release.set()

    def wait_until_done(self, timeout: float) -> bool:
        return self._done.wait(timeout=timeout)
