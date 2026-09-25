"""Tests for AER chunk emission (quick-260421-fj2).

Focus areas:

1. ``metadata["source"]`` is forwarded on every ``chunk_type="text"``
   call to OE's /stream/chunk endpoint.
2. The thinking-token filter keeps separate state per source so the
   root agent's mid-``<think>`` buffer can't swallow a subagent's first
   token.
3. ``subagent_start`` / ``subagent_end`` are emitted as distinct
   ``chunk_type`` values carrying ``subagent_name`` and ``tool_call_id``.
4. ``subagent_start`` / ``subagent_end`` are always emitted unconditionally.
5. The AER drains one post-terminal SDK stream iteration so cleanup after
   ``result`` / ``suspend`` can run before the request returns.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext, StreamEvent

from agent_engine_runner_shared.custom_events import (
    clear_custom_event_transport,
    get_custom_event_transport,
    set_custom_event_transport,
)
from agent_engine_runner_shared.server.aer import AERServer

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal BaseAgent-shaped stand-in whose ``execute(ctx, input)``
    yields a scripted sequence of StreamEvents."""

    def __init__(self, events: list[StreamEvent]):
        self._events = events

    def execute(self, _ctx: RequestContext, _input: AgentInput) -> "_FakeAgent":
        # ``execute`` is used in ``async for`` so expose __aiter__ on self.
        return self

    def __aiter__(self) -> "_FakeAgent":
        self._i = 0
        return self

    async def __anext__(self) -> StreamEvent:
        if self._i >= len(self._events):
            raise StopAsyncIteration
        evt = self._events[self._i]
        self._i += 1
        return evt


class _ResultThenPostYieldCleanupAgent:
    """Agent whose cleanup runs only when the consumer asks for the next item.

    This mirrors SDK async generators that yield the terminal ``result`` before
    running stream/checkpointer cleanup. Returning immediately from the AER's
    stream loop skips this post-yield continuation.
    """

    def __init__(self) -> None:
        self.post_yield_cleanup_ran = False

    def execute(self, _ctx: RequestContext, _input: AgentInput):
        return self._events()

    async def _events(self):
        yield StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        )
        self.post_yield_cleanup_ran = True


class _ResultThenRepeatingPostTerminalAgent:
    """Agent that keeps producing events after the terminal result."""

    def __init__(self) -> None:
        self.post_terminal_events = 0
        self._step = 0

    def execute(
        self, _ctx: RequestContext, _input: AgentInput
    ) -> "_ResultThenRepeatingPostTerminalAgent":
        return self

    def __aiter__(self) -> "_ResultThenRepeatingPostTerminalAgent":
        return self

    async def __anext__(self) -> StreamEvent:
        if self._step == 0:
            self._step += 1
            return StreamEvent(
                event="result",
                data={"response": "done", "messages": [], "resumed": False},
            )
        await asyncio.sleep(0.001)
        self.post_terminal_events += 1
        return StreamEvent(event="token", data={"content": "late", "source": ""})


class _ResultThenHangingPostTerminalAgent:
    """Agent whose post-terminal next item never arrives until closed."""

    def __init__(self) -> None:
        self.closed = False
        self._step = 0
        self._released = asyncio.Event()

    def execute(
        self, _ctx: RequestContext, _input: AgentInput
    ) -> "_ResultThenHangingPostTerminalAgent":
        return self

    def __aiter__(self) -> "_ResultThenHangingPostTerminalAgent":
        return self

    async def __anext__(self) -> StreamEvent:
        if self._step == 0:
            self._step += 1
            return StreamEvent(
                event="result",
                data={"response": "done", "messages": [], "resumed": False},
            )
        await self._released.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        self._released.set()


class _ResultThenDrainErrorAgent:
    """Agent whose cleanup raises after yielding a terminal result."""

    def __init__(self) -> None:
        self._step = 0

    def execute(self, _ctx: RequestContext, _input: AgentInput) -> "_ResultThenDrainErrorAgent":
        return self

    def __aiter__(self) -> "_ResultThenDrainErrorAgent":
        return self

    async def __anext__(self) -> StreamEvent:
        if self._step == 0:
            self._step += 1
            return StreamEvent(
                event="result",
                data={"response": "done", "messages": [], "resumed": False},
            )
        raise RuntimeError("cleanup failed")


class _ResultThenMalformedPostTerminalAgent:
    """Agent whose post-terminal drain yields a malformed event."""

    def __init__(self) -> None:
        self._step = 0

    def execute(
        self, _ctx: RequestContext, _input: AgentInput
    ) -> "_ResultThenMalformedPostTerminalAgent":
        return self

    def __aiter__(self) -> "_ResultThenMalformedPostTerminalAgent":
        return self

    async def __anext__(self) -> object:
        if self._step == 0:
            self._step += 1
            return StreamEvent(
                event="result",
                data={"response": "done", "messages": [], "resumed": False},
            )
        return object()


class _MalformedEventAgent:
    """Agent that raises in AER stream processing before a terminal event."""

    def __init__(self) -> None:
        self.closed = False
        self._step = 0

    def execute(self, _ctx: RequestContext, _input: AgentInput) -> "_MalformedEventAgent":
        return self

    def __aiter__(self) -> "_MalformedEventAgent":
        return self

    async def __anext__(self) -> StreamEvent:
        if self._step == 0:
            self._step += 1
            return StreamEvent(event="token", data="not a dict")
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


def _agent_input() -> AgentInput:
    return AgentInput(
        payload={
            "message": "hi",
            "thread_id": "t-1",
            "session_id": "t-1",
            "resume": False,
        }
    )


def _ctx(execution_id: str) -> RequestContext:
    return RequestContext(
        execution_id=execution_id,
        user_id="test-user",
        request_headers={},
    )


def _make_server() -> tuple[AERServer, AsyncMock]:
    """Construct an AERServer wired with mocks sufficient to exercise
    ``_execute_via_agent_stream`` without touching FastAPI or HTTPX.

    Returns the server and the ``_send_stream_chunk`` mock that the
    test inspects.
    """
    runtime = MagicMock()
    # Avoid on_startup / super().__init__ doing real work:
    server = AERServer.__new__(AERServer)
    server.runtime = runtime  # type: ignore[attr-defined]
    server._client = None  # type: ignore[attr-defined]
    import asyncio

    server._client_lock = asyncio.Lock()  # type: ignore[attr-defined]

    send_mock = AsyncMock()
    server._send_stream_chunk = send_mock  # type: ignore[assignment]
    return server, send_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_text_chunks_carry_metadata_source_attribution() -> None:
    """Every text chunk forwarded to OE should include
    ``metadata["source"]`` — empty for root tokens, subagent name for
    sourced tokens."""
    events = [
        StreamEvent(event="token", data={"content": "root says ", "source": ""}),
        StreamEvent(event="token", data={"content": "hi ", "source": ""}),
        StreamEvent(event="token", data={"content": "sub says ok", "source": "sec"}),
        StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        ),
    ]
    server, send_mock = _make_server()
    agent = _FakeAgent(events)

    await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-1"), _agent_input(), "http://oe", "exec-1"
    )

    text_calls = [
        call for call in send_mock.await_args_list if call.kwargs.get("chunk_type") == "text"
    ]
    assert len(text_calls) == 3
    # Each text call must carry a ``metadata.source`` string.
    sources = [call.kwargs.get("metadata", {}).get("source") for call in text_calls]
    assert sources == ["", "", "sec"]
    contents = [call.kwargs.get("content") for call in text_calls]
    assert contents == ["root says ", "hi ", "sub says ok"]


async def test_terminal_result_drains_one_more_event_for_post_yield_cleanup() -> None:
    """A terminal result should not make AER return from the stream loop
    immediately.

    The old implementation returned as soon as it saw ``event="result"``. That
    left SDK cleanup after the terminal yield to forced generator close rather
    than normal iteration, which is the shape that showed up as the next invoke
    hanging after a successful stream.
    """
    server, _ = _make_server()
    agent = _ResultThenPostYieldCleanupAgent()

    result = await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-cleanup"), _agent_input(), "http://oe", "exec-cleanup"
    )

    assert result.content == "done"
    assert agent.post_yield_cleanup_ran is True


async def test_terminal_result_drains_contextvar_cleanup_in_same_task() -> None:
    """Post-terminal drain must finalize ContextVar tokens in the request task.

    ``asyncio.wait_for`` runs ``__anext__`` in a child Task whose Context is a
    copy. Resetting a Token created in the parent then raises ValueError and
    leaves the transport installed. ``asyncio.timeout`` keeps finalization on
    the same task.
    """

    class _ResultWithTransportCleanupAgent:
        def __init__(self) -> None:
            self.cleanup_error: BaseException | None = None

        def execute(self, _ctx: RequestContext, _input: AgentInput):
            return self._events()

        async def _events(self):
            token = set_custom_event_transport(MagicMock())
            try:
                yield StreamEvent(
                    event="result",
                    data={"response": "done", "messages": [], "resumed": False},
                )
            finally:
                try:
                    clear_custom_event_transport(token)
                except BaseException as exc:  # noqa: BLE001 — capture for assertion
                    self.cleanup_error = exc

    server, _ = _make_server()
    agent = _ResultWithTransportCleanupAgent()

    result = await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent,
        _ctx("exec-ctxvar-drain"),
        _agent_input(),
        "http://oe",
        "exec-ctxvar-drain",
    )

    assert result.content == "done"
    assert agent.cleanup_error is None
    assert get_custom_event_transport() is None


async def test_terminal_result_ignores_only_one_post_terminal_event() -> None:
    """AER must not keep draining indefinitely after a terminal outcome."""
    server, _ = _make_server()
    agent = _ResultThenRepeatingPostTerminalAgent()

    result = await asyncio.wait_for(
        server._execute_via_agent_stream(  # type: ignore[attr-defined]
            agent, _ctx("exec-repeat"), _agent_input(), "http://oe", "exec-repeat"
        ),
        timeout=0.2,
    )

    assert result.content == "done"
    assert agent.post_terminal_events == 1


async def test_terminal_result_timeout_closes_post_terminal_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out post-terminal drain must close the underlying iterator."""
    monkeypatch.setattr(
        "agent_engine_runner_shared.server.aer.AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT_S",
        0.01,
    )
    server, _ = _make_server()
    agent = _ResultThenHangingPostTerminalAgent()

    result = await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent,
        _ctx("exec-timeout-drain"),
        _agent_input(),
        "http://oe",
        "exec-timeout-drain",
    )

    assert result.content == "done"
    assert agent.closed is True


async def test_post_terminal_drain_exception_returns_terminal_result() -> None:
    """Cleanup errors after a terminal result should not overwrite success."""
    server, _ = _make_server()
    agent = _ResultThenDrainErrorAgent()

    result = await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-drain-error"), _agent_input(), "http://oe", "exec-drain-error"
    )

    assert result.content == "done"


async def test_post_terminal_malformed_event_returns_terminal_result() -> None:
    """Malformed post-terminal events should not overwrite success."""
    server, _ = _make_server()
    agent = _ResultThenMalformedPostTerminalAgent()

    result = await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent,
        _ctx("exec-drain-malformed"),
        _agent_input(),
        "http://oe",
        "exec-drain-malformed",
    )

    assert result.content == "done"


async def test_pre_terminal_exception_closes_stream_iterator() -> None:
    """Manual iterator consumption must still close the stream on exceptions."""
    server, _ = _make_server()
    agent = _MalformedEventAgent()

    with pytest.raises(TypeError, match="Expected dict event data"):
        await server._execute_via_agent_stream(  # type: ignore[attr-defined]
            agent, _ctx("exec-malformed"), _agent_input(), "http://oe", "exec-malformed"
        )

    assert agent.closed is True


async def test_per_source_thinking_filter_does_not_leak_state() -> None:
    """A mid-``<think>`` root must not eat a subagent's first token.

    Scripted scenario:
      - Root opens a ``<think>`` block (no close) → should buffer in root's
        own state slot and emit nothing.
      - Subagent streams ``<think>a</think>foo`` → filtered to ``foo`` in
        the subagent's own state slot (not the root's mid-think buffer).
      - Subagent streams ``bar`` → passes through cleanly.
      - Root emits ``</think>after`` → closes its think block and emits
        ``after``.
    """
    events = [
        StreamEvent(event="token", data={"content": "<think>internal", "source": ""}),
        StreamEvent(event="token", data={"content": "<think>a</think>foo", "source": "sub"}),
        StreamEvent(event="token", data={"content": "bar", "source": "sub"}),
        StreamEvent(event="token", data={"content": "</think>after", "source": ""}),
        StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        ),
    ]
    server, send_mock = _make_server()
    agent = _FakeAgent(events)

    await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-2"), _agent_input(), "http://oe", "exec-2"
    )

    text_calls = [
        call for call in send_mock.await_args_list if call.kwargs.get("chunk_type") == "text"
    ]
    pairs: list[tuple[str, str]] = [
        (
            str(call.kwargs.get("metadata", {}).get("source", "")),
            str(call.kwargs.get("content", "")),
        )
        for call in text_calls
    ]

    # The subagent's ``foo`` must survive even though the root is still
    # inside a ``<think>`` block, and the root's later ``after`` must
    # arrive cleanly.
    sub_texts = [content for source, content in pairs if source == "sub"]
    root_texts = [content for source, content in pairs if source == ""]

    sub_joined = "".join(sub_texts)
    root_joined = "".join(root_texts)
    # The subagent's post-``</think>`` text must make it through.
    assert "foo" in sub_joined
    assert "bar" in sub_joined
    # The subagent's own ``<think>a</think>`` reasoning must have been
    # stripped — neither the tag nor the reasoning token ``a`` (as a
    # standalone slice between tags) should appear. We check that the
    # total subagent content equals only the non-thinking portions.
    assert sub_joined == "foobar"
    assert "<think>" not in sub_joined

    assert "after" in root_joined
    assert "<think>" not in root_joined


async def test_subagent_start_and_end_chunks_are_emitted_with_metadata() -> None:
    events = [
        StreamEvent(
            event="subagent_start",
            data={
                "source": "sec",
                "subagent_name": "sec",
                "tool_call_id": "call-1",
                "description": "Review security.",
            },
        ),
        StreamEvent(event="token", data={"content": "ok", "source": "sec"}),
        StreamEvent(
            event="subagent_end",
            data={
                "source": "sec",
                "subagent_name": "sec",
                "tool_call_id": "call-1",
            },
        ),
        StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        ),
    ]
    server, send_mock = _make_server()
    agent = _FakeAgent(events)

    await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-3"), _agent_input(), "http://oe", "exec-3"
    )

    chunk_types = [call.kwargs.get("chunk_type") for call in send_mock.await_args_list]
    assert chunk_types.count("subagent_start") == 1
    assert chunk_types.count("subagent_end") == 1
    start_call = next(
        call
        for call in send_mock.await_args_list
        if call.kwargs.get("chunk_type") == "subagent_start"
    )
    # Structural chunks carry no text content — description lives in metadata
    # so CLI/UI accumulators that append chunk.content don't pick it up.
    assert start_call.kwargs.get("content") == ""
    meta = start_call.kwargs.get("metadata")
    assert isinstance(meta, dict)
    assert meta["description"] == "Review security."
    assert meta["subagent_name"] == "sec"
    assert meta["tool_call_id"] == "call-1"
    assert meta["source"] == "sec"

    end_call = next(
        call
        for call in send_mock.await_args_list
        if call.kwargs.get("chunk_type") == "subagent_end"
    )
    assert end_call.kwargs.get("content") == ""
    end_meta = end_call.kwargs.get("metadata")
    assert isinstance(end_meta, dict)
    assert end_meta["subagent_name"] == "sec"
    assert end_meta["tool_call_id"] == "call-1"


async def test_text_chunks_carry_tool_call_id_metadata() -> None:
    """``tool_call_id`` from the SDK token event must reach OE so the UI
    can route per-tc_id (and disambiguate parallel same-name dispatches).
    """
    events = [
        StreamEvent(
            event="token",
            data={"content": "root", "source": "", "tool_call_id": ""},
        ),
        StreamEvent(
            event="token",
            data={
                "content": "sub-a",
                "source": "research",
                "tool_call_id": "call-a",
            },
        ),
        StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        ),
    ]
    server, send_mock = _make_server()
    agent = _FakeAgent(events)

    await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-tcid"), _agent_input(), "http://oe", "exec-tcid"
    )

    text_calls = [
        call for call in send_mock.await_args_list if call.kwargs.get("chunk_type") == "text"
    ]
    assert [call.kwargs["metadata"]["tool_call_id"] for call in text_calls] == [
        "",
        "call-a",
    ]


async def test_subagent_boundary_handles_missing_keys() -> None:
    """Missing/None values in the SDK event_data must coerce to empty strings
    in the wire chunk metadata — never to literal ``"None"``."""
    events = [
        StreamEvent(
            event="subagent_start",
            data={
                # source / subagent_name / tool_call_id / description all absent
            },
        ),
        StreamEvent(
            event="subagent_end",
            data={
                "subagent_name": "x",
                # tool_call_id None, summary None
                "tool_call_id": None,
                "summary": None,
            },
        ),
        StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        ),
    ]
    server, send_mock = _make_server()
    agent = _FakeAgent(events)

    await server._execute_via_agent_stream(  # type: ignore[attr-defined]
        agent, _ctx("exec-none"), _agent_input(), "http://oe", "exec-none"
    )

    boundary_calls = [
        call
        for call in send_mock.await_args_list
        if call.kwargs.get("chunk_type") in ("subagent_start", "subagent_end")
    ]
    assert len(boundary_calls) == 2
    for call in boundary_calls:
        meta = call.kwargs.get("metadata") or {}
        for key in ("source", "subagent_name", "tool_call_id"):
            assert isinstance(meta.get(key), str)
            assert meta.get(key) != "None", f"None-coercion regression on {key}: {meta}"
        # Structural chunks always emit empty content.
        assert call.kwargs.get("content") == ""
        # The content_key value (description/summary) must also coerce — never "None".
        for ck in ("description", "summary"):
            if ck in meta:
                assert meta[ck] != "None", f"None-coercion regression on {ck}: {meta}"
