"""Tests for the session finish request primitive.

Tests verify the unavailable/requested/already_requested lifecycle, and that
a mutation of the holder from a copied context (as LangGraph uses for nodes
and worker threads) is visible to the parent frame that reads it later.
"""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from agent_engine_runner_shared.context import (
    SessionFinishStatus,
    clear_execution_context,
    close_session_finish_latch,
    is_session_finish_requested,
    request_session_finish,
    set_execution_context,
)

_EXEC_ID = "test-exec-001"
_WRAPPER = MagicMock()
_OE_URL = "http://localhost:8000"


def test_request_session_finish_outside_execution_is_unavailable():
    assert request_session_finish() is SessionFinishStatus.UNAVAILABLE
    assert is_session_finish_requested() is False


def test_request_session_finish_is_idempotent_within_an_execution():
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        assert request_session_finish() is SessionFinishStatus.REQUESTED
        assert request_session_finish() is SessionFinishStatus.ALREADY_REQUESTED
        assert is_session_finish_requested() is True
    finally:
        clear_execution_context(tokens)


def test_request_from_a_child_context_is_visible_to_the_parent():
    """LangGraph runs nodes in copied contexts and worker threads; a
    ContextVar.set there would be invisible here, a holder mutation is not."""
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            ctx = contextvars.copy_context()
            assert (
                pool.submit(ctx.run, request_session_finish).result()
                is SessionFinishStatus.REQUESTED
            )
        assert is_session_finish_requested() is True
    finally:
        clear_execution_context(tokens)


def test_request_after_the_turn_closed_the_latch_is_unavailable():
    """Post-turn work (asyncio.create_task, a thread joined after the turn)
    that calls request_session_finish() from a copied context must not get
    REQUESTED once the AER's `finally` has closed the latch."""
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        ctx = contextvars.copy_context()
        close_session_finish_latch()
        assert ctx.run(request_session_finish) is SessionFinishStatus.UNAVAILABLE
    finally:
        clear_execution_context(tokens)


def test_request_from_a_thread_joined_after_the_turn_closed_is_unavailable():
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(request_session_finish,))
        thread.start()
        thread.join()
        close_session_finish_latch()
        assert ctx.run(request_session_finish) is SessionFinishStatus.UNAVAILABLE
    finally:
        clear_execution_context(tokens)


def test_a_run_that_raised_still_closes_the_latch():
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            pass
        finally:
            close_session_finish_latch()
        assert request_session_finish() is SessionFinishStatus.UNAVAILABLE
    finally:
        clear_execution_context(tokens)


def test_concurrent_requests_on_one_holder_yield_exactly_one_requested():
    """N threads racing request_session_finish() on the same holder must
    serialize on the lock: exactly one REQUESTED, the rest ALREADY_REQUESTED."""
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        ctx = contextvars.copy_context()
        barrier = threading.Barrier(16)
        results = [None] * 16

        def call(i):
            barrier.wait()
            results[i] = ctx.run(request_session_finish)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(SessionFinishStatus.REQUESTED) == 1
        assert results.count(SessionFinishStatus.ALREADY_REQUESTED) == 15
    finally:
        clear_execution_context(tokens)


def test_concurrent_request_and_close_never_observe_an_inconsistent_pair():
    """A thread calling request_session_finish() racing the AER's close()
    must never see REQUESTED after closed was already set."""
    tokens = set_execution_context(
        execution_id=_EXEC_ID, wrapper=_WRAPPER, oe_url=_OE_URL, session_id="sess-1"
    )
    try:
        ctx = contextvars.copy_context()
        barrier = threading.Barrier(2)
        result = [None]

        def requester():
            barrier.wait()
            result[0] = ctx.run(request_session_finish)

        def closer():
            barrier.wait()
            ctx.run(close_session_finish_latch)

        t1 = threading.Thread(target=requester)
        t2 = threading.Thread(target=closer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Whichever won the race, a subsequent call must be UNAVAILABLE -
        # closed is terminal, so a REQUESTED result never resurrects.
        assert result[0] in (
            SessionFinishStatus.REQUESTED,
            SessionFinishStatus.UNAVAILABLE,
        )
        assert ctx.run(request_session_finish) is SessionFinishStatus.UNAVAILABLE
    finally:
        clear_execution_context(tokens)
