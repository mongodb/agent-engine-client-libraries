"""Tests for the independent attempt-lease heartbeat loop."""

from __future__ import annotations

import threading
import time

import pytest

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    AttemptHeartbeatRequest,
)
from agent_engine_runner_shared.workflow.heartbeat import (
    AttemptHeartbeat,
    attempt_heartbeat_request_from_context,
)


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
        heartbeat_interval_ms=1000,
    )


def test_heartbeat_request_carries_the_attempt_fence() -> None:
    request = attempt_heartbeat_request_from_context(_attempt())

    assert request.workflow_identity.execution_id == "execution-1"
    assert request.attempt_id == "attempt-1"
    assert request.fencing_token == 7
    assert request.owner_id == "aer-1"


def test_heartbeat_request_requires_execution_identity() -> None:
    with pytest.raises(ValueError):
        attempt_heartbeat_request_from_context(
            AttemptContext(attempt_id="attempt-1", fencing_token=1)
        )


def test_initial_heartbeat_failure_fails_closed() -> None:
    def send(request: AttemptHeartbeatRequest) -> None:
        raise ConnectionError("OE unreachable")

    heartbeat = AttemptHeartbeat(attempt=_attempt(), interval_seconds=0.01, send=send)

    with pytest.raises(RuntimeError, match="initial"):
        heartbeat.start()
    assert heartbeat.is_alive is False


def test_heartbeat_keeps_renewing_while_caller_blocks() -> None:
    beats = threading.Event()
    count = 0
    lock = threading.Lock()

    def send(request: AttemptHeartbeatRequest) -> None:
        nonlocal count
        with lock:
            count += 1
            if count >= 4:
                beats.set()

    heartbeat = AttemptHeartbeat(attempt=_attempt(), interval_seconds=0.01, send=send)
    heartbeat.start()
    try:
        # Simulate tenant work blocking this thread longer than the interval.
        assert beats.wait(timeout=5.0)
    finally:
        heartbeat.stop()

    assert heartbeat.is_alive is False
    assert count >= 4


def test_later_heartbeat_failures_are_retried() -> None:
    recovered = threading.Event()
    calls = 0

    def send(request: AttemptHeartbeatRequest) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("transient blip")
        if calls >= 3:
            recovered.set()

    heartbeat = AttemptHeartbeat(attempt=_attempt(), interval_seconds=0.01, send=send)
    heartbeat.start()
    try:
        assert recovered.wait(timeout=5.0)
    finally:
        heartbeat.stop()


def test_permanent_lease_loss_stops_renewals() -> None:
    from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
        WORKFLOW_ERROR_CODE_STALE_FENCE,
    )
    from agent_engine_runner_shared.workflow import WorkflowClientError

    calls = 0

    def send(request: AttemptHeartbeatRequest) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise WorkflowClientError(WORKFLOW_ERROR_CODE_STALE_FENCE, "superseded")

    heartbeat = AttemptHeartbeat(attempt=_attempt(), interval_seconds=0.01, send=send)
    heartbeat.start()
    try:
        deadline = time.monotonic() + 5.0
        while heartbeat.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        heartbeat.stop()

    # The loop halted itself on the coded rejection and exposed the loss.
    assert heartbeat.lease_lost is True
    settled = calls
    time.sleep(0.05)
    assert calls == settled


def test_stop_halts_renewals() -> None:
    count = 0

    def send(request: AttemptHeartbeatRequest) -> None:
        nonlocal count
        count += 1

    heartbeat = AttemptHeartbeat(attempt=_attempt(), interval_seconds=0.01, send=send)
    heartbeat.start()
    heartbeat.stop()
    settled = count
    time.sleep(0.05)

    assert count == settled
    assert heartbeat.is_alive is False


def test_start_twice_is_rejected() -> None:
    heartbeat = AttemptHeartbeat(
        attempt=_attempt(), interval_seconds=0.01, send=lambda request: None
    )
    heartbeat.start()
    try:
        with pytest.raises(RuntimeError):
            heartbeat.start()
    finally:
        heartbeat.stop()
