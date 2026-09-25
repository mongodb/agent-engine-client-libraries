"""Independent attempt-lease renewal for OE-owned durable executions.

The loop runs on its own daemon thread so tenant LLM/tool work may block
longer than a lease while heartbeats keep ownership alive. Heartbeat
intervals renew the lease only; they never append workflow events.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_NOT_FOUND,
    WORKFLOW_ERROR_CODE_STALE_FENCE,
    WORKFLOW_ERROR_CODE_UNAUTHORIZED,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    AttemptHeartbeatRequest,
)
from agent_engine_runner_shared.workflow.client import WorkflowClient, WorkflowClientError

__all__ = ["AttemptHeartbeat", "attempt_heartbeat_request_from_context"]

logger = logging.getLogger(__name__)

_START_TIMEOUT_SECONDS = 10.0

# Coded rejections that mean the lease is permanently gone: the attempt was
# superseded or revoked, so further renewal is pointless and any outcome this
# worker reports will be fenced out.
_LEASE_LOST_CODES = frozenset(
    {
        WORKFLOW_ERROR_CODE_STALE_FENCE,
        WORKFLOW_ERROR_CODE_UNAUTHORIZED,
        WORKFLOW_ERROR_CODE_NOT_FOUND,
    }
)


def attempt_heartbeat_request_from_context(attempt: AttemptContext) -> AttemptHeartbeatRequest:
    """Build the ProtoJSON heartbeat request for one OE-issued attempt."""
    if not attempt.workflow_identity.execution_id:
        raise ValueError("AttemptContext.workflow_identity is required for attempt heartbeats")
    return AttemptHeartbeatRequest(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        owner_id=attempt.owner_id,
    )


class AttemptHeartbeat:
    """Renew one attempt lease from a thread independent of agent work.

    The initial renewal is awaited and fail-closed: `start()` raises when OE
    rejects or cannot be reached, so an attempt never runs without a live
    lease. Later failures are logged and retried on the next interval so a
    transient OE blip does not kill healthy work.
    """

    def __init__(
        self,
        *,
        attempt: AttemptContext,
        interval_seconds: float,
        oe_url: str | None = None,
        send: Callable[[AttemptHeartbeatRequest], None] | None = None,
        start_timeout_seconds: float = _START_TIMEOUT_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if send is None and not oe_url:
            raise ValueError("oe_url or send is required")
        self._request = attempt_heartbeat_request_from_context(attempt)
        self._interval_seconds = interval_seconds
        self._oe_url = oe_url
        self._send = send
        self._start_timeout_seconds = start_timeout_seconds
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lease_lost = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        """Return whether the heartbeat thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def lease_lost(self) -> bool:
        """True once OE permanently rejected the lease (e.g. stale fence)."""
        return self._lease_lost.is_set()

    def start(self) -> None:
        """Send the first renewal fail-closed, then renew until `stop()`."""
        if self.is_alive:
            raise RuntimeError("attempt heartbeat is already running")
        self._stop.clear()
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"workflow-heartbeat-{self._request.attempt_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=self._start_timeout_seconds):
            self.stop()
            raise RuntimeError("initial attempt heartbeat timed out")
        if self._startup_error is not None:
            self.stop()
            raise RuntimeError("initial attempt heartbeat failed") from self._startup_error

    def stop(self) -> None:
        """Stop and join the heartbeat thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval_seconds * 2))
            if self._thread.is_alive():
                # A send stuck past the join budget: the daemon thread exits at
                # its next interval check, but flag it so a lingering renewal
                # after cleanup is visible in logs.
                logger.warning(
                    "Attempt heartbeat thread did not exit within the stop "
                    "timeout; it will halt at its next interval check",
                    extra={"attempt_id": self._request.attempt_id},
                )

    def _run(self) -> None:
        try:
            if self._send is not None:
                self._run_loop(self._send)
            else:
                assert self._oe_url is not None
                with WorkflowClient(self._oe_url) as client:
                    self._run_loop(client.heartbeat)
        except Exception as error:  # noqa: BLE001 - reported to the starting thread
            self._mark_startup_failed(error)

    def _run_loop(self, send: Callable[[AttemptHeartbeatRequest], None]) -> None:
        try:
            send(self._request)
        except Exception as error:  # noqa: BLE001 - activation must fail closed
            self._mark_startup_failed(error)
            return
        self._ready.set()
        while not self._stop.wait(self._interval_seconds):
            try:
                send(self._request)
            except WorkflowClientError as error:
                if error.code in _LEASE_LOST_CODES:
                    # The attempt was superseded or revoked; renewing further
                    # is pointless and this worker's outcomes will be fenced
                    # out. Stop and expose the loss to the request lifecycle.
                    self._lease_lost.set()
                    logger.error(
                        "Attempt lease permanently lost (%s); stopping renewals",
                        error.message,
                        extra={"attempt_id": self._request.attempt_id},
                    )
                    return
                logger.exception(
                    "Failed to renew durable attempt lease",
                    extra={"attempt_id": self._request.attempt_id},
                )
            except Exception:  # noqa: BLE001 - transient failures retried until shutdown
                logger.exception(
                    "Failed to renew durable attempt lease",
                    extra={"attempt_id": self._request.attempt_id},
                )

    def _mark_startup_failed(self, error: Exception) -> None:
        if self._ready.is_set():
            logger.exception(
                "Attempt heartbeat stopped unexpectedly",
                extra={"attempt_id": self._request.attempt_id},
            )
            return
        self._startup_error = error
        self._ready.set()
