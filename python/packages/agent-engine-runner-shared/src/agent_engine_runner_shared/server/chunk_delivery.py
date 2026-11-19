"""Ordered, coalescing, budget-bounded delivery of a turn's stream chunks.

When the engine negotiates durable acknowledgement (it stamps
``durable_ack_enabled`` on the execute request because the workspace advertised
the ``durable_ack`` capability), a 200 response to ``/stream/chunk`` means every
chunk in the request is durably persisted — a delivery-identity dedup hit
counts as persisted. This module implements the runner half of that contract:

- chunks are posted **in production order** through one poster, so durability
  order matches production order even across failed requests and takeovers;
- chunks produced while a request is in flight are **coalesced** into the next
  request (bounded batch), so a slow ack does not add a POST per token;
- a batch is **retried unchanged** until the engine acknowledges it or the
  turn's budget is exhausted — never resent after a 200, never dropped
  silently;
- the waiting queue is **bounded**, and producers block while it is full, so
  memory stays bounded at any token rate; a failed sender wakes blocked
  producers with the failure;
- a failed sender is terminal for the turn: the settlement paths refuse to
  present the turn as complete.

**Journal mode** (``journal_mode=True``, set when the engine asked for the
runner's stream journal) keeps the poster and its ordering guarantees but
reverses the durability obligations above, because the journal — not live
delivery — is the durable copy of the turn:

- producers never block and never fail the turn: a full queue drops the chunk
  from live delivery (the journal keeps it);
- an exhausted retry budget drops the chunk and delivery continues with the
  next one, instead of failing the sender;
- settlement does not wait for delivery to drain, so an engine outage that
  exhausts live delivery never turns a successful execution into a failed one.

Journal mode also posts one chunk per request in the legacy raw shape: a
journal dispatch never negotiates durable acknowledgement, and the engine parses
a batch envelope only under that negotiation, so a batch would be ingested as a
single unknown frame. One chunk in flight at a time is also all live delivery
needs — the journal supplies everything older.

The not-negotiated path does not construct a sender and is untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import httpx

from agent_engine_runner_shared.server.http_retry import (
    RetryBudgetExhausted,
    is_retryable_callback_error,
    post_json_with_retries_until,
)

logger = logging.getLogger(__name__)

# One request maps onto at most one engine publication batch: the engine's
# own per-batch bound is 32 chunks, and a request over it is rejected.
DURABLE_ACK_MAX_BATCH_CHUNKS = 32
# The engine caps a request body at 1 MiB and rejects an oversized batch, so
# a batch is also bounded by its serialized request size. The estimate is
# exact: the payload is compact JSON with UTF-8 encoding, which is what the
# HTTP client puts on the wire.
DURABLE_ACK_MAX_BATCH_BYTES = 1 << 20
# Bounded waiting memory. The queue cap plus one in-flight batch keeps the
# total unacknowledged window at or below the engine's resume-dedup budget
# (its FIFO dedup set is capped at 1024, and only unacknowledged chunks are
# ever resent): 96 queued + 32 in flight = 128.
DURABLE_ACK_MAX_QUEUED_CHUNKS = 96
# Retry backoff for the in-order batch loop. The cap keeps the settlement
# drain's wake-up latency bounded when the engine is slow or down.
DURABLE_ACK_BASE_DELAY_S = 0.1
DURABLE_ACK_MAX_DELAY_S = 2.0


def _serialized_size(value: Any) -> int:
    """Wire size of one JSON value in the request body (compact, UTF-8)."""
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


class DurableAckError(Exception):
    """The turn's chunk stream could not be acknowledged.

    Raised when the sender could not confirm durable delivery within its
    budget, or the engine answered with a non-retryable failure. The turn must
    not be presented as complete.
    """


class DurableChunkSender:
    """One ordered poster per execution for the negotiated durable-ack path.

    ``send`` enqueues a chunk (blocking while the bounded queue is full, which
    is the backpressure that bounds memory); the poster drains the queue in
    order, coalescing up to ``max_batch_chunks`` per request. ``settle`` waits
    until everything enqueued so far is acknowledged; the terminal/settlement
    paths call it before presenting the turn, and a raise from it means the
    turn must be failed instead.

    In journal mode (``journal_mode=True``) the poster keeps the batch form but
    live delivery is best effort: ``send`` never blocks or raises (a full queue
    drops the chunk), a failed batch is dropped and delivery continues, and
    ``settle`` returns without waiting for the drain.
    """

    def __init__(
        self,
        *,
        execution_id: str,
        chunk_url: str,
        client_getter: Callable[[], Awaitable[httpx.AsyncClient]],
        deadline: float,
        owner_url_getter: Optional[Callable[[], Optional[str]]] = None,
        on_owner_failure: Optional[Callable[[], None]] = None,
        max_batch_chunks: int = DURABLE_ACK_MAX_BATCH_CHUNKS,
        max_batch_bytes: int = DURABLE_ACK_MAX_BATCH_BYTES,
        max_queued_chunks: int = DURABLE_ACK_MAX_QUEUED_CHUNKS,
        base_delay: float = DURABLE_ACK_BASE_DELAY_S,
        max_delay: float = DURABLE_ACK_MAX_DELAY_S,
        journal_mode: bool = False,
    ) -> None:
        self._execution_id = execution_id
        self._chunk_url = chunk_url
        self._client_getter = client_getter
        self._deadline = deadline
        self._owner_url_getter = owner_url_getter
        self._on_owner_failure = on_owner_failure
        self._max_batch_chunks = max_batch_chunks
        self._max_batch_bytes = max_batch_bytes
        self._max_queued_chunks = max_queued_chunks
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._journal_mode = journal_mode

        # The envelope size depends only on the execution id: the request body
        # is ``{"execution_id": ..., "chunks": [<chunk>, ...]}``, so its size is
        # this base plus each chunk's serialized size plus one comma between
        # chunks.
        self._envelope_bytes = _serialized_size({"execution_id": execution_id, "chunks": []})
        self._pending: deque[tuple[dict[str, Any], int]] = deque()
        self._enqueued = 0
        self._acked = 0
        self._dropped = 0
        self._failed: Optional[BaseException] = None
        self._closed = False
        self._condition = asyncio.Condition()
        self._poster = asyncio.create_task(self._run(), name=f"durable-chunk-sender:{execution_id}")

    @property
    def failed(self) -> Optional[BaseException]:
        return self._failed

    @property
    def journal_mode(self) -> bool:
        """True when live delivery is best effort (the journal is the durable copy)."""
        return self._journal_mode

    @property
    def dropped(self) -> int:
        """Chunks abandoned by journal-mode live delivery (kept in the journal)."""
        return self._dropped

    async def send(self, chunk: dict[str, Any]) -> None:
        """Queue one chunk for ordered delivery.

        Returns once the chunk is queued — not once it is acknowledged — so a
        POST in flight does not stall production; it blocks only while the
        bounded queue is full (backpressure). Raises :class:`DurableAckError`
        when the sender has failed or been closed: the chunk can no longer be
        delivered, and the caller must fail the turn rather than drop it.

        In journal mode nothing blocks and nothing raises: a full queue, a
        failed poster, or a closed sender drops the chunk from live delivery
        and returns, because the journal — not live delivery — is the durable
        copy of the turn.
        """
        async with self._condition:
            if self._journal_mode:
                if (
                    len(self._pending) >= self._max_queued_chunks
                    or self._failed is not None
                    or self._closed
                ):
                    self._drop(reason="queue full, failed, or closed")
                    return
                self._pending.append((chunk, _serialized_size(chunk)))
                self._enqueued += 1
                self._condition.notify_all()
                return
            while (
                len(self._pending) >= self._max_queued_chunks
                and self._failed is None
                and not self._closed
            ):
                await self._condition.wait()
            self._raise_if_unusable()
            self._pending.append((chunk, _serialized_size(chunk)))
            self._enqueued += 1
            self._condition.notify_all()

    async def settle(self, *, timeout: Optional[float] = None) -> None:
        """Wait until every chunk enqueued so far is acknowledged.

        ``timeout`` bounds the wait below the sender's turn budget; the error
        paths use it so a failing turn's terminal is not held on the network.
        Raises :class:`DurableAckError` when the sender failed or the budget
        ran out with chunks still unacknowledged — the turn must not be
        presented as complete.

        In journal mode settlement never waits for the poster to drain (and so
        never raises): an engine outage that exhausts live delivery must not
        turn a successful execution into a failed one, because the journal
        already holds the turn's chunks.
        """
        if self._journal_mode:
            return
        deadline = self._deadline
        if timeout is not None:
            deadline = min(deadline, time.monotonic() + timeout)
        async with self._condition:
            while self._acked < self._enqueued:
                self._raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._incomplete_error_and_poison()
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    # The poster may be sleeping between retries past the
                    # deadline; re-check before giving up on the wait.
                    self._raise_if_failed()
                    if self._acked < self._enqueued:
                        raise self._incomplete_error_and_poison() from None
            self._raise_if_failed()

    async def aclose(self) -> None:
        """Stop the poster and abandon anything still queued.

        Called from the execution teardown: the terminal paths have already
        settled (or failed the turn), so unacknowledged chunks can no longer
        be presented.
        """
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._poster.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._poster

    def _raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed

    def _raise_if_unusable(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise DurableAckError(
                f"durable chunk sender for execution {self._execution_id} is closed"
            )

    def _incomplete_error(self) -> DurableAckError:
        return DurableAckError(
            f"durable chunk acknowledgement incomplete for execution "
            f"{self._execution_id}: {self._enqueued - self._acked} chunk(s) unacknowledged"
        )

    def _incomplete_error_and_poison(self) -> DurableAckError:
        """Build the incomplete error, poisoning the sender when the turn's own
        budget (not a bounded drain timeout) is what expired: any later send
        must fail rather than queue into a turn that can no longer be
        presented. Called with the condition held."""
        error = self._incomplete_error()
        if time.monotonic() >= self._deadline:
            self._set_failure(error)
        return error

    async def _run(self) -> None:
        try:
            while self._failed is None:
                batch = await self._next_batch()
                if batch is None:
                    return
                await self._deliver(batch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the poster must never die silently
            if self._journal_mode:
                # Journal mode never fails the turn: stop live delivery and
                # leave the chunks to the journal.
                logger.warning(
                    "Journal-mode live delivery for execution %s stopped after an "
                    "unexpected failure; the journal keeps the turn's chunks: %s",
                    self._execution_id,
                    exc,
                )
                return
            await self._fail(
                exc
                if isinstance(exc, DurableAckError)
                else DurableAckError(
                    f"durable chunk delivery for execution {self._execution_id} "
                    f"failed unexpectedly: {exc}"
                )
            )

    async def _next_batch(self) -> Optional[list[dict[str, Any]]]:
        """Pop the next coalesced batch, or ``None`` once delivery is over.

        The batch is the largest ordered prefix that fits both the engine's
        chunk-count bound and its request-size bound, so one request maps onto
        at most one publication batch. A chunk that cannot fit the size bound
        even alone can never be delivered: it fails the sender, or — in journal
        mode, where live delivery is best effort — is dropped and delivery
        moves on to the rest of the queue.
        """
        async with self._condition:
            while True:
                while not self._pending and not self._closed and self._failed is None:
                    await self._condition.wait()
                if not self._pending or self._failed is not None:
                    return None
                batch: list[dict[str, Any]] = []
                # Journal mode posts the legacy single-chunk shape. A journal
                # dispatch never negotiates durable acknowledgement, and the
                # engine parses a batch envelope only under that negotiation, so
                # a batch would be ingested as one unknown frame. One raw chunk
                # per request keeps live delivery on the shape the engine
                # already accepts, which is what the journal replaces.
                size = 0 if self._journal_mode else self._envelope_bytes
                max_chunks = 1 if self._journal_mode else self._max_batch_chunks
                while self._pending and len(batch) < max_chunks:
                    chunk, chunk_bytes = self._pending[0]
                    if batch:
                        chunk_bytes += 1  # comma between elements
                    if size + chunk_bytes > self._max_batch_bytes:
                        break
                    self._pending.popleft()
                    batch.append(chunk)
                    size += chunk_bytes
                if batch:
                    # Popping freed queue space: wake any producer blocked on
                    # the cap.
                    self._condition.notify_all()
                    return batch
                # The head chunk alone exceeds the request cap: it can never be
                # sent.
                _, chunk_bytes = self._pending.popleft()
                if self._journal_mode:
                    self._drop(
                        reason=f"chunk of {chunk_bytes} bytes exceeds the request size limit"
                    )
                    continue
                self._set_failure(
                    DurableAckError(
                        f"chunk of {chunk_bytes} bytes for execution "
                        f"{self._execution_id} exceeds the engine's request size "
                        f"limit of {self._max_batch_bytes} bytes"
                    )
                )
                return None

    async def _deliver(self, batch: list[dict[str, Any]]) -> None:
        """Post one batch, retrying the same chunks in order until acknowledged."""
        # Journal mode always carries exactly one chunk (see _next_batch) and
        # posts it in the legacy raw shape the unnegotiated engine parses.
        payload: dict[str, Any]
        if self._journal_mode:
            payload = batch[0]
        else:
            payload = {"execution_id": self._execution_id, "chunks": batch}
        try:
            client = await self._client_getter()
            owner_url = self._owner_url_getter() if self._owner_url_getter else None
            await post_json_with_retries_until(
                client,
                self._chunk_url,
                payload,
                deadline=self._deadline,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
                owner_url=owner_url,
                on_owner_failure=self._on_owner_failure if owner_url else None,
                # Delivery is idempotent (the engine dedupes by delivery
                # identity), so the durable path retries the broader
                # established-connection failure set — a read/write failure on
                # a live connection must not end a turn that can still land.
                is_retryable=is_retryable_callback_error,
            )
        except RetryBudgetExhausted:
            if self._journal_mode:
                # Journal mode: live delivery is best effort, so an exhausted
                # retry budget drops this batch and delivery moves on.
                self._drop(reason="retry budget exhausted", count=len(batch))
                return
            await self._fail(
                DurableAckError(
                    f"durable chunk acknowledgement timed out for execution "
                    f"{self._execution_id} after {self._acked} of "
                    f"{self._enqueued} chunk(s) were acknowledged"
                )
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any other failure is terminal for the turn
            if self._journal_mode:
                self._drop(reason=str(exc), count=len(batch))
                return
            await self._fail(
                DurableAckError(
                    f"durable chunk delivery for execution {self._execution_id} failed: {exc}"
                )
            )
            return
        async with self._condition:
            self._acked += len(batch)
            self._condition.notify_all()

    def _drop(self, *, reason: str, count: int = 1) -> None:
        """Abandon ``count`` chunk(s) from journal-mode live delivery.

        Journal mode only: the chunks are already recorded in the runner's
        stream journal, so dropping them from live delivery costs at most a
        live ``seq`` gap, never a turn. Called with the condition held.
        """
        self._dropped += count
        logger.debug(
            "Journal-mode live delivery for execution %s dropped %d chunk(s) "
            "(%s); the journal keeps them",
            self._execution_id,
            count,
            reason,
        )

    async def _fail(self, error: BaseException) -> None:
        async with self._condition:
            self._set_failure(error)

    def _set_failure(self, error: BaseException) -> None:
        """Record a terminal sender failure; called with the condition held."""
        if self._failed is None:
            self._failed = error
            logger.warning(
                "Durable chunk delivery failed for execution %s with %d of %d "
                "chunk(s) acknowledged; the turn must not be presented as complete: %s",
                self._execution_id,
                self._acked,
                self._enqueued,
                error,
            )
        self._condition.notify_all()
