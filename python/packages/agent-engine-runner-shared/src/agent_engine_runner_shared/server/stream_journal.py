"""Runner-side stream journal: the runner's own on-disk record of a turn's chunks.

When the engine stamps ``stream_journal_enabled`` on the execute request, the
runner keeps a journal of every chunk it emits for that lifecycle on its own
disk, so any engine replica — including one that never received a chunk — can
read the turn's tail after a hard owner death, without the engine writing
anything per chunk.

One journal per ``(execution_id, generation)``. A lifecycle closes its journal
with an end marker on completion, error, suspension and abort. The end marker
says only that this generation stopped producing; it says nothing about whether
the execution is terminal.

The write path never blocks the producer and never fails the turn: a chunk goes
onto a bounded queue, and one writer appends it on a worker thread so no
blocking disk call runs on the event loop. Appends are not fsynced — the journal
dies with the VM. A bound hit, a lost write, or an overflow marks the journal
``incomplete`` at the last *written* sequence and stops recording; entries up to
that point stay readable.

Records are a 4-byte big-endian length prefix, a UTF-8 JSON body, and a 4-byte
CRC32 of the body. A chunk record carries ``kind: "chunk"``, its ``seq``, its
``generation`` and the ``chunk``. A lifecycle end is a ``kind: "end"`` record
carrying the ``generation``, the last written ``seq`` and the close ``reason``.
A torn final record is ignored on read, and its presence is reported as
``incomplete`` rather than a clean end.

Accounting: each open journal reserves its potential size (the chunk bound plus
an end-marker allowance) against the runner budget, so several journals opened
before any writes cannot over-admit. The reservation is released, and the
journal becomes eligible for eviction, only after the writer has settled — a
journal still draining keeps its reservation and cannot be evicted.

A reader replays a journal's recorded prefix from a sequence cursor and then
follows live appends on the same connection, so a resumed turn sees contiguous
events rather than a snapshot taken behind the producer. The connection ends
with exactly one terminal event: ``end`` when the journal closed cleanly, or
``incomplete`` when its prefix cannot be trusted (a bound hit, a lost write, a
torn final record, or a close that never wrote an end marker). A reader slot is
accounted per journal and per runner, so a stuck or hostile engine replica
cannot pin unbounded follower loops on one runner.

The HTTP routes that expose this to the engine live in the AER server; this
module owns the record format, the writer, the whole-runner budget, eviction
and retention, the incremental reader, and reader accounting.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import struct
import zlib
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_DIR = os.environ.get(
    "AGENTENGINE_STREAM_JOURNAL_DIR", "/scratch/agent-engine-stream-journal"
)

# Per journal: 64 MiB or 50,000 chunks of chunk records, matching the engine's
# stream buffer. The lifecycle end record is additional, bounded by the reserve.
DEFAULT_JOURNAL_MAX_BYTES = 64 << 20
DEFAULT_JOURNAL_MAX_CHUNKS = 50_000
# Bytes reserved beyond the chunk bound for the lifecycle end record.
END_MARKER_RESERVE = 512
# Whole runner: 256 MiB of journal data; closed journals are evicted oldest-first.
DEFAULT_RUNNER_MAX_BYTES = 256 << 20
# Closed journals kept per session.
DEFAULT_MAX_CLOSED_PER_SESSION = 2
# Bounded producer queue ahead of the writer.
DEFAULT_QUEUE_MAX_CHUNKS = 4096
# How long close() waits for the writer to settle before returning.
CLOSE_DRAIN_TIMEOUT_S = 2.0
# How long admission waits for closed-journal reclamation before proceeding.
ADMISSION_TIMEOUT_S = 1.0
# Concurrent live-follow readers, per journal and per runner. A follower is a
# long-lived connection, so the cap is what keeps a stuck or hostile engine
# replica from pinning unbounded read loops on one runner.
DEFAULT_MAX_READERS_PER_JOURNAL = 4
DEFAULT_MAX_READERS_PER_RUNNER = 32
# A follower is woken on every append and close; this bound is only the
# backstop that keeps it re-reading if a wake-up is ever missed.
READER_WAKE_TIMEOUT_S = 0.5

REASON_COMPLETED = "completed"
REASON_ERROR = "error"
REASON_SUSPENDED = "suspended"
REASON_ABORTED = "aborted"
REASON_INCOMPLETE = "incomplete"

KIND_CHUNK = "chunk"
KIND_END = "end"

_HEADER = struct.Struct(">I")
_TRAILER = struct.Struct(">I")


class ReaderLimitExceeded(Exception):
    """The per-journal or per-runner live-follow reader cap is already taken."""


class ReaderLease:
    """A reserved reader slot, released exactly once when a follow ends."""

    __slots__ = ("journal_key", "released")

    def __init__(self, journal_key: Tuple[str, int]) -> None:
        self.journal_key = journal_key
        self.released = False


def journal_file_name(execution_id: str, generation: int) -> str:
    """Encode a journal filename from a tenant-influenced execution id.

    base64url is path-safe by construction: it never emits ``/``, ``.`` or
    ``..``, so an execution id can never escape the journal root.
    """
    encoded = base64.urlsafe_b64encode(execution_id.encode("utf-8")).decode("ascii")
    return f"{encoded.rstrip('=')}.{generation}.journal"


def _encode_frame(record: Dict[str, Any]) -> bytes:
    """Encode one record: length prefix, UTF-8 JSON body, CRC32 of the body."""
    body = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _HEADER.pack(len(body)) + body + _TRAILER.pack(zlib.crc32(body) & 0xFFFFFFFF)


def _read_records(path: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """Read every complete, checksummed record.

    Returns ``(chunks, end, torn)`` where ``chunks`` are the chunk records in
    order, ``end`` is the lifecycle end record if one was written, and ``torn``
    is true when the file ends in a short, CRC-mismatched, or undecodable
    trailing record: an interrupted append that a reader must report as
    ``incomplete`` rather than a clean end.
    """
    chunks: List[Dict[str, Any]] = []
    end: Optional[Dict[str, Any]] = None
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return chunks, end, False
    except OSError:
        return chunks, end, True
    offset = 0
    n = len(data)
    while offset < n:
        if offset + _HEADER.size > n:
            return chunks, end, True
        (length,) = _HEADER.unpack_from(data, offset)
        start = offset + _HEADER.size
        record_end = start + length
        if record_end + _TRAILER.size > n:
            return chunks, end, True
        body = data[start:record_end]
        (crc,) = _TRAILER.unpack_from(data, record_end)
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            return chunks, end, True
        try:
            record = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return chunks, end, True
        if record.get("kind") == KIND_END:
            end = record
        else:
            chunks.append(record)
        offset = record_end + _TRAILER.size
    return chunks, end, False


def _read_records_from(
    path: Path, offset: int
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int, bool]:
    """Read the complete records appended after ``offset``.

    Returns ``(chunks, end, next_offset, torn)``. ``next_offset`` advances only
    past records that were fully written, so a live follower re-reads a partial
    trailing record once the writer finishes it instead of serving a truncated
    body. ``torn`` is true when the bytes after ``next_offset`` are a short,
    CRC-mismatched or undecodable record: harmless while the journal is live,
    and ``incomplete`` once it has closed.
    """
    chunks: List[Dict[str, Any]] = []
    end: Optional[Dict[str, Any]] = None
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except FileNotFoundError:
        return chunks, end, offset, False
    except OSError:
        return chunks, end, offset, True
    position = 0
    length_ = len(data)
    while position < length_:
        if position + _HEADER.size > length_:
            return chunks, end, offset + position, True
        (body_length,) = _HEADER.unpack_from(data, position)
        body_start = position + _HEADER.size
        body_end = body_start + body_length
        if body_end + _TRAILER.size > length_:
            return chunks, end, offset + position, True
        body = data[body_start:body_end]
        (crc,) = _TRAILER.unpack_from(data, body_end)
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            return chunks, end, offset + position, True
        try:
            record = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return chunks, end, offset + position, True
        if record.get("kind") == KIND_END:
            end = record
        else:
            chunks.append(record)
        position = body_end + _TRAILER.size
    return chunks, end, offset + position, False


def _consume_task_result(task: "asyncio.Future[Any]") -> None:
    """Retrieve a background task's result so a failure is never silently lost."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Stream journal background reclamation failed: %s", exc)


class JournalWriter:
    """One ``(execution_id, generation)`` journal and its single writer task."""

    def __init__(
        self,
        registry: "StreamJournalRegistry",
        execution_id: str,
        session_id: str,
        generation: int,
        path: Path,
    ) -> None:
        self.registry = registry
        self.execution_id = execution_id
        self.session_id = session_id
        self.generation = generation
        self.path = path
        # The written prefix: advanced only after a record reaches the file.
        self.last_seq = 0
        self.bytes = 0
        # Why recording stopped; None while the journal is still open, so the
        # index reports a null reason for a live generation (the TypeScript twin
        # reports the same, and a consumer must not decode two shapes).
        self.reason: Optional[str] = None
        # Producer admission counters, kept separate from the written prefix so
        # the bounds still hold while records are queued.
        self.accepted_chunks = 0
        self.accepted_bytes = 0
        # Capacity reserved against the runner budget while this journal is
        # open; on settle it becomes the journal's written size.
        self.reserved = 0
        self.retained = True
        self.incomplete = False
        self.closed = False
        # True once the writer has settled and the journal has been finalized.
        self.finalized = False
        # True while a removal is in flight, so it is never selected twice.
        self._evicting = False
        # Creation order across the whole runner, for oldest-first eviction.
        self.order = 0
        # Encoded frames queued for the writer: (seq, bytes). Encoding once at
        # admission keeps the byte bound honest and makes the record immutable
        # to a caller that mutates the chunk after record() returns.
        self._queue: List[Tuple[int, bytes]] = []
        self._writer: Optional[asyncio.Task[None]] = None
        self._notify = asyncio.Event()
        self._done = asyncio.Event()
        self._write_failed = False
        # Live followers are woken per reader (a shared Event would let one
        # follower consume another's wake-up), and the version lets a follower
        # tell "changed while I was reading" from "nothing new".
        self._change_version = 0
        self._change_waiters: List[asyncio.Event] = []

    def start(self) -> None:
        """Start the writer task. Filesystem work happens on the writer."""
        self._writer = asyncio.create_task(
            self._run(), name=f"stream-journal:{self.execution_id}:{self.generation}"
        )

    def start_incomplete(self) -> None:
        """Start a journal that records nothing (the runner budget is full)."""
        self.incomplete = True
        self.reason = REASON_INCOMPLETE

    def _ensure_dir(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:  # noqa: BLE001 - the journal must never fail the turn
            logger.debug("Stream journal dir unavailable for %s: %s", self.execution_id, exc)
            return False

    def record(self, chunk: Dict[str, Any]) -> None:
        """Enqueue one chunk for the journal. Never blocks the producer.

        The frame is encoded once, here, so the byte bound reflects exactly what
        reaches the file and a caller mutating ``chunk`` afterwards cannot change
        what is journaled.
        """
        if self.incomplete or self.closed:
            return
        seq = int(chunk.get("seq") or 0)
        if seq <= 0:
            return
        data = _encode_frame(
            {
                "kind": KIND_CHUNK,
                "seq": seq,
                "generation": self.generation,
                "chunk": chunk,
            }
        )
        if len(self._queue) >= self.registry.queue_max_chunks:
            self._mark_incomplete()
            return
        if (
            self.accepted_chunks + 1 > self.registry.journal_max_chunks
            or self.accepted_bytes + len(data) > self.registry.journal_max_bytes
        ):
            self._mark_incomplete()
            return
        self.accepted_chunks += 1
        self.accepted_bytes += len(data)
        self._queue.append((seq, data))
        self._wake()

    def _mark_incomplete(self) -> None:
        if self.incomplete:
            return
        self.incomplete = True
        self.reason = REASON_INCOMPLETE
        self._wake()
        self._bump_change()

    def _wake(self) -> None:
        self._notify.set()

    def _bump_change(self) -> None:
        """Wake every live follower after the file or the journal state moved."""
        self._change_version += 1
        for waiter in self._change_waiters:
            waiter.set()

    async def wait_for_change(self, version: int, timeout: float = READER_WAKE_TIMEOUT_S) -> None:
        """Wait until the journal changes after ``version``, or ``timeout``.

        The version check and the waiter registration happen without an await
        between them, so a change can never slip through the gap. Each follower
        waits on its own event: a shared one would let the fastest follower
        consume a wake-up meant for the others.
        """
        if self._change_version != version or self.closed or self.incomplete:
            return
        waiter = asyncio.Event()
        self._change_waiters.append(waiter)
        try:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(waiter.wait(), timeout=timeout)
        finally:
            self._change_waiters.remove(waiter)

    def _append_bytes(self, data: bytes) -> None:
        """Blocking append, run on a worker thread so the loop never stalls."""
        with self.path.open("ab") as fh:
            fh.write(data)

    async def _run(self) -> None:
        try:
            if not await asyncio.to_thread(self._ensure_dir):
                self.incomplete = True
                self.reason = REASON_INCOMPLETE
                return
            while True:
                while self._queue:
                    seq, data = self._queue.pop(0)
                    try:
                        await asyncio.to_thread(self._append_bytes, data)
                    except OSError as exc:
                        logger.debug(
                            "Stream journal write failed for %s: %s", self.execution_id, exc
                        )
                        self._write_failed = True
                        self._mark_incomplete()
                        return
                    self.bytes += len(data)
                    self.last_seq = seq
                    self._bump_change()
                if self._write_failed:
                    return
                if self.closed:
                    await self._write_end_marker()
                    return
                self._notify.clear()
                if self._queue or self.closed:
                    continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._notify.wait(), timeout=0.1)
        except asyncio.CancelledError:
            raise
        finally:
            # The writer will consume nothing more: drop any payload it can no
            # longer write so a retired journal does not pin the memory.
            self._queue.clear()
            await self._finalize()
            self._done.set()
            # Last, so a follower that wakes on it sees the settled state.
            self._bump_change()

    async def _write_end_marker(self) -> None:
        """Append the lifecycle end record. Best-effort; never fails the turn."""
        record = {
            "kind": KIND_END,
            "generation": self.generation,
            "last_seq": self.last_seq,
            "reason": self.reason or REASON_COMPLETED,
        }
        data = _encode_frame(record)
        try:
            await asyncio.to_thread(self._append_bytes, data)
        except OSError as exc:
            logger.debug(
                "Stream journal end marker write failed for %s: %s", self.execution_id, exc
            )
            self._write_failed = True
            self._mark_incomplete()
            return
        self.bytes += len(data)

    async def _finalize(self) -> None:
        """Release the unused reservation and register the body for retention.

        Runs exactly once, after the writer has settled, so a journal that is
        still draining keeps its reservation and is never evicted.
        """
        if self.finalized:
            return
        self.finalized = True
        if not self._write_failed:
            self.registry.runner_bytes -= self.reserved - self.bytes
            self.reserved = self.bytes
        # On an uncertain append the file may hold a torn prefix longer than the
        # written prefix; keep the reservation until the body is deleted.
        await self.registry._on_close(self)

    async def close(self, reason: str) -> None:
        """Close the journal and let the writer settle. Idempotent.

        Waits a bounded time for the writer. On timeout the journal is marked
        incomplete and close returns: the writer keeps its reservation until it
        settles, and finalizes then.
        """
        if self.closed:
            return
        self.closed = True
        if not self.incomplete:
            self.reason = reason
        self._wake()
        self._bump_change()
        if self._writer is not None:
            try:
                await asyncio.wait_for(
                    self._done.wait(), timeout=self.registry.close_drain_timeout_s
                )
            except asyncio.TimeoutError:
                self._mark_incomplete()
                return
            return
        # No writer: bound the finalization too, so retention removal on slow
        # storage cannot stall the turn. The outstanding cleanup keeps its
        # charge until it settles.
        task = asyncio.ensure_future(self._finalize())
        task.add_done_callback(_consume_task_result)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(task), timeout=self.registry.close_drain_timeout_s
            )

    async def iter_events(self, after_seq: int) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Yield this journal's events past ``after_seq``, then one terminal event.

        Replays the recorded prefix in sequence order and then follows live
        appends until the journal closes, so a resumed turn sees contiguous
        events instead of a snapshot taken behind the producer. The terminal
        event is ``end`` for a clean close and ``incomplete`` whenever the
        prefix cannot be trusted — a bound hit, a lost write, a torn final
        record, or a close that never wrote an end marker. An ``incomplete``
        journal yields it as soon as the flag is set rather than after the
        turn ends: nothing more can be recorded, so a follower that waited
        would hang for the rest of the turn to learn the prefix is untrustworthy.
        """
        sent = after_seq
        offset = 0
        # The end record may be decoded in the same read that still had chunks
        # to hand the client, so carry it to the terminal decision below.
        last_end: Optional[Dict[str, Any]] = None
        while True:
            # close() flips `closed` before the writer drains; a reader must
            # judge the *settled* file, or it would report the last queued
            # chunk as missing. Bounded, so a writer that never settles is read
            # as-is (and then reported incomplete).
            if self.closed and self._writer is not None and not self._done.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._done.wait(), timeout=self.registry.close_drain_timeout_s
                    )
            version = self._change_version
            chunks, end, offset, torn = await asyncio.to_thread(
                _read_records_from, self.path, offset
            )
            if end is not None:
                last_end = end
            emitted = 0
            for record in chunks:
                seq = int(record.get("seq") or 0)
                if seq <= sent:
                    continue
                sent = seq
                emitted += 1
                yield (
                    "chunk",
                    {
                        "generation": self.generation,
                        "seq": seq,
                        "chunk": record.get("chunk"),
                    },
                )
            if emitted or version != self._change_version:
                # The journal moved while this read was in flight — a chunk or
                # the end marker was appended, or `incomplete` was set. Re-read
                # so `end`/`torn`/`incomplete` describe the file as it is now
                # instead of a state this read raced past.
                continue
            if self.incomplete:
                yield (
                    "incomplete",
                    {
                        "generation": self.generation,
                        "last_seq": sent,
                        "reason": REASON_INCOMPLETE,
                    },
                )
                return
            if self.closed:
                if torn or last_end is None:
                    yield (
                        "incomplete",
                        {
                            "generation": self.generation,
                            "last_seq": sent,
                            "reason": REASON_INCOMPLETE,
                        },
                    )
                else:
                    yield (
                        "end",
                        {
                            "generation": self.generation,
                            "last_seq": sent,
                            "reason": last_end.get("reason") or REASON_COMPLETED,
                        },
                    )
                return
            await self.wait_for_change(self._change_version)

    async def evict(self) -> bool:
        """Select the body for removal and reclaim its charge once it is gone.

        Returns True once the body is reclaimed (or was already gone). The charge
        is released only after the file is confirmed absent, and ``retained``
        flips only then too, so a transient removal failure leaves the body
        selectable for a later retry instead of stranding it and its charge.
        """
        if self._evicting:
            return False
        if not self.retained:
            return True
        self._evicting = True
        # Flip immediately so the body is never selected twice while the
        # removal is in flight.
        self.retained = False
        try:
            if not await asyncio.to_thread(self._unlink):
                # The file is still on disk: keep the charge and make the body
                # selectable again so a later admission can retry.
                self.retained = True
                return False
            self.registry.runner_bytes -= self.reserved
            self.reserved = 0
            return True
        finally:
            self._evicting = False

    def _unlink(self) -> bool:
        """Remove the body. True once it is confirmed absent."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.warning("Stream journal eviction failed for %s: %s", self.execution_id, exc)
            return False
        return True

    def index_entry(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "last_seq": self.last_seq,
            "reason": self.reason,
            "retained": self.retained,
        }


class StreamJournalRegistry:
    """Process-global journal registry for one runner."""

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        queue_max_chunks: int = DEFAULT_QUEUE_MAX_CHUNKS,
        journal_max_bytes: int = DEFAULT_JOURNAL_MAX_BYTES,
        journal_max_chunks: int = DEFAULT_JOURNAL_MAX_CHUNKS,
        runner_max_bytes: int = DEFAULT_RUNNER_MAX_BYTES,
        max_closed_per_session: int = DEFAULT_MAX_CLOSED_PER_SESSION,
        end_marker_reserve: int = END_MARKER_RESERVE,
        admission_timeout_s: float = ADMISSION_TIMEOUT_S,
        close_drain_timeout_s: float = CLOSE_DRAIN_TIMEOUT_S,
        max_readers_per_journal: int = DEFAULT_MAX_READERS_PER_JOURNAL,
        max_readers_per_runner: int = DEFAULT_MAX_READERS_PER_RUNNER,
    ) -> None:
        self._root = Path(root or DEFAULT_JOURNAL_DIR)
        self.queue_max_chunks = queue_max_chunks
        self.journal_max_bytes = journal_max_bytes
        self.journal_max_chunks = journal_max_chunks
        self.runner_max_bytes = runner_max_bytes
        self.max_closed_per_session = max_closed_per_session
        self.end_marker_reserve = end_marker_reserve
        self.admission_timeout_s = admission_timeout_s
        self.close_drain_timeout_s = close_drain_timeout_s
        self.max_readers_per_journal = max_readers_per_journal
        self.max_readers_per_runner = max_readers_per_runner

        # Sum of open journals' reservations plus settled journals' written size.
        self.runner_bytes = 0
        self._sequence = 0
        self._journals: Dict[str, Dict[int, JournalWriter]] = {}
        self._closed_by_session: Dict[str, List[JournalWriter]] = {}
        # Live-follow reader slots, per journal and per runner.
        self._runner_readers = 0
        self._journal_readers: Dict[Tuple[str, int], int] = {}
        # In-flight admissions per (execution_id, generation), so a duplicate
        # open of the same tuple awaits the original decision instead of
        # observing a journal that is not admitted yet.
        self._admissions: Dict[Tuple[str, int], "asyncio.Future[JournalWriter]"] = {}

    @property
    def journal_reservation(self) -> int:
        """Capacity one open journal reserves against the runner budget."""
        return self.journal_max_bytes + self.end_marker_reserve

    async def open_journal(
        self, *, execution_id: str, session_id: str, generation: int
    ) -> JournalWriter:
        """Open a journal for a dispatch. Idempotent per (execution, generation).

        A duplicate open while the first admission is still reclaiming capacity
        awaits that same admission and receives the same journal, so neither
        caller can close a journal whose writer has not been started yet.
        """
        key = (execution_id, generation)
        # The in-flight admission is authoritative: the journal is published
        # into ``_journals`` before the writer starts, so a duplicate must
        # consult the admission before the published journal.
        inflight = self._admissions.get(key)
        if inflight is not None:
            # Shield the shared admission: cancelling one duplicate must not
            # cancel the admission the other callers are waiting on.
            return await asyncio.shield(inflight)
        existing = self._journals.get(execution_id, {}).get(generation)
        if existing is not None:
            return existing
        admission: "asyncio.Future[JournalWriter]" = asyncio.get_running_loop().create_future()
        admission.add_done_callback(_consume_task_result)
        self._admissions[key] = admission
        try:
            journal = await self._admit_journal(
                execution_id=execution_id, session_id=session_id, generation=generation
            )
        except BaseException as exc:
            admission.set_exception(exc)
            raise
        finally:
            self._admissions.pop(key, None)
        admission.set_result(journal)
        return journal

    async def _admit_journal(
        self, *, execution_id: str, session_id: str, generation: int
    ) -> JournalWriter:
        gens = self._journals.setdefault(execution_id, {})
        existing = gens.get(generation)
        if existing is not None:
            return existing
        journal = JournalWriter(
            self,
            execution_id,
            session_id,
            generation,
            self._root / journal_file_name(execution_id, generation),
        )
        gens[generation] = journal
        journal.order = self._sequence
        self._sequence += 1
        # Make room for the new journal's whole potential size by evicting
        # settled closed journals oldest-first. Bounded: if scratch storage is
        # too slow to reclaim in time, the new journal starts incomplete and the
        # turn proceeds; the outstanding removal keeps its charge until it
        # settles. A journal that is still draining is never evictable.
        try:
            await self._evict_closed_to_fit_bounded(self.journal_reservation)
        except BaseException:
            # Cancellation before the writer started: withdraw the published
            # journal, so no reader can discover a body nothing will ever write
            # or close, and terminate any follower that already captured it.
            gens.pop(generation, None)
            journal.retained = False
            journal.incomplete = True
            journal.reason = REASON_INCOMPLETE
            journal.closed = True
            journal._bump_change()
            raise
        if self.runner_bytes + self.journal_reservation > self.runner_max_bytes:
            journal.start_incomplete()
            return journal
        journal.reserved = self.journal_reservation
        self.runner_bytes += self.journal_reservation
        journal.start()
        return journal

    def _admission_pending(self, execution_id: str, generation: int) -> bool:
        """True while this ``(execution, generation)`` journal is being admitted."""
        return (execution_id, generation) in self._admissions

    def has_generation(self, execution_id: str, generation: int) -> bool:
        # A journal that is still being admitted has no writer and no
        # reservation yet; a follower must not be admitted onto it.
        if self._admission_pending(execution_id, generation):
            return False
        journal = self._journals.get(execution_id, {}).get(generation)
        return journal is not None and journal.retained

    def read_index(self, execution_id: str) -> Optional[Dict[str, Any]]:
        gens = self._journals.get(execution_id)
        if not gens:
            return None
        entries = [g.index_entry() for g in sorted(gens.values(), key=lambda j: j.generation)]
        return {"execution_id": execution_id, "generations": entries}

    async def stream(
        self, execution_id: str, generation: int, after_seq: int
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Serve one generation's journal, or report it unavailable.

        A body evicted between the route's admission check and this call must
        not become an empty 200 that a reader could mistake for a clean, empty
        stream: it answers ``incomplete``, which is what a reader takes as
        "cannot be trusted, fall back to the snapshot".
        """
        journal = self._journals.get(execution_id, {}).get(generation)
        if (
            journal is None
            or not journal.retained
            or self._admission_pending(execution_id, generation)
        ):
            yield (
                "incomplete",
                {
                    "generation": generation,
                    "last_seq": after_seq,
                    "reason": REASON_INCOMPLETE,
                },
            )
            return
        async for event in journal.iter_events(after_seq):
            yield event

    # -- reader accounting -------------------------------------------------

    def acquire_reader(self, execution_id: str, generation: int) -> ReaderLease:
        """Reserve a live-follow slot, or raise :class:`ReaderLimitExceeded`.

        Every route that holds a connection open for a journal's lifetime takes
        one, so the runner-wide and per-journal caps bound how many followers a
        single tenant's streams can pin.
        """
        key = (execution_id, generation)
        if self._runner_readers >= self.max_readers_per_runner:
            raise ReaderLimitExceeded("runner reader limit reached")
        if self._journal_readers.get(key, 0) >= self.max_readers_per_journal:
            raise ReaderLimitExceeded("journal reader limit reached")
        self._journal_readers[key] = self._journal_readers.get(key, 0) + 1
        self._runner_readers += 1
        return ReaderLease(key)

    def release_reader(self, lease: ReaderLease) -> None:
        """Release a slot. Idempotent, so a disconnect path can call it twice."""
        if lease.released:
            return
        lease.released = True
        remaining = self._journal_readers.get(lease.journal_key, 1) - 1
        if remaining <= 0:
            self._journal_readers.pop(lease.journal_key, None)
        else:
            self._journal_readers[lease.journal_key] = remaining
        self._runner_readers = max(0, self._runner_readers - 1)

    # -- retention and eviction -------------------------------------------

    async def _on_close(self, journal: JournalWriter) -> None:
        closed = self._closed_by_session.setdefault(journal.session_id, [])
        closed.append(journal)
        # Only retained bodies count against the per-session cap, and retention
        # is by creation order, not settlement order, so a writer that settles
        # late cannot outrank a newer turn's body. A body already removed by
        # budget eviction must not consume a cap slot.
        retained = sorted((j for j in closed if j.retained), key=lambda j: j.order)
        for victim in retained[: max(0, len(retained) - self.max_closed_per_session)]:
            if not await victim.evict():
                # A failed removal keeps the body and its charge; stop rather
                # than spin, and let a later admission retry.
                break
        # Prune in place: a concurrent _on_close for this session holds the same
        # list object, so replacing it would drop a body registered during the
        # awaits above. Keep a body whose removal is still in flight: it is not
        # retained yet, but it is still on disk and must stay discoverable.
        closed[:] = [j for j in closed if j.retained or j._evicting]

    async def _evict_closed_to_fit(self, need: int) -> None:
        """Evict settled closed journals, oldest first, until ``need`` fits.

        A draining journal is never a candidate, so this stops as soon as the
        only journals left are open or unsettled. The caller decides what to do
        if the budget still cannot fit ``need``.
        """
        while self.runner_bytes + need > self.runner_max_bytes:
            victim = self._oldest_closed_retained()
            if victim is None:
                return
            if not await victim.evict():
                # The removal failed; the body stays charged and selectable for
                # a later retry. Stop rather than select it again immediately.
                return

    async def _evict_closed_to_fit_bounded(self, need: int) -> None:
        """Run reclamation with a deadline so admission cannot stall a turn.

        On timeout the reclamation keeps running in the background: the
        outstanding removal holds its charge until it settles, and the caller
        starts the new journal incomplete.
        """
        task = asyncio.ensure_future(self._evict_closed_to_fit(need))
        task.add_done_callback(_consume_task_result)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=self.admission_timeout_s)

    def _oldest_closed_retained(self) -> Optional[JournalWriter]:
        candidates: List[JournalWriter] = []
        for closed in self._closed_by_session.values():
            candidates.extend(j for j in closed if j.retained)
        if not candidates:
            return None
        return min(candidates, key=lambda j: j.order)
