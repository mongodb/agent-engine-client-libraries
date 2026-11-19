"""Tests for the runner stream journal: the writer, index and runner budget.

Covers the writer layer only (the read routes and the AER wiring are separate
changes):

  - the record framing round-trips a chunk with its seq and generation;
  - a torn/short final record is ignored on read;
  - every lifecycle end appends a framed end record naming the reason;
  - the index reports the written prefix, not the queued sequence, after a
    write failure;
  - a producer-queue overflow marks the journal incomplete at the last
    recorded sequence and stops recording;
  - the index reports ``retained=false`` once a body is evicted;
  - the whole-runner budget reserves an open journal's capacity and marks a
    new journal incomplete when it cannot fit;
  - a draining writer keeps its reservation and is never evicted, and a
    delayed writer settles into creation-order retention;
  - a body is charged until its file is confirmed deleted;
  - an execution id cannot escape the journal root.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agent_engine_runner_shared.server import stream_journal as sj
from agent_engine_runner_shared.server.stream_journal import (
    REASON_ABORTED,
    REASON_COMPLETED,
    REASON_ERROR,
    REASON_SUSPENDED,
    StreamJournalRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path, **overrides: Any) -> StreamJournalRegistry:
    return StreamJournalRegistry(root=str(tmp_path), **overrides)


def _chunk(seq: int, content: str = "") -> dict[str, Any]:
    return {"execution_id": "exec-1", "chunk_type": "text", "content": content, "seq": seq}


def _chunks(writer: sj.JournalWriter) -> list[dict[str, Any]]:
    chunks, _end, _torn = sj._read_records(writer.path)
    return chunks


def _end(writer: sj.JournalWriter) -> dict[str, Any] | None:
    _chunks, end, _torn = sj._read_records(writer.path)
    return end


async def _wait_for(predicate: Any, *, timeout: float = 5.0) -> None:
    """Poll until ``predicate()`` is true — deterministic, unlike a fixed sleep."""
    for _ in range(int(timeout / 0.01) + 1):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met")


def _chunk_record_bytes(seq: int = 1, generation: int = 1) -> int:
    return len(
        sj._encode_frame(
            {"kind": sj.KIND_CHUNK, "seq": seq, "generation": generation, "chunk": _chunk(seq)}
        )
    )


def _end_record_bytes(last_seq: int, reason: str = REASON_COMPLETED, generation: int = 1) -> int:
    return len(
        sj._encode_frame(
            {"kind": sj.KIND_END, "generation": generation, "last_seq": last_seq, "reason": reason}
        )
    )


# ---------------------------------------------------------------------------
# Record format and writer
# ---------------------------------------------------------------------------


async def test_records_round_trip_with_seq_and_generation(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=3)
    for seq in range(1, 6):
        writer.record(_chunk(seq, content=f"c{seq}"))
    await writer.close(REASON_COMPLETED)

    chunks = _chunks(writer)
    assert [record["seq"] for record in chunks] == [1, 2, 3, 4, 5]
    assert [record["generation"] for record in chunks] == [3, 3, 3, 3, 3]
    assert chunks[0]["chunk"]["content"] == "c1"
    assert writer.last_seq == 5


async def test_torn_final_record_is_ignored(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=0)

    # Simulate an interrupted append: two good records followed by a short,
    # CRC-less trailing record.
    with open(writer.path, "wb") as f:
        f.write(
            sj._encode_frame({"kind": sj.KIND_CHUNK, "seq": 1, "generation": 0, "chunk": _chunk(1)})
        )
        f.write(
            sj._encode_frame({"kind": sj.KIND_CHUNK, "seq": 2, "generation": 0, "chunk": _chunk(2)})
        )
        f.write(b"\x00\x00\x00\x40short")

    chunks, end, torn = sj._read_records(writer.path)
    assert [record["seq"] for record in chunks] == [1, 2]
    assert end is None
    assert torn is True


@pytest.mark.parametrize(
    "reason", [REASON_COMPLETED, REASON_ERROR, REASON_SUSPENDED, REASON_ABORTED]
)
async def test_close_appends_a_framed_end_record(tmp_path: Path, reason: str) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    writer.record(_chunk(1))
    writer.record(_chunk(2))
    await writer.close(reason)

    chunks, end, torn = sj._read_records(writer.path)
    assert [record["seq"] for record in chunks] == [1, 2]
    assert torn is False
    assert end == {"kind": sj.KIND_END, "generation": 1, "last_seq": 2, "reason": reason}


async def test_empty_lifecycle_still_writes_the_end_marker(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=2)
    await writer.close(REASON_SUSPENDED)

    chunks, end, torn = sj._read_records(writer.path)
    assert chunks == []
    assert torn is False
    assert end == {"kind": sj.KIND_END, "generation": 2, "last_seq": 0, "reason": REASON_SUSPENDED}


async def test_close_is_idempotent(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    writer.record(_chunk(1))
    await writer.close(REASON_COMPLETED)
    size = writer.path.stat().st_size
    await writer.close(REASON_ERROR)
    assert writer.path.stat().st_size == size
    assert _end(writer) is not None


async def test_index_reports_written_prefix_after_a_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    real = sj.JournalWriter._append_bytes
    calls = {"n": 0}

    def flaky(self: sj.JournalWriter, data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(self, data)

    monkeypatch.setattr(sj.JournalWriter, "_append_bytes", flaky)
    for seq in range(1, 4):
        writer.record(_chunk(seq))
    await writer.close(REASON_ERROR)

    # Only the first record reached the file; the index must not advertise the
    # accepted-but-unwritten sequences.
    assert [record["seq"] for record in _chunks(writer)] == [1]
    index = registry.read_index("exec-1")
    assert index is not None
    assert index["generations"][0]["last_seq"] == 1
    assert index["generations"][0]["reason"] == sj.REASON_INCOMPLETE
    # The retired writer released the payload it could no longer write.
    assert writer._queue == []


async def test_producer_overflow_marks_journal_incomplete(tmp_path: Path) -> None:
    # Queue cap of 2: the third record overflows and drops, the journal is
    # marked incomplete, and recording stops.
    registry = _registry(tmp_path, queue_max_chunks=2)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=0)
    for seq in range(1, 6):
        writer.record(_chunk(seq))
    await writer.close(REASON_ERROR)

    assert writer.incomplete is True
    assert writer.reason == sj.REASON_INCOMPLETE
    assert [record["seq"] for record in _chunks(writer)] == [1, 2]


async def test_index_reports_generations_before_and_after_close(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.read_index("exec-1") is None

    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=0)
    writer.record(_chunk(1))
    await _wait_for(lambda: writer.last_seq == 1)

    open_index = registry.read_index("exec-1")
    assert open_index == {
        "execution_id": "exec-1",
        "generations": [
            {"generation": 0, "last_seq": 1, "reason": None, "retained": True},
        ],
    }

    await writer.close(REASON_COMPLETED)
    closed_index = registry.read_index("exec-1")
    assert closed_index is not None
    assert closed_index["generations"][0]["reason"] == REASON_COMPLETED
    assert closed_index["generations"][0]["retained"] is True


# ---------------------------------------------------------------------------
# Budget, eviction and retention
# ---------------------------------------------------------------------------


async def test_eviction_marks_body_not_retained(tmp_path: Path) -> None:
    # At most one closed journal retained per session, so the older one is
    # evicted when the second closes.
    registry = _registry(tmp_path, max_closed_per_session=1)

    first = await registry.open_journal(execution_id="exec-1", session_id="s", generation=0)
    first.record(_chunk(1))
    await first.close(REASON_COMPLETED)

    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=0)
    second.record(_chunk(1))
    await second.close(REASON_COMPLETED)

    first_index = registry.read_index("exec-1")
    assert first_index is not None
    assert first_index["generations"][0]["retained"] is False
    assert registry.has_generation("exec-1", 0) is False
    assert not first.path.exists()

    second_index = registry.read_index("exec-2")
    assert second_index is not None
    assert second_index["generations"][0]["retained"] is True


async def test_open_reserves_capacity_so_two_journals_cannot_over_admit(
    tmp_path: Path,
) -> None:
    # The runner budget fits exactly one journal's reservation. Two journals
    # opened before either writes: the first reserves the capacity, the second
    # starts incomplete and records nothing.
    registry = _registry(
        tmp_path,
        journal_max_bytes=1000,
        journal_max_chunks=1000,
        runner_max_bytes=1000 + sj.END_MARKER_RESERVE,
    )
    first = await registry.open_journal(execution_id="exec-1", session_id="s", generation=0)
    assert first.incomplete is False

    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=0)
    assert second.incomplete is True
    second.record(_chunk(1))
    await second.close(REASON_COMPLETED)
    assert _chunks(second) == []


async def test_closed_journal_is_evicted_to_fit_a_new_journal(tmp_path: Path) -> None:
    # A settled journal's body plus one more reservation must not fit, so
    # opening the second journal evicts the first (oldest-first).
    one = _chunk_record_bytes()
    end = _end_record_bytes(last_seq=3)
    capacity = 3 * one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=3 * one,
        journal_max_chunks=1000,
        runner_max_bytes=(3 * one + end) + capacity - 1,
    )

    first = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    for seq in (1, 2, 3):
        first.record(_chunk(seq))
    await first.close(REASON_COMPLETED)

    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=1)
    second.record(_chunk(1))

    first_index = registry.read_index("exec-1")
    assert first_index is not None
    assert first_index["generations"][0]["retained"] is False
    assert second.incomplete is False
    assert not first.path.exists()


async def test_execution_id_cannot_escape_the_journal_root(tmp_path: Path) -> None:
    root = tmp_path / "journals"
    registry = StreamJournalRegistry(root=str(root))
    writer = await registry.open_journal(execution_id="../../escape", session_id="s", generation=1)
    writer.record(_chunk(1))
    await writer.close(REASON_COMPLETED)

    assert writer.path.parent == root
    assert "/" not in writer.path.name
    assert ".." not in writer.path.name
    assert writer.path.exists()
    # Nothing was created outside the journal root.
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path.parent / "escape").exists()


def test_journal_file_name_is_path_safe() -> None:
    for execution_id in ("/tmp/evil", "../evil", "a/b", "a.b"):
        name = sj.journal_file_name(execution_id, 1)
        assert "/" not in name
        assert ".." not in name
    # Distinct ids cannot alias one another.
    assert sj.journal_file_name("a/b", 1) != sj.journal_file_name("a\\b", 1)


# ---------------------------------------------------------------------------
# Draining writers, delayed settlement and reclamation ownership
# ---------------------------------------------------------------------------


async def test_a_draining_journal_keeps_its_reservation_until_it_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=capacity + 1,
        close_drain_timeout_s=0.05,
        admission_timeout_s=0.05,
    )

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._append_bytes

    def gated(self: sj.JournalWriter, data: bytes) -> None:
        entered.set()
        release.wait(5)
        real(self, data)

    monkeypatch.setattr(sj.JournalWriter, "_append_bytes", gated)

    draining = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    draining.record(_chunk(1))
    await _wait_for(entered.is_set)

    # Close times out while the append is held: the journal is incomplete, not
    # finalized, and still holds its whole reservation.
    await draining.close(REASON_COMPLETED)
    assert draining.incomplete is True
    assert draining.finalized is False
    assert registry.runner_bytes == capacity

    # Admission while draining: the draining body cannot be evicted, so the new
    # journal starts incomplete instead of reusing its capacity.
    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=1)
    assert second.incomplete is True
    assert draining.retained is True
    assert registry.runner_bytes == capacity

    # Release the append: the writer settles exactly once, releases the unused
    # reservation, and keeps the readable prefix.
    release.set()
    await _wait_for(lambda: draining.finalized)
    assert registry.runner_bytes == draining.bytes
    assert draining.reserved == draining.bytes
    assert [record["seq"] for record in _chunks(draining)] == [1]
    assert _end(draining) is not None
    assert _end(draining)["reason"] == sj.REASON_INCOMPLETE


@contextlib.asynccontextmanager
async def _delayed_writer_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner_max_bytes: int
) -> AsyncIterator[
    tuple[
        StreamJournalRegistry, sj.JournalWriter, sj.JournalWriter, sj.JournalWriter, threading.Event
    ]
]:
    """Start A (pending), settle B and C, and own the gate until cleanup.

    The gate is released and every writer awaited on exit, so a setup or
    assertion failure cannot leave a held append (or its cleanup) running past
    the test.
    """
    one = _chunk_record_bytes()
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=runner_max_bytes,
        max_closed_per_session=2,
        close_drain_timeout_s=0.5,
    )

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._append_bytes

    def gated(self: sj.JournalWriter, data: bytes) -> None:
        # Only A is gated, so B and C settle normally while A stays pending.
        if self.execution_id == "exec-a":
            entered.set()
            release.wait(5)
        real(self, data)

    monkeypatch.setattr(sj.JournalWriter, "_append_bytes", gated)

    journals: list[sj.JournalWriter] = []
    try:
        a = await registry.open_journal(execution_id="exec-a", session_id="s", generation=1)
        journals.append(a)
        a.record(_chunk(1))
        await _wait_for(entered.is_set)
        await a.close(REASON_COMPLETED)
        assert a.finalized is False

        b = await registry.open_journal(execution_id="exec-b", session_id="s", generation=1)
        journals.append(b)
        b.record(_chunk(1))
        await b.close(REASON_COMPLETED)
        c = await registry.open_journal(execution_id="exec-c", session_id="s", generation=1)
        journals.append(c)
        c.record(_chunk(1))
        await c.close(REASON_COMPLETED)

        # B and C are settled while A is still draining.
        assert b.finalized is True
        assert c.finalized is True
        assert a.finalized is False
        yield registry, a, b, c, release
    finally:
        # Own the cleanup: release the gate, close anything still open, then
        # wait for every writer to finish before the fixture tears down. A
        # writer-less journal has no task to await. Non-settlement is reported
        # explicitly (the original failure stays chained) rather than treated
        # as success.
        release.set()
        for journal in journals:
            if not journal.closed:
                with contextlib.suppress(Exception):
                    await journal.close(REASON_ABORTED)
        unsettled: list[str] = []
        for journal in journals:
            if journal._writer is None:
                continue
            try:
                await asyncio.wait_for(journal._done.wait(), timeout=5)
            except asyncio.TimeoutError:
                unsettled.append(journal.execution_id)
        if unsettled:
            raise RuntimeError(f"stream journal cleanup did not settle: {unsettled}")


async def test_a_delayed_writer_settles_into_creation_order_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity = _chunk_record_bytes() + sj.END_MARKER_RESERVE
    async with _delayed_writer_scenario(tmp_path, monkeypatch, 4 * capacity) as (
        registry,
        a,
        b,
        c,
        release,
    ):
        # A settles last: retention is by creation order, so the two newer
        # bodies (B, C) survive and the oldest (A) is evicted. Wait for the
        # deletion and the released charge, not just the selection flag.
        release.set()
        await _wait_for(lambda: a.reserved == 0)
        assert a.retained is False
        assert b.retained is True
        assert c.retained is True
        assert not a.path.exists()
        assert registry.runner_bytes == sum(p.stat().st_size for p in (b.path, c.path))
        # The surviving tails are still readable.
        assert [record["seq"] for record in _chunks(b)] == [1]
        assert _end(b) is not None and _end(b)["reason"] == REASON_COMPLETED
        assert [record["seq"] for record in _chunks(c)] == [1]
        assert _end(c) is not None and _end(c)["reason"] == REASON_COMPLETED


async def test_retention_counts_only_bodies_still_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity = _chunk_record_bytes() + sj.END_MARKER_RESERVE
    async with _delayed_writer_scenario(tmp_path, monkeypatch, 2 * capacity + 1) as (
        registry,
        a,
        b,
        c,
        release,
    ):
        # C's admission budget-evicted B while A was still draining.
        assert b.retained is False

        # A settling must not evict another body to make room for the
        # already-gone B: the cap counts only bodies still on disk.
        release.set()
        await _wait_for(lambda: a.reserved == a.bytes)
        assert a.retained is True
        assert c.retained is True
        assert registry.runner_bytes == sum(p.stat().st_size for p in (a.path, c.path))
        # A's timed-out close still left its accepted chunk and an incomplete
        # end record; C is a normal completed tail.
        assert [record["seq"] for record in _chunks(a)] == [1]
        assert _end(a) is not None and _end(a)["reason"] == sj.REASON_INCOMPLETE
        assert [record["seq"] for record in _chunks(c)] == [1]
        assert _end(c) is not None and _end(c)["reason"] == REASON_COMPLETED


async def test_a_pending_eviction_keeps_its_charge_until_the_file_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    end = _end_record_bytes(last_seq=1)
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=2 * (one + end) + capacity - 1,
        admission_timeout_s=0.05,
        close_drain_timeout_s=0.5,
    )

    first = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    first.record(_chunk(1))
    await first.close(REASON_COMPLETED)
    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=1)
    second.record(_chunk(1))
    await second.close(REASON_COMPLETED)
    charged = registry.runner_bytes

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._unlink

    def gated(self: sj.JournalWriter) -> bool:
        entered.set()
        release.wait(5)
        return real(self)

    monkeypatch.setattr(sj.JournalWriter, "_unlink", gated)

    # Admission needs room, selects the oldest body, and times out while its
    # removal is held: the new journal starts incomplete and the selected body
    # keeps its charge because its file is still there.
    third = await registry.open_journal(execution_id="exec-3", session_id="s", generation=1)
    assert entered.is_set()
    assert third.incomplete is True
    assert first.retained is False
    assert first.path.exists()
    assert registry.runner_bytes == charged

    release.set()
    await _wait_for(lambda: not first.path.exists())
    await _wait_for(lambda: first.reserved == 0)
    assert registry.runner_bytes == charged - (one + end)


async def test_a_failed_removal_keeps_the_body_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    end = _end_record_bytes(last_seq=1)
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=2 * (one + end) + capacity - 1,
        admission_timeout_s=0.05,
    )

    first = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    first.record(_chunk(1))
    await first.close(REASON_COMPLETED)
    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=1)
    second.record(_chunk(1))
    await second.close(REASON_COMPLETED)
    charged = registry.runner_bytes

    monkeypatch.setattr(sj.JournalWriter, "_unlink", lambda self: False)

    third = await registry.open_journal(execution_id="exec-3", session_id="s", generation=1)
    assert third.incomplete is True
    assert first.path.exists()
    assert registry.runner_bytes == charged


async def test_a_writer_less_close_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(
        tmp_path,
        journal_max_bytes=1,
        runner_max_bytes=0,
        close_drain_timeout_s=0.05,
    )
    journal = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    # The budget fits nothing, so the journal has no writer.
    assert journal.incomplete is True

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._finalize

    async def gated(self: sj.JournalWriter) -> None:
        entered.set()
        await asyncio.to_thread(release.wait, 5)
        await real(self)

    monkeypatch.setattr(sj.JournalWriter, "_finalize", gated)

    await asyncio.wait_for(journal.close(REASON_COMPLETED), timeout=1.0)
    assert entered.is_set()
    assert journal.finalized is False

    release.set()
    await _wait_for(lambda: journal.finalized)


async def test_a_torn_append_keeps_the_body_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    capacity = 10 * one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=10 * one,
        journal_max_chunks=1000,
        runner_max_bytes=2 * capacity - 1,
    )
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)

    def partial(self: sj.JournalWriter, data: bytes) -> None:
        # Write a real prefix before failing: the file is left torn on disk.
        with self.path.open("ab") as fh:
            fh.write(data[: len(data) // 2])
        raise OSError("ENOSPC")

    monkeypatch.setattr(sj.JournalWriter, "_append_bytes", partial)
    writer.record(_chunk(1))
    await writer.close(REASON_ERROR)

    assert writer.incomplete is True
    assert writer.last_seq == 0
    _chunks_, _end_, torn = sj._read_records(writer.path)
    assert torn is True
    assert writer.path.stat().st_size > 0
    # The occupied file stays charged at the captured reservation, so a later
    # admission must reclaim it before it can use that capacity.
    assert registry.runner_bytes == capacity

    second = await registry.open_journal(execution_id="exec-2", session_id="s", generation=1)
    assert second.incomplete is False
    assert writer.reserved == 0
    assert not writer.path.exists()
    assert registry.runner_bytes == capacity


async def test_a_torn_end_marker_keeps_the_body_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    capacity = 10 * one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=10 * one,
        journal_max_chunks=1000,
        runner_max_bytes=2 * capacity - 1,
    )
    writer = await registry.open_journal(execution_id="exec-1", session_id="s", generation=1)
    real = sj.JournalWriter._append_bytes
    calls = {"n": 0}

    def partial_on_end(self: sj.JournalWriter, data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            real(self, data)  # the chunk lands
            return
        with self.path.open("ab") as fh:
            fh.write(data[: len(data) // 2])
        raise OSError("ENOSPC")

    monkeypatch.setattr(sj.JournalWriter, "_append_bytes", partial_on_end)
    writer.record(_chunk(1))
    await writer.close(REASON_COMPLETED)

    assert writer.incomplete is True
    assert writer.last_seq == 1
    chunks, end, torn = sj._read_records(writer.path)
    assert [record["seq"] for record in chunks] == [1]
    assert end is None
    assert torn is True
    # The partial end frame is on disk, so the reservation stays charged.
    assert registry.runner_bytes == capacity


async def test_retention_survives_a_held_unlink_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=20 * capacity,
        max_closed_per_session=2,
        close_drain_timeout_s=0.5,
    )

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._unlink

    def gated(self: sj.JournalWriter) -> bool:
        # Only A's removal is held, so later same-session finalizations finish
        # while C's cleanup is still pending.
        if self.execution_id == "exec-a":
            entered.set()
            release.wait(5)
        return real(self)

    monkeypatch.setattr(sj.JournalWriter, "_unlink", gated)

    async def settle(execution_id: str) -> sj.JournalWriter:
        journal = await registry.open_journal(
            execution_id=execution_id, session_id="s", generation=1
        )
        journal.record(_chunk(1))
        await journal.close(REASON_COMPLETED)
        return journal

    a = await settle("exec-a")
    b = await settle("exec-b")
    c = await settle("exec-c")  # selects A for eviction, whose removal is held
    await _wait_for(entered.is_set)
    d = await settle("exec-d")
    e = await settle("exec-e")
    f = await settle("exec-f")

    try:
        release.set()
        # Await C's completed cleanup: A's file is gone and its charge released.
        await _wait_for(lambda: not a.path.exists())
        await _wait_for(lambda: a.reserved == 0)

        # No more than two bodies survive, and every survivor stays charged
        # exactly for its file.
        survivors = [j for j in (b, c, d, e, f) if j.retained]
        assert len(survivors) == 2
        assert registry.runner_bytes == sum(j.bytes for j in survivors)
        assert sum(j.bytes for j in survivors) == sum(j.path.stat().st_size for j in survivors)

        # Every survivor stays discoverable: admission pressure reclaims one.
        registry.runner_max_bytes = registry.runner_bytes + registry.journal_reservation - 1
        reclaimed = await registry.open_journal(execution_id="exec-g", session_id="s", generation=1)
        assert reclaimed.incomplete is False
        assert sum(1 for j in survivors if j.retained) == 1
    finally:
        release.set()


async def test_duplicate_open_awaits_the_original_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    end = _end_record_bytes(last_seq=1)
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=(one + end) + capacity - 1,
        admission_timeout_s=5.0,
        close_drain_timeout_s=0.5,
    )
    # A settled body the next admission must reclaim.
    prior = await registry.open_journal(execution_id="exec-prior", session_id="s", generation=1)
    prior.record(_chunk(1))
    await prior.close(REASON_COMPLETED)

    entered = threading.Event()
    release = threading.Event()
    real = sj.JournalWriter._unlink

    def gated(self: sj.JournalWriter) -> bool:
        entered.set()
        release.wait(5)
        return real(self)

    monkeypatch.setattr(sj.JournalWriter, "_unlink", gated)

    # Two concurrent opens of the same tuple while the first admission is
    # reclaiming capacity.
    first = asyncio.ensure_future(
        registry.open_journal(execution_id="exec-dup", session_id="s", generation=1)
    )
    await _wait_for(entered.is_set)
    second = asyncio.ensure_future(
        registry.open_journal(execution_id="exec-dup", session_id="s", generation=1)
    )
    # The duplicate must not resolve while reclamation is still held: it awaits
    # the same admission instead of exposing an unadmitted journal.
    await asyncio.sleep(0.05)
    assert not second.done()

    # A cancelled duplicate must not cancel the shared admission.
    cancelled = asyncio.ensure_future(
        registry.open_journal(execution_id="exec-dup", session_id="s", generation=1)
    )
    await asyncio.sleep(0.05)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    release.set()
    a = await first
    b = await second

    # The duplicate awaited the same admission: it did not observe a
    # writer-less, unadmitted journal.
    assert a is b
    assert b._writer is not None
    assert b.incomplete is False
    assert b.reserved == capacity

    # Closing through the duplicate settles the same admitted journal and
    # reconciles its reservation to the file exactly once.
    b.record(_chunk(1))
    await b.close(REASON_COMPLETED)
    assert a.finalized is True
    assert a.retained is True
    assert _end(a) is not None
    assert a.reserved == a.path.stat().st_size
    assert registry.runner_bytes == a.reserved


async def test_a_failed_removal_stays_registered_for_a_later_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = _chunk_record_bytes()
    end = _end_record_bytes(last_seq=1)
    capacity = one + sj.END_MARKER_RESERVE
    registry = _registry(
        tmp_path,
        journal_max_bytes=one,
        journal_max_chunks=1000,
        runner_max_bytes=(one + end) + capacity - 1,
        admission_timeout_s=0.05,
        close_drain_timeout_s=0.5,
    )
    # A: a settled body whose removal will be held and then fail.
    a = await registry.open_journal(execution_id="exec-a", session_id="s", generation=1)
    a.record(_chunk(1))
    await a.close(REASON_COMPLETED)
    charged = registry.runner_bytes

    entered = threading.Event()
    release = threading.Event()
    attempted = {"done": False}
    real = sj.JournalWriter._unlink

    def gated(self: sj.JournalWriter) -> bool:
        if self.execution_id == "exec-a" and not attempted["done"]:
            attempted["done"] = True
            entered.set()
            release.wait(5)
            return False  # the removal fails
        return real(self)

    monkeypatch.setattr(sj.JournalWriter, "_unlink", gated)

    # B's admission selects A for removal and holds it, then starts incomplete.
    b = await registry.open_journal(execution_id="exec-b", session_id="s", generation=1)
    assert entered.is_set()
    assert b.incomplete is True

    # A same-session writer-less close during the hold must not drop A's
    # registration.
    await b.close(REASON_COMPLETED)

    # The held removal now fails: A keeps its charge and stays discoverable.
    release.set()
    await _wait_for(lambda: a.retained is True)
    assert a.path.exists()
    assert registry.runner_bytes == charged

    # Storage recovers: a later admission retries and reclaims A.
    c = await registry.open_journal(execution_id="exec-c", session_id="s", generation=1)
    assert c.incomplete is False
    assert not a.path.exists()
    assert a.reserved == 0


# ---------------------------------------------------------------------------
# Read layer: replay-then-follow, the terminal event, and reader accounting
# ---------------------------------------------------------------------------


async def test_events_replay_from_the_cursor_and_end_cleanly(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=4)
    for seq in range(1, 4):
        writer.record(_chunk(seq, content=f"c{seq}"))
    await writer.close(REASON_COMPLETED)

    events = [event async for event in writer.iter_events(0)]
    assert [event for event, _data in events] == ["chunk", "chunk", "chunk", "end"]
    assert [data["seq"] for event, data in events if event == "chunk"] == [1, 2, 3]
    assert {data["generation"] for _event, data in events} == {4}
    end = events[-1][1]
    assert end["reason"] == REASON_COMPLETED
    assert end["last_seq"] == 3

    # A resume cursor replays only what the client has not seen.
    resumed = [event async for event in writer.iter_events(2)]
    assert [data["seq"] for event, data in resumed if event == "chunk"] == [3]
    assert resumed[-1][0] == "end"


async def test_events_follow_chunks_appended_after_the_reader_started(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    writer.record(_chunk(1, content="a"))

    events: list[tuple[str, dict[str, Any]]] = []
    stream = writer.iter_events(0)
    events.append(await stream.__anext__())

    # Appended while the follower is parked: the wake-up must deliver it, and
    # the close must terminate the stream after the terminal chunk.
    writer.record(_chunk(2, content="b"))
    await writer.close(REASON_COMPLETED)
    async for event in stream:
        events.append(event)

    assert [event for event, _data in events] == ["chunk", "chunk", "end"]
    assert [data["chunk"]["content"] for event, data in events if event == "chunk"] == ["a", "b"]


async def test_incomplete_journal_ends_the_stream_without_waiting_for_the_turn(
    tmp_path: Path,
) -> None:
    # A bound hit means nothing more can be recorded, so a follower must learn
    # the prefix is untrustworthy now rather than hang until the turn ends.
    registry = _registry(tmp_path, journal_max_chunks=1)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    writer.record(_chunk(1))
    writer.record(_chunk(2))
    assert writer.incomplete is True

    events = [event async for event in writer.iter_events(0)]
    assert events[-1][0] == "incomplete"
    assert events[-1][1]["reason"] == sj.REASON_INCOMPLETE
    await writer.close(REASON_COMPLETED)


async def test_torn_end_record_is_reported_incomplete_rather_than_a_clean_end(
    tmp_path: Path,
) -> None:
    # An interrupted end-marker append: the chunks are intact but the journal
    # cannot prove it closed, so the reader must not present it as complete.
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    writer.record(_chunk(1))
    await writer.close(REASON_COMPLETED)
    with writer.path.open("r+b") as fh:
        fh.truncate(writer.path.stat().st_size - 2)

    events = [event async for event in writer.iter_events(0)]
    assert [event for event, _data in events] == ["chunk", "incomplete"]
    assert events[-1][1]["last_seq"] == 1


async def test_unknown_generation_is_reported_unavailable_not_a_clean_stream(
    tmp_path: Path,
) -> None:
    # A body that was never opened (or that eviction removed) must answer
    # `incomplete`, so a reader never mistakes it for a clean, empty stream.
    registry = _registry(tmp_path)
    events = [event async for event in registry.stream("exec-unknown", 1, 0)]
    assert events == [
        ("incomplete", {"generation": 1, "last_seq": 0, "reason": sj.REASON_INCOMPLETE})
    ]


def test_reader_slots_are_capped_per_journal_and_released_idempotently(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, max_readers_per_journal=1, max_readers_per_runner=4)
    lease = registry.acquire_reader("exec-1", 1)
    with pytest.raises(sj.ReaderLimitExceeded):
        registry.acquire_reader("exec-1", 1)

    registry.release_reader(lease)
    # A disconnect path may release twice without stealing another slot.
    registry.release_reader(lease)
    registry.release_reader(registry.acquire_reader("exec-1", 1))


def test_reader_slots_are_capped_across_the_runner(tmp_path: Path) -> None:
    registry = _registry(tmp_path, max_readers_per_journal=4, max_readers_per_runner=1)
    lease = registry.acquire_reader("exec-1", 1)
    with pytest.raises(sj.ReaderLimitExceeded):
        registry.acquire_reader("exec-2", 1)
    registry.release_reader(lease)
    registry.release_reader(registry.acquire_reader("exec-2", 1))


async def test_a_close_that_races_a_read_is_re_read_before_reporting_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A follower's read can complete just after close() set `closed` and just
    # before the end marker reached the file. The reader must re-read the
    # settled file, not report the trusted prefix as untrusted.
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    writer.record(_chunk(1, content="a"))
    await _wait_for(lambda: writer.last_seq == 1)
    await writer.close(REASON_COMPLETED)

    real_read = sj._read_records_from
    reads = {"n": 0}

    def racing_read(path: Path, offset: int) -> Any:
        reads["n"] += 1
        if reads["n"] == 1:
            # Model the change landing mid-read: this read saw the file before
            # the end marker, and the journal moved before it returned.
            writer._bump_change()
            return ([], None, offset, False)
        return real_read(path, offset)

    monkeypatch.setattr(sj, "_read_records_from", racing_read)

    events = [event async for event in writer.iter_events(0)]
    assert reads["n"] >= 2
    assert events[-1][0] == "end", events
    assert events[-1][1]["reason"] == REASON_COMPLETED


async def test_stream_reports_an_evicted_body_as_incomplete_not_empty(
    tmp_path: Path,
) -> None:
    # Evicted between the route's admission check and the follower starting:
    # an empty 200 would read as a clean empty stream.
    registry = _registry(tmp_path)
    writer = await registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    writer.record(_chunk(1))
    await writer.close(REASON_COMPLETED)
    await writer.evict()

    events = [event async for event in registry.stream("exec-1", 1, 0)]
    assert events == [
        ("incomplete", {"generation": 1, "last_seq": 0, "reason": sj.REASON_INCOMPLETE})
    ]


async def test_cancelled_admission_withdraws_the_published_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cancelled execute task must not leave a published journal that nothing
    # will write or close: a reader would follow it forever.
    registry = _registry(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_eviction(need: int) -> None:
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)

    monkeypatch.setattr(registry, "_evict_closed_to_fit_bounded", held_eviction)

    admission = asyncio.ensure_future(
        registry.open_journal(execution_id="exec-1", session_id="s-1", generation=1)
    )
    await asyncio.wait_for(entered.wait(), timeout=5)

    # A journal still being admitted has no writer yet, so a follower must not
    # be admitted onto it: it would have no terminal owner if the admission
    # never completes.
    assert registry.has_generation("exec-1", 1) is False
    early = [event async for event in registry.stream("exec-1", 1, 0)]
    assert early == [
        ("incomplete", {"generation": 1, "last_seq": 0, "reason": sj.REASON_INCOMPLETE})
    ]

    admission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await admission
    release.set()

    assert registry.read_index("exec-1") is None
    assert registry.runner_bytes == 0
    events = [event async for event in registry.stream("exec-1", 1, 0)]
    assert events[-1][0] == "incomplete"
