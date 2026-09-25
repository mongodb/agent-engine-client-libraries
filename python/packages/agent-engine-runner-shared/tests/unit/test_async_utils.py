from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from agent_engine_runner_shared.async_utils import iterate_in_thread


@pytest.mark.asyncio
async def test_iterate_in_thread_yields_items_and_closes_iterator() -> None:
    class ClosableIterator(Iterator[int]):
        def __init__(self) -> None:
            self._values = iter([1, 2])
            self.closed = False

        def __next__(self) -> int:
            return next(self._values)

        def close(self) -> None:
            self.closed = True

    iterator = ClosableIterator()

    assert [item async for item in iterate_in_thread(iterator)] == [1, 2]
    assert iterator.closed is True


@pytest.mark.asyncio
async def test_iterate_in_thread_closes_abandoned_iterator() -> None:
    closed = False

    def values() -> Iterator[int]:
        nonlocal closed
        try:
            yield 1
            yield 2
        finally:
            closed = True

    stream = iterate_in_thread(values())
    async for item in stream:
        assert item == 1
        break
    await stream.aclose()

    assert closed is True


@pytest.mark.asyncio
async def test_iterate_in_thread_propagates_failure_and_closes_once() -> None:
    failure = RuntimeError("stream failed")

    class FailingIterator(Iterator[int]):
        def __init__(self) -> None:
            self.calls = 0
            self.close_calls = 0

        def __next__(self) -> int:
            self.calls += 1
            if self.calls == 1:
                return 1
            raise failure

        def close(self) -> None:
            self.close_calls += 1

    iterator = FailingIterator()
    stream = iterate_in_thread(iterator)

    assert await anext(stream) == 1
    with pytest.raises(RuntimeError) as raised:
        await anext(stream)
    assert raised.value is failure
    assert iterator.close_calls == 1


@pytest.mark.asyncio
async def test_iterate_in_thread_waits_for_active_next_before_closing() -> None:
    class BlockingIterator(Iterator[int]):
        def __init__(self) -> None:
            self.next_started = threading.Event()
            self.release_next = threading.Event()
            self.close_calls = 0
            self.close_overlapped_next = False
            self._next_active = False
            self._iterator = self._values()

        def _values(self) -> Iterator[int]:
            self.next_started.set()
            self.release_next.wait()
            yield 1

        def __next__(self) -> int:
            self._next_active = True
            try:
                return next(self._iterator)
            finally:
                self._next_active = False

        def close(self) -> None:
            self.close_calls += 1
            self.close_overlapped_next = self._next_active
            self._iterator.close()

    iterator = BlockingIterator()
    stream = iterate_in_thread(iterator)
    consumer = asyncio.create_task(anext(stream))
    await asyncio.to_thread(iterator.next_started.wait)

    consumer.cancel()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(consumer), timeout=0.05)
    finally:
        iterator.release_next.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert iterator.close_overlapped_next is False
    assert iterator.close_calls == 1
