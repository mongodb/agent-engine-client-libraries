"""Small async bridges shared by framework adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import TypeVar, cast

T = TypeVar("T")


async def iterate_in_thread(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Consume a blocking iterator without blocking the event loop."""
    sentinel = object()
    try:
        while True:
            next_task = asyncio.create_task(asyncio.to_thread(next, iterator, sentinel))
            try:
                item = await asyncio.shield(next_task)
            except asyncio.CancelledError:
                # Cancellation cannot stop the worker. Wait before closing so
                # iterator methods never run concurrently.
                while not next_task.done():
                    try:
                        await asyncio.shield(next_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not next_task.cancelled():
                    next_task.exception()
                raise
            if item is sentinel:
                return
            yield cast(T, item)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
