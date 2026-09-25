"""Best-effort application graph warming for an AER process.

The workspace warm poke asks OE to create and maintain a paired AER and Tool
standby. OE then calls ``POST /warm-up`` before pausing the executor so the first
request does not pay the graph-construction cost.

AER readiness can precede availability of dependencies used only while building
the application graph. A short retry sequence lets that optional work recover
within OE's bounded warm-up request. Exhaustion is still non-fatal: the normal
lazy request build remains the fallback.
"""

from __future__ import annotations

import asyncio
import logging
from asyncio import sleep
from collections.abc import Callable
from time import monotonic

logger = logging.getLogger(__name__)

# Attempt slots are measured from the start of the sequence. A slow build can
# consume a slot, but it cannot shift every later attempt toward OE's 60-second
# standby warm-up timeout. The final target precedes a separate latest-start
# boundary so normal scheduling jitter remains harmless while every started
# build retains at least 30 seconds beneath OE's timeout.
_GRAPH_WARM_ATTEMPT_OFFSETS_S = (0.0, 5.0, 15.0, 25.0)
_GRAPH_WARM_LATEST_START_S = 30.0


class GraphWarmer:
    """Run one process-scoped graph warm-up sequence on explicit request."""

    def __init__(self, warm_up: Callable[[], bool | None]) -> None:
        self._warm_up = warm_up
        self._task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Start on first request and let concurrent callers share the attempt.

        Shielding keeps a disconnected or timed-out HTTP caller from cancelling
        the task that owns ``asyncio.to_thread``. The caller may stop waiting,
        but Python cannot cancel the graph-building thread itself.
        """
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await asyncio.shield(self._task)

    async def _run(self) -> None:
        scheduled_attempts = len(_GRAPH_WARM_ATTEMPT_OFFSETS_S)
        started_at = monotonic()
        attempts_started = 0

        for scheduled_attempt, offset in enumerate(_GRAPH_WARM_ATTEMPT_OFFSETS_S, start=1):
            if offset:
                delay = started_at + offset - monotonic()
                if delay < 0:
                    logger.info(
                        "Agent graph warm-up skipped scheduled attempt %d/%d; slot elapsed",
                        scheduled_attempt,
                        scheduled_attempts,
                    )
                    continue
                if delay > 0:
                    logger.info(
                        "Agent graph warm-up will retry in %.0fs (scheduled attempt %d/%d)",
                        delay,
                        scheduled_attempt,
                        scheduled_attempts,
                    )
                    await sleep(delay)

            if monotonic() - started_at >= _GRAPH_WARM_LATEST_START_S:
                logger.warning(
                    "Agent graph warm-up stopped after %d attempts; latest start elapsed; "
                    "lazy build remains available",
                    attempts_started,
                )
                return

            attempts_started += 1
            try:
                warmed = await asyncio.to_thread(self._warm_up)
            except Exception:
                warmed = False
                logger.warning(
                    "Agent graph warm-up attempt %d failed",
                    attempts_started,
                    exc_info=True,
                )

            if warmed is not False:
                logger.info("Agent graph warm-up completed on attempt %d", attempts_started)
                return

        logger.warning(
            "Agent graph warm-up exhausted %d attempts across %d scheduled slots; "
            "lazy build remains available",
            attempts_started,
            scheduled_attempts,
        )
