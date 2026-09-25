"""Process-local executor callback delivery and redelivery."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, NamedTuple, Optional

import httpx

from agent_engine_runner_shared.server.http_retry import (
    is_retryable_callback_error,
    post_json_with_retries,
)

logger = logging.getLogger(__name__)

CALLBACK_POST_ATTEMPTS = 3
CALLBACK_POST_BASE_DELAY_S = 0.1
CALLBACK_RETRY_MAX_DELAY_S = 5.0


class _PendingCallback(NamedTuple):
    oe_url: str
    body: dict[str, Any]


class _CallbackKey(NamedTuple):
    execution_id: str
    kind: str
    suspend_generation: Optional[int] = None


def _callback_key(callback_body: dict[str, Any]) -> Optional[_CallbackKey]:
    execution_id = str(callback_body["execution_id"])
    if callback_body.get("status") != "SUSPENDED":
        return _CallbackKey(execution_id, "terminal")

    suspend_generation = callback_body.get("suspend_generation")
    if suspend_generation is None:
        # Rolling-upgrade compatibility for OE versions that do not yet send a
        # suspension generation. Remove this bounded legacy path once every
        # supported OE consistently supplies suspend_generation.
        return None
    return _CallbackKey(execution_id, "suspension", int(suspend_generation))


class CallbackDelivery:
    """Retain idempotent executor callbacks until OE acknowledges them."""

    def __init__(
        self,
        get_client: Callable[[str], Awaitable[httpx.AsyncClient]],
    ) -> None:
        self._get_client = get_client
        self._pending: dict[_CallbackKey, _PendingCallback] = {}
        self._tasks: dict[_CallbackKey, asyncio.Task[None]] = {}
        self._stopping = False

    async def send(
        self,
        oe_url: str,
        callback_body: dict[str, Any],
        *,
        owner_url: Optional[str] = None,
        on_owner_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        """Deliver a callback and retain retryable failures when replay is safe."""
        client = await self._get_client(oe_url)
        execution_id = str(callback_body.get("execution_id", "") or "")
        callback_key = _callback_key(callback_body)
        pending: Optional[_PendingCallback] = None
        if callback_key is not None:
            pending = self._reserve(callback_key, oe_url, callback_body)
            if pending is None:
                return

        retain_pending = False
        try:
            await post_json_with_retries(
                client,
                f"{oe_url}/executor/callback",
                callback_body,
                max_attempts=CALLBACK_POST_ATTEMPTS,
                base_delay=CALLBACK_POST_BASE_DELAY_S,
                owner_url=f"{owner_url}/executor/callback" if owner_url else None,
                on_owner_failure=on_owner_failure,
                is_retryable=(is_retryable_callback_error if callback_key is not None else None),
                should_attempt=(
                    None
                    if callback_key is not None
                    else lambda: _CallbackKey(execution_id, "terminal") not in self._pending
                ),
            )
        except Exception as exc:
            if callback_key is not None and is_retryable_callback_error(exc):
                self._start_redelivery(callback_key)
                retain_pending = True
                logger.warning(
                    "Retaining callback for execution %s after transient delivery failure",
                    execution_id,
                    exc_info=True,
                )
                return
            if callback_key is not None:
                logger.error(
                    "Callback for execution %s failed with a non-retryable error",
                    execution_id,
                    exc_info=True,
                )
            else:
                logger.error(
                    "Suspension callback for execution %s failed; not retaining because "
                    "suspend_generation is absent",
                    execution_id,
                    exc_info=True,
                )
        finally:
            if (
                not retain_pending
                and pending is not None
                and callback_key is not None
                and self._pending.get(callback_key) is pending
            ):
                self._pending.pop(callback_key, None)

    async def shutdown(self) -> None:
        """Cancel retry work before the AER closes its HTTP clients."""
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._pending.clear()

    async def wait_for_terminal(self, execution_id: str) -> bool:
        """Wait for retained terminal delivery before session release."""
        callback_key = _CallbackKey(execution_id, "terminal")
        task = self._tasks.get(callback_key)
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if self._stopping:
                    return False
                raise
        return not self._stopping and callback_key not in self._pending

    def _reserve(
        self,
        callback_key: _CallbackKey,
        oe_url: str,
        callback_body: dict[str, Any],
    ) -> Optional[_PendingCallback]:
        execution_id = callback_key.execution_id
        if self._stopping:
            logger.error(
                "Cannot retain callback for execution %s during shutdown",
                execution_id,
            )
            return None

        pending = _PendingCallback(oe_url=oe_url, body=callback_body)
        existing = self._pending.get(callback_key)
        if existing is not None:
            if existing != pending:
                logger.error(
                    "Refusing to replace conflicting callback for execution %s",
                    execution_id,
                )
            return None

        self._pending[callback_key] = pending
        return pending

    def _start_redelivery(self, callback_key: _CallbackKey) -> None:
        task = self._tasks.get(callback_key)
        if task is None or task.done():
            self._tasks[callback_key] = asyncio.create_task(
                self._redeliver(callback_key),
                name=(
                    f"callback-redelivery-{callback_key.execution_id}-"
                    f"{callback_key.kind}-{callback_key.suspend_generation}"
                ),
            )

    async def _post(self, oe_url: str, callback_body: dict[str, Any]) -> None:
        client = await self._get_client(oe_url)
        await post_json_with_retries(
            client,
            f"{oe_url}/executor/callback",
            callback_body,
            max_attempts=CALLBACK_POST_ATTEMPTS,
            base_delay=CALLBACK_POST_BASE_DELAY_S,
            is_retryable=is_retryable_callback_error,
        )

    async def _redeliver(self, callback_key: _CallbackKey) -> None:
        execution_id = callback_key.execution_id
        delay = CALLBACK_POST_BASE_DELAY_S
        try:
            while not self._stopping:
                await asyncio.sleep(delay)
                pending = self._pending.get(callback_key)
                if pending is None:
                    return
                try:
                    await self._post(pending.oe_url, pending.body)
                except Exception as exc:
                    if is_retryable_callback_error(exc):
                        delay = min(delay * 2, CALLBACK_RETRY_MAX_DELAY_S)
                        logger.warning(
                            "Callback redelivery for execution %s remains pending",
                            execution_id,
                            exc_info=True,
                        )
                        continue
                    logger.error(
                        "Dropping callback for execution %s after a non-retryable redelivery error",
                        execution_id,
                        exc_info=True,
                    )
                    if self._pending.get(callback_key) == pending:
                        self._pending.pop(callback_key, None)
                    return

                if self._pending.get(callback_key) == pending:
                    self._pending.pop(callback_key, None)
                return
        finally:
            if self._tasks.get(callback_key) is asyncio.current_task():
                self._tasks.pop(callback_key, None)
