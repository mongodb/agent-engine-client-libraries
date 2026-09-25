/** Process-local executor callback delivery and redelivery. */

import type { ExecutorCallbackRequest } from "../models.js";
import { getLogger } from "../logger.js";
import { postWithRetries, RetryablePostError } from "./http_retry.js";

const logger = getLogger("agent_engine_runner_shared.server.callback_delivery");

const CALLBACK_POST_BASE_DELAY_MS = 100;
const CALLBACK_RETRY_MAX_DELAY_MS = 5_000;

interface PendingCallback {
  oeUrl: string;
  callback: ExecutorCallbackRequest;
}

function callbackKey(callback: ExecutorCallbackRequest): string | null {
  if (callback.status !== "SUSPENDED") {
    return terminalCallbackKey(callback.execution_id);
  }
  // Rolling-upgrade compatibility for OE versions that do not yet send a
  // suspension generation. Remove this bounded legacy path once every
  // supported OE consistently supplies suspend_generation.
  if (callback.suspend_generation === undefined) return null;
  return JSON.stringify([
    "suspension",
    callback.execution_id,
    callback.suspend_generation,
  ]);
}

function terminalCallbackKey(executionId: string): string {
  return JSON.stringify(["terminal", executionId]);
}

function waitForRetry(ms: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(false);
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      resolve(false);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve(true);
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export class CallbackDelivery {
  private pending = new Map<string, PendingCallback>();
  private tasks = new Map<string, Promise<void>>();
  private retryController = new AbortController();

  async send(
    oeUrl: string,
    callback: ExecutorCallbackRequest,
    ownerUrl?: string | null,
    onOwnerFailure?: (() => void) | null,
  ): Promise<void> {
    const key = callbackKey(callback);
    let pending: PendingCallback | null = null;
    if (key !== null) {
      pending = this.reserve(key, oeUrl, callback);
      if (pending === null) return;
    }

    try {
      await postWithRetries(
        `${oeUrl}/executor/callback`,
        callback,
        ownerUrl ? `${ownerUrl}/executor/callback` : null,
        onOwnerFailure,
        undefined,
        undefined,
        undefined,
        key === null
          ? () => !this.pending.has(terminalCallbackKey(callback.execution_id))
          : undefined,
      );
    } catch (error) {
      if (error instanceof RetryablePostError && key !== null) {
        this.startRedelivery(key, callback.execution_id);
        logger.warn(
          { err: error, execution_id: callback.execution_id },
          "Retaining callback after transient delivery failure",
        );
        return;
      }
      logger.error(
        { err: error, execution_id: callback.execution_id },
        key !== null
          ? "Callback failed with a non-retryable error"
          : "Suspension callback failed; not retaining because suspend_generation is absent",
      );
    }

    if (pending !== null && key !== null && this.pending.get(key) === pending) {
      this.pending.delete(key);
    }
  }

  async shutdown(): Promise<void> {
    this.retryController.abort();
    await Promise.allSettled(this.tasks.values());
    this.tasks.clear();
    this.pending.clear();
  }

  async waitForTerminal(executionId: string): Promise<boolean> {
    const key = terminalCallbackKey(executionId);
    const task = this.tasks.get(key);
    if (task !== undefined) await task;
    return !this.retryController.signal.aborted && !this.pending.has(key);
  }

  private reserve(
    key: string,
    oeUrl: string,
    callback: ExecutorCallbackRequest,
  ): PendingCallback | null {
    const executionId = callback.execution_id;
    if (this.retryController.signal.aborted) {
      logger.error(
        { execution_id: executionId },
        "Cannot retain callback during shutdown",
      );
      return null;
    }

    const pending = { oeUrl, callback };
    const existing = this.pending.get(key);
    if (existing !== undefined) {
      if (
        existing.oeUrl !== oeUrl ||
        JSON.stringify(existing.callback) !== JSON.stringify(callback)
      ) {
        logger.error(
          { execution_id: executionId },
          "Refusing to replace conflicting callback",
        );
      }
      return null;
    }

    this.pending.set(key, pending);
    return pending;
  }

  private startRedelivery(key: string, executionId: string): void {
    if (!this.tasks.has(key)) {
      const task = this.redeliver(
        key,
        executionId,
        this.retryController.signal,
      ).finally(() => {
        this.tasks.delete(key);
      });
      this.tasks.set(key, task);
    }
  }

  private async redeliver(
    key: string,
    executionId: string,
    signal: AbortSignal,
  ): Promise<void> {
    let delayMs = CALLBACK_POST_BASE_DELAY_MS;
    while (!signal.aborted) {
      if (!(await waitForRetry(delayMs, signal))) return;
      const pending = this.pending.get(key);
      if (pending === undefined) return;
      try {
        await postWithRetries(
          `${pending.oeUrl}/executor/callback`,
          pending.callback,
          null,
          null,
          signal,
        );
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof RetryablePostError) {
          delayMs = Math.min(delayMs * 2, CALLBACK_RETRY_MAX_DELAY_MS);
          logger.warn(
            { err: error, execution_id: executionId },
            "Callback redelivery remains pending",
          );
          continue;
        }
        logger.error(
          { err: error, execution_id: executionId },
          "Dropping callback after a non-retryable redelivery error",
        );
        if (this.pending.get(key) === pending) this.pending.delete(key);
        return;
      }

      if (this.pending.get(key) === pending) this.pending.delete(key);
      return;
    }
  }
}
