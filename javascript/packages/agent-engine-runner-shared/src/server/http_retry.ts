/**
 * Shared retry-policy and bounded POST helpers for OE transports.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.http_retry`. The AER chunk/callback
 * path (`server/aer.ts`) and the tool result settlement loop
 * (`secure_wrapper.ts`) share policy here so their cooperative back-pressure
 * behaviour stays identical.
 */

import { getLogger } from "../logger.js";
import { ownerDelivered as ownerPreAttemptDelivered } from "../owner_callback.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "../tls_client.js";
import { getRequestTimeout } from "../utils.js";

const logger = getLogger("agent_engine_runner_shared.server.http_retry");

const REQUEST_TIMEOUT_MS = getRequestTimeout() * 1000;
const DEFAULT_POST_ATTEMPTS = 3;
const DEFAULT_POST_BASE_DELAY_MS = 100;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

export function waitForRetry(
  ms: number,
  signal: AbortSignal,
): Promise<boolean> {
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

export class RetryablePostError extends Error {}

/**
 * Raised when a deadline-bounded retry loop runs out of budget before any
 * attempt succeeded. The caller must treat the payload as not delivered.
 */
export class RetryBudgetExhausted extends Error {}

// Ceiling on an honored `Retry-After` wait: a cooperative 503 hint is worth a
// short pause, but the runner must not park a callback for an OE-chosen
// duration.
export const RETRY_AFTER_MAX_WAIT_MS = 10_000;

/** Release an unread fetch body without changing callback delivery semantics. */
export async function discardResponseBody(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // The response outcome is authoritative; cleanup failure must not replace it.
  }
}

/**
 * Parse a delta-seconds `Retry-After` header; ignore the HTTP-date form.
 *
 * Returns the non-negative integer seconds, or `null` when the header is
 * absent or not in delta-seconds form (an HTTP-date, or garbage). Callers fall
 * back to their normal backoff when this returns `null`.
 */
export function parseRetryAfterSeconds(
  headerValue: string | null | undefined,
): number | null {
  if (headerValue == null) return null;
  const value = headerValue.trim();
  // Only ASCII digits (matches Python's `str.isdigit()`): no sign, decimal, or
  // HTTP-date. An empty string fails, so a blank header falls through.
  if (!/^\d+$/.test(value)) return null;
  return Number(value);
}

/**
 * The wait (in ms) an honored delta-seconds `Retry-After` implies, capped at
 * {@link RETRY_AFTER_MAX_WAIT_MS}, or `null` when the header is not honorable.
 */
export function retryAfterWaitMs(
  headerValue: string | null | undefined,
): number | null {
  const seconds = parseRetryAfterSeconds(headerValue);
  if (seconds === null) return null;
  return Math.min(seconds * 1000, RETRY_AFTER_MAX_WAIT_MS);
}

/**
 * POST to OE with one optional owner attempt and bounded service retries.
 *
 * Transport failures and 5xx responses are retried; 4xx responses fail
 * immediately. An owner URL gets one best-effort pre-attempt without consuming
 * the service retry budget. A 503 `Retry-After` delta overrides the normal
 * exponential delay up to the configured cap.
 */
export async function postWithRetries(
  url: string,
  body: unknown,
  ownerUrl?: string | null,
  onOwnerFailure?: (() => void) | null,
  cancellationSignal?: AbortSignal,
  maxAttempts = DEFAULT_POST_ATTEMPTS,
  baseDelayMs = DEFAULT_POST_BASE_DELAY_MS,
  shouldAttempt?: () => boolean,
): Promise<void> {
  const serialized = JSON.stringify(body);
  if (shouldAttempt?.() === false) return;
  if (ownerUrl != null) {
    if (await ownerDelivered(ownerUrl, url, serialized)) return;
    onOwnerFailure?.();
  }

  let lastError: unknown = null;
  const tlsOptions = getFetchOptionsWithTLS(url);
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (shouldAttempt?.() === false) return;
    let retryWaitMs: number | null = null;
    let response: Response | undefined;
    try {
      const requestSignal = cancellationSignal
        ? AbortSignal.any([
            cancellationSignal,
            AbortSignal.timeout(REQUEST_TIMEOUT_MS),
          ])
        : AbortSignal.timeout(REQUEST_TIMEOUT_MS);
      response = await fetchPlatform(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serialized,
        signal: requestSignal,
        ...tlsOptions,
      });
    } catch (error) {
      if (cancellationSignal?.aborted) throw error;
      lastError = new RetryablePostError(
        error instanceof Error ? error.message : String(error),
      );
    }
    if (response) {
      const ok = response.ok;
      const status = response.status;
      const retryAfter = response.headers?.get?.("Retry-After") ?? null;
      await discardResponseBody(response);
      if (ok) return;
      if (status < 500) throw new Error(`HTTP ${status} from ${url}`);
      lastError = new RetryablePostError(`HTTP ${status} from ${url}`);
      if (status === 503) retryWaitMs = retryAfterWaitMs(retryAfter);
    }
    if (attempt < maxAttempts - 1) {
      const delayMs = retryWaitMs ?? baseDelayMs * 2 ** attempt;
      if (cancellationSignal) {
        if (!(await waitForRetry(delayMs, cancellationSignal))) {
          throw cancellationSignal.reason;
        }
      } else {
        await sleep(delayMs);
      }
    }
  }
  throw (
    lastError ??
    new Error(
      `postWithRetries exhausted ${maxAttempts} attempts but recorded no error`,
    )
  );
}

async function ownerDelivered(
  ownerUrl: string,
  serviceUrl: string,
  serialized: string,
  cancellationSignal?: AbortSignal,
): Promise<boolean> {
  return ownerPreAttemptDelivered(
    ownerUrl,
    () => ({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: serialized,
      signal: cancellationSignal
        ? AbortSignal.any([
            cancellationSignal,
            AbortSignal.timeout(REQUEST_TIMEOUT_MS),
          ])
        : AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      ...getFetchOptionsWithTLS(ownerUrl),
    }),
    (message) => logger.warn(`${message} URL ${serviceUrl}`),
  );
}

/** Options for {@link postWithRetriesUntil}. */
export interface PostUntilOptions {
  /**
   * Monotonic deadline (`performance.now()` based). Retryable failures are
   * retried until it passes; a deadline already in the past fails immediately.
   */
  deadlineMs: number;
  maxDelayMs: number;
  baseDelayMs?: number;
  ownerUrl?: string | null;
  onOwnerFailure?: (() => void) | null;
  /** Aborts an in-flight attempt and the backoff wait (used by sender close). */
  cancellationSignal?: AbortSignal;
}

/**
 * POST to OE, retrying retryable failures until a deadline.
 *
 * Mirrors Python's `post_json_with_retries_until`. Same owner-first policy and
 * retry classification as {@link postWithRetries} — the difference is the
 * budget: instead of a bounded attempt count, transport failures and 5xx
 * responses are retried until `deadlineMs` passes, so a caller whose payload
 * must not be dropped (the ordered durable chunk stream) can ride out a long
 * outage. A 4xx response fails immediately (a caller bug, not a transient
 * condition). Throws the last retryable failure, or {@link RetryBudgetExhausted}
 * when the deadline passed before an attempt succeeded.
 */
export async function postWithRetriesUntil(
  url: string,
  body: unknown,
  options: PostUntilOptions,
): Promise<void> {
  const {
    deadlineMs,
    maxDelayMs,
    baseDelayMs = DEFAULT_POST_BASE_DELAY_MS,
    ownerUrl,
    onOwnerFailure,
    cancellationSignal,
  } = options;
  const serialized = JSON.stringify(body);
  if (ownerUrl != null) {
    if (await ownerDelivered(ownerUrl, url, serialized, cancellationSignal))
      return;
    // A cancelled execution must not start the service fallback: surface the
    // cancellation reason instead of posting again.
    if (cancellationSignal?.aborted) throw cancellationSignal.reason;
    onOwnerFailure?.();
  }

  let backoffMs = baseDelayMs;
  const tlsOptions = getFetchOptionsWithTLS(url);
  while (true) {
    const remainingMs = deadlineMs - performance.now();
    if (remainingMs <= 0) {
      throw new RetryBudgetExhausted(
        `retry budget exhausted before ${url} acknowledged the payload`,
      );
    }
    let retryWaitMs: number | null = null;
    let response: Response | undefined;
    try {
      const requestSignal = cancellationSignal
        ? AbortSignal.any([
            cancellationSignal,
            AbortSignal.timeout(REQUEST_TIMEOUT_MS),
          ])
        : AbortSignal.timeout(REQUEST_TIMEOUT_MS);
      response = await fetchPlatform(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serialized,
        signal: requestSignal,
        ...tlsOptions,
      });
    } catch (error) {
      if (cancellationSignal?.aborted) throw error;
      // Transport failure: retried below unless the deadline has passed.
    }
    if (response) {
      const ok = response.ok;
      const status = response.status;
      const retryAfter = response.headers?.get?.("Retry-After") ?? null;
      await discardResponseBody(response);
      if (ok) return;
      if (status < 500) throw new Error(`HTTP ${status} from ${url}`);
      if (status === 503) retryWaitMs = retryAfterWaitMs(retryAfter);
    }
    // Doubling a running cap (rather than `base * 2 ** attempt`) keeps the
    // exponent from overflowing on a long outage, and never sleeps past the
    // deadline: the loop re-checks it on wake.
    const waitMs = Math.min(
      retryWaitMs ?? backoffMs,
      Math.max(0, deadlineMs - performance.now()),
    );
    backoffMs = Math.min(backoffMs * 2, maxDelayMs);
    if (waitMs > 0) {
      if (cancellationSignal) {
        if (!(await waitForRetry(waitMs, cancellationSignal))) {
          throw cancellationSignal.reason;
        }
      } else {
        await sleep(waitMs);
      }
    }
  }
}
