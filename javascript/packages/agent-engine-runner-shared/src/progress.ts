/**
 * emit / emitStep — send a mid-execution chunk from within a tool function.
 *
 * Tool functions running inside a Tool Pod call these to stream events to the
 * client while the tool is still executing. Both executionId and oeUrl are
 * injected into the execution context by the Tool Pod server before the tool
 * runs (see `server/tool.ts`).
 *
 * The call is best-effort: network errors are logged and swallowed so a failed
 * chunk emit never aborts the tool. Mirrors `agent_engine_runner_shared/progress.py`.
 *
 * Unlike the Python port there is no module-level client to pool: TLS agent
 * reuse and cert-rotation invalidation are already handled by
 * `getFetchOptionsWithTLS` (tls_client.ts), which caches one keep-alive agent
 * per origin — so a plain `fetch` here reuses the same pooled connection the
 * Python `_get_client` machinery provides.
 */

import {
  getCurrentExecutionId,
  getCurrentOeOwnerUrl,
  getCurrentOeUrl,
  reportOeOwnerUrlFailure,
  withExecutionSignal,
} from "./context.js";
import { getLogger } from "./logger.js";
import { ownerDelivered } from "./owner_callback.js";
import {
  DONE,
  ERROR,
  STEP,
  SUBAGENT_END,
  SUBAGENT_START,
  TEXT,
} from "./server/chunk_types.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "./tls_client.js";

const logger = getLogger("agent_engine_runner_shared.progress");

// Short timeout: chunk emits are best-effort fire-and-forget. A slow or
// unreachable OE must not delay the tool function itself.
const PROGRESS_TIMEOUT_SECS = 5.0;

// Chunk types reserved for infrastructure use. Tool code emitting these would
// either close the SSE stream prematurely (DONE/ERROR are terminal) or spoof
// infrastructure-only chunk types (TEXT/SUBAGENT_*).
const RESERVED_EVENTS: ReadonlySet<string> = new Set([
  DONE,
  ERROR,
  TEXT,
  SUBAGENT_START,
  SUBAGENT_END,
]);

/**
 * POST `body` to `ownerUrl` first, falling back to `serviceUrl`.
 *
 * Owner preference per {@link ownerDelivered}: one best-effort attempt against
 * the replica-specific owner URL; any owner failure falls back to the trusted
 * service URL and latches via {@link reportOeOwnerUrlFailure} so later emits
 * in this execution skip the owner instead of re-paying its timeout. Returns
 * `null` when the owner delivered (nothing left to inspect), else the service
 * response. Best-effort: the caller inspects `resp.ok` and swallows failures.
 */
async function postChunkOwnerFirst(
  serviceUrl: string,
  ownerUrl: string | null,
  body: string,
): Promise<Response | null> {
  const init = (url: string): RequestInit => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: withExecutionSignal(
      AbortSignal.timeout(PROGRESS_TIMEOUT_SECS * 1000),
    ),
    ...getFetchOptionsWithTLS(url),
  });
  if (ownerUrl !== null) {
    if (
      await ownerDelivered(
        ownerUrl,
        () => init(ownerUrl),
        (msg) => logger.debug(`emit: ${msg}`),
      )
    ) {
      return null;
    }
    reportOeOwnerUrlFailure();
  }
  return await fetchPlatform(serviceUrl, init(serviceUrl));
}

/**
 * Emit a chunk of the given event type to the client stream.
 *
 * General-purpose API. For the common textual-step case, prefer
 * {@link emitStep}.
 *
 * @param event Chunk type identifier (e.g. `"step"`). Reserved infrastructure
 *   event types (`done`, `error`, `text`, `subagent_start`, `subagent_end`)
 *   reject with an `Error` to prevent tool code from prematurely closing or
 *   spoofing the stream. The reserved-event check fires even outside an
 *   execution context — a programmer error wins over the no-op.
 *   `"custom_event"` is not Atlas Agent Engine custom events; those require
 *   `emit_custom_event` and `features.use_custom_parser` in `agent.yaml`.
 *   Gateway drops `chunk_type: "custom_event"` unless that feature is on.
 * @param data Payload string for the event.
 *
 * Transport failures (including a non-2xx response) are logged at debug and
 * swallowed — best-effort delivery must never abort the tool function. No-op
 * outside an execution context.
 */
export async function emit(event: string, data: string): Promise<void> {
  if (RESERVED_EVENTS.has(event)) {
    throw new Error(
      `emit: event '${event}' is reserved for infrastructure use; ` +
        `choose a non-terminal event type such as 'step'`,
    );
  }

  const executionId = getCurrentExecutionId();
  const oeUrl = getCurrentOeUrl();
  const ownerUrl = getCurrentOeOwnerUrl();
  if (!executionId || !oeUrl) return;

  try {
    const body = JSON.stringify({
      execution_id: executionId,
      chunk_type: event,
      content: data,
      metadata: {},
    });
    // Owner preference: the owner URL is already validated against oeUrl by the
    // tool-pod server, so trust it as-is. Each request gets its own timeout
    // signal combined with the execution-wide abort signal so an owner timeout
    // cannot pre-abort the service fallback.
    const resp = await postChunkOwnerFirst(
      `${oeUrl.replace(/\/$/, "")}/stream/chunk`,
      ownerUrl ? `${ownerUrl.replace(/\/$/, "")}/stream/chunk` : null,
      body,
    );
    if (resp !== null) {
      // A non-2xx is surfaced at debug but not thrown — emit is fire-and-forget.
      if (!resp.ok) {
        logger.debug(`emit: OE returned HTTP ${resp.status} for chunk`);
      }
      // We never read the body; cancel it so undici can release/reuse the
      // connection instead of leaking it under frequent emits.
      await resp.body?.cancel();
    }
  } catch (exc) {
    // Best-effort: a failed chunk must never abort the tool function. Debug so
    // operators can diagnose OE connectivity without noise.
    logger.debug(`emit: failed to send chunk: ${String(exc)}`);
  }
}

/**
 * Emit a textual progress step from within a running tool function.
 *
 * Convenience wrapper around `emit("step", message)`.
 *
 * @example
 * ```ts
 * import { emitStep } from "@mongodb-js/agent-engine-runner-shared";
 *
 * app.tool({ isLocal: false })(async function crawlWebsite(url: string) {
 *   await emitStep("Fetching page...");
 *   const html = await fetch(url);
 *   await emitStep(`Parsing links from ${url}...`);
 *   return parse(html);
 * });
 * ```
 */
export async function emitStep(message: string): Promise<void> {
  await emit(STEP, message);
}
