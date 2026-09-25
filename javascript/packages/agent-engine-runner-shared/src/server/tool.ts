/**
 * Tool Executor Pod server — executes tool functions in isolated pods.
 *
 * Responsibilities:
 * - Execute registered tool functions
 * - Invoke LLMs routed from AER via SecureWrappedLLM
 * - Return results to OE
 *
 * Mirrors Python's `ToolServer` in `server/tool.py`.
 */

import os from "node:os";
import { once } from "node:events";
import type { ServerResponse } from "node:http";
import type { FastifyInstance } from "fastify";
import type { JsonValue, LLMStreamChunk } from "@mongodb-js/agent-engine-sdk";
import type { LLMTokenUsage } from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "../logger.js";
import { withMetrics } from "../metrics.js";
import {
  getCurrentExecutionMetadata,
  getRequestedSuspend,
  runWithCustomerOrigin,
  runWithExecutionContext,
} from "../context.js";
import {
  getLLMAdapterFactory,
  getNamedLlm,
  populateLlmRegistryFromEntrypoint,
} from "../hooks.js";
import {
  isConfiguredMcpSdkToolName,
  makeMcpToolCallable,
  MCPConfigError,
  resolveConfiguredMcpToolBinding,
} from "../mcp_tools.js";
import {
  GuardrailCheckRequestSchema,
  LLMPodInvokeRequestSchema,
  LLMPodInvokeResponseSchema,
  LLMPodStreamEventSchema,
  mergeTokenUsage,
  normalizeLLMPodInvokeResponse,
  ToolPodExecuteRequestSchema,
  type GuardrailCheckRequest,
  type GuardrailCheckResponse,
  type LLMPodInvokeRequest,
  type LLMPodInvokeResponse,
  type LLMPodStreamEvent,
  type ToolPodExecuteRequest,
  type ToolPodExecuteResponse,
  type ToolsListResponse,
} from "../models.js";
import { redactText } from "../error_reporting.js";
import { evaluateGuardrailCheck } from "../guardrails_evaluator/index.js";
import { SecureLLMProxy } from "../secure_llm_proxy.js";
import { closeAllTLSAgents } from "../tls_client.js";
import {
  executeErrorResponse,
  requestCredentialValues,
  requestNamedCredentialValues,
} from "../tool_api_error.js";
import {
  withMetadataEnv,
  withMetadataEnvGen,
  mergeMetadataEnv,
} from "./metadata.js";
import {
  LLM_BACKOFF_MULTIPLIER,
  LLM_INITIAL_BACKOFF,
  LLM_MAX_BACKOFF,
  LLM_MAX_RETRIES,
  formatLlmError,
  isLlmCredentialRejection,
  isRetryableError,
  logToolRequest,
  normalizeToolCallArgs,
  toolRedactFields,
} from "../utils.js";
import { BaseServer, type ITenantRuntime, type ServerToolFn } from "./base.js";
import { LLM_CREDENTIAL_REJECTED_ERROR_CODE } from "./chunk_types.js";
import {
  BUILTIN_TOOL_NAMES,
  registerBuiltinTools,
} from "../toolpod_handlers.js";
import { resolveOeUrl } from "./oe_url.js";
import { resolveOwnerUrl } from "./owner_url.js";

const logger = getLogger("agent_engine_runner_shared.server.tool");

/**
 * Max time to wait for a slow SSE client to drain before aborting the stream.
 * Prevents unbounded buffering and long-lived holds on the credential gate
 * when a peer never reads.
 */
export const SSE_DRAIN_TIMEOUT_MS = 30_000;

/**
 * Pipe SSE `data:` lines to a hijacked Node response with disconnect cancel,
 * backpressure, and no process-crashing unhandled `error` events.
 *
 * Fastify's `reply.hijack()` removes framework lifecycle management; this
 * helper owns the raw socket for the rest of the stream.
 *
 * Disconnect is detected via the *response* `close` event when the peer drops
 * before `end()` — not via `IncomingMessage` `close`, which also fires after a
 * normally-consumed POST body and would abort a valid stream.
 *
 * `lines` may be an AsyncIterable or a factory that receives the pipe's
 * AbortSignal so upstream generation (retries/backoff) can stop promptly.
 *
 * Force-close paths (peer drop, drain timeout, socket error) call
 * `res.destroy()` rather than `res.end()`: end only queues FIN behind data
 * already stuck in the writable buffer, which never drains for a never-reading
 * peer and leaks the socket/FD indefinitely.
 */
export async function pipeSseLinesToResponse(
  res: ServerResponse,
  lines:
    | AsyncIterable<string>
    | ((signal: AbortSignal) => AsyncIterable<string>),
  opts: {
    drainTimeoutMs?: number;
    signal?: AbortSignal;
    // Fired exactly once when the generator has truly finished: synchronously
    // on the graceful path, or whenever the deferred return() settles after a
    // force-close — possibly never for a generator parked on an await that
    // ignores the signal. The drain receiver keys its work release off this:
    // socket teardown must not masquerade as quiescence.
    onGeneratorSettled?: () => void;
  } = {},
): Promise<void> {
  const drainTimeoutMs = opts.drainTimeoutMs ?? SSE_DRAIN_TIMEOUT_MS;
  const ac = new AbortController();
  // `ac` carries the pipe's own lifecycle (peer drop, socket error, drain
  // timeout); an external signal (e.g. an execution drain) combines in so
  // either source unblocks a parked iterator.next() — including when the
  // upstream adapter ignores the signal it was handed.
  const signal = opts.signal
    ? combineAbortSignals(ac.signal, opts.signal)
    : ac.signal;
  let iterator: AsyncIterator<string>;
  try {
    const iterable = typeof lines === "function" ? lines(signal) : lines;
    iterator = iterable[Symbol.asyncIterator]();
  } catch (err) {
    // Nothing ever started, so nothing is pending: settle immediately.
    opts.onGeneratorSettled?.();
    throw err;
  }

  const onResError = (err: Error): void => {
    logger.warn(`SSE response error: ${err.message}`);
    if (!ac.signal.aborted) ac.abort();
  };
  const onResClose = (): void => {
    // Normal completion also emits `close` after `end()`; only abort when the
    // peer dropped the connection before we finished writing.
    if (!res.writableEnded && !ac.signal.aborted) {
      ac.abort();
    }
  };

  // Without this listener, write/end on a destroyed socket becomes an
  // unhandled stream 'error' and terminates the Node process.
  res.on("error", onResError);
  res.on("close", onResClose);

  // Graceful end only when the iterable finishes without abort/timeout.
  let forceClose = false;

  try {
    while (true) {
      if (signal.aborted || res.destroyed || res.writableEnded) {
        forceClose = true;
        break;
      }

      // Race next() against abort so a hung provider / backoff sleep cannot
      // keep the credential gate locked after the peer is gone.
      const stepped = await nextOrAbort(iterator, signal);
      if (stepped === "aborted") {
        forceClose = true;
        break;
      }
      if (stepped.done) break;

      const line = stepped.value;
      if (signal.aborted || res.destroyed || res.writableEnded) {
        forceClose = true;
        break;
      }
      const ok = res.write(line);
      if (!ok) {
        const drained = await waitForDrainOrAbort(res, signal, drainTimeoutMs);
        if (!drained || signal.aborted || res.destroyed || res.writableEnded) {
          forceClose = true;
          if (!ac.signal.aborted) ac.abort();
          break;
        }
      }
    }
  } catch (err) {
    forceClose = true;
    if (!ac.signal.aborted) {
      logger.warn(
        `SSE stream failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  } finally {
    // Stop aborting on the post-end `close`; keep `error` attached through
    // close/destroy and the following turn so async EPIPE/hangup is still
    // handled.
    res.off("close", onResClose);
    try {
      if (forceClose) {
        // A generator parked at an unresolved await (e.g. an adapter ignoring
        // the abort signal) never settles return(); teardown must not wait on
        // it. Fire-and-forget — and notify settlement only when the generator
        // actually unwinds, so a drain cannot report completed while provider
        // work is still pending; it times out honestly instead.
        Promise.resolve(iterator.return?.())
          .catch(() => {
            /* ignore */
          })
          .finally(() => opts.onGeneratorSettled?.());
      } else {
        await iterator.return?.();
        opts.onGeneratorSettled?.();
      }
    } catch (err) {
      logger.warn(
        `SSE iterator return failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      // A thrown return() means the generator finished with that error.
      opts.onGeneratorSettled?.();
    }
    if (!res.writableEnded && !res.destroyed) {
      try {
        if (forceClose) {
          res.destroy();
        } else {
          res.end();
        }
      } catch (err) {
        logger.warn(
          `SSE response ${forceClose ? "destroy" : "end"} failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
    setImmediate(() => {
      res.off("error", onResError);
    });
  }
}

/**
 * Pull the next iterator value, or resolve `"aborted"` if `signal` fires
 * first. Does not cancel the in-flight `next()`; callers should still
 * `return()` the iterator in `finally` to unwind generators.
 */
function nextOrAbort<T>(
  iterator: AsyncIterator<T>,
  signal: AbortSignal,
): Promise<IteratorResult<T> | "aborted"> {
  if (signal.aborted) return Promise.resolve("aborted");
  return new Promise((resolve, reject) => {
    const onAbort = (): void => {
      cleanup();
      resolve("aborted");
    };
    const cleanup = (): void => {
      signal.removeEventListener("abort", onAbort);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    iterator.next().then(
      (result) => {
        cleanup();
        // Abort may have won the race and already resolved; ignore late next.
        if (signal.aborted) return;
        resolve(result);
      },
      (err: unknown) => {
        cleanup();
        if (signal.aborted) return;
        reject(err);
      },
    );
  });
}

/**
 * Abort the returned controller when either source fires. Local
 * `AbortSignal.any` — both signals stay independent of each other otherwise.
 */
function combineAbortSignals(a: AbortSignal, b: AbortSignal): AbortSignal {
  const combined = new AbortController();
  if (a.aborted || b.aborted) {
    combined.abort();
    return combined.signal;
  }
  const abort = (): void => combined.abort();
  a.addEventListener("abort", abort, { once: true });
  b.addEventListener("abort", abort, { once: true });
  return combined.signal;
}

/**
 * Wait until the response drains, the abort signal fires, or the drain
 * timeout elapses. Returns true only when drain won.
 */
function waitForDrainOrAbort(
  res: ServerResponse,
  signal: AbortSignal,
  timeoutMs: number,
): Promise<boolean> {
  if (signal.aborted || res.destroyed || res.writableEnded) {
    return Promise.resolve(false);
  }
  return once(res, "drain", {
    signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
  }).then(
    () => true,
    () => false,
  );
}

function isAbortError(err: unknown): boolean {
  return (
    (err instanceof Error && err.name === "AbortError") ||
    (typeof DOMException !== "undefined" &&
      err instanceof DOMException &&
      err.name === "AbortError")
  );
}

/** Abortable sleep used by LLM stream retry backoff. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("This operation was aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new DOMException("This operation was aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Promise-chain mutex serializing async work within one ToolServer instance.
 * Mirrors Python's asyncio.Lock on ToolServer._execute_gate.
 * Exported for tests that build a ToolServer without its constructor.
 */
export class AsyncMutex {
  private locked: Promise<void> = Promise.resolve();

  private async acquire(): Promise<() => void> {
    const previous = this.locked;
    let release!: () => void;
    this.locked = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    return release;
  }

  async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Generator form of runExclusive: the lock is held for the lifetime of the
   * iteration and released when it completes, throws, or is closed early.
   */
  async *runExclusiveGen<T>(
    genFactory: () => AsyncGenerator<T>,
  ): AsyncGenerator<T> {
    const release = await this.acquire();
    try {
      yield* genFactory();
    } finally {
      release();
    }
  }
}

/**
 * Normalize adapter stream output into an LLMStreamChunk shape.
 *
 * Mirrors Python's `_coerce_stream_chunk`. The framework adapter
 * (`agent-engine-sdk-langgraph-ts`) translates raw LangChain chunks into the
 * camelCase sdk-core `LLMStreamChunk` contract before yielding (see its
 * `chunkToLlmStreamChunk`), so this reads camelCase keys only. Python reads
 * snake_case because it inspects raw LangChain objects directly; the TS
 * adapter pre-translates, so no snake_case ever reaches here. Always emits
 * camelCase.
 */
function coerceStreamChunk(chunk: unknown): LLMStreamChunk {
  if (chunk !== null && typeof chunk === "object" && !Array.isArray(chunk)) {
    const obj = chunk as Record<string, unknown>;

    // Prefer streaming tool-call chunks, then fall back to merged tool calls.
    const rawChunks = (obj["toolCallChunks"] as unknown[] | undefined) ?? [];
    let toolCalls: LLMStreamChunk["toolCalls"] =
      rawChunks.length > 0
        ? rawChunks.flatMap((tc, index) => {
            if (typeof tc !== "object" || tc === null) return [];
            const t = tc as Record<string, unknown>;
            return [
              {
                id: t["id"] as string | undefined,
                name: t["name"] as string | undefined,
                args: normalizeToolCallArgs(t["args"]) ?? undefined,
                type: t["type"] as string | undefined,
                index: (t["index"] as number | undefined) ?? index,
              },
            ];
          })
        : undefined;

    if (!toolCalls) {
      const rawCalls = (obj["toolCalls"] as unknown[] | undefined) ?? [];
      const built = rawCalls.flatMap((tc, index) => {
        if (typeof tc !== "object" || tc === null) return [];
        const t = tc as Record<string, unknown>;
        return [
          {
            id: t["id"] as string | undefined,
            name: t["name"] as string | undefined,
            args:
              normalizeToolCallArgs(t["args"] ?? t["arguments"]) ?? undefined,
            type: t["type"] as string | undefined,
            index: (t["index"] as number | undefined) ?? index,
          },
        ];
      });
      toolCalls = built.length > 0 ? built : undefined;
    }

    return {
      content: typeof obj["content"] === "string" ? obj["content"] : undefined,
      toolCalls,
      usage: obj["usage"] as LLMTokenUsage | undefined,
      id: obj["id"] as string | undefined,
      name: obj["name"] as string | undefined,
      responseMetadata: obj["responseMetadata"] as
        | Record<string, JsonValue>
        | undefined,
      additionalKwargs: obj["additionalKwargs"] as
        | Record<string, JsonValue>
        | undefined,
    };
  }

  return {};
}

function streamChunkHasPayload(streamChunk: LLMStreamChunk): boolean {
  return Boolean(
    streamChunk.content ||
    streamChunk.toolCalls ||
    streamChunk.id !== undefined ||
    streamChunk.name !== undefined ||
    streamChunk.responseMetadata !== undefined ||
    streamChunk.additionalKwargs !== undefined,
  );
}

/** BaseLLM-like interface expected from the registered adapter factory. */
interface AdaptedLLM {
  stream(
    messages: unknown[],
    kwargs?: Record<string, unknown>,
  ): AsyncIterable<unknown>;
}

// Bound on the entrypoint-derived text echoed to the caller. The text comes
// from the agent's own code, which makes it useful, but it is unbounded
// user-authored input on a caller-facing path.
const MAX_ENTRYPOINT_ERROR_TEXT = 512;

/**
 * Stands for "the entrypoint load has not failed" in `llmRegistryLoadError`.
 *
 * A unique symbol rather than `null`: JavaScript permits `throw null`, and a
 * null initializer would make that failure indistinguishable from a successful
 * load, so its bare registration error would be reported instead of the real
 * cause. No user throwable can equal this value.
 */
const NO_RECORDED_LOAD_ERROR = Symbol("noRecordedLlmRegistryLoadError");

/**
 * Error class name, for parity with Python's `type(error).__name__`.
 *
 * `error.name` is inherited as "Error" unless a subclass assigns `this.name`,
 * so an unmarked `class ProviderError extends Error {}` would render as
 * "Error" and lose the specificity the Python implementation reports for the
 * same agent code. Prefer an explicit `this.name` override when one is set,
 * otherwise fall back to the constructor's class name.
 */
function thrownName(error: Error): string {
  const explicit = error.name;
  if (explicit && explicit !== "Error") return explicit;
  return error.constructor?.name || explicit || "Error";
}

/**
 * Render a thrown value for display. JavaScript permits throwing anything, so
 * this must not assume an Error: `null` becomes "null" rather than the
 * misleading "object: null" that `typeof` would produce.
 */
function describeThrown(error: unknown): string {
  if (error instanceof Error) return `${thrownName(error)}: ${error.message}`;
  return String(error);
}

/**
 * Render an entrypoint failure for a caller-facing error, redacted.
 *
 * The cause is the agent's own code failing, which is what makes it worth
 * surfacing at all; but its text can echo a connection string, an API key, or a
 * credential-bearing URL. Two layers, both used for caller-facing text in this
 * package: the shared pattern redactor (URL userinfo, bearer tokens, inline
 * secrets), then exact replacement of the credential values a name marks as
 * secrets (`requestNamedCredentialValues`).
 *
 * Both choices are load-bearing:
 *
 * - Patterns run on the pristine text. Replacing values verbatim first can
 *   destroy the `://` the userinfo patterns anchor on, leaving the credential
 *   in place.
 * - Candidates are selected by variable *name*, not by value length. Tenant env
 *   is mostly not secret and includes single-character values ("1", "/"), and a
 *   length threshold cannot tell those from a short token -- it drops the token
 *   too. Selecting by name keeps "/" out (never redacted) while redacting a
 *   7-character secret (always redacted).
 *
 * Deliberately renders `name: message` rather than `formatLlmError`: that
 * helper unpacks a provider error's `.details`/`.body` wholesale, which is the
 * highest-risk payload to echo, and the error name plus message already
 * carries the actionable part. Mirrors Python's _safe_entrypoint_error_text.
 */
function safeEntrypointErrorText(error: unknown): string {
  let text = redactText(describeThrown(error));
  const values = requestNamedCredentialValues().sort(
    (a, b) => b.length - a.length,
  );
  for (const value of values) {
    text = text.split(value).join("<redacted>");
  }
  if (text.length > MAX_ENTRYPOINT_ERROR_TEXT) {
    text = text.slice(0, MAX_ENTRYPOINT_ERROR_TEXT - 1) + "…";
  }
  return text;
}

/**
 * Wire code for a provider credential rejection, else undefined.
 *
 * Deliberately narrow: only a rejection authenticated by the provider's own
 * HTTP status earns the code, so a rate limit, outage, or agent-side bug
 * keeps its existing generic classification instead of being relabeled as a
 * customer-secret problem. Mirrors Python's _llm_failure_error_code.
 */
function llmFailureErrorCode(error: unknown): string | undefined {
  return isLlmCredentialRejection(error)
    ? LLM_CREDENTIAL_REJECTED_ERROR_CODE
    : undefined;
}

/**
 * Thrown when an LLM lookup fails because the entrypoint registry load failed.
 *
 * The named-LLM registry is populated by running the user entrypoint. When that
 * run throws, the registry is left empty (or holding only import-time
 * registrations) and every lookup of an LLM the entrypoint would have
 * registered fails with a bare "not registered" error -- which points the agent
 * developer at their `app.llm()` calls instead of at the transient cause that
 * actually broke the load. The real cause is named in the message only; it is
 * deliberately not attached as `cause`, because the caller-facing response is
 * rendered by `formatLlmError`, which walks `cause` and would dump a raw
 * provider `.details`/`.body` from there, bypassing the message redaction.
 * Mirrors Python's LLMRegistryLoadError.
 */
export class LLMRegistryLoadError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "LLMRegistryLoadError";
  }
}

export class ToolServer extends BaseServer {
  // Serialize the credential-bearing endpoints (/execute, /invoke_llm,
  // /invoke_llm/stream): applyMetadataEnv mutates process-global process.env,
  // so per-call delegated credentials and request context must never
  // overlap. Restriction is off, so the gate is skipped.
  private readonly executeGate = new AsyncMutex();
  private readonly restrictionDisabled: boolean;
  // Guards one-time registry loading across startup and request paths.
  private readonly llmRegistryLock = new AsyncMutex();
  private llmRegistryLoaded = false;
  // Why the last entrypoint run failed, or NO_RECORDED_LOAD_ERROR after a
  // successful load. Read by createLlmForPod to replace a downstream "not
  // registered" throw with the real cause. Mirrors Python's
  // _llm_registry_load_error, except that the sentinel is distinct from every
  // throwable value: a JavaScript entrypoint may legally `throw null`, and a
  // null initializer would make that failure indistinguishable from success.
  private llmRegistryLoadError: unknown = NO_RECORDED_LOAD_ERROR;

  constructor(runtime: ITenantRuntime) {
    super(runtime);
    this.restrictionDisabled = true;
  }

  /**
   * Run `fn` under restricted or unrestricted secret handling.
   *
   * Restricted (default): take `executeGate` and apply/restore metadata env.
   * Unrestricted: merge metadata without restore and skip the gate.
   */
  private async withCredentialIsolation<T>(fn: () => Promise<T>): Promise<T> {
    if (this.restrictionDisabled) {
      mergeMetadataEnv();
      return fn();
    }
    return this.executeGate.runExclusive(() => withMetadataEnv(fn));
  }

  /**
   * Streaming variant of `withCredentialIsolation` for `/invoke_llm/stream`.
   */
  private async *withCredentialIsolationGen<T>(
    genFactory: () => AsyncGenerator<T>,
  ): AsyncGenerator<T> {
    if (this.restrictionDisabled) {
      mergeMetadataEnv();
      yield* genFactory();
      return;
    }
    yield* this.executeGate.runExclusiveGen(() =>
      withMetadataEnvGen(genFactory),
    );
  }

  get modeName(): string {
    return "tool";
  }

  async onStartup(): Promise<void> {
    // `features.deep_agent` gates the built-in filesystem + shell handlers.
    // Tenants that don't opt in never get a file/shell attack surface
    // registered on their Tool Pod, so the platform stays agent-type-agnostic
    // and tenants pay only for what they use.
    const deepAgentEnabled = this.runtime
      .getAgentConfig()
      .featureEnabled("deep_agent", false);
    if (deepAgentEnabled) {
      registerBuiltinTools(this.runtime);
      const missing = [...BUILTIN_TOOL_NAMES].filter(
        (name) => this.runtime.tools[name] === undefined,
      );
      if (missing.length > 0) {
        throw new Error(
          `ToolServer startup: missing built-in tool handlers: ${JSON.stringify(missing.sort())}. ` +
            `Expected all of ${JSON.stringify([...BUILTIN_TOOL_NAMES].sort())} to be registered by registerBuiltinTools().`,
        );
      }
      logger.info(
        `Deep-agent built-in tools registered (features.deep_agent=true): ${JSON.stringify([...BUILTIN_TOOL_NAMES].sort())}`,
      );
    } else {
      logger.info(
        "Deep-agent built-in tools NOT registered (features.deep_agent is off in agent.yaml).",
      );
    }

    const tools = Object.keys(this.runtime.tools);
    logger.info(`Registered tools: ${JSON.stringify(tools)}`);

    // Populate the registry while the guest warms, before the first model request.
    await this.ensureLlmRegistryLoaded();
  }

  /**
   * Populate the registry at startup, retrying failed construction on requests.
   * The lock ensures subsequent calls reuse successful preparation.
   */
  private async ensureLlmRegistryLoaded(): Promise<void> {
    if (this.llmRegistryLoaded) return;
    await this.llmRegistryLock.runExclusive(async () => {
      if (this.llmRegistryLoaded) return;
      let loadError: unknown = NO_RECORDED_LOAD_ERROR;
      this.llmRegistryLoaded = populateLlmRegistryFromEntrypoint(
        this.runtime.graphBuilder,
        logger,
        {
          entrypointFailed:
            "Failed to run entrypoint for LLM registration; " +
            "/invoke_llm will fail if no LLM was registered",
          noLlmRegistered:
            "Entrypoint ran but did not call app.llm(); " +
            "/invoke_llm will fail until an LLM is registered",
        },
        (error) => {
          loadError = error;
        },
      );
      // A successful load clears any cause recorded by an earlier failure, so
      // a later lookup miss is reported as a genuine registration error. The
      // sink only fires on failure, so a success leaves the sentinel in place.
      this.llmRegistryLoadError = loadError;
    });
  }

  async onShutdown(): Promise<void> {
    // Close all cached TLS agents for graceful shutdown (parity with Python)
    closeAllTLSAgents();
  }

  getHealthDetails(): Record<string, unknown> {
    return {
      tools: Object.keys(this.runtime.tools),
      tool_count: Object.keys(this.runtime.tools).length,
    };
  }

  registerRoutes(app: FastifyInstance): void {
    app.post("/execute", async (request, reply) => {
      const body = ToolPodExecuteRequestSchema.parse(request.body);
      // Tool functions receive no AbortSignal (ServerToolFn takes only args),
      // so a drain can only wait this work out: count-only registration, and
      // outliving the deadline is an honest timed_out. Throws 409 if the
      // execution was drained after OE dispatched. The step number makes the
      // call addressable by the per-call interrupt even though tool functions
      // have no signal channel (it reports not_cancellable honestly).
      this.drainRegistry.beginWork(
        body.execution_id,
        undefined,
        body.step_number,
      );
      try {
        const result = await this.handleExecute(body);
        return reply.send(result);
      } finally {
        this.drainRegistry.endWork(
          body.execution_id,
          undefined,
          body.step_number,
        );
      }
    });

    app.post("/invoke_llm", async (request, reply) => {
      const body = LLMPodInvokeRequestSchema.parse(request.body);
      const drainController = new AbortController();
      this.drainRegistry.beginWork(
        body.execution_id,
        drainController,
        body.step_number,
      );
      try {
        const result = await this.handleInvokeLlm(body, drainController.signal);
        // Mirror Python's `_sync_usage_with_result` model validator: ensure
        // `result.usage` is populated (OE reads `result["usage"]`), not just the
        // top-level `usage`.
        return reply.send(
          normalizeLLMPodInvokeResponse(
            LLMPodInvokeResponseSchema.parse(result),
          ),
        );
      } finally {
        this.drainRegistry.endWork(
          body.execution_id,
          drainController,
          body.step_number,
        );
      }
    });

    app.post("/invoke_llm/stream", async (request, reply) => {
      const body = LLMPodInvokeRequestSchema.parse(request.body);
      // Checked here so a drained execution gets a 409 rather than an SSE
      // error event after the stream has already started.
      this.drainRegistry.checkAdmission(body.execution_id);
      reply.hijack();
      const res = reply.raw;
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();
      const drainController = new AbortController();
      this.drainRegistry.beginWork(
        body.execution_id,
        drainController,
        body.step_number,
      );
      // The registration is released when the generator has truly settled,
      // not when the socket closes: a signal-ignoring generator can outlive
      // force-close, and a drain must never report completed while provider
      // work is still pending — it times out honestly instead (mirrors the
      // Python twin, whose end_work runs only when the task unwinds).
      //
      // Hijack removes Fastify lifecycle management; pipe owns disconnect
      // cancel, backpressure, and safe end so a dropped client cannot crash
      // the shared tool pod. The drain signal rides
      // the pipe's combined signal, so a drain unblocks the iterator even
      // when the LLM adapter ignores the signal it was handed.
      await pipeSseLinesToResponse(
        res,
        (signal) => this.handleInvokeLlmStream(body, signal),
        {
          signal: drainController.signal,
          onGeneratorSettled: () =>
            this.drainRegistry.endWork(
              body.execution_id,
              drainController,
              body.step_number,
            ),
        },
      );
    });

    app.get(
      "/tools",
      async (): Promise<ToolsListResponse> => ({
        tools: Object.entries(this.runtime.toolDefinitions).map(
          ([name, defn]) => ({ name, ...defn }),
        ),
        count: Object.keys(this.runtime.toolDefinitions).length,
      }),
    );

    app.post("/guardrails/check", async (request, reply) => {
      const body = GuardrailCheckRequestSchema.parse(request.body);
      // Count-only registration: the evaluation is synchronous and fast, so
      // there is no abort channel to signal; a drain waits it out.
      this.drainRegistry.beginWork(body.execution_id);
      try {
        return reply.send(await this.handleGuardrailsCheck(body));
      } finally {
        this.drainRegistry.endWork(body.execution_id);
      }
    });
  }

  private handleGuardrailsCheck = withMetrics(
    "tool_pod_guardrails_check",
    (request: GuardrailCheckRequest): GuardrailCheckResponse =>
      evaluateGuardrailCheck(request),
    { recordErrors: false },
  );

  private handleExecute = withMetrics(
    "tool_pod_execute",
    (request: ToolPodExecuteRequest) => this.doHandleExecute(request),
    { recordErrors: false },
  );

  private async doHandleExecute(
    request: ToolPodExecuteRequest,
  ): Promise<ToolPodExecuteResponse> {
    const podName = os.hostname();

    logToolRequest(
      request.tool_name ?? "(empty)",
      request.arguments,
      0,
      "TOOL_POD",
      toolRedactFields(this.runtime.toolDefinitions, request.tool_name),
    );

    if (!request.tool_name) {
      return {
        status: "error",
        error: "Tool name cannot be empty",
        pod_name: podName,
        metadata: {},
      };
    }

    const resolved = this.resolveTool(request.tool_name, request.metadata);
    if ("error" in resolved) {
      return {
        status: "error",
        error: resolved.error,
        pod_name: podName,
        metadata: {},
      };
    }
    const func = resolved.func;

    // Pin the callback base to the runner's own configured OE (deploy-time
    // OE_URL) before it is used as the callback base or the owner-URL trust
    // anchor. The request field is only honoured when no OE_URL is stamped
    // (local dev / tests). Resolving once here means the owner URL is validated
    // against a trusted anchor, not the raw request value.
    const oeUrl = resolveOeUrl(request.oe_url ?? "");

    return this.withCredentialIsolation(() =>
      runWithExecutionContext(
        {
          executionId: request.execution_id,
          traceId: request.platform_trace_id,
          wrapper: null,
          oeUrl,
          // Emit() from the tool prefers this owner replica, validated against
          // the trusted resolved oeUrl above.
          oeOwnerUrl: resolveOwnerUrl(request.oe_owner_url, oeUrl),
          userId: request.user_id,
          sessionId: request.session_id,
          authorization: request.authorization,
          customHeaders: request.custom_headers,
          payload: request.payload,
        },
        async () => {
          // Sync tools run on the event loop (no asyncio.to_thread equivalent
          // in TS). This diverges from Python which off-loads sync functions to
          // a thread pool.
          try {
            const maybePromise = runWithCustomerOrigin(() =>
              func(request.arguments),
            );
            const result =
              maybePromise instanceof Promise
                ? await maybePromise
                : maybePromise;
            const metadata = getCurrentExecutionMetadata();
            // Author-intended suspend is signaled out of band via
            // suspendPayloadToJson, never inferred from result content — so
            // relayed untrusted data cannot forge a HITL suspend.
            const suspendMarker = getRequestedSuspend();
            if (suspendMarker !== null) {
              return {
                status: "suspend",
                result: JSON.stringify(suspendMarker),
                pod_name: podName,
                kind: metadata["memory"] ? "memory" : undefined,
                metadata,
                oob_suspend_supported: true,
              };
            }
            return {
              status: "success",
              result,
              pod_name: podName,
              kind: metadata["memory"] ? "memory" : undefined,
              metadata,
              oob_suspend_supported: true,
            };
          } catch (e) {
            return executeErrorResponse(
              e,
              this.runtime.toolDefinitions[request.tool_name],
              podName,
              getCurrentExecutionMetadata(),
              requestCredentialValues(),
            );
          }
        },
      ),
    );
  }

  /**
   * Resolve a tool name to a callable, falling back to a configured MCP tool.
   *
   * Mirrors Python's `ToolServer._resolve_tool()`. AER discovers remote MCP
   * schemas at startup and forwards `mcp_server`/`mcp_tool` call metadata
   * when invoking a Tool-Pod-dispatched MCP tool; the direct registry lookup
   * misses for those names because MCP tools aren't pre-registered on the
   * Tool Pod side (it skips `tools/list` startup discovery).
   */
  private resolveTool(
    toolName: string,
    metadata: Record<string, unknown>,
  ): { func: ServerToolFn } | { error: string } {
    const direct = this.runtime.tools[toolName];
    if (direct) return { func: direct };

    const mcpConfig = this.runtime.getAgentConfig().mcp;
    const mcpServerName = metadata["mcp_server"];
    const mcpToolName = metadata["mcp_tool"];
    if (typeof mcpServerName === "string" && typeof mcpToolName === "string") {
      try {
        const binding = resolveConfiguredMcpToolBinding(
          mcpConfig,
          toolName,
          mcpServerName,
          mcpToolName,
        );
        if (binding === null) {
          return { error: `Unknown tool: ${toolName}` };
        }
        const callable = makeMcpToolCallable(binding);
        return { func: (args: Record<string, unknown>) => callable(args) };
      } catch (err) {
        if (err instanceof MCPConfigError) return { error: err.message };
        throw err;
      }
    }

    if (isConfiguredMcpSdkToolName(mcpConfig, toolName)) {
      return {
        error:
          `Tool '${toolName}' is a configured MCP tool but the request did ` +
          "not include mcp_server/mcp_tool call metadata",
      };
    }

    return { error: `Unknown tool: ${toolName}` };
  }

  /**
   * Create a BaseLLM-compatible instance for tool pod execution.
   *
   * Retrieves the named LLM from the registry (registered via
   * `app.llm(llm, { llmId })`) and wraps it through the registered adapter
   * factory. No SecureWrappedLLM needed — OE approval already happened in AER.
   */
  private createLlmForPod(
    llmId: string,
    tools?: unknown[],
    toolChoice?: unknown,
  ): AdaptedLLM {
    let llm: unknown;
    try {
      llm = getNamedLlm(llmId);
    } catch (error) {
      const loadError = this.llmRegistryLoadError;
      if (loadError === NO_RECORDED_LOAD_ERROR) throw error;
      // The lookup miss is a symptom of a failed entrypoint load, so reporting
      // it as a missing app.llm() call would misdirect the agent developer.
      // Surface the real cause instead -- redacted, since this text is returned
      // to the caller. This branch is reachable only for a lookup that already
      // failed, so it can never turn a working call into a failure -- in
      // particular the import-time-snapshot path still resolves an llmId the
      // snapshot holds even when the entrypoint run threw.
      //
      // No `cause`: the caller-facing error is rendered by formatLlmError,
      // which walks `error.cause` and would dump a provider error's raw
      // `.details`/`.body` from there, bypassing the redaction in the message.
      throw new LLMRegistryLoadError(
        `llm_id ${JSON.stringify(llmId)} is not registered because the agent ` +
          `entrypoint failed when the LLM registry was loaded. The next LLM ` +
          `call re-runs the entrypoint, so this recovers once the underlying ` +
          `cause clears. Underlying failure: ${safeEntrypointErrorText(loadError)}`,
      );
    }
    // tool_choice forces a specific bound tool (e.g. the schema bound by
    // withStructuredOutput); forward it so the adapter's bindTools call can
    // translate it. OE approval already happened in AER.
    return getLLMAdapterFactory()(llm, {
      tools,
      tool_choice: toolChoice,
    }) as AdaptedLLM;
  }

  private handleInvokeLlm = withMetrics(
    "llm_pod_invoke",
    (request: LLMPodInvokeRequest, signal?: AbortSignal) =>
      this.doHandleInvokeLlm(request, signal),
    { recordErrors: false },
  );

  private async doHandleInvokeLlm(
    request: LLMPodInvokeRequest,
    signal?: AbortSignal,
  ): Promise<LLMPodInvokeResponse> {
    const podName = os.hostname();
    const startTime = performance.now();

    try {
      // Usually a no-op after startup; keep any registry load inside the
      // request's credential isolation, as with model execution below.
      const chunks = await runWithExecutionContext(
        {
          executionId: request.execution_id ?? "",
          wrapper: null,
          oeUrl: "",
          traceId: request.platform_trace_id,
        },
        () =>
          this.withCredentialIsolation(async () => {
            await this.ensureLlmRegistryLoaded();
            const collected: LLMStreamChunk[] = [];
            for await (const chunk of this.streamLlmChunks(request, signal)) {
              collected.push(chunk);
            }
            return collected;
          }),
      );
      const llmResponse = SecureLLMProxy.responseFromStreamChunks(chunks);
      const durationMs = performance.now() - startTime;
      return {
        status: "success",
        result: {
          content: llmResponse.content,
          tool_calls:
            (llmResponse.toolCalls?.length ?? 0) > 0
              ? llmResponse.toolCalls
              : undefined,
          metadata: llmResponse.metadata,
          id: llmResponse.id,
          name: llmResponse.name,
          additional_kwargs: llmResponse.additionalKwargs,
          response_metadata: llmResponse.responseMetadata,
        } as LLMPodInvokeResponse["result"],
        pod_name: podName,
        duration_ms: durationMs,
        usage: llmResponse.usage,
      };
    } catch (e) {
      const durationMs = performance.now() - startTime;
      return {
        status: "error",
        error: formatLlmError(e),
        error_code: llmFailureErrorCode(e),
        pod_name: podName,
        duration_ms: durationMs,
      };
    }
  }

  /**
   * ensureLlmRegistryLoaded() then streamLlmChunks(), both inside the
   * request's credential isolation window. Registry loading is normally a
   * no-op after startup; the gate still covers the complete stream.
   */
  private async *loadRegistryThenStreamLlmChunks(
    request: LLMPodInvokeRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<LLMStreamChunk> {
    yield* this.registryLoadedLlmChunks(request, signal);
  }

  private async *registryLoadedLlmChunks(
    request: LLMPodInvokeRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<LLMStreamChunk> {
    await this.ensureLlmRegistryLoaded();
    yield* this.streamLlmChunks(request, signal);
  }

  private async *streamLlmChunks(
    request: LLMPodInvokeRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<LLMStreamChunk> {
    const invokeArgs = request.arguments;
    const llm = this.createLlmForPod(
      invokeArgs.llm_id,
      invokeArgs.tools as unknown[],
      invokeArgs.tool_choice,
    );
    const messages = invokeArgs.messages;
    const extraKwargs: Record<string, unknown> =
      invokeArgs.options?.toModelKwargs() ?? {};
    // `stop` is the wire key the LLM adapter expects (Python Pydantic serialization alias).
    extraKwargs["stop"] = invokeArgs.stop_sequences;
    // Forward abort so LangChain adapters that honor `signal` can cancel the
    // in-flight provider request (best-effort; not all adapters support it).
    if (signal !== undefined) {
      extraKwargs["signal"] = signal;
    }

    let lastError: unknown = null;
    let streamStarted = false;

    for (let attempt = 0; attempt < LLM_MAX_RETRIES; attempt++) {
      let usage: LLMTokenUsage | undefined;
      if (signal?.aborted) {
        throw new DOMException("This operation was aborted", "AbortError");
      }
      try {
        const stream = llm.stream(messages, extraKwargs);
        for await (const chunk of stream) {
          if (signal?.aborted) {
            throw new DOMException("This operation was aborted", "AbortError");
          }
          const streamChunk = coerceStreamChunk(chunk);
          if (streamChunkHasPayload(streamChunk)) {
            streamStarted = true;
            yield streamChunk;
          }
          if (streamChunk.usage !== undefined) {
            usage = mergeTokenUsage(usage, streamChunk.usage) ?? usage;
          }
        }

        if (usage !== undefined) {
          yield { usage };
        }
        return;
      } catch (e) {
        if (isAbortError(e) || signal?.aborted) {
          throw isAbortError(e)
            ? e
            : new DOMException("This operation was aborted", "AbortError");
        }
        lastError = e;
        if (
          !streamStarted &&
          isRetryableError(e) &&
          attempt < LLM_MAX_RETRIES - 1
        ) {
          const backoff = Math.min(
            LLM_INITIAL_BACKOFF * Math.pow(LLM_BACKOFF_MULTIPLIER, attempt),
            LLM_MAX_BACKOFF,
          );
          logger.warn(
            `LLM Pod Stream: Transient provider failure (attempt ${attempt + 1}/${LLM_MAX_RETRIES}). ` +
              `Retrying in ${backoff.toFixed(1)}s...`,
          );
          await sleep(backoff * 1000, signal);
          continue;
        }
        break;
      }
    }

    if (lastError !== null) throw lastError;
    throw new Error("Failed to create LLM stream");
  }

  private async *handleInvokeLlmStream(
    request: LLMPodInvokeRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<string> {
    const podName = os.hostname();
    const startTime = performance.now();
    const ctx = {
      executionId: request.execution_id ?? "",
      wrapper: null,
      oeUrl: "",
      traceId: request.platform_trace_id,
    };
    const gen = this.invokeLlmStreamEvents(request, signal, podName, startTime);
    try {
      while (true) {
        const step = await runWithExecutionContext(ctx, () => gen.next());
        if (step.done) {
          return;
        }
        yield step.value;
      }
    } finally {
      await gen.return(undefined);
    }
  }

  private async *invokeLlmStreamEvents(
    request: LLMPodInvokeRequest,
    signal: AbortSignal | undefined,
    podName: string,
    startTime: number,
  ): AsyncGenerator<string> {
    try {
      let usage: LLMTokenUsage | undefined;
      // See the same-gate rationale in doHandleInvokeLlm above.
      for await (const streamChunk of this.withCredentialIsolationGen(() =>
        this.loadRegistryThenStreamLlmChunks(request, signal),
      )) {
        if (!streamChunkHasPayload(streamChunk)) {
          if (streamChunk.usage !== undefined) {
            usage = streamChunk.usage;
          }
          continue;
        }
        const chunkEvent: LLMPodStreamEvent = {
          content: streamChunk.content ?? undefined,
          tool_call_chunks: streamChunk.toolCalls ?? undefined,
          id: streamChunk.id,
          name: streamChunk.name,
          response_metadata: streamChunk.responseMetadata as
            | Record<string, JsonValue>
            | undefined,
          additional_kwargs: streamChunk.additionalKwargs as
            | Record<string, JsonValue>
            | undefined,
        };
        yield `data: ${JSON.stringify(LLMPodStreamEventSchema.parse(chunkEvent))}\n\n`;
      }

      const durationMs = performance.now() - startTime;
      const summary: LLMPodStreamEvent = {
        done: true,
        pod_name: podName,
        duration_ms: durationMs,
        usage,
      };
      yield `data: ${JSON.stringify(LLMPodStreamEventSchema.parse(summary))}\n\n`;
    } catch (e) {
      // Abort is expected on peer drop / drain timeout — no error frame.
      if (isAbortError(e) || signal?.aborted) {
        return;
      }
      // Log even when the SSE consumer is already gone: otherwise adapter
      // teardown failures after disconnect are invisible server-side.
      logger.warn(
        `LLM stream failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      const errorEvent: LLMPodStreamEvent = {
        error: formatLlmError(e),
        error_code: llmFailureErrorCode(e),
      };
      yield `data: ${JSON.stringify(LLMPodStreamEventSchema.parse(errorEvent))}\n\n`;
    }
  }
}
