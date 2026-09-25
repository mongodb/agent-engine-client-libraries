/**
 * SecureToolWrapper — routes all tool calls through OE for logging and policy enforcement.
 *
 * Provides:
 *   SecureToolWrapper: wraps tool execution calls to route through OE
 *   createSecureToolFunction: wraps a tool function with runtime context lookup
 *   requestOeApproval / reportOeResult: low-level OE HTTP helpers (exported for SecureLLMProxy)
 *   PolicyDeniedException / ToolExecutionError / LLMInvocationError: exception classes
 */

import { isDeepStrictEqual } from "node:util";

import { getLogger } from "./logger.js";
import { ownerDelivered } from "./owner_callback.js";
import {
  SuspendPayloadSchema,
  ToolExecuteResponseSchema,
  coerceTokenUsage,
  type GuardrailMeta,
  type ToolExecuteResponse,
  type ToolResultRequest,
} from "./models.js";
import {
  classifyToolAPIError,
  requestCredentialValues,
  ToolAPIErrorSchema,
  type ToolAPIError,
} from "./tool_api_error.js";
import {
  getRequestTimeout,
  getToolReadTimeout,
  logToolRequest,
  logToolResult,
  logCachedResult,
  logPolicyBlocked,
  OE_RETRYABLE_MAX_ATTEMPTS,
} from "./utils.js";
import {
  getCurrentExecutionId,
  getCurrentOeOwnerUrl,
  getCurrentUserId,
  getCurrentWrapper,
  getRequestedSuspend,
  runWithCustomerOrigin,
  reportOeOwnerUrlFailure,
  runWithSuspendRequestContext,
  withExecutionSignal,
} from "./context.js";
import { getSuspendHandler } from "./hooks.js";
import {
  fetchPlatform,
  getFetchOptionsForLongCall,
  getFetchOptionsWithTLS,
} from "./tls_client.js";
import {
  discardResponseBody,
  retryAfterWaitMs,
  waitForRetry,
} from "./server/http_retry.js";
import { SpanStatusCode } from "@opentelemetry/api";
import { getCurrentTraceContext, getTracer } from "./tracing/setup.js";
import { runWithToolMemoryReadOwnership } from "./tool_memory_ownership.js";
import {
  ActivityKind,
  DurableActivityDeniedError,
  DurableActivityInterrupted,
  type DurableMemoryState,
  ReplayedActivityFailedError,
  WorkflowClient,
  activityRequiresReconstruction,
  allocateActivityOrdinal,
  currentAttemptContext,
  runSerialActivity,
  toolActivityKey,
} from "./workflow/index.js";
import { semanticInputFromJson, valueToJson } from "./workflow/activity.js";

const logger = getLogger("agent_engine_runner_shared.secure_wrapper");
const DURABLE_TOOL_RESULT_JSON_ERROR =
  "durable workflow values must be JSON-safe";

// =============================================================================
// Token usage extraction
// =============================================================================

/**
 * Normalized usage shape returned by `extractUsage` / `extractPodUsage`.
 * Mirrors Python's dict-shaped return so downstream consumers can pass these
 * fields straight into `reportOeResult`.
 */
export interface ExtractedUsage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  model: string | null;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Extract token usage from an LLM response's `response_metadata`.
 *
 * Handles provider-variant key names:
 *   - OpenAI:   `response_metadata.usage.{prompt_tokens, completion_tokens, total_tokens}`
 *   - Anthropic: `response_metadata.usage.{input_tokens, output_tokens}`
 *   - Some providers: `response_metadata.token_usage.{...}`
 *
 * Returns an object with keys `prompt_tokens`, `completion_tokens`,
 * `total_tokens`, `model`. All values may be `null` if unavailable.
 *
 * Mirrors Python's `agent_engine_runner_shared.secure_wrapper.extract_usage` 1:1.
 */
export function extractUsage(
  response: unknown,
  fallbackModel: string | null = null,
): ExtractedUsage {
  const result: ExtractedUsage = {
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    model: fallbackModel,
  };

  const respObj = isObject(response) ? response : {};
  const rm = isObject(respObj["response_metadata"])
    ? respObj["response_metadata"]
    : {};

  // Coerce each key independently so an empty or malformed `usage`
  // still falls through to `token_usage`.
  let usageTu = coerceTokenUsage(rm["usage"]);
  if (usageTu === undefined) {
    usageTu = coerceTokenUsage(rm["token_usage"]);
  }
  if (usageTu === undefined) {
    usageTu = coerceTokenUsage(respObj["usage_metadata"]);
  }
  if (usageTu === undefined) {
    usageTu = coerceTokenUsage(respObj["usage"]);
  }

  if (usageTu !== undefined) {
    const prompt = usageTu.promptTokens ?? usageTu.inputTokens;
    const completion = usageTu.completionTokens ?? usageTu.outputTokens;
    const total = usageTu.totalTokens;
    if (prompt != null) result.prompt_tokens = prompt;
    if (completion != null) result.completion_tokens = completion;
    if (total != null) {
      result.total_tokens = total;
    } else if (
      result.prompt_tokens !== null &&
      result.completion_tokens !== null
    ) {
      result.total_tokens = result.prompt_tokens + result.completion_tokens;
    }
    if (usageTu.model) result.model = usageTu.model;
  }

  // Python: `rm.get("model_name") or rm.get("model") or fallback_model`
  const modelName =
    (rm["model_name"] as string | undefined) ||
    (rm["model"] as string | undefined) ||
    result.model ||
    fallbackModel;
  result.model = modelName ?? null;

  return result;
}

/**
 * Extract token usage from a tool-pod usage dict into `reportOeResult` kwargs.
 *
 * Uses `!= null` checks instead of truthy `||` so a valid `0` token count is
 * not treated as missing. Computes `total_tokens` when the provider omits it.
 *
 * Mirrors Python's `_extract_pod_usage`. Python keeps the leading underscore
 * as a "private" convention and leaves the function unexported. TS exports
 * this (TS `noUnusedLocals` flags unused module-level functions, and the
 * leading-underscore identifier convention doesn't translate cleanly to TS).
 *
 * @internal — call sites are expected to live inside the agent-engine-runner-shared
 * package or downstream framework SDKs; not part of the stable public API.
 */
export function extractPodUsage(
  usage: Record<string, unknown> | null | undefined,
  fallbackModel: string | null = null,
): ExtractedUsage {
  if (!usage || Object.keys(usage).length === 0) {
    return {
      prompt_tokens: null,
      completion_tokens: null,
      total_tokens: null,
      model: fallbackModel,
    };
  }

  // Note: prefers `input_tokens` over `prompt_tokens` (Anthropic-first),
  // which is the opposite of `extractUsage`. Matches Python verbatim.
  let prompt = usage["input_tokens"] as number | null | undefined;
  if (prompt == null)
    prompt = usage["prompt_tokens"] as number | null | undefined;

  let completion = usage["output_tokens"] as number | null | undefined;
  if (completion == null)
    completion = usage["completion_tokens"] as number | null | undefined;

  let total = usage["total_tokens"] as number | null | undefined;
  if (total == null && prompt != null && completion != null) {
    total = prompt + completion;
  }

  let model = usage["model"] as string | null | undefined;
  if (!model) model = fallbackModel;

  return {
    prompt_tokens: prompt ?? null,
    completion_tokens: completion ?? null,
    total_tokens: total ?? null,
    model: model ?? null,
  };
}

// =============================================================================
// Exceptions
// =============================================================================

export class PolicyDeniedException extends Error {
  readonly reason: string;
  /** Guardrail policy identity when the denial came from a guardrail policy. */
  readonly guardrailMeta: GuardrailMeta | null;
  constructor(
    reason: string,
    guardrailMeta: GuardrailMeta | null = null,
    options?: ErrorOptions,
  ) {
    super(`Policy denied: ${reason}`, options);
    this.name = "PolicyDeniedException";
    this.reason = reason;
    this.guardrailMeta = guardrailMeta;
  }
}

class OERetryAfterError extends Error {
  readonly waitMs: number;
  constructor(waitMs: number, options?: ErrorOptions) {
    super(`OE retry after ${waitMs}ms`, options);
    this.name = "OERetryAfterError";
    this.waitMs = waitMs;
  }
}

/** Raised when tool execution fails. */
export class ToolExecutionError extends Error {
  readonly error: string;
  constructor(error: string, options?: ErrorOptions) {
    super(`Tool execution error: ${error}`, options);
    this.name = "ToolExecutionError";
    this.error = error;
  }
}

const TERMINAL_EXECUTION_REASONS: Record<string, string> = {
  "execution already error":
    "This execution already ended in error; later tool calls are rejected.",
  "execution already completed":
    "This execution already completed; later tool calls are rejected.",
  "execution already cancelled":
    "This execution was cancelled; later tool calls are rejected.",
};

export class TerminalExecutionError extends ToolExecutionError {
  constructor(error: string, options?: ErrorOptions) {
    super(error, options);
    this.name = "TerminalExecutionError";
  }
}

export function raiseForOeRejection(
  reason: string | null | undefined,
  guardrailMeta: GuardrailMeta | null = null,
): never {
  const text = reason ?? "Policy denied";
  if (Object.hasOwn(TERMINAL_EXECUTION_REASONS, text)) {
    throw new TerminalExecutionError(TERMINAL_EXECUTION_REASONS[text]);
  }
  throw new PolicyDeniedException(text, guardrailMeta);
}

/**
 * Raised when a tool call outlives its deadline.
 *
 * Distinct from PolicyDeniedException on purpose. A timeout means the call was
 * permitted and ran — it just ran too long — so reporting it as a denial sends
 * the developer to debug governance instead of their tool.
 */
export class ToolCallTimeoutError extends ToolExecutionError {
  readonly toolName: string;
  readonly timeoutSeconds: number;
  readonly elapsedSeconds: number;
  constructor(
    toolName: string,
    timeoutSeconds: number,
    elapsedSeconds: number,
    options?: ErrorOptions,
  ) {
    const message =
      `Tool '${toolName}' timed out after ${elapsedSeconds.toFixed(1)}s ` +
      `(deadline ${timeoutSeconds.toFixed(0)}s). It can be given longer via ` +
      `RUNNER_TOOL_READ_TIMEOUT, or interrupted while running.`;
    super(message, options);
    this.name = "ToolCallTimeoutError";
    this.message = message;
    this.toolName = toolName;
    this.timeoutSeconds = timeoutSeconds;
    this.elapsedSeconds = elapsedSeconds;
  }
}

/**
 * Whether an error is this call outliving a deadline, as opposed to a transport
 * failure. Covers the caller's AbortSignal.timeout and Undici's own
 * headers/body budgets, which surface as a TypeError carrying a code.
 */
function isCallDeadlineError(exc: Error): boolean {
  if (exc.name === "TimeoutError") return true;
  const code = (exc as { cause?: { code?: string } }).cause?.code;
  return code === "UND_ERR_HEADERS_TIMEOUT" || code === "UND_ERR_BODY_TIMEOUT";
}

class DurableToolActivityFailedError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "DurableToolActivityFailedError";
  }
}

const DURABLE_TOOL_FAILURE_PREFIX = "__agentic_durable_tool_failure_v1__:";
const DURABLE_TOOL_FAILURE_FALLBACK = "durable activity failed";

function toolErrorMessage(error: unknown): string {
  const message =
    error instanceof ToolExecutionError
      ? error.error
      : error instanceof Error
        ? error.message
        : String(error);
  return message || DURABLE_TOOL_FAILURE_FALLBACK;
}

function encodeDurableToolFailure(error: unknown): string {
  const payload: Record<string, unknown> = {
    kind: "execution",
    message: toolErrorMessage(error),
  };
  if (error instanceof ToolCallTimeoutError) {
    payload["kind"] = "timeout";
    payload["tool_name"] = error.toolName;
    payload["timeout_seconds"] = error.timeoutSeconds;
    payload["elapsed_seconds"] = error.elapsedSeconds;
  } else if (error instanceof ExternalAPICallError) {
    payload["kind"] = "external_api";
    payload["tool_api_error"] = error.tool_api_error;
  }
  return `${DURABLE_TOOL_FAILURE_PREFIX}${JSON.stringify(payload)}`;
}

function decodeDurableToolFailure(
  message: string,
  cause: Error,
): ToolExecutionError {
  if (!message.startsWith(DURABLE_TOOL_FAILURE_PREFIX)) {
    return new ToolExecutionError(message || DURABLE_TOOL_FAILURE_FALLBACK, {
      cause,
    });
  }
  let payload: unknown;
  try {
    payload = JSON.parse(message.slice(DURABLE_TOOL_FAILURE_PREFIX.length));
  } catch {
    return new ToolExecutionError(message, { cause });
  }
  if (!isObject(payload)) return new ToolExecutionError(message, { cause });
  const failureMessage =
    typeof payload["message"] === "string" && payload["message"]
      ? payload["message"]
      : DURABLE_TOOL_FAILURE_FALLBACK;
  const kind = payload["kind"];
  if (kind === "external_api") {
    const parsed = ToolAPIErrorSchema.safeParse(payload["tool_api_error"]);
    if (!parsed.success) {
      return new ToolExecutionError(failureMessage, { cause });
    }
    return new ExternalAPICallError(failureMessage, parsed.data, { cause });
  }
  if (kind !== "timeout") {
    return new ToolExecutionError(failureMessage, { cause });
  }
  const toolName = payload["tool_name"];
  const timeoutSeconds = payload["timeout_seconds"];
  const elapsedSeconds = payload["elapsed_seconds"];
  if (
    typeof toolName !== "string" ||
    typeof timeoutSeconds !== "number" ||
    !Number.isFinite(timeoutSeconds) ||
    typeof elapsedSeconds !== "number" ||
    !Number.isFinite(elapsedSeconds)
  ) {
    return new ToolExecutionError(failureMessage, { cause });
  }
  return new ToolCallTimeoutError(toolName, timeoutSeconds, elapsedSeconds, {
    cause,
  });
}

export class ExternalAPICallError extends ToolExecutionError {
  readonly tool_api_error: ToolAPIError;
  constructor(
    error: string,
    toolApiError: ToolAPIError,
    options?: ErrorOptions,
  ) {
    super(error, options);
    this.name = "ExternalAPICallError";
    this.tool_api_error = toolApiError;
  }
}

export class LLMInvocationError extends Error {
  readonly error: string;
  /** Invoke-owner attribution. Only `"llm"` means the provider failed. */
  readonly source?: string;
  /**
   * Machine-readable classification the tool pod stamped on the failure
   * (e.g. a provider credential rejection), when it did. Travels to the
   * OE/UI on the ERROR chunk metadata instead of the generic invocation
   * code so consumers can classify without string-matching prose.
   */
  readonly error_code?: string;
  constructor(
    error: string,
    options?: ErrorOptions & { source?: string; error_code?: string },
  ) {
    super(`LLM invocation error: ${error}`, options);
    this.name = "LLMInvocationError";
    this.error = error;
    this.source = options?.source;
    this.error_code = options?.error_code;
  }
}

// =============================================================================
// OE Communication Helpers
// =============================================================================

export interface RequestOeApprovalArgs {
  oeUrl: string;
  executionId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  step: number;
  kind?: string | null;
  isLocal?: boolean;
  /** Top-level tool argument names to redact from execution logs. */
  redactFields?: readonly string[];
  providerType?: string | null;
  scopes?: readonly string[];
  metadata?: Record<string, unknown>;
  /** Stable LLM tool-call id, forwarded to OE for the execution-log join key. */
  toolCallId?: string;
  /** Caller-provided headers carried only for this active request. */
  customHeaders?: Record<string, string>;
  /**
   * Overall request deadline in milliseconds. Defaults to `getToolReadTimeout()`.
   * The LLM proxy passes a longer value (`LLM_READ_TIMEOUT`) for `invoke_llm`,
   * because OE holds `/tool/execute` open while the LLM call completes — the
   * analog of Python's `httpx.Timeout(get_request_timeout(), read=LLM_READ_TIMEOUT)`.
   */
  timeoutMs?: number;
}

/**
 * Request approval from OE before executing a tool/LLM call.
 * Any HTTP or network error becomes PolicyDeniedException — fail-safe.
 */
export async function requestOeApproval(
  args: RequestOeApprovalArgs,
): Promise<ToolExecuteResponse> {
  try {
    return await requestOeApprovalRaw(args);
  } catch (exc) {
    if (exc instanceof OERetryAfterError) {
      throw new PolicyDeniedException(
        "OE unreachable; blocking for safety",
        null,
        { cause: exc },
      );
    }
    throw exc;
  }
}

async function requestOeApprovalRaw(
  args: RequestOeApprovalArgs,
): Promise<ToolExecuteResponse> {
  const {
    oeUrl,
    executionId,
    toolName,
    arguments: toolArgs,
    step,
    kind,
    isLocal = true,
    redactFields,
    providerType,
    scopes,
    metadata,
    toolCallId,
    customHeaders,
    timeoutMs,
  } = args;

  const { traceId, spanId } = getCurrentTraceContext();
  const body: Record<string, unknown> = {
    execution_id: executionId,
    tool_name: toolName,
    arguments: toolArgs,
    step_number: step,
    tool_call_id: toolCallId ?? null,
    kind: kind ?? null,
    is_local: isLocal,
    redact_fields: redactFields ?? [],
    provider_type: providerType ?? null,
    scopes: scopes ?? [],
    metadata: metadata ?? {},
    custom_headers: customHeaders,
    trace_id: traceId ?? undefined,
    span_id: spanId ?? undefined,
  };

  let resp: Response;
  // The OE keeps /tool/execute open until the tool returns, so this deadline
  // bounds the tool's own runtime. Callers with a different profile (the LLM
  // proxy) pass their own timeoutMs.
  const deadlineMs = timeoutMs ?? getToolReadTimeout() * 1000;
  const startedAt = Date.now();
  try {
    // Undici's own headers/body budgets have to be raised to the call deadline
    // too, or they cap it independently of the AbortSignal. The connect budget
    // stays short so an unreachable OE still fails fast.
    const tlsOptions = getFetchOptionsForLongCall(oeUrl, deadlineMs);
    resp = await fetchPlatform(`${oeUrl.replace(/\/$/, "")}/tool/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // Combine the per-call read deadline with the execution-wide abort signal
      // so an AER execution timeout cancels this in-flight OE call (Python gets
      // this from asyncio.wait_for cancelling the inner coroutine).
      signal: withExecutionSignal(AbortSignal.timeout(deadlineMs)),
      ...tlsOptions,
    });
    if (!resp.ok) {
      const status = resp.status;
      const statusText = resp.statusText;
      const retryAfter = resp.headers?.get?.("Retry-After") ?? null;
      await discardResponseBody(resp);
      if (status === 503) {
        const waitMs = retryAfterWaitMs(retryAfter);
        if (waitMs !== null) {
          throw new OERetryAfterError(waitMs);
        }
      }
      throw new Error(`HTTP ${status}: ${statusText}`);
    }
  } catch (exc) {
    if (exc instanceof OERetryAfterError) {
      throw exc;
    }
    // AbortSignal.any propagates the reason of whichever signal fired, so a
    // TimeoutError is this call's own deadline; an AbortError is the execution
    // being torn down (cancel/interrupt), which is not a timeout. Undici raises
    // its own headers/body deadline as a TypeError with a code rather than a
    // TimeoutError, so match that too — it is still the call outliving a
    // deadline, and reporting it as an unreachable OE is what this replaces.
    if (exc instanceof Error && isCallDeadlineError(exc)) {
      const elapsedSeconds = (Date.now() - startedAt) / 1000;
      logger.error(
        { step, toolName, elapsedSeconds },
        `Step ${step}: Tool '${toolName}' timed out`,
      );
      throw new ToolCallTimeoutError(
        toolName,
        deadlineMs / 1000,
        elapsedSeconds,
        { cause: exc },
      );
    }
    logger.error(
      { step, err: String(exc) },
      `Step ${step}: Failed to request OE approval`,
    );
    throw new PolicyDeniedException(
      "OE unreachable; blocking for safety",
      null,
      {
        cause: exc,
      },
    );
  }

  // Reading the body is part of the fail-safe contract: a 200 with a
  // non-JSON/truncated body (e.g. a proxy error page) must still block rather
  // than surface a raw SyntaxError that bypasses the policy-denied path.
  let data: unknown;
  try {
    data = await resp.json();
  } catch (exc) {
    logger.error(
      { step, err: String(exc) },
      `Step ${step}: Failed to read OE approval response`,
    );
    throw new PolicyDeniedException(
      "OE response unreadable; blocking for safety",
      null,
      { cause: exc },
    );
  }
  // A schema mismatch is a programming/contract bug, not a network error, so
  // let the ZodError propagate rather than masking it as a policy denial.
  return ToolExecuteResponseSchema.parse(data);
}

/**
 * Repeat the same /tool/execute while OE says the failure is retryable.
 * Reservation-loss returns status=error with retryable=true after releasing
 * the step stamp. Retrying the same step here keeps durable activity from
 * recording FAILED on a call that can still succeed.
 */
export async function requestOeApprovalRetryable(
  args: RequestOeApprovalArgs,
): Promise<ToolExecuteResponse> {
  // Started as the active span before the first requestOeApproval call, so
  // its trace_id/span_id — read via getCurrentTraceContext() inside
  // requestOeApproval — reflect this span, not whatever was active before it.
  // Without a span here, a retried call (OE-side reservation loss) is
  // invisible: the surrounding tool-node span just looks slower, with no
  // record of how many attempts happened or how long each took.
  return getTracer().startActiveSpan(
    "secure_wrapper.request_oe_approval",
    async (span) => {
      let attemptCount = 0;
      try {
        let response: ToolExecuteResponse | undefined;
        for (let attempt = 0; attempt < OE_RETRYABLE_MAX_ATTEMPTS; attempt++) {
          try {
            response = await requestOeApprovalRaw(args);
          } catch (err) {
            if (!(err instanceof OERetryAfterError)) {
              throw err;
            }
            attemptCount = attempt + 1;
            if (attempt >= OE_RETRYABLE_MAX_ATTEMPTS - 1) {
              throw new PolicyDeniedException(
                "OE unreachable; blocking for safety",
                null,
                { cause: err },
              );
            }
            const waited = await waitForRetry(
              err.waitMs,
              withExecutionSignal(new AbortController().signal),
            );
            if (!waited) {
              throw new PolicyDeniedException(
                "OE unreachable; blocking for safety",
                null,
                { cause: err },
              );
            }
            continue;
          }
          attemptCount = attempt + 1;
          if (response.status !== "error" || !response.retryable) {
            break;
          }
        }
        if (response === undefined) {
          throw new PolicyDeniedException(
            "OE unreachable; blocking for safety",
          );
        }
        return response;
      } catch (err) {
        span.recordException(err as Error);
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw err;
      } finally {
        span.setAttribute("attempt_count", attemptCount);
        span.end();
      }
    },
  );
}

export interface ReportOeResultArgs {
  oeUrl: string;
  executionId: string;
  toolName: string;
  step: number;
  toolCallId?: string;
  status: string;
  result?: unknown;
  error?: string | null;
  durationMs: number;
  podName?: string | null;
  promptTokens?: number | null;
  completionTokens?: number | null;
  totalTokens?: number | null;
  model?: string | null;
  kind?: string | null;
  metadata?: Record<string, unknown>;
  /**
   * Validated replica-specific OE owner base URL, already checked against
   * `oeUrl` by the caller. When set, a single owner pre-attempt runs before the
   * service loop and does not consume the service retry budget; the owner is
   * best-effort, so any failure — a transport error OR any non-2xx response —
   * marks the replica unusable and falls through to the trusted `oeUrl` loop.
   * The owner is never retried and an owner response never throws.
   */
  ownerUrl?: string | null;
  /** Called once when the owner pre-attempt fails, before service fallback. */
  onOwnerFailure?: (() => void) | null;
  toolApiError?: ToolAPIError | null;
}

/**
 * Report an execution result and require OE to acknowledge settlement.
 *
 * Started as the active span before getCurrentTraceContext() is read below,
 * so the trace_id/span_id put on the wire reflect this span. Without a span
 * here, a slow-to-ack OE (or a retried settlement) is invisible: the
 * surrounding tool-node span just looks slower, with no record of how many
 * attempts happened.
 */
export async function reportOeResult(args: ReportOeResultArgs): Promise<void> {
  return getTracer().startActiveSpan(
    "secure_wrapper.report_oe_result",
    async (span) => {
      let attemptCount = 0;
      try {
        await reportOeResultAttempts(args, (count) => {
          attemptCount = count;
        });
      } catch (err) {
        span.recordException(err as Error);
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw err;
      } finally {
        span.setAttribute("attempt_count", attemptCount);
        span.end();
      }
    },
  );
}

async function reportOeResultAttempts(
  args: ReportOeResultArgs,
  onAttempt: (attemptCount: number) => void,
): Promise<void> {
  const {
    oeUrl,
    executionId,
    toolName,
    step,
    status,
    result,
    error,
    durationMs,
    podName,
    promptTokens,
    completionTokens,
    totalTokens,
    model,
    kind,
    metadata,
    toolCallId,
    ownerUrl,
    onOwnerFailure,
    toolApiError,
  } = args;

  const { traceId, spanId } = getCurrentTraceContext();
  const body: ToolResultRequest = {
    execution_id: executionId,
    step_number: step,
    tool_name: toolName,
    tool_call_id: toolCallId ?? undefined,
    status,
    result,
    error: error ?? undefined,
    duration_ms: durationMs,
    pod_name: podName ?? undefined,
    prompt_tokens: promptTokens ?? undefined,
    completion_tokens: completionTokens ?? undefined,
    total_tokens: totalTokens ?? undefined,
    model: model ?? undefined,
    kind: kind ?? undefined,
    metadata: metadata ?? {},
    trace_id: traceId ?? undefined,
    span_id: spanId ?? undefined,
    tool_api_error: toolApiError ?? undefined,
  };

  const serializedBody = JSON.stringify(body);
  const serviceUrl = `${oeUrl.replace(/\/$/, "")}/tool/result`;

  // Owner pre-attempt per the shared owner-callback policy (owner_callback.ts):
  // a single best-effort POST that does not consume the service retry budget.
  const ownerResultUrl = ownerUrl
    ? `${ownerUrl.replace(/\/$/, "")}/tool/result`
    : null;
  const attemptOffset = ownerResultUrl === null ? 0 : 1;
  if (ownerResultUrl !== null) onAttempt(1);
  if (
    ownerResultUrl !== null &&
    (await ownerDelivered(
      ownerResultUrl,
      () => ({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serializedBody,
        signal: AbortSignal.timeout(getRequestTimeout() * 1000),
        ...getFetchOptionsWithTLS(ownerResultUrl),
      }),
      (msg) => logger.warn(`/tool/result ${msg} URL ${serviceUrl}`),
    ))
  ) {
    return;
  }
  if (ownerResultUrl !== null) {
    onOwnerFailure?.();
  }

  const tlsOptions = getFetchOptionsWithTLS(serviceUrl);
  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    let retryWaitMs: number | null = null;
    onAttempt(attemptOffset + attempt + 1);
    let response: Response | undefined;
    try {
      response = await fetchPlatform(serviceUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serializedBody,
        signal: AbortSignal.timeout(getRequestTimeout() * 1000),
        ...tlsOptions,
      });
    } catch (error) {
      lastError = error;
    }
    if (response !== undefined) {
      const ok = response.ok;
      const responseStatus = response.status;
      const retryAfter = response.headers?.get?.("Retry-After") ?? null;
      await discardResponseBody(response);
      if (ok) return;
      const error = new ToolExecutionError(
        `OE rejected result settlement with HTTP ${responseStatus}`,
      );
      if (responseStatus >= 400 && responseStatus < 500) {
        throw error;
      }
      lastError = error;
      if (responseStatus === 503) {
        retryWaitMs = retryAfterWaitMs(retryAfter);
      }
    }
    if (attempt < 2) {
      await new Promise((resolve) =>
        setTimeout(resolve, retryWaitMs ?? 100 * 2 ** attempt),
      );
    }
  }
  throw new ToolExecutionError(
    `OE did not acknowledge tool result settlement: ${String(lastError)}`,
  );
}

// =============================================================================
// Interrupt marker
// =============================================================================

// Marker key for an OE-triggered call interruption, kept out of the plain
// shape a real tool result could return, and namespaced to avoid collision
// with a tool's own artifact keys.
//
import { CALL_INTERRUPTED_ARTIFACT_KEY } from "./call_interrupted.js";

// Re-exported so existing `secure_wrapper.js` importers keep working; the
// canonical owner is `./call_interrupted.js` (see its comment).
export { CALL_INTERRUPTED_ARTIFACT_KEY };
export const INTERRUPTED_CALL_CONTENT =
  "This call was stopped before completing.";

// Identity-checked sentinel distinguishing an OE-triggered interrupt from any
// real tool return value. Never shape-checked — a tool cannot return this by accident.
class CallInterrupted {}
const CALL_INTERRUPTED = new CallInterrupted();

const INTERRUPTED_WIRE_MARKER = Object.freeze({
  [CALL_INTERRUPTED_ARTIFACT_KEY]: true,
});

/** Canonical tool inputs used for durable identity and replay matching. */
interface ToolActivityInput {
  arguments: Record<string, unknown>;
  is_local: boolean;
  metadata?: Record<string, unknown>;
  provider_type?: string;
  scopes?: string[];
  tool_call_id?: string;
}

function isLocalTool(options: ExecuteToolOptions): boolean {
  return (
    (options.isLocal ?? true) &&
    !options.providerType &&
    (options.scopes?.length ?? 0) === 0
  );
}

function buildToolActivityInput(
  args: Record<string, unknown>,
  options: ExecuteToolOptions,
): ToolActivityInput {
  const input: ToolActivityInput = {
    arguments: args,
    is_local: isLocalTool(options),
  };
  if (options.metadata !== undefined) input.metadata = options.metadata;
  if (options.providerType != null) {
    input.provider_type = options.providerType;
  }
  if (options.scopes !== undefined && options.scopes.length > 0) {
    input.scopes = [...options.scopes].sort();
  }
  if (options.toolCallId) input.tool_call_id = options.toolCallId;
  return input;
}

function encodeToolActivityResult(value: unknown): unknown {
  let encoded: unknown;
  if (value === CALL_INTERRUPTED) {
    encoded = INTERRUPTED_WIRE_MARKER;
  } else if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value) ||
    !(CALL_INTERRUPTED_ARTIFACT_KEY in value)
  ) {
    encoded = value;
  } else {
    const result = { ...(value as Record<string, unknown>) };
    delete result[CALL_INTERRUPTED_ARTIFACT_KEY];
    encoded = result;
  }
  assertDurableToolResultJsonSafe(encoded);
  return encoded;
}

function assertDurableToolResultJsonSafe(value: unknown): void {
  let roundTripped: unknown;
  try {
    roundTripped = valueToJson(semanticInputFromJson(value));
  } catch {
    throw new TypeError(DURABLE_TOOL_RESULT_JSON_ERROR);
  }
  if (!isDeepStrictEqual(value, roundTripped)) {
    throw new TypeError(DURABLE_TOOL_RESULT_JSON_ERROR);
  }
}

function decodeToolActivityResult(
  value: unknown,
  rawOnInterrupt: boolean,
): unknown {
  const isInterrupt =
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    Object.keys(value).length === 1 &&
    (value as Record<string, unknown>)[CALL_INTERRUPTED_ARTIFACT_KEY] === true;
  if (!isInterrupt) return value;
  return rawOnInterrupt ? CALL_INTERRUPTED : { interrupted: true };
}

// =============================================================================
// OperationalStepAllocator
// =============================================================================

/**
 * Process-local operational step_number mint for one AER execution.
 *
 * Shared by SecureToolWrapper and SecureLLMProxy so parallel tools and LLM
 * calls cannot collide under a single AER. Not a multi-OE authority.
 * Node's single-threaded event loop makes a plain counter
 * safe across concurrent async tasks as long as minting stays synchronous
 * before the first await.
 */
export interface OperationalStepSource {
  next(): number;
  observeAtLeast(n: number): void;
  current(): number;
}

export class OperationalStepAllocator implements OperationalStepSource {
  private n = 0;

  next(): number {
    this.n += 1;
    return this.n;
  }

  observeAtLeast(n: number): void {
    if (n <= 0) return;
    if (n > this.n) this.n = n;
  }

  current(): number {
    return this.n;
  }
}

// =============================================================================
// SecureToolWrapper
// =============================================================================

/**
 * Optional args for {@link SecureToolWrapper.executeTool}, grouped into an
 * object rather than positional parameters so future additions cannot silently
 * shift values with overlapping types into the wrong slot.
 */
export interface ExecuteToolOptions {
  metadata?: Record<string, unknown> | undefined;
  providerType?: string | null | undefined;
  scopes?: readonly string[] | undefined;
  toolCallId?: string | undefined;
  /**
   * When true, an OE `interrupted` status resolves to the raw `CALL_INTERRUPTED`
   * sentinel instead of `{ interrupted: true }`. Only the LangGraph tool-wrapping
   * path (`createSecureToolFunction`) requests this — it coerces the sentinel
   * into a safe shape immediately, never returning it to the caller.
   */
  rawOnInterrupt?: boolean | undefined;
  /** The tool's `redact_fields` policy, applied to the debug argument dump. */
  redactFields?: readonly string[] | undefined;
  /** Whether OE should return the approved call to this AER. */
  isLocal?: boolean | undefined;
  /** Invoke the registered tool on the current framework stack. */
  localExecutor?: (() => unknown | Promise<unknown>) | undefined;
  /** Identify framework-owned native control-flow errors. */
  isFrameworkControlFlow?: ((error: unknown) => boolean) | undefined;
}

export class SecureToolWrapper {
  readonly oeUrl: string;
  readonly executionId: string;
  readonly customHeaders: Record<string, string>;
  /**
   * Validated replica-specific OE owner base URL, or null. Forwarded
   * to `reportOeResult` for the in-process (local-callback) tool-result path so
   * settlement prefers the owning OE replica.
   */
  readonly oeOwnerUrl: string | null;
  private ownerUrlFailed = false;
  /** Shared with SecureLLMProxy for this execution (single-AER assumption). */
  readonly operationalSteps: OperationalStepSource;
  durableMemory: DurableMemoryState | null = null;

  constructor(
    oeUrl: string,
    executionId: string,
    customHeaders?: Record<string, string>,
    oeOwnerUrl?: string | null,
  ) {
    this.oeUrl = oeUrl.replace(/\/$/, "");
    this.executionId = executionId;
    this.customHeaders = customHeaders ?? {};
    this.oeOwnerUrl = oeOwnerUrl ?? null;
    this.operationalSteps = new OperationalStepAllocator();
  }

  private currentOeOwnerUrl(): string | null {
    if (this.ownerUrlFailed) return null;
    if (getCurrentExecutionId() === this.executionId) {
      return getCurrentOeOwnerUrl();
    }
    return this.oeOwnerUrl;
  }

  private reportOeOwnerFailure(): void {
    this.ownerUrlFailed = true;
    if (getCurrentExecutionId() === this.executionId) {
      reportOeOwnerUrlFailure();
    }
  }

  /** Current operational-step watermark for compatibility readers. */
  get stepCounter(): number {
    return this.operationalSteps.current();
  }

  /** Allocate the next operational step_number for this execution. */
  nextOperationalStep(): number {
    return this.operationalSteps.next();
  }

  /** Raise the allocator watermark (e.g. from OE latest_step_number). */
  observeOperationalStep(n: number): void {
    this.operationalSteps.observeAtLeast(n);
  }

  async close(): Promise<void> {
    // no-op — available for future cleanup
  }

  /**
   * Execute a tool call through OE.
   *
   * Flow:
   *   1. POST /tool/execute — OE approves or denies
   *   2. OE returns a final outcome or routes the call back in process
   *   3. Wrapper logs the outcome and converts suspend payloads back into framework interrupts
   */
  async executeTool(
    toolName: string,
    args: Record<string, unknown>,
    options: ExecuteToolOptions = {},
  ): Promise<unknown> {
    const step = this.nextOperationalStep();
    // The tool's redact_fields policy must reach the debug argument dump —
    // it ships to the centralized log sink even at debug level.
    logToolRequest(toolName, args, step, "TOOL", options.redactFields ?? []);

    if (currentAttemptContext() === null) {
      return this.executeToolNative(toolName, args, step, options);
    }
    return this.executeToolDurably(toolName, args, step, options);
  }

  private async executeToolDurably(
    toolName: string,
    args: Record<string, unknown>,
    step: number,
    options: ExecuteToolOptions,
  ): Promise<unknown> {
    const { toolCallId, rawOnInterrupt = false } = options;
    const durableMemory = this.durableMemory;
    const activityKey = toolCallId ? toolActivityKey(toolCallId) : undefined;
    const activityOrdinal = allocateActivityOrdinal(activityKey);
    try {
      const result = await runSerialActivity({
        client: new WorkflowClient(this.oeUrl),
        kind: ActivityKind.TOOL,
        name: toolName,
        activityOrdinal,
        semanticInput: buildToolActivityInput(args, options),
        execute: (context) =>
          this.executeToolActivity(
            toolName,
            args,
            step,
            options,
            context.activityId,
          ),
        onActivityResolved:
          durableMemory === null
            ? undefined
            : (client, context, value) =>
                durableMemory.synchronizeTool(
                  client as WorkflowClient,
                  context,
                  value,
                  getCurrentUserId(),
                  toolCallId,
                  toolName,
                ),
        exclusive: activityKey === undefined,
      });
      return decodeToolActivityResult(result, rawOnInterrupt);
    } catch (error) {
      if (error instanceof DurableActivityDeniedError) {
        if (error.cause instanceof PolicyDeniedException) throw error.cause;
        throw new PolicyDeniedException(error.message, null, {
          cause: error,
        });
      }
      if (error instanceof ReplayedActivityFailedError) {
        throw decodeDurableToolFailure(error.message, error);
      }
      if (error instanceof DurableToolActivityFailedError) {
        throw decodeDurableToolFailure(error.message, error);
      }
      throw error;
    }
  }

  private async executeToolActivity(
    toolName: string,
    args: Record<string, unknown>,
    step: number,
    options: ExecuteToolOptions,
    activityId: string,
  ): Promise<unknown> {
    try {
      if (
        isLocalTool(options) &&
        options.localExecutor !== undefined &&
        activityRequiresReconstruction(activityId)
      ) {
        // OE already resolved this exact admitted activity. Re-enter its
        // captured callback so LangGraph can reconstruct interrupt().
        return encodeToolActivityResult(await options.localExecutor());
      }
      const result = await this.executeToolNative(
        toolName,
        args,
        step,
        { ...options, rawOnInterrupt: true },
        "reject",
      );
      return encodeToolActivityResult(result);
    } catch (error) {
      if (error instanceof PolicyDeniedException) {
        throw new DurableActivityDeniedError(error.reason, { cause: error });
      }
      if (options.isFrameworkControlFlow?.(error) === true) {
        throw new DurableActivityInterrupted(error);
      }
      throw new DurableToolActivityFailedError(
        encodeDurableToolFailure(error),
        {
          cause: error,
        },
      );
    }
  }

  private async executeToolNative(
    toolName: string,
    args: Record<string, unknown>,
    step: number,
    options: ExecuteToolOptions,
    interruptMode: "framework" | "reject" = "framework",
  ): Promise<unknown> {
    const {
      metadata,
      providerType,
      scopes,
      toolCallId,
      rawOnInterrupt = false,
      isLocal = true,
      localExecutor,
      isFrameworkControlFlow,
    } = options;
    const effectiveIsLocal = isLocalTool({
      isLocal,
      providerType,
      scopes,
    });
    const effectiveLocalExecutor = effectiveIsLocal ? localExecutor : undefined;

    const response = await requestOeApprovalRetryable({
      oeUrl: this.oeUrl,
      executionId: this.executionId,
      toolName,
      arguments: args,
      step,
      isLocal: effectiveIsLocal,
      redactFields: options.redactFields,
      providerType,
      scopes,
      toolCallId,
      metadata,
      customHeaders: this.customHeaders,
    });

    if (response.elicitation) {
      if (interruptMode === "reject") {
        throw new ToolExecutionError(
          "durable Tool activities support terminal outcomes only; " +
            "the framework adapter must own authorization interrupts",
        );
      }
      const interrupt = getSuspendHandler();
      if (interrupt == null) {
        throw new Error(
          "No suspend handler registered. Ensure the framework SDK calls registerSuspendHandler() before run().",
        );
      }
      interrupt({
        suspend_reason: "authorization_required",
        suspend_context: {
          authorization_url: response.elicitation.authorization_url,
          elicitation_id: response.elicitation.elicitation_id,
          message: response.elicitation.message,
          created: response.elicitation.created,
        },
      });
      throw new Error(
        "authorization_required interrupt must halt execution; suspend handler should throw (e.g. LangGraph GraphInterrupt).",
      );
    }

    if (!response.proceed) {
      const reason = response.reason ?? "Policy denied";
      if (!Object.hasOwn(TERMINAL_EXECUTION_REASONS, reason)) {
        logPolicyBlocked(toolName, step, reason);
      }
      raiseForOeRejection(reason, response.guardrail_meta ?? null);
    }

    if (response.latest_step_number != null) {
      this.observeOperationalStep(response.latest_step_number);
    }

    if (response.from_cache) {
      logCachedResult(toolName, step);
    }

    if (response.route_to === "callback") {
      if (!effectiveIsLocal) {
        throw new ToolExecutionError(
          "OE returned a local callback route for a tool declared as remote",
        );
      }
      if (effectiveLocalExecutor == null) {
        // Registered-tool adapters always install this executor. Merely
        // running in AER is not enough for a low-level caller: the wrapper
        // still needs the original framework tool object to invoke.
        throw new ToolExecutionError(
          "Local callback route is missing its registered tool executor",
        );
      }
      const startedAt = performance.now();
      let localResult: unknown;
      let suspendMarker: Record<string, unknown> | null = null;
      try {
        await runWithSuspendRequestContext(async () => {
          localResult = await runWithCustomerOrigin(() =>
            effectiveLocalExecutor(),
          );
          suspendMarker = getRequestedSuspend();
        });
      } catch (error) {
        const interrupted =
          isFrameworkControlFlow?.(error) === true ||
          (error instanceof Error && error.name === "AbortError");
        const propagatedError = error;
        const durationMs = performance.now() - startedAt;
        const status = interrupted ? "interrupted" : "error";
        const classified =
          status === "error"
            ? await classifyToolAPIError(
                propagatedError,
                undefined,
                requestCredentialValues(),
              )
            : undefined;
        const errorText = classified
          ? classified.message
          : propagatedError instanceof Error
            ? `${propagatedError.name}: ${propagatedError.message}`
            : String(propagatedError);
        logToolResult(toolName, step, status, null, errorText, durationMs);
        try {
          await reportOeResult({
            oeUrl: this.oeUrl,
            ownerUrl: this.currentOeOwnerUrl(),
            onOwnerFailure: () => this.reportOeOwnerFailure(),
            executionId: this.executionId,
            toolName,
            step,
            toolCallId,
            status,
            result: null,
            error: errorText,
            durationMs,
            metadata,
            toolApiError: classified?.toolApiError,
          });
        } catch (settlementError) {
          throw new AggregateError(
            [propagatedError, settlementError],
            "In-process tool result settlement failed",
            { cause: settlementError },
          );
        }
        throw propagatedError;
      }
      const durationMs = performance.now() - startedAt;
      if (suspendMarker !== null && interruptMode === "reject") {
        const error = new ToolExecutionError(
          "durable Tool activities support terminal outcomes only; " +
            "the framework adapter must own suspension and resume",
        );
        logToolResult(toolName, step, "error", null, error.message, durationMs);
        await reportOeResult({
          oeUrl: this.oeUrl,
          ownerUrl: this.currentOeOwnerUrl(),
          onOwnerFailure: () => this.reportOeOwnerFailure(),
          executionId: this.executionId,
          toolName,
          step,
          toolCallId,
          status: "error",
          result: null,
          error: error.message,
          durationMs,
          metadata,
        });
        throw error;
      }
      const status = suspendMarker === null ? "success" : "suspend";
      const reportedResult =
        suspendMarker === null ? localResult : JSON.stringify(suspendMarker);
      logToolResult(toolName, step, status, reportedResult, null, durationMs);
      await reportOeResult({
        oeUrl: this.oeUrl,
        ownerUrl: this.currentOeOwnerUrl(),
        onOwnerFailure: () => this.reportOeOwnerFailure(),
        executionId: this.executionId,
        toolName,
        step,
        toolCallId,
        status,
        result: reportedResult,
        error: null,
        durationMs,
        metadata,
      });
      if (suspendMarker !== null) {
        // suspendPayloadToJson records this out of band, so relayed tool output
        // cannot forge a wait.
        return this.handleSuspend(reportedResult);
      }
      return localResult;
    }
    if (response.route_to != null) {
      throw new ToolExecutionError(
        `Unexpected OE tool route: ${response.route_to}`,
      );
    }

    const status = response.status ?? "error";
    const result = response.result;
    const error = response.error;
    const durationMs = response.duration_ms ?? 0;

    logToolResult(toolName, step, status, result, error ?? null, durationMs);

    if (status === "interrupted") {
      // Only the LangGraph tool-wrapping path requests the raw sentinel and
      // coerces it immediately; every other caller keeps the plain dict.
      return rawOnInterrupt ? CALL_INTERRUPTED : { interrupted: true };
    }
    if (status === "error") {
      if (response.tool_api_error) {
        throw new ExternalAPICallError(
          error ?? "Unknown error",
          response.tool_api_error,
        );
      }
      throw new ToolExecutionError(error ?? "Unknown error");
    }
    // Suspend is honored only from the OE-confirmed status channel, whose
    // provenance is the tool author's suspendPayloadToJson call — never from
    // sniffing result content, which a relayed untrusted payload controls.
    // A successful result is returned verbatim.
    if (status === "suspend") {
      if (interruptMode === "reject") {
        throw new ToolExecutionError(
          "durable Tool activities support terminal outcomes only; " +
            "the framework adapter must own suspension and resume",
        );
      }
      return this.handleSuspend(result);
    }
    if (status !== "success") {
      throw new ToolExecutionError(`Unexpected OE tool status: ${status}`);
    }

    return result;
  }

  /** Fire the framework HITL interrupt for an OE-confirmed suspend. */
  private handleSuspend(result: unknown): unknown {
    let parsed = result;
    if (typeof result === "string") {
      try {
        parsed = JSON.parse(result);
      } catch {
        /* not JSON, keep as-is */
      }
    }
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      throw new ToolExecutionError(
        "OE reported suspend but the result is not a suspend payload",
      );
    }
    const interrupt = getSuspendHandler();
    if (interrupt == null) {
      throw new Error(
        "No suspend handler registered. Ensure the framework SDK calls registerSuspendHandler() before run().",
      );
    }
    const payload = SuspendPayloadSchema.parse(parsed);
    const humanDecision = interrupt(payload as Record<string, unknown>);
    if (
      typeof humanDecision !== "object" ||
      humanDecision === null ||
      Array.isArray(humanDecision)
    ) {
      throw new TypeError(
        `Expected object from HITL interrupt, got ${typeof humanDecision}`,
      );
    }
    return JSON.stringify(humanDecision);
  }
}

// =============================================================================
// content_and_artifact coercion
// =============================================================================

export type ToolResponseFormat = "content" | "content_and_artifact";

function isTwoElementArray(value: unknown): value is [unknown, unknown] {
  return Array.isArray(value) && value.length === 2;
}

/**
 * Shapes a tool result for LangChain's content_and_artifact wire format.
 * `toolDeclaredFormat` (the tool's own original format, independent of the
 * forced wire-level `responseFormat`) decides whether a real result gets
 * shape-matched into (content, artifact) — otherwise it's never split.
 */
function coerceContentAndArtifact(
  result: unknown,
  responseFormat: ToolResponseFormat,
  toolDeclaredFormat: ToolResponseFormat = "content",
): unknown {
  if (result === CALL_INTERRUPTED) {
    if (responseFormat === "content_and_artifact") {
      return [
        INTERRUPTED_CALL_CONTENT,
        { [CALL_INTERRUPTED_ARTIFACT_KEY]: true },
      ];
    }
    return { interrupted: true };
  }
  if (responseFormat !== "content_and_artifact") return result;
  if (
    toolDeclaredFormat === "content_and_artifact" &&
    isTwoElementArray(result)
  ) {
    const [content, artifact] = result;
    // Only the interrupted branch above may set this key — a tool's own
    // artifact must never be able to forge it.
    if (
      typeof artifact === "object" &&
      artifact !== null &&
      CALL_INTERRUPTED_ARTIFACT_KEY in artifact
    ) {
      const stripped = { ...(artifact as Record<string, unknown>) };
      delete stripped[CALL_INTERRUPTED_ARTIFACT_KEY];
      return [content, stripped];
    }
    return result;
  }
  return [result, null];
}

// =============================================================================
// Context-Aware Tool Wrapper
// =============================================================================

/**
 * The subset of the LangChain JS run config a wrapped tool reads. When a tool
 * is invoked with a ToolCall, ToolNode populates `toolCall` (whose `id` is the
 * stable tool-call id) and mirrors the id onto `configurable.tool_call_id`.
 */
export interface SecureToolRunConfig {
  toolCall?: { id?: unknown };
  configurable?: Record<string, unknown>;
}

/**
 * Optional args for {@link createSecureToolFunction}, grouped into an object
 * for the same reason as {@link ExecuteToolOptions}.
 */
export interface CreateSecureToolFunctionOptions {
  metadata?: Record<string, unknown> | undefined;
  providerType?: string | null | undefined;
  scopes?: readonly string[] | undefined;
  /** The wire format the caller's StructuredTool is built with. See {@link coerceContentAndArtifact}. */
  responseFormat?: ToolResponseFormat | undefined;
  /**
   * The tool author's own original format, used only to decide whether a
   * normal result gets shape-matched into a tuple. Defaults to `responseFormat`.
   */
  toolDeclaredFormat?: ToolResponseFormat | undefined;
  /** The tool's `redact_fields` policy, forwarded to the wrapper's debug argument dump. */
  redactFields?: readonly string[] | undefined;
  /** Whether to execute the approved tool on the current AER stack. */
  isLocal?: boolean | undefined;
  /** Identify framework-owned native control-flow errors. */
  isFrameworkControlFlow?: ((error: unknown) => boolean) | undefined;
}

/**
 * Create a wrapped tool function that routes through SecureToolWrapper.
 *
 * The wrapper is looked up from AsyncLocalStorage context at call time,
 * so the tool can be built before execution context exists.
 *
 * The registered tool must expose an `invoke` method for both approved local
 * execution and the direct-execution debugging path. Direct execution is
 * UNSAFE and only for debugging.
 */
export function createSecureToolFunction(
  originalTool: unknown,
  toolName: string,
  allowDirect = false,
  options: CreateSecureToolFunctionOptions = {},
): (
  kwargs?: Record<string, unknown>,
  config?: SecureToolRunConfig,
) => Promise<unknown> {
  const {
    metadata,
    providerType,
    scopes,
    responseFormat = "content",
    toolDeclaredFormat,
    redactFields,
    isLocal = true,
    isFrameworkControlFlow,
  } = options;
  const effectiveToolFormat = toolDeclaredFormat ?? responseFormat;
  return async (
    kwargs: Record<string, unknown> = {},
    config?: SecureToolRunConfig,
  ) => {
    // When a tool is invoked with a ToolCall, LangChain JS exposes it on
    // config.toolCall (toolCall.id is the stable id) and also mirrors the id
    // onto config.configurable.tool_call_id. Prefer config.toolCall.id — that is
    // the documented shape ToolNode produces — and fall back to configurable.
    // Forward it to OE for the execution-log join key.
    const rawToolCallId =
      config?.toolCall?.id ?? config?.configurable?.tool_call_id;
    const toolCallId =
      typeof rawToolCallId === "string" ? rawToolCallId : undefined;
    const wrapper = getCurrentWrapper() as SecureToolWrapper | null;
    if (wrapper == null) {
      if (!allowDirect) {
        throw new Error(
          `SecureToolWrapper missing for tool ${toolName}; ` +
            "execution not allowed (RUNNER_ALLOW_DIRECT_TOOL_EXECUTION=false)",
        );
      }
      logger.warn(
        `No wrapper for tool ${toolName}, executing directly (UNSAFE)`,
      );
      const tool = originalTool as {
        invoke: (kwargs: Record<string, unknown>) => unknown;
      };
      return coerceContentAndArtifact(
        await tool.invoke(kwargs),
        responseFormat,
        effectiveToolFormat,
      );
    }

    // Capture the registered tool object and this invocation's arguments. OE
    // approves first; invoking this closure afterward keeps execution on the
    // framework's original stack with its live native context.
    const localExecutor = isLocal
      ? () =>
          runWithToolMemoryReadOwnership(() => {
            const tool = originalTool as {
              invoke: (input: Record<string, unknown>) => unknown;
            };
            return tool.invoke(kwargs);
          })
      : undefined;

    try {
      const result = await wrapper.executeTool(toolName, kwargs, {
        metadata,
        providerType,
        scopes,
        toolCallId,
        rawOnInterrupt: true,
        redactFields,
        isLocal,
        localExecutor,
        isFrameworkControlFlow,
      });
      return coerceContentAndArtifact(
        result,
        responseFormat,
        effectiveToolFormat,
      );
    } catch (exc) {
      // Both branches re-throw; the only distinction is that a
      // ToolExecutionError is logged as it propagates to the graph.
      if (exc instanceof ToolExecutionError) {
        logger.info(
          { tool_name: toolName, error: exc.error },
          `Propagating tool execution error to graph for ${toolName}: ${exc.error}`,
        );
      }
      throw exc;
    }
  };
}
