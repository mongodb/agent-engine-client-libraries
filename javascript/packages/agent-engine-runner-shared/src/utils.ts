/**
 * Utility functions for Runner SDK.
 */

import { getLogger } from "./logger.js";

// Module-level logger — equivalent to Python `logging.getLogger(__name__)`.
const logger = getLogger("agent_engine_runner_shared.utils");

// =============================================================================
// Runtime Mode
// =============================================================================

/** Runtime mode for the Runner SDK. */
export enum RuntimeMode {
  AER = "aer",
  TOOL = "tool",
  // Tool role, per-call (function) lifecycle: the process boots, runs a
  // single named tool from the registry, reports the result, and exits.
  // TOOL is the same role as a long-running server handling many /execute
  // calls. The value carries the role because the function lifecycle is
  // orthogonal to it (a future per-call AER would be e.g. "aer_function").
  // The invocation is read once from the guest metadata channel, not env.
  TOOL_FUNCTION = "tool_function",
}

const MODE_MAP: Record<string, RuntimeMode> = {
  [RuntimeMode.AER]: RuntimeMode.AER,
  [RuntimeMode.TOOL]: RuntimeMode.TOOL,
  [RuntimeMode.TOOL_FUNCTION]: RuntimeMode.TOOL_FUNCTION,
};

/**
 * Get the current runtime mode from environment variable.
 *
 * @returns RuntimeMode based on RUNNER_MODE environment variable.
 * @throws Error If RUNNER_MODE is not set or is not a valid mode.
 */
export function getRuntimeMode(): RuntimeMode {
  const raw = process.env["RUNNER_MODE"];
  if (raw === undefined) {
    throw new Error(
      `RUNNER_MODE environment variable is required. Valid values: ${Object.keys(MODE_MAP).join(", ")}`,
    );
  }
  const mode = raw.toLowerCase();
  const resolved = MODE_MAP[mode];
  if (!resolved) {
    throw new Error(
      `Unknown RUNNER_MODE '${mode}'. Valid values: ${Object.keys(MODE_MAP).join(", ")}`,
    );
  }
  return resolved;
}

// =============================================================================
// Platform-owned environment variables
// =============================================================================
//
// Used by `agent_config.ts` to validate fields that name an env var (e.g.
// `mcp.servers.*.auth.token_env`) and to build the substitution mapping for
// `loadRuntimeAgentConfig(envVars=...)` so tenant `agent.yaml` cannot use
// `${VAR}` interpolation to dereference platform secrets or internal service
// addresses (e.g. `url: https://attacker/${OPENAI_API_KEY}`).
//
// Maintenance: whenever a new platform-owned env var is introduced in
// `agent-engine-runner-shared` (or a sibling package whose env reaches the runtime
// container), add its exact name to `PLATFORM_ENV_VARS` or, if it shares a
// common prefix with related platform vars, extend
// `PLATFORM_ENV_VAR_PREFIXES`. The prefix list does most of the work — prefer
// adding a prefix over enumerating individual names. Keep in sync with
// `agent_engine_runner_shared/utils.py`'s `_PLATFORM_ENV_VAR_PREFIXES` / `_PLATFORM_ENV_VARS`.
const PLATFORM_ENV_VAR_PREFIXES: readonly string[] = [
  // `AGENTIC_*` is the platform's reserved namespace. Tenants are documented
  // to use unprefixed names for their own vars.
  "AGENTIC_",
  // Component-scoped families
  "MONGOMEM_", // memory server config
  "MONGODB_", // MongoDB URI + client tunables
  "VOYAGE_", // platform-owned embedder config
  "LLM_", // retry tunables in this module
  "OE_", // orchestration engine
  "AER_", // Agent Execution Runtime endpoints
  "TOOL_", // Tool Pod endpoints
  "ECP_", // Executor Control Plane
  "RUNNER_", // runner-internal tunables (mode, stream timeouts, ...)
  "GUARDRAILS_", // guardrails server config + LLM key
  "FILESYSTEM_", // Tool Pod sandbox limits
  "SHELL_", // Tool Pod sandbox limits
  "MAX_", // Tool Pod sandbox limits (MAX_LS_ENTRIES, MAX_GREP_*, ...)
  // Auth / observability
  "OKTA_",
  "OIDC_",
  "OTEL_",
  // Test infra (must not leak into production tenant interpolation either)
  "E2E_",
  // Container / orchestration infra
  "KUBERNETES_",
  "HELIX_",
  // POSIX
  "LC_",
];

const PLATFORM_ENV_VARS: ReadonlySet<string> = new Set([
  // Multi-tenant identity
  "ORG_ID",
  "PROJECT_ID",
  "ATLAS_GROUP_ID",
  "TENANT_ID",
  "APP_ID",
  // Operator-stamped MaaS observability identity (keep in sync with
  // utils.py's `_PLATFORM_ENV_VARS`).
  "AGENT_ENGINE_ENVIRONMENT",
  // Platform LLM provider credentials. Read by the standalone memory server
  // for extraction-pipeline provider detection, but potentially present in
  // any runtime container.
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "CEREBRAS_API_KEY",
  "GEMINI_API_KEY",
  // Platform LLM endpoints (not secret, but leaking them via tenant YAML is
  // still pointless and a potential SSRF vector).
  "OPENAI_BASE_URL",
  "ANTHROPIC_BASE_URL",
  // Auth / inter-service secrets
  "A2A_JWT_SECRET",
  // TLS / mTLS certificate paths (operator-mounted for AER→OE HTTPS)
  "TLS_CERT_PATH",
  "TLS_KEY_PATH",
  "TLS_CA_CERT_PATH",
  // TLS / mTLS PEM content (VM mode via SecretKeyRef, kubelet decodes automatically)
  // CRITICAL: Must be excluded to prevent tenant exfiltration via ${TLS_KEY_PEM}
  "TLS_CERT_PEM",
  "TLS_KEY_PEM",
  "TLS_CA_CERT_PEM",
  // Server / network config
  "APP_HOST",
  "APP_PORT",
  "LOG_LEVEL",
  "LOG_DIR", // operator-injected via render.WorkloadInjectedEnv
  "STRUCTURED_LOGGING", // operator-injected via render.StructuredLoggingEnv
  "CORS_ALLOWED_ORIGINS", // OE-injected (forward-defensive on AER)
  "USE_MEMORY_CLIENT",
  "AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT",
  "SHUTDOWN_GRACE_PERIOD_MS",
  // Runner launcher entrypoint baked into the Dockerfile ENV by ECP
  "AGENT_ENTRYPOINT",
  // Platform persistence
  "PLATFORM_DATABASE",
  "MEMORY_DATABASE_NAME",
  "MDB_AGENTIC_STORE_DB",
  "CHECKPOINT_DB_NAME",
  "DB_NAME",
  "CHECKPOINTER_SERVER_SELECTION_TIMEOUT",
  "CHECKPOINTER_CONNECT_TIMEOUT",
  "CHECKPOINTER_SOCKET_TIMEOUT",
  "TEST_MONGO_URI",
  // Tool Pod sandbox (covered partially by prefix, but the no-prefix
  // `WORKSPACE_DIR` / `DOWNLOAD_MAX_BYTES` names live here).
  "WORKSPACE_DIR",
  "DOWNLOAD_MAX_BYTES",
  // Feature toggles
  "ENABLE_MEMORY",
  "ENABLE_TRACING",
  "ENABLE_VECTOR_SEARCH_TESTS",
  // Dev / test infra
  "SEED_DATA",
  "CI",
  // Container / pod infra
  "POD_NAME",
  "HOSTNAME",
  // Generic POSIX. Substituting these into a URL is never what the tenant
  // means; blocking prevents accidental leakage of process identity /
  // filesystem layout.
  "PATH",
  "HOME",
  "USER",
  "PWD",
  "LANG",
  "TERM",
  "SHELL",
  "SHLVL",
  "_",
]);

/**
 * Return `true` if `name` matches a platform-owned env var.
 *
 * Public-API view of the same membership check used by `tenantEnvVars`.
 */
export function isPlatformEnvVar(name: string): boolean {
  return (
    PLATFORM_ENV_VARS.has(name) ||
    PLATFORM_ENV_VAR_PREFIXES.some((prefix) => name.startsWith(prefix))
  );
}

/**
 * Return the tenant-owned subset of environment variables.
 *
 * `source` defaults to `process.env`. Names in `PLATFORM_ENV_VARS` or
 * matching any prefix in `PLATFORM_ENV_VAR_PREFIXES` are excluded, so the
 * result is safe to pass as the substitution mapping to
 * `loadRuntimeAgentConfig({ envVars: ... })`.
 */
export function tenantEnvVars(
  source?: Record<string, string | undefined>,
): Record<string, string> {
  const from = source ?? (process.env as Record<string, string | undefined>);
  const result: Record<string, string> = {};
  for (const [name, value] of Object.entries(from)) {
    if (value !== undefined && !isPlatformEnvVar(name)) {
      result[name] = value;
    }
  }
  return result;
}

// =============================================================================
// Environment Helpers
// =============================================================================

/** Get environment variable with default. */
export function getEnv(name: string, defaultValue = ""): string {
  return process.env[name] ?? defaultValue;
}

/** Get integer environment variable with default. */
export function getEnvInt(name: string, defaultValue: number): number {
  const val = process.env[name];
  if (!val) return defaultValue;
  const parsed = parseInt(val, 10);
  return isNaN(parsed) ? defaultValue : parsed;
}

/** Get float environment variable with default. */
export function getEnvFloat(name: string, defaultValue: number): number {
  const val = process.env[name];
  if (!val) return defaultValue;
  const parsed = parseFloat(val);
  return isNaN(parsed) ? defaultValue : parsed;
}

/** Get boolean environment variable with default. */
export function getEnvBool(name: string, defaultValue = false): boolean {
  const raw = process.env[name];
  const value = (raw ?? String(defaultValue)).toLowerCase();
  return ["true", "1", "yes", "on"].includes(value);
}

/**
 * Get the HTTP request timeout from environment.
 *
 * Uses RUNNER_REQUEST_TIMEOUT env var, defaults to 60.0 seconds.
 */
export function getRequestTimeout(): number {
  return getEnvFloat("RUNNER_REQUEST_TIMEOUT", 60.0);
}

/**
 * Get the tool-call read timeout from environment.
 *
 * The OE holds /tool/execute open until the tool result comes back, so this
 * bounds the tool's own runtime rather than the handshake. It matches the OE's
 * own tool deadline: a smaller value abandons a tool the platform is still
 * happily running, leaving the SDK with no result to report.
 *
 * Uses RUNNER_TOOL_READ_TIMEOUT env var, defaults to 600.0 seconds.
 */
export function getToolReadTimeout(): number {
  return getEnvFloat("RUNNER_TOOL_READ_TIMEOUT", 600.0);
}

// =============================================================================
// LLM Retry Configuration
// =============================================================================

// LLM retry configuration for rate limit errors.
export const LLM_MAX_RETRIES = getEnvInt("LLM_MAX_RETRIES", 3);
export const LLM_INITIAL_BACKOFF = getEnvFloat("LLM_INITIAL_BACKOFF", 1.0); // seconds
export const LLM_BACKOFF_MULTIPLIER = getEnvFloat(
  "LLM_BACKOFF_MULTIPLIER",
  2.0,
);
export const LLM_MAX_BACKOFF = getEnvFloat("LLM_MAX_BACKOFF", 30.0); // seconds

/** Same-step retries of /tool/execute (and the SSE relay) after OE advertises retryable=true, and after an SSE transport disconnect. 1 initial + 2 extras; lockstep with Python. */
export const OE_RETRYABLE_MAX_ATTEMPTS = 3;

/**
 * Wait after a dropped SSE connection so the next attempt can land after
 * dispatch_heartbeat TTL (30s). Do not reuse HTTP Retry-After's 10s cap.
 * Lockstep with Python.
 */
export const OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS = 30_000;
export const OE_DISPATCH_RETRY_MAX_WAIT_MS = 60_000;

/** Honor a server-provided SSE retry_after_ms, capped. Absent means retry immediately. */
export function oeStreamRetryDelayMs(retryAfterMs?: number | null): number {
  if (
    retryAfterMs == null ||
    !Number.isFinite(retryAfterMs) ||
    retryAfterMs <= 0
  ) {
    return 0;
  }
  return Math.min(retryAfterMs, OE_DISPATCH_RETRY_MAX_WAIT_MS);
}

/** Sleep before an SSE same-URL retry. Tests spy this to avoid wall-clock waits. */
export const oeStreamRetry = {
  async sleep(delayMs: number): Promise<void> {
    if (delayMs <= 0) return;
    await new Promise<void>((resolve) => {
      setTimeout(resolve, delayMs);
    });
  },
};

// LLM read timeout — LLM generation can take much longer than a typical HTTP request.
// Applies to both streaming (per-chunk wait) and non-streaming (full-response wait) paths.
export const LLM_READ_TIMEOUT = getEnvFloat(
  "RUNNER_LLM_READ_TIMEOUT",
  getEnvFloat("RUNNER_STREAM_READ_TIMEOUT", 300.0),
); // seconds

/** Check if an error is retryable (rate limit, server overload, etc.). */
export function isRetryableError(error: unknown): boolean {
  const errorStr = String(error).toLowerCase();
  const retryablePatterns = [
    "too_many_requests",
    "rate_limit",
    "429",
    "500",
    "503",
    "server error",
    "queue_exceeded",
    "high traffic",
    "overloaded",
  ];
  return retryablePatterns.some((p) => errorStr.includes(p));
}

/**
 * Render an LLM provider error for display, without escaped-JSON text.
 *
 * Some provider SDKs attach the raw API error body as an object on the
 * error or its `cause` (`.details`, `.body`). That object's values are
 * often themselves JSON-encoded strings containing real newlines (e.g. a
 * pretty-printed nested error payload), so `String()`/template-literal
 * stringification ends up producing a repr that turns the newlines into
 * literal `\n` sequences. Decode any JSON-encoded string values
 * first and re-serialize with `JSON.stringify(..., null, 2)` so the result
 * renders as readable, indented JSON instead.
 */
export function formatLlmError(error: unknown): string {
  const candidates = [error, error instanceof Error ? error.cause : undefined];
  for (const candidate of candidates) {
    if (candidate === null || candidate === undefined) continue;
    const c = candidate as Record<string, unknown>;
    try {
      let body = c["details"];
      if (typeof body !== "object" || body === null || Array.isArray(body)) {
        body = c["body"];
      }
      if (typeof body === "object" && body !== null && !Array.isArray(body)) {
        return JSON.stringify(decodeNestedJson(body), jsonReplacer, 2);
      }
    } catch {
      // A provider-supplied .details/.body can be anything -- a plain
      // property, a getter that throws, a dict with circular refs or exotic
      // values. Never let formatting itself throw — fall back to the plain
      // message below.
      break;
    }
  }
  return error instanceof Error ? error.message : String(error);
}

/** `JSON.stringify` replacer that stringifies values it can't serialize natively. */
function jsonReplacer(_key: string, value: unknown): unknown {
  if (
    typeof value === "bigint" ||
    typeof value === "function" ||
    typeof value === "symbol"
  ) {
    return String(value);
  }
  return value;
}

/** Recursively `JSON.parse()` any string value that looks like JSON. */
function decodeNestedJson(value: unknown): unknown {
  if (typeof value === "string") {
    const stripped = value.trim();
    if (stripped && (stripped[0] === "{" || stripped[0] === "[")) {
      try {
        return decodeNestedJson(JSON.parse(stripped));
      } catch {
        return value;
      }
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(decodeNestedJson);
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        k,
        decodeNestedJson(v),
      ]),
    );
  }
  return value;
}

/**
 * True when an LLM provider error is an auth rejection (HTTP 401/403).
 *
 * Reads only the provider SDK's own status fields — `status` (openai-node /
 * anthropic-node style), `status_code`, `response.status` /
 * `response.status_code` (fetch/axios style), and an integer `code` —
 * walking the `cause` chain because adapters occasionally re-throw. Text is
 * never matched: the caller maps a positive result onto the wire-level
 * credential error code, and anything unrecognized returns false so the
 * failure keeps its existing generic classification.
 */
export function isLlmCredentialRejection(error: unknown): boolean {
  const seen = new Set<unknown>();
  const stack: unknown[] = [error];
  while (stack.length > 0) {
    const current = stack.pop();
    if (current === null || current === undefined || seen.has(current)) {
      continue;
    }
    seen.add(current);
    const status = exceptionHttpStatus(current);
    if (status === 401 || status === 403) {
      return true;
    }
    if (current instanceof Error) {
      stack.push(current.cause);
    }
  }
  return false;
}

/** Extract the HTTP status a provider SDK attached to its error. */
function exceptionHttpStatus(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) {
    return undefined;
  }
  const e = error as Record<string, unknown>;
  const response = e["response"];
  const responseStatus =
    typeof response === "object" && response !== null
      ? ((response as Record<string, unknown>)["status"] ??
        (response as Record<string, unknown>)["status_code"])
      : undefined;
  for (const candidate of [
    e["status"],
    e["status_code"],
    responseStatus,
    e["code"],
  ]) {
    if (typeof candidate === "number" && Number.isInteger(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

// =============================================================================
// Content Normalization
// =============================================================================

/**
 * Normalize LLM message content to a string.
 *
 * LLM content can be:
 * - A string (normal text)
 * - A list (multimodal content with text and other parts)
 * - null/undefined or empty
 *
 * This handles cases where LangChain's AIMessageChunk.content is a list
 * of content blocks (e.g., from Gemini multimodal responses) rather than
 * a simple string.
 *
 * @param content The content to normalize (string, list, or null/undefined)
 * @returns A string representation of the content
 */
export function normalizeContent(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (typeof part === "string") {
        parts.push(part);
      } else if (typeof part === "object" && part !== null) {
        const p = part as Record<string, unknown>;
        if (p["type"] === "text" && typeof p["text"] === "string") {
          parts.push(p["text"]);
        }
      }
    }
    return parts.join("");
  }
  return String(content);
}

/** Normalize streamed tool-call args into the wire-format string payload. */
export function normalizeToolCallArgs(args: unknown): string | null {
  if (args === null || args === undefined) return null;
  if (typeof args === "string") return args;
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}

/**
 * Trim a string tool-metadata value, collapsing blank/non-string to null.
 *
 * Shared by the framework SDKs' tool-wrapping code so metadata values written
 * as "" or "  " are treated the same as absent rather
 * than sent to the OE as a non-empty-looking but meaningless string.
 */
export function normalizeOptionalStr(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length > 0) return trimmed;
  }
  return null;
}

// =============================================================================
// Thinking Token Filter
// =============================================================================

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

/**
 * Length of the longest suffix of *buffer* that is a *proper* prefix of
 * `<think>` (i.e. `<`, `<t`, … `<think` but never the full tag — a full
 * match is handled by `indexOf`). Used to hold back a marker that was split
 * across streamed chunks so its tail can't leak as plain text. Returns 0
 * when no partial opener trails the buffer.
 */
function partialOpenSuffixLength(buffer: string): number {
  const max = Math.min(buffer.length, THINK_OPEN.length - 1);
  for (let k = max; k > 0; k--) {
    if (buffer.endsWith(THINK_OPEN.slice(0, k))) return k;
  }
  return 0;
}

/** Remove `<think>...</think>` blocks and unclosed `<think>` tails. */
export function stripThinking(text: string): string {
  if (!text) return "";
  let cleaned = text.replace(/<think>.*?<\/think>/gs, "");
  cleaned = cleaned.replace(/<think>.*$/s, "");
  return cleaned.trim();
}

/**
 * Filter `<think>` blocks from a stream of token chunks.
 *
 * Accumulates text in *buffer* until we can determine whether content
 * is inside a thinking block. Returns `[streamable, newBuffer, inside]`
 * where *streamable* is the text safe to send to the client.
 */
export function filterThinkingTokens(
  token: string,
  buffer: string,
  inside: boolean,
): [streamable: string, newBuffer: string, inside: boolean] {
  buffer += token;
  const streamableParts: string[] = [];

  while (true) {
    if (inside) {
      const closeIdx = buffer.indexOf(THINK_CLOSE);
      if (closeIdx === -1) return [streamableParts.join(""), buffer, true];
      buffer = buffer.slice(closeIdx + THINK_CLOSE.length);
      inside = false;
    } else {
      const openIdx = buffer.indexOf(THINK_OPEN);
      if (openIdx === -1) {
        // No complete `<think>` opener. The buffer may still end with a
        // partial opener split across chunk boundaries (e.g. "…<thi"); flush
        // everything before it but hold the partial back, otherwise the rest
        // of the marker ("nk>secret…") would arrive next chunk and stream as
        // plain text, leaking the hidden block.
        const held = partialOpenSuffixLength(buffer);
        const flushEnd = buffer.length - held;
        streamableParts.push(buffer.slice(0, flushEnd));
        return [streamableParts.join(""), buffer.slice(flushEnd), false];
      }
      streamableParts.push(buffer.slice(0, openIdx));
      buffer = buffer.slice(openIdx + THINK_OPEN.length);
      inside = true;
    }
  }
}

// =============================================================================
// Structured Logging Utilities
// =============================================================================
//
// `setupLogging` is intentionally NOT exported from this file (or from
// logger.ts) — full setup lives in `structured_logging.ts` alongside the
// agent-log JSON contract and stdout/stderr capture. Modules only need
// `getLogger(name)` for now.

// Truncation limits for log readability.
const MAX_CONTENT_LENGTH = 200;

const MAX_LOGGED_PAYLOAD_FIELDS = 20;
const MAX_LOGGED_FIELD_NAME_CHARS = 64;
const MAX_REDACT_FIELDS = 100;

function isLogUnsafeCodePoint(codePoint: number): boolean {
  // C0, DEL, and C1 (including CSI U+009B) must not reach the stdout log sink.
  return codePoint < 32 || (codePoint >= 127 && codePoint <= 0x9f);
}

function boundedFieldName(key: string): string {
  const prefix = key.slice(0, MAX_LOGGED_FIELD_NAME_CHARS);
  let safe = "";
  for (const char of prefix) {
    const codePoint = char.codePointAt(0) ?? 0;
    safe += isLogUnsafeCodePoint(codePoint) ? "?" : char;
  }
  return safe + (key.length > MAX_LOGGED_FIELD_NAME_CHARS ? "..." : "");
}

function shallowValueSummary(value: unknown): string {
  if (typeof value === "string") {
    return `string chars=${value.length}`;
  }
  if (value instanceof Uint8Array) {
    return `bytes length=${value.byteLength}`;
  }
  if (Array.isArray(value)) {
    return `array items=${value.length}`;
  }
  if (value instanceof Set) {
    return `array items=${value.size}`;
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "boolean") {
    return "boolean";
  }
  if (typeof value === "number") {
    return "number";
  }
  if (typeof value === "object") {
    return "object";
  }
  return typeof value;
}

/** Describe a payload without serializing or recursively walking it. */
function payloadDebugSummary(
  payload: unknown,
  fieldsToRedact: readonly string[] = [],
): string {
  if (
    payload === null ||
    typeof payload !== "object" ||
    Array.isArray(payload) ||
    payload instanceof Uint8Array ||
    payload instanceof Set ||
    (Object.getPrototypeOf(payload) !== Object.prototype &&
      Object.getPrototypeOf(payload) !== null)
  ) {
    return `type=${shallowValueSummary(payload)} values_omitted=true`;
  }

  const redactAll = fieldsToRedact.length > MAX_REDACT_FIELDS;
  const redactedFields = redactAll
    ? new Set<string>()
    : new Set(fieldsToRedact);
  const fields: string[] = [];
  let fieldsTruncated = false;
  const record = payload as Record<string, unknown>;

  for (const key in record) {
    if (!Object.prototype.hasOwnProperty.call(record, key)) continue;
    if (fields.length >= MAX_LOGGED_PAYLOAD_FIELDS) {
      fieldsTruncated = true;
      break;
    }
    const summary =
      redactAll || redactedFields.has(key)
        ? "redacted"
        : shallowValueSummary(record[key]);
    fields.push(`${boundedFieldName(key)}=<${summary}>`);
  }

  return `fields=[${fields.join(", ")}] fields_truncated=${fieldsTruncated} values_omitted=true`;
}

/** Describe a result without exposing dynamic object keys. */
function resultDebugSummary(result: unknown): string {
  return `type=${shallowValueSummary(result)} values_omitted=true`;
}

/** Log a minor section separator (for individual operations). */
export function logSeparator(): void {
  logger.info("-".repeat(40));
}

/** Log a major section separator (for execution boundaries). */
export function logSection(): void {
  logger.info("=".repeat(60));
}

/**
 * Log LLM conversation messages in a consistent format.
 *
 * @param messages List of message dicts or LangChain message objects
 * @param prefix Log line prefix (e.g., "OE", "LLM", "AER")
 * @param countOnly If true, only log message count (for info level)
 */
export function logLLMMessages(
  messages: unknown[],
  prefix = "LLM",
  countOnly = false,
): void {
  if (countOnly) {
    logger.info(`${prefix}: ${messages.length} messages`);
    return;
  }
  logger.debug(`${prefix}: ${messages.length} messages:`);
  messages.forEach((msg, i) => {
    const m = msg as Record<string, unknown>;
    const msgType = m["type"] ?? "unknown";
    const content = m["content"] ?? "";
    const toolCalls = m["tool_calls"];
    const preview = String(content).slice(0, MAX_CONTENT_LENGTH) || "(empty)";
    if (toolCalls && Array.isArray(toolCalls)) {
      logger.debug(
        `${prefix}:   [${i}] ${msgType}: ${preview}... + ${toolCalls.length} tool_calls`,
      );
    } else {
      logger.debug(`${prefix}:   [${i}] ${msgType}: ${preview}...`);
    }
  });
}

/**
 * Log an LLM response in a consistent format.
 *
 * @param result LLM response (dict or AIMessage)
 * @param step Step number
 * @param prefix Log line prefix
 */
export function logLLMResponse(
  result: unknown,
  step: number,
  prefix = "LLM",
): void {
  if (!result) return;
  const r = result as Record<string, unknown>;
  const content = r["content"] ?? "";
  const toolCalls = r["tool_calls"];
  const preview = String(content).slice(0, MAX_CONTENT_LENGTH) || "(empty)";
  logger.debug(`${prefix}: Step ${step} - Response: ${preview}...`);
  if (toolCalls && Array.isArray(toolCalls)) {
    const names = toolCalls.map((tc) => {
      const t = tc as Record<string, unknown>;
      return t["name"] ?? String(tc);
    });
    logger.info(`${prefix}: Step ${step} - Tool calls: ${names.join(", ")}`);
  }
}

// Mirrors Python's agent_engine_runner_shared.logging.redact_fields.
/**
 * Redact sensitive fields from an arguments record.
 *
 * Returns a copy with each listed field replaced by "[REDACTED]"; unlisted
 * fields pass through unchanged, and an empty list returns the input as-is.
 */
export function redactFields(
  data: Record<string, unknown>,
  fieldsToRedact: readonly string[],
): Record<string, unknown> {
  if (fieldsToRedact.length === 0) return data;
  const result = { ...data };
  for (const field of fieldsToRedact) {
    if (field in result) {
      result[field] = "[REDACTED]";
    }
  }
  return result;
}

/**
 * Extract a tool's `redact_fields` policy from its registered definition.
 * The definition is an opaque metadata record (framework-SDK populated), so
 * non-string-array values degrade to no redaction rather than throwing.
 */
export function toolRedactFields(
  toolDefinitions: Record<string, Record<string, unknown>>,
  toolName: string | undefined,
): readonly string[] {
  const raw = toolName
    ? toolDefinitions[toolName]?.["redact_fields"]
    : undefined;
  if (!Array.isArray(raw)) return [];
  return raw.filter((f): f is string => typeof f === "string");
}

/**
 * Log a tool execution request.
 *
 * @param toolName Name of the tool
 * @param args Tool arguments
 * @param step Step number
 * @param prefix Log line prefix
 * @param fieldsToRedact Catalog sensitive-field policy. Matching fields omit
 *   even their type/length metadata; no values are logged.
 */
export function logToolRequest(
  toolName: string,
  args: Record<string, unknown>,
  step: number,
  prefix = "TOOL",
  fieldsToRedact: readonly string[] = [],
): void {
  logSeparator();
  logger.info(`${prefix}: Step ${step} - ${toolName}`);
  if (logger.isLevelEnabled("debug")) {
    logger.debug(
      `${prefix}: Step ${step} - Arguments: ${payloadDebugSummary(args, fieldsToRedact)}`,
    );
  }
}

/**
 * Log a tool execution result.
 *
 * @param toolName Name of the tool
 * @param step Step number
 * @param status Execution status (success, error, suspend, interrupted)
 * @param result Tool result
 * @param error Error message if failed
 * @param durationMs Execution duration in milliseconds
 * @param prefix Log line prefix
 */
export function logToolResult(
  toolName: string,
  step: number,
  status: string,
  result: unknown = null,
  error: string | null = null,
  durationMs = 0,
  prefix = "TOOL",
): void {
  logger.info(
    `${prefix}: Step ${step} - ${toolName} ${status} (${durationMs.toFixed(0)}ms)`,
  );
  if (error) {
    logger.error(`${prefix}: Step ${step} - Error: ${error}`);
  } else if (result !== null) {
    if (logger.isLevelEnabled("debug")) {
      logger.debug(
        `${prefix}: Step ${step} - Result: ${resultDebugSummary(result)}`,
      );
    }
  }
}

/** Log that a cached result is being used (replay scenario). */
export function logCachedResult(
  toolName: string,
  step: number,
  prefix = "TOOL",
): void {
  logger.info(
    `${prefix}: Step ${step} - ${toolName} using CACHED result (replay)`,
  );
}

/** Log that a tool call was blocked by policy. */
export function logPolicyBlocked(
  toolName: string,
  step: number,
  reason: string,
  prefix = "TOOL",
): void {
  logger.warn(
    `${prefix}: Step ${step} - ${toolName} BLOCKED by policy: ${reason}`,
  );
}

/** Log the start of an execution. */
export function logExecutionStart(
  executionId: string,
  inputKeys: string[],
  prefix = "OE",
): void {
  logSection();
  logger.info(`${prefix}: Starting execution ${executionId.slice(0, 8)}...`);
  logger.debug(`${prefix}: Input keys: ${inputKeys.join(", ")}`);
}

/** Log an execution callback (completion/suspension/error). */
export function logExecutionCallback(
  executionId: string,
  status: string,
  result: unknown = null,
  error: string | null = null,
  suspendReason: string | null = null,
  prefix = "OE",
): void {
  logSection();
  logger.info(`${prefix}: Execution ${executionId.slice(0, 8)}... → ${status}`);
  if (suspendReason) logger.info(`${prefix}: Suspend reason: ${suspendReason}`);
  if (error) logger.error(`${prefix}: Error: ${error}`);
  if (result) {
    const preview = String(result).slice(0, 300);
    logger.debug(`${prefix}: Result: ${preview}...`);
  }
}
