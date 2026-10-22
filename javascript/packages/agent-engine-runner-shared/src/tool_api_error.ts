/**
 * Classify external API call failures into structured, safe metadata.
 *
 * Mirrors Python's `agent_engine_runner_shared/tool_api_error.py`.
 *
 * Native `fetch` (Node ≥ 24, backed by Undici) does NOT throw on non-2xx
 * responses — it only rejects for transport/abort failures. So HTTP-status
 * classification requires a thrown error that retains a numeric `status`,
 * `statusCode`, or `response.status` property (e.g. an Axios-shaped error
 * or a custom wrapper around a fetch Response). Transport failures
 * (timeout, connection refused) are classified from error names and
 * `cause.code` values that Undici/Node produces.
 */

import { z } from "zod";
import { getCurrentAuthorization } from "./context.js";
import { redactText } from "./error_reporting.js";
import type { ToolPodExecuteResponse } from "./models.js";
import { tenantEnvVars } from "./utils.js";

const MAX_ERROR_CODE_LEN = 128;
const MAX_REASON_LEN = 256;

const STATUS_CLASSIFICATIONS: Record<number, [string, boolean]> = {
  401: ["AUTH_FAILED", false],
  403: ["AUTH_FAILED", false],
  429: ["RATE_LIMITED", true],
  503: ["PROVIDER_UNAVAILABLE", true],
};

const TIMEOUT_CODES = new Set([
  "ETIMEDOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_CONNECT_TIMEOUT",
]);

const CONNECTION_CODES = new Set([
  "ECONNREFUSED",
  "ECONNRESET",
  "ENOTFOUND",
  "EAI_AGAIN",
  "UND_ERR_SOCKET",
  "EHOSTUNREACH",
  "ENETUNREACH",
]);

const MAX_ENVELOPE_BODY = 4096;

// Rudimentary URL drop: a scheme-bearing URL or a protocol-relative reference
// with a non-whitespace authority. Encoded or nested spellings are accepted
// risk, not chased.
const PROTOCOL_RELATIVE_URL = /(?<![A-Za-z0-9])\/\/(?=\S)/;
const URL_MARKER = new RegExp(`://|${PROTOCOL_RELATIVE_URL.source}`);

/**
 * Request-local credential values that must never be surfaced. Combines the
 * tenant environment secrets with the delegated authorization token installed
 * for this execution; both are values a provider can echo.
 *
 * Reads pod-level tenant env; the ToolServer runs with secret restriction
 * disabled (merge, no restore), so these values persist for the request. If
 * per-request apply/restore is ever re-enabled, capture the values inside the
 * credential window instead of reading them at classification time.
 */
export function requestCredentialValues(): string[] {
  const values = Object.values(tenantEnvVars()).filter((value) => value);
  const token = getCurrentAuthorization()?.token;
  if (token && !values.includes(token)) values.push(token);
  return values;
}

// Name segments marking a tenant env var's value as credential-bearing. Env
// names are conventionally SCREAMING_SNAKE with the credential kind last
// (OPENAI_API_KEY, GITHUB_TOKEN, DB_PASSWORD), so the name is what selects
// credential values. Selecting on the value's shape instead -- its length, or
// whether it looks alphanumeric -- cannot separate a 7-character token from the
// working directory or the string "1", and redacting those shreds the message
// the redaction exists to make readable.
const CREDENTIAL_NAME_MARKERS = [
  "KEY",
  "TOKEN",
  "SECRET",
  "PASSWORD",
  "CREDENTIAL",
  "CREDS",
  "AUTH",
];

export function isCredentialName(name: string): boolean {
  const upper = name.toUpperCase();
  return CREDENTIAL_NAME_MARKERS.some((marker) => upper.includes(marker));
}

/**
 * Credential values a name marks as secrets, at any length.
 *
 * A narrower view of `requestCredentialValues`: ordinary tenant env (`PATH`,
 * `HOSTNAME`, `SHLVL`) is excluded, while a value under a credential-named
 * variable is treated as a credential however short it is. Use this where the
 * value is inserted into text a human reads, so that redaction cannot corrupt
 * the message; use `requestCredentialValues` where the sink tolerates a blunt
 * replacement. The delegated authorization token counts as credential-named.
 */
export function requestNamedCredentialValues(): string[] {
  const values = Object.entries(tenantEnvVars())
    .filter(([name, value]) => Boolean(value) && isCredentialName(name))
    .map(([, value]) => value);
  const token = getCurrentAuthorization()?.token;
  if (token && !values.includes(token)) values.push(token);
  return values;
}

export const ToolAPIErrorSchema = z.object({
  provider_type: z.string().nullish().describe("Provider identity when known"),
  classification: z
    .string()
    .describe(
      "AUTH_FAILED, RATE_LIMITED, PROVIDER_UNAVAILABLE, TIMEOUT, CONNECTION_ERROR, or UNKNOWN",
    ),
  http_status: z
    .number()
    .int()
    .nullish()
    .describe("HTTP status when the failure was an HTTP response"),
  retryable: z
    .boolean()
    .describe(
      "Provider-condition guidance only. Must not trigger replay of an already-dispatched tool call.",
    ),
  error_code: z
    .string()
    .nullish()
    .describe("Bounded redacted provider error code"),
  reason: z
    .string()
    .nullish()
    .describe("Bounded redacted provider reason; URLs dropped"),
});
export type ToolAPIError = z.infer<typeof ToolAPIErrorSchema>;

export interface ClassifiedToolAPIError {
  toolApiError: ToolAPIError;
  message: string;
}

/**
 * Walk the `.cause` chain of an error, yielding each node up to a
 * cycle-safe maximum depth of five. Non-Error objects with a `cause`
 * property are included — Undici wraps transport failures as
 * `TypeError("fetch failed")` with a plain-object `cause` carrying the
 * relevant `code`.
 */
function* walkCauseChain(
  error: unknown,
  maxDepth = 5,
): Generator<Record<string, unknown>> {
  let current: unknown = error;
  let depth = 0;
  const seen = new Set<unknown>();
  while (current != null && depth < maxDepth && !seen.has(current)) {
    seen.add(current);
    yield current as Record<string, unknown>;
    current = (current as { cause?: unknown })?.cause;
    depth++;
  }
}

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function findHttpStatus(error: unknown): number | null {
  for (const node of walkCauseChain(error)) {
    if (isNum(node["status"])) return node["status"] as number;
    if (isNum(node["statusCode"])) return node["statusCode"] as number;
    const resp = node["response"];
    if (resp && typeof resp === "object") {
      const r = resp as Record<string, unknown>;
      if (isNum(r["status"])) return r["status"] as number;
      if (isNum(r["statusCode"])) return r["statusCode"] as number;
    }
  }
  return null;
}

function findResponse(error: unknown): unknown | null {
  for (const node of walkCauseChain(error)) {
    const resp = node["response"];
    if (resp != null) return resp;
  }
  return null;
}

function extractFromBody(
  body: Record<string, unknown>,
  credentials: readonly string[] = [],
): [string | null, string | null] {
  return [
    envelopeErrorCode(body, credentials),
    envelopeReason(body, credentials),
  ];
}

function envelopeErrorCode(
  body: Record<string, unknown>,
  credentials: readonly string[] = [],
): string | null {
  const rawCode = body["errorCode"];
  if (typeof rawCode === "string") {
    return safeEnvelopeText(rawCode, MAX_ERROR_CODE_LEN, credentials);
  }
  const nested = asRecord(body["error"]);
  if (nested && typeof nested["code"] === "string") {
    return safeEnvelopeText(
      nested["code"] as string,
      MAX_ERROR_CODE_LEN,
      credentials,
    );
  }
  return null;
}

/**
 * The provider's own explanation from a recognized error envelope. Shapes
 * follow the common conventions: Atlas `reason`, Jira `errorMessages`/`errors`,
 * a top-level `message`, and a nested `error.message` (OpenAI/Stripe/Anthropic).
 */
function envelopeReason(
  body: Record<string, unknown>,
  credentials: readonly string[] = [],
): string | null {
  const candidates: string[] = [];
  const rawReason = body["reason"];
  if (typeof rawReason === "string") candidates.push(rawReason);
  const messages = body["errorMessages"];
  if (Array.isArray(messages)) {
    for (const message of messages) {
      if (typeof message === "string") candidates.push(message);
    }
  }
  const errors = asRecord(body["errors"]);
  if (errors) {
    for (const value of Object.values(errors)) {
      if (typeof value === "string") candidates.push(value);
    }
  }
  const rawMessage = body["message"];
  if (typeof rawMessage === "string") candidates.push(rawMessage);
  const nested = asRecord(body["error"]);
  if (nested && typeof nested["message"] === "string") {
    candidates.push(nested["message"] as string);
  }
  for (const candidate of candidates) {
    const safe = safeEnvelopeText(candidate, MAX_REASON_LEN, credentials);
    if (safe) return safe;
  }
  return null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * Rudimentary hygiene for one provider-authored string. Best-effort, not a
 * guarantee: control characters are dropped, URL-shaped text is rejected,
 * exact credential values are redacted, and the result is bounded. Other
 * encodings and provider-specific content are accepted risk.
 */
function safeEnvelopeText(
  text: string,
  maxLen: number,
  credentials: readonly string[] = [],
): string | null {
  let printable = withoutControlCharacters(text);
  if (printable.includes("://") || PROTOCOL_RELATIVE_URL.test(printable)) {
    return null;
  }
  const ordered = credentials
    .filter((value) => value)
    .sort((a, b) => b.length - a.length);
  for (const value of ordered) {
    printable = printable.split(value).join("<redacted>");
  }
  const out = redactText(printable).slice(0, maxLen);
  return out || null;
}

async function extractEnvelope(
  response: unknown,
  credentials: readonly string[] = [],
): Promise<[string | null, string | null]> {
  if (!response || typeof response !== "object") return [null, null];
  const r = response as Record<string, unknown>;

  // Axios-shaped: response.data is the already-parsed body. It gets the same
  // encoded-size limit as a native streamed body, so an oversized document
  // cannot supply a visible reason. A body Axios could not parse as JSON
  // arrives as a string.
  const data = r["data"];
  if (data && typeof data === "object" && !Array.isArray(data)) {
    if (!isWithinEnvelopeLimit(data)) return [null, null];
    return extractFromBody(data as Record<string, unknown>, credentials);
  }
  if (typeof data === "string") {
    const raw = new TextEncoder().encode(data);
    if (raw.byteLength > MAX_ENVELOPE_BODY) return [null, null];
    return [null, plainTextReason(raw, credentials)];
  }

  const raw = await boundedBody(r);
  if (raw === null) return [null, null];
  let body: unknown;
  try {
    body = JSON.parse(Buffer.from(raw).toString("utf8"));
  } catch {
    return [null, plainTextReason(raw, credentials)];
  }
  if (body && typeof body === "object" && !Array.isArray(body)) {
    return extractFromBody(body as Record<string, unknown>, credentials);
  }
  return [null, null];
}

/**
 * The explanation in a body that is not JSON. Some providers and gateways
 * refuse with a bare sentence, whatever content type they declare, often
 * followed by documentation links. Only the text before the first link is
 * kept, so the explanation survives the URL rule. Markup is skipped so an HTML
 * error page never becomes the reason, and an undecodable body supplies none,
 * including one a caller already decoded with replacement characters.
 */
function plainTextReason(
  raw: Uint8Array,
  credentials: readonly string[] = [],
): string | null {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch {
    return null;
  }
  if (text.includes("\ufffd")) return null;
  text = normalizeText(text);
  if (text.startsWith("<")) return null;
  const marker = URL_MARKER.exec(text);
  if (marker) {
    text = text
      .slice(0, text.lastIndexOf(" ", marker.index) + 1)
      .replace(/[\s([:,;-]+$/, "");
  }
  if (!text) return null;
  // The markup and link checks read the provider's own text; redaction then
  // covers each credential as sent and as the normalization above respells it.
  const spellings = [...credentials, ...credentials.map(normalizeText)];
  return safeEnvelopeText(text, MAX_REASON_LEN, spellings);
}

/** `text` with whitespace runs collapsed and control characters dropped. */
function normalizeText(text: string): string {
  return withoutControlCharacters(text.split(/\s+/).filter(Boolean).join(" "));
}

/** `text` without control characters, so none can hide a URL marker. */
function withoutControlCharacters(text: string): string {
  return text.replace(/\p{C}/gu, "");
}

// The 4 KB limit is a per-binding heuristic: Axios exposes only parsed data,
// so this measures the re-serialized document rather than wire bytes, while
// streamed and Python bodies are measured on raw bytes.
function isWithinEnvelopeLimit(value: unknown): boolean {
  try {
    const encoded = JSON.stringify(value);
    if (typeof encoded !== "string") return false;
    return new TextEncoder().encode(encoded).byteLength <= MAX_ENVELOPE_BODY;
  } catch {
    return false;
  }
}

/** A streamed body's bytes, or null when absent, unreadable, or over 4 KB. */
async function boundedBody(
  r: Record<string, unknown>,
): Promise<Uint8Array | null> {
  const stream = r["body"] as ReadableStream<Uint8Array> | null | undefined;
  if (stream && typeof stream.getReader === "function") {
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let n = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        n += value.byteLength;
        if (n > MAX_ENVELOPE_BODY) {
          await reader.cancel();
          return null;
        }
        chunks.push(value);
      }
    } catch {
      return null;
    }
    return Buffer.concat(chunks);
  }
  return null;
}

function classifyTransport(
  error: unknown,
): { classification: string; retryable: boolean } | null {
  for (const node of walkCauseChain(error)) {
    const name = node["name"];
    if (name === "TimeoutError") {
      return { classification: "TIMEOUT", retryable: true };
    }
    const code = node["code"];
    if (typeof code === "string") {
      if (TIMEOUT_CODES.has(code)) {
        return { classification: "TIMEOUT", retryable: true };
      }
      if (CONNECTION_CODES.has(code)) {
        return { classification: "CONNECTION_ERROR", retryable: true };
      }
    }
  }
  return null;
}

function msg(providerType: string | undefined, detail: string): string {
  return `${providerType || "External"} API call failed: ${detail}`;
}

/**
 * Return structured metadata and a safe message, or `undefined` when the
 * error is not a recognized HTTP or transport failure.
 *
 * `credentials` are the request-local secret values applied for this call;
 * they are matched exactly against recognized provider text so a provider
 * echoing an unlabeled credential cannot surface it.
 */
export async function classifyToolAPIError(
  error: unknown,
  providerType?: string,
  credentials: readonly string[] = [],
): Promise<ClassifiedToolAPIError | undefined> {
  const transport = classifyTransport(error);
  if (transport) {
    return {
      toolApiError: {
        provider_type: providerType ?? null,
        classification: transport.classification,
        http_status: null,
        retryable: transport.retryable,
      },
      message: msg(providerType, transport.classification),
    };
  }

  const httpStatus = findHttpStatus(error);
  if (httpStatus !== null) {
    const response = findResponse(error);
    const [errorCode, reason] = await extractEnvelope(response, credentials);
    const [classification, retryable] = STATUS_CLASSIFICATIONS[httpStatus] ?? [
      "UNKNOWN",
      false,
    ];
    const base = msg(providerType, `HTTP ${httpStatus} ${classification}`);
    return {
      toolApiError: {
        provider_type: providerType ?? null,
        classification,
        http_status: httpStatus,
        retryable,
        error_code: errorCode,
        reason,
      },
      // The provider's own explanation is the fastest path to the cause; it
      // reaches the user and the agent, and is bounded/redacted above.
      message: reason ? `${base} — ${reason}` : base,
    };
  }

  return undefined;
}

function providerTypeFromToolDef(toolDef: unknown): string | undefined {
  const providerType =
    toolDef && typeof toolDef === "object"
      ? (toolDef as Record<string, unknown>)["provider_type"]
      : undefined;
  return typeof providerType === "string" ? providerType : undefined;
}

/** Error `ToolPodExecuteResponse` for a thrown tool, with classification when recognized. */
export async function executeErrorResponse(
  error: unknown,
  toolDef: unknown,
  podName: string,
  metadata: Record<string, unknown>,
  credentials: readonly string[] = [],
): Promise<ToolPodExecuteResponse> {
  const classified = await classifyToolAPIError(
    error,
    providerTypeFromToolDef(toolDef),
    credentials,
  );
  return {
    status: "error",
    error: classified
      ? classified.message
      : error instanceof Error
        ? error.message
        : String(error),
    pod_name: podName,
    kind: metadata["memory"] ? "memory" : undefined,
    metadata,
    ...(classified ? { tool_api_error: classified.toolApiError } : {}),
  };
}
