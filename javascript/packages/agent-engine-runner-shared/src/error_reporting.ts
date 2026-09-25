/**
 * Shared local error reporting helpers for CLI-generated runtime flows.
 *
 * Mirrors `agent_engine_runner_shared/error_reporting.py`. Reports agent runtime/build
 * failures to Sentry, gated by env (`AGENTIC_SENTRY_ENABLED=1` + a DSN). All
 * outbound payloads are scrubbed for secrets and the caller's home directory
 * via `beforeSend` plus per-value redaction before capture.
 *
 * `@sentry/node` is an *optional* dependency (a devDependency here for types
 * and tests): it is loaded via a guarded dynamic import only when reporting is
 * enabled, so agents that never flip `AGENTIC_SENTRY_ENABLED` never install or
 * pay for it. When the module isn't present, or anything in the reporting path
 * faults, every function degrades to a no-op — reporting must never block an
 * agent from starting or mask its real failure.
 */

import { homedir } from "node:os";

// Type-only import (erased at build): keeps the internal `sentry` handle typed
// without making @sentry/node a value dependency — it's loaded lazily below.
import type * as SentryNode from "@sentry/node";

// Public API stays decoupled from @sentry/node's types so consumers needn't
// install it. Local mirror of Sentry's severity levels.
type SeverityLevel = "fatal" | "error" | "warning" | "log" | "info" | "debug";

// The lazily-imported @sentry/node module; null until a successful enable.
let sentry: typeof SentryNode | null = null;
let enabled = false;

const MAX_SUMMARY_LINE_LENGTH = 240;
const MAX_OUTPUT_TAIL_CHARS = 8 * 1024;
// Cap redaction recursion so a self-referential or pathologically nested
// caller-supplied `extra` can't drive `redactValue` into a stack overflow.
const MAX_REDACT_DEPTH = 12;

// `key=value` / `key: value` inline secrets. Case-insensitive; keeps the key
// and separator, redacts the value up to the next whitespace/comma/semicolon.
const SECRET_INLINE_PATTERN =
  /\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)(=|:)\s*([^\s,;]+)/gi;
const BEARER_PATTERN = /\b(bearer)\s+[A-Za-z0-9._-]+/gi;
const URL_USERINFO_PATTERN = /\b([a-z][a-z0-9+.-]*:\/\/)[^:/\s@]+:[^@\s/]+@/gi;
// Bare-token form (scheme://token@host, no ":" separator) — e.g. a GitHub PAT
// embedded as `https://ghp_xxx@github.com`. Applied after URL_USERINFO_PATTERN
// so a user:pass@ pair is never double-matched by this looser pattern.
const URL_USERINFO_BARE_PATTERN = /\b([a-z][a-z0-9+.-]*:\/\/)[^:/\s@]+@/gi;

function resolveHomeDir(): string {
  try {
    return homedir().trim();
  } catch {
    return (process.env["HOME"] ?? "").trim();
  }
}

let homeDir = resolveHomeDir();

/** @internal — test-only override of the resolved home directory. */
export function setHomeDirForTest(dir: string): void {
  homeDir = dir;
}

export interface InitErrorReportingArgs {
  surface: string;
  component?: string | null;
  mode?: string | null;
}

/**
 * Initialize Sentry error reporting. No-op (returns `false`) unless
 * `AGENTIC_SENTRY_ENABLED=1` and `AGENTIC_SENTRY_DSN` are both set, and
 * `@sentry/node` is installed. Tags every event with `surface` and, when
 * provided, `component` / `mode`.
 *
 * Never throws: a missing module or a faulting `Sentry.init` leaves reporting
 * disabled rather than propagating — enabling observability must not turn into
 * a boot blocker.
 */
export async function initErrorReporting(
  args: InitErrorReportingArgs,
): Promise<boolean> {
  enabled = false;
  sentry = null;
  const dsn = (process.env["AGENTIC_SENTRY_DSN"] ?? "").trim();
  if (!(process.env["AGENTIC_SENTRY_ENABLED"] === "1" && dsn)) return false;

  try {
    sentry = await import("@sentry/node");
  } catch {
    // @sentry/node not installed — reporting is opt-in, so stay disabled.
    return false;
  }

  try {
    sentry.init({
      dsn,
      environment: process.env["AGENTIC_SENTRY_ENVIRONMENT"] ?? "local-dev",
      release: process.env["AGENTIC_SENTRY_RELEASE"] || undefined,
      tracesSampleRate: 0.0,
      debug: process.env["AGENTIC_SENTRY_DEBUG"] === "1",
      // Our scrubber operates on a generic event dict and returns the same
      // (mutated) object; narrow the return to the callback's event type.
      beforeSend: (event, hint) =>
        beforeSend(
          event as unknown as Record<string, unknown>,
          hint,
        ) as unknown as typeof event,
    });
    sentry.setTag("surface", args.surface);
    if (args.component) sentry.setTag("component", args.component);
    if (args.mode) sentry.setTag("mode", args.mode);
  } catch {
    // A malformed DSN/environment or any init fault disables reporting rather
    // than crashing the agent at boot.
    sentry = null;
    return false;
  }

  enabled = true;
  return true;
}

export interface CaptureExceptionArgs {
  summary?: string | null;
  extra?: Record<string, unknown>;
}

/**
 * Capture an exception. When `summary` is given, the reported error message is
 * the (redacted) summary with the original exception's type/message preserved
 * as scoped extras — matching Python's `_exception_for_capture`.
 */
export function captureException(
  exc: unknown,
  args: CaptureExceptionArgs = {},
): void {
  if (!enabled || !sentry) return;
  const s = sentry;
  // Reporting must never throw into the caller — a fault here would mask the
  // very failure the caller is trying to report.
  try {
    const { summary, extra } = args;
    const captured = exceptionForCapture(exc, summary);
    s.withScope((scope) => {
      if (summary) {
        scope.setExtra(
          "original_exception_type",
          exc instanceof Error ? exc.name : typeof exc,
        );
        scope.setExtra("original_exception_message", redactText(String(exc)));
      }
      for (const [key, value] of Object.entries(extra ?? {})) {
        scope.setExtra(key, redactValue(value));
      }
      s.captureException(captured);
    });
  } catch {
    /* swallow — best-effort reporting */
  }
}

export function captureMessage(
  message: string,
  level: SeverityLevel = "error",
  extra: Record<string, unknown> = {},
): void {
  if (!enabled || !sentry) return;
  const s = sentry;
  try {
    s.withScope((scope) => {
      for (const [key, value] of Object.entries(extra)) {
        scope.setExtra(key, redactValue(value));
      }
      s.captureMessage(redactText(message), level);
    });
  } catch {
    /* swallow — best-effort reporting */
  }
}

export async function flush(timeoutMs = 2000): Promise<void> {
  if (!enabled || !sentry) return;
  try {
    await sentry.flush(timeoutMs);
  } catch {
    /* swallow — best-effort reporting */
  }
}

// =============================================================================
// Subprocess-failure summarization (build/spawn flows)
// =============================================================================

export interface SubprocessFailure {
  /** The command that failed — array (argv) or a single string. */
  command: string | readonly string[];
  /** Process exit code, if known. */
  exitCode?: number | null;
  stdout?: string | Uint8Array | null;
  stderr?: string | Uint8Array | null;
}

/**
 * A one-line summary of a failed subprocess: the command label plus the most
 * meaningful line of its output, or the exit code when no output stands out.
 */
export function summarizeSubprocessFailure(exc: SubprocessFailure): string {
  const command = commandLabel(exc.command);
  const headline = failureHeadline(subprocessOutputTail(exc));
  if (headline) return `${command} failed: ${headline}`;
  return `${command} failed with exit ${exc.exitCode ?? "unknown"}`;
}

/** The trailing `MAX_OUTPUT_TAIL_CHARS` of a subprocess's stdout+stderr. */
export function subprocessOutputTail(exc: SubprocessFailure): string {
  const chunks: string[] = [];
  for (const value of [exc.stdout, exc.stderr]) {
    if (value === null || value === undefined) continue;
    chunks.push(
      typeof value === "string" ? value : Buffer.from(value).toString("utf-8"),
    );
  }
  if (chunks.length === 0) return "";
  return tailText(chunks.join("\n"));
}

// =============================================================================
// Redaction — internal, but pure and unit-tested directly
// =============================================================================

/** @internal — exported only for tests; not part of the public API. */
export function beforeSend(
  event: Record<string, unknown>,
  _hint?: unknown,
): Record<string, unknown> {
  const data = event;
  data["server_name"] = null;
  if ("message" in data) {
    data["message"] = redactText(String(data["message"]));
  }
  for (const section of [
    "request",
    "user",
    "tags",
    "extra",
    "contexts",
    "exception",
    "breadcrumbs",
    "threads",
    "logentry",
    "modules",
  ]) {
    const payload = data[section];
    if (payload !== null && typeof payload === "object") {
      data[section] = redactValue(payload);
    }
  }
  return event;
}

/**
 * Recursively redact strings in a value. Guards against caller-supplied
 * `extra` that is self-referential (WeakSet cycle check) or pathologically
 * deep (`MAX_REDACT_DEPTH`): either would otherwise overflow the stack and
 * take down the reporting path. A cycle/over-depth node is replaced with a
 * sentinel rather than recursed into.
 */
function redactValue(
  value: unknown,
  depth = 0,
  seen: WeakSet<object> = new WeakSet(),
): unknown {
  if (typeof value === "string") return redactText(value);
  if (value === null || typeof value !== "object") return value;

  if (seen.has(value)) return "[Circular]";
  if (depth >= MAX_REDACT_DEPTH) return "[MaxDepth]";
  seen.add(value);

  const result = Array.isArray(value)
    ? value.map((item) => redactValue(item, depth + 1, seen))
    : Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([k, v]) => [
          k,
          redactValue(v, depth + 1, seen),
        ]),
      );
  // Allow the same object reached again via a sibling branch (a DAG, not a
  // cycle) to serialize in full: only nodes on the current path are guarded.
  seen.delete(value);
  return result;
}

/** @internal — exported only for tests; not part of the public API. */
export function redactText(text: string): string {
  if (!text) return text;
  let redacted = text;
  if (homeDir) redacted = redacted.split(homeDir).join("<home>");
  redacted = redacted.replace(URL_USERINFO_PATTERN, "$1<redacted>:<redacted>@");
  redacted = redacted.replace(URL_USERINFO_BARE_PATTERN, "$1<redacted>@");
  redacted = redacted.replace(SECRET_INLINE_PATTERN, "$1$2<redacted>");
  redacted = redacted.replace(BEARER_PATTERN, "$1 <redacted>");
  return redacted;
}

function exceptionForCapture(exc: unknown, summary?: string | null): unknown {
  if (!summary) return exc;
  const wrapped = new Error(redactText(summary));
  // Preserve the original stack so Sentry groups on the real failure site.
  if (exc instanceof Error && exc.stack) wrapped.stack = exc.stack;
  return wrapped;
}

function commandLabel(cmd: string | readonly string[]): string {
  if (Array.isArray(cmd)) {
    const parts = cmd.map(String).filter((p) => p.trim());
    if (parts.length === 0) return "subprocess";
    if (parts.length >= 3 && parts[0] === "uv" && parts[1] === "pip") {
      return parts.slice(0, 3).join(" ");
    }
    if (parts.length >= 2) return parts.slice(0, 2).join(" ");
    return parts[0] as string;
  }
  const text = String(cmd).trim();
  return text || "subprocess";
}

function failureHeadline(output: string): string {
  const normalized = output
    .split("\n")
    .map(normalizeFailureLine)
    .filter((line) => line);
  for (let i = normalized.length - 1; i >= 0; i--) {
    const line = normalized[i] as string;
    const lower = line.toLowerCase();
    if (
      lower.includes("failed") ||
      lower.includes("error") ||
      lower.includes("not found") ||
      lower.includes("no matching") ||
      lower.includes("exception")
    ) {
      return truncateSummaryLine(line);
    }
  }
  if (normalized.length === 0) return "";
  return truncateSummaryLine(normalized[normalized.length - 1] as string);
}

function normalizeFailureLine(line: string): string {
  let cleaned = line
    .trim()
    .replace(/^[│>]+/, "")
    .trim();
  if (cleaned.startsWith("×")) cleaned = cleaned.slice(1).trim();
  if (cleaned.startsWith("╰─▶")) cleaned = cleaned.slice(3).trim();
  if (cleaned.toLowerCase().startsWith("error:"))
    cleaned = cleaned.slice(6).trim();
  return cleaned;
}

function tailText(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= MAX_OUTPUT_TAIL_CHARS) return trimmed;
  return trimmed.slice(-MAX_OUTPUT_TAIL_CHARS);
}

function truncateSummaryLine(line: string): string {
  if (line.length <= MAX_SUMMARY_LINE_LENGTH) return line;
  return line.slice(0, MAX_SUMMARY_LINE_LENGTH - 3) + "...";
}
