/**
 * The structured-logging layout: turns a `LoggingEvent` into the single-line JSON
 * envelope (timestamp, level, logger, message, service, tenant/project/
 * workspace/execution/session ids, pod, source, fields).
 */

import { basename } from "node:path";

import type { LoggingEvent } from "log4js";

import {
  getCurrentExecutionId,
  getCurrentLogOrigin,
  getCurrentTraceId,
  getCurrentSessionId,
  getCurrentWorkspaceId,
} from "../context.js";
import { DEFAULT_RECORD_SOURCE, MAX_TRACEBACK_BYTES } from "./constants.js";
import type { FormatterEnv } from "./env.js";
import { jsonSafeStringify, truncateUtf8 } from "./serialize.js";

// log4js spells some levels differently from Python's stdlib `logging`
// (WARN vs WARNING, FATAL vs CRITICAL, TRACE has no stdlib peer). The Splunk
// dashboards and the playground UI logs tab were built against the Python
// SDK's level strings, so normalize log4js names to the stdlib names here —
// otherwise a TypeScript agent's warnings would ship as `"WARN"` and miss a
// `level=WARNING` filter / render inconsistently next to Python agents.
const LEVEL_NAME_BY_LOG4JS: Record<string, string> = {
  WARN: "WARNING",
  FATAL: "CRITICAL",
  TRACE: "DEBUG",
};

function normalizeLevelName(levelStr: string): string {
  return LEVEL_NAME_BY_LOG4JS[levelStr] ?? levelStr;
}

interface ParsedMessage {
  message: string;
  extra: Record<string, unknown>;
  err: Error | undefined;
}

/**
 * Split log4js's `event.data` into a message string, an `extra` field bag,
 * and an optional `Error`. Matches the common call shape used in this
 * codebase: `logger.info({...extra}, "message")`. If no leading object is
 * present, all args are stringified and joined.
 */
function parseEventData(data: unknown[]): ParsedMessage {
  let extra: Record<string, unknown> = {};
  let parts: unknown[] = data;
  let err: Error | undefined;

  const first = data[0];
  if (
    first &&
    typeof first === "object" &&
    !(first instanceof Error) &&
    !Array.isArray(first)
  ) {
    extra = { ...(first as Record<string, unknown>) };
    parts = data.slice(1);
  } else if (first instanceof Error) {
    err = first;
    parts = data.slice(1);
  }

  const extraErr = extra["err"];
  if (extraErr instanceof Error) {
    err = extraErr;
    delete extra["err"];
  }

  const message = parts
    .map((p) => {
      if (typeof p === "string") return p;
      // Use the cycle/BigInt-safe serializer — a BigInt or circular value in a
      // non-leading message arg must not throw and take down the log line.
      try {
        return jsonSafeStringify(p) ?? String(p);
      } catch {
        return String(p);
      }
    })
    .join(" ");
  return { message, extra, err };
}

export function buildStructuredLayout(
  env: FormatterEnv,
): (event: LoggingEvent) => string {
  // Tenant context is read from the AsyncLocalStorage contextvars in
  // `context.ts` at layout time so records emitted from inside an
  // execution carry executionId/sessionId automatically; records emitted
  // outside an execution (startup, idle) leave those fields `null`.
  return (event: LoggingEvent): string => {
    const { message, extra, err } = parseEventData(event.data);
    // Intentional divergence from the Python formatter: structured_logging.py
    // ignores `extra=` kwargs on `logger.info(...)`, so Python callers have
    // no way to attach structured key/value pairs to `fields`. log4js's
    // idiomatic call shape *is* `logger.info({...extra}, "msg")`, so we
    // spread the first-arg object into `fields` and let TS callers do what
    // Python callers cannot. The `source` key is special-cased below and
    // stripped from `fields` so it never appears twice on the wire.
    const fields: Record<string, unknown> = {
      component: env.component,
      ...extra,
    };
    if (err) {
      // Capture the formatted stack so on-call can debug from the
      // archived JSON without reproducing. Truncated to keep S3 batch
      // sizes manageable; truncation is marked so consumers know the
      // tail was cut.
      fields["exc_type"] = err.name;
      fields["exc_message"] = err.message;
      if (err.stack)
        fields["exc_traceback"] = truncateUtf8(err.stack, MAX_TRACEBACK_BYTES);
    }
    const source =
      typeof extra["source"] === "string"
        ? (extra["source"] as string)
        : DEFAULT_RECORD_SOURCE;
    delete fields["source"];

    // Match Python's `record.filename` / `record.lineno` / `record.funcName`
    // (structured_logging.py:125-128): include caller-site info only for
    // logger-emitted records, not for stdout/stderr capture lines whose
    // call site is the producer that wrote to the stream — meaningless for
    // debugging. log4js populates these on LoggingEvent only when
    // `enableCallStack: true` is set on the category (done in the
    // configuration below); generating a stack trace per record is not free
    // but the volume is bounded by LOG_LEVEL filtering on the default
    // category, which keeps the cost acceptable in the steady state.
    if (source === DEFAULT_RECORD_SOURCE) {
      if (event.fileName)
        // Python's `LogRecord.filename` is the basename, not the full path.
        // log4js returns the absolute path (or a `file://` URL under ESM);
        // `basename` handles both string forms.
        fields["filename"] = basename(event.fileName);
      if (typeof event.lineNumber === "number")
        fields["lineno"] = event.lineNumber;
      if (event.functionName) fields["funcName"] = event.functionName;
    }

    const origin = getCurrentLogOrigin();
    const traceId = getCurrentTraceId();
    return jsonSafeStringify({
      timestamp: event.startTime.toISOString(),
      level: normalizeLevelName(event.level.levelStr),
      logger: event.categoryName,
      message,
      service: env.service,
      tenantId: env.tenantId,
      projectId: env.projectId,
      workspaceId: getCurrentWorkspaceId() ?? env.workspaceIdEnv,
      executionId: getCurrentExecutionId(),
      sessionId: getCurrentSessionId(),
      bootId: env.bootId,
      podName: env.podName,
      source,
      fields,
      ...(origin !== null ? { origin } : {}),
      ...(traceId &&
      /^[0-9a-f]{32}$/.test(traceId) &&
      traceId !== "0".repeat(32)
        ? { traceId }
        : {}),
    });
  };
}
