/**
 * Distinguish an inbound caller trace from a root a runner created.
 *
 * A propagated local root arrives as a valid remote span context. W3C
 * traceparent cannot say who created that root, so runners add
 * `agentengine=root` to tracestate when they create the trace. Guardrail
 * content is recorded only when that marker is absent.
 */

import {
  context,
  createContextKey,
  isSpanContextValid,
  trace,
  type Context,
  type Span,
  type SpanContext,
} from "@opentelemetry/api";

/** Simple W3C tracestate member set when a runner creates the trace root. */
export const RUNNER_ROOT_TRACESTATE_KEY = "agentengine";
export const RUNNER_ROOT_TRACESTATE_VALUE = "root";

const RUNNER_ROOTED_TRACE_KEY = createContextKey(
  "agentengine.runner_rooted_trace",
);

export function traceStateMarksRunnerRoot(
  spanContext: SpanContext | undefined,
): boolean {
  return (
    spanContext?.traceState?.get(RUNNER_ROOT_TRACESTATE_KEY) ===
    RUNNER_ROOT_TRACESTATE_VALUE
  );
}

/** True when this span context was extracted from an inbound caller trace. */
export function inboundCallerSpanContext(
  spanContext: SpanContext | undefined,
): boolean {
  if (
    !spanContext ||
    !isSpanContextValid(spanContext) ||
    spanContext.isRemote !== true
  ) {
    return false;
  }
  return !traceStateMarksRunnerRoot(spanContext);
}

export function mergeRunnerRootTraceState(
  existing: string | undefined,
): string {
  const prefix = `${RUNNER_ROOT_TRACESTATE_KEY}=`;
  const kept = (existing ?? "")
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0 && !part.startsWith(prefix));
  return [`${prefix}${RUNNER_ROOT_TRACESTATE_VALUE}`, ...kept].join(",");
}

export function parentContextIsRunnerRooted(parent: Context): boolean {
  const spanContext = trace.getSpanContext(parent);
  if (!spanContext || !isSpanContextValid(spanContext)) {
    return true;
  }
  if (traceStateMarksRunnerRoot(spanContext)) {
    return true;
  }
  if (spanContext.isRemote === true) {
    return false;
  }
  return true;
}

export function bindRunnerRootedTrace(ctx: Context, rooted: boolean): Context {
  return ctx.setValue(RUNNER_ROOTED_TRACE_KEY, rooted);
}

function parentSpanContext(span: Span): SpanContext | undefined {
  const candidate = span as Span & {
    parentSpanContext?: SpanContext;
    parent?: SpanContext;
  };
  return candidate.parentSpanContext ?? candidate.parent;
}

/**
 * True when the active trace was created by a runner.
 *
 * An explicit binding from the span that joined or rooted the trace wins.
 * Otherwise a remote parent without the marker is an inbound caller, and a
 * local root (or a child of one) is not.
 */
export function activeTraceIsRunnerRooted(
  active: Context = context.active(),
): boolean {
  const explicit = active.getValue(RUNNER_ROOTED_TRACE_KEY);
  if (explicit === true) {
    return true;
  }
  if (explicit === false) {
    return false;
  }
  const span = trace.getSpan(active);
  if (!span) {
    return false;
  }
  const spanContext = span.spanContext();
  if (traceStateMarksRunnerRoot(spanContext)) {
    return true;
  }
  const parent = parentSpanContext(span);
  if (parent && isSpanContextValid(parent)) {
    if (traceStateMarksRunnerRoot(parent)) {
      return true;
    }
    if (parent.isRemote === true) {
      return false;
    }
    return true;
  }
  if (!isSpanContextValid(spanContext)) {
    return false;
  }
  if (spanContext.isRemote === true) {
    return traceStateMarksRunnerRoot(spanContext);
  }
  return true;
}
