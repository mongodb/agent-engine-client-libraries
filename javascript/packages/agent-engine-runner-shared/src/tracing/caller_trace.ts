import {
  context,
  createContextKey,
  isSpanContextValid,
  trace,
  type Context,
  type Span,
  type SpanContext,
} from "@opentelemetry/api";

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
