/**
 * Inbound W3C trace-context extraction for request boundaries.
 *
 * The OE injects a `traceparent` on every tool-pod and AER call. Without
 * extracting it, the spans produced while a tool or routed LLM call runs
 * (framework instrumentation, outbound HTTP) root a disconnected trace instead
 * of joining the invocation's. Mirrors Python's
 * `ToolServer._run_with_inbound_trace_context` /
 * `_stream_with_inbound_trace_context` and the AER's
 * `_execute_with_inbound_trace_context`.
 */

import {
  ROOT_CONTEXT,
  SpanStatusCode,
  context,
  defaultTextMapGetter,
  trace,
  type Attributes,
  type Span,
} from "@opentelemetry/api";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import {
  bindRunnerRootedTrace,
  parentContextIsRunnerRooted,
} from "../tracing/caller_trace.js";
import { getTracer } from "../tracing/setup.js";
import {
  OPENINFERENCE_SPAN_KIND,
  type OpenInferenceSpanKind,
} from "../span_names.js";

// Local propagator instance (matching guardrail_span.ts) rather than the
// global one, so extraction never depends on whatever the process registered.
const traceContextPropagator = new W3CTraceContextPropagator();

export type InboundHeaders = Record<string, string | string[] | undefined>;

export interface InboundSpanSpec {
  /** Span name, shared with the Python counterpart so both render one span. */
  name: string;
  /** OpenInference kind the span declares, so a backend can classify it. */
  kind: OpenInferenceSpanKind;
  /** Instrumentation scope, matching the Python counterpart's `get_tracer`. */
  tracerName: string;
  /** Extra attributes set at span start (e.g. the execution id). */
  attributes?: Attributes;
}

/**
 * Extract the inbound traceparent and bind runner-root provenance.
 *
 * A valid traceparent is not an inbound caller by itself: a runner roots a
 * trace with the `agentengine=root` marker and propagates it, so it arrives
 * here remote. Nested platform calls use the binding to tell the two apart.
 */
function boundParentContext(headers: InboundHeaders) {
  const parent = traceContextPropagator.extract(
    ROOT_CONTEXT,
    headers,
    defaultTextMapGetter,
  );
  return bindRunnerRootedTrace(parent, parentContextIsRunnerRooted(parent));
}

function recordSpanError(span: Span, err: unknown): void {
  span.recordException(err instanceof Error ? err : String(err));
  span.setStatus({
    code: SpanStatusCode.ERROR,
    message: err instanceof Error ? err.message : String(err),
  });
}

/**
 * Run `handler` inside a span that joins the inbound trace, or roots a fresh
 * one when no traceparent arrived. The span is handed to `handler` so callers
 * that must stamp identity resolved *after* the span opened (the AER's session
 * and workspace ids) act on the run root explicitly, rather than reading
 * `trace.getActiveSpan()` and silently landing on whatever span is active.
 * Always ends the span; records the exception and sets ERROR status on failure.
 */
export function runWithInboundTraceContext<T>(
  spec: InboundSpanSpec,
  headers: InboundHeaders,
  handler: (span: Span) => Promise<T>,
): Promise<T> {
  const bound = boundParentContext(headers);
  const tracer = getTracer(spec.tracerName);
  return tracer.startActiveSpan(
    spec.name,
    {
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: spec.kind,
        ...spec.attributes,
      },
    },
    bound,
    async (span) => {
      try {
        return await handler(span);
      } catch (err) {
        recordSpanError(span, err);
        throw err;
      } finally {
        span.end();
      }
    },
  );
}

/**
 * Streaming variant of `runWithInboundTraceContext`.
 *
 * The SSE body runs after the route handler returns, so the span must wrap the
 * generator itself: it starts under the extracted parent when iteration begins
 * and ends when the stream is exhausted, errors, or the consumer stops early.
 * Each pull runs inside the span's context so provider spans created while the
 * source advances nest under it.
 *
 * `source` is a factory handed the boundary span, mirroring the non-streaming
 * variant's handler. A source that converts failures into in-band frames (the
 * LLM stream's `error` event) rather than throwing can then mark its own
 * boundary ERROR, so a failed call is not rendered as OK.
 */
export async function* streamWithInboundTraceContext<T>(
  spec: InboundSpanSpec,
  headers: InboundHeaders,
  source: (span: Span) => AsyncIterable<T>,
): AsyncGenerator<T> {
  const bound = boundParentContext(headers);
  const tracer = getTracer(spec.tracerName);
  const span = tracer.startSpan(
    spec.name,
    {
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: spec.kind,
        ...spec.attributes,
      },
    },
    bound,
  );
  const spanContext = trace.setSpan(bound, span);
  const iterator = source(span)[Symbol.asyncIterator]();
  try {
    while (true) {
      const next = await context.with(spanContext, () => iterator.next());
      if (next.done === true) return;
      yield next.value;
    }
  } catch (err) {
    recordSpanError(span, err);
    throw err;
  } finally {
    // Close the source under the boundary context and before ending the span,
    // so any spans it emits while unwinding (disconnect cleanup, drain release)
    // still nest under it instead of rooting a disconnected trace.
    await context.with(spanContext, () => closeIterator(iterator));
    span.end();
  }
}

async function closeIterator<T>(iterator: AsyncIterator<T>): Promise<void> {
  if (typeof iterator.return !== "function") return;
  try {
    await iterator.return();
  } catch {
    // Best-effort: the consumer already stopped pulling, so a close failure
    // must not mask the real stream outcome.
  }
}
