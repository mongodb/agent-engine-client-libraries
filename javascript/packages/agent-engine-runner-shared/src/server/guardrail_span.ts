import {
  ROOT_CONTEXT,
  SpanStatusCode,
  defaultTextMapGetter,
  isSpanContextValid,
  trace,
  type Span,
} from "@opentelemetry/api";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import {
  bindRunnerRootedTrace,
  inboundCallerSpanContext,
  parentContextIsRunnerRooted,
} from "../tracing/caller_trace.js";
import { getTracer } from "../tracing/setup.js";
import {
  OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind,
} from "../span_names.js";
import type {
  GuardrailCheckRequest,
  GuardrailCheckResponse,
} from "../models.js";

/** Name shared with the Python tool pod so both runners render one span. */
export const GUARDRAIL_CHECK_SPAN_NAME = "tool.guardrails_check";

// One large prompt must not reject the whole span document. The traces
// collection keeps this text; metadata-only export redacts input.value.
const GUARDRAIL_SPAN_TEXT_LIMIT = 4000;

const traceContextPropagator = new W3CTraceContextPropagator();

/**
 * Runs a guardrail evaluation inside a span that joins the invocation's trace.
 *
 * The OE injects a traceparent on every tool-pod call; without extracting it
 * the span would root a disconnected trace. Mirrors the Python tool pod's
 * `tool.guardrails_check` span and its bounded input/output attributes.
 */
export function evaluateGuardrailCheckTraced(
  headers: Record<string, string | string[] | undefined>,
  request: GuardrailCheckRequest,
  evaluate: (request: GuardrailCheckRequest) => GuardrailCheckResponse,
): GuardrailCheckResponse {
  const parent = traceContextPropagator.extract(
    ROOT_CONTEXT,
    headers,
    defaultTextMapGetter,
  );
  const spanContext = trace.getSpanContext(parent);
  // No caller trace means no span. Rooting one here would mint a disconnected
  // tool.guardrails_check trace on every check that arrives without a parent.
  if (!spanContext || !isSpanContextValid(spanContext)) {
    return evaluate(request);
  }
  // A valid traceparent is not an inbound caller by itself. AER roots
  // aer.execute with the flag off and tls_client propagates that root, so it
  // arrives here remote. Join the span either way; record content only when
  // a caller trace was already bound.
  const recordContent = inboundCallerSpanContext(spanContext);
  const bound = bindRunnerRootedTrace(
    parent,
    parentContextIsRunnerRooted(parent),
  );
  const tracer = getTracer("agent-engine-runner-shared.tool");
  return tracer.startActiveSpan(
    GUARDRAIL_CHECK_SPAN_NAME,
    { attributes: { [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL } },
    bound,
    (span) => {
      try {
        const response = evaluate(request);
        if (span.isRecording() && recordContent) {
          recordGuardrailSpanIO(span, request, response);
        }
        return response;
      } catch (err) {
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: err instanceof Error ? err.message : String(err),
        });
        throw err;
      } finally {
        span.end();
      }
    },
  );
}

function recordGuardrailSpanIO(
  span: Span,
  request: GuardrailCheckRequest,
  response: GuardrailCheckResponse,
): void {
  const text = request.input.text;
  const payloadOut: Record<string, unknown> = {
    decision: response.decision,
    allowed: response.allowed,
    triggered_policy_ids: [...response.triggered_policy_ids],
    reason: response.reason ?? null,
    evidence: response.evidence.map((item) => ({
      policy_id: item.policy_id,
      message: item.message,
    })),
  };
  const transformed = response.transformed_text;
  // Empty string is a real rewrite (a modify/redact policy cleared the text).
  // Only null or undefined means the evaluator left the field unset.
  if (transformed != null && transformed !== text) {
    payloadOut.transformed_text = transformed.slice(
      0,
      GUARDRAIL_SPAN_TEXT_LIMIT,
    );
    payloadOut.transformed_text_truncated =
      transformed.length > GUARDRAIL_SPAN_TEXT_LIMIT;
  }

  span.setAttribute(
    "input.value",
    JSON.stringify({
      stage: request.stage,
      text: text.slice(0, GUARDRAIL_SPAN_TEXT_LIMIT),
      text_truncated: text.length > GUARDRAIL_SPAN_TEXT_LIMIT,
      policies: request.policies.map((policy) => ({
        id: policy.id,
        type: policy.type,
        action: policy.action,
      })),
    }),
  );
  span.setAttribute("output.value", JSON.stringify(payloadOut));
  span.setStatus({ code: SpanStatusCode.OK });
}
