/**
 * The TypeScript platform exporter sends the payload the Python one sends.
 *
 * Both runner test suites decode their own request and compare it with the
 * shared fixture, so a field one runner drops or renames fails here instead of
 * in the OE.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, test, vi } from "vitest";
import { resourceFromAttributes } from "@opentelemetry/resources";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
// The generated protobuf root has no public export; the test needs it to decode.
// eslint-disable-next-line @typescript-eslint/no-require-imports
import otlpRoot from "@opentelemetry/otlp-transformer/build/src/generated/root.js";

import { PlatformSpanExporter } from "../../src/tracing/exporters.js";

const FIXTURE = fileURLToPath(
  new URL(
    "../../../../../shared-test-fixtures/runner_trace_payload.json",
    import.meta.url,
  ),
);

const TRACE_ID = "0af7651916cd43dd8448eb211c80319c";

function fixedSpan(): ReadableSpan {
  return {
    spanContext: () => ({
      traceId: TRACE_ID,
      spanId: "b7ad6b7169203331",
      traceFlags: 1,
      isRemote: false,
    }),
    parentSpanContext: {
      traceId: TRACE_ID,
      spanId: "00f067aa0ba902b7",
      traceFlags: 1,
    },
    name: "agent.run",
    kind: 0,
    startTime: [1700000000, 1],
    endTime: [1700000001, 500000002],
    duration: [1, 500000001],
    status: { code: 1 },
    attributes: {
      "session.id": "sess-1",
      "execution.id": "exec-1",
      "thread.id": "sess-1",
      "workspace.id": "ws-1",
      "llm.token_count.total": 42,
      "custom.flag": true,
      ratio: 0.5,
      "user.note": "contact jane@x.com",
    },
    resource: resourceFromAttributes({
      "service.name": "runner-sdk",
      "agentic_platform.project_id": "proj-1", // open-source-refs:ignore — data contract
    }),
    instrumentationScope: { name: "test" },
    events: [],
    links: [],
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

type ProtoAttr = { key: string; value: Record<string, unknown> };

function anyValue(value: Record<string, unknown>): unknown {
  for (const field of [
    "stringValue",
    "intValue",
    "boolValue",
    "doubleValue",
  ] as const) {
    if (value[field] !== undefined && value[field] !== null) {
      return field === "intValue" ? Number(value[field]) : value[field];
    }
  }
  throw new Error(`unexpected attribute value ${JSON.stringify(value)}`);
}

function attributes(pairs: ProtoAttr[]): Record<string, unknown> {
  return Object.fromEntries(pairs.map((p) => [p.key, anyValue(p.value)]));
}

interface RequestCodec {
  decode(bytes: Uint8Array): unknown;
  toObject(message: unknown, options: object): unknown;
}

interface OtlpRoot {
  opentelemetry: {
    proto: {
      collector: { trace: { v1: { ExportTraceServiceRequest: RequestCodec } } };
    };
  };
}

function normalize(body: Uint8Array): { spans: unknown[] } {
  const proto = (otlpRoot as unknown as OtlpRoot).opentelemetry.proto.collector
    .trace.v1.ExportTraceServiceRequest;
  const request = proto.toObject(proto.decode(body), {
    longs: String,
    enums: String,
    bytes: String,
    defaults: true,
  }) as {
    resourceSpans: Array<{
      resource: { attributes: ProtoAttr[] };
      scopeSpans: Array<{ spans: Array<Record<string, unknown>> }>;
    }>;
  };
  const hex = (b64: string) => Buffer.from(b64, "base64").toString("hex");
  const spans: unknown[] = [];
  for (const rs of request.resourceSpans) {
    const resource = attributes(rs.resource.attributes);
    for (const ss of rs.scopeSpans) {
      for (const span of ss.spans) {
        spans.push({
          trace_id: hex(span["traceId"] as string),
          span_id: hex(span["spanId"] as string),
          parent_span_id: hex(span["parentSpanId"] as string),
          name: span["name"],
          kind: span["kind"],
          start_time_unix_nano: span["startTimeUnixNano"],
          end_time_unix_nano: span["endTimeUnixNano"],
          status_code: (span["status"] as { code: string }).code,
          attributes: attributes(span["attributes"] as ProtoAttr[]),
          resource,
        });
      }
    }
  }
  return { spans };
}

describe("platform trace payload parity", () => {
  test("payload matches the fixture the Python exporter is also tested against", async () => {
    const bodies: Uint8Array[] = [];
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      bodies.push(new Uint8Array(init.body as Uint8Array));
      return new Response(null, { status: 200 });
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
    });

    const result = await new Promise<{ code: number }>((resolve) =>
      exporter.export([fixedSpan()], resolve),
    );

    expect(result.code).toBe(0);
    expect(bodies).toHaveLength(1);
    const expected = JSON.parse(readFileSync(FIXTURE, "utf8")) as {
      spans: unknown[];
    };
    expect(normalize(bodies[0] as Uint8Array)).toEqual({
      spans: expected.spans,
    });
  });
});
