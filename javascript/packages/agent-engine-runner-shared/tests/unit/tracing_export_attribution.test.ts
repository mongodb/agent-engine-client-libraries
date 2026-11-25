import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { propagation, trace } from "@opentelemetry/api";

import {
  getTracer,
  runWithCustomerOrigin,
  runWithExecutionContext,
  setupTracing,
  shutdownTracing,
} from "../../src/index.js";
import { captureJsonLogs, type JsonLogCapture } from "./json_log_capture.js";

let logs: JsonLogCapture;
let exportedBodies: string[];

beforeEach(() => {
  vi.stubEnv("OE_URL", "http://oe.test:8000");
  exportedBodies = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init: RequestInit) => {
      exportedBodies.push(
        Buffer.from(init.body as Uint8Array).toString("latin1"),
      );
      return new Response(null, { status: 400 });
    }),
  );
  logs = captureJsonLogs();
});

afterEach(async () => {
  await shutdownTracing();
  trace.disable();
  propagation.disable();
  logs.restore();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

test("an export failure after an execution ends carries none of its identity", async () => {
  await setupTracing({ serviceName: "svc-export-attribution" });
  const failures = () =>
    logs.records.filter((r) =>
      r.message.startsWith("Trace export: OE returned a terminal status"),
    );

  const traceIds = {
    "exec-a": "0123456789abcdef0123456789abcdef",
    "exec-b": "abcdef0123456789abcdef0123456789",
  };
  for (const [executionId, traceId] of Object.entries(traceIds)) {
    const seen = failures().length;
    runWithExecutionContext(
      {
        executionId,
        sessionId: `sess-${executionId}`,
        traceId,
        wrapper: null,
        oeUrl: "http://oe.test:8000",
      },
      () =>
        runWithCustomerOrigin(() =>
          getTracer("test").startSpan(`span-${executionId}`).end(),
        ),
    );
    await vi.waitFor(() => expect(failures().length).toBeGreaterThan(seen), {
      timeout: 5000,
    });
  }

  for (const record of failures()) {
    expect(record).toMatchObject({ executionId: null, sessionId: null });
    expect(record).not.toHaveProperty("traceId");
    expect(record).not.toHaveProperty("origin");
  }
  for (const executionId of Object.keys(traceIds)) {
    expect(
      exportedBodies.some(
        (body) =>
          body.includes(`span-${executionId}`) &&
          body.includes(`sess-${executionId}`),
      ),
    ).toBe(true);
  }
});
