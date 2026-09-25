/**
 * Integration test for the TS runner's OTLP export leg.
 *
 * Runs the real OTLPTraceExporter (from `@opentelemetry/exporter-trace-otlp-http`)
 * against a real local HTTP server acting as the collector — no mocking of the
 * OTLP transport itself — so this proves spans actually leave the process over
 * the wire, and that metadata-only redaction holds in the real wire payload,
 * not just at the exporter-unit level. This is as close to the ticket's "verify
 * spans reach the platform-debug/customer OTLP destination" acceptance
 * criterion as a live tenant/cluster is not available in this environment;
 * live verification against a real dev-cell tenant is tracked separately.
 *
 * Protobuf bodies aren't decoded here — OTLP/HTTP protobuf string fields are
 * length-delimited raw UTF-8, so a substring search on the raw request body is
 * sufficient to prove presence/absence of specific content without pulling in
 * the full OTLP proto schema.
 */

import { describe, test, expect, afterEach } from "vitest";
import { createServer } from "node:http";
import type { Server } from "node:http";
import { propagation, trace } from "@opentelemetry/api";
import type { Collection } from "mongodb";

import { setupTracing, shutdownTracing, getTracer } from "../../src/index.js";

const REDACTED_SECRET = "must-not-cross-service-boundary-integration";
const METADATA_VALUE = "gpt-integration-test-model";

function startCollector(): Promise<{
  server: Server;
  url: string;
  bodies: () => Buffer[];
}> {
  const bodies: Buffer[] = [];
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        bodies.push(Buffer.concat(chunks));
        res.writeHead(200, { "Content-Type": "application/x-protobuf" });
        res.end();
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      resolve({
        server,
        url: `http://127.0.0.1:${port}/v1/traces`,
        bodies: () => bodies,
      });
    });
  });
}

describe("OTLP export integration", () => {
  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
    propagation.disable();
    delete process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"];
    delete process.env["AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE"];
  });

  test("a real span reaches the OTLP collector with content redacted (metadata-only default)", async () => {
    const { server, url, bodies } = await startCollector();
    try {
      process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = url;

      await setupTracing({ serviceName: "svc-otlp-integration" });

      const span = getTracer("test").startSpan("otlp-integration-span");
      span.setAttribute(
        "llm.input_messages.0.message.content",
        REDACTED_SECRET,
      );
      span.setAttribute("model.name", METADATA_VALUE);
      span.end();

      await shutdownTracing();

      expect(bodies().length).toBeGreaterThan(0);
      const combined = Buffer.concat(bodies()).toString("latin1");
      expect(combined).toContain("otlp-integration-span");
      expect(combined).toContain(METADATA_VALUE);
      expect(combined).not.toContain(REDACTED_SECRET);
    } finally {
      server.close();
    }
  });

  test("full content-capture mode sends the unredacted attribute over OTLP", async () => {
    const { server, url, bodies } = await startCollector();
    try {
      process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = url;
      process.env["AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE"] = "full";

      await setupTracing({ serviceName: "svc-otlp-full" });

      const span = getTracer("test").startSpan("otlp-full-span");
      span.setAttribute(
        "llm.input_messages.0.message.content",
        REDACTED_SECRET,
      );
      span.end();

      await shutdownTracing();

      const combined = Buffer.concat(bodies()).toString("latin1");
      expect(combined).toContain(REDACTED_SECRET);
    } finally {
      server.close();
    }
  });

  test("Mongo export stays unredacted and independent of the OTLP leg's redaction", async () => {
    const { server, url, bodies } = await startCollector();
    const insertMany = async (docs: unknown[]) => {
      mongoDocs.push(...(docs as Record<string, unknown>[]));
      return { insertedCount: docs.length };
    };
    const mongoDocs: Record<string, unknown>[] = [];
    const collection = { insertMany } as unknown as Collection;

    try {
      process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = url;

      // setupTracing's mongodbUri path builds its own MongoClient; this test
      // only needs to prove the *exporter* stays unredacted, so it exercises
      // MongoDBSpanExporter directly against a fake collection rather than
      // standing up a real Mongo connection here.
      const { MongoDBSpanExporter } = await import("../../src/index.js");
      await setupTracing({ serviceName: "svc-otlp-mongo-independent" });

      const span = getTracer("test").startSpan("otlp-mongo-span");
      span.setAttribute(
        "llm.input_messages.0.message.content",
        REDACTED_SECRET,
      );
      span.end();
      await shutdownTracing();

      // OTLP leg redacted it.
      const otlpBody = Buffer.concat(bodies()).toString("latin1");
      expect(otlpBody).not.toContain(REDACTED_SECRET);

      // Mongo's own exporter — independent of the OTLP ContentPolicy wrapper —
      // never redacts (customer's own BYOC DB is the canonical unredacted
      // store).
      const mongoExporter = new MongoDBSpanExporter(collection);
      await new Promise<void>((resolve) => {
        mongoExporter.export(
          [
            {
              name: "mongo-span",
              spanContext: () => ({
                traceId: "0123456789abcdef0123456789abcdef",
                spanId: "0123456789abcdef",
                traceFlags: 1,
                isRemote: false,
              }),
              kind: 0,
              startTime: [1, 0],
              endTime: [2, 0],
              status: { code: 1 },
              attributes: {
                "llm.input_messages.0.message.content": REDACTED_SECRET,
              },
              resource: { attributes: {} },
              events: [],
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
            } as any,
          ],
          () => resolve(),
        );
      });
      expect(mongoDocs[0]?.["attributes"]).toMatchObject({
        "llm.input_messages.0.message.content": REDACTED_SECRET,
      });
    } finally {
      server.close();
    }
  });
});
