/**
 * Integration test for the TS runner's OTLP export leg.
 *
 * Runs the real OTLPTraceExporter (from `@opentelemetry/exporter-trace-otlp-proto`)
 * against a real local HTTP server acting as the collector — no mocking of the
 * OTLP transport itself — so this proves spans actually leave the process over
 * the wire, and that metadata-only redaction holds in the real wire payload,
 * not just at the exporter-unit level. This is as close to the ticket's "verify
 * spans reach the platform-debug/customer OTLP destination" acceptance
 * criterion as a live tenant/cluster is not available in this environment;
 * live verification against a real dev-cell tenant is tracked separately.
 *
 * The collector records each request's Content-Type, because the OE intake
 * accepts only application/x-protobuf and rejects any other format with 415.
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

import { setupTracing, shutdownTracing, getTracer } from "../../src/index.js";

const REDACTED_SECRET = "must-not-cross-service-boundary-integration";
const METADATA_VALUE = "gpt-integration-test-model";

function startCollector(): Promise<{
  server: Server;
  url: string;
  bodies: () => Buffer[];
  contentTypes: () => (string | undefined)[];
}> {
  const bodies: Buffer[] = [];
  const contentTypes: (string | undefined)[] = [];
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        bodies.push(Buffer.concat(chunks));
        contentTypes.push(req.headers["content-type"]);
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
        contentTypes: () => contentTypes,
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
    delete process.env["OE_URL"];
  });

  test("a real span reaches the OTLP collector with content redacted (metadata-only default)", async () => {
    const { server, url, bodies, contentTypes } = await startCollector();
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
      for (const contentType of contentTypes()) {
        expect(contentType).toBe("application/x-protobuf");
      }
      const combined = Buffer.concat(bodies()).toString("latin1");
      expect(combined).toContain("svc-otlp-integration");
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

  test("platform export stays unredacted and independent of the OTLP leg's redaction", async () => {
    const collector = await startCollector();
    const oe = await startCollector();
    try {
      process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = collector.url;
      process.env["OE_URL"] = new URL(oe.url).origin;

      await setupTracing({ serviceName: "svc-otlp-platform-independent" });

      const span = getTracer("test").startSpan("otlp-platform-span");
      span.setAttribute(
        "llm.input_messages.0.message.content",
        REDACTED_SECRET,
      );
      span.end();
      await shutdownTracing();

      const otlpBody = Buffer.concat(collector.bodies()).toString("latin1");
      expect(otlpBody).toContain("otlp-platform-span");
      expect(otlpBody).not.toContain(REDACTED_SECRET);

      // The OE stores the trace the playground shows, so it gets the content.
      const platformBody = Buffer.concat(oe.bodies()).toString("latin1");
      expect(platformBody).toContain("otlp-platform-span");
      expect(platformBody).toContain(REDACTED_SECRET);
    } finally {
      collector.server.close();
      oe.server.close();
    }
  });
});
