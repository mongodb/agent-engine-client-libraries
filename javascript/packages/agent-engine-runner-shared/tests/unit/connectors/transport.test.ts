import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { brotliCompressSync, createGzip, gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import { classifyToolAPIError } from "../../../src/tool_api_error.js";
import { ConnectorExecutor } from "../../../src/connectors/executor.js";
import {
  loadToolDefs,
  type Operation,
  type ToolDefinitions,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

function definitions(baseUrl: string): ToolDefinitions {
  const value = structuredClone(loadToolDefs(FIXTURE));
  value.source.base_url = baseUrl;
  value.auth = null;
  const operation: Operation = {
    name: "status",
    description: "",
    method: "GET",
    path: "/status",
    params: {
      path: [],
      query: [],
      header: [],
      cookie: [],
      body: [],
      wrapped: false,
      query_style: {},
      form_style: {},
    },
    inputSchema: { type: "object", properties: {} },
    body_encoding: "json",
  };
  value.tools = [operation];
  return value;
}

function definitionsWithBody(baseUrl: string): ToolDefinitions {
  const value = definitions(baseUrl);
  const operation = value.tools[0];
  if (operation === undefined) throw new Error("fixture has no operation");
  operation.params = {
    path: [],
    query: [],
    header: [],
    cookie: [],
    body: [],
    wrapped: true,
    query_style: {},
    form_style: {},
  };
  operation.inputSchema = {
    type: "object",
    properties: { body: { type: "object" } },
    required: ["body"],
  };
  return value;
}

async function serve(
  handler: Parameters<typeof createServer>[0],
): Promise<{ url: string; close: () => Promise<void> }> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      ),
  };
}

describe("ConnectorExecutor HTTP transport", () => {
  const cleanups: Array<() => Promise<void>> = [];
  afterEach(async () => {
    await Promise.all(cleanups.splice(0).map((cleanup) => cleanup()));
  });

  it("decodes compressed responses", async () => {
    const server = await serve((_request, response) => {
      response.writeHead(200, {
        "content-encoding": "gzip",
        "content-type": "text/plain",
      });
      response.end(gzipSync("ok"));
    });
    cleanups.push(server.close);
    const executor = new ConnectorExecutor(definitions(server.url));
    cleanups.push(() => executor.close());
    await expect(executor.execute("status", {})).resolves.toBe("ok");
  });

  it("sends a GET body through the production transport", async () => {
    let received = "";
    const server = await serve((request, response) => {
      expect(request.method).toBe("GET");
      request.setEncoding("utf8");
      request.on("data", (chunk: string) => {
        received += chunk;
      });
      request.on("end", () => {
        response.writeHead(200, { "content-type": "text/plain" });
        response.end("ok");
      });
    });
    cleanups.push(server.close);
    const executor = new ConnectorExecutor(definitionsWithBody(server.url));
    cleanups.push(() => executor.close());
    await expect(
      executor.execute("status", { body: { q: "search" } }),
    ).resolves.toBe("ok");
    expect(received).toBe('{"q":"search"}');
  });

  it("decodes stacked content encodings", async () => {
    const server = await serve((_request, response) => {
      response.writeHead(200, {
        "content-encoding": "gzip, br",
        "content-type": "text/plain",
      });
      response.end(brotliCompressSync(gzipSync("ok")));
    });
    cleanups.push(server.close);
    const executor = new ConnectorExecutor(definitions(server.url));
    cleanups.push(() => executor.close());
    await expect(executor.execute("status", {})).resolves.toBe("ok");
  });

  it("returns an empty body for bodyless gzip-advertising HEAD responses", async () => {
    const server = await serve((_request, response) => {
      response.writeHead(200, { "content-encoding": "gzip" });
      response.end();
    });
    cleanups.push(server.close);
    const value = definitions(server.url);
    const first = value.tools[0];
    if (first === undefined) throw new Error("fixture has no operation");
    value.tools[0] = { ...first, method: "HEAD" };
    const executor = new ConnectorExecutor(value);
    cleanups.push(() => executor.close());
    await expect(executor.execute("status", {})).resolves.toBe("");
  });

  it("rejects malformed compressed responses without leaking their body", async () => {
    const server = await serve((_request, response) => {
      response.writeHead(200, { "content-encoding": "gzip" });
      response.end("unsafe malformed gzip content");
    });
    cleanups.push(server.close);
    const executor = new ConnectorExecutor(definitions(server.url));
    cleanups.push(() => executor.close());

    let caught: unknown;
    try {
      await executor.execute("status", {});
    } catch (error) {
      caught = error;
    }
    const classified = await classifyToolAPIError(caught, "jira");
    expect(classified?.toolApiError.classification).toBe("CONNECTION_ERROR");
    expect(classified?.message).not.toContain("unsafe");
  });

  it("closes the upstream response when a decoded body is truncated", async () => {
    let upstreamClosedResolve: (() => void) | undefined;
    const upstreamClosed = new Promise<void>((resolve) => {
      upstreamClosedResolve = resolve;
    });
    const server = await serve((_request, response) => {
      response.writeHead(200, { "content-encoding": "gzip" });
      const gzip = createGzip();
      gzip.pipe(response);
      const timer = setInterval(() => {
        gzip.write("x".repeat(64 * 1024));
      }, 1);
      response.on("close", () => {
        clearInterval(timer);
        gzip.destroy();
        upstreamClosedResolve?.();
      });
    });
    cleanups.push(server.close);
    const executor = new ConnectorExecutor(definitions(server.url));
    cleanups.push(() => executor.close());
    const result = await executor.execute("status", {});
    expect(result.endsWith("\n...[truncated]")).toBe(true);
    await Promise.race([
      upstreamClosed,
      new Promise<never>((_resolve, reject) =>
        setTimeout(
          () => reject(new Error("upstream response was not closed")),
          2_000,
        ),
      ),
    ]);
    await executor.close();
  });
});
