/**
 * End-to-end protocol-era compatibility for the runner's MCP client.
 *
 * The unit tests mock the SDK `Client`; this test runs the real
 * `@modelcontextprotocol/client` against local streamable-HTTP servers:
 *
 * - a 2026-07-28 server that answers `server/discover`, so `mode: "auto"`
 *   selects the modern era; and
 * - a 2025-11-25 server that rejects `server/discover` and only speaks
 *   `initialize`, so the client must fall back to the legacy handshake.
 *
 * It proves discovery and invocation work across both eras, that the matching
 * `tools/call` request carries the negotiated revision, and that pagination
 * stops before the tool cap even when the catalog uses SSE framing.
 */

import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";

import { afterEach, describe, expect, test } from "vitest";

import {
  RuntimeMCPConfigSchema,
  callMcpTool,
  discoverMcpTools,
} from "../../src/index.js";

type Era = "modern" | "legacy";

interface RecordedRequest {
  method: string;
  protocolVersion: string | undefined;
}

interface EraServerOptions {
  pages?: number;
  toolsPerPage?: number;
  sse?: boolean;
  waitForPing?: boolean;
  resumable?: boolean;
  lineEnding?: "lf" | "cr" | "crlf";
  sseContentType?: string;
}

interface EraServer {
  server: Server;
  url: string;
  requests: RecordedRequest[];
  pingReplies: () => number;
}

const TOOL = {
  name: "annotate",
  description: "Return a marker for the negotiated era",
  inputSchema: {
    type: "object",
    properties: { text: { type: "string" } },
    required: ["text"],
  },
};

interface JsonRpcMessage {
  jsonrpc: string;
  id?: unknown;
  method?: string;
  params?: { cursor?: unknown };
}

interface PendingSse {
  res: ServerResponse;
  id: unknown;
  result: unknown;
}

function startEraServer(
  era: Era,
  options: EraServerOptions = {},
): Promise<EraServer> {
  const {
    pages = 1,
    toolsPerPage = 1,
    sse = false,
    waitForPing = false,
    resumable = false,
    lineEnding = "lf",
    sseContentType = "text/event-stream",
  } = options;
  const requests: RecordedRequest[] = [];
  let pingReplies = 0;
  let pendingSse: PendingSse | undefined;
  let pendingResume: { id: unknown; result: unknown } | undefined;

  const server = createServer((req, res) => {
    void handle(req, res);
  });

  function startSse(res: ServerResponse): void {
    res.writeHead(200, {
      "content-type": sseContentType,
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
  }

  function writeSseEvent(res: ServerResponse, payload: unknown): void {
    const json = JSON.stringify(payload);
    // Split the JSON payload across two data lines to exercise SSE
    // reassembly; the newline is legal whitespace between object members.
    const splitAt = json.indexOf(',"tools"');
    const lines =
      splitAt > 0
        ? [json.slice(0, splitAt + 1), json.slice(splitAt + 1)]
        : [json];
    const eol =
      lineEnding === "cr" ? "\r" : lineEnding === "crlf" ? "\r\n" : "\n";
    res.write(
      `event: message${eol}${lines
        .map((line) => `data: ${line}`)
        .join(eol)}${eol}${eol}`,
    );
  }

  async function handle(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    if (
      req.method === "GET" &&
      resumable &&
      pendingResume !== undefined &&
      req.headers["last-event-id"] === "page-1"
    ) {
      startSse(res);
      writeSseEvent(res, {
        jsonrpc: "2.0",
        id: pendingResume.id,
        result: pendingResume.result,
      });
      res.end();
      pendingResume = undefined;
      return;
    }

    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(chunk as Buffer);
    }
    const raw = Buffer.concat(chunks).toString("utf8");
    const message = raw === "" ? null : (JSON.parse(raw) as JsonRpcMessage);

    if (message !== null && typeof message.method === "string") {
      requests.push({
        method: message.method,
        protocolVersion: req.headers["mcp-protocol-version"] as
          | string
          | undefined,
      });
    }

    if (message === null) {
      // The transport's listening GET has no reply body.
      res.writeHead(202).end();
      return;
    }

    if (message.method === undefined) {
      // A client response, such as the automatic reply to an in-stream ping.
      res.writeHead(202).end();
      if (waitForPing && message.id === "ping-1" && pendingSse !== undefined) {
        pingReplies++;
        writeSseEvent(pendingSse.res, {
          jsonrpc: "2.0",
          id: pendingSse.id,
          result: pendingSse.result,
        });
        pendingSse.res.end();
        pendingSse = undefined;
      }
      return;
    }

    if (message.id === undefined) {
      // Notifications have no reply body.
      res.writeHead(202).end();
      return;
    }

    const send = (result: unknown): void => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
    };
    const fail = (code: number, text: string): void => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          jsonrpc: "2.0",
          id: message.id,
          error: { code, message: text },
        }),
      );
    };

    switch (message.method) {
      case "server/discover":
        if (era === "legacy") {
          fail(-32601, "Method not found");
          return;
        }
        send({
          supportedVersions: ["2026-07-28"],
          capabilities: { tools: {} },
        });
        return;
      case "initialize":
        send({
          protocolVersion: "2025-11-25",
          capabilities: { tools: {} },
          serverInfo: { name: "era-test", version: "1.0.0" },
        });
        return;
      case "notifications/initialized":
        res.writeHead(202).end();
        return;
      case "tools/list": {
        const cursor = message.params?.cursor;
        const page = typeof cursor === "string" ? Number(cursor) : 0;
        const perPage = pages > 1 ? toolsPerPage : 1;
        const pageTools =
          pages > 1
            ? Array.from({ length: perPage }, (_, index) => ({
                ...TOOL,
                name: `tool_${page}_${index}`,
              }))
            : [TOOL];
        const nextCursor = page + 1 < pages ? String(page + 1) : undefined;
        // 2026-07-28 requires the explicit result type and cache metadata.
        const result = {
          ...(era === "modern"
            ? { resultType: "complete", ttlMs: 0, cacheScope: "private" }
            : {}),
          tools: pageTools,
          ...(nextCursor === undefined ? {} : { nextCursor }),
        };

        if (resumable) {
          startSse(res);
          const eol =
            lineEnding === "cr" ? "\r" : lineEnding === "crlf" ? "\r\n" : "\n";
          res.write(`id: page-1${eol}retry: 5${eol}${eol}`);
          pendingResume = { id: message.id, result };
          res.end();
          return;
        }
        if (sse) {
          startSse(res);
          if (waitForPing) {
            writeSseEvent(res, {
              jsonrpc: "2.0",
              id: "ping-1",
              method: "ping",
            });
            pendingSse = { res, id: message.id, result };
            return;
          }
          writeSseEvent(res, { jsonrpc: "2.0", id: message.id, result });
          res.end();
          return;
        }
        send(result);
        return;
      }
      case "tools/call":
        send(
          era === "modern"
            ? {
                resultType: "complete",
                content: [{ type: "text", text: "modern-result" }],
              }
            : { content: [{ type: "text", text: "legacy-result" }] },
        );
        return;
      default:
        fail(-32601, "Method not found");
    }
  }

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        throw new Error("era test server did not bind a TCP port");
      }
      resolve({
        server,
        url: `http://127.0.0.1:${address.port}/mcp`,
        requests,
        pingReplies: () => pingReplies,
      });
    });
  });
}

const servers: Server[] = [];

afterEach(() => {
  for (const server of servers.splice(0)) {
    server.close();
  }
});

async function discoverAndCall(url: string): Promise<unknown> {
  const config = RuntimeMCPConfigSchema.parse({ servers: { era: { url } } });
  const bindings = await discoverMcpTools(config);
  expect(bindings.map((binding) => binding.toolName)).toEqual(["annotate"]);

  const binding = bindings[0];
  if (binding === undefined) throw new Error("expected a discovered binding");
  const result = await callMcpTool(binding, { text: "hello" });

  return (result.content[0] as { text?: unknown } | undefined)?.text;
}

describe("MCP protocol-era compatibility", () => {
  test("negotiates 2026-07-28 and attributes it to the tool call", async () => {
    const { server, url, requests } = await startEraServer("modern");
    servers.push(server);

    const text = await discoverAndCall(url);

    expect(text).toBe("modern-result");
    expect(requests.map((request) => request.method)).not.toContain(
      "initialize",
    );
    const calls = requests.filter((request) => request.method === "tools/call");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.protocolVersion).toBe("2026-07-28");
  });

  test("falls back to 2025-11-25 when the probe is rejected", async () => {
    const { server, url, requests } = await startEraServer("legacy");
    servers.push(server);

    const text = await discoverAndCall(url);

    expect(text).toBe("legacy-result");
    expect(requests.map((request) => request.method)).toContain("initialize");
    const calls = requests.filter((request) => request.method === "tools/call");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.protocolVersion).toBe("2025-11-25");
  });

  test("discovers a terminating catalog beyond the SDK's default 64 pages", async () => {
    const { server, url } = await startEraServer("modern", { pages: 65 });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });
    const bindings = await discoverMcpTools(config);

    expect(bindings).toHaveLength(65);
  });

  test("rejects a catalog beyond the runner's 100-page bound", async () => {
    const { server, url } = await startEraServer("modern", { pages: 101 });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });

    await expect(discoverMcpTools(config)).rejects.toMatchObject({
      name: "MCPConfigError",
      message: expect.stringContaining("tools/list"),
    });
  });

  test("stops pagination once the catalog exceeds the tool cap", async () => {
    const { server, url, requests } = await startEraServer("modern", {
      pages: 3,
      toolsPerPage: 6_000,
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });

    await expect(discoverMcpTools(config)).rejects.toMatchObject({
      name: "MCPConfigError",
      message: expect.stringContaining("more than 10000 tools"),
    });

    const listCalls = requests.filter(
      (request) => request.method === "tools/list",
    );
    expect(listCalls).toHaveLength(2);
  });

  test("counts multiline SSE events and stops at the tool cap", async () => {
    const { server, url, requests } = await startEraServer("modern", {
      pages: 3,
      toolsPerPage: 6_000,
      sse: true,
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });

    await expect(discoverMcpTools(config)).rejects.toMatchObject({
      name: "MCPConfigError",
      message: expect.stringContaining("more than 10000 tools"),
    });

    const listCalls = requests.filter(
      (request) => request.method === "tools/list",
    );
    expect(listCalls).toHaveLength(2);
  });

  test("answers an in-stream ping before a legacy catalog completes", async () => {
    const { server, url, pingReplies } = await startEraServer("legacy", {
      sse: true,
      waitForPing: true,
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });
    const bindings = await discoverMcpTools(config);

    expect(bindings.map((binding) => binding.toolName)).toEqual(["annotate"]);
    expect(pingReplies()).toBe(1);
  });

  test("counts CR-delimited SSE events and stops at the tool cap", async () => {
    const { server, url, requests } = await startEraServer("modern", {
      pages: 3,
      toolsPerPage: 6_000,
      sse: true,
      lineEnding: "cr",
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });

    await expect(discoverMcpTools(config)).rejects.toMatchObject({
      name: "MCPConfigError",
      message: expect.stringContaining("more than 10000 tools"),
    });

    const listCalls = requests.filter(
      (request) => request.method === "tools/list",
    );
    expect(listCalls).toHaveLength(2);
  });

  test("resumes a primed legacy catalog within the configured timeout", async () => {
    const { server, url } = await startEraServer("legacy", {
      sse: true,
      resumable: true,
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url, timeout_seconds: 1 } },
    });
    const bindings = await discoverMcpTools(config);

    expect(bindings.map((binding) => binding.toolName)).toEqual(["annotate"]);
  });

  test("stops a resumable catalog at the tool cap", async () => {
    const { server, url, requests } = await startEraServer("legacy", {
      sse: true,
      resumable: true,
      pages: 3,
      toolsPerPage: 6_000,
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url, timeout_seconds: 1 } },
    });

    await expect(discoverMcpTools(config)).rejects.toMatchObject({
      name: "MCPConfigError",
      message: expect.stringContaining("more than 10000 tools"),
    });

    const listCalls = requests.filter(
      (request) => request.method === "tools/list",
    );
    expect(listCalls).toHaveLength(2);
  });

  test("answers a CR-delimited in-stream ping before a legacy catalog completes", async () => {
    const { server, url, pingReplies } = await startEraServer("legacy", {
      sse: true,
      waitForPing: true,
      lineEnding: "cr",
      sseContentType: "Text/Event-Stream",
    });
    servers.push(server);

    const config = RuntimeMCPConfigSchema.parse({
      servers: { era: { url } },
    });
    const bindings = await discoverMcpTools(config);

    expect(bindings.map((binding) => binding.toolName)).toEqual(["annotate"]);
    expect(pingReplies()).toBe(1);
  });
});
