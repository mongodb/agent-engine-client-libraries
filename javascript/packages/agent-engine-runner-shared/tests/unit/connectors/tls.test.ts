import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "node:https";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

import { ConnectorExecutor } from "../../../src/connectors/executor.js";
import {
  loadToolDefs,
  type Operation,
  type ToolDefinitions,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);
const CERT = fileURLToPath(
  new URL("./fixtures/tls/server.crt", import.meta.url),
);
const KEY = fileURLToPath(
  new URL("./fixtures/tls/server.key", import.meta.url),
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

describe("ConnectorExecutor TLS trust", () => {
  const originalFile = process.env["SSL_CERT_FILE"];
  const originalDir = process.env["SSL_CERT_DIR"];
  const originalProxy = process.env["HTTPS_PROXY"];

  afterEach(() => {
    if (originalFile === undefined) delete process.env["SSL_CERT_FILE"];
    else process.env["SSL_CERT_FILE"] = originalFile;
    if (originalDir === undefined) delete process.env["SSL_CERT_DIR"];
    else process.env["SSL_CERT_DIR"] = originalDir;
    if (originalProxy === undefined) delete process.env["HTTPS_PROXY"];
    else process.env["HTTPS_PROXY"] = originalProxy;
  });

  it("uses SSL_CERT_FILE without following proxy settings", async () => {
    const server = createServer(
      {
        cert: readFileSync(CERT),
        key: readFileSync(KEY),
      },
      (_request, response) => {
        response.writeHead(200, { "content-type": "text/plain" });
        response.end("ok");
      },
    );
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const address = server.address() as AddressInfo;
    const value = definitions(`https://127.0.0.1:${address.port}`);

    try {
      delete process.env["SSL_CERT_FILE"];
      delete process.env["SSL_CERT_DIR"];
      process.env["HTTPS_PROXY"] = "http://127.0.0.1:9";

      const untrusted = new ConnectorExecutor(value);
      await expect(untrusted.execute("status", {})).rejects.toThrow();
      await untrusted.close();

      process.env["SSL_CERT_FILE"] = CERT;
      const trusted = new ConnectorExecutor(value);
      await expect(trusted.execute("status", {})).resolves.toBe("ok");
      await trusted.close();
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    }
  });

  it("uses SSL_CERT_DIR", async () => {
    const server = createServer(
      {
        cert: readFileSync(CERT),
        key: readFileSync(KEY),
      },
      (_request, response) => {
        response.writeHead(200, { "content-type": "text/plain" });
        response.end("ok");
      },
    );
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const address = server.address() as AddressInfo;
    const value = definitions(`https://127.0.0.1:${address.port}`);
    const directory = mkdtempSync(join(tmpdir(), "connector-ca-"));
    copyFileSync(CERT, join(directory, "ca.pem"));

    try {
      delete process.env["SSL_CERT_FILE"];
      process.env["SSL_CERT_DIR"] = directory;
      const trusted = new ConnectorExecutor(value);
      await expect(trusted.execute("status", {})).resolves.toBe("ok");
      await trusted.close();
    } finally {
      rmSync(directory, { recursive: true, force: true });
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    }
  });
});
