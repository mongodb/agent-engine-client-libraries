import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { execFileSync, spawn } from "node:child_process";
import { createServer, type Server } from "node:http";
import { createServer as createSecureServer } from "node:https";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import type { TLSSocket } from "node:tls";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const bootId = "6f1c0000-1234-4567-8901-000000000001";
const packageRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const launcherPath = join(packageRoot, "dist/launcher.js");
let directory: string;
let server: Server;
let oeUrl: string;
let reports: Array<{ path: string; body: Record<string, unknown> }>;
let responseStatus: number;
let delayHeaders: number;
let trickleBody: boolean;

beforeEach(async () => {
  directory = mkdtempSync(join(tmpdir(), "launcher-reports-"));
  reports = [];
  responseStatus = 201;
  delayHeaders = 0;
  trickleBody = false;
  server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => {
      body += String(chunk);
    });
    request.on("end", () => {
      reports.push({ path: request.url ?? "", body: JSON.parse(body) });
      const timer = setTimeout(() => {
        response.writeHead(responseStatus, {
          Location: "/redirected",
          "Content-Length": "100000",
        });
        response.flushHeaders();
        if (trickleBody) {
          const interval = setInterval(() => response.write("x"), 100);
          response.on("close", () => clearInterval(interval));
        }
      }, delayHeaders);
      response.on("close", () => clearTimeout(timer));
    });
  });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("Missing HTTP address");
  oeUrl = `http://127.0.0.1:${address.port}`;
});

afterEach(async () => {
  server.closeAllConnections();
  await new Promise<void>((done) => server.close(() => done()));
  rmSync(directory, { recursive: true, force: true });
});

async function launch(
  source: string,
  environment: Record<string, string> = {},
) {
  const modulePath = join(directory, "agent.mjs");
  writeFileSync(modulePath, source);
  const started = performance.now();
  const child = spawn(process.execPath, [launcherPath], {
    env: {
      PATH: process.env.PATH,
      HOME: directory,
      AGENT_ENTRYPOINT: `${modulePath}:main`,
      AGENTIC_MCP_OAUTH_DIR: join(directory, "oauth"),
      AGENTIC_BOOT_ID: bootId,
      OE_URL: oeUrl,
      ...environment,
    },
    stdio: "ignore",
  });
  const timer = setTimeout(() => child.kill("SIGKILL"), 8000);
  const code = await new Promise<number | null>((done, reject) => {
    child.on("error", reject);
    child.on("close", done);
  });
  clearTimeout(timer);
  return { code, elapsed: performance.now() - started };
}

describe("launcher startup reports", () => {
  test.each([
    "https://alice:quiet-pass@db.example",
    "https://alice:quiet-pass" + "x".repeat(5000) + "@db.example",
  ])("scrubbing preserves URI credential defenses", async (message) => {
    const result = await launch(`throw new Error(${JSON.stringify(message)})`, {
      TENANT_PROTOCOL: "https",
    });
    expect(result.code).toBe(82);
    expect(reports).toHaveLength(1);
    expect(JSON.stringify(reports[0].body)).not.toContain("quiet-pass");
  });

  test("scrubs a bearer value the pattern only partially matches", async () => {
    const result = await launch(
      "throw new Error('Authorization: Bearer abc+tail/xyz== rejected')",
      { TENANT_API_KEY: "abc+tail/xyz==" },
    );
    expect(result.code).toBe(82);
    const body = JSON.stringify(reports[0].body);
    expect(body).not.toContain("tail");
    expect(body).not.toContain("xyz");
  });

  test("fully scrubs a tenant password inside a URI", async () => {
    await launch(
      "throw new Error('connect mongodb://alice:quiet-pass@db.example failed')",
      { TENANT_DB_PASSWORD: "quiet-pass" },
    );
    expect(reports[0].body.message).not.toContain("alice");
    expect(reports[0].body.message).not.toContain("quiet-pass");
  });

  test("empty OAuth fields do not garble the message", async () => {
    const encoded = Buffer.from(
      JSON.stringify({
        server_url: "https://mcp.example",
        access_token: "opaque",
        scope: "",
      }),
    ).toString("base64");
    await launch("throw new Error('MCP discovery rejected')", {
      AGENTIC_MCP_OAUTH_B64_TEST: encoded,
    });
    expect(reports[0].body.message).toContain("MCP discovery rejected");
  });

  test("password-only userinfo is not sent", async () => {
    const result = await launch("throw new Error('import failed')", {
      OE_URL: oeUrl.replace("http://", "http://:pass@"),
    });
    expect(result.code).toBe(82);
    expect(reports).toHaveLength(0);
  });

  test("a malformed OE URL keeps the classified exit", async () => {
    const result = await launch("throw new Error('import failed')", {
      OE_URL: "http://[::1",
    });
    expect(result.code).toBe(82);
    expect(reports).toHaveLength(0);
  });
  test("https performs a real client-certificate exchange", async () => {
    const certPath = join(directory, "cert.pem");
    const keyPath = join(directory, "key.pem");
    execFileSync(
      "openssl",
      [
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        keyPath,
        "-out",
        certPath,
        "-days",
        "1",
        "-subj",
        "/CN=localhost",
        "-addext",
        "subjectAltName=DNS:localhost",
      ],
      { env: { PATH: process.env.PATH }, stdio: "ignore" },
    );
    const cert = readFileSync(certPath);
    let authorized = false;
    const secure = createSecureServer(
      {
        cert,
        key: readFileSync(keyPath),
        ca: cert,
        requestCert: true,
        rejectUnauthorized: true,
      },
      (request, response) => {
        authorized = (request.socket as TLSSocket).authorized;
        request.resume();
        response.writeHead(201);
        response.end();
      },
    );
    await new Promise<void>((done) => secure.listen(0, "127.0.0.1", done));
    const address = secure.address();
    if (!address || typeof address === "string")
      throw new Error("Missing HTTPS address");
    try {
      const result = await launch("throw new Error('import failed')", {
        OE_URL: `https://localhost:${address.port}`,
        TLS_CERT_PATH: certPath,
        TLS_KEY_PATH: keyPath,
        TLS_CA_CERT_PATH: certPath,
      });
      expect(result.code).toBe(82);
      expect(authorized).toBe(true);
    } finally {
      secure.closeAllConnections();
      await new Promise<void>((done) => secure.close(() => done()));
    }
  });
  test.each([
    ["throw new Error('import failed')", 82],
    ["export const other = 1", 83],
    [
      "export function main() { throw new Error('MCP discovery rejected') }",
      84,
    ],
    [
      "export function main() { setImmediate(() => { throw new Error('stray exception') }); return new Promise(() => {}) }",
      84,
    ],
    [
      "export function main() { Promise.reject(new Error('stray rejection')); return new Promise(() => {}) }",
      84,
    ],
  ])(
    "reports one startup failure and preserves its exit code (%s)",
    async (source, code) => {
      expect((await launch(source)).code).toBe(code);
      expect(reports).toHaveLength(1);
      expect(reports[0]).toMatchObject({
        path: `/boots/${bootId}/errors`,
        body: { sandbox: "agent", exitCode: code },
      });
      expect(reports[0].body.message).not.toContain("at main");
    },
  );

  test.each(["OE_URL", "AGENTIC_BOOT_ID"])("skips without %s", async (name) => {
    expect(
      (await launch("throw new Error('import failed')", { [name]: "" })).code,
    ).toBe(82);
    expect(reports).toHaveLength(0);
  });

  test.each([201, 307, 404, 409, 500])(
    "uses only status %i without following redirects or reading bodies",
    async (status) => {
      responseStatus = status;
      trickleBody = true;
      const result = await launch("throw new Error('import failed')");
      expect(result.code).toBe(82);
      expect(reports).toHaveLength(1);
    },
  );

  test("delayed headers cannot extend the total exit deadline", async () => {
    delayHeaders = 10_000;
    const result = await launch("throw new Error('import failed')");
    expect(result.code).toBe(82);
    expect(result.elapsed).toBeGreaterThan(1900);
    expect(result.elapsed).toBeLessThan(6000);
    expect(reports).toHaveLength(1);
  });

  test("concurrent failures cannot exit before the single report completes", async () => {
    delayHeaders = 300;
    const result = await launch(`export function main() {
      setImmediate(() => { Promise.reject(new Error('second')); throw new Error('first') });
      return new Promise(() => {});
    }`);
    expect(result.code).toBe(84);
    expect(result.elapsed).toBeGreaterThan(300);
    expect(reports).toHaveLength(1);
  });

  test.each(["true", ""])(
    "post-listen exceptions keep existing behavior with structured logging %s",
    async (structured) => {
      const state = pathToFileURL(
        join(packageRoot, "dist/startup_state.js"),
      ).href;
      const result = await launch(
        `import { markStartupComplete } from ${JSON.stringify(state)};
      export function main() { markStartupComplete(); setImmediate(() => { throw new Error('serving crash') }); }
    `,
        { STRUCTURED_LOGGING: structured },
      );
      expect(result.code).toBe(1);
      expect(reports).toHaveLength(0);
    },
  );

  test("scrubs tenant, TLS and already-removed OAuth values and bounds UTF-8", async () => {
    const token = "opaque-oauth-value";
    const encoded = Buffer.from(
      JSON.stringify({
        server_url: "https://mcp.example",
        access_token: token,
      }),
    ).toString("base64");
    const result = await launch(
      `export function main() {
      if (process.env.AGENTIC_MCP_OAUTH_B64_TEST) throw new Error('credential was not removed');
      throw new Error(${JSON.stringify(`port 1: ${token} opaque-tenant-value private-key-material\n\x1b[31m` + "界".repeat(1000))});
    }`,
      {
        AGENTIC_MCP_OAUTH_B64_TEST: encoded,
        TENANT_API_KEY: "opaque-tenant-value",
        TENANT_DEBUG: "1",
        TLS_KEY_PEM: "private-key-material",
      },
    );
    expect(result.code).toBe(84);
    expect(reports).toHaveLength(1);
    const body = reports[0].body;
    for (const value of [
      token,
      encoded,
      "opaque-tenant-value",
      "private-key-material",
    ]) {
      expect(JSON.stringify(body)).not.toContain(value);
    }
    expect(Buffer.byteLength(String(body.message))).toBeLessThanOrEqual(512);
    expect(body.message).toContain("port 1: ");
    expect(body.message).not.toContain("\n");
    expect(body.message).not.toContain("\x1b");
  });
});
