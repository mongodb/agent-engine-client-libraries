/**
 * Tests for deployed MCP OAuth secret materialization.
 *
 * Ports tests/unit/test_mcp_oauth_secret.py — verifies AGENTIC_MCP_OAUTH_B64_*
 * env secrets decode into the file cache with the same layout the reader
 * (mcp_oauth.ts) and the Go CLI expect.
 */

import { test, expect, beforeEach, afterEach } from "vitest";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  RuntimeMCPServerConfigSchema,
  makeMcpOauthAuth,
  materializeMcpOauthSecretCache,
  mcpOauthCacheName,
} from "../../src/index.js";

let cacheDir: string;
const savedEnv: Record<string, string | undefined> = {};

function trackEnv(...names: string[]): void {
  for (const name of names) savedEnv[name] = process.env[name];
}

beforeEach(() => {
  // A parent dir; tests point defaultCacheDir at a not-yet-created child so
  // mkdir behavior is exercised (matching the Python tmp_path/"cache" pattern).
  cacheDir = join(mkdtempSync(join(tmpdir(), "mcp-oauth-secret-")), "cache");
  trackEnv("AGENTIC_MCP_OAUTH_DIR");
  delete process.env["AGENTIC_MCP_OAUTH_DIR"];
});

afterEach(() => {
  rmSync(cacheDir, { recursive: true, force: true });
  for (const [name, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
    delete savedEnv[name];
  }
});

function b64(value: string): string {
  return Buffer.from(value, "utf-8").toString("base64");
}

test("sets the runtime cache dir without any secret", async () => {
  const materialized = await materializeMcpOauthSecretCache({
    defaultCacheDir: cacheDir,
  });

  expect(materialized).toBe(0);
  expect(process.env["AGENTIC_MCP_OAUTH_DIR"]).toBe(cacheDir);
  expect(statSync(cacheDir).isDirectory()).toBe(true);
});

test("writes the decoded cache file using the payload server_name", async () => {
  const payload = {
    schema_version: 1,
    server_name: "github-enterprise-team",
    server_url: "https://github.example.com/mcp/",
    tokens: { access_token: "access-token", refresh_token: "refresh-token" },
  };
  const raw = JSON.stringify(payload, null, 2);
  const envName = "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE_TEAM";
  trackEnv(envName);
  process.env[envName] = b64(raw);

  const materialized = await materializeMcpOauthSecretCache({
    defaultCacheDir: cacheDir,
  });

  const cacheFile = join(
    cacheDir,
    `${mcpOauthCacheName(payload.server_name, payload.server_url)}.json`,
  );
  expect(materialized).toBe(1);
  expect(process.env["AGENTIC_MCP_OAUTH_DIR"]).toBe(cacheDir);
  expect(readFileSync(cacheFile, "utf-8")).toBe(raw);
  expect(statSync(cacheFile).mode & 0o777).toBe(0o600);
  // Consumed env var is removed so it can't leak downstream.
  expect(process.env[envName]).toBeUndefined();
});

test("falls back to the env suffix as the cache name", async () => {
  const payload = {
    server_url: "https://github.example.com/mcp/",
    tokens: { refresh_token: "refresh-token" },
  };
  const raw = JSON.stringify(payload);
  const envName = "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE";
  trackEnv(envName);
  process.env[envName] = b64(raw);

  const materialized = await materializeMcpOauthSecretCache({
    defaultCacheDir: cacheDir,
  });

  expect(materialized).toBe(1);
  expect(
    readFileSync(
      join(
        cacheDir,
        `${mcpOauthCacheName("github_enterprise", payload.server_url)}.json`,
      ),
      "utf-8",
    ),
  ).toBe(raw);
});

test("reader resolves the same cache dir the writer set (barrel import)", async () => {
  // Regression guard: the reader's cache dir must be resolved at call time,
  // not frozen when mcp_oauth.ts is first imported. Importing everything from
  // the package barrel (as here) loads mcp_oauth.ts before materialize runs;
  // a frozen dir would leave the reader looking in the wrong place.
  const serverUrl = "https://mcp.example.com/github";
  const payload = {
    server_name: "github",
    server_url: serverUrl,
    tokens: { access_token: "access-token" },
  };
  const envName = "AGENTIC_MCP_OAUTH_B64_GITHUB";
  trackEnv(envName);
  process.env[envName] = b64(JSON.stringify(payload));

  await materializeMcpOauthSecretCache({ defaultCacheDir: cacheDir });

  // No explicit cacheDir — the provider must pick up AGENTIC_MCP_OAUTH_DIR
  // that materialize just set.
  const config = RuntimeMCPServerConfigSchema.parse({
    url: serverUrl,
    auth: { type: "oauth" },
  });
  const provider = makeMcpOauthAuth("github", config);
  expect((await provider.tokens())?.access_token).toBe("access-token");
});

test("rejects an invalid secret without leaking its value", async () => {
  const envName = "AGENTIC_MCP_OAUTH_B64_GITHUB";
  trackEnv(envName);
  process.env[envName] = "not base64!!!";

  let thrown: Error | undefined;
  try {
    await materializeMcpOauthSecretCache({ defaultCacheDir: cacheDir });
  } catch (e) {
    thrown = e as Error;
  }

  expect(thrown).toBeInstanceOf(Error);
  expect(thrown?.message).toContain(envName);
  expect(thrown?.message).not.toContain("not base64");
  expect(existsSync(join(cacheDir, "github.json"))).toBe(false);
});

test("preserves a legacy payload server name identity", async () => {
  // A payload minted before any naming rule can carry an arbitrary alias; the
  // materialized file must land exactly where the runtime reader for that
  // same alias looks.
  const payload = {
    server_name: "GitHub Enterprise",
    server_url: "https://github.example.com/mcp/",
    tokens: { refresh_token: "token" },
  };
  const envName = "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE";
  trackEnv(envName);
  process.env[envName] = b64(JSON.stringify(payload));

  await materializeMcpOauthSecretCache({ defaultCacheDir: cacheDir });

  expect(
    readFileSync(
      join(
        cacheDir,
        `${mcpOauthCacheName(payload.server_name, payload.server_url)}.json`,
      ),
      "utf-8",
    ),
  ).toBe(JSON.stringify(payload));
});
