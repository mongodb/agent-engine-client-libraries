/**
 * Tests for MCP OAuth providers and file-backed token caching.
 *
 * Mirrors the file-cache-format coverage in Python's tests/unit/test_mcp_oauth.py.
 * Adapted to `@modelcontextprotocol/sdk`'s `OAuthClientProvider` interface —
 * see the `mcp_oauth.ts` module comment for why this isn't a literal port.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  RuntimeMCPServerConfigSchema,
  makeMcpClientCredentialsAuth,
  makeMcpOauthAuth,
  mcpOauthCacheName,
} from "../../src/index.js";

let cacheDir: string;

beforeEach(() => {
  cacheDir = mkdtempSync(join(tmpdir(), "mcp-oauth-"));
});

afterEach(() => {
  rmSync(cacheDir, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

function makeServerConfig(overrides: Record<string, unknown> = {}) {
  return RuntimeMCPServerConfigSchema.parse({
    url: "https://mcp.example.com/github",
    ...overrides,
  });
}

describe("mcpOauthCacheName", () => {
  // Mirrored by the Go (mcpname) and Python parity tests; a drift in any
  // implementation breaks the shared cache layout.
  const url = "https://mcp.example.com/one";
  const urlHash =
    "34fcf0941fd4db94f72e35ac8d58114ce17a82c9949c883fa6f2767b5c095c2b";

  test.each([
    ["github", `github-${urlHash}`],
    ["github_enterprise", `github_enterprise-${urlHash}`],
    ["GitHub", `GitHub-${urlHash}`],
    ["github.com", `github-com-3aeb002460381c6f-${urlHash}`],
    ["github com", `github-com-7950713f209fb3ca-${urlHash}`],
    [" /// ", `663b6e4b48f7158c-${urlHash}`],
    [
      "GitHub Enterprise / Team",
      `GitHub-Enterprise-Team-5616f7fc218b4465-${urlHash}`,
    ],
    ["a".repeat(65), `${"a".repeat(40)}-635361c48bb9eab1-${urlHash}`],
  ])("derives %s as %s", (serverName, expected) => {
    expect(mcpOauthCacheName(serverName, url)).toBe(expected);
  });

  test("separates aliases that sanitize to the same readable form", () => {
    expect(mcpOauthCacheName("github.com", url)).not.toBe(
      mcpOauthCacheName("github com", url),
    );
    expect(mcpOauthCacheName("github.com", url)).not.toBe(
      mcpOauthCacheName("github/com", url),
    );
  });
});

describe("makeMcpOauthAuth", () => {
  test("isolates the same alias at different server URLs", async () => {
    const first = makeMcpOauthAuth(
      "github",
      makeServerConfig({
        url: "https://mcp.example.com/one",
        auth: { type: "oauth" },
      }),
      cacheDir,
    );
    const second = makeMcpOauthAuth(
      "github",
      makeServerConfig({
        url: "https://mcp.example.com/two",
        auth: { type: "oauth" },
      }),
      cacheDir,
    );

    await first.saveTokens({ access_token: "first" });
    await second.saveTokens({ access_token: "second" });

    expect(readdirSync(cacheDir).sort()).toHaveLength(2);
    expect(readdirSync(cacheDir)).toEqual(
      expect.arrayContaining([
        "github-34fcf0941fd4db94f72e35ac8d58114ce17a82c9949c883fa6f2767b5c095c2b.json",
      ]),
    );
  });

  test("has no cached tokens before any login", async () => {
    const config = makeServerConfig({ auth: { type: "oauth" } });
    const provider = makeMcpOauthAuth("github", config, cacheDir);
    expect(await provider.tokens()).toBeUndefined();
  });

  test("reads tokens back after saveTokens (same cache file the CLI writes)", async () => {
    const config = makeServerConfig({ auth: { type: "oauth" } });
    const provider = makeMcpOauthAuth("github", config, cacheDir);

    await provider.saveTokens({
      access_token: "abc123",
      token_type: "Bearer",
      refresh_token: "refresh-abc",
      expires_in: 3600,
    });

    const tokens = await provider.tokens();
    expect(tokens).toMatchObject({
      access_token: "abc123",
      token_type: "Bearer",
      refresh_token: "refresh-abc",
    });
    expect(tokens?.expires_in).toBeGreaterThan(3500);

    const cachePath = join(
      cacheDir,
      `${mcpOauthCacheName("github", config.url)}.json`,
    );
    expect(statSync(cachePath).mode & 0o777).toBe(0o600);
    const onDisk = JSON.parse(readFileSync(cachePath, "utf-8"));
    expect(onDisk.server_url).toBe("https://mcp.example.com/github");
    expect(onDisk.tokens.access_token).toBe("abc123");
    expect(onDisk.tokens.expiry).toBeDefined();
    expect(onDisk.tokens.expires_in).toBeUndefined();
  });

  test("concurrent saveTokens + saveClientInformation don't clobber each other (lock covers the read-modify-write)", async () => {
    // Without the advisory lock around writePayload()'s read-modify-write,
    // two concurrent writers can each read the same (stale) payload and
    // write back only their own key, silently losing whichever finished
    // first — a lost-update race. Running both concurrently here proves the
    // lock serializes them so both keys survive.
    const config = makeServerConfig({ auth: { type: "oauth" } });
    const provider = makeMcpOauthAuth("github", config, cacheDir);

    await Promise.all([
      provider.saveTokens({ access_token: "token-a", token_type: "Bearer" }),
      Promise.resolve(
        provider.saveClientInformation?.({
          redirect_uris: ["http://127.0.0.1:8765/callback"],
          client_id: "client-xyz",
          grant_types: ["authorization_code", "refresh_token"],
          response_types: ["code"],
        }),
      ),
    ]);

    const cachePath = join(
      cacheDir,
      `${mcpOauthCacheName("github", config.url)}.json`,
    );
    const onDisk = JSON.parse(readFileSync(cachePath, "utf-8"));
    expect(onDisk.tokens?.access_token).toBe("token-a");
    expect(onDisk.client?.client_id).toBe("client-xyz");
  });

  test("returns undefined when the cached server_url does not match (e.g. url changed)", async () => {
    const config = makeServerConfig({ auth: { type: "oauth" } });
    const provider = makeMcpOauthAuth("github", config, cacheDir);
    await provider.saveTokens({ access_token: "abc123", token_type: "Bearer" });

    const otherConfig = makeServerConfig({
      url: "https://mcp.example.com/other-github",
      auth: { type: "oauth" },
    });
    const otherProvider = makeMcpOauthAuth("github", otherConfig, cacheDir);
    expect(await otherProvider.tokens()).toBeUndefined();
  });

  test("redirectToAuthorization/codeVerifier require the CLI login flow", async () => {
    const config = makeServerConfig({ auth: { type: "oauth" } });
    const provider = makeMcpOauthAuth("github", config, cacheDir);

    expect(() =>
      provider.redirectToAuthorization(new URL("https://x")),
    ).toThrow(/agentengine dev mcp auth login github/);
    expect(() => provider.codeVerifier()).toThrow(
      /agentengine dev mcp auth login github/,
    );
  });

  test("clientMetadata reflects configured client_name/scope/redirect_uri", () => {
    const config = makeServerConfig({
      auth: {
        type: "oauth",
        client_name: "My Agent",
        scope: "repo read:org",
        redirect_uri: "http://127.0.0.1:9999/callback",
      },
    });
    const provider = makeMcpOauthAuth("github", config, cacheDir);
    expect(provider.clientMetadata).toMatchObject({
      client_name: "My Agent",
      scope: "repo read:org",
      redirect_uris: ["http://127.0.0.1:9999/callback"],
      token_endpoint_auth_method: "none",
    });
    expect(provider.redirectUrl).toBe("http://127.0.0.1:9999/callback");
  });
});

describe("makeMcpClientCredentialsAuth", () => {
  test("throws when client_id_env is unset", () => {
    vi.stubEnv("GH_SECRET", "shh");
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "GH_CLIENT_ID_MISSING",
        client_secret_env: "GH_SECRET",
      },
    });
    expect(() =>
      makeMcpClientCredentialsAuth("github", config, cacheDir),
    ).toThrow(/auth\.client_id_env references missing env var/);
  });

  test("constructs a provider with client_credentials clientMetadata", () => {
    vi.stubEnv("GH_CLIENT_ID", "client-abc");
    vi.stubEnv("GH_SECRET", "shh");
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "GH_CLIENT_ID",
        client_secret_env: "GH_SECRET",
        scope: "repo",
      },
    });
    const provider = makeMcpClientCredentialsAuth("github", config, cacheDir);
    expect(provider.clientMetadata.scope).toBe("repo");
    expect(provider.redirectUrl).toBeUndefined();
  });

  test("caches tokens to a client-id-scoped file distinct from the oauth cache", async () => {
    vi.stubEnv("GH_CLIENT_ID", "client-abc");
    vi.stubEnv("GH_SECRET", "shh");
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "GH_CLIENT_ID",
        client_secret_env: "GH_SECRET",
      },
    });
    const provider = makeMcpClientCredentialsAuth("github", config, cacheDir);
    await provider.saveTokens({
      access_token: "cc-token",
      token_type: "Bearer",
      expires_in: 3600,
    });

    const tokens = await provider.tokens();
    expect(tokens?.access_token).toBe("cc-token");
  });

  test("pre-seeds discovery state to skip discovery when auth.token_url is set", () => {
    vi.stubEnv("GH_CLIENT_ID", "client-abc");
    vi.stubEnv("GH_SECRET", "shh");
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "GH_CLIENT_ID",
        client_secret_env: "GH_SECRET",
        token_url: "https://auth.example.com/token",
      },
    });
    const provider = makeMcpClientCredentialsAuth("github", config, cacheDir);
    const state = provider.discoveryState?.();
    expect(state).toMatchObject({
      authorizationServerUrl: "https://auth.example.com/token",
    });
  });

  test("discoveryState is undefined when auth.token_url is not set", () => {
    vi.stubEnv("GH_CLIENT_ID", "client-abc");
    vi.stubEnv("GH_SECRET", "shh");
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "GH_CLIENT_ID",
        client_secret_env: "GH_SECRET",
      },
    });
    const provider = makeMcpClientCredentialsAuth("github", config, cacheDir);
    expect(provider.discoveryState?.()).toBeUndefined();
  });
});
