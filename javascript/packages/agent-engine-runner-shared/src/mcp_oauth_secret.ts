/**
 * Materialize deployed MCP OAuth secrets into the file-backed token cache.
 *
 * Port of `agent_engine_runner_shared/mcp_oauth_secret.py`. On deploy the platform injects
 * each server's OAuth cache as a base64-encoded `AGENTIC_MCP_OAUTH_B64_*` env
 * var; this decodes them into the cache directory that `mcp_oauth.ts` reads,
 * keeping the on-disk layout byte-identical to the Python side so either SDK
 * (and the Go CLI) can read the same files.
 */

import * as fs from "node:fs";
import * as path from "node:path";

import { lock } from "proper-lockfile";

import { mcpOauthCacheName } from "./mcp_oauth.js";

const MCP_OAUTH_DIR_ENV = "AGENTIC_MCP_OAUTH_DIR";
const MCP_OAUTH_SECRET_ENV_PREFIX = "AGENTIC_MCP_OAUTH_B64_";
const PLATFORM_MCP_OAUTH_CACHE_DIR = "/tmp/agentic/mcp-oauth";
const SAFE_ENV_CACHE_NAME_RE = /^[A-Z0-9_]+$/;
const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/;

interface MaterializeOptions {
  defaultCacheDir?: string;
}

interface DecodedSecret {
  rawCache: Buffer;
  payload: Record<string, unknown>;
}

/**
 * Establish the writable runtime cache and decode MCP OAuth secrets into it.
 *
 * Sets `AGENTIC_MCP_OAUTH_DIR` (if unset) so the reader in `mcp_oauth.ts`
 * resolves the same directory. Returns the number of secrets materialized.
 */
export async function materializeMcpOauthSecretCache({
  defaultCacheDir = PLATFORM_MCP_OAUTH_CACHE_DIR,
}: MaterializeOptions = {}): Promise<number> {
  const cacheDir = process.env[MCP_OAUTH_DIR_ENV] ?? defaultCacheDir;
  if (process.env[MCP_OAUTH_DIR_ENV] === undefined) {
    process.env[MCP_OAUTH_DIR_ENV] = cacheDir;
  }
  fs.mkdirSync(cacheDir, { recursive: true, mode: 0o700 });

  let materialized = 0;
  for (const [envName, encodedCache] of mcpOauthSecretEnv()) {
    const { rawCache, payload } = decodeCacheSecret(envName, encodedCache);
    const cacheName = cacheNameFor(envName, payload);
    await writeCacheFile(cacheDir, cacheName, rawCache);
    delete process.env[envName];
    materialized += 1;
  }

  return materialized;
}

function mcpOauthSecretEnv(): Array<[string, string]> {
  return Object.entries(process.env)
    .filter(
      (entry): entry is [string, string] =>
        entry[0].startsWith(MCP_OAUTH_SECRET_ENV_PREFIX) &&
        entry[1] !== undefined,
    )
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

function decodeCacheSecret(
  envName: string,
  encodedCache: string,
): DecodedSecret {
  // Strict base64: mirror Python's `validate=True`. Node's decoder silently
  // drops out-of-alphabet characters, so reject them up front rather than
  // materializing a truncated cache.
  if (encodedCache.length % 4 !== 0 || !BASE64_RE.test(encodedCache)) {
    throw new Error(
      `invalid MCP OAuth secret ${envName}: value is not valid base64`,
    );
  }
  const rawCache = Buffer.from(encodedCache, "base64");

  let payload: unknown;
  try {
    payload = JSON.parse(rawCache.toString("utf-8"));
  } catch (err) {
    throw new Error(
      `invalid MCP OAuth secret ${envName}: decoded value is not valid JSON`,
      { cause: err },
    );
  }
  if (
    payload === null ||
    typeof payload !== "object" ||
    Array.isArray(payload)
  ) {
    throw new Error(
      `invalid MCP OAuth secret ${envName}: decoded value must be a JSON object`,
    );
  }
  return { rawCache, payload: payload as Record<string, unknown> };
}

function cacheNameFor(
  envName: string,
  payload: Record<string, unknown>,
): string {
  // The runtime looks the cache up by the configured server URL's hash, so a
  // payload without server_url could never be found — fail loudly instead of
  // materializing an unreadable file.
  const serverUrl = payload["server_url"];
  if (typeof serverUrl !== "string" || serverUrl.trim() === "") {
    throw new Error(`invalid MCP OAuth secret ${envName}: missing server_url`);
  }

  const payloadServerName = payload["server_name"];
  if (
    typeof payloadServerName === "string" &&
    payloadServerName.trim() !== ""
  ) {
    return mcpOauthCacheName(payloadServerName, serverUrl);
  }

  const envCacheName = envName.slice(MCP_OAUTH_SECRET_ENV_PREFIX.length);
  if (envCacheName === "" || !SAFE_ENV_CACHE_NAME_RE.test(envCacheName)) {
    throw new Error(`invalid MCP OAuth secret ${envName}: invalid cache name`);
  }
  return mcpOauthCacheName(envCacheName.toLowerCase(), serverUrl);
}

async function writeCacheFile(
  cacheDir: string,
  cacheName: string,
  rawCache: Buffer,
): Promise<void> {
  const cacheFile = path.join(cacheDir, `${cacheName}.json`);

  let release: (() => Promise<void>) | undefined;
  try {
    // Same lock target/params as `mcp_oauth.ts`'s writePayload, so a
    // materialize write and a runtime token refresh can't clobber each other.
    release = await lock(cacheFile, {
      realpath: false,
      stale: 5000,
      retries: { retries: 10, minTimeout: 20, maxTimeout: 200 },
    });
  } catch (err) {
    throw new Error(
      `failed to acquire MCP OAuth cache lock for ${JSON.stringify(cacheFile)}: ` +
        `${err instanceof Error ? err.message : String(err)}`,
      { cause: err },
    );
  }

  const tmpName = path.join(
    cacheDir,
    `.tmp-${cacheName}-${process.pid}-${Date.now()}`,
  );
  try {
    try {
      fs.writeFileSync(tmpName, rawCache, { mode: 0o600 });
      fs.renameSync(tmpName, cacheFile);
    } finally {
      try {
        fs.unlinkSync(tmpName);
      } catch {
        // Already renamed into place, or never created — nothing to clean up.
      }
    }
  } finally {
    await release();
  }
}
