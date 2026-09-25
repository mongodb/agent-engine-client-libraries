/**
 * OAuth provider and file-backed token storage for remote MCP servers.
 *
 * Port of `agent_engine_runner_shared/mcp_oauth.py`, adapted to `@modelcontextprotocol/sdk`'s
 * `OAuthClientProvider` interface — a lower-level, method-based contract
 * (`tokens()`/`saveTokens()`/`clientInformation()`/...) rather than Python's
 * httpx-`Auth`-flow-based `OAuthClientProvider`. The cache file format/path
 * is kept byte-identical to the Python side so `agentengine dev mcp auth
 * login/status/upload` (Go CLI) and this runtime read/write the same file
 * regardless of which SDK the agent uses.
 */

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { lock } from "proper-lockfile";

import { ClientCredentialsProvider } from "@modelcontextprotocol/sdk/client/auth-extensions.js";
import type { OAuthClientProvider } from "@modelcontextprotocol/sdk/client/auth.js";
import type {
  OAuthClientInformationFull,
  OAuthClientMetadata,
  OAuthTokens,
} from "@modelcontextprotocol/sdk/shared/auth.js";

import type { RuntimeMCPServerConfig } from "./agent_config.js";

export const DEFAULT_MCP_OAUTH_CLIENT_NAME =
  "Atlas Agent Engine Dev MCP Client";
export const DEFAULT_MCP_OAUTH_REDIRECT_URI = "http://127.0.0.1:8765/callback";
/**
 * Cache directory, resolved at call time — not a module-level constant.
 *
 * `materializeMcpOauthSecretCache()` (in `mcp_oauth_secret.ts`) sets
 * `AGENTIC_MCP_OAUTH_DIR` at startup. A frozen import-time const would capture
 * the value from before that runs whenever this module is imported first (e.g.
 * via the package barrel), so the reader and writer could resolve to different
 * directories. Reading the env on each call keeps them in lockstep regardless
 * of import order.
 */
export function mcpOauthCacheDir(): string {
  return (
    process.env["AGENTIC_MCP_OAUTH_DIR"] ??
    path.join(os.homedir(), ".agentic", "mcp-oauth")
  );
}

const CACHE_SCHEMA_VERSION = 1;

interface CachePayload {
  schema_version?: number;
  server_name?: string;
  server_url?: string;
  updated_at?: string;
  client_id?: string;
  configured_scope?: string | null;
  tokens?: Record<string, unknown>;
  client?: Record<string, unknown>;
}

function secondsUntilExpiry(rawExpiry: unknown): number | undefined {
  if (typeof rawExpiry !== "string" || rawExpiry === "") return undefined;
  const expiry = new Date(rawExpiry);
  if (Number.isNaN(expiry.getTime())) return undefined;
  return Math.max(0, Math.floor((expiry.getTime() - Date.now()) / 1000));
}

/** Token storage backed by the shared `agentengine dev` cache file. */
class FileOAuthTokenStorage {
  constructor(
    private readonly serverName: string,
    private readonly serverUrl: string,
    private readonly cacheDir: string = mcpOauthCacheDir(),
    private readonly clientId?: string,
    private readonly configuredScope?: string,
  ) {}

  get path(): string {
    return path.join(
      this.cacheDir,
      `${mcpOauthCacheName(this.serverName, this.serverUrl)}.json`,
    );
  }

  private readPayload(): CachePayload {
    try {
      const raw = fs.readFileSync(this.path, "utf-8");
      const parsed: unknown = JSON.parse(raw);
      return parsed !== null && typeof parsed === "object"
        ? (parsed as CachePayload)
        : {};
    } catch {
      return {};
    }
  }

  private payloadMatches(payload: CachePayload): boolean {
    if (payload.server_url !== this.serverUrl) return false;
    if (this.clientId !== undefined) {
      return (
        payload.client_id === this.clientId &&
        payload.configured_scope === (this.configuredScope ?? null)
      );
    }
    return true;
  }

  /**
   * Read-modify-write under an advisory file lock.
   *
   * The final write is atomic (tmp file + rename), but without a lock around
   * the read-modify-write as a whole, two processes racing a token refresh
   * for the same server (e.g. two concurrent Tool Pod invocations) can
   * clobber each other's update: the second writer's `readPayload()` can run
   * before the first writer's rename lands, silently discarding it.
   */
  private async writePayload(values: Partial<CachePayload>): Promise<void> {
    fs.mkdirSync(this.cacheDir, { recursive: true, mode: 0o700 });

    let release: (() => Promise<void>) | undefined;
    try {
      // `realpath: false` — the cache file may not exist yet on first write,
      // and resolving symlinks isn't needed for a same-process-family lock.
      // Retries require the async API (`lockSync` rejects a `retries` option).
      release = await lock(this.path, {
        realpath: false,
        stale: 5000,
        retries: { retries: 10, minTimeout: 20, maxTimeout: 200 },
      });
    } catch (err) {
      throw new Error(
        `failed to acquire MCP OAuth cache lock for ${JSON.stringify(this.path)}: ` +
          `${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }

    try {
      const payload = this.readPayload();
      Object.assign(payload, {
        schema_version: CACHE_SCHEMA_VERSION,
        server_name: this.serverName,
        server_url: this.serverUrl,
        updated_at: new Date().toISOString(),
      });
      if (this.clientId !== undefined) {
        payload.client_id = this.clientId;
        payload.configured_scope = this.configuredScope ?? null;
      }
      Object.assign(payload, values);

      const tmpName = path.join(
        this.cacheDir,
        `.tmp-${mcpOauthCacheName(this.serverName, this.serverUrl)}-${process.pid}-${Date.now()}`,
      );
      fs.writeFileSync(tmpName, `${JSON.stringify(payload, null, 2)}\n`, {
        encoding: "utf-8",
        mode: 0o600,
      });
      fs.renameSync(tmpName, this.path);
    } finally {
      await release();
    }
  }

  getTokens(): OAuthTokens | undefined {
    const payload = this.readPayload();
    if (!this.payloadMatches(payload)) return undefined;
    const tokenData = payload.tokens;
    if (
      !tokenData ||
      typeof tokenData !== "object" ||
      !tokenData["access_token"]
    ) {
      return undefined;
    }
    let expiresIn = secondsUntilExpiry(tokenData["expiry"]);
    if (
      expiresIn === undefined &&
      typeof tokenData["expires_in"] === "number"
    ) {
      expiresIn = tokenData["expires_in"];
    }
    return {
      access_token: String(tokenData["access_token"]),
      token_type: (tokenData["token_type"] as string | undefined) || "Bearer",
      refresh_token: tokenData["refresh_token"] as string | undefined,
      expires_in: expiresIn,
      scope: tokenData["scope"] as string | undefined,
    } as OAuthTokens;
  }

  async saveTokens(tokens: OAuthTokens): Promise<void> {
    const tokenData: Record<string, unknown> = { ...tokens };
    if (typeof tokenData["expires_in"] === "number") {
      const expiry = new Date(
        Date.now() + (tokenData["expires_in"] as number) * 1000,
      );
      tokenData["expiry"] = expiry.toISOString();
      delete tokenData["expires_in"];
    }
    await this.writePayload({ tokens: tokenData });
  }

  getClientInfo(): OAuthClientInformationFull | undefined {
    const payload = this.readPayload();
    if (!this.payloadMatches(payload)) return undefined;
    const clientData = payload.client;
    if (
      !clientData ||
      typeof clientData !== "object" ||
      !clientData["client_id"]
    ) {
      return undefined;
    }
    return {
      redirect_uris: (clientData["redirect_uris"] as string[] | undefined) ?? [
        DEFAULT_MCP_OAUTH_REDIRECT_URI,
      ],
      token_endpoint_auth_method: clientData["token_endpoint_auth_method"] as
        | string
        | undefined,
      grant_types: (clientData["grant_types"] as string[] | undefined) ?? [
        "authorization_code",
        "refresh_token",
      ],
      response_types: (clientData["response_types"] as
        | string[]
        | undefined) ?? ["code"],
      scope: clientData["scope"] as string | undefined,
      client_name: clientData["client_name"] as string | undefined,
      client_id: clientData["client_id"] as string,
      client_secret: clientData["client_secret"] as string | undefined,
      client_id_issued_at: clientData["client_id_issued_at"] as
        | number
        | undefined,
      client_secret_expires_at: clientData["client_secret_expires_at"] as
        | number
        | undefined,
    } as OAuthClientInformationFull;
  }

  async saveClientInfo(clientInfo: OAuthClientInformationFull): Promise<void> {
    await this.writePayload({
      client: clientInfo as unknown as Record<string, unknown>,
    });
  }
}

const READABLE_NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;
const UNSAFE_NAME_CHAR_RE = /[^A-Za-z0-9_-]+/g;
const LEADING_TRAILING_DASH_UNDERSCORE_RE = /^[-_]+|[-_]+$/g;

/**
 * Return a collision-free cache basename scoped to alias and endpoint.
 * agent.yaml imposes no charset on aliases, so uniqueness comes from hashing
 * the raw alias whenever the readable form would lose information, and from
 * the endpoint hash that stops the same alias sharing credentials across
 * different MCP servers.
 */
export function mcpOauthCacheName(
  serverName: string,
  serverUrl: string,
): string {
  return `${cacheBase(serverName)}-${sha256Hex(serverUrl)}`;
}

function cacheBase(name: string): string {
  if (READABLE_NAME_RE.test(name)) return name;
  let safe = name
    .trim()
    .replace(UNSAFE_NAME_CHAR_RE, "-")
    .replace(LEADING_TRAILING_DASH_UNDERSCORE_RE, "");
  if (safe.length > 40) {
    safe = safe.slice(0, 40).replace(LEADING_TRAILING_DASH_UNDERSCORE_RE, "");
  }
  const rawHash = sha256Hex(name).slice(0, 16);
  return safe === "" ? rawHash : `${safe}-${rawHash}`;
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

const INTERACTIVE_LOGIN_REQUIRED_MESSAGE = (serverName: string): string =>
  `mcp server '${serverName}' requires cached OAuth credentials; run 'agentengine dev mcp auth login ${serverName}'`;

/**
 * Non-interactive OAuth provider for the interactive `oauth` auth mode.
 *
 * Reads/refreshes tokens the CLI's `agentengine dev mcp auth login` already
 * cached to `mcpOauthCacheDir()`. Never initiates a fresh authorization
 * flow itself — the runtime process has no browser/loopback listener, so any
 * attempt to do so raises telling the user to run the CLI login command.
 */
class FileBackedOAuthProvider implements OAuthClientProvider {
  private readonly storage: FileOAuthTokenStorage;
  private readonly _clientMetadata: OAuthClientMetadata;

  constructor(
    private readonly serverName: string,
    config: RuntimeMCPServerConfig,
    redirectUri: string,
    cacheDir?: string,
  ) {
    this.storage = new FileOAuthTokenStorage(
      serverName,
      config.url,
      cacheDir,
      undefined,
      config.auth.scope ?? undefined,
    );
    this._clientMetadata = {
      redirect_uris: [redirectUri],
      client_name: config.auth.client_name ?? DEFAULT_MCP_OAUTH_CLIENT_NAME,
      scope: config.auth.scope,
      grant_types: ["authorization_code", "refresh_token"],
      response_types: ["code"],
      // A loopback client cannot keep a client secret. Omitting this lets
      // servers apply RFC 7591's client_secret_basic default and mint a
      // secret the token exchange then has no way to send.
      token_endpoint_auth_method: "none",
    } as OAuthClientMetadata;
  }

  get redirectUrl(): string {
    const [redirectUri] = this._clientMetadata.redirect_uris;
    if (redirectUri === undefined) {
      throw new Error("MCP OAuth client metadata has no redirect_uris");
    }
    return redirectUri;
  }

  get clientMetadata(): OAuthClientMetadata {
    return this._clientMetadata;
  }

  clientInformation(): OAuthClientInformationFull | undefined {
    return this.storage.getClientInfo();
  }

  async saveClientInformation(info: OAuthClientInformationFull): Promise<void> {
    await this.storage.saveClientInfo(info);
  }

  tokens(): OAuthTokens | undefined {
    return this.storage.getTokens();
  }

  async saveTokens(tokens: OAuthTokens): Promise<void> {
    await this.storage.saveTokens(tokens);
  }

  redirectToAuthorization(): void {
    throw new Error(INTERACTIVE_LOGIN_REQUIRED_MESSAGE(this.serverName));
  }

  saveCodeVerifier(): void {
    throw new Error(INTERACTIVE_LOGIN_REQUIRED_MESSAGE(this.serverName));
  }

  codeVerifier(): string {
    throw new Error(INTERACTIVE_LOGIN_REQUIRED_MESSAGE(this.serverName));
  }
}

/**
 * Return a non-interactive OAuth provider for a configured MCP server.
 *
 * `cacheDir` overrides `mcpOauthCacheDir()` — used by tests and any caller
 * that needs an isolated cache location instead of the shared `agentengine dev`
 * cache directory.
 */
export function makeMcpOauthAuth(
  serverName: string,
  config: RuntimeMCPServerConfig,
  cacheDir?: string,
): OAuthClientProvider {
  const redirectUri =
    config.auth.redirect_uri ?? DEFAULT_MCP_OAUTH_REDIRECT_URI;
  return new FileBackedOAuthProvider(serverName, config, redirectUri, cacheDir);
}

/**
 * `client_credentials` OAuth provider with file-backed token caching.
 *
 * Wraps the SDK's `ClientCredentialsProvider` (which only caches tokens
 * in-memory) so refreshed tokens persist across process restarts, matching
 * Python's `CachedClientCredentialsOAuthProvider`. When `auth.token_url` is
 * configured (Atlas-style out-of-band token endpoint), pre-seeds discovery
 * state so the SDK's `auth()` orchestrator skips RFC 9728/8414 discovery and
 * posts directly to that endpoint — mirroring Python's
 * `DirectClientCredentialsOAuthProvider`.
 */
class FileBackedClientCredentialsProvider implements OAuthClientProvider {
  private readonly inner: ClientCredentialsProvider;
  private readonly storage: FileOAuthTokenStorage;
  private readonly tokenUrl: string | null;

  constructor(
    serverName: string,
    config: RuntimeMCPServerConfig,
    clientId: string,
    clientSecret: string,
    cacheDir?: string,
  ) {
    this.inner = new ClientCredentialsProvider({
      clientId,
      clientSecret,
      scope: config.auth.scope ?? undefined,
    });
    this.storage = new FileOAuthTokenStorage(
      serverName,
      config.url,
      cacheDir,
      clientId,
      config.auth.scope ?? undefined,
    );
    this.tokenUrl = config.auth.token_url;
  }

  get redirectUrl(): undefined {
    return undefined;
  }

  get clientMetadata(): OAuthClientMetadata {
    return this.inner.clientMetadata;
  }

  clientInformation() {
    return this.inner.clientInformation();
  }

  saveClientInformation(info: OAuthClientInformationFull): void {
    this.inner.saveClientInformation(info);
  }

  tokens(): OAuthTokens | undefined {
    return this.storage.getTokens();
  }

  async saveTokens(tokens: OAuthTokens): Promise<void> {
    await this.storage.saveTokens(tokens);
    this.inner.saveTokens(tokens);
  }

  redirectToAuthorization(): void {
    this.inner.redirectToAuthorization();
  }

  saveCodeVerifier(): void {
    this.inner.saveCodeVerifier();
  }

  codeVerifier(): string {
    return this.inner.codeVerifier();
  }

  prepareTokenRequest(scope?: string) {
    return this.inner.prepareTokenRequest(scope);
  }

  discoveryState() {
    if (this.tokenUrl === null) return undefined;
    return {
      authorizationServerUrl: this.tokenUrl,
      authorizationServerMetadata: {
        issuer: this.tokenUrl,
        authorization_endpoint: this.tokenUrl,
        token_endpoint: this.tokenUrl,
        response_types_supported: ["token"],
      },
    };
  }

  saveDiscoveryState(): void {
    // No-op: the token URL is static config, not something discovery
    // produces — nothing to persist beyond what discoveryState() returns.
  }
}

function requiredEnvValue(
  serverName: string,
  fieldName: string,
  envName: string | null,
): string {
  const value = envName === null ? undefined : process.env[envName];
  if (value === undefined || value.trim() === "") {
    throw new Error(
      `mcp server '${serverName}' auth.${fieldName} references missing env var ${envName}`,
    );
  }
  return value.trim();
}

/**
 * Return a client-credentials OAuth provider for a configured MCP server.
 *
 * `cacheDir` overrides `mcpOauthCacheDir()` — used by tests and any caller
 * that needs an isolated cache location instead of the shared `agentengine dev`
 * cache directory.
 */
export function makeMcpClientCredentialsAuth(
  serverName: string,
  config: RuntimeMCPServerConfig,
  cacheDir?: string,
): OAuthClientProvider {
  const auth = config.auth;
  const clientId = requiredEnvValue(
    serverName,
    "client_id_env",
    auth.client_id_env,
  );
  const clientSecret = requiredEnvValue(
    serverName,
    "client_secret_env",
    auth.client_secret_env,
  );
  return new FileBackedClientCredentialsProvider(
    serverName,
    config,
    clientId,
    clientSecret,
    cacheDir,
  );
}
