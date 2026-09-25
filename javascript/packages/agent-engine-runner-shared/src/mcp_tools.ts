/**
 * Framework-neutral helpers for remote MCP tool discovery and execution.
 *
 * Port of `agent_engine_runner_shared/mcp_tools.py`. Built directly on
 * `@modelcontextprotocol/sdk` (mirroring Python's `mcp` PyPI package) rather
 * than a framework-specific adapter, since the platform needs custom result
 * normalization, egress host derivation, and the Tool-Pod/AER dispatch split
 * that a framework wrapper doesn't support.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { OAuthClientProvider } from "@modelcontextprotocol/sdk/client/auth.js";

import type {
  RuntimeMCPConfig,
  RuntimeMCPServerConfig,
} from "./agent_config.js";

const UNSAFE_TOOL_NAME_CHARS_RE = /[^a-zA-Z0-9_]/g;
const LEADING_TRAILING_UNDERSCORES_RE = /^_+|_+$/g;
const MAX_LIST_TOOL_PAGES = 100;
const MAX_TOOLS_PER_SERVER = 10_000;

/** Raised when remote MCP configuration cannot be resolved. */
export class MCPConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MCPConfigError";
  }
}

/** Raised when a remote MCP tool returns an MCP error result or a transport failure occurs. */
export class MCPToolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MCPToolError";
  }
}

/** Discovered MCP tool bound to a configured server. */
export interface MCPToolBinding {
  readonly serverName: string;
  readonly toolName: string;
  readonly sdkToolName: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
  readonly serverConfig: RuntimeMCPServerConfig;
}

/** JSON/BSON-safe result returned by an MCP `tools/call` invocation. */
export interface MCPToolResult {
  readonly content: unknown[];
  readonly structuredContent: Record<string, unknown> | null;
  readonly isError: boolean;
}

/** Return the server-prefixed SDK name for a remote MCP tool. */
export function makeMcpSdkToolName(
  serverName: string,
  toolName: string,
): string {
  const serverPart = sanitizeToolNamePart(serverName);
  const toolPart = sanitizeToolNamePart(toolName);
  return `${serverPart}__${toolPart}`;
}

// Provider names can contain punctuation that downstream LLM tool-call APIs
// reject, so the SDK exposes a stable identifier and keeps the original MCP
// tool name only for the outbound tools/call request.
function sanitizeToolNamePart(value: string): string {
  let sanitized = value
    .trim()
    .replace(UNSAFE_TOOL_NAME_CHARS_RE, "_")
    .replace(LEADING_TRAILING_UNDERSCORES_RE, "");
  if (sanitized === "") {
    throw new MCPConfigError(
      "mcp server and tool names must contain alphanumeric characters",
    );
  }
  if (/^[0-9]/.test(sanitized)) {
    sanitized = `mcp_${sanitized}`;
  }
  return sanitized;
}

/** Resolve headers for a remote MCP server without exposing secret values in config. */
export function resolveMcpHeaders(
  serverName: string,
  config: RuntimeMCPServerConfig,
): Record<string, string> {
  const headers = { ...config.headers };
  if (
    config.auth.type === "none" ||
    config.auth.type === "oauth" ||
    config.auth.type === "client_credentials"
  ) {
    return headers;
  }

  const tokenEnv = config.auth.token_env;
  if (tokenEnv === null) {
    throw new MCPConfigError(
      `mcp server '${serverName}' auth.token_env is required`,
    );
  }
  const token = process.env[tokenEnv];
  if (token === undefined || token.trim() === "") {
    throw new MCPConfigError(
      `mcp server '${serverName}' references missing env var ${tokenEnv}`,
    );
  }
  headers["Authorization"] = `Bearer ${token.trim()}`;
  return headers;
}

/** Resolve an MCP SDK OAuth provider for the configured auth mode, if any. */
export async function resolveMcpAuth(
  serverName: string,
  config: RuntimeMCPServerConfig,
): Promise<OAuthClientProvider | undefined> {
  if (config.auth.type === "none" || config.auth.type === "bearer_env") {
    return undefined;
  }

  const { makeMcpOauthAuth, makeMcpClientCredentialsAuth } =
    await import("./mcp_oauth.js");

  try {
    if (config.auth.type === "oauth") {
      return makeMcpOauthAuth(serverName, config);
    }
    if (config.auth.type === "client_credentials") {
      return makeMcpClientCredentialsAuth(serverName, config);
    }
  } catch (err) {
    throw new MCPConfigError(err instanceof Error ? err.message : String(err));
  }

  throw new MCPConfigError(
    `mcp server '${serverName}' has unsupported auth.type ${config.auth.type}`,
  );
}

/** Return the outbound hosts needed to reach a configured MCP server. */
export function mcpServerNetworkHosts(
  config: RuntimeMCPServerConfig,
): string[] {
  const hosts = [new URL(config.url).hostname];
  if (config.auth.token_url !== null) {
    const tokenHost = new URL(config.auth.token_url).hostname;
    if (!hosts.includes(tokenHost)) hosts.push(tokenHost);
  }
  return hosts;
}

interface DiscoveredTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

/**
 * Discover remote MCP tools for all configured servers.
 *
 * Mirrors common MCP client adapters: initialize each server, call
 * `tools/list`, then expose the returned schemas as framework-native tools.
 */
export async function discoverMcpTools(
  config: RuntimeMCPConfig,
): Promise<MCPToolBinding[]> {
  const bindings: MCPToolBinding[] = [];
  const usedNames = new Map<string, string>();

  for (const [serverName, serverConfig] of Object.entries(config.servers)) {
    let tools = await listServerTools(serverName, serverConfig);
    const availableToolNames = new Set(tools.map((tool) => tool.name));
    const allowedTools = serverConfig.allowed_tools;

    if (allowedTools !== null) {
      const missing = allowedTools
        .filter((name) => !availableToolNames.has(name))
        .sort();
      if (missing.length > 0) {
        throw new MCPConfigError(
          `mcp server '${serverName}' did not expose configured allowed_tools: ${missing.join(", ")}`,
        );
      }
      const allowedSet = new Set(allowedTools);
      tools = tools.filter((tool) => allowedSet.has(tool.name));
    }

    for (const tool of tools) {
      const sdkToolName = makeMcpSdkToolName(serverName, tool.name);
      const previous = usedNames.get(sdkToolName);
      if (previous !== undefined) {
        throw new MCPConfigError(
          `mcp tool name collision for '${sdkToolName}': ${previous} and ${serverName}.${tool.name}`,
        );
      }
      usedNames.set(sdkToolName, `${serverName}.${tool.name}`);
      bindings.push({
        serverName,
        toolName: tool.name,
        sdkToolName,
        description: tool.description ?? "",
        inputSchema: tool.inputSchema,
        serverConfig,
      });
    }
  }

  return bindings;
}

/**
 * Map an SDK-visible tool name back to configured MCP call metadata.
 *
 * AER discovers remote MCP schemas at startup and exposes server-prefixed
 * names such as `github__search_issues` to the LLM. Tool Pods skip
 * `tools/list` startup discovery, so call-time execution resolves that SDK
 * name against `agent.yaml` and calls the original MCP tool name.
 */
export function resolveConfiguredMcpToolBinding(
  config: RuntimeMCPConfig,
  sdkToolName: string,
  mcpServerName: string,
  mcpToolName: string,
): MCPToolBinding | null {
  const serverConfig = config.servers[mcpServerName];
  if (serverConfig === undefined) return null;

  const expectedSdkToolName = makeMcpSdkToolName(mcpServerName, mcpToolName);
  if (expectedSdkToolName !== sdkToolName) {
    throw new MCPConfigError(
      `mcp call metadata for '${mcpServerName}'.'${mcpToolName}' does not match '${sdkToolName}'`,
    );
  }

  if (
    serverConfig.allowed_tools !== null &&
    !serverConfig.allowed_tools.includes(mcpToolName)
  ) {
    throw new MCPConfigError(
      `mcp server '${mcpServerName}' tool '${mcpToolName}' is not in allowed_tools`,
    );
  }

  return {
    serverName: mcpServerName,
    toolName: mcpToolName,
    sdkToolName,
    description: "",
    inputSchema: {},
    serverConfig,
  };
}

/** Return whether `sdkToolName` belongs to a configured MCP server. */
export function isConfiguredMcpSdkToolName(
  config: RuntimeMCPConfig,
  sdkToolName: string,
): boolean {
  for (const serverName of Object.keys(config.servers)) {
    const serverPrefix = `${sanitizeToolNamePart(serverName)}__`;
    if (
      sdkToolName.startsWith(serverPrefix) &&
      sdkToolName.length > serverPrefix.length
    ) {
      return true;
    }
  }
  return false;
}

/** Execute a remote MCP tool and normalize its result. */
export async function callMcpTool(
  binding: MCPToolBinding,
  args: Record<string, unknown>,
): Promise<MCPToolResult> {
  return withMcpSession(
    binding.serverName,
    binding.serverConfig,
    `tool '${binding.toolName}'`,
    async (client) => {
      const result = await client.callTool({
        name: binding.toolName,
        arguments: args,
      });
      // `callTool()`'s return type also covers the deprecated `{ toolResult }`
      // shape from pre-2025-06-18 servers; MCP tool servers we target return
      // the standard `content`/`structuredContent`/`isError` shape.
      return normalizeMcpToolResult(
        result as {
          content?: unknown[];
          structuredContent?: unknown;
          isError?: boolean;
        },
      );
    },
    MCPToolError,
  );
}

/** Create the async callable used to invoke a discovered MCP tool. */
export function makeMcpToolCallable(
  binding: MCPToolBinding,
): (args: Record<string, unknown>) => Promise<MCPToolResult> {
  return (args: Record<string, unknown>) => callMcpTool(binding, args);
}

/** Convert an MCP `CallToolResult` into a JSON-safe result, raising on an error result. */
export function normalizeMcpToolResult(result: {
  content?: unknown[];
  structuredContent?: unknown;
  isError?: boolean;
}): MCPToolResult {
  const content = result.content ?? [];
  ensureJsonSafe("content", content);

  const rawStructuredContent = result.structuredContent ?? null;
  if (rawStructuredContent !== null) {
    if (
      typeof rawStructuredContent !== "object" ||
      Array.isArray(rawStructuredContent)
    ) {
      throw new MCPToolError(
        "MCP tool returned structuredContent that is not an object",
      );
    }
    ensureJsonSafe("structuredContent", rawStructuredContent);
  }

  const normalized: MCPToolResult = {
    content,
    structuredContent: rawStructuredContent as Record<string, unknown> | null,
    isError: Boolean(result.isError),
  };

  if (result.isError) {
    const textParts: string[] = [];
    for (const block of content) {
      if (block !== null && typeof block === "object") {
        const b = block as Record<string, unknown>;
        if (
          b["type"] === "text" &&
          typeof b["text"] === "string" &&
          b["text"]
        ) {
          textParts.push(b["text"]);
        }
      }
    }
    if (textParts.length > 0) {
      throw new MCPToolError(
        `MCP tool returned error: ${textParts.join("\n")}`,
      );
    }
    throw new MCPToolError("MCP tool returned an error");
  }

  return normalized;
}

function ensureJsonSafe(fieldName: string, value: unknown): void {
  try {
    JSON.stringify(value);
  } catch {
    throw new MCPToolError(
      `MCP tool returned non-JSON-serializable ${fieldName}`,
    );
  }
}

async function listServerTools(
  serverName: string,
  config: RuntimeMCPServerConfig,
): Promise<DiscoveredTool[]> {
  return withMcpSession(serverName, config, "tools/list", async (client) => {
    const tools: DiscoveredTool[] = [];
    let cursor: string | undefined;
    for (let page = 0; page < MAX_LIST_TOOL_PAGES; page++) {
      const result = await client.listTools({ cursor });
      tools.push(
        ...result.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema as Record<string, unknown>,
        })),
      );
      if (tools.length > MAX_TOOLS_PER_SERVER) {
        throw new MCPConfigError(
          `mcp server '${serverName}' returned more than ${MAX_TOOLS_PER_SERVER} tools`,
        );
      }
      cursor = result.nextCursor;
      if (cursor === undefined) return tools;
    }
    throw new MCPConfigError(
      `mcp server '${serverName}' tools/list exceeded ${MAX_LIST_TOOL_PAGES} pages`,
    );
  });
}

async function withMcpSession<T>(
  serverName: string,
  config: RuntimeMCPServerConfig,
  operation: string,
  fn: (client: Client) => Promise<T>,
  // Discovery (tools/list) failures are config-adjacent — they happen once at
  // startup against a server the tenant configured — so they wrap as
  // MCPConfigError. Invocation (tools/call) failures are per-call runtime
  // errors and wrap as MCPToolError, matching its docstring ("...or a
  // transport failure occurs") and Python's mcp_tools.py split between
  // _list_server_tools (MCPConfigError) and _call_mcp_tool (MCPToolError).
  transportErrorClass:
    | typeof MCPConfigError
    | typeof MCPToolError = MCPConfigError,
): Promise<T> {
  const headers = resolveMcpHeaders(serverName, config);
  const authProvider = await resolveMcpAuth(serverName, config);
  const transport = new StreamableHTTPClientTransport(new URL(config.url), {
    requestInit: {
      headers,
      signal: AbortSignal.timeout(config.timeout_seconds * 1000),
    },
    authProvider,
  });
  const client = new Client({
    name: "agent-engine-runner",
    version: "1.0.0",
  });

  try {
    await client.connect(transport);
    return await fn(client);
  } catch (err) {
    if (err instanceof MCPConfigError || err instanceof MCPToolError) throw err;
    throw new transportErrorClass(
      mcpTransportErrorMessage(serverName, operation, err),
    );
  } finally {
    await client.close().catch(() => {});
  }
}

function mcpTransportErrorMessage(
  serverName: string,
  operation: string,
  err: unknown,
): string {
  const message = err instanceof Error ? err.message : String(err);
  return `mcp server '${serverName}' ${operation} failed: ${message}`;
}
