/**
 * Framework-neutral helpers for remote MCP tool discovery and execution.
 *
 * Port of `agent_engine_runner_shared/mcp_tools.py`. Built directly on
 * `@modelcontextprotocol/client` (mirroring Python's `mcp` PyPI package) rather
 * than a framework-specific adapter, since the platform needs custom result
 * normalization, egress host derivation, and the Tool-Pod/AER dispatch split
 * that a framework wrapper doesn't support.
 */

import { TextDecoder, TextEncoder } from "node:util";

import {
  Client,
  StreamableHTTPClientTransport,
  type FetchLike,
  type OAuthClientProvider,
} from "@modelcontextprotocol/client";

import type {
  RuntimeMCPConfig,
  RuntimeMCPServerConfig,
} from "./agent_config.js";
import { toolsFromPayload } from "./mcp_tool_pages.js";

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
  readonly annotations?: Record<string, unknown>;
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
  annotations?: Record<string, unknown>;
}

/**
 * Return a JSON-safe copy of MCP `Tool.annotations`, or `undefined` when the
 * value is missing or not a serializable plain object.
 */
export function normalizeMcpToolAnnotations(
  value: unknown,
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  try {
    const parsed: unknown = JSON.parse(JSON.stringify(value));
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return undefined;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return undefined;
  }
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
        ...(tool.annotations !== undefined && {
          annotations: tool.annotations,
        }),
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
    async (client, timeoutMs) => {
      const result = await client.callTool(
        {
          name: binding.toolName,
          arguments: args,
        },
        { timeout: timeoutMs },
      );
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
  return withMcpSession(
    serverName,
    config,
    "tools/list",
    async (client, timeoutMs) => {
      // v2's no-cursor path aggregates every page and applies its SEP-2243
      // tool filtering; listMaxPages keeps the runner's existing page bound.
      const result = await client.listTools(undefined, { timeout: timeoutMs });
      const tools: DiscoveredTool[] = result.tools.map((tool) => {
        const annotations = normalizeMcpToolAnnotations(tool.annotations);
        return {
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema as Record<string, unknown>,
          ...(annotations !== undefined && { annotations }),
        };
      });
      if (tools.length > MAX_TOOLS_PER_SERVER) {
        throw new MCPConfigError(
          `mcp server '${serverName}' returned more than ${MAX_TOOLS_PER_SERVER} tools`,
        );
      }
      return tools;
    },
  );
}

interface ToolListLimitState {
  listed: number;
  exceeded: boolean;
}

/**
 * Count each `tools/list` page as it arrives and abort before the SDK fetches
 * another page once the catalog exceeds MAX_TOOLS_PER_SERVER.
 *
 * The SDK's no-cursor `listTools` path owns pagination (and its SEP-2243 tool
 * filtering), so counting at the fetch seam is the only way to stop mid-walk
 * without replacing that result handling with a lower-level request path.
 */
function createToolsListFetch(
  serverName: string,
  state: ToolListLimitState,
): FetchLike {
  const baseFetch: FetchLike = (input, init) => globalThis.fetch(input, init);
  return async (input, init) => {
    const response = await baseFetch(input, init);
    const request = await mcpRequest(init);
    const contentType = response.headers.get("content-type") ?? "";
    const isSse = contentType.toLowerCase().includes("text/event-stream");

    if (request?.method === "tools/list") {
      if (isSse) {
        // Return a streaming tee: the SDK must receive pings and incremental
        // results while tools/list is still open, so counting cannot wait for
        // the stream to end.
        return streamToolsListResponse(response, serverName, state, request.id);
      }

      const count = await countToolsInPage(response.clone());
      if (count === undefined) return response;
      state.listed += count;
      if (state.listed > MAX_TOOLS_PER_SERVER) {
        state.exceeded = true;
        throw toolLimitError(serverName);
      }
      return response;
    }

    if (isSse && init?.method?.toUpperCase() === "GET") {
      // A resumed tools/list stream arrives on a bodyless GET; count any
      // tools/list results it carries before forwarding their cursors.
      return streamToolsListResponse(response, serverName, state, undefined);
    }

    return response;
  };
}

function toolLimitError(serverName: string): Error {
  return new Error(
    `mcp server '${serverName}' returned more than ${MAX_TOOLS_PER_SERVER} tools`,
  );
}

interface ParsedSseEvent {
  id?: string;
  event?: string;
  data: string;
}

/**
 * Minimal SSE line parser that preserves priming events.
 *
 * `eventsource-parser` (which the SDK uses) only calls `onEvent` for events
 * with data, so an id-only priming event and its `retry` directive would be
 * lost when re-framing. This parser accepts LF, CRLF, and bare-CR line
 * endings, reassembles multi-line `data:` fields, and exposes both.
 */
class SseEventParser {
  private buffer = "";
  private atLineStart = true;
  private id: string | undefined;
  private event: string | undefined;
  private dataLines: string[] = [];

  constructor(
    private readonly onEvent: (event: ParsedSseEvent) => void,
    private readonly onRetry: (milliseconds: number) => void,
  ) {}

  feed(chunk: string): void {
    this.buffer += chunk;
    this.drain(false);
  }

  end(): void {
    this.drain(true);
  }

  private drain(atEnd: boolean): void {
    while (true) {
      const next = takeSseLine(this.buffer, atEnd, this.atLineStart);
      if (next === undefined) {
        // The remaining buffer is a partial line (or a deferred CR), not a
        // fresh line boundary.
        if (this.buffer !== "") this.atLineStart = false;
        return;
      }
      this.buffer = next.rest;
      this.processLine(next.line);
      this.atLineStart = true;
    }
  }

  private processLine(line: string): void {
    if (line === "") {
      this.dispatch();
      return;
    }
    if (line.startsWith(":")) return;

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    switch (field) {
      case "id":
        this.id = value.includes("\0") ? undefined : value;
        break;
      case "event":
        this.event = value === "" ? undefined : value;
        break;
      case "data":
        this.dataLines.push(value);
        break;
      case "retry":
        if (/^\d+$/.test(value)) this.onRetry(Number(value));
        break;
      default:
        break;
    }
  }

  private dispatch(): void {
    const id = this.id;
    const event = this.event;
    const hasData = this.dataLines.length > 0;
    const data = this.dataLines.join("\n");
    this.id = undefined;
    this.event = undefined;
    this.dataLines = [];

    if (hasData) {
      this.onEvent({ id, event, data });
    } else if (id !== undefined || event !== undefined) {
      this.onEvent({ id, event, data: "" });
    }
  }
}

function takeSseLine(
  buffer: string,
  atEnd: boolean,
  atLineStart: boolean,
): { line: string; rest: string } | undefined {
  for (let index = 0; index < buffer.length; index++) {
    const char = buffer[index];
    if (char === "\n") {
      return { line: buffer.slice(0, index), rest: buffer.slice(index + 1) };
    }
    if (char === "\r") {
      if (index + 1 < buffer.length) {
        const skip = buffer[index + 1] === "\n" ? 2 : 1;
        return {
          line: buffer.slice(0, index),
          rest: buffer.slice(index + skip),
        };
      }
      // A CR at the start of a line is a blank-line terminator, even when the
      // previous line ended with a bare CR (`\r\r`). Only a CR after content
      // can be the first half of a CRLF split across chunks.
      if (atLineStart && index === 0) {
        return { line: "", rest: "" };
      }
      // Defer until more data or EOF resolves it.
      if (atEnd) return { line: buffer.slice(0, index), rest: "" };
      return undefined;
    }
  }
  return undefined;
}

/**
 * Forward an SSE `tools/list` response while counting complete events.
 *
 * Complete events are re-framed as they arrive: in-stream pings reach -- and
 * are answered by -- the SDK before the server finishes the catalog, priming
 * ids and retry directives survive, and a capped page is replaced with a
 * JSON-RPC error instead of waiting or forwarding its cursor.
 */
function streamToolsListResponse(
  response: Awaited<ReturnType<FetchLike>>,
  serverName: string,
  state: ToolListLimitState,
  requestId: unknown,
): Awaited<ReturnType<FetchLike>> {
  if (response.body === null) return response;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const sendLimitError = (
    controller: ReadableStreamDefaultController<Uint8Array>,
    id: unknown,
  ): void => {
    // Settle the in-flight JSON-RPC request explicitly; erroring the body
    // instead can leave the SDK's SSE reader waiting for a response.
    const payload = JSON.stringify({
      jsonrpc: "2.0",
      id: id ?? null,
      error: {
        code: -32603,
        message: `mcp server '${serverName}' returned more than ${MAX_TOOLS_PER_SERVER} tools`,
      },
    });
    controller.enqueue(encoder.encode(`event: message\ndata: ${payload}\n\n`));
    controller.close();
  };

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let exceeded = false;
      let limitRequestId = requestId;
      const parser = new SseEventParser(
        (event) => {
          if (exceeded) return;
          const tools = parseSseTools(event);
          if (tools !== undefined) {
            state.listed += tools.count;
            if (state.listed > MAX_TOOLS_PER_SERVER) {
              exceeded = true;
              limitRequestId = tools.id ?? requestId;
              return;
            }
          }
          controller.enqueue(encoder.encode(sseFrame(event)));
        },
        (milliseconds) => {
          if (!exceeded) {
            controller.enqueue(encoder.encode(`retry: ${milliseconds}\n\n`));
          }
        },
      );

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            parser.feed(decoder.decode());
            parser.end();
            if (exceeded) {
              sendLimitError(controller, limitRequestId);
              return;
            }
            controller.close();
            return;
          }

          parser.feed(decoder.decode(value, { stream: true }));
          if (exceeded) {
            // Do not forward the page that crossed the cap; the SDK must not
            // see its cursor and request another page.
            await reader.cancel().catch(() => {});
            sendLimitError(controller, limitRequestId);
            return;
          }
        }
      } catch (err) {
        controller.error(err);
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });

  const headers = new Headers(response.headers);
  // The wrapper re-frames the events, so the original byte length no longer
  // describes this body.
  headers.delete("content-length");

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function sseFrame(event: ParsedSseEvent): string {
  let frame = "";
  if (event.id !== undefined) frame += `id: ${event.id}\n`;
  if (event.event !== undefined) frame += `event: ${event.event}\n`;
  if (event.data !== "") {
    for (const line of event.data.split("\n")) {
      frame += `data: ${line}\n`;
    }
  } else if (event.id !== undefined || event.event !== undefined) {
    // An id-only priming event needs an (empty) data field to make the SDK's
    // SSE parser dispatch it; otherwise the resumption id is dropped.
    frame += "data: \n";
  }
  return `${frame}\n`;
}

function parseSseTools(
  event: ParsedSseEvent,
): { id: unknown; count: number } | undefined {
  if (event.data === "") return undefined;
  try {
    return toolsFromPayload(JSON.parse(event.data));
  } catch {
    return undefined;
  }
}

async function mcpRequest(
  init: Parameters<FetchLike>[1],
): Promise<{ method?: string; id?: unknown } | undefined> {
  const body = init?.body;
  if (typeof body !== "string") return undefined;
  try {
    const message = JSON.parse(body) as { method?: unknown; id?: unknown };
    return {
      ...(typeof message.method === "string" && { method: message.method }),
      id: message.id,
    };
  } catch {
    return undefined;
  }
}

async function countToolsInPage(
  response: Awaited<ReturnType<FetchLike>>,
): Promise<number | undefined> {
  try {
    return toolsFromPayload(await response.json())?.count ?? 0;
  } catch {
    return undefined;
  }
}

async function withMcpSession<T>(
  serverName: string,
  config: RuntimeMCPServerConfig,
  operation: string,
  fn: (client: Client, timeoutMs: number) => Promise<T>,
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
  const isDiscovery = operation === "tools/list";
  const toolLimitState: ToolListLimitState = { listed: 0, exceeded: false };
  const timeoutMs = config.timeout_seconds * 1000;
  const transport = new StreamableHTTPClientTransport(new URL(config.url), {
    requestInit: {
      headers,
      // v2's OAuth metadata/token fetches do not inherit the per-request
      // timeouts passed to connect/listTools/callTool; this bounds them too.
      signal: AbortSignal.timeout(timeoutMs),
    },
    authProvider,
    // The fetch seam is only needed where tools/list pages are counted;
    // tool-call sessions keep the plain fetch and their SSE streams untouched.
    ...(isDiscovery
      ? { fetch: createToolsListFetch(serverName, toolLimitState) }
      : {}),
  });
  const client = new Client(
    {
      name: "agent-engine-runner",
      version: "1.0.0",
    },
    {
      // v2 negotiates legacy by default; auto probes server/discover and
      // falls back to initialize, so both protocol eras stay reachable.
      versionNegotiation: { mode: "auto" },
      // The no-cursor listTools path aggregates pages client-side; keep the
      // runner's existing 100-page bound instead of the SDK's default of 64.
      listMaxPages: MAX_LIST_TOOL_PAGES,
    },
  );

  try {
    await client.connect(transport, { timeout: timeoutMs });
    return await fn(client, timeoutMs);
  } catch (err) {
    // The fetch seam aborts aggregation once the catalog crosses the cap; the
    // SDK may wrap that error, so report the limit from the captured state.
    if (toolLimitState.exceeded) {
      throw new MCPConfigError(
        `mcp server '${serverName}' returned more than ${MAX_TOOLS_PER_SERVER} tools`,
      );
    }
    if (isListPaginationExceeded(err)) {
      // Keep the runner-owned error text instead of the SDK's internal
      // listMaxPages wording, which can change on any minor bump.
      throw new MCPConfigError(
        `mcp server '${serverName}' ${operation} exceeded ${MAX_LIST_TOOL_PAGES} pages`,
      );
    }
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

function isListPaginationExceeded(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const code = (err as { code?: unknown }).code;
  return (
    code === "LIST_PAGINATION_EXCEEDED" ||
    /exceeded listMaxPages/i.test(err.message)
  );
}
