/**
 * Tests for MCP tool discovery/invocation helpers.
 *
 * Mirrors the pure-logic coverage in Python's tests/unit/test_mcp_tools.py.
 * The transport/session-level paths (`discoverMcpTools`'s pagination and
 * caps, `withMcpSession`'s error-class selection) are covered below in
 * "discoverMcpTools / callMcpTool (mocked transport)" by mocking
 * `@modelcontextprotocol/client`'s `Client`, mirroring how Python's
 * `test_mcp_tools.py` monkeypatches the `mcp` client for the same scenarios.
 */

import { describe, test, expect, vi, afterEach } from "vitest";
import {
  MCPConfigError,
  MCPToolError,
  RuntimeMCPConfigSchema,
  RuntimeMCPServerConfigSchema,
  callMcpTool,
  discoverMcpTools,
  isConfiguredMcpSdkToolName,
  makeMcpSdkToolName,
  mcpServerNetworkHosts,
  normalizeMcpToolAnnotations,
  normalizeMcpToolResult,
  resolveConfiguredMcpToolBinding,
  resolveMcpHeaders,
} from "../../src/index.js";
import { toolsFromPayload } from "../../src/mcp_tool_pages.js";

const { mockConnect, mockListTools, mockCallTool, mockClose } = vi.hoisted(
  () => ({
    mockConnect: vi.fn(async () => {}),
    mockListTools: vi.fn(),
    mockCallTool: vi.fn(),
    mockClose: vi.fn(async () => {}),
  }),
);

vi.mock("@modelcontextprotocol/client", () => ({
  // A regular `function` (not an arrow function) so `new Client(...)` in the
  // source under test can invoke this as a constructor — returning an object
  // from a constructor function replaces the `this` instance with it.
  Client: vi.fn().mockImplementation(function MockClient() {
    return {
      connect: mockConnect,
      listTools: mockListTools,
      callTool: mockCallTool,
      close: mockClose,
    };
  }),
  // The production code constructs a transport before calling `connect`;
  // the mocked client never uses it, but the import must resolve.
  StreamableHTTPClientTransport: vi
    .fn()
    .mockImplementation(function MockTransport() {}),
}));

afterEach(() => {
  vi.unstubAllEnvs();
  mockConnect.mockReset().mockImplementation(async () => {});
  mockListTools.mockReset();
  mockCallTool.mockReset();
  mockClose.mockReset().mockImplementation(async () => {});
});

function makeServerConfig(overrides: Record<string, unknown> = {}) {
  return RuntimeMCPServerConfigSchema.parse({
    url: "https://mcp.example.com/github",
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// makeMcpSdkToolName
// ---------------------------------------------------------------------------

describe("makeMcpSdkToolName", () => {
  test("joins sanitized server + tool name", () => {
    expect(makeMcpSdkToolName("github", "search_issues")).toBe(
      "github__search_issues",
    );
  });

  test("sanitizes punctuation to underscores", () => {
    expect(makeMcpSdkToolName("my-server.v2", "list-repos")).toBe(
      "my_server_v2__list_repos",
    );
  });

  test("prefixes a leading digit", () => {
    expect(makeMcpSdkToolName("1server", "1tool")).toBe(
      "mcp_1server__mcp_1tool",
    );
  });

  test("throws when a name has no alphanumeric characters", () => {
    expect(() => makeMcpSdkToolName("---", "tool")).toThrow(MCPConfigError);
  });
});

// ---------------------------------------------------------------------------
// resolveMcpHeaders
// ---------------------------------------------------------------------------

describe("resolveMcpHeaders", () => {
  test("returns static headers unchanged for auth.type none", () => {
    const config = makeServerConfig({ headers: { "X-Custom": "value" } });
    expect(resolveMcpHeaders("github", config)).toEqual({
      "X-Custom": "value",
    });
  });

  test("adds a Bearer header for bearer_env", () => {
    vi.stubEnv("GITHUB_MCP_TOKEN", "secret-token");
    const config = makeServerConfig({
      auth: { type: "bearer_env", token_env: "GITHUB_MCP_TOKEN" },
    });
    expect(resolveMcpHeaders("github", config)).toEqual({
      Authorization: "Bearer secret-token",
    });
  });

  test("throws when the bearer_env env var is unset", () => {
    const config = makeServerConfig({
      auth: { type: "bearer_env", token_env: "MISSING_TOKEN_VAR" },
    });
    expect(() => resolveMcpHeaders("github", config)).toThrow(MCPConfigError);
  });

  test("does not add a header for oauth/client_credentials (handled by the auth provider)", () => {
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "ID",
        client_secret_env: "SECRET",
      },
    });
    expect(resolveMcpHeaders("github", config)).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// mcpServerNetworkHosts
// ---------------------------------------------------------------------------

describe("mcpServerNetworkHosts", () => {
  test("returns the server hostname", () => {
    const config = makeServerConfig({ url: "https://mcp.example.com/github" });
    expect(mcpServerNetworkHosts(config)).toEqual(["mcp.example.com"]);
  });

  test("includes the token_url host when configured", () => {
    const config = makeServerConfig({
      auth: {
        type: "client_credentials",
        client_id_env: "ID",
        client_secret_env: "SECRET",
        token_url: "https://auth.example.com/token",
      },
    });
    expect(mcpServerNetworkHosts(config)).toEqual([
      "mcp.example.com",
      "auth.example.com",
    ]);
  });

  test("deduplicates when the token host matches the server host", () => {
    const config = makeServerConfig({
      url: "https://mcp.example.com/github",
      auth: {
        type: "client_credentials",
        client_id_env: "ID",
        client_secret_env: "SECRET",
        token_url: "https://mcp.example.com/token",
      },
    });
    expect(mcpServerNetworkHosts(config)).toEqual(["mcp.example.com"]);
  });
});

// ---------------------------------------------------------------------------
// resolveConfiguredMcpToolBinding / isConfiguredMcpSdkToolName
// ---------------------------------------------------------------------------

describe("resolveConfiguredMcpToolBinding", () => {
  function makeConfig() {
    return RuntimeMCPConfigSchema.parse({
      servers: {
        github: { url: "https://mcp.example.com/github" },
      },
    });
  }

  test("resolves a binding for a configured server/tool", () => {
    const config = makeConfig();
    const binding = resolveConfiguredMcpToolBinding(
      config,
      "github__search_issues",
      "github",
      "search_issues",
    );
    expect(binding).toMatchObject({
      serverName: "github",
      toolName: "search_issues",
      sdkToolName: "github__search_issues",
    });
  });

  test("returns null for an unconfigured server", () => {
    const config = makeConfig();
    expect(
      resolveConfiguredMcpToolBinding(config, "x", "not_configured", "y"),
    ).toBeNull();
  });

  test("throws when the sdk tool name does not match the derived name", () => {
    const config = makeConfig();
    expect(() =>
      resolveConfiguredMcpToolBinding(
        config,
        "wrong_name",
        "github",
        "search_issues",
      ),
    ).toThrow(MCPConfigError);
  });

  test("throws when the tool is not in allowed_tools", () => {
    const config = RuntimeMCPConfigSchema.parse({
      servers: {
        github: {
          url: "https://mcp.example.com/github",
          allowed_tools: ["search_issues"],
        },
      },
    });
    expect(() =>
      resolveConfiguredMcpToolBinding(
        config,
        "github__delete_repo",
        "github",
        "delete_repo",
      ),
    ).toThrow(/not in allowed_tools/);
  });

  test("isConfiguredMcpSdkToolName matches only configured server prefixes", () => {
    const config = makeConfig();
    expect(isConfiguredMcpSdkToolName(config, "github__search_issues")).toBe(
      true,
    );
    expect(isConfiguredMcpSdkToolName(config, "notion__search")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// normalizeMcpToolResult
// ---------------------------------------------------------------------------

describe("normalizeMcpToolResult", () => {
  test("passes through content and structuredContent on success", () => {
    const result = normalizeMcpToolResult({
      content: [{ type: "text", text: "hello" }],
      structuredContent: { count: 3 },
      isError: false,
    });
    expect(result).toEqual({
      content: [{ type: "text", text: "hello" }],
      structuredContent: { count: 3 },
      isError: false,
    });
  });

  test("defaults content to [] and structuredContent to null", () => {
    const result = normalizeMcpToolResult({});
    expect(result).toEqual({
      content: [],
      structuredContent: null,
      isError: false,
    });
  });

  test("throws MCPToolError with the text block when isError is true", () => {
    expect(() =>
      normalizeMcpToolResult({
        content: [{ type: "text", text: "rate limited" }],
        isError: true,
      }),
    ).toThrow(/rate limited/);
  });

  test("throws a generic MCPToolError when isError is true with no text block", () => {
    expect(() =>
      normalizeMcpToolResult({ content: [], isError: true }),
    ).toThrow(MCPToolError);
  });

  test("throws when structuredContent is not an object", () => {
    expect(() =>
      normalizeMcpToolResult({
        content: [],
        structuredContent: "not-an-object",
      }),
    ).toThrow(/not an object/);
  });
});

// ---------------------------------------------------------------------------
// discoverMcpTools / callMcpTool (mocked transport)
// ---------------------------------------------------------------------------
//
// Mocks @modelcontextprotocol/client's `Client` so listServerTools' pagination
// and caps, and withMcpSession's MCPConfigError-vs-MCPToolError selection,
// are exercised without a real network dependency — mirroring the monkeypatch
// coverage in Python's test_mcp_tools.py.

function makeDiscoveryConfig(overrides: Record<string, unknown> = {}) {
  return RuntimeMCPConfigSchema.parse({
    servers: {
      github: { url: "https://mcp.example.com/github", ...overrides },
    },
  });
}

describe("toolsFromPayload", () => {
  test("counts a JSON-RPC batch tools page", () => {
    expect(
      toolsFromPayload([
        {
          jsonrpc: "2.0",
          id: 7,
          result: { tools: [{ name: "a" }, { name: "b" }] },
        },
      ]),
    ).toEqual({ id: 7, count: 2 });
  });

  test("sums multiple tools results in one batch", () => {
    expect(
      toolsFromPayload([
        { id: 1, result: { tools: [{}] } },
        { id: 2, result: { tools: [{}, {}] } },
      ]),
    ).toEqual({ id: 2, count: 3 });
  });

  test("ignores payloads without a tools result", () => {
    expect(toolsFromPayload({ id: 1, result: {} })).toBeUndefined();
  });
});

describe("discoverMcpTools — pagination and caps", () => {
  test("uses the SDK's aggregated page result in one listTools call", async () => {
    mockListTools.mockResolvedValueOnce({
      tools: [
        { name: "tool_a", inputSchema: { type: "object" } },
        { name: "tool_b", inputSchema: { type: "object" } },
      ],
    });

    const bindings = await discoverMcpTools(makeDiscoveryConfig());

    expect(mockListTools).toHaveBeenCalledTimes(1);
    expect(mockListTools).toHaveBeenCalledWith(undefined, {
      timeout: 30_000,
    });
    expect(bindings.map((b) => b.toolName).sort()).toEqual([
      "tool_a",
      "tool_b",
    ]);
  });

  test("wraps the SDK pagination-limit error as MCPConfigError", async () => {
    mockListTools.mockRejectedValueOnce(
      new Error(
        "tools/list: exceeded listMaxPages (100); server pagination did not terminate",
      ),
    );

    await expect(discoverMcpTools(makeDiscoveryConfig())).rejects.toMatchObject(
      {
        name: "MCPConfigError",
        message: expect.stringContaining("tools/list exceeded 100 pages"),
      },
    );
  });

  test("rejects when a server returns more than MAX_TOOLS_PER_SERVER tools", async () => {
    const tooManyTools = Array.from({ length: 10_001 }, (_, i) => ({
      name: `tool_${i}`,
      inputSchema: { type: "object" },
    }));
    mockListTools.mockResolvedValueOnce({
      tools: tooManyTools,
      nextCursor: undefined,
    });

    await expect(discoverMcpTools(makeDiscoveryConfig())).rejects.toThrow(
      /returned more than 10000 tools/,
    );
  });

  test("rejects when allowed_tools names a tool the server doesn't expose", async () => {
    mockListTools.mockResolvedValueOnce({
      tools: [{ name: "search_issues", inputSchema: { type: "object" } }],
      nextCursor: undefined,
    });

    await expect(
      discoverMcpTools(
        makeDiscoveryConfig({ allowed_tools: ["search_issues", "missing"] }),
      ),
    ).rejects.toThrow(/did not expose configured allowed_tools: missing/);
  });

  test("wraps a connect failure as MCPConfigError with the operation in the message", async () => {
    mockConnect.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    await expect(discoverMcpTools(makeDiscoveryConfig())).rejects.toMatchObject(
      {
        name: "MCPConfigError",
        message: expect.stringContaining("tools/list"),
      },
    );
  });
});

describe("normalizeMcpToolAnnotations", () => {
  test("returns a JSON-safe copy of a plain annotations object", () => {
    const annotations = {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
      title: "Search",
    };
    expect(normalizeMcpToolAnnotations(annotations)).toEqual(annotations);
  });

  test("omits missing, null, and malformed values instead of inventing defaults", () => {
    expect(normalizeMcpToolAnnotations(undefined)).toBeUndefined();
    expect(normalizeMcpToolAnnotations(null)).toBeUndefined();
    expect(normalizeMcpToolAnnotations("read-only")).toBeUndefined();
    expect(normalizeMcpToolAnnotations(["readOnlyHint"])).toBeUndefined();
    expect(normalizeMcpToolAnnotations(true)).toBeUndefined();
  });

  test("omits non-JSON-serializable objects rather than failing discovery", () => {
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;
    expect(normalizeMcpToolAnnotations(circular)).toBeUndefined();
  });
});

describe("discoverMcpTools — annotations", () => {
  test("preserves standard hints and extra keys from tools/list", async () => {
    mockListTools.mockResolvedValueOnce({
      tools: [
        {
          name: "search_issues",
          description: "Search issues",
          inputSchema: { type: "object" },
          annotations: {
            readOnlyHint: true,
            destructiveHint: false,
            idempotentHint: true,
            openWorldHint: false,
            title: "Search issues",
          },
        },
        {
          name: "create_issue",
          inputSchema: { type: "object" },
          annotations: {
            readOnlyHint: false,
            destructiveHint: false,
          },
        },
        {
          name: "delete_issue",
          inputSchema: { type: "object" },
          annotations: {
            readOnlyHint: false,
            destructiveHint: true,
          },
        },
        {
          name: "unannotated",
          inputSchema: { type: "object" },
        },
      ],
      nextCursor: undefined,
    });

    const bindings = await discoverMcpTools(makeDiscoveryConfig());
    const byName = Object.fromEntries(bindings.map((b) => [b.toolName, b]));

    expect(byName["search_issues"]?.annotations).toEqual({
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
      title: "Search issues",
    });
    expect(byName["create_issue"]?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
    });
    expect(byName["delete_issue"]?.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: true,
    });
    expect(byName["unannotated"]?.annotations).toBeUndefined();
  });

  test("omits malformed annotations and still discovers the tool", async () => {
    mockListTools.mockResolvedValueOnce({
      tools: [
        {
          name: "weird",
          inputSchema: { type: "object" },
          annotations: "not-an-object",
        },
      ],
      nextCursor: undefined,
    });

    const bindings = await discoverMcpTools(makeDiscoveryConfig());
    expect(bindings).toHaveLength(1);
    expect(bindings[0]?.toolName).toBe("weird");
    expect(bindings[0]?.annotations).toBeUndefined();
  });

  test("keeps annotations on tools that survive allowed_tools filtering", async () => {
    mockListTools.mockResolvedValueOnce({
      tools: [
        {
          name: "search_issues",
          inputSchema: { type: "object" },
          annotations: { readOnlyHint: true },
        },
        {
          name: "create_issue",
          inputSchema: { type: "object" },
          annotations: { destructiveHint: false },
        },
      ],
      nextCursor: undefined,
    });

    const bindings = await discoverMcpTools(
      makeDiscoveryConfig({ allowed_tools: ["search_issues"] }),
    );
    expect(bindings).toHaveLength(1);
    expect(bindings[0]?.toolName).toBe("search_issues");
    expect(bindings[0]?.annotations).toEqual({ readOnlyHint: true });
  });
});

describe("callMcpTool — error classification", () => {
  function makeBinding() {
    return resolveConfiguredMcpToolBinding(
      makeDiscoveryConfig(),
      "github__search_issues",
      "github",
      "search_issues",
    );
  }

  test("returns a normalized result on success", async () => {
    mockCallTool.mockResolvedValueOnce({
      content: [{ type: "text", text: "ok" }],
      isError: false,
    });

    const result = await callMcpTool(makeBinding(), { query: "bug" });

    expect(result).toEqual({
      content: [{ type: "text", text: "ok" }],
      structuredContent: null,
      isError: false,
    });
    expect(mockCallTool).toHaveBeenCalledWith(
      {
        name: "search_issues",
        arguments: { query: "bug" },
      },
      { timeout: 30_000 },
    );
  });

  test("wraps a transport failure during tools/call as MCPToolError, not MCPConfigError", async () => {
    mockCallTool.mockRejectedValueOnce(new Error("socket hang up"));

    await expect(
      callMcpTool(makeBinding(), { query: "bug" }),
    ).rejects.toMatchObject({
      name: "MCPToolError",
      message: expect.stringContaining("socket hang up"),
    });
  });

  test("propagates MCPToolError from an MCP-level error result unwrapped", async () => {
    mockCallTool.mockResolvedValueOnce({
      content: [{ type: "text", text: "rate limited upstream" }],
      isError: true,
    });

    await expect(
      callMcpTool(makeBinding(), { query: "bug" }),
    ).rejects.toMatchObject({
      name: "MCPToolError",
      message: expect.stringContaining("rate limited upstream"),
    });
  });
});
