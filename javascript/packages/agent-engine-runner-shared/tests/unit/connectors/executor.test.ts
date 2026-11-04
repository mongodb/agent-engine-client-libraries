import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";

import { classifyToolAPIError } from "../../../src/tool_api_error.js";
import {
  ConnectorExecutor,
  ConnectorHTTPError,
  type ConnectorTransport,
  type ConnectorTransportRequest,
  type ConnectorTransportResponse,
} from "../../../src/connectors/executor.js";
import {
  loadToolDefs,
  type Operation,
  type ToolDefinitions,
} from "../../../src/connectors/index.js";

const FIXTURE = fileURLToPath(
  new URL("./fixtures/jira.tool_defs.yaml", import.meta.url),
);

function definitions(): ToolDefinitions {
  return structuredClone(loadToolDefs(FIXTURE));
}

function response(
  status = 200,
  body = "",
  headers: Record<string, string> = {},
): ConnectorTransportResponse {
  const bytes = new TextEncoder().encode(body);
  return {
    status,
    headers: new Headers(headers),
    body: (async function* () {
      yield bytes;
    })(),
  };
}

function transport(
  handler: (
    request: ConnectorTransportRequest,
  ) => Promise<ConnectorTransportResponse>,
): { transport: ConnectorTransport; requests: ConnectorTransportRequest[] } {
  const requests: ConnectorTransportRequest[] = [];
  return {
    requests,
    transport: async (request) => {
      requests.push(request);
      return handler(request);
    },
  };
}

function execute(
  executor: ConnectorExecutor,
  name: string,
  arguments_: Record<string, unknown>,
  env: Record<string, string> = { JIRA_TOKEN: "token" },
  token?: string,
): Promise<string> {
  return executor.execute(name, arguments_, {
    env,
    ...(token !== undefined && { authorization: { token } }),
  });
}

function customDefinitions(operation: Operation): ToolDefinitions {
  const value = definitions();
  value.tools = [operation];
  value.auth = null;
  return value;
}

describe("ConnectorExecutor", () => {
  it("executes the Jira read and write operations", async () => {
    const calls = transport(async () => response(200, '{"id":"42"}'));
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: "DEMO-1" }),
    ).resolves.toBe('{"id":"42"}');
    await expect(
      execute(executor, "jiradc_addComment", {
        issueIdOrKey: "DEMO-1",
        body: "Hello",
      }),
    ).resolves.toBe('{"id":"42"}');

    expect(calls.requests).toHaveLength(2);
    expect(calls.requests[0]?.method).toBe("GET");
    expect(calls.requests[0]?.url).toBe(
      "https://jira.example/rest/api/2/issue/DEMO-1/comment",
    );
    expect(calls.requests[0]?.headers["Authorization"]).toBe("Bearer token");
    expect(calls.requests[1]?.method).toBe("POST");
    expect(new TextDecoder().decode(calls.requests[1]?.body)).toBe(
      '{"body":"Hello"}',
    );
  });

  it("rejects an unknown tool and invalid arguments before I/O", async () => {
    const calls = transport(async () => response());
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    await expect(execute(executor, "missing", {})).rejects.toThrow(
      "unknown connector tool",
    );
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: 1 }),
    ).rejects.toThrow("arguments do not match the compiled input schema");
    expect(calls.requests).toHaveLength(0);
  });

  it("uses static auth and ignores a delegated token", async () => {
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    await execute(
      executor,
      "jiradc_getComments",
      { issueIdOrKey: "DEMO-1" },
      { JIRA_TOKEN: "static-token" },
      "delegated-token",
    );
    expect(calls.requests[0]?.headers["Authorization"]).toBe(
      "Bearer static-token",
    );
  });

  it("requires a valid delegated token for OAuth", async () => {
    const value = definitions();
    value.auth = { provider: "github", scopes: ["repo"] };
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(value, {
      transport: calls.transport,
    });
    await expect(
      execute(
        executor,
        "jiradc_getComments",
        { issueIdOrKey: "DEMO-1" },
        {},
        "alice-token",
      ),
    ).resolves.toBe("ok");
    expect(calls.requests[0]?.headers["Authorization"]).toBe(
      "Bearer alice-token",
    );
    for (const token of [
      "",
      "with space",
      "injection\r\nheader",
      "nonascii-é",
    ]) {
      await expect(
        execute(
          executor,
          "jiradc_getComments",
          { issueIdOrKey: "DEMO-1" },
          {},
          token,
        ),
      ).rejects.toThrow("delegated connector token is missing or invalid");
    }
  });

  it("applies header, query, cookie, and path static credentials", async () => {
    const cases = [
      {
        location: "header" as const,
        baseUrl: "https://jira.example/rest",
        expectedUrl: "https://jira.example/rest/api/2/issue/DEMO-1/comment",
        header: ["api_key", "secret"],
      },
      {
        location: "query" as const,
        baseUrl: "https://jira.example/rest",
        expectedUrl:
          "https://jira.example/rest/api/2/issue/DEMO-1/comment?api_key=secret",
        header: null,
      },
      {
        location: "cookie" as const,
        baseUrl: "https://jira.example/rest",
        expectedUrl: "https://jira.example/rest/api/2/issue/DEMO-1/comment",
        header: ["Cookie", "api_key=secret"],
      },
      {
        location: "path" as const,
        baseUrl: "https://jira.example/bot{api_key}",
        expectedUrl:
          "https://jira.example/botsecret/api/2/issue/DEMO-1/comment",
        header: null,
      },
    ];
    for (const item of cases) {
      const value = definitions();
      value.source.base_url = item.baseUrl;
      value.auth = {
        api_keys: [
          {
            env: "API_KEY",
            location: item.location,
            name: "api_key",
            prefix: "",
          },
        ],
      };
      const calls = transport(async () => response(200, "ok"));
      const executor = new ConnectorExecutor(value, {
        transport: calls.transport,
      });
      await execute(
        executor,
        "jiradc_getComments",
        { issueIdOrKey: "DEMO-1" },
        { API_KEY: "secret" },
      );
      expect(calls.requests[0]?.url).toBe(item.expectedUrl);
      if (item.header) {
        expect(calls.requests[0]?.headers[item.header[0]]).toBe(item.header[1]);
      }
    }
  });

  it("supports basic and compound API-key auth", async () => {
    const basic = definitions();
    basic.auth = {
      basic: { username: "JIRA_USER", password: "JIRA_PASSWORD" },
    };
    const basicCalls = transport(async () => response(200, "ok"));
    const basicExecutor = new ConnectorExecutor(basic, {
      transport: basicCalls.transport,
    });
    await execute(
      basicExecutor,
      "jiradc_getComments",
      { issueIdOrKey: "DEMO-1" },
      { JIRA_USER: "alice", JIRA_PASSWORD: "secret" },
    );
    expect(basicCalls.requests[0]?.headers["Authorization"]).toBe(
      `Basic ${Buffer.from("alice:secret").toString("base64")}`,
    );
    await expect(
      execute(
        basicExecutor,
        "jiradc_getComments",
        { issueIdOrKey: "DEMO-1" },
        { JIRA_USER: "alice:admin", JIRA_PASSWORD: "secret" },
      ),
    ).rejects.toThrow("basic username is invalid");

    const compound = definitions();
    compound.auth = {
      api_keys: [
        { env: "FIRST", location: "header", name: "X-First", prefix: "" },
        { env: "SECOND", location: "query", name: "second", prefix: "" },
        { env: "THIRD", location: "cookie", name: "third", prefix: "" },
      ],
    };
    const compoundCalls = transport(async () => response(200, "ok"));
    const compoundExecutor = new ConnectorExecutor(compound, {
      transport: compoundCalls.transport,
    });
    await execute(
      compoundExecutor,
      "jiradc_getComments",
      { issueIdOrKey: "DEMO-1" },
      { FIRST: "one", SECOND: "two", THIRD: "three" },
    );
    expect(compoundCalls.requests[0]?.headers["X-First"]).toBe("one");
    expect(compoundCalls.requests[0]?.headers["Cookie"]).toBe("third=three");
    expect(
      new URL(compoundCalls.requests[0]?.url ?? "").searchParams.get("second"),
    ).toBe("two");
  });

  it("serializes query styles and header and cookie values", async () => {
    const value = customDefinitions({
      name: "query_tool",
      description: "",
      method: "GET",
      path: "/items",
      params: {
        path: [],
        query: ["tag", "ids", "filter"],
        header: ["X-Tags"],
        cookie: ["session"],
        body: [],
        wrapped: false,
        query_style: { ids: "comma", filter: "json" },
        form_style: {},
      },
      inputSchema: {
        type: "object",
        properties: {
          tag: { type: "string" },
          ids: { type: "array", items: { type: "string" } },
          filter: { type: "object" },
          "X-Tags": { type: "array", items: { type: "string" } },
          session: { type: "string" },
        },
      },
      body_encoding: "json",
    });
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(value, {
      transport: calls.transport,
    });
    await execute(
      executor,
      "query_tool",
      {
        tag: "a b",
        ids: ["one", "two"],
        filter: { a: 1 },
        "X-Tags": ["x", "y"],
        session: "a b",
      },
      {},
    );
    const url = new URL(calls.requests[0]?.url ?? "");
    expect(url.searchParams.get("tag")).toBe("a b");
    expect(url.searchParams.get("ids")).toBe("one,two");
    expect(url.searchParams.get("filter")).toBe('{"a":1}');
    expect(calls.requests[0]?.headers["X-Tags"]).toBe("x,y");
    expect(calls.requests[0]?.headers["Cookie"]).toBe("session=a%20b");
  });

  it("encodes form, multipart, text, binary, and unwrapped JSON bodies", async () => {
    const base = {
      description: "",
      method: "POST" as const,
      path: "/body",
      params: {
        path: [],
        query: [],
        header: [],
        cookie: [],
        body: [],
        wrapped: true,
        query_style: {},
        form_style: {},
      },
      inputSchema: {
        type: "object",
        properties: { body: {} },
        required: ["body"],
      },
    };
    const cases: Array<[Operation, unknown, string, (body: string) => void]> = [
      [
        { ...base, name: "form_tool", body_encoding: "form" },
        { a: "one", tags: ["x", "y"] },
        "application/x-www-form-urlencoded",
        (body) => {
          expect(body).toContain("a=one");
          expect(body).toContain("tags=x");
          expect(body).toContain("tags=y");
        },
      ],
      [
        { ...base, name: "multipart_tool", body_encoding: "multipart" },
        { a: "one", meta: { k: 2 } },
        "multipart/form-data; boundary=",
        (body) => {
          expect(body).toContain('name="a"');
          expect(body).toContain('{"k":2}');
        },
      ],
      [
        { ...base, name: "text_tool", body_encoding: "text" },
        "hello",
        "text/plain",
        (body) => expect(body).toBe("hello"),
      ],
      [
        { ...base, name: "binary_tool", body_encoding: "binary" },
        "aGVsbG8=",
        "application/octet-stream",
        (body) => expect(body).toBe("hello"),
      ],
    ];
    for (const [operation, body, contentType, check] of cases) {
      const calls = transport(async () => response(200, "ok"));
      const executor = new ConnectorExecutor(customDefinitions(operation), {
        transport: calls.transport,
      });
      await execute(executor, operation.name, { body }, {});
      const request = calls.requests[0];
      expect(request?.headers["Content-Type"]).toContain(contentType);
      check(new TextDecoder().decode(request?.body));
    }

    const unwrapped: Operation = {
      ...base,
      name: "json_tool",
      body_encoding: "json",
      params: { ...base.params, wrapped: false, body: ["value"] },
      inputSchema: {
        type: "object",
        properties: { value: { type: "string" } },
      },
    };
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(customDefinitions(unwrapped), {
      transport: calls.transport,
    });
    await execute(executor, "json_tool", { value: "v" }, {});
    expect(calls.requests[0]?.headers["Content-Type"]).toBe("application/json");
    expect(new TextDecoder().decode(calls.requests[0]?.body)).toBe(
      '{"value":"v"}',
    );
  });

  it("encodes CR/LF in multipart field names", async () => {
    const operation: Operation = {
      name: "multipart_tool",
      description: "",
      method: "POST",
      path: "/body",
      params: {
        path: [],
        query: [],
        header: [],
        cookie: [],
        body: [],
        wrapped: true,
        query_style: {},
        form_style: {},
      },
      inputSchema: {
        type: "object",
        properties: { body: { type: "object" } },
        required: ["body"],
      },
      body_encoding: "multipart",
    };
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(customDefinitions(operation), {
      transport: calls.transport,
    });
    await execute(
      executor,
      "multipart_tool",
      { body: { "safe\r\nX-Injected: 1": "value" } },
      {},
    );
    const body = new TextDecoder().decode(calls.requests[0]?.body);
    expect(body).toContain('name="safe%0D%0AX-Injected: 1"');
    expect(body).not.toContain("\r\nX-Injected");
  });

  it("follows same-origin redirects and rewrites POST 303 to GET", async () => {
    let calls = 0;
    const recorded = transport(async () => {
      calls += 1;
      if (calls === 1) {
        return response(303, "", { location: "/next" });
      }
      return response(200, "ok");
    });
    const value = definitions();
    const tool = structuredClone(value.tools[0] as Operation);
    tool.method = "POST";
    tool.description = "";
    tool.path = "/start";
    tool.params = {
      path: [],
      query: [],
      header: [],
      cookie: [],
      body: [],
      wrapped: false,
      query_style: {},
      form_style: {},
    };
    tool.inputSchema = { type: "object", properties: {} };
    value.tools = [tool];
    const executor = new ConnectorExecutor(value, {
      transport: recorded.transport,
    });
    await expect(
      execute(executor, tool.name, {}, { JIRA_TOKEN: "token" }),
    ).resolves.toBe("ok");
    expect(recorded.requests[1]?.method).toBe("GET");
    expect(recorded.requests[1]?.url).toBe("https://jira.example/next");
    expect(recorded.requests[1]?.body).toBeUndefined();
    expect(recorded.requests[1]?.headers["Content-Type"]).toBeUndefined();
  });

  it("rejects cross-origin and path-authenticated redirects", async () => {
    const crossOrigin = transport(async () =>
      response(302, "", { location: "https://evil.example/next" }),
    );
    const executor = new ConnectorExecutor(definitions(), {
      transport: crossOrigin.transport,
    });
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: "DEMO-1" }),
    ).rejects.toThrow("connector API returned HTTP 302");
    expect(crossOrigin.requests).toHaveLength(1);

    const pathAuth = definitions();
    pathAuth.source.base_url = "https://jira.example/bot{api_key}";
    pathAuth.auth = {
      api_keys: [
        { env: "API_KEY", location: "path", name: "api_key", prefix: "" },
      ],
    };
    const redirect = transport(async () =>
      response(302, "", { location: "/next" }),
    );
    const pathExecutor = new ConnectorExecutor(pathAuth, {
      transport: redirect.transport,
    });
    await expect(
      execute(
        pathExecutor,
        "jiradc_getComments",
        { issueIdOrKey: "DEMO-1" },
        { API_KEY: "secret" },
      ),
    ).rejects.toThrow("connector API returned HTTP 302");
    expect(redirect.requests).toHaveLength(1);
  });

  it("reapplies query credentials on redirects", async () => {
    const value = definitions();
    value.auth = {
      api_keys: [
        { env: "API_KEY", location: "query", name: "api_key", prefix: "" },
      ],
    };
    const calls = transport(async () => {
      if (calls.requests.length === 1) {
        return response(302, "", { location: "/next?x=1&api_key=old" });
      }
      return response(200, "ok");
    });
    const executor = new ConnectorExecutor(value, {
      transport: calls.transport,
    });
    await execute(
      executor,
      "jiradc_getComments",
      { issueIdOrKey: "DEMO-1" },
      { API_KEY: "secret" },
    );
    const redirected = new URL(calls.requests[1]?.url ?? "");
    expect(redirected.pathname).toBe("/next");
    expect(redirected.searchParams.get("x")).toBe("1");
    expect(redirected.searchParams.get("api_key")).toBe("secret");
  });

  it("retries only safe methods and honors Retry-After", async () => {
    const clock = {
      now: 0,
      sleeps: [] as number[],
    };
    const calls = transport(async () => {
      if (calls.requests.length === 1) {
        return response(503, "", { "retry-after": "10" });
      }
      return response(200, "ok");
    });
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
      now: () => clock.now,
      sleep: async (ms) => {
        clock.sleeps.push(ms);
        clock.now += ms;
      },
    });
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: "DEMO-1" }),
    ).resolves.toBe("ok");
    expect(clock.sleeps).toEqual([10_000]);
  });

  it("never retries writes and caps response bodies", async () => {
    const write = transport(async () => response(503, "unavailable"));
    const executor = new ConnectorExecutor(definitions(), {
      transport: write.transport,
    });
    await expect(
      execute(executor, "jiradc_addComment", {
        issueIdOrKey: "DEMO-1",
        body: "hello",
      }),
    ).rejects.toThrow("connector API returned HTTP 503");
    expect(write.requests).toHaveLength(1);

    const large = transport(async () => response(200, "x".repeat(200_001)));
    const largeExecutor = new ConnectorExecutor(definitions(), {
      transport: large.transport,
    });
    const result = await execute(largeExecutor, "jiradc_getComments", {
      issueIdOrKey: "DEMO-1",
    });
    expect(result).toHaveLength(200_000 + "\n...[truncated]".length);
    expect(result.endsWith("\n...[truncated]")).toBe(true);
  });

  it("retains a bounded provider error envelope for classification", async () => {
    const calls = transport(async () =>
      response(
        401,
        JSON.stringify({ reason: "Rejected credential platform-secret" }),
      ),
    );
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    let caught: unknown;
    try {
      await execute(
        executor,
        "jiradc_getComments",
        { issueIdOrKey: "DEMO-1" },
        { JIRA_TOKEN: "platform-secret" },
      );
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ConnectorHTTPError);
    const classified = await classifyToolAPIError(caught, "jira", [
      "platform-secret",
    ]);
    expect(classified?.message).toContain("HTTP 401 AUTH_FAILED");
    expect(classified?.message).toContain("<redacted>");
    expect(classified?.message).not.toContain("platform-secret");
  });

  it.each([
    [403, "AUTH_FAILED"],
    [404, "UNKNOWN"],
    [429, "RATE_LIMITED"],
    [503, "PROVIDER_UNAVAILABLE"],
  ])(
    "matches the shared error-message vocabulary for HTTP %i",
    async (status, classification) => {
      const classified = await classifyToolAPIError(
        new ConnectorHTTPError(status),
        "jira",
      );
      expect(classified?.toolApiError.classification).toBe(classification);
      expect(classified?.message).toBe(
        `jira API call failed: HTTP ${status} ${classification}`,
      );
    },
  );

  it("classifies a wrapped transport timeout as TIMEOUT", async () => {
    const wrapped = new TypeError("fetch failed");
    (wrapped as { cause?: unknown }).cause = {
      code: "UND_ERR_CONNECT_TIMEOUT",
    };
    const calls = transport(async () => {
      throw wrapped;
    });
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    let caught: unknown;
    try {
      await execute(executor, "jiradc_getComments", {
        issueIdOrKey: "DEMO-1",
      });
    } catch (error) {
      caught = error;
    }
    const classified = await classifyToolAPIError(caught, "jira");
    expect(classified?.toolApiError.classification).toBe("TIMEOUT");
    expect(classified?.message).not.toContain("fetch failed");
  });

  it("sanitizes transport and response-decoding failures", async () => {
    const requestFailure = transport(async () => {
      throw new Error("unsafe provider detail");
    });
    const requestExecutor = new ConnectorExecutor(definitions(), {
      transport: requestFailure.transport,
    });
    let requestError: unknown;
    try {
      await execute(requestExecutor, "jiradc_getComments", {
        issueIdOrKey: "DEMO-1",
      });
    } catch (error) {
      requestError = error;
    }
    const requestClassified = await classifyToolAPIError(requestError, "jira");
    expect(requestClassified?.toolApiError.classification).toBe(
      "CONNECTION_ERROR",
    );
    expect(requestClassified?.message).not.toContain("unsafe");

    const bodyFailure = transport(async () => ({
      status: 200,
      headers: new Headers(),
      body: {
        [Symbol.asyncIterator]() {
          return {
            next: async (): Promise<IteratorResult<Uint8Array>> => {
              throw new Error("unsafe body detail");
            },
          };
        },
      },
    }));
    const bodyExecutor = new ConnectorExecutor(definitions(), {
      transport: bodyFailure.transport,
    });
    let bodyError: unknown;
    try {
      await execute(bodyExecutor, "jiradc_getComments", {
        issueIdOrKey: "DEMO-1",
      });
    } catch (error) {
      bodyError = error;
    }
    const bodyClassified = await classifyToolAPIError(bodyError, "jira");
    expect(bodyClassified?.toolApiError.classification).toBe(
      "CONNECTION_ERROR",
    );
    expect(bodyClassified?.message).not.toContain("unsafe");
  });

  it("times out when the deadline expires while reading", async () => {
    const clock = { now: 0 };
    const calls = transport(async () => {
      clock.now = 2_000;
      return response(200, "late");
    });
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
      timeoutSeconds: 1,
      now: () => clock.now,
    });
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: "DEMO-1" }),
    ).rejects.toThrow("connector request timed out");
  });

  it("requires path credentials to be materialized", () => {
    const value = definitions();
    value.source.base_url = "https://jira.example/{tenant}/rest";
    value.auth = null;
    expect(() => new ConnectorExecutor(value)).toThrow(
      "connector path credential must be materialized",
    );
  });

  it("closes once and rejects execution after close", async () => {
    const calls = transport(async () => response(200, "ok"));
    const executor = new ConnectorExecutor(definitions(), {
      transport: calls.transport,
    });
    await executor.close();
    await executor.close();
    await expect(
      execute(executor, "jiradc_getComments", { issueIdOrKey: "DEMO-1" }),
    ).rejects.toThrow("connector executor is closed");
  });
});
