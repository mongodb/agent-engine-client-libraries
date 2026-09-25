import { afterEach, describe, expect, it, vi } from "vitest";

import { MemoryNotSupportedError } from "@mongodb-js/agent-engine-sdk-memory";

import {
  AppBoundCrudClient,
  AppBoundRuntime,
} from "../../src/memory_appbound.js";
import { runWithExecutionContext } from "../../src/context.js";

interface Recorded {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

function stubFetch(json: unknown, status = 200): { calls: Recorded[] } {
  const calls: Recorded[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    calls.push({
      url,
      method: init.method ?? "GET",
      headers: (init.headers as Record<string, string>) ?? {},
      body: typeof init.body === "string" ? JSON.parse(init.body) : null,
    });
    return new Response(JSON.stringify(json), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  return { calls };
}

function withContext<T>(
  args: {
    oeUrl: string;
    executionId?: string;
    userId?: string;
    sessionId?: string;
    workspaceId?: string;
  },
  fn: () => T,
): T {
  return runWithExecutionContext(
    {
      executionId: args.executionId ?? "exec-1",
      wrapper: null,
      oeUrl: args.oeUrl,
      userId: args.userId ?? null,
      sessionId: args.sessionId ?? null,
      workspaceId: args.workspaceId ?? null,
    },
    fn,
  );
}

describe("AppBoundRuntime", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("requestContext reflects the ambient user/session", () => {
    const rt = new AppBoundRuntime();
    const ctx = withContext(
      { oeUrl: "http://oe.test", userId: "u1", sessionId: "s1" },
      () => rt.requestContext(),
    );
    expect(ctx).toEqual({ userId: "u1", sessionId: "s1" });
  });

  it("throws MemoryNotSupportedError when no OE_URL is in context", async () => {
    const rt = new AppBoundRuntime();
    await expect(rt.searchSemantic({ query: "q" })).rejects.toBeInstanceOf(
      MemoryNotSupportedError,
    );
  });

  it("routes searchSemantic through the OE proxy with the execution-id header", async () => {
    const { calls } = stubFetch({ memories: [] });
    const rt = new AppBoundRuntime();
    await withContext(
      { oeUrl: "http://oe.test", executionId: "exec-42", userId: "u1" },
      () => rt.searchSemantic({ query: "q", userId: "u1" }),
    );
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/search");
    expect(calls[0].body).toMatchObject({
      type: "semantic",
      query: "q",
      user_id: "u1",
    });
    expect(calls[0].headers["X-Agent-Engine-Execution-Id"]).toBe("exec-42");
    // Legacy name is dual-sent during the rename transition.
    expect(calls[0].headers["X-Agentic-Execution-Id"]).toBe("exec-42");
  });

  it("sends workspace identity under both names when present", async () => {
    const { calls } = stubFetch({ memories: [] });
    const rt = new AppBoundRuntime();
    await withContext(
      { oeUrl: "http://oe.test", executionId: "exec-42", workspaceId: "ws-7" },
      () => rt.searchSemantic({ query: "q" }),
    );
    expect(calls[0].headers["X-Agent-Engine-Agent-Id"]).toBe("ws-7");
    expect(calls[0].headers["X-Agentic-Agent-Id"]).toBe("ws-7");
  });
});

const CONTEXT_OK = {
  formatted_context: "ctx",
  metadata: { token_count: 0, memory_counts: {}, timing: {} },
};

describe("AppBoundRuntime buildContext maxTokens", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards explicit maxTokens to the terminal OE body as max_tokens", async () => {
    const { calls } = stubFetch(CONTEXT_OK);
    const rt = new AppBoundRuntime();
    await withContext(
      { oeUrl: "http://oe.test", executionId: "exec-42", userId: "u1" },
      () =>
        rt.buildContext({
          query: "q",
          userId: "u1",
          topK: 10,
          maxTokens: 2048,
        }),
    );

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/context");
    expect(calls[0].method).toBe("POST");
    expect(calls[0].headers["X-Agent-Engine-Execution-Id"]).toBe("exec-42");
    // Legacy name is dual-sent during the rename transition.
    expect(calls[0].headers["X-Agentic-Execution-Id"]).toBe("exec-42");
    const body = calls[0].body as Record<string, unknown>;
    expect(body.max_tokens).toBe(2048);
    expect(body.top_k).toBe(10);
    expect(body.query).toBe("q");
    expect(body.user_id).toBe("u1");
  });

  it("omits max_tokens from the terminal OE body when maxTokens is absent", async () => {
    const { calls } = stubFetch(CONTEXT_OK);
    const rt = new AppBoundRuntime();
    await withContext(
      { oeUrl: "http://oe.test", executionId: "exec-42", userId: "u1" },
      () =>
        rt.buildContext({
          query: "q",
          userId: "u1",
          topK: 10,
        }),
    );

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/context");
    const body = calls[0].body as Record<string, unknown>;
    expect("max_tokens" in body).toBe(false);
    expect(body.top_k).toBe(10);
    expect(body.query).toBe("q");
    expect(body.user_id).toBe("u1");
  });
});

describe("AppBoundCrudClient", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("creates a semantic memory through the OE proxy CRUD route", async () => {
    const { calls } = stubFetch({
      id: "1",
      label: "fav",
      has_embedding: true,
      acknowledged: true,
    });
    const client = new AppBoundCrudClient();
    const res = await withContext(
      { oeUrl: "http://oe.test", userId: "u1" },
      () => client.createSemantic({ label: "fav", text: "teal", userId: "u1" }),
    );
    expect(res.id).toBe("1");
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/semantic");
    expect(calls[0].body).toMatchObject({
      org_id: "",
      project_id: "",
      label: "fav",
    });
  });

  it("createCustom posts to the OE types route with the execution header", async () => {
    const { calls } = stubFetch(
      { id: "m1", type: "tickets", tags: {}, has_embedding: true },
      201,
    );
    const client = new AppBoundCrudClient();
    await withContext({ oeUrl: "http://oe.test", executionId: "ex-1" }, () =>
      client.createCustom({ memoryType: "tickets", content: "c" }),
    );
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/types/tickets");
    expect(calls[0].headers["X-Agent-Engine-Execution-Id"]).toBe("ex-1");
    expect(calls[0].headers["X-Agentic-Execution-Id"]).toBe("ex-1");
    expect(calls[0].body).toEqual({ content: "c" });
  });

  it("retrieveCustom posts to the OE retrieve route", async () => {
    const { calls } = stubFetch({ results: [], count: 0 });
    const client = new AppBoundCrudClient();
    await withContext({ oeUrl: "http://oe.test" }, () =>
      client.retrieveCustom({ memoryType: "tickets", query: "q" }),
    );
    expect(calls[0].url).toBe(
      "http://oe.test/api/v1/memory/types/tickets/retrieve",
    );
    expect(calls[0].body).toEqual({ query: "q", top_k: 10 });
  });
});
