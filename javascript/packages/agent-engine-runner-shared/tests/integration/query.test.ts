/**
 * Integration tests for the AER's session query routes.
 *
 * Mirrors Python's tests/unit/test_query_routes.py, but exercises the real
 * Fastify routes in-process via inject() (matching executions.test.ts).
 * The plugin itself is stubbed — its real LangGraph-backed implementation
 * is covered by agent-engine-sdk-langgraph-ts's MongoDB-backed integration tests.
 */

import { describe, test, expect, afterEach, beforeEach } from "vitest";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type {
  SessionMessagesResponse,
  SessionsSummaryResponse,
} from "@mongodb-js/agent-engine-sdk";
import { AERServer } from "../../src/server/aer.js";
import type { AERQueryPlugin } from "../../src/server/query.js";
import { registerQueryPlugin, resetHooks } from "../../src/hooks.js";

// ---------------------------------------------------------------------------
// Stub plugin — records calls, returns canned responses
// ---------------------------------------------------------------------------

class StubPlugin implements AERQueryPlugin {
  summariesCalls: string[][] = [];
  messagesCalls: string[] = [];

  constructor(
    private readonly summaries: SessionsSummaryResponse = { sessions: [] },
    private readonly messages: SessionMessagesResponse = { messages: [] },
    private readonly failWith: Error | null = null,
  ) {}

  async getSummariesForSessions(
    sessionIds: string[],
  ): Promise<SessionsSummaryResponse> {
    this.summariesCalls.push(sessionIds);
    if (this.failWith) throw this.failWith;
    return this.summaries;
  }

  async getMessagesForSession(
    sessionId: string,
  ): Promise<SessionMessagesResponse> {
    this.messagesCalls.push(sessionId);
    if (this.failWith) throw this.failWith;
    return this.messages;
  }
}

// ---------------------------------------------------------------------------
// App factory — real AERServer route registration, stubbed runtime
// ---------------------------------------------------------------------------

function makeApp(): FastifyInstance {
  const server = Object.create(AERServer.prototype) as AERServer & {
    runtime: {
      getAgent: () => unknown;
      toolDefinitions: Record<string, never>;
    };
  };
  server.runtime = {
    getAgent: () => ({}),
    toolDefinitions: {},
  };
  const app = Fastify({ logger: false });
  server.registerRoutes(app);
  return app;
}

function sessionIdsQuery(ids: string[]): string {
  return ids.map((id) => `session_ids=${encodeURIComponent(id)}`).join("&");
}

describe("GET /query/sessions", () => {
  let app: FastifyInstance;

  beforeEach(() => {
    app = makeApp();
  });

  afterEach(() => {
    resetHooks();
  });

  test("returns 501 when no plugin is registered", async () => {
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s-1",
    });
    expect(resp.statusCode).toBe(501);
    expect(resp.json()).toEqual({
      detail: "session queries are not supported by this framework",
    });
  });

  test("returns empty list for no ids even without a plugin", async () => {
    const resp = await app.inject({ method: "GET", url: "/query/sessions" });
    expect(resp.statusCode).toBe(200);
    expect(resp.json()).toEqual({ sessions: [] });
  });

  test("does not invoke the plugin for empty ids", async () => {
    const plugin = new StubPlugin();
    registerQueryPlugin(plugin);
    const resp = await app.inject({ method: "GET", url: "/query/sessions" });
    expect(resp.statusCode).toBe(200);
    expect(plugin.summariesCalls).toEqual([]);
  });

  test("forwards repeated session_ids params to the plugin as a list", async () => {
    const plugin = new StubPlugin({
      sessions: [
        {
          session_id: "s-1",
          last_activity: "2026-06-09T00:00:00.000Z",
          created_at: "2026-06-08T00:00:00.000Z",
          message_count: 3,
          first_message_preview: "hello",
        },
      ],
    });
    registerQueryPlugin(plugin);
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s-1&session_ids=s-2",
    });
    expect(resp.statusCode).toBe(200);
    expect(plugin.summariesCalls).toEqual([["s-1", "s-2"]]);
    expect(resp.json()).toEqual({
      sessions: [
        {
          session_id: "s-1",
          last_activity: "2026-06-09T00:00:00.000Z",
          created_at: "2026-06-08T00:00:00.000Z",
          message_count: 3,
          first_message_preview: "hello",
        },
      ],
    });
  });

  test("forwards a single session_ids param as a one-element list", async () => {
    const plugin = new StubPlugin();
    registerQueryPlugin(plugin);
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=only-one",
    });
    expect(resp.statusCode).toBe(200);
    expect(plugin.summariesCalls).toEqual([["only-one"]]);
  });

  test("rejects an over-length session_ids list with 422", async () => {
    const plugin = new StubPlugin();
    registerQueryPlugin(plugin);
    const ids = Array.from({ length: 201 }, (_, i) => `s-${i}`);
    const resp = await app.inject({
      method: "GET",
      url: `/query/sessions?${sessionIdsQuery(ids)}`,
    });
    expect(resp.statusCode).toBe(422);
    expect(plugin.summariesCalls).toEqual([]);
  });

  test("accepts exactly the maximum number of session_ids", async () => {
    const plugin = new StubPlugin();
    registerQueryPlugin(plugin);
    const ids = Array.from({ length: 200 }, (_, i) => `s-${i}`);
    const resp = await app.inject({
      method: "GET",
      url: `/query/sessions?${sessionIdsQuery(ids)}`,
    });
    expect(resp.statusCode).toBe(200);
    expect(plugin.summariesCalls).toEqual([ids]);
  });

  test("surfaces a plugin exception as a generic 500", async () => {
    registerQueryPlugin(
      new StubPlugin({ sessions: [] }, { messages: [] }, new Error("boom")),
    );
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s-1",
    });
    expect(resp.statusCode).toBe(500);
    // Plugin error text (which can echo persisted user content) must not
    // reach the response body.
    expect(resp.json()).toEqual({ detail: "Internal Server Error" });
  });
});

describe("GET /query/sessions/:session_id/messages", () => {
  let app: FastifyInstance;

  beforeEach(() => {
    app = makeApp();
  });

  afterEach(() => {
    resetHooks();
  });

  test("returns 501 when no plugin is registered", async () => {
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions/s-1/messages",
    });
    expect(resp.statusCode).toBe(501);
    expect(resp.json()).toEqual({
      detail: "session queries are not supported by this framework",
    });
  });

  test("forwards the path param to the plugin", async () => {
    const plugin = new StubPlugin(
      { sessions: [] },
      {
        messages: [
          {
            id: "m-1",
            role: "user",
            content: "hi",
            timestamp: "2026-06-09T00:00:00.000Z",
            session_id: "sess-abc",
            name: "",
            tool_calls: null,
            tool_call_id: null,
            additional_kwargs: null,
          },
          {
            id: "a-1",
            role: "assistant",
            content: "",
            timestamp: "2026-06-09T00:00:01.000Z",
            session_id: "sess-abc",
            name: "",
            tool_calls: [
              {
                id: "call_1",
                name: "search",
                args: { q: "tokyo" },
              },
            ],
            tool_call_id: null,
            additional_kwargs: null,
          },
          {
            id: "t-1",
            role: "tool",
            content: "result",
            timestamp: "2026-06-09T00:00:02.000Z",
            session_id: "sess-abc",
            name: "search",
            tool_calls: null,
            tool_call_id: "call_1",
            additional_kwargs: null,
          },
        ],
      },
    );
    registerQueryPlugin(plugin);
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions/sess-abc/messages",
    });
    expect(resp.statusCode).toBe(200);
    expect(plugin.messagesCalls).toEqual(["sess-abc"]);
    expect(resp.json()).toEqual({
      messages: [
        {
          id: "m-1",
          role: "user",
          content: "hi",
          timestamp: "2026-06-09T00:00:00.000Z",
          session_id: "sess-abc",
          name: "",
          tool_calls: null,
          tool_call_id: null,
          additional_kwargs: null,
        },
        {
          id: "a-1",
          role: "assistant",
          content: "",
          timestamp: "2026-06-09T00:00:01.000Z",
          session_id: "sess-abc",
          name: "",
          tool_calls: [
            {
              id: "call_1",
              name: "search",
              args: { q: "tokyo" },
            },
          ],
          tool_call_id: null,
          additional_kwargs: null,
        },
        {
          id: "t-1",
          role: "tool",
          content: "result",
          timestamp: "2026-06-09T00:00:02.000Z",
          session_id: "sess-abc",
          name: "search",
          tool_calls: null,
          tool_call_id: "call_1",
          additional_kwargs: null,
        },
      ],
    });
  });

  test("returns an empty list when the plugin finds nothing", async () => {
    registerQueryPlugin(new StubPlugin());
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions/unknown-session/messages",
    });
    expect(resp.statusCode).toBe(200);
    expect(resp.json()).toEqual({ messages: [] });
  });

  test("surfaces a plugin exception as a generic 500", async () => {
    registerQueryPlugin(
      new StubPlugin({ sessions: [] }, { messages: [] }, new Error("boom")),
    );
    const resp = await app.inject({
      method: "GET",
      url: "/query/sessions/s-1/messages",
    });
    expect(resp.statusCode).toBe(500);
    expect(resp.json()).toEqual({ detail: "Internal Server Error" });
  });
});
