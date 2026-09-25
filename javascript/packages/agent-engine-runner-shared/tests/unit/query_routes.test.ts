/**
 * Tests for the AER's framework-agnostic session query routes (aer.ts).
 *
 * Mirrors runner-shared/tests/unit/test_query_routes.py.
 *
 * Ported cases:
 *   - 501 + QUERY_PLUGIN_UNSUPPORTED_DETAIL when no plugin registered
 *     (both /query/sessions with ids and /query/sessions/:id/messages).
 *   - empty session_ids short-circuits to { sessions: [] } without a plugin.
 *   - empty session_ids does NOT invoke a registered plugin.
 *   - session_ids forwarded verbatim to getSummariesForSessions.
 *   - :session_id path param forwarded to getMessagesForSession.
 *   - plugin exception surfaces as 500 (route does not swallow).
 *   - > MAX_QUERY_SESSION_IDS (200) rejected; plugin not reached.
 *   - exactly 200 accepted.
 *
 * Python→TS adaptation notes:
 *   - FastAPI TestClient → Fastify app + app.inject (pattern from tool.test.ts).
 *   - AERServer is built via Object.create(prototype) to skip the ctor
 *     (pattern from aer_policy_denied.test.ts); registerRoutes only touches
 *     runtime inside the /tools + /execute handlers, never the query routes.
 *   - The plugin is registered in the global hook registry
 *     (registerQueryPlugin) and cleared via resetHooks() in afterEach so tests
 *     don't leak the plugin across cases. Python passed the plugin through a
 *     per-request mocked runtime.get_query_plugin instead.
 *   - Over-length rejection: Python's FastAPI Query(max_length=200) yields 422;
 *     the TS route also returns 422 (asserted here).
 */

import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import { describe, test, expect, vi, afterEach } from "vitest";
import type {
  SessionMessagesResponse,
  SessionsSummaryResponse,
} from "@mongodb-js/agent-engine-sdk";
import {
  AERServer,
  registerQueryPlugin,
  getQueryPlugin,
  resetHooks,
  type AERQueryPlugin,
} from "../../src/index.js";

const QUERY_PLUGIN_UNSUPPORTED_DETAIL =
  "session queries are not supported by this framework";

interface StubPlugin extends AERQueryPlugin {
  getSummariesForSessions: ReturnType<typeof vi.fn>;
  getMessagesForSession: ReturnType<typeof vi.fn>;
}

function makeStubPlugin(opts: {
  summary?: SessionsSummaryResponse;
  messages?: SessionMessagesResponse;
  summaryError?: Error;
  messagesError?: Error;
}): StubPlugin {
  const summary = opts.summary ?? ({ sessions: [] } as SessionsSummaryResponse);
  const messages =
    opts.messages ?? ({ messages: [] } as SessionMessagesResponse);
  return {
    getSummariesForSessions: vi.fn(async (_ids: string[]) => {
      if (opts.summaryError) throw opts.summaryError;
      return summary;
    }),
    getMessagesForSession: vi.fn(async (_id: string) => {
      if (opts.messagesError) throw opts.messagesError;
      return messages;
    }),
  };
}

/**
 * Build a Fastify app with the AER routes wired up, skipping the AERServer
 * constructor (Object.create), and optionally register a query plugin.
 */
async function makeApp(
  plugin: AERQueryPlugin | null,
): Promise<FastifyInstance> {
  // Start from a clean registry so the no-plugin (501) cases are deterministic
  // even if another test file left a query plugin registered (global state).
  resetHooks();
  if (plugin !== null) registerQueryPlugin(plugin);
  const server = Object.create(AERServer.prototype) as AERServer;
  const app = Fastify();
  server.registerRoutes(app);
  await app.ready();
  return app;
}

afterEach(() => {
  // Query plugin lives in the global hook registry; clear it so it does not
  // leak into the next test (which asserts the no-plugin 501 path).
  resetHooks();
  expect(getQueryPlugin()).toBeNull();
});

// ---------------------------------------------------------------------------
// /query/sessions
// ---------------------------------------------------------------------------

describe("GET /query/sessions", () => {
  test("returns 501 with the unsupported detail when no plugin is registered", async () => {
    const app = await makeApp(null);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s1",
    });
    expect(res.statusCode).toBe(501);
    expect(res.json()).toEqual({ detail: QUERY_PLUGIN_UNSUPPORTED_DETAIL });
    await app.close();
  });

  test("empty session_ids short-circuits to an empty list without a plugin", async () => {
    const app = await makeApp(null);
    const res = await app.inject({ method: "GET", url: "/query/sessions" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ sessions: [] });
    await app.close();
  });

  test("empty session_ids does not invoke a registered plugin", async () => {
    // Guards against a refactor that moves the early return below the lookup.
    const plugin = makeStubPlugin({});
    const app = await makeApp(plugin);
    const res = await app.inject({ method: "GET", url: "/query/sessions" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ sessions: [] });
    expect(plugin.getSummariesForSessions).not.toHaveBeenCalled();
    await app.close();
  });

  test("forwards all repeated session_ids to getSummariesForSessions", async () => {
    const plugin = makeStubPlugin({
      summary: {
        sessions: [
          {
            session_id: "s1",
            last_activity: "2026-05-22T10:00:00+00:00",
            created_at: "2026-05-22T09:00:00+00:00",
            message_count: 3,
            first_message_preview: "hello",
          },
        ],
      } as SessionsSummaryResponse,
    });
    const app = await makeApp(plugin);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s1&session_ids=s2",
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.sessions[0].session_id).toBe("s1");
    expect(body.sessions[0].message_count).toBe(3);
    // Both repeated query params must reach the plugin, in order.
    expect(plugin.getSummariesForSessions).toHaveBeenCalledExactlyOnceWith([
      "s1",
      "s2",
    ]);
    await app.close();
  });

  test("plugin exception surfaces as 500 (route does not swallow)", async () => {
    const plugin = makeStubPlugin({
      summaryError: new Error("checkpoint store down"),
    });
    const app = await makeApp(plugin);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions?session_ids=s1",
    });
    expect(res.statusCode).toBe(500);
    // The plugin was actually invoked (route delegated, then caught).
    expect(plugin.getSummariesForSessions).toHaveBeenCalledExactlyOnceWith([
      "s1",
    ]);
    await app.close();
  });

  test("rejects an over-length session_ids list (> 200) without reaching the plugin", async () => {
    const plugin = makeStubPlugin({});
    const app = await makeApp(plugin);
    const tooMany = Array.from({ length: 201 }, (_, i) => `session_ids=s${i}`);
    const res = await app.inject({
      method: "GET",
      url: `/query/sessions?${tooMany.join("&")}`,
    });
    expect(res.statusCode).toBe(422);
    expect(plugin.getSummariesForSessions).not.toHaveBeenCalled();
    await app.close();
  });

  test("accepts exactly 200 session_ids (cap is inclusive)", async () => {
    const plugin = makeStubPlugin({});
    const app = await makeApp(plugin);
    const maxIds = Array.from({ length: 200 }, (_, i) => `session_ids=s${i}`);
    const res = await app.inject({
      method: "GET",
      url: `/query/sessions?${maxIds.join("&")}`,
    });
    expect(res.statusCode).toBe(200);
    expect(plugin.getSummariesForSessions).toHaveBeenCalledTimes(1);
    expect(plugin.getSummariesForSessions.mock.calls[0]?.[0]).toHaveLength(200);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// /query/sessions/:session_id/messages
// ---------------------------------------------------------------------------

describe("GET /query/sessions/:session_id/messages", () => {
  test("returns 501 when no plugin is registered", async () => {
    const app = await makeApp(null);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions/s1/messages",
    });
    expect(res.statusCode).toBe(501);
    expect(res.json()).toEqual({ detail: QUERY_PLUGIN_UNSUPPORTED_DETAIL });
    await app.close();
  });

  test("forwards the :session_id path param to getMessagesForSession", async () => {
    const plugin = makeStubPlugin({
      messages: {
        messages: [
          {
            id: "m1",
            role: "user",
            content: "hi",
            timestamp: "2026-05-22T09:00:00+00:00",
            session_id: "s1",
          },
          {
            id: "m2",
            role: "assistant",
            content: "hello",
            timestamp: "2026-05-22T09:00:05+00:00",
            session_id: "s1",
          },
        ],
      } as SessionMessagesResponse,
    });
    const app = await makeApp(plugin);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions/s1/messages",
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0].role).toBe("user");
    expect(body.messages[1].role).toBe("assistant");
    expect(plugin.getMessagesForSession).toHaveBeenCalledExactlyOnceWith("s1");
    await app.close();
  });

  test("plugin exception surfaces as 500", async () => {
    const plugin = makeStubPlugin({ messagesError: new Error("serde drift") });
    const app = await makeApp(plugin);
    const res = await app.inject({
      method: "GET",
      url: "/query/sessions/s1/messages",
    });
    expect(res.statusCode).toBe(500);
    expect(plugin.getMessagesForSession).toHaveBeenCalledExactlyOnceWith("s1");
    await app.close();
  });
});
