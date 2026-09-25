/**
 * Tests for the MemoryClient-backed adapter: route convention, empty-tenancy
 * stamping, and the HTTP-error → typed-error translation the facade relies on.
 */

import { describe, expect, it } from "vitest";

import {
  MemoryAuthError,
  MemoryBadRequestError,
  MemoryNotProvisionedError,
  MemoryNotSupportedError,
  MemoryRouteNotFoundError,
  MemoryServerError,
} from "../src/errors.js";
import { MemoryClientAdapter } from "../src/http/client_adapter.js";
import { fakeFetch } from "./helpers.js";

function adapter(
  opts: Partial<ConstructorParameters<typeof MemoryClientAdapter>[0]> & {
    fetchImpl: ConstructorParameters<
      typeof MemoryClientAdapter
    >[0]["fetchImpl"];
  },
): MemoryClientAdapter {
  return new MemoryClientAdapter({
    baseUrl: "https://backend.test",
    apiPrefix: "/api/v1/memory",
    routeStyle: "aliased",
    projectScoped: false,
    ...opts,
  });
}

describe("MemoryClientAdapter routing", () => {
  it("uses the aliased /search route with a type body field", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    await adapter({ fetchImpl }).searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toBe("https://backend.test/api/v1/memory/search");
    expect((calls[0].body as Record<string, unknown>).type).toBe("semantic");
  });

  it("uses the aliased /turns route and forwards the idempotency key", async () => {
    const { fetchImpl, calls } = fakeFetch([
      {
        json: {
          id: "1",
          session_id: "s",
          turn_seq: 0,
          acknowledged: true,
          has_embedding: true,
        },
      },
    ]);
    const result = await adapter({ fetchImpl }).recordTurn({
      role: "user",
      content: "hi",
      sessionId: "s",
      userId: "u",
      idempotencyKey: "idem-1",
    });
    expect(calls[0].url).toBe("https://backend.test/api/v1/memory/turns");
    expect((calls[0].body as Record<string, unknown>).idempotency_key).toBe(
      "idem-1",
    );
    // has_embedding must survive schema parsing so TS callers can observe
    // whether the turn was embedded on write.
    expect(result.has_embedding).toBe(true);
  });

  it("stamps empty tenancy on creates for the backend to override", async () => {
    const { fetchImpl, calls } = fakeFetch([
      {
        status: 201,
        json: { id: "1", label: "l", has_embedding: true, acknowledged: true },
      },
    ]);
    await adapter({ fetchImpl }).createSemantic({
      label: "l",
      text: "t",
      userId: "u",
    });
    const body = calls[0].body as Record<string, unknown>;
    expect(body.org_id).toBe("");
    expect(body.project_id).toBe("");
    expect(body.user_id).toBe("u");
  });

  it("forwards caller metadata on createSemantic", async () => {
    const { fetchImpl, calls } = fakeFetch([
      {
        status: 201,
        json: { id: "1", label: "l", has_embedding: true, acknowledged: true },
      },
    ]);
    await adapter({ fetchImpl }).createSemantic({
      label: "l",
      text: "t",
      userId: "u",
      metadata: { channel: "web" },
    });
    const body = calls[0].body as Record<string, unknown>;
    expect(body.metadata).toEqual({ channel: "web" });
  });
});

describe("MemoryClientAdapter error translation", () => {
  it("maps 401 to MemoryAuthError", async () => {
    const { fetchImpl } = fakeFetch([{ status: 401, json: { error: "nope" } }]);
    await expect(
      adapter({ fetchImpl }).searchSemantic({ query: "q", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryAuthError);
  });

  it("maps a not-provisioned code to MemoryNotProvisionedError", async () => {
    const { fetchImpl } = fakeFetch([
      { status: 503, json: { code: "AGENT_NOT_DEPLOYED", error: "starting" } },
    ]);
    await expect(
      adapter({ fetchImpl }).searchSemantic({ query: "q", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryNotProvisionedError);
  });

  it("maps a core-loop 404 to MemoryRouteNotFoundError", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, text: "not found" }]);
    await expect(
      adapter({ fetchImpl }).buildContext({ query: "q", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryRouteNotFoundError);
  });

  it("maps 5xx to MemoryServerError (after retries)", async () => {
    // 500 is not retried; a single response suffices.
    const { fetchImpl } = fakeFetch([{ status: 500, text: "boom" }]);
    await expect(
      adapter({ fetchImpl }).searchSemantic({ query: "q", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryServerError);
  });
});

const CONTEXT_OK = {
  json: {
    formatted_context: "ctx",
    metadata: { token_count: 0, memory_counts: {}, timing: {} },
  },
};

const PROFILES: Array<
  [string, Partial<ConstructorParameters<typeof MemoryClientAdapter>[0]>]
> = [
  [
    "gateway",
    {
      apiPrefix: "/api/v1/projects/p1/memory",
      routeStyle: "aliased",
      projectScoped: true,
    },
  ],
  [
    "oe",
    {
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
    },
  ],
];

describe("MemoryClientAdapter buildContext maxTokens", () => {
  it.each(PROFILES)(
    "serializes max_tokens when explicit (%s)",
    async (_name, profile) => {
      const { fetchImpl, calls } = fakeFetch([CONTEXT_OK]);
      await adapter({ fetchImpl, ...profile }).buildContext({
        query: "q",
        userId: "u",
        topK: 10,
        maxTokens: 2048,
      });
      const body = calls[0].body as Record<string, unknown>;
      expect(body.max_tokens).toBe(2048);
      expect(body.top_k).toBe(10);
    },
  );

  it.each(PROFILES)(
    "omits max_tokens when absent (%s)",
    async (_name, profile) => {
      const { fetchImpl, calls } = fakeFetch([CONTEXT_OK]);
      await adapter({ fetchImpl, ...profile }).buildContext({
        query: "q",
        userId: "u",
        topK: 10,
      });
      const body = calls[0].body as Record<string, unknown>;
      expect("max_tokens" in body).toBe(false);
      expect(body.top_k).toBe(10);
    },
  );
});

describe("custom-type ops", () => {
  it("createCustom posts to the types route and parses the echo", async () => {
    const { fetchImpl, calls } = fakeFetch([
      {
        status: 201,
        json: {
          id: "m1",
          type: "tickets",
          tags: { open: true },
          has_embedding: true,
        },
      },
    ]);
    const adapter = new MemoryClientAdapter({
      baseUrl: "http://oe.test",
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
      fetchImpl,
    });
    const r = await adapter.createCustom({
      memoryType: "tickets",
      content: "c",
      tags: { open: true },
    });
    expect(calls[0].url).toBe("http://oe.test/api/v1/memory/types/tickets");
    expect(calls[0].body).toEqual({ content: "c", tags: { open: true } });
    expect(r.tags.open).toBe(true);
  });

  it("retrieveCustom maps a bare 404 to MemoryNotSupportedError", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, text: "Not Found" }]);
    const adapter = new MemoryClientAdapter({
      baseUrl: "http://oe.test",
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
      fetchImpl,
    });
    await expect(
      adapter.retrieveCustom({ memoryType: "tickets", query: "q" }),
    ).rejects.toThrow(MemoryNotSupportedError);
  });

  it("retrieveCustom passes through the server's unknown-type 404", async () => {
    const { fetchImpl } = fakeFetch([
      { status: 404, json: { error: "unknown custom memory type 'tickets'" } },
    ]);
    const adapter = new MemoryClientAdapter({
      baseUrl: "http://oe.test",
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
      fetchImpl,
    });
    await expect(
      adapter.retrieveCustom({ memoryType: "tickets", query: "q" }),
    ).rejects.toThrow(MemoryBadRequestError);
  });
});

describe("custom-type 405 handling", () => {
  // A gateway that does not know the route may reject the method rather than
  // the path, so 405 must reach the same unsupported-platform error as 404.
  it("maps a bare 405 to MemoryNotSupportedError", async () => {
    const { fetchImpl } = fakeFetch([
      { status: 405, text: "Method Not Allowed" },
    ]);
    const adapter = new MemoryClientAdapter({
      baseUrl: "http://oe.test",
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
      fetchImpl,
    });
    await expect(
      adapter.createCustom({ memoryType: "tickets", content: "c" }),
    ).rejects.toThrow(MemoryNotSupportedError);
  });

  it("passes through a 405 that names an unknown type", async () => {
    const { fetchImpl } = fakeFetch([
      { status: 405, json: { detail: "unknown custom memory type 'tickets'" } },
    ]);
    const adapter = new MemoryClientAdapter({
      baseUrl: "http://oe.test",
      apiPrefix: "/api/v1/memory",
      routeStyle: "aliased",
      projectScoped: false,
      fetchImpl,
    });
    await expect(
      adapter.retrieveCustom({ memoryType: "tickets", query: "q" }),
    ).rejects.toThrow(MemoryBadRequestError);
  });
});
