/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_route_selection.py
 *
 * Ported cases:
 * - projectId set -> project-scoped route + Bearer auth
 * - no projectId -> flat route + no auth
 * - serviceAccountToken without projectId -> still flat (projectId, not auth, discriminates)
 * - blank / whitespace projectId (kwarg + env) -> flat
 * - surrounding whitespace stripped ("  p1  " -> /projects/p1/)
 * - stale projectId 404 -> MemoryRouteNotFoundError hinting to unset projectId
 * - missing projectId against a gateway 404 -> MemoryRouteNotFoundError hinting
 *   to set projectId (not "unset")
 *
 * Adaptation notes:
 * - The TS port has no _profile/_base_path internals. Route selection is asserted
 *   from OBSERVED requests: calls[0].url and calls[0].headers.Authorization
 *   (capital A) via the fakeFetch helper.
 * - AGENTIC_MEMORY_* env is saved/restored around each test.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { MemoryRouteNotFoundError } from "../src/errors.js";
import { Memory } from "../src/memory.js";
import { fakeFetch } from "./helpers.js";

const ENV_KEYS = [
  "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN",
  "AGENTIC_MEMORY_API_KEY",
  "AGENTIC_MEMORY_BASE_URL",
  "AGENTIC_MEMORY_PROJECT_ID",
];

const GATEWAY_URL = "https://gw.example.com";
const OE_URL = "http://oe.local";
const API_KEY = "k";

describe("route selection from observed requests", () => {
  const saved: Record<string, string | undefined> = {};
  beforeEach(() => {
    for (const k of ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
  });
  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it("uses a project-scoped route and a bearer token when projectId is set", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: GATEWAY_URL,
      serviceAccountToken: API_KEY,
      projectId: "p123",
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/projects/p123/memory/");
    expect(calls[0].headers.Authorization).toBe(`Bearer ${API_KEY}`);
  });

  it("uses a flat route and no auth when projectId is absent", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ baseUrl: OE_URL, fetchImpl });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/memory/");
    expect(calls[0].url).not.toContain("/projects/");
    expect(calls[0].headers.Authorization).toBeUndefined();
  });

  it("stays flat with a token but no projectId (auth is not the discriminator)", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: OE_URL,
      serviceAccountToken: API_KEY,
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).not.toContain("/projects/");
    expect(calls[0].headers.Authorization).toBe(`Bearer ${API_KEY}`);
  });

  it.each(["", "   "])(
    "treats a blank projectId kwarg (%j) as unset (flat)",
    async (blank) => {
      const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
      const mem = new Memory({ baseUrl: OE_URL, projectId: blank, fetchImpl });
      await mem.searchSemantic({ query: "q", userId: "u" });
      expect(calls[0].url).not.toContain("/projects/");
    },
  );

  it("treats a blank AGENTIC_MEMORY_PROJECT_ID env as unset (flat)", async () => {
    process.env.AGENTIC_MEMORY_PROJECT_ID = "";
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ baseUrl: OE_URL, fetchImpl });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).not.toContain("/projects/");
  });

  it("strips surrounding whitespace from a projectId kwarg", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: GATEWAY_URL,
      serviceAccountToken: API_KEY,
      projectId: "  p1  ",
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/projects/p1/memory/");
  });

  it("strips surrounding whitespace from the projectId env var", async () => {
    process.env.AGENTIC_MEMORY_PROJECT_ID = "  p1  ";
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: GATEWAY_URL,
      serviceAccountToken: API_KEY,
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/projects/p1/memory/");
  });

  it("maps a stale-projectId 404 to an actionable 'unset projectId' error", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, json: { error: "nope" } }]);
    const mem = new Memory({
      baseUrl: OE_URL,
      serviceAccountToken: API_KEY,
      projectId: "p1",
      fetchImpl,
    });
    await expect(mem.buildContext({ query: "q", userId: "u" })).rejects.toThrow(
      /unset projectId/i,
    );
    await expect(
      new Memory({
        baseUrl: OE_URL,
        serviceAccountToken: API_KEY,
        projectId: "p1",
        fetchImpl: fakeFetch([{ status: 404, json: {} }]).fetchImpl,
      }).buildContext({ query: "q", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryRouteNotFoundError);
  });

  it("maps a missing-projectId 404 to a 'set projectId' error (not 'unset')", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, json: { error: "nope" } }]);
    const mem = new Memory({ baseUrl: GATEWAY_URL, fetchImpl });
    const err = await mem.buildContext({ query: "q", userId: "u" }).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(MemoryRouteNotFoundError);
    const msg = String((err as Error).message).toLowerCase();
    expect(msg).toContain("set projectid");
    expect(msg).not.toContain("unset");
  });
});
