/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_direct_crud.py
 *
 * Ported cases:
 * - all 9 CRUD forwards carry empty tenancy (writes: body org_id/project_id="";
 *   reads: query params org_id=/project_id=)
 * - createEpisodic summaryText null -> body summary_text === ""
 * - create on a transient 503 surfaces MemoryServerError after exactly 1 request
 *   (creates use the non-retrying client)
 * - createProcedural updateExisting truthy -> MemoryNotSupportedError before any
 *   request is issued
 * - default createProcedural does NOT put updateExisting/update_existing on the wire
 * - malformed 2xx body -> MemoryServerError (parity with Python)
 *
 * Adaptation notes:
 * - Uses the adapter({fetchImpl}) helper (from client_adapter.test.ts) with a
 *   branching FetchLike (fakeMemoryServer pattern from cross_session.test.ts).
 */

import { describe, expect, it } from "vitest";

import { MemoryNotSupportedError, MemoryServerError } from "../src/errors.js";
import { MemoryClientAdapter } from "../src/http/client_adapter.js";
import type { FetchLike } from "../src/transport.js";

interface RecordedRequest {
  method: string;
  path: string;
  params: URLSearchParams;
  body: Record<string, unknown>;
}

function adapter(fetchImpl: FetchLike): MemoryClientAdapter {
  return new MemoryClientAdapter({
    baseUrl: "http://oe.local",
    apiPrefix: "/api/v1/memory",
    routeStyle: "aliased",
    projectScoped: false,
    fetchImpl,
  });
}

/** A branching stub that records each request and returns a minimal valid body. */
function recordingServer(): {
  fetchImpl: FetchLike;
  requests: RecordedRequest[];
} {
  const requests: RecordedRequest[] = [];
  const json = (data: unknown, status = 200) =>
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  const fetchImpl: FetchLike = async (url, init) => {
    const parsed = new URL(url);
    const method = init.method ?? "GET";
    const body =
      typeof init.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : {};
    requests.push({
      method,
      path: parsed.pathname,
      params: parsed.searchParams,
      body,
    });
    const path = parsed.pathname;

    if (method === "POST" && path === "/api/v1/memory/semantic")
      return json(
        { id: "1", label: "l", has_embedding: false, acknowledged: true },
        201,
      );
    if (method === "GET" && path === "/api/v1/memory/semantic")
      return json({ entries: [] });
    if (method === "POST" && path === "/api/v1/memory/episodic")
      return json(
        { id: "1", title: "t", has_embedding: false, acknowledged: true },
        201,
      );
    if (method === "GET" && path === "/api/v1/memory/episodic")
      return json({ entries: [] });
    if (method === "POST" && path === "/api/v1/memory/taxonomic")
      return json(
        {
          id: "1",
          domain: "d",
          term: "x",
          has_embedding: false,
          acknowledged: true,
        },
        201,
      );
    if (method === "GET" && path === "/api/v1/memory/taxonomic")
      return json({ entries: [] });
    if (method === "GET" && path === "/api/v1/memory/taxonomic/domains")
      return json({ domains: ["d"] });
    if (method === "POST" && path === "/api/v1/memory/procedural")
      return json(
        { id: "1", procedure: "p", has_embedding: false, acknowledged: true },
        201,
      );
    if (method === "GET" && path === "/api/v1/memory/procedural")
      return json({ entries: [] });
    return json({}, 404);
  };
  return { fetchImpl, requests };
}

interface CrudCase {
  name: string;
  method: "POST" | "GET";
  path: string;
  run: (a: MemoryClientAdapter) => Promise<unknown>;
}

const CRUD_CASES: CrudCase[] = [
  {
    name: "createSemantic",
    method: "POST",
    path: "/api/v1/memory/semantic",
    run: (a) => a.createSemantic({ label: "l", text: "t", userId: "u" }),
  },
  {
    name: "getSemantic",
    method: "GET",
    path: "/api/v1/memory/semantic",
    run: (a) => a.getSemantic({ label: "l", userId: "u" }),
  },
  {
    name: "createEpisodic",
    method: "POST",
    path: "/api/v1/memory/episodic",
    run: (a) =>
      a.createEpisodic({
        title: "t",
        content: "c",
        userId: "u",
        sessionId: "s",
      }),
  },
  {
    name: "listEpisodic",
    method: "GET",
    path: "/api/v1/memory/episodic",
    run: (a) => a.listEpisodic({ userId: "u" }),
  },
  {
    name: "createTaxonomic",
    method: "POST",
    path: "/api/v1/memory/taxonomic",
    run: (a) =>
      a.createTaxonomic({
        domain: "d",
        term: "x",
        definition: "def",
        userId: "u",
      }),
  },
  {
    name: "getTaxonomic",
    method: "GET",
    path: "/api/v1/memory/taxonomic",
    run: (a) => a.getTaxonomic({ domain: "d", term: "x" }),
  },
  {
    name: "getDistinctDomains",
    method: "GET",
    path: "/api/v1/memory/taxonomic/domains",
    run: (a) => a.getDistinctDomains({}),
  },
  {
    name: "createProcedural",
    method: "POST",
    path: "/api/v1/memory/procedural",
    run: (a) =>
      a.createProcedural({
        procedure: "p",
        description: "d",
        content: "c",
        userId: "u",
      }),
  },
  {
    name: "getProcedural",
    method: "GET",
    path: "/api/v1/memory/procedural",
    run: (a) => a.getProcedural({ procedure: "p", userId: "u" }),
  },
];

describe("MemoryClientAdapter empty-tenancy CRUD forwards", () => {
  it.each(CRUD_CASES)(
    "$name forwards to $method $path with empty tenancy",
    async ({ method, path, run }) => {
      const { fetchImpl, requests } = recordingServer();
      await run(adapter(fetchImpl));
      const req = requests[requests.length - 1];
      expect(req.method).toBe(method);
      expect(req.path).toBe(path);
      if (method === "POST") {
        expect(req.body.org_id).toBe("");
        expect(req.body.project_id).toBe("");
      } else {
        expect(req.params.get("org_id")).toBe("");
        expect(req.params.get("project_id")).toBe("");
      }
    },
  );

  it("createEpisodic sends an empty summary when summaryText is null", async () => {
    const { fetchImpl, requests } = recordingServer();
    await adapter(fetchImpl).createEpisodic({
      title: "t",
      content: "c",
      userId: "u",
      sessionId: "s",
      summaryText: null,
    });
    expect(requests[requests.length - 1].body.summary_text).toBe("");
  });
});

describe("MemoryClientAdapter create semantics", () => {
  it("does not retry a create on a transient 5xx (exactly 1 request)", async () => {
    const requests: RecordedRequest[] = [];
    const fetchImpl: FetchLike = async (url, init) => {
      const parsed = new URL(url);
      requests.push({
        method: init.method ?? "GET",
        path: parsed.pathname,
        params: parsed.searchParams,
        body: {},
      });
      return new Response(JSON.stringify({ error: "unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    };
    await expect(
      adapter(fetchImpl).createSemantic({ label: "l", text: "t", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryServerError);
    expect(requests).toHaveLength(1);
  });

  it("rejects createProcedural updateExisting before issuing any request", async () => {
    const { fetchImpl, requests } = recordingServer();
    await expect(
      adapter(fetchImpl).createProcedural({
        procedure: "p",
        description: "d",
        content: "c",
        userId: "u",
        updateExisting: true,
      }),
    ).rejects.toThrow(/updateExisting/);
    await expect(
      adapter(fetchImpl).createProcedural({
        procedure: "p",
        description: "d",
        content: "c",
        userId: "u",
        updateExisting: true,
      }),
    ).rejects.toBeInstanceOf(MemoryNotSupportedError);
    expect(requests).toHaveLength(0);
  });

  it("does not put updateExisting/update_existing on the wire by default", async () => {
    const { fetchImpl, requests } = recordingServer();
    await adapter(fetchImpl).createProcedural({
      procedure: "p",
      description: "d",
      content: "c",
      userId: "u",
    });
    const body = requests[requests.length - 1].body;
    expect("update_existing" in body).toBe(false);
    expect("updateExisting" in body).toBe(false);
  });

  it("maps a malformed 2xx body to MemoryServerError (parity with Python)", async () => {
    // A success status with an unparseable body is a server contract violation,
    // not a transport failure — mapped to MemoryServerError like the Python SDK.
    const fetchImpl: FetchLike = async () =>
      new Response("not json", {
        status: 201,
        headers: { "Content-Type": "application/json" },
      });
    await expect(
      adapter(fetchImpl).createSemantic({ label: "l", text: "t", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryServerError);
  });
});
