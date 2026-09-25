/** Mirrors sdk-core/tests/clients/test_memory.py. Mocks global fetch. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryClient } from "../../src/clients/memory.js";

type FetchMock = ReturnType<typeof vi.fn>;

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let fetchMock: FetchMock;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

function getCallUrl(call: unknown): string {
  const [url] = call as [string];
  return url;
}

function getCallBody(call: unknown): Record<string, unknown> {
  const [, init] = call as [string, RequestInit];
  return JSON.parse(init.body as string) as Record<string, unknown>;
}

describe("MemoryClient init", () => {
  it("strips trailing slash", () => {
    const client = new MemoryClient("http://localhost:8081/");
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );
    return client
      .writeTurn({
        sessionId: "s1",
        role: "user",
        orgId: "o",
        userId: "u",
        projectId: "p",
      })
      .then(() => {
        const url = getCallUrl(fetchMock.mock.calls[0]);
        expect(
          url.startsWith("http://localhost:8081/api/v1/memory/stm/turns"),
        ).toBe(true);
      });
  });

  it("works without trailing slash", () => {
    const client = new MemoryClient("http://localhost:8081");
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );
    return client
      .writeTurn({
        sessionId: "s1",
        role: "user",
        orgId: "o",
        userId: "u",
        projectId: "p",
      })
      .then(() => {
        const url = getCallUrl(fetchMock.mock.calls[0]);
        expect(
          url.startsWith("http://localhost:8081/api/v1/memory/stm/turns"),
        ).toBe(true);
      });
  });
});

describe("writeTurn", () => {
  it("success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "turn123",
        session_id: "sess1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.writeTurn({
      sessionId: "sess1",
      role: "user",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      content: "Hello",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/stm/turns");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["session_id"]).toBe("sess1");
    expect(body["role"]).toBe("user");
    expect(body["project_id"]).toBe("proj1");
    expect(result.id).toBe("turn123");
    expect(result.turn_seq).toBe(1);
  });

  it("sends a null session_id when blank (sessionless turn)", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "",
        turn_seq: 0,
        acknowledged: true,
      }),
    );
    const client = new MemoryClient("http://localhost:8081");
    await client.writeTurn({
      sessionId: "",
      role: "user",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      content: "Hello",
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["session_id"]).toBeNull();
  });
});

describe("SemanticMemory", () => {
  it("createSemantic success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "mem123",
        label: "test",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.createSemantic({
      label: "test",
      text: "Test memory",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/semantic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    // upsert defaults to false at the client layer (Python parity).
    expect(body["upsert"]).toBe(false);
    expect(result.id).toBe("mem123");
  });

  it("createSemantic forwards upsert=true", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "mem123",
        label: "test",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.createSemantic({
      label: "test",
      text: "Test memory",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      upsert: true,
    });

    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["upsert"]).toBe(true);
  });

  it("createSemantic includes metadata when provided", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "mem123",
        label: "test",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.createSemantic({
      label: "test",
      text: "Test memory",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      metadata: { channel: "web" },
    });

    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["metadata"]).toEqual({ channel: "web" });
  });

  it("fetchSemanticMemories uses correct endpoint", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ memories: [] }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.fetchSemanticMemories({
      query: "test query",
      orgId: "org1",
      projectId: "proj1",
      userId: "user1",
      visibility: "org",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/retrieval/semantic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["query"]).toBe("test query");
    expect(body["project_id"]).toBe("proj1");
    expect(body["user_id"]).toBe("user1");
    expect(body["visibility"]).toBe("org");
    expect(Array.isArray(result)).toBe(true);
  });

  it("getSemantic by label", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        entries: [{ id: "mem1", label: "test" }],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getSemantic({
      orgId: "org1",
      projectId: "proj1",
      label: "test",
      userId: "user1",
      visibility: "private",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url.startsWith("http://localhost:8081/api/v1/memory/semantic")).toBe(
      true,
    );
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("user_id=user1");
    expect(url).toContain("visibility=private");
    expect((result as Record<string, unknown>)["label"]).toBe("test");
  });

  it("getSemantic by id forwards project_id", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ id: "mem1", label: "test" }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.getSemantic({
      orgId: "org1",
      projectId: "proj1",
      id: "mem1",
      userId: "user1",
      visibility: "private",
    });

    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("/api/v1/memory/semantic/mem1");
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("user_id=user1");
    expect(url).toContain("visibility=private");
  });

  it("getSemantic returns null on 404", async () => {
    fetchMock.mockResolvedValueOnce(new Response("not found", { status: 404 }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getSemantic({
      orgId: "org1",
      projectId: "proj1",
      id: "missing-id",
    });

    expect(result).toBeNull();
  });
});

describe("EpisodicMemory", () => {
  it("createEpisodic success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "ep123",
        title: "Test episode",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.createEpisodic({
      title: "Test episode",
      content: "Content",
      summaryText: "Summary",
      orgId: "org1",
      userId: "user1",
      sessionId: "sess1",
      projectId: "proj1",
      snapshotRefId: "snap1",
      summaryType: "llm",
      sourceAgent: "agent",
      participants: ["Customer", "Alex"],
      tags: ["policy_created"],
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/episodic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["session_id"]).toBe("sess1");
    expect(body["project_id"]).toBe("proj1");
    expect(body["snapshot_ref_id"]).toBe("snap1");
    expect(body["summary_type"]).toBe("llm");
    expect(body["source_agent"]).toBe("agent");
    expect(body["participants"]).toEqual(["Customer", "Alex"]);
    expect(body["tags"]).toEqual(["policy_created"]);
    // metadata omitted -> absent from body (unchanged behavior for callers)
    expect(body["metadata"]).toBeUndefined();
    expect(result.id).toBe("ep123");
  });

  it("createEpisodic includes metadata when provided", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "ep123",
        title: "Test episode",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.createEpisodic({
      title: "Test episode",
      content: "Content",
      summaryText: "Summary",
      orgId: "org1",
      userId: "user1",
      sessionId: "sess1",
      projectId: "proj1",
      metadata: { channel: "web" },
    });

    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["metadata"]).toEqual({ channel: "web" });
  });

  it("createEpisodic requires session_id", async () => {
    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.createEpisodic({
        title: "T",
        content: "C",
        summaryText: "S",
        orgId: "org1",
        userId: "user1",
        sessionId: "",
        projectId: "proj1",
      }),
    ).rejects.toThrow(/session_id is required/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetchEpisodicMemories uses correct endpoint", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ memories: [] }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.fetchEpisodicMemories({
      query: "test query",
      orgId: "org1",
      projectId: "proj1",
      visibility: "private",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/retrieval/episodic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(body["visibility"]).toBe("private");
    expect(Array.isArray(result)).toBe(true);
  });

  it("listEpisodic returns entries", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        entries: [{ id: "ep1" }, { id: "ep2" }],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.listEpisodic({
      orgId: "org1",
      projectId: "proj1",
      userId: "user1",
      visibility: "private",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url.startsWith("http://localhost:8081/api/v1/memory/episodic")).toBe(
      true,
    );
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("visibility=private");
    expect(result).toHaveLength(2);
  });
});

describe("TaxonomicMemory", () => {
  it("createTaxonomic success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "tax123",
        domain: "insurance",
        term: "premium",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.createTaxonomic({
      domain: "insurance",
      term: "premium",
      definition: "Amount paid",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/taxonomic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(result.id).toBe("tax123");
  });

  it("fetchTaxonomicMemories uses taxonomic endpoint, not procedural", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ memories: [] }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.fetchTaxonomicMemories({
      query: "test query",
      orgId: "org1",
      projectId: "proj1",
      domain: "insurance",
      visibility: "org",
      // Accepted for API-surface parity but must NOT reach the wire.
      userId: "user1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/retrieval/taxonomic");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["query"]).toBe("test query");
    expect(body["domain"]).toBe("insurance");
    expect(body["project_id"]).toBe("proj1");
    expect(body["visibility"]).toBe("org");
    // Parity: taxonomic search is org-scoped; user_id is a dead param that the
    // client drops before the wire (matches Python's **kwargs absorption).
    expect("user_id" in body).toBe(false);
    expect(Array.isArray(result)).toBe(true);
  });

  it("getTaxonomic by term", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        entries: [
          { term: "premium", definition: "Amount paid" },
          { term: "deductible", definition: "Out of pocket" },
        ],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getTaxonomic({
      orgId: "org1",
      projectId: "proj1",
      domain: "insurance",
      term: "premium",
      userId: "user1",
      visibility: "org",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(
      url.startsWith("http://localhost:8081/api/v1/memory/taxonomic"),
    ).toBe(true);
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("user_id=user1");
    expect(url).toContain("visibility=org");
    expect((result as Record<string, unknown>)["term"]).toBe("premium");
  });

  it("getDistinctDomains", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        domains: ["insurance", "banking"],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getDistinctDomains({
      orgId: "org1",
      projectId: "proj1",
      visibility: "org",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(
      url.startsWith("http://localhost:8081/api/v1/memory/taxonomic/domains"),
    ).toBe(true);
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("visibility=org");
    expect(result).toEqual(["insurance", "banking"]);
  });
});

describe("ContextBuilding", () => {
  it("buildContext success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        formatted_context: "Context text",
        metadata: {
          token_count: 10,
          memory_counts: { semantic: 5 },
          timing: { total: 0.1 },
        },
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.buildContext({
      query: "test query",
      sessionId: "sess1",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/retrieval/context");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(result.formatted_context).toBe("Context text");
  });

  it("buildContext sends a null session_id when blank (sessionless context)", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        formatted_context: "",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      }),
    );
    const client = new MemoryClient("http://localhost:8081");
    await client.buildContext({
      query: "test query",
      sessionId: "",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["session_id"]).toBeNull();
  });

  it("buildContext serializes max_tokens when explicit", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        formatted_context: "Context text",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      }),
    );
    const client = new MemoryClient("http://localhost:8081");
    await client.buildContext({
      query: "test query",
      sessionId: "sess1",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      maxTokens: 2048,
    });
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["max_tokens"]).toBe(2048);
    expect(body["top_k"]).toBe(50);
  });

  it("buildContext omits max_tokens when absent", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        formatted_context: "Context text",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      }),
    );
    const client = new MemoryClient("http://localhost:8081");
    await client.buildContext({
      query: "test query",
      sessionId: "sess1",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
    });
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect("max_tokens" in body).toBe(false);
    expect(body["top_k"]).toBe(50);
  });

  it.each([
    0,
    -1,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    1.5,
  ])(
    "buildContext rejects invalid maxTokens (%s) before fetch",
    async (maxTokens) => {
      const client = new MemoryClient("http://localhost:8081");
      let err: unknown;
      try {
        await client.buildContext({
          query: "test query",
          sessionId: "sess1",
          orgId: "org1",
          userId: "user1",
          projectId: "proj1",
          maxTokens,
        });
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RangeError);
      expect((err as Error).message).toBe(
        "maxTokens must be a positive integer",
      );
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );
});

describe("ProceduralMemory", () => {
  it("createProcedural success", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "proc123",
        procedure: "deploy-app",
        has_embedding: true,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.createProcedural({
      procedure: "deploy-app",
      description: "Deploy the application",
      content: "Step 1: build. Step 2: push.",
      orgId: "org1",
      userId: "user1",
      projectId: "proj1",
      tags: ["deploy", "ci"],
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/procedural");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["procedure"]).toBe("deploy-app");
    expect(body["project_id"]).toBe("proj1");
    expect(body["tags"]).toEqual(["deploy", "ci"]);
    expect(result.id).toBe("proc123");
  });

  it("getProcedural by id — 200", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ id: "proc1", procedure: "deploy-app" }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "proc1",
      userId: "user1",
      visibility: "private",
    });

    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("/api/v1/memory/procedural/proc1");
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("user_id=user1");
    expect(url).toContain("visibility=private");
    expect((result as Record<string, unknown>)["id"]).toBe("proc1");
  });

  it("getProcedural by id — 404 returns null", async () => {
    fetchMock.mockResolvedValueOnce(new Response("not found", { status: 404 }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "missing",
    });

    expect(result).toBeNull();
  });

  it("getProcedural by procedure name", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        entries: [{ id: "proc1", procedure: "deploy-app" }],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getProcedural({
      orgId: "org1",
      projectId: "proj1",
      procedure: "deploy-app",
      userId: "user1",
      visibility: "private",
    });

    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("project_id=proj1");
    expect(url).toContain("procedure=deploy-app");
    expect(url).toContain("user_id=user1");
    expect(url).toContain("visibility=private");
    expect((result as Record<string, unknown>)["id"]).toBe("proc1");
  });

  it("getProcedural returns null when procedure name not in entries", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ entries: [{ id: "p1", procedure: "other" }] }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.getProcedural({
      orgId: "org1",
      projectId: "proj1",
      procedure: "deploy-app",
    });

    expect(result).toBeNull();
  });

  it("updateProcedural by id — single PATCH call, project_id in body", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ id: "proc1", content: "updated" }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.updateProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "proc1",
      content: "updated",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("/api/v1/memory/procedural/proc1");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(body["content"]).toBe("updated");
  });

  it("updateProcedural by procedure name — GET then PATCH, project_id on both", async () => {
    const existing = { id: "proc1", procedure: "deploy-app" };
    fetchMock
      .mockResolvedValueOnce(mockJsonResponse({ entries: [existing] }))
      .mockResolvedValueOnce(mockJsonResponse({ id: "proc1", content: "new" }));

    const client = new MemoryClient("http://localhost:8081");
    await client.updateProcedural({
      orgId: "org1",
      projectId: "proj1",
      procedure: "deploy-app",
      content: "new",
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const getUrl = getCallUrl(fetchMock.mock.calls[0]);
    expect(getUrl).toContain("project_id=proj1");
    const patchBody = getCallBody(fetchMock.mock.calls[1]);
    expect(patchBody["project_id"]).toBe("proj1");
    expect(patchBody["content"]).toBe("new");
  });

  it("updateProcedural throws when procedure not found", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ entries: [] }));

    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.updateProcedural({
        orgId: "org1",
        projectId: "proj1",
        procedure: "missing-proc",
      }),
    ).rejects.toThrow(/Procedural memory not found/);
  });

  it("updateProcedural throws when neither id nor procedure supplied", async () => {
    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.updateProcedural({ orgId: "org1", projectId: "proj1" }),
    ).rejects.toThrow(/id or procedure is required/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("deleteProcedural by id — single DELETE call", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ deleted_count: 1, acknowledged: true }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.deleteProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "proc1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("/api/v1/memory/procedural/proc1");
    expect(url).toContain("project_id=proj1");
    expect(result.deleted_count).toBe(1);
  });

  it("deleteProcedural by procedure name — GET then DELETE, project_id on both", async () => {
    const existing = { id: "proc1", procedure: "deploy-app" };
    fetchMock
      .mockResolvedValueOnce(mockJsonResponse({ entries: [existing] }))
      .mockResolvedValueOnce(
        mockJsonResponse({ deleted_count: 1, acknowledged: true }),
      );

    const client = new MemoryClient("http://localhost:8081");
    await client.deleteProcedural({
      orgId: "org1",
      projectId: "proj1",
      procedure: "deploy-app",
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const getUrl = getCallUrl(fetchMock.mock.calls[0]);
    expect(getUrl).toContain("project_id=proj1");
    const deleteUrl = getCallUrl(fetchMock.mock.calls[1]);
    expect(deleteUrl).toContain("/api/v1/memory/procedural/proc1");
    expect(deleteUrl).toContain("project_id=proj1");
  });

  it("deleteProcedural throws when procedure not found", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ entries: [] }));

    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.deleteProcedural({
        orgId: "org1",
        projectId: "proj1",
        procedure: "missing-proc",
      }),
    ).rejects.toThrow(/Procedural memory not found/);
  });

  it("deleteProcedural throws when neither id nor procedure supplied", async () => {
    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.deleteProcedural({ orgId: "org1", projectId: "proj1" }),
    ).rejects.toThrow(/id or procedure is required/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("discoverProcedures returns projected results", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        memories: [
          {
            content: "Step 1: build",
            similarity_score: 0.92,
            metadata: { procedure: "deploy-app", author: "ci-bot" },
          },
        ],
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.discoverProcedures({
      query: "how to deploy",
      orgId: "org1",
      projectId: "proj1",
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(result).toHaveLength(1);
    expect(result[0]?.["procedure"]).toBe("deploy-app");
    expect(result[0]?.["score"]).toBe(0.92);
    expect(result[0]?.["author"]).toBe("ci-bot");
  });

  it("fetchProceduralMemories uses retrieval endpoint", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ memories: [] }));

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.fetchProceduralMemories({
      query: "deploy steps",
      orgId: "org1",
      projectId: "proj1",
    });

    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe(
      "http://localhost:8081/api/v1/memory/retrieval/procedural",
    );
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body["project_id"]).toBe("proj1");
    expect(Array.isArray(result)).toBe(true);
  });
});

describe("updateSemantic", () => {
  it("GET then PATCH — two calls, project_id on both", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({ entries: [{ id: "sem1", label: "api-docs" }] }),
      )
      .mockResolvedValueOnce(mockJsonResponse({ id: "sem1", text: "updated" }));

    const client = new MemoryClient("http://localhost:8081");
    await client.updateSemantic({
      orgId: "org1",
      projectId: "proj1",
      label: "api-docs",
      text: "updated",
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const getUrl = getCallUrl(fetchMock.mock.calls[0]);
    expect(getUrl).toContain("project_id=proj1");
    const patchUrl = getCallUrl(fetchMock.mock.calls[1]);
    expect(patchUrl).toContain("/api/v1/memory/semantic/sem1");
    const patchBody = getCallBody(fetchMock.mock.calls[1]);
    expect(patchBody["project_id"]).toBe("proj1");
    expect(patchBody["text"]).toBe("updated");
  });

  it("throws when label not found", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ entries: [] }));

    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.updateSemantic({
        orgId: "org1",
        projectId: "proj1",
        label: "missing-label",
        text: "x",
      }),
    ).rejects.toThrow(/Semantic memory not found/);
  });

  it("throws when server entry has no id field", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ entries: [{ label: "api-docs" }] }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.updateSemantic({
        orgId: "org1",
        projectId: "proj1",
        label: "api-docs",
      }),
    ).rejects.toThrow(/no id field/);
  });
});

describe("memory id URL encoding", () => {
  // Regression tests for authenticated request forgery via unencoded memory
  // ids: '?' must not inject query params, '#' must not truncate the tenant
  // scoping query, and '../' must not reroute the request outside the memory
  // route via URL dot-segment normalization.

  it("getSemantic encodes '?' and '#' in the id path segment", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ id: "mem1" }));

    const client = new MemoryClient("http://localhost:8081");
    await client.getSemantic({
      orgId: "org1",
      projectId: "proj1",
      id: "fake-id?org_id=attacker-org#frag",
    });

    const url = new URL(getCallUrl(fetchMock.mock.calls[0]));
    expect(url.pathname).toBe(
      "/api/v1/memory/semantic/fake-id%3Forg_id%3Dattacker-org%23frag",
    );
    // Tenant scoping survives: the injected '?' created no second query
    // string and '#' swallowed nothing into a fragment.
    expect(url.searchParams.get("org_id")).toBe("org1");
    expect(url.searchParams.get("project_id")).toBe("proj1");
  });

  it("getProcedural encodes '?' and '#' in the id path segment", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ id: "proc1" }));

    const client = new MemoryClient("http://localhost:8081");
    await client.getProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "fake-id?org_id=attacker-org#frag",
    });

    const url = new URL(getCallUrl(fetchMock.mock.calls[0]));
    expect(url.pathname).toBe(
      "/api/v1/memory/procedural/fake-id%3Forg_id%3Dattacker-org%23frag",
    );
    expect(url.searchParams.get("org_id")).toBe("org1");
    expect(url.searchParams.get("project_id")).toBe("proj1");
  });

  it("updateProcedural keeps '../' traversal inside the procedural route", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ id: "proc1", content: "updated" }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.updateProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "../../admin/dangerous-endpoint",
      content: "updated",
    });

    const url = new URL(getCallUrl(fetchMock.mock.calls[0]));
    expect(url.pathname).toBe(
      "/api/v1/memory/procedural/..%2F..%2Fadmin%2Fdangerous-endpoint",
    );
  });

  it("deleteProcedural keeps '../' traversal inside the procedural route", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ deleted_count: 1, acknowledged: true }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.deleteProcedural({
      orgId: "org1",
      projectId: "proj1",
      id: "../../admin/dangerous-endpoint",
    });

    const url = new URL(getCallUrl(fetchMock.mock.calls[0]));
    expect(url.pathname).toBe(
      "/api/v1/memory/procedural/..%2F..%2Fadmin%2Fdangerous-endpoint",
    );
  });

  it("updateSemantic encodes a traversal id read back from the server", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({
          entries: [{ id: "../../evil", label: "api-docs" }],
        }),
      )
      .mockResolvedValueOnce(mockJsonResponse({ id: "../../evil" }));

    const client = new MemoryClient("http://localhost:8081");
    await client.updateSemantic({
      orgId: "org1",
      projectId: "proj1",
      label: "api-docs",
      text: "updated",
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const patchUrl = new URL(getCallUrl(fetchMock.mock.calls[1]));
    expect(patchUrl.pathname).toBe("/api/v1/memory/semantic/..%2F..%2Fevil");
  });

  it.each([".", ".."])(
    "rejects a bare dot-segment id (%j) before any request",
    async (id) => {
      const client = new MemoryClient("http://localhost:8081");
      await expect(
        client.getSemantic({ orgId: "org1", projectId: "proj1", id }),
      ).rejects.toThrow(/bare dot segment/);
      await expect(
        client.getProcedural({ orgId: "org1", projectId: "proj1", id }),
      ).rejects.toThrow(/bare dot segment/);
      await expect(
        client.deleteProcedural({ orgId: "org1", projectId: "proj1", id }),
      ).rejects.toThrow(/bare dot segment/);
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("rejects a bare dot-segment id read back from the server", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ entries: [{ id: "..", label: "api-docs" }] }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await expect(
      client.updateSemantic({
        orgId: "org1",
        projectId: "proj1",
        label: "api-docs",
        text: "updated",
      }),
    ).rejects.toThrow(/bare dot segment/);
    // Only the lookup GET went out; the PATCH was never fired.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("MemoryClient AbortSignal propagation", () => {
  it("passes caller signal to fetch via AbortSignal.any composition", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );

    const controller = new AbortController();
    const client = new MemoryClient("http://localhost:8081");
    await client.writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
      signal: controller.signal,
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    // AbortSignal.any returns a composite; it is a distinct object from either input
    expect(init.signal).toBeDefined();
    expect(init.signal).not.toBe(controller.signal);
  });

  it("rejects immediately when a pre-aborted signal is passed", async () => {
    const controller = new AbortController();
    controller.abort();

    const client = new MemoryClient("http://localhost:8081");
    // fetch is not called when the signal is already aborted
    fetchMock.mockRejectedValueOnce(
      new DOMException("This operation was aborted", "AbortError"),
    );

    await expect(
      client.writeTurn({
        sessionId: "s1",
        role: "user",
        orgId: "o",
        userId: "u",
        projectId: "p",
        signal: controller.signal,
      }),
    ).rejects.toThrow(/aborted/i);
  });
});

describe("identity headers", () => {
  function getCallHeaders(call: unknown): Record<string, string> {
    const [, init] = call as [string, RequestInit];
    return init.headers as Record<string, string>;
  }

  it("staticHeaders are sent on every POST request", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081", 30, {
      "X-Agent-Engine-Agent-Id": "agent-123",
    });
    await client.writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });

    const headers = getCallHeaders(fetchMock.mock.calls[0]);
    expect(headers["X-Agent-Engine-Agent-Id"]).toBe("agent-123");
  });

  it("staticHeaders are sent on GET requests via request()", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ entries: [{ id: "m1", label: "test" }] }),
    );

    const client = new MemoryClient("http://localhost:8081", 30, {
      "X-Agent-Engine-Agent-Id": "agent-456",
    });
    await client.getSemantic({ orgId: "o", projectId: "p", label: "test" });

    const headers = getCallHeaders(fetchMock.mock.calls[0]);
    expect(headers["X-Agent-Engine-Agent-Id"]).toBe("agent-456");
  });

  it("staticHeaders are sent on direct fetch paths (getSemantic by id)", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ id: "m1", label: "test" }),
    );

    const client = new MemoryClient("http://localhost:8081", 30, {
      "X-Agent-Engine-Agent-Id": "agent-789",
    });
    await client.getSemantic({ orgId: "o", projectId: "p", id: "m1" });

    const headers = getCallHeaders(fetchMock.mock.calls[0]);
    expect(headers["X-Agent-Engine-Agent-Id"]).toBe("agent-789");
  });

  it("dynamicHeaders are merged over staticHeaders on every request", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );

    let execId = "exec-aaa";
    const client = new MemoryClient(
      "http://localhost:8081",
      30,
      { "X-Agent-Engine-Agent-Id": "agent-static" },
      "/api/v1/memory",
      () => ({ "X-Agent-Engine-Execution-Id": execId }),
    );
    await client.writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });

    const headers = getCallHeaders(fetchMock.mock.calls[0]);
    expect(headers["X-Agent-Engine-Agent-Id"]).toBe("agent-static");
    expect(headers["X-Agent-Engine-Execution-Id"]).toBe("exec-aaa");

    // dynamicHeaders is called per request — a new value appears on the next call
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t2",
        session_id: "s1",
        turn_seq: 2,
        acknowledged: true,
      }),
    );
    execId = "exec-bbb";
    await client.writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });

    const headers2 = getCallHeaders(fetchMock.mock.calls[1]);
    expect(headers2["X-Agent-Engine-Execution-Id"]).toBe("exec-bbb");
  });

  it("dynamicHeaders returning empty object adds no extra headers", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 1,
        acknowledged: true,
      }),
    );

    const client = new MemoryClient(
      "http://localhost:8081",
      30,
      { "X-Agent-Engine-Agent-Id": "agent-static" },
      "/api/v1/memory",
      () => ({}),
    );
    await client.writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });

    const headers = getCallHeaders(fetchMock.mock.calls[0]);
    expect(headers["X-Agent-Engine-Agent-Id"]).toBe("agent-static");
    expect(headers["X-Agent-Engine-Execution-Id"]).toBeUndefined();
  });
});

describe("createCustom", () => {
  it("POSTs to /types/{name} with a snake_case body and no identity fields", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "m1",
        type: "tickets",
        tags: { queue: "billing", priority: 3 },
        has_embedding: true,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    const result = await client.createCustom({
      memoryType: "tickets",
      content: "c",
      tags: { queue: "billing", priority: 3 },
      contextualMetadata: { k: "v" },
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe("http://localhost:8081/api/v1/memory/types/tickets");
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body).toEqual({
      content: "c",
      tags: { queue: "billing", priority: 3 },
      contextual_metadata: { k: "v" },
    });
    expect(result.tags.priority).toBe(3);
  });

  it("omits absent optional fields", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "m1",
        type: "tickets",
        tags: {},
        has_embedding: false,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.createCustom({ memoryType: "tickets", content: "c" });

    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body).toEqual({ content: "c" });
  });

  it("URL-encodes the type name", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "m1",
        type: "a b",
        tags: {},
        has_embedding: false,
      }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.createCustom({ memoryType: "a b", content: "c" });

    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toContain("/types/a%20b");
  });
});

describe("retrieveCustom", () => {
  it("POSTs query/tags/top_k to /types/{name}/retrieve", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ results: [], count: 0 }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.retrieveCustom({
      memoryType: "tickets",
      query: "q",
      tags: { open: true },
      topK: 5,
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const url = getCallUrl(fetchMock.mock.calls[0]);
    expect(url).toBe(
      "http://localhost:8081/api/v1/memory/types/tickets/retrieve",
    );
    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body).toEqual({ query: "q", tags: { open: true }, top_k: 5 });
  });

  it("defaults top_k to 10", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ results: [], count: 0 }),
    );

    const client = new MemoryClient("http://localhost:8081");
    await client.retrieveCustom({ memoryType: "tickets", query: "q" });

    const body = getCallBody(fetchMock.mock.calls[0]);
    expect(body).toEqual({ query: "q", top_k: 10 });
  });
});

describe("routeStyle: aliased (Gateway convention)", () => {
  function aliasedClient(): MemoryClient {
    return new MemoryClient(
      "http://gw.test",
      30,
      {},
      "/api/v1/projects/p1/memory",
      undefined,
      undefined,
      0,
      "aliased",
    );
  }

  it("writeTurn posts to /turns", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        id: "t1",
        session_id: "s1",
        turn_seq: 0,
        acknowledged: true,
      }),
    );
    await aliasedClient().writeTurn({
      sessionId: "s1",
      role: "user",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });
    expect(getCallUrl(fetchMock.mock.calls[0])).toBe(
      "http://gw.test/api/v1/projects/p1/memory/turns",
    );
  });

  it("buildContext posts to /context", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        formatted_context: "",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      }),
    );
    await aliasedClient().buildContext({
      query: "q",
      sessionId: "s1",
      orgId: "o",
      userId: "u",
      projectId: "p",
    });
    expect(getCallUrl(fetchMock.mock.calls[0])).toBe(
      "http://gw.test/api/v1/projects/p1/memory/context",
    );
  });

  it("fetchSemanticMemories posts to /search with a type field", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ memories: [] }));
    await aliasedClient().fetchSemanticMemories({
      query: "q",
      orgId: "o",
      projectId: "p",
    });
    expect(getCallUrl(fetchMock.mock.calls[0])).toBe(
      "http://gw.test/api/v1/projects/p1/memory/search",
    );
    expect(getCallBody(fetchMock.mock.calls[0])["type"]).toBe("semantic");
  });
});
