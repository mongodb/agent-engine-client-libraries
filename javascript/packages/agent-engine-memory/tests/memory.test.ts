import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MemoryIdentityError, MemoryNotSupportedError } from "../src/errors.js";
import { Memory } from "../src/memory.js";
import type { MemoryChunk } from "../src/models.js";
import type {
  AmbientIdentityRuntime,
  MemoryCrudClient,
  MemoryRequestContext,
  MemoryRuntime,
} from "../src/transport.js";
import { fakeFetch } from "./helpers.js";

const ENV_KEYS = [
  "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN",
  "AGENTIC_MEMORY_API_KEY",
  "AGENTIC_MEMORY_BASE_URL",
  "AGENTIC_MEMORY_PROJECT_ID",
];

describe("Memory construction", () => {
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

  it("throws without a resolvable url or api key", () => {
    expect(() => new Memory()).toThrow(/requires baseUrl/);
  });

  it("selects flat OE routes from baseUrl with no projectId", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ baseUrl: "https://oe.test", fetchImpl });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/memory/search");
  });

  it("selects project-scoped routes and attaches a bearer token", async () => {
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: "https://gw.test",
      serviceAccountToken: "secret",
      projectId: "p1",
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].url).toContain("/api/v1/projects/p1/memory/search");
    expect(calls[0].headers.Authorization).toBe("Bearer secret");
  });

  it("reads env vars as fallback", async () => {
    process.env.AGENTIC_MEMORY_BASE_URL = "https://env.test";
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ fetchImpl });
    await mem.searchSemantic({ query: "q" });
    expect(calls[0].url).toContain("https://env.test/api/v1/memory/search");
  });

  it("reads the service-account token env var as fallback", async () => {
    process.env.AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN = "token-from-env";
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ fetchImpl });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].headers.Authorization).toBe("Bearer token-from-env");
  });

  it("accepts the legacy apiKey option with a deprecation warning", async () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({
      baseUrl: "https://gw.test",
      apiKey: "secret",
      fetchImpl,
    });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].headers.Authorization).toBe("Bearer secret");
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("serviceAccountToken"),
      "DeprecationWarning",
    );
    warn.mockRestore();
  });

  it("accepts the legacy AGENTIC_MEMORY_API_KEY env var with a deprecation warning", () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    process.env.AGENTIC_MEMORY_API_KEY = "legacy-from-env";
    new Memory({});
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN"),
      "DeprecationWarning",
    );
    warn.mockRestore();
  });

  it("rejects both new and legacy auth options", () => {
    expect(
      () => new Memory({ serviceAccountToken: "new", apiKey: "legacy" }),
    ).toThrow(/only one/);
  });

  it("rejects a new auth option combined with the legacy env var", () => {
    process.env.AGENTIC_MEMORY_API_KEY = "legacy-from-env";
    expect(() => new Memory({ serviceAccountToken: "new" })).toThrow(
      /only one/,
    );
  });

  it("rejects a blank serviceAccountToken", () => {
    expect(() => new Memory({ serviceAccountToken: "   " })).toThrow(
      /serviceAccountToken must be a non-empty string/,
    );
  });

  it("rejects a blank legacy apiKey without warning", () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    expect(() => new Memory({ apiKey: " " })).toThrow(
      /apiKey must be a non-empty string/,
    );
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("treats an empty-exported legacy env var as unset (no conflict, no warning)", async () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    process.env.AGENTIC_MEMORY_API_KEY = "";
    const { fetchImpl, calls } = fakeFetch([{ json: { memories: [] } }]);
    const mem = new Memory({ serviceAccountToken: "secret", fetchImpl });
    await mem.searchSemantic({ query: "q", userId: "u" });
    expect(calls[0].headers.Authorization).toBe("Bearer secret");
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("treats an empty-exported service-account env var as unset", () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    process.env.AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN = "";
    const mem = new Memory({ baseUrl: "https://gw.test", apiKey: "secret" });
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN"),
      "DeprecationWarning",
    );
    warn.mockRestore();
    expect(mem).toBeInstanceOf(Memory);
  });

  it("treats blank auth env vars as unset for the no-config error", () => {
    process.env.AGENTIC_MEMORY_API_KEY = "";
    process.env.AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN = "  ";
    expect(() => new Memory()).toThrow(/requires baseUrl/);
  });
});

// A minimal injected runtime/CRUD to exercise the facade's orchestration.
function stubRuntime(): { runtime: MemoryRuntime; searched: string[] } {
  const searched: string[] = [];
  const chunk = (id: string, score: number | null): MemoryChunk => ({
    id,
    content: id,
    source: "semantic",
    timestamp: new Date(0),
    similarity_score: score,
  });
  const runtime: MemoryRuntime = {
    recordTurn: async () => ({
      id: "",
      session_id: "s",
      turn_seq: 0,
      acknowledged: true,
    }),
    buildContext: async () => ({
      formatted_context: "",
      metadata: { token_count: 0, memory_counts: {}, timing: {} },
    }),
    searchSemantic: async () => {
      searched.push("semantic");
      return [chunk("a", 0.2), chunk("b", 0.9)];
    },
    searchEpisodes: async () => {
      searched.push("episodic");
      return [{ ...chunk("c", null), source: "episodic" }];
    },
    searchTaxonomic: async () => {
      searched.push("taxonomic");
      return [];
    },
    discoverProcedures: async () => {
      searched.push("procedural");
      return [];
    },
  };
  return { runtime, searched };
}

describe("Memory facade orchestration", () => {
  it("recordTurn auto-generates an idempotency key", async () => {
    let seenKey: string | null | undefined;
    const runtime = stubRuntime().runtime;
    runtime.recordTurn = async (a) => {
      seenKey = a.idempotencyKey;
      return { id: "", session_id: "s", turn_seq: 0, acknowledged: true };
    };
    const mem = new Memory({ runtime });
    await mem.recordTurn({ role: "user", content: "hi", sessionId: "s" });
    expect(typeof seenKey).toBe("string");
    expect((seenKey as string).length).toBeGreaterThan(0);
  });

  it("search merges sources and sorts by score, unscored last, truncating to topK", async () => {
    const { runtime, searched } = stubRuntime();
    const mem = new Memory({ runtime });
    const results = await mem.search({ query: "q", userId: "u", topK: 2 });
    expect(searched).toEqual(["semantic", "episodic"]); // default sources
    expect(results.map((r) => r.id)).toEqual(["b", "a"]); // 0.9, 0.2; null (c) dropped by topK
  });

  it("requireClient throws MemoryNotSupportedError when no CRUD client is injected", async () => {
    const mem = new Memory({ runtime: stubRuntime().runtime });
    await expect(
      mem.saveSemantic({ text: "t", label: "l", userId: "u" }),
    ).rejects.toBeInstanceOf(MemoryNotSupportedError);
  });

  it("delegates saveSemantic to the injected CRUD client", async () => {
    let received: unknown;
    const client = {
      createSemantic: async (a: unknown) => {
        received = a;
        return { id: "1", label: "l", has_embedding: true, acknowledged: true };
      },
    } as unknown as MemoryCrudClient;
    const mem = new Memory({ runtime: stubRuntime().runtime, client });
    const res = await mem.saveSemantic({
      text: "t",
      label: "l",
      userId: "u",
      metadata: { channel: "web" },
    });
    expect(res.id).toBe("1");
    expect(received).toMatchObject({
      text: "t",
      label: "l",
      userId: "u",
      metadata: { channel: "web" },
      upsert: true,
    });
  });

  it("bind supplies ambient identity to subsequent calls", async () => {
    let seenUser: string | null | undefined;
    const runtime = stubRuntime().runtime;
    runtime.searchSemantic = async (a) => {
      seenUser = a.userId;
      return [];
    };
    const mem = new Memory({ runtime }).bind({ userId: "bound-user" });
    await mem.searchSemantic({ query: "q" });
    expect(seenUser).toBe("bound-user");
  });

  it("buildContext forwards explicit maxTokens to the runtime", async () => {
    let seen: Parameters<MemoryRuntime["buildContext"]>[0] | undefined;
    const runtime = stubRuntime().runtime;
    runtime.buildContext = async (a) => {
      seen = a;
      return {
        formatted_context: "ctx",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      };
    };
    const mem = new Memory({ runtime });
    await mem.buildContext({ query: "q", maxTokens: 2048 });
    expect(seen?.maxTokens).toBe(2048);
    expect("topK" in (seen ?? {})).toBe(false);
  });

  it("buildContext omits maxTokens from runtime options when absent", async () => {
    let seen: Parameters<MemoryRuntime["buildContext"]>[0] | undefined;
    const runtime = stubRuntime().runtime;
    runtime.buildContext = async (a) => {
      seen = a;
      return {
        formatted_context: "ctx",
        metadata: { token_count: 0, memory_counts: {}, timing: {} },
      };
    };
    const mem = new Memory({ runtime });
    await mem.buildContext({ query: "q", sessionId: "s" });
    expect(seen).toBeDefined();
    expect("maxTokens" in (seen ?? {})).toBe(false);
  });

  it.each([
    0,
    -1,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    1.5,
  ])(
    "buildContext rejects invalid maxTokens (%s) before runtime delegation",
    async (maxTokens) => {
      let called = false;
      const runtime = stubRuntime().runtime;
      runtime.buildContext = async () => {
        called = true;
        return {
          formatted_context: "",
          metadata: { token_count: 0, memory_counts: {}, timing: {} },
        };
      };
      const mem = new Memory({ runtime });
      let err: unknown;
      try {
        await mem.buildContext({ query: "q", maxTokens });
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(RangeError);
      expect((err as Error).message).toBe(
        "maxTokens must be a positive integer",
      );
      expect(called).toBe(false);
    },
  );
});

describe("app-bound ambient identity is authoritative", () => {
  // A runtime that advertises ambient identity (like AppBoundRuntime): the
  // trusted user is resolved from the execution context, not the caller.
  function ambientRuntime(ctx: MemoryRequestContext): {
    runtime: MemoryRuntime & AmbientIdentityRuntime;
    seenUser: () => string | null | undefined;
  } {
    let seenUser: string | null | undefined;
    const runtime = {
      ...stubRuntime().runtime,
      requestContext: () => ctx,
      searchSemantic: async (a: { userId?: string | null }) => {
        seenUser = a.userId;
        return [];
      },
    } as MemoryRuntime & AmbientIdentityRuntime;
    return { runtime, seenUser: () => seenUser };
  }

  it("rejects a caller userId that mismatches the ambient identity", async () => {
    const { runtime } = ambientRuntime({ userId: "ambient-user" });
    const mem = new Memory({ runtime });
    await expect(
      mem.searchSemantic({ query: "q", userId: "someone-else" }),
    ).rejects.toBeInstanceOf(MemoryIdentityError);
  });

  it("allows a caller userId equal to the ambient identity", async () => {
    const { runtime, seenUser } = ambientRuntime({ userId: "ambient-user" });
    const mem = new Memory({ runtime });
    await mem.searchSemantic({ query: "q", userId: "ambient-user" });
    expect(seenUser()).toBe("ambient-user");
  });

  it("uses the ambient identity when the caller omits userId", async () => {
    const { runtime, seenUser } = ambientRuntime({ userId: "ambient-user" });
    const mem = new Memory({ runtime });
    await mem.searchSemantic({ query: "q" });
    expect(seenUser()).toBe("ambient-user");
  });
});
