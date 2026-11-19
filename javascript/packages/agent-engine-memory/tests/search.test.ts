/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_search.py
 *
 * Ported cases:
 * - defaults to semantic + episodic only (taxonomic not queried)
 * - merge + sort by score desc with null-score chunks last
 * - topK truncation (explicit topK, and the default of 10)
 * - procedural -> MemoryChunk adaptation (content precedence content ?? description ?? procedure)
 * - string and enum sources; a bare string treated as one source
 * - an unsupported source method propagates MemoryNotSupportedError
 * - an unknown source throws RangeError (TS's toSearchSource, vs Python's ValueError)
 * - a server error in one source leaves the others' results; on every source it throws the first
 * - a non-server error (auth) from one source still throws
 *
 * Adaptation notes:
 * - Python raises ValueError for an unknown source; the TS port narrows via
 *   toSearchSource, which throws RangeError. Asserted as RangeError here.
 * - The facade is async in TS, so the unknown-source and unsupported-source
 *   errors surface as rejected promises rather than synchronous raises.
 * - Injected via `new Memory({ runtime })` with a per-source result + call
 *   recorder (the stubRuntime pattern from memory.test.ts, extended).
 */

import { describe, expect, it, vi } from "vitest";

import {
  MemoryAuthError,
  MemoryNotSupportedError,
  MemoryServerError,
} from "../src/errors.js";
import { Memory } from "../src/memory.js";
import { SearchSource, type MemoryChunk } from "../src/models.js";
import type { MemoryRuntime } from "../src/transport.js";

function chunk(
  source: MemoryChunk["source"],
  score: number | null,
  id = "i",
): MemoryChunk {
  return {
    id,
    content: "c",
    source,
    timestamp: new Date(0),
    similarity_score: score,
  };
}

/** A MemoryRuntime stand-in returning preset results per source + a call log. */
function fakeRuntime(opts: {
  semantic?: MemoryChunk[];
  episodic?: MemoryChunk[];
  taxonomic?: MemoryChunk[];
  procedures?: Array<Record<string, unknown>>;
  taxonomicError?: Error;
  semanticError?: Error;
}): { runtime: MemoryRuntime; calls: string[] } {
  const calls: string[] = [];
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
      calls.push("semantic");
      if (opts.semanticError) throw opts.semanticError;
      return [...(opts.semantic ?? [])];
    },
    searchEpisodes: async () => {
      calls.push("episodic");
      return [...(opts.episodic ?? [])];
    },
    searchTaxonomic: async () => {
      calls.push("taxonomic");
      if (opts.taxonomicError) throw opts.taxonomicError;
      return [...(opts.taxonomic ?? [])];
    },
    discoverProcedures: async () => {
      calls.push("procedural");
      return [...(opts.procedures ?? [])];
    },
  };
  return { runtime, calls };
}

describe("Memory.search fan-out, merge, and adaptation", () => {
  it("defaults to semantic and episodic only", async () => {
    const { runtime, calls } = fakeRuntime({
      semantic: [chunk("semantic", 0.5)],
      episodic: [chunk("episodic", 0.9)],
      taxonomic: [chunk("taxonomic", 0.99)],
    });
    const results = await new Memory({ runtime }).search({
      query: "q",
      userId: "u",
    });
    expect(calls).toEqual(["semantic", "episodic"]);
    expect(results.map((c) => c.source)).toEqual(["episodic", "semantic"]);
  });

  it("merges and sorts by score, with unscored chunks last", async () => {
    const { runtime } = fakeRuntime({
      semantic: [chunk("semantic", 0.2, "a"), chunk("semantic", null, "b")],
      episodic: [chunk("episodic", 0.8, "c")],
    });
    const results = await new Memory({ runtime }).search({ query: "q" });
    expect(results.map((c) => c.id)).toEqual(["c", "a", "b"]);
  });

  it("truncates to an explicit topK", async () => {
    const { runtime } = fakeRuntime({
      semantic: [chunk("semantic", 0.9, "a"), chunk("semantic", 0.8, "b")],
      episodic: [chunk("episodic", 0.7, "c")],
    });
    const results = await new Memory({ runtime }).search({
      query: "q",
      topK: 2,
    });
    expect(results.map((c) => c.id)).toEqual(["a", "b"]);
  });

  it("truncates to the default topK of 10", async () => {
    const many = Array.from({ length: 11 }, (_, i) =>
      chunk("semantic", 1 - i / 100, `s${i}`),
    );
    const { runtime } = fakeRuntime({ semantic: many });
    const results = await new Memory({ runtime }).search({ query: "q" });
    expect(results).toHaveLength(10);
  });

  it("adapts a procedure dict into a chunk (description as content fallback)", async () => {
    const proc = { procedure: "start-quote", description: "d", score: 0.7 };
    const { runtime } = fakeRuntime({ procedures: [proc] });
    const results = await new Memory({ runtime }).search({
      query: "q",
      sources: ["procedural"],
    });
    expect(results).toHaveLength(1);
    const c = results[0];
    expect(c.source).toBe("procedural");
    expect(c.content).toBe("d");
    expect(c.similarity_score).toBe(0.7);
    expect(c.metadata).toEqual(proc);
  });

  it("honors content precedence content ?? description ?? procedure", async () => {
    const cases: Array<[Record<string, unknown>, string]> = [
      [{ content: "C", description: "D", procedure: "P" }, "C"],
      [{ description: "D", procedure: "P" }, "D"],
      [{ procedure: "P" }, "P"],
    ];
    for (const [proc, expected] of cases) {
      const { runtime } = fakeRuntime({ procedures: [proc] });
      const results = await new Memory({ runtime }).search({
        query: "q",
        sources: ["procedural"],
      });
      expect(results[0].content).toBe(expected);
    }
  });

  it("accepts string and enum sources", async () => {
    const { runtime: r1 } = fakeRuntime({
      taxonomic: [chunk("taxonomic", 0.5)],
    });
    expect(
      await new Memory({ runtime: r1 }).search({
        query: "q",
        sources: ["taxonomic"],
      }),
    ).toHaveLength(1);

    const { runtime: r2 } = fakeRuntime({
      taxonomic: [chunk("taxonomic", 0.5)],
    });
    expect(
      await new Memory({ runtime: r2 }).search({
        query: "q",
        sources: [SearchSource.TAXONOMIC],
      }),
    ).toHaveLength(1);
  });

  it("treats a bare string source as a single source", async () => {
    // A bare string must not be iterated character by character.
    const { runtime } = fakeRuntime({ taxonomic: [chunk("taxonomic", 0.5)] });
    expect(
      await new Memory({ runtime }).search({
        query: "q",
        sources: "taxonomic",
      }),
    ).toHaveLength(1);
  });

  it("propagates MemoryNotSupportedError from an unsupported source method", async () => {
    const { runtime } = fakeRuntime({
      semantic: [chunk("semantic", 0.5)],
      taxonomicError: new MemoryNotSupportedError(
        "no taxonomic in api-key mode",
      ),
    });
    await expect(
      new Memory({ runtime }).search({
        query: "q",
        sources: ["semantic", "taxonomic"],
      }),
    ).rejects.toBeInstanceOf(MemoryNotSupportedError);
  });

  it("throws RangeError (not a plain error) for an unknown source", async () => {
    const { runtime } = fakeRuntime({});
    await expect(
      new Memory({ runtime }).search({ query: "q", sources: ["stm"] }),
    ).rejects.toBeInstanceOf(RangeError);
  });

  it("returns the other sources' results when one fails on the server", async () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    const { runtime } = fakeRuntime({
      semantic: [chunk("semantic", 0.5, "s")],
      taxonomicError: new MemoryServerError("detail \u001b]52;c;boom\u0007", {
        status: 500,
      }),
    });
    const results = await new Memory({ runtime }).search({
      query: "q",
      sources: ["semantic", "taxonomic"],
    });
    expect(results.map((c) => c.id)).toEqual(["s"]);
    const message = String(warn.mock.calls[0]?.[0]);
    expect(message).toContain("taxonomic source failed (HTTP 500)");
    // Server-supplied text stays out of the warning.
    expect(message).not.toContain("detail");
    expect(message).not.toContain("\u001b");
    warn.mockRestore();
  });

  it("throws the first error when every requested source fails", async () => {
    const warn = vi.spyOn(process, "emitWarning").mockImplementation(() => {});
    const first = new MemoryServerError("semantic down", { status: 500 });
    const { runtime } = fakeRuntime({
      semanticError: first,
      taxonomicError: new MemoryServerError("taxonomic down", { status: 500 }),
    });
    await expect(
      new Memory({ runtime }).search({
        query: "q",
        sources: ["semantic", "taxonomic"],
      }),
    ).rejects.toBe(first);
    // No partial-result warning when nothing is returned.
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it("still throws an auth error from one source", async () => {
    const { runtime } = fakeRuntime({
      semantic: [chunk("semantic", 0.5)],
      taxonomicError: new MemoryAuthError("token expired", { status: 401 }),
    });
    await expect(
      new Memory({ runtime }).search({
        query: "q",
        sources: ["semantic", "taxonomic"],
      }),
    ).rejects.toBeInstanceOf(MemoryAuthError);
  });

  it.each([null, 502, 503, 504])(
    "still throws a retryable or malformed server error (status %s)",
    async (status) => {
      // null is a malformed success body; 502-504 tell the caller to retry.
      const { runtime } = fakeRuntime({
        semantic: [chunk("semantic", 0.5)],
        taxonomicError: new MemoryServerError("not a source fault", { status }),
      });
      await expect(
        new Memory({ runtime }).search({
          query: "q",
          sources: ["semantic", "taxonomic"],
        }),
      ).rejects.toBeInstanceOf(MemoryServerError);
    },
  );
});
