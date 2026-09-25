/*
 * Mirrors runner ../agent-engine-sdk-memory/tests/test_direct_parity.py
 * (PORTABLE SUBSET ONLY).
 *
 * Ported cases — the 3 cross-profile call-site tests, each asserted under BOTH
 * the gateway shape ({baseUrl, serviceAccountToken, projectId, fetchImpl}) and the OE shape
 * ({baseUrl, fetchImpl}), served by one branching FetchLike:
 * - recordTurn returns a WriteTurnResult
 * - searchEpisodes returns MemoryChunk[]
 * - buildContext returns a ContextResponse
 *
 * Parked (no TS meaning):
 * - test_runtime_methods_are_keyword_only (TS has no positional/keyword split;
 *   all facade args are object properties)
 * - test_runtime_satisfies_protocol_under_both_profiles (Python isinstance on a
 *   runtime Protocol; TS interfaces are structural + erased)
 */

import { describe, expect, it } from "vitest";

import { Memory } from "../src/memory.js";
import type { FetchLike } from "../src/transport.js";

const GW_PREFIX = "/api/v1/projects/proj-1/memory";
const OE_PREFIX = "/api/v1/memory";

/** One handler serving both the gateway and flat OE aliased route shapes. */
function handler(): FetchLike {
  const json = (data: unknown) =>
    new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  return async (url) => {
    const path = new URL(url).pathname;
    if (path === `${GW_PREFIX}/turns` || path === `${OE_PREFIX}/turns`)
      return json({
        id: "t1",
        session_id: "s",
        turn_seq: 1,
        acknowledged: true,
      });
    if (path === `${GW_PREFIX}/context` || path === `${OE_PREFIX}/context`)
      return json({ formatted_context: "", metadata: {} });
    if (path === `${GW_PREFIX}/search` || path === `${OE_PREFIX}/search`)
      return json({ memories: [] });
    return new Response(JSON.stringify({ error: `unhandled ${path}` }), {
      status: 404,
    });
  };
}

function gatewayMemory(): Memory {
  return new Memory({
    baseUrl: "http://gw",
    serviceAccountToken: "k",
    projectId: "proj-1",
    fetchImpl: handler(),
  });
}

function oeMemory(): Memory {
  return new Memory({ baseUrl: "http://oe", fetchImpl: handler() });
}

const PROFILES: Array<[string, () => Memory]> = [
  ["gateway", gatewayMemory],
  ["oe", oeMemory],
];

describe("cross-profile call-site parity", () => {
  it.each(PROFILES)(
    "recordTurn returns a WriteTurnResult (%s)",
    async (_n, make) => {
      const result = await make()
        .bind({ userId: "u", sessionId: "s" })
        .recordTurn({
          role: "user",
          content: "hi",
        });
      expect(result.id).toBe("t1");
      expect(result.session_id).toBe("s");
      expect(result.acknowledged).toBe(true);
    },
  );

  it.each(PROFILES)(
    "searchEpisodes returns MemoryChunk[] (%s)",
    async (_n, make) => {
      const result = await make().searchEpisodes({ query: "q", userId: "u" });
      expect(Array.isArray(result)).toBe(true);
    },
  );

  it.each(PROFILES)(
    "buildContext returns a ContextResponse (%s)",
    async (_n, make) => {
      const result = await make().buildContext({ query: "q", userId: "u" });
      expect(result).toHaveProperty("formatted_context");
      expect(result).toHaveProperty("metadata");
    },
  );
});
