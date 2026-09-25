/**
 * Demonstration: memory persists across turns and sessions.
 *
 * Uses a fake in-process memory server (a Map-backed fetch stub, no network) that
 * implements the semantic create / get / search routes the OE proxy exposes. An
 * agent in "session A" saves a user preference; a fresh "session B" (new session
 * id, same user) recalls it — proving persistence is scoped to the user, not the
 * session. This demonstration is intentionally test-only.
 */

import { describe, expect, it } from "vitest";

import { Memory } from "../src/memory.js";
import type { FetchLike } from "../src/transport.js";

interface StoredSemantic {
  id: string;
  label: string;
  text: string;
  user_id: string;
}

/** A minimal Map-backed stand-in for the memory server's semantic routes. */
function fakeMemoryServer(): FetchLike {
  const store = new Map<string, StoredSemantic>();
  let nextId = 1;
  const key = (userId: string, label: string) => `${userId}::${label}`;

  return async (url, init) => {
    const parsed = new URL(url);
    const path = parsed.pathname;
    const method = init.method ?? "GET";
    const body =
      typeof init.body === "string"
        ? (JSON.parse(init.body) as Record<string, unknown>)
        : {};
    const json = (data: unknown, status = 200) =>
      new Response(JSON.stringify(data), {
        status,
        headers: { "Content-Type": "application/json" },
      });

    // Create (upsert) a semantic memory.
    if (path.endsWith("/memory/semantic") && method === "POST") {
      const userId = String(body.user_id);
      const label = String(body.label);
      const existing = store.get(key(userId, label));
      const id = existing?.id ?? String(nextId++);
      store.set(key(userId, label), {
        id,
        label,
        text: String(body.text),
        user_id: userId,
      });
      return json({ id, label, has_embedding: true, acknowledged: true }, 201);
    }

    // Get a semantic memory by label (returns an entries envelope).
    if (path.endsWith("/memory/semantic") && method === "GET") {
      const userId = parsed.searchParams.get("user_id") ?? "";
      const label = parsed.searchParams.get("label") ?? "";
      const found = store.get(key(userId, label));
      return json({ entries: found ? [found] : [] });
    }

    // Search: return the user's memories whose text matches the query substring.
    if (path.endsWith("/memory/search") && method === "POST") {
      const userId = String(body.user_id);
      const query = String(body.query).toLowerCase();
      const memories = [...store.values()]
        .filter(
          (m) => m.user_id === userId && m.text.toLowerCase().includes(query),
        )
        .map((m) => ({
          id: m.id,
          content: m.text,
          source: "semantic",
          timestamp: "2026-01-01T00:00:00Z",
          similarity_score: 0.9,
          metadata: { label: m.label },
        }));
      return json({ memories });
    }

    return json({ error: `unhandled ${method} ${path}` }, 404);
  };
}

/** Build a Memory over the fake server (flat OE routes, no projectId). */
function memoryForSession(fetchImpl: FetchLike): Memory {
  return new Memory({ baseUrl: "http://oe.local", fetchImpl });
}

describe("memory persists across turns and sessions", () => {
  it("recalls a user preference saved in an earlier session", async () => {
    const fetchImpl = fakeMemoryServer();
    const userId = "user-42";

    // --- Session A: the user states a preference; the agent saves it. ---
    const sessionA = memoryForSession(fetchImpl).bind({
      userId,
      sessionId: "session-A",
    });
    const saved = await sessionA.saveSemantic({
      text: "favorite color is teal",
      label: "favorite_color",
    });
    expect(saved.acknowledged).toBe(true);

    // --- Session B: a brand-new conversation, same user. ---
    const sessionB = memoryForSession(fetchImpl).bind({
      userId,
      sessionId: "session-B",
    });

    // Recall by exact label.
    const byLabel = (await sessionB.getSemantic({
      label: "favorite_color",
    })) as {
      text: string;
    } | null;
    expect(byLabel?.text).toBe("favorite color is teal");

    // Recall by semantic search.
    const hits = await sessionB.searchSemantic({ query: "color" });
    expect(hits.map((h) => h.content)).toContain("favorite color is teal");
  });

  it("does not leak another user's memory", async () => {
    const fetchImpl = fakeMemoryServer();
    await memoryForSession(fetchImpl)
      .bind({ userId: "alice", sessionId: "s1" })
      .saveSemantic({ text: "alice likes teal", label: "favorite_color" });

    const bob = memoryForSession(fetchImpl).bind({
      userId: "bob",
      sessionId: "s2",
    });
    expect(await bob.getSemantic({ label: "favorite_color" })).toBeNull();
    expect(await bob.searchSemantic({ query: "teal" })).toEqual([]);
  });
});
