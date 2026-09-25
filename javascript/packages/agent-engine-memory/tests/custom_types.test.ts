import { describe, expect, it } from "vitest";

import { Memory } from "../src/memory.js";
import {
  MemoryBadRequestError,
  MemoryClientError,
  MemoryNotSupportedError,
} from "../src/errors.js";
import { fakeFetch } from "./helpers.js";

describe("Memory.save (custom types)", () => {
  it("posts to the project-scoped types route with bearer auth", async () => {
    const { fetchImpl, calls } = fakeFetch([
      {
        status: 201,
        json: {
          id: "m1",
          type: "tickets",
          tags: { "profile.tier": "gold" },
          has_embedding: true,
        },
      },
    ]);
    const mem = new Memory({
      baseUrl: "https://gw.test",
      serviceAccountToken: "secret",
      projectId: "p1",
      fetchImpl,
    });
    const r = await mem.save("tickets", "c", {
      tags: { profile: { tier: "gold" } },
    });
    expect(calls[0].url).toBe(
      "https://gw.test/api/v1/projects/p1/memory/types/tickets",
    );
    expect(calls[0].headers.Authorization).toBe("Bearer secret");
    expect(calls[0].body).toEqual({
      content: "c",
      tags: { profile: { tier: "gold" } },
    });
    expect(r.tags["profile.tier"]).toBe("gold");
  });

  it("rejects built-in type names before any HTTP call", async () => {
    const { fetchImpl, calls } = fakeFetch([]);
    const mem = new Memory({ baseUrl: "https://gw.test", fetchImpl });
    await expect(mem.save("semantic", "c")).rejects.toThrow(
      "'semantic' is a built-in memory type and is not accepted by this operation",
    );
    expect(calls).toHaveLength(0);
  });

  it("rejects bad tag syntax before any HTTP call", async () => {
    const { fetchImpl, calls } = fakeFetch([]);
    const mem = new Memory({ baseUrl: "https://gw.test", fetchImpl });
    await expect(
      mem.save("tickets", "c", { tags: { "a.b.c": "x" } }),
    ).rejects.toThrow(MemoryClientError);
    expect(calls).toHaveLength(0);
  });
});

describe("Memory.retrieve (custom types)", () => {
  it("posts query/tags/top_k to the flat OE retrieve route", async () => {
    const { fetchImpl, calls } = fakeFetch([
      { json: { results: [], count: 0 } },
    ]);
    const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
    await mem.retrieve("tickets", "q", { tags: { open: true }, topK: 5 });
    expect(calls[0].url).toBe(
      "http://oe.test/api/v1/memory/types/tickets/retrieve",
    );
    expect(calls[0].body).toEqual({
      query: "q",
      tags: { open: true },
      top_k: 5,
    });
  });

  it("maps a bare 404 to a clear unsupported-platform error", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, text: "Not Found" }]);
    const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
    await expect(mem.retrieve("tickets", "q")).rejects.toThrow(
      MemoryNotSupportedError,
    );
  });

  it("keeps the underlying HTTP error as the cause of the mapped error", async () => {
    const { fetchImpl } = fakeFetch([{ status: 404, text: "Not Found" }]);
    const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
    const err = await mem.retrieve("tickets", "q").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(MemoryNotSupportedError);
    expect((err as Error).cause).toBeInstanceOf(MemoryBadRequestError);
    expect(((err as Error).cause as MemoryBadRequestError).status).toBe(404);
  });
});

describe("feature-flag-off 400 handling", () => {
  // The gateway rejects flag-off deployments with a 400 whose code is the
  // generic INVALID_REQUEST, so the message substring is the only stable
  // discriminator; that 400 must surface as MemoryNotSupportedError.
  const flagOffResponse = {
    status: 400,
    json: {
      code: "INVALID_REQUEST",
      error: "custom_memory_types is not enabled on this deployment",
    },
  };

  it("maps the flag-off 400 to a disabled-deployment error for both operations", async () => {
    for (const call of ["save", "retrieve"] as const) {
      const { fetchImpl } = fakeFetch([flagOffResponse]);
      const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
      const op =
        call === "save"
          ? mem.save("tickets", "c")
          : mem.retrieve("tickets", "q");
      const err = await op.catch((e: unknown) => e);
      expect(err).toBeInstanceOf(MemoryNotSupportedError);
      expect((err as Error).message).toContain("disabled on this deployment");
      expect((err as Error).cause).toBeInstanceOf(MemoryBadRequestError);
    }
  });

  it("passes other 400s through untouched", async () => {
    const { fetchImpl } = fakeFetch([
      {
        status: 400,
        json: {
          error: "tag 'urgent' is not declared for custom type 'tickets'",
        },
      },
    ]);
    const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
    const err = await mem
      .save("tickets", "c", { tags: { urgent: "true" } })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(MemoryBadRequestError);
    expect((err as Error).message).toContain("is not declared");
  });
});

describe("server score field", () => {
  // The server still projects `score`; the SDK deliberately does not expose
  // it. Parsing must tolerate the extra key so the two can diverge safely.
  it("ignores it rather than failing to parse", async () => {
    const { fetchImpl } = fakeFetch([
      {
        json: {
          results: [
            {
              id: "m1",
              type: "tickets",
              content: "c",
              tags: { queue: "billing" },
              score: 0.87,
              user_id: "u1",
            },
          ],
          count: 1,
        },
      },
    ]);
    const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
    const r = await mem.retrieve("tickets", "q");
    expect(r.count).toBe(1);
    expect(r.results[0]).not.toHaveProperty("score");
    expect(r.results[0].tags).toEqual({ queue: "billing" });
  });
});

describe("Memory facade 405 handling", () => {
  it("reports an unsupported platform for both operations", async () => {
    for (const call of ["save", "retrieve"] as const) {
      const { fetchImpl } = fakeFetch([
        { status: 405, text: "Method Not Allowed" },
      ]);
      const mem = new Memory({ baseUrl: "http://oe.test", fetchImpl });
      const op =
        call === "save"
          ? mem.save("tickets", "c")
          : mem.retrieve("tickets", "q");
      await expect(op).rejects.toThrow(MemoryNotSupportedError);
    }
  });
});
