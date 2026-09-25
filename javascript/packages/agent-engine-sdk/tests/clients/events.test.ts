/** Tests for EventClient HTTP wrapper. No Python equivalent; mirrors the
 *  mock-fetch pattern from memory.test.ts. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EventClient } from "../../src/clients/events.js";

type FetchMock = ReturnType<typeof vi.fn>;

function mockJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const sampleEvent = {
  event_id: "evt_001",
  session_id: "sess-1",
  actor_id: "user",
  payload: { text: "hello" },
  timestamp: "2024-01-15T10:30:00Z",
};

let fetchMock: FetchMock;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("EventClient init", () => {
  it("strips trailing slash from oeUrl", async () => {
    const client = new EventClient("http://localhost:8080/");
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));
    await client.appendEvent("sess-1", "user", { text: "hi" });
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8080/v1/sessions/sess-1/events");
  });
});

describe("appendEvent", () => {
  it("posts to /v1/sessions/<id>/events with body", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    const ev = await client.appendEvent("sess-1", "user", { text: "hello" });

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8080/v1/sessions/sess-1/events");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body["actor_id"]).toBe("user");
    expect(body["payload"]).toEqual({ text: "hello" });
    expect(ev.event_id).toBe("evt_001");
    expect(ev.timestamp).toBeInstanceOf(Date);
  });

  it("includes optional fields when provided", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    await client.appendEvent(
      "sess-1",
      "user",
      { text: "hi" },
      "evt_parent",
      { step: "reply" },
      { name: "branch-a", root_event_id: "evt_root" },
    );

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body["parent_event_id"]).toBe("evt_parent");
    expect(body["metadata"]).toEqual({ step: "reply" });
    expect(body["branch"]).toEqual({
      name: "branch-a",
      root_event_id: "evt_root",
    });
  });

  it("throws on non-OK response", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("bad request", { status: 400 }),
    );

    const client = new EventClient("http://localhost:8080");
    await expect(
      client.appendEvent("sess-1", "user", { text: "hi" }),
    ).rejects.toThrow(/HTTP 400/);
  });
});

describe("listEvents", () => {
  it("GETs /v1/sessions/<id>/events without query when unfiltered", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({ events: [sampleEvent] }),
    );

    const client = new EventClient("http://localhost:8080");
    const events = await client.listEvents("sess-1");

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("http://localhost:8080/v1/sessions/sess-1/events");
    expect(events).toHaveLength(1);
    expect(events[0]?.event_id).toBe("evt_001");
  });

  it("includes branch filter as query param", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ events: [] }));

    const client = new EventClient("http://localhost:8080");
    await client.listEvents("sess-1", "flight-search");

    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("branch=flight-search");
  });

  it("includes metadata.* filters as query params", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ events: [] }));

    const client = new EventClient("http://localhost:8080");
    await client.listEvents("sess-1", undefined, {
      step: "search",
      source: "agent",
    });

    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("metadata.step=search");
    expect(url).toContain("metadata.source=agent");
  });
});

describe("getEvent", () => {
  it("returns parsed event on 200", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    const ev = await client.getEvent("sess-1", "evt_001");

    expect(ev).not.toBeNull();
    expect(ev?.event_id).toBe("evt_001");
    expect(ev?.timestamp).toBeInstanceOf(Date);
  });

  it("returns null on 404", async () => {
    fetchMock.mockResolvedValueOnce(new Response("not found", { status: 404 }));

    const client = new EventClient("http://localhost:8080");
    const ev = await client.getEvent("sess-1", "missing");

    expect(ev).toBeNull();
  });

  it("throws on other error statuses", async () => {
    fetchMock.mockResolvedValueOnce(new Response("boom", { status: 500 }));

    const client = new EventClient("http://localhost:8080");
    await expect(client.getEvent("sess-1", "evt_001")).rejects.toThrow(
      /HTTP 500/,
    );
  });
});

describe("session/event id URL encoding", () => {
  // Regression tests for cross-session event read/write: sessionId and
  // eventId are caller-supplied wire values, so '/', '..', '?', '#' must not
  // rewrite the request path, query, or fragment.

  it("appendEvent encodes path traversal in sessionId", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    await client.appendEvent("../../admin/sessions/victim", "user", {
      text: "hi",
    });

    const [rawUrl] = fetchMock.mock.calls[0] as [string];
    const url = new URL(rawUrl);
    expect(url.pathname).toBe(
      "/v1/sessions/..%2F..%2Fadmin%2Fsessions%2Fvictim/events",
    );
  });

  it("listEvents encodes '#' so filters are not stripped into a fragment", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ events: [] }));

    const client = new EventClient("http://localhost:8080");
    await client.listEvents("sess-1#", "flight-search");

    const [rawUrl] = fetchMock.mock.calls[0] as [string];
    const url = new URL(rawUrl);
    expect(url.pathname).toBe("/v1/sessions/sess-1%23/events");
    // The branch filter survives as a query param, not swallowed by '#'.
    expect(url.searchParams.get("branch")).toBe("flight-search");
  });

  it("getEvent encodes path traversal in eventId", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    await client.getEvent("sess-1", "../../other-session/events/evt-target");

    const [rawUrl] = fetchMock.mock.calls[0] as [string];
    const url = new URL(rawUrl);
    expect(url.pathname).toBe(
      "/v1/sessions/sess-1/events/..%2F..%2Fother-session%2Fevents%2Fevt-target",
    );
  });

  it("encodes '/' in sessionId for all three methods", async () => {
    fetchMock
      .mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }))
      .mockResolvedValueOnce(mockJsonResponse({ events: [] }))
      .mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    await client.appendEvent("a/b", "user", { text: "hi" });
    await client.listEvents("a/b");
    await client.getEvent("a/b", "e/f");

    for (const call of fetchMock.mock.calls) {
      const [rawUrl] = call as [string];
      const url = new URL(rawUrl);
      expect(url.pathname.startsWith("/v1/sessions/a%2Fb/events")).toBe(true);
    }
  });

  it("leaves typical IDs (UUIDs, alphanumerics) unchanged", async () => {
    fetchMock.mockResolvedValueOnce(mockJsonResponse({ event: sampleEvent }));

    const client = new EventClient("http://localhost:8080");
    await client.getEvent("01908f3e-7c2a-7b3d-9e4f-2a1b3c4d5e6f", "evt_001");

    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe(
      "http://localhost:8080/v1/sessions/01908f3e-7c2a-7b3d-9e4f-2a1b3c4d5e6f/events/evt_001",
    );
  });

  it.each([".", ".."])(
    "rejects a bare dot-segment session/event id (%j) before any request",
    async (id) => {
      const client = new EventClient("http://localhost:8080");
      await expect(
        client.appendEvent(id, "user", { text: "hi" }),
      ).rejects.toThrow(/bare dot segment/);
      await expect(client.listEvents(id)).rejects.toThrow(/bare dot segment/);
      await expect(client.getEvent("sess-1", id)).rejects.toThrow(
        /bare dot segment/,
      );
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );
});
