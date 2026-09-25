import { describe, expect, it } from "vitest";

import {
  scopedThreadId,
  sessionIdFromThreadId,
  threadIdsForQuery,
  threadIdsForSessionsQuery,
} from "../src/thread_id.js";

describe("scopedThreadId", () => {
  it("appends workspace suffix", () => {
    expect(scopedThreadId("sess-1", "ws-1")).toBe("sess-1:ws-1");
  });

  it("returns bare session when workspace is unset", () => {
    expect(scopedThreadId("sess-1", "")).toBe("sess-1");
    expect(scopedThreadId("sess-1", null)).toBe("sess-1");
  });
});

describe("sessionIdFromThreadId", () => {
  it("strips workspace suffix", () => {
    expect(sessionIdFromThreadId("sess-1:ws-1", "ws-1")).toBe("sess-1");
  });

  it("leaves legacy unscoped thread ids intact", () => {
    expect(sessionIdFromThreadId("legacy-session", "ws-1")).toBe(
      "legacy-session",
    );
  });
});

describe("threadIdsForQuery", () => {
  it("returns only the bare session key when workspace is unset", () => {
    expect(threadIdsForQuery("sess-1", "")).toEqual(["sess-1"]);
    expect(threadIdsForQuery("sess-1", null)).toEqual(["sess-1"]);
  });

  it("returns only the scoped key when workspace is set", () => {
    // Bare keys are shared across workspaces; reads must never include them
    // once a scope is known.
    expect(threadIdsForQuery("sess-1", "ws-1")).toEqual(["sess-1:ws-1"]);
  });

  it("dedupes session thread ids across sessions", () => {
    expect(threadIdsForSessionsQuery(["s1", "s2"], "ws-1")).toEqual([
      "s1:ws-1",
      "s2:ws-1",
    ]);
  });
});
