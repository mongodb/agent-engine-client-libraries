/**
 * Tests for per-execution context management (context.ts).
 *
 * Mirrors the behaviour documented in Python's agent_engine_runner_shared/context.py:
 * each concurrent execution gets its own isolated frame, accessors read from
 * the current frame, and execution metadata is derived from it.
 *
 * TS-vs-Python note:
 *   Python uses `set_execution_context(...)` + `try/finally
 *   clear_execution_context(tokens)` backed by contextvars. TS collapses that
 *   token pair into `runWithExecutionContext(args, fn)` backed by
 *   AsyncLocalStorage#run, which scopes the frame to the callback and tears it
 *   down automatically — even on throw or across async boundaries. These tests
 *   assert that isolation guarantee, which is the reason the enterWith-based
 *   token API was removed.
 */

import { describe, test, expect } from "vitest";
import {
  runWithExecutionContext,
  getCurrentExecutionId,
  getCurrentWrapper,
  getCurrentOeUrl,
  getCurrentRequestId,
  getCurrentTraceId,
  getCurrentUserId,
  getCurrentSessionId,
  getCurrentWorkspaceId,
  getCurrentCustomHeaders,
  getAllCustomHeaders,
  getCurrentExecutionMetadata,
  recordCurrentMemoryMetadata,
} from "../../src/index.js";

const WRAPPER = {}; // analog of MagicMock() — wrapper is `unknown` in TS

function baseArgs(overrides: Record<string, unknown> = {}) {
  return {
    executionId: "exec-1",
    wrapper: WRAPPER,
    oeUrl: "http://localhost:8000",
    ...overrides,
  } as Parameters<typeof runWithExecutionContext>[0];
}

describe("accessors with no active context", () => {
  test("return null / empty defaults", () => {
    expect(getCurrentExecutionId()).toBeNull();
    expect(getCurrentWrapper()).toBeNull();
    expect(getCurrentOeUrl()).toBeNull();
    expect(getCurrentTraceId()).toBeNull();
    expect(getCurrentRequestId()).toBeNull();
    expect(getCurrentUserId()).toBeNull();
    expect(getCurrentSessionId()).toBeNull();
    expect(getCurrentWorkspaceId()).toBeNull();
    expect(getCurrentCustomHeaders()).toEqual({});
    expect(getCurrentExecutionMetadata()).toEqual({});
  });
});

describe("runWithExecutionContext", () => {
  test("exposes all fields inside the frame, returns the callback value, and tears down", () => {
    const result = runWithExecutionContext(
      baseArgs({
        executionId: "exec-abc",
        userId: "user-1",
        sessionId: "session-1",
        workspaceId: "ws-1",
        traceId: "0123456789abcdef0123456789abcdef",
        requestId: "legacy-request",
      }),
      () => {
        expect(getCurrentExecutionId()).toBe("exec-abc");
        expect(getCurrentWrapper()).toBe(WRAPPER);
        expect(getCurrentOeUrl()).toBe("http://localhost:8000");
        expect(getCurrentTraceId()).toBe("0123456789abcdef0123456789abcdef");
        expect(getCurrentRequestId()).toBe("legacy-request");
        expect(getCurrentUserId()).toBe("user-1");
        expect(getCurrentSessionId()).toBe("session-1");
        expect(getCurrentWorkspaceId()).toBe("ws-1");
        return 42;
      },
    );
    expect(result).toBe(42);
    expect(getCurrentExecutionId()).toBeNull();
    expect(getCurrentTraceId()).toBeNull();
  });

  test("auto-generates a req- request id without fabricating a trace id", () => {
    runWithExecutionContext(baseArgs(), () => {
      expect(getCurrentRequestId()).toMatch(/^req-[0-9a-f]{12}$/);
      expect(getCurrentTraceId()).toBeNull();
    });
  });

  test("tears the frame down even when the callback throws", () => {
    expect(() =>
      runWithExecutionContext(
        baseArgs({ traceId: "0123456789abcdef0123456789abcdef" }),
        () => {
          throw new Error("boom");
        },
      ),
    ).toThrow(/boom/);
    expect(getCurrentExecutionId()).toBeNull();
    expect(getCurrentTraceId()).toBeNull();
  });

  test("nested frames restore the parent frame on exit", () => {
    const traceId = "0123456789abcdef0123456789abcdef";
    runWithExecutionContext(baseArgs({ executionId: "outer", traceId }), () => {
      expect(getCurrentExecutionId()).toBe("outer");
      expect(getCurrentTraceId()).toBe(traceId);
      runWithExecutionContext(baseArgs({ executionId: "inner" }), () => {
        expect(getCurrentExecutionId()).toBe("inner");
        expect(getCurrentTraceId()).toBeNull();
      });
      // Inner frame must not mutate the parent.
      expect(getCurrentExecutionId()).toBe("outer");
      expect(getCurrentTraceId()).toBe(traceId);
    });
  });

  test("concurrent async executions do not cross-contaminate", async () => {
    // The core guarantee that motivated removing the enterWith token API:
    // two overlapping executions must each observe only their own context.
    const observe = (id: string, headerVal: string) =>
      runWithExecutionContext(
        baseArgs({ executionId: id, customHeaders: { "x-id": headerVal } }),
        async () => {
          // Yield so the two executions interleave on the event loop.
          await new Promise((r) => setTimeout(r, 0));
          return {
            execId: getCurrentExecutionId(),
            header: getCurrentCustomHeaders()["x-id"],
          };
        },
      );

    const [a, b] = await Promise.all([
      observe("exec-A", "A"),
      observe("exec-B", "B"),
    ]);

    expect(a).toEqual({ execId: "exec-A", header: "A" });
    expect(b).toEqual({ execId: "exec-B", header: "B" });
  });
});

describe("custom headers a2a stripping", () => {
  // Mirrors Python context.py: get_current_custom_headers() hides platform-internal
  // `a2a-` headers from agent code, while get_all_custom_headers() returns everything.
  const headers = {
    authorization: "Bearer token",
    "x-tenant-id": "tenant-1",
    "a2a-token": "secret",
    "a2a-route": "internal",
  };

  test("getCurrentCustomHeaders strips a2a- prefixed entries", () => {
    runWithExecutionContext(baseArgs({ customHeaders: headers }), () => {
      expect(getCurrentCustomHeaders()).toEqual({
        authorization: "Bearer token",
        "x-tenant-id": "tenant-1",
      });
    });
  });

  test("getAllCustomHeaders returns every entry including a2a-", () => {
    runWithExecutionContext(baseArgs({ customHeaders: headers }), () => {
      expect(getAllCustomHeaders()).toEqual(headers);
    });
  });

  test("both return empty object with no active frame", () => {
    expect(getCurrentCustomHeaders()).toEqual({});
    expect(getAllCustomHeaders()).toEqual({});
  });
});

describe("recordCurrentMemoryMetadata", () => {
  test("appends to memory_events and sets latest memory inside a frame", () => {
    runWithExecutionContext(baseArgs(), () => {
      recordCurrentMemoryMetadata({
        action: "read",
        memoryType: "semantic",
        content: "fact",
        relevanceScore: 0.9,
      });
      recordCurrentMemoryMetadata({ action: "write", memoryType: "episodic" });

      const metadata = getCurrentExecutionMetadata();
      expect(metadata["memory_events"]).toEqual([
        {
          action: "read",
          type: "semantic",
          content: "fact",
          relevance_score: 0.9,
        },
        { action: "write", type: "episodic" },
      ]);
      // `memory` holds the most recent event.
      expect(metadata["memory"]).toEqual({ action: "write", type: "episodic" });
    });
  });

  test("records query when provided and omits it otherwise", () => {
    runWithExecutionContext(baseArgs(), () => {
      recordCurrentMemoryMetadata({
        action: "recall",
        memoryType: "episodic",
        content: "fact",
        query: "customer interactions for u1",
      });
      recordCurrentMemoryMetadata({ action: "write", memoryType: "semantic" });

      const metadata = getCurrentExecutionMetadata();
      expect(metadata["memory_events"]).toEqual([
        {
          action: "recall",
          type: "episodic",
          content: "fact",
          query: "customer interactions for u1",
        },
        { action: "write", type: "semantic" },
      ]);
    });
  });

  test("is a no-op outside any execution frame", () => {
    expect(() =>
      recordCurrentMemoryMetadata({ action: "read", memoryType: "semantic" }),
    ).not.toThrow();
    expect(getCurrentExecutionMetadata()).toEqual({});
  });

  test("redacts credential-shaped fragments from the recorded query and content", () => {
    // Query and content are user/LLM-derived and this metadata is returned to
    // the UI, so credential-shaped fragments must be scrubbed before storage.
    runWithExecutionContext(baseArgs(), () => {
      recordCurrentMemoryMetadata({
        action: "recall",
        memoryType: "semantic",
        content: "see https://user:p%40ss@internal.example.com/db",
        query:
          "status of https://user:p%40ss@internal.example.com/db" +
          " with Authorization: Bearer abc123.def456 and api_key=sk-live-999",
      });

      const event = (
        getCurrentExecutionMetadata()["memory_events"] as Record<
          string,
          unknown
        >[]
      )[0];
      const stored = event["query"] as string;
      const content = event["content"] as string;
      for (const fragment of ["p%40ss", "abc123.def456", "sk-live-999"]) {
        expect(stored).not.toContain(fragment);
        expect(content).not.toContain(fragment);
      }
      expect(stored).toContain(
        "https://<redacted>:<redacted>@internal.example.com",
      );
      expect(content).toContain(
        "https://<redacted>:<redacted>@internal.example.com",
      );
      expect(stored).toContain("Bearer <redacted>");
      expect(stored).toContain("api_key=<redacted>");
    });
  });
});
