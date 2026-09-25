/**
 * Tests for caller-payload context access (context.ts).
 *
 * Mirrors runner-shared/tests/unit/test_payload_context.py.
 *
 * Ported cases:
 *   - set_execution_context(payload=...) → get_current_payload returns it
 *     → runWithExecutionContext({ payload }, fn); getCurrentPayload() inside fn.
 *   - without payload → get_current_payload returns default.
 *   - clear_execution_context resets payload → frame torn down after fn returns.
 *   - sequential contexts do not leak payload → sequential scopes.
 *   - get_current_payload before any context set.
 *
 * Python→TS adaptation notes:
 *   - Python's set_execution_context(...) + try/finally clear_execution_context()
 *     token pair collapses into runWithExecutionContext(args, fn), which scopes
 *     the frame to the callback (see context.test.ts for the rationale).
 *   - DIVERGENCE: the TS default payload is `null`, NOT Python's `{}`. Python's
 *     get_current_payload() returns {} outside a context and when no payload was
 *     set; the TS accessor returns null in those cases (EMPTY_STORE.payload is
 *     null and args.payload ?? null). These tests assert the actual TS null.
 */

import { describe, test, expect } from "vitest";
import { runWithExecutionContext, getCurrentPayload } from "../../src/index.js";

const WRAPPER = {}; // analog of MagicMock(); wrapper is `unknown` in TS

function baseArgs(overrides: Record<string, unknown> = {}) {
  return {
    executionId: "test-exec-001",
    wrapper: WRAPPER,
    oeUrl: "http://localhost:8000",
    ...overrides,
  } as Parameters<typeof runWithExecutionContext>[0];
}

describe("getCurrentPayload with no active context", () => {
  test("returns null (TS default; Python returns {})", () => {
    expect(getCurrentPayload()).toBeNull();
  });
});

describe("getCurrentPayload inside an execution frame", () => {
  test("returns the payload passed to runWithExecutionContext", () => {
    const payload = { message: "Hello", extra: { screen: "policy-list" } };
    runWithExecutionContext(baseArgs({ payload }), () => {
      expect(getCurrentPayload()).toEqual(payload);
    });
  });

  test("returns null when no payload is supplied (TS default, not {})", () => {
    runWithExecutionContext(baseArgs(), () => {
      expect(getCurrentPayload()).toBeNull();
    });
  });

  test("resets to null after the frame tears down", () => {
    runWithExecutionContext(baseArgs({ payload: { extra: "v" } }), () => {
      expect(getCurrentPayload()).toEqual({ extra: "v" });
    });
    // Frame torn down on return — equivalent of Python's clear_execution_context.
    expect(getCurrentPayload()).toBeNull();
  });
});

describe("payload isolation across frames", () => {
  test("sequential frames do not leak payload", () => {
    runWithExecutionContext(baseArgs({ payload: { extra: "first" } }), () => {
      expect(getCurrentPayload()).toEqual({ extra: "first" });
    });

    runWithExecutionContext(baseArgs(), () => {
      const result = getCurrentPayload();
      expect(result).toBeNull();
    });
  });

  test("a nested frame's payload does not leak into the parent", () => {
    runWithExecutionContext(baseArgs({ payload: { extra: "outer" } }), () => {
      expect(getCurrentPayload()).toEqual({ extra: "outer" });
      runWithExecutionContext(baseArgs({ payload: { extra: "inner" } }), () => {
        expect(getCurrentPayload()).toEqual({ extra: "inner" });
      });
      // Inner frame must not mutate the parent's payload.
      expect(getCurrentPayload()).toEqual({ extra: "outer" });
    });
  });
});
