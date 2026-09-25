/**
 * Tests for custom_headers context functions.
 *
 * Mirrors Python's tests/unit/test_custom_headers_context.py.
 *
 * Verifies isolation, get/set semantics, and that headers from one execution
 * context do not leak into another.
 *
 * TS-vs-Python note:
 *   Python: set_execution_context(exec_id, wrapper, oe_url, custom_headers=None)
 *           + try/finally clear_execution_context(tokens)   (positional + token pair)
 *   TS:     runWithExecutionContext({ executionId, wrapper, oeUrl, customHeaders? }, fn)
 *           (single options object; the frame is scoped to the callback)
 *   Same semantics; the set/clear token pair is collapsed into a callback.
 */

import { describe, test, expect } from "vitest";
import {
  runWithExecutionContext,
  getCurrentCustomHeaders,
} from "../../src/index.js";

const EXEC_ID = "test-exec-001";
const WRAPPER = {}; // analog of MagicMock() — wrapper is `unknown` in TS
const OE_URL = "http://localhost:8000";

describe("custom_headers context", () => {
  test("set with custom_headers makes getCurrentCustomHeaders return them", () => {
    // test_set_execution_context_with_custom_headers_returns_them
    const headers = {
      authorization: "Bearer token-abc",
      "x-tenant-id": "tenant-1",
    };
    runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        customHeaders: headers,
      },
      () => {
        expect(getCurrentCustomHeaders()).toEqual(headers);
      },
    );
  });

  test("set without custom_headers returns empty object", () => {
    // test_set_execution_context_without_custom_headers_returns_empty_dict
    runWithExecutionContext(
      { executionId: EXEC_ID, wrapper: WRAPPER, oeUrl: OE_URL },
      () => {
        expect(getCurrentCustomHeaders()).toEqual({});
      },
    );
  });

  test("frame teardown resets custom headers to empty", () => {
    // test_clear_execution_context_resets_custom_headers_to_empty
    runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        customHeaders: { "x-some-header": "value" },
      },
      () => {
        // Precondition: headers visible inside the frame.
        expect(getCurrentCustomHeaders()).toEqual({ "x-some-header": "value" });
      },
    );

    // Outside the frame the context is torn down automatically.
    expect(getCurrentCustomHeaders()).toEqual({});
  });

  test("sequential contexts do not leak headers", () => {
    // test_sequential_contexts_do_not_leak_headers
    const firstHeaders = {
      authorization: "Bearer first-secret",
      "x-request-id": "req-1",
    };
    runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        customHeaders: firstHeaders,
      },
      () => {
        expect(getCurrentCustomHeaders()).toEqual(firstHeaders);
      },
    );

    // Second context established without headers.
    runWithExecutionContext(
      { executionId: EXEC_ID, wrapper: WRAPPER, oeUrl: OE_URL },
      () => {
        const result = getCurrentCustomHeaders();
        expect(
          result,
          `Headers from the first context leaked: ${JSON.stringify(result)}`,
        ).toEqual({});
      },
    );
  });

  test("sequential contexts with different headers do not cross-contaminate", () => {
    // test_sequential_contexts_with_different_headers_do_not_cross_contaminate
    const firstHeaders = { "x-tenant-id": "tenant-A" };
    const secondHeaders = {
      "x-tenant-id": "tenant-B",
      "x-request-id": "req-2",
    };

    runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        customHeaders: firstHeaders,
      },
      () => {
        expect(getCurrentCustomHeaders()).toEqual(firstHeaders);
      },
    );

    runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        customHeaders: secondHeaders,
      },
      () => {
        const result = getCurrentCustomHeaders();
        expect(result).toEqual(secondHeaders);
        expect(
          result["x-tenant-id"],
          `Second context should see its own tenant-id, not the first context's`,
        ).toBe("tenant-B");
      },
    );
  });

  test("returns empty object before any context is set", () => {
    // test_get_current_custom_headers_returns_empty_dict_before_any_context_set
    expect(getCurrentCustomHeaders()).toEqual({});
  });
});
