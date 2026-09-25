/**
 * Tests for custom_headers fields on ExecuteRequest and InvokeRequest
 * models.
 *
 * Mirrors Python's tests/unit/test_custom_headers_models.py.
 *
 * TS-vs-Python parity notes:
 *   - Python uses Pydantic Optional[X] (X | None). TS uses Zod .optional()
 *     which produces X | undefined. There's no null value — JSON.stringify
 *     omits undefined natively.
 *   - Python `req.custom_headers is None` → TS `req.custom_headers === undefined`.
 *   - Python `model_dump(exclude_none=True)` (omit None keys) → TS
 *     JSON.stringify naturally omits undefined keys, so a round-trip via
 *     JSON.stringify + JSON.parse achieves the same effect.
 *   - Python constructs with `ExecuteRequest(**fields)`; TS parses via
 *     `ExecuteRequestSchema.parse(fields)` since the type is a Zod-inferred
 *     interface, not a class.
 */

import { describe, test, expect } from "vitest";
import {
  ExecuteRequestSchema,
  InvokeRequestSchema,
  ToolExecuteRequestSchema,
} from "../../src/index.js";

const EXECUTE_REQUIRED = {
  execution_id: "exec-001",
  message: "test message",
  platform_api_url: "http://localhost:8000",
};
const INVOKE_REQUIRED = {
  message: "test message",
};
const TOOL_EXECUTE_REQUIRED = {
  execution_id: "exec-001",
  tool_name: "get_custom_headers",
  arguments: {},
  step_number: 1,
};

// ---------------------------------------------------------------------------
// ExecuteRequest
// ---------------------------------------------------------------------------

describe("ExecuteRequest custom_headers", () => {
  test("accepts a custom_headers object and stores it", () => {
    // test_execute_request_accepts_custom_headers
    const headers = {
      authorization: "Bearer secret",
      "x-tenant-id": "tenant-42",
    };
    const req = ExecuteRequestSchema.parse({
      ...EXECUTE_REQUIRED,
      custom_headers: headers,
    });

    expect(req.custom_headers).toEqual(headers);
  });

  test("survives a JSON.stringify / parse round-trip", () => {
    // test_execute_request_round_trips_custom_headers
    const headers = {
      "x-request-id": "req-999",
      "x-feature-flag": "dark-mode",
    };
    const req = ExecuteRequestSchema.parse({
      ...EXECUTE_REQUIRED,
      custom_headers: headers,
    });

    const dumped = JSON.parse(JSON.stringify(req));
    const restored = ExecuteRequestSchema.parse(dumped);

    expect(restored.custom_headers).toEqual(headers);
  });

  test("defaults custom_headers to undefined when not provided", () => {
    // test_execute_request_defaults_custom_headers_to_none
    const req = ExecuteRequestSchema.parse(EXECUTE_REQUIRED);

    expect(req.custom_headers).toBeUndefined();
  });

  test("JSON.stringify omits custom_headers key when undefined", () => {
    // test_execute_request_custom_headers_omitted_when_none_in_serialization
    const req = ExecuteRequestSchema.parse(EXECUTE_REQUIRED);

    const dumped = JSON.parse(JSON.stringify(req));

    expect(dumped).not.toHaveProperty("custom_headers");
  });

  test("JSON.stringify includes custom_headers key when set", () => {
    // test_execute_request_custom_headers_included_when_set_in_serialization
    const headers = { "x-tenant-id": "tenant-xyz" };
    const req = ExecuteRequestSchema.parse({
      ...EXECUTE_REQUIRED,
      custom_headers: headers,
    });

    const dumped = JSON.parse(JSON.stringify(req));

    expect(dumped["custom_headers"]).toEqual(headers);
  });

  test("accepts an empty object for custom_headers (distinct from undefined)", () => {
    // test_execute_request_accepts_empty_custom_headers_dict
    const req = ExecuteRequestSchema.parse({
      ...EXECUTE_REQUIRED,
      custom_headers: {},
    });

    expect(req.custom_headers).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// ToolExecuteRequest
// ---------------------------------------------------------------------------

describe("ToolExecuteRequest custom_headers", () => {
  test("survives a JSON.stringify / parse round-trip", () => {
    const headers = {
      authorization: "Bearer request-token",
      "x-tenant-id": "tenant-42",
    };
    const req = ToolExecuteRequestSchema.parse({
      ...TOOL_EXECUTE_REQUIRED,
      custom_headers: headers,
    });

    const restored = ToolExecuteRequestSchema.parse(
      JSON.parse(JSON.stringify(req)),
    );

    expect(restored.custom_headers).toEqual(headers);
  });

  test("JSON.stringify omits custom_headers when not provided", () => {
    const req = ToolExecuteRequestSchema.parse(TOOL_EXECUTE_REQUIRED);

    expect(JSON.parse(JSON.stringify(req))).not.toHaveProperty(
      "custom_headers",
    );
  });
});

// ---------------------------------------------------------------------------
// InvokeRequest
// ---------------------------------------------------------------------------

describe("InvokeRequest custom_headers", () => {
  test("accepts a custom_headers object and stores it", () => {
    // test_invoke_request_accepts_custom_headers
    const headers = {
      authorization: "Bearer invoke-token",
      "x-correlation-id": "corr-1",
    };
    const req = InvokeRequestSchema.parse({
      ...INVOKE_REQUIRED,
      custom_headers: headers,
    });

    expect(req.custom_headers).toEqual(headers);
  });

  test("survives a JSON.stringify / parse round-trip", () => {
    // test_invoke_request_round_trips_custom_headers
    const headers = { "x-user-id": "user-7", "x-session-id": "sess-88" };
    const req = InvokeRequestSchema.parse({
      ...INVOKE_REQUIRED,
      custom_headers: headers,
    });

    const dumped = JSON.parse(JSON.stringify(req));
    const restored = InvokeRequestSchema.parse(dumped);

    expect(restored.custom_headers).toEqual(headers);
  });

  test("defaults custom_headers to undefined when not provided", () => {
    // test_invoke_request_defaults_custom_headers_to_none
    const req = InvokeRequestSchema.parse(INVOKE_REQUIRED);

    expect(req.custom_headers).toBeUndefined();
  });

  test("JSON.stringify omits custom_headers key when undefined", () => {
    // test_invoke_request_custom_headers_omitted_when_none_in_serialization
    const req = InvokeRequestSchema.parse(INVOKE_REQUIRED);

    const dumped = JSON.parse(JSON.stringify(req));

    expect(dumped).not.toHaveProperty("custom_headers");
  });

  test("JSON.stringify includes custom_headers key when set", () => {
    // test_invoke_request_custom_headers_included_when_set_in_serialization
    const headers = { "x-tenant-id": "tenant-abc" };
    const req = InvokeRequestSchema.parse({
      ...INVOKE_REQUIRED,
      custom_headers: headers,
    });

    const dumped = JSON.parse(JSON.stringify(req));

    expect(dumped["custom_headers"]).toEqual(headers);
  });

  test("accepts an empty object for custom_headers (distinct from undefined)", () => {
    // test_invoke_request_accepts_empty_custom_headers_dict
    const req = InvokeRequestSchema.parse({
      ...INVOKE_REQUIRED,
      custom_headers: {},
    });

    expect(req.custom_headers).toEqual({});
  });
});
