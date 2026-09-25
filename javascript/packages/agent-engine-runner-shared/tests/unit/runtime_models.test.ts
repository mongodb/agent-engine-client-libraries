/**
 * Unit tests for agent-engine-runner-shared API models + TenantRuntime.get_agent.
 *
 * Mirrors Python's tests/unit/test_runtime_models.py.
 *
 * TS-vs-Python adaptations:
 *   - Python constructs models via `Model(**fields)`; TS uses
 *     `ModelSchema.parse(fields)` since Zod-inferred types are interfaces.
 *   - Python `is None` → TS `=== undefined` (Zod `.optional()` produces
 *     undefined, not null — same parity note as custom_headers_models.test.ts).
 *   - Python `__new__` to skip __init__ → TS `Object.create(prototype)`,
 *     then inject `graphBuilder` via cast (same pattern as
 *     `hitl_interrupt.test.ts`).
 *   - `get_agent(callbacks=[...])` in Python → `getAgent({ callbacks: [...] })`
 *     in TS (options-object convention). `BaseExecutionCallback[]` cast as
 *     `unknown[]` since the test only verifies argument passing.
 */

import { describe, test, expect, vi } from "vitest";
import {
  InvokeRequestSchema,
  InvokeResponseSchema,
  HealthResponseSchema,
  HealthStatus,
  StreamChunkSchema,
  TenantRuntime,
} from "../../src/index.js";
import type { BaseExecutionCallback } from "@mongodb-js/agent-engine-sdk";

// ---------------------------------------------------------------------------
// InvokeRequest
// ---------------------------------------------------------------------------

describe("InvokeRequest", () => {
  test("constructs with explicit fields", () => {
    // test_invoke_request_model
    const request = InvokeRequestSchema.parse({
      message: "Hello",
      session_id: "session-123",
      user_id: "user-456",
    });

    expect(request.message).toBe("Hello");
    expect(request.session_id).toBe("session-123");
    expect(request.user_id).toBe("user-456");
  });

  test("defaults: session_id/user_id undefined, wait=true", () => {
    // test_invoke_request_defaults
    const request = InvokeRequestSchema.parse({ message: "Hello" });

    expect(request.message).toBe("Hello");
    expect(request.session_id).toBeUndefined();
    expect(request.user_id).toBeUndefined();
    expect(request.wait).toBe(true);
  });

  test("JSON serialization preserves set fields", () => {
    // test_invoke_request_model_dump
    const request = InvokeRequestSchema.parse({
      message: "Test message",
      session_id: "s-123",
      user_id: "u-456",
    });

    const data = JSON.parse(JSON.stringify(request));

    expect(data.message).toBe("Test message");
    expect(data.session_id).toBe("s-123");
    expect(data.user_id).toBe("u-456");
  });
});

// ---------------------------------------------------------------------------
// InvokeResponse
// ---------------------------------------------------------------------------

describe("InvokeResponse", () => {
  test("constructs with string result", () => {
    // test_invoke_response_model
    const response = InvokeResponseSchema.parse({
      result: "Hello back!",
      session_id: "session-123",
      user_id: "user-456",
    });

    expect(response.result).toBe("Hello back!");
    expect(response.session_id).toBe("session-123");
    expect(response.user_id).toBe("user-456");
  });

  test("preserves an object result verbatim", () => {
    // test_invoke_response_with_dict_result
    const response = InvokeResponseSchema.parse({
      result: { key: "value", count: 42 },
      session_id: "session-123",
    });

    expect(response.result).toEqual({ key: "value", count: 42 });
  });

  test("optional fields default to undefined", () => {
    // test_invoke_response_optional_fields
    const response = InvokeResponseSchema.parse({ result: "Result only" });

    expect(response.result).toBe("Result only");
    expect(response.session_id).toBeUndefined();
    expect(response.user_id).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// HealthResponse
// ---------------------------------------------------------------------------

describe("HealthResponse", () => {
  test("constructs with explicit fields", () => {
    // test_health_response_model
    const response = HealthResponseSchema.parse({
      status: HealthStatus.HEALTHY,
      component: "Test Agent",
      mode: "orchestrator",
      version: "1.0.0",
    });

    expect(response.status).toBe(HealthStatus.HEALTHY);
    expect(response.component).toBe("Test Agent");
    expect(response.mode).toBe("orchestrator");
    expect(response.version).toBe("1.0.0");
  });

  test("preserves details map", () => {
    // test_health_response_with_details
    const response = HealthResponseSchema.parse({
      status: HealthStatus.DEGRADED,
      component: "Test Agent",
      mode: "aer",
      details: { memory: "low", cpu: "high" },
    });

    expect(response.status).toBe(HealthStatus.DEGRADED);
    expect(response.details).toEqual({ memory: "low", cpu: "high" });
  });
});

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

describe("StreamChunk", () => {
  test("text content", () => {
    // test_stream_chunk_text
    const chunk = StreamChunkSchema.parse({
      chunk_type: "text",
      content: "Hello, world!",
    });

    expect(chunk.chunk_type).toBe("text");
    expect(chunk.content).toBe("Hello, world!");
    expect(chunk.metadata).toEqual({});
    expect(chunk.error).toBeUndefined();
  });

  test("metadata-only chunk defaults content to empty", () => {
    // test_stream_chunk_metadata
    const chunk = StreamChunkSchema.parse({
      chunk_type: "metadata",
      metadata: { thread_id: "123", user_id: "456" },
    });

    expect(chunk.chunk_type).toBe("metadata");
    expect(chunk.metadata).toEqual({ thread_id: "123", user_id: "456" });
    expect(chunk.content).toBe("");
  });

  test("error chunk preserves error string", () => {
    // test_stream_chunk_error
    const chunk = StreamChunkSchema.parse({
      chunk_type: "error",
      error: "Something went wrong",
    });

    expect(chunk.chunk_type).toBe("error");
    expect(chunk.error).toBe("Something went wrong");
    expect(chunk.content).toBe("");
  });

  test("tool_call chunk preserves tool_name + tool_call_id", () => {
    // test_stream_chunk_tool_call
    const chunk = StreamChunkSchema.parse({
      chunk_type: "tool_call",
      tool_name: "my_tool",
      tool_call_id: "tc-123",
      content: '{"arg": "value"}',
    });

    expect(chunk.chunk_type).toBe("tool_call");
    expect(chunk.tool_name).toBe("my_tool");
    expect(chunk.tool_call_id).toBe("tc-123");
  });

  test("done marker has empty content + metadata", () => {
    // test_stream_chunk_done
    const chunk = StreamChunkSchema.parse({ chunk_type: "done" });

    expect(chunk.chunk_type).toBe("done");
    expect(chunk.content).toBe("");
    expect(chunk.metadata).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// TenantRuntime.get_agent
// ---------------------------------------------------------------------------

/** Build a runtime without running its constructor — Python `__new__` analog. */
function makeRuntime(graphBuilder: unknown): TenantRuntime {
  const rt = Object.create(TenantRuntime.prototype) as TenantRuntime;
  (rt as unknown as { graphBuilder: unknown }).graphBuilder = graphBuilder;
  return rt;
}

describe("TenantRuntime.get_agent", () => {
  test("throws when no App is registered (graphBuilder is null)", () => {
    // test_raises_when_no_app_registered
    const rt = makeRuntime(null);

    expect(() => rt.getAgent()).toThrow(/No App registered/);
  });

  test("delegates to graphBuilder.getAgent and forwards callbacks", () => {
    // test_delegates_to_app_get_agent
    const sentinel = { __sentinel: "agent" };
    const mockApp = { getAgent: vi.fn().mockReturnValue(sentinel) };
    const rt = makeRuntime(mockApp);

    const result = rt.getAgent({
      callbacks: ["cb1"] as unknown as BaseExecutionCallback[],
    });

    expect(result).toBe(sentinel);
    expect(mockApp.getAgent).toHaveBeenCalledOnce();
    expect(mockApp.getAgent).toHaveBeenCalledWith({ callbacks: ["cb1"] });
  });

  test("throws for plain callable graph builder (no getAgent method)", () => {
    // test_plain_callable_raises
    const rt = makeRuntime(() => "graph");

    expect(() => rt.getAgent()).toThrow(/must be a BaseApp instance/);
  });
});
