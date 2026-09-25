/**
 * Tests for asynchronous tool registration via TenantRuntime.
 *
 * Mirrors Python's tests/integration/test_secure_async.py.
 *
 * Note: The @app.tool() decorator now lives in App (agent-engine-sdk-langgraph).
 * TenantRuntime exposes registerTool() for raw function registration.
 *
 * RUNNER_MODE=aer is set globally by tests/conftest.ts.
 *
 * TS note: Python's register_tool accepts any callable; TS enforces the
 * ServerToolFn contract `(args: Record<string, unknown>) => unknown`.
 */

import { describe, test, expect } from "vitest";
import { TenantRuntime } from "../../src/index.js";
import type { ServerToolFn } from "../../src/index.js";

describe("TenantRuntime async tool registration", () => {
  test("registering an async function via registerTool()", async () => {
    // test_async_tool_register
    const runtime = new TenantRuntime();

    const asyncAdd: ServerToolFn = async (args) =>
      (args["x"] as number) + (args["y"] as number);

    runtime.registerTool("async_add", asyncAdd, {});

    expect(runtime.tools).toHaveProperty("async_add");
    await expect(asyncAdd({ x: 2, y: 3 })).resolves.toBe(5);
  });

  test("registering an async function with tool metadata", async () => {
    // test_async_tool_register_with_metadata
    const runtime = new TenantRuntime();

    const asyncSummarize: ServerToolFn = async (args) =>
      (args["text"] as string).toUpperCase();

    runtime.registerTool("async_summarize", asyncSummarize, {
      timeout_seconds: 120,
    });

    expect(runtime.tools).toHaveProperty("async_summarize");
    await expect(asyncSummarize({ text: "hello" })).resolves.toBe("HELLO");
  });
});
