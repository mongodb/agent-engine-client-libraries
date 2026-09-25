/**
 * Tests for synchronous tool registration via TenantRuntime.
 *
 * Mirrors Python's tests/integration/test_secure_sync.py.
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

describe("TenantRuntime sync tool registration", () => {
  test("registering a sync function via registerTool()", () => {
    // test_sync_tool_register
    const runtime = new TenantRuntime();

    const add: ServerToolFn = (args) =>
      (args["x"] as number) + (args["y"] as number);

    runtime.registerTool("add", add, {});

    expect(runtime.tools).toHaveProperty("add");
    expect(add({ x: 2, y: 3 })).toBe(5);
  });

  test("registering a sync function with tool metadata", () => {
    // test_sync_tool_register_with_metadata
    const runtime = new TenantRuntime();

    const summarize: ServerToolFn = (args) =>
      (args["text"] as string).toUpperCase();

    runtime.registerTool("summarize", summarize, { timeout_seconds: 120 });

    expect(runtime.tools).toHaveProperty("summarize");
    expect(summarize({ text: "hello" })).toBe("HELLO");
  });

  test("registerTool() with redact_fields metadata", () => {
    // test_sync_tool_register_with_redact_metadata
    const runtime = new TenantRuntime();

    const login: ServerToolFn = (args) =>
      args["username"] === "admin" && args["password"] === "secret";

    runtime.registerTool("login", login, { redact_fields: ["password"] });

    expect(runtime.tools).toHaveProperty("login");
    expect(login({ username: "admin", password: "secret" })).toBe(true);
  });
});
