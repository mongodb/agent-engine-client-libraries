/**
 * Integration tests for the TenantRuntime.
 *
 * Mirrors Python's tests/integration/test_runtime.py.
 *
 * Note: tool registration via @app.tool() now lives in App (agent-engine-sdk-langgraph).
 * TenantRuntime exposes registerTool() for raw function registration.
 *
 * RUNNER_MODE=aer is set globally by tests/conftest.ts.
 *
 * TS note: Python's register_tool accepts any callable; TS enforces the
 * ServerToolFn contract `(args: Record<string, unknown>) => unknown`.
 * Test tool functions accept a single args object — same as how production
 * code invokes them via JSON request bodies.
 */

import { describe, test, expect, vi } from "vitest";

import { TenantRuntime } from "../../src/index.js";
import type { ServerToolFn } from "../../src/index.js";

describe("TenantRuntime", () => {
  test("basic runtime creation", () => {
    // test_runtime_creation
    const app = new TenantRuntime({
      appName: "Test Agent",
      appVersion: "1.0.0",
    });

    expect(app.appName).toBe("Test Agent");
    expect(app.appVersion).toBe("1.0.0");
  });

  test("constructor orgId/projectId are deprecated and ignored (env wins)", () => {
    // A hardcoded orgId/projectId must never override the
    // platform-injected env vars, otherwise memory writes get tagged with the
    // wrong tenant and the platform UI silently sees nothing. Mirrors Python's
    // test_constructor_org_id_and_project_id_are_ignored. vi.stubEnv auto-
    // restores after the test.
    vi.stubEnv("ORG_ID", "env_org");
    vi.stubEnv("PROJECT_ID", "env_proj");

    const runtime = new TenantRuntime({
      appName: "Test",
      orgId: "hardcoded_org",
      projectId: "hardcoded_proj",
    });

    expect(runtime.orgId).toBe("env_org");
    expect(runtime.projectId).toBe("env_proj");
  });

  test("registerTool stores raw function", () => {
    // test_runtime_register_tool
    const runtime = new TenantRuntime();

    const myTool: ServerToolFn = (args) => (args["x"] as number) * 2;

    runtime.registerTool("my_tool", myTool, {});

    expect(runtime.tools).toHaveProperty("my_tool");
    expect(runtime.tools["my_tool"]).toBe(myTool);
    expect(myTool({ x: 5 })).toBe(10);
  });

  test("registerTool with metadata", () => {
    // test_runtime_register_tool_with_metadata
    const runtime = new TenantRuntime();

    const secureOperation: ServerToolFn = (args) =>
      (args["data"] as string).toUpperCase();

    runtime.registerTool("secure_operation", secureOperation, {
      network: ["api.openai.com"],
    });

    expect(runtime.tools).toHaveProperty("secure_operation");
    expect(runtime.toolDefinitions["secure_operation"]?.["network"]).toEqual([
      "api.openai.com",
    ]);
    expect(secureOperation({ data: "hello" })).toBe("HELLO");
  });

  test("registerTool registers multiple tools", () => {
    // test_runtime_register_multiple_tools
    const runtime = new TenantRuntime();

    const toolOne: ServerToolFn = (args) => (args["x"] as number) + 1;
    const toolTwo: ServerToolFn = (args) => (args["x"] as number) * 2;
    const toolThree: ServerToolFn = (args) => (args["x"] as number) - 1;

    runtime.registerTool("tool_one", toolOne, {});
    runtime.registerTool("tool_two", toolTwo, {});
    runtime.registerTool("tool_three", toolThree, { timeout: 60 });

    expect(Object.keys(runtime.tools)).toHaveLength(3);
    expect(runtime.tools).toHaveProperty("tool_one");
    expect(runtime.tools).toHaveProperty("tool_two");
    expect(runtime.tools).toHaveProperty("tool_three");

    expect(toolOne({ x: 5 })).toBe(6);
    expect(toolTwo({ x: 5 })).toBe(10);
    expect(toolThree({ x: 5 })).toBe(4);
  });
});
