/**
 * Tests for TenantRuntime.registerTool() — raw tool registration (runtime.ts).
 *
 * Mirrors runner-shared/tests/unit/test_runtime_register_tool.py.
 *
 * Ported cases:
 *   - stores the raw function by name (identity: runtime.tools[name] === fn)
 *     ← test_stores_raw_function (Python asserts runtime._tools[name] is fn).
 *   - stores the metadata dict; getToolMetadata(name) returns it
 *     ← test_stores_metadata.
 *   - registerTool does not wrap the fn into a framework/LangChain tool
 *     ← test_does_not_create_lc_tools (Python checks no _lc_tools entry).
 *
 * Added TS case:
 *   - getToolMetadata("missing") returns {} — the documented not-found default
 *     (runtime.ts: `return this.toolDefinitions[name] ?? {}`).
 *
 * Python→TS adaptation notes:
 *   - Python's private `_tools` / `_tool_definitions` are the public `tools` /
 *     `toolDefinitions` fields in TS; the identity + metadata guarantees are the
 *     same.
 *   - Python's "_lc_tools" absence check becomes: the stored value is the raw
 *     fn (identity), so no wrapping/transformation happened. TS has no _lc_tools
 *     concept — framework tool construction is the SDK's job, not the runtime's.
 */

import { describe, test, expect } from "vitest";
import { TenantRuntime, type ServerToolFn } from "../../src/index.js";

describe("TenantRuntime.registerTool", () => {
  test("stores the raw function by name (identity preserved)", () => {
    const runtime = new TenantRuntime({ appName: "Test" });
    const myFunc: ServerToolFn = (args) => args["x"];

    runtime.registerTool("my_func", myFunc, { is_local: true });

    // Raw fn stored verbatim — same reference, not a wrapper.
    expect(runtime.tools["my_func"]).toBe(myFunc);
  });

  test("stores the metadata object; getToolMetadata returns it", () => {
    const runtime = new TenantRuntime({ appName: "Test" });
    const myFunc: ServerToolFn = (args) => args["x"];
    const metadata = {
      name: "my_func",
      is_local: true,
      network: ["api.openai.com"],
      timeout_seconds: 60,
    };

    runtime.registerTool("my_func", myFunc, metadata);

    expect(runtime.toolDefinitions["my_func"]).toEqual(metadata);
    expect(runtime.getToolMetadata("my_func")).toEqual(metadata);
  });

  test("getToolMetadata returns {} for an unregistered tool", () => {
    const runtime = new TenantRuntime({ appName: "Test" });
    expect(runtime.getToolMetadata("missing")).toEqual({});
  });

  test("does not wrap the fn into a framework tool (no transformation)", () => {
    const runtime = new TenantRuntime({ appName: "Test" });
    const myFunc: ServerToolFn = (args) => args["x"];

    runtime.registerTool("my_func", myFunc, {});

    // The stored value is callable and is the same object we registered —
    // registerTool performs no LangChain/framework wrapping.
    const stored = runtime.tools["my_func"];
    expect(stored).toBe(myFunc);
    expect(typeof stored).toBe("function");
    expect(stored?.({ x: "hi" })).toBe("hi");
  });
});
