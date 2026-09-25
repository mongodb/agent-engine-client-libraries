/**
 * ToolServer startup / built-in-handler registration — TS port of the
 * deep-agent-relevant cases in Python's `test_tool_server_startup.py`.
 *
 * Covers the `features.deep_agent` gate: on → all 8 handlers registered with
 * metadata, user-name collisions overridden with a warning; off → nothing
 * registered and user tools preserved.
 */

import { describe, expect, it, vi } from "vitest";

import { ToolServer } from "../../src/index.js";
import type { ITenantRuntime, ServerToolFn } from "../../src/index.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { BUILTIN_TOOL_NAMES } from "../../src/toolpod_handlers.js";
import { getLogger } from "../../src/logger.js";

// log4js hands out a fresh wrapper per getLogger() call, but the level methods
// live on the shared Logger prototype — spying there intercepts the module's
// own instance too.
const loggerProto = Object.getPrototypeOf(getLogger("x")) as {
  warn: (msg: string) => void;
};

interface MockRuntime {
  tools: Record<string, ServerToolFn>;
  toolDefinitions: Record<string, Record<string, unknown>>;
  graphBuilder: null;
  getAgentConfig: () => RuntimeAgentConfig;
}

function makeRuntime(deepAgent: boolean | null): MockRuntime {
  return {
    tools: {},
    toolDefinitions: {},
    graphBuilder: null,
    getAgentConfig: () =>
      new RuntimeAgentConfig({
        features: { playground: null, deep_agent: deepAgent },
      }),
  };
}

function makeServer(runtime: MockRuntime): ToolServer {
  return new ToolServer(runtime as unknown as ITenantRuntime);
}

// Registration mkdir's the module-level WORKSPACE_DIR (bound at import to the
// default `/tmp/agent-workspace`); these tests assert registration wiring, not
// filesystem paths, so the default workspace is fine. Path/session behavior is
// covered by session_filesystem_isolation.test.ts.

describe("ToolServer built-in handler registration", () => {
  it("registers all 8 handlers with metadata when deep_agent is on", async () => {
    const runtime = makeRuntime(true);
    await makeServer(runtime).onStartup();
    for (const name of BUILTIN_TOOL_NAMES) {
      expect(typeof runtime.tools[name]).toBe("function");
      expect(runtime.toolDefinitions[name]).toMatchObject({
        name,
        is_local: false,
      });
    }
    expect(Object.keys(runtime.tools).sort()).toEqual(
      [...BUILTIN_TOOL_NAMES].sort(),
    );
  });

  it("does not register handlers when deep_agent is off", async () => {
    const runtime = makeRuntime(false);
    await makeServer(runtime).onStartup();
    for (const name of BUILTIN_TOOL_NAMES) {
      expect(runtime.tools[name]).toBeUndefined();
    }
  });

  it("does not register handlers when deep_agent is unset (default off)", async () => {
    const runtime = makeRuntime(null);
    await makeServer(runtime).onStartup();
    expect(Object.keys(runtime.tools)).toHaveLength(0);
  });

  it("preserves user-registered tools when the flag is off", async () => {
    const runtime = makeRuntime(false);
    const userTool: ServerToolFn = () => undefined;
    runtime.tools["my_tool"] = userTool;
    await makeServer(runtime).onStartup();
    expect(runtime.tools["my_tool"]).toBe(userTool);
  });

  it("keeps unrelated user tools alongside the built-ins when the flag is on", async () => {
    const runtime = makeRuntime(true);
    const userTool: ServerToolFn = () => undefined;
    runtime.tools["my_tool"] = userTool;
    await makeServer(runtime).onStartup();
    expect(runtime.tools["my_tool"]).toBe(userTool);
    expect(typeof runtime.tools["shell_execute"]).toBe("function");
  });

  it("overrides a user tool colliding with a built-in name and warns", async () => {
    const runtime = makeRuntime(true);
    const userShell: ServerToolFn = () => "user";
    runtime.tools["shell_execute"] = userShell;
    const warn = vi
      .spyOn(loggerProto, "warn")
      .mockImplementation(() => undefined);
    await makeServer(runtime).onStartup();
    expect(runtime.tools["shell_execute"]).not.toBe(userShell);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("overriding user-registered tool"),
    );
    warn.mockRestore();
  });
});
