/**
 * Port of `agent-engine-sdk-langgraph/tests/test_runtime.py`.
 *
 * Covers App initialization, tool decorator, checkpointer,
 * suspend, getAgent, and AER-mode tool
 * wrapping. Mirrors the Python test surface so regressions in the public API
 * — including the one where `app.tool()` silently shipped empty descriptions
 * and schemas — get caught by CI.
 *
 * NOT ported (deliberate omissions):
 * - Deep-agent tests (deep-agent support is out of scope for this PR; tracked
 *   for a later phase).
 * - app.llm() AER/TOOL coverage — already in `app_llm.test.ts`.
 *
 * MCP tool registration is covered below in "App — MCP tool registration",
 * mocking `discoverMcpTools` to avoid a live network dependency.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRequire } from "node:module";
import { create } from "@bufbuild/protobuf";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { BaseApp } from "@mongodb-js/agent-engine-sdk";
import {
  AttemptContextSchema,
  RuntimeMCPAuthConfigSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  clearWorkflowAdapter,
  getWorkflowAdapter,
  runWithAttemptContext,
  runWithExecutionContext,
  type TenantRuntime,
} from "@mongodb-js/agent-engine-runner-shared";
import type * as RunnerSharedTs from "@mongodb-js/agent-engine-runner-shared";

import { App } from "../src/runtime.js";
import { LangGraphBaseAgent } from "../src/agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import { withCallInterruptSupport } from "../src/call_interrupt.js";

const { discoverMcpToolsMock } = vi.hoisted(() => ({
  discoverMcpToolsMock: vi.fn(),
}));

vi.mock("@mongodb-js/agent-engine-runner-shared", async (importOriginal) => {
  const actual = await importOriginal<typeof RunnerSharedTs>();
  return { ...actual, discoverMcpTools: discoverMcpToolsMock };
});

// ---------------------------------------------------------------------------
// Global env setup — TenantRuntime requires RUNNER_MODE on every construction.
// Default to AER for tests that don't care about mode; per-test overrides go
// through `withMode(...)`.
// ---------------------------------------------------------------------------

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) {
    process.env["RUNNER_MODE"] = "aer";
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function privateOf<T extends object>(app: App): T {
  return app as unknown as T;
}

// This package's release version, read the same way production does — the
// registered adapter version must track it (never the unknown sentinel).
function packageVersion(): string {
  const require = createRequire(import.meta.url);
  return (require("../package.json") as { version: string }).version;
}

function withMode(mode: "aer" | "tool", fn: () => void): void {
  const prev = process.env["RUNNER_MODE"];
  process.env["RUNNER_MODE"] = mode;
  try {
    fn();
  } finally {
    if (prev === undefined) delete process.env["RUNNER_MODE"];
    else process.env["RUNNER_MODE"] = prev;
  }
}

// ---------------------------------------------------------------------------
// TestApp — App initialization, identity, tool decorator
// ---------------------------------------------------------------------------

describe("App — initialization", () => {
  it("is a BaseApp subclass", () => {
    // Mirrors Python test_is_base_app_subclass — guards the public class hierarchy.
    expect(App.prototype instanceof BaseApp).toBe(true);
  });

  it("creates an underlying TenantRuntime on init", () => {
    const app = new App({ appName: "Test Agent" });
    const internals = privateOf<{
      runtime: TenantRuntime;
      builderFn: unknown;
    }>(app);

    expect(app.name).toBe("Test Agent");
    expect(internals.runtime).toBeDefined();
    expect(internals.builderFn).toBeNull();
  });
});

describe("App — entrypoint decorator", () => {
  it("stores the builder function and returns it unchanged", () => {
    const app = new App({ appName: "Test Agent" });
    const buildAgent = (): string => "test_graph";

    const decorated = app.entrypoint(buildAgent);

    const internals = privateOf<{ builderFn: typeof buildAgent | null }>(app);
    expect(internals.builderFn).toBe(buildAgent);
    expect(decorated).toBe(buildAgent);
  });
});

// ---------------------------------------------------------------------------
// app.prepareAgentInput() decorator
// ---------------------------------------------------------------------------

describe("App — prepareAgentInput decorator", () => {
  it("stores the hook and returns it unchanged", () => {
    // Python: test_prepare_agent_input_decorator_stores_fn.
    const app = new App({ appName: "Test Agent" });
    const prepare = (): Record<string, unknown> => ({ messages: [] });

    const decorated = app.prepareAgentInput(prepare);

    const internals = privateOf<{ prepareInputFn: typeof prepare | null }>(app);
    expect(internals.prepareInputFn).toBe(prepare);
    expect(decorated).toBe(prepare);
  });
});

// ---------------------------------------------------------------------------
// app.resolveThreadId() decorator
// ---------------------------------------------------------------------------

describe("App — resolveThreadId decorator", () => {
  it("stores the hook and returns it unchanged", () => {
    // Python: test_resolve_thread_id_decorator_stores_fn.
    const app = new App({ appName: "Test Agent" });
    const resolve = (): string => "custom-thread";

    const decorated = app.resolveThreadId(resolve);

    const internals = privateOf<{ resolveThreadIdFn: typeof resolve | null }>(
      app,
    );
    expect(internals.resolveThreadIdFn).toBe(resolve);
    expect(decorated).toBe(resolve);
  });
});

// ---------------------------------------------------------------------------
// app.tool() decorator — the bug-catching tests
// ---------------------------------------------------------------------------

describe("App.tool() decorator", () => {
  it("registers the raw function on the runtime with metadata", () => {
    // Python: test_tool_decorator_registers_on_runtime
    const app = new App({ appName: "Test Agent" });
    const internals = privateOf<{ runtime: TenantRuntime }>(app);
    const registerSpy = vi.spyOn(internals.runtime, "registerTool");

    const myTool = app.tool({
      isLocal: false,
      timeout: 60,
      description: "Search for stuff.",
      schema: z.object({ query: z.string() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    expect(registerSpy).toHaveBeenCalledOnce();
    const [name, func, metadata] = registerSpy.mock.calls[0] ?? [];
    expect(name).toBe("my_tool");
    expect(func).toBe(myTool);
    const m = metadata as Record<string, unknown>;
    expect(m["is_local"]).toBe(false);
    expect(m["timeout_seconds"]).toBe(60);
  });

  it("creates a LangChain tool in App.lcTools", () => {
    // Python: test_tool_decorator_creates_lc_tool
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "Search for stuff.",
      schema: z.object({ query: z.string() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    const internals = privateOf<{ lcTools: Map<string, { name?: string }> }>(
      app,
    );
    expect(internals.lcTools.has("my_tool")).toBe(true);
    expect(internals.lcTools.get("my_tool")?.name).toBe("my_tool");
  });

  it("populates getToolDefinitions() with description, timeout, and remote flag", () => {
    // Python: test_tool_decorator_captures_tool_definition.
    // This is THE test that would have caught the bug — the description must
    // round-trip from the decorator options into ToolDefinition.description.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      isLocal: false,
      timeout: 60,
      description: "Search for stuff.",
      schema: z.object({ query: z.string() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    const defs = app.getToolDefinitions();
    expect(defs).toHaveLength(1);
    expect(defs[0]?.name).toBe("my_tool");
    expect(defs[0]?.description).toBe("Search for stuff.");
    expect(defs[0]?.remote).toBe(true);
    expect(defs[0]?.timeout_seconds).toBe(60);
  });

  it("derives a non-empty args_schema from the Zod schema", () => {
    // Python: test_tool_decorator_derives_args_schema.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "Search for stuff.",
      schema: z.object({ query: z.string(), limit: z.number().optional() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    const schema = app.getToolDefinitions()[0]?.args_schema as {
      type?: string;
      properties?: Record<string, { type?: string }>;
      required?: string[];
    };
    expect(schema.type).toBe("object");
    expect(Object.keys(schema.properties ?? {}).sort()).toEqual([
      "limit",
      "query",
    ]);
    expect(schema.properties?.["query"]?.type).toBe("string");
    expect(schema.required).toEqual(["query"]);
  });

  it("registers a parameterless tool with an object schema", () => {
    // Python: test_tool_decorator_no_params_registers_object_schema.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "No args.",
      schema: z.object({}),
    })(function ping(): string {
      return "pong";
    });

    const schema = app.getToolDefinitions()[0]?.args_schema as {
      type?: string;
      properties?: Record<string, unknown>;
    };
    expect(schema.type).toBe("object");
    expect(schema.properties ?? {}).toEqual({});
  });

  it("registers cleanly with an empty schema when none is provided", () => {
    const app = new App({ appName: "Test Agent" });
    app.tool({ description: "No schema." })(function bare(): string {
      return "ok";
    });

    expect(app.getToolDefinitions()[0]?.args_schema).toEqual({});
  });

  it("falls back to an empty schema when JSON Schema derivation throws", () => {
    // Python: test_tool_decorator_schema_derivation_failure_falls_back.
    // z.date() is unrepresentable in JSON Schema, so Zod 4's toJSONSchema
    // throws by default; the decorator must degrade to {} rather than crash.
    const app = new App({ appName: "Test Agent" });
    app.tool({ description: "Bad.", schema: z.object({ when: z.date() }) })(
      function bad_tool(): string {
        return "ok";
      },
    );

    expect(app.getToolDefinitions()[0]?.args_schema).toEqual({});
  });

  it("tools() returns the registered LangChain tools", () => {
    // Python: test_tools_returns_lc_tools.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "Search.",
      schema: z.object({ query: z.string() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    const result = app.tools();
    expect(result).toHaveLength(1);
    expect((result[0] as { name: string }).name).toBe("my_tool");
  });

  it("getToolSchemas() returns the original (unwrapped) LangChain tools", () => {
    // Python: test_get_tool_schemas_returns_lc_tools.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "Search.",
      schema: z.object({ query: z.string() }),
    })(function my_tool({ query }: { query: string }): string {
      return query;
    });

    const result = app.getToolSchemas();
    expect(result).toHaveLength(1);
    expect((result[0] as { name: string }).name).toBe("my_tool");
  });

  it("rejects duplicate user tool registration", () => {
    // Python: test_register_tool_definition_rejects_duplicate_user_tool.
    const app = new App({ appName: "Test Agent" });

    app.tool({
      description: "First.",
      schema: z.object({}),
    })(function duplicate_tool(): string {
      return "first";
    });

    // The second registration uses the same function name — must throw.
    expect(() =>
      app.tool({
        description: "Second.",
        schema: z.object({}),
      })(function duplicate_tool(): string {
        return "second";
      }),
    ).toThrow(/already registered/);
  });

  it("trims whitespace from the description", () => {
    // Python: test_register_tool_definition_strips_description.
    // The TS implementation trims via `description.trim()` in
    // registerToolDefinition; this verifies the public path runs it too.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      description: "  Spaced description  ",
      schema: z.object({}),
    })(function spaced_tool(): string {
      return "ok";
    });

    expect(app.getToolDefinitions()[0]?.description).toBe("Spaced description");
  });

  it("supports explicit name override", () => {
    // The TS SDK adds an explicit `name` option that Python derives from
    // `fn.__name__`. Verifies the override path.
    const app = new App({ appName: "Test Agent" });
    app.tool({
      name: "renamed_tool",
      description: "Renamed",
      schema: z.object({}),
    })(function anonymousImpl(): string {
      return "ok";
    });

    expect(app.getToolDefinitions()[0]?.name).toBe("renamed_tool");
  });
});

// ---------------------------------------------------------------------------
// app.getTools() — AER-mode wrapping
// ---------------------------------------------------------------------------

describe("App.getTools() — AER mode wrapping", () => {
  it("wraps registered tools when runtime mode is AER", () => {
    // Python: test_get_tools_wraps_in_aer_mode.
    withMode("aer", () => {
      const app = new App({ appName: "Test Agent" });
      app.tool({
        isLocal: true,
        description: "A local tool.",
        schema: z.object({ x: z.string() }),
      })(function local_tool({ x }: { x: string }): string {
        return x;
      });
      app.tool({
        isLocal: false,
        description: "A remote tool.",
        schema: z.object({ x: z.string() }),
      })(function remote_tool({ x }: { x: string }): string {
        return x;
      });

      const tools = app.getTools();
      expect(tools).toHaveLength(2);
      const names = new Set(tools.map((t) => (t as { name: string }).name));
      expect(names).toEqual(new Set(["local_tool", "remote_tool"]));
    });
  });

  it("forwards each tool's redaction fields to OE", async () => {
    const prev = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    try {
      const app = new App({ appName: "Test Agent" });
      app.tool({
        isLocal: true,
        redactFields: ["card_number"],
        description: "Charge a customer.",
        schema: z.object({ card_number: z.string() }),
      })(function charge_customer({
        card_number,
      }: {
        card_number: string;
      }): string {
        return `charged ${card_number.length}`;
      });

      const [tool] = app.getTools() as unknown as {
        name: string;
        invoke: (input: unknown) => Promise<unknown>;
      }[];
      if (tool === undefined) throw new Error("tool not found");
      const executeTool = vi.fn().mockResolvedValue("ok");
      await runWithExecutionContext(
        {
          executionId: "exec-1",
          wrapper: { executeTool },
          oeUrl: "http://localhost:8080",
        },
        async () => {
          await tool.invoke({
            name: "charge_customer",
            args: { card_number: "4111-1111" },
            id: "call_1",
            type: "tool_call",
          });
        },
      );
      expect(executeTool).toHaveBeenCalledOnce();
      const callArgs = executeTool.mock.calls[0] ?? [];
      const options = callArgs[2] as { redactFields?: string[] } | undefined;
      expect(options?.redactFields).toEqual(["card_number"]);
    } finally {
      if (prev === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = prev;
    }
  });

  it("carries the per-call Stop opt-in from registration through to the wrapper", async () => {
    const prev = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    try {
      const app = new App({ appName: "Test Agent" });
      app.tool({
        isLocal: true,
        description: "A cooperative slow check.",
        schema: z.object({ x: z.string() }),
      })(
        withCallInterruptSupport(function slow_check({
          x,
        }: {
          x: string;
        }): string {
          return x;
        }),
      );
      app.tool({
        isLocal: true,
        description: "A plain tool.",
        schema: z.object({ x: z.string() }),
      })(function plain_tool({ x }: { x: string }): string {
        return x;
      });

      const tools = app.getTools() as unknown as {
        name: string;
        invoke: (input: unknown) => Promise<unknown>;
      }[];
      const executeTool = vi.fn().mockResolvedValue("ok");
      const declared: Record<string, unknown> = {};
      await runWithExecutionContext(
        {
          executionId: "exec-1",
          wrapper: { executeTool },
          oeUrl: "http://localhost:8080",
        },
        async () => {
          for (const tool of tools) {
            await tool.invoke({
              name: tool.name,
              args: { x: "a" },
              id: `call_${tool.name}`,
              type: "tool_call",
            });
            const options = executeTool.mock.calls.at(-1)?.[2] as
              | { supportsCallInterrupt?: boolean }
              | undefined;
            // The wrapper forwards the flag only when set — absence means the
            // call registers without a controller and answers not_cancellable.
            declared[tool.name] = options?.supportsCallInterrupt === true;
          }
        },
      );
      // The brand on the registered callable must survive App.tool()'s
      // LangChain-tool construction; unbranded tools stay not_cancellable.
      expect(declared).toEqual({ slow_check: true, plain_tool: false });
    } finally {
      if (prev === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = prev;
    }
  });

  it("forwards the ToolCall id to OE through a real lcTool invocation", async () => {
    // Exercises the actual LangChain ToolCall/config shape end to end (not a
    // mocked config): a real ToolNode-style invocation must surface
    // tool_call_id to the OE so execution logs carry the execution-log join key.
    const prev = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    try {
      const app = new App({ appName: "Test Agent" });
      app.tool({
        isLocal: true,
        description: "Get weather.",
        schema: z.object({ city: z.string() }),
      })(function get_weather({ city }: { city: string }): string {
        return `weather ${city}`;
      });

      const wrapped = app.getTools()[0] as {
        invoke: (input: unknown) => Promise<unknown>;
      };

      const executeTool = vi.fn().mockResolvedValue("ok");
      await runWithExecutionContext(
        {
          executionId: "exec-1",
          wrapper: { executeTool },
          oeUrl: "http://localhost:8080",
        },
        async () => {
          await wrapped.invoke({
            name: "get_weather",
            args: { city: "Tokyo" },
            id: "call_real1",
            type: "tool_call",
          });
        },
      );

      expect(executeTool).toHaveBeenCalledOnce();
      const callArgs = executeTool.mock.calls[0] ?? [];
      expect(callArgs[0]).toBe("get_weather");
      expect(callArgs[1]).toEqual({ city: "Tokyo" });
      // executeTool(toolName, args, options) — toolCallId lives on the options object.
      const options = callArgs[2] as { toolCallId?: string } | undefined;
      expect(options?.toolCallId).toBe("call_real1");
    } finally {
      if (prev === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = prev;
    }
  });

  it("assigns platform Tool results a stable durable replay id", async () => {
    const app = new App({ appName: "Test Agent" });
    app.tool({
      isLocal: true,
      description: "Get weather.",
      schema: z.object({ city: z.string() }),
    })(function get_weather({ city }: { city: string }): string {
      return `weather ${city}`;
    });
    const wrapped = app.getTools()[0] as {
      invoke: (input: unknown) => Promise<unknown>;
    };
    const identity = create(WorkflowIdentitySchema, {
      tenantScope: create(TenantScopeSchema, {
        orgId: "org",
        projectId: "project",
        workspaceId: "workspace",
      }),
      sessionId: "session",
      executionId: "exec-1",
    });
    const toolCall = {
      name: "get_weather",
      args: { city: "Tokyo" },
      id: "call_1",
      type: "tool_call",
    };
    const executeTool = vi.fn().mockResolvedValue("ok");
    const invoke = (attemptId: string, fence: bigint) =>
      runWithExecutionContext(
        {
          executionId: "exec-1",
          wrapper: { executeTool },
          oeUrl: "http://oe",
        },
        () =>
          runWithAttemptContext(
            create(AttemptContextSchema, {
              attemptId,
              fencingToken: fence,
              workflowIdentity: identity,
            }),
            () => wrapped.invoke(toolCall),
          ),
      );

    const first = await invoke("attempt-1", 1n);
    const replay = await invoke("attempt-2", 2n);

    expect(first).toBeInstanceOf(ToolMessage);
    expect((first as ToolMessage).id).toBe(
      "durable-tool-result:exec-1:1:call_1",
    );
    expect((replay as ToolMessage).id).toBe((first as ToolMessage).id);
  });

  it("returns raw tools when runtime mode is not AER", () => {
    // Complement to the AER test — non-AER modes must skip wrapping.
    withMode("tool", () => {
      const app = new App({ appName: "Test Agent" });
      app.tool({
        description: "A tool.",
        schema: z.object({ x: z.string() }),
      })(function plain_tool({ x }: { x: string }): string {
        return x;
      });

      const tools = app.getTools();
      const internals = privateOf<{ lcTools: Map<string, unknown> }>(app);
      expect(tools).toHaveLength(1);
      // In non-AER, getTools returns the unwrapped tools directly from lcTools.
      expect(tools[0]).toBe(internals.lcTools.get("plain_tool"));
    });
  });
});

// ---------------------------------------------------------------------------
// App — deprecated orgId option
// ---------------------------------------------------------------------------

describe("App — deprecated orgId option", () => {
  it("ignores the orgId option: ORG_ID env var wins", () => {
    // A hardcoded orgId must never override the platform-injected env
    // var. The warning is emitted at App construction (visible in test output);
    // this assertion guards the safety invariant — the wrong org must never
    // be forwarded to the runtime.
    vi.stubEnv("ORG_ID", "env_org");
    const app = new App({ appName: "Test", orgId: "hardcoded_org" });
    const internals = app as unknown as { runtime: { orgId: string | null } };
    expect(internals.runtime.orgId).toBe("env_org");
  });
});

// ---------------------------------------------------------------------------
// app.suspend()
// ---------------------------------------------------------------------------

describe("App.suspend()", () => {
  it("returns a JSON string carrying suspend_reason and suspend_context", () => {
    // Python: test_suspend_returns_json_string.
    const app = new App({ appName: "Test Agent" });
    const result = app.suspend("needs_approval", { ticket: "INC-123" });

    const parsed = JSON.parse(result) as Record<string, unknown>;
    expect(parsed["suspend_reason"]).toBe("needs_approval");
    expect(parsed["suspend_context"]).toEqual({ ticket: "INC-123" });
  });
});

// ---------------------------------------------------------------------------
// app.finishSession()
// ---------------------------------------------------------------------------

describe("App.finishSession()", () => {
  it("reports requested then already_requested", async () => {
    const app = new App({ appName: "Test Agent" });
    await runWithExecutionContext(
      {
        executionId: "exec-1",
        wrapper: {},
        oeUrl: "http://oe",
        sessionId: "sess-1",
      },
      async () => {
        expect(app.finishSession()).toBe("requested");
        expect(app.finishSession()).toBe("already_requested");
      },
    );
  });

  it("reports unavailable outside a run", () => {
    const app = new App({ appName: "Test Agent" });
    expect(app.finishSession()).toBe("unavailable");
  });
});

// ---------------------------------------------------------------------------
// app.getCurrentUserId() — direct delegation
// ---------------------------------------------------------------------------

describe("App.getCurrentUserId()", () => {
  it("delegates to TenantRuntime.getCurrentUserId", () => {
    // Python: test_get_current_user_id_delegates_to_runtime.
    const app = new App({ appName: "Test Agent" });
    const internals = privateOf<{ runtime: TenantRuntime }>(app);
    const spy = vi
      .spyOn(internals.runtime, "getCurrentUserId")
      .mockReturnValue("user_123");

    expect(app.getCurrentUserId()).toBe("user_123");
    expect(spy).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// app.getAgent()
// ---------------------------------------------------------------------------

describe("App.getAgent()", () => {
  it("returns a LangGraphBaseAgent wrapping the built graph", () => {
    // Python: test_get_agent_returns_langgraph_base_agent.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);

    const agent = app.getAgent();
    expect(agent).toBeInstanceOf(LangGraphBaseAgent);
  });

  it("throws when no entrypoint is registered", () => {
    // Python: test_get_agent_raises_without_entrypoint.
    const app = new App({ appName: "Test Agent" });
    expect(() => app.getAgent()).toThrow(/No entrypoint registered/);
  });

  it("wraps execution callbacks in LangGraphCallbackAdapter", () => {
    // Python: test_get_agent_wraps_execution_callbacks.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);

    const executionCallback = {
      onNodeStart: vi.fn(),
      onNodeEnd: vi.fn(),
      onNodeError: vi.fn(),
      onNodeSuspend: vi.fn(),
    };
    const agent = app.getAgent({ callbacks: [executionCallback] });
    const callbacks = (agent as unknown as { callbacks: unknown[] }).callbacks;
    expect(callbacks).toHaveLength(1);
    // It should be adapted, not the raw callback.
    expect(callbacks[0]).not.toBe(executionCallback);
  });

  it("passes the prepareAgentInput hook to the adapter", () => {
    // Python: test_get_agent_passes_prepare_input_to_adapter.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);
    const prepare = (): Record<string, unknown> => ({ messages: [] });
    app.prepareAgentInput(prepare);

    const agent = app.getAgent();
    expect((agent as unknown as { prepareInput: unknown }).prepareInput).toBe(
      prepare,
    );
  });

  it("passes the resolveThreadId hook to the adapter", () => {
    // Python: test_get_agent_passes_resolve_thread_id_to_adapter.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);
    const resolve = (): string => "custom-thread";
    app.resolveThreadId(resolve);

    const agent = app.getAgent();
    expect(
      (agent as unknown as { resolveThreadIdFn: unknown }).resolveThreadIdFn,
    ).toBe(resolve);
  });

  it("leaves the resolveThreadId hook null when none is registered", () => {
    // Python: test_get_agent_without_resolve_thread_id_leaves_hook_none.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);

    const agent = app.getAgent();
    expect(
      (agent as unknown as { resolveThreadIdFn: unknown }).resolveThreadIdFn,
    ).toBeNull();
  });

  it("leaves the prepareInput hook null when none is registered", () => {
    // Python: test_get_agent_without_prepare_input_leaves_hook_none.
    const app = new App({ appName: "Test Agent" });
    const mockGraph = { invoke: vi.fn() };
    app.entrypoint(() => mockGraph);

    const agent = app.getAgent();
    expect(
      (agent as unknown as { prepareInput: unknown }).prepareInput,
    ).toBeNull();
  });

  it("registers the langgraph workflow adapter only for the platform checkpointer", () => {
    // Python: test_get_agent_registers_workflow_adapter_only_for_platform_checkpointer.
    clearWorkflowAdapter();

    const eligible = new App({ appName: "Test Agent" });
    eligible.entrypoint(() => ({
      invoke: vi.fn(),
      checkpointer: new PlatformCheckpointer({ native: null }),
    }));
    eligible.getAgent();
    expect(getWorkflowAdapter()?.name).toBe("langgraph");
    // The registered version tracks this package's release (Python parity);
    // the "0.0.0" unknown sentinel must never masquerade as a release.
    expect(getWorkflowAdapter()?.version).toBe(packageVersion());
    // Interleave an ineligible materialization: a build-only registration
    // would leave the first result in place and hide the stale-state bug,
    // so the cached eligible graph must restore the declaration below.
    const nativeOnly = new App({ appName: "Test Agent" });
    nativeOnly.entrypoint(() => ({ invoke: vi.fn() }));
    nativeOnly.getAgent();
    expect(getWorkflowAdapter()).toBeNull();
    eligible.getAgent();
    expect(getWorkflowAdapter()?.name).toBe("langgraph");

    const customSaver = new App({ appName: "Test Agent" });
    customSaver.entrypoint(() => ({
      invoke: vi.fn(),
      checkpointer: { getTuple: vi.fn() },
    }));
    customSaver.getAgent();
    expect(getWorkflowAdapter()).toBeNull();

    clearWorkflowAdapter();
  });
});

// ---------------------------------------------------------------------------
// App.checkpointer()
// ---------------------------------------------------------------------------

describe("App.checkpointer() — mode + URI gating", () => {
  let savedUri: string | undefined;

  beforeEach(() => {
    savedUri = process.env["MONGODB_URI"];
  });

  afterEach(() => {
    if (savedUri === undefined) delete process.env["MONGODB_URI"];
    else process.env["MONGODB_URI"] = savedUri;
  });

  it("returns null in TOOL mode", () => {
    // Python: test_checkpointer_returns_none_outside_aer_mode.
    // TS RUNNER_MODE only recognizes `aer` and `tool` — Python's
    // MEMORY_SERVER mode isn't a TS surface, so we only assert TOOL here.
    withMode("tool", () => {
      const app = new App({ appName: "Test Agent" });
      expect(app.checkpointer()).toBeNull();
    });
  });

  it("returns null in AER mode when MONGODB_URI is unset", () => {
    // Python: test_checkpointer_returns_none_when_aer_mode_but_no_uri.
    delete process.env["MONGODB_URI"];
    withMode("aer", () => {
      const app = new App({ appName: "Test Agent" });
      expect(app.checkpointer()).toBeNull();
    });
  });

  it("wraps and caches the native MongoDB saver in AER mode", async () => {
    process.env["MONGODB_URI"] = "mongodb://localhost:27017";
    const app = new App({ appName: "Test Agent" });
    try {
      const checkpointer = app.checkpointer();
      expect(checkpointer).toBeInstanceOf(PlatformCheckpointer);
      expect(checkpointer?.native).toBeDefined();
      expect(app.checkpointer()).toBe(checkpointer);
    } finally {
      await app.close();
    }
  });

  it("close() is a no-op when no checkpointer was created", async () => {
    // Verifies the cleanup path doesn't reject when there's nothing to close.
    const app = new App({ appName: "Test Agent" });
    await expect(app.close()).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// App — MCP tool registration
// ---------------------------------------------------------------------------

describe("App — MCP tool registration", () => {
  let tmpDir: string;

  beforeEach(() => {
    discoverMcpToolsMock.mockReset();
    tmpDir = mkdtempSync(join(tmpdir(), "mcp-runtime-"));
    // AGENTIC_AGENT_WORKDIR takes priority over cwd in loadRuntimeAgentConfig's
    // candidate search, so this points TenantRuntime at tmpDir's agent.yaml
    // without mutating global process.cwd() (flaky under Vitest's parallel
    // file execution).
    vi.stubEnv("AGENTIC_AGENT_WORKDIR", tmpDir);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  function writeAgentYamlWithMcpServer(): void {
    writeFileSync(
      join(tmpDir, "agent.yaml"),
      [
        "entrypoint: mcp.agent:app",
        "mcp:",
        "  servers:",
        "    github:",
        "      url: https://mcp.example.com/github",
      ].join("\n"),
      "utf-8",
    );
  }

  it("does not call discoverMcpTools when no MCP servers are configured", async () => {
    const app = new App({ appName: "Test Agent" });
    await app.ready();
    expect(discoverMcpToolsMock).not.toHaveBeenCalled();
  });

  it("registers discovered MCP tools as tool definitions and LangChain tools", async () => {
    writeAgentYamlWithMcpServer();
    discoverMcpToolsMock.mockResolvedValue([
      {
        serverName: "github",
        toolName: "search_issues",
        sdkToolName: "github__search_issues",
        description: "Search GitHub issues",
        inputSchema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
        },
        serverConfig: {
          transport: "streamable_http",
          url: "https://mcp.example.com/github",
          headers: {},
          auth: RuntimeMCPAuthConfigSchema.parse({}),
          allowed_tools: null,
          timeout_seconds: 30,
        },
      },
    ]);

    const app = new App({ appName: "Test Agent" });
    await app.ready();

    expect(discoverMcpToolsMock).toHaveBeenCalledOnce();

    const toolDefs = app.getToolDefinitions();
    const mcpDef = toolDefs.find((def) => def.name === "github__search_issues");
    expect(mcpDef).toMatchObject({
      name: "github__search_issues",
      description: "Search GitHub issues",
      remote: true,
    });

    const schemas = app.getToolSchemas() as { name?: string }[];
    expect(schemas.some((tool) => tool.name === "github__search_issues")).toBe(
      true,
    );
  });

  it("forwards mcp_server/mcp_tool call metadata for AER Tool-Pod dispatch", async () => {
    writeAgentYamlWithMcpServer();
    discoverMcpToolsMock.mockResolvedValue([
      {
        serverName: "github",
        toolName: "search_issues",
        sdkToolName: "github__search_issues",
        description: "Search GitHub issues",
        inputSchema: { type: "object", properties: {} },
        serverConfig: {
          transport: "streamable_http",
          url: "https://mcp.example.com/github",
          headers: {},
          auth: RuntimeMCPAuthConfigSchema.parse({}),
          allowed_tools: null,
          timeout_seconds: 30,
        },
      },
    ]);

    // The top-level `beforeEach` defaults RUNNER_MODE to "aer" when unset.
    const app = new App({ appName: "Test Agent" });
    await app.ready();

    const internals = privateOf<{ runtime: TenantRuntime }>(app);
    const metadata = internals.runtime.getToolMetadata("github__search_issues");
    expect(metadata).toMatchObject({
      mcp_server: "github",
      mcp_tool: "search_issues",
    });
  });
});
