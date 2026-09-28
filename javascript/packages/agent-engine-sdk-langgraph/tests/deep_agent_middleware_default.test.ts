/**
 * `createAgentEngineDeepAgent` registers `DurableDeepAgentMiddleware` by
 * default so durable task dispatch gets deterministic attribution. A stopped
 * tool call deliberately gets no default short-circuit: the batch reaches the
 * model as ordinary interrupted ToolMessages, matching the Python twin.
 * Mirrors Python's test_factory_sdk.py coverage.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { create } from "@bufbuild/protobuf";
import {
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
  runWithExecutionContext,
  type AttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";
import type * as Deepagents from "deepagents";

const { createDeepAgent } = vi.hoisted(() => ({
  createDeepAgent: vi.fn().mockReturnValue({}),
}));
vi.mock("deepagents", async (importOriginal) => {
  const actual = await importOriginal<typeof Deepagents>();
  return { ...actual, createDeepAgent };
});

import { createAgentEngineDeepAgent } from "../src/deep_agent.js";

function model(): ChatOpenAI {
  return new ChatOpenAI({ model: "gpt-4o", apiKey: "test-key" });
}

function attempt(): AttemptContext {
  return create(AttemptContextSchema, {
    attemptId: "attempt-1",
    fencingToken: 7n,
    replayMode: false,
    workflowIdentity: create(WorkflowIdentitySchema, {
      tenantScope: create(TenantScopeSchema, {
        orgId: "org",
        projectId: "project",
        workspaceId: "workspace",
      }),
      sessionId: "session",
      executionId: "execution",
    }),
  });
}

/** The default durable middleware instance captured from the factory call. */
function durableMiddleware(): Record<string, unknown> {
  const [call] = createDeepAgent.mock.calls;
  if (call === undefined) throw new Error("createDeepAgent was not called");
  const middleware = call[0].middleware as Record<string, unknown>[];
  const instance = middleware.find(
    (m) => m.name === "durableDeepAgentMiddleware",
  );
  if (instance === undefined) {
    throw new Error("durable middleware was not wired by the factory");
  }
  return instance;
}

describe("createAgentEngineDeepAgent middleware defaults", () => {
  beforeEach(() => {
    createDeepAgent.mockClear();
  });

  it("registers only the durable middleware when no middleware is supplied", () => {
    createAgentEngineDeepAgent(model(), undefined);
    expect(createDeepAgent).toHaveBeenCalledTimes(1);
    const [call] = createDeepAgent.mock.calls;
    if (call === undefined) throw new Error("createDeepAgent was not called");
    const { middleware } = call[0];
    expect(middleware).toHaveLength(1);
    expect(middleware[0]).toMatchObject({
      name: "durableDeepAgentMiddleware",
    });
  });

  it("keeps the default middleware first, ahead of caller-supplied middleware", () => {
    const custom = { name: "custom" };
    createAgentEngineDeepAgent(model(), undefined, {
      middleware: [custom as never],
    });
    expect(createDeepAgent).toHaveBeenCalledTimes(1);
    const [call] = createDeepAgent.mock.calls;
    if (call === undefined) throw new Error("createDeepAgent was not called");
    const { middleware } = call[0];
    expect(middleware).toHaveLength(2);
    expect(middleware[1]).toBe(custom);
  });

  it("rejects a durable task dispatch for a compiled subagent with its own checkpointer", async () => {
    // Factory-level regression: the guard lives in the derived
    // unsupportedSubagentNames wiring, not in the middleware itself, so the
    // manually-constructed middleware tests cannot see a wiring regression.
    createAgentEngineDeepAgent(model(), undefined, {
      subagents: [
        {
          name: "research",
          description: "research specialist",
          runnable: { checkpointer: new MemorySaver() },
        } as never,
      ],
    });

    let handlerCalled = false;
    await expect(
      runWithExecutionContext(
        { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
        () =>
          runWithAttemptContext(attempt(), async () =>
            (
              durableMiddleware().wrapToolCall as (
                request: never,
                handler: never,
              ) => Promise<unknown>
            )(
              {
                toolCall: {
                  name: "task",
                  id: "call_task_1",
                  args: { subagent_type: "research", description: "go" },
                },
                state: {},
              } as never,
              (() => {
                handlerCalled = true;
                return { content: "done", tool_call_id: "call_task_1" };
              }) as never,
            ),
          ),
      ),
    ).rejects.toThrow(/checkpointer=None/);
    expect(handlerCalled).toBe(false);
  });
});
