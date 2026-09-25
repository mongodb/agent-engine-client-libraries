/**
 * Durable operation-path scoping for Deep Agent `task` dispatch.
 *
 * Port of `test_durable_deep_agent.py`, plus an end-to-end durable attempt
 * that drives a real `createAgentEngineDeepAgent` graph through subagent
 * delegation (the compiled-subgraph suites in durable_nested_paths.test.ts
 * cannot cover the middleware-owned `task` boundary).
 */

import { create } from "@bufbuild/protobuf";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { Runnable } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { Command } from "@langchain/langgraph";
import { z } from "zod";
import {
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  UnsupportedChildOperationFanOutError,
  currentOperationPath,
  currentPendingChildOperationBatch,
  runWithAttemptContext,
  runWithExecutionContext,
  type AttemptContext,
  type CompleteExecutionCommand,
  type FinalizeStepCommand,
} from "@mongodb-js/agent-engine-runner-shared";
import type { AgentInput, RequestContext } from "@mongodb-js/agent-engine-sdk";
import { describe, expect, it, vi } from "vitest";
import type { AnySubAgent } from "deepagents";

import { LangGraphBaseAgent } from "../src/agent.js";
import {
  createDurableDeepAgentMiddleware,
  stampDeepAgentMessageIds,
} from "../src/durable_deep_agent.js";
import { createAgentEngineDeepAgent } from "../src/deep_agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import {
  channelValuesToStateSnapshot,
  stateSnapshotToChannelValues,
} from "../src/workflow_state.js";

type Middleware = ReturnType<typeof createDurableDeepAgentMiddleware>;

type ToolCallLike = {
  name: string;
  id: string;
  args: Record<string, unknown>;
};

interface WrapRequest {
  toolCall: ToolCallLike;
  state: { messages?: BaseMessage[] };
}

function middleware(): Middleware {
  return createDurableDeepAgentMiddleware({});
}

function request(
  overrides: {
    name?: string;
    toolCallId?: string;
    subagentType?: string;
    state?: { messages?: BaseMessage[] };
  } = {},
): WrapRequest {
  return {
    toolCall: {
      name: overrides.name ?? "task",
      id: overrides.toolCallId ?? "task-a",
      args: {
        subagent_type: overrides.subagentType ?? "research",
        description: "research",
      },
    },
    state: overrides.state ?? {},
  };
}

function segments(): [string, number][] {
  return currentOperationPath().segments.map((segment) => [
    segment.name,
    Number(segment.ordinal),
  ]);
}

function resultMessage(req: WrapRequest): ToolMessage {
  return new ToolMessage({
    content: "done",
    tool_call_id: req.toolCall.id,
  });
}

/** Drive the wrapToolCall hook without importing langchain's handler types. */
function wrapToolCall(
  m: Middleware,
  req: WrapRequest,
  handler: (req: WrapRequest) => unknown,
): unknown {
  return m.wrapToolCall?.(req as never, handler as never);
}

function runAfterModel(
  m: Middleware,
  state: { messages?: BaseMessage[] },
): unknown {
  const hook = m.afterModel as unknown as
    | ((state: unknown, runtime: unknown) => unknown)
    | undefined;
  return hook?.(state, undefined);
}

function attemptContext(): AttemptContext {
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

describe("DurableDeepAgentMiddleware operation paths", () => {
  it.each([
    ["native task tool", false, "task"],
    ["durable non-task tool", true, "search_corpus"],
  ] as const)(
    "%s remains at the parent path",
    (_description, durable, toolName) => {
      const m = middleware();
      const seen: [string, number][][] = [];
      const handler = (req: WrapRequest) => {
        seen.push(segments());
        return resultMessage(req);
      };

      if (durable) {
        runWithAttemptContext(attemptContext(), () => {
          wrapToolCall(m, request({ name: toolName }), handler);
        });
      } else {
        wrapToolCall(m, request({ name: toolName }), handler);
      }

      expect(seen).toEqual([[["agent", 1]]]);
    },
  );

  it("scopes task descendants and restores the parent path", () => {
    const m = middleware();
    const seen: [string, number][][] = [];
    const handler = (req: WrapRequest) => {
      seen.push(segments());
      return resultMessage(req);
    };

    runWithAttemptContext(attemptContext(), () => {
      wrapToolCall(m, request(), handler);
      expect(segments()).toEqual([["agent", 1]]);
    });

    expect(seen).toEqual([
      [
        ["agent", 1],
        ["research", 1],
      ],
    ]);
  });

  it("keeps the scope across await and restores it on error", async () => {
    const m = middleware();
    const seen: [string, number][][] = [];
    const handler = async (_req: WrapRequest) => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      seen.push(segments());
      throw new Error("subagent failed");
    };

    await expect(
      runWithAttemptContext(
        attemptContext(),
        () => wrapToolCall(m, request(), handler) as Promise<unknown>,
      ),
    ).rejects.toThrow("subagent failed");
    expect(segments()).toEqual([["agent", 1]]);
    expect(seen).toEqual([
      [
        ["agent", 1],
        ["research", 1],
      ],
    ]);
  });

  it.each([
    ["", "research", "tool call ID"],
    ["task-a", "", "subagent type"],
  ])(
    "requires stable identity (%s, %s)",
    (toolCallId, subagentType, missing) => {
      const m = middleware();
      const handler = (req: WrapRequest) => resultMessage(req);

      expect(() =>
        runWithAttemptContext(attemptContext(), () => {
          wrapToolCall(m, request({ toolCallId, subagentType }), handler);
        }),
      ).toThrow(missing);
    },
  );

  it("rejects a compiled subagent with its own checkpointer on a durable attempt", () => {
    const m = createDurableDeepAgentMiddleware({
      unsupportedSubagentNames: new Set(["research"]),
    });
    let handlerCalled = false;
    const handler = (req: WrapRequest) => {
      handlerCalled = true;
      return resultMessage(req);
    };

    expect(() =>
      runWithAttemptContext(attemptContext(), () => {
        wrapToolCall(m, request(), handler);
      }),
    ).toThrow(/checkpointer=None/);
    expect(handlerCalled).toBe(false);
  });

  it("allows a compiled subagent with its own checkpointer outside durable execution", () => {
    const m = createDurableDeepAgentMiddleware({
      unsupportedSubagentNames: new Set(["research"]),
    });
    const seen: string[] = [];
    const handler = (req: WrapRequest) => {
      seen.push(req.toolCall.id);
      return resultMessage(req);
    };

    wrapToolCall(m, request(), handler);

    expect(seen).toEqual(["task-a"]);
  });
});

describe("DurableDeepAgentMiddleware message ids", () => {
  it("stamps child state and the parent result message on a durable attempt", () => {
    const m = middleware();
    const childMessages = [
      new HumanMessage({ content: "research", id: "random-input" }),
      new AIMessage({ content: "", id: "provider-run" }),
      new ToolMessage({
        content: "updated",
        tool_call_id: "write-todos-a",
        id: "random-tool-result",
      }),
    ];

    let settled: unknown;
    runWithAttemptContext(attemptContext(), () => {
      settled = wrapToolCall(m, request(), (req) => {
        stampDeepAgentMessageIds(childMessages);
        return new Command({
          update: {
            messages: [
              new ToolMessage({
                content: "done",
                tool_call_id: req.toolCall.id,
              }),
            ],
          },
        });
      });
    });

    expect(childMessages.map((message) => message.id)).toEqual([
      "durable-deep-agent-input:task-a:0",
      "provider-run",
      "durable-deep-agent-tool-result:task-a:write-todos-a",
    ]);
    const command = settled as Command;
    const resultMessages = (command.update as { messages: BaseMessage[] })
      .messages;
    expect(resultMessages[0]?.id).toBe("durable-deep-agent-task-result:task-a");
  });

  it("does not rewrite child messages for native task calls", () => {
    const m = middleware();
    const message = new HumanMessage({
      content: "research",
      id: "native-input",
    });

    wrapToolCall(m, request(), (req) => {
      stampDeepAgentMessageIds([message]);
      return resultMessage(req);
    });

    expect(message.id).toBe("native-input");
  });
});

function parentState(...toolCalls: ToolCallLike[]) {
  return {
    messages: [
      new AIMessage({ content: "", tool_calls: toolCalls as never }),
    ] as BaseMessage[],
  };
}

function taskCall(toolCallId: string, subagentType = "research"): ToolCallLike {
  return {
    name: "task",
    id: toolCallId,
    args: { subagent_type: subagentType, description: "research" },
  };
}

describe("DurableDeepAgentMiddleware same-step batches", () => {
  it("keeps message order under inverted start", () => {
    const m = middleware();
    const seen: Record<string, [string, number][]> = {};
    const handler = (req: WrapRequest) => {
      seen[req.toolCall.id] = segments();
      return resultMessage(req);
    };
    const state = parentState(taskCall("task-a"), taskCall("task-b"));

    runWithAttemptContext(attemptContext(), () => {
      runAfterModel(m, state);
      wrapToolCall(m, request({ toolCallId: "task-b", state }), handler);
      wrapToolCall(m, request({ toolCallId: "task-a", state }), handler);
    });

    expect(seen["task-a"]).toEqual([
      ["agent", 1],
      ["research", 1],
    ]);
    expect(seen["task-b"]).toEqual([
      ["agent", 1],
      ["research", 2],
    ]);
  });

  it("preallocates from the wrap state without afterModel", () => {
    const m = middleware();
    const seen: Record<string, [string, number][]> = {};
    const handler = (req: WrapRequest) => {
      seen[req.toolCall.id] = segments();
      return resultMessage(req);
    };
    const state = parentState(taskCall("task-a"), taskCall("task-b"));

    runWithAttemptContext(attemptContext(), () => {
      wrapToolCall(m, request({ toolCallId: "task-b", state }), handler);
      wrapToolCall(m, request({ toolCallId: "task-a", state }), handler);
    });

    expect(seen["task-a"]).toEqual([
      ["agent", 1],
      ["research", 1],
    ]);
    expect(seen["task-b"]).toEqual([
      ["agent", 1],
      ["research", 2],
    ]);
  });

  it("preallocates from the afterModel batch without wrap state", () => {
    const m = middleware();
    const seen: Record<string, [string, number][]> = {};
    const handler = (req: WrapRequest) => {
      seen[req.toolCall.id] = segments();
      return resultMessage(req);
    };

    runWithAttemptContext(attemptContext(), () => {
      runAfterModel(m, parentState(taskCall("task-a"), taskCall("task-b")));
      wrapToolCall(m, request({ toolCallId: "task-b" }), handler);
      wrapToolCall(m, request({ toolCallId: "task-a" }), handler);
    });

    expect(seen["task-a"]).toEqual([
      ["agent", 1],
      ["research", 1],
    ]);
    expect(seen["task-b"]).toEqual([
      ["agent", 1],
      ["research", 2],
    ]);
  });

  it("clears the batch after a successful preallocate", () => {
    const m = middleware();
    const handler = (req: WrapRequest) => resultMessage(req);

    runWithAttemptContext(attemptContext(), () => {
      runAfterModel(m, parentState(taskCall("task-a"), taskCall("task-b")));
      wrapToolCall(m, request({ toolCallId: "task-b" }), handler);
      expect(currentPendingChildOperationBatch()).toEqual([]);
    });
  });

  it("discards the pending batch when the attempt holder is gone", () => {
    const m = middleware();
    runWithAttemptContext(attemptContext(), () => {
      runAfterModel(m, parentState(taskCall("task-a"), taskCall("task-b")));
      expect(currentPendingChildOperationBatch()).not.toEqual([]);
    });
    expect(currentPendingChildOperationBatch()).toEqual([]);
  });

  it("rejects duplicate same-step task ids before entering a child handler", () => {
    const m = middleware();
    let handlerCalled = false;
    const handler = (req: WrapRequest) => {
      handlerCalled = true;
      return resultMessage(req);
    };
    const state = parentState(taskCall("task-a"), taskCall("task-a"));

    expect(() =>
      runWithAttemptContext(attemptContext(), () => {
        runAfterModel(m, state);
        wrapToolCall(m, request({ toolCallId: "task-a", state }), handler);
      }),
    ).toThrow(UnsupportedChildOperationFanOutError);
    expect(handlerCalled).toBe(false);
  });

  it("rejects an unreadable afterModel batch before entering a child handler", () => {
    const m = middleware();

    expect(() =>
      runWithAttemptContext(attemptContext(), () => {
        runAfterModel(m, parentState(taskCall("")));
      }),
    ).toThrow("tool call ID");
  });

  it("rejects an unreadable wrap tool call before entering a child handler", () => {
    const m = middleware();
    let handlerCalled = false;
    const handler = (req: WrapRequest) => {
      handlerCalled = true;
      return resultMessage(req);
    };

    expect(() =>
      runWithAttemptContext(attemptContext(), () => {
        wrapToolCall(m, request({ toolCallId: "" }), handler);
      }),
    ).toThrow("tool call ID");
    expect(handlerCalled).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// End-to-end durable attempt through createAgentEngineDeepAgent
// ---------------------------------------------------------------------------

interface ScriptedTurn {
  message: AIMessage;
  text: string;
}

type TurnScript = (turn: number, received: BaseMessage[]) => ScriptedTurn;

/**
 * Minimal tool-calling fake chat model with per-turn scripts and explicit
 * message ids, so two replay attempts produce identical published state.
 */
class ScriptedToolModel extends BaseChatModel {
  calls = 0;
  private readonly script: TurnScript;

  constructor(script: TurnScript) {
    super({});
    this.script = script;
  }

  reset(): void {
    this.calls = 0;
  }

  _llmType(): string {
    return "scripted-tool-fake";
  }

  override bindTools(): Runnable {
    return this as unknown as Runnable;
  }

  async _generate(
    messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    this.calls += 1;
    const scripted = this.script(this.calls, [...messages]);
    return {
      generations: [{ message: scripted.message, text: scripted.text }],
    };
  }
}

function durableAttempt() {
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
    previousState: channelValuesToStateSnapshot({
      messages: [new AIMessage({ content: "Go", id: "previous-1" })],
    }),
  });
}

describe("deep agent subagent delegation on a durable attempt", () => {
  it("attributes subagent work to the task boundary, stamps message ids, and replays identically", async () => {
    const finalizeStep = vi.fn(async (_command: FinalizeStepCommand) => []);
    const completeExecution = vi.fn(
      async (_command: CompleteExecutionCommand) => undefined,
    );
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: () => ({ finalizeStep, completeExecution }),
    });

    // A plain (non-platform) tool inside the subagent that records the
    // resolved durable operation path at activity-admission time.
    const seenPaths: string[] = [];
    const probe = tool(
      async () => {
        seenPaths.push(
          currentOperationPath()
            .segments.map(
              (segment) => `${segment.name}/${Number(segment.ordinal)}`,
            )
            .join("/"),
        );
        return "probe-ok";
      },
      { name: "probe", schema: z.object({}) },
    );

    const researcherModel = new ScriptedToolModel((turn) => {
      if (turn === 1) {
        return {
          message: new AIMessage({
            content: "",
            tool_calls: [
              {
                id: "call_probe_1",
                name: "probe",
                args: {},
                type: "tool_call",
              },
            ],
            id: "researcher-ai-1",
          }),
          text: "",
        };
      }
      return {
        message: new AIMessage({
          content: "PROBE_DONE",
          id: "researcher-ai-2",
        }),
        text: "PROBE_DONE",
      };
    });
    const researcher = {
      name: "researcher",
      description: "research specialist",
      systemPrompt: "Investigate and report.",
      model: researcherModel,
      tools: [probe],
    } as unknown as AnySubAgent;

    const parentModel = new ScriptedToolModel((turn) => {
      if (turn === 1) {
        return {
          message: new AIMessage({
            content: "",
            tool_calls: [
              {
                id: "call_task_1",
                name: "task",
                args: {
                  subagent_type: "researcher",
                  description: "Investigate the topic.",
                },
                type: "tool_call",
              },
            ],
            id: "parent-ai-1",
          }),
          text: "",
        };
      }
      return {
        message: new AIMessage({
          content: "RESEARCH_DONE",
          id: "parent-ai-2",
        }),
        text: "RESEARCH_DONE",
      };
    });

    const graph = createAgentEngineDeepAgent(
      parentModel as unknown as never,
      undefined,
      {
        subagents: [researcher],
        systemPrompt: "You are an orchestrator. Delegate research.",
        checkpointer: saver as never,
      },
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const ctx = {
      sessionId: "session",
      workspaceId: "workspace",
    } as RequestContext;
    const input = { payload: { message: "Go" } } as AgentInput;

    const run = () => {
      parentModel.reset();
      researcherModel.reset();
      return runWithExecutionContext(
        { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
        () =>
          runWithAttemptContext(durableAttempt(), () =>
            agent.invoke(ctx, input),
          ),
      );
    };
    await run();
    await run();

    expect(completeExecution).toHaveBeenCalledTimes(2);
    // The probe ran inside the subagent and attributed to the task boundary.
    expect(seenPaths).toEqual(["agent/1/researcher/1", "agent/1/researcher/1"]);

    const published = [0, 1].map((index) => {
      const call = completeExecution.mock.calls[index]?.[0];
      if (call === undefined || call.state === undefined) {
        throw new Error("missing completeExecution call state");
      }
      return (
        stateSnapshotToChannelValues(call.state)["messages"] as BaseMessage[]
      ).map((message) => [
        message.getType(),
        String(message.content),
        message.id,
      ]);
    });
    expect(published[0]).toEqual([
      ["ai", "Go", "previous-1"],
      ["human", "Go", "durable-input:execution"],
      ["ai", "", "parent-ai-1"],
      ["tool", "PROBE_DONE", "durable-deep-agent-task-result:call_task_1"],
      ["ai", "RESEARCH_DONE", "parent-ai-2"],
    ]);
    // A replacement attempt rebuilds the identical message list.
    expect(published[1]).toEqual(published[0]);
    expect(finalizeStep).toHaveBeenCalled();
  });

  it("shares the platform scratch with the deep-agent checkpointer view", async () => {
    const finalizeStep = vi.fn(async (_command: FinalizeStepCommand) => []);
    const completeExecution = vi.fn(
      async (_command: CompleteExecutionCommand) => undefined,
    );
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: () => ({ finalizeStep, completeExecution }),
    });

    const researcher = {
      name: "researcher",
      description: "research specialist",
      systemPrompt: "Investigate and report.",
      model: new ScriptedToolModel(() => ({
        message: new AIMessage({
          content: "PROBE_DONE",
          id: "researcher-ai-1",
        }),
        text: "PROBE_DONE",
      })),
    } as unknown as AnySubAgent;
    const parentModel = new ScriptedToolModel(() => ({
      message: new AIMessage({ content: "RESEARCH_DONE", id: "parent-ai-1" }),
      text: "RESEARCH_DONE",
    }));

    const graph = createAgentEngineDeepAgent(
      parentModel as unknown as never,
      undefined,
      {
        subagents: [researcher],
        checkpointer: saver as never,
      },
    );
    const agent = new LangGraphBaseAgent(graph as never);
    const ctx = {
      sessionId: "session",
      workspaceId: "workspace",
    } as RequestContext;
    const input = { payload: { message: "Go" } } as AgentInput;
    const attempt = durableAttempt();

    await runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () => runWithAttemptContext(attempt, () => agent.invoke(ctx, input)),
    );
    expect(completeExecution).toHaveBeenCalledOnce();

    // close() releases the platform scratch, so the deep-agent writes must
    // have landed on the SAME scratch the platform view owns.
    await saver.releaseScratch(attempt);
    await runWithAttemptContext(attempt, async () => {
      expect(
        await saver.getTuple({ configurable: { thread_id: "caller" } }),
      ).toBeUndefined();
    });
  });
});
