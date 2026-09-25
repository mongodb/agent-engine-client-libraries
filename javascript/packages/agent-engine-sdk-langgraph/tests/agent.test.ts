/**
 * Port of `agent-engine-sdk-langgraph/tests/test_agent.py`.
 *
 * Covers LangGraphBaseAgent and native execution-session behavior — init,
 * input/config preparation, invoke, invoke_suspended, resume, stream tokens,
 * stream empty-messages, stream first-interrupt, stream resume-without-data,
 * checkpoint-id fallback on error, Overwrite unwrap, Command update with/without
 * messages, and the standard add_messages regression.
 *
 * The Python tests use `mock_graph.astream(...)`; TS uses an awaited stream
 * function returning an `AsyncIterable`. Helper `mockGraph(...)` adapts both.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  Command,
  END,
  MemorySaver,
  MessagesAnnotation,
  Overwrite,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
import type {
  AgentInput,
  AgentOutput,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";

import { LangGraphBaseAgent } from "../src/agent.js";
import { copyCheckpoint } from "../src/checkpoint_branch.js";
import type { ExecutionSession } from "../src/execution_session.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) {
    process.env["RUNNER_MODE"] = "aer";
  }
});

interface MockGraph {
  invoke: ReturnType<typeof vi.fn>;
  getState: ReturnType<typeof vi.fn>;
  stream: ReturnType<typeof vi.fn>;
  checkpointer: unknown;
}

function makeMockGraph(): MockGraph {
  return {
    invoke: vi.fn(),
    getState: vi.fn().mockResolvedValue({ next: [] }),
    stream: vi.fn(),
    checkpointer: null,
  };
}

function sampleInput(extra: Record<string, unknown> = {}): AgentInput {
  return {
    payload: {
      message: "Hello, world!",
      user_id: "user-123",
      session_id: "session-456",
      thread_id: "thread-789",
      ...extra,
    },
  };
}

async function* yieldFrom(
  events: ReadonlyArray<readonly [readonly string[], string, unknown]>,
): AsyncIterable<unknown> {
  for (const event of events) yield event;
}

async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const v of iter) out.push(v);
  return out;
}

// LangGraphBaseAgent's private helpers — accessed via plain cast (intersection
// with a class that has private fields collapses to `never` in TS, so we use
// an unrelated structural shape instead).
interface AgentPrivate {
  callbacks: readonly unknown[];
  session(ctx: RequestContext, input: AgentInput): ExecutionSession;
}

function asPrivate(agent: LangGraphBaseAgent): AgentPrivate {
  return agent as unknown as AgentPrivate;
}

async function preparedRun(
  agent: LangGraphBaseAgent,
  ctx: RequestContext,
  input: AgentInput = sampleInput(),
) {
  return asPrivate(agent).session(ctx, input).prepareRun();
}

// ---------------------------------------------------------------------------
// LangGraphBaseAgent — construction
// ---------------------------------------------------------------------------

describe("LangGraphBaseAgent — construction", () => {
  it("stores callbacks when provided", () => {
    // Python: test_init_with_callbacks.
    const callbacks = [{ cb: 1 }, { cb: 2 }];
    const agent = new LangGraphBaseAgent(makeMockGraph() as never, callbacks);
    expect(asPrivate(agent).callbacks).toEqual(callbacks);
  });

  it("defaults callbacks to an empty list", () => {
    // Python: test_init_without_callbacks.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    expect(asPrivate(agent).callbacks).toEqual([]);
  });
});

describe("LangGraphBaseAgent — LangGraph checkpoint branching", () => {
  function makeAgent() {
    const builder = makeBuilder();
    const graph = builder.compile({ checkpointer: new MemorySaver() });
    return { agent: new LangGraphBaseAgent(graph as never), graph };
  }

  function makeBuilder() {
    return new StateGraph(MessagesAnnotation)
      .addNode("respond", async (state) => {
        const human = [...state.messages]
          .reverse()
          .find((message) => message instanceof HumanMessage);
        if (human === undefined) throw new Error("missing human message");
        return {
          messages: [new AIMessage({ content: `reply:${human.content}` })],
        };
      })
      .addEdge(START, "respond")
      .addEdge("respond", END);
  }

  function makeInterruptingAgent() {
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", async (state) => {
        const human = [...state.messages]
          .reverse()
          .find((message) => message instanceof HumanMessage);
        if (human === undefined) throw new Error("missing human message");
        if (human.content === "pause") {
          const decision = interrupt({ question: "continue?" });
          return {
            messages: [new AIMessage({ content: `reply:${String(decision)}` })],
          };
        }
        return {
          messages: [new AIMessage({ content: `reply:${human.content}` })],
        };
      })
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: new MemorySaver() });
    return { agent: new LangGraphBaseAgent(graph as never), graph };
  }

  it("copies the selected checkpoint into the new session thread", async () => {
    const { agent, graph } = makeAgent();
    const sourceCtx: RequestContext = {
      sessionId: "source",
      workspaceId: "workspace",
    };
    const first = await agent.invoke(sourceCtx, {
      payload: { message: "one" },
    });
    const firstResponse = first.response as Record<string, unknown>;
    const firstMetadata = firstResponse["metadata"] as Record<string, unknown>;
    const source = firstMetadata["langgraph_checkpoint"] as Record<
      string,
      string
    >;
    const sourceThreadId = source["thread_id"];
    const sourceCheckpointId = source["checkpoint_id"];
    if (sourceThreadId === undefined || sourceCheckpointId === undefined) {
      throw new Error("source checkpoint metadata is missing");
    }

    await agent.invoke(sourceCtx, { payload: { message: "two" } });

    const branch = await agent.invoke(
      {
        sessionId: "branch",
        workspaceId: "workspace",
        metadata: {
          langgraph_branch_point: {
            thread_id: sourceThreadId,
            checkpoint_id: sourceCheckpointId,
          },
        },
      },
      { payload: { message: "branch" } },
    );
    const branchMetadata = (branch.response as Record<string, unknown>)[
      "metadata"
    ] as Record<string, unknown>;
    const branchCheckpoint = branchMetadata["langgraph_checkpoint"] as Record<
      string,
      string
    >;
    expect(branchCheckpoint["thread_id"]).toBe("branch:workspace");

    const branchState = await graph.getState({
      configurable: { thread_id: "branch:workspace" },
    });
    expect(
      (branchState.values.messages as BaseMessage[]).map(
        (message) => message.content,
      ),
    ).toEqual(["one", "reply:one", "branch", "reply:branch"]);
    const sourceState = await graph.getState({
      configurable: { thread_id: "source:workspace" },
    });
    expect(
      (sourceState.values.messages as BaseMessage[]).map(
        (message) => message.content,
      ),
    ).toEqual(["one", "reply:one", "two", "reply:two"]);
  });

  it("leaves an initialized destination intact instead of copying again", async () => {
    const { agent, graph } = makeAgent();
    const source = await agent.invoke(
      { sessionId: "source", workspaceId: "workspace" },
      { payload: { message: "one" } },
    );
    const sourceCheckpoint = (
      (source.response as Record<string, unknown>)["metadata"] as Record<
        string,
        unknown
      >
    )["langgraph_checkpoint"] as Record<string, string>;
    const branchCtx: RequestContext = {
      sessionId: "branch",
      workspaceId: "workspace",
      metadata: { langgraph_branch_point: sourceCheckpoint },
    };
    await agent.invoke(branchCtx, { payload: { message: "branch" } });
    const before = await graph.getState({
      configurable: { thread_id: "branch:workspace" },
    });

    // OE dispatches every later dest invoke with the source branch point;
    // the in-app fork already materialized dest, so the copy is skipped and
    // the turn runs normally (Python initialize_checkpoint_branch parity).
    await agent.invoke(branchCtx, { payload: { message: "do not retry" } });

    const after = await graph.getState({
      configurable: { thread_id: "branch:workspace" },
    });
    const beforeMessages = ((before.values as Record<string, unknown>)[
      "messages"
    ] ?? []) as unknown[];
    const afterMessages = ((after.values as Record<string, unknown>)[
      "messages"
    ] ?? []) as unknown[];
    expect(afterMessages.length).toBeGreaterThan(beforeMessages.length);
  });

  it("resumes an initialized branch without copying it again", async () => {
    const { agent } = makeInterruptingAgent();
    const source = await agent.invoke(
      { sessionId: "source", workspaceId: "workspace" },
      { payload: { message: "source" } },
    );
    const sourceCheckpoint = (
      (source.response as Record<string, unknown>)["metadata"] as Record<
        string,
        unknown
      >
    )["langgraph_checkpoint"] as Record<string, string>;
    const metadata = { langgraph_branch_point: sourceCheckpoint };
    const suspended = await agent.invoke(
      { sessionId: "branch", workspaceId: "workspace", metadata },
      { payload: { message: "pause" } },
    );
    expect((suspended.response as Record<string, unknown>)["status"]).toBe(
      "suspended",
    );

    const resumed = await agent.invoke(
      {
        sessionId: "branch",
        workspaceId: "workspace",
        metadata,
        resume: true,
        resumeData: "approved",
      },
      { payload: { message: "" } },
    );
    expect((resumed.response as Record<string, unknown>)["status"]).toBe(
      "completed",
    );
    expect((resumed.response as Record<string, unknown>)["response"]).toBe(
      "reply:approved",
    );
  });

  it("copies a completed source with a historical interrupt", async () => {
    const { agent, graph } = makeInterruptingAgent();
    const sourceCtx: RequestContext = {
      sessionId: "source",
      workspaceId: "workspace",
    };
    const suspended = await agent.invoke(sourceCtx, {
      payload: { message: "pause" },
    });
    expect((suspended.response as Record<string, unknown>)["status"]).toBe(
      "suspended",
    );
    const completed = await agent.invoke(
      { ...sourceCtx, resume: true, resumeData: "approved" },
      { payload: { message: "" } },
    );
    const sourceCheckpoint = (
      (completed.response as Record<string, unknown>)["metadata"] as Record<
        string,
        unknown
      >
    )["langgraph_checkpoint"] as Record<string, string>;

    const branch = await agent.invoke(
      {
        sessionId: "branch",
        workspaceId: "workspace",
        metadata: { langgraph_branch_point: sourceCheckpoint },
      },
      { payload: { message: "branch" } },
    );

    expect((branch.response as Record<string, unknown>)["status"]).toBe(
      "completed",
    );
    const branchState = await graph.getState({
      configurable: { thread_id: "branch:workspace" },
    });
    expect(
      (branchState.values.messages as BaseMessage[]).map(
        (message) => message.content,
      ),
    ).toEqual(["pause", "reply:approved", "branch", "reply:branch"]);
  });

  it.each([
    { error: "failed", state: undefined, message: "failed history" },
    {
      error: undefined,
      state: { configurable: {} },
      message: "persistent subgraph history",
    },
  ])("rejects $message before copying", async ({ error, state, message }) => {
    const source = {
      threadId: "source:workspace",
      checkpointId: "source-checkpoint",
    };
    const sourceConfig: RunnableConfig = {
      configurable: {
        thread_id: source.threadId,
        checkpoint_ns: "",
        checkpoint_id: source.checkpointId,
      },
    };
    const parentConfig: RunnableConfig = {
      configurable: {
        thread_id: source.threadId,
        checkpoint_ns: "",
        checkpoint_id: "parent-checkpoint",
      },
    };
    const sourceValues = { messages: ["source"] };
    const selected = {
      config: sourceConfig,
      parentConfig,
      next: [],
      tasks: [],
      values: sourceValues,
    };
    const parent = {
      config: parentConfig,
      parentConfig: undefined,
      next: [],
      tasks: [
        {
          id: "task-1",
          name: "respond",
          error,
          state,
          interrupts: [],
          result: undefined,
        },
      ],
      values: { messages: ["parent"] },
    };
    async function* emptyList(): AsyncIterable<never> {}
    const bulkUpdateState = vi.fn();
    const graph = {
      checkpointer: {
        list: vi.fn(emptyList),
        getTuple: vi.fn().mockResolvedValue({ pendingWrites: [] }),
      },
      getState: vi
        .fn()
        .mockResolvedValueOnce(selected)
        .mockResolvedValueOnce(selected)
        .mockResolvedValueOnce(parent),
      bulkUpdateState,
    };

    await expect(
      copyCheckpoint(
        graph as never,
        { configurable: { thread_id: "branch:workspace" } },
        source,
      ),
    ).rejects.toThrow(message);

    expect(bulkUpdateState).not.toHaveBeenCalled();
    expect(selected.values).toEqual(sourceValues);
  });
});

// ---------------------------------------------------------------------------
// buildGraphInput
// ---------------------------------------------------------------------------

describe("native execution-session input", () => {
  it("constructs a fresh-execution payload with a HumanMessage", async () => {
    // Python: test_build_graph_input_fresh. Identity comes from ctx.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { graphInput } = await preparedRun(
      agent,
      { userId: "user-1", sessionId: "session-1" } as RequestContext,
      { payload: { message: "Hello" } },
    );
    const result = graphInput as Record<string, unknown>;

    const msgs = result["messages"] as BaseMessage[];
    expect(msgs).toHaveLength(1);
    expect(msgs[0]).toBeInstanceOf(HumanMessage);
    expect(msgs[0]?.content).toBe("Hello");
    expect(result["user_id"]).toBe("user-1");
    expect(result["session_id"]).toBe("session-1");
  });

  it("constructs a Command(resume=...) when ctx.resume is true", async () => {
    // Python: test_build_graph_input_resume_via_ctx. Resume state lives on ctx.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { graphInput: result } = await preparedRun(
      agent,
      {
        resume: true,
        resumeData: { decision: "approved" },
      } as RequestContext,
      { payload: { message: "" } },
    );

    expect(result).toBeInstanceOf(Command);
    expect((result as unknown as { resume: unknown }).resume).toEqual({
      decision: "approved",
    });
  });

  it("delegates fresh-execution input to the prepareInput hook when registered", async () => {
    // Python: test_build_graph_input_uses_prepare_input_hook.
    const sentinel = { custom: "graph-state" };
    const prepare = vi.fn(() => sentinel);
    const agent = new LangGraphBaseAgent(makeMockGraph() as never, [], prepare);
    const ctx = { executionId: "inv-1" } as RequestContext;
    const input: AgentInput = { payload: { message: "Hello" } };

    const { graphInput: result } = await preparedRun(agent, ctx, input);

    expect(result).toBe(sentinel);
    expect(prepare).toHaveBeenCalledOnce();
    expect(prepare).toHaveBeenCalledWith(input, ctx);
  });

  it("does not call the prepareInput hook on resume (resume stays platform-managed)", async () => {
    // Resume is built by the adapter before the hook is consulted.
    const prepare = vi.fn(() => ({ custom: true }));
    const agent = new LangGraphBaseAgent(makeMockGraph() as never, [], prepare);

    const { graphInput: result } = await preparedRun(
      agent,
      {
        resume: true,
        resumeData: { decision: "approved" },
      } as RequestContext,
      { payload: {} },
    );

    expect(result).toBeInstanceOf(Command);
    expect(prepare).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// buildConfig
// ---------------------------------------------------------------------------

describe("native execution-session config", () => {
  it("includes thread_id in configurable", async () => {
    // Python: test_build_config. session_id is the checkpoint thread id.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { config } = await preparedRun(agent, {
      sessionId: "thread-123",
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["thread_id"],
    ).toBe("thread-123");
  });

  it("scopes thread_id to workspace", async () => {
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { config } = await preparedRun(agent, {
      sessionId: "sess-1",
      workspaceId: "ws-1",
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["thread_id"],
    ).toBe("sess-1:ws-1");
  });

  it("isolates the same session across workspaces", async () => {
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { config: a } = await preparedRun(agent, {
      sessionId: "sess-1",
      workspaceId: "ws-a",
    } as RequestContext);
    const { config: b } = await preparedRun(agent, {
      sessionId: "sess-1",
      workspaceId: "ws-b",
    } as RequestContext);
    expect(
      (a["configurable"] as Record<string, unknown>)["thread_id"],
    ).not.toBe((b["configurable"] as Record<string, unknown>)["thread_id"]);
  });

  it("propagates a non-null checkpoint_id from ctx.metadata", async () => {
    // Python: test_build_config_with_checkpoint_in_metadata.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { config } = await preparedRun(agent, {
      sessionId: "thread-123",
      metadata: { checkpoint_id: "ckpt-456" },
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["checkpoint_id"],
    ).toBe("ckpt-456");
  });

  it("ignores a null checkpoint_id so latest-checkpoint lookup proceeds", async () => {
    // Python: test_build_config_ignores_none_checkpoint_value.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const { config } = await preparedRun(agent, {
      sessionId: "thread-123",
      metadata: { checkpoint_id: null },
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["checkpoint_id"],
    ).toBeUndefined();
  });

  it("attaches callbacks when the agent was constructed with any", async () => {
    // Python: test_build_config_includes_callbacks.
    const callbacks = [{ id: "cb-1" }];
    const agent = new LangGraphBaseAgent(makeMockGraph() as never, callbacks);
    const { config } = await preparedRun(agent, {} as RequestContext);
    expect(config["callbacks"]).toEqual(callbacks);
  });

  it("uses a resolveThreadId hook verbatim, skipping workspace scoping", async () => {
    // Python: test_build_config_resolve_thread_id_hook_overrides_default.
    const agent = new LangGraphBaseAgent(
      makeMockGraph() as never,
      undefined,
      null,
      (ctx) => `${ctx.sessionId}__actor`,
    );
    const { config } = await preparedRun(agent, {
      sessionId: "sess-1",
      workspaceId: "ws-1",
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["thread_id"],
    ).toBe("sess-1__actor");
  });

  it("recomputes resolveThreadId on resume", async () => {
    // Python: test_build_config_resolve_thread_id_hook_used_on_resume.
    const agent = new LangGraphBaseAgent(
      makeMockGraph() as never,
      undefined,
      null,
      (ctx) => `${ctx.sessionId}__actor`,
    );
    const { config } = await preparedRun(agent, {
      sessionId: "sess-1",
      workspaceId: "ws-1",
      resume: true,
      resumeData: { decision: "approved" },
    } as RequestContext);
    expect(
      (config["configurable"] as Record<string, unknown>)["thread_id"],
    ).toBe("sess-1__actor");
  });

  it("rejects an empty resolveThreadId return value", async () => {
    // Python: test_build_config_resolve_thread_id_rejects_empty.
    const agent = new LangGraphBaseAgent(
      makeMockGraph() as never,
      undefined,
      null,
      () => "   ",
    );
    await expect(
      preparedRun(agent, { sessionId: "sess-1" } as RequestContext),
    ).rejects.toThrow(/resolveThreadId must return a non-empty string/);
  });

  it("rejects a non-string resolveThreadId return value", async () => {
    // Python: test_build_config_resolve_thread_id_rejects_non_string.
    const agent = new LangGraphBaseAgent(
      makeMockGraph() as never,
      undefined,
      null,
      () => 42 as unknown as string,
    );
    await expect(
      preparedRun(agent, { sessionId: "sess-1" } as RequestContext),
    ).rejects.toThrow(/resolveThreadId must return a non-empty string/);
  });

  it("names resolveThreadId even when the value is not JSON-serializable", async () => {
    const agent = new LangGraphBaseAgent(
      makeMockGraph() as never,
      undefined,
      null,
      () => 1n as unknown as string,
    );
    await expect(
      preparedRun(agent, { sessionId: "sess-1" } as RequestContext),
    ).rejects.toThrow(
      /App\.resolveThreadId must return a non-empty string; got bigint/,
    );
  });

  it("injects RequestContext into configurable.request_context", async () => {
    // Python: test_build_config_injects_request_context.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const ctx: RequestContext = { executionId: "inv-1", userId: "u-1" };
    const { config } = await preparedRun(agent, ctx);
    expect(
      (config["configurable"] as Record<string, unknown>)["request_context"],
    ).toBe(ctx);
  });
});

// ---------------------------------------------------------------------------
// invoke / invoke_suspended / resume
// ---------------------------------------------------------------------------

describe("LangGraphBaseAgent.invoke", () => {
  it("returns response='Hi there!' and status='completed'", async () => {
    // Python: test_invoke.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [
        new HumanMessage({ content: "Hello" }),
        new AIMessage({ content: "Hi there!" }),
      ],
    });
    graph.getState.mockResolvedValue({ next: [] });

    const agent = new LangGraphBaseAgent(graph as never);
    const result = (await agent.invoke(
      {} as RequestContext,
      sampleInput(),
    )) as AgentOutput;

    const resp = result.response as Record<string, unknown>;
    expect(resp["response"]).toBe("Hi there!");
    expect(resp["status"]).toBe("completed");
    expect(graph.invoke).toHaveBeenCalledOnce();
  });

  it("extracts response from AIMessageChunk when provider returns chunk instead of AIMessage", async () => {
    // Regression: LangChain JS AIMessageChunk is not a subclass of AIMessage
    // (unlike Python), so instanceof AIMessage would silently return "" for
    // providers that aggregate stream chunks via _streamResponseChunks.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [
        new HumanMessage({ content: "Hello" }),
        new AIMessageChunk({ content: "Hi from chunk!" }),
      ],
    });
    graph.getState.mockResolvedValue({ next: [] });

    const agent = new LangGraphBaseAgent(graph as never);
    const result = (await agent.invoke(
      {} as RequestContext,
      sampleInput(),
    )) as AgentOutput;

    const resp = result.response as Record<string, unknown>;
    expect(resp["response"]).toBe("Hi from chunk!");
    expect(resp["status"]).toBe("completed");
  });

  it("returns status='suspended' with interrupt_values when interrupt fires", async () => {
    // Python: test_invoke_suspended.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [
        new HumanMessage({ content: "Hello" }),
        new AIMessage({ content: "Checking..." }),
      ],
    });
    graph.getState.mockResolvedValue({
      next: ["some_node"],
      tasks: [{ interrupts: [{ value: { reason: "approval_needed" } }] }],
    });

    const agent = new LangGraphBaseAgent(graph as never);
    const result = (await agent.invoke(
      {} as RequestContext,
      sampleInput(),
    )) as AgentOutput;

    const resp = result.response as Record<string, unknown>;
    expect(resp["status"]).toBe("suspended");
    expect(resp["interrupts"]).toBeUndefined();
    expect(resp["resume_schema"]).toBeUndefined();
    const sc = resp["suspend_context"] as Record<string, unknown>;
    expect(sc["interrupt_values"]).toEqual([{ reason: "approval_needed" }]);
  });

  it("rejects a non-serializable interrupt value from invoke", async () => {
    // Python: test_invoke_rejects_non_serializable_interrupt_value.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({ messages: [] });
    graph.getState.mockResolvedValue({
      next: ["some_node"],
      tasks: [{ interrupts: [{ id: "approval", value: 1n }] }],
    });

    const agent = new LangGraphBaseAgent(graph as never);
    await expect(
      agent.invoke({} as RequestContext, sampleInput()),
    ).rejects.toThrow(
      'LangGraph interrupt "approval" has a non-JSON-serializable value of type bigint',
    );
  });

  it("passes resolveThreadId into graph config and execution_id", async () => {
    // Python: test_invoke_uses_resolve_thread_id_for_graph_and_execution_id.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [new AIMessage({ content: "done" })],
    });
    graph.getState.mockResolvedValue({ next: [] });

    const agent = new LangGraphBaseAgent(
      graph as never,
      undefined,
      null,
      (ctx) => `${ctx.sessionId}__actor`,
    );
    const result = (await agent.invoke(
      { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext,
      sampleInput(),
    )) as AgentOutput;

    const invokeConfig = graph.invoke.mock.calls[0]?.[1] as {
      configurable: Record<string, unknown>;
    };
    expect(invokeConfig.configurable["thread_id"]).toBe("sess-1__actor");
    expect((result.response as Record<string, unknown>)["execution_id"]).toBe(
      "sess-1__actor",
    );
  });

  it("looks up checkpoints with resolveThreadId on suspend", async () => {
    // Python: test_invoke_suspended_looks_up_checkpoint_with_resolve_thread_id.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [new AIMessage({ content: "Checking..." })],
    });
    graph.getState.mockResolvedValue({
      next: ["some_node"],
      tasks: [{ interrupts: [{ value: { reason: "approval_needed" } }] }],
    });
    const getTuple = vi.fn().mockResolvedValue({
      checkpoint: { id: "ckpt-custom" },
    });
    graph.checkpointer = { getTuple };

    const agent = new LangGraphBaseAgent(
      graph as never,
      undefined,
      null,
      (ctx) => `${ctx.sessionId}__actor`,
    );
    const result = (await agent.invoke(
      { sessionId: "sess-1", workspaceId: "ws-1" } as RequestContext,
      sampleInput(),
    )) as AgentOutput;

    expect(getTuple).toHaveBeenCalledOnce();
    const lookupConfig = getTuple.mock.calls[0]?.[0] as {
      configurable: Record<string, unknown>;
    };
    expect(lookupConfig.configurable["thread_id"]).toBe("sess-1__actor");
    const resp = result.response as Record<string, unknown>;
    expect(resp["execution_id"]).toBe("sess-1__actor");
    expect(
      (resp["suspend_context"] as Record<string, unknown>)["checkpoint_id"],
    ).toBe("ckpt-custom");
  });
});

describe("LangGraphBaseAgent.resume", () => {
  it("delegates to invoke with resumed=true", async () => {
    // Python: test_resume.
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [new AIMessage({ content: "Approved and done" })],
    });
    graph.getState.mockResolvedValue({ next: [] });

    const agent = new LangGraphBaseAgent(graph as never);
    const result = (await agent.resume(
      {} as RequestContext,
      sampleInput(),
      "approved",
    )) as AgentOutput;

    const resp = result.response as Record<string, unknown>;
    expect(resp["response"]).toBe("Approved and done");
    expect(resp["status"]).toBe("completed");
    expect(resp["resumed"]).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// stream — happy paths
// ---------------------------------------------------------------------------

describe("LangGraphBaseAgent.stream — tokens", () => {
  it("yields token events with empty source for root-agent tokens and a final result", async () => {
    // Python: test_stream_tokens.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Hello" }), {}]],
        [[], "messages", [new AIMessageChunk({ content: " world" }), {}]],
        [
          [],
          "updates",
          { node: { messages: [new AIMessage({ content: "Hello world" })] } },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const events: StreamEvent[] = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(3);
    expect(events[0]?.event).toBe("token");
    expect(events[0]?.data).toEqual({
      content: "Hello",
      source: "",
      tool_call_id: "",
    });
    expect(events[1]?.event).toBe("token");
    expect(events[2]?.event).toBe("result");
    expect((events[2]?.data as { response: string }).response).toBe(
      "Hello world",
    );
  });
});

describe("LangGraphBaseAgent.stream — HITL suspend", () => {
  it("includes checkpoint_id in the suspend event metadata when the checkpointer resolves it", async () => {
    // Python: test_stream_suspend_includes_checkpoint_id.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Checking..." }), {}]],
        [
          [],
          "updates",
          {
            __interrupt__: [
              {
                id: "approval",
                value: {
                  suspend_reason: "Need approval",
                  suspend_context: { amount: 1000 },
                },
              },
            ],
            agent: {
              messages: [new AIMessage({ content: "Waiting for approval" })],
            },
          },
        ],
      ]),
    );
    graph.checkpointer = {
      getTuple: vi.fn().mockResolvedValue({ checkpoint: { id: "ckpt-789" } }),
    };

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(2);
    expect(events[0]?.event).toBe("token");
    expect(events[1]?.event).toBe("suspend");
    const d = events[1]?.data as Record<string, unknown>;
    expect(
      (d["suspend_payload"] as Record<string, unknown>)["suspend_reason"],
    ).toBe("Need approval");
    // Resume state travels in the opaque metadata dict, never top-level.
    expect(d["checkpoint_id"]).toBeUndefined();
    expect(d["metadata"]).toEqual({ checkpoint_id: "ckpt-789" });
    expect(d["interrupts"]).toEqual([
      {
        id: "approval",
        value: {
          suspend_reason: "Need approval",
          suspend_context: { amount: 1000 },
        },
      },
    ]);
    expect(d["messages"]).toEqual([
      expect.objectContaining({
        role: "assistant",
        content: "Waiting for approval",
      }),
    ]);
  });

  it("reports the full pending interrupt snapshot", async () => {
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [
          [],
          "updates",
          {
            __interrupt__: [
              {
                id: "payment",
                value: { task: "approve_payment", amount: 100 },
              },
              {
                id: "refund",
                value: { task: "approve_refund", amount: 50 },
              },
              { id: "identity", value: { task: "verify_identity" } },
            ],
          },
        ],
      ]),
    );
    graph.checkpointer = null;

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(1);
    expect(events[0]?.event).toBe("suspend");
    const data = events[0]?.data as Record<string, unknown>;
    expect(data["suspend_payload"]).toEqual({
      task: "approve_payment",
      amount: 100,
    });
    expect(data["interrupts"]).toEqual([
      {
        id: "payment",
        value: { task: "approve_payment", amount: 100 },
      },
      { id: "refund", value: { task: "approve_refund", amount: 50 } },
      { id: "identity", value: { task: "verify_identity" } },
    ]);
    expect(data["resume_schema"]).toMatchObject({
      required: ["resume_map"],
      properties: {
        resume_map: { required: ["payment", "refund", "identity"] },
      },
    });
  });

  it("rejects duplicate interrupt ids", async () => {
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [
          [],
          "updates",
          {
            __interrupt__: [
              { id: "approval", value: { task: "first" } },
              { id: "approval", value: { task: "second" } },
            ],
          },
        ],
      ]),
    );
    const agent = new LangGraphBaseAgent(graph as never);
    await expect(
      collect(agent.stream({} as RequestContext, sampleInput())),
    ).rejects.toThrow("LangGraph returned duplicate interrupt ids");
  });

  it("emits a resume_map schema for a single pending interrupt", async () => {
    // The resume_map must require the exact interrupt id.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [
          [],
          "updates",
          {
            __interrupt__: [
              {
                id: "approval",
                value: { task: "approve_payment", amount: 100 },
              },
            ],
          },
        ],
      ]),
    );
    graph.checkpointer = null;

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(1);
    expect(events[0]?.event).toBe("suspend");
    const data = events[0]?.data as Record<string, unknown>;
    expect(data["resume_schema"]).toEqual({
      type: "object",
      required: ["resume_map"],
      properties: {
        resume_map: {
          type: "object",
          required: ["approval"],
          properties: { approval: {} },
          additionalProperties: false,
        },
      },
      additionalProperties: true,
    });
  });

  it("rejects non-serializable interrupt values before emitting suspend", async () => {
    // Python: test_stream_rejects_non_serializable_interrupt_value.
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;

    for (const [value, valueType] of [
      [1n, "bigint"],
      [Number.NaN, "number"],
      [Number.POSITIVE_INFINITY, "number"],
      [undefined, "undefined"],
      [() => undefined, "function"],
      [Symbol("approval"), "symbol"],
      [{ decision: undefined }, "Object"],
      [[undefined], "Array"],
      [new Map([["decision", "approve"]]), "Map"],
      [new Set(["approve"]), "Set"],
      [circular, "Object"],
    ] as const) {
      const graph = makeMockGraph();
      graph.stream.mockReturnValue(
        yieldFrom([
          [[], "updates", { __interrupt__: [{ id: "approval", value }] }],
        ]),
      );
      const events: StreamEvent[] = [];
      const agent = new LangGraphBaseAgent(graph as never);

      await expect(
        (async () => {
          for await (const event of agent.stream(
            {} as RequestContext,
            sampleInput(),
          )) {
            events.push(event);
          }
        })(),
      ).rejects.toThrow(
        `LangGraph interrupt "approval" has a non-JSON-serializable value of type ${valueType}`,
      );
      expect(events).toEqual([]);
    }
  });

  it("still emits suspend with checkpoint_id=null when checkpoint lookup throws", async () => {
    // Python: test_stream_emits_suspend_event_when_checkpoint_lookup_fails.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [
          [],
          "updates",
          {
            __interrupt__: [
              {
                id: "approval",
                value: {
                  suspend_reason: "Need approval",
                  suspend_context: { amount: 1000 },
                },
              },
            ],
          },
        ],
      ]),
    );
    graph.checkpointer = {
      getTuple: vi.fn().mockRejectedValue(new Error("transient")),
    };

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(1);
    expect(events[0]?.event).toBe("suspend");
    const d = events[0]?.data as Record<string, unknown>;
    expect(
      (d["metadata"] as Record<string, unknown>)["checkpoint_id"],
    ).toBeNull();
    expect(
      (d["suspend_payload"] as Record<string, unknown>)["suspend_reason"],
    ).toBe("Need approval");
  });
});

describe("LangGraphBaseAgent.stream — resume", () => {
  it("emits resumed=true on the result event when ctx.resume is true", async () => {
    // Python: test_stream_resume_via_ctx.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Resumed" }), {}]],
        [
          [],
          "updates",
          {
            node: {
              messages: [new AIMessage({ content: "Resumed and completed" })],
            },
          },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const resumeCtx: RequestContext = {
      userId: "user-123",
      sessionId: "thread-789",
      resume: true,
      resumeData: { decision: "approved" },
      metadata: { checkpoint_id: "ckpt-123" },
    };
    const resumeInput: AgentInput = { payload: { message: "" } };
    const events = await collect(agent.stream(resumeCtx, resumeInput));

    expect(events).toHaveLength(2);
    expect(events[0]?.event).toBe("token");
    expect(events[1]?.event).toBe("result");
    const d = events[1]?.data as Record<string, unknown>;
    expect(d["response"]).toBe("Resumed and completed");
    expect(d["resumed"]).toBe(true);
  });

  it("recomputes resolveThreadId into stream config on resume", async () => {
    // Python: test_stream_resume_recomputes_resolve_thread_id.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [
          [],
          "updates",
          { node: { messages: [new AIMessage({ content: "Resumed" })] } },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(
      graph as never,
      undefined,
      null,
      (ctx) => `${ctx.sessionId}__actor`,
    );
    const resumeCtx: RequestContext = {
      sessionId: "sess-1",
      workspaceId: "ws-1",
      resume: true,
      resumeData: { decision: "approved" },
    };
    const events = await collect(
      agent.stream(resumeCtx, { payload: { message: "" } }),
    );

    const streamConfig = graph.stream.mock.calls[0]?.[1] as {
      configurable: Record<string, unknown>;
    };
    expect(streamConfig.configurable["thread_id"]).toBe("sess-1__actor");
    expect(events).toHaveLength(1);
    expect(events[0]?.event).toBe("result");
    expect((events[0]?.data as Record<string, unknown>)["resumed"]).toBe(true);
  });

  it("throws when resume=true but resume_data is missing", async () => {
    // Python: test_stream_resume_without_data_raises.
    const agent = new LangGraphBaseAgent(makeMockGraph() as never);
    const badCtx: RequestContext = {
      userId: "user-123",
      sessionId: "thread-789",
      resume: true,
      metadata: { checkpoint_id: "ckpt-123" },
    };
    const badInput: AgentInput = { payload: { message: "" } };
    await expect(collect(agent.stream(badCtx, badInput))).rejects.toThrow(
      /resume_data/,
    );
  });
});

describe("LangGraphBaseAgent.stream — error cases", () => {
  it("throws when graph completes without producing any messages", async () => {
    // Python: test_stream_empty_messages_raises.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(yieldFrom([[[], "updates", { node: {} }]]));

    const agent = new LangGraphBaseAgent(graph as never);
    await expect(
      collect(agent.stream({} as RequestContext, sampleInput())),
    ).rejects.toThrow(/without producing messages/);
  });
});

// ---------------------------------------------------------------------------
// stream — Overwrite + Command update unwrap
// ---------------------------------------------------------------------------

describe("LangGraphBaseAgent.stream — Overwrite + Command unwrap", () => {
  it.each([
    [
      "a real Overwrite",
      () => new Overwrite([new AIMessage({ content: "Overwritten" })]),
    ],
    [
      "a serialized Overwrite",
      () => ({ __overwrite__: [new AIMessage({ content: "Overwritten" })] }),
    ],
  ])("unwraps %s so its messages reach the result", async (_kind, messages) => {
    // Python: test_stream_overwrite_wrapped_messages.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Overwritten" }), {}]],
        [
          [],
          "updates",
          {
            patch_tool_calls: {
              messages: messages(),
            },
          },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(2);
    expect(events[0]?.event).toBe("token");
    expect(events[1]?.event).toBe("result");
    expect((events[1]?.data as Record<string, unknown>)["response"]).toBe(
      "Overwritten",
    );
  });

  // Command.update messages: `stream()` now checks `instanceof Command` before
  // the generic-object branch, so Command instances are routed to
  // `processCommandUpdate` instead of being swallowed by the `'messages' in
  // node` check (Command's payload lives under `.update`, not `.messages`).
  it("extracts messages from Command.update", async () => {
    // Python: test_stream_command_update_with_messages.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Sub result" }), {}]],
        [
          [],
          "updates",
          {
            tools: new Command({
              update: {
                messages: [
                  new ToolMessage({ content: "done", tool_call_id: "tc-1" }),
                  new AIMessage({ content: "Sub result" }),
                ],
                custom_field: "val",
              },
            }),
          },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(2);
    expect(events[1]?.event).toBe("result");
    const d = events[1]?.data as Record<string, unknown>;
    expect(d["response"]).toBe("Sub result");
    expect(d["message_count"]).toBe(2);
  });

  it("skips Command updates that have no `messages` key", async () => {
    // Python: test_stream_command_update_without_messages.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Hello" }), {}]],
        [
          [],
          "updates",
          {
            summarize: new Command({
              update: { _summarization_event: { cutoff_index: 5 } },
            }),
          },
        ],
        [
          [],
          "updates",
          { agent: { messages: [new AIMessage({ content: "Hello" })] } },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(2);
    expect(events[0]?.event).toBe("token");
    expect(events[1]?.event).toBe("result");
    expect((events[1]?.data as Record<string, unknown>)["response"]).toBe(
      "Hello",
    );
  });

  it("regression: standard add_messages pattern still produces token + result", async () => {
    // Python: test_stream_standard_add_messages_regression.
    const graph = makeMockGraph();
    graph.stream.mockReturnValue(
      yieldFrom([
        [[], "messages", [new AIMessageChunk({ content: "Normal" }), {}]],
        [
          [],
          "updates",
          {
            agent: {
              messages: [new AIMessage({ content: "Normal response" })],
            },
          },
        ],
      ]),
    );

    const agent = new LangGraphBaseAgent(graph as never);
    const events = await collect(
      agent.stream({} as RequestContext, sampleInput()),
    );

    expect(events).toHaveLength(2);
    expect(events[0]?.event).toBe("token");
    expect((events[1]?.data as Record<string, unknown>)["response"]).toBe(
      "Normal response",
    );
  });
});

// ---------------------------------------------------------------------------
// getCheckpointId — error fallback
// ---------------------------------------------------------------------------

describe("native checkpoint metadata", () => {
  it("does not fail a completed execution when checkpoint lookup fails", async () => {
    const graph = makeMockGraph();
    graph.invoke.mockResolvedValue({
      messages: [new AIMessage({ content: "completed" })],
    });
    graph.checkpointer = {
      getTuple: vi.fn().mockRejectedValue(new Error("DB connection failed")),
    };

    const result = await new LangGraphBaseAgent(graph as never).invoke(
      { sessionId: "thread-123" } as RequestContext,
      sampleInput(),
    );

    expect((result.response as Record<string, unknown>)["status"]).toBe(
      "completed",
    );
    expect(result.response).not.toHaveProperty("metadata");
  });
});

describe("fenceCancelledPredecessorWrites", () => {
  type Doc = Record<string, unknown>;

  function makeFakeSaver(ckptDocs: Doc[], writesDocs: Doc[]) {
    const ckptCol = {
      distinct: async (key: string, flt: Doc) => [
        ...new Set(
          ckptDocs
            .filter((d) => d.thread_id === flt.thread_id)
            .map((d) => d[key]),
        ),
      ],
      findOne: async (flt: Doc, _opts: Doc) => {
        const matches = ckptDocs.filter((d) =>
          Object.entries(flt).every(([k, v]) => d[k] === v),
        );
        if (matches.length === 0) return null;
        return matches.reduce((a, b) =>
          String(a.checkpoint_id) > String(b.checkpoint_id) ? a : b,
        );
      },
      deleteMany: async () => ({ deletedCount: 0 }),
    };
    const writesCol = {
      distinct: async () => [],
      findOne: async () => null,
      deleteMany: async (flt: Doc) => {
        const before = writesDocs.length;
        for (let i = writesDocs.length - 1; i >= 0; i--) {
          if (Object.entries(flt).every(([k, v]) => writesDocs[i]?.[k] === v)) {
            writesDocs.splice(i, 1);
          }
        }
        return { deletedCount: before - writesDocs.length };
      },
    };
    return {
      db: {
        collection: (name: string) =>
          name === "checkpoints" ? ckptCol : writesCol,
      },
      checkpointCollectionName: "checkpoints",
      checkpointWritesCollectionName: "checkpoint_writes",
    };
  }

  function agentWithSaver(saver: unknown): {
    fence: (ctx: RequestContext) => Promise<void>;
  } {
    const graph = { checkpointer: saver };
    const agent = new LangGraphBaseAgent(graph as never);
    return {
      fence: async (ctx: RequestContext) => {
        await preparedRun(agent, ctx);
      },
    };
  }

  it("prunes only the latest checkpoint's writes per namespace", async () => {
    const thread = "sess-1:ws-1";
    const ckpts = [
      { thread_id: thread, checkpoint_ns: "", checkpoint_id: "c1" },
      { thread_id: thread, checkpoint_ns: "", checkpoint_id: "c2" },
      { thread_id: thread, checkpoint_ns: "sub:x", checkpoint_id: "s1" },
    ];
    const writes = [
      {
        thread_id: thread,
        checkpoint_ns: "",
        checkpoint_id: "c1",
        task_id: "old",
      },
      {
        thread_id: thread,
        checkpoint_ns: "",
        checkpoint_id: "c2",
        task_id: "dead",
      },
      {
        thread_id: thread,
        checkpoint_ns: "sub:x",
        checkpoint_id: "s1",
        task_id: "dead-sub",
      },
      {
        thread_id: "other",
        checkpoint_ns: "",
        checkpoint_id: "c2",
        task_id: "foreign",
      },
    ];
    const { fence } = agentWithSaver(
      new PlatformCheckpointer({
        native: makeFakeSaver(ckpts, writes) as never,
      }),
    );

    await fence({
      sessionId: "sess-1",
      workspaceId: "ws-1",
      previousExecutionCancelled: true,
    });

    expect(writes.map((w) => w.task_id).sort()).toEqual(["foreign", "old"]);
  });

  it("is a no-op without the flag, on resume, and without a Mongo saver", async () => {
    const thread = "sess-1:ws-1";
    const ckpts = [
      { thread_id: thread, checkpoint_ns: "", checkpoint_id: "c1" },
    ];
    const writes = [
      {
        thread_id: thread,
        checkpoint_ns: "",
        checkpoint_id: "c1",
        task_id: "w",
      },
    ];

    const { fence } = agentWithSaver(makeFakeSaver(ckpts, writes));
    await fence({ sessionId: "sess-1", workspaceId: "ws-1" });
    await fence({
      sessionId: "sess-1",
      workspaceId: "ws-1",
      previousExecutionCancelled: true,
      resume: true,
      resumeData: {},
    });
    expect(writes).toHaveLength(1);

    // Saver without Mongo internals: must simply return.
    const bare = agentWithSaver(undefined);
    await bare.fence({
      sessionId: "sess-1",
      workspaceId: "ws-1",
      previousExecutionCancelled: true,
    });

    const durable = agentWithSaver(new PlatformCheckpointer({ native: null }));
    await durable.fence({
      sessionId: "sess-1",
      workspaceId: "ws-1",
      previousExecutionCancelled: true,
    });
  });

  it("never fails the invoke when the prune errors", async () => {
    const saver = {
      db: {
        collection: () => ({
          distinct: async () => {
            throw new Error("mongo down");
          },
          findOne: async () => null,
          deleteMany: async () => ({ deletedCount: 0 }),
        }),
      },
      checkpointCollectionName: "checkpoints",
      checkpointWritesCollectionName: "checkpoint_writes",
    };
    const { fence } = agentWithSaver(saver);
    await expect(
      fence({
        sessionId: "sess-1",
        workspaceId: "ws-1",
        previousExecutionCancelled: true,
      }),
    ).resolves.toBeUndefined();
  });
});
