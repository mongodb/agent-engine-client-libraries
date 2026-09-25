import { create } from "@bufbuild/protobuf";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  Annotation,
  END,
  emptyCheckpoint,
  interrupt,
  MemorySaver,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import {
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
  runWithExecutionContext,
  type CompleteExecutionCommand,
  type FinalizeStepCommand,
} from "@mongodb-js/agent-engine-runner-shared";
import type { RequestContext } from "@mongodb-js/agent-engine-sdk";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LangGraphBaseAgent } from "../src/agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import {
  forkNativeSession,
  LangGraphForkPlugin,
  takeContinueWithoutUserMessage,
} from "../src/session_fork.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});

function durableAttempt(replayMode = false) {
  return create(AttemptContextSchema, {
    attemptId: "attempt-1",
    fencingToken: 7n,
    replayMode,
    workflowIdentity: identity,
  });
}

function clients() {
  const finalizeStep = vi.fn(async (_command: FinalizeStepCommand) => []);
  const completeExecution = vi.fn(
    async (_command: CompleteExecutionCommand) => undefined,
  );
  return {
    finalizeStep,
    completeExecution,
    factory: () => ({ finalizeStep, completeExecution }),
  };
}

function durableGraph(saver: PlatformCheckpointer) {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("respond", () => ({
      messages: [new AIMessage({ content: "reply", id: "assistant-1" })],
    }))
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile({ checkpointer: saver as never });
  return { agent: new LangGraphBaseAgent(graph as never), graph };
}

async function inDurableAttempt<T>(
  fn: () => Promise<T>,
  replayMode = false,
): Promise<T> {
  return runWithExecutionContext(
    { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
    () => runWithAttemptContext(durableAttempt(replayMode), fn),
  );
}

interface RecordedRequest {
  url: string;
  body: Record<string, unknown>;
}

function stubFetch(
  status = 201,
  payload: Record<string, unknown> = {
    session_id: "branch-session",
    execution_id: "branch-exec",
  },
): { calls: RecordedRequest[]; restore: () => void } {
  const calls: RecordedRequest[] = [];
  const mock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    calls.push({
      url: String(input),
      body: JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>,
    });
    return {
      status,
      json: async () => payload,
      text: async () => JSON.stringify(payload),
    };
  });
  vi.stubGlobal("fetch", mock);
  return {
    calls,
    restore: () => {
      vi.unstubAllGlobals();
    },
  };
}

function branchRequests(calls: RecordedRequest[]): RecordedRequest[] {
  return calls.filter((call) => call.url.includes("/branches"));
}

/** Answer successive CreateBranch calls with one payload each. */
function stubFetchSequence(payloads: ReadonlyArray<Record<string, unknown>>): {
  calls: RecordedRequest[];
  restore: () => void;
} {
  const calls: RecordedRequest[] = [];
  const mock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    calls.push({
      url: String(input),
      body: JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>,
    });
    const payload = payloads[Math.min(calls.length - 1, payloads.length - 1)];
    return {
      status: 201,
      json: async () => payload,
      text: async () => JSON.stringify(payload),
    };
  });
  vi.stubGlobal("fetch", mock);
  return {
    calls,
    restore: () => {
      vi.unstubAllGlobals();
    },
  };
}

function messageContents(snapshot: { values: unknown }): unknown[] {
  const values = snapshot.values as Record<string, unknown>;
  return ((values["messages"] ?? []) as Array<{ content: unknown }>).map(
    (message) => message.content,
  );
}

const callerConfig = (): RunnableConfig => ({
  configurable: { thread_id: "session" },
});

describe("wrapSessionForkUpdateState — durable sessions", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it.each([false, true])(
    "forks a committed root step into a new OE branch (replay=%s)",
    async (replayMode) => {
      const fake = clients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: fake.factory,
      });
      const { graph } = durableGraph(saver);
      const stub = stubFetch();

      await inDurableAttempt(async () => {
        await saver.put(
          callerConfig(),
          {
            ...emptyCheckpoint(),
            id: "root-1",
            channel_values: {
              messages: [new HumanMessage({ content: "seed", id: "human-1" })],
            },
          },
          { source: "loop", step: 0, parents: {} },
          {},
        );
        const destConfig = await graph.updateState(callerConfig(), null);
        expect(destConfig.configurable).toMatchObject({
          thread_id: "branch-session:workspace",
          checkpoint_ns: "",
        });
      }, replayMode);

      const request = branchRequests(stub.calls)[0];
      expect(request?.url).toBe(
        "http://oe/workflow/orgs/org/projects/project/workspaces/" +
          "workspace/sessions/session/executions/execution/branches",
      );
      expect(request?.body["step_ordinal"]).toBe(1);
      expect(String(request?.body["branch_key"])).toMatch(/^update-/);
      const state = request?.body["state"] as Record<string, unknown>;
      expect(state).toMatchObject({
        properties: expect.anything(),
        message_encoding_version: 1,
      });
      const messages = state["messages"] as Array<Record<string, unknown>>;
      expect(messages).toHaveLength(1);
      expect("source_message" in (messages[0] ?? {})).toBe(replayMode);
    },
  );

  it("reuses the branch for a repeated fork of the same step", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { graph } = durableGraph(saver);
    const stub = stubFetch();

    await inDurableAttempt(async () => {
      await saver.put(
        callerConfig(),
        {
          ...emptyCheckpoint(),
          id: "root-1",
          channel_values: { messages: [] },
        },
        { source: "loop", step: 0, parents: {} },
        {},
      );
      await graph.updateState(callerConfig(), null);
      await graph.updateState(callerConfig(), null);
    });

    const requests = branchRequests(stub.calls);
    expect(requests).toHaveLength(2);
    expect(requests[0]?.body["branch_key"]).toBe(
      requests[1]?.body["branch_key"],
    );
  });

  it("rejects a state patch before creating a branch", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { graph } = durableGraph(saver);
    const stub = stubFetch();

    await inDurableAttempt(async () => {
      await saver.put(
        callerConfig(),
        {
          ...emptyCheckpoint(),
          id: "root-1",
          channel_values: { messages: [] },
        },
        { source: "loop", step: 0, parents: {} },
        {},
      );
      await expect(
        graph.updateState(callerConfig(), { count: 2 }),
      ).rejects.toThrow(/does not support an updateState patch/);
    });

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("rejects a custom resolveThreadId on the durable fork path", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [new AIMessage({ content: "reply", id: "assistant-1" })],
      }))
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: saver as never });
    new LangGraphBaseAgent(graph as never, [], null, () => "custom-thread");
    const stub = stubFetch();

    await inDurableAttempt(async () => {
      await expect(graph.updateState(callerConfig(), null)).rejects.toThrow(
        /default thread_id formula/,
      );
    });

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("fails closed when the target scratch step never committed", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { graph } = durableGraph(saver);
    const stub = stubFetch();

    await inDurableAttempt(async () => {
      await saver.put(
        callerConfig(),
        {
          ...emptyCheckpoint(),
          id: "root-1",
          channel_values: { messages: [] },
        },
        { source: "input", step: -1, parents: {} },
        {},
      );
      await expect(graph.updateState(callerConfig(), null)).rejects.toThrow(
        /not a committed root step/,
      );
    });

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("fails closed when CreateBranch rejects the snapshot", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { graph } = durableGraph(saver);
    stubFetch(409, { error: { message: "state hash mismatch" } });

    await inDurableAttempt(async () => {
      await saver.put(
        callerConfig(),
        {
          ...emptyCheckpoint(),
          id: "root-1",
          channel_values: { messages: [] },
        },
        { source: "loop", step: 0, parents: {} },
        {},
      );
      await expect(graph.updateState(callerConfig(), null)).rejects.toThrow(
        /CreateBranch failed with HTTP 409/,
      );
    });
  });
});

describe("wrapSessionForkUpdateState — native sessions", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  function nativeGraph() {
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [new AIMessage({ content: "reply", id: "assistant-1" })],
      }))
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: new MemorySaver() });
    return { agent: new LangGraphBaseAgent(graph as never), graph };
  }

  function liveConfig(): RunnableConfig {
    return {
      configurable: {
        thread_id: "session:workspace",
        request_context: {
          sessionId: "session",
          workspaceId: "workspace",
          executionId: "execution",
        } as RequestContext,
      },
    };
  }

  /** A graph whose patch channel holds plain objects for branch-key tests. */
  function metaGraph() {
    const graph = new StateGraph(
      Annotation.Root({
        meta: Annotation<Record<string, unknown>>({
          reducer: (
            _current: Record<string, unknown>,
            next: Record<string, unknown>,
          ) => next,
          default: () => ({}),
        }),
      }),
    )
      .addNode("respond", () => ({ meta: { reply: true } }))
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: new MemorySaver() });
    new LangGraphBaseAgent(graph as never);
    return { graph };
  }

  it("copies the selected history onto a fresh branch session", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    const stub = stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    const destConfig = await graph.updateState(liveConfig(), null);

    expect(branchRequests(stub.calls)).toHaveLength(1);
    const request = branchRequests(stub.calls)[0];
    expect(request?.url).toBe(
      "http://oe/workflow/orgs/org/projects/project/workspaces/workspace/" +
        "sessions/session/executions/execution/branches",
    );
    expect(request?.body).toMatchObject({
      checkpoint_id: expect.any(String),
    });
    const destThreadId = destConfig.configurable?.["thread_id"];
    expect(destThreadId).toBe("branch-session:workspace");
    // bulkUpdateState rebuilds messages with new LangChain ids — compare
    // content only.
    expect(messageContents(await graph.getState(destConfig))).toEqual(
      messageContents(await graph.getState(liveConfig())),
    );
  });

  it("arms the once-only dest continue mark for the new thread", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    await graph.updateState(liveConfig(), null);

    expect(takeContinueWithoutUserMessage("branch-session:workspace")).toBe(
      true,
    );
    expect(takeContinueWithoutUserMessage("branch-session:workspace")).toBe(
      false,
    );
  });

  it("applies a state patch to the branch destination only", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    const destConfig = await graph.updateState(
      liveConfig(),
      { messages: [new HumanMessage({ content: "patch", id: "patch-1" })] },
      "respond",
    );

    const sourceMessages = (
      (await graph.getState(liveConfig())).values as Record<string, unknown>
    )["messages"];
    const destMessages = (
      (await graph.getState(destConfig)).values as Record<string, unknown>
    )["messages"];
    // The source run produced [human, AI reply]; the patch belongs to the
    // destination only.
    expect(sourceMessages).toHaveLength(2);
    expect(destMessages).toHaveLength(3);
  });

  it("passes through updateState for an unidentifiable source", async () => {
    const { graph } = nativeGraph();
    const stub = stubFetch();
    const bareConfig: RunnableConfig = {
      configurable: { thread_id: "plain-thread" },
    };

    await graph.updateState(
      bareConfig,
      {
        messages: [new HumanMessage({ content: "seed", id: "human-1" })],
      },
      "respond",
    );

    expect(branchRequests(stub.calls)).toEqual([]);
    const state = await graph.getState(bareConfig);
    expect((state.values as Record<string, unknown>)["messages"]).toHaveLength(
      1,
    );
  });

  it("fails before dispatch when the history point does not exist", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    const { graph } = nativeGraph();
    const stub = stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );

    await expect(
      graph.updateState(
        {
          configurable: {
            thread_id: "session:workspace",
            checkpoint_id: "missing-checkpoint",
            request_context: {
              sessionId: "session",
              workspaceId: "workspace",
              executionId: "execution",
            } as RequestContext,
          },
        },
        null,
      ),
    ).rejects.toThrow(/has no checkpoint history/);

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("fails before dispatch when the history point is still running", async () => {
    vi.stubEnv("OE_URL", "http://oe");
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
        return { messages: [new AIMessage({ content: "reply" })] };
      })
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: new MemorySaver() });
    new LangGraphBaseAgent(graph as never);
    const stub = stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "pause" })] },
      liveConfig(),
    );
    await expect(graph.updateState(liveConfig(), null)).rejects.toThrow(
      /not a completed root checkpoint/,
    );

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it.each([
    { error: "incomplete", state: undefined, message: "failed history" },
    {
      error: undefined,
      state: { configurable: {} },
      message: "persistent subgraph history",
    },
  ])(
    "fails before dispatch on $message in the replay history",
    async ({ error, state, message }) => {
      const selected = {
        config: {
          configurable: {
            thread_id: "session:workspace",
            checkpoint_ns: "",
            checkpoint_id: "source-checkpoint",
          },
        },
        values: { messages: ["source"] },
        next: [],
        tasks: [],
        parentConfig: {
          configurable: {
            thread_id: "session:workspace",
            checkpoint_ns: "",
            checkpoint_id: "parent-checkpoint",
          },
        },
      };
      const parent = {
        config: {
          configurable: {
            thread_id: "session:workspace",
            checkpoint_ns: "",
            checkpoint_id: "parent-checkpoint",
          },
        },
        values: { messages: ["parent"] },
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
        parentConfig: undefined,
      };
      const graph = {
        checkpointer: {
          getTuple: vi.fn().mockResolvedValue({ pendingWrites: [] }),
        },
        getState: vi.fn(async (config: RunnableConfig) =>
          config.configurable?.["checkpoint_id"] === "parent-checkpoint"
            ? parent
            : selected,
        ),
      };
      const stub = stubFetch();

      await expect(
        forkNativeSession(graph as never, {
          ctx: {
            sessionId: "session",
            workspaceId: "workspace",
            executionId: "execution",
          } as RequestContext,
          historyId: "source-checkpoint",
        }),
      ).rejects.toThrow(message);

      expect(selected.values).toEqual({ messages: ["source"] });
      expect(branchRequests(stub.calls)).toEqual([]);
    },
  );

  it("seals a populated destination on a repeated native fork", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    stubFetchSequence([
      { session_id: "branch-session", execution_id: "branch-exec" },
      { session_id: "branch-session", execution_id: "branch-exec" },
    ]);

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    await graph.updateState(liveConfig(), null);

    // Python `fork_native_session` parity: the deterministic branch already
    // exists, and its populated destination stays sealed instead of being
    // reused, verified, or repaired.
    await expect(graph.updateState(liveConfig(), null)).rejects.toThrow(
      /destination thread is not empty/,
    );
  });

  it("seals a repeated patched fork after an unrelated destination update", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    stubFetchSequence([
      { session_id: "branch-session", execution_id: "branch-exec" },
      { session_id: "branch-session", execution_id: "branch-exec" },
    ]);
    const destThread: RunnableConfig = {
      configurable: { thread_id: "branch-session:workspace" },
    };
    const patch = {
      messages: [new HumanMessage({ content: "patch", id: "patch-1" })],
    };

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    await graph.updateState(liveConfig(), null);
    // An unrelated same-thread update after the fork must never be mistaken
    // for the requested patch on a retry.
    await graph.updateState(
      destThread,
      { messages: [new HumanMessage({ content: "manual", id: "manual-1" })] },
      "respond",
    );

    await expect(
      graph.updateState(liveConfig(), patch, "respond"),
    ).rejects.toThrow(/destination thread is not empty/);
  });

  it("fails before dispatch when a multi-node patch omits asNode", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [new AIMessage({ content: "reply" })],
      }))
      .addNode("summarize", () => ({
        messages: [new AIMessage({ content: "summary" })],
      }))
      .addEdge(START, "respond")
      .addEdge("respond", "summarize")
      .addEdge("summarize", END)
      .compile({ checkpointer: new MemorySaver() });
    new LangGraphBaseAgent(graph as never);
    const stub = stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );

    await expect(
      graph.updateState(liveConfig(), {
        messages: [new HumanMessage({ content: "patch" })],
      }),
    ).rejects.toThrow(/requires a destination node/);

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("fails before dispatch when a patch names an unknown node", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [new AIMessage({ content: "reply" })],
      }))
      .addNode("summarize", () => ({
        messages: [new AIMessage({ content: "summary" })],
      }))
      .addEdge(START, "respond")
      .addEdge("respond", "summarize")
      .addEdge("summarize", END)
      .compile({ checkpointer: new MemorySaver() });
    new LangGraphBaseAgent(graph as never);
    const stub = stubFetch();

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );

    await expect(
      graph.updateState(
        liveConfig(),
        { messages: [new HumanMessage({ content: "patch" })] },
        "missing",
      ),
    ).rejects.toThrow(/is not a node in this graph/);

    expect(branchRequests(stub.calls)).toEqual([]);
  });

  it("derives the same branch key for equivalent patch state with different key order", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = metaGraph();
    const stub = stubFetchSequence([
      { session_id: "branch-session", execution_id: "branch-exec" },
      { session_id: "branch-session", execution_id: "branch-exec" },
    ]);

    await graph.invoke({ meta: { seed: true } }, liveConfig());
    await graph.updateState(
      liveConfig(),
      { meta: { a: 1, b: { c: 2, d: 3 } } },
      "respond",
    );
    // The equivalent patch reaches the same branch key, whose destination is
    // sealed (Python parity); the recorded key is what this test pins.
    await expect(
      graph.updateState(
        liveConfig(),
        { meta: { b: { d: 3, c: 2 }, a: 1 } },
        "respond",
      ),
    ).rejects.toThrow(/destination thread is not empty/);

    const requests = branchRequests(stub.calls);
    expect(requests).toHaveLength(2);
    expect(requests[0]?.body["branch_key"]).toBe(
      requests[1]?.body["branch_key"],
    );
  });

  it("keeps distinct branch keys when patch array order differs", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = metaGraph();
    const stub = stubFetchSequence([
      { session_id: "branch-a", execution_id: "exec-a" },
      { session_id: "branch-b", execution_id: "exec-b" },
    ]);

    await graph.invoke({ meta: { seed: true } }, liveConfig());
    await graph.updateState(
      liveConfig(),
      { meta: { list: [1, 2] } },
      "respond",
    );
    await graph.updateState(
      liveConfig(),
      { meta: { list: [2, 1] } },
      "respond",
    );

    const requests = branchRequests(stub.calls);
    expect(requests).toHaveLength(2);
    expect(requests[0]?.body["branch_key"]).not.toBe(
      requests[1]?.body["branch_key"],
    );
  });

  it("continues a forked destination without an empty user turn", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    vi.stubEnv("RUNNER_MODE", "aer");
    const { agent, graph } = nativeGraph();
    stubFetchSequence([
      {
        session_id: "branch-session",
        execution_id: "branch-exec",
      },
    ]);

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    await graph.updateState(liveConfig(), null);
    await agent.invoke(
      {
        sessionId: "branch-session",
        workspaceId: "workspace",
        executionId: "branch-exec",
      } as RequestContext,
      { payload: { message: "" } },
    );

    const dest = await graph.getState({
      configurable: { thread_id: "branch-session:workspace" },
    });
    expect(messageContents(dest)).toEqual(["one", "reply"]);
  });

  it("still rejects a new branch whose destination is occupied", async () => {
    vi.stubEnv("OE_URL", "http://oe");
    vi.stubEnv("ORG_ID", "org");
    vi.stubEnv("PROJECT_ID", "project");
    const { graph } = nativeGraph();
    stubFetchSequence([
      {
        session_id: "branch-session",
        execution_id: "branch-exec",
      },
    ]);

    await graph.invoke(
      { messages: [new HumanMessage({ content: "one" })] },
      liveConfig(),
    );
    await graph.invoke(
      { messages: [new HumanMessage({ content: "occupied" })] },
      { configurable: { thread_id: "branch-session:workspace" } },
    );

    await expect(graph.updateState(liveConfig(), null)).rejects.toThrow(
      /destination thread is not empty/,
    );
  });
});

describe("LangGraphForkPlugin", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("forks durably and validates request identity against the attempt", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { agent } = durableGraph(saver);
    const plugin = new LangGraphForkPlugin({
      getAgent: () => agent,
      workspaceIdResolver: () => "workspace",
    });
    const stub = stubFetch();

    await inDurableAttempt(async () => {
      await saver.put(
        callerConfig(),
        {
          ...emptyCheckpoint(),
          id: "root-1",
          channel_values: { messages: [] },
        },
        { source: "loop", step: 0, parents: {} },
        {},
      );
      const branch = await plugin.forkSession({
        sessionId: "session",
        executionId: "execution",
        historyId: null,
      });
      expect(branch).toEqual({
        sessionId: "branch-session",
        executionId: "branch-exec",
      });
    });

    expect(branchRequests(stub.calls)).toHaveLength(1);
  });

  it("rejects a durable fork request that does not match the attempt", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { agent } = durableGraph(saver);
    const plugin = new LangGraphForkPlugin({
      getAgent: () => agent,
      workspaceIdResolver: () => "workspace",
    });
    stubFetch();

    await inDurableAttempt(async () => {
      await expect(
        plugin.forkSession({
          sessionId: "other-session",
          executionId: "execution",
          historyId: null,
        }),
      ).rejects.toThrow(/does not match the current OE attempt/);
    });
  });

  it("rejects a durable fork carrying a state patch", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const { agent } = durableGraph(saver);
    const plugin = new LangGraphForkPlugin({
      getAgent: () => agent,
      workspaceIdResolver: () => "workspace",
    });
    stubFetch();

    await inDurableAttempt(async () => {
      await expect(
        plugin.forkSession({
          sessionId: "session",
          executionId: "execution",
          historyId: null,
          state: { count: 2 },
        }),
      ).rejects.toThrow(/does not support an updateState patch/);
    });
  });
});
