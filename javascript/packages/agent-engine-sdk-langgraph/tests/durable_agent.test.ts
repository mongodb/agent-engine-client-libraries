import { create } from "@bufbuild/protobuf";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  END,
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
import type { AgentInput, RequestContext } from "@mongodb-js/agent-engine-sdk";
import { describe, expect, it, vi } from "vitest";

import { LangGraphBaseAgent } from "../src/agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import {
  channelValuesToStateSnapshot,
  stateSnapshotToChannelValues,
} from "../src/workflow_state.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});

function durableAttempt(
  args: {
    replay?: boolean;
    attemptId?: string;
    fencingToken?: bigint;
  } = {},
) {
  return create(AttemptContextSchema, {
    attemptId: args.attemptId ?? "attempt-1",
    fencingToken: args.fencingToken ?? 7n,
    replayMode: args.replay ?? false,
    workflowIdentity: identity,
    previousState: channelValuesToStateSnapshot({
      messages: [new HumanMessage({ content: "previous", id: "previous-1" })],
      count: 1,
    }),
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

function request(): { ctx: RequestContext; input: AgentInput } {
  return {
    ctx: {
      sessionId: "session",
      workspaceId: "workspace",
    } as RequestContext,
    input: { payload: { message: "current" } },
  };
}

function compiledDurableAgent(saver: PlatformCheckpointer) {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("respond", async (state) => {
      const current = [...state.messages]
        .reverse()
        .find((message) => message instanceof HumanMessage);
      if (current === undefined) throw new Error("missing current message");
      return {
        messages: [
          new AIMessage({
            content: `reply:${current.content}`,
            id: "assistant-current",
          }),
        ],
      };
    })
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile({ checkpointer: saver as never });
  return { agent: new LangGraphBaseAgent(graph as never), graph };
}

function publishedMessages(
  completeExecution: ReturnType<typeof clients>["completeExecution"],
  callIndex = 0,
): BaseMessage[] {
  const state = completeExecution.mock.calls[callIndex]?.[0].state;
  if (state === undefined) throw new Error("missing published state");
  return stateSnapshotToChannelValues(state)["messages"] as BaseMessage[];
}

async function inDurableAttempt<T>(
  attempt: ReturnType<typeof durableAttempt>,
  fn: () => Promise<T>,
): Promise<T> {
  return runWithExecutionContext(
    { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
    () => runWithAttemptContext(attempt, fn),
  );
}

async function expectScratchReleased(
  saver: PlatformCheckpointer,
  attempt: ReturnType<typeof durableAttempt>,
): Promise<void> {
  await runWithAttemptContext(attempt, async () => {
    expect(
      await saver.getTuple({ configurable: { thread_id: "caller" } }),
    ).toBeUndefined();
  });
}

describe("LangGraphBaseAgent durable lifecycle", () => {
  it.each(["invoke", "stream"] as const)(
    "keeps same-named compiled-subgraph messages distinct across replay during %s",
    async (mode) => {
      const fake = clients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: fake.factory,
      });
      const child = (content: string) =>
        new StateGraph(MessagesAnnotation)
          .addNode("respond", () => ({
            messages: [new AIMessage({ content })],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", END)
          .compile();
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("left", child("left"))
        .addNode("right", child("right"))
        .addEdge(START, "left")
        .addEdge(START, "right")
        .addEdge("left", END)
        .addEdge("right", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      const requestValues = request();

      const run = async () => {
        if (mode === "invoke") {
          await agent.invoke(requestValues.ctx, requestValues.input);
          return;
        }
        for await (const _event of agent.stream(
          requestValues.ctx,
          requestValues.input,
        )) {
          // Drain the stream so the durable execution publishes terminal state.
        }
      };

      await inDurableAttempt(durableAttempt(), run);
      await inDurableAttempt(
        durableAttempt({
          replay: true,
          attemptId: "attempt-2",
          fencingToken: 8n,
        }),
        run,
      );

      const emitted = (callIndex: number) =>
        publishedMessages(fake.completeExecution, callIndex)
          .filter(
            (message) =>
              message.content === "left" || message.content === "right",
          )
          .map((message) => ({ content: message.content, id: message.id }));
      const first = emitted(0);
      const replay = emitted(1);

      expect(first).toHaveLength(2);
      expect(first).toEqual(replay);
      expect(first[0]?.id).not.toBe(first[1]?.id);
      expect(
        first.every((message) => message.id?.startsWith("durable-message:")),
      ).toBe(true);
    },
  );

  it.each([
    ["checkpointer=true", true],
    ["checkpointer=false", false],
    ["an independent saver", new MemorySaver()],
  ])("rejects a compiled child with %s", async (_description, checkpointer) => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const childNode = vi.fn(() => ({
      messages: [new AIMessage({ content: "child" })],
    }));
    const child = new StateGraph(MessagesAnnotation)
      .addNode("respond", childNode)
      .addEdge(START, "respond")
      .addEdge("respond", END)
      .compile({ checkpointer: checkpointer as never });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("child", child)
      .addEdge(START, "child")
      .addEdge("child", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    const requestValues = request();

    await expect(
      inDurableAttempt(durableAttempt(), () =>
        agent.invoke(requestValues.ctx, requestValues.input),
      ),
    ).rejects.toThrow(/must use the default checkpointer=undefined/);
    expect(childNode).not.toHaveBeenCalled();
    expect(fake.completeExecution).not.toHaveBeenCalled();
  });

  it.each([
    ["no checkpointer", undefined],
    ["a native LangGraph checkpointer", new MemorySaver()],
  ])(
    "rejects %s before preparing input or invoking the graph",
    async (_description, checkpointer) => {
      const prepare = vi.fn(() => ({
        messages: [new HumanMessage("current")],
      }));
      const invoke = vi.fn();
      const agent = new LangGraphBaseAgent(
        { checkpointer, invoke } as never,
        [],
        prepare,
      );
      const { ctx, input } = request();

      await expect(
        inDurableAttempt(durableAttempt(), () => agent.invoke(ctx, input)),
      ).rejects.toThrow(/requires the platform checkpointer/);
      expect(prepare).not.toHaveBeenCalled();
      expect(invoke).not.toHaveBeenCalled();
    },
  );

  it.each([
    {
      control: "scalar native resume",
      ctx: { resume: true, resumeData: "approved" },
      error: /activity-id-keyed resume map/,
    },
    {
      control: "explicit checkpoint targeting",
      ctx: { metadata: { checkpoint_id: "checkpoint-1" } },
      error: /explicit checkpoint targeting is not supported/,
    },
    {
      control: "checkpoint branching",
      ctx: {
        metadata: {
          langgraph_branch_point: {
            thread_id: "source",
            checkpoint_id: "checkpoint-1",
          },
        },
      },
      error: /checkpoint branching is not supported/,
    },
  ])(
    "rejects $control at the durable-session boundary",
    async ({ ctx, error }) => {
      const fake = clients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: fake.factory,
      });
      const prepare = vi.fn(() => ({
        messages: [new HumanMessage("current")],
      }));
      const invoke = vi.fn();
      const agent = new LangGraphBaseAgent(
        { checkpointer: saver, invoke } as never,
        [],
        prepare,
      );
      const requestValues = request();

      await expect(
        inDurableAttempt(durableAttempt(), () =>
          agent.invoke(
            { ...requestValues.ctx, ...ctx } as RequestContext,
            requestValues.input,
          ),
        ),
      ).rejects.toThrow(error);
      expect(prepare).not.toHaveBeenCalled();
      expect(invoke).not.toHaveBeenCalled();
    },
  );

  it.each(["invoke", "stream"] as const)(
    "%s rebuilds stable input, publishes the compiled graph state once, and releases scratch",
    async (mode) => {
      const fake = clients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: fake.factory,
      });
      const { agent, graph } = compiledDurableAgent(saver);
      const attempt = durableAttempt();
      const { ctx, input } = request();

      if (mode === "invoke") {
        const invoke = vi.spyOn(graph, "invoke");
        const result = await inDurableAttempt(attempt, () =>
          agent.invoke(ctx, input),
        );
        expect(result.response).toMatchObject({
          response: "reply:current",
          status: "completed",
          resumed: false,
        });
        expect(invoke.mock.calls[0]?.[1]).toMatchObject({ durability: "sync" });
      } else {
        const stream = vi.spyOn(graph, "stream");
        const events = await inDurableAttempt(attempt, async () => {
          const collected = [];
          for await (const event of agent.stream(ctx, input)) {
            collected.push(event);
          }
          return collected;
        });
        expect(events.at(-1)).toMatchObject({
          event: "result",
          data: { response: "reply:current", resumed: false },
        });
        expect(stream.mock.calls[0]?.[1]).toMatchObject({ durability: "sync" });
      }

      expect(fake.completeExecution).toHaveBeenCalledOnce();
      expect(
        fake.completeExecution.mock.calls[0]?.[0].workflowIdentity?.executionId,
      ).toBe("execution");
      expect(
        publishedMessages(fake.completeExecution).map((message) => ({
          content: message.content,
          id: message.id,
          type: message.getType(),
        })),
      ).toEqual([
        { content: "previous", id: "previous-1", type: "human" },
        {
          content: "current",
          id: "durable-input:execution",
          type: "human",
        },
        {
          content: "reply:current",
          id: "assistant-current",
          type: "ai",
        },
      ]);
      await expectScratchReleased(saver, attempt);
    },
  );

  it("rejects cross-version Command input returned by the fresh-input hook", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const prepare = vi.fn(() => ({
      lg_name: "Command",
      update: { messages: [new HumanMessage("again")] },
    }));
    const invoke = vi.fn();
    const agent = new LangGraphBaseAgent(
      { checkpointer: saver, invoke } as never,
      [],
      prepare,
    );
    const attempt = durableAttempt({ replay: true });
    const { ctx, input } = request();

    await expect(
      inDurableAttempt(attempt, () => agent.invoke(ctx, input)),
    ).rejects.toThrow(/Command input is not supported/);
    expect(invoke).not.toHaveBeenCalled();
    expect(fake.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt);
  });

  it("does not publish on execution error and still releases scratch", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const graph = {
      checkpointer: saver,
      invoke: vi.fn(async () => {
        throw new Error("graph failed");
      }),
      getState: vi.fn(),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    const attempt = durableAttempt();
    const { ctx, input } = request();

    await expect(
      inDurableAttempt(attempt, () => agent.invoke(ctx, input)),
    ).rejects.toThrow("graph failed");
    expect(fake.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt);
  });

  it("rejects durable static pauses without publishing terminal state", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const graph = {
      checkpointer: saver,
      invoke: vi.fn(async () => ({ messages: [new AIMessage("waiting")] })),
      getState: vi.fn(async () => ({ next: ["approval"], tasks: [] })),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    const attempt = durableAttempt();
    const { ctx, input } = request();

    await expect(
      inDurableAttempt(attempt, () => agent.invoke(ctx, input)),
    ).rejects.toThrow(/interrupt_before or interrupt_after/);
    expect(fake.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt);
  });

  it("rejects durable stream interrupts without publishing terminal state", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const graph = {
      checkpointer: saver,
      stream: vi.fn(async () =>
        (async function* () {
          yield [
            [],
            "updates",
            {
              __interrupt__: [
                { id: "approval", value: { question: "Proceed?" } },
              ],
            },
          ];
        })(),
      ),
      getState: vi.fn(async () => ({
        next: ["approval"],
        tasks: [
          { interrupts: [{ id: "approval", value: { question: "Proceed?" } }] },
        ],
      })),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    const attempt = durableAttempt();
    const { ctx, input } = request();

    await expect(
      inDurableAttempt(attempt, async () => {
        for await (const _event of agent.stream(ctx, input)) {
          // A durable interrupt rejects before yielding a public event.
        }
      }),
    ).rejects.toThrow(/requires a stable LangGraph task path/);
    expect(fake.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt);
  });

  it("releases scratch when a streaming consumer abandons the generator", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const graph = {
      checkpointer: saver,
      stream: vi.fn(async () =>
        (async function* () {
          yield [[], "messages", [new AIMessageChunk("partial")]];
          await new Promise<never>(() => undefined);
        })(),
      ),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    const attempt = durableAttempt();
    const { ctx, input } = request();

    await inDurableAttempt(attempt, async () => {
      const stream = agent.stream(ctx, input)[Symbol.asyncIterator]();
      expect((await stream.next()).value).toMatchObject({ event: "token" });
      await stream.return?.();
    });

    expect(fake.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt);
  });

  it("rebuilds replay input instead of treating recovery as native resume", async () => {
    const fake = clients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: fake.factory,
    });
    const preparedInputs: unknown[] = [];
    const prepare = vi.fn(() => ({
      messages: [
        ["system", "instructions"],
        new HumanMessage({ content: "kept", id: "caller-message" }),
        { role: "user", content: "again" },
      ],
    }));
    const graph = {
      checkpointer: saver,
      invoke: vi.fn(async (input: unknown, config: RunnableConfig) => {
        expect(input).not.toHaveProperty("resume");
        preparedInputs.push(input);
        expect(config.configurable?.["request_context"]).toMatchObject({
          resume: false,
          resumeData: { ignored: true },
          metadata: { checkpoint_id: "ignored" },
        });
        return { messages: [new AIMessage("done")] };
      }),
      getState: vi.fn(async () => ({ next: [] })),
    };
    const agent = new LangGraphBaseAgent(graph as never, [], prepare);
    const attempt = durableAttempt({ replay: true });
    const { input } = request();
    const ctx = {
      sessionId: "session",
      workspaceId: "workspace",
      resume: false,
      resumeData: { ignored: true },
      metadata: { checkpoint_id: "ignored" },
    } as RequestContext;

    await inDurableAttempt(attempt, () => agent.invoke(ctx, input));

    expect(prepare).toHaveBeenCalledWith(
      input,
      expect.objectContaining({
        resume: false,
        resumeData: null,
        metadata: null,
      }),
    );
    expect(
      (
        preparedInputs[0] as {
          messages: BaseMessage[];
        }
      ).messages.map((message) => message.id),
    ).toEqual([
      "durable-input:execution:0",
      "caller-message",
      "durable-input:execution:2",
    ]);
  });
});
