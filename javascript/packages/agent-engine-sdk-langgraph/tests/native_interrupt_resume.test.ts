/** Real-LangGraph coverage for native interrupt suspend and resume. */

import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import {
  Command,
  END,
  MemorySaver,
  MessagesAnnotation,
  ParentCommand,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
import type {
  AgentInput,
  JsonValue,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import {
  registerSuspendHandler,
  resetHooks,
  runWithExecutionContext,
  SecureToolWrapper,
  ToolExecutionError,
} from "@mongodb-js/agent-engine-runner-shared";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../src/agent.js";
import { CANCELLED_TURN_MESSAGE_ID_METADATA_KEY } from "../src/cancelled_turn.js";
import { App } from "../src/runtime.js";

const input: AgentInput = { payload: { message: "start" } };

async function collect<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const values: T[] = [];
  for await (const value of iterable) values.push(value);
  return values;
}

function mustGet<T>(value: T | undefined): T {
  expect(value).toBeDefined();
  if (value === undefined) throw new Error("expected a value");
  return value;
}

async function suspend(
  agent: LangGraphBaseAgent,
  sessionId: string,
): Promise<{
  data: Record<string, unknown>;
  metadata: Record<string, JsonValue>;
}> {
  const events = await collect(
    agent.stream({ sessionId } as RequestContext, input),
  );
  expect(events.map((event) => event.event)).toEqual(["suspend"]);
  const data = events[0]?.data as Record<string, unknown>;
  const metadata = data["metadata"] as Record<string, JsonValue>;
  return { data, metadata };
}

async function resume(
  agent: LangGraphBaseAgent,
  sessionId: string,
  metadata: Record<string, JsonValue>,
  resumeData: JsonValue,
): Promise<StreamEvent[]> {
  return collect(
    agent.stream(
      {
        sessionId,
        resume: true,
        resumeData,
        metadata,
      },
      { payload: { message: "" } },
    ),
  );
}

function singleInterruptAgent(
  payload: JsonValue,
  received: JsonValue[],
): LangGraphBaseAgent {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("gate", async () => {
      const answer = interrupt(payload) as JsonValue;
      received.push(answer);
      return { messages: [new AIMessage({ content: "resumed" })] };
    })
    .addEdge(START, "gate")
    .addEdge("gate", END)
    .compile({ checkpointer: new MemorySaver() });
  return new LangGraphBaseAgent(graph as never);
}

/** gate (real HITL interrupt) → step_one (intermediate message) → gated
 * (blocks). A resume runs step_one, commits it, then parks in `gated` — the
 * window in which a Stop must still recover the persisted opener. */
function gatedResumeAgent(
  received: JsonValue[],
  onEnter: () => void,
  wait: Promise<void>,
): LangGraphBaseAgent {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("gate", async () => {
      const answer = interrupt("question") as JsonValue;
      received.push(answer);
      return {};
    })
    .addNode("step_one", async () => ({
      messages: [new AIMessage({ content: "step one", id: "a-1" })],
    }))
    .addNode("gated", async () => {
      onEnter();
      await wait;
      return { messages: [new AIMessage({ content: "step two", id: "a-2" })] };
    })
    .addEdge(START, "gate")
    .addEdge("gate", "step_one")
    .addEdge("step_one", "gated")
    .addEdge("gated", END)
    .compile({ checkpointer: new MemorySaver() });
  return new LangGraphBaseAgent(graph as never);
}

const jsonFixtures: JsonValue[] = [
  "approve?",
  42,
  true,
  null,
  ["a", 2, false, null],
  { nested: { items: [1, 1.5], enabled: true } },
];

describe("LangGraphBaseAgent native interrupt resume", () => {
  it.each(jsonFixtures)(
    "accepts arbitrary JSON interrupt payload %#",
    async (payload) => {
      const received: JsonValue[] = [];
      const agent = singleInterruptAgent(payload, received);
      const sessionId = `payload-${JSON.stringify(payload)}`;

      const { data, metadata } = await suspend(agent, sessionId);
      const interrupts = data["interrupts"] as Array<{
        id: string;
        value: JsonValue;
      }>;
      const pendingInterrupt = mustGet(interrupts[0]);
      expect(pendingInterrupt.value).toEqual(payload);

      const events = await resume(agent, sessionId, metadata, {
        [pendingInterrupt.id]: { accepted: true },
      });

      expect(events.map((event) => event.event)).toEqual(["result"]);
      expect(received).toEqual([{ accepted: true }]);
    },
  );

  it("resumes one interrupt with a plain message", async () => {
    const received: JsonValue[] = [];
    const agent = singleInterruptAgent("question", received);

    const { metadata } = await suspend(agent, "plain-message");
    const events = await resume(agent, "plain-message", metadata, "approved");

    expect(events.map((event) => event.event)).toEqual(["result"]);
    expect(received).toEqual(["approved"]);
  });

  it("records the persisted opener when a native resume is cancelled", async () => {
    // A resume opens no new user message: the cancel must record the opener
    // the first execution persisted, not append a second copy of the prompt.
    const received: JsonValue[] = [];
    const agent = singleInterruptAgent("question", received);
    const sessionId = "resume-cancel";
    const { metadata } = await suspend(agent, sessionId);

    // OE dispatches exec.Message even on resume, so the payload carries the
    // original prompt.
    const resumeInput: AgentInput = {
      payload: { message: "the original prompt" },
    };
    const resumeCtx: RequestContext = {
      sessionId,
      resume: true,
      resumeData: "approved",
      metadata,
    };
    await collect(agent.stream(resumeCtx, resumeInput));

    const coordinate = await agent.commitCancelledTurn(resumeCtx, resumeInput);
    if (coordinate === null) {
      throw new Error("expected a cancelled-turn coordinate");
    }

    const saver = agent.compiledGraph.checkpointer as unknown as MemorySaver;
    const tip = await saver.getTuple({
      configurable: { thread_id: coordinate.thread_id },
    });
    const messages = mustGet(tip).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    const humans = messages.filter((message) => message.getType() === "human");
    expect(humans).toHaveLength(1);
    expect(
      (mustGet(tip).metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(humans[0]?.id);
  });

  it("records the persisted opener when a resume is cancelled during preparation", async () => {
    // The Stop beats the resumed step: the commit recovers the opener from the
    // checkpoint it already loaded, so no interruptible lookup can miss it.
    const received: JsonValue[] = [];
    const agent = singleInterruptAgent("question", received);
    const sessionId = "resume-cancel-prep";
    await suspend(agent, sessionId);

    const resumeInput: AgentInput = {
      payload: { message: "the original prompt" },
    };
    const resumeCtx: RequestContext = {
      sessionId,
      resume: true,
      resumeData: "approved",
    };
    // No resume stream runs — the cancel beats the resumed step.
    const coordinate = await agent.commitCancelledTurn(resumeCtx, resumeInput);
    if (coordinate === null) {
      throw new Error("expected a cancelled-turn coordinate");
    }

    const saver = agent.compiledGraph.checkpointer as unknown as MemorySaver;
    const tip = await saver.getTuple({
      configurable: { thread_id: coordinate.thread_id },
    });
    const messages = mustGet(tip).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    const humans = messages.filter((message) => message.getType() === "human");
    expect(humans).toHaveLength(1);
    expect(
      (mustGet(tip).metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(humans[0]?.id);
  });

  it("records the persisted opener when a resume is cancelled mid-continuation", async () => {
    // The continuation is still running when Stop lands: step_one has
    // committed its intermediate message and the next step is gated. The
    // teardown commit must recover the persisted opener and retain the
    // completed step, in order.
    const received: JsonValue[] = [];
    let onEnter: () => void = () => {};
    const entered = new Promise<void>((resolve) => {
      onEnter = resolve;
    });
    let abortGated: (reason: unknown) => void = () => {};
    const gatedWait = new Promise<void>((_resolve, reject) => {
      abortGated = reject;
    });
    const agent = gatedResumeAgent(received, onEnter, gatedWait);
    const sessionId = "resume-mid-continuation";
    const { metadata } = await suspend(agent, sessionId);

    const resumeInput: AgentInput = {
      payload: { message: "the original prompt" },
    };
    const resumeCtx: RequestContext = {
      sessionId,
      resume: true,
      resumeData: "approved",
      metadata,
    };

    // Drive the resumed continuation until the gated step is executing.
    const pump = (async () => {
      for await (const event of agent.stream(resumeCtx, resumeInput)) {
        void event; // consume; the gated step blocks before the stream completes
      }
    })();
    await entered;

    // Stop tears the invocation down mid-continuation, then it settles.
    abortGated(new Error("stopped"));
    await pump.catch(() => undefined);

    const coordinate = await agent.commitCancelledTurn(resumeCtx, resumeInput);
    if (coordinate === null) {
      throw new Error("expected a cancelled-turn coordinate");
    }

    const saver = agent.compiledGraph.checkpointer as unknown as MemorySaver;
    const tip = await saver.getTuple({
      configurable: { thread_id: coordinate.thread_id },
    });
    const messages = mustGet(tip).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    expect(messages.map((m) => m.content)).toEqual(["start", "step one"]);
    const humans = messages.filter((message) => message.getType() === "human");
    expect(humans).toHaveLength(1);
    expect(humans[0]?.content).toBe("start");
    expect(
      (mustGet(tip).metadata as Record<string, unknown>)[
        CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
      ],
    ).toBe(humans[0]?.id);

    // A fresh turn preserves the stopped opener and the completed step.
    await collect(
      agent.stream({ sessionId } as RequestContext, {
        payload: { message: "next" },
      }),
    );
    const after = await saver.getTuple({
      configurable: { thread_id: coordinate.thread_id },
    });
    const afterMessages = mustGet(after).checkpoint.channel_values[
      "messages"
    ] as BaseMessage[];
    expect(
      afterMessages
        .filter((m) => m.getType() === "human")
        .map((m) => m.content),
    ).toEqual(["start", "next"]);
  });

  it("delivers a legacy human_review dict as one interrupt answer", async () => {
    const received: JsonValue[] = [];
    const agent = singleInterruptAgent("legacy review", received);
    const humanReview = {
      human_review: { decision: "approve", reviewer_notes: "looks good" },
    };

    const { metadata } = await suspend(agent, "legacy-human-review");
    const events = await resume(
      agent,
      "legacy-human-review",
      metadata,
      humanReview,
    );

    expect(events.map((event) => event.event)).toEqual(["result"]);
    expect(received).toEqual([humanReview]);
  });

  it.each([
    "text",
    42,
    1.5,
    true,
    null,
    ["a", 2, false, null],
    { nested: { items: [1, 1.5], enabled: true, empty: null } },
  ] satisfies JsonValue[])(
    "preserves structured resume type %#",
    async (answer) => {
      const received: JsonValue[] = [];
      const agent = singleInterruptAgent({ question: "value?" }, received);
      const sessionId = `answer-${JSON.stringify(answer)}`;

      const { data, metadata } = await suspend(agent, sessionId);
      const interrupts = data["interrupts"] as Array<{ id: string }>;
      const pendingInterrupt = mustGet(interrupts[0]);
      const events = await resume(agent, sessionId, metadata, {
        [pendingInterrupt.id]: answer,
      });

      expect(events.map((event) => event.event)).toEqual(["result"]);
      expect(received).toEqual([answer]);
      expect(typeof received[0]).toBe(typeof answer);
      expect(Array.isArray(received[0])).toBe(Array.isArray(answer));
    },
  );

  it.each([
    {
      consented: false,
      expected:
        "error: Tool execution error: OAuth authorization is still required after continuation",
    },
    { consented: true, expected: '["qualification-repo"]' },
  ])(
    "never runs an OAuth tool without a token after Continue (consented: $consented)",
    async ({ consented, expected }) => {
      registerSuspendHandler(interrupt as never);
      const challenge = {
        proceed: false,
        reason: "authorization required",
        status: "suspend",
        elicitation: {
          elicitation_id: "elic-1",
          authorization_url: "https://idp.example/authorize?state=elic-1",
          message: "Connect Keycloak",
          created: true,
        },
      };
      // Before consent the resumed node is challenged again, and once more
      // after interrupt() returns the answer; after consent its gate proceeds.
      const replies: unknown[] = consented
        ? [
            challenge,
            {
              proceed: true,
              status: "success",
              result: '["qualification-repo"]',
              duration_ms: 1,
            },
          ]
        : [challenge, challenge, challenge];
      const approvals = replies.length;
      const fetchMock = vi.fn(async () =>
        Promise.resolve(
          new Response(JSON.stringify(replies.shift()), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        ),
      );
      vi.stubGlobal("fetch", fetchMock);
      const wrapper = new SecureToolWrapper("http://oe:8000", "exec-oauth");
      const received: JsonValue[] = [];
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("tool", async () => {
          try {
            received.push(
              (await wrapper.executeTool(
                "list_repos",
                {},
                { providerType: "keycloak", scopes: ["repo"], isLocal: false },
              )) as JsonValue,
            );
          } catch (error) {
            if (!(error instanceof ToolExecutionError)) throw error;
            received.push(`error: ${error.message}`);
          }
          return { messages: [new AIMessage({ content: "tool done" })] };
        })
        .addEdge(START, "tool")
        .addEdge("tool", END)
        .compile({ checkpointer: new MemorySaver() });
      const agent = new LangGraphBaseAgent(graph as never);

      try {
        const { data, metadata } = await suspend(agent, "oauth-continue");
        const interrupts = data["interrupts"] as Array<{
          id: string;
          value: Record<string, JsonValue>;
        }>;
        const pendingInterrupt = mustGet(interrupts[0]);
        expect(pendingInterrupt.value["suspend_reason"]).toBe(
          "authorization_required",
        );

        const events = await resume(agent, "oauth-continue", metadata, {
          [pendingInterrupt.id]: { decision: "approve" },
        });

        expect(events.map((event) => event.event)).toEqual(["result"]);
        expect(received).toEqual([expected]);
        expect(fetchMock).toHaveBeenCalledTimes(approvals);
      } finally {
        vi.unstubAllGlobals();
        resetHooks();
      }
    },
  );

  it("resumes a tool-result suspend with the decision as its result", async () => {
    const previousMode = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    registerSuspendHandler(interrupt as never);
    const app = new App({ appName: "Native interrupt tool test" });
    const reviewClaim = app.tool({
      isLocal: false,
      description: "Request a human decision for a claim.",
      schema: z.object({ claimId: z.string() }),
    })(function review_claim({ claimId }: { claimId: string }): string {
      return app.suspend("claim_review", {
        claim_id: claimId,
        allowed_decisions: ["approve", "deny"],
      });
    });
    const suspendWire = reviewClaim({ claimId: "claim-1" });
    const wrappedTool = app.getTools()[0] as {
      invoke: (input: unknown) => Promise<unknown>;
    };
    const wrapper = new SecureToolWrapper("http://oe:8000", "exec-tool");
    const received: JsonValue[] = [];
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("tool", async () => {
        const result = await wrappedTool.invoke({ claimId: "claim-1" });
        const content = Array.isArray(result) ? result[0] : result;
        received.push(JSON.parse(String(content)) as JsonValue);
        return { messages: [new AIMessage({ content: "tool resumed" })] };
      })
      .addEdge(START, "tool")
      .addEdge("tool", END)
      .compile({ checkpointer: new MemorySaver() });
    const agent = new LangGraphBaseAgent(graph as never);
    const fetchMock = vi.fn(async () =>
      Promise.resolve(
        new Response(
          // HITL suspend is gated on OE's status channel, so
          // the mock must report status "suspend" to fire the interrupt.
          JSON.stringify({
            proceed: true,
            status: "suspend",
            result: suspendWire,
            duration_ms: 1,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    try {
      await runWithExecutionContext(
        { executionId: "exec-tool", wrapper, oeUrl: "http://oe:8000" },
        async () => {
          const { data, metadata } = await suspend(
            agent,
            "tool-result-suspend",
          );
          const interrupts = data["interrupts"] as Array<{
            id: string;
            value: JsonValue;
          }>;
          const pendingInterrupt = mustGet(interrupts[0]);
          expect(pendingInterrupt.value).toEqual({
            suspend_reason: "claim_review",
            suspend_context: {
              claim_id: "claim-1",
              allowed_decisions: ["approve", "deny"],
            },
          });
          const decision = {
            decision: "approve",
            reviewer_notes: "verified",
          };

          const events = await resume(agent, "tool-result-suspend", metadata, {
            [pendingInterrupt.id]: decision,
          });

          expect(events.map((event) => event.event)).toEqual(["result"]);
          expect(received).toEqual([decision]);
        },
      );
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      vi.unstubAllGlobals();
      resetHooks();
      if (previousMode === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = previousMode;
    }
  });

  it("runs a local native interrupt on the live graph stack", async () => {
    const previousMode = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    const app = new App({ appName: "In-process native interrupt tool test" });
    app.tool({
      isLocal: true,
      description: "Request a native framework decision for a claim.",
      schema: z.object({ claimId: z.string() }),
    })(function review_claim({ claimId }: { claimId: string }): JsonValue {
      return interrupt({
        claim_id: claimId,
        question: "approve?",
      }) as JsonValue;
    });
    const wrappedTool = app.getTools()[0] as {
      invoke: (input: unknown) => Promise<unknown>;
    };
    const wrapper = new SecureToolWrapper("http://oe:8000", "exec-tool");
    const received: JsonValue[] = [];
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("tool", async () => {
        const result = await wrappedTool.invoke({ claimId: "claim-1" });
        received.push(
          (Array.isArray(result) ? result[0] : result) as JsonValue,
        );
        return { messages: [new AIMessage({ content: "tool resumed" })] };
      })
      .addEdge(START, "tool")
      .addEdge("tool", END)
      .compile({ checkpointer: new MemorySaver() });
    const agent = new LangGraphBaseAgent(graph as never);
    const fetchMock = vi.fn(
      async (input: string | URL | Request, _init?: RequestInit) => {
        const url = String(input);
        const body = url.endsWith("/tool/result")
          ? { ok: true }
          : {
              proceed: true,
              route_to: "callback",
              latest_step_number: 1,
            };
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    try {
      await runWithExecutionContext(
        { executionId: "exec-tool", wrapper, oeUrl: "http://oe:8000" },
        async () => {
          const { data, metadata } = await suspend(
            agent,
            "local-tool-native-interrupt",
          );
          const interrupts = data["interrupts"] as Array<{
            id: string;
            value: JsonValue;
          }>;
          const pendingInterrupt = mustGet(interrupts[0]);
          expect(pendingInterrupt.value).toEqual({
            claim_id: "claim-1",
            question: "approve?",
          });
          const decision = { decision: "approve" };

          const events = await resume(
            agent,
            "local-tool-native-interrupt",
            metadata,
            { [pendingInterrupt.id]: decision },
          );

          expect(events.map((event) => event.event)).toEqual(["result"]);
          expect(received).toEqual([decision]);
        },
      );

      const executeCalls = fetchMock.mock.calls.filter(([request]) =>
        String(request).endsWith("/tool/execute"),
      );
      expect(executeCalls).toHaveLength(2);
      for (const [, init] of executeCalls) {
        const body = JSON.parse(String(init?.body)) as { is_local?: boolean };
        expect(body.is_local).toBe(true);
      }
      expect(
        fetchMock.mock.calls.filter(([request]) =>
          String(request).endsWith("/tool/result"),
        ),
      ).toHaveLength(2);
      const resultStatuses = fetchMock.mock.calls
        .filter(([request]) => String(request).endsWith("/tool/result"))
        .map(
          ([, init]) =>
            (JSON.parse(String(init?.body)) as { status: string }).status,
        );
      expect(resultStatuses).toEqual(["interrupted", "success"]);
    } finally {
      vi.unstubAllGlobals();
      resetHooks();
      if (previousMode === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = previousMode;
    }
  });

  it("settles non-interrupt LangGraph bubble-up control flow as interrupted", async () => {
    const previousMode = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
    const parentCommand = new ParentCommand(
      new Command({ graph: Command.PARENT, goto: "parent_node" }),
    );
    const app = new App({ appName: "Parent command local tool test" });
    app.tool({
      description: "Route control to a parent graph.",
      schema: z.object({}),
    })(function route_to_parent(): never {
      throw parentCommand;
    });
    const wrappedTool = app.getTools()[0] as {
      invoke: (input: unknown) => Promise<unknown>;
    };
    const wrapper = new SecureToolWrapper("http://oe:8000", "exec-tool");
    const fetchMock = vi.fn(
      async (input: string | URL | Request, _init?: RequestInit) =>
        new Response(
          JSON.stringify(
            String(input).endsWith("/tool/result")
              ? { ok: true }
              : { proceed: true, route_to: "callback" },
          ),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);

    try {
      await runWithExecutionContext(
        { executionId: "exec-tool", wrapper, oeUrl: "http://oe:8000" },
        async () => {
          await expect(wrappedTool.invoke({})).rejects.toBe(parentCommand);
        },
      );
      const resultCall = mustGet(
        fetchMock.mock.calls.find(([request]) =>
          String(request).endsWith("/tool/result"),
        ),
      );
      const resultBody = JSON.parse(String(resultCall[1]?.body)) as {
        status: string;
      };
      expect(resultBody.status).toBe("interrupted");
    } finally {
      vi.unstubAllGlobals();
      resetHooks();
      if (previousMode === undefined) delete process.env["RUNNER_MODE"];
      else process.env["RUNNER_MODE"] = previousMode;
    }
  });

  it("resumes parallel branches in one map", async () => {
    const received: Record<string, JsonValue> = {};
    const builder = new StateGraph(MessagesAnnotation);
    for (const name of ["alpha", "beta", "gamma"]) {
      builder
        .addNode(name, async () => {
          received[name] = interrupt({ branch: name }) as JsonValue;
          return {
            messages: [new AIMessage({ content: `resumed:${name}` })],
          };
        })
        .addEdge(START, name)
        .addEdge(name, END);
    }
    const agent = new LangGraphBaseAgent(
      builder.compile({ checkpointer: new MemorySaver() }) as never,
    );

    const { data, metadata } = await suspend(agent, "parallel");
    const interrupts = data["interrupts"] as Array<{
      id: string;
      value: { branch: string };
    }>;
    expect(interrupts.map((item) => item.value.branch).sort()).toEqual([
      "alpha",
      "beta",
      "gamma",
    ]);
    const resumeMap = Object.fromEntries(
      interrupts.map((item, index) => [
        item.id,
        {
          branch: item.value.branch,
          position: index,
          approved: index % 2 === 0,
        },
      ]),
    ) as Record<string, JsonValue>;

    const events = await resume(agent, "parallel", metadata, resumeMap);

    expect(events.map((event) => event.event)).toEqual(["result"]);
    expect(received).toEqual(
      Object.fromEntries(
        Object.values(resumeMap).map((answer) => [
          (answer as { branch: string }).branch,
          answer,
        ]),
      ),
    );
  });
});

describe("LangGraphBaseAgent nested subgraph native interrupt", () => {
  function nestedInterruptAgent(
    payload: JsonValue,
    received: JsonValue[],
  ): LangGraphBaseAgent {
    const child = new StateGraph(MessagesAnnotation)
      .addNode("gate", async () => {
        const answer = interrupt(payload) as JsonValue;
        received.push(answer);
        return { messages: [new AIMessage({ content: "resumed" })] };
      })
      .addEdge(START, "gate")
      .addEdge("gate", END)
      .compile();
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("nested", child)
      .addEdge(START, "nested")
      .addEdge("nested", END)
      .compile({ checkpointer: new MemorySaver() });
    return new LangGraphBaseAgent(graph as never);
  }

  it("projects a nested interrupt once and resumes the child occurrence", async () => {
    const received: JsonValue[] = [];
    const agent = nestedInterruptAgent(
      { question: "nested approve?" },
      received,
    );
    const sessionId = "nested-native-interrupt";
    const suspended = await suspend(agent, sessionId);
    const interrupts = suspended.data["interrupts"] as Array<{
      id: string;
      value: unknown;
    }>;
    expect(interrupts).toHaveLength(1);
    expect(interrupts[0]?.value).toEqual({ question: "nested approve?" });

    const events = await resume(
      agent,
      sessionId,
      suspended.metadata,
      "approved",
    );
    expect(events.at(-1)?.event).toBe("result");
    expect(received).toEqual(["approved"]);
  });

  it("projects nested interrupt values once in invoke mode", async () => {
    const received: JsonValue[] = [];
    const agent = nestedInterruptAgent(
      { question: "invoke approve?" },
      received,
    );
    const sessionId = "nested-native-invoke";
    const output = mustGet(
      await agent
        .invoke({ sessionId } as RequestContext, input)
        .then((result) => result.response as Record<string, unknown>),
    );
    expect(output["status"]).toBe("suspended");
    const suspendContext = output["suspend_context"] as {
      interrupt_values: unknown[];
    };
    expect(suspendContext.interrupt_values).toHaveLength(1);
    expect(suspendContext.interrupt_values[0]).toEqual({
      question: "invoke approve?",
    });
  });

  it("rejects conflicting values for one interrupt id in invoke mode", async () => {
    const graph = {
      invoke: vi.fn(async () => ({ messages: [new AIMessage("waiting")] })),
      getState: vi.fn(async () => ({
        next: ["waiter"],
        tasks: [
          {
            path: ["__pregel_pull", "waiter"],
            interrupts: [{ id: "wait-1", value: { question: "prompt A" } }],
          },
          {
            path: ["__pregel_pull", "other"],
            interrupts: [{ id: "wait-1", value: { question: "prompt B" } }],
          },
        ],
      })),
      checkpointer: new MemorySaver(),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    await expect(
      agent.invoke(
        { sessionId: "conflicting-invoke" } as RequestContext,
        input,
      ),
    ).rejects.toThrow("LangGraph returned duplicate interrupt ids");
    expect(graph.invoke).toHaveBeenCalledOnce();
  });
});
