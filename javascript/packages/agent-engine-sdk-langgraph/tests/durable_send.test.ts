/**
 * Durable LangGraph `Send` fan-out: each worker replays its own recorded
 * model and tool results when a replacement attempt starts the workers in a
 * different order.
 *
 * Ports the `Send` cases of Python's `test_durable_subgraphs_integration.py`.
 * The model and tool wrappers run for real against a fake OE that records
 * activity outcomes by position and replays them on later attempts.
 */

import { create } from "@bufbuild/protobuf";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  HumanMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import {
  Annotation,
  END,
  MessagesAnnotation,
  START,
  Send,
  StateGraph,
} from "@langchain/langgraph";
import type {
  AgentInput,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import {
  AttemptContextSchema,
  SecureLLMProxy,
  SecureToolWrapper,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
  runWithExecutionContext,
  type CompleteExecutionCommand,
} from "@mongodb-js/agent-engine-runner-shared";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LangGraphBaseAgent } from "../src/agent.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import { SecureWrappedLLM } from "../src/secure_llm.js";
import { stateSnapshotToChannelValues } from "../src/workflow_state.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});
const input: AgentInput = { payload: { message: "start" } };
const baseContext = {
  sessionId: "session",
  workspaceId: "workspace",
} as RequestContext;
const CASES = ["billing", "access"];

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function attempt(attemptId: string, replay: boolean) {
  return create(AttemptContextSchema, {
    attemptId,
    fencingToken: replay ? 8n : 7n,
    replayMode: replay,
    workflowIdentity: identity,
  });
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

interface ActivityStart {
  readonly kind: string;
  readonly name: string;
  readonly path: string[];
  readonly ordinal: number;
}

/** OE fake: records activity outcomes by position and replays them. */
function fakeOe() {
  const starts: ActivityStart[] = [];
  const recorded = new Map<string, Record<string, unknown>>();
  const positionByActivity = new Map<string, string>();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
      const url = String(request);
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      if (url.endsWith("/executor/activity/start")) {
        const position = body["position"] as {
          operation_path?: { segments?: { name: string; ordinal: string }[] };
          activity_ordinal: string;
        };
        starts.push({
          kind: String(body["activity_kind"]),
          name: String(body["activity_name"]),
          path: (position.operation_path?.segments ?? []).map(
            (segment) => `${segment.name}/${segment.ordinal}`,
          ),
          ordinal: Number(position.activity_ordinal),
        });
        const key = JSON.stringify(position);
        const provenance = {
          workflow_identity: body["workflow_identity"],
          attempt_id: body["attempt_id"],
          fencing_token: body["fencing_token"],
        };
        const outcome = recorded.get(key);
        if (outcome !== undefined) {
          return jsonResponse({ outcome: { ...outcome, ...provenance } });
        }
        const activityId = `activity-${positionByActivity.size + 1}`;
        positionByActivity.set(activityId, key);
        return jsonResponse({
          activity_context: { ...provenance, activity_id: activityId },
        });
      }
      if (url.endsWith("/executor/activity/outcome")) {
        const key = positionByActivity.get(String(body["activity_id"]));
        if (key === undefined) throw new Error("unknown activity");
        recorded.set(key, body);
        return jsonResponse({});
      }
      if (url.endsWith("/tool/execute")) {
        return jsonResponse({ proceed: true, route_to: "callback" });
      }
      if (url.endsWith("/tool/result")) return jsonResponse({});
      throw new Error(`unexpected URL: ${url}`);
    }),
  );
  return { starts };
}

/**
 * Make the worker named first in each round call its model first. A case id
 * is `<worker>` or `<worker>#<round>`.
 */
class OrderGate {
  private first = "";
  private readonly released = new Map<
    string,
    { promise: Promise<void>; resolve: () => void }
  >();

  reset(first: string): void {
    this.first = first;
    this.released.clear();
  }

  async turn<T>(caseId: string, fn: () => Promise<T>): Promise<T> {
    const [name, round = ""] = caseId.split("#");
    let released = this.released.get(round);
    if (released === undefined) {
      let resolve = () => {};
      const promise = new Promise<void>((done) => {
        resolve = done;
      });
      released = { promise, resolve };
      this.released.set(round, released);
    }
    if (name !== this.first) await released.promise;
    try {
      return await fn();
    } finally {
      if (name === this.first) released.resolve();
    }
  }
}

/**
 * A durable model whose provider replies `resolved:<prompt>`, or with one
 * ToolCall when `toolCall` names one for the prompt.
 */
function durableModel(
  dispatched: string[],
  toolCall: (prompt: string) => { id: string; name: string } | null = () =>
    null,
) {
  vi.spyOn(
    SecureLLMProxy.prototype as unknown as {
      streamLlm: (request: unknown) => AsyncGenerator<unknown>;
    },
    "streamLlm",
  ).mockImplementation(async function* (request: unknown) {
    const messages = (request as { messages: { content: unknown }[] }).messages;
    const prompt = String(messages.at(-1)?.content);
    dispatched.push(prompt);
    const call = toolCall(prompt);
    yield call === null
      ? { content: `resolved:${prompt}` }
      : { toolCalls: [{ ...call, args: "{}", index: 0 }] };
  });
  const holder = { wrapper: new SecureToolWrapper("http://oe", "execution") };
  const llm = new SecureWrappedLLM(
    {
      model: "test-model",
      stream: () => undefined,
    } as unknown as BaseChatModel,
    () => holder.wrapper,
  );
  return { llm, holder };
}

function platformSaver() {
  const completeExecution = vi.fn(
    async (_command: CompleteExecutionCommand) => undefined,
  );
  const saver = new PlatformCheckpointer({
    native: null,
    clientFactory: () => ({
      finalizeStep: vi.fn(async () => []),
      completeExecution,
    }),
  });
  return { saver, completeExecution };
}

type Mode = "invoke" | "stream";

/**
 * Run the turn twice: a first attempt, then a replacement attempt that starts
 * the workers in reverse order. Each attempt is a fresh process, so the tool
 * wrapper's operational steps restart.
 */
async function runAttempts(
  agent: LangGraphBaseAgent,
  gate: OrderGate,
  holder: { wrapper: SecureToolWrapper },
  mode: Mode = "invoke",
): Promise<void> {
  for (const [first, current] of [
    ["billing", attempt("attempt-1", false)],
    ["access", attempt("attempt-2", true)],
  ] as const) {
    gate.reset(first);
    holder.wrapper = new SecureToolWrapper("http://oe", "execution");
    await runWithExecutionContext(
      {
        executionId: "execution",
        wrapper: holder.wrapper,
        oeUrl: "http://oe",
        userId: "user-1",
      },
      () =>
        runWithAttemptContext(current, async () => {
          if (mode === "invoke") {
            await agent.invoke(baseContext, input);
            return;
          }
          const events: StreamEvent[] = [];
          for await (const event of agent.stream(baseContext, input)) {
            events.push(event);
          }
          expect(events.at(-1)?.event).toBe("result");
        }),
    );
  }
}

const RoundState = Annotation.Root({
  ...MessagesAnnotation.spec,
  rounds: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0,
  }),
  done: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
});

const WorkState = Annotation.Root({
  caseId: Annotation<string>,
  done: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
});

type Shape = "plain" | "compiled" | "invoked";

/** A Send worker that asks the model about its case, in the given shape. */
function roundWorker(
  llm: SecureWrappedLLM,
  gate: OrderGate,
  answers: [string, string][],
  shape: Shape,
) {
  const ask = async (state: typeof WorkState.State) => {
    const reply = await gate.turn(state.caseId, () =>
      llm.invoke([new HumanMessage(`work for ${state.caseId}`)]),
    );
    answers.push([state.caseId, String(reply.content)]);
    return { done: [state.caseId] };
  };
  if (shape === "plain") return ask;
  const compiled = new StateGraph(WorkState)
    .addNode("ask", ask)
    .addEdge(START, "ask")
    .addEdge("ask", END)
    .compile();
  if (shape === "compiled") return compiled;
  // A plain worker that runs its own graph, outside the compiled topology.
  return async (state: typeof WorkState.State) => ({
    done: (await compiled.invoke(state)).done,
  });
}

function sendRound(round: number): Send[] {
  return CASES.map((name) => new Send("work", { caseId: `${name}#${round}` }));
}

/**
 * A root graph whose compiled coordinator runs two `Send` rounds of `worker`.
 */
function coordinatedRounds(worker: unknown, saver: PlatformCheckpointer) {
  const coordinator = new StateGraph(RoundState)
    .addNode("work", worker as never)
    .addNode("gather", (state: typeof RoundState.State) => ({
      rounds: state.rounds + 1,
    }))
    .addConditionalEdges(START, () => sendRound(1), ["work"])
    .addEdge("work", "gather")
    .addConditionalEdges(
      "gather",
      (state: typeof RoundState.State) =>
        state.rounds < 2 ? sendRound(2) : END,
      ["work", END],
    )
    .compile();
  return new StateGraph(RoundState)
    .addNode("coordinate", coordinator)
    .addEdge(START, "coordinate")
    .addEdge("coordinate", END)
    .compile({ checkpointer: saver as never });
}

function expectEachWorkerGotItsOwnReply(
  answers: [string, string][],
  dispatched: string[],
  cases: string[],
): void {
  const sorted = (items: unknown[]) =>
    items.map((item) => JSON.stringify(item)).sort();
  // Both attempts saw every worker's own reply...
  expect(sorted(answers)).toEqual(
    sorted([...cases, ...cases].map((id) => [id, `resolved:work for ${id}`])),
  );
  // ...while the provider ran once per worker.
  expect(sorted(dispatched)).toEqual(
    sorted(cases.map((id) => `work for ${id}`)),
  );
}

describe("durable Send workers", () => {
  describe.each(["invoke", "stream"] as const)("%s", (mode) => {
    it.each(["plain", "compiled", "invoked"] as const)(
      "replay their own model results in any start order (%s worker)",
      async (shape) => {
        const oe = fakeOe();
        const dispatched: string[] = [];
        const { llm, holder } = durableModel(dispatched);
        const gate = new OrderGate();
        const answers: [string, string][] = [];
        const { saver } = platformSaver();
        const graph = new StateGraph(RoundState)
          .addNode("work", roundWorker(llm, gate, answers, shape) as never)
          // Streaming requires the turn to end with a message.
          .addNode("finish", () => ({ messages: [new AIMessage("finished")] }))
          .addConditionalEdges(START, () => sendRound(1), ["work"])
          .addEdge("work", "finish")
          .addEdge("finish", END)
          .compile({ checkpointer: saver as never });

        await runAttempts(
          new LangGraphBaseAgent(graph as never),
          gate,
          holder,
          mode,
        );

        expectEachWorkerGotItsOwnReply(answers, dispatched, [
          "billing#1",
          "access#1",
        ]);
        expect(
          new Set(oe.starts.map((start) => start.path.slice(0, 2).join(" "))),
        ).toEqual(
          new Set([
            "agent/1 langgraph.task:__pregel_push:1:0/1",
            "agent/1 langgraph.task:__pregel_push:1:1/1",
          ]),
        );
      },
    );
  });

  it.each(["plain", "compiled"] as const)(
    "replay per worker across Send rounds inside a compiled coordinator (%s worker)",
    async (shape) => {
      fakeOe();
      const dispatched: string[] = [];
      const { llm, holder } = durableModel(dispatched);
      const gate = new OrderGate();
      const answers: [string, string][] = [];
      const { saver } = platformSaver();
      const graph = coordinatedRounds(
        roundWorker(llm, gate, answers, shape),
        saver,
      );

      await runAttempts(new LangGraphBaseAgent(graph as never), gate, holder);

      expectEachWorkerGotItsOwnReply(answers, dispatched, [
        "billing#1",
        "access#1",
        "billing#2",
        "access#2",
      ]);
    },
  );

  // A worker that runs its own graph inside a compiled parent is outside the
  // compiled topology, so it cannot be named and the turn fails before any
  // worker records an activity, as it did before Send support.
  it("reject a worker that runs its own graph inside a compiled coordinator", async () => {
    const oe = fakeOe();
    const dispatched: string[] = [];
    const { llm, holder } = durableModel(dispatched);
    const gate = new OrderGate();
    const { saver, completeExecution } = platformSaver();
    const graph = coordinatedRounds(
      roundWorker(llm, gate, [], "invoked"),
      saver,
    );

    await expect(
      runAttempts(new LangGraphBaseAgent(graph as never), gate, holder),
    ).rejects.toThrow(/unsupported LangGraph checkpoint namespace/);
    expect(oe.starts).toEqual([]);
    expect(dispatched).toEqual([]);
    expect(completeExecution).not.toHaveBeenCalled();
  });

  it("keep distinct message ids that replay unchanged", async () => {
    fakeOe();
    const { saver, completeExecution } = platformSaver();
    const graph = new StateGraph(RoundState)
      .addNode("work", (state: typeof WorkState.State) => ({
        // No id: the adapter derives one from the worker's operation path.
        messages: [new AIMessage(`done ${state.caseId}`)],
      }))
      .addConditionalEdges(START, () => sendRound(1), ["work"])
      .addEdge("work", END)
      .compile({ checkpointer: saver as never });

    await runAttempts(new LangGraphBaseAgent(graph as never), new OrderGate(), {
      wrapper: new SecureToolWrapper("http://oe", "execution"),
    });

    const workerMessages = (call: number): Record<string, string> => {
      const state = completeExecution.mock.calls[call]?.[0].state;
      if (state === undefined) throw new Error("missing published state");
      const messages = stateSnapshotToChannelValues(state)[
        "messages"
      ] as BaseMessage[];
      return Object.fromEntries(
        messages
          .filter((message) => message.type === "ai")
          .map((message) => [String(message.content), String(message.id)]),
      );
    };
    const first = workerMessages(0);
    // LangGraph's messages reducer replaces a message whose id collides.
    expect(Object.keys(first).sort()).toEqual([
      "done access#1",
      "done billing#1",
    ]);
    expect(new Set(Object.values(first)).size).toBe(2);
    expect(workerMessages(1)).toEqual(first);
  });
});

const ToolState = Annotation.Root({
  ...MessagesAnnotation.spec,
  // Compiled workers return their whole state, so parallel workers write
  // these channels in the same step.
  caseId: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  call: Annotation<{ id: string; name: string } | null>({
    reducer: (_left, right) => right,
    default: () => null,
  }),
  calls: Annotation<{ id: string; name: string }[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  done: Annotation<string[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
});
type ToolStateValue = typeof ToolState.State;

describe("durable Send worker tools", () => {
  function toolHarness() {
    const oe = fakeOe();
    const effects: string[] = [];
    const { llm, holder } = durableModel(effects, (prompt) => {
      const name = prompt.split(" ").at(-1) as string;
      return prompt.startsWith("plan ")
        ? { id: `call-${name}`, name: `lookup_${name}` }
        : null;
    });
    const gate = new OrderGate();
    const results: [string, unknown][] = [];
    const plan = async (state: ToolStateValue) => {
      const reply = await gate.turn(state.caseId, () =>
        llm.invoke([new HumanMessage(`plan ${state.caseId}`)]),
      );
      return {
        calls: (reply.tool_calls ?? []).map((call) => ({
          id: String(call.id),
          name: call.name,
        })),
      };
    };
    const runTool = async (state: ToolStateValue) => {
      if (state.call === null) throw new Error("missing Send payload");
      const { id, name } = state.call;
      const result = await gate.turn(name.replace("lookup_", ""), () =>
        holder.wrapper.executeTool(
          name,
          {},
          {
            toolCallId: id,
            isLocal: true,
            localExecutor: () => {
              effects.push(name);
              return { answer: name };
            },
          },
        ),
      );
      results.push([name, result]);
      return { done: [name] };
    };
    const dispatch = (state: ToolStateValue) =>
      [...state.calls].reverse().map((call) => new Send("tool", { call }));
    return { oe, effects, holder, gate, results, plan, runTool, dispatch };
  }

  it.each([false, true])(
    "replay at their model's position from other Send tasks (compiled worker: %s)",
    async (compiledWorker) => {
      const h = toolHarness();
      const { saver } = platformSaver();
      let root;
      if (compiledWorker) {
        const worker = new StateGraph(ToolState)
          .addNode("plan", h.plan)
          .addNode("tool", h.runTool)
          .addEdge(START, "plan")
          .addConditionalEdges("plan", h.dispatch, ["tool"])
          .addEdge("tool", END)
          .compile();
        root = new StateGraph(ToolState)
          .addNode("work", worker)
          .addEdge("work", END);
      } else {
        // One task sends every worker's Tool, so the Tool runs in a task
        // other than its model's, at a different Send index.
        root = new StateGraph(ToolState)
          .addNode("work", h.plan)
          .addNode("collect", () => ({}))
          .addNode("tool", h.runTool)
          .addEdge("work", "collect")
          .addConditionalEdges("collect", h.dispatch, ["tool"])
          .addEdge("tool", END);
      }
      const graph = root
        .addConditionalEdges(
          START,
          () => CASES.map((caseId) => new Send("work", { caseId })),
          ["work"],
        )
        .compile({ checkpointer: saver as never });

      await runAttempts(
        new LangGraphBaseAgent(graph as never),
        h.gate,
        h.holder,
      );

      expect(h.effects.sort()).toEqual([
        "lookup_access",
        "lookup_billing",
        "plan access",
        "plan billing",
      ]);
      expect(h.results.map(([name]) => name).sort()).toEqual([
        "lookup_access",
        "lookup_access",
        "lookup_billing",
        "lookup_billing",
      ]);
      // Every Tool replays at the position its model reserved, in the
      // model's path, and returns the result its first run recorded.
      expect(
        new Set(
          h.results.map(([name, result]) =>
            [name, JSON.stringify(result)].join("="),
          ),
        ).size,
      ).toBe(2);
      const modelPaths = new Map<string, string>();
      for (const start of h.oe.starts) {
        if (start.kind === "ACTIVITY_KIND_LLM") {
          modelPaths.set(start.path.join(" "), start.path.join(" "));
        }
      }
      const tools = h.oe.starts.filter(
        (start) => start.kind === "ACTIVITY_KIND_TOOL",
      );
      expect(tools).toHaveLength(4);
      for (const tool of tools) {
        expect(modelPaths.has(tool.path.join(" "))).toBe(true);
        expect(tool.ordinal).toBe(2);
      }
    },
  );

  it("record where they run when no Send is involved", async () => {
    const h = toolHarness();
    const { saver } = platformSaver();
    const runTools = async (state: ToolStateValue) => {
      for (const call of state.calls) {
        await h.holder.wrapper.executeTool(
          call.name,
          {},
          { toolCallId: call.id, isLocal: true, localExecutor: () => "ok" },
        );
      }
      return { done: state.calls.map((call) => call.name) };
    };
    const planner = new StateGraph(ToolState)
      .addNode("plan", h.plan)
      .addEdge(START, "plan")
      .addEdge("plan", END)
      .compile();
    const executor = new StateGraph(ToolState)
      .addNode("run", runTools)
      .addEdge(START, "run")
      .addEdge("run", END)
      .compile();
    const graph = new StateGraph(ToolState)
      .addNode("planner", planner)
      .addNode("tools", executor)
      .addNode("seed", () => ({ caseId: "billing" }))
      .addEdge(START, "seed")
      .addEdge("seed", "planner")
      .addEdge("planner", "tools")
      .addEdge("tools", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);

    h.gate.reset("billing");
    await runWithExecutionContext(
      {
        executionId: "execution",
        wrapper: h.holder.wrapper,
        oeUrl: "http://oe",
        userId: "user-1",
      },
      () =>
        runWithAttemptContext(attempt("attempt-1", false), () =>
          agent.invoke(baseContext, input),
        ),
    );

    expect(
      h.oe.starts.map((start) => [
        start.kind,
        start.path.map((segment) => segment.split("/")[0]),
        start.ordinal,
      ]),
    ).toEqual([
      ["ACTIVITY_KIND_LLM", ["agent", "planner"], 1],
      ["ACTIVITY_KIND_TOOL", ["agent", "tools"], 1],
    ]);
  });
});
