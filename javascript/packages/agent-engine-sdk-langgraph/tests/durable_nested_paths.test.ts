/**
 * Durable nested operation paths across compiled subgraphs, reused graphs,
 * and nested suspension replay, including pauses in Send workers.
 *
 * Ports the replay cases of Python's `test_durable_subgraphs_integration.py`.
 * Send fan-out replay itself is covered in `durable_send.test.ts`.
 */

import { create } from "@bufbuild/protobuf";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Annotation,
  END,
  MessagesAnnotation,
  Send,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import type { AgentInput, StreamEvent } from "@mongodb-js/agent-engine-sdk";
import {
  ActivityKind,
  ActivityDispatch,
  ActivityReplay,
  ActivityContextSchema,
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runSerialActivity,
  runWithAttemptContext,
  runWithExecutionContext,
  runWithOperationPathResolver,
  type ActivityCommand,
  type ActivityOutcome,
} from "@mongodb-js/agent-engine-runner-shared";

import { LangGraphBaseAgent } from "../src/agent.js";
import { DurableSubgraphResolver } from "../src/durable_subgraphs.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";

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
} as never as Parameters<LangGraphBaseAgent["invoke"]>[0];

type Path = ReadonlyArray<readonly [string, string]>;

function attempt(attemptId: string, replay = false) {
  return create(AttemptContextSchema, {
    attemptId,
    fencingToken: replay ? 8n : 7n,
    replayMode: replay,
    workflowIdentity: identity,
  });
}

function pathKey(command: ActivityCommand): Path {
  return (command.position?.operationPath?.segments ?? []).map((segment) => [
    segment.name,
    segment.ordinal.toString(),
  ]);
}

/** Port of Python's `_ReplayClient`: records commands, replays by position. */
class ReplayClient {
  readonly commands: ActivityCommand[] = [];
  private readonly outcomes = new Map<string, ActivityOutcome>();
  private readonly positionsByActivity = new Map<string, Path>();

  startActivity = async (
    command: ActivityCommand,
  ): Promise<ActivityDispatch | ActivityReplay> => {
    this.commands.push(command);
    const key = JSON.stringify(pathKey(command));
    const outcome = this.outcomes.get(key);
    if (outcome !== undefined) return new ActivityReplay(outcome);
    const activityId = `activity-${this.commands.length}`;
    this.positionsByActivity.set(activityId, pathKey(command));
    return new ActivityDispatch(
      create(ActivityContextSchema, {
        workflowIdentity: command.workflowIdentity,
        activityId,
        attemptId: command.attemptId,
        fencingToken: command.fencingToken,
      }),
    );
  };

  reportOutcome = async (outcome: ActivityOutcome): Promise<void> => {
    const activityId = outcome.activityId;
    const position = this.positionsByActivity.get(activityId);
    if (position === undefined) return;
    this.outcomes.set(JSON.stringify(position), outcome);
  };
}

function activityNode(
  client: ReplayClient,
  name: string,
  effects: string[],
): () => Promise<Record<string, never>> {
  return async () => {
    await runSerialActivity({
      client,
      kind: ActivityKind.TOOL,
      name,
      activityOrdinal: 1,
      semanticInput: {},
      execute: () => {
        effects.push(name);
        return { source: name };
      },
    });
    return {};
  };
}

function nestedGraph(client: ReplayClient, effects: string[]) {
  const policy = new StateGraph(MessagesAnnotation)
    .addNode("read_policy", activityNode(client, "read_policy", effects))
    .addEdge(START, "read_policy")
    .addEdge("read_policy", END)
    .compile();

  const investigation = new StateGraph(MessagesAnnotation)
    .addNode(
      "collect_evidence",
      activityNode(client, "collect_evidence", effects),
    )
    .addNode("policy_analysis", policy)
    .addEdge(START, "collect_evidence")
    .addEdge("collect_evidence", "policy_analysis")
    .addEdge("policy_analysis", END)
    .compile();

  return new StateGraph(MessagesAnnotation)
    .addNode(
      "coordinate_case",
      activityNode(client, "coordinate_case", effects),
    )
    .addNode("case_investigation", investigation)
    .addEdge(START, "coordinate_case")
    .addEdge("coordinate_case", "case_investigation")
    .addEdge("case_investigation", END)
    .compile();
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("durable compiled-subgraph paths", () => {
  it("reconstructs child and grandchild paths on replay", async () => {
    const client = new ReplayClient();
    const effects: string[] = [];
    const graph = nestedGraph(client, effects);
    const resolver = new DurableSubgraphResolver(graph as never);

    for (const current of [attempt("attempt-1"), attempt("attempt-2", true)]) {
      await runWithAttemptContext(current, async () =>
        runWithOperationPathResolver(resolver.resolve, () =>
          graph.invoke({ messages: [] }, { configurable: { thread_id: "t" } }),
        ),
      );
    }

    const expected: Path[] = [
      [["agent", "1"]],
      [
        ["agent", "1"],
        ["case_investigation", "1"],
      ],
      [
        ["agent", "1"],
        ["case_investigation", "1"],
        ["policy_analysis", "1"],
      ],
    ];
    expect(client.commands.map(pathKey)).toEqual([...expected, ...expected]);
    // The replacement attempt replays recorded outcomes without re-running
    // the effects.
    expect(effects).toEqual([
      "coordinate_case",
      "collect_evidence",
      "read_policy",
    ]);
  });

  it("keeps reused graph occurrences distinct and stable across replay", async () => {
    const client = new ReplayClient();
    const effects: string[] = [];
    const research = new StateGraph(MessagesAnnotation)
      .addNode("tools", activityNode(client, "tools", effects))
      .addEdge(START, "tools")
      .addEdge("tools", END)
      .compile();

    const graph = new StateGraph(MessagesAnnotation)
      .addNode("seed", activityNode(client, "seed", effects))
      .addNode("research_a", research)
      .addNode("research_b", research)
      .addEdge(START, "seed")
      .addEdge("seed", "research_a")
      .addEdge("research_a", "research_b")
      .addEdge("research_b", END)
      .compile();
    const resolver = new DurableSubgraphResolver(graph as never);

    for (const current of [attempt("attempt-1"), attempt("attempt-2", true)]) {
      await runWithAttemptContext(current, async () =>
        runWithOperationPathResolver(resolver.resolve, () =>
          graph.invoke({ messages: [] }, { configurable: { thread_id: "t" } }),
        ),
      );
    }

    expect(client.commands.map(pathKey)).toEqual([
      [["agent", "1"]],
      [
        ["agent", "1"],
        ["research_a", "1"],
      ],
      [
        ["agent", "1"],
        ["research_b", "1"],
      ],
      [["agent", "1"]],
      [
        ["agent", "1"],
        ["research_a", "1"],
      ],
      [
        ["agent", "1"],
        ["research_b", "1"],
      ],
    ]);
    expect(effects).toEqual(["seed", "tools", "tools"]);
  });
});

describe("durable nested suspension replay", () => {
  function nestedInterruptAgent(saver: PlatformCheckpointer) {
    const child = new StateGraph(MessagesAnnotation)
      .addNode("waiter", async () => {
        interrupt({ question: "child approve?" });
        return { messages: [] };
      })
      .addEdge(START, "waiter")
      .addEdge("waiter", END)
      .compile();

    const graph = new StateGraph(MessagesAnnotation)
      .addNode("nested", child)
      .addEdge(START, "nested")
      .addEdge("nested", END)
      .compile({ checkpointer: saver as never });
    return new LangGraphBaseAgent(graph as never);
  }

  async function collect(iterable: AsyncIterable<StreamEvent>) {
    const events: StreamEvent[] = [];
    for await (const event of iterable) events.push(event);
    return events;
  }

  type SuspensionBody = {
    suspensions?: {
      position: {
        operation_path?: { segments?: { name: string }[] };
        activity_ordinal?: string | number;
      };
    }[];
  };

  /** Stubs OE settlement; each suspension settles as `settlement.kind`. */
  function stubOeSettlement() {
    const settlement = {
      kind: "suspended" as "suspended" | "completed",
      bodies: [] as SuspensionBody[],
      /** The resume value OE returns for a completed suspension. */
      result: (_segment: string): unknown => "approved",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: unknown, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body)) as {
          suspensions?: { position: Record<string, unknown> }[];
          workflow_identity?: unknown;
          attempt_id?: string;
          fencing_token?: string;
        };
        settlement.bodies.push(body as SuspensionBody);
        return new Response(
          JSON.stringify({
            entries: (body.suspensions ?? []).map((suspension, index) => ({
              position: suspension.position,
              outcome: {
                workflow_identity: body.workflow_identity,
                activity_id: `wait-${index + 1}`,
                attempt_id: body.attempt_id,
                fencing_token: body.fencing_token,
                outcome_kind:
                  settlement.kind === "suspended"
                    ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
                    : "ACTIVITY_OUTCOME_KIND_COMPLETED",
                ...(settlement.kind === "completed"
                  ? {
                      result: settlement.result(
                        suspendedSegments({ suspensions: [suspension] })[0] ??
                          "",
                      ),
                    }
                  : {}),
              },
            })),
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }),
    );
    return settlement;
  }

  function suspendedSegments(body: SuspensionBody | undefined): string[] {
    return (
      body?.suspensions?.[0]?.position.operation_path?.segments ?? []
    ).map((segment) => segment.name);
  }

  function platformSaver() {
    const completeExecution = vi.fn(async () => undefined);
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: () => ({
        finalizeStep: vi.fn(async () => []),
        completeExecution,
      }),
    });
    return { saver, completeExecution };
  }

  async function inAttempt<T>(
    attemptId: string,
    replay: boolean,
    fn: () => Promise<T>,
  ): Promise<T> {
    return runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () => runWithAttemptContext(attempt(attemptId, replay), fn),
    );
  }

  const resumeContext = {
    ...baseContext,
    resume: true,
    resumeData: { "wait-1": "approved" },
  } as never;

  it("suspends inside a nested graph, records the child occurrence, and resumes", async () => {
    const { saver, completeExecution } = platformSaver();
    const agent = nestedInterruptAgent(saver);
    const settlement = stubOeSettlement();

    await inAttempt("attempt-1", false, async () => {
      const events = await collect(agent.stream(baseContext, input));
      expect(events.map((event) => event.event)).toEqual(["suspend"]);
      const data = events[0]?.data as {
        interrupts?: Array<{ id: string; position?: unknown }>;
      };
      expect(data.interrupts).toEqual([
        { id: "wait-1", value: { question: "child approve?" } },
      ]);
    });
    // The synthetic interrupt position is the root hosting task path —
    // the occurrence that owns the suspended child.
    expect(suspendedSegments(settlement.bodies[0])).toEqual([
      'langgraph.task:["__pregel_pull","nested"]',
    ]);
    // A single pause inside a compiled child is the task's first.
    expect(
      Number(settlement.bodies[0]?.suspensions?.[0]?.position.activity_ordinal),
    ).toBe(1);

    settlement.kind = "completed";
    await inAttempt("attempt-2", true, async () => {
      const events = await collect(agent.stream(resumeContext, input));
      expect(events[events.length - 1]?.event).toBe("result");
    });
    expect(completeExecution).toHaveBeenCalledTimes(1);
  });

  it("suspends and resumes a pause inside a Send-dispatched task", async () => {
    // LangChain agents dispatch each tool call with Send, so a
    // human-in-the-loop pause inside a tool runs in a PUSH task.
    const { saver, completeExecution } = platformSaver();
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("fan", async () => ({ messages: [] }))
      .addNode("worker", async () => {
        const answer = interrupt({ question: "send approve?" }) as string;
        return { messages: [new AIMessage({ content: `worker:${answer}` })] };
      })
      .addEdge(START, "fan")
      .addConditionalEdges("fan", () => [new Send("worker", { messages: [] })])
      .addEdge("worker", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    const settlement = stubOeSettlement();

    await inAttempt("attempt-1", false, async () => {
      const events = await collect(agent.stream(baseContext, input));
      expect(events.map((event) => event.event)).toEqual(["suspend"]);
    });
    // LangGraph 1.5 appends `false` to the Send task path.
    expect(suspendedSegments(settlement.bodies[0])).toEqual([
      expect.stringMatching(/^langgraph\.task:\["__pregel_push",0(,false)?\]$/),
    ]);

    settlement.kind = "completed";
    await inAttempt("attempt-2", true, async () => {
      const events = await collect(agent.stream(resumeContext, input));
      expect(events[events.length - 1]?.event).toBe("result");
      expect(JSON.stringify(events[events.length - 1]?.data)).toContain(
        "worker:approved",
      );
    });
    expect(completeExecution).toHaveBeenCalledTimes(1);
  });

  it("pauses and resumes each Send worker with its own answer", async () => {
    const { saver, completeExecution } = platformSaver();
    const Worker = Annotation.Root({
      ...MessagesAnnotation.spec,
      caseId: Annotation<string>,
    });
    const graph = new StateGraph(Worker)
      .addNode("worker", async (state: typeof Worker.State) => {
        const answer = interrupt({ question: `approve ${state.caseId}?` });
        return {
          messages: [new AIMessage(`${state.caseId}:${String(answer)}`)],
        };
      })
      .addConditionalEdges(
        START,
        () =>
          ["billing", "access"].map((caseId) => new Send("worker", { caseId })),
        ["worker"],
      )
      .addEdge("worker", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    const settlement = stubOeSettlement();

    await inAttempt("attempt-1", false, async () => {
      const events = await collect(agent.stream(baseContext, input));
      expect(events.map((event) => event.event)).toEqual(["suspend"]);
    });
    const paused = (settlement.bodies[0]?.suspensions ?? []).map(
      (suspension) => suspendedSegments({ suspensions: [suspension] })[0] ?? "",
    );
    expect(paused.sort()).toEqual([
      expect.stringMatching(/^langgraph\.task:\["__pregel_push",0(,false)?\]$/),
      expect.stringMatching(/^langgraph\.task:\["__pregel_push",1(,false)?\]$/),
    ]);

    // Each worker's pause settles with an answer naming its own task index.
    settlement.kind = "completed";
    settlement.result = (segment) =>
      `answer-${segment.includes('"__pregel_push",0') ? 0 : 1}`;
    await inAttempt("attempt-2", true, async () => {
      const events = await collect(
        agent.stream(
          {
            ...baseContext,
            resume: true,
            resumeData: { "wait-1": "unused", "wait-2": "unused" },
          } as never,
          input,
        ),
      );
      expect(events[events.length - 1]?.event).toBe("result");
      const result = JSON.stringify(events[events.length - 1]?.data);
      // Send index 0 is billing and index 1 is access.
      expect(result).toContain("billing:answer-0");
      expect(result).toContain("access:answer-1");
    });
    expect(completeExecution).toHaveBeenCalledTimes(1);
  });

  /**
   * A graph whose only task pauses at `path`, as getState reports it. The
   * stream event and getState snapshot mirror what the adapter reads from
   * LangGraph; the real-Send test above is the check that they still match.
   */
  function pausedTaskAgent(saver: PlatformCheckpointer, path: unknown[]) {
    const pause = { id: "approval", value: { question: "Proceed?" } };
    const graph = {
      checkpointer: saver,
      stream: vi.fn(async () =>
        (async function* () {
          yield [[], "updates", { __interrupt__: [pause] }];
        })(),
      ),
      getState: vi.fn(async () => ({
        next: ["worker"],
        tasks: [{ path, interrupts: [pause] }],
      })),
    };
    return new LangGraphBaseAgent(graph as never);
  }

  it.each([
    {
      kind: "Send task",
      path: ["__pregel_push", 0, false],
      segment: 'langgraph.task:["__pregel_push",0,false]',
    },
    {
      kind: "functional-API call",
      path: ["__pregel_push", ["__pregel_pull", "agent"], 0, true],
      segment:
        'langgraph.task:["__pregel_push",["__pregel_pull","agent"],0,true]',
    },
  ])(
    "keys a pause in a $kind by its full task path",
    async ({ path, segment }) => {
      const { saver } = platformSaver();
      const agent = pausedTaskAgent(saver, path);
      const settlement = stubOeSettlement();

      await inAttempt("attempt-1", false, async () => {
        const events = await collect(agent.stream(baseContext, input));
        expect(events.map((event) => event.event)).toEqual(["suspend"]);
      });
      expect(suspendedSegments(settlement.bodies[0])).toEqual([segment]);
    },
  );

  it.each([
    ["a fractional index", ["__pregel_push", 0.5]],
    ["an object nested in the parent path", ["__pregel_push", [{}], 0, true]],
  ])("rejects a pause whose task path has %s", async (_case, path) => {
    const { saver, completeExecution } = platformSaver();
    const agent = pausedTaskAgent(saver, path);
    const settlement = stubOeSettlement();

    await expect(
      inAttempt("attempt-1", false, () =>
        collect(agent.stream(baseContext, input)),
      ),
    ).rejects.toThrow(/requires a stable LangGraph task path/);
    expect(settlement.bodies).toEqual([]);
    expect(completeExecution).not.toHaveBeenCalled();
  });
});
