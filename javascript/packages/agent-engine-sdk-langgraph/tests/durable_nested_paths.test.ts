/**
 * Durable nested operation paths across compiled subgraphs, reused graphs,
 * nested suspension replay, and the durable Send rejection.
 *
 * Ports the replay cases of Python's `test_durable_subgraphs_integration.py`
 * and `test_platform_checkpointer` Send rejection.
 */

import { create } from "@bufbuild/protobuf";
import { describe, expect, it, vi } from "vitest";
import {
  END,
  MessagesAnnotation,
  Send,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
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

describe("durable Send rejection", () => {
  it("fails closed when a durable checkpoint carries Send fan-out", async () => {
    const finalizeStep = vi.fn(async () => []);
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: () => ({
        finalizeStep,
        completeExecution: async () => undefined,
      }),
    });
    const baseCheckpoint: Record<string, unknown> = {
      v: 1,
      id: "chk-1",
      ts: "2026-09-14",
      channel_values: {},
      channel_versions: {},
      versions_seen: {},
      parents: {},
    };
    const config = { configurable: { thread_id: "t" } } as never;

    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ entries: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
      ),
    );
    await runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () =>
        runWithAttemptContext(attempt("attempt-1"), async () => {
          await expect(
            saver.put(
              config,
              baseCheckpoint as never,
              { source: "loop", step: 1, parents: {} },
              {},
            ),
          ).resolves.toBeDefined();
          const finalizedBefore = finalizeStep.mock.calls.length;
          await expect(
            saver.put(
              config,
              {
                ...baseCheckpoint,
                // Topic.checkpoint() stores the TASKS channel as
                // [seen, values]; the Send packets live in `values`.
                channel_values: {
                  __pregel_tasks: [[], [new Send("worker", { messages: [] })]],
                },
              } as never,
              { source: "loop", step: 1, parents: {} },
              {},
            ),
          ).rejects.toThrow(/LangGraph Send is not supported/);
          // The rejection fires before the producer step can finalize.
          expect(finalizeStep).toHaveBeenCalledTimes(finalizedBefore);
          await expect(
            saver.put(
              config,
              {
                ...baseCheckpoint,
                channel_values: {
                  __pregel_tasks: [new Send("worker", { messages: [] })],
                },
              } as never,
              { source: "loop", step: 1, parents: {} },
              {},
            ),
          ).rejects.toThrow(/LangGraph Send is not supported/);
          await expect(
            saver.putWrites(
              config,
              [
                ["__pregel_tasks", new Send("worker", { messages: [] })],
              ] as never,
              "task-1",
            ),
          ).rejects.toThrow(/LangGraph Send is not supported/);
          // A Send constructed from a different module copy of
          // @langchain/langgraph fails instanceof; the SendInterface shape
          // ({node, args}) must still reject.
          await expect(
            saver.put(
              config,
              {
                ...baseCheckpoint,
                channel_values: {
                  __pregel_tasks: [
                    ["seen-hash"],
                    [{ node: "worker", args: { messages: [] } }],
                  ],
                },
              } as never,
              { source: "loop", step: 1, parents: {} },
              {},
            ),
          ).rejects.toThrow(/LangGraph Send is not supported/);
          expect(finalizeStep).toHaveBeenCalledTimes(1);
        }),
    );
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

  it("suspends inside a nested graph, records the child occurrence, and resumes", async () => {
    let settlement: "suspended" | "completed" = "suspended";
    const finalizeStep = vi.fn(async () => []);
    const completeExecution = vi.fn(async () => undefined);
    const finalizeBodies: Record<string, unknown>[] = [];
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: () => ({ finalizeStep, completeExecution }),
    });
    const agent = nestedInterruptAgent(saver);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: unknown, init?: RequestInit) => {
        const body = JSON.parse(String(init?.body)) as {
          suspensions?: { position: Record<string, unknown> }[];
          workflow_identity?: unknown;
          attempt_id?: string;
          fencing_token?: string;
        };
        finalizeBodies.push(body);
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
                  settlement === "suspended"
                    ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
                    : "ACTIVITY_OUTCOME_KIND_COMPLETED",
                ...(settlement === "completed" ? { result: "approved" } : {}),
              },
            })),
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }),
    );

    await runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () =>
        runWithAttemptContext(attempt("attempt-1"), async () => {
          const events = await collect(agent.stream(baseContext, input));
          expect(events.map((event) => event.event)).toEqual(["suspend"]);
          const data = events[0]?.data as {
            interrupts?: Array<{ id: string; position?: unknown }>;
          };
          expect(data.interrupts).toEqual([
            { id: "wait-1", value: { question: "child approve?" } },
          ]);
          // The synthetic interrupt position is the root hosting task path —
          // the occurrence that owns the suspended child.
          const position = (
            finalizeBodies[0] as {
              suspensions?: {
                position: {
                  operation_path?: { segments?: { name: string }[] };
                };
              }[];
            }
          )?.suspensions?.[0]?.position;
          expect(
            position?.operation_path?.segments?.map((segment) => segment.name),
          ).toEqual(['langgraph.task:["__pregel_pull","nested"]']);
        }),
    );

    settlement = "completed";
    await runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () =>
        runWithAttemptContext(attempt("attempt-2", true), async () => {
          const events = await collect(
            agent.stream(
              {
                ...baseContext,
                resume: true,
                resumeData: { "wait-1": "approved" },
              } as never,
              input,
            ),
          );
          expect(events[events.length - 1]?.event).toBe("result");
        }),
    );
    expect(completeExecution).toHaveBeenCalledTimes(1);
  });
});
