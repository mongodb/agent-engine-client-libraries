import { create } from "@bufbuild/protobuf";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import { tool as lcTool } from "@langchain/core/tools";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
  interrupt,
  isGraphBubbleUp,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import type {
  AgentInput,
  AgentOutput,
  JsonValue,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import {
  AttemptContextSchema,
  DurableMemoryState,
  PolicyDeniedException,
  SecureLLMProxy,
  SecureToolWrapper,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  createSecureToolFunction,
  currentAttemptContext,
  registerSuspendHandler,
  resetHooks,
  runWithAttemptContext,
  runWithExecutionContext,
  type CompleteExecutionCommand,
  type FinalizeStepCommand,
} from "@mongodb-js/agent-engine-runner-shared";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../src/agent.js";
import type { DurableSession } from "../src/durable_session.js";
import { withDurableToolResultIdentity } from "../src/durable_tools.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";
import { SecureWrappedLLM } from "../src/secure_llm.js";
import { executionSession } from "../src/session_factory.js";
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

interface SuspensionJson {
  readonly position: Record<string, unknown>;
  readonly semantic_input?: unknown;
}

interface FinalizeBody {
  readonly workflow_identity?: unknown;
  readonly attempt_id?: string;
  readonly fencing_token?: string;
  readonly suspensions?: SuspensionJson[];
  readonly observed_activity_positions?: Record<string, unknown>[];
  readonly state?: unknown;
}

function attempt(attemptId: string, replay = false) {
  return create(AttemptContextSchema, {
    attemptId,
    fencingToken: replay ? 8n : 7n,
    replayMode: replay,
    workflowIdentity: identity,
  });
}

function platformClients() {
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

function publishedMessages(
  completeExecution: ReturnType<typeof platformClients>["completeExecution"],
  callIndex = 0,
): BaseMessage[] {
  const state = completeExecution.mock.calls[callIndex]?.[0].state;
  if (state === undefined) throw new Error("missing published state");
  return stateSnapshotToChannelValues(state)["messages"] as BaseMessage[];
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestBody(init?: RequestInit): FinalizeBody {
  if (typeof init?.body !== "string") throw new Error("missing request body");
  return JSON.parse(init.body) as FinalizeBody;
}

function wireBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") throw new Error("missing request body");
  return JSON.parse(init.body) as Record<string, unknown>;
}

function settlementResponse(
  body: FinalizeBody,
  kind: "suspended" | "completed",
  answerByActivityId: Readonly<Record<string, JsonValue>> = {},
  provenance: {
    readonly attemptId?: string;
    readonly fencingToken?: string;
  } = {},
): Response {
  const suspensions = body.suspensions ?? [];
  return jsonResponse({
    entries: suspensions.map((suspension, index) => {
      const semanticInput = suspension.semantic_input as
        | { value?: { branch?: string } }
        | undefined;
      const branch = semanticInput?.value?.branch;
      const activityId = branch ? `wait-${branch}` : `wait-${index + 1}`;
      return {
        position: suspension.position,
        outcome: {
          workflow_identity: body.workflow_identity,
          activity_id: activityId,
          attempt_id: provenance.attemptId ?? body.attempt_id,
          fencing_token: provenance.fencingToken ?? body.fencing_token,
          outcome_kind:
            kind === "suspended"
              ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
              : "ACTIVITY_OUTCOME_KIND_COMPLETED",
          ...(kind === "completed"
            ? { result: answerByActivityId[activityId] }
            : {}),
        },
      };
    }),
  });
}

function present<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("expected a value");
  return value;
}

function statefulOe() {
  const ids = new Map<string, string>();
  const answers = new Map<string, JsonValue>();
  const bodies: FinalizeBody[] = [];
  const fetch = vi.fn(async (_request: unknown, init?: RequestInit) => {
    const body = requestBody(init);
    bodies.push(body);
    return jsonResponse({
      entries: (body.suspensions ?? []).map((suspension) => {
        const key = JSON.stringify(suspension.position);
        const activityId = ids.get(key) ?? `wait-${ids.size + 1}`;
        ids.set(key, activityId);
        const answer = answers.get(activityId);
        return {
          position: suspension.position,
          outcome: {
            workflow_identity: body.workflow_identity,
            activity_id: activityId,
            attempt_id: body.attempt_id,
            fencing_token: body.fencing_token,
            outcome_kind:
              answer === undefined
                ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
                : "ACTIVITY_OUTCOME_KIND_COMPLETED",
            ...(answer === undefined ? {} : { result: answer }),
          },
        };
      }),
    });
  });
  return { fetch, answers, bodies };
}

async function inAttempt<T>(
  current: ReturnType<typeof attempt>,
  fn: () => Promise<T>,
): Promise<T> {
  return runWithExecutionContext(
    { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
    () => runWithAttemptContext(current, fn),
  );
}

async function inToolAttempt<T>(
  current: ReturnType<typeof attempt>,
  wrapper: SecureToolWrapper,
  fn: () => Promise<T>,
): Promise<T> {
  return runWithExecutionContext(
    {
      executionId: "execution",
      wrapper,
      oeUrl: "http://oe",
      userId: "user-1",
    },
    () => runWithAttemptContext(current, fn),
  );
}

async function collect(iterable: AsyncIterable<StreamEvent>) {
  const events: StreamEvent[] = [];
  for await (const event of iterable) events.push(event);
  return events;
}

async function expectScratchReleased(
  saver: PlatformCheckpointer,
  current: ReturnType<typeof attempt>,
): Promise<void> {
  await runWithAttemptContext(current, async () => {
    expect(
      await saver.getTuple({ configurable: { thread_id: "caller" } }),
    ).toBeUndefined();
  });
}

function singleInterruptAgent(
  saver: PlatformCheckpointer,
  received: JsonValue[],
  value: JsonValue = { question: "approve?" },
) {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("approval", async () => {
      const answer = interrupt(value) as JsonValue;
      received.push(answer);
      return {
        messages: [new AIMessage({ content: `resumed:${String(answer)}` })],
      };
    })
    .addEdge(START, "approval")
    .addEdge("approval", END)
    .compile({ checkpointer: saver as never });
  return new LangGraphBaseAgent(graph as never);
}

function suspendedData(
  result: AgentOutput | StreamEvent[],
): Record<string, unknown> {
  if (Array.isArray(result)) {
    expect(result.map((event) => event.event)).toEqual(["suspend"]);
    return result[0]?.data as Record<string, unknown>;
  }
  return result.response as Record<string, unknown>;
}

afterEach(() => {
  resetHooks();
  vi.unstubAllGlobals();
  // Tests here spy on SecureLLMProxy.prototype to stand in for the provider;
  // without restoring, the stub leaks into later tests in this file.
  vi.restoreAllMocks();
});

describe("durable root interrupt resume", () => {
  describe.each(["invoke", "stream"] as const)("%s integer range", (mode) => {
    it.each([
      { value: 2 ** 53, accepted: false },
      { value: -(2 ** 53), accepted: false },
      { value: Number.MAX_SAFE_INTEGER, accepted: true },
      { value: Number.MIN_SAFE_INTEGER, accepted: true },
      { value: 1.5, accepted: true },
    ])(
      "validates nested $value (accepted=$accepted)",
      async ({ value, accepted }) => {
        const clients = platformClients();
        const saver = new PlatformCheckpointer({
          native: null,
          clientFactory: clients.factory,
        });
        const agent = singleInterruptAgent(saver, [], { nested: [value] });
        const fetch = vi.fn(async (_request: unknown, init?: RequestInit) =>
          settlementResponse(requestBody(init), "suspended"),
        );
        vi.stubGlobal("fetch", fetch);
        const current = attempt("attempt-1");
        const result = inAttempt<AgentOutput | StreamEvent[]>(current, () =>
          mode === "invoke"
            ? agent.invoke(baseContext, input)
            : collect(agent.stream(baseContext, input)),
        );
        if (accepted) {
          expect(suspendedData(await result)["interrupts"]).toEqual([
            { id: "wait-1", value: { nested: [value] } },
          ]);
          expect(fetch).toHaveBeenCalledOnce();
        } else {
          await expect(result).rejects.toThrow(
            "integer exceeds the exact ProtoJSON range",
          );
          expect(fetch).not.toHaveBeenCalled();
          for (const [command] of clients.finalizeStep.mock.calls) {
            expect(command.suspensions).toEqual([]);
          }
        }
        expect(clients.completeExecution).not.toHaveBeenCalled();
        await expectScratchReleased(saver, current);
      },
    );
  });

  it.each([
    { mode: "invoke", replayWait: false },
    { mode: "stream", replayWait: false },
    { mode: "invoke", replayWait: true },
    { mode: "stream", replayWait: true },
  ] as const)(
    "$mode rejects an unreconstructed resume frontier (replayWait=$replayWait)",
    async ({ mode, replayWait }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const reachedEnd = vi.fn();
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("review", () => {
          if (replayWait) interrupt({ question: "earlier wait" });
          reachedEnd();
          return { messages: [new AIMessage("graph finished")] };
        })
        .addEdge(START, "review")
        .addEdge("review", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      const fetch = vi.fn(async (_request: unknown, init?: RequestInit) =>
        settlementResponse(requestBody(init), "completed", {
          "wait-1": "earlier answer",
        }),
      );
      vi.stubGlobal("fetch", fetch);
      const current = attempt("replacement", true);
      const ctx = {
        ...baseContext,
        resume: true,
        resumeData: { "wait-stale": "approved" },
      } as RequestContext;
      const events: StreamEvent[] = [];

      await expect(
        inAttempt(current, async () => {
          if (mode === "invoke") {
            await agent.invoke(ctx, input);
          } else {
            for await (const event of agent.stream(ctx, input)) {
              events.push(event);
            }
          }
        }),
      ).rejects.toThrow(
        "durable resume did not reconstruct its suspension frontier",
      );
      expect(reachedEnd).toHaveBeenCalledOnce();
      expect(fetch).toHaveBeenCalledTimes(replayWait ? 1 : 0);
      expect(clients.completeExecution).not.toHaveBeenCalled();
      expect(events.some((event) => event.event === "result")).toBe(false);
      await expectScratchReleased(saver, current);
    },
  );

  it.each(["invoke", "stream"] as const)(
    "%s publishes an OE wait id, resumes it, and closes durable Memory",
    async (mode) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const received: JsonValue[] = [];
      const agent = singleInterruptAgent(saver, received);
      const wrapper = new SecureToolWrapper("http://oe", "execution");
      wrapper.durableMemory = new DurableMemoryState();
      const finalizeBodies: FinalizeBody[] = [];
      const memoryBodies: Record<string, unknown>[] = [];
      let settlement: "suspended" | "completed" = "suspended";
      vi.stubGlobal(
        "fetch",
        vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
          if (String(request) === "http://oe/executor/activity/memory") {
            memoryBodies.push(wireBody(init));
            return jsonResponse({});
          }
          expect(String(request)).toBe("http://oe/executor/step/finalize");
          const body = requestBody(init);
          finalizeBodies.push(body);
          return settlementResponse(body, settlement, {
            "wait-1": "approved",
          });
        }),
      );

      const firstAttempt = attempt("attempt-1");
      const first = await inToolAttempt<AgentOutput | StreamEvent[]>(
        firstAttempt,
        wrapper,
        () =>
          mode === "invoke"
            ? agent.invoke(baseContext, input)
            : collect(agent.stream(baseContext, input)),
      );
      const data = suspendedData(first);
      expect(data).toMatchObject({
        interrupts: [{ id: "wait-1", value: { question: "approve?" } }],
        resumed: false,
      });
      if (mode === "invoke") expect(data["status"]).toBe("suspended");
      expect(data["resume_schema"]).toMatchObject({
        properties: { resume_map: { required: ["wait-1"] } },
      });
      if (mode === "invoke") {
        expect(
          (data["suspend_context"] as Record<string, unknown>)["checkpoint_id"],
        ).toBeNull();
      } else {
        expect(data["metadata"]).toEqual({ checkpoint_id: null });
      }
      expect(clients.completeExecution).not.toHaveBeenCalled();
      await expectScratchReleased(saver, firstAttempt);

      settlement = "completed";
      const replacement = attempt("attempt-2", true);
      const resumeContext = {
        ...baseContext,
        resume: true,
        resumeData: { "wait-1": "request data is not the winning OE answer" },
      } as RequestContext;
      const resumed = await inToolAttempt<AgentOutput | StreamEvent[]>(
        replacement,
        wrapper,
        () =>
          mode === "invoke"
            ? agent.invoke(resumeContext, input)
            : collect(agent.stream(resumeContext, input)),
      );

      if (Array.isArray(resumed)) {
        expect(resumed.at(-1)).toMatchObject({
          event: "result",
          data: { response: "resumed:approved", resumed: true },
        });
      } else {
        expect(resumed.response).toMatchObject({
          response: "resumed:approved",
          status: "completed",
          resumed: true,
        });
      }
      expect(received).toEqual(["approved"]);
      expect(clients.completeExecution).toHaveBeenCalledOnce();
      expect(finalizeBodies).toHaveLength(2);
      expect(finalizeBodies[1]?.suspensions?.[0]?.position).toEqual(
        finalizeBodies[0]?.suspensions?.[0]?.position,
      );
      expect(finalizeBodies[0]?.observed_activity_positions).toEqual([
        finalizeBodies[0]?.suspensions?.[0]?.position,
      ]);
      expect(finalizeBodies[0]?.state).toBeUndefined();
      expect(memoryBodies).toHaveLength(1);
      expect(memoryBodies[0]?.["activity_id"]).toBe("wait-1");
      expect(memoryBodies[0]?.["memory_writes"]).toEqual([
        expect.objectContaining({
          id: "workflow:execution:wait-1:tool:0",
        }),
      ]);
      await expectScratchReleased(saver, replacement);
    },
  );

  it.each(["invoke", "stream"] as const)(
    "%s posts no Memory writes when the resume map misses the suspension frontier",
    async (mode) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const agent = singleInterruptAgent(saver, []);
      const wrapper = new SecureToolWrapper("http://oe", "execution");
      wrapper.durableMemory = new DurableMemoryState();
      const memoryBodies: Record<string, unknown>[] = [];
      vi.stubGlobal(
        "fetch",
        vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
          if (String(request) === "http://oe/executor/activity/memory") {
            memoryBodies.push(wireBody(init));
            return jsonResponse({});
          }
          expect(String(request)).toBe("http://oe/executor/step/finalize");
          return settlementResponse(requestBody(init), "completed", {
            "wait-1": "approved",
          });
        }),
      );

      // The resolved interrupt plus one stale ID: the frontier check must
      // reject before any durable Memory side effect is posted.
      const replacement = attempt("attempt-9", true);
      const resumeContext = {
        ...baseContext,
        resume: true,
        resumeData: {
          "wait-1": "approved",
          "wait-stale": "no such wait",
        },
      } as RequestContext;

      await expect(
        inToolAttempt<AgentOutput | StreamEvent[]>(replacement, wrapper, () =>
          mode === "invoke"
            ? agent.invoke(resumeContext, input)
            : collect(agent.stream(resumeContext, input)),
        ),
      ).rejects.toThrow(
        "durable resume map does not match the resolved suspension frontier",
      );
      expect(memoryBodies).toHaveLength(0);
      expect(clients.completeExecution).not.toHaveBeenCalled();
      await expectScratchReleased(saver, replacement);
    },
  );

  it("settles and resumes parallel root waits as one complete frontier", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const received: Record<string, JsonValue> = {};
    const builder = new StateGraph(MessagesAnnotation);
    for (const branch of ["alpha", "beta"]) {
      builder
        .addNode(branch, async () => {
          received[branch] = interrupt({ branch }) as JsonValue;
          return {
            messages: [new AIMessage({ content: `resumed:${branch}` })],
          };
        })
        .addEdge(START, branch)
        .addEdge(branch, END);
    }
    const agent = new LangGraphBaseAgent(
      builder.compile({ checkpointer: saver as never }) as never,
    );
    const finalizeBodies: FinalizeBody[] = [];
    let settlement: "suspended" | "completed" = "suspended";
    const answers = {
      "wait-alpha": { branch: "alpha", approved: true },
      "wait-beta": { branch: "beta", approved: false },
    } satisfies Record<string, JsonValue>;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: string | URL | Request, init?: RequestInit) => {
        const body = requestBody(init);
        finalizeBodies.push(body);
        const response = (await settlementResponse(
          body,
          settlement,
          answers,
        ).json()) as { entries: unknown[] };
        return jsonResponse({ entries: response.entries.reverse() });
      }),
    );

    const first = attempt("attempt-1");
    const suspended = await inAttempt(first, () =>
      agent.invoke(baseContext, input),
    );
    const pending = (suspended.response as Record<string, unknown>)[
      "interrupts"
    ] as Array<{ id: string; value: { branch: string } }>;
    expect(pending.map((item) => item.value.branch).sort()).toEqual([
      "alpha",
      "beta",
    ]);
    expect(pending.map((item) => item.id).sort()).toEqual([
      "wait-alpha",
      "wait-beta",
    ]);
    expect(finalizeBodies[0]?.suspensions).toHaveLength(2);
    expect(clients.completeExecution).not.toHaveBeenCalled();

    settlement = "completed";
    const replacement = attempt("attempt-2", true);
    const result = await inAttempt(replacement, () =>
      agent.invoke(
        {
          ...baseContext,
          resume: true,
          resumeData: answers,
        } as RequestContext,
        input,
      ),
    );

    expect(result.response).toMatchObject({
      status: "completed",
      resumed: true,
    });
    expect(received).toEqual({
      alpha: answers["wait-alpha"],
      beta: answers["wait-beta"],
    });
    expect(finalizeBodies[1]?.suspensions).toHaveLength(2);
    expect(
      finalizeBodies[1]?.suspensions?.map((item) => item.position),
    ).toEqual(finalizeBodies[0]?.suspensions?.map((item) => item.position));
    expect(clients.completeExecution).toHaveBeenCalledOnce();
  });

  it.each([
    { mode: "invoke", payloads: "distinct", nested: false },
    { mode: "stream", payloads: "distinct", nested: false },
    { mode: "invoke", payloads: "identical", nested: false },
    { mode: "stream", payloads: "identical", nested: false },
    { mode: "invoke", payloads: "distinct", nested: true },
    { mode: "stream", payloads: "distinct", nested: true },
  ] as const)(
    "$mode resumes each pause of one task with its own answer (payloads=$payloads, nested=$nested)",
    async ({ mode, payloads, nested }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const received: JsonValue[][] = [];
      const pause = (ordinal: number): JsonValue =>
        payloads === "identical"
          ? { question: "approve?" }
          : { pause: ordinal };
      const inner = new StateGraph(MessagesAnnotation)
        .addNode("review", async () => {
          const first = interrupt(pause(1)) as JsonValue;
          const second = interrupt(pause(2)) as JsonValue;
          const third = interrupt(pause(3)) as JsonValue;
          received.push([first, second, third]);
          return {
            messages: [
              new AIMessage(
                `${String(first)}|${String(second)}|${String(third)}`,
              ),
            ],
          };
        })
        .addEdge(START, "review")
        .addEdge("review", END);
      const graph = nested
        ? new StateGraph(MessagesAnnotation)
            .addNode("child", inner.compile())
            .addEdge(START, "child")
            .addEdge("child", END)
            .compile({ checkpointer: saver as never })
        : inner.compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      const oe = statefulOe();
      vi.stubGlobal("fetch", oe.fetch);

      const run = (current: ReturnType<typeof attempt>, ctx: RequestContext) =>
        inAttempt<AgentOutput | StreamEvent[]>(current, () =>
          mode === "invoke"
            ? agent.invoke(ctx, input)
            : collect(agent.stream(ctx, input)),
        );

      let result = await run(attempt("attempt-1"), baseContext);
      const answered: string[] = [];
      for (const ordinal of [1, 2, 3]) {
        const interrupts = suspendedData(result)["interrupts"] as {
          id: string;
          value: JsonValue;
        }[];
        expect(interrupts).toHaveLength(1);
        const waiting = present(interrupts[0]);
        expect(waiting.value).toEqual(pause(ordinal));
        expect(answered).not.toContain(waiting.id);
        answered.push(waiting.id);
        oe.answers.set(waiting.id, `answer-${ordinal}`);
        // Each answer arrives on a fresh attempt; only OE history survives.
        result = await run(attempt(`attempt-${ordinal + 1}`, true), {
          ...baseContext,
          resume: true,
          resumeData: { [waiting.id]: `answer-${ordinal}` },
        } as RequestContext);
      }

      if (Array.isArray(result)) {
        expect(result.at(-1)).toMatchObject({
          event: "result",
          data: { response: "answer-1|answer-2|answer-3" },
        });
      } else {
        expect(result.response).toMatchObject({
          response: "answer-1|answer-2|answer-3",
          status: "completed",
        });
      }
      expect(received.at(-1)).toEqual(["answer-1", "answer-2", "answer-3"]);
      expect(clients.completeExecution).toHaveBeenCalledOnce();
      // Three pauses, three durable positions in one task.
      const positions = new Set(
        oe.bodies.flatMap((body) =>
          (body.suspensions ?? []).map((item) => JSON.stringify(item.position)),
        ),
      );
      expect(positions.size).toBe(3);
      expect(
        new Set(
          [...positions].map((key) => {
            const position = JSON.parse(key) as {
              operation_path?: unknown;
              activity_ordinal?: string;
            };
            return JSON.stringify(position.operation_path);
          }),
        ).size,
      ).toBe(1);
    },
  );

  it.each(
    [false, true].flatMap((nested) =>
      (["invoke", "stream"] as const).map((mode) => ({ nested, mode })),
    ),
  )(
    "$mode resumes tasks that pause again while a parallel task finishes (nested=$nested)",
    async ({ nested, mode }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const onceAnswers: JsonValue[] = [];
      const pausesTwice = (task: string) => async () => {
        const first = interrupt({ task, pause: 1 }) as JsonValue;
        const second = interrupt({ task, pause: 2 }) as JsonValue;
        return {
          messages: [
            new AIMessage(`${task}:${String(first)}|${String(second)}`),
          ],
        };
      };
      const parallel = new StateGraph(MessagesAnnotation)
        .addNode("twice", pausesTwice("twice"))
        .addNode("again", pausesTwice("again"))
        .addNode("once", async () => {
          const answer = interrupt({ task: "once", pause: 1 }) as JsonValue;
          onceAnswers.push(answer);
          return { messages: [new AIMessage(`once:${String(answer)}`)] };
        })
        .addEdge(START, "twice")
        .addEdge(START, "again")
        .addEdge(START, "once")
        .addEdge("twice", END)
        .addEdge("again", END)
        .addEdge("once", END);
      // Through a compiled child, LangGraph projects only one of the
      // branches' pauses to the parent task that runs the child.
      const graph = nested
        ? new StateGraph(MessagesAnnotation)
            .addNode("child", parallel.compile())
            .addEdge(START, "child")
            .addEdge("child", END)
            .compile({ checkpointer: saver as never })
        : parallel.compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      const oe = statefulOe();
      vi.stubGlobal("fetch", oe.fetch);
      const run = async (
        current: ReturnType<typeof attempt>,
        ctx: RequestContext,
      ): Promise<Record<string, unknown>> => {
        if (mode === "invoke") {
          const output = await inAttempt(current, () =>
            agent.invoke(ctx, input),
          );
          return output.response as Record<string, unknown>;
        }
        const events = await inAttempt(current, () =>
          collect(agent.stream(ctx, input)),
        );
        return present(events.at(-1)).data as Record<string, unknown>;
      };
      const waitingByTask = (response: Record<string, unknown>) =>
        Object.fromEntries(
          (
            response["interrupts"] as {
              id: string;
              value: { task: string; pause: number };
            }[]
          ).map((item) => [`${item.value.task}:${item.value.pause}`, item.id]),
        );

      const first = waitingByTask(await run(attempt("attempt-1"), baseContext));
      expect(Object.keys(first).sort()).toEqual([
        "again:1",
        "once:1",
        "twice:1",
      ]);
      const answers = {
        [present(first["twice:1"])]: "t1",
        [present(first["again:1"])]: "a1",
        [present(first["once:1"])]: "o1",
      };
      for (const [id, answer] of Object.entries(answers))
        oe.answers.set(id, answer);

      const second = waitingByTask(
        await run(attempt("attempt-2", true), {
          ...baseContext,
          resume: true,
          resumeData: answers,
        } as RequestContext),
      );
      // Both tasks that paused again wait; no answered pause returns, and
      // the finished task's leftover interrupt is not offered again.
      expect(Object.keys(second).sort()).toEqual(["again:2", "twice:2"]);
      const secondAnswers = {
        [present(second["twice:2"])]: "t2",
        [present(second["again:2"])]: "a2",
      };
      for (const id of Object.keys(secondAnswers))
        expect(Object.keys(answers)).not.toContain(id);
      for (const [id, answer] of Object.entries(secondAnswers))
        oe.answers.set(id, answer);

      const completed = await run(attempt("attempt-3", true), {
        ...baseContext,
        resume: true,
        resumeData: secondAnswers,
      } as RequestContext);
      expect(completed).not.toHaveProperty("interrupts");
      const messages = publishedMessages(clients.completeExecution).map((m) =>
        String(m.content),
      );
      expect(messages).toContain("twice:t1|t2");
      expect(messages).toContain("again:a1|a2");
      expect(messages).toContain("once:o1");
      expect(new Set(onceAnswers)).toEqual(new Set(["o1"]));

      // Every attempt reported each pause at one position. Under a compiled
      // child the pauses share the root task that runs the child and take
      // consecutive ordinals: siblings in path order, then later pauses.
      const positionsByPause = new Map<string, Set<string>>();
      for (const body of oe.bodies) {
        for (const suspension of body.suspensions ?? []) {
          const value = (
            suspension.semantic_input as {
              value: { task: string; pause: number };
            }
          ).value;
          const position = suspension.position as {
            operation_path?: { segments?: { name: string }[] };
            activity_ordinal?: string | number;
          };
          const key = `${value.task}:${value.pause}`;
          const at = `${(position.operation_path?.segments ?? [])
            .map((segment) => segment.name)
            .join("/")}#${Number(position.activity_ordinal)}`;
          positionsByPause.set(
            key,
            (positionsByPause.get(key) ?? new Set()).add(at),
          );
        }
      }
      const task = (name: string) =>
        `langgraph.task:${JSON.stringify(["__pregel_pull", name])}`;
      expect(
        Object.fromEntries(
          [...positionsByPause].map(([key, at]) => [key, [...at]]),
        ),
      ).toEqual(
        nested
          ? {
              "again:1": [`${task("child")}#1`],
              "once:1": [`${task("child")}#2`],
              "twice:1": [`${task("child")}#3`],
              "again:2": [`${task("child")}#4`],
              "twice:2": [`${task("child")}#5`],
            }
          : {
              "again:1": [`${task("again")}#1`],
              "once:1": [`${task("once")}#1`],
              "twice:1": [`${task("twice")}#1`],
              "again:2": [`${task("again")}#2`],
              "twice:2": [`${task("twice")}#2`],
            },
      );
    },
  );

  it.each([
    { mode: "invoke", secondInterrupt: false },
    { mode: "stream", secondInterrupt: false },
    { mode: "invoke", secondInterrupt: true },
    { mode: "stream", secondInterrupt: true },
  ] as const)(
    "$mode reconstructs a registered local-tool interrupt without redispatch (secondInterrupt=$secondInterrupt)",
    async ({ mode, secondInterrupt }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const received: JsonValue[] = [];
      const wrappedTool = createSecureToolFunction(
        {
          invoke: () => {
            const answer = interrupt({ claim_id: "CLM-1" }) as JsonValue;
            received.push(answer);
            if (secondInterrupt) {
              return interrupt({ claim_id: "CLM-1", question: "again?" });
            }
            return answer;
          },
        },
        "review_claim",
        false,
        { isLocal: true, isFrameworkControlFlow: isGraphBubbleUp },
      );
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("tool", async () => {
          const answer = await wrappedTool({ claimId: "CLM-1" });
          return {
            messages: [
              new AIMessage({ content: `tool:${JSON.stringify(answer)}` }),
            ],
          };
        })
        .addEdge(START, "tool")
        .addEdge("tool", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      const wrapper = new SecureToolWrapper("http://oe", "execution");
      const toolResults: Record<string, unknown>[] = [];
      const toolExecutes: Record<string, unknown>[] = [];
      const activityStarts: Record<string, unknown>[] = [];
      const activityOutcomes: Record<string, unknown>[] = [];
      const stepFinalizations: FinalizeBody[] = [];
      let settlement: "suspended" | "completed" = "suspended";
      vi.stubGlobal(
        "fetch",
        vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
          const url = String(request);
          const body = wireBody(init);
          if (url.endsWith("/executor/activity/start")) {
            activityStarts.push(body);
            if (settlement === "suspended") {
              return jsonResponse({
                activity_context: {
                  workflow_identity: body["workflow_identity"],
                  activity_id: "tool-wait",
                  attempt_id: body["attempt_id"],
                  fencing_token: body["fencing_token"],
                },
              });
            }
            return jsonResponse({
              outcome: {
                workflow_identity: body["workflow_identity"],
                activity_id: "tool-wait",
                attempt_id: body["attempt_id"],
                fencing_token: body["fencing_token"],
                outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
                result: { decision: "approve" },
              },
            });
          }
          if (url.endsWith("/tool/execute")) {
            toolExecutes.push(body);
            return jsonResponse({ proceed: true, route_to: "callback" });
          }
          if (url.endsWith("/tool/result")) {
            toolResults.push(body);
            return jsonResponse({});
          }
          if (url.endsWith("/executor/activity/outcome")) {
            activityOutcomes.push(body);
            return jsonResponse({});
          }
          if (url.endsWith("/executor/step/finalize")) {
            const finalize = body as FinalizeBody;
            stepFinalizations.push(finalize);
            const suspension = finalize.suspensions?.[0];
            if (suspension === undefined) throw new Error("missing suspension");
            return jsonResponse({
              entries: [
                {
                  position: suspension.position,
                  outcome: {
                    workflow_identity: finalize.workflow_identity,
                    activity_id: "tool-wait",
                    attempt_id: finalize.attempt_id,
                    fencing_token: finalize.fencing_token,
                    outcome_kind:
                      settlement === "suspended"
                        ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
                        : "ACTIVITY_OUTCOME_KIND_COMPLETED",
                    ...(settlement === "completed"
                      ? { result: { decision: "approve" } }
                      : {}),
                  },
                },
              ],
            });
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      const firstAttempt = attempt("attempt-1");
      const first = await inToolAttempt<AgentOutput | StreamEvent[]>(
        firstAttempt,
        wrapper,
        () =>
          mode === "invoke"
            ? agent.invoke(baseContext, input)
            : collect(agent.stream(baseContext, input)),
      );
      expect(suspendedData(first)["interrupts"]).toEqual([
        { id: "tool-wait", value: { claim_id: "CLM-1" } },
      ]);
      expect(clients.completeExecution).not.toHaveBeenCalled();

      settlement = "completed";
      const replacement = attempt("attempt-2", true);
      const resumeContext = {
        ...baseContext,
        resume: true,
        resumeData: { "tool-wait": { decision: "approve" } },
      } as RequestContext;
      const resume = inToolAttempt<AgentOutput | StreamEvent[]>(
        replacement,
        wrapper,
        () =>
          mode === "invoke"
            ? agent.invoke(resumeContext, input)
            : collect(agent.stream(resumeContext, input)),
      );

      if (secondInterrupt) {
        await expect(resume).rejects.toThrow(
          "cannot raise another native interrupt",
        );
        expect(toolExecutes).toHaveLength(1);
        expect(toolResults).toHaveLength(1);
        expect(activityOutcomes).toEqual([]);
        expect(clients.completeExecution).not.toHaveBeenCalled();
        await expectScratchReleased(saver, replacement);
        return;
      }

      const resumed = await resume;

      if (Array.isArray(resumed)) {
        expect(resumed.at(-1)?.event).toBe("result");
      } else {
        expect(resumed.response).toMatchObject({ status: "completed" });
      }
      expect(received).toEqual([{ decision: "approve" }]);
      expect(activityStarts.map((body) => body["attempt_id"])).toEqual([
        "attempt-1",
        "attempt-2",
        "attempt-2",
      ]);
      expect(toolExecutes).toHaveLength(1);
      expect(toolResults).toHaveLength(1);
      expect(activityOutcomes).toEqual([]);
      expect(stepFinalizations).toHaveLength(2);
      expect(clients.completeExecution).toHaveBeenCalledOnce();
      await expectScratchReleased(saver, replacement);
    },
  );

  it("replays an earlier resolved step before resuming the requested frontier", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const received: unknown[] = [];
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("first", async () => {
        received.push(interrupt({ branch: "first" }));
        return { messages: [new AIMessage("first resolved")] };
      })
      .addNode("second", async () => {
        received.push(interrupt({ branch: "second" }));
        return { messages: [new AIMessage("second resolved")] };
      })
      .addEdge(START, "first")
      .addEdge("first", "second")
      .addEdge("second", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    const finalizations: FinalizeBody[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: unknown, init?: RequestInit) => {
        const body = requestBody(init);
        finalizations.push(body);
        return settlementResponse(body, "completed", {
          "wait-first": "one",
          "wait-second": "two",
        });
      }),
    );

    const result = await inAttempt(attempt("replacement", true), () =>
      agent.invoke(
        {
          ...baseContext,
          resume: true,
          resumeData: { "wait-second": "two" },
        } as RequestContext,
        input,
      ),
    );

    expect(result.response).toMatchObject({
      status: "completed",
      response: "second resolved",
      resumed: true,
    });
    expect(received).toEqual(["one", "two"]);
    expect(finalizations).toHaveLength(2);
    expect(
      finalizations[0]?.suspensions?.[0]?.position["step_ordinal"],
    ).not.toEqual(finalizations[1]?.suspensions?.[0]?.position["step_ordinal"]);
    expect(clients.completeExecution).toHaveBeenCalledOnce();
  });

  it.each([
    ["incomplete", "incomplete frontier"],
    ["missing position", "without a position"],
    ["wrong position", "unknown or duplicate position"],
    ["wrong workflow", "cross-workflow outcome"],
    ["stale fence", "cross-attempt outcome"],
    ["missing outcome", "invalid activity outcome"],
  ])(
    "rejects %s OE settlement without completing",
    async (malformation, error) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const agent = singleInterruptAgent(saver, []);
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_request: unknown, init?: RequestInit) => {
          const body = requestBody(init);
          const response = (await settlementResponse(
            body,
            "suspended",
          ).json()) as {
            entries: Array<{
              position?: Record<string, unknown>;
              outcome?: Record<string, unknown>;
            }>;
          };
          const entry = response.entries[0];
          if (!entry?.outcome) throw new Error("invalid fixture");
          switch (malformation) {
            case "incomplete":
              response.entries = [];
              break;
            case "missing position":
              delete entry.position;
              break;
            case "wrong position":
              entry.position = { ...entry.position, activity_ordinal: "999" };
              break;
            case "wrong workflow":
              entry.outcome["workflow_identity"] = {
                session_id: "other",
                execution_id: "other",
              };
              break;
            case "stale fence":
              entry.outcome["fencing_token"] = "6";
              break;
            case "missing outcome":
              delete entry.outcome;
              break;
          }
          return jsonResponse(response);
        }),
      );
      const current = attempt("attempt-1");

      await expect(
        inAttempt(current, () => agent.invoke(baseContext, input)),
      ).rejects.toThrow(error);
      expect(clients.completeExecution).not.toHaveBeenCalled();
      await expectScratchReleased(saver, current);
    },
  );

  it("rejects a cross-attempt OE settlement outcome", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const agent = singleInterruptAgent(saver, []);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: string | URL | Request, init?: RequestInit) => {
        const body = requestBody(init);
        return settlementResponse(
          body,
          "suspended",
          {},
          {
            attemptId: "attempt-other",
          },
        );
      }),
    );
    const current = attempt("attempt-1");

    await expect(
      inAttempt(current, () => agent.invoke(baseContext, input)),
    ).rejects.toThrow("cross-attempt outcome");
    expect(clients.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, current);
  });

  it.each([
    ["mixed", "mixed suspended and resolved outcomes"],
    ["duplicate position", "unknown or duplicate position"],
    [
      "duplicate activity",
      "durable interrupt settlement returned duplicate activity ids",
    ],
    [
      "resolved duplicate activity",
      "durable interrupt settlement returned duplicate activity ids",
    ],
  ])(
    "rejects a %s frontier instead of surfacing an ambiguous wait",
    async (malformation, error) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const graph = {
        checkpointer: saver,
        invoke: vi.fn(async () => ({ messages: [] })),
        getState: vi.fn(async () => ({
          next: ["alpha", "beta"],
          tasks: ["alpha", "beta"].map((branch) => ({
            path: ["__pregel_pull", branch],
            interrupts: [{ id: branch, value: { branch } }],
          })),
        })),
      };
      const agent = new LangGraphBaseAgent(graph as never);
      const fetch = vi.fn(async (_request: unknown, init?: RequestInit) => {
        const response = (await settlementResponse(
          requestBody(init),
          malformation === "resolved duplicate activity"
            ? "completed"
            : "suspended",
        ).json()) as {
          entries: Array<{
            position: unknown;
            outcome: Record<string, unknown>;
          }>;
        };
        const first = response.entries[0];
        const second = response.entries[1];
        if (!first || !second) throw new Error("invalid fixture");
        if (malformation === "mixed")
          second.outcome["outcome_kind"] = "ACTIVITY_OUTCOME_KIND_COMPLETED";
        if (malformation === "duplicate position")
          second.position = first.position;
        if (
          malformation === "duplicate activity" ||
          malformation === "resolved duplicate activity"
        )
          second.outcome["activity_id"] = first.outcome["activity_id"];
        return jsonResponse(response);
      });
      vi.stubGlobal("fetch", fetch);
      await expect(
        inAttempt(attempt("attempt-1"), () => agent.invoke(baseContext, input)),
      ).rejects.toThrow(error);
      expect(fetch).toHaveBeenCalledOnce();
      expect(clients.completeExecution).not.toHaveBeenCalled();
    },
  );

  it.each([false, true])(
    "handles duplicate native projections (conflicting=%s)",
    async (conflicting) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const graph = {
        checkpointer: saver,
        invoke: vi.fn(async () => ({ messages: [] })),
        getState: vi.fn(async () => ({
          next: ["approval"],
          tasks: [
            {
              path: ["__pregel_pull", "approval"],
              interrupts: [
                { id: "native", value: { a: 1, b: 2 } },
                { id: "native", value: { b: 2, a: conflicting ? 3 : 1 } },
              ],
            },
          ],
        })),
      };
      const agent = new LangGraphBaseAgent(graph as never);
      const fetch = vi.fn(async (_request: unknown, init?: RequestInit) =>
        settlementResponse(requestBody(init), "suspended"),
      );
      vi.stubGlobal("fetch", fetch);
      const result = inAttempt(attempt("attempt-1"), () =>
        agent.invoke(baseContext, input),
      );
      if (conflicting) {
        await expect(result).rejects.toThrow("conflicting values");
        expect(fetch).not.toHaveBeenCalled();
      } else {
        expect(suspendedData(await result)["interrupts"]).toEqual([
          { id: "wait-1", value: { a: 1, b: 2 } },
        ]);
        expect(fetch).toHaveBeenCalledOnce();
      }
      expect(clients.completeExecution).not.toHaveBeenCalled();
    },
  );

  it("rejects a resume map that does not match the resolved frontier", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const agent = singleInterruptAgent(saver, []);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_request: string | URL | Request, init?: RequestInit) => {
        const body = requestBody(init);
        return settlementResponse(body, "completed", {
          "wait-1": "approved",
        });
      }),
    );
    const current = attempt("attempt-2", true);

    await expect(
      inAttempt(current, () =>
        agent.invoke(
          {
            ...baseContext,
            resume: true,
            resumeData: { "wait-1": "approved", extra: "forged" },
          } as RequestContext,
          input,
        ),
      ),
    ).rejects.toThrow("resume map does not match");
    expect(clients.completeExecution).not.toHaveBeenCalled();
  });

  it("collapses a nested interrupt repeated by its parent snapshot into one frontier", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const graph = {
      checkpointer: saver,
      invoke: vi.fn(async () => ({ messages: [new AIMessage("waiting")] })),
      getState: vi.fn(async () => ({
        next: ["parent"],
        tasks: [
          {
            path: ["__pregel_pull", "parent"],
            interrupts: [{ id: "child-wait", value: { question: "approve?" } }],
            state: {
              tasks: [
                {
                  path: ["__pregel_pull", "child"],
                  interrupts: [
                    { id: "child-wait", value: { question: "approve?" } },
                  ],
                },
              ],
            },
          },
        ],
      })),
    };
    const agent = new LangGraphBaseAgent(graph as never);
    const fetch = vi.fn(async (_request: unknown, init?: RequestInit) =>
      settlementResponse(requestBody(init), "suspended"),
    );
    vi.stubGlobal("fetch", fetch);

    const result = await inAttempt(attempt("attempt-1"), () =>
      agent.invoke(baseContext, input),
    );
    const data = result.response as Record<string, unknown>;
    expect(data["interrupts"]).toEqual([
      { id: "wait-1", value: { question: "approve?" } },
    ]);
    expect(fetch).toHaveBeenCalledTimes(1);
    for (const [command] of clients.finalizeStep.mock.calls) {
      expect(command.suspensions).toHaveLength(1);
    }
    expect(clients.completeExecution).not.toHaveBeenCalled();
  });
});

describe("durable ToolNode batch parallel tools", () => {
  interface ToolCallSpec {
    readonly name: string;
    readonly args: Record<string, unknown>;
    readonly id: string;
  }

  interface BatchOe {
    readonly answers: Record<string, JsonValue>;
    readonly results: Record<string, JsonValue>;
    readonly activityStarts: Record<string, unknown>[];
    readonly activityOutcomes: Record<string, unknown>[];
    readonly toolExecutes: Record<string, unknown>[];
    readonly stepFinalizations: FinalizeBody[];
    /** Whether activity/start dispatches live or replays a recorded outcome. */
    dispatch: "live" | "replay";
    settlement: "suspended" | "completed";
  }

  function durableLocalTool(
    name: string,
    body: (args: Record<string, unknown>) => unknown,
  ) {
    const secure = createSecureToolFunction({ invoke: body }, name, false, {
      isLocal: true,
      responseFormat: "content_and_artifact",
      isFrameworkControlFlow: isGraphBubbleUp,
    });
    return lcTool(withDurableToolResultIdentity(secure, name) as never, {
      name,
      description: `${name} tool`,
      schema: z.object({ orderId: z.string() }),
    });
  }

  function compileBatchGraph(
    saver: PlatformCheckpointer,
    tools: ReturnType<typeof durableLocalTool>[],
    planToolCalls: ToolCallSpec[],
  ) {
    return new StateGraph(MessagesAnnotation)
      .addNode("plan", () => ({
        messages: [
          new AIMessage({
            content: "",
            id: "assistant-plan",
            tool_calls: planToolCalls,
          }),
        ],
      }))
      .addNode("tools", new ToolNode(tools))
      .addEdge(START, "plan")
      .addEdge("plan", "tools")
      .addEdge("tools", END)
      .compile({ checkpointer: saver as never });
  }

  function toolBatchAgent(
    saver: PlatformCheckpointer,
    tools: ReturnType<typeof durableLocalTool>[],
    planToolCalls: ToolCallSpec[],
  ) {
    return new LangGraphBaseAgent(
      compileBatchGraph(saver, tools, planToolCalls) as never,
    );
  }

  function wireBatchOe(oe: BatchOe): void {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
        const url = String(request);
        const body = wireBody(init);
        if (url.endsWith("/executor/activity/start")) {
          oe.activityStarts.push(body);
          const semanticInput = body["semantic_input"] as Record<
            string,
            unknown
          >;
          const activityId = `wait-${String(semanticInput["tool_call_id"])}`;
          if (oe.dispatch === "live") {
            return jsonResponse({
              activity_context: {
                workflow_identity: body["workflow_identity"],
                activity_id: activityId,
                attempt_id: body["attempt_id"],
                fencing_token: body["fencing_token"],
              },
            });
          }
          return jsonResponse({
            outcome: {
              workflow_identity: body["workflow_identity"],
              activity_id: activityId,
              attempt_id: body["attempt_id"],
              fencing_token: body["fencing_token"],
              outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
              result: oe.results[activityId] ?? null,
            },
          });
        }
        if (url.endsWith("/tool/execute")) {
          oe.toolExecutes.push(body);
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          oe.activityOutcomes.push(body);
          return jsonResponse({});
        }
        if (url.endsWith("/executor/step/finalize")) {
          const finalize = body as FinalizeBody;
          oe.stepFinalizations.push(finalize);
          const suspensions = finalize.suspensions ?? [];
          return jsonResponse({
            entries: suspensions.map((suspension) => {
              const semanticInput = suspension.semantic_input as
                | { tool_call_id?: string }
                | undefined;
              const activityId = `wait-${semanticInput?.tool_call_id}`;
              return {
                position: suspension.position,
                outcome: {
                  workflow_identity: finalize.workflow_identity,
                  activity_id: activityId,
                  attempt_id: finalize.attempt_id,
                  fencing_token: finalize.fencing_token,
                  outcome_kind:
                    oe.settlement === "suspended"
                      ? "ACTIVITY_OUTCOME_KIND_SUSPENDED"
                      : "ACTIVITY_OUTCOME_KIND_COMPLETED",
                  ...(oe.settlement === "completed"
                    ? { result: oe.answers[activityId] }
                    : {}),
                },
              };
            }),
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );
  }

  it("fails closed when two parallel tools in one batch raise interrupt()", async () => {
    // Both interrupts live in the same graph task, so LangGraph.js mints the
    // same native interrupt id for each (namespace hash only — identical to
    // Python's Interrupt.from_ns). Durable identity cannot join two activities
    // to one native id, so the surface must reject explicitly instead of
    // guessing a subset.
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const reviewTool = durableLocalTool("review_order", ({ orderId }) => {
      interrupt({ orderId, question: "review?" });
      return { approved: true };
    });
    const escalateTool = durableLocalTool("escalate_order", ({ orderId }) => {
      interrupt({ orderId, question: "escalate?" });
      return { approved: true };
    });
    const agent = toolBatchAgent(
      saver,
      [reviewTool, escalateTool],
      [
        { name: "review_order", args: { orderId: "A-1" }, id: "call-a" },
        { name: "escalate_order", args: { orderId: "B-2" }, id: "call-b" },
      ],
    );
    const oe: BatchOe = {
      answers: {},
      results: {},
      activityStarts: [],
      activityOutcomes: [],
      toolExecutes: [],
      stepFinalizations: [],
      dispatch: "live",
      settlement: "suspended",
    };
    wireBatchOe(oe);

    await expect(
      inToolAttempt(
        attempt("attempt-1"),
        new SecureToolWrapper("http://oe", "execution"),
        () => agent.invoke(baseContext, input),
      ),
    ).rejects.toThrow("invalid or duplicate identity");
    expect(oe.stepFinalizations).toEqual([]);
    expect(clients.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, attempt("attempt-1"));
  });

  it("settles a batch with one suspended sibling as one frontier and resumes it exactly once", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const wrapper = new SecureToolWrapper("http://oe", "execution");
    const answers = {
      "wait-call-b": { decision: "approve-b" },
    } satisfies Record<string, JsonValue>;
    let placeCalls = 0;
    let reviewAnswer: JsonValue | undefined;
    const placeTool = durableLocalTool("review_order", () => {
      placeCalls += 1;
      return { placed: true };
    });
    const reviewTool = durableLocalTool("escalate_order", ({ orderId }) => {
      const answer = interrupt({ orderId, question: "escalate?" }) as JsonValue;
      reviewAnswer = answer;
      return { approved: answer };
    });
    const agent = toolBatchAgent(
      saver,
      [placeTool, reviewTool],
      [
        { name: "review_order", args: { orderId: "A-1" }, id: "call-a" },
        { name: "escalate_order", args: { orderId: "B-2" }, id: "call-b" },
      ],
    );
    const oe: BatchOe = {
      answers,
      results: {
        "wait-call-a": { placed: true },
        "wait-call-b": { approved: { decision: "approve-b" } },
      },
      activityStarts: [],
      activityOutcomes: [],
      toolExecutes: [],
      stepFinalizations: [],
      dispatch: "live",
      settlement: "suspended",
    };
    wireBatchOe(oe);

    const firstAttempt = attempt("attempt-1");
    const suspended = await inToolAttempt<AgentOutput>(
      firstAttempt,
      wrapper,
      () => agent.invoke(baseContext, input),
    );
    expect(suspendedData(suspended)["interrupts"]).toEqual([
      { id: "wait-call-b", value: { orderId: "B-2", question: "escalate?" } },
    ]);
    expect(placeCalls).toBe(1);
    expect(reviewAnswer).toBeUndefined();
    expect(oe.activityOutcomes).toHaveLength(1);
    expect(oe.stepFinalizations).toHaveLength(1);
    expect(oe.stepFinalizations[0]?.suspensions).toHaveLength(1);
    expect(clients.completeExecution).not.toHaveBeenCalled();
    await expectScratchReleased(saver, firstAttempt);

    oe.settlement = "completed";
    oe.dispatch = "replay";
    const replacement = attempt("attempt-2", true);
    const resumed = await inToolAttempt<AgentOutput>(replacement, wrapper, () =>
      agent.invoke(
        {
          ...baseContext,
          resume: true,
          resumeData: answers,
        } as RequestContext,
        input,
      ),
    );
    expect(resumed.response).toMatchObject({
      status: "completed",
      resumed: true,
    });
    expect(placeCalls).toBe(1);
    expect(reviewAnswer).toEqual({ decision: "approve-b" });
    expect(oe.activityOutcomes).toHaveLength(1);
    expect(clients.completeExecution).toHaveBeenCalledOnce();
    await expectScratchReleased(saver, replacement);
  });

  it("terminates a mid-batch lost attempt without a frontier or terminal outcome", async () => {
    // Process-loss model: attempt-1 is terminated mid-superstep after
    // call-b's outcome is durably recorded while call-a is started but
    // unterminal. The harness cancels attempt-1 through the graph's abort
    // signal — the invocation settles with no suspension finalization and no
    // terminal activity outcome, exactly like a lost runner process. The
    // recorded state (one terminal sibling, one started-but-unterminal) is
    // what OE's replacement admission fence refuses with OUTCOME_UNKNOWN;
    // that fail-closed boundary is pinned at the WorkflowClient layer in
    // agent-engine-runner-shared.
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const wrapper = new SecureToolWrapper("http://oe", "execution");
    const startEvents: {
      attemptId: string;
      callId: string;
      position: Record<string, unknown>;
    }[] = [];
    const outcomeEvents: { attemptId: string; callId: string }[] = [];
    const finalizeEvents: FinalizeBody[] = [];
    // Stateful OE fixture: what durable history holds per sibling.
    const recorded = new Map<
      string,
      { state: "started" | "terminal"; result?: unknown }
    >();

    const slowTool = durableLocalTool("review_order", () => {
      if (currentAttemptContext()?.attemptId === "attempt-1") {
        // The lost sibling parks mid-effect and never settles.
        return new Promise<never>(() => {});
      }
      // The replacement fence must prevent this continuation from running.
      return { approved: { decision: "approve-a" } };
    });
    const fastTool = durableLocalTool("escalate_order", () => {
      return { approved: { decision: "approve-b" } };
    });
    const compiled = compileBatchGraph(
      saver,
      [slowTool, fastTool],
      [
        { name: "review_order", args: { orderId: "A-1" }, id: "call-a" },
        { name: "escalate_order", args: { orderId: "B-2" }, id: "call-b" },
      ],
    );

    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
        const url = String(request);
        const body = wireBody(init);
        const semanticInput = body["semantic_input"] as
          | Record<string, unknown>
          | undefined;
        const callId = String(semanticInput?.["tool_call_id"]);
        const attemptId = String(body["attempt_id"]);
        if (url.endsWith("/executor/activity/start")) {
          startEvents.push({
            attemptId,
            callId,
            position: body["position"] as Record<string, unknown>,
          });
          const entry = recorded.get(callId);
          if (entry === undefined) {
            recorded.set(callId, { state: "started" });
            return jsonResponse({
              activity_context: {
                workflow_identity: body["workflow_identity"],
                activity_id: `wait-${callId}`,
                attempt_id: attemptId,
                fencing_token: body["fencing_token"],
              },
            });
          }
          // Replacement attempt: respond from recorded activity state.
          if (entry.state === "terminal") {
            return jsonResponse({
              outcome: {
                workflow_identity: body["workflow_identity"],
                activity_id: `wait-${callId}`,
                attempt_id: attemptId,
                fencing_token: body["fencing_token"],
                outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
                result: entry.result,
              },
            });
          }
          // OE's fence: a started-but-unterminal position refuses
          // re-admission with a coded OUTCOME_UNKNOWN failure.
          return jsonResponse(
            {
              error: {
                code: "WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN",
                message:
                  "execution has unfinished activities from a previous attempt",
              },
            },
            409,
          );
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          const call = String(body["activity_id"]).replace(/^wait-/, "");
          outcomeEvents.push({ attemptId, callId: call });
          recorded.set(call, { state: "terminal", result: body["result"] });
          return jsonResponse({});
        }
        if (url.endsWith("/executor/step/finalize")) {
          finalizeEvents.push(body as FinalizeBody);
          throw new Error("the lost attempt must not finalize a frontier");
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    // Attempt-1 is driven through the exported session factory so the harness
    // can cancel it deterministically at the graph boundary.
    const firstAttempt = attempt("attempt-1");
    await runWithExecutionContext(
      {
        executionId: "execution",
        wrapper,
        oeUrl: "http://oe",
        userId: "user-1",
      },
      () =>
        runWithAttemptContext(firstAttempt, async () => {
          const session = executionSession(
            compiled as never,
            { ...baseContext } as RequestContext,
            input,
            {
              callbacks: [],
              prepareInput: null,
              resolveThreadId: null,
              durableSubgraphs: null,
            },
          ) as DurableSession;
          const { graphInput, config } = await session.prepareRun();
          const controller = new AbortController();
          const lostRun = session.runInGraphScope(() =>
            (
              compiled as never as {
                invoke(
                  input: unknown,
                  config: RunnableConfig,
                ): Promise<unknown>;
              }
            ).invoke(graphInput, { ...config, signal: controller.signal }),
          );
          for (
            let waited = 0;
            (startEvents.length < 2 || outcomeEvents.length < 1) &&
            waited < 10_000;
            waited += 10
          ) {
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
          expect(startEvents).toHaveLength(2);
          expect(outcomeEvents).toHaveLength(1);
          expect(finalizeEvents).toHaveLength(0);

          // Terminate the lost attempt: the abort settles the invocation
          // with no suspension finalization and no terminal outcome for the
          // unfinished sibling.
          controller.abort();
          await expect(lostRun).rejects.toThrow(/abort/i);
          await session.close();
        }),
    );
    await expectScratchReleased(saver, firstAttempt);
    expect(clients.completeExecution).not.toHaveBeenCalled();
    // Production never admits a replacement over this history: admission
    // fails with OUTCOME_UNKNOWN before any replacement graph, sibling
    // replay, or root commit can run (pinned at the WorkflowClient layer in
    // agent-engine-runner-shared). The loss state asserted here is exactly what that
    // fence refuses.
    expect(startEvents.map((start) => start.callId).sort()).toEqual([
      "call-a",
      "call-b",
    ]);
    expect(outcomeEvents).toEqual([
      { attemptId: "attempt-1", callId: "call-b" },
    ]);
    expect(finalizeEvents).toHaveLength(0);
  }, 30_000);

  it("keeps batch dispatch validation failures ordinary ToolMessages without an Activity", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const placed: string[] = [];
    const placeTool = durableLocalTool("review_order", ({ orderId }) => {
      placed.push(String(orderId));
      return { placed: true };
    });
    const agent = toolBatchAgent(
      saver,
      [placeTool],
      [
        { name: "unknown_tool", args: { orderId: "X-9" }, id: "call-x" },
        { name: "review_order", args: { orderId: "A-1" }, id: "call-a" },
      ],
    );
    const oe: BatchOe = {
      answers: { "wait-call-a": { placed: true } },
      results: { "wait-call-a": { placed: true } },
      activityStarts: [],
      activityOutcomes: [],
      toolExecutes: [],
      stepFinalizations: [],
      dispatch: "live",
      settlement: "completed",
    };
    wireBatchOe(oe);

    const current = attempt("attempt-1");
    const result = await inToolAttempt<AgentOutput>(
      current,
      new SecureToolWrapper("http://oe", "execution"),
      () => agent.invoke(baseContext, input),
    );

    expect(result.response).toMatchObject({ status: "completed" });
    expect(placed).toEqual(["A-1"]);
    // The unknown tool produced no durable Activity; only the valid sibling
    // dispatched one.
    expect(oe.activityStarts).toHaveLength(1);
    expect(oe.activityStarts[0]?.["activity_name"]).toBe("review_order");
    expect(oe.activityOutcomes).toHaveLength(1);
    expect(clients.completeExecution).toHaveBeenCalledOnce();

    // The handled error is visible to the model in the committed conversation:
    // an error ToolMessage correlated to call-x beside the valid sibling's
    // success ToolMessage.
    const messages = publishedMessages(clients.completeExecution);
    expect(messages).toHaveLength(4);
    expect(messages[0]).toMatchObject({ id: "durable-input:execution" });
    expect(messages[1]).toMatchObject({ id: "assistant-plan" });
    expect(messages[2]).toBeInstanceOf(ToolMessage);
    expect(messages[2]).toMatchObject({
      tool_call_id: "call-x",
      status: "error",
    });
    expect(String((messages[2] as ToolMessage).content)).toMatch(
      /unknown_tool/,
    );
    expect(messages[3]).toBeInstanceOf(ToolMessage);
    expect(messages[3]).toMatchObject({
      tool_call_id: "call-a",
      status: "success",
    });
    await expectScratchReleased(saver, current);
  });
});

describe("durable wrapped LLM calls around pauses", () => {
  it.each(
    ["invoke", "stream"].flatMap((mode) =>
      ["invoke", "stream"].flatMap((call) =>
        [
          "pause-model",
          "model-pause",
          "pause-model-pause",
          "model-pause-model",
        ].map((layout) => ({ mode, call, layout })),
      ),
    ),
  )(
    "$mode dispatches each llm.$call once across attempts (layout=$layout)",
    async ({ mode, call, layout }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      // Only the provider round trip is replaced; activity admission, ordinal
      // allocation, and replay run for real against the fake OE below.
      const dispatched: string[] = [];
      vi.spyOn(
        SecureLLMProxy.prototype as unknown as {
          streamLlm: (request: unknown) => AsyncGenerator<unknown>;
        },
        "streamLlm",
      ).mockImplementation(async function* (request: unknown) {
        const messages = (request as { messages: { content: unknown }[] })
          .messages;
        const prompt = String(messages.at(-1)?.content);
        dispatched.push(prompt);
        yield { content: `resolved:${prompt}` };
      });
      let wrapper = new SecureToolWrapper("http://oe", "execution");
      const llm = new SecureWrappedLLM(
        // A streaming-capable inner model, so `llm.stream` takes the wrapper's
        // streaming branch instead of delegating to the generate path.
        {
          model: "test-model",
          stream: () => undefined,
        } as unknown as BaseChatModel,
        () => wrapper,
      );
      const steps = layout.split("-");
      const replies: [number, string][] = [];
      const modelReply = async (prompt: string): Promise<string> => {
        const messages = [new HumanMessage(prompt)];
        if (call === "invoke") {
          return String((await llm.invoke(messages)).content);
        }
        let text = "";
        for await (const chunk of await llm.stream(messages)) {
          text += String(chunk.content);
        }
        return text;
      };
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("review", async () => {
          // Each step sees the previous step's output, so a model call's
          // prompt records which answers reached the node before it.
          let seen = "start";
          for (const [index, step] of steps.entries()) {
            if (step === "pause") {
              seen = String(interrupt({ pause: index }));
            } else {
              seen = await modelReply(`call-${index} after ${seen}`);
              replies.push([index, seen]);
            }
          }
          return { messages: [new AIMessage(seen)] };
        })
        .addEdge(START, "review")
        .addEdge("review", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);

      // Fake OE: waits answer through finalize; LLM activities replay by position.
      const waits = statefulOe();
      const recorded = new Map<string, Record<string, unknown>>();
      const positionByActivity = new Map<string, string>();
      vi.stubGlobal(
        "fetch",
        vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
          const url = String(request);
          if (url.endsWith("/executor/step/finalize")) {
            return waits.fetch(request, init);
          }
          const body = wireBody(init);
          if (url.endsWith("/executor/activity/start")) {
            const key = JSON.stringify(body["position"]);
            const outcome = recorded.get(key);
            const provenance = {
              workflow_identity: body["workflow_identity"],
              attempt_id: body["attempt_id"],
              fencing_token: body["fencing_token"],
            };
            if (outcome !== undefined) {
              return jsonResponse({ outcome: { ...outcome, ...provenance } });
            }
            const activityId = `llm-${positionByActivity.size + 1}`;
            positionByActivity.set(activityId, key);
            return jsonResponse({
              activity_context: { ...provenance, activity_id: activityId },
            });
          }
          if (url.endsWith("/executor/activity/outcome")) {
            const key = present(
              positionByActivity.get(String(body["activity_id"])),
            );
            recorded.set(key, body);
            return jsonResponse({});
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      const run = (
        current: ReturnType<typeof attempt>,
        ctx: RequestContext,
      ) => {
        // Each attempt is a fresh process: operational steps restart.
        wrapper = new SecureToolWrapper("http://oe", "execution");
        return inToolAttempt<AgentOutput | StreamEvent[]>(
          current,
          wrapper,
          () =>
            mode === "invoke"
              ? agent.invoke(ctx, input)
              : collect(agent.stream(ctx, input)),
        );
      };

      let result = await run(attempt("attempt-1"), baseContext);
      const pauses = steps.filter((step) => step === "pause").length;
      for (let number = 1; number <= pauses; number += 1) {
        // A streamed model call before the pause emits token events first.
        const suspended = Array.isArray(result)
          ? result.filter((event) => event.event !== "token")
          : result;
        const interrupts = suspendedData(suspended)["interrupts"] as {
          id: string;
        }[];
        expect(interrupts).toHaveLength(1);
        const waiting = present(interrupts[0]);
        waits.answers.set(waiting.id, `answer-${number}`);
        result = await run(attempt(`attempt-${number + 1}`, true), {
          ...baseContext,
          resume: true,
          resumeData: { [waiting.id]: `answer-${number}` },
        } as RequestContext);
      }

      // Replay the node's logic with the recorded answers to get the one
      // prompt each model call must have been dispatched with.
      const expectedPrompts: string[] = [];
      const expectedReplies = new Map<number, string>();
      let seen = "start";
      let answered = 0;
      for (const [index, step] of steps.entries()) {
        if (step === "pause") {
          answered += 1;
          seen = `answer-${answered}`;
        } else if (step === "model") {
          const prompt = `call-${index} after ${seen}`;
          expectedPrompts.push(prompt);
          seen = `resolved:${prompt}`;
          expectedReplies.set(index, seen);
        }
      }
      // Every execution of the node, live or replayed, received the one
      // recorded reply for each model call.
      expect(new Set(replies.map(([index]) => index))).toEqual(
        new Set(expectedReplies.keys()),
      );
      for (const [index, reply] of replies) {
        expect(reply).toBe(expectedReplies.get(index));
      }
      const firstCall = Math.min(...expectedReplies.keys());
      if (steps.slice(firstCall + 1).some((step) => step !== "model")) {
        // A pause after a model call re-executes the node, so that
        // call's reply was observed again from the recorded activity.
        expect(replies.length).toBeGreaterThan(expectedReplies.size);
      }
      if (Array.isArray(result)) {
        expect(result.at(-1)).toMatchObject({
          event: "result",
          data: { response: seen },
        });
      } else {
        expect(result.response).toMatchObject({
          response: seen,
          status: "completed",
        });
      }
      // Later attempts and re-entries of the node replay each recorded model
      // call, and separate calls in the node stay separate activities.
      expect(dispatched).toEqual(expectedPrompts);
      expect(recorded.size).toBe(expectedPrompts.length);
    },
  );

  it("rejects a durable graph whose node sets a retry policy", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("review", async () => ({ messages: [new AIMessage("done")] }), {
        retryPolicy: { maxAttempts: 2 },
      })
      .addEdge(START, "review")
      .addEdge("review", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    vi.stubGlobal("fetch", statefulOe().fetch);

    await expect(
      inAttempt(attempt("attempt-1"), () => agent.invoke(baseContext, input)),
    ).rejects.toThrow(/"review" sets a retry policy/);
  });

  it("rejects a durable graph whose nodes inherit a retry policy", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const builder = new StateGraph(MessagesAnnotation);
    builder.setNodeDefaults({ retryPolicy: { maxAttempts: 2 } });
    const graph = builder
      .addNode("review", async () => ({ messages: [new AIMessage("done")] }))
      .addEdge(START, "review")
      .addEdge("review", END)
      .compile({ checkpointer: saver as never });
    const agent = new LangGraphBaseAgent(graph as never);
    vi.stubGlobal("fetch", statefulOe().fetch);

    await expect(
      inAttempt(attempt("attempt-1"), () => agent.invoke(baseContext, input)),
    ).rejects.toThrow(/"review" sets a retry policy/);
  });

  it("rejects a durable graph that sets a default retry policy", async () => {
    const clients = platformClients();
    const saver = new PlatformCheckpointer({
      native: null,
      clientFactory: clients.factory,
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("review", async () => ({ messages: [new AIMessage("done")] }))
      .addEdge(START, "review")
      .addEdge("review", END)
      .compile({ checkpointer: saver as never });
    // A compiled graph's own policy is the default for all its nodes.
    (graph as { retryPolicy?: unknown }).retryPolicy = { maxAttempts: 2 };
    const agent = new LangGraphBaseAgent(graph as never);
    vi.stubGlobal("fetch", statefulOe().fetch);

    await expect(
      inAttempt(attempt("attempt-1"), () => agent.invoke(baseContext, input)),
    ).rejects.toThrow(/"root" sets a default retry policy/);
  });
});

/** Decode one durable Memory command body into (role, content) turns. */
function memoryBatch(body: Record<string, unknown>): Array<[string, string]> {
  const writes = (body["memory_writes"] ?? []) as Record<string, unknown>[];
  return writes.map((write) => {
    const payload = JSON.parse(
      Buffer.from(String(write["payload_json"] ?? ""), "base64").toString(
        "utf-8",
      ),
    ) as Record<string, unknown>;
    return [String(payload["role"]), String(payload["content"])] as [
      string,
      string,
    ];
  });
}

describe("durable guardrail review", () => {
  it.each([
    { decision: "approve", call: "invoke" },
    { decision: "approve", call: "stream" },
    { decision: "deny", call: "invoke" },
    { decision: "deny", call: "stream" },
  ] as const)(
    "pauses after the halted activity and resolves the review ($decision, $call)",
    async ({ decision, call }) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const waits = statefulOe();
      // The runtime registers LangGraph's interrupt as the pause at startup.
      registerSuspendHandler(((payload: unknown) =>
        interrupt(payload as never)) as never);

      const recorded = new Map<string, Record<string, unknown>>();
      const positionByActivity = new Map<string, string>();
      const oeRequests: Record<string, unknown>[] = [];
      const memoryCommands: Record<string, unknown>[] = [];

      // Fake OE: the call halts for review; the resolving call gets the
      // decided outcome. LLM activities replay by position.
      vi.stubGlobal(
        "fetch",
        vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
          const url = String(request);
          if (url.endsWith("/executor/step/finalize")) {
            return waits.fetch(request, init);
          }
          const body = wireBody(init);
          if (url.endsWith("/executor/activity/start")) {
            const key = JSON.stringify(body["position"]);
            const outcome = recorded.get(key);
            const provenance = {
              workflow_identity: body["workflow_identity"],
              attempt_id: body["attempt_id"],
              fencing_token: body["fencing_token"],
            };
            if (outcome !== undefined) {
              return jsonResponse({ outcome: { ...outcome, ...provenance } });
            }
            const activityId = `llm-${positionByActivity.size + 1}`;
            positionByActivity.set(activityId, key);
            return jsonResponse({
              activity_context: { ...provenance, activity_id: activityId },
            });
          }
          if (url.endsWith("/executor/activity/outcome")) {
            const key = present(
              positionByActivity.get(String(body["activity_id"])),
            );
            recorded.set(key, body);
            return jsonResponse({});
          }
          if (url.endsWith("/executor/activity/memory")) {
            memoryCommands.push(body);
            return jsonResponse({});
          }
          if (url.endsWith("/tool/execute")) {
            oeRequests.push(body);
            if (body["review_id"] === undefined) {
              return jsonResponse({
                proceed: false,
                status: "require_review",
                reason: "output needs human review",
                guardrail_review: {
                  review_id: "review-1",
                  allowed_decisions: ["approve", "deny"],
                },
              });
            }
            if (decision === "approve") {
              return jsonResponse({
                proceed: true,
                status: "success",
                result: { content: "the held answer" },
              });
            }
            return jsonResponse({
              proceed: false,
              status: "blocked",
              reason: "guardrail review denied by reviewer",
            });
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      let wrapper = new SecureToolWrapper("http://oe", "execution");
      const llm = new SecureWrappedLLM(
        {
          model: "test-model",
          stream: () => undefined,
        } as unknown as BaseChatModel,
        () => wrapper,
      );
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("answer_claim", async () => {
          const messages = [new HumanMessage("summarize the claim")];
          let content = "";
          if (call === "invoke") {
            content = String((await llm.invoke(messages)).content);
          } else {
            for await (const chunk of await llm.stream(messages)) {
              content += String(chunk.content);
            }
          }
          return { messages: [new AIMessage(content)] };
        })
        .addEdge(START, "answer_claim")
        .addEdge("answer_claim", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);

      const run = (
        current: ReturnType<typeof attempt>,
        ctx: RequestContext,
      ) => {
        // Each attempt is a fresh process: operational steps restart, and the
        // turn's user input is pending again until memory has it.
        wrapper = new SecureToolWrapper("http://oe", "execution");
        wrapper.durableMemory = new DurableMemoryState("summarize the claim");
        return inToolAttempt<AgentOutput | StreamEvent[]>(
          current,
          wrapper,
          () => agent.invoke(ctx, input),
        );
      };

      // Attempt 1: OE halts the call. The halt is the LLM activity's result
      // and the node pauses for the review it names.
      const paused = await run(attempt("attempt-1"), baseContext);
      const interrupts = suspendedData(paused)["interrupts"] as {
        id: string;
        value: unknown;
      }[];
      expect(interrupts).toHaveLength(1);
      const waiting = present(interrupts[0]);
      expect(waiting.value).toEqual({
        guardrail_review: { review_id: "review-1" },
      });
      expect(oeRequests[0]?.["review_protocol"]).toBe(1);
      expect(oeRequests[0] ?? {}).not.toHaveProperty("review_id");
      // The halt is closed in memory with nothing written.
      expect(memoryCommands.map(memoryBatch)).toEqual([[]]);
      expect(
        [...recorded.values()].map((outcome) => outcome["outcome_kind"]),
      ).toEqual(["ACTIVITY_OUTCOME_KIND_COMPLETED"]);

      // The reviewer answers, and a new attempt picks the turn up.
      const answer = {
        guardrail_review: { review_id: "review-1", decision },
      };
      waits.answers.set(waiting.id, answer);
      const resumed = run(attempt("attempt-2", true), {
        ...baseContext,
        resume: true,
        resumeData: { [waiting.id]: answer },
      } as RequestContext);

      if (decision === "deny") {
        await expect(resumed).rejects.toBeInstanceOf(PolicyDeniedException);
      } else {
        const result = (await resumed) as AgentOutput;
        expect(result.response).toMatchObject({ response: "the held answer" });
      }

      // The halted call replayed from its recorded result. Only the resolving
      // call reached OE, naming the review, at a step after the halted one.
      expect(oeRequests).toHaveLength(2);
      const halted = present(oeRequests[0]);
      const resolving = present(oeRequests[1]);
      expect(resolving["review_id"]).toBe("review-1");
      expect(resolving["review_protocol"]).toBe(1);
      expect(Number(resolving["step_number"])).toBeGreaterThan(
        Number(halted["step_number"]),
      );
      expect(recorded.size).toBe(2);

      // The answered wait is closed in memory under the attempt that resumed
      // it, with nothing written.
      expect(
        memoryCommands
          .filter((command) => command["activity_id"] === waiting.id)
          .map(memoryBatch),
      ).toEqual([[]]);

      // Neither the halt nor the reviewer's decision is conversation: the
      // only content memory ever receives is the user's input with the
      // released response.
      const written = memoryCommands
        .map(memoryBatch)
        .filter((batch) => batch.length > 0);
      if (decision === "approve") {
        expect(written).toEqual([
          [
            ["user", "summarize the claim"],
            ["assistant", "the held answer"],
          ],
        ]);
      } else {
        expect(written).toEqual([]);
      }
    },
  );
});

describe("durable streamed LLM abandonment", () => {
  it("closing a live streamed call cancels its transport and settles the activity", async () => {
    const starts: string[] = [];
    const outcomes: Record<string, unknown>[] = [];
    let streamOpened = false;
    let streamCancelled = false;
    let oeCalls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: string | URL | Request, init?: RequestInit) => {
        const url = String(request);
        if (url.endsWith("/executor/activity/start")) {
          const body = wireBody(init);
          const provenance = {
            workflow_identity: body["workflow_identity"],
            attempt_id: body["attempt_id"],
            fencing_token: body["fencing_token"],
          };
          const activityId = `llm-${starts.length + 1}`;
          starts.push(activityId);
          return jsonResponse({
            activity_context: { ...provenance, activity_id: activityId },
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomes.push(wireBody(init));
          return jsonResponse({});
        }
        if (url.endsWith("/tool/execute")) {
          oeCalls += 1;
          if (oeCalls === 1) {
            return jsonResponse({
              proceed: true,
              status: "success",
              route_to: "http://oe/live-stream",
            });
          }
          return jsonResponse({
            proceed: true,
            status: "success",
            result: { content: "second answer" },
          });
        }
        if (url === "http://oe/live-stream") {
          streamOpened = true;
          const encoder = new TextEncoder();
          return new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                // One visible chunk, then the stream stays open.
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({ content: "partial" })}\n\n`,
                  ),
                );
              },
              cancel() {
                streamCancelled = true;
              },
            }),
            { headers: { "Content-Type": "text/event-stream" } },
          );
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const wrapper = new SecureToolWrapper("http://oe", "execution");
    const llm = new SecureWrappedLLM(
      {
        model: "test-model",
        stream: () => undefined,
      } as unknown as BaseChatModel,
      () => wrapper,
    );

    await inToolAttempt(attempt("attempt-1"), wrapper, async () => {
      // Drive the adapter iterator directly. The public `llm.stream()` wrapper
      // eagerly starts its next pull, and LangChain's IterableReadableStream
      // queues the consumer's cancel behind that pending pull on an endless
      // source, so a break there would deadlock regardless of this adapter.
      const iterator = llm._streamIterator(
        [new HumanMessage("summarize the claim")] as never,
        {} as never,
      );
      const first = await iterator.next();
      expect(String(first.value?.content)).toBe("partial");
      await iterator.return(undefined);

      // The attempt is still usable: a following call dispatches and completes.
      const answer = await llm.invoke([new HumanMessage("second question")]);
      expect(String(answer.content)).toBe("second answer");
    });

    expect(streamOpened).toBe(true);
    expect(streamCancelled).toBe(true);
    expect(starts).toHaveLength(2);
    const abandoned = outcomes.filter(
      (outcome) =>
        outcome["outcome_kind"] === "ACTIVITY_OUTCOME_KIND_FAILED" &&
        JSON.stringify(outcome["error"] ?? "").includes("abandoned"),
    );
    expect(abandoned).toHaveLength(1);
    expect(outcomes).toHaveLength(2);
  });
});
