import { create } from "@bufbuild/protobuf";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
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
  SecureToolWrapper,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  createSecureToolFunction,
  currentAttemptContext,
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

  it.each(["first?", "second?"])(
    "rejects a second same-task interrupt with payload %s",
    async (secondQuestion) => {
      const clients = platformClients();
      const saver = new PlatformCheckpointer({
        native: null,
        clientFactory: clients.factory,
      });
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("approval", async () => {
          interrupt({ question: "first?" });
          interrupt({ question: secondQuestion });
          return { messages: [new AIMessage("resumed")] };
        })
        .addEdge(START, "approval")
        .addEdge("approval", END)
        .compile({ checkpointer: saver as never });
      const agent = new LangGraphBaseAgent(graph as never);
      let settlement: "suspended" | "completed" = "suspended";
      vi.stubGlobal(
        "fetch",
        vi.fn(async (_request: string | URL | Request, init?: RequestInit) => {
          const body = requestBody(init);
          return settlementResponse(body, settlement, { "wait-1": "approved" });
        }),
      );

      const first = await inAttempt(attempt("attempt-1"), () =>
        agent.invoke(baseContext, input),
      );
      expect(suspendedData(first)["interrupts"]).toEqual([
        { id: "wait-1", value: { question: "first?" } },
      ]);

      settlement = "completed";
      await expect(
        inAttempt(attempt("attempt-2", true), () =>
          agent.invoke(
            {
              ...baseContext,
              resume: true,
              resumeData: { "wait-1": "approved" },
            } as RequestContext,
            input,
          ),
        ),
      ).rejects.toThrow("multiple sequential direct interrupts");
      expect(clients.completeExecution).not.toHaveBeenCalled();
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
