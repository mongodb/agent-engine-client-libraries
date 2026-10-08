/**
 * Guardrail review of a model call on a durable session.
 *
 * Port of `agent-engine-runner-shared/tests/unit/test_durable_guardrail_review.py`.
 * OE halts the call and names a review. The halt is the call's recorded result,
 * the pause happens after it, and a second call that names the review gets the
 * decided outcome from OE.
 */

import { create, fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import type { LLMStreamChunk, Message } from "@mongodb-js/agent-engine-sdk";
import { afterEach, describe, expect, test, vi } from "vitest";

import { runWithExecutionContext } from "../../src/context.js";
import { registerSuspendHandler, resetHooks } from "../../src/hooks.js";
import {
  SecureLLMProxy,
  guardrailReviewWaitId,
} from "../../src/secure_llm_proxy.js";
import {
  GUARDRAIL_REVIEW_INVALID_CODE,
  LLMInvocationError,
  OperationalStepAllocator,
  PolicyDeniedException,
  requestOeApproval,
} from "../../src/secure_wrapper.js";
import {
  ActivityContextSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  noteGuardrailReviewWait,
  runWithAttemptContext,
} from "../../src/workflow/index.js";
import { DurableMemoryState } from "../../src/workflow/memory.js";

const REVIEW_ID = "review-1";
const MESSAGES: Message[] = [{ role: "user", content: "summarize the claim" }];
const HELD = { content: "the held answer" };

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org-1",
    projectId: "project-1",
    workspaceId: "workspace-1",
  }),
  sessionId: "session-1",
  executionId: "execution-1",
});

const attempt = create(AttemptContextSchema, {
  attemptId: "attempt",
  fencingToken: 7n,
  ownerId: "aer-1",
  workflowIdentity: identity,
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function dispatchResponse(activityId: string): Response {
  const context = create(ActivityContextSchema, {
    workflowIdentity: identity,
    activityId,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
  });
  return jsonResponse({
    activity_context: toJson(ActivityContextSchema, context, {
      useProtoFieldName: true,
    }),
  });
}

function replayResponse(activityId: string, result: unknown): Response {
  const outcome = create(ActivityOutcomeSchema, {
    workflowIdentity: identity,
    activityId,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
    outcomeKind: ActivityOutcomeKind.COMPLETED,
    result: fromJson(ValueSchema, result ?? null),
  });
  return jsonResponse({
    outcome: toJson(ActivityOutcomeSchema, outcome, {
      useProtoFieldName: true,
    }),
  });
}

function requestBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  return JSON.parse(init.body) as Record<string, unknown>;
}

/** A workflow history shared by attempts: a recorded position replays. */
class History {
  readonly recorded = new Map<number, unknown>();
  readonly kinds = new Map<number, string>();
  readonly memoryCommands: Record<string, unknown>[] = [];
  private readonly dispatched = new Map<string, number>();

  startActivity(ordinal: number): Response {
    if (this.recorded.has(ordinal)) {
      return replayResponse(`activity-${ordinal}`, this.recorded.get(ordinal));
    }
    const activityId = `activity-${ordinal}`;
    this.dispatched.set(activityId, ordinal);
    return dispatchResponse(activityId);
  }

  reportOutcome(body: Record<string, unknown>): void {
    const activityId = String(body["activity_id"]);
    const ordinal = this.dispatched.get(activityId);
    if (ordinal === undefined) {
      throw new Error(`outcome for unknown activity: ${activityId}`);
    }
    this.recorded.set(
      ordinal,
      body["result"] === undefined ? null : body["result"],
    );
    this.kinds.set(ordinal, String(body["outcome_kind"]));
  }

  recordedHalt(ordinal: number): Record<string, unknown> {
    const result = this.recorded.get(ordinal);
    if (result === null || typeof result !== "object") {
      throw new Error(`recorded result ${ordinal} is not an object`);
    }
    return result as Record<string, unknown>;
  }
}

class Paused extends Error {
  constructor(readonly value: unknown) {
    super("paused");
    this.name = "Paused";
  }
}

function haltResponse(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    proceed: false,
    status: "require_review",
    reason: "output needs human review",
    latest_step_number: 1,
    guardrail_review: {
      review_id: REVIEW_ID,
      allowed_decisions: ["approve", "deny"],
      reason: "output needs human review",
      guardrails: [
        { id: "policy-1", name: "pii", category: "output_validation" },
      ],
    },
    ...overrides,
  };
}

interface Run {
  readonly history: History;
  readonly oeBodies: Record<string, unknown>[];
  queueOe(response: unknown): void;
  stream(options?: {
    protocol?: boolean;
    stepsBefore?: number;
    steps?: OperationalStepAllocator;
    memory?: DurableMemoryState | null;
  }): Promise<LLMStreamChunk[]>;
}

// One fetch stub routes every run by its OE origin, so several runs can be
// live in one test (parallel branches share a superstep).
const runHandlers = new Map<
  string,
  (url: string, init?: RequestInit) => Response | Promise<Response>
>();
let fetchRouterInstalled = false;

function installFetchRouter(): void {
  if (fetchRouterInstalled) return;
  fetchRouterInstalled = true;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const handler = runHandlers.get(new URL(url).origin);
      if (handler === undefined) {
        throw new Error(`unexpected URL: ${url}`);
      }
      return handler(url, init);
    }),
  );
}

/** One attempt of the node: a fresh proxy over a shared history and OE. */
function makeRun(
  memory: DurableMemoryState | null = null,
  oeUrl = "http://oe:8000",
): Run {
  const history = new History();
  const oeResponses: unknown[] = [];
  const oeBodies: Record<string, unknown>[] = [];

  installFetchRouter();
  runHandlers.set(new URL(oeUrl).origin, (url, init) => {
    if (url.endsWith("/executor/activity/start")) {
      const body = requestBody(init);
      const position = body["position"] as Record<string, unknown> | undefined;
      return history.startActivity(Number(position?.["activity_ordinal"]));
    }
    if (url.endsWith("/tool/execute")) {
      oeBodies.push(requestBody(init));
      const response = oeResponses.shift();
      if (response === undefined) {
        throw new Error("no OE response queued");
      }
      return jsonResponse(response);
    }
    if (url.endsWith("/executor/activity/outcome")) {
      history.reportOutcome(requestBody(init));
      return jsonResponse({});
    }
    if (url.endsWith("/executor/activity/memory")) {
      history.memoryCommands.push(requestBody(init));
      return jsonResponse({});
    }
    throw new Error(`unexpected URL: ${url}`);
  });

  return {
    history,
    oeBodies,
    queueOe: (response: unknown) => {
      oeResponses.push(response);
    },
    stream: async (options = {}) => {
      const steps = options.steps ?? new OperationalStepAllocator();
      if (options.steps === undefined) {
        steps.observeAtLeast(options.stepsBefore ?? 0);
      }
      const proxy = new SecureLLMProxy({
        oeUrl,
        executionId: "execution-1",
        modelName: "gpt-test",
        operationalSteps: steps,
        durableMemory: options.memory ?? memory,
        guardrailReviewProtocol: options.protocol ?? true,
      });
      return runWithExecutionContext(
        {
          executionId: "execution-1",
          wrapper: null,
          oeUrl,
          userId: "user-1",
          sessionId: "session-1",
        },
        () =>
          runWithAttemptContext(attempt, async () => {
            const chunks: LLMStreamChunk[] = [];
            for await (const chunk of proxy.stream(
              MESSAGES,
              null,
              null,
              null,
              "llm:task-1:1",
            )) {
              chunks.push(chunk);
            }
            return chunks;
          }),
      );
    },
  };
}

/** Register a handler that pauses, and return the values it paused with. */
function pauseOnReview(): unknown[] {
  const paused: unknown[] = [];
  registerSuspendHandler(((value: Record<string, unknown>) => {
    paused.push(value);
    throw new Paused(value);
  }) as never);
  return paused;
}

function answerReview(decision: string, reviewId = REVIEW_ID): void {
  registerSuspendHandler(((value: Record<string, unknown>) => ({
    guardrail_review: {
      ...(value["guardrail_review"] as Record<string, unknown>),
      review_id: reviewId,
      decision,
    },
  })) as never);
}

/** Attempt 1: the call is halted for review and the node pauses. */
async function halt(run: Run): Promise<void> {
  run.queueOe(haltResponse());
  pauseOnReview();
  await expect(run.stream()).rejects.toBeInstanceOf(Paused);
  run.oeBodies.length = 0;
}

function memoryWrites(run: Run): Array<Array<[string, string]>> {
  return run.history.memoryCommands.map((command) => {
    const writes = (command["memory_writes"] ?? []) as Array<
      Record<string, unknown>
    >;
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
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  resetHooks();
  runHandlers.clear();
  fetchRouterInstalled = false;
});

describe("durable guardrail review", () => {
  test("halt is the call's recorded result and the pause follows it", async () => {
    const run = makeRun(new DurableMemoryState("summarize the claim"));
    run.queueOe(haltResponse());
    const paused = pauseOnReview();

    await expect(run.stream()).rejects.toBeInstanceOf(Paused);

    // The halted call opted in to the protocol and named no review.
    expect(run.oeBodies).toHaveLength(1);
    expect(run.oeBodies[0]?.["review_protocol"]).toBe(1);
    expect(run.oeBodies[0] ?? {}).not.toHaveProperty("review_id");
    // The activity completed with the halt; the pause names only the review.
    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
    const recorded = run.history.recordedHalt(1);
    expect(
      (recorded["guardrail_review_halt"] as Record<string, unknown>)[
        "review_id"
      ],
    ).toBe(REVIEW_ID);
    expect(recorded["guardrail_review_halt_step"]).toBe(1);
    expect(paused).toEqual([{ guardrail_review: { review_id: REVIEW_ID } }]);
    // A halt is not a model response: its activity is closed in memory with
    // nothing written.
    expect(memoryWrites(run)).toEqual([[]]);
  });

  test("approve resolves with a second call that names the review", async () => {
    const run = makeRun(new DurableMemoryState("summarize the claim"));
    await halt(run);
    run.queueOe({
      proceed: true,
      status: "success",
      result: HELD,
      latest_step_number: 2,
    });
    answerReview("approve");

    const chunks = await run.stream();

    expect(chunks.map((chunk) => chunk.content ?? "").join("")).toBe(
      "the held answer",
    );
    // The halted call replayed from history; only the resolving call reached
    // OE, at a step after the halted one.
    expect(run.oeBodies).toHaveLength(1);
    expect(run.oeBodies[0]?.["review_protocol"]).toBe(1);
    expect(run.oeBodies[0]?.["review_id"]).toBe(REVIEW_ID);
    expect(run.oeBodies[0]?.["step_number"]).toBe(2);
    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
    expect(run.history.kinds.get(2)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
    expect(run.history.recorded.get(2)).toMatchObject(HELD);
    // The released response is the turn's model response.
    expect(memoryWrites(run)).toEqual([
      [],
      [],
      [
        ["user", "summarize the claim"],
        ["assistant", "the held answer"],
      ],
    ]);
  });

  test("resolving call takes a step after the halted one when the attempt counts fewer", async () => {
    // The attempt that halted re-ran an earlier node to rebuild its interrupt,
    // so it counted four steps before this call. The attempt that answers the
    // review replays that interrupt without the re-run and counts two, then
    // replays this halted call. OE resolves a review only at a later step.
    const run = makeRun();
    run.queueOe(haltResponse());
    pauseOnReview();
    await expect(run.stream({ stepsBefore: 4 })).rejects.toBeInstanceOf(Paused);
    expect(run.history.recordedHalt(1)["guardrail_review_halt_step"]).toBe(5);
    run.oeBodies.length = 0;
    run.queueOe({ proceed: true, status: "success", result: HELD });
    answerReview("approve");

    const chunks = await run.stream({ stepsBefore: 2 });

    expect(chunks.map((chunk) => chunk.content ?? "").join("")).toBe(
      "the held answer",
    );
    expect(run.oeBodies[0]?.["review_id"]).toBe(REVIEW_ID);
    expect(run.oeBodies[0]?.["step_number"]).toBe(6);
  });

  test("parallel halted calls resolve at steps neither of them halted at", async () => {
    // Two branches of one superstep share the step counter and each halts.
    const first = makeRun(null, "http://oe-a");
    const second = makeRun(null, "http://oe-b");
    const halting = new OperationalStepAllocator();
    halting.observeAtLeast(4);
    pauseOnReview();
    for (const [run, reviewId] of [
      [first, "review-a"],
      [second, "review-b"],
    ] as const) {
      run.queueOe(haltResponse({ guardrail_review: { review_id: reviewId } }));
      await expect(run.stream({ steps: halting })).rejects.toBeInstanceOf(
        Paused,
      );
      run.oeBodies.length = 0;
      run.queueOe({ proceed: true, status: "success", result: HELD });
    }

    // The answering attempt counted fewer steps. Both branches replay their
    // halt and pause before the framework re-runs them with the answers.
    const answering = new OperationalStepAllocator();
    answering.observeAtLeast(2);
    pauseOnReview();
    for (const run of [first, second]) {
      await expect(run.stream({ steps: answering })).rejects.toBeInstanceOf(
        Paused,
      );
    }
    registerSuspendHandler(((value: Record<string, unknown>) => ({
      guardrail_review: {
        ...(value["guardrail_review"] as Record<string, unknown>),
        decision: "approve",
      },
    })) as never);
    await first.stream({ steps: answering });
    await second.stream({ steps: answering });

    const resolved = [first.oeBodies[0], second.oeBodies[0]].map((body) =>
      Number(body?.["step_number"]),
    );
    expect(new Set(resolved).size).toBe(2);
    expect(Math.min(...resolved)).toBeGreaterThan(6);
  });

  test("a halt with no suspend handler is an error, not a denial", async () => {
    const run = makeRun();
    run.queueOe(haltResponse());

    const consume = run.stream();

    await expect(consume).rejects.toBeInstanceOf(LLMInvocationError);
    await expect(consume).rejects.toThrow("no suspend handler registered");
  });

  test("deny ends the call as a policy denial", async () => {
    const run = makeRun();
    await halt(run);
    run.queueOe({
      proceed: false,
      status: "blocked",
      reason: "guardrail review denied by reviewer",
      guardrail_id: "policy-1",
      guardrail_category: "output_validation",
    });
    answerReview("deny");

    const error = await run.stream().catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(PolicyDeniedException);
    expect((error as PolicyDeniedException).guardrailMeta).toEqual({
      guardrail_id: "policy-1",
      guardrail_category: "output_validation",
    });
    expect(run.oeBodies[0]?.["review_id"]).toBe(REVIEW_ID);
    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
    expect(run.history.kinds.get(2)).toBe("ACTIVITY_OUTCOME_KIND_DENIED");
    expect(memoryWrites(run).every((writes) => writes.length === 0)).toBe(true);
  });

  test("a resolved review replays without reaching OE", async () => {
    const run = makeRun();
    await halt(run);
    run.queueOe({ proceed: true, status: "success", result: HELD });
    answerReview("approve");
    await run.stream();
    run.oeBodies.length = 0;

    const chunks = await run.stream();

    expect(chunks.map((chunk) => chunk.content ?? "").join("")).toBe(
      "the held answer",
    );
    expect(run.oeBodies).toHaveLength(0);
  });

  test("an answer for another review resolves nothing", async () => {
    const run = makeRun();
    await halt(run);
    answerReview("approve", "review-2");

    await expect(run.stream()).rejects.toThrow("different review");

    expect(run.oeBodies).toHaveLength(0);
    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
  });

  test.each(["approve", null, { decision: "approve" }])(
    "an answer that names no review resolves nothing (%j)",
    async (answer) => {
      const run = makeRun();
      await halt(run);
      registerSuspendHandler(
        ((_value: Record<string, unknown>) => answer) as never,
      );

      await expect(run.stream()).rejects.toBeInstanceOf(LLMInvocationError);

      expect(run.oeBodies).toHaveLength(0);
    },
  );

  test("a halt that names no review fails closed", async () => {
    const run = makeRun();
    run.queueOe(haltResponse({ guardrail_review: null }));
    const paused = pauseOnReview();

    await expect(run.stream()).rejects.toThrow("named no review");

    expect(paused).toEqual([]);
    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_FAILED");
  });

  test("a halt in place of a resolution fails closed", async () => {
    const run = makeRun();
    await halt(run);
    run.queueOe(haltResponse());
    answerReview("approve");

    await expect(run.stream()).rejects.toThrow("named no review");

    expect(run.history.kinds.get(1)).toBe("ACTIVITY_OUTCOME_KIND_COMPLETED");
    expect(run.history.kinds.get(2)).toBe("ACTIVITY_OUTCOME_KIND_FAILED");
    expect(memoryWrites(run).every((writes) => writes.length === 0)).toBe(true);
  });

  test("a resolving activity that replays a halt fails closed", async () => {
    const run = makeRun();
    await halt(run);
    // A halt-shaped result recorded at the resolving activity's position does
    // not belong to that activity; it must not become a second review pause.
    run.history.recorded.set(2, {
      guardrail_review_halt: {
        review_id: REVIEW_ID,
        allowed_decisions: ["approve", "deny"],
        guardrails: [],
      },
      guardrail_review_halt_step: 1,
    });
    answerReview("approve");

    await expect(run.stream()).rejects.toThrow(/resolving activity/);
  });

  test("without the adapter opt-in the call does not use the protocol", async () => {
    const run = makeRun();
    run.queueOe({ proceed: true, status: "success", result: HELD });

    await run.stream({ protocol: false });

    expect(run.oeBodies[0] ?? {}).not.toHaveProperty("review_protocol");
  });

  test("outside a durable attempt the call does not use the protocol", async () => {
    const oeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/tool/execute")) {
          oeBodies.push(requestBody(init));
          return jsonResponse({
            proceed: true,
            status: "success",
            result: HELD,
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );
    const proxy = new SecureLLMProxy({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      modelName: "gpt-test",
      guardrailReviewProtocol: true,
    });

    for await (const _chunk of proxy.stream(MESSAGES)) {
      // drain
    }

    expect(oeBodies[0] ?? {}).not.toHaveProperty("review_protocol");
  });

  test("a pause is a review wait only when this attempt recorded its halt", async () => {
    const value = { guardrail_review: { review_id: REVIEW_ID } };

    // Outside an attempt, and for a value an application wrote, it is not one.
    expect(guardrailReviewWaitId(value)).toBeNull();
    await runWithAttemptContext(attempt, async () => {
      expect(guardrailReviewWaitId(value)).toBeNull();
      noteGuardrailReviewWait(REVIEW_ID);
      expect(guardrailReviewWaitId(value)).toBe(REVIEW_ID);
      expect(
        guardrailReviewWaitId({ guardrail_review: { review_id: "another" } }),
      ).toBeNull();
      expect(guardrailReviewWaitId("approve")).toBeNull();
    });
    // A later attempt starts with nothing recorded.
    await runWithAttemptContext(attempt, async () => {
      expect(guardrailReviewWaitId(value)).toBeNull();
    });
  });
});

describe("review fields on the wire", () => {
  function stubOe(response: Response): {
    bodies: Record<string, unknown>[];
  } {
    const bodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/tool/execute")) {
          bodies.push(requestBody(init));
          return response.clone();
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );
    return { bodies };
  }

  function response(status: number, body: unknown): Response {
    return jsonResponse(body, status);
  }

  test("review fields are sent only when set", async () => {
    const ok = response(200, { proceed: true });

    const without = stubOe(ok);
    await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
    });
    expect(without.bodies[0] ?? {}).not.toHaveProperty("review_protocol");
    expect(without.bodies[0] ?? {}).not.toHaveProperty("review_id");

    const withFields = stubOe(ok);
    await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
      reviewProtocol: 1,
      reviewId: REVIEW_ID,
    });
    expect(withFields.bodies[0]?.["review_protocol"]).toBe(1);
    expect(withFields.bodies[0]?.["review_id"]).toBe(REVIEW_ID);

    // Each field is independent, as in Python: a resolving call that names a
    // review without opting in must still reach OE's refusal.
    const reviewOnly = stubOe(ok);
    await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
      reviewId: REVIEW_ID,
    });
    expect(reviewOnly.bodies[0] ?? {}).not.toHaveProperty("review_protocol");
    expect(reviewOnly.bodies[0]?.["review_id"]).toBe(REVIEW_ID);
  });

  test("a halt response carries the review", async () => {
    stubOe(response(200, haltResponse()));

    const result = await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
    });

    expect(result.guardrail_review?.review_id).toBe(REVIEW_ID);
    expect(result.guardrail_review?.guardrails[0]?.id).toBe("policy-1");
  });

  test("a refused resolution is a failure, not an unreachable OE", async () => {
    stubOe(
      response(409, {
        error: "guardrail review cannot resolve this call",
        code: GUARDRAIL_REVIEW_INVALID_CODE,
      }),
    );

    const error = await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
      reviewProtocol: 1,
      reviewId: REVIEW_ID,
    }).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(LLMInvocationError);
    expect((error as LLMInvocationError).error_code).toBe(
      GUARDRAIL_REVIEW_INVALID_CODE,
    );
  });

  test("another conflict still blocks as before", async () => {
    stubOe(response(409, { error: "something else" }));

    const error = await requestOeApproval({
      oeUrl: "http://oe:8000",
      executionId: "execution-1",
      toolName: "invoke_llm",
      arguments: { messages: [] },
      step: 3,
    }).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(PolicyDeniedException);
    expect((error as PolicyDeniedException).message).toContain(
      "OE unreachable",
    );
  });
});

describe("durable review conversation memory", () => {
  test("the halt is closed with nothing written and the input stays pending", async () => {
    const run = makeRun(new DurableMemoryState("summarize the claim"));
    await halt(run);

    // Every completed activity needs an acknowledged batch before its step
    // can commit; the halt's batch is empty.
    expect(memoryWrites(run)).toEqual([[]]);
  });

  test("the released response is written with the user's input", async () => {
    const run = makeRun(new DurableMemoryState("summarize the claim"));
    await halt(run);
    run.queueOe({ proceed: true, status: "success", result: HELD });
    answerReview("approve");

    await run.stream();

    const [haltLive, haltReplayed, released] = memoryWrites(run);
    expect(haltLive).toEqual([]);
    expect(haltReplayed).toEqual([]);
    expect(released).toEqual([
      ["user", "summarize the claim"],
      ["assistant", "the held answer"],
    ]);
  });

  test("a denied response is never written", async () => {
    const run = makeRun(new DurableMemoryState("summarize the claim"));
    await halt(run);
    run.queueOe({
      proceed: false,
      status: "blocked",
      reason: "guardrail review denied by reviewer",
    });
    answerReview("deny");

    await run.stream().catch(() => undefined);

    expect(memoryWrites(run).every((writes) => writes.length === 0)).toBe(true);
  });
});
