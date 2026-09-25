import { create, fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import type { LLMStreamChunk, Message } from "@mongodb-js/agent-engine-sdk";
import { afterEach, describe, expect, test, vi } from "vitest";

import { runWithExecutionContext } from "../../src/context.js";
import { AppBoundRuntime } from "../../src/memory_appbound.js";
import { suspendPayloadToJson } from "../../src/models.js";
import { SecureLLMProxy } from "../../src/secure_llm_proxy.js";
import { registerSuspendHandler, resetHooks } from "../../src/hooks.js";
import {
  CALL_INTERRUPTED_ARTIFACT_KEY,
  ExternalAPICallError,
  LLMInvocationError,
  PolicyDeniedException,
  SecureToolWrapper,
  ToolCallTimeoutError,
  ToolExecutionError,
} from "../../src/secure_wrapper.js";
import {
  ActivityContextSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  AttemptContextSchema,
  ReplayedActivityFailedError,
  TenantScopeSchema,
  WorkflowClientError,
  WorkflowErrorCode,
  WorkflowErrorSchema,
  WorkflowIdentitySchema,
  preallocateActivityOrdinals,
  runWithAttemptContext,
  toolActivityKey,
} from "../../src/workflow/index.js";

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session",
  executionId: "execution",
});

const attempt = create(AttemptContextSchema, {
  attemptId: "attempt",
  fencingToken: 5n,
  ownerId: "owner",
  workflowIdentity: identity,
  heartbeatIntervalMs: 1000n,
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function dispatchResponse(activityId = "activity"): Response {
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
    result: fromJson(ValueSchema, result),
  });
  return jsonResponse({
    outcome: toJson(ActivityOutcomeSchema, outcome, {
      useProtoFieldName: true,
    }),
  });
}

function failedReplayResponse(activityId: string, message: string): Response {
  const outcome = create(ActivityOutcomeSchema, {
    workflowIdentity: identity,
    activityId,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
    outcomeKind: ActivityOutcomeKind.FAILED,
    error: create(WorkflowErrorSchema, {
      code: WorkflowErrorCode.OUTCOME_UNKNOWN,
      message,
    }),
  });
  return jsonResponse({
    outcome: toJson(ActivityOutcomeSchema, outcome, {
      useProtoFieldName: true,
    }),
  });
}

function sseResponse(events: unknown[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const event of events) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
          );
        }
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

interface FetchCall {
  url: string;
  body: Record<string, unknown>;
}

function requestBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  return JSON.parse(init.body) as Record<string, unknown>;
}

function runWithDurableExecution<T>(fn: () => T): T {
  return runWithExecutionContext(
    {
      executionId: "execution",
      wrapper: null,
      oeUrl: "http://oe",
      userId: "user",
      sessionId: "session",
    },
    () => runWithAttemptContext(attempt, fn),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  resetHooks();
});

describe("durable Tool effects", () => {
  test("replay returns the recorded Tool result without external dispatch", async () => {
    const calls: FetchCall[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        calls.push({ url, body: requestBody(init) });
        if (url.endsWith("/executor/activity/start")) {
          return replayResponse("tool-replay", { cached: true });
        }
        throw new Error(`unexpected external dispatch: ${url}`);
      }),
    );

    const result = await runWithAttemptContext(attempt, () =>
      new SecureToolWrapper("http://oe", "execution").executeTool(
        "lookup",
        { q: "teal" },
        {
          metadata: { source: "agent" },
          providerType: "oauth",
          scopes: ["write", "read"],
          toolCallId: "call-1",
        },
      ),
    );

    expect(result).toEqual({ cached: true });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("http://oe/executor/activity/start");
    expect(calls[0]?.body["semantic_input"]).toEqual({
      arguments: { q: "teal" },
      is_local: false,
      metadata: { source: "agent" },
      provider_type: "oauth",
      scopes: ["read", "write"],
      tool_call_id: "call-1",
    });
  });

  test("replayed Tool failure preserves the public error type", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return failedReplayResponse("tool-replay", "recorded Tool failure");
        }
        throw new Error(`unexpected external dispatch: ${url}`);
      }),
    );

    const error = await runWithAttemptContext(attempt, () =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool("lookup", {}, { toolCallId: "call-1" })
        .catch((caught: unknown) => caught),
    );

    expect(error).toBeInstanceOf(ToolExecutionError);
    expect(error).toHaveProperty("error", "recorded Tool failure");
    expect(error).toHaveProperty(
      "cause",
      expect.any(ReplayedActivityFailedError),
    );
  });

  test("preallocated Tool call ids preserve their ordinals", async () => {
    const startBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (!url.endsWith("/executor/activity/start")) {
          throw new Error(`unexpected external dispatch: ${url}`);
        }
        startBodies.push(requestBody(init));
        return replayResponse(`tool-${startBodies.length}`, { ok: true });
      }),
    );

    await runWithAttemptContext(attempt, async () => {
      preallocateActivityOrdinals([
        toolActivityKey("call-a"),
        toolActivityKey("call-b"),
      ]);
      const wrapper = new SecureToolWrapper("http://oe", "execution");
      await wrapper.executeTool("second", {}, { toolCallId: "call-b" });
      await wrapper.executeTool("first", {}, { toolCallId: "call-a" });
    });

    expect(
      startBodies.map(
        (body) =>
          (body["position"] as Record<string, unknown>)["activity_ordinal"],
      ),
    ).toEqual(["2", "1"]);
  });

  test("keyed Tool activities may overlap within one attempt", async () => {
    const starts: string[] = [];
    const bothStarted = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (!url.endsWith("/executor/activity/start")) {
          throw new Error(`unexpected external dispatch: ${url}`);
        }
        const body = requestBody(init);
        const semanticInput = body["semantic_input"] as Record<string, unknown>;
        starts.push(String(semanticInput["tool_call_id"]));
        const startIndex = starts.length;
        if (starts.length === 2) bothStarted.resolve();
        await release.promise;
        return replayResponse(`tool-${startIndex}`, { ok: true });
      }),
    );

    const results = runWithAttemptContext(attempt, async () => {
      const wrapper = new SecureToolWrapper("http://oe", "execution");
      return Promise.all([
        wrapper.executeTool("first", {}, { toolCallId: "call-a" }),
        wrapper.executeTool("second", {}, { toolCallId: "call-b" }),
      ]);
    });

    await bothStarted.promise;
    expect(starts).toEqual(["call-a", "call-b"]);
    release.resolve();
    await expect(results).resolves.toEqual([{ ok: true }, { ok: true }]);
  });

  // Both outcomes are recorded before the replay leg, so this pins
  // list-order ordinal stability and replay-without-re-execution for a
  // fully completed inverted batch. Loss with a sibling still in flight is
  // covered by the adapter-level mixed replay/live test.
  test("replay of a completed inverted parallel Tool batch keeps positions and skips re-execution", async () => {
    const liveAttempt = create(AttemptContextSchema, {
      attemptId: "attempt-live",
      fencingToken: 7n,
      ownerId: "owner",
      workflowIdentity: identity,
      heartbeatIntervalMs: 1000n,
    });
    const replayAttempt = create(AttemptContextSchema, {
      attemptId: "attempt-replay",
      fencingToken: 8n,
      replayMode: true,
      ownerId: "owner",
      workflowIdentity: identity,
      heartbeatIntervalMs: 1000n,
    });
    const resultsByCall: Record<string, unknown> = {
      "call-a": { ok: "a" },
      "call-b": { ok: "b" },
    };
    const startBodies: Record<string, unknown>[] = [];
    const outcomeBodies: Record<string, unknown>[] = [];
    const toolExecutes: string[] = [];
    let effects = 0;

    function startResponse(
      body: Record<string, unknown>,
      replay: boolean,
    ): Response {
      const semanticInput = body["semantic_input"] as Record<string, unknown>;
      const callId = String(semanticInput["tool_call_id"]);
      if (!replay) {
        return jsonResponse({
          activity_context: {
            workflow_identity: body["workflow_identity"],
            activity_id: `wait-${callId}`,
            attempt_id: body["attempt_id"],
            fencing_token: body["fencing_token"],
          },
        });
      }
      return jsonResponse({
        outcome: {
          workflow_identity: body["workflow_identity"],
          activity_id: `wait-${callId}`,
          attempt_id: body["attempt_id"],
          fencing_token: body["fencing_token"],
          outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
          result: resultsByCall[callId],
        },
      });
    }

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        const body = requestBody(init);
        if (url.endsWith("/executor/activity/start")) {
          startBodies.push(body);
          return startResponse(body, false);
        }
        if (url.endsWith("/tool/execute")) {
          toolExecutes.push(String(body["tool_name"]));
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(body);
          return jsonResponse({});
        }
        throw new Error(`unexpected external dispatch: ${url}`);
      }),
    );

    // The recorded batch is [call-a, call-b] in list order, but call-b
    // completes first while call-a is still in flight.
    const aGate = Promise.withResolvers<void>();
    const batch = runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () =>
        runWithAttemptContext(liveAttempt, async () => {
          preallocateActivityOrdinals([
            toolActivityKey("call-a"),
            toolActivityKey("call-b"),
          ]);
          const wrapper = new SecureToolWrapper("http://oe", "execution");
          const first = wrapper.executeTool(
            "slow_first",
            {},
            {
              toolCallId: "call-a",
              isLocal: true,
              localExecutor: async () => {
                effects += 1;
                await aGate.promise;
                return resultsByCall["call-a"];
              },
            },
          );
          const second = wrapper
            .executeTool(
              "fast_second",
              {},
              {
                toolCallId: "call-b",
                isLocal: true,
                localExecutor: async () => {
                  effects += 1;
                  return resultsByCall["call-b"];
                },
              },
            )
            .finally(() => aGate.resolve());
          return Promise.all([first, second]);
        }),
    );
    await expect(batch).resolves.toEqual([{ ok: "a" }, { ok: "b" }]);

    const positionsFromStarts = (bodies: Record<string, unknown>[]): string[] =>
      bodies.map(
        (body) =>
          ((body["semantic_input"] as Record<string, unknown>)[
            "tool_call_id"
          ] as string) +
          `:${(body["position"] as Record<string, unknown>)["activity_ordinal"]}`,
      );
    expect(positionsFromStarts(startBodies).sort()).toEqual([
      "call-a:1",
      "call-b:2",
    ]);
    expect(toolExecutes.sort()).toEqual(["fast_second", "slow_first"]);
    expect(outcomeBodies).toHaveLength(2);
    const effectsAfterLoss = effects;

    // Replacement attempt: OE replays both recorded outcomes. No callback
    // re-execution, no new approval, no new outcome report.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        const body = requestBody(init);
        if (url.endsWith("/executor/activity/start")) {
          startBodies.push(body);
          return startResponse(body, true);
        }
        throw new Error(`unexpected replay dispatch: ${url}`);
      }),
    );

    await runWithExecutionContext(
      { executionId: "execution", wrapper: null, oeUrl: "http://oe" },
      () =>
        runWithAttemptContext(replayAttempt, async () => {
          preallocateActivityOrdinals([
            toolActivityKey("call-a"),
            toolActivityKey("call-b"),
          ]);
          const wrapper = new SecureToolWrapper("http://oe", "execution");
          return Promise.all([
            wrapper.executeTool(
              "slow_first",
              {},
              {
                toolCallId: "call-a",
                isLocal: true,
                localExecutor: async () => {
                  effects += 1;
                  return resultsByCall["call-a"];
                },
              },
            ),
            wrapper.executeTool(
              "fast_second",
              {},
              {
                toolCallId: "call-b",
                isLocal: true,
                localExecutor: async () => {
                  effects += 1;
                  return resultsByCall["call-b"];
                },
              },
            ),
          ]);
        }),
    );

    expect(effects).toBe(effectsAfterLoss);
    expect(toolExecutes).toHaveLength(2);
    expect(outcomeBodies).toHaveLength(2);
    expect(positionsFromStarts(startBodies.slice(2)).sort()).toEqual([
      "call-a:1",
      "call-b:2",
    ]);
  });

  // OE's fence (transition.go StartActivity): a position recorded
  // started-but-unterminal by a lost attempt refuses re-admission, so the
  // unknown external-effect state can never be re-dispatched into a
  // duplicate. The lost-attempt state itself is proven reachable by the
  // adapter-level abort test.
  test("a start for a started-but-unterminal lost sibling fails closed with OUTCOME_UNKNOWN", async () => {
    const lostEffect = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
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
        throw new Error(`unexpected dispatch: ${url}`);
      }),
    );

    const error = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool(
          "slow_first",
          {},
          { toolCallId: "call-a", isLocal: true, localExecutor: lostEffect },
        )
        .catch((caught: unknown) => caught),
    );

    expect(error).toBeInstanceOf(WorkflowClientError);
    expect((error as WorkflowClientError).code).toBe(
      WorkflowErrorCode.OUTCOME_UNKNOWN,
    );
    // Fail-closed: the unknown sibling's effect never re-ran and no outcome
    // was reported for it.
    expect(lostEffect).not.toHaveBeenCalled();
  });

  test.each([
    { firstToolCallId: "call-1", secondToolCallId: undefined },
    { firstToolCallId: undefined, secondToolCallId: "call-1" },
  ])(
    "keyed and unkeyed Tool activities conflict in either order",
    async ({ firstToolCallId, secondToolCallId }) => {
      const firstStarted = Promise.withResolvers<void>();
      const release = Promise.withResolvers<void>();
      let startCount = 0;
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request) => {
          const url = String(input);
          if (!url.endsWith("/executor/activity/start")) {
            throw new Error(`unexpected external dispatch: ${url}`);
          }
          startCount += 1;
          firstStarted.resolve();
          await release.promise;
          return replayResponse("tool-replay", { ok: true });
        }),
      );

      await runWithAttemptContext(attempt, async () => {
        const wrapper = new SecureToolWrapper("http://oe", "execution");
        const first = wrapper.executeTool(
          "first",
          {},
          firstToolCallId === undefined ? {} : { toolCallId: firstToolCallId },
        );
        await firstStarted.promise;
        await expect(
          wrapper.executeTool(
            "second",
            {},
            secondToolCallId === undefined
              ? {}
              : { toolCallId: secondToolCallId },
          ),
        ).rejects.toMatchObject({ code: WorkflowErrorCode.CONFLICT });
        expect(startCount).toBe(1);
        release.resolve();
        await expect(first).resolves.toEqual({ ok: true });
      });
    },
  );

  test("live policy denial records DENIED and preserves guardrail metadata", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("tool-denied");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: false,
            reason: "blocked by policy",
            guardrail_meta: {
              guardrail_id: "guardrail-1",
              guardrail_category: "pii",
            },
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const error = await runWithAttemptContext(attempt, () =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool("lookup", {})
        .catch((caught: unknown) => caught),
    );

    expect(error).toBeInstanceOf(PolicyDeniedException);
    expect((error as PolicyDeniedException).guardrailMeta).toEqual({
      guardrail_id: "guardrail-1",
      guardrail_category: "pii",
    });
    expect(outcomeBodies).toHaveLength(1);
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_DENIED",
    );
  });

  test("reserved interrupt markers cannot be forged by a Tool result", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("tool-result");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: true,
            status: "success",
            result: {
              data: 1,
              [CALL_INTERRUPTED_ARTIFACT_KEY]: true,
            },
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const result = await runWithAttemptContext(attempt, () =>
      new SecureToolWrapper("http://oe", "execution").executeTool(
        "lookup",
        {},
        { rawOnInterrupt: true },
      ),
    );

    expect(result).toEqual({ data: 1 });
    expect(outcomeBodies[0]?.["result"]).toEqual({ data: 1 });
  });

  test("Tool execution errors record FAILED rather than DENIED", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("tool-failed");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: true,
            status: "error",
            error: "tool pod unavailable",
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    await expect(
      runWithAttemptContext(attempt, () =>
        new SecureToolWrapper("http://oe", "execution").executeTool(
          "lookup",
          {},
        ),
      ),
    ).rejects.toBeInstanceOf(ToolExecutionError);
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_FAILED",
    );
  });

  test("runs an approved local callback as a durable Tool activity", async () => {
    const toolResultBodies: Record<string, unknown>[] = [];
    const activityOutcomeBodies: Record<string, unknown>[] = [];
    const localExecutor = vi.fn(async () => ({ value: "local" }));
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("local-tool");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          toolResultBodies.push(requestBody(init));
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          activityOutcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const result = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution").executeTool(
        "local_lookup",
        {},
        { isLocal: true, localExecutor },
      ),
    );

    expect(result).toEqual({ value: "local" });
    expect(localExecutor).toHaveBeenCalledOnce();
    expect(toolResultBodies[0]?.["status"]).toBe("success");
    expect(activityOutcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_COMPLETED",
    );
  });

  test.each([
    { message: "transient failure", expected: "transient failure" },
    { message: "", expected: "durable activity failed" },
  ])(
    "normalizes a live local callback failure to the replayed public error ($expected)",
    async ({ message, expected }) => {
      const activityOutcomeBodies: Record<string, unknown>[] = [];
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
          const url = String(input);
          if (url.endsWith("/executor/activity/start")) {
            return dispatchResponse("local-tool-failed");
          }
          if (url.endsWith("/tool/execute")) {
            return jsonResponse({ proceed: true, route_to: "callback" });
          }
          if (url.endsWith("/tool/result")) return jsonResponse({});
          if (url.endsWith("/executor/activity/outcome")) {
            activityOutcomeBodies.push(requestBody(init));
            return jsonResponse({});
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      const error = await runWithDurableExecution(() =>
        new SecureToolWrapper("http://oe", "execution")
          .executeTool(
            "submit_claim_update",
            {},
            {
              isLocal: true,
              localExecutor: async () => {
                throw new Error(message);
              },
            },
          )
          .catch((caught: unknown) => caught),
      );

      expect(error).toBeInstanceOf(ToolExecutionError);
      expect(error).toHaveProperty("error", expected);
      expect(activityOutcomeBodies[0]?.["outcome_kind"]).toBe(
        "ACTIVITY_OUTCOME_KIND_FAILED",
      );
      const recordedMessage = (
        activityOutcomeBodies[0]?.["error"] as Record<string, unknown>
      )?.["message"];
      expect(typeof recordedMessage).toBe("string");

      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request) => {
          const url = String(input);
          if (url.endsWith("/executor/activity/start")) {
            return failedReplayResponse(
              "local-tool-failed",
              String(recordedMessage),
            );
          }
          throw new Error(`unexpected replay dispatch: ${url}`);
        }),
      );
      const replayed = await runWithDurableExecution(() =>
        new SecureToolWrapper("http://oe", "execution")
          .executeTool("submit_claim_update", {}, { isLocal: true })
          .catch((caught: unknown) => caught),
      );

      expect(replayed).toBeInstanceOf(ToolExecutionError);
      expect(replayed).toHaveProperty("error", expected);
      expect(String(replayed)).toBe(String(error));
    },
  );

  test("preserves a timeout discriminator and fields live and on replay", async () => {
    const activityOutcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("local-tool-timeout");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) return jsonResponse({});
        if (url.endsWith("/executor/activity/outcome")) {
          activityOutcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const timeout = new ToolCallTimeoutError("slow_tool", 600, 601.2);
    const live = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool(
          "slow_tool",
          {},
          {
            isLocal: true,
            localExecutor: async () => {
              throw timeout;
            },
          },
        )
        .catch((caught: unknown) => caught),
    );

    expect(live).toBeInstanceOf(ToolCallTimeoutError);
    expect(live).toBeInstanceOf(ToolExecutionError);
    expect(live).toMatchObject({
      error: timeout.error,
      toolName: "slow_tool",
      timeoutSeconds: 600,
      elapsedSeconds: 601.2,
    });
    const recordedMessage = (
      activityOutcomeBodies[0]?.["error"] as Record<string, unknown>
    )?.["message"];
    expect(typeof recordedMessage).toBe("string");

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return failedReplayResponse(
            "local-tool-timeout",
            String(recordedMessage),
          );
        }
        throw new Error(`unexpected replay dispatch: ${url}`);
      }),
    );
    const replayed = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool("slow_tool", {}, { isLocal: true })
        .catch((caught: unknown) => caught),
    );

    expect(replayed).toBeInstanceOf(ToolCallTimeoutError);
    expect(replayed).toMatchObject({
      error: timeout.error,
      toolName: "slow_tool",
      timeoutSeconds: 600,
      elapsedSeconds: 601.2,
    });
  });

  test("preserves structured external API failures live and on replay", async () => {
    const activityOutcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("external-api-failure");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: true,
            status: "error",
            error: "atlas API call failed: HTTP 429 RATE_LIMITED",
            duration_ms: 3,
            tool_api_error: {
              provider_type: "atlas",
              classification: "RATE_LIMITED",
              http_status: 429,
              retryable: true,
              error_code: "TOO_MANY_REQUESTS",
              reason: "Request quota exceeded",
            },
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          activityOutcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const live = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool("atlas_lookup", {})
        .catch((caught: unknown) => caught),
    );

    expect(live).toBeInstanceOf(ExternalAPICallError);
    const expectedToolApiError = {
      provider_type: "atlas",
      classification: "RATE_LIMITED",
      http_status: 429,
      retryable: true,
      error_code: "TOO_MANY_REQUESTS",
      reason: "Request quota exceeded",
    };
    expect(live).toMatchObject({
      error: "atlas API call failed: HTTP 429 RATE_LIMITED",
      tool_api_error: expectedToolApiError,
    });
    const recordedMessage = (
      activityOutcomeBodies[0]?.["error"] as Record<string, unknown>
    )?.["message"];
    expect(typeof recordedMessage).toBe("string");

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return failedReplayResponse(
            "external-api-failure",
            String(recordedMessage),
          );
        }
        throw new Error(`unexpected replay dispatch: ${url}`);
      }),
    );
    const replayed = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution")
        .executeTool("atlas_lookup", {})
        .catch((caught: unknown) => caught),
    );

    expect(replayed).toBeInstanceOf(ExternalAPICallError);
    expect(replayed).toMatchObject({
      error: "atlas API call failed: HTTP 429 RATE_LIMITED",
      tool_api_error: expectedToolApiError,
    });
  });

  test.each([
    ["BigInt", { count: 1n }],
    ["top-level undefined", undefined],
    ["an undefined object property", { value: undefined }],
    ["an undefined array element", [undefined]],
  ])(
    "rejects a non-JSON Tool result containing %s live and on replay without redispatch",
    async (_description, toolResult) => {
      const activityOutcomeBodies: Record<string, unknown>[] = [];
      const executeNative = vi
        .spyOn(
          SecureToolWrapper.prototype as unknown as {
            executeToolNative: (...args: unknown[]) => Promise<unknown>;
          },
          "executeToolNative",
        )
        .mockResolvedValue(toolResult);
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
          const url = String(input);
          if (url.endsWith("/executor/activity/start")) {
            return dispatchResponse("non-json-result");
          }
          if (url.endsWith("/executor/activity/outcome")) {
            activityOutcomeBodies.push(requestBody(init));
            return jsonResponse({});
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      const live = await runWithDurableExecution(() =>
        new SecureToolWrapper("http://oe", "execution")
          .executeTool("large_counter", {})
          .catch((caught: unknown) => caught),
      );

      expect(live).toBeInstanceOf(ToolExecutionError);
      expect(live).toHaveProperty(
        "error",
        "durable workflow values must be JSON-safe",
      );
      const recordedMessage = (
        activityOutcomeBodies[0]?.["error"] as Record<string, unknown>
      )?.["message"];
      expect(typeof recordedMessage).toBe("string");

      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request) => {
          const url = String(input);
          if (url.endsWith("/executor/activity/start")) {
            return failedReplayResponse(
              "non-json-result",
              String(recordedMessage),
            );
          }
          throw new Error(`unexpected replay dispatch: ${url}`);
        }),
      );
      const replayed = await runWithDurableExecution(() =>
        new SecureToolWrapper("http://oe", "execution")
          .executeTool("large_counter", {})
          .catch((caught: unknown) => caught),
      );

      expect(executeNative).toHaveBeenCalledOnce();
      expect(replayed).toBeInstanceOf(ToolExecutionError);
      expect(replayed).toHaveProperty(
        "error",
        (live as ToolExecutionError).error,
      );
      executeNative.mockRestore();
    },
  );

  test("converts local callback suspension into consistent terminal failures", async () => {
    const toolResultBodies: Record<string, unknown>[] = [];
    const activityOutcomeBodies: Record<string, unknown>[] = [];
    const interrupt = vi.fn();
    registerSuspendHandler(interrupt);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("local-tool-suspend");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          toolResultBodies.push(requestBody(init));
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          activityOutcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    await expect(
      runWithDurableExecution(() =>
        new SecureToolWrapper("http://oe", "execution").executeTool(
          "review_claim",
          {},
          {
            localExecutor: async () =>
              suspendPayloadToJson({
                suspend_reason: "review",
                suspend_context: {},
              }),
          },
        ),
      ),
    ).rejects.toThrow("framework adapter must own suspension");

    expect(interrupt).not.toHaveBeenCalled();
    expect(toolResultBodies[0]?.["status"]).toBe("error");
    expect(activityOutcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_FAILED",
    );
  });

  test("preserves native callback control flow for adapter settlement", async () => {
    const toolResultBodies: Record<string, unknown>[] = [];
    const activityOutcomeBodies: Record<string, unknown>[] = [];
    const nativeControlFlow = new Error("native interrupt");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("local-tool-interrupt");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/tool/result")) {
          toolResultBodies.push(requestBody(init));
          return jsonResponse({});
        }
        if (url.endsWith("/executor/activity/outcome")) {
          activityOutcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const error = await runWithDurableExecution(() =>
      new SecureToolWrapper("http://oe", "execution").executeTool(
        "native_command",
        {},
        {
          localExecutor: async () => {
            throw nativeControlFlow;
          },
          isFrameworkControlFlow: (error) => error === nativeControlFlow,
        },
      ),
    ).then(
      () => undefined,
      (caught: unknown) => caught,
    );

    expect(error).toBe(nativeControlFlow);

    expect(toolResultBodies[0]?.["status"]).toBe("interrupted");
    expect(activityOutcomeBodies).toEqual([]);
  });

  test.each([
    {
      name: "Tool suspension",
      response: {
        proceed: true,
        status: "suspend",
        result: {
          suspend_reason: "awaiting_human_review",
          suspend_context: { claim_id: "claim-1" },
        },
      },
      message: "framework adapter must own suspension",
    },
    {
      name: "authorization elicitation",
      response: {
        proceed: false,
        status: "suspend",
        elicitation: {
          elicitation_id: "elicitation-1",
          authorization_url: "https://example.com/authorize",
        },
      },
      message: "framework adapter must own authorization interrupts",
    },
  ])(
    "rejects $name before invoking framework hooks",
    async ({ response, message }) => {
      const outcomeBodies: Record<string, unknown>[] = [];
      const interrupt = vi.fn();
      registerSuspendHandler(interrupt);
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
          const url = String(input);
          if (url.endsWith("/executor/activity/start")) {
            return dispatchResponse("tool-wait");
          }
          if (url.endsWith("/tool/execute")) return jsonResponse(response);
          if (url.endsWith("/executor/activity/outcome")) {
            outcomeBodies.push(requestBody(init));
            return jsonResponse({});
          }
          throw new Error(`unexpected URL: ${url}`);
        }),
      );

      await expect(
        runWithAttemptContext(attempt, () =>
          new SecureToolWrapper("http://oe", "execution").executeTool(
            "review_claim",
            {},
          ),
        ),
      ).rejects.toThrow(message);

      expect(interrupt).not.toHaveBeenCalled();
      expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
        "ACTIVITY_OUTCOME_KIND_FAILED",
      );
    },
  );
});

describe("durable LLM effects", () => {
  const messages: Message[] = [{ role: "user", content: "Hello" }];

  test("replay restores the recorded response and preallocates Tool call ordinals", async () => {
    const startBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (!url.endsWith("/executor/activity/start")) {
          throw new Error(`unexpected external dispatch: ${url}`);
        }
        const body = requestBody(init);
        startBodies.push(body);
        if (startBodies.length === 1) {
          return replayResponse("llm-replay", {
            content: "cached",
            metadata: {},
            id: "message-1",
            name: "assistant",
            response_metadata: { finish_reason: "tool_calls" },
            additional_kwargs: { provider: "recorded" },
            usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
            tool_calls: [
              { id: "call-a", name: "first", args: {} },
              { id: "call-b", name: "second", args: {} },
            ],
          });
        }
        return replayResponse(`tool-${startBodies.length}`, { ok: true });
      }),
    );

    const { response, replayChunks } = await runWithAttemptContext(
      attempt,
      async () => {
        const llm = new SecureLLMProxy({
          oeUrl: "http://oe",
          executionId: "execution",
          modelName: "gpt-4o",
        });
        const replayChunks: LLMStreamChunk[] = [];
        for await (const chunk of llm.stream(messages)) {
          replayChunks.push(chunk);
        }
        const response = SecureLLMProxy.responseFromStreamChunks(replayChunks);
        const tools = new SecureToolWrapper("http://oe", "execution");
        await tools.executeTool("second", {}, { toolCallId: "call-b" });
        await tools.executeTool("first", {}, { toolCallId: "call-a" });
        return { response, replayChunks };
      },
    );

    expect(replayChunks).toHaveLength(1);
    expect(replayChunks[0]?.content).toBe("cached");
    expect(replayChunks[0]?.toolCalls).toHaveLength(2);
    expect(replayChunks[0]?.usage).toMatchObject({
      inputTokens: 2,
      outputTokens: 1,
      totalTokens: 3,
    });
    expect(response.content).toBe("cached");
    expect(response.id).toBe("message-1");
    expect(response.name).toBe("assistant");
    expect(response.responseMetadata).toEqual({ finish_reason: "tool_calls" });
    expect(response.additionalKwargs).toEqual({ provider: "recorded" });
    expect(response.usage).toMatchObject({
      inputTokens: 2,
      outputTokens: 1,
      totalTokens: 3,
    });
    expect(startBodies[0]?.["semantic_input"]).toMatchObject({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
    });
    expect(
      startBodies.map(
        (body) =>
          (body["position"] as Record<string, unknown>)["activity_ordinal"],
      ),
    ).toEqual(["1", "3", "2"]);
  });

  test("replayed LLM failure preserves the public error type", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return failedReplayResponse("llm-replay", "recorded LLM failure");
        }
        throw new Error(`unexpected external dispatch: ${url}`);
      }),
    );

    const error = await runWithAttemptContext(attempt, async () => {
      try {
        for await (const _chunk of new SecureLLMProxy({
          oeUrl: "http://oe",
          executionId: "execution",
        }).stream(messages)) {
          // A replayed failure yields no chunks.
        }
      } catch (caught) {
        return caught;
      }
      return null;
    });

    expect(error).toBeInstanceOf(LLMInvocationError);
    expect(error).toHaveProperty("error", "recorded LLM failure");
    expect(error).toHaveProperty(
      "cause",
      expect.any(ReplayedActivityFailedError),
    );
  });

  test("live LLM failure preserves the public error type and records FAILED", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-failed");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: true,
            status: "error",
            error: "provider unavailable",
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const consume = runWithAttemptContext(attempt, async () => {
      for await (const _chunk of new SecureLLMProxy({
        oeUrl: "http://oe",
        executionId: "execution",
      }).stream(messages)) {
        // A live failure yields no chunks.
      }
    });

    await expect(consume).rejects.toBeInstanceOf(LLMInvocationError);
    await expect(consume).rejects.toThrow("provider unavailable");
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_FAILED",
    );
  });

  test("live chunks are yielded before the folded outcome is recorded", async () => {
    const sseUrl = "http://oe/tool/stream/execution/1";
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-live");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: true,
            route_to: sseUrl,
            result: null,
            cached_result: null,
          });
        }
        if (url === sseUrl) {
          return sseResponse([
            { content: "Hel", tool_call_chunks: [] },
            { content: "lo", tool_call_chunks: [] },
            {
              done: true,
              usage: {
                input_tokens: 2,
                output_tokens: 1,
                total_tokens: 3,
              },
            },
          ]);
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    await runWithAttemptContext(attempt, async () => {
      const stream = new SecureLLMProxy({
        oeUrl: "http://oe",
        executionId: "execution",
        modelName: "gpt-4o",
      }).stream(messages);

      expect((await stream.next()).value?.content).toBe("Hel");
      expect(outcomeBodies).toHaveLength(0);
      expect((await stream.next()).value?.content).toBe("lo");
      expect(outcomeBodies).toHaveLength(0);
      while (!(await stream.next()).done) {
        // Consume the usage chunk and complete the activity.
      }
    });

    expect(outcomeBodies).toHaveLength(1);
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_COMPLETED",
    );
    expect(outcomeBodies[0]?.["result"]).toMatchObject({
      content: "Hello",
      usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
    });
  });

  test("terminal fold merges streamed Tool-call fragments", async () => {
    const sseUrl = "http://oe/tool/stream/execution/1";
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-tools");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: sseUrl });
        }
        if (url === sseUrl) {
          return sseResponse([
            {
              tool_call_chunks: [
                {
                  id: "call-1",
                  name: "lookup",
                  args: '{"q": ',
                  index: 0,
                },
              ],
            },
            { tool_call_chunks: [{ args: '"teal"}', index: 0 }] },
            { done: true },
          ]);
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    await runWithAttemptContext(attempt, async () => {
      for await (const _chunk of new SecureLLMProxy({
        oeUrl: "http://oe",
        executionId: "execution",
      }).stream(messages)) {
        // Consume the complete stream.
      }
    });

    expect(outcomeBodies[0]?.["result"]).toMatchObject({
      tool_calls: [{ id: "call-1", name: "lookup", args: { q: "teal" } }],
    });
  });

  test("abandoning a live model stream records FAILED", async () => {
    const sseUrl = "http://oe/tool/stream/execution/1";
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-abandoned");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: sseUrl });
        }
        if (url === sseUrl) {
          return sseResponse([
            { content: "first", tool_call_chunks: [] },
            { content: "second", tool_call_chunks: [] },
            { done: true },
          ]);
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    await runWithAttemptContext(attempt, async () => {
      const stream = new SecureLLMProxy({
        oeUrl: "http://oe",
        executionId: "execution",
      }).stream(messages);
      expect((await stream.next()).value?.content).toBe("first");
      await stream.return(undefined);
    });

    expect(outcomeBodies).toHaveLength(1);
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_FAILED",
    );
  });

  test("live model policy denial records DENIED and preserves metadata", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-denied");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: false,
            reason: "model blocked",
            guardrail_meta: {
              guardrail_id: "guardrail-llm",
              guardrail_category: "model",
            },
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const error = await runWithAttemptContext(attempt, async () => {
      try {
        for await (const _chunk of new SecureLLMProxy({
          oeUrl: "http://oe",
          executionId: "execution",
        }).stream(messages)) {
          // A denial yields no chunks.
        }
      } catch (caught) {
        return caught;
      }
      return null;
    });

    expect(error).toBeInstanceOf(PolicyDeniedException);
    expect((error as PolicyDeniedException).guardrailMeta).toEqual({
      guardrail_id: "guardrail-llm",
      guardrail_category: "model",
    });
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_DENIED",
    );
  });

  test("rejects require_review before invoking a framework interrupt", async () => {
    const outcomeBodies: Record<string, unknown>[] = [];
    const interrupt = vi.fn();
    registerSuspendHandler(interrupt);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          return dispatchResponse("llm-review");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({
            proceed: false,
            status: "require_review",
            reason: "review required",
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const consume = runWithAttemptContext(attempt, async () => {
      for await (const _chunk of new SecureLLMProxy({
        oeUrl: "http://oe",
        executionId: "execution",
      }).stream(messages)) {
        // The durable activity must fail before yielding review content.
      }
    });

    await expect(consume).rejects.toBeInstanceOf(LLMInvocationError);
    await expect(consume).rejects.toThrow("framework adapter must own review");
    expect(interrupt).not.toHaveBeenCalled();
    expect(outcomeBodies[0]?.["outcome_kind"]).toBe(
      "ACTIVITY_OUTCOME_KIND_FAILED",
    );
  });
});

describe("durable Memory effects", () => {
  test("live dispatch records and returns schema-decoded context", async () => {
    const startBodies: Record<string, unknown>[] = [];
    const memoryBodies: Record<string, unknown>[] = [];
    const outcomeBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          startBodies.push(requestBody(init));
          return dispatchResponse("memory-live");
        }
        if (url.endsWith("/api/v1/memory/context")) {
          memoryBodies.push(requestBody(init));
          return jsonResponse({
            formatted_context: "live memory",
            metadata: { token_count: 1, memory_counts: {}, timing: {} },
            selected_memories: [
              {
                id: "memory-1",
                content: "teal",
                source: "semantic",
                timestamp: "2026-08-26T12:00:00.000Z",
              },
            ],
          });
        }
        if (url.endsWith("/executor/activity/outcome")) {
          outcomeBodies.push(requestBody(init));
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const result = await runWithDurableExecution(() =>
      new AppBoundRuntime().buildContext({
        query: "favorite color",
        userId: "user",
        enabledSources: new Set(["semantic"]),
      }),
    );

    expect(result.formatted_context).toBe("live memory");
    expect(result.selected_memories?.[0]?.timestamp).toEqual(
      new Date("2026-08-26T12:00:00.000Z"),
    );
    expect(startBodies[0]).toMatchObject({
      activity_kind: "ACTIVITY_KIND_MEMORY",
      activity_name: "memory.build_context",
      semantic_input: {
        query: "favorite color",
        user_id: "user",
        enabled_sources: ["semantic"],
      },
    });
    expect(memoryBodies).toEqual([
      expect.objectContaining({
        query: "favorite color",
        user_id: "user",
        enabled_sources: ["semantic"],
      }),
    ]);
    expect(outcomeBodies[0]).toMatchObject({
      outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
      result: {
        formatted_context: "live memory",
        selected_memories: [
          expect.objectContaining({
            id: "memory-1",
            timestamp: "2026-08-26T12:00:00.000Z",
          }),
        ],
      },
    });
  });

  test("replay returns recorded context without calling the Memory proxy", async () => {
    const calls: FetchCall[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        calls.push({ url, body: requestBody(init) });
        if (url.endsWith("/executor/activity/start")) {
          return replayResponse("memory-replay", {
            formatted_context: "cached memory",
            metadata: { token_count: 1, memory_counts: {}, timing: {} },
            selected_memories: [
              {
                id: "memory-1",
                content: "teal",
                source: "semantic",
                timestamp: "2026-08-26T12:00:00.000Z",
              },
            ],
          });
        }
        throw new Error(`unexpected Memory dispatch: ${url}`);
      }),
    );

    const result = await runWithDurableExecution(() =>
      new AppBoundRuntime().buildContext({
        query: "favorite color",
        userId: "user",
        sessionId: "session",
        enabledSources: new Set(["semantic", "episodic"]),
        topK: 4,
        maxTokens: 1200,
      }),
    );

    expect(result.formatted_context).toBe("cached memory");
    expect(result.selected_memories?.[0]?.timestamp).toEqual(
      new Date("2026-08-26T12:00:00.000Z"),
    );
    expect(calls).toHaveLength(1);
    expect(calls[0]?.body).toMatchObject({
      activity_name: "memory.build_context",
      semantic_input: {
        query: "favorite color",
        user_id: "user",
        session_id: "session",
        visibility: null,
        metadata_filter: null,
        enabled_sources: ["episodic", "semantic"],
        top_k: 4,
        max_tokens: 1200,
      },
    });
  });

  test("short-term Memory writes remain outside the durable activity protocol", async () => {
    const memoryBodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/v1/memory/turns")) {
          memoryBodies.push(requestBody(init));
          return jsonResponse({
            id: "turn-1",
            session_id: "session",
            turn_seq: 1,
            has_embedding: true,
            acknowledged: true,
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const result = await runWithDurableExecution(() =>
      new AppBoundRuntime().recordTurn({
        role: "assistant",
        content: "teal",
        sessionId: "session",
        userId: "user",
      }),
    );

    expect(result.id).toBe("turn-1");
    expect(memoryBodies).toHaveLength(1);
    expect(memoryBodies[0]).toMatchObject({
      role: "assistant",
      content: "teal",
      session_id: "session",
      user_id: "user",
    });
  });
});
