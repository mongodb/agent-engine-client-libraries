import { create } from "@bufbuild/protobuf";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  ActivityContextSchema,
  ActivityKind,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  StepSuspensionEntrySchema,
  type ActivityCommand,
  type ActivityContext,
  type ActivityOutcome,
} from "../../src/generated/workflow/v1/activity_pb.js";
import {
  TenantScopeSchema,
  WorkflowErrorCode,
  WorkflowErrorSchema,
  WorkflowIdentitySchema,
} from "../../src/generated/workflow/v1/common_pb.js";
import { AttemptContextSchema } from "../../src/generated/workflow/v1/runtime_pb.js";
import { StateSnapshotSchema } from "../../src/generated/workflow/v1/state_pb.js";
import { getLogger } from "../../src/logger.js";
import {
  DurableMemoryState,
  type WorkflowMemoryClient,
} from "../../src/workflow/memory.js";
import type { ActivityMemoryCommand } from "../../src/generated/workflow/v1/activity_pb.js";
import {
  ActivityDispatch,
  ActivityReplay,
  DurableActivityDeniedError,
  DurableActivityInterrupted,
  ReplayedActivityFailedError,
  WorkflowClientError,
  advanceStepOrdinal,
  buildActivityCommand,
  completedOutcome,
  failedOutcome,
  finalizeCurrentStepCommand,
  finalizeCurrentStepSuspensionsCommand,
  interruptedActivities,
  recordObservedActivity,
  runSerialActivity,
  runStreamingActivity,
  runWithAttemptContext,
  runWithOperationPathResolver,
  setActivityReconstructionIds,
  unwrapActivityOutcome,
  valueToJson,
} from "../../src/workflow/index.js";

afterEach(() => {
  vi.restoreAllMocks();
});

const identity = create(WorkflowIdentitySchema, {
  tenantScope: create(TenantScopeSchema, {
    orgId: "org",
    projectId: "project",
    workspaceId: "workspace",
  }),
  sessionId: "session-1",
  executionId: "execution-1",
});

function attempt() {
  return create(AttemptContextSchema, {
    attemptId: "attempt-1",
    fencingToken: 7n,
    ownerId: "aer-1",
    workflowIdentity: identity,
  });
}

function activityContext(): ActivityContext {
  return create(ActivityContextSchema, {
    workflowIdentity: identity,
    activityId: "activity-1",
    attemptId: "attempt-1",
    fencingToken: 7n,
  });
}

class FakeClient {
  readonly commands: ActivityCommand[] = [];
  readonly outcomes: ActivityOutcome[] = [];
  reportError: WorkflowClientError | undefined;

  constructor(private readonly started: ActivityDispatch | ActivityReplay) {}

  startActivity(
    command: ActivityCommand,
  ): Promise<ActivityDispatch | ActivityReplay> {
    this.commands.push(command);
    return Promise.resolve(this.started);
  }

  reportOutcome(outcome: ActivityOutcome): Promise<void> {
    this.outcomes.push(outcome);
    if (this.reportError !== undefined) {
      return Promise.reject(this.reportError);
    }
    return Promise.resolve();
  }
}

describe("buildActivityCommand", () => {
  test("builds a root serial position from the attempt", () => {
    const command = buildActivityCommand({
      attempt: attempt(),
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 3,
      stepOrdinal: 1,
      semanticInput: { messages: [{ role: "user", content: "hi" }] },
    });
    expect(command.attemptId).toBe("attempt-1");
    expect(command.fencingToken).toBe(7n);
    expect(command.activityKind).toBe(ActivityKind.LLM);
    expect(
      command.position?.operationPath?.segments.map((segment) => [
        segment.name,
        segment.ordinal,
      ]),
    ).toEqual([["agent", 1n]]);
    expect(command.position?.activityOrdinal).toBe(3n);
    expect(command.position?.stepOrdinal).toBe(1n);
    expect(valueToJson(command.semanticInput)).toEqual({
      messages: [{ role: "user", content: "hi" }],
    });
  });

  test("requires workflow identity", () => {
    expect(() =>
      buildActivityCommand({
        attempt: create(AttemptContextSchema, {
          attemptId: "attempt-1",
          fencingToken: 1n,
        }),
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
      }),
    ).toThrow(
      expect.objectContaining({ code: WorkflowErrorCode.INVALID_ARGUMENT }),
    );
  });

  test("requires a positive activity ordinal", () => {
    expect(() =>
      buildActivityCommand({
        attempt: attempt(),
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 0,
        stepOrdinal: 1,
        semanticInput: null,
      }),
    ).toThrow(
      expect.objectContaining({ code: WorkflowErrorCode.INVALID_ARGUMENT }),
    );
  });

  test("omitted step ordinal uses the advanced request counter", () => {
    let command: ActivityCommand | undefined;
    runWithAttemptContext(attempt(), () => {
      advanceStepOrdinal(1);
      command = buildActivityCommand({
        attempt: attempt(),
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        semanticInput: null,
      });
    });
    expect(command?.position?.stepOrdinal).toBe(2n);
  });

  test("omitted operation path snapshots the current nested path", () => {
    let command: ActivityCommand | undefined;
    runWithAttemptContext(attempt(), () =>
      runWithOperationPathResolver(
        () => [
          { name: "case_investigation", occurrenceKey: "task-a" },
          { name: "policy_analysis", occurrenceKey: "task-b" },
        ],
        () => {
          command = buildActivityCommand({
            attempt: attempt(),
            kind: ActivityKind.TOOL,
            name: "lookup_policy",
            activityOrdinal: 1,
            semanticInput: { policy: "refund" },
          });
        },
      ),
    );
    expect(
      command?.position?.operationPath?.segments.map((segment) => [
        segment.name,
        segment.ordinal,
      ]),
    ).toEqual([
      ["agent", 1n],
      ["case_investigation", 1n],
      ["policy_analysis", 1n],
    ]);
  });

  test("omitted operation path survives await before admission", async () => {
    let command: ActivityCommand | undefined;
    await runWithAttemptContext(attempt(), async () =>
      runWithOperationPathResolver(
        () => [{ name: "case_investigation", occurrenceKey: "task-a" }],
        async () => {
          await Promise.resolve();
          command = buildActivityCommand({
            attempt: attempt(),
            kind: ActivityKind.TOOL,
            name: "lookup_policy",
            activityOrdinal: 1,
            semanticInput: { policy: "refund" },
          });
        },
      ),
    );
    expect(
      command?.position?.operationPath?.segments.map((segment) => [
        segment.name,
        segment.ordinal,
      ]),
    ).toEqual([
      ["agent", 1n],
      ["case_investigation", 1n],
    ]);
  });
});

describe("unwrapActivityOutcome", () => {
  test("completed returns result JSON", () => {
    const outcome = completedOutcome(activityContext(), { claim_id: "CLM-1" });
    expect(outcome.outcomeKind).toBe(ActivityOutcomeKind.COMPLETED);
    expect(unwrapActivityOutcome(outcome)).toEqual({ claim_id: "CLM-1" });
  });

  test("denied raises a stable denial", () => {
    const outcome = create(ActivityOutcomeSchema, {
      outcomeKind: ActivityOutcomeKind.DENIED,
      error: create(WorkflowErrorSchema, { message: "policy denied" }),
    });
    expect(() => unwrapActivityOutcome(outcome)).toThrow(
      DurableActivityDeniedError,
    );
    expect(() => unwrapActivityOutcome(outcome)).toThrow("policy denied");
  });

  test("failed without a code maps to outcome unknown", () => {
    const outcome = create(ActivityOutcomeSchema, {
      outcomeKind: ActivityOutcomeKind.FAILED,
      error: create(WorkflowErrorSchema, { message: "boom" }),
    });
    expect(() => unwrapActivityOutcome(outcome)).toThrow(
      expect.objectContaining({ code: WorkflowErrorCode.OUTCOME_UNKNOWN }),
    );
    expect(() => unwrapActivityOutcome(outcome)).toThrow(
      ReplayedActivityFailedError,
    );
  });

  test("suspended outcomes require a framework adapter", () => {
    const outcome = create(ActivityOutcomeSchema, {
      outcomeKind: ActivityOutcomeKind.SUSPENDED,
    });
    expect(() => unwrapActivityOutcome(outcome)).toThrow(
      expect.objectContaining({
        code: WorkflowErrorCode.INVALID_ARGUMENT,
        message: expect.stringContaining("framework adapters"),
      }),
    );
  });

  test("unspecified kind is invalid", () => {
    expect(() =>
      unwrapActivityOutcome(create(ActivityOutcomeSchema, {})),
    ).toThrow(
      expect.objectContaining({ code: WorkflowErrorCode.INVALID_ARGUMENT }),
    );
  });
});

describe("runSerialActivity", () => {
  test("does not execute when replay requires adapter-owned suspension", async () => {
    const recorded = create(ActivityOutcomeSchema, {
      outcomeKind: ActivityOutcomeKind.SUSPENDED,
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const calls: string[] = [];
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "lookup",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: { q: 1 },
        execute: () => {
          calls.push("ran");
          return { fresh: true };
        },
        attempt: attempt(),
      }),
    ).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
    });
    expect(calls).toEqual([]);
    expect(client.outcomes).toEqual([]);
  });

  test("replays a recorded outcome without executing", async () => {
    const recorded = completedOutcome(activityContext(), { cached: true });
    const client = new FakeClient(new ActivityReplay(recorded));
    const calls: string[] = [];
    const result = await runSerialActivity({
      client,
      kind: ActivityKind.TOOL,
      name: "lookup",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { q: 1 },
      execute: () => {
        calls.push("ran");
        return { fresh: true };
      },
      attempt: attempt(),
    });
    expect(result).toEqual({ cached: true });
    expect(calls).toEqual([]);
    expect(client.outcomes).toEqual([]);
  });

  test("runs the post-outcome hook for a replayed result", async () => {
    const recorded = completedOutcome(activityContext(), { cached: true });
    const client = new FakeClient(new ActivityReplay(recorded));
    const resolved: Array<[ActivityContext, unknown]> = [];

    const result = await runSerialActivity({
      client,
      kind: ActivityKind.TOOL,
      name: "lookup",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { q: 1 },
      execute: () => {
        throw new Error("replay executed the tool");
      },
      onActivityResolved: (_client, context, value) => {
        resolved.push([context, value]);
      },
      attempt: attempt(),
    });

    expect(result).toEqual({ cached: true });
    expect(resolved).toEqual([[activityContext(), { cached: true }]]);
    expect(client.outcomes).toEqual([]);
  });

  test("reconstructed tool activities synchronize through durable Memory", async () => {
    const recorded = completedOutcome(activityContext(), { stale: true });
    const client = new FakeClient(new ActivityReplay(recorded));
    const memory = new DurableMemoryState();
    const written: ActivityMemoryCommand[] = [];
    const memClient: WorkflowMemoryClient = {
      ensureMemoryWritten: async (command) => {
        written.push(command);
      },
    };

    const result = await runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["activity-1"]);
      return runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "lookup",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: { q: 1 },
        execute: () => ({ temperature: 72 }),
        onActivityResolved: (_client, context, value) =>
          memory.synchronizeTool(
            memClient,
            context,
            value,
            "user-1",
            "call-1",
            "get_temperature",
          ),
        attempt: attempt(),
      });
    });

    expect(result).toEqual({ temperature: 72 });
    expect(written).toHaveLength(1);
    const contents = (written[0]?.memoryWrites ?? []).map((write) =>
      JSON.parse(new TextDecoder().decode(write.payloadJson as Uint8Array)),
    );
    expect(contents).toMatchObject([
      {
        role: "tool",
        tool_call_id: "call-1",
        content: JSON.stringify({ temperature: 72 }),
      },
    ]);
  });

  test("re-enters only a selected reconstructed activity", async () => {
    const recorded = completedOutcome(activityContext(), {
      decision: "approve",
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const calls: ActivityContext[] = [];

    const result = await runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["activity-1"]);
      return runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "review",
        activityOrdinal: 1,
        semanticInput: { claim: "CLM-1" },
        execute: (context) => {
          calls.push(context);
          return { toolResult: "continued" };
        },
      });
    });

    expect(result).toEqual({ toolResult: "continued" });
    expect(calls).toEqual([activityContext()]);
    expect(client.outcomes).toEqual([]);
  });

  test("propagates a reconstructed callback failure without reporting another outcome", async () => {
    const recorded = completedOutcome(activityContext(), {
      decision: "approve",
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const failure = new Error("callback failed");

    await runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["activity-1"]);
      await expect(
        runSerialActivity({
          client,
          kind: ActivityKind.TOOL,
          name: "review",
          activityOrdinal: 1,
          semanticInput: { claim: "CLM-1" },
          execute: () => {
            throw failure;
          },
        }),
      ).rejects.toBe(failure);
    });

    expect(client.outcomes).toEqual([]);
  });

  test("propagates a reconstructed callback denial without reporting another outcome", async () => {
    const recorded = completedOutcome(activityContext(), {
      decision: "approve",
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const denial = new DurableActivityDeniedError("callback denied");

    await runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["activity-1"]);
      await expect(
        runSerialActivity({
          client,
          kind: ActivityKind.TOOL,
          name: "review",
          activityOrdinal: 1,
          semanticInput: { claim: "CLM-1" },
          execute: () => {
            throw denial;
          },
        }),
      ).rejects.toBe(denial);
    });

    expect(client.outcomes).toEqual([]);
  });

  test("records control flow and rejects a second reconstructed interrupt", async () => {
    const recorded = completedOutcome(activityContext(), {
      decision: "approve",
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const first = new Error("first interrupt");
    const second = new Error("second interrupt");

    await runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["activity-1"]);
      await expect(
        runSerialActivity({
          client,
          kind: ActivityKind.TOOL,
          name: "review",
          activityOrdinal: 1,
          semanticInput: { claim: "CLM-1" },
          execute: () => {
            throw new DurableActivityInterrupted(first);
          },
        }),
      ).rejects.toBe(first);
      await expect(
        runSerialActivity({
          client,
          kind: ActivityKind.TOOL,
          name: "review",
          activityOrdinal: 1,
          semanticInput: { claim: "CLM-1" },
          execute: () => {
            throw new DurableActivityInterrupted(second);
          },
        }),
      ).rejects.toThrow("cannot raise another native interrupt");
      expect(interruptedActivities(1)).toHaveLength(1);
    });

    expect(client.outcomes).toEqual([]);
  });

  test("response-lost retry reuses the original position", async () => {
    const recorded = completedOutcome(activityContext(), { cached: true });
    const first = new FakeClient(new ActivityDispatch(activityContext()));
    await runSerialActivity({
      client: first,
      kind: ActivityKind.TOOL,
      name: "lookup",
      activityOrdinal: 4,
      stepOrdinal: 2,
      semanticInput: { q: 1 },
      execute: () => ({ fresh: true }),
      attempt: attempt(),
    });
    const retry = new FakeClient(new ActivityReplay(recorded));
    await runSerialActivity({
      client: retry,
      kind: ActivityKind.TOOL,
      name: "lookup",
      activityOrdinal: 4,
      stepOrdinal: 2,
      semanticInput: { q: 1 },
      execute: () => ({ fresh: true }),
      attempt: attempt(),
    });
    expect(first.commands[0]?.position).toEqual(retry.commands[0]?.position);
    expect(retry.outcomes).toEqual([]);
  });

  test("dispatch executes and reports completion", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const result = await runSerialActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 2,
      stepOrdinal: 1,
      semanticInput: { prompt: "hi" },
      execute: () => ({ answer: 42 }),
      attempt: attempt(),
    });
    expect(result).toEqual({ answer: 42 });
    const [outcome] = client.outcomes;
    expect(outcome?.outcomeKind).toBe(ActivityOutcomeKind.COMPLETED);
    expect(
      outcome === undefined ? undefined : unwrapActivityOutcome(outcome),
    ).toEqual({
      answer: 42,
    });
  });

  test("runs the post-outcome hook after reporting a fresh result", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const order: string[] = [];
    const originalReportOutcome = client.reportOutcome.bind(client);
    client.reportOutcome = async (outcome) => {
      order.push("outcome");
      await originalReportOutcome(outcome);
    };

    await runSerialActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 2,
      stepOrdinal: 1,
      semanticInput: { prompt: "hi" },
      execute: () => ({ answer: 42 }),
      onActivityResolved: () => {
        order.push("resolved");
      },
      attempt: attempt(),
    });

    expect(order).toEqual(["outcome", "resolved"]);
  });

  test("overlapping work fails before a second admission", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    let releaseFirst!: () => void;
    const firstRunning = Promise.withResolvers<void>();
    const first = runSerialActivity({
      client,
      kind: ActivityKind.TOOL,
      name: "tool-a",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: null,
      execute: async () => {
        firstRunning.resolve();
        await new Promise<void>((resolve) => {
          releaseFirst = resolve;
        });
        return "done";
      },
      attempt: attempt(),
    });
    await firstRunning.promise;
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "tool-b",
        activityOrdinal: 2,
        stepOrdinal: 1,
        semanticInput: null,
        execute: () => "second",
        attempt: attempt(),
      }),
    ).rejects.toMatchObject({
      code: WorkflowErrorCode.CONFLICT,
      message: "this attempt already has an in-flight activity",
    });
    releaseFirst();
    await expect(first).resolves.toBe("done");
    expect(client.commands).toHaveLength(1);
  });

  test("nonexclusive activities may overlap", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const bothRunning = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    let running = 0;
    const start = (name: string, activityOrdinal: number) =>
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name,
        activityOrdinal,
        stepOrdinal: 1,
        semanticInput: null,
        exclusive: false,
        execute: async () => {
          running += 1;
          if (running === 2) bothRunning.resolve();
          await release.promise;
          return name;
        },
        attempt: attempt(),
      });

    const first = start("tool:a", 1);
    const second = start("tool:b", 2);
    await bothRunning.promise;
    expect(client.commands).toHaveLength(2);
    release.resolve();
    await expect(Promise.all([first, second])).resolves.toEqual([
      "tool:a",
      "tool:b",
    ]);
  });

  test.each([
    { firstExclusive: false, secondExclusive: true },
    { firstExclusive: true, secondExclusive: false },
  ])(
    "exclusive and nonexclusive activities conflict in either order",
    async ({ firstExclusive, secondExclusive }) => {
      const client = new FakeClient(new ActivityDispatch(activityContext()));
      const firstRunning = Promise.withResolvers<void>();
      const release = Promise.withResolvers<void>();
      const first = runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "first",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
        exclusive: firstExclusive,
        execute: async () => {
          firstRunning.resolve();
          await release.promise;
          return "first";
        },
        attempt: attempt(),
      });
      await firstRunning.promise;

      await expect(
        runSerialActivity({
          client,
          kind: ActivityKind.TOOL,
          name: "second",
          activityOrdinal: 2,
          stepOrdinal: 1,
          semanticInput: null,
          exclusive: secondExclusive,
          execute: () => "second",
          attempt: attempt(),
        }),
      ).rejects.toMatchObject({ code: WorkflowErrorCode.CONFLICT });
      expect(client.commands).toHaveLength(1);
      release.resolve();
      await expect(first).resolves.toBe("first");
    },
  );

  test("overlapping same-name child scopes fail with conflict before path allocation", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    let releaseFirst!: () => void;
    const firstRunning = Promise.withResolvers<void>();
    await runWithAttemptContext(attempt(), async () => {
      const first = runWithOperationPathResolver(
        () => [{ name: "case_investigation", occurrenceKey: "task-a" }],
        () =>
          runSerialActivity({
            client,
            kind: ActivityKind.TOOL,
            name: "lookup",
            activityOrdinal: 1,
            semanticInput: { q: 1 },
            execute: async () => {
              firstRunning.resolve();
              await new Promise<void>((resolve) => {
                releaseFirst = resolve;
              });
              return "done";
            },
            attempt: attempt(),
          }),
      );
      await firstRunning.promise;
      await expect(
        runWithOperationPathResolver(
          () => [{ name: "case_investigation", occurrenceKey: "task-b" }],
          () =>
            runSerialActivity({
              client,
              kind: ActivityKind.TOOL,
              name: "lookup",
              activityOrdinal: 2,
              semanticInput: { q: 2 },
              execute: () => "second",
              attempt: attempt(),
            }),
        ),
      ).rejects.toMatchObject({ code: WorkflowErrorCode.CONFLICT });
      releaseFirst();
      await expect(first).resolves.toBe("done");
    });
    expect(client.commands).toHaveLength(1);
  });

  test("worker failure reports failed and reraises", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
        execute: () => {
          throw new Error("worker blew up");
        },
        attempt: attempt(),
      }),
    ).rejects.toThrow("worker blew up");
    expect(client.outcomes[0]?.outcomeKind).toBe(ActivityOutcomeKind.FAILED);
  });

  test("stale-fence outcome report logs the original worker denial", async () => {
    const activityLogger = getLogger(
      "agent_engine_runner_shared.workflow.activity",
    );
    const errorSpy = vi
      .spyOn(Object.getPrototypeOf(activityLogger), "error")
      .mockImplementation(() => activityLogger);
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    client.reportError = new WorkflowClientError(
      WorkflowErrorCode.STALE_FENCE,
      "attempt no longer owns the execution",
    );
    const denial = new DurableActivityDeniedError("policy denied");
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
        execute: () => {
          throw denial;
        },
        attempt: attempt(),
      }),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.STALE_FENCE });
    expect(errorSpy).toHaveBeenCalledWith(
      "Failed to report denied activity outcome",
      denial,
      client.reportError,
    );
  });

  test("live denial is recorded as a denied outcome", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
        execute: () => {
          throw new DurableActivityDeniedError("policy denied");
        },
        attempt: attempt(),
      }),
    ).rejects.toBeInstanceOf(DurableActivityDeniedError);
    const [denied] = client.outcomes;
    expect(denied?.outcomeKind).toBe(ActivityOutcomeKind.DENIED);
    expect(denied?.error?.code).toBe(WorkflowErrorCode.UNAUTHORIZED);
    expect(() => {
      if (denied === undefined) throw new Error("missing outcome");
      unwrapActivityOutcome(denied);
    }).toThrow("policy denied");
  });

  test("root commit includes observed activity positions", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    await runWithAttemptContext(attempt(), async () => {
      await runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "lookup",
        activityOrdinal: 1,
        semanticInput: { q: 1 },
        execute: () => ({ ok: true }),
        attempt: attempt(),
      });
      const command = finalizeCurrentStepCommand(
        attempt(),
        create(StateSnapshotSchema, {}),
      );
      expect(command.stepOrdinal).toBe(1n);
      expect(command.observedActivityPositions).toHaveLength(1);
      expect(command.observedActivityPositions[0]?.activityOrdinal).toBe(1n);
    });
  });

  test("requires an attempt context", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "tool",
        activityOrdinal: 1,
        stepOrdinal: 1,
        semanticInput: null,
        execute: () => null,
      }),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.INVALID_ARGUMENT });
  });
});

describe("suspension frontier finalization", () => {
  test("includes every current-step observation and all supplied waits without advancing the step", () => {
    const current = attempt();
    runWithAttemptContext(current, () => {
      const makePosition = (ordinal: number) => {
        const position = buildActivityCommand({
          attempt: current,
          kind: ActivityKind.TOOL,
          name: "review",
          activityOrdinal: ordinal,
          semanticInput: null,
        }).position;
        if (position === undefined)
          throw new Error("missing activity position");
        return position;
      };
      recordObservedActivity(makePosition(1));
      advanceStepOrdinal(1);
      const completed = makePosition(1);
      const firstWait = makePosition(2);
      const secondWait = makePosition(3);
      for (const position of [secondWait, completed, firstWait, completed]) {
        recordObservedActivity(position);
      }
      const waits = [firstWait, secondWait].map((position) =>
        create(StepSuspensionEntrySchema, {
          position,
          activityKind: ActivityKind.TOOL,
          activityName: "review",
          suspension: { reason: "approval", context: { question: "approve?" } },
        }),
      );
      const command = finalizeCurrentStepSuspensionsCommand(current, waits);
      expect(command).toMatchObject({
        workflowIdentity: identity,
        attemptId: "attempt-1",
        fencingToken: 7n,
        stepOrdinal: 2n,
      });
      expect(command.observedActivityPositions).toEqual([
        completed,
        firstWait,
        secondWait,
      ]);
      expect(command.suspensions).toEqual(waits);
      expect(command.state).toBeUndefined();
      expect(finalizeCurrentStepSuspensionsCommand(current, waits)).toEqual(
        command,
      );
    });
  });

  test("rejects empty, missing-position, and cross-step frontiers", () => {
    const current = attempt();
    runWithAttemptContext(current, () => {
      expect(() => finalizeCurrentStepSuspensionsCommand(current, [])).toThrow(
        "at least one entry",
      );
      for (const entry of [
        create(StepSuspensionEntrySchema),
        create(StepSuspensionEntrySchema, { position: { stepOrdinal: 2n } }),
      ]) {
        expect(() =>
          finalizeCurrentStepSuspensionsCommand(current, [entry]),
        ).toThrow("another workflow step");
      }
    });
  });
});

describe("failedOutcome", () => {
  test("preserves a coded failure", () => {
    const outcome = failedOutcome(
      activityContext(),
      "boom",
      WorkflowErrorCode.CONFLICT,
    );
    expect(() => unwrapActivityOutcome(outcome)).toThrow(
      expect.objectContaining({
        code: WorkflowErrorCode.CONFLICT,
        message: "boom",
      }),
    );
  });
});

describe("runStreamingActivity", () => {
  test("yields fresh items before publishing the folded outcome", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "hel";
        yield "lo";
      },
      replay: async function* (result) {
        yield String(result);
      },
      fold: (items) => items.join(""),
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toEqual({
      value: "hel",
      done: false,
    });
    expect(client.outcomes).toHaveLength(0);
    await expect(stream.next()).resolves.toEqual({
      value: "lo",
      done: false,
    });
    expect(client.outcomes).toHaveLength(0);
    await expect(stream.next()).resolves.toEqual({
      value: undefined,
      done: true,
    });
    expect(client.outcomes).toHaveLength(1);
    const [outcome] = client.outcomes;
    if (outcome === undefined) throw new Error("missing outcome");
    expect(unwrapActivityOutcome(outcome)).toBe("hello");
  });

  test("holds exclusive admission while a fresh stream is paused", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "first";
        yield "second";
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toEqual({
      value: "first",
      done: false,
    });
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "lookup",
        activityOrdinal: 2,
        stepOrdinal: 1,
        semanticInput: {},
        execute: () => "unexpected",
        attempt: attempt(),
      }),
    ).rejects.toMatchObject({
      code: WorkflowErrorCode.CONFLICT,
      message: "this attempt already has an in-flight activity",
    });

    await stream.return(undefined);
    expect(client.commands).toHaveLength(1);
  });

  test("allows an identified stream to overlap an identified sibling", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "first";
        yield "second";
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
      exclusive: false,
    });

    await expect(stream.next()).resolves.toEqual({
      value: "first",
      done: false,
    });
    await expect(
      runSerialActivity({
        client,
        kind: ActivityKind.TOOL,
        name: "lookup",
        activityOrdinal: 2,
        stepOrdinal: 1,
        semanticInput: {},
        execute: () => "found",
        attempt: attempt(),
        exclusive: false,
      }),
    ).resolves.toBe("found");

    await stream.return(undefined);
    expect(client.commands).toHaveLength(2);
  });

  test("replays recorded items without executing the worker", async () => {
    const execute = vi.fn(async function* () {
      yield "live";
    });
    const client = new FakeClient(
      new ActivityReplay(completedOutcome(activityContext(), ["recorded"])),
    );
    const items: string[] = [];
    for await (const item of runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute,
      replay: async function* (result) {
        for (const item of result as string[]) yield item;
      },
      fold: (values) => values,
      attempt: attempt(),
    })) {
      items.push(item);
    }

    expect(items).toEqual(["recorded"]);
    expect(execute).not.toHaveBeenCalled();
    expect(client.outcomes).toHaveLength(0);
  });

  test("reports a denied outcome for a live policy denial", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield await Promise.reject(
          new DurableActivityDeniedError("model blocked"),
        );
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
    });

    await expect(stream.next()).rejects.toThrow("model blocked");
    expect(client.outcomes.map((outcome) => outcome.outcomeKind)).toEqual([
      ActivityOutcomeKind.DENIED,
    ]);
  });

  test("terminal-fails an abandoned fresh stream", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "first";
        yield "second";
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toEqual({
      value: "first",
      done: false,
    });
    await stream.return(undefined);
    expect(client.outcomes).toHaveLength(1);
    expect(client.outcomes[0]?.outcomeKind).toBe(ActivityOutcomeKind.FAILED);
    expect(client.outcomes[0]?.error?.message).toContain("stream abandoned");
  });

  test("rejects stream closure when abandonment cannot be reported", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    client.reportError = new WorkflowClientError(
      WorkflowErrorCode.STALE_FENCE,
      "attempt no longer owns the execution",
    );
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "first";
        yield "second";
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toMatchObject({ value: "first" });
    await expect(stream.return(undefined)).rejects.toMatchObject({
      code: WorkflowErrorCode.STALE_FENCE,
    });
    expect(client.outcomes).toHaveLength(1);
    expect(client.outcomes[0]?.outcomeKind).toBe(ActivityOutcomeKind.FAILED);
  });

  test("terminal-fails a fold error", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "complete";
      },
      replay: () => [],
      fold: () => {
        throw new Error("cannot fold stream");
      },
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toEqual({
      value: "complete",
      done: false,
    });
    await expect(stream.next()).rejects.toThrow("cannot fold stream");
    expect(client.outcomes.map((outcome) => outcome.outcomeKind)).toEqual([
      ActivityOutcomeKind.FAILED,
    ]);
  });

  test("terminal-fails an ordinary streaming worker error", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield await Promise.reject(new Error("provider unavailable"));
      },
      replay: () => [],
      fold: (items) => items,
      attempt: attempt(),
    });

    await expect(stream.next()).rejects.toThrow("provider unavailable");
    expect(client.outcomes.map((outcome) => outcome.outcomeKind)).toEqual([
      ActivityOutcomeKind.FAILED,
    ]);
  });

  test("terminal-fails a non-JSON folded result", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "complete";
      },
      replay: () => [],
      fold: () => 1n,
      attempt: attempt(),
    });

    await expect(stream.next()).resolves.toEqual({
      value: "complete",
      done: false,
    });
    await expect(stream.next()).rejects.toThrow(/BigInt/);
    expect(client.outcomes.map((outcome) => outcome.outcomeKind)).toEqual([
      ActivityOutcomeKind.FAILED,
    ]);
  });

  test("runs the post-outcome hook after reporting a fresh streamed outcome", async () => {
    const client = new FakeClient(new ActivityDispatch(activityContext()));
    const memory = new DurableMemoryState("user-hi");
    const written: ActivityMemoryCommand[] = [];
    const memClient: WorkflowMemoryClient = {
      ensureMemoryWritten: async (command) => {
        written.push(command);
      },
    };
    const order: string[] = [];
    const origReport = client.reportOutcome.bind(client);
    client.reportOutcome = async (outcome) => {
      order.push("outcome");
      return origReport(outcome);
    };

    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "assist";
        yield "ant-lo";
      },
      replay: async function* (result) {
        yield String(result);
      },
      fold: (items) => ({ content: items.join("") }),
      onActivityResolved: (_client, context, result) => {
        order.push("resolved");
        return memory.synchronizeLlm(memClient, context, result, "user-1");
      },
      attempt: attempt(),
    });

    const seen: unknown[] = [];
    for await (const item of stream) seen.push(item);
    expect(seen).toEqual(["assist", "ant-lo"]);
    expect(order).toEqual(["outcome", "resolved"]);
    expect(written).toHaveLength(1);
    const contents = (written[0]?.memoryWrites ?? []).map((write) =>
      JSON.parse(new TextDecoder().decode(write.payloadJson as Uint8Array)),
    );
    expect(contents).toMatchObject([
      { role: "user", content: "user-hi" },
      { role: "assistant", content: "assistant-lo" },
    ]);
  });

  test("replayed streams synchronize the recorded result through durable Memory", async () => {
    const recorded = completedOutcome(activityContext(), {
      content: "recorded",
    });
    const client = new FakeClient(new ActivityReplay(recorded));
    const memory = new DurableMemoryState("user-hi");
    const written: ActivityMemoryCommand[] = [];
    const memClient: WorkflowMemoryClient = {
      ensureMemoryWritten: async (command) => {
        written.push(command);
      },
    };

    const stream = runStreamingActivity({
      client,
      kind: ActivityKind.LLM,
      name: "gpt-test",
      activityOrdinal: 1,
      stepOrdinal: 1,
      semanticInput: { messages: [] },
      execute: async function* () {
        yield "unreachable";
        throw new Error("replay executed the stream");
      },
      replay: async function* () {
        yield "recorded";
      },
      fold: (items) => ({ content: items.join("") }),
      onActivityResolved: (_client, context, result) =>
        memory.synchronizeLlm(memClient, context, result, "user-1"),
      attempt: attempt(),
    });

    const seen: unknown[] = [];
    for await (const item of stream) seen.push(item);
    expect(seen).toEqual(["recorded"]);
    expect(written).toHaveLength(1);
    const contents = (written[0]?.memoryWrites ?? []).map((write) =>
      JSON.parse(new TextDecoder().decode(write.payloadJson as Uint8Array)),
    );
    expect(contents).toMatchObject([
      { role: "user", content: "user-hi" },
      { role: "assistant", content: "recorded" },
    ]);
  });
});
