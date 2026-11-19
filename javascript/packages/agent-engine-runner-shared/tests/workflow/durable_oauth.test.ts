import { create, fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  ActivityPositionSchema,
  WorkflowErrorCode,
} from "../../src/generated/workflow/v1/common_pb.js";
import { registerSuspendHandler, resetHooks } from "../../src/hooks.js";
import {
  OAUTH_AUTHORIZATION_ACTIVITY_NAME,
  SecureToolWrapper,
  ToolExecutionError,
  ToolPlatformError,
} from "../../src/secure_wrapper.js";
import {
  ActivityContextSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  interruptedActivities,
  markInterruptedActivitiesAnswered,
  preallocateActivityOrdinals,
  runWithAttemptContext,
  toolActivityKey,
  WorkflowClientError,
} from "../../src/workflow/index.js";
import type { DurableMemoryState } from "../../src/workflow/memory.js";

type Body = Record<string, unknown>;

const ELICITATION = {
  elicitation_id: "elicitation-1",
  authorization_url: "https://example.com/authorize",
};
const RECORDED_ELICITATION = { ...ELICITATION, message: "", created: false };
const CHALLENGE = {
  proceed: false,
  status: "suspend",
  elicitation: ELICITATION,
  latest_step_number: 5,
};

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
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
}

function requestBody(init?: RequestInit): Body {
  if (typeof init?.body !== "string") return {};
  return JSON.parse(init.body) as Body;
}

function resourceResult(value: unknown): Body {
  return { proceed: true, status: "success", result: value };
}

function segmentNames(body: Body): string[] {
  const position = body["position"] as {
    operation_path?: { segments?: { name: string }[] };
  };
  return (position.operation_path?.segments ?? []).map((s) => s.name);
}

class Paused extends Error {
  constructor(readonly value: unknown) {
    super("paused");
  }
}

class History {
  readonly recorded = new Map<string, unknown>();
  readonly outcomeKinds = new Map<string, string>();
  readonly starts: Body[] = [];
  readonly toolBodies: Body[] = [];
  private readonly dispatched = new Map<string, string>();

  constructor(private readonly respond: (toolBody: Body) => unknown) {}

  install(): void {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        const body = requestBody(init);
        if (url.endsWith("/executor/activity/start")) return this.start(body);
        if (url.endsWith("/executor/activity/outcome")) {
          const key = this.dispatched.get(String(body["activity_id"]));
          if (key === undefined)
            throw new Error("outcome for unknown activity");
          this.recorded.set(key, body["result"] ?? null);
          this.outcomeKinds.set(key, String(body["outcome_kind"]));
          return jsonResponse({});
        }
        if (url.endsWith("/tool/execute")) {
          this.toolBodies.push(body);
          return jsonResponse(this.respond(body));
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );
  }

  answer(wait: Body, value: unknown): void {
    this.recorded.set(JSON.stringify(wait["position"]), value);
  }

  waits(): Body[] {
    return this.starts.filter((body) =>
      segmentNames(body).at(-1)?.startsWith("oauth:authorization"),
    );
  }

  recordedAt(start: Body): unknown {
    return this.recorded.get(JSON.stringify(start["position"]));
  }

  private start(body: Body): Response {
    this.starts.push(body);
    const key = JSON.stringify(body["position"]);
    const activityId = `activity-${this.starts.length}`;
    if (this.recorded.has(key)) {
      const outcome = create(ActivityOutcomeSchema, {
        workflowIdentity: identity,
        activityId,
        attemptId: attempt.attemptId,
        fencingToken: attempt.fencingToken,
        outcomeKind: ActivityOutcomeKind.COMPLETED,
        result: fromJson(ValueSchema, this.recorded.get(key) as never),
      });
      return jsonResponse({
        outcome: toJson(ActivityOutcomeSchema, outcome, {
          useProtoFieldName: true,
        }),
      });
    }
    this.dispatched.set(activityId, key);
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
}

function fakeMemory() {
  return {
    acknowledge: vi.fn(async () => {}),
    synchronizeTool: vi.fn(async () => {}),
  };
}

interface AttemptOptions {
  memory?: ReturnType<typeof fakeMemory>;
  answered?: readonly Body[];
  calls?: readonly string[];
}

function runAttempt<T>(
  options: AttemptOptions,
  fn: (call: (toolCallId?: string) => Promise<unknown>) => Promise<T>,
): Promise<T> {
  return runWithAttemptContext(attempt, () => {
    preallocateActivityOrdinals(
      (options.calls ?? ["call-1"]).map(toolActivityKey),
    );
    markInterruptedActivitiesAnswered(
      (options.answered ?? []).map((wait) =>
        fromJson(ActivityPositionSchema, wait["position"] as never),
      ),
    );
    const wrapper = new SecureToolWrapper("http://oe", "execution");
    if (options.memory !== undefined) {
      wrapper.durableMemory = options.memory as unknown as DurableMemoryState;
    }
    return fn((toolCallId = "call-1") =>
      wrapper.executeTool(
        "list_repositories",
        { owner: toolCallId },
        {
          provider: "github",
          toolCallId,
          isFrameworkControlFlow: (error) => error instanceof Paused,
        },
      ),
    );
  });
}

async function pauseForConsent(history: History): Promise<Body> {
  await expect(runAttempt({}, (call) => call())).rejects.toBeInstanceOf(Paused);
  const [wait] = history.waits();
  if (wait === undefined) throw new Error("no consent wait started");
  return wait;
}

afterEach(() => {
  vi.unstubAllGlobals();
  resetHooks();
});

describe("durable OAuth consent", () => {
  test("records the challenge, then pauses on its own wait activity", async () => {
    const history = new History(() => CHALLENGE);
    history.install();
    registerSuspendHandler((value) => {
      throw new Paused(value);
    });
    const memory = fakeMemory();

    const { paused, interrupted } = await runAttempt(
      { memory },
      async (call) => {
        const paused = await call().catch((error: unknown) => error);
        const wait = history.waits()[0] as Body;
        const step = (wait["position"] as Body)["step_ordinal"] ?? 0;
        return { paused, interrupted: interruptedActivities(Number(step)) };
      },
    );

    const [toolStart, wait] = history.starts as [Body, Body];
    expect(history.recordedAt(toolStart)).toEqual({
      kind: "authorization_required",
      elicitation: RECORDED_ELICITATION,
      activity_id: "activity-1",
      step: 5,
    });
    const value = {
      suspend_reason: "authorization_required",
      suspend_context: RECORDED_ELICITATION,
      oauth_authorization_wait: "activity-1",
    };
    expect((paused as Paused).value).toEqual(value);
    expect(segmentNames(wait).at(-1)).toBe("oauth:authorization");
    expect(wait["semantic_input"]).toEqual({ value });
    expect(interrupted.map((item) => item.command.activityName)).toEqual([
      OAUTH_AUTHORIZATION_ACTIVITY_NAME,
    ]);
    expect(memory.acknowledge).toHaveBeenCalledTimes(1);
    expect(memory.synchronizeTool).not.toHaveBeenCalled();
  });

  test("after the answer, runs the resource once and records its result", async () => {
    const resource = { kind: "authorization_required", repos: ["teal"] };
    const history = new History(() =>
      history.toolBodies.length === 1 ? CHALLENGE : resourceResult(resource),
    );
    history.install();
    registerSuspendHandler((value) => {
      throw new Paused(value);
    });
    const wait = await pauseForConsent(history);
    history.answer(wait, { approved: true });
    const memory = fakeMemory();

    const result = await runAttempt({ memory, answered: [wait] }, (call) =>
      call(),
    );

    expect(result).toEqual(resource);
    expect(history.toolBodies.map((body) => body["step_number"])).toEqual([
      1, 6,
    ]);
    const resourceStart = history.starts.at(-1) as Body;
    expect(segmentNames(resourceStart).at(-1)).toBe("oauth:resource");
    expect(history.recordedAt(resourceStart)).toEqual({
      kind: "result",
      result: resource,
    });
    expect(memory.synchronizeTool).toHaveBeenCalledTimes(1);
    expect(memory.synchronizeTool.mock.calls[0]).toEqual([
      expect.anything(),
      expect.anything(),
      resource,
      null,
      "call-1",
      "list_repositories",
    ]);

    await expect(
      runAttempt({ answered: [wait] }, (call) => call()),
    ).resolves.toEqual(resource);
    expect(history.toolBodies).toHaveLength(2);
  });

  test("fails the call when consent is still required after the answer", async () => {
    const history = new History(() => CHALLENGE);
    history.install();
    registerSuspendHandler((value) => {
      throw new Paused(value);
    });
    const wait = await pauseForConsent(history);
    history.answer(wait, { approved: true });

    const error = await runAttempt({ answered: [wait] }, (call) =>
      call().catch((caught: unknown) => caught),
    );

    expect(error).toBeInstanceOf(ToolExecutionError);
    expect(String((error as ToolExecutionError).message)).toContain(
      "still required",
    );
    const resourceStart = history.starts.at(-1) as Body;
    expect(
      history.outcomeKinds.get(JSON.stringify(resourceStart["position"])),
    ).toBe("ACTIVITY_OUTCOME_KIND_FAILED");
  });

  test("fails the turn when memory cannot acknowledge a replayed consent wait", async () => {
    const history = new History(() => CHALLENGE);
    history.install();
    registerSuspendHandler((value) => {
      throw new Paused(value);
    });
    const wait = await pauseForConsent(history);
    history.answer(wait, { approved: true });
    const memory = fakeMemory();
    memory.acknowledge.mockImplementation(async () => {
      const latest = history.starts.at(-1) as Body;
      if (segmentNames(latest).at(-1) === "oauth:authorization") {
        throw new WorkflowClientError(
          WorkflowErrorCode.STALE_FENCE,
          "fence moved",
        );
      }
    });

    const error = await runAttempt({ memory, answered: [wait] }, (call) =>
      call().catch((caught: unknown) => caught),
    );

    expect(error).toBeInstanceOf(ToolPlatformError);
    expect(history.toolBodies).toHaveLength(1);
  });

  test("a wait answered in an earlier attempt pauses once until this attempt settles it", async () => {
    const history = new History(() => CHALLENGE);
    history.install();
    const interrupt = vi.fn((value: unknown) => {
      throw new Paused(value);
    });
    registerSuspendHandler(interrupt);
    const wait = await pauseForConsent(history);
    history.answer(wait, { approved: true });

    await expect(runAttempt({}, (call) => call())).rejects.toBeInstanceOf(
      Paused,
    );

    expect(interrupt).toHaveBeenCalledTimes(2);
    expect(history.toolBodies).toHaveLength(1);
  });

  test("parallel calls keep their own waits and results across reordered attempts", async () => {
    const calls = ["call-a", "call-b"];
    const history = new History((body) => {
      const owner = (body["arguments"] as Body)["owner"];
      const resumed = history.toolBodies.filter(
        (seen) => (seen["arguments"] as Body)["owner"] === owner,
      );
      return resumed.length === 1
        ? CHALLENGE
        : resourceResult({ owner, repos: [`${String(owner)}-repo`] });
    });
    history.install();
    registerSuspendHandler((value) => {
      throw new Paused(value);
    });

    const paused = await runAttempt({ calls }, (call) =>
      Promise.allSettled([call("call-a"), call("call-b")]),
    );
    expect(paused.map((settled) => settled.status)).toEqual([
      "rejected",
      "rejected",
    ]);
    const waits = history.waits();
    expect(waits).toHaveLength(2);
    expect(
      new Set(waits.map((wait) => JSON.stringify(wait["position"]))).size,
    ).toBe(2);
    expect(
      waits.map(
        (wait) =>
          ((wait["semantic_input"] as Body)["value"] as Body)[
            "oauth_authorization_wait"
          ],
      ),
    ).toEqual(["activity-1", "activity-2"]);
    for (const wait of waits) history.answer(wait, { approved: true });

    const live = await runAttempt({ calls, answered: waits }, async (call) => {
      const b = await call("call-b");
      const a = await call("call-a");
      return { a, b };
    });
    const replayed = await runAttempt(
      { calls, answered: waits },
      async (call) => {
        const a = await call("call-a");
        const b = await call("call-b");
        return { a, b };
      },
    );

    const expected = {
      a: { owner: "call-a", repos: ["call-a-repo"] },
      b: { owner: "call-b", repos: ["call-b-repo"] },
    };
    expect(live).toEqual(expected);
    expect(replayed).toEqual(expected);
    expect(history.toolBodies).toHaveLength(4);
  });
});
