import { create, toJson } from "@bufbuild/protobuf";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  ActivityCommandSchema,
  ActivityKind,
  ActivityMemoryCommandSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
} from "../../src/generated/workflow/v1/activity_pb.js";
import {
  TenantScopeSchema,
  WorkflowErrorCode,
  WorkflowIdentitySchema,
} from "../../src/generated/workflow/v1/common_pb.js";
import {
  AttemptContextSchema,
  AttemptHeartbeatRequestSchema,
  AttemptStartRequestSchema,
} from "../../src/generated/workflow/v1/runtime_pb.js";
import {
  CompleteExecutionCommandSchema,
  FinalizeStepCommandSchema,
  StateSnapshotSchema,
} from "../../src/generated/workflow/v1/state_pb.js";
import {
  clearWorkflowAdapter,
  registerWorkflowAdapter,
  resetHooks,
} from "../../src/hooks.js";
import {
  ActivityDispatch,
  ActivityReplay,
  AttemptHeartbeat,
  attemptStartRequestFromExecute,
  completeExecutionCommand,
  currentAttemptContext,
  newDurabilityOwnerId,
  runWithAttemptContext,
  WorkflowClient,
  WorkflowClientError,
} from "../../src/workflow/index.js";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  resetHooks();
});

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
  fencingToken: 7n,
  ownerId: "owner",
  workflowIdentity: identity,
  heartbeatIntervalMs: 1000n,
});

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("WorkflowClient", () => {
  test("parses OE-issued int64 attempt context and sends snake-case ProtoJSON", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        attempt_context: toJson(AttemptContextSchema, attempt, {
          useProtoFieldName: true,
        }),
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    const startRequest = create(AttemptStartRequestSchema, {
      workflowIdentity: identity,
      ownerId: "owner",
    });
    const result = await client.startAttempt(startRequest);

    expect(result?.fencingToken).toBe(7n);
    const init = fetchMock.mock.calls[0]?.[1];
    expect(String(init?.body)).toContain('"workflow_identity"');
    expect(String(init?.body)).not.toContain("workflowIdentity");
  });

  test("does not treat a coded bare 404 as old-OE native fallback", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse(
          { code: "WORKFLOW_ERROR_CODE_STALE_FENCE", message: "stale" },
          404,
        ),
      );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.STALE_FENCE });
  });

  test("a replacement attempt over a lost mid-batch execution fails admission with OUTCOME_UNKNOWN", async () => {
    // OE's recorded history for the lost attempt: one sibling terminal, one
    // started-but-unterminal (proven reachable by the durable-effects abort
    // test). Admission is the production fail-closed boundary: the fence
    // fires before any replacement graph, tool, or sibling replay can run.
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: "WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN",
            message:
              "execution has unfinished activities from a previous attempt",
          },
        },
        409,
      ),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.OUTCOME_UNKNOWN });
  });

  test("keeps a bare 404 on the native checkpoint path", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("", { status: 404 }));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).resolves.toBeNull();
  });

  test("keeps an uncoded JSON 404 envelope on the native checkpoint path", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ error: "Not Found" }, 404));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).resolves.toBeNull();
  });

  test("rejects a 200 start response that is not JSON", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("not json", { status: 200 }));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.INVALID_ARGUMENT });
  });

  test("rejects a 200 start response with neither attempt context nor error", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({}, 200));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.INVALID_ARGUMENT });
  });

  test("rejects a 200 start response whose attempt context is not ProtoJSON", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ attempt_context: "garbage" }, 200));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.startAttempt(
        create(AttemptStartRequestSchema, {
          workflowIdentity: identity,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.INVALID_ARGUMENT });
  });

  test("fails a 2xx heartbeat whose error member is not a WorkflowError", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ error: "lease rejected" }, 200));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.heartbeat(
        create(AttemptHeartbeatRequestSchema, {
          workflowIdentity: identity,
          attemptId: "attempt",
          fencingToken: 7n,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.OUTCOME_UNKNOWN });
  });

  test("fails a 2xx heartbeat that carries a coded WorkflowError", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          error: {
            code: "WORKFLOW_ERROR_CODE_STALE_FENCE",
            message: "superseded",
          },
        },
        200,
      ),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.heartbeat(
        create(AttemptHeartbeatRequestSchema, {
          workflowIdentity: identity,
          attemptId: "attempt",
          fencingToken: 7n,
          ownerId: "owner",
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.STALE_FENCE });
  });
});

describe("attempt start declaration", () => {
  const request = {
    org_id: "org",
    project_id: "project",
    workspace_id: "workspace",
    execution_id: "execution",
  } as Parameters<typeof attemptStartRequestFromExecute>[0];

  test("declares an attempt only after a framework adapter materializes", () => {
    expect(
      attemptStartRequestFromExecute(
        request,
        "session",
        "owner",
        "app",
        "version",
      ),
    ).toBeNull();

    registerWorkflowAdapter("langgraph", "1");
    expect(
      attemptStartRequestFromExecute(
        request,
        "session",
        "owner",
        "app",
        "version",
      )?.declaration,
    ).toMatchObject({
      workflowName: "app",
      workflowVersion: "version",
      adapterName: "langgraph",
      adapterVersion: "1",
    });
  });

  test("omitted application version uses the default workflow version", () => {
    registerWorkflowAdapter("langgraph", "1");
    expect(
      attemptStartRequestFromExecute(request, "session", "owner", "app")
        ?.declaration?.workflowVersion,
    ).toBe("1");
  });

  test("pins Memory enablement in the workflow declaration", () => {
    registerWorkflowAdapter("langgraph", "1");

    expect(
      attemptStartRequestFromExecute(
        request,
        "session",
        "owner",
        "app",
        "version",
        true,
      )?.declaration?.memoryEnabled,
    ).toBe(true);
    expect(
      attemptStartRequestFromExecute(request, "session", "owner", "app")
        ?.declaration?.memoryEnabled,
    ).toBe(false);
  });

  test("incomplete tenant identity stays native even with an adapter", () => {
    registerWorkflowAdapter("langgraph", "1");
    expect(
      attemptStartRequestFromExecute(
        { ...request, org_id: undefined },
        "session",
        "owner",
        "app",
      ),
    ).toBeNull();
  });

  test("whitespace-only identity stays native even with an adapter", () => {
    registerWorkflowAdapter("langgraph", "1");
    expect(
      attemptStartRequestFromExecute(
        { ...request, org_id: " " },
        "session",
        "owner",
        "app",
      ),
    ).toBeNull();
    expect(
      attemptStartRequestFromExecute(request, "  ", "owner", "app"),
    ).toBeNull();
  });

  test("an ineligible materialization after an eligible one stays native", () => {
    registerWorkflowAdapter("langgraph", "1");
    expect(
      attemptStartRequestFromExecute(
        request,
        "session",
        "owner",
        "app",
        "version",
      ),
    ).not.toBeNull();
    clearWorkflowAdapter();
    expect(
      attemptStartRequestFromExecute(
        request,
        "session",
        "owner",
        "app",
        "version",
      ),
    ).toBeNull();
  });

  test("owner ids are unique per call", () => {
    const first = newDurabilityOwnerId();
    const second = newDurabilityOwnerId();
    expect(first).not.toBe(second);
    expect(first.split(":").length).toBeGreaterThanOrEqual(3);
  });
});

describe("AttemptHeartbeat", () => {
  test("initial renewal failure fails closed", async () => {
    const heartbeatCall = vi
      .fn()
      .mockRejectedValue(
        new WorkflowClientError(
          WorkflowErrorCode.OUTCOME_UNKNOWN,
          "OE unreachable",
        ),
      );
    const heartbeat = new AttemptHeartbeat(attempt, {
      heartbeat: heartbeatCall,
    } as unknown as WorkflowClient);

    await expect(heartbeat.start()).rejects.toThrow(/initial/);
    expect(heartbeatCall).toHaveBeenCalledTimes(1);
  });

  test.each([
    ["stale fence", WorkflowErrorCode.STALE_FENCE],
    ["unauthorized", WorkflowErrorCode.UNAUTHORIZED],
    ["not found", WorkflowErrorCode.NOT_FOUND],
  ] as const)(
    "retries transient renewals and stops on %s",
    async (_name, leaseLostCode) => {
      vi.useFakeTimers();
      const heartbeatCall = vi
        .fn()
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(
          new WorkflowClientError(
            WorkflowErrorCode.OUTCOME_UNKNOWN,
            "temporary transport loss",
          ),
        )
        .mockRejectedValueOnce(
          new WorkflowClientError(leaseLostCode, "lease lost"),
        );
      const heartbeat = new AttemptHeartbeat(attempt, {
        heartbeat: heartbeatCall,
      } as unknown as WorkflowClient);

      await heartbeat.start();
      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(1000);
      expect(heartbeatCall).toHaveBeenCalledTimes(3);
      await vi.advanceTimersByTimeAsync(1000);
      expect(heartbeatCall).toHaveBeenCalledTimes(3);
      await heartbeat.stop();
    },
  );
});

describe("request-local attempt context", () => {
  test("binds attempt identity only inside the request-local scope", async () => {
    expect(currentAttemptContext()).toBeNull();
    await runWithAttemptContext(attempt, async () => {
      expect(currentAttemptContext()?.attemptId).toBe("attempt");
    });
    expect(currentAttemptContext()).toBeNull();
  });

  test("detached descendants cannot use the attempt after the request settles", async () => {
    let seenAfterSettle: string | null | undefined;
    await runWithAttemptContext(attempt, async () => {
      setTimeout(() => {
        seenAfterSettle = currentAttemptContext()?.attemptId ?? null;
      }, 0);
      expect(currentAttemptContext()?.attemptId).toBe("attempt");
    });
    expect(currentAttemptContext()).toBeNull();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(seenAfterSettle).toBeNull();
  });
});

describe("WorkflowClient activity endpoints", () => {
  const command = create(ActivityCommandSchema, {
    workflowIdentity: identity,
    attemptId: "attempt-1",
    fencingToken: 7n,
    activityKind: ActivityKind.TOOL,
    activityName: "lookup",
  });

  function matchingActivityContext(
    overrides: Record<string, unknown> = {},
  ): Record<string, unknown> {
    return {
      workflow_identity: toJson(WorkflowIdentitySchema, identity, {
        useProtoFieldName: true,
      }),
      activity_id: "activity-1",
      attempt_id: "attempt-1",
      fencing_token: "7",
      ...overrides,
    };
  }

  test("dispatches a fresh activity from activity_context", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        activity_context: matchingActivityContext(),
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    const started = await client.startActivity(command);
    expect(started).toBeInstanceOf(ActivityDispatch);
    if (!(started instanceof ActivityDispatch)) return;
    expect(started.context.activityId).toBe("activity-1");
    expect(started.context.workflowIdentity?.executionId).toBe("execution");
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://oe/executor/activity/start",
    );
  });

  test("replays a recorded outcome without a new activity context", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        outcome: {
          activity_id: "activity-1",
          outcome_kind: "ACTIVITY_OUTCOME_KIND_COMPLETED",
          result: { cached: true },
        },
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    const started = await client.startActivity(command);
    expect(started).toBeInstanceOf(ActivityReplay);
    if (!(started instanceof ActivityReplay)) return;
    expect(started.outcome.activityId).toBe("activity-1");
  });

  test("posts durable Memory writes to the activity memory endpoint", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({}));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });

    await client.ensureMemoryWritten(
      create(ActivityMemoryCommandSchema, {
        workflowIdentity: identity,
        activityId: "activity-1",
        attemptId: "attempt-1",
        fencingToken: 7n,
        memoryWrites: [],
      }),
    );

    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://oe/executor/activity/memory",
    );
    expect(String(fetchMock.mock.calls[0]?.[1]?.body)).toContain(
      '"activity_id":"activity-1"',
    );
  });

  test("rejects an activity start that carries neither context nor outcome", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({}));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
    });
  });

  test("rejects a proto-default activity_context without identity", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ activity_context: {} }));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
      message:
        "activity start response activity_context is missing identity fields",
    });
  });

  test("rejects an activity_context missing workflow identity", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        activity_context: {
          activity_id: "activity-1",
          attempt_id: "attempt-1",
          fencing_token: "7",
        },
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
      message:
        "activity start response activity_context is missing identity fields",
    });
  });

  test("rejects an activity_context whose attempt does not match the command", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        activity_context: matchingActivityContext({
          attempt_id: "attempt-other",
        }),
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
      message:
        "activity start response activity_context does not match the submitted command",
    });
  });

  test("rejects an activity_context whose fence does not match the command", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        activity_context: matchingActivityContext({ fencing_token: "8" }),
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
      message:
        "activity start response activity_context does not match the submitted command",
    });
  });

  test("rejects an activity_context whose workflow identity does not match the command", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        activity_context: matchingActivityContext({
          workflow_identity: toJson(
            WorkflowIdentitySchema,
            create(WorkflowIdentitySchema, {
              tenantScope: identity.tenantScope,
              sessionId: identity.sessionId,
              executionId: "execution-other",
            }),
            { useProtoFieldName: true },
          ),
        }),
      }),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(client.startActivity(command)).rejects.toMatchObject({
      code: WorkflowErrorCode.INVALID_ARGUMENT,
      message:
        "activity start response activity_context does not match the submitted command",
    });
  });

  test("maps a failed outcome body read to outcome unknown", async () => {
    const response = jsonResponse({});
    vi.spyOn(response, "text").mockRejectedValue(new TypeError("truncated"));
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(response);
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await expect(
      client.reportOutcome(
        create(ActivityOutcomeSchema, {
          activityId: "activity-1",
          attemptId: "attempt-1",
          fencingToken: 7n,
          outcomeKind: ActivityOutcomeKind.COMPLETED,
        }),
      ),
    ).rejects.toMatchObject({
      code: WorkflowErrorCode.OUTCOME_UNKNOWN,
    });
  });

  test("reports an outcome to the activity outcome path", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({}));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    await client.reportOutcome(
      create(ActivityOutcomeSchema, {
        activityId: "activity-1",
        attemptId: "attempt-1",
        fencingToken: 7n,
        outcomeKind: ActivityOutcomeKind.COMPLETED,
      }),
    );
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://oe/executor/activity/outcome",
    );
  });

  test("finalizes a settled step", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({}));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    const entries = await client.finalizeStep(
      create(FinalizeStepCommandSchema, {
        workflowIdentity: identity,
        attemptId: "attempt-1",
        fencingToken: 7n,
        stepOrdinal: 1n,
      }),
    );
    expect(entries).toEqual([]);
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://oe/executor/step/finalize",
    );
  });

  test("publishes one fenced terminal state", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({}));
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });
    const command = create(CompleteExecutionCommandSchema, {
      workflowIdentity: identity,
      attemptId: "attempt-1",
      fencingToken: 7n,
      state: create(StateSnapshotSchema, {}),
    });

    await client.completeExecution(command);

    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      "http://oe/executor/complete",
    );
    expect(String(fetchMock.mock.calls[0]?.[1]?.body)).toContain(
      '"fencing_token":"7"',
    );
  });

  test("fails closed when terminal publication loses its fence", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          code: "WORKFLOW_ERROR_CODE_STALE_FENCE",
          message: "stale",
        },
        412,
      ),
    );
    const client = new WorkflowClient("http://oe", { fetch: fetchMock });

    await expect(
      client.completeExecution(
        create(CompleteExecutionCommandSchema, {
          workflowIdentity: identity,
          attemptId: "attempt-1",
          fencingToken: 7n,
          state: create(StateSnapshotSchema, {}),
        }),
      ),
    ).rejects.toMatchObject({ code: WorkflowErrorCode.STALE_FENCE });
  });
});

test("completeExecutionCommand preserves the OE-issued identity and fence", () => {
  const state = create(StateSnapshotSchema, {});
  const command = completeExecutionCommand(attempt, state);

  expect(command.workflowIdentity).toEqual(identity);
  expect(command.attemptId).toBe("attempt");
  expect(command.fencingToken).toBe(7n);
  expect(command.state).toBe(state);
});
