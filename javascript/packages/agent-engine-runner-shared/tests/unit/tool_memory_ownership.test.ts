import { create, fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { afterEach, describe, expect, test, vi } from "vitest";

import { runWithExecutionContext } from "../../src/context.js";
import { AppBoundRuntime } from "../../src/memory_appbound.js";
import {
  createSecureToolFunction,
  SecureToolWrapper,
} from "../../src/secure_wrapper.js";
import {
  ActivityContextSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
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
});

const memoryContext = {
  formatted_context: "customer context",
  metadata: { token_count: 0, memory_counts: {}, timing: {} },
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
}

function dispatchResponse(activityId: string): Response {
  return jsonResponse({
    activity_context: toJson(
      ActivityContextSchema,
      create(ActivityContextSchema, {
        workflowIdentity: identity,
        activityId,
        attemptId: attempt.attemptId,
        fencingToken: attempt.fencingToken,
      }),
      { useProtoFieldName: true },
    ),
  });
}

function replayResponse(activityId: string, result: unknown): Response {
  return jsonResponse({
    outcome: toJson(
      ActivityOutcomeSchema,
      create(ActivityOutcomeSchema, {
        workflowIdentity: identity,
        activityId,
        attemptId: attempt.attemptId,
        fencingToken: attempt.fencingToken,
        outcomeKind: ActivityOutcomeKind.COMPLETED,
        result: fromJson(ValueSchema, result),
      }),
      { useProtoFieldName: true },
    ),
  });
}

function requestBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  return JSON.parse(init.body) as Record<string, unknown>;
}

function runDurably<T>(wrapper: SecureToolWrapper, fn: () => T): T {
  return runWithExecutionContext(
    {
      executionId: "execution",
      wrapper,
      oeUrl: "http://oe",
      userId: "user",
      sessionId: "session",
    },
    () => runWithAttemptContext(attempt, fn),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("Tool-owned Memory reads", () => {
  test("suppresses only the nested Memory activity for a live registered Tool", async () => {
    const starts: Record<string, unknown>[] = [];
    const memoryRequests: Record<string, unknown>[] = [];
    let toolStarts = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          const body = requestBody(init);
          starts.push(body);
          if (body["activity_kind"] === "ACTIVITY_KIND_TOOL") {
            toolStarts += 1;
            return toolStarts === 1
              ? dispatchResponse("tool-live")
              : replayResponse("tool-replay", memoryContext);
          }
          return dispatchResponse("memory-standalone");
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/api/v1/memory/context")) {
          memoryRequests.push(requestBody(init));
          return jsonResponse(memoryContext);
        }
        if (
          url.endsWith("/tool/result") ||
          url.endsWith("/executor/activity/outcome")
        ) {
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    const invoke = vi.fn(() =>
      new AppBoundRuntime().buildContext({
        query: "customer C-001",
        userId: "user",
      }),
    );
    const wrapped = createSecureToolFunction({ invoke }, "recall_customer");

    const liveWrapper = new SecureToolWrapper("http://oe", "execution");
    expect(await runDurably(liveWrapper, () => wrapped())).toEqual(
      memoryContext,
    );

    const replayWrapper = new SecureToolWrapper("http://oe", "execution");
    expect(await runDurably(replayWrapper, () => wrapped())).toEqual(
      memoryContext,
    );

    const standaloneWrapper = new SecureToolWrapper("http://oe", "execution");
    expect(
      await runDurably(standaloneWrapper, () =>
        new AppBoundRuntime().buildContext({
          query: "standalone question",
          userId: "user",
        }),
      ),
    ).toEqual(memoryContext);

    expect(invoke).toHaveBeenCalledOnce();
    expect(starts.map((body) => body["activity_kind"])).toEqual([
      "ACTIVITY_KIND_TOOL",
      "ACTIVITY_KIND_TOOL",
      "ACTIVITY_KIND_MEMORY",
    ]);
    expect(memoryRequests).toHaveLength(2);
  });

  test("invalidates ownership in detached work after the Tool returns", async () => {
    const starts: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/executor/activity/start")) {
          starts.push(requestBody(init));
          return dispatchResponse(`activity-${starts.length}`);
        }
        if (url.endsWith("/tool/execute")) {
          return jsonResponse({ proceed: true, route_to: "callback" });
        }
        if (url.endsWith("/api/v1/memory/context")) {
          return jsonResponse(memoryContext);
        }
        if (
          url.endsWith("/tool/result") ||
          url.endsWith("/executor/activity/outcome")
        ) {
          return jsonResponse({});
        }
        throw new Error(`unexpected URL: ${url}`);
      }),
    );

    let releaseDescendant: (() => void) | undefined;
    const descendantGate = new Promise<void>((resolve) => {
      releaseDescendant = resolve;
    });
    let descendant: Promise<unknown> | undefined;
    const wrapped = createSecureToolFunction(
      {
        invoke: () => {
          descendant = (async () => {
            await descendantGate;
            return new AppBoundRuntime().buildContext({
              query: "after Tool",
              userId: "user",
            });
          })();
          return "done";
        },
      },
      "start_background_read",
    );

    const wrapper = new SecureToolWrapper("http://oe", "execution");
    await runDurably(wrapper, async () => {
      expect(await wrapped()).toBe("done");
      releaseDescendant?.();
      expect(await descendant).toEqual(memoryContext);
    });

    expect(starts.map((body) => body["activity_kind"])).toEqual([
      "ACTIVITY_KIND_TOOL",
      "ACTIVITY_KIND_MEMORY",
    ]);
  });
});
