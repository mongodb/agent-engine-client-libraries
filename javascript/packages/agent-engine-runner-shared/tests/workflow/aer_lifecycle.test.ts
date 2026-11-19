import { create, toJson } from "@bufbuild/protobuf";
import { afterEach, describe, expect, test, vi } from "vitest";

import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { resetHooks, registerWorkflowAdapter } from "../../src/hooks.js";
import { getCurrentExecutionId } from "../../src/context.js";
import type {
  ExecuteRequest,
  ExecutorCallbackRequest,
  InterruptResult,
  StreamingResult,
} from "../../src/models.js";
import { AERServer } from "../../src/server/aer.js";
import type { DrainRegistry } from "../../src/server/drain.js";
import type { ITenantRuntime } from "../../src/server/base.js";
import type { TurnMemoryWriter } from "../../src/memory_writer.js";
import { SecureToolWrapper } from "../../src/secure_wrapper.js";
import { DurableMemoryState } from "../../src/workflow/memory.js";
import type { ActivityMemoryCommand } from "../../src/generated/workflow/v1/activity_pb.js";
import { ActivityContextSchema } from "../../src/generated/workflow/v1/activity_pb.js";
import {
  AttemptHeartbeat,
  AttemptContextSchema,
  AttemptStartResponseSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  currentAttemptContext,
  currentCancelledHistory,
  WorkflowClient,
} from "../../src/workflow/index.js";

interface TestServer {
  runtime: ITenantRuntime;
  durabilityOwnerId: string;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  pendingSessionFinish: Set<string>;
  drainRegistry: DrainRegistry;
  executeViaAgentStream: ReturnType<typeof vi.fn>;
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  doHandleExecute(request: ExecuteRequest): Promise<unknown>;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  resetHooks();
});

function identity() {
  return create(WorkflowIdentitySchema, {
    tenantScope: create(TenantScopeSchema, {
      orgId: "org",
      projectId: "project",
      workspaceId: "workspace",
    }),
    sessionId: "session",
    executionId: "execution",
  });
}

function executeRequest(): ExecuteRequest {
  return {
    execution_id: "execution",
    session_id: "session",
    workspace_id: "workspace",
    org_id: "org",
    project_id: "project",
    user_id: "user",
    message: "hello",
    platform_api_url: "http://oe",
    suspend_generation: 7,
    resume: false,
  } as ExecuteRequest;
}

function stubDurableAttemptStart(): void {
  const attempt = create(AttemptContextSchema, {
    attemptId: "attempt",
    fencingToken: 8n,
    ownerId: "owner",
    workflowIdentity: identity(),
    heartbeatIntervalMs: 60_000n,
  });
  vi.spyOn(AttemptHeartbeat.prototype, "start").mockResolvedValue(undefined);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request) => {
      const path = new URL(String(input)).pathname;
      if (path !== "/executor/attempt/start") {
        throw new Error(`unexpected request to ${path}`);
      }
      return new Response(
        JSON.stringify(
          toJson(
            AttemptStartResponseSchema,
            create(AttemptStartResponseSchema, { attemptContext: attempt }),
            { useProtoFieldName: true },
          ),
        ),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }),
  );
}

function makeServer(
  registerAdapter: boolean,
  durableWorkflow: boolean | null = true,
): TestServer {
  const server = Object.create(AERServer.prototype) as TestServer;
  server.runtime = {
    appName: "test-app",
    appVersion: "1",
    orgId: "org",
    projectId: "project",
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () =>
      new RuntimeAgentConfig({
        language: "typescript",
        framework: "langgraph",
        features: {
          memory: null,
          guardrails: null,
          playground: null,
          deep_agent: null,
          use_custom_parser: null,
          durable_workflow: durableWorkflow,
        },
      }),
    getMongodbUri: () => null,
    getAgent: () => {
      if (registerAdapter) registerWorkflowAdapter("langgraph", "1");
      return {} as ReturnType<ITenantRuntime["getAgent"]>;
    },
  };
  server.durabilityOwnerId = "owner";
  server.chunkSeq = new Map();
  server.durableAckSenders = new Map();
  server.ownerCallbackUrl = new Map();
  server.pendingSessionFinish = new Set();
  server.executeViaAgentStream = vi.fn(async () => {
    return { content: "done", messages: [] } as StreamingResult;
  });
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  return server;
}

describe("AER durable attempt lifecycle", () => {
  test.each([50, 5000])(
    "bounds cancellation history delivery by the deadline and 3s cap (%sms)",
    async (windowMs) => {
      vi.useFakeTimers();
      stubDurableAttemptStart();
      let entered!: () => void;
      const reportEntered = new Promise<void>((resolve) => {
        entered = resolve;
      });
      vi.spyOn(
        WorkflowClient.prototype,
        "reportCancelledHistory",
      ).mockImplementation(() => {
        entered();
        return new Promise<void>(() => {});
      });
      const server = makeServer(true);
      server.runtime.getAgent = () => {
        registerWorkflowAdapter("langgraph", "1");
        return {
          captureCancelledHistory: () =>
            currentCancelledHistory()?.capture() ?? null,
        } as unknown as ReturnType<ITenantRuntime["getAgent"]>;
      };
      server.drainRegistry.beginWork("execution");
      server.executeViaAgentStream.mockImplementation(async () => {
        server.drainRegistry.apply({
          execution_id: "execution",
          request_id: "bounded-report",
          reason: "execution_cancelled",
          deadline_at_ms: Date.now() + windowMs,
        });
        throw new Error("stopped");
      });
      try {
        let settled = false;
        const pending = server
          .doHandleExecute(executeRequest())
          .then((result) => {
            settled = true;
            return result;
          });
        await reportEntered;
        const bound = Math.min(windowMs, 3000);
        await vi.advanceTimersByTimeAsync(bound - 1);
        expect(settled).toBe(false);
        await vi.advanceTimersByTimeAsync(1);
        expect(await pending).toEqual({ status: "cancelled" });
        expect(server.reportCallback).not.toHaveBeenCalled();
      } finally {
        server.drainRegistry.endWork("execution");
        vi.useRealTimers();
      }
    },
  );

  test("does not retain cancelled history for native-checkpoint executions", async () => {
    const report = vi
      .spyOn(WorkflowClient.prototype, "reportCancelledHistory")
      .mockResolvedValue(undefined);
    const server = makeServer(true, false);
    server.runtime.getAgent = () =>
      ({
        captureCancelledHistory: () => {
          throw new Error("native capture must not run");
        },
      }) as unknown as ReturnType<ITenantRuntime["getAgent"]>;
    server.drainRegistry.beginWork("execution");
    server.executeViaAgentStream.mockImplementation(async () => {
      server.drainRegistry.apply({
        execution_id: "execution",
        request_id: "native-stop",
        reason: "execution_cancelled",
        deadline_at_ms: Date.now() + 5000,
      });
      throw new Error("stopped");
    });
    try {
      expect(await server.doHandleExecute(executeRequest())).toEqual({
        status: "cancelled",
      });
      expect(report).not.toHaveBeenCalled();
    } finally {
      server.drainRegistry.endWork("execution");
    }
  });

  test.each(["execution_cancelled", "rolling_restart", "ordinary"] as const)(
    "retains display history only for durable execution cancellation (%s)",
    async (reason) => {
      stubDurableAttemptStart();
      const report = vi
        .spyOn(WorkflowClient.prototype, "reportCancelledHistory")
        .mockResolvedValue(undefined);
      const server = makeServer(true);
      server.runtime.getAgent = () => {
        registerWorkflowAdapter("langgraph", "1");
        return {
          captureCancelledHistory: () =>
            currentCancelledHistory()?.capture() ?? null,
        } as unknown as ReturnType<ITenantRuntime["getAgent"]>;
      };
      server.drainRegistry.beginWork("execution");
      server.executeViaAgentStream.mockImplementation(async () => {
        if (reason !== "ordinary") {
          server.drainRegistry.apply({
            execution_id: "execution",
            request_id: "cancel-history",
            reason,
            deadline_at_ms: Date.now() + 5000,
          });
        }
        throw new Error("stopped");
      });
      try {
        if (reason === "ordinary") {
          await expect(
            server.doHandleExecute(executeRequest()),
          ).rejects.toThrow("stopped");
        } else {
          expect(await server.doHandleExecute(executeRequest())).toEqual({
            status: "cancelled",
          });
        }
        if (reason === "execution_cancelled") {
          expect(report).toHaveBeenCalledTimes(1);
          expect(report.mock.calls[0]?.[0]).toMatchObject({
            opener_message_id: "durable-input:execution",
            accepted_step_ordinal: "0",
            messages: [{ content: "hello", id: "durable-input:execution" }],
          });
          expect(server.reportCallback).not.toHaveBeenCalled();
        } else {
          expect(report).not.toHaveBeenCalled();
        }
      } finally {
        server.drainRegistry.endWork("execution");
      }
    },
  );

  test("retains the opener when a Stop lands during initial heartbeat activation", async () => {
    // The reporting finally cannot run until the initial renewal settles, and
    // OE's drain window is shorter than the transport timeout. A Stop must
    // therefore interrupt the renewal in flight, or the stopped turn is dropped
    // even though the Stop was delivered. The renewal here is held pending and
    // never released: only the drain can end it.
    vi.useFakeTimers();
    const attempt = create(AttemptContextSchema, {
      attemptId: "attempt",
      fencingToken: 8n,
      ownerId: "owner",
      workflowIdentity: identity(),
      heartbeatIntervalMs: 60_000n,
    });
    let renewalEntered!: () => void;
    const renewalStarted = new Promise<void>((resolve) => {
      renewalEntered = resolve;
    });
    let heartbeatRequests = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const path = new URL(String(input)).pathname;
        if (path === "/executor/attempt/start") {
          return new Response(
            JSON.stringify(
              toJson(
                AttemptStartResponseSchema,
                create(AttemptStartResponseSchema, { attemptContext: attempt }),
                { useProtoFieldName: true },
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        if (path === "/executor/attempt/heartbeat") {
          heartbeatRequests += 1;
          renewalEntered();
          const signal = init?.signal;
          return new Promise<Response>((_resolve, reject) => {
            if (signal?.aborted === true) {
              reject(signal.reason);
              return;
            }
            signal?.addEventListener("abort", () => reject(signal.reason), {
              once: true,
            });
          });
        }
        throw new Error(`unexpected request to ${path}`);
      }),
    );
    const report = vi
      .spyOn(WorkflowClient.prototype, "reportCancelledHistory")
      .mockResolvedValue(undefined);

    const server = makeServer(true);
    server.runtime.getAgent = () => {
      registerWorkflowAdapter("langgraph", "1");
      return {
        captureCancelledHistory: () =>
          currentCancelledHistory()?.capture() ?? null,
      } as unknown as ReturnType<ITenantRuntime["getAgent"]>;
    };
    server.drainRegistry.beginWork("execution");
    try {
      const pending = server.doHandleExecute(executeRequest());
      await renewalStarted;
      server.drainRegistry.apply({
        execution_id: "execution",
        request_id: "stop-during-activation",
        reason: "execution_cancelled",
        deadline_at_ms: Date.now() + 3000,
      });
      await expect(pending).resolves.toEqual({ status: "cancelled" });
      // No tenant code ran, and the drain stages no terminal callback.
      expect(server.executeViaAgentStream).not.toHaveBeenCalled();
      expect(server.reportCallback).not.toHaveBeenCalled();
      expect(report).toHaveBeenCalledTimes(1);
      expect(report.mock.calls[0]?.[0]).toMatchObject({
        opener_message_id: "durable-input:execution",
        accepted_step_ordinal: "0",
        messages: [
          {
            role: "MESSAGE_ROLE_USER",
            content: "hello",
            id: "durable-input:execution",
          },
        ],
      });
      // No heartbeat survives activation: the interval never renews again.
      await vi.advanceTimersByTimeAsync(120_000);
      expect(heartbeatRequests).toBe(1);
    } finally {
      server.drainRegistry.endWork("execution");
      vi.useRealTimers();
    }
  });

  test("reports provisional display history when a durable execution suspends", async () => {
    // A suspension ends the attempt, so a later Stop finds no execute handler:
    // the AER reports the display transcript now, marked provisional.
    stubDurableAttemptStart();
    const report = vi
      .spyOn(WorkflowClient.prototype, "reportCancelledHistory")
      .mockResolvedValue(undefined);
    const server = makeServer(true);
    server.runtime.getAgent = () => {
      registerWorkflowAdapter("langgraph", "1");
      return {
        captureCancelledHistory: () =>
          currentCancelledHistory()?.capture() ?? null,
      } as unknown as ReturnType<ITenantRuntime["getAgent"]>;
    };
    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: {
          suspend_reason: "awaiting_human_review",
          suspend_context: {},
        },
        metadata: {},
      } as InterruptResult;
    });
    server.reportCallback = vi.fn(async () => {});

    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({
      status: "suspended",
    });
    expect(report).toHaveBeenCalledTimes(1);
    expect(report.mock.calls[0]?.[0]).toMatchObject({
      opener_message_id: "durable-input:execution",
      provisional: true,
    });
  });

  test("cancelled history transport failures do not change cancellation settlement", async () => {
    stubDurableAttemptStart();
    vi.spyOn(
      WorkflowClient.prototype,
      "reportCancelledHistory",
    ).mockRejectedValue(new Error("transport unavailable"));
    const server = makeServer(true);
    server.runtime.getAgent = () => {
      registerWorkflowAdapter("langgraph", "1");
      return {
        captureCancelledHistory: () =>
          currentCancelledHistory()?.capture() ?? null,
      } as unknown as ReturnType<ITenantRuntime["getAgent"]>;
    };
    server.drainRegistry.beginWork("execution");
    server.executeViaAgentStream.mockImplementation(async () => {
      server.drainRegistry.apply({
        execution_id: "execution",
        request_id: "failed-report",
        reason: "execution_cancelled",
        deadline_at_ms: Date.now() + 5000,
      });
      throw new Error("stopped");
    });
    try {
      expect(await server.doHandleExecute(executeRequest())).toEqual({
        status: "cancelled",
      });
      expect(server.reportCallback).not.toHaveBeenCalled();
    } finally {
      server.drainRegistry.endWork("execution");
    }
  });

  test("materializes the adapter, binds the OE attempt, and heartbeats", async () => {
    vi.stubEnv("APP_ID", "workspace");
    const attempt = create(AttemptContextSchema, {
      attemptId: "attempt",
      fencingToken: 8n,
      ownerId: "owner",
      workflowIdentity: identity(),
      heartbeatIntervalMs: 60_000n,
    });
    const paths: string[] = [];
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
      },
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const path = new URL(String(input)).pathname;
        paths.push(path);
        // The AER's one-shot capability advertise fires on this first
        // /execute using the deployment's configured workspace identity.
        if (path === "/agent/capabilities") {
          return new Response(null, { status: 200 });
        }
        if (path === "/executor/attempt/start") {
          return new Response(
            JSON.stringify(
              toJson(
                AttemptStartResponseSchema,
                create(AttemptStartResponseSchema, {
                  attemptContext: attempt,
                }),
                { useProtoFieldName: true },
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        if (path === "/executor/attempt/heartbeat") {
          return new Response(null, { status: 204 });
        }
        throw new Error(`unexpected request to ${path}`);
      }),
    );

    const server = makeServer(true);
    const delivered: ExecutorCallbackRequest[] = [];
    const captureCallback = async (
      _oeUrl: string,
      _executionId: string,
      _status: string,
      _fields: Record<string, unknown>,
      stagedCallback?: ExecutorCallbackRequest,
    ) => {
      expect(getCurrentExecutionId()).toBeNull();
      expect(currentAttemptContext()).toBeNull();
      if (stagedCallback !== undefined) delivered.push(stagedCallback);
      settlementOrder.push("callback");
    };
    server.reportCallback = vi.fn(captureCallback);
    server.executeViaAgentStream = vi.fn(async () => {
      expect(currentAttemptContext()).toMatchObject({
        attemptId: "attempt",
        fencingToken: 8n,
      });
      return { content: "done", messages: [] } as StreamingResult;
    });

    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({ status: "completed", result: "done" });

    expect(currentAttemptContext()).toBeNull();
    expect(paths).toEqual([
      "/agent/capabilities",
      "/executor/attempt/start",
      "/executor/attempt/heartbeat",
    ]);
    expect(delivered).toHaveLength(1);
    expect(delivered[0]?.suspend_generation).toBe(7);
    expect(delivered[0]).toMatchObject({
      attempt_id: "attempt",
      attempt_fence: 8,
    });
    expect(settlementOrder).toEqual(["wrapper-close", "callback"]);
  });

  test("seeds a fresh durable Memory wrapper with the pending user message", async () => {
    vi.stubEnv("APP_ID", "workspace");
    const attempt = create(AttemptContextSchema, {
      attemptId: "attempt",
      fencingToken: 8n,
      ownerId: "owner",
      workflowIdentity: identity(),
      heartbeatIntervalMs: 60_000n,
    });
    vi.spyOn(AttemptHeartbeat.prototype, "start").mockResolvedValue(undefined);
    const startDeclarations: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const path = new URL(String(input)).pathname;
        if (path === "/agent/capabilities") {
          return new Response(null, { status: 200 });
        }
        if (path === "/executor/attempt/start") {
          startDeclarations.push(
            JSON.parse(String(init?.body ?? "{}"))?.declaration,
          );
          return new Response(
            JSON.stringify(
              toJson(
                AttemptStartResponseSchema,
                create(AttemptStartResponseSchema, {
                  attemptContext: attempt,
                }),
                { useProtoFieldName: true },
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        if (path === "/executor/attempt/heartbeat") {
          return new Response(null, { status: 204 });
        }
        throw new Error(`unexpected request to ${path}`);
      }),
    );

    const wrappers: SecureToolWrapper[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async function (this: SecureToolWrapper) {
        wrappers.push(this);
      },
    );
    const server = makeServer(true);
    server.runtime = {
      ...server.runtime,
      memoryWriter: {} as TurnMemoryWriter,
    };
    server.reportCallback = vi.fn(async () => {});

    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({
      status: "completed",
    });
    // A wrapper per execute, each with its own Memory state.
    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({
      status: "completed",
    });

    expect(wrappers).toHaveLength(2);
    const states = wrappers.map((wrapper) => wrapper.durableMemory);
    expect(states[0]).toBeInstanceOf(DurableMemoryState);
    expect(states[1]).toBeInstanceOf(DurableMemoryState);
    expect(states[0]).not.toBe(states[1]);

    // The pending message is the request message, proven by driving one
    // synchronization through the provisioned state.
    const written: ActivityMemoryCommand[] = [];
    await states[0]?.synchronizeLlm(
      {
        ensureMemoryWritten: async (command) => {
          written.push(command);
        },
      },
      create(ActivityContextSchema, {
        workflowIdentity: identity(),
        activityId: "activity-1",
        attemptId: "attempt",
        fencingToken: 8n,
      }),
      { content: "agent reply" },
      "user",
    );
    const contents = (written[0]?.memoryWrites ?? []).map((write) =>
      JSON.parse(new TextDecoder().decode(write.payloadJson as Uint8Array)),
    );
    expect(contents).toMatchObject([
      { role: "user", content: "hello" },
      { role: "assistant", content: "agent reply" },
    ]);

    // A present writer must declare Memory enabled to OE on both executes.
    expect(startDeclarations).toHaveLength(2);
    for (const declaration of startDeclarations) {
      expect(declaration).toMatchObject({ memory_enabled: true });
    }
  });

  test("leaves durable Memory unset without a Memory writer", async () => {
    // The runtime field is optional: both an absent writer (undefined) and
    // an explicit null must behave like a disabled writer.
    vi.stubEnv("APP_ID", "workspace");
    const attempt = create(AttemptContextSchema, {
      attemptId: "attempt",
      fencingToken: 8n,
      ownerId: "owner",
      workflowIdentity: identity(),
      heartbeatIntervalMs: 60_000n,
    });
    vi.spyOn(AttemptHeartbeat.prototype, "start").mockResolvedValue(undefined);
    const startDeclarations: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
        const path = new URL(String(input)).pathname;
        if (path === "/agent/capabilities") {
          return new Response(null, { status: 200 });
        }
        if (path === "/executor/attempt/start") {
          startDeclarations.push(
            JSON.parse(String(init?.body ?? "{}"))?.declaration,
          );
          return new Response(
            JSON.stringify(
              toJson(
                AttemptStartResponseSchema,
                create(AttemptStartResponseSchema, {
                  attemptContext: attempt,
                }),
                { useProtoFieldName: true },
              ),
            ),
            { status: 200, headers: { "Content-Type": "application/json" } },
          );
        }
        if (path === "/executor/attempt/heartbeat") {
          return new Response(null, { status: 204 });
        }
        throw new Error(`unexpected request to ${path}`);
      }),
    );

    const wrappers: SecureToolWrapper[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async function (this: SecureToolWrapper) {
        wrappers.push(this);
      },
    );
    const server = makeServer(true);
    // Default fixture omits the optional writer entirely (undefined).
    server.reportCallback = vi.fn(async () => {});

    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({
      status: "completed",
    });
    expect(wrappers).toHaveLength(1);
    expect(wrappers[0]?.durableMemory).toBeNull();

    server.runtime = { ...server.runtime, memoryWriter: null };
    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({
      status: "completed",
    });
    expect(wrappers).toHaveLength(2);
    expect(wrappers[1]?.durableMemory).toBeNull();

    // Both absent and null writers must declare Memory disabled to OE.
    // (protojson omits the default-false field, so absence is the
    // disabled signal — presence with true is the enabled one.)
    expect(startDeclarations).toHaveLength(2);
    for (const declaration of startDeclarations) {
      expect(declaration).not.toMatchObject({ memory_enabled: true });
    }
  });

  test("preserves the native path when no adapter is registered", async () => {
    vi.stubEnv("APP_ID", "workspace");
    const paths: string[] = [];
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const path = new URL(String(input)).pathname;
      paths.push(path);
      // The AER's one-shot capability advertise fires on this first
      // /execute using APP_ID; only the
      // durable-workflow routes are off-limits on the native path.
      if (path === "/agent/capabilities") {
        return new Response(null, { status: 200 });
      }
      throw new Error(
        `native path must not contact OE workflow routes: ${path}`,
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
      },
    );
    const server = makeServer(false);
    const captureCallback = async () => {
      expect(getCurrentExecutionId()).toBeNull();
      expect(currentAttemptContext()).toBeNull();
      settlementOrder.push("callback");
    };
    server.reportCallback = vi.fn(captureCallback);
    await expect(
      server.doHandleExecute(executeRequest()),
    ).resolves.toMatchObject({ status: "completed", result: "done" });

    expect(currentAttemptContext()).toBeNull();
    expect(paths).toEqual(["/agent/capabilities"]);
    expect(server.executeViaAgentStream).toHaveBeenCalledOnce();
    expect(settlementOrder).toEqual(["wrapper-close", "callback"]);
  });

  test("delivers the callback metadata snapshot staged before cleanup", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("native path must not contact OE workflow routes");
      }),
    );

    const outcome = {
      content: "done",
      messages: [],
      metadata: { nested: { value: "before-cleanup" } },
    } as StreamingResult;
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        if (outcome.metadata === undefined) {
          throw new Error("test outcome metadata is required");
        }
        const nested = outcome.metadata.nested as { value: string };
        nested.value = "after-cleanup";
      },
    );

    const delivered: ExecutorCallbackRequest[] = [];
    const server = makeServer(false);
    server.executeViaAgentStream = vi.fn(async () => outcome);
    server.reportCallback = vi.fn(
      async (
        _oeUrl: string,
        executionId: string,
        status: string,
        fields: Record<string, unknown>,
        stagedCallback?: ExecutorCallbackRequest,
      ) => {
        delivered.push(
          stagedCallback ?? { execution_id: executionId, status, ...fields },
        );
      },
    );

    const request = executeRequest();
    (
      request as unknown as { suspend_generation: number | null }
    ).suspend_generation = null;
    await expect(server.doHandleExecute(request)).resolves.toMatchObject({
      status: "completed",
      result: "done",
    });

    expect(delivered).toHaveLength(1);
    expect(delivered[0]?.metadata).toEqual({
      nested: { value: "before-cleanup" },
    });
    expect(delivered[0]).not.toHaveProperty("suspend_generation");
  });

  test("reports completion before propagating wrapper cleanup failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("native path must not contact OE workflow routes");
      }),
    );
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
        throw new Error("wrapper-close");
      },
    );

    const delivered: ExecutorCallbackRequest[] = [];
    const server = makeServer(false);
    server.reportCallback = vi.fn(
      async (
        _oeUrl: string,
        _executionId: string,
        _status: string,
        _fields: Record<string, unknown>,
        stagedCallback?: ExecutorCallbackRequest,
      ) => {
        settlementOrder.push("callback");
        if (stagedCallback !== undefined) delivered.push(stagedCallback);
      },
    );

    await expect(server.doHandleExecute(executeRequest())).rejects.toThrow(
      "wrapper-close",
    );

    expect(delivered).toHaveLength(1);
    expect(delivered[0]?.status).toBe("COMPLETED");
    expect(settlementOrder).toEqual(["wrapper-close", "callback"]);
  });

  test("reports suspension before propagating wrapper cleanup failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("native path must not contact OE workflow routes");
      }),
    );
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
        throw new Error("wrapper-close");
      },
    );

    const delivered: ExecutorCallbackRequest[] = [];
    const server = makeServer(false);
    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: {
          suspend_reason: "awaiting_human_review",
          suspend_context: { checkpoint: "ready" },
        },
        metadata: {},
      } as InterruptResult;
    });
    server.reportCallback = vi.fn(
      async (
        _oeUrl: string,
        _executionId: string,
        _status: string,
        _fields: Record<string, unknown>,
        stagedCallback?: ExecutorCallbackRequest,
      ) => {
        settlementOrder.push("callback");
        if (stagedCallback !== undefined) delivered.push(stagedCallback);
      },
    );

    await expect(server.doHandleExecute(executeRequest())).rejects.toThrow(
      "wrapper-close",
    );

    expect(delivered).toHaveLength(1);
    expect(delivered[0]).toMatchObject({
      status: "SUSPENDED",
      suspend_generation: 7,
      suspend_reason: "awaiting_human_review",
      suspend_context: { checkpoint: "ready" },
    });
    expect(settlementOrder).toEqual(["wrapper-close", "callback"]);
  });

  test("preserves execution failure when wrapper cleanup also fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("native path must not contact OE workflow routes");
      }),
    );
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
        throw new Error("wrapper-close");
      },
    );

    const executionError = new Error("execution-failed");
    const delivered: ExecutorCallbackRequest[] = [];
    const server = makeServer(false);
    server.executeViaAgentStream = vi.fn(async () => {
      throw executionError;
    });
    server.reportCallback = vi.fn(
      async (
        _oeUrl: string,
        _executionId: string,
        _status: string,
        _fields: Record<string, unknown>,
        stagedCallback?: ExecutorCallbackRequest,
      ) => {
        settlementOrder.push("callback");
        if (stagedCallback !== undefined) delivered.push(stagedCallback);
      },
    );

    await expect(server.doHandleExecute(executeRequest())).rejects.toBe(
      executionError,
    );

    expect(delivered).toHaveLength(1);
    expect(delivered[0]).toMatchObject({
      status: "ERROR",
      error: "execution-failed",
    });
    expect(settlementOrder).toEqual(["wrapper-close", "callback"]);
  });

  test("preserves the first cleanup failure when heartbeat shutdown also fails", async () => {
    stubDurableAttemptStart();
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
        throw new Error("wrapper-close");
      },
    );
    vi.spyOn(AttemptHeartbeat.prototype, "stop").mockImplementation(
      async () => {
        settlementOrder.push("heartbeat-stop");
        throw new Error("heartbeat-stop");
      },
    );

    const server = makeServer(true);
    server.reportCallback = vi.fn(async () => {
      settlementOrder.push("callback");
    });

    await expect(server.doHandleExecute(executeRequest())).rejects.toThrow(
      "wrapper-close",
    );
    expect(settlementOrder).toEqual([
      "wrapper-close",
      "heartbeat-stop",
      "callback",
    ]);
  });

  test("preserves execution failure when heartbeat shutdown also fails", async () => {
    stubDurableAttemptStart();
    const settlementOrder: string[] = [];
    vi.spyOn(SecureToolWrapper.prototype, "close").mockImplementation(
      async () => {
        settlementOrder.push("wrapper-close");
      },
    );
    vi.spyOn(AttemptHeartbeat.prototype, "stop").mockImplementation(
      async () => {
        settlementOrder.push("heartbeat-stop");
        throw new Error("heartbeat-stop");
      },
    );

    const executionError = new Error("execution-failed");
    const server = makeServer(true);
    server.executeViaAgentStream = vi.fn(async () => {
      throw executionError;
    });
    server.reportCallback = vi.fn(
      async (_url, _id, _status, _fields, callback) => {
        expect(callback).toMatchObject({
          status: "ERROR",
          attempt_id: "attempt",
          attempt_fence: 8,
          error: "execution-failed",
        });
        settlementOrder.push("callback");
      },
    );

    await expect(server.doHandleExecute(executeRequest())).rejects.toBe(
      executionError,
    );
    expect(settlementOrder).toEqual([
      "wrapper-close",
      "heartbeat-stop",
      "callback",
    ]);
  });

  test.each([
    ["omitted", null],
    ["disabled", false],
  ] as const)(
    "defers to OE when the manifest feature is %s",
    async (_label, durableWorkflow) => {
      vi.stubEnv("APP_ID", "workspace");
      const paths: string[] = [];
      const fetchMock = vi.fn(async (input: string | URL | Request) => {
        const path = new URL(String(input)).pathname;
        paths.push(path);
        if (path === "/agent/capabilities") {
          return new Response(null, { status: 200 });
        }
        expect(path).toBe("/executor/attempt/start");
        return new Response(null, { status: 404 });
      });
      vi.stubGlobal("fetch", fetchMock);

      const server = makeServer(true, durableWorkflow);
      await expect(
        server.doHandleExecute(executeRequest()),
      ).resolves.toMatchObject({ status: "completed", result: "done" });

      expect(currentAttemptContext()).toBeNull();
      expect(paths).toEqual(["/agent/capabilities", "/executor/attempt/start"]);
      expect(server.executeViaAgentStream).toHaveBeenCalledOnce();
    },
  );
});
