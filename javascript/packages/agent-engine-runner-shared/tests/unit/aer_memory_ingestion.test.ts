import { afterEach, describe, expect, test, vi } from "vitest";

import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { resetHooks } from "../../src/hooks.js";
import type {
  ExecuteRequest,
  InterruptResult,
  StreamingResult,
} from "../../src/models.js";
import { AERServer } from "../../src/server/aer.js";
import type { ITenantRuntime } from "../../src/server/base.js";
import type { TurnMemoryWriter } from "../../src/memory_writer.js";

interface TestServer {
  runtime: ITenantRuntime;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  pendingSessionFinish: Set<string>;
  executeViaAgentStream: ReturnType<typeof vi.fn>;
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  doHandleExecute(request: ExecuteRequest): Promise<unknown>;
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  resetHooks();
});

function request(resume = false): ExecuteRequest {
  return {
    execution_id: resume ? "exec-resume" : "exec-fresh",
    message: resume ? "" : "File a claim",
    platform_api_url: "http://oe:8000",
    resume,
    session_id: "session-1",
    user_id: "user-1",
  } as ExecuteRequest;
}

function makeServer(): {
  server: TestServer;
  writer: TurnMemoryWriter & {
    writeTurnAsync: ReturnType<typeof vi.fn>;
  };
} {
  vi.stubEnv("APP_ID", "");
  const writer = {
    pendingWrites: 0,
    writeTurnAsync: vi.fn(),
    drain: vi.fn().mockResolvedValue(true),
    shutdown: vi.fn().mockResolvedValue(undefined),
  };
  const server = Object.create(AERServer.prototype) as TestServer;
  server.runtime = {
    appName: "memory-test",
    appVersion: "1",
    orgId: "org-1",
    projectId: "project-1",
    memoryWriter: writer,
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => new RuntimeAgentConfig(),
    getMongodbUri: () => null,
    getAgent: () => ({}) as ReturnType<ITenantRuntime["getAgent"]>,
  };
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.pendingSessionFinish = new Set();
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  return { server, writer };
}

describe("AER native Memory ingestion", () => {
  test("queues a completed conversation turn", async () => {
    const { server, writer } = makeServer();
    const messages = [
      { role: "assistant", content: "Checking" },
      { role: "assistant", content: "Approved" },
    ] as StreamingResult["messages"];
    server.executeViaAgentStream = vi.fn(async () => ({
      content: "Approved",
      messages,
    }));

    await expect(server.doHandleExecute(request())).resolves.toMatchObject({
      status: "completed",
    });

    expect(writer.writeTurnAsync).toHaveBeenCalledOnce();
    expect(writer.writeTurnAsync).toHaveBeenCalledWith({
      message: "File a claim",
      resultMessages: messages,
      userId: "user-1",
      sessionId: "session-1",
      includeUserTurn: true,
    });
  });

  test("writes the suspend leg, then omits the user turn after replacement resume", async () => {
    const preSuspend = makeServer();
    const preSuspendMessages = [
      { role: "assistant", content: "Waiting for approval" },
    ] as InterruptResult["messages"];
    preSuspend.server.executeViaAgentStream = vi.fn(async () => ({
      suspend_payload: {
        suspend_reason: "approval",
        suspend_context: {},
      },
      metadata: { checkpoint_id: "checkpoint-1" },
      messages: preSuspendMessages,
    }));

    await expect(
      preSuspend.server.doHandleExecute(request()),
    ).resolves.toMatchObject({ status: "suspended" });
    expect(preSuspend.writer.writeTurnAsync).toHaveBeenCalledWith({
      message: "File a claim",
      resultMessages: preSuspendMessages,
      userId: "user-1",
      sessionId: "session-1",
      includeUserTurn: true,
    });

    // This new server has no process-local state from the first leg. The
    // resume flag alone must suppress the already-recorded user prompt.
    const replacement = makeServer();
    const resumedMessages = [
      { role: "assistant", content: "Approved after review" },
    ] as StreamingResult["messages"];
    replacement.server.executeViaAgentStream = vi.fn(async () => ({
      content: "Approved after review",
      messages: resumedMessages,
    }));

    await expect(
      replacement.server.doHandleExecute(request(true)),
    ).resolves.toMatchObject({ status: "completed" });
    expect(replacement.writer.writeTurnAsync).toHaveBeenCalledWith({
      message: "",
      resultMessages: resumedMessages,
      userId: "user-1",
      sessionId: "session-1",
      includeUserTurn: false,
    });
  });

  test("a queue failure does not replace a successful agent result", async () => {
    const { server, writer } = makeServer();
    writer.writeTurnAsync.mockImplementation(() => {
      throw new Error("queue unavailable");
    });
    server.executeViaAgentStream = vi.fn(async () => ({
      content: "Approved",
      messages: [{ role: "assistant", content: "Approved" }],
    }));

    await expect(server.doHandleExecute(request())).resolves.toEqual({
      status: "completed",
      result: "Approved",
    });
  });
});
