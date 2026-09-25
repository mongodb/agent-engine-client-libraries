/**
 * AER surfaces a deadline breach as a timeout, not a generic failure.
 *
 * Mirrors Python's tests/unit/test_aer_timeout_signal.py. A tool that overruns
 * is neither a crash nor a policy denial: the work was permitted and ran, it
 * just ran too long. Consumers need to tell those apart to offer "give it more
 * time / retry" instead of surfacing a bug, so the AER tags the ERROR chunk
 * with metadata.error_code === "timeout" and answers 504 rather than 500.
 */

import { describe, test, expect, vi } from "vitest";

import {
  AERServer,
  ToolCallTimeoutError,
  type ExecuteRequest,
} from "../../src/index.js";
import {
  POLICY_DENIED_ERROR_CODE,
  TIMEOUT_ERROR_CODE,
} from "../../src/server/chunk_types.js";

interface AerPrivates {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent?: (opts?: unknown) => unknown;
  };
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
}

function makeServer(): AerPrivates {
  const server = Object.create(AERServer.prototype) as AerPrivates;
  server.runtime = { graphBuilder: {}, orgId: null, getAgent: () => ({}) };
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  return server;
}

function makeRequest(executionId: string): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: "http://oe:8000",
  } as ExecuteRequest;
}

describe("AER tool-timeout structured signal", () => {
  test("a timeout emits ERROR chunk + callback tagged timeout, and 504", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new ToolCallTimeoutError("slow_tool", 600, 601.4);
    }) as unknown as typeof server.executeViaAgentStream;

    const err = (await server
      .doHandleExecute(makeRequest("exec-tool-timeout"))
      .catch((e: unknown) => e)) as Error & { statusCode?: number };

    // 504, not 500: the caller can reasonably retry or allow more time.
    expect(err.statusCode).toBe(504);
    expect(err.message).toContain("slow_tool");

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(1);
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).toBe(TIMEOUT_ERROR_CODE);
    expect(meta.tool_name).toBe("slow_tool");
    expect(meta.elapsed_seconds).toBeCloseTo(601.4);

    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(TIMEOUT_ERROR_CODE);
  });

  test("a timeout is not tagged as a policy denial", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new ToolCallTimeoutError("slow_tool", 600, 601);
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-not-denied"))
      .catch(() => {});

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).not.toBe(POLICY_DENIED_ERROR_CODE);
  });
});
