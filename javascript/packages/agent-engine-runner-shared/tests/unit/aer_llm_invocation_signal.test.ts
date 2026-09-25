/**
 * AER surfaces an LLM-provider failure with structured owner metadata.
 *
 * Mirrors Python's tests/unit/test_aer_llm_invocation_signal.py. A 401/404
 * from the tenant's LLM after OE approved the call is not a platform crash.
 * Consumers classify it without string-matching the error prose, so the AER
 * tags the ERROR chunk with metadata.error_code === "llm_invocation_failed"
 * and source === "llm" only when LLMInvocationError.source is provider-owned.
 */

import { describe, test, expect, vi } from "vitest";

import {
  AERServer,
  ExternalAPICallError,
  LLMInvocationError,
  type ExecuteRequest,
} from "../../src/index.js";
import {
  LLM_CREDENTIAL_REJECTED_ERROR_CODE,
  LLM_INVOCATION_ERROR_CODE,
  LLM_INVOCATION_ERROR_SOURCE,
  POLICY_DENIED_ERROR_CODE,
  TIMEOUT_ERROR_CODE,
  TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
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

describe("AER LLM-invocation structured signal", () => {
  test("an LLM failure emits ERROR chunk + callback tagged llm, and 500", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new LLMInvocationError("Resource not found", {
        source: LLM_INVOCATION_ERROR_SOURCE,
      });
    }) as unknown as typeof server.executeViaAgentStream;

    const err = (await server
      .doHandleExecute(makeRequest("exec-llm-404"))
      .catch((e: unknown) => e)) as Error & { statusCode?: number };

    expect(err.statusCode).toBe(500);
    expect(err.message).toContain("Resource not found");

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(1);
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).toBe(LLM_INVOCATION_ERROR_CODE);
    expect(meta.code).toBe(LLM_INVOCATION_ERROR_CODE);
    expect(meta.source).toBe(LLM_INVOCATION_ERROR_SOURCE);

    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(LLM_INVOCATION_ERROR_CODE);
    expect(callbackArgs.metadata?.source).toBe(LLM_INVOCATION_ERROR_SOURCE);
  });

  test("an LLM failure is not tagged as a timeout or policy denial", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new LLMInvocationError("missing subscription key", {
        source: LLM_INVOCATION_ERROR_SOURCE,
      });
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-llm-not-other"))
      .catch(() => {});

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).not.toBe(TIMEOUT_ERROR_CODE);
    expect(meta.error_code).not.toBe(POLICY_DENIED_ERROR_CODE);
  });

  test("an uncoded LLMInvocationError does not stamp provider owner", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new LLMInvocationError(
        "SSE stream ended without done signal (truncated response)",
      );
    }) as unknown as typeof server.executeViaAgentStream;

    const err = (await server
      .doHandleExecute(makeRequest("exec-llm-truncated"))
      .catch((e: unknown) => e)) as Error & { statusCode?: number };

    expect(err.statusCode).toBe(500);

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(0);

    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).not.toBe(
      LLM_INVOCATION_ERROR_CODE,
    );
    expect(callbackArgs.metadata?.source).not.toBe(LLM_INVOCATION_ERROR_SOURCE);
  });

  test("a pod-stamped credential code replaces the generic code and drops source", async () => {
    // `source` must NOT ride along: the gateway checks source first, and
    // `llm` attribution would read as provider flakiness when the fix is the
    // customer's own project secret.
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new LLMInvocationError("provider rejected the configured API key", {
        source: LLM_INVOCATION_ERROR_SOURCE,
        error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
      });
    }) as unknown as typeof server.executeViaAgentStream;

    await server.doHandleExecute(makeRequest("exec-llm-401")).catch(() => {});

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(1);
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).toBe(LLM_CREDENTIAL_REJECTED_ERROR_CODE);
    expect(meta.code).toBe(LLM_CREDENTIAL_REJECTED_ERROR_CODE);
    expect(meta.source).toBeUndefined();

    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(
      LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    );
    expect(callbackArgs.metadata?.source).toBeUndefined();
  });

  function toolAuthFailure(): ExternalAPICallError {
    return new ExternalAPICallError(
      "OpenWeather API call failed: HTTP 401 AUTH_FAILED",
      { classification: "AUTH_FAILED", http_status: 401, retryable: false },
    );
  }

  test("a tool credential rejection is tagged on the terminal callback", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw toolAuthFailure();
    }) as unknown as typeof server.executeViaAgentStream;

    await server.doHandleExecute(makeRequest("exec-tool-401")).catch(() => {});

    // The generic catch reports only through the terminal callback.
    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(0);

    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(
      TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
    );
    expect(callbackArgs.metadata?.code).toBe(
      TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
    );
    expect(callbackArgs.metadata?.source).toBeUndefined();
  });

  test("a framework-wrapped tool credential rejection still classifies", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new Error("graph node failed", { cause: toolAuthFailure() });
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-tool-401-wrapped"))
      .catch(() => {});

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(
      TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
    );
  });

  test("a framework-wrapped credential-coded LLM error keeps the pod's code", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new Error("graph node failed", {
        cause: new LLMInvocationError(
          "provider rejected the configured API key",
          {
            source: LLM_INVOCATION_ERROR_SOURCE,
            error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
          },
        ),
      });
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-llm-401-wrapped"))
      .catch(() => {});

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(
      LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    );
  });

  test("a plain crash stays uncoded", async () => {
    // Fail-open: a plain crash must not be relabeled as a secret problem.
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new Error("plain agent crash");
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-plain-crash"))
      .catch(() => {});

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata).toBeUndefined();
  });

  test("an LLMInvocationError with an unrecognized code falls back to generic", async () => {
    // The AER honors only the pod's credential stamp: the exception type is
    // agent-raisable and the live relay forwards pod frames verbatim, so an
    // unrecognized error_code degrades to llm_invocation_failed — matching
    // the unary and replay paths, where the OE's closed-set sanitizer drops
    // it.
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new LLMInvocationError("model exploded", {
        source: LLM_INVOCATION_ERROR_SOURCE,
        error_code: "platform_outage",
      });
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-llm-unknown-code"))
      .catch(() => {});

    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(1);
    const meta = errCalls[0]?.[5] as Record<string, unknown>;
    expect(meta.error_code).toBe(LLM_INVOCATION_ERROR_CODE);
    expect(meta.source).toBe(LLM_INVOCATION_ERROR_SOURCE);

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata?.error_code).toBe(LLM_INVOCATION_ERROR_CODE);
    expect(callbackArgs.metadata?.source).toBe(LLM_INVOCATION_ERROR_SOURCE);
  });

  test("a framework-wrapped LLM error with a forged code stays uncoded", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new Error("graph node failed", {
        cause: new LLMInvocationError("model exploded", {
          source: LLM_INVOCATION_ERROR_SOURCE,
          error_code: TIMEOUT_ERROR_CODE,
        }),
      });
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-llm-forged-code-wrapped"))
      .catch(() => {});

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata).toBeUndefined();
  });

  test("an unrelated error raised while handling a tool auth failure stays uncoded", async () => {
    // Parity with Python's __cause__-only walk: a bare re-throw inside a
    // catch block attaches no `cause`, so the handled AUTH_FAILED error must
    // not relabel what is really a code bug.
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      try {
        throw toolAuthFailure();
      } catch {
        throw new Error("cache bug");
      }
    }) as unknown as typeof server.executeViaAgentStream;

    await server
      .doHandleExecute(makeRequest("exec-tool-context-bug"))
      .catch(() => {});

    const callbackArgs = server.reportCallback.mock.calls[0]?.[3] as {
      metadata?: Record<string, unknown>;
    };
    expect(callbackArgs.metadata).toBeUndefined();
  });
});
