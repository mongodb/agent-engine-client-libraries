/**
 * AER surfaces OE policy-engine denials as a structured signal.
 *
 * Mirrors Python's tests/unit/test_aer_policy_denied.py. When the OE blocks a
 * tool or LLM call, the agent stream throws PolicyDeniedException; the AER must
 * turn that into a structured signal — an ERROR chunk carrying
 * metadata.error_code === "policy_denied" plus the reason — so streaming
 * consumers can render a "blocked by policy" outcome distinctly instead of
 * string-matching a generic error message. The same metadata is attached to the
 * terminal callback (forward-compatible; see the handler comment for the OE
 * terminal-metadata caveat).
 *
 * TS-vs-Python: AERServer is built via Object.create(prototype) (skip ctor);
 * private methods/fields are reached via cast-through-unknown. doHandleExecute
 * is the TS counterpart of Python's _handle_execute.
 */

import { describe, test, expect, vi } from "vitest";

import {
  AERServer,
  PolicyDeniedException,
  type ExecuteRequest,
} from "../../src/index.js";
import { POLICY_DENIED_ERROR_CODE } from "../../src/server/chunk_types.js";

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
  // executeViaAgentStream is stubbed per-test, so getAgent's return is unused.
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

describe("AER policy-denied structured signal", () => {
  test("denial emits ERROR chunk + callback tagged policy_denied with reason", async () => {
    const server = makeServer();
    const reason =
      'AUTHORIZED_MODELS policy denied model "gpt-5" at generation 5';
    server.executeViaAgentStream = vi.fn(async () => {
      throw new PolicyDeniedException(reason);
    }) as unknown as typeof server.executeViaAgentStream;

    // The denial still rejects (terminal error) — the improvement is the
    // structured metadata, asserted below.
    await expect(
      server.doHandleExecute(makeRequest("exec-policy-denied")),
    ).rejects.toThrow();

    // ERROR chunk: sendStreamChunk(oeUrl, execId, chunkType, content, error, metadata)
    const errCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errCalls).toHaveLength(1);
    const meta = errCalls[0][5] as Record<string, string>;
    // The human-readable error field keeps the full exception message
    // ("Policy denied: <reason>") for backward compatibility; the bare reason
    // rides in metadata.
    expect(errCalls[0][4]).toBe(`Policy denied: ${reason}`);
    expect(meta.error_code).toBe(POLICY_DENIED_ERROR_CODE);
    expect(meta.reason).toBe(reason);

    // Terminal callback: reportCallback(oeUrl, execId, status, fields)
    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const cb = server.reportCallback.mock.calls[0];
    expect(cb[2]).toBe("ERROR");
    const cbMeta = (cb[3] as { metadata?: Record<string, string> }).metadata;
    expect(cbMeta?.error_code).toBe(POLICY_DENIED_ERROR_CODE);
    expect(cbMeta?.reason).toBe(reason);
  });

  test("generic (non-policy) error is not tagged policy_denied", async () => {
    const server = makeServer();
    server.executeViaAgentStream = vi.fn(async () => {
      throw new Error("tool pod crashed");
    }) as unknown as typeof server.executeViaAgentStream;

    await expect(
      server.doHandleExecute(makeRequest("exec-generic-error")),
    ).rejects.toThrow();

    const cb = server.reportCallback.mock.calls[0];
    expect(cb[2]).toBe("ERROR");
    const cbMeta = (cb[3] as { metadata?: Record<string, string> }).metadata;
    expect(cbMeta?.error_code).not.toBe(POLICY_DENIED_ERROR_CODE);
  });
});
