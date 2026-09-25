/**
 * Credential-rejection classification on the invoke_llm path.
 *
 * Mirrors Python's tests/unit/test_llm_credential_classification.py. When the
 * LLM provider rejects the configured key (401/403), the tool pod stamps
 * `error_code: "llm_credential_rejected"` on the failure so downstream
 * consumers (OE persist, gateway owner attribution, playground) can classify
 * it as a customer-secret problem without string-matching provider prose.
 * The predicate is deliberately status-only: anything unrecognized must keep
 * the existing generic classification.
 */

import { describe, test, expect, vi, afterEach } from "vitest";

import type { Message } from "@mongodb-js/agent-engine-sdk";
import {
  SecureLLMProxy,
  ToolServer,
  type ITenantRuntime,
  type LLMStreamChunk,
} from "../../src/index.js";
import { LLM_CREDENTIAL_REJECTED_ERROR_CODE } from "../../src/server/chunk_types.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { isLlmCredentialRejection } from "../../src/utils.js";

class StatusError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

class StatusCodeError extends Error {
  constructor(
    message: string,
    readonly status_code: number,
  ) {
    super(message);
  }
}

class ResponseError extends Error {
  response: { status: number };
  constructor(message: string, status: number) {
    super(message);
    this.response = { status };
  }
}

class CodeError extends Error {
  constructor(
    message: string,
    readonly code: number,
  ) {
    super(message);
  }
}

describe("isLlmCredentialRejection", () => {
  test.each([401, 403])("status attr %i", (status) => {
    expect(isLlmCredentialRejection(new StatusError("nope", status))).toBe(
      true,
    );
  });

  test.each([401, 403])("status_code attr %i", (status) => {
    expect(isLlmCredentialRejection(new StatusCodeError("nope", status))).toBe(
      true,
    );
  });

  test.each([401, 403])("response.status attr %i", (status) => {
    expect(isLlmCredentialRejection(new ResponseError("nope", status))).toBe(
      true,
    );
  });

  test.each([401, 403])("int code attr %i", (code) => {
    expect(isLlmCredentialRejection(new CodeError("nope", code))).toBe(true);
  });

  test("walks the cause chain", () => {
    const err = new Error("adapter failed", {
      cause: new StatusError("bad key", 401),
    });
    expect(isLlmCredentialRejection(err)).toBe(true);
  });

  test("a re-throw without an explicit cause stays generic", () => {
    // JS has no implicit __context__: only an explicit `cause` is walked.
    // Pins parity with Python's __cause__-only walk — an unrelated failure
    // raised while handling an auth error must not inherit the rejection.
    let thrown: unknown;
    try {
      try {
        throw new StatusError("bad key", 403);
      } catch {
        throw new Error("unannotated re-throw");
      }
    } catch (outer) {
      thrown = outer;
    }
    expect(isLlmCredentialRejection(thrown)).toBe(false);
  });

  test.each([400, 404, 429, 500, 503])("status %i stays generic", (status) => {
    expect(isLlmCredentialRejection(new StatusError("nope", status))).toBe(
      false,
    );
  });

  test("no status stays generic", () => {
    expect(isLlmCredentialRejection(new Error("plain crash"))).toBe(false);
  });

  test("non-Error values stay generic", () => {
    expect(isLlmCredentialRejection("401")).toBe(false);
    expect(isLlmCredentialRejection(null)).toBe(false);
    expect(isLlmCredentialRejection(undefined)).toBe(false);
  });

  test("a string code is provider prose, not a status — never matched", () => {
    const err = new Error("nope") as Error & { code: string };
    err.code = "UNAUTHENTICATED";
    expect(isLlmCredentialRejection(err)).toBe(false);
  });

  test("cyclic cause chain terminates", () => {
    const first = new Error("first");
    const second = new Error("second", { cause: first });
    (first as Error & { cause?: unknown }).cause = second;
    expect(isLlmCredentialRejection(first)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Tool pod emission
// ---------------------------------------------------------------------------

function makeToolServer(failure: unknown): {
  doHandleInvokeLlm: (
    request: unknown,
  ) => Promise<{ status: string; error_code?: string }>;
  handleInvokeLlmStream: (request: unknown) => AsyncGenerator<string>;
} {
  const runtime = {
    tools: {},
    toolDefinitions: {},
    graphBuilder: null,
    getAgentConfig: () => new RuntimeAgentConfig(),
  };
  const server = new ToolServer(runtime as unknown as ITenantRuntime);
  (
    server as unknown as {
      streamLlmChunks: () => AsyncGenerator<LLMStreamChunk>;
    }
  ).streamLlmChunks = async function* () {
    throw failure;
    yield undefined as unknown as LLMStreamChunk; // marks this an async generator
  };
  return server as unknown as {
    doHandleInvokeLlm: (
      request: unknown,
    ) => Promise<{ status: string; error_code?: string }>;
    handleInvokeLlmStream: (request: unknown) => AsyncGenerator<string>;
  };
}

const invokeRequest = {
  execution_id: "exec-llm-cred",
  arguments: { llm_id: "primary", messages: [] },
};

describe("tool pod error_code emission", () => {
  test("non-streaming /invoke_llm stamps the credential code", async () => {
    const server = makeToolServer(new StatusCodeError("bad key", 401));

    const response = await server.doHandleInvokeLlm(invokeRequest);

    expect(response.status).toBe("error");
    expect(response.error_code).toBe(LLM_CREDENTIAL_REJECTED_ERROR_CODE);
  });

  test("streaming /invoke_llm/stream stamps the credential code", async () => {
    const server = makeToolServer(new StatusError("forbidden", 403));

    const events: Array<Record<string, unknown>> = [];
    for await (const frame of server.handleInvokeLlmStream(invokeRequest)) {
      events.push(JSON.parse(frame.slice("data: ".length)));
    }

    expect(events).toHaveLength(1);
    expect(events[0]?.error_code).toBe(LLM_CREDENTIAL_REJECTED_ERROR_CODE);
  });

  test.each([false, true])(
    "a non-auth failure carries no code (streaming=%s)",
    async (streaming) => {
      const server = makeToolServer(new StatusError("slow down", 429));

      if (streaming) {
        const events: Array<Record<string, unknown>> = [];
        for await (const frame of server.handleInvokeLlmStream(invokeRequest)) {
          events.push(JSON.parse(frame.slice("data: ".length)));
        }
        expect(events).toHaveLength(1);
        expect(events[0]?.error).toBeTruthy();
        expect(events[0]).not.toHaveProperty("error_code");
      } else {
        const response = await server.doHandleInvokeLlm(invokeRequest);
        expect(response.status).toBe("error");
        expect(response.error_code).toBeUndefined();
      }
    },
  );
});

// ---------------------------------------------------------------------------
// SecureLLMProxy propagation
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sseResponse(events: unknown[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
        );
      }
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

const baseMessage: Message = { role: "user", content: "Hello" };

describe("SecureLLMProxy error_code propagation", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("a unary error response carries the code onto the raised error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "error",
          error: "provider rejected the key",
          error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toMatchObject({
      source: "llm",
      error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    });
  });

  test("an in-band SSE error event carries the code onto the raised error", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        return Promise.resolve(
          sseResponse([
            {
              error: "provider rejected the key",
              error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
            },
          ]),
        );
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toMatchObject({
      source: "llm",
      error_code: LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    });
  });

  test("an SSE error event without a code stays uncoded", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        return Promise.resolve(sseResponse([{ error: "model exploded" }]));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toMatchObject({ source: "llm", error_code: undefined });
  });
});
