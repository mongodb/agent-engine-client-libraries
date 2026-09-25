/**
 * Tests for HITL interrupt migration: interrupt() / Command(resume=...) flow.
 *
 * Mirrors Python's tests/unit/test_hitl_interrupt.py.
 *
 * TS-vs-Python adaptations:
 *   - Python `SuspendPayload.to_json()` → TS `suspendPayloadToJson(payload)`.
 *     SuspendPayload is a Zod-inferred plain object, not a class with methods.
 *   - Python `SuspendPayload.model_validate(data)` → TS `SuspendPayloadSchema.parse(data)`.
 *     The wire `__suspend__` field is stripped by the schema (not in shape).
 *   - Python `_process_result` / `_execute_via_agent_stream` are accessed via
 *     cast-through-unknown for the TS `private` visibility (compile-time only).
 *   - AERServer construction: Python `_make_aer_server()` builds via `__new__`
 *     and stubs runtime. TS uses `Object.create(AERServer.prototype)` for the
 *     same effect (skip ctor, stub the runtime field).
 *   - `RequestContext` is a plain object literal (Python is a class).
 *   - The `ValidationError` (Pydantic) Python checks for is Zod's `ZodError`
 *     in TS; we assert via `.rejects` / `.toThrow()`.
 *   - The Python `_MockRunResult` adapter wraps an async-iterable; the TS
 *     equivalent `FakeRunResult` implements both `PromiseLike` and
 *     `AsyncIterable<StreamEvent>` to satisfy `ExecutionResult`.
 *   - Python `hooks.reset_hooks()` autouse fixture → TS `afterEach(resetHooks)`.
 */

import { describe, test, expect, vi, afterEach } from "vitest";
import type {
  AgentInput,
  AgentOutput,
  ExecutionResult,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import {
  AERServer,
  SecureToolWrapper,
  SuspendPayloadSchema,
  suspendPayloadToJson,
  registerSuspendHandler,
  resetHooks,
  getCurrentWrapper,
  type ExecuteRequest,
  type InterruptResult,
  type StreamingResult,
} from "../../src/index.js";

// ---------------------------------------------------------------------------
// SuspendPayload tests
// ---------------------------------------------------------------------------

describe("SuspendPayload", () => {
  test("requires suspend_reason", () => {
    // test_requires_suspend_reason
    expect(() => SuspendPayloadSchema.parse({})).toThrow();
  });

  test("defaults suspend_context to empty dict", () => {
    // test_defaults_context_to_empty_dict
    const p = SuspendPayloadSchema.parse({ suspend_reason: "review" });
    expect(p.suspend_context).toEqual({});
  });

  test("to_json includes suspend flag", () => {
    // test_to_json_includes_suspend_flag
    const json = suspendPayloadToJson({
      suspend_reason: "awaiting_human_review",
      suspend_context: { claim_id: "C-123" },
    });
    const data = JSON.parse(json);
    expect(data["__suspend__"]).toBe(true);
    expect(data["suspend_reason"]).toBe("awaiting_human_review");
    expect(data["suspend_context"]).toEqual({ claim_id: "C-123" });
  });

  test("round trips through schema parse", () => {
    // test_round_trip_through_model_validate
    const original = {
      suspend_reason: "review",
      suspend_context: { task_id: "T-1" },
    };
    const data = JSON.parse(suspendPayloadToJson(original));
    const restored = SuspendPayloadSchema.parse(data);
    expect(restored.suspend_reason).toBe(original.suspend_reason);
    expect(restored.suspend_context).toEqual(original.suspend_context);
  });
});

// ---------------------------------------------------------------------------
// SecureToolWrapper.handleSuspend tests — fires the framework interrupt for an
// OE-confirmed suspend. Suspend provenance (only the author's suspend API can
// trigger it, never relayed tool-result content) is enforced at the tool-pod
// layer (tool.test.ts) and the AER status gate (secure_wrapper.test.ts).
// ---------------------------------------------------------------------------

interface WrapperWithPrivates {
  handleSuspend: (result: unknown) => unknown;
}

function makeWrapper(): WrapperWithPrivates {
  return new SecureToolWrapper(
    "http://oe:8000",
    "test-exec-123",
  ) as unknown as WrapperWithPrivates;
}

describe("SecureToolWrapper.handleSuspend with interrupt()", () => {
  afterEach(() => {
    resetHooks();
  });

  test("suspend calls interrupt with validated payload", () => {
    // test_suspend_calls_interrupt_with_validated_payload
    const wrapper = makeWrapper();
    const suspendResult = suspendPayloadToJson({
      suspend_reason: "awaiting_human_review",
      suspend_context: { task_id: "t1", claim_id: "c1" },
    });

    const mockInterrupt = vi.fn().mockReturnValue({ decision: "approved" });
    registerSuspendHandler(mockInterrupt);
    const result = wrapper.handleSuspend(suspendResult);

    expect(mockInterrupt).toHaveBeenCalledTimes(1);
    expect(mockInterrupt).toHaveBeenCalledWith({
      suspend_reason: "awaiting_human_review",
      suspend_context: { task_id: "t1", claim_id: "c1" },
    });
    expect(JSON.parse(result as string)).toEqual({ decision: "approved" });
  });

  test("suspend returns human decision as json", () => {
    // test_suspend_returns_human_decision_as_json
    const wrapper = makeWrapper();
    const suspendResult = suspendPayloadToJson({
      suspend_reason: "review",
      suspend_context: {},
    });

    const humanDecision = { decision: "rejected", notes: "insufficient docs" };
    registerSuspendHandler(vi.fn().mockReturnValue(humanDecision));
    const result = wrapper.handleSuspend(suspendResult);

    expect(JSON.parse(result as string)).toEqual(humanDecision);
  });

  test("suspend without reason raises validation error", () => {
    // test_suspend_without_reason_raises_validation_error
    const wrapper = makeWrapper();
    registerSuspendHandler(vi.fn());
    const badPayload = JSON.stringify({ __suspend__: true });

    expect(() => wrapper.handleSuspend(badPayload)).toThrow();
  });

  test("suspend without registered handler raises runtime error", () => {
    // test_suspend_without_registered_handler_raises_runtime_error
    const wrapper = makeWrapper();
    const suspendResult = suspendPayloadToJson({
      suspend_reason: "awaiting_human_review",
      suspend_context: { claim_id: "c1" },
    });

    expect(() => wrapper.handleSuspend(suspendResult)).toThrow(
      /No suspend handler registered/,
    );
  });

  test("suspend with a non-payload result raises ToolExecutionError", () => {
    // test_suspend_with_non_payload_result_raises
    const wrapper = makeWrapper();
    registerSuspendHandler(vi.fn());

    expect(() => wrapper.handleSuspend("plain text")).toThrow(
      /not a suspend payload/,
    );
  });
});

// ---------------------------------------------------------------------------
// Helpers for AERServer streaming tests (mirror Python helpers)
// ---------------------------------------------------------------------------

class FakeRunResult implements ExecutionResult {
  constructor(private readonly _events: StreamEvent[]) {}

  then<T1 = AgentOutput, T2 = never>(
    onfulfilled?: ((value: AgentOutput) => T1 | PromiseLike<T1>) | null,
    onrejected?: ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return Promise.resolve({ response: "done" } as AgentOutput).then(
      onfulfilled,
      onrejected,
    );
  }

  [Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
    const events = this._events;
    let i = 0;
    return {
      async next(): Promise<IteratorResult<StreamEvent>> {
        if (i < events.length) return { value: events[i++], done: false };
        return { value: undefined as unknown as StreamEvent, done: true };
      },
    };
  }
}

class FakeAgent {
  constructor(private readonly events: StreamEvent[]) {}
  execute(_ctx: RequestContext, _input: AgentInput): ExecutionResult {
    return new FakeRunResult(this.events);
  }
}

interface AerServerPrivates {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent?: (opts?: unknown) => FakeAgent;
  };
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback?: ReturnType<typeof vi.fn>;
  executeViaAgentStream: (
    agent: FakeAgent,
    ctx: RequestContext,
    agentInput: AgentInput,
    oeUrl: string,
    executionId: string,
  ) => Promise<InterruptResult | StreamingResult>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
}

function makeAerServer(): AerServerPrivates {
  const server = Object.create(AERServer.prototype) as AerServerPrivates;
  server.runtime = {};
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  return server;
}

function makeAgentInput(threadId = "thread-1"): AgentInput {
  return {
    payload: {
      message: "do something",
      user_id: "user-1",
      session_id: "sess-1",
      thread_id: threadId,
    },
  };
}

// ---------------------------------------------------------------------------
// TestExecuteViaAgentStream
// ---------------------------------------------------------------------------

describe("AERServer.executeViaAgentStream", () => {
  test("suspend returns InterruptResult", async () => {
    // test_suspend_returns_interrupt
    const server = makeAerServer();
    const suspendPayload = {
      suspend_reason: "awaiting_human_review",
      suspend_context: { claim_id: "CLM-100" },
    };
    const events: StreamEvent[] = [
      { event: "token", data: { content: "Processing" } },
      {
        event: "suspend",
        data: {
          suspend_payload: suspendPayload,
          metadata: { checkpoint_id: "ckpt-123" },
          messages: [{ role: "assistant", content: "Waiting for approval" }],
        },
      },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      {},
      makeAgentInput(),
      "http://oe:8000",
      "exec-suspend-001",
    );

    expect("suspend_payload" in result).toBe(true);
    const interrupt = result as InterruptResult;
    expect(interrupt.metadata).toEqual({ checkpoint_id: "ckpt-123" });
    expect(interrupt.suspend_payload).toEqual(suspendPayload);
    expect(interrupt.messages).toEqual([
      { role: "assistant", content: "Waiting for approval" },
    ]);
  });

  test("suspend without adapter metadata yields an empty metadata dict", async () => {
    // The adapter owns all resume state; the AER never synthesizes any.
    // A suspend event with no metadata round-trips as an empty dict.
    const server = makeAerServer();
    const suspendPayload = { suspend_reason: "review", suspend_context: {} };
    const events: StreamEvent[] = [
      { event: "suspend", data: { suspend_payload: suspendPayload } },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      { sessionId: "my-thread-id" },
      makeAgentInput(),
      "http://oe:8000",
      "exec-fallback-001",
    );

    const interrupt = result as InterruptResult;
    expect(interrupt.metadata).toEqual({});
    expect(interrupt.suspend_payload).toEqual(suspendPayload);
  });

  test("result event returns StreamingResult with messages", async () => {
    // test_result_returns_streaming_result_with_messages
    const server = makeAerServer();
    const platformMessages: Array<Record<string, string>> = [
      { role: "user", content: "File a claim" },
      { role: "assistant", content: "Processing your claim..." },
      { role: "tool", content: '{"status":"approved"}', tool_call_id: "tc1" },
      { role: "assistant", content: "Your claim CLM-100 has been approved!" },
    ];
    const events: StreamEvent[] = [
      {
        event: "result",
        data: {
          response: "Your claim CLM-100 has been approved!",
          messages: platformMessages,
        },
      },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      {},
      makeAgentInput(),
      "http://oe:8000",
      "exec-resume-001",
    );

    const streaming = result as StreamingResult;
    expect(streaming.messages).toHaveLength(4);
  });

  test("result content is preserved", async () => {
    // test_result_content_preserved
    const server = makeAerServer();
    const events: StreamEvent[] = [
      {
        event: "result",
        data: {
          response: "Claim approved — $14,500 payout.",
          messages: [
            { role: "assistant", content: "Claim approved — $14,500 payout." },
          ],
        },
      },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      {},
      makeAgentInput(),
      "http://oe:8000",
      "exec-resume-002",
    );

    const streaming = result as StreamingResult;
    expect(streaming.content).toContain("Claim approved");
    expect(streaming.content).toContain("$14,500");
  });

  test("no result or suspend raises RuntimeError", async () => {
    // test_no_result_raises_runtime_error
    const server = makeAerServer();
    const events: StreamEvent[] = [
      { event: "token", data: { content: "Processing..." } },
    ];

    await expect(
      server.executeViaAgentStream(
        new FakeAgent(events),
        {},
        makeAgentInput(),
        "http://oe:8000",
        "exec-empty-001",
      ),
    ).rejects.toThrow(/without result or suspend/);
  });

  test("thinking tokens are not streamed live", async () => {
    // test_thinking_tokens_not_streamed
    const server = makeAerServer();
    const events: StreamEvent[] = [
      { event: "token", data: { content: "<think>" } },
      { event: "token", data: { content: "internal reasoning" } },
      { event: "token", data: { content: "</think>" } },
      { event: "token", data: { content: "Hello!" } },
      {
        event: "result",
        data: {
          response: "<think>internal reasoning</think>Hello!",
          messages: [
            {
              role: "assistant",
              content: "<think>internal reasoning</think>Hello!",
            },
          ],
        },
      },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      {},
      makeAgentInput(),
      "http://oe:8000",
      "exec-think-001",
    );

    const streaming = result as StreamingResult;
    expect(streaming.content).toBe("Hello!");

    // _sendStreamChunk(oeUrl, executionId, chunkType, content, error, metadata)
    const streamedContents = server.sendStreamChunk.mock.calls.map(
      (args) => (args[3] as string) ?? "",
    );
    for (const content of streamedContents) {
      expect(content).not.toContain("<think>");
      expect(content).not.toContain("internal reasoning");
    }
  });

  test("thinking tokens stripped from final content", async () => {
    // test_thinking_tokens_stripped_from_final_content
    const server = makeAerServer();
    const events: StreamEvent[] = [
      {
        event: "result",
        data: {
          response: "<think>Let me think about this</think>The answer is 42.",
          messages: [
            {
              role: "assistant",
              content:
                "<think>Let me think about this</think>The answer is 42.",
            },
          ],
        },
      },
    ];

    const result = await server.executeViaAgentStream(
      new FakeAgent(events),
      {},
      makeAgentInput(),
      "http://oe:8000",
      "exec-think-002",
    );

    const streaming = result as StreamingResult;
    expect(streaming.content).toBe("The answer is 42.");
    expect(streaming.content).not.toContain("<think>");
  });
});

// ---------------------------------------------------------------------------
// TestResumeFromStep — wrapper.stepCounter advancement via _handleExecute
// ---------------------------------------------------------------------------

describe("AERServer resume_from_step advances wrapper.stepCounter", () => {
  test("resume_from_step sets wrapper.stepCounter", async () => {
    // test_resume_from_step_sets_wrapper_step_counter
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);

    const captured: { value?: number } = {};
    server.executeViaAgentStream = vi.fn(
      async (_agent, _ctx, _input, _oeUrl, _execId) => {
        const wrapper = getCurrentWrapper() as { stepCounter: number } | null;
        captured.value = wrapper?.stepCounter;
        return {
          content: "Resumed.",
          messages: [{ role: "assistant", content: "Resumed." }],
        } as unknown as StreamingResult;
      },
    ) as unknown as typeof server.executeViaAgentStream;

    const request: ExecuteRequest = {
      execution_id: "exec-resume-step-001",
      message: "Continue after suspend",
      platform_api_url: "http://oe:8000",
      resume: true,
      resume_from_step: 7,
      session_id: "session-resume-1",
    };

    const result = (await server.doHandleExecute(request)) as {
      status: string;
    };
    expect(result.status).toBe("completed");
    expect(captured.value).toBe(7);
  });

  test("fresh execution starts at step 0", async () => {
    // test_fresh_execution_starts_at_step_zero
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);

    const captured: { value?: number } = {};
    server.executeViaAgentStream = vi.fn(
      async (_agent, _ctx, _input, _oeUrl, _execId) => {
        const wrapper = getCurrentWrapper() as { stepCounter: number } | null;
        captured.value = wrapper?.stepCounter;
        return {
          content: "Fresh run.",
          messages: [{ role: "assistant", content: "Fresh run." }],
        } as unknown as StreamingResult;
      },
    ) as unknown as typeof server.executeViaAgentStream;

    const request: ExecuteRequest = {
      execution_id: "exec-fresh-001",
      message: "Hello",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "session-fresh-1",
    };

    const result = (await server.doHandleExecute(request)) as {
      status: string;
    };
    expect(result.status).toBe("completed");
    expect(captured.value).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// TestHandleExecuteMetadataResolution — payload preservation + ctx plumbing
// Mirrors Python tests/unit/test_aer_metadata_payload.py.
// ---------------------------------------------------------------------------

describe("AERServer payload/context contract", () => {
  interface Captured {
    ctx?: RequestContext;
    payload?: Record<string, unknown>;
  }

  function wireCapture(server: AerServerPrivates, captured: Captured): void {
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    server.executeViaAgentStream = vi.fn(async (_agent, ctx, input) => {
      captured.ctx = ctx;
      captured.payload = input.payload as Record<string, unknown>;
      return {
        content: "ok",
        messages: [],
      } as unknown as StreamingResult;
    }) as unknown as typeof server.executeViaAgentStream;
  }

  test("session_id used when present", async () => {
    // test_session_id_used_when_present
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-sess-prio",
      message: "hello",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "sess-from-oe",
    });
    expect(captured.ctx?.sessionId).toBe("sess-from-oe");
  });

  test("empty-string session_id falls through to execution_id", async () => {
    // Mirror Python's `or`: a zero-value "" session_id must not become the
    // session key (which would collapse unrelated executions).
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-empty-sess",
      message: "hello",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "",
    });
    expect(captured.ctx?.sessionId).toBe("exec-empty-sess");
  });

  test("session_id falls back to execution_id when absent", async () => {
    // test_falls_back_to_execution_id_when_no_session_id
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-no-session",
      message: "hello",
      platform_api_url: "http://oe:8000",
      resume: false,
    });
    expect(captured.ctx?.sessionId).toBe("exec-no-session");
  });

  test("message is read from payload when top-level is absent", async () => {
    // test_reads_message_from_payload_when_top_level_absent
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-payload-msg",
      message: "",
      platform_api_url: "http://oe:8000",
      resume: false,
      payload: { message: "from-payload" },
    });
    expect(captured.payload?.["message"]).toBe("from-payload");
  });

  test("message falls back to top-level when payload absent", async () => {
    // test_falls_back_to_top_level_message_when_payload_absent
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-msg-fallback",
      message: "top-level-msg",
      platform_api_url: "http://oe:8000",
      resume: false,
    });
    expect(captured.payload?.["message"]).toBe("top-level-msg");
  });

  test("opaque caller payload is preserved apart from the normalized message", async () => {
    // Python preserves dict(request.payload) and only normalizes `message`.
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-payload-preserve",
      message: "hi",
      platform_api_url: "http://oe:8000",
      resume: false,
      payload: { context: "from-payload", nested: { a: 1 }, count: 42 },
    });
    // Caller fields survive untouched; message is normalized in. Platform
    // plumbing (user_id/session_id/thread_id/resume/checkpoint_id) must NOT
    // leak into the payload — it travels on ctx.
    expect(captured.payload).toEqual({
      context: "from-payload",
      nested: { a: 1 },
      count: 42,
      message: "hi",
    });
  });

  test("resume and checkpoint metadata travel on ctx, not payload", async () => {
    // test_reads_checkpoint_id_from_metadata_on_resume
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-ckpt-meta",
      message: "resume msg",
      platform_api_url: "http://oe:8000",
      resume: true,
      resume_data: { decision: "approve" },
      metadata: { checkpoint_id: "meta-ckpt-1", thread_id: "t-1" },
    });
    expect(captured.ctx?.resume).toBe(true);
    expect(captured.ctx?.resumeData).toEqual({ decision: "approve" });
    expect(captured.ctx?.metadata).toEqual({
      checkpoint_id: "meta-ckpt-1",
      thread_id: "t-1",
    });
    // checkpoint_id is opaque framework state — it must not be in the payload.
    expect(captured.payload?.["checkpoint_id"]).toBeUndefined();
  });

  test("legacy scalar resume messages are ignored", async () => {
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-resume-message",
      message: "",
      platform_api_url: "http://oe:8000",
      resume: true,
      resume_message: "approve",
    });
    expect(captured.ctx?.resumeData).toBeNull();
  });

  test("structured resume data reaches the framework adapter", async () => {
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    await server.doHandleExecute({
      execution_id: "exec-resume-precedence",
      message: "",
      platform_api_url: "http://oe:8000",
      resume: true,
      resume_data: { decision: "approve" },
    });
    expect(captured.ctx?.resumeData).toEqual({ decision: "approve" });
  });

  test("rejects an ambiguous message (top-level and payload) with 400", async () => {
    // test_conflicting_message_top_level_and_payload_rejected — fail fast,
    // before any side effects, and never echo the payload in the error.
    const server = makeAerServer();
    const captured: Captured = {};
    wireCapture(server, captured);
    let thrown: (Error & { statusCode?: number }) | undefined;
    try {
      await server.doHandleExecute({
        execution_id: "exec-conflict",
        message: "top-level",
        platform_api_url: "http://oe:8000",
        resume: false,
        payload: { message: "from-payload" },
      });
    } catch (e) {
      thrown = e as Error & { statusCode?: number };
    }
    expect(thrown?.statusCode).toBe(400);
    expect(thrown?.message).toContain("Ambiguous request");
    // The opaque payload must never appear in the surfaced error.
    expect(thrown?.message).not.toContain("from-payload");
    // The agent never ran — rejection happens before side effects.
    expect(captured.ctx).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Callback metadata round-trip (HITL resume)
// ---------------------------------------------------------------------------

describe("AERServer callback metadata", () => {
  test("legacy interrupt envelope forwards suspend payload", async () => {
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    const interrupts = [
      {
        id: "int-1",
        value: {
          suspend_reason: "awaiting_human_review",
          suspend_context: { allowed_decisions: ["approve", "reject"] },
        },
      },
    ];
    const resumeSchema = { type: "object", required: ["message"] };

    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: interrupts[0]?.value,
        interrupts,
        resume_schema: resumeSchema,
        metadata: { checkpoint_id: "ckpt-legacy" },
      } as unknown as InterruptResult;
    }) as unknown as typeof server.executeViaAgentStream;

    await server.doHandleExecute({
      execution_id: "exec-legacy-envelope",
      message: "review this",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "session-suspend-1",
    });

    const call = server.reportCallback?.mock.calls.find(
      (candidate) => candidate[2] === "SUSPENDED",
    );
    expect(call?.[3]).toMatchObject({
      suspend_reason: "awaiting_human_review",
      suspend_context: { allowed_decisions: ["approve", "reject"] },
      interrupts,
      resume_schema: resumeSchema,
    });
  });

  test("non-legacy interrupt envelope uses generic suspend reason", async () => {
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    const interrupts = [{ id: "int-1", value: { question: "Approve?" } }];
    const resumeSchema = { type: "object", required: ["message"] };

    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: interrupts[0]?.value,
        interrupts,
        resume_schema: resumeSchema,
        metadata: { checkpoint_id: "ckpt-nonlegacy" },
      } as unknown as InterruptResult;
    }) as unknown as typeof server.executeViaAgentStream;

    await server.doHandleExecute({
      execution_id: "exec-nonlegacy-envelope",
      message: "review this",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "session-suspend-1",
    });

    const call = server.reportCallback?.mock.calls.find(
      (candidate) => candidate[2] === "SUSPENDED",
    );
    expect(call?.[3]).toMatchObject({
      suspend_reason: "agent_interrupt",
      interrupts,
      resume_schema: resumeSchema,
    });
    expect(call?.[3]?.suspend_context).toBeUndefined();
  });

  test("multi-interrupt envelope and metadata are forwarded unchanged", async () => {
    // Parity: Python forwards the adapter's opaque metadata on SUSPENDED so
    // OE can persist opaque executor state and replay it on resume. The AER
    // never adds fields of its own — checkpoint_id only exists inside the
    // adapter-owned metadata dict, never top-level.
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);

    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: {
          suspend_reason: "awaiting_human_review",
          suspend_context: { claim_id: "CLM-1" },
        },
        interrupts: [
          { id: "int-1", value: { question: "Approve?" } },
          { id: "int-2", value: ["a", "b"] },
        ],
        resume_schema: {
          type: "object",
          required: ["resume_map"],
        },
        metadata: { checkpoint_id: "ckpt-xyz" },
      } as unknown as InterruptResult;
    }) as unknown as typeof server.executeViaAgentStream;

    const request: ExecuteRequest = {
      execution_id: "exec-suspend-001",
      message: "review this",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "session-suspend-1",
    };

    const result = (await server.doHandleExecute(request)) as {
      status: string;
      checkpoint_id?: string;
    };
    expect(result.status).toBe("suspended");
    // The suspended HTTP response no longer surfaces a checkpoint_id.
    expect(result.checkpoint_id).toBeUndefined();

    const call = server.reportCallback?.mock.calls.find(
      (c) => c[2] === "SUSPENDED",
    );
    expect(call).toBeDefined();
    expect(call?.[3]).toMatchObject({
      suspend_reason: "agent_interrupt",
      interrupts: [
        { id: "int-1", value: { question: "Approve?" } },
        { id: "int-2", value: ["a", "b"] },
      ],
      resume_schema: {
        type: "object",
        required: ["resume_map"],
      },
      metadata: { checkpoint_id: "ckpt-xyz" },
    });
    expect(call?.[3]?.suspend_context).toBeUndefined();
    expect(
      (call?.[3] as Record<string, unknown>)["checkpoint_id"],
    ).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Adapter validation error reporting
// ---------------------------------------------------------------------------

describe("AERServer adapter validation errors", () => {
  test("non-serializable interrupt errors report a terminal callback", async () => {
    // Python: test_non_serializable_interrupt_error_reports_terminal_callback.
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    const errorMessage =
      'LangGraph interrupt "approval" has a non-JSON-serializable value of type bigint';
    server.executeViaAgentStream = vi
      .fn()
      .mockRejectedValue(new Error(errorMessage));

    await expect(
      server.doHandleExecute({
        execution_id: "exec-non-serializable",
        message: "review this",
        platform_api_url: "http://oe:8000",
        suspend_generation: 5,
        resume: false,
        session_id: "session-suspend-1",
      }),
    ).rejects.toMatchObject({ message: errorMessage, statusCode: 500 });

    expect(server.reportCallback).toHaveBeenCalledOnce();
    expect(server.reportCallback).toHaveBeenCalledWith(
      "http://oe:8000",
      "exec-non-serializable",
      "ERROR",
      { error: errorMessage, suspend_generation: 5 },
      {
        execution_id: "exec-non-serializable",
        status: "ERROR",
        suspend_generation: 5,
        error: errorMessage,
      },
      { failed: false },
      null,
    );
  });
});

// ---------------------------------------------------------------------------
// Execution-timeout error reporting (single terminal callback)
// ---------------------------------------------------------------------------

describe("AERServer empty interrupt snapshot reports ERROR before failing", () => {
  test("empty interrupts envelope reports exactly one ERROR callback and rejects with 500", async () => {
    // Report ERROR before propagating the 500, or OE leaves the execution
    // running indefinitely.
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    server.executeViaAgentStream = vi.fn(async () => {
      return {
        suspend_payload: {},
        interrupts: [],
        resume_schema: { type: "object" },
      } as unknown as InterruptResult;
    }) as unknown as typeof server.executeViaAgentStream;

    const request: ExecuteRequest = {
      execution_id: "exec-empty-snapshot",
      message: "hello",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "sess-empty-snapshot",
    };

    await expect(server.doHandleExecute(request)).rejects.toMatchObject({
      statusCode: 500,
      message: expect.stringContaining("empty interrupt snapshot"),
    });

    const reportCallback = server.reportCallback;
    expect(reportCallback).toHaveBeenCalledTimes(1);
    expect(reportCallback?.mock.calls[0]?.[2]).toBe("ERROR");
    const fields = reportCallback?.mock.calls[0]?.[3] as { error?: string };
    expect(String(fields?.error ?? "")).toContain("empty interrupt snapshot");
  });
});

describe("AERServer execution timeout reports exactly one ERROR callback", () => {
  test("timeout does not double-report the ERROR callback", async () => {
    // The execution-timeout branch sends the terminal ERROR chunk + callback,
    // then throws a 504. The outer catch must NOT report a second time
    // (mirrors Python's `except HTTPException: raise`).
    const server = makeAerServer();
    server.runtime.graphBuilder = {};
    server.runtime.orgId = null;
    server.runtime.getAgent = () => new FakeAgent([]);
    server.reportCallback = vi.fn().mockResolvedValue(undefined);
    server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);

    // Simulate a hanging agent that only unwinds when the timeout aborts it.
    server.executeViaAgentStream = vi.fn((_agent, ctx: RequestContext) => {
      return new Promise<never>((_resolve, reject) => {
        ctx.signal?.addEventListener(
          "abort",
          () => {
            const err = new Error("aborted") as Error & { name: string };
            err.name = "AbortError";
            reject(err);
          },
          { once: true },
        );
      });
    }) as unknown as typeof server.executeViaAgentStream;

    const request: ExecuteRequest = {
      execution_id: "exec-timeout-001",
      message: "slow request",
      platform_api_url: "http://oe:8000",
      resume: false,
      session_id: "session-timeout-1",
    };

    // Mirror the module's timeout computation so the test is robust to env.
    const timeoutMs =
      (Number(process.env.RUNNER_EXECUTION_TIMEOUT) || 600) * 1000;

    vi.useFakeTimers();
    try {
      const p = server.doHandleExecute(request);
      // Surface the eventual rejection without an unhandled-rejection warning.
      const settled = p.then(
        () => ({ ok: true as const }),
        (e: unknown) => ({ ok: false as const, error: e }),
      );
      await vi.advanceTimersByTimeAsync(timeoutMs + 1000);
      const outcome = await settled;
      expect(outcome.ok).toBe(false);
    } finally {
      vi.useRealTimers();
    }

    // Exactly one terminal ERROR callback and one ERROR stream chunk.
    const reportCallback = server.reportCallback;
    expect(reportCallback).toHaveBeenCalledTimes(1);
    expect(reportCallback?.mock.calls[0]?.[2]).toBe("ERROR");
    // sendStreamChunk's 3rd positional arg is the ERROR chunk-type constant
    // (value "error"); reportCallback's 3rd arg is the "ERROR" status string.
    const errorChunkCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === "error",
    );
    expect(errorChunkCalls).toHaveLength(1);
  });
});
