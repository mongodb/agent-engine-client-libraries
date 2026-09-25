/**
 * Tests for AER chunk emission (quick-260421-fj2).
 *
 * Mirrors Python's tests/unit/test_aer_chunk_emission.py.
 *
 * Focus areas:
 *   1. `metadata.source` is forwarded on every text chunk.
 *   2. The thinking-token filter keeps separate state per (source, tool_call_id).
 *   3. `tool_call_id` reaches metadata so UI can route per-tool-call-id.
 *
 * TS-vs-Python adaptations:
 *   - AERServer construction: Python `__new__` → TS `Object.create(prototype)`.
 *     Same intent — skip the real constructor.
 *   - `_sendStreamChunk` is private; tests override it via cast-through-unknown
 *     with a `vi.fn()` spy. Python uses `MagicMock()` direct attribute assign.
 *   - `_sendStreamChunk` signature: TS uses positional args
 *     `(oeUrl, executionId, chunkType, content, error, metadata)`. Python uses
 *     kwargs. Tests extract args by index.
 *   - `BaseAgent.execute` returns `ExecutionResult` (PromiseLike + AsyncIterable).
 *     `FakeAgent` exposes both; only the AsyncIterable is exercised here.
 */

import { describe, test, expect, vi } from "vitest";
import type {
  AgentInput,
  AgentOutput,
  ExecutionResult,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import { AERServer } from "../../src/index.js";

// ---------------------------------------------------------------------------
// FakeAgent — minimal BaseAgent-shaped stand-in
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

// ---------------------------------------------------------------------------
// AERServer harness — executeViaAgentStream with _sendStreamChunk spied
// ---------------------------------------------------------------------------

interface ServerWithPrivates {
  sendStreamChunk: ReturnType<typeof vi.fn>;
  executeViaAgentStream: (
    agent: FakeAgent,
    ctx: RequestContext,
    agentInput: AgentInput,
    oeUrl: string,
    executionId: string,
  ) => Promise<unknown>;
}

function makeServer(): {
  server: ServerWithPrivates;
  sendSpy: ReturnType<typeof vi.fn>;
} {
  const server = Object.create(AERServer.prototype) as ServerWithPrivates;
  const sendSpy = vi.fn().mockResolvedValue(undefined);
  server.sendStreamChunk = sendSpy;
  return { server, sendSpy };
}

function agentInput(): AgentInput {
  return {
    payload: {
      message: "hi",
      thread_id: "t-1",
      session_id: "t-1",
      resume: false,
    },
  };
}

function ctx(executionId: string): RequestContext {
  return {
    executionId: executionId,
    userId: "test-user",
    requestHeaders: {},
  };
}

/**
 * Extract (chunkType, content, metadata) from a `_sendStreamChunk` call.
 * Signature: (oeUrl, executionId, chunkType, content, error, metadata).
 */
interface SendCall {
  chunkType: string;
  content: string;
  metadata: Record<string, string>;
}
function capture(spy: ReturnType<typeof vi.fn>): SendCall[] {
  return spy.mock.calls.map((args) => ({
    chunkType: args[2] as string,
    content: (args[3] as string) ?? "",
    metadata: (args[5] as Record<string, string>) ?? {},
  }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AER chunk emission", () => {
  test("text chunks carry metadata.source attribution", async () => {
    // test_text_chunks_carry_metadata_source_attribution
    const events: StreamEvent[] = [
      { event: "token", data: { content: "root says ", source: "" } },
      { event: "token", data: { content: "hi ", source: "" } },
      { event: "token", data: { content: "sub says ok", source: "sec" } },
      {
        event: "result",
        data: { response: "done", messages: [], resumed: false },
      },
    ];
    const { server, sendSpy } = makeServer();

    await server.executeViaAgentStream(
      new FakeAgent(events),
      ctx("exec-1"),
      agentInput(),
      "http://oe",
      "exec-1",
    );

    const textCalls = capture(sendSpy).filter((c) => c.chunkType === "text");
    expect(textCalls).toHaveLength(3);
    expect(textCalls.map((c) => c.metadata["source"])).toEqual(["", "", "sec"]);
    expect(textCalls.map((c) => c.content)).toEqual([
      "root says ",
      "hi ",
      "sub says ok",
    ]);
  });

  test("per-source thinking filter does not leak state across sources", async () => {
    // test_per_source_thinking_filter_does_not_leak_state
    // Root opens <think> (no close) → buffers in root's slot.
    // Subagent streams <think>a</think>foo → only 'foo' emitted from sub's slot.
    // Subagent streams bar → passes through.
    // Root emits </think>after → closes root's think, 'after' emitted.
    const events: StreamEvent[] = [
      { event: "token", data: { content: "<think>internal", source: "" } },
      {
        event: "token",
        data: { content: "<think>a</think>foo", source: "sub" },
      },
      { event: "token", data: { content: "bar", source: "sub" } },
      { event: "token", data: { content: "</think>after", source: "" } },
      {
        event: "result",
        data: { response: "done", messages: [], resumed: false },
      },
    ];
    const { server, sendSpy } = makeServer();

    await server.executeViaAgentStream(
      new FakeAgent(events),
      ctx("exec-2"),
      agentInput(),
      "http://oe",
      "exec-2",
    );

    const textCalls = capture(sendSpy).filter((c) => c.chunkType === "text");
    const subTexts = textCalls
      .filter((c) => c.metadata["source"] === "sub")
      .map((c) => c.content);
    const rootTexts = textCalls
      .filter((c) => c.metadata["source"] === "")
      .map((c) => c.content);

    expect(subTexts.join("")).toBe("foobar");
    expect(subTexts.join("")).not.toContain("<think>");
    expect(rootTexts.join("")).toContain("after");
    expect(rootTexts.join("")).not.toContain("<think>");
  });

  test("text chunks carry tool_call_id metadata", async () => {
    // test_text_chunks_carry_tool_call_id_metadata
    const events: StreamEvent[] = [
      {
        event: "token",
        data: { content: "root", source: "", tool_call_id: "" },
      },
      {
        event: "token",
        data: { content: "sub-a", source: "research", tool_call_id: "call-a" },
      },
      {
        event: "result",
        data: { response: "done", messages: [], resumed: false },
      },
    ];
    const { server, sendSpy } = makeServer();

    await server.executeViaAgentStream(
      new FakeAgent(events),
      ctx("exec-tcid"),
      agentInput(),
      "http://oe",
      "exec-tcid",
    );

    const textCalls = capture(sendSpy).filter((c) => c.chunkType === "text");
    expect(textCalls.map((c) => c.metadata["tool_call_id"])).toEqual([
      "",
      "call-a",
    ]);
  });
});
