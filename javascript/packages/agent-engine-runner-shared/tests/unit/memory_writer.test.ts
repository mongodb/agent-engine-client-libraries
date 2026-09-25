import { afterEach, describe, expect, test, vi } from "vitest";

import type { Message } from "@mongodb-js/agent-engine-sdk";

import { runWithExecutionContext } from "../../src/context.js";
import { AppBoundRuntime } from "../../src/memory_appbound.js";
import { extractTurnMessages, MemoryWriter } from "../../src/memory_writer.js";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("extractTurnMessages", () => {
  test("records the input user turn and preserves assistant/tool details", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call-1", name: "lookup", args: { id: 42 } }],
      },
      {
        role: "tool",
        content: "approved",
        toolCallId: "call-1",
        name: "lookup",
      },
      { role: "assistant", content: "Claim approved" },
    ];

    expect(extractTurnMessages("Check my claim", messages)).toEqual([
      { role: "user", content: "Check my claim" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          {
            id: "call-1",
            name: "lookup",
            arguments: { id: 42 },
          },
        ],
      },
      {
        role: "tool",
        content: "approved",
        toolCallId: "call-1",
        toolName: "lookup",
        isError: false,
      },
      { role: "assistant", content: "Claim approved", toolCalls: null },
    ]);
  });

  test("starts after an echoed current prompt instead of writing prior history", () => {
    const messages: Message[] = [
      { role: "user", content: "first" },
      { role: "assistant", content: "first answer" },
      { role: "user", content: [{ type: "text", text: "second" }] },
      { role: "assistant", content: "second answer" },
    ];

    expect(extractTurnMessages("second", messages)).toEqual([
      { role: "user", content: "second" },
      { role: "assistant", content: "second answer", toolCalls: null },
    ]);
  });

  test("starts after the latest matching prompt when a prompt is repeated", () => {
    const messages: Message[] = [
      { role: "user", content: "yes" },
      { role: "assistant", content: "old answer" },
      { role: "user", content: "yes" },
      { role: "assistant", content: "new answer" },
    ];

    expect(extractTurnMessages("yes", messages)).toEqual([
      { role: "user", content: "yes" },
      { role: "assistant", content: "new answer", toolCalls: null },
    ]);
  });

  test("preserves failed tool status", () => {
    const messages: Message[] = [
      {
        role: "tool",
        content: "lookup failed",
        toolCallId: "call-1",
        name: "lookup",
        isError: true,
      },
    ];

    expect(extractTurnMessages("find it", messages)).toEqual([
      { role: "user", content: "find it" },
      {
        role: "tool",
        content: "lookup failed",
        toolCallId: "call-1",
        toolName: "lookup",
        isError: true,
      },
    ]);
  });

  test("resume legs suppress every replayed user message", () => {
    const messages: Message[] = [
      { role: "user", content: "original prompt" },
      { role: "assistant", content: "resumed answer" },
    ];

    expect(extractTurnMessages("", messages, false)).toEqual([
      { role: "assistant", content: "resumed answer", toolCalls: null },
    ]);
  });
});

describe("MemoryWriter", () => {
  test("retains the execution context for non-blocking OE writes", async () => {
    const calls: Array<{
      url: string;
      headers: Record<string, string>;
      body: Record<string, unknown>;
    }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>;
        calls.push({
          url,
          headers: init.headers as Record<string, string>,
          body,
        });
        return new Response(
          JSON.stringify({
            id: `turn-${calls.length}`,
            session_id: body["session_id"],
            turn_seq: calls.length,
            acknowledged: true,
            has_embedding: true,
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        );
      }),
    );

    const writer = new MemoryWriter(new AppBoundRuntime());
    runWithExecutionContext(
      {
        executionId: "exec-42",
        wrapper: null,
        oeUrl: "http://oe.test",
        userId: "user-1",
        sessionId: "session-1",
      },
      () => {
        writer.writeTurnAsync({
          message: "hello",
          resultMessages: [{ role: "assistant", content: "hi" }],
          userId: "user-1",
          sessionId: "session-1",
        });
      },
    );

    await expect(writer.drain()).resolves.toBe(true);
    expect(calls.map((call) => call.url)).toEqual([
      "http://oe.test/api/v1/memory/turns",
      "http://oe.test/api/v1/memory/turns",
    ]);
    expect(
      calls.every(
        (call) =>
          call.headers["X-Agent-Engine-Execution-Id"] === "exec-42" &&
          // Legacy name is dual-sent during the rename transition.
          call.headers["X-Agentic-Execution-Id"] === "exec-42",
      ),
    ).toBe(true);
    expect(calls.map((call) => call.body)).toEqual([
      expect.objectContaining({
        role: "user",
        content: "hello",
        user_id: "user-1",
        session_id: "session-1",
        idempotency_key: expect.any(String),
      }),
      expect.objectContaining({
        role: "assistant",
        content: "hi",
        user_id: "user-1",
        session_id: "session-1",
        idempotency_key: expect.any(String),
      }),
    ]);
  });

  test("swallows a failed write and drains the queue", async () => {
    const writer = new MemoryWriter({
      recordTurn: vi.fn().mockRejectedValue(new Error("memory unavailable")),
    });

    writer.writeTurnAsync({
      message: "hello",
      resultMessages: [{ role: "assistant", content: "hi" }],
      userId: "user-1",
      sessionId: "session-1",
    });

    await expect(writer.drain()).resolves.toBe(true);
    expect(writer.pendingWrites).toBe(0);
  });

  test("writes a valid prefix before a malformed tool result stops the batch", async () => {
    const recordTurn = vi.fn().mockResolvedValue({
      id: "turn-1",
      session_id: "session-1",
      turn_seq: 1,
      acknowledged: true,
      has_embedding: true,
    });
    const writer = new MemoryWriter({ recordTurn });

    writer.writeTurnAsync({
      message: "hello",
      resultMessages: [
        { role: "assistant", content: "working" },
        { role: "tool", content: "", toolCallId: "call-1" },
      ],
      userId: "user-1",
      sessionId: "session-1",
    });

    await expect(writer.drain()).resolves.toBe(true);
    expect(recordTurn).toHaveBeenCalledTimes(2);
    expect(recordTurn.mock.calls.map(([turn]) => turn.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(writer.pendingWrites).toBe(0);
  });

  test("drain returns false when a write exceeds its timeout", async () => {
    vi.useFakeTimers();
    const writer = new MemoryWriter({
      recordTurn: vi.fn(() => new Promise(() => undefined)),
    });
    writer.writeTurnAsync({
      message: "hello",
      resultMessages: [],
      userId: "user-1",
      sessionId: "session-1",
    });

    const settled = vi.fn();
    const draining = writer.drain(100).then((result) => {
      settled(result);
      return result;
    });
    await vi.advanceTimersByTimeAsync(99);
    expect(settled).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    await expect(draining).resolves.toBe(false);
    expect(settled).toHaveBeenCalledWith(false);
    expect(writer.pendingWrites).toBe(1);
  });

  test("graceful shutdown waits for a queued write before app replacement", async () => {
    let completeWrite!: () => void;
    const recordTurn = vi.fn(
      () =>
        new Promise<{
          id: string;
          session_id: string;
          turn_seq: number;
          acknowledged: boolean;
          has_embedding: boolean;
        }>((resolve) => {
          completeWrite = () =>
            resolve({
              id: "turn-1",
              session_id: "session-1",
              turn_seq: 1,
              acknowledged: true,
              has_embedding: true,
            });
        }),
    );
    const writer = new MemoryWriter({ recordTurn });
    writer.writeTurnAsync({
      message: "before suspend",
      resultMessages: [],
      userId: "user-1",
      sessionId: "session-1",
    });

    const settled = vi.fn();
    const shutdown = writer.shutdown().then(settled);
    await vi.waitFor(() => expect(recordTurn).toHaveBeenCalledOnce());
    expect(settled).not.toHaveBeenCalled();

    completeWrite();
    await shutdown;
    expect(settled).toHaveBeenCalledOnce();
    expect(writer.pendingWrites).toBe(0);
  });
});
