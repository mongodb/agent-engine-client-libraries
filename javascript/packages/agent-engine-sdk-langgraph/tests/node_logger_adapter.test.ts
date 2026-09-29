/**
 * Port of `agent-engine-sdk-langgraph/tests/test_node_logger_adapter.py`.
 *
 * Verifies LangGraphCallbackAdapter delegation to BaseExecutionCallback.
 */

import { describe, expect, it, vi } from "vitest";
import { GraphInterrupt } from "@langchain/langgraph";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { createMiddleware } from "langchain";
import { createDeepAgent } from "deepagents";
import { z } from "zod";
import type { BaseExecutionCallback } from "@mongodb-js/agent-engine-sdk";

import { LangGraphCallbackAdapter } from "../src/node_logger_adapter.js";

function mockCallback(): BaseExecutionCallback & {
  onNodeSuspend?: (
    nodeName: string,
    opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ) => void;
  onNodeInterrupted?: (
    nodeName: string,
    opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ) => void;
} {
  return {
    onNodeStart: vi.fn(),
    onNodeEnd: vi.fn(),
    onNodeError: vi.fn(),
    onNodeSuspend: vi.fn(),
    onNodeInterrupted: vi.fn(),
  };
}

describe("LangGraphCallbackAdapter delegation", () => {
  it("handleChainStart extracts node name from metadata.langgraph_node and delegates", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      { name: "fallback" } as never,
      { key: "value" },
      "run-1",
      undefined,
      ["graph:step:1", "some_tag"],
      { langgraph_node: "agent_node" },
    );

    expect(cb.onNodeStart).toHaveBeenCalledOnce();
    expect(cb.onNodeStart).toHaveBeenCalledWith(
      "agent_node",
      { key: "value" },
      expect.objectContaining({
        runId: "run-1",
        metadata: { langgraph_node: "agent_node" },
      }),
    );
  });

  it("handleChainStart falls back to a non-seq:step tag for node name", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      { name: "fallback" } as never,
      {},
      "run-1",
      undefined,
      ["seq:step:1", "seq:step:2", "my_node"],
      undefined,
    );

    expect(cb.onNodeStart).toHaveBeenCalledOnce();
    expect(
      (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls[0]?.[0],
    ).toBe("my_node");
  });

  it("handleChainStart skips seq:step tags beyond three", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      { name: "fallback" } as never,
      {},
      "run-1",
      undefined,
      ["seq:step:4", "seq:step:5", "my_node"],
      undefined,
    );

    expect(cb.onNodeStart).toHaveBeenCalledOnce();
    expect(
      (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls[0]?.[0],
    ).toBe("my_node");
  });

  it("handleChainStart skips RunnableSequence nodes", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      { name: "RunnableSequence" } as never,
      {},
      "run-1",
      undefined,
      undefined,
      undefined,
    );

    expect(cb.onNodeStart).not.toHaveBeenCalled();
  });

  it("handleChainStart skips when no node name can be extracted", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      undefined,
      undefined,
    );

    expect(cb.onNodeStart).not.toHaveBeenCalled();
  });

  it("handleChainEnd delegates to onNodeEnd using stored metadata", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    // Store metadata for run-1
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    adapter.handleChainEnd({ result: "done" }, "run-1");

    expect(cb.onNodeEnd).toHaveBeenCalledOnce();
    expect((cb.onNodeEnd as ReturnType<typeof vi.fn>).mock.calls[0]?.[0]).toBe(
      "agent_node",
    );
    expect(
      (cb.onNodeEnd as ReturnType<typeof vi.fn>).mock.calls[0]?.[1],
    ).toEqual({ result: "done" });
  });

  it("handleChainError delegates non-GraphInterrupt errors to onNodeError", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    adapter.handleChainError(new Error("something broke"), "run-1");

    expect(cb.onNodeError).toHaveBeenCalledOnce();
    const args = (cb.onNodeError as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("agent_node");
    expect(args?.[1]).toBe("something broke");
  });

  it("handleChainError routes GraphInterrupt to onNodeSuspend (not onNodeError)", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    adapter.handleChainError(new GraphInterrupt(), "run-1");

    expect(cb.onNodeSuspend).toHaveBeenCalledOnce();
    expect(cb.onNodeError).not.toHaveBeenCalled();
    const args = (cb.onNodeSuspend as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("agent_node");
  });

  it("GraphInterrupt falls back to onNodeEnd when callback lacks onNodeSuspend", () => {
    // Legacy callback — no onNodeSuspend method.
    const cb = {
      onNodeStart: vi.fn(),
      onNodeEnd: vi.fn(),
      onNodeError: vi.fn(),
    } as BaseExecutionCallback;
    const adapter = new LangGraphCallbackAdapter(cb);
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    adapter.handleChainError(new GraphInterrupt(), "run-1");

    expect(cb.onNodeEnd).toHaveBeenCalledOnce();
    expect(cb.onNodeError).not.toHaveBeenCalled();
    const args = (cb.onNodeEnd as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("agent_node");
    expect(args?.[1]).toEqual({}); // outputs is empty when falling back from suspend
  });

  it("handleChainError routes AbortError to onNodeInterrupted (not onNodeError)", () => {
    // Cooperative cancellation (per-call stop, run drain, teardown) closes the
    // node as interrupted; an error row would double-count the stop.
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    const abort = new Error("This call was stopped before completing.");
    abort.name = "AbortError";
    adapter.handleChainError(abort, "run-1");

    expect(cb.onNodeError).not.toHaveBeenCalled();
    expect(cb.onNodeInterrupted).toHaveBeenCalledOnce();
    const args = (cb.onNodeInterrupted as ReturnType<typeof vi.fn>).mock
      .calls[0];
    expect(args?.[0]).toBe("agent_node");
  });

  it("AbortError falls back to onNodeEnd when callback lacks onNodeInterrupted", () => {
    // Legacy callback — no onNodeInterrupted method.
    const cb = {
      onNodeStart: vi.fn(),
      onNodeEnd: vi.fn(),
      onNodeError: vi.fn(),
    } as BaseExecutionCallback;
    const adapter = new LangGraphCallbackAdapter(cb);
    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["graph:step:1"],
      {
        langgraph_node: "agent_node",
      },
    );

    const abort = new Error("execution is draining");
    abort.name = "AbortError";
    adapter.handleChainError(abort, "run-1");

    expect(cb.onNodeEnd).toHaveBeenCalledOnce();
    expect(cb.onNodeError).not.toHaveBeenCalled();
    const args = (cb.onNodeEnd as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("agent_node");
  });

  it("propagates parentRunId through to the callback", () => {
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      {} as never,
      {},
      "child-run",
      undefined,
      ["graph:step:1"],
      { langgraph_node: "node" },
      undefined,
      "parent-run",
    );

    const args = (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[2]).toEqual(
      expect.objectContaining({
        runId: "child-run",
        parentRunId: "parent-run",
      }),
    );
  });

  // -------------------------------------------------------------------------
  // Additional ports from test_node_logger_adapter.py
  // -------------------------------------------------------------------------

  // NOTE: Python's adapter coerces `run_id` with `str(run_id)` because
  // LangChain Python types it as a `UUID` object. LangChain JS types `runId`
  // as `string` in its callback contract, so there is nothing to coerce — the
  // corresponding Python test (test_converts_uuid_to_str) has no TS analog.

  it("extracts node name from serialized.name when metadata is absent", () => {
    // Python: test_extracts_from_serialized_name.
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      { name: "custom_node" } as never,
      {},
      "run-1",
      undefined,
      undefined,
      undefined,
    );

    const args = (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("custom_node");
  });

  it("extracts node name from tags when metadata and serialized.name are absent", () => {
    // Python: test_extracts_from_tags.
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart(
      {} as never,
      {},
      "run-1",
      undefined,
      ["my_node_tag"],
      undefined,
    );

    const args = (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(args?.[0]).toBe("my_node_tag");
  });

  it("returns without calling the callback when no node name can be derived from any source", () => {
    // Python: test_returns_none_when_nothing_found.
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart({} as never, {}, "run-1", undefined, [], {});

    expect(cb.onNodeStart).not.toHaveBeenCalled();
  });

  it("drops runs carrying langgraph_node metadata without a graph:step tag, including their end and error", () => {
    // Pregel stamps real node task runs with graph:step:N; a run with the
    // metadata but not the tag is a conditional-edge router evaluation that
    // inherited the node's metadata.
    const cb = mockCallback();
    const adapter = new LangGraphCallbackAdapter(cb);

    adapter.handleChainStart({} as never, {}, "run-1", undefined, [], {
      langgraph_node: "agent_node",
    });
    // A nameable tag proves the end is dropped by the skip, not by extraction.
    adapter.handleChainEnd({ result: "done" }, "run-1", undefined, [
      "graph:step:1",
    ]);

    adapter.handleChainStart({} as never, {}, "run-2", undefined, undefined, {
      langgraph_node: "agent_node",
    });
    adapter.handleChainError(new Error("boom"), "run-2");

    expect(cb.onNodeStart).not.toHaveBeenCalled();
    expect(cb.onNodeEnd).not.toHaveBeenCalled();
    expect(cb.onNodeError).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Driven regression: conditional-edge router runs must not become node rows
// ---------------------------------------------------------------------------

/** Tool call on the first turn, plain text on the second. */
class ToolThenTextModel extends BaseChatModel {
  calls = 0;

  constructor() {
    super({});
  }

  _llmType(): string {
    return "tool-then-text-fake";
  }

  override bindTools(): Runnable {
    return this as unknown as Runnable;
  }

  async _generate(): Promise<ChatResult> {
    this.calls += 1;
    if (this.calls === 1) {
      return {
        generations: [
          {
            message: new AIMessage({
              content: "",
              tool_calls: [
                {
                  id: "call-1",
                  name: "get_weather",
                  args: {},
                  type: "tool_call",
                },
              ],
            }),
            text: "",
          },
        ],
      };
    }
    return {
      generations: [
        { message: new AIMessage({ content: "done" }), text: "done" },
      ],
    };
  }
}

describe("conditional-edge router runs", () => {
  it("emits one node event per hook execution, whether or not the hook declares jump targets", async () => {
    // A hook with `canJumpTo` is wired as a conditional edge; LangGraph runs
    // the edge's router inside the source node's task, so the router's chain
    // run inherits the node's `langgraph_node` metadata. It is a route
    // evaluation, not a node execution, and must produce no events.
    const weather = tool(async () => "sunny", {
      name: "get_weather",
      description: "Get the weather",
      schema: z.object({}),
    });
    const jumpy = createMiddleware({
      name: "JumpMiddleware",
      beforeModel: { canJumpTo: ["end"], hook: () => undefined },
    });
    const plain = createMiddleware({
      name: "NoJumpMiddleware",
      beforeModel: () => undefined,
    });
    const agent = createDeepAgent({
      model: new ToolThenTextModel(),
      tools: [weather],
      middleware: [jumpy, plain],
    });

    const cb = mockCallback();
    await agent.invoke(
      { messages: [new HumanMessage("weather?")] },
      { callbacks: [new LangGraphCallbackAdapter(cb)] },
    );

    const startCalls = (cb.onNodeStart as ReturnType<typeof vi.fn>).mock.calls;
    const endCalls = (cb.onNodeEnd as ReturnType<typeof vi.fn>).mock.calls;
    const starts = startCalls.map((c) => c[0]);
    const ends = endCalls.map((c) => c[0]);

    // Two agent-loop passes (tool call, then final answer) execute each
    // before_model hook twice.
    expect(
      starts.filter((n) => n === "JumpMiddleware.before_model"),
    ).toHaveLength(2);
    expect(
      starts.filter((n) => n === "NoJumpMiddleware.before_model"),
    ).toHaveLength(2);
    expect(
      ends.filter((n) => n === "JumpMiddleware.before_model"),
    ).toHaveLength(2);
  });
});
