/**
 * Tests for StoppedToolCallMiddleware. Mirrors Python's test_stopped_tool_call_middleware.py.
 */

import { describe, expect, it } from "vitest";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { CALL_INTERRUPTED_ARTIFACT_KEY } from "@mongodb-js/agent-engine-runner-shared";
import {
  ALL_INTERRUPTED_MESSAGE,
  StoppedToolCallMiddleware,
  latestToolBatch,
} from "../src/stopped_tool_call_middleware.js";

const INTERRUPTED_ARTIFACT = { [CALL_INTERRUPTED_ARTIFACT_KEY]: true };

function aiWithToolCalls(...callIds: string[]): AIMessage {
  return new AIMessage({
    content: "",
    tool_calls: callIds.map((id) => ({ name: "t", args: {}, id })),
  });
}

describe("latestToolBatch", () => {
  it("returns an empty batch for no messages", () => {
    expect(latestToolBatch([])).toEqual([]);
  });

  it("returns no batch after a plain assistant reply", () => {
    const messages = [new HumanMessage("hi"), new AIMessage("hello there")];
    expect(latestToolBatch(messages)).toEqual([]);
  });

  it("returns tool messages in original order", () => {
    const tool1 = new ToolMessage({ content: "a", tool_call_id: "1" });
    const tool2 = new ToolMessage({ content: "b", tool_call_id: "2" });
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1", "2"),
      tool1,
      tool2,
    ];
    expect(latestToolBatch(messages)).toEqual([tool1, tool2]);
  });

  it("only considers the most recent batch", () => {
    const staleTool = new ToolMessage({ content: "stale", tool_call_id: "0" });
    const currentTool = new ToolMessage({
      content: "current",
      tool_call_id: "1",
    });
    const messages = [
      aiWithToolCalls("0"),
      staleTool,
      new AIMessage("following up"),
      aiWithToolCalls("1"),
      currentTool,
    ];
    expect(latestToolBatch(messages)).toEqual([currentTool]);
  });
});

describe("StoppedToolCallMiddleware", () => {
  function run(messages: unknown[]) {
    const middleware = StoppedToolCallMiddleware();
    const hook = (
      middleware.beforeModel as { hook: (state: unknown) => unknown }
    ).hook;
    return hook({ messages }) as
      | { jumpTo: string; messages: AIMessage[] }
      | undefined;
  }

  it("is a no-op with no tool batch", () => {
    const messages = [new HumanMessage("hi"), new AIMessage("hello")];
    expect(run(messages)).toBeUndefined();
  });

  it("ends the turn deterministically when every call is interrupted", () => {
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1", "2"),
      new ToolMessage({
        content: "stopped",
        artifact: INTERRUPTED_ARTIFACT,
        tool_call_id: "1",
      }),
      new ToolMessage({
        content: "stopped",
        artifact: INTERRUPTED_ARTIFACT,
        tool_call_id: "2",
      }),
    ];

    const result = run(messages);

    expect(result?.jumpTo).toBe("end");
    expect(result?.messages).toHaveLength(1);
    expect(result?.messages[0]?.content).toBe(ALL_INTERRUPTED_MESSAGE);
  });

  it("ends the turn for a single interrupted call", () => {
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1"),
      new ToolMessage({
        content: "stopped",
        artifact: INTERRUPTED_ARTIFACT,
        tool_call_id: "1",
      }),
    ];

    expect(run(messages)?.jumpTo).toBe("end");
  });

  it("defers to the model on a partial interruption", () => {
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1", "2"),
      new ToolMessage({ content: "real result", tool_call_id: "1" }),
      new ToolMessage({
        content: "stopped",
        artifact: INTERRUPTED_ARTIFACT,
        tool_call_id: "2",
      }),
    ];

    expect(run(messages)).toBeUndefined();
  });

  it("is a no-op when nothing was interrupted", () => {
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1"),
      new ToolMessage({ content: "real result", tool_call_id: "1" }),
    ];

    expect(run(messages)).toBeUndefined();
  });

  it("never mistakes a tool's own artifact for an interrupt marker", () => {
    const messages = [
      new HumanMessage("hi"),
      aiWithToolCalls("1"),
      new ToolMessage({
        content: "real result",
        artifact: ["some", "real", "artifact"],
        tool_call_id: "1",
      }),
    ];

    expect(run(messages)).toBeUndefined();
  });

  it("declares end as a valid jump target", () => {
    const middleware = StoppedToolCallMiddleware();
    expect(
      (middleware.beforeModel as { canJumpTo: string[] }).canJumpTo,
    ).toContain("end");
  });
});
