import { create } from "@bufbuild/protobuf";
import {
  AIMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  Annotation,
  Command,
  MessagesAnnotation,
  Overwrite,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  AttemptContextSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../src/agent.js";

function attempt(attemptId: string, fencingToken: bigint) {
  return create(AttemptContextSchema, {
    attemptId,
    fencingToken,
    workflowIdentity: create(WorkflowIdentitySchema, {
      executionId: "execution-1",
    }),
  });
}

describe("durable message identity", () => {
  it("stabilizes missing AIMessage ids and preserves explicit ids", async () => {
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [
          new AIMessage({ content: "first" }),
          new AIMessage({ content: "second" }),
          new AIMessage({ content: "provider", id: "provider-id" }),
        ],
      }))
      .addEdge(START, "respond")
      .compile();
    new LangGraphBaseAgent(graph as never);

    const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
      graph.invoke({ messages: [] }),
    );
    const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
      graph.invoke({ messages: [] }),
    );

    expect(first.messages.map((message) => message.id)).toEqual(
      replay.messages.map((message) => message.id),
    );
    expect(first.messages[0]?.id).not.toBe(first.messages[1]?.id);
    expect(first.messages[0]?.id).toMatch(/^durable-message:execution-1:1:/);
    expect(first.messages[2]?.id).toBe("provider-id");
  });

  it("stabilizes an ordinary ToolMessage from the stock ToolNode", async () => {
    const lookupOrder = tool(async ({ orderId }) => `found:${orderId}`, {
      name: "lookup_order",
      description: "Look up an order.",
      schema: z.object({ orderId: z.string() }),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("tools", new ToolNode([lookupOrder]))
      .addEdge(START, "tools")
      .compile();
    new LangGraphBaseAgent(graph as never);
    const graphInput = () => ({
      messages: [
        new AIMessage({
          content: "",
          id: "assistant-plan",
          tool_calls: [
            {
              name: "lookup_order",
              args: { orderId: "123" },
              id: "tool-call-1",
            },
          ],
        }),
      ],
    });

    const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
      graph.invoke(graphInput()),
    );
    const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
      graph.invoke(graphInput()),
    );
    const firstTool = first.messages.at(-1);
    const replayTool = replay.messages.at(-1);

    expect(firstTool).toBeInstanceOf(ToolMessage);
    expect(replayTool).toBeInstanceOf(ToolMessage);
    expect(firstTool?.id).toBe(replayTool?.id);
    expect(firstTool?.id).toMatch(/^durable-message:execution-1:1:/);
    expect((firstTool as ToolMessage).tool_call_id).toBe("tool-call-1");
  });

  it.each(["class", "serialized"] as const)(
    "stabilizes messages carried through a %s Overwrite before a later update",
    async (overwriteFormat) => {
      const overwrittenMessages = (messages: BaseMessage[]) =>
        overwriteFormat === "class"
          ? new Overwrite(messages)
          : { __overwrite__: messages };
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("patch_tool_calls", (state) => ({
          messages: overwrittenMessages([
            ...state.messages,
            new ToolMessage({
              content: "cancelled",
              name: "lookup_order",
              tool_call_id: "tool-call-1",
            }),
          ]),
        }))
        .addNode("respond", () => ({
          messages: [new AIMessage({ content: "continued" })],
        }))
        .addEdge(START, "patch_tool_calls")
        .addEdge("patch_tool_calls", "respond")
        .compile();
      new LangGraphBaseAgent(graph as never);
      const graphInput = () => ({
        messages: [
          new AIMessage({
            content: "",
            id: "assistant-plan",
            tool_calls: [
              {
                name: "lookup_order",
                args: { orderId: "123" },
                id: "tool-call-1",
              },
            ],
          }),
        ],
      });

      const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
        graph.invoke(graphInput()),
      );
      const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
        graph.invoke(graphInput()),
      );
      const firstTool = first.messages[1];
      const replayTool = replay.messages[1];

      expect(firstTool).toBeInstanceOf(ToolMessage);
      expect(replayTool).toBeInstanceOf(ToolMessage);
      expect(firstTool?.id).toBe(replayTool?.id);
      expect(firstTool?.id).toMatch(/^durable-message:execution-1:1:/);
      expect((firstTool as ToolMessage).tool_call_id).toBe("tool-call-1");
    },
  );

  it("stabilizes tuple-form messages across replacement attempts", async () => {
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("respond", () => ({
        messages: [["human", "hello"]] as [["human", string]],
      }))
      .addEdge(START, "respond")
      .compile();
    new LangGraphBaseAgent(graph as never);

    const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
      graph.invoke({ messages: [] }),
    );
    const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
      graph.invoke({ messages: [] }),
    );

    expect(first.messages[0]?.content).toBe("hello");
    expect(first.messages[0]?.id).toBe(replay.messages[0]?.id);
    expect(first.messages[0]?.id).toMatch(/^durable-message:execution-1:1:/);
  });

  it.each(["assistant", "user"] as const)(
    "stabilizes type=%s message records across replacement attempts",
    async (messageType) => {
      const graph = new StateGraph(MessagesAnnotation)
        .addNode("respond", () => ({
          messages: [{ type: messageType, content: "hello" }] as never,
        }))
        .addEdge(START, "respond")
        .compile();
      new LangGraphBaseAgent(graph as never);

      const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
        graph.invoke({ messages: [] }),
      );
      const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
        graph.invoke({ messages: [] }),
      );

      expect(first.messages[0]?.id).toBe(replay.messages[0]?.id);
      expect(first.messages[0]?.id).toMatch(/^durable-message:execution-1:1:/);
    },
  );

  it("stabilizes messages in mapping-form Command updates", async () => {
    const graph = new StateGraph(MessagesAnnotation)
      .addNode(
        "respond",
        () =>
          new Command({
            update: {
              messages: [new AIMessage({ content: "replacement" })],
            },
          }),
      )
      .addEdge(START, "respond")
      .compile();
    new LangGraphBaseAgent(graph as never);

    const first = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
      graph.invoke({ messages: [] }),
    );
    const replay = await runWithAttemptContext(attempt("attempt-2", 2n), () =>
      graph.invoke({ messages: [] }),
    );

    expect(first.messages[0]?.content).toBe("replacement");
    expect(first.messages[0]?.id).toBe(replay.messages[0]?.id);
    expect(first.messages[0]?.id).toMatch(/^durable-message:execution-1:1:/);
  });

  it("does not interpret role-like ordinary state as messages", async () => {
    const State = Annotation.Root({
      ...MessagesAnnotation.spec,
      profile: Annotation<string[]>(),
      descriptor: Annotation<Record<string, string>>(),
    });
    const graph = new StateGraph(State)
      .addNode("respond", () => ({
        messages: [new AIMessage({ content: "done" })],
        profile: ["user", "admin"],
        descriptor: { type: "assistant", content: "ordinary state" },
      }))
      .addEdge(START, "respond")
      .compile();
    new LangGraphBaseAgent(graph as never);

    const result = await runWithAttemptContext(attempt("attempt-1", 1n), () =>
      graph.invoke({ messages: [], profile: [], descriptor: {} }),
    );

    expect(result.profile).toEqual(["user", "admin"]);
    expect(result.descriptor).toEqual({
      type: "assistant",
      content: "ordinary state",
    });
  });
});
