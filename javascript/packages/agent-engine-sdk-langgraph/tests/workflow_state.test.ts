import { create } from "@bufbuild/protobuf";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { emptyCheckpoint, MemorySaver } from "@langchain/langgraph";
import {
  MessageRole,
  StateSnapshotSchema,
  WorkflowMessageSchema,
} from "@mongodb-js/agent-engine-runner-shared";
import { describe, expect, it } from "vitest";

import {
  channelValuesToStateSnapshot,
  stateSnapshotToChannelValues,
} from "../src/workflow_state.js";
import { jsonValueFromProto } from "../src/workflow_json.js";

describe("LangGraph durable workflow state", () => {
  it("requires explicit review when the durable message contract changes", () => {
    expect(Object.keys(StateSnapshotSchema.field).sort()).toEqual(
      [
        "messageEncodingVersion",
        "messages",
        "properties",
        "replayProperties",
      ].sort(),
    );
    expect(Object.keys(WorkflowMessageSchema.field).sort()).toEqual(
      [
        "additionalKwargs",
        "artifacts",
        "content",
        "id",
        "invalidToolCalls",
        "isError",
        "name",
        "platformArtifacts",
        "responseMetadata",
        "role",
        "sourceMessage",
        "toolArtifact",
        "toolCallId",
        "toolCalls",
      ].sort(),
    );
  });

  it("round-trips application channels and complete message identity", () => {
    const snapshot = channelValuesToStateSnapshot({
      messages: [
        new HumanMessage({ content: "hello", id: "human-1" }),
        new AIMessage({
          content: "checking",
          id: "ai-1",
          tool_calls: [{ id: "call-1", name: "lookup", args: { q: 1 } }],
        }),
        new ToolMessage({
          content: "found",
          id: "tool-1",
          tool_call_id: "call-1",
        }),
      ],
      count: 3,
      profile: { tier: "gold" },
      __pregel_scratch: "ignored",
      "branch:agent:tools": "ignored",
      "start:agent": "ignored",
    });

    expect(snapshot.properties).toEqual({
      count: 3,
      profile: { tier: "gold" },
    });
    expect(snapshot.messageEncodingVersion).toBe(1);
    expect(snapshot.messages.map((message) => message.id)).toEqual([
      "human-1",
      "ai-1",
      "tool-1",
    ]);

    const restored = stateSnapshotToChannelValues(snapshot);
    expect(restored["count"]).toBe(3);
    expect(restored).not.toHaveProperty("__pregel_scratch");
    const messages = restored["messages"] as Array<{
      id?: string;
      type: string;
      tool_call_id?: string;
    }>;
    expect(messages.map((message) => [message.type, message.id])).toEqual([
      ["human", "human-1"],
      ["ai", "ai-1"],
      ["tool", "tool-1"],
    ]);
    expect(messages[2]?.tool_call_id).toBe("call-1");
  });

  it("preserves canonical durable state across a checkpoint round-trip", async () => {
    const channelValues = {
      messages: [
        new HumanMessage({
          content: [{ type: "text", text: "hello" }],
          id: "human-1",
          name: "customer",
          additional_kwargs: { nullable: null },
          response_metadata: { source: "test" },
        }),
        new SystemMessage({ content: "follow policy", id: "system-1" }),
        new AIMessage({
          content: "",
          id: "assistant-1",
          name: "assistant",
          tool_calls: [
            {
              name: "lookup",
              args: { policy: "POL-1" },
              id: "call-1",
              type: "tool_call",
            },
          ],
          invalid_tool_calls: [
            {
              name: "search",
              args: '{"query":',
              id: "call-invalid",
              error: "Malformed arguments",
              type: "invalid_tool_call",
            },
          ],
          additional_kwargs: { provider_extension: { enabled: true } },
          response_metadata: { finish_reason: "tool_calls" },
        }),
        new ToolMessage({
          content: "not found",
          tool_call_id: "call-1",
          id: "tool-1",
          name: "lookup",
          status: "error",
        }),
        new ToolMessage({
          content: "no machine-readable result",
          tool_call_id: "call-2",
          artifact: null,
        }),
        new ToolMessage({
          content: "chart generated",
          tool_call_id: "call-3",
          artifact: { series: [1, null, 3] },
          additional_kwargs: {
            artifacts: [{ id: "chart-1", kind: "chart" }],
          },
        }),
      ],
      count: 3,
      nullable: null,
      nested: { items: ["one", { two: 2 }] },
    };
    const saver = new MemorySaver();
    const config = { configurable: { thread_id: "canonical-state" } };
    await saver.put(
      config,
      {
        ...emptyCheckpoint(),
        id: "canonical-checkpoint",
        channel_values: channelValues,
      },
      { source: "loop", step: 0, parents: {} },
    );
    const checkpointed = await saver.get(config);
    if (checkpointed === undefined) throw new Error("expected checkpoint");

    expect(channelValuesToStateSnapshot(checkpointed.channel_values)).toEqual(
      channelValuesToStateSnapshot(channelValues),
    );
  });

  it("reads legacy source messages but omits them from new state", () => {
    const original = new AIMessage({
      content: "legacy",
      id: "message-legacy",
      additional_kwargs: { claim_evidence_stage: "verified" },
      response_metadata: { model: "legacy-model" },
    });

    const legacy = channelValuesToStateSnapshot({ messages: [original] }, true);
    const restored = stateSnapshotToChannelValues(legacy);
    const compact = channelValuesToStateSnapshot(restored);

    expect(legacy.messages[0]?.sourceMessage).toBeDefined();
    expect(compact.messages[0]?.sourceMessage).toBeUndefined();
    expect(compact.messages[0]?.additionalKwargs).toEqual({
      claim_evidence_stage: "verified",
    });
    expect(compact.messages[0]?.responseMetadata).toEqual({
      model: "legacy-model",
    });
  });

  it("omits absent optional channels but rejects non-JSON application state", () => {
    expect(
      channelValuesToStateSnapshot({ optional: undefined }).properties,
    ).toEqual({});
    expect(() => channelValuesToStateSnapshot({ value: 1n })).toThrow(
      "LangGraph application state must be a JSON object",
    );
  });

  it("canonicalizes live and replayed streaming tool-call metadata", () => {
    const fields = {
      content: "",
      id: "ai-tools",
      tool_calls: [
        {
          id: "call-1",
          name: "lookup",
          args: { q: 1 },
          type: "tool_call" as const,
        },
      ],
      response_metadata: { finish_reason: "tool_calls" },
    };
    const live = new AIMessageChunk({
      ...fields,
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14,
        input_token_details: {},
        output_token_details: {},
      },
      tool_call_chunks: [
        {
          id: "call-1",
          name: "lookup",
          args: '{"q": 1}',
          index: 0,
          type: "tool_call_chunk",
        },
      ],
      additional_kwargs: {
        tool_calls: [
          {
            id: "call-1",
            index: 0,
            type: "function",
            function: { name: "lookup", arguments: '{"q": 1}' },
          },
        ],
        refusal: null,
      },
    } as never);
    const replay = new AIMessageChunk({
      ...fields,
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14,
      },
      tool_call_chunks: [
        {
          id: "call-1",
          name: "lookup",
          args: '{"q":1}',
          index: 0,
          type: "tool_call_chunk",
        },
      ],
      additional_kwargs: {
        tool_calls: [{ index: 0, function: { arguments: "1}" } }],
        refusal: null,
      } as never,
    } as never);

    const liveSnapshot = channelValuesToStateSnapshot({ messages: [live] });
    const replaySnapshot = channelValuesToStateSnapshot({ messages: [replay] });

    expect(replaySnapshot.messages).toEqual(liveSnapshot.messages);
    expect(liveSnapshot.messages[0]?.additionalKwargs).toEqual({
      refusal: null,
    });
    expect(liveSnapshot.messages[0]?.responseMetadata).toEqual({
      finish_reason: "tool_calls",
    });
    const restored = stateSnapshotToChannelValues(liveSnapshot)[
      "messages"
    ] as AIMessage[];
    expect(restored[0]?.tool_calls).toEqual(fields.tool_calls);
    expect(restored[0]?.additional_kwargs).toEqual({ refusal: null });
    expect(
      (restored[0] as AIMessage & { usage_metadata?: unknown }).usage_metadata,
    ).toBeUndefined();
  });

  it("preserves JSON message artifacts", () => {
    const message = new HumanMessage({
      content: "see attachment",
      additional_kwargs: {
        artifacts: [{ type: "document", name: "report.pdf" }],
      },
    });

    const snapshot = channelValuesToStateSnapshot({ messages: [message] });
    expect(snapshot.messages[0]?.platformArtifacts).toEqual([
      { type: "document", name: "report.pdf" },
    ]);
    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as HumanMessage[];
    expect(restored[0]?.additional_kwargs["artifacts"]).toEqual([
      { type: "document", name: "report.pdf" },
    ]);
  });

  it.each(["not-an-array", null])(
    "rejects a message artifact field that is not an array: %#",
    (artifacts) => {
      const message = new HumanMessage({
        content: "see attachment",
        additional_kwargs: { artifacts },
      });

      expect(() =>
        channelValuesToStateSnapshot({ messages: [message] }),
      ).toThrow("message additional_kwargs.artifacts must be an array");
    },
  );

  it("rejects an unspecified message role from OE state", () => {
    const snapshot = create(StateSnapshotSchema, {
      messages: [
        create(WorkflowMessageSchema, {
          role: MessageRole.UNSPECIFIED,
        }),
      ],
    });

    expect(() => stateSnapshotToChannelValues(snapshot)).toThrow(
      "messages[0]: previous workflow message has an unspecified role",
    );
  });

  it.each(["messages", "__pregel_tasks", "branch:agent:tools", "start:agent"])(
    "rejects reserved channel %s from OE properties",
    (name) => {
      const snapshot = create(StateSnapshotSchema, {
        properties: { [name]: "forged" },
      });

      expect(() => stateSnapshotToChannelValues(snapshot)).toThrow(
        `previous workflow state contains reserved LangGraph channel "${name}"`,
      );
    },
  );

  it("uses explicit message identity as the reconstruction source", () => {
    const snapshot = channelValuesToStateSnapshot({
      messages: [new HumanMessage({ content: "hello", id: "human-1" })],
    });
    const message = snapshot.messages[0];
    if (message === undefined) throw new Error("expected workflow message");
    message.id = "forged-id";

    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as HumanMessage[];
    expect(restored[0]?.id).toBe("forged-id");
  });

  it("round-trips tool errors through the explicit field", () => {
    const snapshot = channelValuesToStateSnapshot({
      messages: [
        new ToolMessage({
          content: "lookup failed",
          tool_call_id: "call-1",
          status: "error",
          artifact: ["machine", { readable: true }],
        }),
      ],
    });

    expect(snapshot.messages[0]?.isError).toBe(true);
    expect(jsonValueFromProto(snapshot.messages[0]?.toolArtifact)).toEqual([
      "machine",
      { readable: true },
    ]);
    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as ToolMessage[];
    expect(restored[0]?.status).toBe("error");
    expect(restored[0]?.artifact).toEqual(["machine", { readable: true }]);
  });

  it("keeps platform attachments separate from a tool artifact", () => {
    const original = new ToolMessage({
      content: "chart generated",
      tool_call_id: "call-1",
      artifact: { series: [1, 2] },
      additional_kwargs: {
        artifacts: [{ id: "chart-1", kind: "chart" }],
      },
    });

    const snapshot = channelValuesToStateSnapshot({ messages: [original] });

    expect(snapshot.messages[0]?.platformArtifacts).toEqual([
      { id: "chart-1", kind: "chart" },
    ]);
    expect(jsonValueFromProto(snapshot.messages[0]?.toolArtifact)).toEqual({
      series: [1, 2],
    });
    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as ToolMessage[];
    expect(restored[0]?.artifact).toEqual({ series: [1, 2] });
    expect(restored[0]?.additional_kwargs["artifacts"]).toEqual([
      { id: "chart-1", kind: "chart" },
    ]);
  });

  it("normalizes explicit null and an absent tool artifact", () => {
    const withNull = channelValuesToStateSnapshot({
      messages: [
        new ToolMessage({
          content: "no machine-readable result",
          tool_call_id: "call-1",
          artifact: null,
        }),
      ],
    });
    const withoutArtifact = channelValuesToStateSnapshot({
      messages: [
        new ToolMessage({
          content: "no machine-readable result",
          tool_call_id: "call-1",
        }),
      ],
    });

    expect(withNull.messages[0]?.toolArtifact).toBeUndefined();
    expect(withoutArtifact.messages[0]?.toolArtifact).toBeUndefined();
    expect(withNull.messages[0]).toEqual(withoutArtifact.messages[0]);
  });

  it("round-trips invalid tool calls independently of valid tool calls", () => {
    const original = new AIMessage({
      content: "",
      invalid_tool_calls: [
        {
          name: "lookup",
          args: '{"query":',
          id: "call-invalid",
          error: "Malformed arguments",
          type: "invalid_tool_call",
        },
      ],
    });

    const snapshot = channelValuesToStateSnapshot({ messages: [original] });
    expect(snapshot.messages[0]?.invalidToolCalls).toEqual(
      original.invalid_tool_calls,
    );
    expect(snapshot.messages[0]?.toolCalls).toEqual([]);
    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as AIMessage[];
    expect(restored[0]?.invalid_tool_calls).toEqual(
      original.invalid_tool_calls,
    );
    expect(restored[0]?.tool_calls).toEqual([]);
  });

  it.each([{ score: Number.NaN }, { detail: undefined }])(
    "rejects a lossy tool artifact before encoding: %#",
    (artifact) => {
      const message = new ToolMessage({
        content: "lookup result",
        tool_call_id: "call-1",
        artifact,
      });

      expect(() =>
        channelValuesToStateSnapshot({ messages: [message] }),
      ).toThrow("message tool_artifact is not JSON-serializable");
    },
  );

  it.each([
    new AIMessage({ content: [{ type: "text", text: "hello" }] }),
    new ToolMessage({
      content: [{ type: "json", rows: [1, 2] }],
      tool_call_id: "call-1",
    }),
  ])("preserves supported structured content for %#", (original) => {
    const snapshot = channelValuesToStateSnapshot({ messages: [original] });

    const restored = stateSnapshotToChannelValues(snapshot)[
      "messages"
    ] as BaseMessage[];
    expect(restored[0]?.content).toEqual(original.content);
  });
});
