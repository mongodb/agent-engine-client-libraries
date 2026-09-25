import { AIMessage } from "@langchain/core/messages";
import type { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import type { MongoClient } from "mongodb";
import { describe, expect, test } from "vitest";
import { LangGraphQueryPlugin } from "../src/query.js";

function fakeClient(): MongoClient {
  return {
    db: () => ({
      collection: () => ({}),
    }),
  } as unknown as MongoClient;
}

function fakeSaver(message: unknown): MongoDBSaver {
  return {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    list: async function* () {
      yield {
        checkpoint: {
          ts: "2026-01-02T03:04:05.000Z",
          channel_values: { messages: [message] },
        },
      };
    },
  } as unknown as MongoDBSaver;
}

/** Saver stub that records the thread_id of every `list` call. */
function recordingSaver(requestedThreadIds: string[]): MongoDBSaver {
  return {
    checkpointCollectionName: "checkpoints",
    checkpointWritesCollectionName: "checkpoint_writes",
    list: async function* (config: { configurable?: { thread_id?: string } }) {
      requestedThreadIds.push(config.configurable?.thread_id ?? "");
      // No checkpoints — the assertion is on which keys were queried.
      yield* [];
    },
  } as unknown as MongoDBSaver;
}

describe("LangGraphQueryPlugin", () => {
  test("round-trips tool call args into a JSON-safe shape", async () => {
    const message = new AIMessage({
      content: "",
      tool_calls: [
        {
          id: "tc-json",
          name: "my_tool",
          args: {
            city: "Kyoto",
            visited_at: new Date("2026-01-02T03:04:05.000Z"),
          },
          type: "tool_call",
        },
      ],
    } as unknown as ConstructorParameters<typeof AIMessage>[0]);
    const plugin = new LangGraphQueryPlugin({
      saver: fakeSaver(message),
      client: fakeClient(),
      dbName: "unused",
    });

    const result = await plugin.getMessagesForSession("sess-json");

    expect(result.messages[0]?.tool_calls?.[0]?.args).toEqual({
      city: "Kyoto",
      visited_at: "2026-01-02T03:04:05.000Z",
    });
  });

  test("queries only the workspace-scoped key when a scope is known", async () => {
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      workspaceIdResolver: () => "ws-1",
    });

    await plugin.getMessagesForSession("sess-1");

    // The bare unscoped key must never be queried once a scope exists —
    // it is readable/writable by every workspace on the shared store.
    expect(requested).toEqual(["sess-1:ws-1"]);
  });

  test("fails closed only when a resolver signals a genuinely unresolvable scope", async () => {
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      // A custom resolver returning null says "I cannot decide" — unlike "",
      // which is a deliberate explicitly-unscoped scope.
      workspaceIdResolver: () => null,
    });

    await expect(plugin.getMessagesForSession("sess-1")).rejects.toThrow(
      /workspace scope unavailable/,
    );
    expect(requested).toEqual([]);
  });

  test("an explicitly unscoped runtime (resolver returns '') reads bare keys", async () => {
    // Local dev / tests: no APP_ID, so writes are bare-keyed and reads must
    // match — this is the normal `agentengine dev up` history path.
    const requested: string[] = [];
    const plugin = new LangGraphQueryPlugin({
      saver: recordingSaver(requested),
      client: fakeClient(),
      dbName: "unused",
      workspaceIdResolver: () => "",
    });

    await plugin.getMessagesForSession("sess-1");

    expect(requested).toEqual(["sess-1"]);
  });
});
