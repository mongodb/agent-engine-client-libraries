/**
 * Integration tests for LangGraphQueryPlugin against a real MongoDB.
 *
 * Unlike Python's stub-based tests/test_query.py, these seed checkpoint data
 * by running a real compiled LangGraph graph with a real `MongoDBSaver`, so
 * the `checkpoints` / `checkpoint_writes` documents are written through the
 * production serde. That covers the class of bug stubs structurally cannot:
 * serde format drift, aggregation behavior on real BSON types ($toDate on
 * ObjectId, Binary write values), and the JS saver's storage layout.
 *
 * Gated on TEST_MONGO_URI (exported by scripts/test.sh, which provisions an
 * atlas-local single-node replica set). Skipped when unset (--skip-mongo).
 */

import {
  describe,
  test,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "vitest";
import { MongoClient, Binary } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import Fastify from "fastify";
import {
  AERServer,
  registerQueryPlugin,
  resetHooks,
} from "@mongodb-js/agent-engine-runner-shared";
import { LangGraphQueryPlugin } from "../../src/query.js";

const TEST_MONGO_URI = process.env["TEST_MONGO_URI"] ?? "";
const TIMEOUT = 30_000;

/** Unwrap a possibly-undefined value, failing the test if absent. */
function mustGet<T>(value: T | undefined | null): T {
  expect(value).not.toBeNull();
  expect(value).toBeDefined();
  if (value === undefined || value === null) {
    throw new Error("expected a value");
  }
  return value;
}

// Casts bridge the mongodb@^7 (our dep) vs mongodb@^6 (the saver's pin)
// type mismatch — same approach as src/runtime.ts `checkpointer()`.
type SaverClient = ConstructorParameters<typeof MongoDBSaver>[0]["client"];

describe.skipIf(TEST_MONGO_URI === "")(
  "LangGraphQueryPlugin (real MongoDB)",
  () => {
    let client: MongoClient;
    let dbName: string;
    let saver: MongoDBSaver;
    let plugin: LangGraphQueryPlugin;

    beforeAll(async () => {
      client = new MongoClient(TEST_MONGO_URI);
      await client.connect();
    });

    afterAll(async () => {
      await client.close();
    });

    beforeEach(() => {
      dbName = `query_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
      saver = new MongoDBSaver({
        client: client as unknown as SaverClient,
        dbName,
      });
      plugin = new LangGraphQueryPlugin({ saver, client, dbName });
    });

    afterEach(async () => {
      await client.db(dbName).dropDatabase();
    });

    /** Compile a one-node graph that appends a canned AI reply. */
    function buildEchoGraph(reply: string) {
      return new StateGraph(MessagesAnnotation)
        .addNode("respond", async () => ({
          messages: [new AIMessage({ content: reply })],
        }))
        .addEdge(START, "respond")
        .addEdge("respond", END)
        .compile({ checkpointer: saver });
    }

    /** Run one conversation turn on `threadId` through a real graph. */
    async function runTurn(
      threadId: string,
      inputMessages: BaseMessage[],
      reply = "ok",
    ): Promise<void> {
      await buildEchoGraph(reply).invoke(
        { messages: inputMessages },
        { configurable: { thread_id: threadId } },
      );
    }

    // -------------------------------------------------------------------
    // getSummariesForSessions
    // -------------------------------------------------------------------

    test(
      "returns one summary per persisted session, omitting unknown ids",
      async () => {
        await runTurn("sess-a", [new HumanMessage("Hello from A")]);
        await runTurn("sess-b", [new HumanMessage("Hello from B")]);

        const result = await plugin.getSummariesForSessions([
          "sess-a",
          "sess-b",
          "sess-unknown",
        ]);

        expect(result.sessions).toHaveLength(2);
        const bySession = new Map(
          result.sessions.map((s) => [s.session_id, s]),
        );
        expect([...bySession.keys()].sort()).toEqual(["sess-a", "sess-b"]);

        const sessA = mustGet(bySession.get("sess-a"));
        expect(sessA.message_count).toBeGreaterThanOrEqual(1);
        expect(sessA.first_message_preview).toBe("Hello from A");
        // ISO-8601 timestamps derived from the auto _id ObjectIds.
        expect(new Date(sessA.created_at).getTime()).not.toBeNaN();
        expect(new Date(sessA.last_activity).getTime()).not.toBeNaN();
        expect(sessA.last_activity >= sessA.created_at).toBe(true);
      },
      TIMEOUT,
    );

    test(
      "scopes the aggregation to the requested session ids",
      async () => {
        await runTurn("sess-only", [new HumanMessage("mine")]);
        await runTurn("sess-other", [new HumanMessage("not requested")]);

        const result = await plugin.getSummariesForSessions(["sess-only"]);

        expect(result.sessions.map((s) => s.session_id)).toEqual(["sess-only"]);
      },
      TIMEOUT,
    );

    test(
      "truncates the first human preview to 80 chars",
      async () => {
        const longMessage = "x".repeat(300);
        await runTurn("sess-long", [new HumanMessage(longMessage)]);

        const result = await plugin.getSummariesForSessions(["sess-long"]);

        expect(result.sessions[0]?.first_message_preview).toBe(
          longMessage.slice(0, 80),
        );
      },
      TIMEOUT,
    );

    test(
      "preview skips non-human messages and finds the first human one",
      async () => {
        // First turn seeds an AI-authored message; the human message arrives
        // in a later write. The preview must skip the AI content.
        await runTurn("sess-mixed", [new AIMessage("I am a bot opener")]);
        await runTurn("sess-mixed", [new HumanMessage("Real question")]);

        const result = await plugin.getSummariesForSessions(["sess-mixed"]);

        expect(result.sessions[0]?.first_message_preview).toBe("Real question");
      },
      TIMEOUT,
    );

    test("returns empty response for empty input without touching the DB", async () => {
      const result = await plugin.getSummariesForSessions([]);
      expect(result).toEqual({ sessions: [] });
    });

    test(
      "skips a corrupt preview write and keeps serving the session",
      async () => {
        // A corrupt write that sorts oldest (inserted before the real turn)
        // must be skipped — not fail the whole summary — and the real human
        // message must still surface as the preview.
        await client
          .db(dbName)
          .collection(saver.checkpointWritesCollectionName)
          .insertOne({
            thread_id: "sess-corrupt",
            checkpoint_ns: "",
            checkpoint_id: "0",
            task_id: "corrupt-task",
            idx: 0,
            channel: "messages",
            type: "json",
            value: new Binary(Buffer.from("{not json", "utf8")),
          });
        await runTurn("sess-corrupt", [
          new HumanMessage("survives corruption"),
        ]);

        const result = await plugin.getSummariesForSessions(["sess-corrupt"]);

        expect(result.sessions).toHaveLength(1);
        expect(result.sessions[0]?.first_message_preview).toBe(
          "survives corruption",
        );
      },
      TIMEOUT,
    );

    test(
      "message_count matches the messages endpoint for a session with tool calls",
      async () => {
        // A tool-calling turn writes several checkpoints (one per graph
        // super-step) while adding only a handful of messages. `message_count`
        // must report the message count, not the checkpoint count, so it agrees
        // with getMessagesForSession.
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("respond", async () => ({
            messages: [
              new AIMessage({
                content: "",
                tool_calls: [
                  {
                    id: "tc-1",
                    name: "my_tool",
                    args: { city: "Tokyo" },
                    type: "tool_call",
                  },
                ],
              }),
            ],
          }))
          .addNode("tool", async () => ({
            messages: [
              new ToolMessage({
                content: "tool output",
                tool_call_id: "tc-1",
                name: "my_tool",
              }),
            ],
          }))
          .addNode("finish", async () => ({
            messages: [new AIMessage({ content: "done" })],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", "tool")
          .addEdge("tool", "finish")
          .addEdge("finish", END)
          .compile({ checkpointer: saver });
        await graph.invoke(
          { messages: [new HumanMessage("use the tool")] },
          { configurable: { thread_id: "sess-tools" } },
        );

        const summary = await plugin.getSummariesForSessions(["sess-tools"]);
        const messages = await plugin.getMessagesForSession("sess-tools");

        // human + assistant(tool_call) + tool + assistant(done) = 4 messages,
        // even though the run persisted more than 4 checkpoints.
        expect(messages.messages).toHaveLength(4);
        expect(summary.sessions[0]?.message_count).toBe(
          messages.messages.length,
        );
      },
      TIMEOUT,
    );

    test(
      "message_count counts empty-content assistant messages",
      async () => {
        // An assistant message with empty content (e.g. a pure tool-call turn)
        // is still a message and must be counted.
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("respond", async () => ({
            messages: [new AIMessage({ content: "" })],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", END)
          .compile({ checkpointer: saver });
        await graph.invoke(
          { messages: [new HumanMessage("hi")] },
          { configurable: { thread_id: "sess-empty-ai" } },
        );

        const summary = await plugin.getSummariesForSessions(["sess-empty-ai"]);
        const messages = await plugin.getMessagesForSession("sess-empty-ai");

        expect(messages.messages).toHaveLength(2);
        expect(summary.sessions[0]?.message_count).toBe(
          messages.messages.length,
        );
      },
      TIMEOUT,
    );

    test(
      "one unreadable session degrades to message_count 0 without failing the list",
      async () => {
        await runTurn("sess-ok", [new HumanMessage("hi")]);
        await runTurn("sess-corrupt", [new HumanMessage("hi")]);
        // Corrupt every checkpoint payload for one session: the saver's serde
        // rejects on load, so its scan throws while the other session's stays
        // readable.
        await client
          .db(dbName)
          .collection(saver.checkpointCollectionName)
          .updateMany(
            { thread_id: "sess-corrupt" },
            { $set: { checkpoint: new Binary(Buffer.from("not-msgpack")) } },
          );

        const summary = await plugin.getSummariesForSessions([
          "sess-ok",
          "sess-corrupt",
        ]);

        const counts = new Map(
          summary.sessions.map((session) => [
            session.session_id,
            session.message_count,
          ]),
        );
        expect(counts.get("sess-corrupt")).toBe(0);
        expect(counts.get("sess-ok")).toBe(2);
      },
      TIMEOUT,
    );

    // -------------------------------------------------------------------
    // getMessagesForSession
    // -------------------------------------------------------------------

    test(
      "returns the decoded conversation history in order",
      async () => {
        await runTurn(
          "sess-conv",
          [new HumanMessage("First question")],
          "First answer",
        );
        await runTurn(
          "sess-conv",
          [new HumanMessage("Second question")],
          "Second answer",
        );

        const result = await plugin.getMessagesForSession("sess-conv");

        expect(result.messages.map((m) => m.role)).toEqual([
          "user",
          "assistant",
          "user",
          "assistant",
        ]);
        expect(result.messages.map((m) => m.content)).toEqual([
          "First question",
          "First answer",
          "Second question",
          "Second answer",
        ]);
        for (const message of result.messages) {
          expect(message.session_id).toBe("sess-conv");
          expect(message.id).not.toBe("");
          expect(new Date(message.timestamp).getTime()).not.toBeNaN();
        }
        // First-appearance attribution: turn-1 messages keep the timestamp
        // of the checkpoint they first appeared in, so they are STRICTLY
        // older than turn-2 messages (a broken attribution that stamps every
        // message with the same checkpoint timestamp must fail this).
        const firstTimestamp = mustGet(result.messages[0]).timestamp;
        const thirdTimestamp = mustGet(result.messages[2]).timestamp;
        expect(firstTimestamp < thirdTimestamp).toBe(true);
      },
      TIMEOUT,
    );

    test(
      "maps system and tool messages to framework-neutral roles",
      async () => {
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("respond", async () => ({
            messages: [
              new AIMessage({
                content: "calling tool",
                tool_calls: [
                  {
                    id: "tc-1",
                    name: "my_tool",
                    args: { city: "Tokyo" },
                    type: "tool_call",
                  },
                ],
              }),
              new ToolMessage({
                content: "tool output",
                tool_call_id: "tc-1",
                name: "my_tool",
              }),
            ],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", END)
          .compile({ checkpointer: saver });
        await graph.invoke(
          {
            messages: [new SystemMessage("be nice"), new HumanMessage("hi")],
          },
          { configurable: { thread_id: "sess-roles" } },
        );

        const result = await plugin.getMessagesForSession("sess-roles");

        expect(result.messages.map((m) => m.role)).toEqual([
          "system",
          "user",
          "assistant",
          "tool",
        ]);
        expect(result.messages[3]?.name).toBe("my_tool");
        expect(result.messages[3]?.content).toBe("tool output");
        expect(result.messages[2]?.tool_calls?.[0]?.id).toBe("tc-1");
        expect(result.messages[2]?.tool_calls?.[0]?.args).toEqual({
          city: "Tokyo",
        });
        expect(result.messages[3]?.tool_call_id).toBe("tc-1");
      },
      TIMEOUT,
    );

    test(
      "preserves tool call arguments alias",
      async () => {
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("respond", async () => ({
            messages: [
              // Some revived/tool-call producer shapes carry the sdk-core
              // alias even though LangChain's TS type only declares `args`.
              new AIMessage({
                content: "",
                tool_calls: [
                  {
                    id: "tc-alias",
                    name: "my_tool",
                    arguments: { city: "Kyoto" },
                    type: "tool_call",
                  },
                ],
              } as unknown as ConstructorParameters<typeof AIMessage>[0]),
            ],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", END)
          .compile({ checkpointer: saver });
        await graph.invoke(
          { messages: [new HumanMessage("call with alias args")] },
          { configurable: { thread_id: "sess-tool-arguments-alias" } },
        );

        const result = await plugin.getMessagesForSession(
          "sess-tool-arguments-alias",
        );

        const assistant = mustGet(
          result.messages.find((m) => m.role === "assistant"),
        );
        expect(assistant.tool_calls?.[0]?.args).toEqual({ city: "Kyoto" });
      },
      TIMEOUT,
    );

    test(
      "preserves JSON-safe additional_kwargs on messages",
      async () => {
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("respond", async () => ({
            messages: [
              new AIMessage({
                content: "with artifact",
                additional_kwargs: { artifact: { kind: "chart", rows: 3 } },
              }),
            ],
          }))
          .addEdge(START, "respond")
          .addEdge("respond", END)
          .compile({ checkpointer: saver });
        await graph.invoke(
          { messages: [new HumanMessage("draw")] },
          { configurable: { thread_id: "sess-kwargs" } },
        );

        const result = await plugin.getMessagesForSession("sess-kwargs");

        const aiMessage = mustGet(
          result.messages.find((m) => m.role === "assistant"),
        );
        expect(aiMessage.additional_kwargs).toEqual({
          artifact: { kind: "chart", rows: 3 },
        });
      },
      TIMEOUT,
    );

    test(
      "returns an empty list for a session with no checkpoints",
      async () => {
        const result = await plugin.getMessagesForSession("sess-nonexistent");
        expect(result).toEqual({ messages: [] });
      },
      TIMEOUT,
    );

    // -------------------------------------------------------------------
    // Full stack: AER routes + real plugin + real MongoDB
    // -------------------------------------------------------------------

    describe("through the AER HTTP routes", () => {
      afterEach(() => {
        resetHooks();
      });

      function makeApp() {
        // Cast through unknown: the query routes never touch the agent, so the
        // runtime stub deliberately satisfies only what registerRoutes reads.
        const server = Object.create(AERServer.prototype) as unknown as {
          runtime: unknown;
          registerRoutes: (app: ReturnType<typeof Fastify>) => void;
        };
        server.runtime = { getAgent: () => ({}), toolDefinitions: {} };
        const app = Fastify({ logger: false });
        server.registerRoutes(app);
        return app;
      }

      test(
        "GET /query/sessions returns summaries seeded by a real graph run",
        async () => {
          await runTurn("sess-http", [new HumanMessage("over the wire")]);
          registerQueryPlugin(plugin);
          const app = makeApp();

          const resp = await app.inject({
            method: "GET",
            url: "/query/sessions?session_ids=sess-http",
          });

          expect(resp.statusCode).toBe(200);
          const body = resp.json() as {
            sessions: Array<Record<string, unknown>>;
          };
          expect(body.sessions).toHaveLength(1);
          expect(body.sessions[0]).toMatchObject({
            session_id: "sess-http",
            first_message_preview: "over the wire",
          });
          expect(typeof body.sessions[0]?.["message_count"]).toBe("number");
        },
        TIMEOUT,
      );

      test(
        "GET /query/sessions/:id/messages returns the conversation",
        async () => {
          await runTurn("sess-http-msgs", [new HumanMessage("ping")], "pong");
          registerQueryPlugin(plugin);
          const app = makeApp();

          const resp = await app.inject({
            method: "GET",
            url: "/query/sessions/sess-http-msgs/messages",
          });

          expect(resp.statusCode).toBe(200);
          const body = resp.json() as {
            messages: Array<Record<string, unknown>>;
          };
          expect(body.messages.map((m) => [m["role"], m["content"]])).toEqual([
            ["user", "ping"],
            ["assistant", "pong"],
          ]);
        },
        TIMEOUT,
      );
    });
  },
);
