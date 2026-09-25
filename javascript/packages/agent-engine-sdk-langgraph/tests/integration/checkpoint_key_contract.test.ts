/**
 * Contract tests for agent-owned LangGraph checkpoint keys and exact database
 * overrides against a real MongoDB.
 *
 * The suspend/resume cases drive the real AERServer path with a real
 * MongoDBSaver. The database cases verify the saver and query plugin use the
 * exact CHECKPOINT_DB_NAME without project-scoped rewriting.
 *
 * Gated on TEST_MONGO_URI (exported by scripts/test.sh).
 */

import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  interrupt,
} from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import Fastify from "fastify";
import {
  AERServer,
  resetHooks,
  type ExecuteRequest,
} from "@mongodb-js/agent-engine-runner-shared";
import {
  describe,
  test,
  expect,
  beforeAll,
  beforeEach,
  afterEach,
  afterAll,
  vi,
} from "vitest";
import { App } from "../../src/runtime.js";
import { LangGraphBaseAgent } from "../../src/agent.js";

const TEST_MONGO_URI = process.env["TEST_MONGO_URI"] ?? "";
const TIMEOUT = 30_000;
const ENV_KEYS = [
  "RUNNER_MODE",
  "MONGODB_URI",
  "CHECKPOINT_DB_NAME",
  "MDB_AGENTIC_STORE_DB",
  "APP_ID",
  "REQUIRE_PROJECT_SCOPED_DB",
] as const;

type SaverClient = ConstructorParameters<typeof MongoDBSaver>[0]["client"];

interface AerServerPrivates {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent: (opts?: unknown) => LangGraphBaseAgent;
    toolDefinitions?: Record<string, never>;
  };
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
  chunkSeq: Map<string, number>;
}

interface QueryServerPrivates {
  runtime: unknown;
  registerRoutes: (app: ReturnType<typeof Fastify>) => void;
}

function mustGet<T>(value: T | undefined | null): T {
  expect(value).not.toBeNull();
  expect(value).toBeDefined();
  if (value === undefined || value === null) {
    throw new Error("expected a value");
  }
  return value;
}

function makeAerServer(agent: LangGraphBaseAgent): AerServerPrivates {
  const server = Object.create(AERServer.prototype) as AerServerPrivates;
  server.runtime = {
    graphBuilder: {},
    orgId: null,
    getAgent: () => agent,
    toolDefinitions: {},
  };
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  return server;
}

function makeQueryApp(): ReturnType<typeof Fastify> {
  const server = Object.create(
    AERServer.prototype,
  ) as unknown as QueryServerPrivates;
  server.runtime = { getAgent: () => ({}), toolDefinitions: {} };
  const app = Fastify({ logger: false });
  server.registerRoutes(app);
  return app;
}

function interruptGraph(saver: MongoDBSaver) {
  return new StateGraph(MessagesAnnotation)
    .addNode("gate", async () => {
      const decision = interrupt({
        suspend_reason: "awaiting_review",
      }) as { decision?: string };
      return {
        messages: [
          new AIMessage({ content: `resumed:${decision.decision ?? ""}` }),
        ],
      };
    })
    .addEdge(START, "gate")
    .addEdge("gate", END)
    .compile({ checkpointer: saver });
}

function echoGraph(saver: MongoDBSaver) {
  return new StateGraph(MessagesAnnotation)
    .addNode("respond", async () => ({
      messages: [new AIMessage({ content: "persisted" })],
    }))
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile({ checkpointer: saver });
}

function suspendedMetadata(server: AerServerPrivates): Record<string, unknown> {
  const suspendedCall = server.reportCallback.mock.calls.find(
    (call) => call[2] === "SUSPENDED",
  );
  const fields = mustGet(suspendedCall)[3] as {
    metadata?: Record<string, unknown>;
  };
  return mustGet(fields.metadata);
}

describe.skipIf(TEST_MONGO_URI === "")(
  "LangGraph checkpoint key and database contract (real MongoDB)",
  () => {
    let client: MongoClient;
    let dbName: string;
    let savedEnv: Partial<Record<(typeof ENV_KEYS)[number], string>>;

    beforeAll(async () => {
      client = new MongoClient(TEST_MONGO_URI);
      await client.connect();
    });

    afterAll(async () => {
      await client.close();
    });

    beforeEach(() => {
      dbName = `checkpoint_key_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
      savedEnv = {};
      for (const key of ENV_KEYS) {
        const value = process.env[key];
        if (value !== undefined) savedEnv[key] = value;
        delete process.env[key];
      }
      process.env["RUNNER_MODE"] = "aer";
      process.env["MONGODB_URI"] = TEST_MONGO_URI;
    });

    afterEach(async () => {
      resetHooks();
      await client.db(dbName).dropDatabase();
      for (const key of ENV_KEYS) {
        const value = savedEnv[key];
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    });

    test(
      "uses a custom resolver key verbatim through suspend and resume",
      async () => {
        const saver = new MongoDBSaver({
          client: client as unknown as SaverClient,
          dbName,
        });
        const graph = interruptGraph(saver);
        const agent = new LangGraphBaseAgent(
          graph as never,
          undefined,
          null,
          (ctx) => `${ctx.sessionId}__actor`,
        );
        const server = makeAerServer(agent);
        const sessionId = "session-custom";
        const threadId = `${sessionId}__actor`;

        const suspendResponse = (await server.doHandleExecute({
          execution_id: "exec-custom-key",
          message: "please review",
          platform_api_url: "http://oe:8000",
          resume: false,
          previous_execution_cancelled: false,
          session_id: sessionId,
          user_id: "user-custom",
          workspace_id: "workspace-custom",
        } as ExecuteRequest)) as { status: string };

        expect(suspendResponse.status).toBe("suspended");
        const metadata = suspendedMetadata(server);
        expect(
          await client
            .db(dbName)
            .collection(saver.checkpointCollectionName)
            .countDocuments({ thread_id: threadId }),
        ).toBeGreaterThan(0);

        const resumeResponse = (await server.doHandleExecute({
          execution_id: "exec-custom-key",
          message: "",
          platform_api_url: "http://oe:8000",
          resume: true,
          previous_execution_cancelled: false,
          resume_data: { decision: "approve" },
          metadata,
          session_id: sessionId,
          user_id: "user-custom",
          workspace_id: "workspace-custom",
        } as ExecuteRequest)) as { status: string; result?: string };

        expect(resumeResponse.status).toBe("completed");
        expect(resumeResponse.result).toBe("resumed:approve");
        expect(
          await client
            .db(dbName)
            .collection(saver.checkpointCollectionName)
            .countDocuments({ thread_id: threadId }),
        ).toBeGreaterThan(0);
      },
      TIMEOUT,
    );

    test(
      "preserves the default session and workspace key through suspend and resume",
      async () => {
        // Python: test_build_config_uses_default_scoped_thread_id.
        const saver = new MongoDBSaver({
          client: client as unknown as SaverClient,
          dbName,
        });
        const graph = interruptGraph(saver);
        const agent = new LangGraphBaseAgent(graph as never);
        const server = makeAerServer(agent);
        const sessionId = "session-default";
        const workspaceId = "workspace-default";
        const threadId = `${sessionId}:${workspaceId}`;

        const suspendResponse = (await server.doHandleExecute({
          execution_id: "exec-default-key",
          message: "please review",
          platform_api_url: "http://oe:8000",
          resume: false,
          previous_execution_cancelled: false,
          session_id: sessionId,
          user_id: "user-default",
          workspace_id: workspaceId,
        } as ExecuteRequest)) as { status: string };

        expect(suspendResponse.status).toBe("suspended");
        const metadata = suspendedMetadata(server);
        expect(
          await client
            .db(dbName)
            .collection(saver.checkpointCollectionName)
            .countDocuments({ thread_id: threadId }),
        ).toBeGreaterThan(0);
        expect(
          await client
            .db(dbName)
            .collection(saver.checkpointCollectionName)
            .countDocuments({ thread_id: sessionId }),
        ).toBe(0);

        const resumeResponse = (await server.doHandleExecute({
          execution_id: "exec-default-key",
          message: "",
          platform_api_url: "http://oe:8000",
          resume: true,
          previous_execution_cancelled: false,
          resume_data: { decision: "approve" },
          metadata,
          session_id: sessionId,
          user_id: "user-default",
          workspace_id: workspaceId,
        } as ExecuteRequest)) as { status: string; result?: string };

        expect(resumeResponse.status).toBe("completed");
        expect(resumeResponse.result).toBe("resumed:approve");
      },
      TIMEOUT,
    );

    test(
      "uses CHECKPOINT_DB_NAME exactly and does not create the base store DB",
      async () => {
        const exactDbName = `checkpoint_exact_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
        const baseDbName = `checkpoint_base_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
        process.env["CHECKPOINT_DB_NAME"] = exactDbName;
        process.env["MDB_AGENTIC_STORE_DB"] = baseDbName;

        const app = new App({
          appName: "checkpoint-db-contract",
          mongodbUri: TEST_MONGO_URI,
        });
        try {
          const saver = mustGet(app.checkpointer()?.native) as MongoDBSaver;
          await echoGraph(saver).invoke(
            { messages: [new HumanMessage("write to exact DB")] },
            { configurable: { thread_id: "exact-db-session" } },
          );

          expect(
            await client
              .db(exactDbName)
              .collection(saver.checkpointCollectionName)
              .countDocuments(),
          ).toBeGreaterThan(0);
          const databases = await client
            .db()
            .admin()
            .listDatabases({ nameOnly: true });
          expect(databases.databases.map(({ name }) => name)).not.toContain(
            baseDbName,
          );
        } finally {
          try {
            await app.close();
          } finally {
            await client.db(exactDbName).dropDatabase();
            await client.db(baseDbName).dropDatabase();
          }
        }
      },
      TIMEOUT,
    );

    test(
      "serves query history from the exact CHECKPOINT_DB_NAME",
      async () => {
        const exactDbName = `checkpoint_query_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
        process.env["CHECKPOINT_DB_NAME"] = exactDbName;

        const app = new App({
          appName: "checkpoint-query-contract",
          mongodbUri: TEST_MONGO_URI,
        });
        try {
          const saver = mustGet(app.checkpointer()?.native) as MongoDBSaver;
          await echoGraph(saver).invoke(
            { messages: [new HumanMessage("query exact DB")] },
            { configurable: { thread_id: "query-exact-session" } },
          );

          const privateApp = app as unknown as {
            registerQueryPlugin: () => void;
          };
          privateApp.registerQueryPlugin();
          const httpApp = makeQueryApp();
          try {
            const response = await httpApp.inject({
              method: "GET",
              url: "/query/sessions?session_ids=query-exact-session",
            });

            expect(response.statusCode).toBe(200);
            expect(response.json()).toMatchObject({
              sessions: [
                {
                  session_id: "query-exact-session",
                  first_message_preview: "query exact DB",
                },
              ],
            });
          } finally {
            await httpApp.close();
          }
        } finally {
          try {
            await app.close();
          } finally {
            await client.db(exactDbName).dropDatabase();
          }
        }
      },
      TIMEOUT,
    );
  },
);
