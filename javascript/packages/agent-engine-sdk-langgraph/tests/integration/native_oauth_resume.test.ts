import { AIMessage, type ToolMessage } from "@langchain/core/messages";
import { tool as lcTool } from "@langchain/core/tools";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
  interrupt,
  isGraphBubbleUp,
} from "@langchain/langgraph";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  AERServer,
  createSecureToolFunction,
  registerSuspendHandler,
  resetHooks,
  type ExecuteRequest,
} from "@mongodb-js/agent-engine-runner-shared";
import { MongoClient } from "mongodb";
import { afterAll, describe, expect, test, vi } from "vitest";
import { z } from "zod";

import { LangGraphBaseAgent } from "../../src/agent.js";
import {
  withDurableToolResultIdentity,
  withOAuthTask,
} from "../../src/durable_tools.js";

const TEST_MONGO_URI = process.env["TEST_MONGO_URI"] ?? "";
const TIMEOUT = 30_000;

type SaverClient = ConstructorParameters<typeof MongoDBSaver>[0]["client"];

interface AerServerPrivates {
  runtime: unknown;
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
  chunkSeq: Map<string, number>;
  durableAckSenders: Map<string, unknown>;
}

interface SuspendedCallback {
  interrupts?: Array<{ id: string; value: Record<string, unknown> }>;
  resume_schema?: {
    properties?: { resume_map?: { required?: string[] } };
  };
  metadata?: Record<string, unknown>;
}

function fakeOe() {
  let consented = false;
  const providerCalls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: unknown, init?: RequestInit) => {
      if (!String(url).endsWith("/tool/execute")) {
        return new Response("{}", { status: 200 });
      }
      const body = JSON.parse(String(init?.body)) as {
        arguments: { owner: string };
        step_number: number;
      };
      if (consented) providerCalls.push(body.arguments.owner);
      const reply = consented
        ? {
            proceed: true,
            status: "success",
            result: JSON.stringify({ repos: ["alice-repo"] }),
            duration_ms: 1,
          }
        : {
            proceed: false,
            status: "suspend",
            latest_step_number: body.step_number,
            elicitation: {
              elicitation_id: "consent-alice",
              authorization_url: "https://idp.example/authorize",
              message: "Connect GitHub",
              created: true,
            },
          };
      return new Response(JSON.stringify(reply), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  return {
    consent: () => {
      consented = true;
    },
    providerCalls,
  };
}

async function startRunner(
  dbName: string,
  afterCall: () => Promise<void>,
): Promise<{ server: AerServerPrivates; close: () => Promise<void> }> {
  const client = new MongoClient(TEST_MONGO_URI);
  await client.connect();
  const saver = new MongoDBSaver({
    client: client as unknown as SaverClient,
    dbName,
  });
  const name = "list_repositories";
  const secure = createSecureToolFunction(
    { invoke: () => Promise.reject(new Error("OE runs provider tools")) },
    name,
    false,
    {
      isLocal: false,
      provider: "github",
      scopes: ["repo"],
      responseFormat: "content_and_artifact",
      isFrameworkControlFlow: isGraphBubbleUp,
    },
  );
  const repositories = lcTool(
    withDurableToolResultIdentity(withOAuthTask(secure, name), name) as never,
    {
      name,
      description: "List a user's repositories",
      schema: z.object({ owner: z.string() }),
      responseFormat: "content_and_artifact",
    },
  );
  const graph = new StateGraph(MessagesAnnotation)
    .addNode("repositories", async () => {
      const result = (await repositories.invoke({
        id: "call-alice",
        name,
        args: { owner: "alice" },
        type: "tool_call",
      })) as ToolMessage;
      await afterCall();
      return {
        messages: [
          result,
          new AIMessage({ content: `completed:${String(result.content)}` }),
        ],
      };
    })
    .addEdge(START, "repositories")
    .addEdge("repositories", END)
    .compile({ checkpointer: saver });
  const agent = new LangGraphBaseAgent(graph as never);
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
  server.durableAckSenders = new Map();
  return { server, close: () => client.close() };
}

function request(overrides: Partial<ExecuteRequest>): ExecuteRequest {
  return {
    execution_id: "exec-native-oauth",
    message: "list my repositories",
    platform_api_url: "http://oe:8000",
    resume: false,
    previous_execution_cancelled: false,
    durable_ack_enabled: false,
    stream_journal_enabled: false,
    stream_generation: 0,
    thread_id: "thread-native-oauth",
    ...overrides,
  } as ExecuteRequest;
}

describe.skipIf(TEST_MONGO_URI === "")(
  "native OAuth continuation across a runner replacement (real MongoDB)",
  () => {
    const dbName = `native_oauth_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const runners: Array<{ close: () => Promise<void> }> = [];

    afterAll(async () => {
      vi.unstubAllGlobals();
      resetHooks();
      const client = new MongoClient(TEST_MONGO_URI);
      await client.connect();
      await client.db(dbName).dropDatabase();
      await client.close();
      for (const runner of runners) await runner.close();
    });

    test(
      "returns the recorded API result without calling the provider again",
      async () => {
        registerSuspendHandler(interrupt as never);
        const oe = fakeOe();
        let recorded!: () => void;
        const callRecorded = new Promise<void>((resolve) => {
          recorded = resolve;
        });
        const first = await startRunner(dbName, () => {
          recorded();
          return new Promise<void>(() => {});
        });
        runners.push(first);

        const suspended = (await first.server.doHandleExecute(request({}))) as {
          status: string;
        };
        expect(suspended.status).toBe("suspended");
        const callback = first.server.reportCallback.mock.calls.find(
          (call) => call[2] === "SUSPENDED",
        )?.[3] as SuspendedCallback | undefined;
        const pending = callback?.interrupts ?? [];
        expect(pending).toHaveLength(1);
        const consentId = pending[0]?.id ?? "";
        expect(pending[0]?.value).toMatchObject({
          kind: "oauth_authorization",
          suspend_reason: "authorization_required",
        });
        expect(
          callback?.resume_schema?.properties?.resume_map?.required,
        ).toEqual([consentId]);

        oe.consent();
        const continuation = request({
          message: "",
          resume: true,
          resume_data: { [consentId]: { confirmed: true } },
          metadata: callback?.metadata ?? {},
        });
        void first.server.doHandleExecute(continuation);
        await callRecorded;
        expect(oe.providerCalls).toEqual(["alice"]);

        const replacement = await startRunner(dbName, () => Promise.resolve());
        runners.push(replacement);
        const completed = (await replacement.server.doHandleExecute(
          continuation,
        )) as { status: string; result?: string };

        expect(completed.status).toBe("completed");
        expect(completed.result).toBe(
          `completed:${JSON.stringify({ repos: ["alice-repo"] })}`,
        );
        expect(oe.providerCalls).toEqual(["alice"]);
      },
      TIMEOUT,
    );
  },
);
