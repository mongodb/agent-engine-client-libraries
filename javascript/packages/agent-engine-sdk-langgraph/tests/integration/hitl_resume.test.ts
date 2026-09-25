/**
 * Integration test for the HITL suspend → resume round-trip with the
 * checkpoint id traveling ONLY inside the opaque metadata dict.
 *
 * Drives the real AERServer execute path with a real LangGraph graph that
 * calls `interrupt()`, backed by a real `MongoDBSaver`:
 *
 *   1. Execute — the graph interrupts; the SUSPENDED callback must carry
 *      the adapter-built metadata (real checkpoint_id) and NO top-level
 *      checkpoint_id.
 *   2. Resume — a new ExecuteRequest with `resume: true` and the captured
 *      metadata (simulating the OE's verbatim round-trip, which
 *      TestDispatcher_InvokeSuspendResumeRoundTrip covers on the Go side)
 *      must restore the graph from the checkpoint and complete.
 *
 * Gated on TEST_MONGO_URI (exported by scripts/test.sh).
 */

import { describe, test, expect, beforeAll, afterAll, vi } from "vitest";
import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  interrupt,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import {
  AERServer,
  type ExecuteRequest,
} from "@mongodb-js/agent-engine-runner-shared";
import { LangGraphBaseAgent } from "../../src/agent.js";

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

describe.skipIf(TEST_MONGO_URI === "")(
  "HITL suspend → resume via metadata (real MongoDB)",
  () => {
    let client: MongoClient;
    const dbName = `hitl_it_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;

    beforeAll(async () => {
      client = new MongoClient(TEST_MONGO_URI);
      await client.connect();
    });

    afterAll(async () => {
      await client.db(dbName).dropDatabase();
      await client.close();
    });

    test(
      "surfaces and resumes parallel interrupts from real LangGraph",
      async () => {
        const saver = new MongoDBSaver({
          client: client as unknown as SaverClient,
          dbName,
        });
        const receivedAnswers: Record<string, string> = {};
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("fanout", async () => ({}))
          .addNode("payment_review", async () => {
            const answer = interrupt({
              node: "payment_review",
              amount: 100,
            }) as string;
            receivedAnswers["payment_review"] = answer;
            return {
              messages: [new AIMessage({ content: `payment:${answer}` })],
            };
          })
          .addNode("refund_review", async () => {
            const answer = interrupt({
              node: "refund_review",
              amount: 50,
            }) as string;
            receivedAnswers["refund_review"] = answer;
            return {
              messages: [new AIMessage({ content: `refund:${answer}` })],
            };
          })
          .addNode("finish", async () => ({
            messages: [
              new AIMessage({
                content: `completed:${receivedAnswers["payment_review"]}|${receivedAnswers["refund_review"]}`,
              }),
            ],
          }))
          .addEdge(START, "fanout")
          .addEdge("fanout", "payment_review")
          .addEdge("fanout", "refund_review")
          .addEdge("payment_review", "finish")
          .addEdge("refund_review", "finish")
          .addEdge("finish", END)
          .compile({ checkpointer: saver });
        const agent = new LangGraphBaseAgent(graph as never);
        const server = makeAerServer(agent);

        const suspendResponse = (await server.doHandleExecute({
          execution_id: "exec-hitl-parallel",
          message: "review both",
          platform_api_url: "http://oe:8000",
          resume: false,
          previous_execution_cancelled: false,
          thread_id: "thread-hitl-parallel",
        } as ExecuteRequest)) as { status: string };

        expect(suspendResponse.status).toBe("suspended");
        const suspendedCall = server.reportCallback.mock.calls.find(
          (call) => call[2] === "SUSPENDED",
        );
        const callbackFields = mustGet(suspendedCall)[3] as {
          interrupts?: Array<{ id: string; value: unknown }>;
          metadata?: Record<string, unknown>;
          resume_schema?: {
            required?: string[];
            properties?: {
              resume_map?: {
                required?: string[];
                properties?: Record<string, unknown>;
              };
            };
          };
        };
        const interrupts = mustGet(callbackFields.interrupts);
        expect(interrupts).toHaveLength(2);
        expect(interrupts.map(({ value }) => value)).toEqual(
          expect.arrayContaining([
            { node: "payment_review", amount: 100 },
            { node: "refund_review", amount: 50 },
          ]),
        );
        const interruptIds = interrupts.map(({ id }) => id);
        expect(interruptIds.every((id) => id.length > 0)).toBe(true);
        expect(new Set(interruptIds).size).toBe(2);

        const resumeMapSchema = mustGet(
          mustGet(callbackFields.resume_schema?.properties).resume_map,
        );
        expect(callbackFields.resume_schema?.required).toEqual(["resume_map"]);
        expect(new Set(resumeMapSchema.required)).toEqual(
          new Set(interruptIds),
        );
        expect(
          new Set(Object.keys(mustGet(resumeMapSchema.properties))),
        ).toEqual(new Set(interruptIds));

        const resumeMap = Object.fromEntries(
          interrupts.map(({ id, value }) => {
            const node = (value as { node: string }).node;
            return [id, node === "payment_review" ? "approve" : "deny"];
          }),
        );
        const resumeResponse = (await server.doHandleExecute({
          execution_id: "exec-hitl-parallel",
          message: "",
          platform_api_url: "http://oe:8000",
          resume: true,
          previous_execution_cancelled: false,
          resume_data: resumeMap,
          metadata: mustGet(callbackFields.metadata),
          thread_id: "thread-hitl-parallel",
        } as ExecuteRequest)) as { status: string; result?: string };

        expect(resumeResponse.status).toBe("completed");
        expect(resumeResponse.result).toBe("completed:approve|deny");
        expect(receivedAnswers).toEqual({
          payment_review: "approve",
          refund_review: "deny",
        });
      },
      TIMEOUT,
    );

    test(
      "round-trips the checkpoint id through metadata only",
      async () => {
        const saver = new MongoDBSaver({
          client: client as unknown as SaverClient,
          dbName,
        });
        const graph = new StateGraph(MessagesAnnotation)
          .addNode("gate", async () => {
            const decision = interrupt({
              suspend_reason: "awaiting_human_review",
              suspend_context: { claim_id: "CLM-9" },
            }) as { decision?: string };
            return {
              messages: [
                new AIMessage({ content: `resumed:${decision.decision}` }),
              ],
            };
          })
          .addEdge(START, "gate")
          .addEdge("gate", END)
          .compile({ checkpointer: saver });
        const agent = new LangGraphBaseAgent(graph as never);
        const server = makeAerServer(agent);

        // --- 1. Execute: the graph interrupts and the AER suspends. ---
        const suspendResponse = (await server.doHandleExecute({
          execution_id: "exec-hitl-1",
          message: "please review",
          platform_api_url: "http://oe:8000",
          resume: false,
          previous_execution_cancelled: false,
          thread_id: "thread-hitl-1",
        } as ExecuteRequest)) as {
          status: string;
          checkpoint_id?: string;
        };

        expect(suspendResponse.status).toBe("suspended");
        expect(suspendResponse.checkpoint_id).toBeUndefined();

        const suspendedCall = server.reportCallback.mock.calls.find(
          (c) => c[2] === "SUSPENDED",
        );
        const callbackFields = mustGet(suspendedCall)[3] as {
          checkpoint_id?: string;
          metadata?: Record<string, unknown>;
          suspend_reason?: string;
          interrupts?: Array<{ id: string; value: unknown }>;
        };
        // checkpoint_id travels ONLY inside the adapter-owned metadata.
        expect(callbackFields.checkpoint_id).toBeUndefined();
        expect(callbackFields.suspend_reason).toBe("awaiting_human_review");
        expect(mustGet(callbackFields.interrupts)[0]?.value).toEqual({
          suspend_reason: "awaiting_human_review",
          suspend_context: { claim_id: "CLM-9" },
        });
        const metadata = mustGet(callbackFields.metadata);
        expect(typeof metadata["checkpoint_id"]).toBe("string");
        expect(metadata["checkpoint_id"]).not.toBe("");

        // --- 2. Resume with the captured metadata (the OE round-trips it
        // verbatim) and no top-level checkpoint_id anywhere. ---
        const resumeResponse = (await server.doHandleExecute({
          execution_id: "exec-hitl-1",
          message: "",
          platform_api_url: "http://oe:8000",
          resume: true,
          previous_execution_cancelled: false,
          resume_data: { decision: "approve" },
          metadata,
          thread_id: "thread-hitl-1",
        } as ExecuteRequest)) as { status: string; result?: string };

        expect(resumeResponse.status).toBe("completed");
        expect(resumeResponse.result).toBe("resumed:approve");

        const completedCall = server.reportCallback.mock.calls.find(
          (c) => c[2] === "COMPLETED",
        );
        expect((mustGet(completedCall)[3] as { result?: unknown }).result).toBe(
          "resumed:approve",
        );
      },
      TIMEOUT,
    );
  },
);
