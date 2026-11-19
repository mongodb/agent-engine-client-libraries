import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import Fastify from "fastify";
import { context, propagation, trace } from "@opentelemetry/api";

const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

const mongo = vi.hoisted(() => {
  const collection = {
    insertMany: vi.fn().mockResolvedValue({ insertedCount: 1 }),
    dbName: "testdb",
    collectionName: "traces",
  };
  const dbObj = {
    command: vi.fn(),
    collection: vi.fn(() => collection),
  };
  const instance = {
    connect: vi.fn().mockResolvedValue(undefined),
    db: vi.fn(() => dbObj),
    close: vi.fn().mockResolvedValue(undefined),
  };
  return { instance, dbObj, collection };
});

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

vi.mock("mongodb", () => ({
  MongoClient: function () {
    return mongo.instance;
  },
}));

import { AERServer, setupTracing, shutdownTracing } from "../../src/index.js";
import type { ExecuteRequest } from "../../src/index.js";
import { forceFlushTracing } from "../../src/tracing/setup.js";
import type { ExecutorCallbackRequest } from "../../src/models.js";
import { AER_EXECUTE } from "../../src/span_names.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";

const URI = "mongodb://user:pass@host:27017";
const CALLER_TRACE_ID = "0af7651916cd43dd8448eb211c80319c";
const TRACEPARENT = `00-${CALLER_TRACE_ID}-b7ad6b7169203331-01`;

interface CallbackDeliveryPrivates {
  pending: Map<string, { oeUrl: string; callback: Record<string, unknown> }>;
  tasks: Map<string, Promise<void>>;
  retryController: AbortController;
}

function makeRouteServer(): AERServer {
  const server = Object.create(AERServer.prototype) as AERServer & {
    runtime: Record<string, unknown>;
    pendingSessionFinish: Map<string, ExecutorCallbackRequest>;
    callbackDelivery: CallbackDeliveryPrivates;
    handleExecute: ReturnType<typeof vi.fn>;
  };
  server.runtime = {};
  server.pendingSessionFinish = new Map();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  server.handleExecute = vi
    .fn()
    .mockResolvedValue({ status: "completed", result: "ok" });
  return server;
}

function executeRequest(executionId: string): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: "http://oe",
    resume: false,
    user_id: "u-1",
    session_id: "session-1",
    workspace_id: "workspace-1",
    thread_id: executionId,
  } as ExecuteRequest;
}

describe("AER run root survives pod teardown", () => {
  beforeEach(async () => {
    mongo.instance.connect.mockResolvedValue(undefined);
    mongo.instance.close.mockResolvedValue(undefined);
    mongo.instance.db.mockReturnValue(mongo.dbObj);
    mongo.dbObj.command.mockReset();
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    mongo.dbObj.collection.mockReturnValue(mongo.collection);
    mongo.collection.insertMany.mockReset();
    mongo.collection.insertMany.mockResolvedValue({ insertedCount: 1 });
    await setupTracing({
      serviceName: "aer-teardown",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });
  });

  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
    propagation.disable();
    context.disable();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  for (const [label, headers] of [
    ["with an inbound traceparent", { traceparent: TRACEPARENT }],
    ["without an inbound traceparent", {}],
  ] as const) {
    test(`shutdownTracing delivers the run root ${label}`, async () => {
      const server = makeRouteServer();
      const app = Fastify();
      server.registerRoutes(app);
      const res = await app.inject({
        method: "POST",
        url: "/execute",
        headers,
        payload: executeRequest("exec-root"),
      });
      await app.close();
      expect(res.statusCode).toBe(200);

      let releaseExport!: () => void;
      mongo.collection.insertMany.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            releaseExport = () => resolve({ insertedCount: 1 });
          }),
      );
      const dispatched = forceFlushTracing(5_000);
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(mongo.collection.insertMany).toHaveBeenCalledTimes(1);

      let settled = false;
      const shutdown = shutdownTracing().then(() => {
        settled = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(settled).toBe(false);

      releaseExport();
      await dispatched;
      await shutdown;
      expect(settled).toBe(true);
      const docs = mongo.collection.insertMany.mock.calls.flatMap(
        (call) => call[0] as Array<{ name: string }>,
      );
      expect(docs.some((doc) => doc.name === AER_EXECUTE)).toBe(true);
    });
  }
});
