/**
 * /health route tests: the tracing degraded/attached/disabled mapping onto the
 * HTTP response. Tracing state is driven through the real
 * setupTracing() path (mocked MongoDB driver) and assertions hit the route via
 * Fastify inject — no internal flag inspection, so a broken health-response
 * mapping or lazy-exporter wiring fails here even when unit tests pass.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { trace } from "@opentelemetry/api";

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

import { BaseServer } from "../../src/server/base.js";
import {
  setupTracing,
  shutdownTracing,
  type ITenantRuntime,
} from "../../src/index.js";

const URI = "mongodb://user:pass@host:27017";

class TestServer extends BaseServer {
  get modeName(): string {
    return "aer";
  }

  registerRoutes(): void {}

  getHealthDetails(): Record<string, unknown> {
    return { tools_registered: 2 };
  }
}

function makeServer(): TestServer {
  return new TestServer({ appName: "test-app" } as unknown as ITenantRuntime);
}

describe("/health tracing status mapping", () => {
  beforeEach(() => {
    mongo.instance.connect.mockResolvedValue(undefined);
    mongo.instance.close.mockResolvedValue(undefined);
    mongo.instance.db.mockReturnValue(mongo.dbObj);
    mongo.dbObj.collection.mockReturnValue(mongo.collection);
    mongo.dbObj.command.mockReset();
  });

  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
  });

  test("degraded store -> HTTP 200, status=degraded, tracing detail nested, mode details preserved", async () => {
    mongo.dbObj.command.mockRejectedValue(new Error("still down"));
    await setupTracing({
      serviceName: "svc-health-degraded",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    const app = makeServer().createApp();
    const res = await app.inject({ method: "GET", url: "/health" });
    await app.close();

    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe("degraded");
    expect(body.details.tracing).toEqual({ database_exporter: "degraded" });
    expect(body.details.tools_registered).toBe(2);
    expect(body.component).toBe("test-app");
    expect(body.mode).toBe("aer");
    expect(body.version).toBe("1.0.0");
  });

  test("attached store -> HTTP 200, status=healthy, tracing=attached", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-health-attached",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    const app = makeServer().createApp();
    const res = await app.inject({ method: "GET", url: "/health" });
    await app.close();

    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe("healthy");
    expect(body.details.tracing).toEqual({ database_exporter: "attached" });
    expect(body.details.tools_registered).toBe(2);
  });

  test("no store configured -> HTTP 200, status=healthy, tracing=disabled", async () => {
    await setupTracing({ serviceName: "svc-health-disabled" });

    const app = makeServer().createApp();
    const res = await app.inject({ method: "GET", url: "/health" });
    await app.close();

    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe("healthy");
    expect(body.details.tracing).toEqual({ database_exporter: "disabled" });
    expect(body.details.tools_registered).toBe(2);
  });
});
