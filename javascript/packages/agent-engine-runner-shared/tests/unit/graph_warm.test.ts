/** TypeScript warm-up compatibility behavior. */

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { FastifyInstance } from "fastify";

import {
  AERServer,
  type ITenantRuntime,
  type RuntimeAgentConfig,
} from "../../src/index.js";

interface GraphWarmServer {
  createApp: () => FastifyInstance;
  ensureOeRegistrations: () => Promise<void>;
  onStartup: () => Promise<void>;
  onShutdown: () => Promise<void>;
}

function makeServer(warmUpAgent: () => void): GraphWarmServer {
  const runtime = {
    appName: "test-agent",
    orgId: "org-test",
    projectId: "project-test",
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => ({}) as RuntimeAgentConfig,
    getMongodbUri: () => null,
    getAgent: () => {
      throw new Error("not used");
    },
    warmUpAgent,
  } as ITenantRuntime;
  const server = new AERServer(runtime) as unknown as GraphWarmServer;
  server.ensureOeRegistrations = vi.fn().mockResolvedValue(undefined);
  return server;
}

describe("AER graph warm-up", () => {
  beforeEach(() => {
    vi.stubEnv("OE_URL", "http://oe:8000");
    vi.stubEnv("APP_ID", "ws-test");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  test("AER startup leaves synchronous graph construction lazy", async () => {
    const warmUp = vi.fn();
    const server = makeServer(warmUp);

    await server.onStartup();

    expect(warmUp).not.toHaveBeenCalled();
    await server.onShutdown();
  });

  test("POST /warm-up is an immediate no-op", async () => {
    const warmUp = vi.fn();
    const server = makeServer(warmUp);
    const app = server.createApp();
    await app.ready();

    const response = await app.inject({ method: "POST", url: "/warm-up" });
    expect(response.statusCode).toBe(204);
    expect(warmUp).not.toHaveBeenCalled();
    await app.close();
  });
});
