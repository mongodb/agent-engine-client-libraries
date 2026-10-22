import { afterEach, describe, expect, it, vi } from "vitest";
import type { FastifyInstance } from "fastify";

import { RuntimeAgentConfig } from "../../../src/agent_config.js";
import { onStartupComplete } from "../../../src/startup_state.js";
import { BaseServer, type ITenantRuntime } from "../../../src/server/base.js";

class TestServer extends BaseServer {
  get modeName(): string {
    return "test";
  }

  registerRoutes(_app: FastifyInstance): void {}
}

const runtime = {
  appName: "test",
  orgId: null,
  graphBuilder: null,
  tools: {},
  toolDefinitions: {},
  getAgentConfig: () => new RuntimeAgentConfig(),
  getMongodbUri: () => null,
  getAgent: () => {
    throw new Error("not used");
  },
} as ITenantRuntime;

describe("BaseServer startup completion", () => {
  afterEach(async () => {
    vi.restoreAllMocks();
  });

  it("marks startup complete only after the listener binds", async () => {
    const server = new TestServer(runtime);
    const onComplete = vi.fn();
    const unsubscribe = onStartupComplete(onComplete);

    try {
      await server.run("127.0.0.1", 0);

      expect(onComplete).toHaveBeenCalledOnce();
    } finally {
      unsubscribe();
      await server.close();
    }
  });

  it("does not mark startup complete when listener binding fails", async () => {
    const server = new TestServer(runtime);
    const onComplete = vi.fn();
    const unsubscribe = onStartupComplete(onComplete);

    try {
      await expect(
        server.run("127.0.0.1", Number.MAX_SAFE_INTEGER),
      ).rejects.toThrow();

      expect(onComplete).not.toHaveBeenCalled();
    } finally {
      unsubscribe();
      await server.close();
    }
  });
});
