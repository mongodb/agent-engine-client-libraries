/**
 * Pin the TypeScript drain receiver to the shared contract fixture.
 *
 * Mirrors Python's tests/unit/test_drain_contract.py: both suites execute the
 * same vectors from `client-libraries/test-fixtures/drain/contract.json`, so
 * the two receivers cannot drift on wire behavior.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { afterAll, beforeAll, describe, expect, test } from "vitest";

import { ToolServer } from "../../src/server/tool.js";
import type { ITenantRuntime } from "../../src/server/base.js";

interface ContractCase {
  name: string;
  setup?: "none" | "ended" | "active";
  request: Record<string, unknown> & { execution_id: string };
  expect: { status: number; outcome?: string; reason_code?: string };
}

const fixturePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../../../../test-fixtures/drain/contract.json",
);
const contract = JSON.parse(readFileSync(fixturePath, "utf8")) as {
  cases: ContractCase[];
};

function fakeRuntime(): ITenantRuntime {
  return {
    appName: "test-app",
    orgId: null,
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () =>
      ({ featureEnabled: () => false }) as unknown as ReturnType<
        ITenantRuntime["getAgentConfig"]
      >,
    getMongodbUri: () => null,
    getAgent: () => {
      throw new Error("no agent in contract tests");
    },
  };
}

describe("drain contract fixture", () => {
  beforeAll(() => {
    process.env["RUNNER_AUTH_TOKEN"] = "s3cret";
    process.env["APP_ID"] = "ws-1";
  });
  afterAll(() => {
    delete process.env["RUNNER_AUTH_TOKEN"];
    delete process.env["APP_ID"];
  });

  for (const contractCase of contract.cases) {
    test(contractCase.name, async () => {
      const server = new ToolServer(fakeRuntime());
      const app = server.createApp();
      await app.ready();

      const executionId = contractCase.request.execution_id;
      if (contractCase.setup === "ended" || contractCase.setup === "active") {
        server.drainRegistry.beginWork(executionId);
      }
      if (contractCase.setup === "ended") {
        server.drainRegistry.endWork(executionId);
      }

      const request = { ...contractCase.request };
      if (typeof request["deadline_offset_ms"] === "number") {
        request["deadline_at_ms"] = Date.now() + request["deadline_offset_ms"];
        delete request["deadline_offset_ms"];
      }

      const resp = await app.inject({
        method: "POST",
        url: "/drain",
        headers: { authorization: "Bearer s3cret" },
        payload: request,
      });

      expect(resp.statusCode, resp.body).toBe(contractCase.expect.status);
      if (contractCase.expect.outcome !== undefined) {
        expect(resp.json().outcome).toBe(contractCase.expect.outcome);
      }
      if (contractCase.expect.reason_code !== undefined) {
        expect(resp.json().reason_code).toBe(contractCase.expect.reason_code);
      }
      await app.close();
    });
  }
});
