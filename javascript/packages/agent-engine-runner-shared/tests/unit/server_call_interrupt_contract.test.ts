/**
 * Pin the TypeScript per-call interrupt receiver to the shared contract
 * fixture.
 *
 * Mirrors Python's tests/unit/test_call_interrupt_contract.py: both suites
 * execute the same vectors from
 * `client-libraries/test-fixtures/interrupt-call/contract.json`, so the two
 * receivers cannot drift on wire behavior.
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
  request: Record<string, unknown> & {
    execution_id: string;
    step_number: number;
  };
  expect: { status: number; outcome?: string };
}

const fixturePath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../../../../test-fixtures/interrupt-call/contract.json",
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

describe("call-interrupt contract fixture", () => {
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

      const { execution_id: executionId, step_number: stepNumber } =
        contractCase.request;
      if (contractCase.setup === "ended" || contractCase.setup === "active") {
        server.drainRegistry.beginWork(executionId, undefined, stepNumber);
      }
      if (contractCase.setup === "ended") {
        server.drainRegistry.endWork(executionId, undefined, stepNumber);
      }

      const resp = await app.inject({
        method: "POST",
        url: "/interrupt/call",
        headers: { authorization: "Bearer s3cret" },
        payload: contractCase.request,
      });

      expect(resp.statusCode).toBe(contractCase.expect.status);
      if (contractCase.expect.outcome !== undefined) {
        expect(resp.json().outcome).toBe(contractCase.expect.outcome);
      }
    });
  }
});
