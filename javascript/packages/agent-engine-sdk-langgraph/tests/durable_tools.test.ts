import { create } from "@bufbuild/protobuf";
import { ToolMessage } from "@langchain/core/messages";
import {
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
  runWithOperationPathResolver,
} from "@mongodb-js/agent-engine-runner-shared";
import { describe, expect, it, vi } from "vitest";

import { withDurableToolResultIdentity } from "../src/durable_tools.js";

const attempt = create(AttemptContextSchema, {
  attemptId: "attempt-1",
  fencingToken: 1n,
  workflowIdentity: create(WorkflowIdentitySchema, {
    tenantScope: create(TenantScopeSchema, {
      orgId: "org",
      projectId: "project",
      workspaceId: "workspace",
    }),
    sessionId: "session",
    executionId: "execution",
  }),
});

describe("withDurableToolResultIdentity", () => {
  it("uses configurable.tool_call_id when ToolNode omits config.toolCall", async () => {
    const wrapped = vi.fn(async () => ["result", { source: "tool" }]);
    const tool = withDurableToolResultIdentity(wrapped, "lookup");

    const result = await runWithAttemptContext(attempt, () =>
      tool({}, { configurable: { tool_call_id: "call-1" } }),
    );

    expect(result).toEqual([expect.any(ToolMessage), null]);
    expect((result as [ToolMessage, null])[0]).toMatchObject({
      id: "durable-tool-result:execution:1:call-1",
      tool_call_id: "call-1",
      name: "lookup",
      artifact: { source: "tool" },
    });
  });

  it("scopes the message id to the Send worker that ran the tool", async () => {
    const wrapped = vi.fn(async () => ["result", null]);
    const tool = withDurableToolResultIdentity(wrapped, "lookup");
    const messageId = (worker: string): Promise<string | undefined> =>
      runWithOperationPathResolver(
        () => [{ name: worker, occurrenceKey: worker, spawnedTask: true }],
        async () => {
          const [message] = (await tool(
            {},
            { configurable: { tool_call_id: "call-1" } },
          )) as [ToolMessage, null];
          return message.id;
        },
      );

    const [first, second, again] = await runWithAttemptContext(
      attempt,
      async () => [
        await messageId("worker-0"),
        await messageId("worker-1"),
        await messageId("worker-0"),
      ],
    );

    // Two Send workers that see one ToolCall ID keep both results, and a
    // worker's result id is the same every time it is built.
    expect(first).toMatch(
      /^durable-tool-result:execution:1:call-1:[0-9a-f]{64}$/,
    );
    expect(second).not.toBe(first);
    expect(again).toBe(first);
  });
});
