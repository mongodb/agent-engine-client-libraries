import { create } from "@bufbuild/protobuf";
import { ToolMessage } from "@langchain/core/messages";
import {
  AttemptContextSchema,
  TenantScopeSchema,
  WorkflowIdentitySchema,
  runWithAttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";
import { describe, expect, it, vi } from "vitest";

import { withDurableToolResultIdentity } from "../src/durable_tools.js";

describe("withDurableToolResultIdentity", () => {
  it("uses configurable.tool_call_id when ToolNode omits config.toolCall", async () => {
    const wrapped = vi.fn(async () => ["result", { source: "tool" }]);
    const tool = withDurableToolResultIdentity(wrapped, "lookup");
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
});
