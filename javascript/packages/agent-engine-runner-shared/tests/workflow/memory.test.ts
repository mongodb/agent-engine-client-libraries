import { create } from "@bufbuild/protobuf";
import { describe, expect, test, vi } from "vitest";

import {
  ActivityContextSchema,
  type ActivityMemoryCommand,
} from "../../src/generated/workflow/v1/activity_pb.js";
import {
  TenantScopeSchema,
  WorkflowIdentitySchema,
} from "../../src/generated/workflow/v1/common_pb.js";
import { CALL_INTERRUPTED_ARTIFACT_KEY } from "../../src/secure_wrapper.js";
import {
  DurableMemoryState,
  validateDurableMemoryIdentity,
} from "../../src/workflow/memory.js";

function context() {
  return create(ActivityContextSchema, {
    workflowIdentity: create(WorkflowIdentitySchema, {
      tenantScope: create(TenantScopeSchema, {
        orgId: "org-1",
        projectId: "project-1",
        workspaceId: "workspace-1",
      }),
      sessionId: "session-1",
      executionId: "execution-1",
    }),
    activityId: "activity-1",
    attemptId: "attempt-1",
    fencingToken: 7n,
  });
}

function client() {
  const commands: ActivityMemoryCommand[] = [];
  return {
    commands,
    ensureMemoryWritten: vi.fn(async (command: ActivityMemoryCommand) => {
      commands.push(command);
    }),
  };
}

function payload(command: ActivityMemoryCommand, index: number) {
  const bytes = command.memoryWrites[index]?.payloadJson;
  if (!bytes) throw new Error("missing Memory payload");
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
}

function commandAt(commands: ActivityMemoryCommand[], index: number) {
  const command = commands[index];
  if (!command) throw new Error("missing Memory command");
  return command;
}

describe("DurableMemoryState", () => {
  test("writes pending input once and each LLM activity result", async () => {
    const memory = new DurableMemoryState("hello");
    const workflow = client();

    await memory.synchronizeLlm(
      workflow,
      context(),
      { content: "hi" },
      "user-1",
    );
    await memory.synchronizeLlm(
      workflow,
      context(),
      { content: "later" },
      "user-1",
    );

    expect(workflow.commands[0]?.memoryWrites.map((write) => write.id)).toEqual(
      [
        "workflow:execution-1:input",
        "workflow:execution-1:activity-1:assistant:0",
      ],
    );
    expect(payload(commandAt(workflow.commands, 0), 0)).toMatchObject({
      role: "user",
      content: "hello",
      session_id: "session-1",
      org_id: "org-1",
      user_id: "user-1",
      project_id: "project-1",
      agent_id: "workspace-1",
      idempotency_key: "workflow:execution-1:input",
    });
    expect(payload(commandAt(workflow.commands, 0), 1)).toMatchObject({
      role: "assistant",
      content: "hi",
    });
    expect(workflow.commands[1]?.memoryWrites.map((write) => write.id)).toEqual(
      ["workflow:execution-1:activity-1:assistant:0"],
    );
    expect(payload(commandAt(workflow.commands, 1), 0)).toMatchObject({
      content: "later",
    });
  });

  test("retains pending input when delivery fails", async () => {
    const memory = new DurableMemoryState("hello");
    const commands: ActivityMemoryCommand[] = [];
    let fail = true;
    const workflow = {
      ensureMemoryWritten: vi.fn(async (command: ActivityMemoryCommand) => {
        commands.push(command);
        if (fail) throw new Error("Memory unavailable");
      }),
    };

    await expect(
      memory.synchronizeLlm(
        workflow,
        context(),
        { content: "first" },
        "user-1",
      ),
    ).rejects.toThrow("Memory unavailable");
    fail = false;
    await memory.synchronizeLlm(
      workflow,
      context(),
      { content: "retry" },
      "user-1",
    );

    expect(commands[0]?.memoryWrites[0]?.id).toBe("workflow:execution-1:input");
    expect(commands[1]?.memoryWrites[0]?.id).toBe("workflow:execution-1:input");
  });

  test("serializes concurrent synchronization so pending input is written once", async () => {
    const memory = new DurableMemoryState("hello");
    const workflow = client();

    await Promise.all([
      memory.synchronizeLlm(workflow, context(), { content: "one" }, "user-1"),
      memory.synchronizeLlm(workflow, context(), { content: "two" }, "user-1"),
    ]);

    expect(
      workflow.commands
        .flatMap((command) => command.memoryWrites)
        .filter((write) => write.id === "workflow:execution-1:input"),
    ).toHaveLength(1);
  });

  test("skips incompatible LLM results without blocking acknowledgement", async () => {
    const memory = new DurableMemoryState("hello");
    const workflow = client();

    await memory.synchronizeLlm(
      workflow,
      context(),
      { content: "", tool_calls: [{ args: ["unsupported"] }] },
      "user-1",
    );

    expect(workflow.commands[0]?.memoryWrites.map((write) => write.id)).toEqual(
      ["workflow:execution-1:input"],
    );
  });

  test("does not publish the exact interrupted tool marker", async () => {
    const memory = new DurableMemoryState("approve the claim");
    const workflow = client();

    await memory.synchronizeTool(
      workflow,
      context(),
      { [CALL_INTERRUPTED_ARTIFACT_KEY]: true },
      "user-1",
      "call-1",
      "approve_claim",
    );

    expect(workflow.commands[0]?.memoryWrites).toHaveLength(1);
    expect(payload(commandAt(workflow.commands, 0), 0)).toMatchObject({
      role: "user",
      content: "approve the claim",
    });
  });

  test("keeps interrupted-shaped tool data with additional fields as content", async () => {
    const memory = new DurableMemoryState();
    const workflow = client();

    await memory.synchronizeTool(
      workflow,
      context(),
      { [CALL_INTERRUPTED_ARTIFACT_KEY]: true, some: "artifact" },
      "user-1",
      "call-1",
      "lookup",
    );

    expect(workflow.commands[0]?.memoryWrites).toHaveLength(1);
    expect(payload(commandAt(workflow.commands, 0), 0)).toMatchObject({
      role: "tool",
      tool_call_id: "call-1",
      tool_name: "lookup",
    });
  });

  test("serializes ordinary object tool results as JSON content", async () => {
    const memory = new DurableMemoryState();
    const workflow = client();

    await memory.synchronizeTool(
      workflow,
      context(),
      { temperature: 72 },
      "user-1",
      "call-1",
      "get_temperature",
    );

    expect(workflow.commands[0]?.memoryWrites).toHaveLength(1);
    expect(payload(commandAt(workflow.commands, 0), 0)).toMatchObject({
      role: "tool",
      tool_call_id: "call-1",
      tool_name: "get_temperature",
      content: JSON.stringify({ temperature: 72 }),
    });
  });

  test("serializes array tool results as JSON content", async () => {
    const memory = new DurableMemoryState();
    const workflow = client();

    await memory.synchronizeTool(
      workflow,
      context(),
      [1, 2],
      "user-1",
      "call-1",
      "list_policies",
    );

    expect(workflow.commands[0]?.memoryWrites).toHaveLength(1);
    expect(payload(commandAt(workflow.commands, 0), 0)).toMatchObject({
      role: "tool",
      tool_call_id: "call-1",
      tool_name: "list_policies",
      content: "[1,2]",
    });
  });
});

describe("validateDurableMemoryIdentity", () => {
  test("rejects incomplete attribution", () => {
    expect(() => validateDurableMemoryIdentity(undefined, "user-1")).toThrow(
      "durable Memory identity is incomplete",
    );
    expect(() =>
      validateDurableMemoryIdentity(context().workflowIdentity, null),
    ).toThrow("durable Memory identity is incomplete");
  });
});
