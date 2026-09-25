/** Durable activity synchronization with short-term Memory. */

import { create } from "@bufbuild/protobuf";
import { LLMResponse } from "@mongodb-js/agent-engine-sdk";

import {
  ActivityMemoryCommandSchema,
  MemoryWriteSchema,
  type ActivityContext,
  type ActivityMemoryCommand,
  type MemoryWrite,
} from "../generated/workflow/v1/activity_pb.js";
import type { WorkflowIdentity } from "../generated/workflow/v1/common_pb.js";
import { getLogger } from "../logger.js";
import { normalizeContent } from "../utils.js";
import { CALL_INTERRUPTED_ARTIFACT_KEY } from "../call_interrupted.js";

const logger = getLogger("agent_engine_runner_shared.workflow.memory");
const encoder = new TextEncoder();

/** JSON-serialize a plain tool result, degrading to String on failure. */
function safeJsonContent(result: object): string {
  try {
    return JSON.stringify(result) ?? String(result);
  } catch {
    return String(result);
  }
}

export interface WorkflowMemoryClient {
  ensureMemoryWritten(command: ActivityMemoryCommand): Promise<void>;
}

export function validateDurableMemoryIdentity(
  identity: WorkflowIdentity | undefined,
  userId: string | null | undefined,
): void {
  const scope = identity?.tenantScope;
  if (
    !userId ||
    !identity?.executionId ||
    !identity.sessionId ||
    !scope?.orgId ||
    !scope.projectId ||
    !scope.workspaceId
  ) {
    throw new Error("durable Memory identity is incomplete");
  }
}

interface TurnIdentity {
  session_id: string;
  org_id: string;
  user_id: string;
  project_id: string;
  agent_id: string;
  idempotency_key: string;
}

class DurableMemoryProjection {
  private constructor(
    readonly executionId: string,
    readonly activityId: string,
    readonly sessionId: string,
    readonly orgId: string,
    readonly projectId: string,
    readonly workspaceId: string,
    readonly userId: string,
  ) {}

  static fromActivity(
    context: ActivityContext,
    userId: string | null | undefined,
  ): DurableMemoryProjection {
    validateDurableMemoryIdentity(context.workflowIdentity, userId);
    if (!context.activityId || !context.workflowIdentity || !userId) {
      throw new Error("durable Memory identity is incomplete");
    }
    const scope = context.workflowIdentity.tenantScope;
    if (!scope) throw new Error("durable Memory identity is incomplete");
    return new DurableMemoryProjection(
      context.workflowIdentity.executionId,
      context.activityId,
      context.workflowIdentity.sessionId,
      scope.orgId,
      scope.projectId,
      scope.workspaceId,
      userId,
    );
  }

  userInput(content: string): MemoryWrite {
    const id = `workflow:${this.executionId}:input`;
    return this.write(id, {
      ...this.turnIdentity(id),
      role: "user",
      content,
    });
  }

  assistant(
    content: string,
    toolCalls: readonly Record<string, unknown>[] | null,
  ): MemoryWrite {
    const id = `workflow:${this.executionId}:${this.activityId}:assistant:0`;
    return this.write(id, {
      ...this.turnIdentity(id),
      role: "assistant",
      content,
      ...(toolCalls === null ? {} : { tool_calls: toolCalls }),
    });
  }

  tool(content: string, toolCallId: string, toolName: string): MemoryWrite {
    const id = `workflow:${this.executionId}:${this.activityId}:tool:0`;
    return this.write(id, {
      ...this.turnIdentity(id),
      role: "tool",
      content,
      tool_call_id: toolCallId,
      tool_name: toolName,
      is_error: false,
    });
  }

  private turnIdentity(id: string): TurnIdentity {
    return {
      session_id: this.sessionId,
      org_id: this.orgId,
      user_id: this.userId,
      project_id: this.projectId,
      agent_id: this.workspaceId,
      idempotency_key: id,
    };
  }

  private write(id: string, payload: Record<string, unknown>): MemoryWrite {
    return create(MemoryWriteSchema, {
      id,
      payloadJson: encoder.encode(JSON.stringify(payload)),
    });
  }
}

/** Own durable Memory projection, delivery, and pending user input. */
export class DurableMemoryState {
  private pendingUserMessage: string | null;
  private synchronization: Promise<void> = Promise.resolve();

  constructor(pendingUserMessage: string | null = null) {
    this.pendingUserMessage = pendingUserMessage;
  }

  synchronizeLlm(
    client: WorkflowMemoryClient,
    context: ActivityContext,
    result: unknown,
    userId: string | null | undefined,
  ): Promise<void> {
    const projection = DurableMemoryProjection.fromActivity(context, userId);
    const writes: MemoryWrite[] = [];
    try {
      if (
        typeof result !== "object" ||
        result === null ||
        Array.isArray(result)
      ) {
        throw new Error("LLM result must be an object");
      }
      const response = LLMResponse.fromRaw(result as Record<string, unknown>);
      const toolCalls = response.toolCalls?.map((toolCall) => {
        const args = toolCall.args ?? {};
        if (
          typeof args !== "string" &&
          (typeof args !== "object" || args === null || Array.isArray(args))
        ) {
          throw new Error("tool-call arguments must be an object or string");
        }
        return {
          id: toolCall.id ?? "",
          name: toolCall.name ?? "",
          arguments: args,
        };
      });
      if (!response.content && !toolCalls?.length) {
        throw new Error("assistant result has no content or tool calls");
      }
      writes.push(projection.assistant(response.content, toolCalls ?? null));
    } catch {
      logger.warn(
        { activity_id: context.activityId },
        "Skipping incompatible durable Memory LLM result",
      );
    }
    return this.synchronize(client, context, projection, writes);
  }

  synchronizeTool(
    client: WorkflowMemoryClient,
    context: ActivityContext,
    result: unknown,
    userId: string | null | undefined,
    toolCallId: string | undefined,
    toolName: string,
  ): Promise<void> {
    const projection = DurableMemoryProjection.fromActivity(context, userId);
    const writes: MemoryWrite[] = [];
    const interrupted =
      typeof result === "object" &&
      result !== null &&
      !Array.isArray(result) &&
      Object.keys(result).length === 1 &&
      (result as Record<string, unknown>)[CALL_INTERRUPTED_ARTIFACT_KEY] ===
        true;
    if (!interrupted) {
      // Plain structured results must survive as JSON (mirroring the
      // ToolMessage path): normalizeContent would reduce objects to
      // "[object Object]" and mangle arrays (multimodal concatenation),
      // losing the result Memory must preserve.
      const content =
        typeof result === "object" && result !== null
          ? safeJsonContent(result)
          : normalizeContent(result);
      if (toolCallId && content) {
        writes.push(projection.tool(content, toolCallId, toolName));
      } else if (toolCallId) {
        logger.warn(
          { activity_id: context.activityId },
          "Skipping incompatible durable Memory tool result",
        );
      }
    }
    return this.synchronize(client, context, projection, writes);
  }

  private synchronize(
    client: WorkflowMemoryClient,
    context: ActivityContext,
    projection: DurableMemoryProjection,
    resultWrites: MemoryWrite[],
  ): Promise<void> {
    const operation = this.synchronization.then(async () => {
      const writes =
        this.pendingUserMessage === null
          ? resultWrites
          : [projection.userInput(this.pendingUserMessage), ...resultWrites];
      await client.ensureMemoryWritten(
        create(ActivityMemoryCommandSchema, {
          workflowIdentity: context.workflowIdentity,
          activityId: context.activityId,
          attemptId: context.attemptId,
          fencingToken: context.fencingToken,
          memoryWrites: writes,
        }),
      );
      this.pendingUserMessage = null;
    });
    this.synchronization = operation.catch(() => {});
    return operation;
  }
}
