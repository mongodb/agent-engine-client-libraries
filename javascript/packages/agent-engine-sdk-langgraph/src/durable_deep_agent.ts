/**
 * Durable operation-path boundaries for Deep Agent task delegation.
 *
 * Deep Agent exposes subagent dispatch through the public `task` tool and
 * middleware hooks, rather than as a compiled child graph:
 * https://docs.langchain.com/oss/javascript/deepagents/subagents
 *
 * Same-step `task` siblings stamp ToolCall-id ordinals after FinalizeStep.
 * Stamping in `afterModel` would key those ordinals to the model step, which
 * FinalizeStep then leaves behind.
 *
 * Port of `durable_deep_agent.py`.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { createMiddleware } from "langchain";
import type { Command } from "@langchain/langgraph";
import {
  type BaseMessage,
  isHumanMessage,
  isToolMessage,
  type ToolMessage,
} from "@langchain/core/messages";
import {
  childOperationBoundaryScope,
  currentAttemptContext,
  preallocateChildOperationOrdinals,
  currentPendingChildOperationBatch,
  setPendingChildOperationBatch,
  type ChildOperationBoundary,
} from "@mongodb-js/agent-engine-runner-shared";

import { parseTaskCall } from "./deep_agent_task.js";
import { UnsupportedDurableGraphError } from "./platform_checkpointer.js";

type ToolCallResult = ToolMessage | Command;

const activeTaskCallId = new AsyncLocalStorage<string>();

function messagesFromState(state: unknown): BaseMessage[] {
  if (state === null || typeof state !== "object") return [];
  const messages = (state as { messages?: unknown }).messages;
  return Array.isArray(messages) ? (messages as BaseMessage[]) : [];
}

/**
 * Assign a deterministic id to one message. LangChain's serializer emits only
 * keys present in `lc_kwargs`, so a late-added `id` must be mirrored there or
 * checkpointed messages silently lose it (same as `assignBaseMessageId`).
 */
function assignMessageId(message: BaseMessage, id: string): void {
  message.id = id;
  const kwargs = (message as { lc_kwargs?: Record<string, unknown> }).lc_kwargs;
  if (kwargs !== undefined && kwargs !== null && typeof kwargs === "object") {
    kwargs["id"] = id;
  }
}

/**
 * Replace Deep Agent's generated message ids before LangGraph persists them.
 * No-op outside a task scope, so native sessions keep provider ids.
 */
export function stampDeepAgentMessageIds(messages: BaseMessage[]): void {
  const taskCallId = activeTaskCallId.getStore();
  if (taskCallId === undefined) return;
  messages.forEach((message, index) => {
    // isHumanMessage/isToolMessage key off getType(), matching messages a
    // nested graph created through its own @langchain/core module copy —
    // instanceof would miss those.
    if (isHumanMessage(message)) {
      assignMessageId(
        message,
        `durable-deep-agent-input:${taskCallId}:${index}`,
      );
    } else if (isToolMessage(message)) {
      assignMessageId(
        message,
        `durable-deep-agent-tool-result:${taskCallId}:${message.tool_call_id}`,
      );
    }
  });
}

function stampTaskResultMessageId(
  result: ToolCallResult,
  toolCallId: string,
): void {
  if (isToolMessage(result)) {
    assignMessageId(result, `durable-deep-agent-task-result:${toolCallId}`);
    return;
  }
  const update = (result as { update?: unknown }).update;
  if (update === null || typeof update !== "object" || Array.isArray(update)) {
    return;
  }
  const messages = (update as Record<string, unknown>)["messages"];
  if (!Array.isArray(messages)) return;
  for (const message of messages) {
    if (isToolMessage(message) && message.tool_call_id === toolCallId) {
      assignMessageId(message, `durable-deep-agent-task-result:${toolCallId}`);
    }
  }
}

export function createDurableDeepAgentMiddleware(options: {
  unsupportedSubagentNames?: ReadonlySet<string>;
  retryPolicySubagentNames?: ReadonlySet<string>;
}) {
  const unsupportedSubagentNames =
    options.unsupportedSubagentNames ?? new Set();
  const retryPolicySubagentNames =
    options.retryPolicySubagentNames ?? new Set();

  const boundaryFromTask = (
    toolCall: unknown,
  ): ChildOperationBoundary | null => {
    const taskCall = parseTaskCall(toolCall);
    if (taskCall === null) return null;
    if (taskCall.toolCallId === "") {
      throw new Error(
        "durable Deep Agent task requires a non-empty tool call ID",
      );
    }
    if (taskCall.subagentName === "") {
      throw new Error(
        "durable Deep Agent task requires a non-empty subagent type",
      );
    }
    if (unsupportedSubagentNames.has(taskCall.subagentName)) {
      throw new UnsupportedDurableGraphError(
        `compiled Deep Agent subagent '${taskCall.subagentName}' must use ` +
          "checkpointer=None; independent child checkpointers are not supported " +
          "during durable execution",
      );
    }
    if (retryPolicySubagentNames.has(taskCall.subagentName)) {
      throw new UnsupportedDurableGraphError(
        `compiled Deep Agent subagent '${taskCall.subagentName}' sets a ` +
          "retry policy; node retry policies are not supported during " +
          "durable execution",
      );
    }
    return {
      name: taskCall.subagentName,
      occurrenceKey: taskCall.toolCallId,
    };
  };

  const taskBoundary = (toolCall: unknown): ChildOperationBoundary | null => {
    if (currentAttemptContext() === null) return null;
    return boundaryFromTask(toolCall);
  };

  const taskBoundariesFromState = (
    state: unknown,
  ): ChildOperationBoundary[] => {
    const messages = messagesFromState(state);
    if (messages.length === 0) return [];
    const last = messages[messages.length - 1];
    const toolCalls =
      (last as { tool_calls?: unknown[] } | undefined)?.tool_calls ?? [];
    if (!Array.isArray(toolCalls)) return [];
    const boundaries: ChildOperationBoundary[] = [];
    for (const toolCall of toolCalls) {
      const candidate = boundaryFromTask(toolCall);
      if (candidate !== null) boundaries.push(candidate);
    }
    return boundaries;
  };

  const pendingBoundaries = (
    state: unknown,
  ): readonly ChildOperationBoundary[] => {
    const fromState = taskBoundariesFromState(state);
    if (fromState.length > 0) return fromState;
    return currentPendingChildOperationBatch();
  };

  const preallocatePendingTaskBatch = (
    toolCall: unknown,
    state: unknown,
  ): void => {
    // Phase 2: stamp on the ToolNode superstep, after FinalizeStep advanced
    // the root step counter. Skip if this wrap is not in the stashed batch
    // (sequential later-step task calls must first-seen-allocate).
    if (currentAttemptContext() === null) return;
    const current = boundaryFromTask(toolCall);
    if (current === null) return;
    const pending = pendingBoundaries(state);
    if (
      pending.length === 0 ||
      !pending.some(
        (candidate) =>
          candidate.name === current.name &&
          candidate.occurrenceKey === current.occurrenceKey,
      )
    ) {
      return;
    }
    preallocateChildOperationOrdinals(pending);
    setPendingChildOperationBatch([]);
  };

  return createMiddleware({
    name: "durableDeepAgentMiddleware",
    afterModel: (state) => {
      // Phase 1: validate and stash. Stamping here would key ordinals to the
      // model step, which FinalizeStep then leaves behind. Mutate the
      // attempt-scoped holder so Pregel worker copies see the batch.
      if (currentAttemptContext() === null) return undefined;
      setPendingChildOperationBatch(taskBoundariesFromState(state));
      return undefined;
    },
    wrapToolCall: (request, handler) => {
      preallocatePendingTaskBatch(request.toolCall, request.state);
      const boundary = taskBoundary(request.toolCall);
      if (boundary === null) return handler(request);
      return activeTaskCallId.run(boundary.occurrenceKey, () => {
        const result = childOperationBoundaryScope(boundary, () =>
          handler(request),
        );
        // Both branches may be a promise; stamp after settlement so the
        // returned Command/ToolMessage carries the deterministic id.
        if (result instanceof Promise) {
          return result.then((settled) => {
            stampTaskResultMessageId(settled, boundary.occurrenceKey);
            return settled;
          });
        }
        stampTaskResultMessageId(result, boundary.occurrenceKey);
        return result;
      });
    },
  });
}
