/** Durable replay identity for platform-executed LangGraph tools. */

import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";

import { ToolMessage } from "@langchain/core/messages";
import { getConfig, isGraphBubbleUp, task } from "@langchain/langgraph";
import {
  ToolExecutionError,
  currentAttemptContext,
  currentOperationBoundaries,
  currentOperationPath,
  currentStepOrdinal,
  inSpawnedTask,
  runWithOperationPathResolver,
  type ChildOperationBoundary,
  type createSecureToolFunction,
} from "@mongodb-js/agent-engine-runner-shared";

import { toToolMessageContent } from "./messages.js";

type SecureToolFunction = ReturnType<typeof createSecureToolFunction>;

function durableToolResultMessageId(toolCallId: string): string | null {
  const attempt = currentAttemptContext();
  if (attempt === null) return null;

  const { executionId } = attempt.workflowIdentity as { executionId: string };
  const messageId =
    `durable-tool-result:${executionId}:` +
    `${currentStepOrdinal()}:${toolCallId}`;
  if (!inSpawnedTask()) return messageId;
  // Send workers in one step may see the same ToolCall ID, so their results
  // also carry the worker's path. Other IDs keep their recorded form.
  const path = JSON.stringify(
    currentOperationPath().segments.map((segment) => [
      segment.name,
      Number(segment.ordinal),
    ]),
  );
  return `${messageId}:${createHash("sha256").update(path).digest("hex")}`;
}

/**
 * Give a replayed platform tool result the same LangGraph message ID as its
 * first attempt, preventing the messages reducer from appending it twice.
 * The model-provided tool-call ID remains unchanged and continues to correlate
 * the ToolMessage with the AIMessage that requested it.
 */
export function withDurableToolResultIdentity(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  return async (kwargs, config) => {
    const result = await wrapped(kwargs, config);
    const toolCallId =
      config?.toolCall?.id ?? config?.configurable?.["tool_call_id"];
    if (typeof toolCallId !== "string" || !toolCallId) return result;

    const messageId = durableToolResultMessageId(toolCallId);
    if (messageId === null) return result;

    const [content, artifact] = result as [unknown, unknown];
    const message = new ToolMessage({
      content: toToolMessageContent(content),
      artifact,
      tool_call_id: toolCallId,
      name: toolName,
      id: messageId,
    });
    // content_and_artifact unwraps this tuple, then returns the ToolMessage
    // unchanged because the artifact is already attached to the message.
    return [message, null];
  };
}

interface OAuthTaskCall {
  readonly kwargs: Record<string, unknown>;
  readonly toolCallId: string;
  readonly boundaries: readonly ChildOperationBoundary[] | undefined;
}

interface OAuthTaskResult {
  readonly kwargs: Record<string, unknown>;
  readonly toolCallId: string;
  readonly result?: unknown;
  /**
   * The call's own failure, which the caller rethrows for ToolNode. A
   * checkpoint keeps it as an `Error` with its message.
   */
  readonly error?: unknown;
}

function inGraphTask(): boolean {
  const configurable = getConfig()?.configurable;
  return typeof configurable?.["__pregel_call"] === "function";
}

/**
 * Run each call of an OAuth tool in its own LangGraph task during a durable
 * attempt.
 *
 * The call's consent wait pauses with `interrupt()`. LangGraph numbers the
 * pauses of one task in the order they happen, and the calls a ToolNode runs
 * in parallel share its task, so their waits would take each other's slots.
 * A task per call gives each wait its own.
 *
 * The task's activities keep the operation position of the graph node that
 * called the tool. The task's cached result is bound to the call it ran for:
 * when parallel calls reach their tasks in another order on a rerun, a call
 * that receives another call's cached slot runs again outside the task, and
 * OE replays its recorded activities.
 *
 * The call's own failure is the task's result, rethrown to the caller: a
 * failed task would fail the graph, while ToolNode reports a tool's failure
 * to the model. Only LangGraph's pauses leave the task as errors.
 */
export function withDurableOAuthTask(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  const dispatch = task(
    `delegated_oauth_${toolName}`,
    async (call: OAuthTaskCall): Promise<OAuthTaskResult> => {
      const run = () =>
        wrapped(call.kwargs, { toolCall: { id: call.toolCallId } });
      const boundaries = call.boundaries;
      const identity = { kwargs: call.kwargs, toolCallId: call.toolCallId };
      try {
        const result =
          boundaries === undefined
            ? await run()
            : await runWithOperationPathResolver(() => boundaries, run);
        return { ...identity, result };
      } catch (error) {
        if (isGraphBubbleUp(error)) throw error;
        return { ...identity, error };
      }
    },
  );
  return async (kwargs = {}, config) => {
    if (currentAttemptContext() === null || !inGraphTask()) {
      return wrapped(kwargs, config);
    }
    const rawToolCallId =
      config?.toolCall?.id ?? config?.configurable?.["tool_call_id"];
    const toolCallId =
      typeof rawToolCallId === "string" && rawToolCallId ? rawToolCallId : null;
    // Without a call id, a rerun from the checkpoint gives the call a new
    // position, so OE would not match it to the consent the user gave.
    if (toolCallId === null) {
      throw new ToolExecutionError(
        `OAuth tool '${toolName}' must be invoked with a tool call during a durable run`,
      );
    }
    // Captured here because inside the task the graph path resolver sees the
    // task's namespace, which is not a graph node.
    const outcome = await dispatch({
      kwargs,
      toolCallId,
      boundaries: currentOperationBoundaries(),
    });
    if (outcome.toolCallId !== toolCallId) {
      return wrapped(kwargs, config);
    }
    if (!isDeepStrictEqual(outcome.kwargs, kwargs)) {
      throw new ToolExecutionError(
        "OAuth tool arguments changed during checkpoint replay",
      );
    }
    if ("error" in outcome) throw outcome.error;
    return outcome.result;
  };
}
