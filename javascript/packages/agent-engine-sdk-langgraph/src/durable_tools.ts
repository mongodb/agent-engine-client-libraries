/** Durable replay identity for platform-executed LangGraph tools. */

import { ToolMessage } from "@langchain/core/messages";
import {
  currentAttemptContext,
  currentStepOrdinal,
  type createSecureToolFunction,
} from "@mongodb-js/agent-engine-runner-shared";

import { toToolMessageContent } from "./messages.js";

type SecureToolFunction = ReturnType<typeof createSecureToolFunction>;

function durableToolResultMessageId(toolCallId: string): string | null {
  const attempt = currentAttemptContext();
  if (attempt === null) return null;

  const { executionId } = attempt.workflowIdentity as { executionId: string };
  return (
    `durable-tool-result:${executionId}:` +
    `${currentStepOrdinal()}:${toolCallId}`
  );
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
