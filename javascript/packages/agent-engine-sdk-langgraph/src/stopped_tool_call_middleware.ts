import { createMiddleware } from "langchain";
import {
  AIMessage,
  type BaseMessage,
  isAIMessage,
  isToolMessage,
  type ToolMessage,
} from "@langchain/core/messages";
import { CALL_INTERRUPTED_ARTIFACT_KEY } from "@mongodb-js/agent-engine-runner-shared";

export const ALL_INTERRUPTED_MESSAGE =
  "This request was stopped before any of its actions completed.";

// `isAIMessage`/`isToolMessage` key off `getType()`, so they also match
// AIMessageChunk/ToolMessageChunk instances a checkpointer may hand back —
// unlike `instanceof`, which only matches the exact class.
function hasToolCalls(message: AIMessage): boolean {
  return Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
}

function isCallInterrupted(message: ToolMessage): boolean {
  const artifact = message.artifact as Record<string, unknown> | undefined;
  return (
    typeof artifact === "object" &&
    artifact !== null &&
    Boolean(artifact[CALL_INTERRUPTED_ARTIFACT_KEY])
  );
}

export function latestToolBatch(messages: readonly unknown[]): ToolMessage[] {
  const batch: ToolMessage[] = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (isToolMessage(message)) {
      batch.push(message);
      continue;
    }
    if (
      isAIMessage(message as BaseMessage) &&
      hasToolCalls(message as AIMessage)
    )
      break;
    return [];
  }
  return batch.reverse();
}

/**
 * Ends the turn deterministically when every tool call in the latest batch
 * was stopped, instead of leaving graph continuation to the model.
 */
export const StoppedToolCallMiddleware = () =>
  createMiddleware({
    name: "StoppedToolCallMiddleware",
    beforeModel: {
      canJumpTo: ["end"],
      hook: (state: { messages?: unknown[] }) => {
        const batch = latestToolBatch(state.messages ?? []);
        if (batch.length === 0) return undefined;

        const interruptedCount = batch.filter(isCallInterrupted).length;
        if (interruptedCount === 0 || interruptedCount !== batch.length) {
          return undefined;
        }

        return {
          jumpTo: "end",
          messages: [new AIMessage(ALL_INTERRUPTED_MESSAGE)],
        };
      },
    },
  });
