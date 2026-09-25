/**
 * Parse Deep Agent `task` tool calls for adapter-owned behavior.
 *
 * Port of `deep_agent_task.py`.
 */

export const DISPATCH_TOOL_NAME = "task";

function toolCallValue(toolCall: unknown, key: string): unknown {
  if (toolCall === null || typeof toolCall !== "object") return undefined;
  return (toolCall as Record<string, unknown>)[key];
}

export interface DeepAgentTaskCall {
  /** Normalized identity and stream metadata for one `task` call. */
  readonly toolCallId: string;
  readonly subagentName: string;
  readonly description: string;
}

/**
 * Normalize one Deep Agent task call, retaining malformed identity as empty.
 * Returns null for tool calls that are not `task` dispatches.
 */
export function parseTaskCall(toolCall: unknown): DeepAgentTaskCall | null {
  if (toolCallValue(toolCall, "name") !== DISPATCH_TOOL_NAME) return null;

  const rawId = toolCallValue(toolCall, "id");
  const args = toolCallValue(toolCall, "args");
  const argsDict =
    args !== null && typeof args === "object" && !Array.isArray(args)
      ? (args as Record<string, unknown>)
      : {};
  return {
    toolCallId: typeof rawId === "string" ? rawId : "",
    subagentName: String(argsDict["subagent_type"] ?? "") || "",
    description: String(argsDict["description"] ?? "") || "",
  };
}

/**
 * Yield task calls with IDs from LangChain messages or chunks.
 *
 * Stream attribution historically ignores incomplete tool-call chunks until
 * their stable ID is assembled. Durable dispatch performs its stricter name
 * and ID validation directly through {@link parseTaskCall}.
 */
export function* iterTaskCalls(
  messagesOrChunks: Iterable<unknown>,
): IterableIterator<DeepAgentTaskCall> {
  for (const message of messagesOrChunks) {
    const toolCalls = (message as { tool_calls?: unknown }).tool_calls;
    if (!Array.isArray(toolCalls)) continue;
    for (const toolCall of toolCalls) {
      const parsed = parseTaskCall(toolCall);
      if (parsed !== null && parsed.toolCallId !== "") yield parsed;
    }
  }
}
