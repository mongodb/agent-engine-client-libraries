/** Checkpointer view for graphs created by Deep Agent. */

import type { BaseCheckpointSaver } from "@langchain/langgraph";

import { PlatformCheckpointer } from "./platform_checkpointer.js";

/**
 * Return the Deep Agent view for a platform checkpointer. LangChain's agent
 * factory routes pending ToolCalls through LangGraph Send packets, and Deep
 * Agent's recorded activity paths predate Send task segments, so the adapter
 * recognizes this view and keeps those tasks on their parent path. The view
 * shares the same platform scratch and native savers. Other checkpointers
 * pass through unchanged.
 */
export function checkpointerForDeepAgent<
  C extends BaseCheckpointSaver<string | number> | boolean | undefined,
>(checkpointer: C): C {
  if (checkpointer instanceof PlatformCheckpointer) {
    // The view shares the same platform state, so platform-owned lifecycle
    // (scratch release, completion) observes exactly what the deep-agent
    // graph writes.
    return checkpointer.deepAgentView() as unknown as C;
  }
  return checkpointer;
}
