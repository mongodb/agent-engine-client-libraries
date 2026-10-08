/**
 * The interrupts LangGraph raised during the latest graph run of an attempt.
 *
 * A run's result lists only the interrupts projected to the root: for a
 * compiled child with parallel branches that is one of them. The checkpoint
 * write for an interrupt happens in the task that raised it, at any depth, so
 * the checkpointer is where every raised interrupt can be seen.
 */

import type { PendingWrite } from "@langchain/langgraph-checkpoint";
import type { AttemptContext } from "@mongodb-js/agent-engine-runner-shared";

const raisedByAttempt = new WeakMap<AttemptContext, Set<string>>();

/** Record the interrupt ids carried by one durable checkpoint write. */
export function noteRaisedInterrupts(
  attempt: AttemptContext,
  writes: readonly PendingWrite[],
): void {
  for (const [channel, value] of writes) {
    if (channel !== "__interrupt__") continue;
    // LangGraph JS writes one interrupt per write; accept a list as well.
    for (const interrupt of Array.isArray(value) ? value : [value]) {
      const id = (interrupt as { id?: unknown } | null)?.id;
      if (typeof id !== "string" || id === "") continue;
      const raised = raisedByAttempt.get(attempt) ?? new Set<string>();
      raised.add(id);
      raisedByAttempt.set(attempt, raised);
    }
  }
}

/** Return the ids raised since the last call, and start a new run's set. */
export function takeRaisedInterruptIds(attempt: AttemptContext): Set<string> {
  const raised = raisedByAttempt.get(attempt) ?? new Set<string>();
  raisedByAttempt.delete(attempt);
  return raised;
}
