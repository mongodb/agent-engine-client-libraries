/**
 * Subagent spec validation for `createAgentEngineDeepAgent`.
 *
 * Hard-fails at agent construction time when a subagent spec uses a string
 * model — string models bypass the `SecureWrappedLLM` seam and therefore the
 * OE audit path. Kept in its own module (mirroring Python's `subagents.py`) so
 * the validator has an importable surface for tests rather than being reachable
 * only through the wrapper factory.
 */

import type { AnySubAgent, SubAgent } from "deepagents";
import { isAsyncSubAgent } from "deepagents";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";

const logger = getLogger("agent_engine_sdk_langgraph.subagents");

/**
 * Recursion is bounded so an adversarial spec (or an accidental cycle) can't
 * blow the stack at agent-construction time.
 */
export const MAX_SUBAGENT_NESTING_DEPTH = 10;

/** A `SubAgent` may nest child specs; the deepagents type does not surface this. */
type NestableSubAgent = SubAgent & { subagents?: readonly AnySubAgent[] };

/**
 * Walk the subagent tree, throwing on any string-model spec.
 *
 * String model specs would bypass OE routing because deepagents' `resolveModel`
 * instantiates a raw provider client with no `SecureToolWrapper`. Only model
 * instances (e.g. `SecureWrappedLLM`) are accepted.
 *
 * Spec kinds handled:
 * - `CompiledSubAgent` (`runnable` set) — pre-built graph; the model is already
 *   bound and unreadable here. Skipped.
 * - `AsyncSubAgent` (`graphId` set) — remote graph we can't validate in-process.
 *   Logged at WARNING and skipped — the receiving end must route through the OE.
 * - Plain `SubAgent` — validated; a string `model` throws.
 *
 * The depth counter is held in a closure rather than a public parameter so
 * callers cannot start recursion mid-tree and bypass the cap.
 *
 * @throws {Error} A subagent spec uses a string `model`, or nesting exceeds
 *   {@link MAX_SUBAGENT_NESTING_DEPTH}.
 */
export function validateSubagentTree(
  subagents: readonly AnySubAgent[] | undefined,
): void {
  const walk = (
    nodes: readonly AnySubAgent[] | undefined,
    depth: number,
  ): void => {
    if (nodes === undefined) return;
    if (depth > MAX_SUBAGENT_NESTING_DEPTH) {
      throw new Error(
        `SubAgent nesting exceeds maximum recursion depth (${MAX_SUBAGENT_NESTING_DEPTH})`,
      );
    }
    for (const spec of nodes) {
      // Pre-built graph — model already bound, nothing to validate.
      if ("runnable" in spec) continue;
      // Remote graph — defer to the receiving end.
      if (isAsyncSubAgent(spec)) {
        logger.warn(
          `AsyncSubAgent '${spec.name ?? "<unnamed>"}' targets remote ` +
            `graphId='${spec.graphId}' — verify the remote graph routes ` +
            `through the OE for audit compliance`,
        );
        continue;
      }
      const sub = spec as NestableSubAgent;
      if (typeof sub.model === "string") {
        throw new Error(
          `SubAgent '${sub.name ?? "<unnamed>"}' uses a string model spec ` +
            `('${sub.model}') which bypasses OE routing. Pass a model ` +
            `instance (e.g. SecureWrappedLLM via app.llm()) instead.`,
        );
      }
      walk(sub.subagents, depth + 1);
    }
  };

  walk(subagents, 0);
}
