/**
 * Deep-agent factory for Atlas Agent Engine AER integration.
 *
 * `createAgentEngineDeepAgent` resolves relative skill paths and validates
 * subagent specs before delegating to deepagents (mirroring Python's
 * `deep_agent.py`):
 *
 * - {@link validateSubagentTree} rejects subagent specs with string models,
 *   which would bypass OE routing (deepagents instantiates a raw, unwrapped LLM
 *   from a string spec).
 *
 */

import * as path from "node:path";

import { createDeepAgent } from "deepagents";
import type {
  AnyBackendProtocol,
  AnySubAgent,
  CreateDeepAgentParams,
} from "deepagents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { validateSubagentTree } from "./subagents.js";
import { checkpointerForDeepAgent } from "./deep_agent_checkpointer.js";
import { createDurableDeepAgentMiddleware } from "./durable_deep_agent.js";
import { rejectNodeRetryPolicies } from "./durable_subgraphs.js";
import { UnsupportedDurableGraphError } from "./platform_checkpointer.js";

/** Names of compiled subagent specs whose runnable carries its own checkpointer. */
function compiledSubagentCheckpointerNames(
  subagents: readonly AnySubAgent[] | undefined,
): Set<string> {
  const names = new Set<string>();
  for (const spec of subagents ?? []) {
    if (
      "runnable" in spec &&
      (spec as { runnable?: { checkpointer?: unknown } }).runnable
        ?.checkpointer != null
    ) {
      names.add((spec as { name: string }).name);
    }
  }
  return names;
}

/**
 * Name the compiled subagents whose graphs set a retry policy. A compiled
 * subagent runs inside the Deep Agent's task tool, not as a node of the agent
 * graph, so the durable session's own check never reaches it.
 */
function compiledSubagentRetryPolicyNames(
  subagents: readonly AnySubAgent[] | undefined,
): Set<string> {
  const names = new Set<string>();
  for (const spec of subagents ?? []) {
    if (!("runnable" in spec)) continue;
    try {
      rejectNodeRetryPolicies((spec as { runnable?: unknown }).runnable);
    } catch (error) {
      if (!(error instanceof UnsupportedDurableGraphError)) throw error;
      names.add((spec as { name: string }).name);
    }
  }
  return names;
}

export interface CreateAgentEngineDeepAgentOptions {
  /** Additional tools for the deep agent. */
  tools?: StructuredTool[];
  /**
   * SubAgent specs. Plain specs with a `model` field must use a model instance,
   * not a string — string models bypass OE routing.
   */
  subagents?: readonly AnySubAgent[];
  /** Custom system instructions. */
  systemPrompt?: string;
  /** Additional middleware, appended after the default durable middleware. */
  middleware?: CreateDeepAgentParams["middleware"];
  /** LangGraph checkpointer for state persistence, HITL, and multi-turn. */
  checkpointer?: BaseCheckpointSaver | boolean;
  /** LangGraph store for skills and shared data. */
  store?: CreateDeepAgentParams["store"];
  /** Parent directories for deepagents' one-level skill discovery. */
  skills?: string[];
  /** Base directory for relative skill paths (set by `App.deepAgent()`). */
  skillsBaseDir?: string;
}

function loadSkillPaths(paths: readonly string[], baseDir?: string): string[] {
  if (baseDir === undefined) return [...paths];
  return paths.map((skillPath) =>
    path.isAbsolute(skillPath) ? skillPath : path.join(baseDir, skillPath),
  );
}

/**
 * Resolve relative per-subagent skill paths against the same base directory as
 * the top-level `skills` list. Compiled (`runnable`) and remote (`graphId`)
 * specs, and specs without a `skills` field, pass through unchanged.
 */
function loadSubagentSkillPaths(
  subagents: readonly AnySubAgent[],
  baseDir?: string,
): AnySubAgent[] {
  return subagents.map((spec) => {
    if ("runnable" in spec || "graphId" in spec || !("skills" in spec)) {
      return spec;
    }
    const skills = (spec as { skills?: readonly string[] }).skills;
    if (skills === undefined) return spec;
    return { ...spec, skills: loadSkillPaths(skills, baseDir) };
  });
}

/**
 * Create a deep-agent graph pre-configured for Atlas Agent Engine AER. Resolves relative
 * skill paths, validates the subagent tree, then delegates to deepagents'
 * `createDeepAgent`.
 *
 * @param secureLlm A `SecureWrappedLLM` instance — every LLM call is OE-audited.
 * @param backend Backend for filesystem/shell ops; resolved by `App.deepAgent()`.
 * @throws {Error} A subagent spec uses a string model, or nesting exceeds the cap.
 */
export function createAgentEngineDeepAgent(
  secureLlm: BaseChatModel,
  backend: AnyBackendProtocol | undefined,
  options: CreateAgentEngineDeepAgentOptions = {},
): ReturnType<typeof createDeepAgent> {
  validateSubagentTree(options.subagents);

  const forwardedSkills =
    options.skills !== undefined
      ? loadSkillPaths(options.skills, options.skillsBaseDir)
      : undefined;
  const forwardedSubagents =
    options.subagents !== undefined
      ? loadSubagentSkillPaths(options.subagents, options.skillsBaseDir)
      : undefined;
  const durableCheckpointer =
    options.checkpointer !== undefined
      ? checkpointerForDeepAgent(options.checkpointer)
      : undefined;

  // Passed straight through as skills= (not built into a middleware
  // instance here, unlike the Python port): createDeepAgent constructs
  // createSkillsMiddleware itself in three places -- the main agent, the
  // auto-injected general-purpose subagent, and any subagent with its own
  // skills= -- and deepagents-js exports that factory as a frozen ESM
  // binding, so there is no way to substitute a traced wrapper for the
  // other two call sites without dropping skills from them entirely.
  // Python can patch the class in place instead (see skills_tracing.py);
  // no equivalent exists here, so this hook stays untraced in TS.

  // Build via conditional spreads so absent options are omitted rather than
  // set to `undefined` — the package is compiled with
  // `exactOptionalPropertyTypes`, which rejects `undefined` on optional props.
  // `backend` is likewise omitted when undefined so deepagents applies its own
  // default (an in-memory StateBackend).
  return createDeepAgent({
    model: secureLlm as unknown as NonNullable<CreateDeepAgentParams["model"]>,
    ...(options.tools !== undefined && { tools: options.tools }),
    ...(forwardedSubagents !== undefined && { subagents: forwardedSubagents }),
    ...(options.systemPrompt !== undefined && {
      systemPrompt: options.systemPrompt,
    }),
    // Prepended so every agent gets durable local-subagent attribution by
    // default. A per-call-stopped tool batch goes to the model as ordinary
    // interrupted ToolMessages — continuation is the model's call, matching
    // the Python twin.
    middleware: [
      createDurableDeepAgentMiddleware({
        unsupportedSubagentNames: compiledSubagentCheckpointerNames(
          options.subagents,
        ),
        retryPolicySubagentNames: compiledSubagentRetryPolicyNames(
          options.subagents,
        ),
      }),
      ...(options.middleware ?? []),
    ],
    ...(durableCheckpointer !== undefined && {
      checkpointer: durableCheckpointer,
    }),
    ...(options.store !== undefined && { store: options.store }),
    ...(forwardedSkills !== undefined && { skills: forwardedSkills }),
    ...(backend !== undefined && { backend }),
  });
}
