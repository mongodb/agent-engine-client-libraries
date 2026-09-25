/**
 * Framework hook registry for runner-shared.
 *
 * Most hooks are registered once at process startup by the framework SDK
 * before the server starts. The workflow adapter is the exception: it is
 * register-or-clear on every graph materialization so eligibility cannot
 * leak from a durable graph into a later native one. Consumers in
 * runner-shared call `get*()` instead of importing framework-specific
 * modules directly.
 *
 * Type aliases use sdk-core interfaces where possible. The suspend handler
 * remains loosely typed because its signature is framework-specific
 * (e.g. langgraph's `interrupt` returns whatever the resume caller provides).
 */

import type { BaseLLM } from "@mongodb-js/agent-engine-sdk";
import type { AERQueryPlugin } from "./server/query.js";
import { resetCheckpointWorkspaceState } from "./checkpoint_workspace.js";

// Accepts a serialized SuspendPayload, returns the human decision.
export type SuspendHandler = (
  payload: Record<string, unknown>,
) => Record<string, unknown>;
// Accepts (rawLlm, { tools, tool_choice }), returns a BaseLLM-compatible adapter.
export type LLMAdapterFactory = (
  rawLlm: unknown,
  options?: { tools?: unknown[]; tool_choice?: unknown },
) => BaseLLM;
// Runs framework-specific OTel instrumentation (e.g. LangChainInstrumentor).
export type Instrumentor = () => void;
export interface WorkflowAdapter {
  readonly name: string;
  readonly version: string;
}

let suspendHandler: SuspendHandler | null = null;
let llmAdapterFactory: LLMAdapterFactory | null = null;
let instrumentor: Instrumentor | null = null;
let queryPlugin: AERQueryPlugin | null = null;
let workflowAdapter: WorkflowAdapter | null = null;
const llmRegistry = new Map<string, unknown>();

// Set while the framework SDK is executing the user's @app.entrypoint
// builder function (see entrypointScope()). app.llm() must only be called
// while this is true -- it must be reproducibly re-derivable by the Tool
// Pod's startup registry construction (see docs/runner/README.md
// "Execution lifecycle & secret availability"), so calls from module top
// level, tool bodies, or any other function are rejected at the call site.
let entrypointActive = false;

/**
 * Mark the dynamic extent of the user's @app.entrypoint call.
 *
 * **Framework-internal — user agent code must never call this.** Users only
 * declare an entrypoint (`app.entrypoint(fn)`); the framework SDK wraps its
 * own evaluation of that function (during AER graph construction and
 * during Tool preparation, retried after failure) in this helper so
 * that `registerLlm()` can reject `app.llm()` calls made outside the
 * entrypoint. Tests that call `app.llm()`/`registerLlm()` directly (bypassing
 * `app.entrypoint` + `getAgent()`) use it to simulate that framework
 * evaluation.
 */
export function entrypointScope<T>(fn: () => T): T {
  const previous = entrypointActive;
  entrypointActive = true;
  try {
    return fn();
  } finally {
    entrypointActive = previous;
  }
}

export function registerSuspendHandler(handler: SuspendHandler): void {
  suspendHandler = handler;
}

export function getSuspendHandler(): SuspendHandler | null {
  return suspendHandler;
}

export function registerLLMAdapterFactory(factory: LLMAdapterFactory): void {
  llmAdapterFactory = factory;
}

export function getLLMAdapterFactory(): LLMAdapterFactory {
  if (llmAdapterFactory === null) {
    throw new Error(
      "No LLM adapter factory registered. Ensure the framework SDK calls registerLLMAdapterFactory() before run().",
    );
  }
  return llmAdapterFactory;
}

export function registerInstrumentor(fn: Instrumentor): void {
  instrumentor = fn;
}

export function getInstrumentor(): Instrumentor | null {
  return instrumentor;
}

/**
 * Register the framework adapter's `AERQueryPlugin`.
 *
 * Mirrors Python's `TenantRuntime.register_query_plugin`. The AER's
 * `/query/sessions` routes return 501 until a plugin is registered.
 */
export function registerQueryPlugin(plugin: AERQueryPlugin): void {
  queryPlugin = plugin;
}

/** Return the registered `AERQueryPlugin`, or `null` if none has been registered. */
export function getQueryPlugin(): AERQueryPlugin | null {
  return queryPlugin;
}

/**
 * Declare that the materialized graph supports OE durable workflow routing.
 *
 * Call this or {@link clearWorkflowAdapter} on every `getAgent()` materialization
 * so a later ineligible graph cannot reuse the previous request's identity.
 */
export function registerWorkflowAdapter(name: string, version: string): void {
  if (!name.trim() || !version.trim()) {
    throw new Error("workflow adapter name and version must be non-empty");
  }
  workflowAdapter = { name, version };
}

/** Drop durable eligibility for the latest materialized graph. */
export function clearWorkflowAdapter(): void {
  workflowAdapter = null;
}

export function getWorkflowAdapter(): WorkflowAdapter | null {
  return workflowAdapter;
}

/**
 * Register an LLM by id. Throws on duplicate id.
 *
 * Framework SDKs call this from `app.llm()` for every LLM the agent uses.
 * The unnamed-LLM convenience case is represented by registering under the
 * sentinel id `"__default__"`; a second unnamed call therefore raises the
 * same duplicate-id error as a second named call with the same id.
 */
export function registerLlm(llmId: string, llm: unknown): void {
  if (!entrypointActive) {
    throw new Error(
      "app.llm(...) must be called inside the function decorated with " +
        "@app.entrypoint. It was called outside the entrypoint's dynamic " +
        "extent. Move this call inside your @app.entrypoint builder function.",
    );
  }
  if (llmRegistry.has(llmId)) {
    throw new Error(
      `llm_id ${JSON.stringify(llmId)} is already registered. ` +
        "Each LLM must have a unique llm_id; pass app.llm(llm, llm_id='...') for each.",
    );
  }
  llmRegistry.set(llmId, llm);
}

export function getNamedLlm(llmId: string): unknown {
  if (!llmRegistry.has(llmId)) {
    throw new Error(
      `llm_id ${JSON.stringify(llmId)} not registered. ` +
        "Call app.llm(llm, llm_id=...) for each LLM before run().",
    );
  }
  return llmRegistry.get(llmId);
}

export function hasNamedLlms(): boolean {
  return llmRegistry.size > 0;
}

export function resetLlmRegistry(): void {
  llmRegistry.clear();
}

/** Return a shallow copy of the current registry. */
export function snapshotLlmRegistry(): Map<string, unknown> {
  return new Map(llmRegistry);
}

export function resetHooks(): void {
  suspendHandler = null;
  llmAdapterFactory = null;
  instrumentor = null;
  queryPlugin = null;
  workflowAdapter = null;
  llmRegistry.clear();
  entrypointActive = false;
  resetCheckpointWorkspaceState();
}

/**
 * Run the user's agent entrypoint to populate the named-LLM registry.
 *
 * Resets the registry before invoking `graphBuilder.getAgent()` so
 * import-time registrations don't collide if the entrypoint re-registers
 * them. If the entrypoint registers nothing, restores the pre-reset snapshot
 * so import-time-only agents still work.
 *
 * Returns false after failed construction, including when a snapshot is restored.
 * Successful construction with no named LLMs is still complete.
 *
 * `onEntrypointError`, when supplied, receives the thrown value on that same
 * path so the caller can attribute a later registry lookup miss to its real
 * cause rather than to a missing `app.llm()` call.
 *
 * Shared by ToolServer.onStartup and ToolFunctionRunner._prepare — mirrors
 * Python's ToolExecution.prepare() which both inherit.
 *
 * @internal — exported for server/ consumers; not part of the public API.
 */
export function populateLlmRegistryFromEntrypoint(
  graphBuilder: { getAgent?: () => unknown } | null,
  logger: { warn(msg: string): void; info(msg: string): void },
  warnings: { entrypointFailed: string; noLlmRegistered: string },
  onEntrypointError?: (error: unknown) => void,
): boolean {
  if (graphBuilder === null || typeof graphBuilder.getAgent !== "function")
    return true;

  const preSnapshot = snapshotLlmRegistry();
  resetLlmRegistry();
  try {
    // Call through the receiver -- graphBuilder.getAgent() (e.g. App.getAgent())
    // reads its own instance state (e.g. `this.builderFn`), so storing the
    // method in a local and invoking it unbound would drop `this` and throw.
    entrypointScope(() => graphBuilder.getAgent?.());
  } catch (error) {
    logger.warn(warnings.entrypointFailed);
    // Discard partial registrations before restoring the last snapshot.
    resetLlmRegistry();
    if (preSnapshot.size > 0) {
      // Runtime-owned restoration, not a user app.llm() call -- exempt from
      // the entrypoint-scope requirement.
      entrypointScope(() => {
        for (const [llmId, llm] of preSnapshot) registerLlm(llmId, llm);
      });
      logger.info(
        `Entrypoint failed; restored ${preSnapshot.size} import-time registration(s)`,
      );
    }
    // Let the caller remember the cause: a later lookup miss inside the LLM
    // route is a symptom of this failure, and reporting it as a missing
    // app.llm() call would misdirect the agent developer. Mirrors Python's
    // _llm_registry_load_error.
    onEntrypointError?.(error);
    return false;
  }

  if (hasNamedLlms()) {
    logger.info("Named LLM registry populated from user entrypoint");
  } else if (preSnapshot.size > 0) {
    entrypointScope(() => {
      for (const [llmId, llm] of preSnapshot) registerLlm(llmId, llm);
    });
    logger.info(
      `Entrypoint registered no LLMs; restored ${preSnapshot.size} import-time registration(s)`,
    );
  } else {
    logger.warn(warnings.noLlmRegistered);
  }

  return true;
}
