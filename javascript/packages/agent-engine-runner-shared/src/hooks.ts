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

import { AsyncLocalStorage } from "node:async_hooks";

import type {
  AgentInput,
  BaseLLM,
  RequestContext,
} from "@mongodb-js/agent-engine-sdk";
import type { AERQueryPlugin } from "./server/query.js";
import { resetCheckpointWorkspaceState } from "./checkpoint_workspace.js";
import { withTimeout } from "./timeout.js";
import { AsyncMutex } from "./async_mutex.js";
import { getEnvFloat } from "./utils.js";

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

// Marks the dynamic extent of the framework SDK's evaluation of the user's
// @app.entrypoint builder. app.llm() must only be called inside it, because the
// Tool Pod's startup registry construction has to be able to reproduce every
// registration (see docs/runner/README.md "Execution lifecycle & secret
// availability"), so calls from module top level, tool bodies, or any other
// function are rejected at the call site.
//
// AsyncLocalStorage rather than a module-global flag: a per-execution builder
// holds this scope across its own I/O, and a process-wide flag would authorize
// any other task running during those awaits to register into the very registry
// the build's id set is validated against. Mirrors Python's ContextVar.
const entrypointStorage = new AsyncLocalStorage<true>();

// Evaluations currently inside the scope. AsyncLocalStorage keeps the store for
// every async resource created inside the callback, including timers and
// floating promises that outlive it, so the store alone would authorize a
// registration landing after the build returned and its id set was already
// validated. Pairing it with a live count bounds the authorization to the
// evaluation's actual lifetime.
let activeEntrypointEvaluations = 0;

/**
 * How long the context-free discovery build may take.
 *
 * Matches the AER's execution bound so a builder that works there works here.
 */
function discoveryTimeoutMs(): number {
  return getEnvFloat("RUNNER_EXECUTION_TIMEOUT", 3600.0) * 1000;
}

function withEntrypointStore<T>(fn: () => T): T {
  // Nested scopes are a no-op, as in runWithCustomerOrigin.
  if (entrypointStorage.getStore() === true) return fn();
  return entrypointStorage.run(true, fn);
}

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
  activeEntrypointEvaluations += 1;
  try {
    return withEntrypointStore(fn);
  } finally {
    activeEntrypointEvaluations -= 1;
  }
}

/**
 * Async variant of {@link entrypointScope}, for a builder that awaits.
 *
 * The scope has to survive the builder's awaits: a per-execution builder does
 * its own I/O before returning its graph, so its `app.llm()` calls land after
 * an await. `AsyncLocalStorage.run` already keeps the store for the async
 * continuations created inside the callback, so this is the same call; the
 * separate name documents which kind of callback is expected.
 */
export async function entrypointScopeAsync<T>(
  fn: () => Promise<T>,
): Promise<T> {
  activeEntrypointEvaluations += 1;
  try {
    return await withEntrypointStore(fn);
  } finally {
    activeEntrypointEvaluations -= 1;
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
  if (
    entrypointStorage.getStore() !== true ||
    activeEntrypointEvaluations === 0
  ) {
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
  // No scope flag to reset: the store is scoped to the async context that
  // opened it, so it cannot outlive the call that ran under it.

  resetCheckpointWorkspaceState();
}

/**
 * Whether `error` is a framework SDK's LLM-registration drift error.
 *
 * Detected by marker property rather than `instanceof`: runner-shared must not
 * depend on a framework SDK, so the SDK's error carries the marker instead.
 */
function isLlmRegistrationDrift(error: unknown): boolean {
  return (
    (error as { isLlmRegistrationDrift?: boolean } | null)
      ?.isLlmRegistrationDrift === true
  );
}

/**
 * Restore the pre-entrypoint snapshot after a failed build.
 *
 * Runtime-owned restoration, not a user `app.llm()` call, so it is exempt from
 * the entrypoint-scope requirement.
 */
function restoreAfterFailedEntrypoint(
  preSnapshot: Map<string, unknown>,
  logger: { info(msg: string): void },
): void {
  if (preSnapshot.size === 0) return;
  entrypointScope(() => {
    for (const [llmId, llm] of preSnapshot) registerLlm(llmId, llm);
  });
  logger.info(
    `Entrypoint failed; restored ${preSnapshot.size} import-time registration(s)`,
  );
}

/** The graph builder a server hands in, narrowed to the two methods used here. */
type RegistryGraphBuilder = {
  getAgent?: () => unknown;
  buildGraph?: (
    agentInput?: AgentInput,
    ctx?: RequestContext,
  ) => Promise<unknown>;
} | null;

/**
 * Serializes the whole populate / reset / restore lifetime across every server
 * in this process. See populateLlmRegistryFromEntrypoint for why it is here
 * rather than in a server.
 */
const llmRegistryPopulationLock = new AsyncMutex();

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
 * Used by ToolServer.onStartup — mirrors Python's ToolExecution.prepare().
 *
 * @internal — exported for server/ consumers; not part of the public API.
 */
export async function populateLlmRegistryFromEntrypoint(
  graphBuilder: RegistryGraphBuilder,
  logger: {
    warn(msg: string): void;
    info(msg: string): void;
    error(msg: string): void;
  },
  warnings: { entrypointFailed: string; noLlmRegistered: string },
  onEntrypointError?: (error: unknown) => void,
  strategy?: { usesPerInvocationBuild: boolean },
): Promise<boolean> {
  if (graphBuilder === null || typeof graphBuilder.getAgent !== "function")
    return true;
  // One population at a time. The registry is process-global and the reset
  // below runs before the build is awaited, so without this lock a second
  // caller clears the registrations a live build is still working from, and
  // that build then captures a baseline which is not its own.
  return llmRegistryPopulationLock.runExclusive(() =>
    populateLlmRegistryLocked(
      graphBuilder,
      logger,
      warnings,
      onEntrypointError,
      strategy,
    ),
  );
}

async function populateLlmRegistryLocked(
  graphBuilder: NonNullable<RegistryGraphBuilder>,
  logger: {
    warn(msg: string): void;
    info(msg: string): void;
    error(msg: string): void;
  },
  warnings: { entrypointFailed: string; noLlmRegistered: string },
  onEntrypointError?: (error: unknown) => void,
  strategy?: { usesPerInvocationBuild: boolean },
): Promise<boolean> {
  const preSnapshot = snapshotLlmRegistry();
  resetLlmRegistry();
  try {
    // Call through the receiver -- graphBuilder.getAgent() (e.g. App.getAgent())
    // reads its own instance state (e.g. `this.builderFn`), so storing the
    // method in a local and invoking it unbound would drop `this` and throw.
    if (
      strategy?.usesPerInvocationBuild &&
      typeof graphBuilder.buildGraph === "function"
    ) {
      // An opted-in builder is async and does its own I/O, so discovery has to
      // await the App's context-free build rather than call getAgent(). The
      // cached getAgent() path would invoke the builder synchronously and store
      // its pending Promise as the graph, leaving the builder to reject
      // unhandled outside this boundary -- which is exactly where the partial
      // registrations below are recovered.
      const buildGraph = graphBuilder.buildGraph;
      // Bounded, because this runs with the caller's registry lock held: an
      // author builder whose I/O never returns would otherwise hold it forever
      // and queue every later model request behind it. The AER bounds the same
      // builder, so the two have to agree or a builder that works on one path
      // fails on the other. A breach lands in the catch below, which keeps the
      // registrations the builder made before it stalled.
      await withTimeout(
        entrypointScopeAsync(() => buildGraph.call(graphBuilder)),
        discoveryTimeoutMs(),
        () =>
          "the Tool Pod's context-free discovery build did not complete in time",
      );
    } else {
      entrypointScope(() => graphBuilder.getAgent?.());
    }
  } catch (error) {
    logger.warn(warnings.entrypointFailed);
    if (isLlmRegistrationDrift(error)) {
      // The build registered a different id set than the pod's baseline, so
      // that registry has drifted and is not a usable partial discovery:
      // discard it rather than serve ids the pod was not built for.
      resetLlmRegistry();
      restoreAfterFailedEntrypoint(preSnapshot, logger);
      logger.error(
        "Per-execution build changed the registered LLM ids; " +
          "discarding the mismatched registry",
      );
    } else if (strategy?.usesPerInvocationBuild) {
      // An opted-in builder registers its LLMs and only then reads invocation
      // context, which this context-free discovery path cannot supply. Keep
      // what it registered before raising, and fill in anything the previous
      // snapshot holds that the partial build did not reach.
      const partial = snapshotLlmRegistry();
      const missing = new Map<string, unknown>();
      for (const [llmId, llm] of preSnapshot) {
        if (!partial.has(llmId)) missing.set(llmId, llm);
      }
      if (missing.size > 0) {
        entrypointScope(() => {
          for (const [llmId, llm] of missing) registerLlm(llmId, llm);
        });
      }
      if (partial.size > 0 || missing.size > 0) {
        logger.info(
          `Entrypoint failed; kept ${partial.size} partial registration(s) and restored ${missing.size} import-time registration(s)`,
        );
      }
    } else {
      // Every other agent keeps the previous contract: a failed build leaves
      // no half-populated registry behind.
      resetLlmRegistry();
      restoreAfterFailedEntrypoint(preSnapshot, logger);
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
