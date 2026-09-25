/**
 * Session fork. See `docs/session-fork.md`.
 *
 * Port of Python's `agent_engine_sdk_langgraph/session_fork.py`. LangGraph.js has no
 * synchronous `update_state` and no `task_id` parameter, so only the async
 * `updateState` surface is wrapped and branch keys omit the task component.
 */

import { createHash } from "node:crypto";
import type { RunnableConfig } from "@langchain/core/runnables";
import type {
  CheckpointMetadata,
  CompiledStateGraph,
} from "@langchain/langgraph";
import {
  currentAttemptContext,
  encodeProtoJson,
  fetchPlatform,
  getFetchOptionsWithTLS,
  getCurrentOeUrl,
  quotePathSegment,
  resolveOeUrl,
  StateSnapshotSchema,
  type StateSnapshot,
  type WorkflowIdentity,
} from "@mongodb-js/agent-engine-runner-shared";
import type { RequestContext } from "@mongodb-js/agent-engine-sdk";

import {
  checkpointReplayPlan,
  completedRootCheckpoint,
  copyCheckpoint,
} from "./checkpoint_branch.js";
import {
  OE_STEP_ORDINAL_METADATA_KEY,
  PlatformCheckpointer,
} from "./platform_checkpointer.js";
import { checkpointThreadId, type ResolveThreadIdHook } from "./thread_id.js";
import { channelValuesToStateSnapshot } from "./workflow_state.js";

/** Response for forking a session into a new independent session. */
export interface SessionForkResponse {
  /** Identifier of the new branch session. */
  sessionId: string;
  /** Pending execution on the new branch session. */
  executionId: string;
}

interface ForkableAgent {
  compiledGraph: CompiledStateGraph<unknown, unknown>;
  resolveThreadId?: ResolveThreadIdHook | null;
}

const CREATE_BRANCH_TIMEOUT_MS = 15_000;
// Dest continue is process-local. Bound so unused dests cannot accumulate;
// an evicted mark behaves like a dest execution that starts in another process.
const CONTINUE_MARK_LIMIT = 256;
const CONTINUE_WITHOUT_USER_MESSAGE = new Map<string, null>();

/** True once: dest should `invoke(null)`. See `docs/session-fork.md`. */
export function takeContinueWithoutUserMessage(threadId: string): boolean {
  if (!CONTINUE_WITHOUT_USER_MESSAGE.has(threadId)) return false;
  CONTINUE_WITHOUT_USER_MESSAGE.delete(threadId);
  return true;
}

function armContinueWithoutUserMessage(threadId: string): void {
  if (CONTINUE_WITHOUT_USER_MESSAGE.has(threadId)) {
    CONTINUE_WITHOUT_USER_MESSAGE.delete(threadId);
  } else if (CONTINUE_WITHOUT_USER_MESSAGE.size >= CONTINUE_MARK_LIMIT) {
    const oldest = CONTINUE_WITHOUT_USER_MESSAGE.keys().next();
    if (!oldest.done) CONTINUE_WITHOUT_USER_MESSAGE.delete(oldest.value);
  }
  CONTINUE_WITHOUT_USER_MESSAGE.set(threadId, null);
}

function requireNonEmpty(
  value: string | undefined | null,
  what: string,
): string {
  if (!value || !value.trim()) {
    throw new Error(`${what} is required to fork a session`);
  }
  return value;
}

function isEmptyValues(state: unknown): boolean {
  return (
    state === null ||
    state === undefined ||
    (typeof state === "object" && Object.keys(state).length === 0)
  );
}

/**
 * Create a branch session and copy the source checkpoint onto it.
 *
 * Port of Python's `fork_native_session`: the selected checkpoint and its
 * replay plan are validated before CreateBranch, and the copy reuses this
 * package's `copyCheckpoint` with that validated plan.
 */
export async function forkNativeSession(
  graph: CompiledStateGraph<unknown, unknown>,
  args: {
    ctx: RequestContext;
    historyId: string | null;
    resolveThreadId?: ResolveThreadIdHook | null;
    state?: unknown;
    asNode?: string | null;
  },
): Promise<[SessionForkResponse, RunnableConfig]> {
  if (currentAttemptContext() !== null) {
    throw new Error(
      "session fork is not supported on durable_workflow sessions",
    );
  }
  const { ctx } = args;
  const sessionId = requireNonEmpty(ctx.sessionId, "session identity");
  const workspaceId = requireNonEmpty(ctx.workspaceId, "workspace identity");
  const executionId = requireNonEmpty(ctx.executionId, "execution identity");

  const source = await completedRootCheckpoint(graph, {
    threadId: checkpointThreadId(ctx, args.resolveThreadId),
    historyId: args.historyId,
  });
  const plan = await checkpointReplayPlan(graph, source);
  const hasPatch = !isEmptyValues(args.state);
  // Resolve and validate the patch node before CreateBranch: LangGraph would
  // only reject an unknown or ambiguous target after the branch and copy
  // already exist.
  const patchNode = hasPatch ? (args.asNode ?? singleNodeName(graph)) : null;
  if (
    hasPatch &&
    args.asNode !== null &&
    args.asNode !== undefined &&
    !graphNodeNames(graph).includes(args.asNode)
  ) {
    throw new Error(
      `session fork state patch node "${args.asNode}" is not a node in this graph`,
    );
  }
  if (hasPatch && patchNode === null) {
    throw new Error(
      "session fork state patch requires a destination node: " +
        "pass asNode or fork a single-node graph",
    );
  }

  const branch = await createNativeBranch({
    orgId: tenantEnvId("ORG_ID"),
    projectId: tenantEnvId("PROJECT_ID"),
    workspaceId,
    sessionId,
    executionId,
    branchKey: branchKey(
      source.checkpointId,
      args.state,
      args.asNode ?? null,
      null,
    ),
    checkpointId: source.checkpointId,
  });

  const destThreadId = checkpointThreadId(
    {
      sessionId: branch.sessionId,
      workspaceId,
      executionId: branch.executionId,
    },
    args.resolveThreadId,
  );
  const destConfig: RunnableConfig = {
    configurable: { thread_id: destThreadId, checkpoint_ns: "" },
  };
  // Python `fork_native_session` parity: copy onto the empty destination,
  // then apply the optional patch. A populated destination (a repeated fork
  // for the same branch key, or a destination left behind by a failed
  // attempt) is sealed by `copyCheckpoint`'s empty-destination guard instead
  // of being repaired, verified, or silently reused.
  await copyCheckpoint(graph, destConfig, source, plan);
  if (patchNode !== null) {
    await graph.updateState(
      destConfig,
      args.state as Record<string, unknown>,
      patchNode,
    );
  }
  const destCheckpointId = await latestCheckpointId(graph, destThreadId);
  const finalConfig: RunnableConfig = {
    ...destConfig,
    configurable: {
      ...destConfig.configurable,
      checkpoint_id: destCheckpointId,
    },
  };
  armContinueWithoutUserMessage(destThreadId);
  return [branch, finalConfig];
}

async function latestCheckpointId(
  graph: CompiledStateGraph<unknown, unknown>,
  threadId: string,
): Promise<string> {
  const snapshot = await graph.getState({
    configurable: { thread_id: threadId },
  });
  const checkpointId = snapshot.config.configurable?.["checkpoint_id"];
  if (typeof checkpointId !== "string" || checkpointId === "") {
    throw new Error("LangGraph branch point was not found");
  }
  return checkpointId;
}

/** Root-loop node names a state patch may target. */
function graphNodeNames(graph: CompiledStateGraph<unknown, unknown>): string[] {
  return Object.keys(
    (graph as unknown as { nodes: Record<string, unknown> }).nodes ?? {},
  ).filter((name) => !name.startsWith("__"));
}

function singleNodeName(
  graph: CompiledStateGraph<unknown, unknown>,
): string | null {
  const names = graphNodeNames(graph);
  return names.length === 1 ? (names[0] ?? null) : null;
}

/**
 * Create an OE-owned branch from a selected committed scratch checkpoint.
 *
 * Port of Python's `_fork_durable_session`.
 */
export async function forkDurableSession(
  graph: CompiledStateGraph<unknown, unknown>,
  args: {
    context: NonNullable<ReturnType<typeof currentAttemptContext>>;
    config: RunnableConfig;
    resolveThreadId?: ResolveThreadIdHook | null;
  },
): Promise<[SessionForkResponse, RunnableConfig]> {
  const { context } = args;
  const identity: WorkflowIdentity | undefined = context.workflowIdentity;
  const scope = identity?.tenantScope;
  if (
    !identity ||
    !scope ||
    !identity.sessionId ||
    !identity.executionId ||
    !scope.orgId ||
    !scope.projectId ||
    !scope.workspaceId
  ) {
    throw new Error(
      "durable session fork requires complete OE attempt identity",
    );
  }

  const checkpointer = graph.checkpointer;
  if (!(checkpointer instanceof PlatformCheckpointer)) {
    throw new Error("durable session fork requires the platform checkpointer");
  }
  const source = await checkpointer.getTuple(args.config);
  if (source === undefined) {
    throw new Error("durable branch point is not a committed root step");
  }
  const stepOrdinal = oeStepOrdinal(source.metadata);
  // Materialized state adds unwritten reducer defaults absent from the OE hash.
  const sourceState = channelValuesToStateSnapshot(
    (source.checkpoint.channel_values ?? {}) as Record<string, unknown>,
    context.replayMode,
  );
  const branchSource = `${identity.executionId}/step/${stepOrdinal}`;
  const branch = await createDurableBranch({
    orgId: scope.orgId,
    projectId: scope.projectId,
    workspaceId: scope.workspaceId,
    sessionId: identity.sessionId,
    executionId: identity.executionId,
    branchKey: branchKey(branchSource, null, null, null),
    stepOrdinal,
    state: sourceState,
  });

  // RequestContext is request-local. Use a minimal value only to resolve the
  // destination thread; the future branch invocation receives its own context.
  const destConfig: RunnableConfig = {
    configurable: {
      thread_id: checkpointThreadId(
        {
          sessionId: branch.sessionId,
          workspaceId: scope.workspaceId,
          executionId: branch.executionId,
        },
        args.resolveThreadId,
      ),
      checkpoint_ns: "",
    },
  };
  return [branch, destConfig];
}

/** Read the absolute OE ordinal recorded with a committed scratch step. */
function oeStepOrdinal(metadata: CheckpointMetadata | undefined): number {
  const values = (metadata ?? {}) as Record<string, unknown>;
  const stepOrdinal = values[OE_STEP_ORDINAL_METADATA_KEY];
  if (stepOrdinal === undefined || stepOrdinal === null) {
    throw new Error("durable branch point is not a committed root step");
  }
  return Number(stepOrdinal);
}

/** Reject patches until OE can record one as a branch-local transition. */
export function requireExactDurableBranch(state: unknown): void {
  if (!isEmptyValues(state)) {
    throw new Error(
      "durable_workflow session fork does not support an updateState patch",
    );
  }
}

function liveSource(
  config: RunnableConfig,
  snapshotConfig: RunnableConfig,
  resolveThreadId: ResolveThreadIdHook | null,
): RequestContext | null {
  if (currentAttemptContext() !== null) return null;
  const checkpointId = snapshotConfig.configurable?.["checkpoint_id"];
  if (typeof checkpointId !== "string" || checkpointId === "") return null;
  const ctx = config.configurable?.["request_context"] as
    | RequestContext
    | undefined;
  if (ctx === undefined || ctx === null) return null;
  if (!ctx.sessionId || !ctx.workspaceId || !ctx.executionId) return null;
  const threadId = config.configurable?.["thread_id"];
  if (threadId !== checkpointThreadId(ctx, resolveThreadId)) return null;
  return ctx;
}

function tenantEnvId(name: string): string {
  const value = (process.env[name] ?? "").trim();
  if (!value) {
    throw new Error(`${name} is required to fork a session`);
  }
  return value;
}

/**
 * Route live `updateState` through the persistence-specific session fork.
 *
 * See `docs/session-fork.md`.
 */
export function wrapSessionForkUpdateState(
  graph: CompiledStateGraph<unknown, unknown>,
  args: { resolveThreadId: ResolveThreadIdHook | null },
): void {
  // Python wraps only when the graph exposes the async update surface; graph
  // test doubles may omit it entirely.
  const originalUpdate = (
    graph as Partial<CompiledStateGraph<unknown, unknown>>
  ).updateState;
  if (typeof originalUpdate !== "function") return;
  const original = originalUpdate.bind(graph);
  const originalGetState = graph.getState.bind(graph);

  graph.updateState = async (
    config: RunnableConfig,
    values: unknown,
    asNode?: string,
  ): Promise<RunnableConfig> => {
    const attempt = currentAttemptContext();
    if (attempt !== null) {
      if (args.resolveThreadId !== null) {
        throw new Error(
          "session fork requires the default thread_id formula; " +
            "custom App.resolveThreadId is not supported on graph.updateState",
        );
      }
      // CreateBranch hash-validates the selected committed snapshot. A
      // state patch needs a separate branch-local transition contract.
      requireExactDurableBranch(values);
      const [, destConfig] = await forkDurableSession(graph, {
        context: attempt,
        config,
        resolveThreadId: args.resolveThreadId,
      });
      return destConfig;
    }
    const snapshot = await originalGetState(config);
    const ctx = liveSource(config, snapshot.config, args.resolveThreadId);
    if (ctx !== null) {
      if (args.resolveThreadId !== null) {
        throw new Error(
          "session fork requires the default thread_id formula; " +
            "custom App.resolveThreadId is not supported on graph.updateState",
        );
      }
      const pinned = config.configurable?.["checkpoint_id"];
      const snapshotId = snapshot.config.configurable?.["checkpoint_id"];
      const historyId =
        typeof pinned === "string" && pinned !== ""
          ? pinned
          : typeof snapshotId === "string"
            ? snapshotId
            : null;
      const [, destConfig] = await forkNativeSession(graph, {
        ctx,
        historyId,
        resolveThreadId: null,
        state: values,
        asNode: asNode ?? null,
      });
      return destConfig;
    }
    return original(config, values, asNode);
  };
}

/** AER write plugin that forks a LangGraph session. */
export class LangGraphForkPlugin {
  private readonly getAgent: () => ForkableAgent;
  private readonly workspaceIdResolver: () => string;

  constructor(args: {
    getAgent: () => ForkableAgent;
    workspaceIdResolver: () => string;
  }) {
    this.getAgent = args.getAgent;
    this.workspaceIdResolver = args.workspaceIdResolver;
  }

  async forkSession(args: {
    sessionId: string;
    executionId: string;
    historyId: string | null;
    state?: unknown;
  }): Promise<SessionForkResponse> {
    const agent = this.getAgent();
    const workspaceId = this.workspaceIdResolver();
    const attempt = currentAttemptContext();
    if (attempt !== null) {
      const identity = attempt.workflowIdentity;
      if (
        !identity ||
        args.sessionId !== identity.sessionId ||
        args.executionId !== identity.executionId ||
        workspaceId !== (identity.tenantScope?.workspaceId ?? "")
      ) {
        throw new Error(
          "durable fork request identity does not match the current OE attempt",
        );
      }
      // Durable CreateBranch accepts the exact committed snapshot. Do not
      // reinterpret an API state patch as part of that source snapshot.
      requireExactDurableBranch(args.state);
      const configurable: Record<string, unknown> = {
        thread_id: checkpointThreadId(
          {
            sessionId: args.sessionId,
            workspaceId,
            executionId: args.executionId,
          },
          agent.resolveThreadId ?? null,
        ),
        checkpoint_ns: "",
      };
      if (args.historyId) {
        configurable["checkpoint_id"] = args.historyId;
      }
      const [branch] = await forkDurableSession(agent.compiledGraph, {
        context: attempt,
        config: { configurable } as RunnableConfig,
        resolveThreadId: agent.resolveThreadId ?? null,
      });
      return branch;
    }
    const [branch] = await forkNativeSession(agent.compiledGraph, {
      ctx: {
        sessionId: args.sessionId,
        workspaceId,
        executionId: args.executionId,
      },
      historyId: args.historyId,
      resolveThreadId: agent.resolveThreadId ?? null,
      state: args.state,
    });
    return branch;
  }
}

async function createNativeBranch(args: {
  orgId: string;
  projectId: string;
  workspaceId: string;
  sessionId: string;
  executionId: string;
  branchKey: string;
  checkpointId: string | null;
}): Promise<SessionForkResponse> {
  const body: Record<string, unknown> = { branch_key: args.branchKey };
  if (args.checkpointId) body["checkpoint_id"] = args.checkpointId;
  return createBranch(args, body);
}

async function createDurableBranch(args: {
  orgId: string;
  projectId: string;
  workspaceId: string;
  sessionId: string;
  executionId: string;
  branchKey: string;
  stepOrdinal: number;
  state: StateSnapshot;
}): Promise<SessionForkResponse> {
  const body = {
    branch_key: args.branchKey,
    step_ordinal: args.stepOrdinal,
    state: JSON.parse(
      encodeProtoJson(StateSnapshotSchema, args.state),
    ) as unknown,
  };
  return createBranch(args, body);
}

async function createBranch(
  args: {
    orgId: string;
    projectId: string;
    workspaceId: string;
    sessionId: string;
    executionId: string;
  },
  body: Record<string, unknown>,
): Promise<SessionForkResponse> {
  const oeUrl = resolveOeUrl(getCurrentOeUrl() ?? "").replace(/\/$/, "");
  if (!oeUrl) {
    throw new Error("OE_URL is not configured; cannot fork a session");
  }
  const path =
    `/workflow/orgs/${quotePathSegment(args.orgId)}` +
    `/projects/${quotePathSegment(args.projectId)}` +
    `/workspaces/${quotePathSegment(args.workspaceId)}` +
    `/sessions/${quotePathSegment(args.sessionId)}` +
    `/executions/${quotePathSegment(args.executionId)}/branches`;
  const url = `${oeUrl}${path}`;
  const response = await fetchPlatform(url, {
    ...getFetchOptionsWithTLS(url),
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CREATE_BRANCH_TIMEOUT_MS),
  });
  if (response.status !== 200 && response.status !== 201) {
    throw new Error(
      `CreateBranch failed with HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`,
    );
  }
  const payload = (await response.json()) as Record<string, unknown>;
  const newSessionId = payload["session_id"];
  const newExecutionId = payload["execution_id"];
  if (
    typeof newSessionId !== "string" ||
    typeof newExecutionId !== "string" ||
    !newSessionId ||
    !newExecutionId
  ) {
    throw new Error("CreateBranch response is missing session identity");
  }
  return { sessionId: newSessionId, executionId: newExecutionId };
}

const BRANCH_KEY_SAFE = /[^A-Za-z0-9._/-]+/g;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Canonicalize a state patch for hashing: plain-object keys are sorted so
 * equivalent patches hash identically, while array order stays significant
 * (Python `json.dumps(..., sort_keys=True)` parity). Non-plain values keep
 * their own serialization contract.
 */
function canonicalStateValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalStateValue);
  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalStateValue(value[key])]),
    );
  }
  return value;
}

/**
 * Deterministic idempotency key for one fork: the same history, state patch,
 * and node always map to the same branch, so retries reuse the OE branch.
 */
function branchKey(
  historyId: string,
  state: unknown,
  asNode: string | null,
  taskId: string | null,
): string {
  const payload = JSON.stringify({
    state: isEmptyValues(state) ? null : canonicalStateValue(state),
    asNode: asNode ?? "",
    taskId: taskId ?? "",
  });
  const digest = createHash("sha256")
    .update(`${historyId}\0${payload}`)
    .digest("hex")
    .slice(0, 16);
  const safeHistory = historyId
    .replace(BRANCH_KEY_SAFE, "-")
    .replace(/^[-._/]+|[-._/]+$/g, "");
  return `update-${safeHistory.slice(0, 64) || "latest"}-${digest}`;
}
