/**
 * Translate LangGraph checkpoint namespaces into durable operation paths.
 *
 * `Send` workers run concurrently in one superstep, so each one gets its own
 * operation-path segment named by its superstep and its index in that
 * superstep's send list. LangGraph orders the send list the same way on every
 * attempt; the task id and namespace change per attempt.
 */

import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { ChainValues } from "@langchain/core/utils/types";
import { CompiledStateGraph, getConfig } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  runWithOperationPathResolver,
  type ChildOperationBoundary,
} from "@mongodb-js/agent-engine-runner-shared";

import { UnsupportedDurableGraphError } from "./platform_checkpointer.js";

/** LangGraph's task-path tag for tasks created by `Send`. */
const PUSH = "__pregel_push";

/**
 * Name a top-level `Send` task by its superstep and send index.
 *
 * The superstep keeps successive `Send` rounds of one graph run apart. Other
 * tasks get no segment.
 */
function sendTaskBoundary(
  metadata: Record<string, unknown> | undefined,
): ChildOperationBoundary | null {
  const taskPath = metadata?.["langgraph_path"];
  const step = metadata?.["langgraph_step"];
  if (
    !Array.isArray(taskPath) ||
    taskPath.length < 2 ||
    taskPath[0] !== PUSH ||
    !isIndex(taskPath[1]) ||
    !isIndex(step)
  ) {
    return null;
  }
  const name = `langgraph.task:${PUSH}:${step}:${taskPath[1]}`;
  return { name, occurrenceKey: name, spawnedTask: true };
}

function isIndex(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value);
}

/**
 * Remember which checkpoint namespace segments belong to `Send` tasks.
 *
 * Inside a compiled child, LangGraph metadata describes only the child's own
 * task, and the namespace names the parent `Send` task by an attempt-local id.
 * A task's start callback carries both its namespace and its task path, so
 * recording it lets the resolver name the enclosing `Send` worker. One
 * recorder lives for one durable invocation and must be in the run's
 * callbacks.
 */
export class SendTaskRecorder extends BaseCallbackHandler {
  override name = "agent_engine_send_task_recorder";
  // Record before the task's own work starts.
  override awaitHandlers = true;
  private readonly boundaries = new Map<string, ChildOperationBoundary>();

  override handleChainStart(
    _chain: Serialized,
    _inputs: ChainValues,
    _runId: string,
    _parentRunId?: string,
    _tags?: string[],
    metadata?: Record<string, unknown>,
  ): void {
    const boundary = sendTaskBoundary(metadata);
    const namespace = metadata?.["langgraph_checkpoint_ns"];
    if (boundary === null || typeof namespace !== "string" || !namespace) {
      return;
    }
    this.boundaries.set(namespace.split("|").at(-1) as string, boundary);
  }

  /** Return the `Send` task boundary for one `node:task_id` segment. */
  boundaryFor(namespaceSegment: string): ChildOperationBoundary | null {
    return this.boundaries.get(namespaceSegment) ?? null;
  }
}

interface Topology {
  readonly nodeNames: ReadonlySet<string>;
  readonly children: ReadonlyMap<string, Topology>;
}

interface GraphNode {
  readonly runnable?: unknown;
  readonly retryPolicy?: unknown;
}

interface GraphBuilder {
  readonly nodes?: Record<string, GraphNode>;
}

type GraphWithBuilder = CompiledStateGraph<unknown, unknown> & {
  readonly builder?: GraphBuilder;
};

/**
 * Translate LangGraph nesting into framework-neutral durable boundaries.
 *
 * Construction discovers the compiled-child topology once. Activity admission
 * later resolves the running node's checkpoint namespace and `Send` task
 * against it.
 */
export class DurableSubgraphResolver {
  private readonly topology: Topology;
  private readonly validationError: string | null;
  private readonly scopeSendTasks: boolean;

  /**
   * @param options.scopeSendTasks False for Deep Agent graphs, whose recorded
   *   paths predate `Send` task segments.
   */
  constructor(
    graph: CompiledStateGraph<unknown, unknown>,
    options: { readonly scopeSendTasks?: boolean } = {},
  ) {
    const errors: string[] = [];
    this.topology = discover(graph as GraphWithBuilder, [], new Set(), errors);
    this.validationError = errors[0] ?? null;
    this.scopeSendTasks = options.scopeSendTasks ?? true;
  }

  get hasCompiledChildren(): boolean {
    return this.topology.children.size > 0;
  }

  validate(): void {
    if (this.validationError !== null) {
      throw new UnsupportedDurableGraphError(this.validationError);
    }
  }

  /**
   * Run `fn` with this resolver and the invocation's `Send` task recorder
   * bound. The recorder must also be in the callbacks of the graph run.
   */
  scope<T>(sendTasks: SendTaskRecorder, fn: () => T): T {
    const recorder = this.scopeSendTasks ? sendTasks : null;
    return runWithOperationPathResolver(() => this.resolve(recorder), fn);
  }

  /**
   * Resolve the LangGraph node that is admitting a durable activity.
   *
   * Reads the running node's config through `getConfig()`. Returns one
   * boundary for every compiled child graph in the active namespace, each
   * preceded by a `Send` task boundary when a `Send` started that child, and
   * a final `Send` task boundary when the running node is itself a `Send`
   * task. In a graph without compiled children, a `Send` task that runs a
   * nested graph itself also contributes its boundary. The root graph
   * returns no boundaries; runner-shared adds the fixed `agent/1` segment.
   */
  resolve = (
    sendTasks: SendTaskRecorder | null = null,
  ): readonly ChildOperationBoundary[] => {
    const config = getConfig() as RunnableConfig | undefined;
    const namespace = config?.configurable?.["checkpoint_ns"] ?? "";
    if (typeof namespace !== "string") {
      throw new UnsupportedDurableGraphError(
        "LangGraph checkpoint namespace must be a string",
      );
    }
    let boundaries: readonly ChildOperationBoundary[] = [];
    if (this.hasCompiledChildren) {
      boundaries = this.resolveNamespace(namespace, sendTasks);
    } else if (sendTasks !== null && namespace !== "") {
      // Without compiled children only Send tasks add segments. A Send worker
      // that runs a graph itself appears as an outer segment.
      boundaries = namespace
        .split("|")
        .slice(0, -1)
        .flatMap((segment) => sendTasks.boundaryFor(segment) ?? []);
    }
    if (!this.scopeSendTasks) return boundaries;
    const sendTask = sendTaskBoundary(
      config?.metadata as Record<string, unknown> | undefined,
    );
    return sendTask === null ? boundaries : [...boundaries, sendTask];
  };

  /**
   * Translate one checkpoint namespace into compiled-child boundaries. A
   * child segment that `sendTasks` recorded as a `Send` task is preceded by
   * that task's boundary.
   */
  private resolveNamespace(
    namespace: string,
    sendTasks: SendTaskRecorder | null = null,
  ): readonly ChildOperationBoundary[] {
    if (namespace === "") return [];

    let topology = this.topology;
    const boundaries: ChildOperationBoundary[] = [];
    const segments = namespace.split("|");
    for (const [index, segment] of segments.entries()) {
      const separator = segment.indexOf(":");
      const name = separator < 0 ? "" : segment.slice(0, separator);
      const occurrenceKey = separator < 0 ? "" : segment.slice(separator + 1);
      if (
        name === "" ||
        occurrenceKey === "" ||
        !topology.nodeNames.has(name)
      ) {
        throw unsupportedNamespace(namespace);
      }
      const child = topology.children.get(name);
      if (child === undefined) {
        if (index !== segments.length - 1) {
          throw unsupportedNamespace(namespace);
        }
        break;
      }
      const sendTask = sendTasks?.boundaryFor(segment) ?? null;
      if (sendTask !== null) boundaries.push(sendTask);
      boundaries.push({ name, occurrenceKey });
      topology = child;
    }
    return boundaries;
  }
}

/**
 * Reject a durable graph that sets a retry policy, on a node or as a default.
 *
 * A retry re-runs the node from the top. A durable node's side effects are
 * recorded activities, which would be replayed or repeated depending on why
 * the node ran again, and its own logic is deterministic, so there is nothing
 * for a node retry to recover.
 */
export function rejectNodeRetryPolicies(graph: unknown): void {
  const seen = new Set<object>();
  const visit = (current: unknown, path: readonly string[]): void => {
    if (typeof current !== "object" || current === null || seen.has(current)) {
      return;
    }
    const nodes = (current as GraphWithBuilder).builder?.nodes;
    if (nodes === undefined) return;
    seen.add(current);
    // A compiled graph's own policy is the default for all its nodes.
    const graphPolicy = (current as { retryPolicy?: unknown }).retryPolicy;
    if (graphPolicy !== undefined && graphPolicy !== null) {
      throw new UnsupportedDurableGraphError(
        `graph ${JSON.stringify(path.join("/") || "root")} sets a default ` +
          "retry policy; node retry policies are not supported by durable_workflow",
      );
    }
    // The compiled node carries the effective policy, which may come from
    // the builder's node defaults instead of the node's own spec.
    const compiledNodes =
      (current as { nodes?: Record<string, { retryPolicy?: unknown }> })
        .nodes ?? {};
    for (const [name, node] of Object.entries(nodes)) {
      const nodePath = [...path, name];
      const policy = node.retryPolicy ?? compiledNodes[name]?.retryPolicy;
      if (policy !== undefined && policy !== null) {
        throw new UnsupportedDurableGraphError(
          `node ${JSON.stringify(nodePath.join("/"))} sets a retry policy; ` +
            "node retry policies are not supported by durable_workflow",
        );
      }
      visit(node.runnable, nodePath);
    }
  };
  visit(graph, []);
}

function discover(
  graph: GraphWithBuilder,
  path: readonly string[],
  ancestors: ReadonlySet<object>,
  errors: string[],
): Topology {
  if (ancestors.has(graph)) {
    errors.push(
      `compiled subgraph topology at ${path.join("/")} is cyclic and unsupported`,
    );
    return { nodeNames: new Set(), children: new Map() };
  }

  const nodes = graph.builder?.nodes ?? {};
  const descendants = new Set(ancestors);
  descendants.add(graph);
  const children = new Map<string, Topology>();
  for (const [name, node] of Object.entries(nodes)) {
    if (!(node.runnable instanceof CompiledStateGraph)) continue;
    const childPath = [...path, name];
    if (node.runnable.checkpointer !== undefined) {
      errors.push(
        `compiled subgraph ${JSON.stringify(childPath.join("/"))} must use ` +
          "the default checkpointer=undefined; checkpointer=true, " +
          "checkpointer=false, and independent child checkpointers are not " +
          "supported by durable_workflow",
      );
    }
    children.set(
      name,
      discover(
        node.runnable as GraphWithBuilder,
        childPath,
        descendants,
        errors,
      ),
    );
  }
  return { nodeNames: new Set(Object.keys(nodes)), children };
}

function unsupportedNamespace(namespace: string): UnsupportedDurableGraphError {
  return new UnsupportedDurableGraphError(
    `unsupported LangGraph checkpoint namespace ${JSON.stringify(namespace)}`,
  );
}
