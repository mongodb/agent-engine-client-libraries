/** Translate LangGraph checkpoint namespaces into durable operation paths. */

import { CompiledStateGraph, getConfig } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { ChildOperationBoundary } from "@mongodb-js/agent-engine-runner-shared";

import { UnsupportedDurableGraphError } from "./platform_checkpointer.js";

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

export class DurableSubgraphResolver {
  private readonly topology: Topology;
  private readonly validationError: string | null;

  constructor(graph: CompiledStateGraph<unknown, unknown>) {
    const errors: string[] = [];
    this.topology = discover(graph as GraphWithBuilder, [], new Set(), errors);
    this.validationError = errors[0] ?? null;
  }

  get hasCompiledChildren(): boolean {
    return this.topology.children.size > 0;
  }

  validate(): void {
    if (this.validationError !== null) {
      throw new UnsupportedDurableGraphError(this.validationError);
    }
  }

  resolve = (): readonly ChildOperationBoundary[] => {
    const config = getConfig() as RunnableConfig | undefined;
    const namespace = config?.configurable?.["checkpoint_ns"] ?? "";
    if (typeof namespace !== "string") {
      throw new UnsupportedDurableGraphError(
        "LangGraph checkpoint namespace must be a string",
      );
    }
    return this.resolveNamespace(namespace);
  };

  resolveNamespace(namespace: string): readonly ChildOperationBoundary[] {
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
