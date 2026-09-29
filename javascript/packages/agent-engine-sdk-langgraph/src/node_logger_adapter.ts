/**
 * LangGraph callback adapter for `BaseExecutionCallback`.
 *
 * Bridges LangChain's `BaseCallbackHandler` interface to the framework-neutral
 * `BaseExecutionCallback` protocol. Extracts node names from LangGraph-specific
 * metadata and forwards events to the platform's NodeExecutionLogger.
 *
 * Port of `agent_engine_sdk_langgraph/node_logger_adapter.py`.
 *
 * **API difference vs Python:** LangChain.js exposes chain callbacks as
 * `handleChainStart` / `handleChainEnd` / `handleChainError` (not `on_chain_*`).
 * Method names and parameter order follow the LangChain.js contract — the
 * Python `on_*` names cannot be reused or LangChain will skip them.
 *
 * LangChain.js's chain-end / chain-error callbacks do **not** pass `metadata`
 * as a parameter (Python's do). To preserve parity, the adapter stores the
 * chain-start metadata keyed by `runId` and looks it up at end / error time.
 */

import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import { isGraphInterrupt } from "@langchain/langgraph";

import type { BaseExecutionCallback } from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";

const logger = getLogger("agent_engine_sdk_langgraph.node_logger_adapter");

// LangChain.js typings for chain values — `ChainValues = Record<string, unknown>`
type ChainValues = Record<string, unknown>;

// ---------------------------------------------------------------------------
// LangGraphCallbackAdapter
// ---------------------------------------------------------------------------

/**
 * Wraps a `BaseExecutionCallback` to satisfy LangChain's callback interface.
 *
 * Used by the LangGraph agent to forward node execution events to the
 * platform's framework-neutral `NodeExecutionLogger`.
 */
export class LangGraphCallbackAdapter extends BaseCallbackHandler {
  override name = "LangGraphCallbackAdapter";
  private readonly callback: BaseExecutionCallback;
  /**
   * Stores per-`runId` metadata captured in `handleChainStart` so `handleChainEnd`
   * and `handleChainError` (which don't receive `metadata` from LangChain.js)
   * can recover Python parity. Entries are deleted on end / error to bound the
   * map's size during long-running graphs.
   */
  private readonly runMetadata = new Map<
    string,
    Readonly<Record<string, unknown>>
  >();

  /**
   * `runId`s identified as conditional-edge router evaluations in
   * `handleChainStart`; their end / error callbacks are dropped so a ghost
   * pair never reaches the callback. Entries are deleted on end / error.
   */
  private readonly skippedRuns = new Set<string>();

  constructor(callback: BaseExecutionCallback) {
    super();
    this.callback = callback;
  }

  override handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    _runType?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    _runName?: string,
    parentRunId?: string,
  ): void {
    if (isRouterEvaluation(metadata, tags)) {
      this.skippedRuns.add(runId);
      return;
    }
    if (metadata !== undefined) {
      this.runMetadata.set(runId, metadata);
    }
    const nodeName = extractNodeName(chain, tags, metadata);
    if (!nodeName || nodeName.startsWith("RunnableSequence")) return;

    this.callback.onNodeStart(nodeName, inputs, {
      runId,
      ...(parentRunId !== undefined && { parentRunId }),
      ...(metadata !== undefined && { metadata }),
    });
  }

  override handleChainEnd(
    outputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
  ): void {
    if (this.skippedRuns.delete(runId)) return;
    const metadata = this.runMetadata.get(runId);
    this.runMetadata.delete(runId);

    const nodeName = extractNodeName(undefined, tags, metadata);
    if (!nodeName || nodeName.startsWith("RunnableSequence")) return;

    this.callback.onNodeEnd(nodeName, outputs, {
      runId,
      ...(parentRunId !== undefined && { parentRunId }),
      ...(metadata !== undefined && { metadata }),
    });
  }

  override handleChainError(
    error: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[],
  ): void {
    if (this.skippedRuns.delete(runId)) return;
    const metadata = this.runMetadata.get(runId);
    this.runMetadata.delete(runId);

    const nodeName = extractNodeName(undefined, tags, metadata);
    if (!nodeName || nodeName.startsWith("RunnableSequence")) return;

    if (isGraphInterrupt(error)) {
      // `onNodeSuspend` lives on `NullExecutionCallback` (and any subclass);
      // it isn't part of the base interface. Duck-type to honour it when present.
      const suspend = (
        this.callback as {
          onNodeSuspend?: (
            n: string,
            o: {
              runId: string;
              parentRunId?: string;
              metadata?: Record<string, unknown>;
            },
          ) => void;
        }
      ).onNodeSuspend;
      if (typeof suspend === "function") {
        suspend.call(this.callback, nodeName, {
          runId,
          ...(parentRunId !== undefined && { parentRunId }),
          ...(metadata !== undefined && { metadata }),
        });
      } else {
        // Backward compat: callbacks that predate `onNodeSuspend` fall back to
        // `onNodeEnd` so the node timing is closed out. The recorded status
        // will be "success" rather than "suspend" — acceptable until the
        // callback implements `onNodeSuspend`.
        logger.warn(
          `${this.callback.constructor.name} does not implement onNodeSuspend; ` +
            `GraphInterrupt on node ${JSON.stringify(nodeName)} recorded as success. ` +
            `Subclass NullExecutionCallback to fix this.`,
        );
        this.callback.onNodeEnd(
          nodeName,
          {},
          {
            runId,
            ...(parentRunId !== undefined && { parentRunId }),
            ...(metadata !== undefined && { metadata }),
          },
        );
      }
      return;
    }

    if (error instanceof Error && error.name === "AbortError") {
      // Cooperative cancellation (per-call stop, run drain, pod teardown), not
      // a node failure. The node closes as interrupted; an error row would
      // double-count the stop the tool-result row already carries.
      const interrupted = (
        this.callback as {
          onNodeInterrupted?: (
            n: string,
            o: {
              runId: string;
              parentRunId?: string;
              metadata?: Record<string, unknown>;
            },
          ) => void;
        }
      ).onNodeInterrupted;
      if (typeof interrupted === "function") {
        interrupted.call(this.callback, nodeName, {
          runId,
          ...(parentRunId !== undefined && { parentRunId }),
          ...(metadata !== undefined && { metadata }),
        });
      } else {
        // Backward compat: callbacks that predate `onNodeInterrupted` fall back
        // to `onNodeEnd` so the node timing is closed out (status "success").
        logger.warn(
          `${this.callback.constructor.name} does not implement onNodeInterrupted; ` +
            `AbortError on node ${JSON.stringify(nodeName)} recorded as success. ` +
            `Subclass NullExecutionCallback to fix this.`,
        );
        this.callback.onNodeEnd(
          nodeName,
          {},
          {
            runId,
            ...(parentRunId !== undefined && { parentRunId }),
            ...(metadata !== undefined && { metadata }),
          },
        );
      }
      return;
    }

    this.callback.onNodeError(
      nodeName,
      error instanceof Error ? error.message : String(error),
      {
        runId,
        ...(parentRunId !== undefined && { parentRunId }),
        ...(metadata !== undefined && { metadata }),
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Pregel stamps every node task run with a `graph:step:N` tag (it builds the
 * per-step child callback manager with it). Conditional-edge routers — any
 * middleware hook declaring `canJumpTo`, or the model node's tools/end edge —
 * run inside their source node's task, so they inherit the node's
 * `langgraph_node` metadata but not the tag. A run carrying the metadata
 * without the tag is a route evaluation, not a node execution.
 */
function isRouterEvaluation(
  metadata: Readonly<Record<string, unknown>> | undefined,
  tags: readonly string[] | undefined,
): boolean {
  if (!metadata || !("langgraph_node" in metadata)) return false;
  return !(tags ?? []).some((tag) => tag.startsWith("graph:step:"));
}

/** Extract the node name from LangGraph callback parameters. */
function extractNodeName(
  serialized: Serialized | undefined,
  tags: readonly string[] | undefined,
  metadata: Readonly<Record<string, unknown>> | undefined,
): string | undefined {
  if (metadata && "langgraph_node" in metadata) {
    const node = metadata["langgraph_node"];
    if (typeof node === "string") return node;
  }
  if (tags) {
    for (const tag of tags) {
      if (!tag.startsWith("seq:step:")) {
        return tag;
      }
    }
  }
  if (serialized && "name" in serialized) {
    // `Serialized` is a tagged union; bridge to plain Record once at the boundary.
    const name = (serialized as unknown as Record<string, unknown>)["name"];
    if (typeof name === "string") return name;
  }
  return undefined;
}
