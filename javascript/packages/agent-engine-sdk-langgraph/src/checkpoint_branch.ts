/** Copy a LangGraph checkpoint branch point into a new thread. */

import { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { CompiledStateGraph, StateSnapshot } from "@langchain/langgraph";
import { Overwrite } from "@langchain/langgraph";
import {
  emptyCheckpoint,
  type CheckpointMetadata,
} from "@langchain/langgraph-checkpoint";
import { isDeepStrictEqual } from "node:util";

import {
  CANCELLED_TURN_MESSAGE_ID_METADATA_KEY,
  CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY,
  CANCELLED_TURN_METADATA_KEY,
  cancelledOpenerIdsFromMetadata,
  clearQueuedTasks,
  terminalizeVersionsSeen,
} from "./cancelled_turn.js";
import { PlatformCheckpointer } from "./platform_checkpointer.js";

export interface LangGraphCheckpoint {
  readonly threadId: string;
  readonly checkpointId: string;
}

export function branchPointFromMetadata(
  metadata: Record<string, unknown> | null | undefined,
): LangGraphCheckpoint | undefined {
  const value = metadata?.["langgraph_branch_point"];
  if (value === undefined) return undefined;
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("LangGraph branch point must be an object");
  }
  const threadId = (value as Record<string, unknown>)["thread_id"];
  const checkpointId = (value as Record<string, unknown>)["checkpoint_id"];
  if (
    typeof threadId !== "string" ||
    threadId.length === 0 ||
    typeof checkpointId !== "string" ||
    checkpointId.length === 0
  ) {
    throw new Error("LangGraph branch point is incomplete");
  }
  return { threadId, checkpointId };
}

export function checkpointMetadata(
  checkpoint: LangGraphCheckpoint | undefined,
):
  | {
      langgraph_checkpoint: {
        thread_id: string;
        checkpoint_id: string;
      };
    }
  | undefined {
  if (checkpoint === undefined) return undefined;
  return {
    langgraph_checkpoint: {
      thread_id: checkpoint.threadId,
      checkpoint_id: checkpoint.checkpointId,
    },
  };
}

/** Root-graph config for one checkpoint coordinate. */
function checkpointConfig(source: LangGraphCheckpoint): RunnableConfig {
  return {
    configurable: {
      thread_id: source.threadId,
      checkpoint_ns: "",
      checkpoint_id: source.checkpointId,
    },
  };
}

/** Root-graph coordinate, or fail closed on a subgraph namespace. */
function rootGraphCheckpoint(config: RunnableConfig): LangGraphCheckpoint {
  const configurable = config.configurable ?? {};
  if ((configurable["checkpoint_ns"] ?? "") !== "") {
    throw new Error("LangGraph checkpoint branching supports root graphs only");
  }
  const threadId = configurable["thread_id"];
  const checkpointId = configurable["checkpoint_id"];
  if (typeof threadId !== "string" || typeof checkpointId !== "string") {
    throw new Error("LangGraph checkpoint coordinate is incomplete");
  }
  return {
    threadId,
    checkpointId,
  };
}

function stateUpdates(
  snapshot: StateSnapshot,
  pendingWrites: readonly (readonly [string, string, unknown])[],
): Array<{ values?: unknown; asNode?: string }> {
  return snapshot.tasks.flatMap((task) => {
    if (task.error !== undefined && task.error !== null) {
      throw new Error(
        "LangGraph checkpoint branching does not support failed history",
      );
    }
    if (task.state !== undefined && task.state !== null) {
      throw new Error(
        "LangGraph checkpoint branching does not support persistent subgraph history",
      );
    }
    if (
      task.interrupts.length > 0 &&
      (task.result === undefined || task.result === null)
    ) {
      const values: Record<string, unknown> = {};
      for (const [taskId, channel, value] of pendingWrites) {
        if (
          taskId !== task.id ||
          channel.startsWith("__") ||
          channel.startsWith("branch:")
        ) {
          continue;
        }
        if (channel in values) {
          throw new Error(
            `Interrupted LangGraph checkpoint task wrote channel ${channel} more than once`,
          );
        }
        values[channel] = value;
      }
      return Object.keys(values).length === 0
        ? []
        : [{ values, asNode: task.name }];
    }
    return [{ values: task.result, asNode: task.name }];
  });
}

/**
 * Graph values comparable after a branch copy. `bulkUpdateState` rebuilds
 * messages with new LangChain ids; those must not fail a content match.
 */
function stateWithoutMessageIds(value: unknown): unknown {
  if (value instanceof BaseMessage) {
    const serialized = value.toDict();
    const data = { ...serialized.data };
    delete data.id;
    return stateWithoutMessageIds({ ...serialized, data });
  }
  if (Array.isArray(value)) return value.map(stateWithoutMessageIds);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nested]) => [
        key,
        stateWithoutMessageIds(nested),
      ]),
    );
  }
  return value;
}

/** Replay plan: the source supersteps a branch copy materializes in order. */
export type CheckpointReplayPlan = Array<{
  updates: Array<{ values?: unknown; asNode?: string }>;
}>;

/**
 * Materialize a cancelled-turn marker tip as one synthetic superstep. The
 * marker checkpoint deliberately carries no checkpoint_writes rows, so
 * task-derived replay would lose its mutations (the rescued user message and
 * the stopped stamp). Overwrite replaces the messages channel wholesale with
 * the tip's authoritative list — the dest replay re-mints message ids, so
 * id-keyed reducer replacement cannot carry the stamp.
 */
function cancelledTipUpdates(
  snapshot: StateSnapshot,
  writerNode: string,
): Array<{ values?: unknown; asNode?: string }> {
  const values = snapshot.values as Record<string, unknown> | undefined;
  const messages = values?.["messages"];
  if (!Array.isArray(messages) || messages.length === 0) return [];
  return [
    { values: { messages: new Overwrite([...messages]) }, asNode: writerNode },
  ];
}

/** Node a synthetic superstep's writes are attributed to. */
function replayWriterNode(graph: CompiledStateGraph<unknown, unknown>): string {
  const names = Object.keys(graph.getGraph().nodes)
    .filter((node) => !node.startsWith("__"))
    .sort();
  return names[0] ?? "__start__";
}

/**
 * Load a completed root checkpoint, or fail closed.
 *
 * Fork validates this before CreateBranch so a missing or incomplete history
 * point does not create a branch session (Python
 * `completed_root_checkpoint` parity).
 */
export async function completedRootCheckpoint(
  graph: CompiledStateGraph<unknown, unknown>,
  args: { threadId: string; historyId: string | null },
): Promise<LangGraphCheckpoint> {
  const configurable: Record<string, unknown> = {
    thread_id: args.threadId,
    checkpoint_ns: "",
  };
  if (args.historyId) configurable["checkpoint_id"] = args.historyId;
  const snapshot = await graph.getState({ configurable });
  const found = snapshot.config.configurable ?? {};
  const checkpointId = found["checkpoint_id"];
  if (typeof checkpointId !== "string" || checkpointId === "") {
    throw new Error("session has no completed history to fork from");
  }
  if (found["checkpoint_ns"]) {
    throw new Error("LangGraph checkpoint branching supports root graphs only");
  }
  if (args.historyId && checkpointId !== args.historyId) {
    throw new Error("LangGraph branch point was not found");
  }
  if (snapshot.next.length > 0 || snapshot.tasks.length > 0) {
    throw new Error("history point is not a completed root checkpoint");
  }
  return { threadId: args.threadId, checkpointId };
}

/**
 * Build the copyable superstep plan, or raise before any destination write.
 *
 * Port of Python's `checkpoint_replay_plan`; fork builds this before
 * CreateBranch so the copied snapshot is the one that was validated.
 */
export interface CheckpointReplayPlanResult {
  supersteps: CheckpointReplayPlan;
  /** Every cancelled-opener id the source history records, for the copy. */
  cancelledOpeners: string[];
}

export async function checkpointReplayPlan(
  graph: CompiledStateGraph<unknown, unknown>,
  source: LangGraphCheckpoint,
): Promise<CheckpointReplayPlanResult> {
  const checkpointer = graph.checkpointer;
  if (checkpointer === undefined || typeof checkpointer === "boolean") {
    throw new Error("LangGraph checkpoint branching requires a checkpointer");
  }

  const sourceConfig = checkpointConfig(source);
  const selected = await graph.getState(sourceConfig);
  if (!isDeepStrictEqual(rootGraphCheckpoint(selected.config), source)) {
    throw new Error("LangGraph branch point was not found");
  }
  if (selected.next.length > 0 || selected.tasks.length > 0) {
    throw new Error(
      "LangGraph branch point must be a completed root checkpoint",
    );
  }

  const history: StateSnapshot[] = [];
  let historyConfig: RunnableConfig | undefined = sourceConfig;
  while (historyConfig !== undefined) {
    const snapshot = await graph.getState(historyConfig);
    rootGraphCheckpoint(snapshot.config);
    history.push(snapshot);
    historyConfig = snapshot.parentConfig;
  }

  const supersteps: CheckpointReplayPlan = [];
  const cancelledOpeners = new Set<string>();
  for (const snapshot of history.reverse()) {
    const metadata = snapshot.metadata as Record<string, unknown> | undefined;
    if (metadata?.[CANCELLED_TURN_METADATA_KEY] === true) {
      // Replay rebuilds checkpoints and drops the per-turn marker metadata, so
      // the copy re-records every opener the source history names — otherwise a
      // forked stopped turn would lose its badge.
      for (const openerId of cancelledOpenerIdsFromMetadata(metadata)) {
        cancelledOpeners.add(openerId);
      }
      // Lazy: the writer node is only needed for marker tips, and normal
      // forks must not require the graph's drawable form.
      const updates = cancelledTipUpdates(snapshot, replayWriterNode(graph));
      if (updates.length > 0) supersteps.push({ updates });
      continue;
    }
    const tuple = await checkpointer.getTuple(snapshot.config);
    const updates = stateUpdates(snapshot, tuple?.pendingWrites ?? []);
    if (updates.length > 0) supersteps.push({ updates });
  }
  if (supersteps.length === 0) {
    throw new Error("LangGraph branch point has no checkpoint history");
  }
  return { supersteps, cancelledOpeners: [...cancelledOpeners].sort() };
}

/**
 * Close the copied marker tip's artificially-triggered work. Replaying a
 * marker tip applies its message state wholesale (no task writes), so nodes
 * that never ran in the source have no `versions_seen` coverage on the dest
 * and would read as pending — a follow-up on the fork would resume the
 * cancelled super-step, and the copy's own source-equality check rejects a
 * non-terminal tip. The written tip keeps the marker: it is itself an
 * authoritative-values, writes-less checkpoint and must replay as one when
 * the fork becomes a fork source.
 */
async function terminalizeCopiedMarkerTip(
  graph: CompiledStateGraph<unknown, unknown>,
  threadId: string,
  cancelledMessageId?: unknown,
  cancelledOpeners: Iterable<string> = [],
): Promise<void> {
  const raw = graph.checkpointer;
  const saver = raw instanceof PlatformCheckpointer ? raw.native : raw;
  if (saver === undefined || saver === null || typeof saver === "boolean") {
    return;
  }
  const latest = await saver.getTuple({
    configurable: { thread_id: threadId },
  });
  if (latest === undefined) return;
  const channelVersions = { ...(latest.checkpoint.channel_versions ?? {}) };
  const versionsSeen: Record<string, Record<string, number | string>> = {};
  for (const [node, seen] of Object.entries(
    latest.checkpoint.versions_seen ?? {},
  )) {
    versionsSeen[node] = { ...seen };
  }
  terminalizeVersionsSeen(
    versionsSeen,
    channelVersions,
    Object.keys(graph.getGraph().nodes),
  );
  const checkpoint = emptyCheckpoint();
  checkpoint.channel_values = clearQueuedTasks({
    ...(latest.checkpoint.channel_values ?? {}),
  });
  checkpoint.channel_versions = channelVersions;
  checkpoint.versions_seen = versionsSeen;
  const metadata: CheckpointMetadata & Record<string, unknown> = {
    source: "loop",
    step: (latest.metadata?.step ?? -1) + 1,
    parents: {},
    [CANCELLED_TURN_METADATA_KEY]: true,
  };
  // Carry the source opener id onto the copy so the history reader still
  // classifies the forked opener as stopped — the reader derives the outcome
  // from this metadata, not from the message.
  if (typeof cancelledMessageId === "string" && cancelledMessageId !== "") {
    metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] = cancelledMessageId;
  }
  // Re-record every opener the source history named: replay drops the per-turn
  // marker metadata, so a fork of later history would otherwise lose the
  // earlier stopped turns.
  const carried = [...new Set(cancelledOpeners)]
    .filter((openerId) => openerId !== "")
    .sort();
  if (carried.length > 0) {
    metadata[CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY] = carried;
  }
  await saver.put(
    {
      configurable: {
        thread_id: threadId,
        checkpoint_ns: "",
        checkpoint_id: latest.checkpoint.id,
      },
    },
    checkpoint,
    metadata,
    channelVersions,
  );
}

/**
 * Rebuild `source` in the empty thread named by `targetConfig`.
 *
 * `plan` is the pre-dispatch validated plan from `checkpointReplayPlan`;
 * callers that validated before a side effect pass it so the copy can only
 * materialize that snapshot.
 */
export async function copyCheckpoint(
  graph: CompiledStateGraph<unknown, unknown>,
  targetConfig: RunnableConfig,
  source: LangGraphCheckpoint,
  plan?: CheckpointReplayPlanResult,
): Promise<void> {
  const checkpointer = graph.checkpointer;
  if (checkpointer === undefined || typeof checkpointer === "boolean") {
    throw new Error("LangGraph checkpoint branching requires a checkpointer");
  }

  const targetThreadId = targetConfig.configurable?.["thread_id"];
  if (typeof targetThreadId !== "string") {
    throw new Error("LangGraph branch destination thread is missing");
  }
  for await (const _ of checkpointer.list(
    { configurable: { thread_id: targetThreadId } },
    { limit: 1 },
  )) {
    throw new Error("LangGraph branch destination thread is not empty");
  }

  const { supersteps, cancelledOpeners } =
    plan ?? (await checkpointReplayPlan(graph, source));
  const sourceConfig = checkpointConfig(source);
  const selected = await graph.getState(sourceConfig);

  const configurable: Record<string, unknown> = {
    ...targetConfig.configurable,
    checkpoint_ns: "",
  };
  delete configurable["checkpoint_id"];
  const copiedConfig = await graph.bulkUpdateState(
    { ...targetConfig, configurable },
    supersteps,
  );
  const selectedMetadata = selected.metadata as
    | Record<string, unknown>
    | undefined;
  const markerTip = selectedMetadata?.[CANCELLED_TURN_METADATA_KEY] === true;
  // Re-record the source's cancelled openers on the copy whenever the history
  // held any — not only when the tip itself is a marker — so a fork of later
  // history keeps every earlier stopped turn classified.
  const copied =
    markerTip || cancelledOpeners.length > 0
      ? await (async () => {
          await terminalizeCopiedMarkerTip(
            graph,
            targetThreadId,
            selectedMetadata?.[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY],
            cancelledOpeners,
          );
          return graph.getState({
            configurable: { thread_id: targetThreadId, checkpoint_ns: "" },
          });
        })()
      : await graph.getState(copiedConfig);
  if (
    copied.next.length > 0 ||
    copied.tasks.length > 0 ||
    !isDeepStrictEqual(
      stateWithoutMessageIds(copied.values),
      stateWithoutMessageIds(selected.values),
    )
  ) {
    throw new Error("Copied LangGraph branch point does not match its source");
  }
}

/** True when `config`'s thread has no checkpoints yet. */
export async function destinationThreadIsEmpty(
  graph: CompiledStateGraph<unknown, unknown>,
  config: RunnableConfig,
): Promise<boolean> {
  const checkpointer = graph.checkpointer;
  if (checkpointer === undefined || typeof checkpointer === "boolean") {
    throw new Error("LangGraph checkpoint branching requires a checkpointer");
  }
  const targetThreadId = config.configurable?.["thread_id"];
  if (typeof targetThreadId !== "string") {
    throw new Error("LangGraph branch destination thread is missing");
  }
  for await (const _ of checkpointer.list(
    { configurable: { thread_id: targetThreadId } },
    { limit: 1 },
  )) {
    return false;
  }
  return true;
}

/**
 * Copy `source` onto an empty dest thread.
 *
 * In-app fork already materializes dest; OE still dispatches later dest
 * invokes with the source branch point. If dest already has checkpoints,
 * leave it alone (Python `initialize_checkpoint_branch` parity).
 */
export async function initializeCheckpointBranch(
  graph: CompiledStateGraph<unknown, unknown>,
  config: RunnableConfig,
  source: LangGraphCheckpoint,
): Promise<void> {
  if (!(await destinationThreadIsEmpty(graph, config))) return;
  await copyCheckpoint(graph, config, source);
}
