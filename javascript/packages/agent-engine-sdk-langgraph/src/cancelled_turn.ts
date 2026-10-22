/**
 * Commit a terminal checkpoint for a cancelled turn so history can serve it.
 *
 * Nothing checkpoints on cancel: LangGraph commits only at super-step
 * boundaries, and the drain tears the loop down mid-super-step, so the turn
 * vanishes from playground history. This writes one synthetic root-namespace
 * checkpoint from the last committed state (never the in-flight super-step —
 * its channel values are mid-mutation), records the turn-opening message id in
 * the checkpoint metadata so the history reader can classify it as stopped,
 * and terminal-izes `versions_seen` for every graph node so the tip presents as
 * completed: the cancelled super-step must never look resumable, and fork
 * validation rejects snapshots with pending tasks.
 *
 * Port of Python's `agent_engine_sdk_langgraph/cancelled_turn.py` — the twins
 * must stay behaviourally identical.
 */

import {
  emptyCheckpoint,
  type Checkpoint,
  type CheckpointMetadata,
  type CheckpointTuple,
} from "@langchain/langgraph-checkpoint";
import { HumanMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { randomUUID } from "node:crypto";

export const CANCELLED_TURN_METADATA_KEY = "agent_engine_cancelled_turn";
export const CANCELLED_TURN_MESSAGE_ID_METADATA_KEY =
  "agent_engine_cancelled_turn_message_id";
/**
 * A fork carries the source thread's whole set of cancelled-opener ids onto its
 * copied tip under this plural key: replay rebuilds checkpoints and drops the
 * per-turn marker metadata, so the fork must re-record the set or a forked
 * stopped turn would lose its badge.
 */
export const CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY =
  "agent_engine_cancelled_turn_message_ids";

/**
 * Framework-owned channel carrying queued Send packets (the deep agents'
 * dispatched tool calls). Rebuilt from channel_values on read; absence reads
 * as an empty queue.
 */
export const TASKS_CHANNEL_KEY = "__pregel_tasks";

/** Duck-typed slice of the native saver the commit needs. */
export interface CancelledTurnSaver {
  getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined>;
  put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    newVersions: Record<string, number | string>,
  ): Promise<unknown>;
  getNextVersion(current: string | number | undefined): string | number;
}

/**
 * Write the cancelled-turn marker checkpoint; returns the turn's
 * user-message id (null when the turn has no user message). `turnMessageId`
 * is the id captured when the turn's graph input was built — identity, not
 * prompt text, so repeated or hook-rewritten prompts can't misidentify the
 * turn. `null` means no opener id was captured at build: a preparation-phase
 * cancel, a resume turn, or a hook return outside the supported message-likes
 * (those are converted and captured at build). A resume recovers its already
 * persisted opener from the loaded checkpoint (`resume`); otherwise the
 * stopped message is appended under a freshly minted id — never matched
 * against an older prompt's content. No `checkpoint_writes` rows are
 * attached — the next invoke's cancelled-predecessor fence stays a no-op on
 * this tip, and the half-dead super-step's orphaned writes are never joined
 * to it.
 */
export async function commitCancelledTurnCheckpoint(
  saver: CancelledTurnSaver,
  {
    threadId,
    userMessage,
    turnMessageId,
    nodeNames,
    resume = false,
  }: {
    threadId: string;
    userMessage: string;
    turnMessageId: string | null;
    nodeNames: Iterable<string>;
    /** The execution is a native HITL resume: its opener is already persisted. */
    resume?: boolean;
  },
): Promise<string | null> {
  const config: RunnableConfig = {
    configurable: { thread_id: threadId, checkpoint_ns: "" },
  };
  const latest = await saver.getTuple(config);

  let channelValues: Record<string, unknown> = {};
  let channelVersions: Record<string, number | string> = {};
  const versionsSeen: Record<string, Record<string, number | string>> = {};
  let parentId: string | undefined;
  let priorStep = -1;
  if (latest !== undefined) {
    channelValues = { ...(latest.checkpoint.channel_values ?? {}) };
    channelVersions = { ...(latest.checkpoint.channel_versions ?? {}) };
    for (const [node, seen] of Object.entries(
      latest.checkpoint.versions_seen ?? {},
    )) {
      versionsSeen[node] = { ...seen };
    }
    parentId = latest.checkpoint.id;
    priorStep = latest.metadata?.step ?? -1;
    clearQueuedTasks(channelValues);
  }

  const messages = [
    ...((channelValues["messages"] as unknown[] | undefined) ?? []),
  ];
  let resolvedId = locateTurnMessage(messages, turnMessageId);
  if (resolvedId === undefined && resume) {
    // A resume opens no new user message: the opener is the last human message
    // already committed to the thread. Recover it here, from the checkpoint the
    // commit already loaded, so a Stop that beats the preamble's capture still
    // stamps the persisted opener instead of appending a duplicate. Gated on
    // resume — a fresh-turn preparation cancel must append, never adopt an
    // older identical prompt.
    resolvedId = lastHumanMessageId(messages);
  }
  if (resolvedId === undefined && userMessage !== "") {
    // The cancel beat even the input checkpoint: inject the user message
    // under its captured id or the turn cannot exist in history at all.
    resolvedId = turnMessageId ?? `msg-cancelled-${randomUUID()}`;
    messages.push(new HumanMessage({ content: userMessage, id: resolvedId }));
    channelVersions["messages"] = saver.getNextVersion(
      channelVersions["messages"],
    );
  }
  channelValues["messages"] = messages;

  // Every graph node reads as caught up — including nodes that never ran and
  // therefore have no versions_seen entry. A node absent there presents as
  // pending, and fork validation rejects a tip with pending work.
  terminalizeVersionsSeen(versionsSeen, channelVersions, nodeNames);

  const checkpoint = emptyCheckpoint();
  checkpoint.channel_values = channelValues;
  checkpoint.channel_versions = channelVersions;
  checkpoint.versions_seen = versionsSeen;

  const metadata: CheckpointMetadata & Record<string, unknown> = {
    source: "loop",
    step: priorStep + 1,
    parents: {},
    [CANCELLED_TURN_METADATA_KEY]: true,
    ...(resolvedId !== undefined
      ? { [CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]: resolvedId }
      : {}),
  };
  await saver.put(
    {
      configurable: {
        thread_id: threadId,
        checkpoint_ns: "",
        checkpoint_id: parentId,
      },
    },
    checkpoint,
    metadata,
    channelVersions,
  );
  return resolvedId ?? null;
}

/**
 * Drop the framework's queued Send packets from copied channel values.
 *
 * The last committed checkpoint can carry Sends dispatched to a still-running
 * tool (the deep agents route tool calls through `__pregel_tasks`), and PUSH
 * work is not consumed by terminalizing `versionsSeen` — a marker built on it
 * would present pending tasks and be rejected by fork validation, and a
 * resume would run the cancelled work. Dropping the channel reads back as an
 * empty queue.
 */
export function clearQueuedTasks(
  channelValues: Record<string, unknown>,
): Record<string, unknown> {
  delete channelValues[TASKS_CHANNEL_KEY];
  return channelValues;
}

/**
 * Mark every node as having seen every current channel version. A node absent
 * from (or behind in) `versionsSeen` presents as pending: resume would
 * schedule it and fork validation rejects a tip with pending work. Shared by
 * the cancelled-turn commit and the fork copy of a marker-tipped thread, which
 * replays values without task writes.
 */
export function terminalizeVersionsSeen(
  versionsSeen: Record<string, Record<string, number | string>>,
  channelVersions: Record<string, number | string>,
  nodeNames: Iterable<string>,
): Record<string, Record<string, number | string>> {
  for (const node of new Set([...nodeNames, ...Object.keys(versionsSeen)])) {
    const seen = (versionsSeen[node] ??= {});
    for (const [channel, version] of Object.entries(channelVersions)) {
      seen[channel] = version;
    }
  }
  return versionsSeen;
}

/**
 * Resolve the turn-opening message by the id captured at input build. No
 * content fallback: an identical earlier prompt would make content matching
 * stamp the previous turn's opener, so a missing captured id is always
 * answered by appending, never by matching.
 */
function locateTurnMessage(
  messages: unknown[],
  turnMessageId: string | null,
): string | undefined {
  if (turnMessageId === null) return undefined;
  for (const message of messages) {
    if ((message as { id?: unknown } | null)?.id === turnMessageId) {
      return turnMessageId;
    }
  }
  return undefined;
}

/**
 * Id of the last human message in the thread, or undefined. Used only to
 * recover a native HITL resume's already-persisted opener (see above); a
 * fresh turn never reaches it, so it can never adopt an older identical prompt.
 */
function lastHumanMessageId(messages: unknown[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as {
      getType?: () => unknown;
      type?: unknown;
      id?: unknown;
    } | null;
    if (message === null || typeof message !== "object") continue;
    const type =
      typeof message.getType === "function" ? message.getType() : message.type;
    if (type !== "human") continue;
    return typeof message.id === "string" && message.id !== ""
      ? message.id
      : undefined;
  }
  return undefined;
}

/**
 * Opener ids a cancelled-turn checkpoint's metadata names.
 *
 * A marker checkpoint carries its own turn's singular opener id; a fork carries
 * the source thread's whole set on its copied tip as a plural list. Never read
 * from a message kwarg an agent could author, so both the reader and the fork
 * resolve the same trusted set.
 */
export function cancelledOpenerIdsFromMetadata(metadata: unknown): Set<string> {
  const ids = new Set<string>();
  if (metadata === null || typeof metadata !== "object") return ids;
  const record = metadata as Record<string, unknown>;
  if (record[CANCELLED_TURN_METADATA_KEY] !== true) return ids;
  const messageId = record[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY];
  if (typeof messageId === "string" && messageId !== "") ids.add(messageId);
  const carried = record[CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY];
  if (Array.isArray(carried)) {
    for (const candidate of carried) {
      if (typeof candidate === "string" && candidate !== "") ids.add(candidate);
    }
  }
  return ids;
}
