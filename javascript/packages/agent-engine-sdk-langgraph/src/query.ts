/**
 * LangGraph-backed implementation of runner-shared's `AERQueryPlugin`.
 *
 * Reads from the same MongoDB collections that LangGraph's `MongoDBSaver`
 * writes to (`checkpoints`, `checkpoint_writes`). Deserialization is
 * delegated to the saver's `serde` so this code does not need to know how
 * LangGraph encodes state on disk.
 *
 * Port of Python's `agent_engine_sdk_langgraph/query.py`. One structural
 * difference: the JS `MongoDBSaver` keeps its `Db` handle protected, so the
 * plugin is constructed with the `MongoClient` + database name instead.
 *
 * Checkpoints are keyed by a workspace-scoped `thread_id` — see
 * `thread_id.ts`. The plugin composes that scope onto the plain `session_id`
 * it receives before querying, and strips it back off before returning.
 * Reads never query the bare unscoped key when a workspace scope is known:
 * bare keys are shared across every workspace on the store.
 * An empty scope is legitimate only on explicitly unscoped runtimes (local
 * dev / tests, no `APP_ID`), whose writes are bare-keyed to match. Managed
 * runtimes carry `REQUIRE_PROJECT_SCOPED_DB` and reject a missing `APP_ID`.
 */

import type { Collection, Document, MongoClient } from "mongodb";
import type { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  LLMToolCall,
  type JsonValue,
  type SessionMessage,
  type SessionMessagesResponse,
  type SessionsSummaryResponse,
  type SessionSummary,
} from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";
import {
  sessionIdFromThreadId,
  threadIdsForQuery,
  threadIdsForSessionsQuery,
} from "./thread_id.js";

const logger = getLogger("agent_engine_sdk_langgraph.query");

const PREVIEW_MAX_CHARS = 80;
const PREVIEW_WRITES_PER_SESSION = 5;

// Cap on the bytes logged from a deserialization exception. langgraph's
// serde errors can carry chunks of the raw on-disk payload (and therefore
// user content) — truncate so a Splunk-visible warning never doubles as a
// data-leak vector.
const DESERIALIZATION_ERROR_MAX_CHARS = 200;

// Cap the per-session checkpoint scan in `getMessagesForSession`. LangGraph
// writes one checkpoint per graph step, so an unbounded scan is O(turns) in
// both DB I/O and deserialization cost. Sessions with more than this many
// checkpoints return the messages present in the most recent
// `MAX_CHECKPOINTS_PER_SESSION` checkpoints; older turns are not surfaced.
// Raise this if real sessions routinely exceed it.
const MAX_CHECKPOINTS_PER_SESSION = 500;

// Time budgets on the read paths so a slow / hung MongoDB fails the request
// fast instead of pinning it.
const AGGREGATION_TIMEOUT_MS = 15_000;
const MESSAGE_READ_TIMEOUT_MS = 20_000;

// Cap on how many per-session checkpoint scans the summaries path runs at
// once. A sessions-list request can name hundreds of sessions; without a bound
// each one would open its own checkpoint cursor simultaneously, so a single
// request could hold hundreds of concurrent Mongo cursors and deserialize
// hundreds of checkpoint windows in parallel.
const MESSAGE_COUNT_CONCURRENCY = 10;

// LangChain message type → platform Message role used on the wire.
// Matches the live invoke/stream vocabulary in sdk-core (`Role`: user, assistant,
// tool, system). "function" is an older LangChain message type collapsed onto
// "tool" so callers do not need to distinguish.
const ROLE_MAP: Record<string, string> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool",
  function: "tool",
};

/** Minimal structural view of a CheckpointTuple from `MongoDBSaver.list()`. */
interface CheckpointTupleLike {
  checkpoint?: unknown;
}

export interface LangGraphQueryPluginParams {
  saver: MongoDBSaver;
  client: MongoClient;
  dbName: string;
  workspaceId?: string;
  workspaceIdResolver?: () => string | null | undefined;
}

/** AERQueryPlugin backed by LangGraph's MongoDBSaver. */
export class LangGraphQueryPlugin {
  private readonly saver: MongoDBSaver;
  private readonly checkpointsCollection: Collection<Document>;
  private readonly writesCollection: Collection<Document>;
  private readonly workspaceId: string;
  private readonly workspaceIdResolver:
    | (() => string | null | undefined)
    | undefined;

  constructor(params: LangGraphQueryPluginParams) {
    this.saver = params.saver;
    const db = params.client.db(params.dbName);
    this.checkpointsCollection = db.collection(
      params.saver.checkpointCollectionName,
    );
    this.writesCollection = db.collection(
      params.saver.checkpointWritesCollectionName,
    );
    this.workspaceId = params.workspaceId ?? "";
    this.workspaceIdResolver = params.workspaceIdResolver;
  }

  /**
   * Resolve the workspace scope for a read. "" is a legitimate scope — an
   * explicitly unscoped runtime (local dev / tests, no APP_ID) whose
   * checkpoints are bare-keyed by construction, matching the write path. The
   * runner rejects this state when managed-runtime scoping is required. A
   * resolver returning null/undefined signals a genuinely unresolvable scope
   * (a custom resolver that cannot make that call) and fails closed rather
   * than silently reading the shared store.
   */
  private effectiveWorkspaceId(): string {
    if (this.workspaceIdResolver !== undefined) {
      const resolved = this.workspaceIdResolver();
      if (resolved === null || resolved === undefined) {
        throw new Error(
          "checkpoint workspace scope unavailable; refusing to serve " +
            "session queries without a workspace",
        );
      }
      return resolved;
    }
    return this.workspaceId;
  }

  async getSummariesForSessions(
    sessionIds: string[],
  ): Promise<SessionsSummaryResponse> {
    if (sessionIds.length === 0) {
      return { sessions: [] };
    }
    return {
      sessions: await this.aggregateSessionSummaries([...sessionIds]),
    };
  }

  async getMessagesForSession(
    sessionId: string,
  ): Promise<SessionMessagesResponse> {
    const authoritative = await this.loadAuthoritativeMessages(sessionId);
    if (authoritative === null) {
      return { messages: [] };
    }

    const { messages, firstAppearanceByIndex } = authoritative;
    return {
      messages: messages.map((message, messageIndex) =>
        toSessionMessage({
          message,
          sessionId,
          messageIndex,
          timestamp: firstAppearanceByIndex.get(messageIndex) ?? "",
        }),
      ),
    };
  }

  /**
   * Resolve a session's authoritative message list: the `messages` channel of
   * the most recent checkpoint that carries a usable one, plus each message
   * index's first-appearance timestamp. Returns null when no checkpoint has a
   * usable `messages` channel.
   *
   * Both the messages endpoint and the sessions-list `message_count` derive
   * from this so the two always agree — `message_count` reports the
   * number of messages, not the number of checkpoints.
   */
  private async loadAuthoritativeMessages(sessionId: string): Promise<{
    messages: unknown[];
    firstAppearanceByIndex: Map<number, string>;
  } | null> {
    // A corrupt checkpoint here propagates to the caller rather than being
    // silently skipped: that's the right call for state-store corruption. On
    // the messages endpoint it surfaces as a 500; `collectMessageCounts`
    // downgrades it per session so one bad session can't fail a whole list.
    const threadIds = threadIdsForQuery(sessionId, this.effectiveWorkspaceId());
    const checkpointsNewestFirst =
      await this.collectCheckpointsForThreadIds(threadIds);

    // The most recent checkpoint with a usable `messages` list is the
    // authoritative message list.
    const checkpointsWithMessagesNewestFirst: Array<
      [CheckpointTupleLike, unknown[]]
    > = [];
    for (const checkpointTuple of checkpointsNewestFirst) {
      const messages = extractMessages(checkpointTuple);
      if (messages !== null) {
        checkpointsWithMessagesNewestFirst.push([checkpointTuple, messages]);
      }
    }

    const newestEntry = checkpointsWithMessagesNewestFirst[0];
    if (newestEntry === undefined) {
      return null;
    }

    return {
      messages: newestEntry[1],
      firstAppearanceByIndex: attributeFirstAppearanceTimestamps(
        checkpointsWithMessagesNewestFirst,
      ),
    };
  }

  /**
   * Materialize a bounded checkpoint window across one or more thread keys.
   *
   * The scan is wrapped in a wall-clock deadline rather than checked between
   * yields: the driver's `MongoClient` is constructed without socket/read
   * timeouts, so a stalled cursor can leave the `for await` parked forever and
   * an in-loop clock check would never run.
   */
  private async collectCheckpointsForThreadIds(
    threadIds: string[],
  ): Promise<CheckpointTupleLike[]> {
    return withDeadline(
      this.streamCheckpointsForThreadIds(threadIds),
      MESSAGE_READ_TIMEOUT_MS,
      `session message read timed out after ${MESSAGE_READ_TIMEOUT_MS}ms`,
    );
  }

  private async streamCheckpointsForThreadIds(
    threadIds: string[],
  ): Promise<CheckpointTupleLike[]> {
    const checkpoints: CheckpointTupleLike[] = [];
    for (const threadId of threadIds) {
      for await (const checkpointTuple of this.saver.list(
        { configurable: { thread_id: threadId } },
        { limit: MAX_CHECKPOINTS_PER_SESSION },
      )) {
        checkpoints.push(checkpointTuple as CheckpointTupleLike);
      }
    }

    checkpoints.sort((left, right) =>
      checkpointTimestamp(right).localeCompare(checkpointTimestamp(left)),
    );
    return checkpoints.slice(0, MAX_CHECKPOINTS_PER_SESSION);
  }

  // ------------------------------------------------------------------
  // Session summaries
  // ------------------------------------------------------------------

  private async aggregateSessionSummaries(
    sessionIds: string[],
  ): Promise<SessionSummary[]> {
    const workspaceId = this.effectiveWorkspaceId();
    const threadIds = threadIdsForSessionsQuery(sessionIds, workspaceId);
    const aggregateDocs = await this.checkpointsCollection
      .aggregate(sessionSummariesPipeline(threadIds), {
        maxTimeMS: AGGREGATION_TIMEOUT_MS,
      })
      .toArray();

    const aggregateDocBySessionId = new Map<string, Document>();
    for (const aggregateDoc of aggregateDocs) {
      const sessionId = sessionIdFromThreadId(
        String(aggregateDoc["_id"] ?? ""),
        workspaceId,
      );
      if (sessionId === "") continue;
      const existing = aggregateDocBySessionId.get(sessionId);
      aggregateDocBySessionId.set(
        sessionId,
        mergeSessionSummaryAggregate(existing, aggregateDoc),
      );
    }

    const summarySessionIds = [...aggregateDocBySessionId.keys()];
    const [previewBySessionId, messageCountBySessionId] = await Promise.all([
      this.collectPreviews(summarySessionIds),
      this.collectMessageCounts(summarySessionIds),
    ]);

    return [...aggregateDocBySessionId.entries()].map(
      ([sessionId, aggregateDoc]) => ({
        session_id: sessionId,
        last_activity: isoOrEmpty(aggregateDoc["latest_ts"]),
        created_at: isoOrEmpty(aggregateDoc["first_ts"]),
        message_count: messageCountBySessionId.get(sessionId) ?? 0,
        first_message_preview: previewBySessionId.get(sessionId) ?? "",
      }),
    );
  }

  /**
   * Count messages per session from the same authoritative list the messages
   * endpoint returns, so a summary's `message_count` matches that endpoint
   * exactly. Sessions with no usable checkpoint count as 0.
   *
   * Unlike the single-session messages endpoint — where a corrupt checkpoint
   * rightly surfaces as a 500 — a failed read here degrades that one session's
   * count to 0 and leaves the rest of the list intact. One unreadable session
   * must not take down a summaries request naming hundreds of others.
   */
  private async collectMessageCounts(
    sessionIds: string[],
  ): Promise<Map<string, number>> {
    const counts = new Map<string, number>();
    for (
      let offset = 0;
      offset < sessionIds.length;
      offset += MESSAGE_COUNT_CONCURRENCY
    ) {
      const chunk = sessionIds.slice(
        offset,
        offset + MESSAGE_COUNT_CONCURRENCY,
      );
      await Promise.all(
        chunk.map(async (sessionId) => {
          try {
            const authoritative =
              await this.loadAuthoritativeMessages(sessionId);
            counts.set(sessionId, authoritative?.messages.length ?? 0);
          } catch (exc) {
            logger.warn(
              { sessionId, error: String(exc) },
              "message count unavailable for session; reporting 0",
            );
            counts.set(sessionId, 0);
          }
        }),
      );
    }
    return counts;
  }

  private async collectPreviews(
    sessionIds: string[],
  ): Promise<Map<string, string>> {
    const previews = new Map<string, string>();
    if (sessionIds.length === 0) return previews;

    const workspaceId = this.effectiveWorkspaceId();
    const threadIds = threadIdsForSessionsQuery(sessionIds, workspaceId);
    const aggregateDocs = await this.writesCollection
      .aggregate(messagePreviewsPipeline(threadIds), {
        maxTimeMS: AGGREGATION_TIMEOUT_MS,
      })
      .toArray();

    const writesBySessionId = new Map<string, Document[]>();
    for (const aggregateDoc of aggregateDocs) {
      const sessionId = sessionIdFromThreadId(
        String(aggregateDoc["_id"] ?? ""),
        workspaceId,
      );
      if (sessionId === "") continue;
      const candidateWrites = Array.isArray(aggregateDoc["writes"])
        ? (aggregateDoc["writes"] as Document[])
        : [];
      const existing = writesBySessionId.get(sessionId) ?? [];
      writesBySessionId.set(sessionId, [...existing, ...candidateWrites]);
    }

    for (const [sessionId, candidateWrites] of writesBySessionId.entries()) {
      const preview = await this.firstNonEmptyHumanPreview(
        candidateWrites,
        sessionId,
      );
      if (preview !== "") previews.set(sessionId, preview);
    }
    return previews;
  }

  /**
   * Return the first non-empty human-message preview across `writes`.
   *
   * `writes` is the per-session accumulation from the preview pipeline,
   * ordered oldest-first. Returns `""` if no write decodes to a human
   * message with text content.
   */
  private async firstNonEmptyHumanPreview(
    candidateWrites: Document[],
    sessionId: string,
  ): Promise<string> {
    for (const writeRecord of candidateWrites) {
      const preview = await this.decodeHumanPreview(writeRecord, sessionId);
      if (preview !== "") return preview;
    }
    return "";
  }

  /**
   * Decode a single `checkpoint_writes` document and return the first
   * human-authored text content it contains, truncated.
   *
   * Returns `""` for non-human writes, malformed writes, and any
   * deserialization failures (logged at WARNING for observability).
   */
  private async decodeHumanPreview(
    writeRecord: Document,
    sessionId: string,
  ): Promise<string> {
    const serdeTypeTag = writeRecord["type"];
    const serdeData = toSerdeData(writeRecord["value"]);
    if (typeof serdeTypeTag !== "string" || serdeTypeTag === "") return "";
    if (serdeData === null) return "";

    let deserialized: unknown;
    try {
      deserialized = await this.saver.serde.loadsTyped(serdeTypeTag, serdeData);
    } catch (exc) {
      // Warn (not debug) so schema drift / data corruption is visible in
      // Splunk instead of silently serving empty previews.
      const excName = exc instanceof Error ? exc.name : typeof exc;
      const excMessage = exc instanceof Error ? exc.message : String(exc);
      logger.warn(
        `preview deserialization failed for session ${sessionId}: ` +
          `${excName}: ${excMessage.slice(0, DESERIALIZATION_ERROR_MAX_CHARS)}`,
      );
      return "";
    }

    const decodedMessages = Array.isArray(deserialized)
      ? deserialized
      : [deserialized];
    for (const decodedMessage of decodedMessages) {
      const content = humanContent(decodedMessage);
      if (content !== "") return content.slice(0, PREVIEW_MAX_CHARS);
    }
    return "";
  }
}

// ----------------------------------------------------------------------
// Module-level helpers — kept stateless and pure so they're easy to read
// and easy to test in isolation.
// ----------------------------------------------------------------------

/**
 * Reject with `message` if `work` has not settled within `timeoutMs`.
 *
 * The abandoned work keeps running until its own I/O settles — there is no
 * cancellation token to hand the saver's async generator — but the caller is
 * released on time instead of waiting on a cursor that may never yield.
 */
async function withDeadline<T>(
  work: Promise<T>,
  timeoutMs: number,
  message: string,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  try {
    return await Promise.race([work, deadline]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function mergeSessionSummaryAggregate(
  existing: Document | undefined,
  newDoc: Document,
): Document {
  if (existing === undefined) return { ...newDoc };

  const latestTs = maxTimestamp(existing["latest_ts"], newDoc["latest_ts"]);
  const firstTs = minTimestamp(existing["first_ts"], newDoc["first_ts"]);

  return {
    _id: existing["_id"] ?? newDoc["_id"],
    latest_ts: latestTs,
    first_ts: firstTs,
  };
}

function maxTimestamp(left: unknown, right: unknown): unknown {
  if (left instanceof Date && right instanceof Date) {
    return left > right ? left : right;
  }
  return left ?? right;
}

function minTimestamp(left: unknown, right: unknown): unknown {
  if (left instanceof Date && right instanceof Date) {
    return left < right ? left : right;
  }
  return left ?? right;
}

/**
 * Aggregation pipeline that reduces per-session checkpoint rows to one
 * summary row per `thread_id` (first/last activity). The message count is
 * derived separately from the authoritative message list (see
 * `collectMessageCounts`) so it matches the messages endpoint.
 */
function sessionSummariesPipeline(threadIds: string[]): Document[] {
  return [
    { $match: { thread_id: { $in: threadIds } } },
    { $addFields: { _ts: { $toDate: "$_id" } } },
    {
      $group: {
        _id: "$thread_id",
        latest_ts: { $max: "$_ts" },
        first_ts: { $min: "$_ts" },
      },
    },
    { $sort: { latest_ts: -1 } },
  ];
}

/**
 * Aggregation pipeline that pulls the oldest few `messages`-channel writes
 * per session so we can extract a human-message preview.
 *
 * `$firstN` caps the per-session write count at the accumulator — the full
 * sorted stream is never materialized in memory. The preceding
 * `$sort: {_id: 1}` makes "first N" mean "oldest N".
 */
function messagePreviewsPipeline(threadIds: string[]): Document[] {
  return [
    {
      $match: {
        thread_id: { $in: threadIds },
        channel: "messages",
      },
    },
    { $sort: { _id: 1 } },
    {
      $group: {
        _id: "$thread_id",
        writes: {
          $firstN: {
            input: { type: "$type", value: "$value" },
            n: PREVIEW_WRITES_PER_SESSION,
          },
        },
      },
    },
  ];
}

/**
 * Normalize a `checkpoint_writes.value` field to what `serde.loadsTyped`
 * accepts. The driver surfaces it as a BSON `Binary`; mirror the saver's own
 * read path (`doc.value.value("utf8")`). Returns null for unusable values.
 */
function toSerdeData(value: unknown): Uint8Array | string | null {
  if (typeof value === "string" || value instanceof Uint8Array) return value;
  if (
    value !== null &&
    typeof value === "object" &&
    typeof (value as { value?: unknown }).value === "function"
  ) {
    return (
      value as { value: (encoding: string) => Uint8Array | string }
    ).value("utf8");
  }
  return null;
}

/**
 * Return the `messages` channel from a CheckpointTuple if it's a list,
 * otherwise null.
 *
 * A null return signals "skip this checkpoint" — the channel is either
 * absent, malformed, or the checkpoint itself isn't an object.
 */
function extractMessages(
  checkpointTuple: CheckpointTupleLike,
): unknown[] | null {
  const checkpoint = checkpointTuple.checkpoint;
  if (checkpoint === null || typeof checkpoint !== "object") return null;
  const channelValues = (checkpoint as Record<string, unknown>)[
    "channel_values"
  ];
  if (channelValues === null || typeof channelValues !== "object") return null;
  const messages = (channelValues as Record<string, unknown>)["messages"];
  return Array.isArray(messages) ? messages : null;
}

/**
 * Read `checkpoint.ts` — LangGraph's own ISO-8601 timestamp field —
 * falling back to the empty string if it's missing.
 */
function checkpointTimestamp(checkpointTuple: CheckpointTupleLike): string {
  const checkpoint = checkpointTuple.checkpoint;
  if (checkpoint === null || typeof checkpoint !== "object") return "";
  const ts = (checkpoint as Record<string, unknown>)["ts"];
  return ts === null || ts === undefined || ts === "" ? "" : String(ts);
}

/**
 * For each message index in the conversation history, return the timestamp
 * of the *earliest* checkpoint where that index existed.
 *
 * The caller passes pairs in newest-first order (the natural `list()`
 * direction); this function walks them oldest-first so each index records
 * its first appearance.
 */
function attributeFirstAppearanceTimestamps(
  checkpointsWithMessagesNewestFirst: Array<[CheckpointTupleLike, unknown[]]>,
): Map<number, string> {
  const firstAppearanceByIndex = new Map<number, string>();
  for (const [checkpointTuple, messagesAtThisStep] of [
    ...checkpointsWithMessagesNewestFirst,
  ].reverse()) {
    const timestamp = checkpointTimestamp(checkpointTuple);
    for (
      let messageIndex = 0;
      messageIndex < messagesAtThisStep.length;
      messageIndex++
    ) {
      if (!firstAppearanceByIndex.has(messageIndex)) {
        firstAppearanceByIndex.set(messageIndex, timestamp);
      }
    }
  }
  return firstAppearanceByIndex;
}

/**
 * Return a LangChain message's type tag ("human", "ai", "tool", ...).
 * Prefers `getType()` (the modern accessor on revived BaseMessage
 * instances); falls back to the `type` property for stand-ins that don't
 * implement the message protocol.
 */
function messageType(message: unknown): string {
  if (message === null || typeof message !== "object") return "";
  const candidate = message as {
    getType?: () => unknown;
    type?: unknown;
  };
  if (typeof candidate.getType === "function") {
    try {
      return String(candidate.getType() ?? "");
    } catch {
      return "";
    }
  }
  return candidate.type === null || candidate.type === undefined
    ? ""
    : String(candidate.type);
}

/**
 * Return a flat-string view of a LangChain message's content.
 *
 * Uses `BaseMessage.text` when available — it collapses string/list content
 * to a string and extracts only `type: "text"` blocks. Falls back to
 * `String(content)` for stand-ins that don't implement the LangChain
 * message protocol (e.g. test doubles).
 */
function messageText(message: unknown): string {
  if (message === null || typeof message !== "object") return "";
  const candidate = message as { text?: unknown; content?: unknown };
  if (typeof candidate.text === "string") return candidate.text;
  const content = candidate.content;
  if (content === null || content === undefined || content === "") return "";
  return typeof content === "string" ? content : String(content);
}

function humanContent(decodedMessage: unknown): string {
  if (messageType(decodedMessage) !== "human") return "";
  return messageText(decodedMessage);
}

function toSessionMessage(args: {
  message: unknown;
  sessionId: string;
  messageIndex: number;
  timestamp: string;
}): SessionMessage {
  const candidate =
    args.message !== null && typeof args.message === "object"
      ? (args.message as {
          id?: unknown;
          name?: unknown;
          additional_kwargs?: unknown;
        })
      : {};
  const rawMessageType = messageType(args.message);
  const role = ROLE_MAP[rawMessageType] ?? rawMessageType;
  const name = typeof candidate.name === "string" ? candidate.name : "";
  const messageId =
    typeof candidate.id === "string" && candidate.id !== ""
      ? candidate.id
      : `msg-${args.sessionId}-${args.messageIndex}`;
  return {
    id: messageId,
    role,
    content: messageText(args.message),
    timestamp: args.timestamp,
    session_id: args.sessionId,
    name,
    tool_calls: toolCalls(args.message),
    tool_call_id: toolCallId(args.message),
    additional_kwargs: jsonSafeMetadata(candidate.additional_kwargs),
  };
}

function toolCalls(message: unknown): LLMToolCall[] | null {
  if (message === null || typeof message !== "object") return null;
  const rawToolCalls = (message as { tool_calls?: unknown }).tool_calls;
  if (!Array.isArray(rawToolCalls)) return null;
  const calls = rawToolCalls
    .filter((toolCall): toolCall is Record<string, unknown> => {
      return toolCall !== null && typeof toolCall === "object";
    })
    .map((toolCall) => {
      const data: ConstructorParameters<typeof LLMToolCall>[0] = {};
      if (typeof toolCall["id"] === "string") data.id = toolCall["id"];
      if (typeof toolCall["name"] === "string") data.name = toolCall["name"];
      const args = toolCall["args"] ?? toolCall["arguments"];
      const safeArgs = jsonSafeValue(args);
      if (safeArgs !== undefined) data.args = safeArgs;
      if (typeof toolCall["type"] === "string") data.type = toolCall["type"];
      if (typeof toolCall["index"] === "number") data.index = toolCall["index"];
      return new LLMToolCall(data);
    });
  return calls.length > 0 ? calls : null;
}

function toolCallId(message: unknown): string | null {
  if (message === null || typeof message !== "object") return null;
  const value = (message as { tool_call_id?: unknown }).tool_call_id;
  return typeof value === "string" && value !== "" ? value : null;
}

function jsonSafeValue(value: unknown): JsonValue | undefined {
  if (value === undefined) return undefined;
  try {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) return undefined;
    return JSON.parse(encoded) as JsonValue;
  } catch {
    return undefined;
  }
}

/**
 * Keep only JSON-serializable metadata fields, mirroring Python's
 * per-field JsonValue validation: a single non-JSON field is dropped (with
 * a warning) without discarding the rest of the metadata.
 */
function jsonSafeMetadata(value: unknown): Record<string, unknown> | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const metadata: Record<string, unknown> = {};
  for (const [key, rawField] of Object.entries(value)) {
    try {
      const encoded = JSON.stringify(rawField);
      if (encoded === undefined) {
        logger.warn(`Skipping non-JSON message metadata field: ${key}`);
        continue;
      }
      metadata[key] = JSON.parse(encoded);
    } catch {
      logger.warn(`Skipping non-JSON message metadata field: ${key}`);
    }
  }
  return Object.keys(metadata).length > 0 ? metadata : null;
}

function isoOrEmpty(value: unknown): string {
  if (value instanceof Date) return value.toISOString();
  return "";
}
