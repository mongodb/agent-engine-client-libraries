/**
 * Trim an LLM span's input messages to the messages added since the last call.
 *
 * Every LLM call receives the whole conversation, so an OpenInference span
 * carries the entire transcript under `llm.input_messages.*`. A run with N
 * calls therefore stores the conversation N times and the trace grows
 * quadratically in the number of steps.
 *
 * The tracker rewrites each span's input messages to the messages added since
 * the previous call in the same trace and stream, and records where that slice
 * starts. A reader replays the slices to rebuild the full context. Spans
 * written before this existed carry no offset and read as a full list, so old
 * and new traces read the same way.
 *
 * Mirrors `agent_engine_runner_shared/tracing/conversation.py` — keep the
 * attribute names and the prefix rule identical.
 */

import { createHash } from "node:crypto";

export const ATTR_INPUT_MESSAGES_OFFSET =
  "agent_engine.internal.input_messages.offset";
export const ATTR_INPUT_MESSAGES_TOTAL =
  "agent_engine.internal.input_messages.total";
export const ATTR_INPUT_MESSAGES_STREAM =
  "agent_engine.internal.input_messages.stream";

const INPUT_MESSAGES_PREFIX = "llm.input_messages.";
const INPUT_MESSAGES_KEY = /^llm\.input_messages\.(\d+)\.(.+)$/;

// Resource attribute carrying the pod's role. The name predates the product
// rename and is the contract the runner already emits on every span, not
// branding: it is how a reader tells the agent process from the tool process.
export const RUNTIME_MODE_ATTRIBUTE = "agentic_platform.runtime_mode"; // open-source-refs:ignore — data contract

/**
 * One entry per in-flight conversation. Bounded because a long-lived pod
 * serves many traces; eviction only costs compression on a later span, never
 * correctness, since a missing entry reads as "store the full list".
 */
const MAX_TRACKED_CONVERSATIONS = 256;

type MessageRow = Record<string, unknown>;

function fingerprint(message: MessageRow): string {
  const payload = Object.keys(message)
    .sort()
    .map((field) => `${field}\u0001${String(message[field])}`)
    .join("\u0000");
  // 64 bits, matching the Python tracker's digest_size. The tracker holds one
  // digest per message, so a full-length digest would dominate its memory.
  return createHash("blake2b512")
    .update(payload, "utf8")
    .digest("hex")
    .slice(0, 16);
}

/** Whether a span carries an OpenInference input-message list. */
export function hasInputMessages(attributes: Record<string, unknown>): boolean {
  return Object.keys(attributes).some((name) =>
    name.startsWith(INPUT_MESSAGES_PREFIX),
  );
}

function collectMessages(
  attributes: Record<string, unknown>,
): Map<number, MessageRow> {
  const messages = new Map<number, MessageRow>();
  for (const [key, value] of Object.entries(attributes)) {
    const match = INPUT_MESSAGES_KEY.exec(key);
    if (!match) continue;
    const index = Number(match[1]);
    const field = match[2] ?? "";
    const row = messages.get(index) ?? {};
    row[field] = value;
    messages.set(index, row);
  }
  return messages;
}

export class ConversationDeltaTracker {
  private byConversation = new Map<string, string[]>();

  constructor(private readonly maxConversations = MAX_TRACKED_CONVERSATIONS) {}

  /**
   * Independent tracker holding the same progress. The exporter trims a batch
   * on a copy and adopts it only after the write succeeds, so a failed write
   * cannot advance progress the store never recorded.
   */
  copy(): ConversationDeltaTracker {
    const clone = new ConversationDeltaTracker(this.maxConversations);
    clone.byConversation = new Map(this.byConversation);
    return clone;
  }

  /**
   * Return `attributes` with only the new input messages kept. `streamId`
   * separates the agent-side span from the tool-pod span for the same call:
   * both carry the conversation, they are exported by different processes,
   * and their message indices must not be compared.
   */
  trim(
    attributes: Record<string, unknown>,
    traceId: string,
    streamId: string,
  ): Record<string, unknown> {
    const messages = collectMessages(attributes);
    if (messages.size === 0) return { ...attributes };

    const total = Math.max(...messages.keys()) + 1;
    const fingerprints = Array.from({ length: total }, (_, index) =>
      fingerprint(messages.get(index) ?? {}),
    );

    const key = `${streamId}\u0000${traceId}`;
    const stored = this.byConversation.get(key) ?? [];
    // Only a strict extension can be expressed as an offset. A retry or a
    // branch that rewrote history stores its full list instead.
    const extendsStored =
      stored.length <= total &&
      stored.every((digest, index) => fingerprints[index] === digest);
    const offset = extendsStored ? stored.length : 0;
    this.remember(key, fingerprints);

    const trimmed: Record<string, unknown> = {};
    for (const [name, value] of Object.entries(attributes)) {
      if (name.startsWith(INPUT_MESSAGES_PREFIX)) continue;
      trimmed[name] = value;
    }
    // OpenInference serializes the same prompt into input.value, so leaving it
    // keeps the trace quadratic; readers replay the trimmed message list
    // instead, and GET /events therefore previews no input for LLM spans.
    delete trimmed["input.value"];
    delete trimmed["input.mime_type"];
    for (let index = offset; index < total; index += 1) {
      for (const [field, value] of Object.entries(messages.get(index) ?? {})) {
        trimmed[`${INPUT_MESSAGES_PREFIX}${index - offset}.${field}`] = value;
      }
    }
    trimmed[ATTR_INPUT_MESSAGES_OFFSET] = offset;
    trimmed[ATTR_INPUT_MESSAGES_TOTAL] = total;
    trimmed[ATTR_INPUT_MESSAGES_STREAM] = streamId;
    return trimmed;
  }

  private remember(key: string, fingerprints: string[]): void {
    this.byConversation.delete(key);
    this.byConversation.set(key, fingerprints);
    while (this.byConversation.size > this.maxConversations) {
      const oldest = this.byConversation.keys().next().value;
      if (oldest === undefined) break;
      this.byConversation.delete(oldest);
    }
  }
}

/** Identify which process wrote a span (see `ConversationDeltaTracker`). */
export function traceStream(resource: Record<string, unknown>): string {
  const runtimeMode =
    resource[RUNTIME_MODE_ATTRIBUTE] ?? process.env["RUNNER_MODE"];
  const parts = [resource["service.name"], runtimeMode].filter(
    (part): part is string => typeof part === "string" && part.length > 0,
  );
  return parts.length > 0 ? parts.join("|") : "unknown";
}
