import { AsyncLocalStorage } from "node:async_hooks";
import { createHash } from "node:crypto";

import type { Span } from "@opentelemetry/sdk-trace-base";

export const ATTR_INPUT_MESSAGES_OFFSET =
  "agent_engine.internal.input_messages.offset";
export const ATTR_INPUT_MESSAGES_TOTAL =
  "agent_engine.internal.input_messages.total";
export const ATTR_INPUT_MESSAGES_STREAM =
  "agent_engine.internal.input_messages.stream";
export const ATTR_INPUT_MESSAGES_RECORDED_BY =
  "agent_engine.internal.input_messages.recorded_by";
export const RECORDED_BY_CALLER = "caller";

const INPUT_MESSAGES_PREFIX = "llm.input_messages.";
const INPUT_MESSAGES_KEY = /^llm\.input_messages\.(\d+)\.(.+)$/;

export const RUNTIME_MODE_ATTRIBUTE = "agentic_platform.runtime_mode"; // open-source-refs:ignore — data contract

const MAX_TRACKED_CONVERSATIONS = 256;

type MessageRow = Record<string, unknown>;

function fingerprint(message: MessageRow): string {
  const payload = Object.keys(message)
    .sort()
    .map((field) => `${field}\u0001${String(message[field])}`)
    .join("\u0000");
  return createHash("blake2b512")
    .update(payload, "utf8")
    .digest("hex")
    .slice(0, 16);
}

export function hasInputMessages(attributes: Record<string, unknown>): boolean {
  return Object.keys(attributes).some((name) =>
    name.startsWith(INPUT_MESSAGES_PREFIX),
  );
}

const callerRecorded = new AsyncLocalStorage<boolean>();

export function runAsCallerRecordedConversation<T>(fn: () => T): T {
  return callerRecorded.run(true, fn);
}

export function markInputRecordedByCaller(span: Span): void {
  if (callerRecorded.getStore() === true) {
    span.setAttribute(ATTR_INPUT_MESSAGES_RECORDED_BY, RECORDED_BY_CALLER);
  }
}

export function omitInputMessages(
  attributes: Record<string, unknown>,
): Record<string, unknown> {
  const omitted: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(attributes)) {
    if (name.startsWith(INPUT_MESSAGES_PREFIX)) continue;
    omitted[name] = value;
  }
  delete omitted["input.value"];
  delete omitted["input.mime_type"];
  omitted[ATTR_INPUT_MESSAGES_RECORDED_BY] = RECORDED_BY_CALLER;
  return omitted;
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

  copy(): ConversationDeltaTracker {
    const clone = new ConversationDeltaTracker(this.maxConversations);
    clone.byConversation = new Map(this.byConversation);
    return clone;
  }

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

export function traceStream(resource: Record<string, unknown>): string {
  const runtimeMode =
    resource[RUNTIME_MODE_ATTRIBUTE] ?? process.env["RUNNER_MODE"];
  const parts = [resource["service.name"], runtimeMode].filter(
    (part): part is string => typeof part === "string" && part.length > 0,
  );
  return parts.length > 0 ? parts.join("|") : "unknown";
}
