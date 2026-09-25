/** Native-checkpoint conversation-turn ingestion for the TypeScript runner. */

import { randomUUID } from "node:crypto";

import type { Message } from "@mongodb-js/agent-engine-sdk";
import type { MemoryRuntime } from "@mongodb-js/agent-engine-sdk-memory";

import { getLogger } from "./logger.js";
import { normalizeContent } from "./utils.js";

const logger = getLogger("agent_engine_runner_shared.memory_writer");

type RecordTurnArgs = Parameters<MemoryRuntime["recordTurn"]>[0];

export interface WriteTurnArgs {
  message: string;
  resultMessages: readonly Message[];
  userId?: string | null;
  sessionId?: string | null;
  includeUserTurn?: boolean;
}

export interface TurnMemoryWriter {
  readonly pendingWrites: number;
  writeTurnAsync(args: WriteTurnArgs): void;
  drain(timeoutMs?: number): Promise<boolean>;
  shutdown(): Promise<void>;
}

/**
 * Extract only the current turn from framework messages.
 *
 * The original user prompt is input, so many LangGraph update streams never
 * emit it. Prepend it explicitly, then skip through an echoed copy when one is
 * present so prior checkpoint history and the current prompt are not written
 * twice. Resume legs suppress every user message because they carry no new
 * user prompt.
 */
export function extractTurnMessages(
  message: string,
  resultMessages: readonly Message[],
  includeUserTurn = true,
): RecordTurnArgs[] {
  const turns: RecordTurnArgs[] = [];
  const normalizedMessage = normalizeContent(message);

  if (includeUserTurn && message) {
    turns.push({ role: "user", content: normalizedMessage });
  }

  let startIndex = 0;
  for (let index = resultMessages.length - 1; index >= 0; index -= 1) {
    const candidate = resultMessages[index];
    if (
      candidate?.role === "user" &&
      normalizeContent(candidate.content) === normalizedMessage
    ) {
      startIndex = index + 1;
      break;
    }
  }

  for (const resultMessage of resultMessages.slice(startIndex)) {
    const content = normalizeContent(resultMessage.content);
    if (resultMessage.role === "user") {
      if (!includeUserTurn) continue;
      turns.push({ role: "user", content });
      continue;
    }

    if (resultMessage.role === "assistant") {
      const toolCalls = resultMessage.toolCalls?.map((toolCall) => ({
        id: toolCall.id ?? "",
        name: toolCall.name ?? "",
        arguments: toolCall.args ?? {},
      }));
      if (!content && !toolCalls?.length) continue;
      turns.push({
        role: "assistant",
        content,
        toolCalls: toolCalls ?? null,
      });
      continue;
    }

    if (resultMessage.role === "tool") {
      const toolCallId = resultMessage.toolCallId ?? "";
      turns.push({
        role: "tool",
        content,
        toolCallId,
        toolName: resultMessage.name ?? "",
        isError: resultMessage.isError ?? false,
      });
    }
  }

  return turns;
}

function validateTurn(turn: RecordTurnArgs): void {
  if (turn.role === "user" && !turn.content) {
    throw new Error("user memory turns require content");
  }
  if (turn.role === "tool" && !turn.content) {
    throw new Error("tool memory turns require content");
  }
  if (turn.role === "tool" && !turn.toolCallId) {
    throw new Error("tool memory turns require a tool call id");
  }
}

/** Write one current-turn batch in message order. */
async function writeTurnToMemory(
  runtime: Pick<MemoryRuntime, "recordTurn">,
  args: WriteTurnArgs,
): Promise<void> {
  const sessionId = args.sessionId?.trim();
  if (!sessionId) {
    logger.warn("session_id required for memory write; skipping turn write");
    return;
  }
  const userId = args.userId?.trim();
  if (!userId) {
    logger.warn("user_id not set; skipping turn write");
    return;
  }

  const turns = extractTurnMessages(
    args.message,
    args.resultMessages,
    args.includeUserTurn ?? true,
  );
  for (const turn of turns) {
    validateTurn(turn);
    await runtime.recordTurn({
      ...turn,
      sessionId,
      userId,
      // The adapter retries transport failures. Keep one key for all attempts
      // of this individual turn so a committed response loss cannot duplicate it.
      idempotencyKey: randomUUID(),
    });
  }
}

/**
 * Process-wide, non-blocking turn writer.
 *
 * Batches are serialized so concurrent executions cannot interleave messages
 * within a conversation turn. Failures are logged and swallowed: Memory is a
 * best-effort side effect and must not replace an otherwise successful agent
 * result. `drain` lets session release and graceful app replacement wait for
 * already-queued writes.
 */
export class MemoryWriter implements TurnMemoryWriter {
  private tail: Promise<void> = Promise.resolve();
  private pending = 0;
  private closed = false;

  constructor(private readonly runtime: Pick<MemoryRuntime, "recordTurn">) {}

  get pendingWrites(): number {
    return this.pending;
  }

  writeTurnAsync(args: WriteTurnArgs): void {
    if (this.closed) {
      logger.warn("MemoryWriter is shut down; skipping turn write");
      return;
    }

    this.pending += 1;
    this.tail = this.tail
      .then(() => writeTurnToMemory(this.runtime, args))
      .catch((error: unknown) => {
        const kind = error instanceof Error ? error.name : typeof error;
        logger.warn(`Failed to write turn to memory (${kind})`);
      })
      .finally(() => {
        this.pending -= 1;
      });
  }

  async drain(timeoutMs?: number): Promise<boolean> {
    const deadline =
      timeoutMs === undefined ? null : Date.now() + Math.max(0, timeoutMs);
    while (this.pending > 0) {
      const pending = this.tail;
      if (deadline === null) {
        await pending;
        continue;
      }

      const remaining = deadline - Date.now();
      if (remaining <= 0) return false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const completed = await Promise.race([
        pending.then(() => true),
        new Promise<boolean>((resolve) => {
          timer = setTimeout(() => resolve(false), remaining);
        }),
      ]);
      if (timer !== undefined) clearTimeout(timer);
      if (!completed) return false;
    }
    return true;
  }

  async shutdown(): Promise<void> {
    if (this.closed) {
      await this.drain();
      return;
    }
    this.closed = true;
    await this.drain();
  }
}
