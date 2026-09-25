/** LangGraph interrupt validation and suspension output helpers. */

import type { BaseMessage } from "@langchain/core/messages";
import type { Interrupt } from "@langchain/langgraph";
import {
  JsonValueSchema,
  type AgentOutput,
  type StreamEvent,
} from "@mongodb-js/agent-engine-sdk";

import { lcMessagesToPlatform } from "./messages.js";

export interface PublicInterrupt {
  readonly id: string;
  readonly value: unknown;
}

function interruptValueType(value: unknown): string {
  if (value === null) return "null";
  if (typeof value !== "object") return typeof value;
  const constructor = (value as { constructor?: { name?: unknown } })
    .constructor;
  return typeof constructor?.name === "string" ? constructor.name : "object";
}

export function validateInterruptValue(id: string, value: unknown): void {
  try {
    if (JsonValueSchema.safeParse(value).success) return;
  } catch {
    // Recursive values can overflow the recursive schema before returning a result.
  }
  throw new Error(
    `LangGraph interrupt "${id}" has a non-JSON-serializable value of type ${interruptValueType(value)}`,
  );
}

export function interruptSnapshot(
  captured: readonly Interrupt[],
): PublicInterrupt[] {
  const interrupts = captured.map((interrupt) => {
    if (typeof interrupt.id !== "string" || interrupt.id.length === 0) {
      throw new Error("LangGraph returned an empty interrupt id");
    }
    validateInterruptValue(interrupt.id, interrupt.value);
    return { id: interrupt.id, value: interrupt.value };
  });
  const interruptIds = interrupts.map((interrupt) => interrupt.id);
  if (new Set(interruptIds).size !== interruptIds.length) {
    throw new Error("LangGraph returned duplicate interrupt ids");
  }
  return interrupts;
}

export function resumeSchema(
  interruptIds: readonly string[],
): Record<string, unknown> {
  const resumeMap = {
    type: "object",
    required: [...interruptIds],
    properties: Object.fromEntries(interruptIds.map((id) => [id, {}])),
    additionalProperties: false,
  };
  return {
    type: "object",
    required: ["resume_map"],
    properties: { resume_map: resumeMap },
    additionalProperties: true,
  };
}

export function invokeSuspendOutput(args: {
  response: string;
  threadId: string;
  checkpointId: string | undefined;
  interruptValues: readonly unknown[];
  interrupts?: readonly PublicInterrupt[];
  messageCount: number;
  resumed: boolean;
}): AgentOutput {
  return {
    response: {
      response: args.response,
      execution_id: args.threadId,
      status: "suspended",
      suspend_context: {
        checkpoint_id: args.checkpointId ?? null,
        interrupt_values: args.interruptValues as never,
      },
      message_count: args.messageCount,
      resumed: args.resumed,
      ...(args.interrupts === undefined
        ? {}
        : {
            interrupts: args.interrupts as never,
            resume_schema: resumeSchema(
              args.interrupts.map((item) => item.id),
            ) as never,
          }),
    },
  };
}

export function streamHitlSuspendEvent(args: {
  interrupts: readonly PublicInterrupt[];
  checkpointId: string | undefined;
  resumed: boolean;
  messages: readonly BaseMessage[];
}): StreamEvent {
  const interruptIds = args.interrupts.map((interrupt) => interrupt.id);
  return {
    data: {
      // Retained so older runners can still consume a new adapter event.
      suspend_payload: args.interrupts[0]?.value as never,
      interrupts: args.interrupts as never,
      resume_schema: resumeSchema(interruptIds) as never,
      resumed: args.resumed,
      messages: lcMessagesToPlatform(args.messages) as never,
      metadata: { checkpoint_id: args.checkpointId ?? null },
    },
    event: "suspend",
  };
}
