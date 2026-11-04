/**
 * Apply the provider-specific shape of one LLM request.
 *
 * The adapter's normal work is provider-neutral: convert platform messages,
 * call the client, convert the response back. A provider can still require a
 * request shape that no earlier stage can express — a marker that has to sit on
 * one specific block, say, which a wrapped model loses long before the provider
 * call.
 *
 * Those rules live in one module per provider. The adapter calls
 * `prepareProviderCall` and never names a provider itself, so this dispatcher
 * is also the place a second provider would be wired in.
 */
import type { BaseMessage } from "@langchain/core/messages";

import type { Message } from "@mongodb-js/agent-engine-sdk";
import * as anthropicCache from "./anthropic_cache.js";
import { platformMessagesToLc } from "./messages.js";

export interface PreparedCall {
  messages: BaseMessage[];
  kwargs: Readonly<Record<string, unknown>>;
}

/**
 * Return the provider's request shape for one call.
 *
 * The caller's `kwargs` are never mutated, and a provider with no rules here
 * gets them back unchanged.
 */
export function prepareProviderCall(
  llm: unknown,
  messages: readonly Message[],
  kwargs: Readonly<Record<string, unknown>>,
): PreparedCall {
  const lcMessages = platformMessagesToLc(messages);
  if (!anthropicCache.matches(llm)) {
    return { messages: lcMessages, kwargs };
  }
  const [shapedMessages, shapedKwargs] = anthropicCache.prepare(
    llm,
    lcMessages,
    kwargs,
  );
  return { messages: shapedMessages, kwargs: shapedKwargs };
}
