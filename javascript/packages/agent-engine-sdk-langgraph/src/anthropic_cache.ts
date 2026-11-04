/**
 * Anthropic prompt caching for the tool-pod LLM adapter.
 *
 * Anthropic caches a *prefix* of the request, in the order tools -> system ->
 * messages, and only where the request carries an explicit `cache_control`
 * breakpoint. Where the breakpoint sits decides what is cached:
 *
 * - on the last block of the system prompt it covers the tools and the system
 *   prompt only, so every conversation built from the same agent can read that
 *   prefix instead of writing it again;
 * - on the last message it covers the whole conversation prefix, which the next
 *   turn of a tool loop extends and re-reads.
 *
 * Nothing earlier in the pipeline can place one. The framework's prompt-caching
 * middleware is gated on the provider client class and the platform's
 * `SecureWrappedLLM` is not a provider client, so the middleware skips every
 * call. The agent pod also rebuilds each content block as `{type, text}`, so a
 * breakpoint an agent writes on a block is dropped before the tool pod sees it.
 *
 * The adapter is the last stop before the provider call, so the breakpoints are
 * placed here: after OE approval, immediately before the provider formats the
 * request, and out of reach of agent input.
 */
import type { BaseMessage } from "@langchain/core/messages";

const CACHE_CONTROL_OPTION = "cache_control";
const EPHEMERAL_BREAKPOINT: Readonly<Record<string, string>> = {
  type: "ephemeral",
};
// Anthropic rejects a breakpoint on thinking blocks, and an empty text block has
// nothing to cache. Every other block type the platform produces is eligible.
const NON_CACHEABLE_BLOCK_TYPES = new Set([
  "thinking",
  "redacted_thinking",
  "reasoning",
  "reasoning_content",
]);

/** Unwrap a `RunnableBinding` produced by `bindTools` to reach the real model. */
function unwrapChatModel(llm: unknown): unknown {
  if (llm !== null && typeof llm === "object") {
    const bound = (llm as { bound?: unknown }).bound;
    if (bound !== undefined) return bound;
  }
  return llm;
}

/**
 * True when the client speaks the Anthropic wire format.
 *
 * Detection is structural rather than an `instanceof`, so the adapter does not
 * depend on `@langchain/anthropic` being installed. `_llmType()` is the same
 * signal the provider itself uses; `getName()` covers versions that only expose
 * the class name.
 */
export function matches(llm: unknown): boolean {
  const model = unwrapChatModel(llm);
  if (model === null || typeof model !== "object") return false;
  const candidate = model as {
    _llmType?: () => unknown;
    getName?: () => unknown;
  };
  if (typeof candidate._llmType === "function") {
    if (candidate._llmType() === "anthropic") return true;
  }
  return (
    typeof candidate.getName === "function" &&
    candidate.getName() === "ChatAnthropic"
  );
}

/** True when the client declares the direct Anthropic API transport. */
function isDirectAnthropicTransport(llm: unknown): boolean {
  const model = unwrapChatModel(llm);
  if (model === null || typeof model !== "object") return false;
  const declared = (model as { _llmType?: () => unknown })._llmType;
  return typeof declared === "function" && declared() === "anthropic";
}

/**
 * Consume the request-level cache option and place the breakpoints.
 *
 * Two breakpoints, deliberately placed:
 *
 * - the system prompt's last block, so a new conversation reads tools + system
 *   from cache instead of rewriting them (`markAnthropicSystemPrompt`);
 * - the conversation tail. The direct API takes the request-level option, which
 *   is the only form that can mark a `tool_result` block; a translated
 *   transport forwards call options unchanged, so it gets a block marker
 *   instead (`markAnthropicBlockTail`) and a tool-result tail stays unmarked,
 *   because the client would nest the marker inside the block where the
 *   provider rejects it.
 *
 * The option is re-emitted with the normalized value so one TTL applies to both
 * breakpoints; Anthropic rejects a request whose longer TTL follows a shorter
 * one. The caller's `kwargs` are copied, never mutated.
 */
export function prepare(
  llm: unknown,
  messages: BaseMessage[],
  kwargs: Readonly<Record<string, unknown>>,
): [BaseMessage[], Record<string, unknown>] {
  const { [CACHE_CONTROL_OPTION]: requested, ...rest } = kwargs;
  const cacheControl =
    requested === undefined || requested === true
      ? EPHEMERAL_BREAKPOINT
      : isPlainObject(requested) && Object.keys(requested).length > 0
        ? requested
        : null;
  if (cacheControl === null) {
    return [messages, rest];
  }
  const hasConversation = messages.some(
    (message) => messageType(message) !== "system",
  );
  const sendTailOption = hasConversation && isDirectAnthropicTransport(llm);
  const marked = markAnthropicSystemPrompt(messages, cacheControl);
  return [
    sendTailOption ? marked : markAnthropicBlockTail(marked, cacheControl),
    sendTailOption ? { ...rest, [CACHE_CONTROL_OPTION]: cacheControl } : rest,
  ];
}

/**
 * Mark the last block of the first system message, or return messages as-is.
 *
 * The first message is the one Anthropic hoists into the request's `system`
 * field; the entry it creates covers the tools and the system prompt, so a
 * conversation that starts later can read it without writing it.
 */
function markAnthropicSystemPrompt(
  messages: readonly BaseMessage[],
  cacheControl: Record<string, unknown>,
): BaseMessage[] {
  const tagged = [...messages];
  const first = tagged[0];
  if (first === undefined) return tagged;
  if (messageType(first) === "system") {
    tagged[0] = attachAnthropicBreakpoint(first, cacheControl);
  }
  return tagged;
}

/**
 * Mark a translated transport's conversation tail with a block breakpoint.
 *
 * A tool-result tail stays unmarked: the client nests a marker placed on the
 * message content inside the `tool_result` block, where the provider rejects it,
 * and the adapter does not synthesize provider blocks.
 */
function markAnthropicBlockTail(
  messages: readonly BaseMessage[],
  cacheControl: Record<string, unknown>,
): BaseMessage[] {
  const tagged = [...messages];
  const lastIndex = tagged.length - 1;
  const last = tagged[lastIndex];
  if (last === undefined) return tagged;
  const lastType = messageType(last);
  if (lastType === "system" || lastType === "tool") return tagged;
  tagged[lastIndex] = attachAnthropicBreakpoint(last, cacheControl);
  return tagged;
}

function messageType(message: BaseMessage): string {
  const candidate = message as { getType?: () => unknown; type?: unknown };
  const type =
    typeof candidate.getType === "function"
      ? candidate.getType()
      : candidate.type;
  return typeof type === "string" ? type : "";
}

/**
 * Return `message` with a breakpoint on its last cacheable block.
 *
 * The message is cloned rather than mutated, so a caller holding the original
 * is unaffected.
 */
function attachAnthropicBreakpoint(
  message: BaseMessage,
  cacheControl: Record<string, unknown>,
): BaseMessage {
  const content = message.content;
  if (typeof content === "string") {
    if (content.length === 0) return message;
    return cloneMessage(message, [
      { type: "text", text: content, cache_control: { ...cacheControl } },
    ]);
  }
  if (!Array.isArray(content)) return message;
  const blocks = attachBreakpointToLastBlock(content, cacheControl);
  return blocks === undefined ? message : cloneMessage(message, blocks);
}

/**
 * Copy `blocks` with a breakpoint on the last cacheable one, or undefined.
 *
 * Undefined means leave the message untouched: no block can carry a breakpoint,
 * or the caller already placed one, in which case the caller's choice wins.
 */
function attachBreakpointToLastBlock(
  blocks: readonly unknown[],
  cacheControl: Record<string, unknown>,
): unknown[] | undefined {
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    if (!isPlainObject(block)) continue;
    const type = block["type"];
    if (typeof type !== "string") continue;
    if (NON_CACHEABLE_BLOCK_TYPES.has(type)) continue;
    if (block[CACHE_CONTROL_OPTION] != null) return undefined;
    if (type === "text" && !block["text"]) continue;
    const tagged = [...blocks];
    tagged[index] = { ...block, cache_control: { ...cacheControl } };
    return tagged;
  }
  return undefined;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Shallow-copy a message with new content, preserving its prototype. */
function cloneMessage(message: BaseMessage, content: unknown): BaseMessage {
  const clone = Object.assign(
    Object.create(Object.getPrototypeOf(message)) as BaseMessage,
    message,
  );
  (clone as unknown as { content: unknown }).content = content;
  return clone;
}
