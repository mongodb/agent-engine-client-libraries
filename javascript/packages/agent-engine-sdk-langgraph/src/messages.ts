/**
 * LangChain message conversion layer.
 *
 * Converts between LangChain message types (AIMessage, HumanMessage, etc.)
 * and the framework-neutral Message model from agent-engine-sdk.
 *
 * Port of `agent_engine_sdk_langgraph/messages.py`.
 */

import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  coerceMessageLikeToMessage,
  HumanMessage,
  type MessageContent,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { Overwrite } from "@langchain/langgraph";
import type {
  ToolCall,
  ToolCallChunk as LCToolCallChunk,
} from "@langchain/core/messages/tool";
import {
  ChatGenerationChunk,
  type ChatGeneration,
  type ChatResult,
} from "@langchain/core/outputs";

import {
  type AnyContentBlock,
  type DocumentBlock,
  type ImageBlock,
  type JsonValue,
  type LLMResponse,
  type LLMStreamChunk,
  LLMTokenUsage,
  LLMToolCall,
  type Message,
  type Role,
  type TextBlock,
} from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";

/** Drop keys whose value is `undefined` — local helper (not in agent-engine-sdk). */
function stripUndefined<T extends Record<string, unknown>>(
  obj: T,
): { [K in keyof T]: Exclude<T[K], undefined> } {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== undefined) result[key] = value;
  }
  return result as { [K in keyof T]: Exclude<T[K], undefined> };
}

/**
 * Build an `LLMTokenUsage` from a raw dict. His agent-engine-sdk class has only a
 * data-constructor (no static `fromDict`); this helper centralises the cast.
 */
function newLLMTokenUsageFromDict(
  data: Record<string, unknown>,
): LLMTokenUsage {
  return new LLMTokenUsage(
    data as ConstructorParameters<typeof LLMTokenUsage>[0],
  );
}

/** `LLMTokenUsage` no longer exposes `.toDict()`; round-trip via JSON-stripping. */
function llmTokenUsageToDict(usage: LLMTokenUsage): Record<string, JsonValue> {
  return stripUndefined({
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    prompt_tokens: usage.promptTokens,
    completion_tokens: usage.completionTokens,
    total_tokens: usage.totalTokens,
    model: usage.model,
  });
}

const logger = getLogger("agent_engine_sdk_langgraph.messages");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const FRAMEWORK_LANGCHAIN = "langchain" as const;

// Keep in sync with Python's langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES,
// consumed by langgraph.prebuilt.tool_node.msg_content_output. Unsupported
// block lists are intentionally JSON-stringified on the durable path.
const TOOL_MESSAGE_BLOCK_TYPES = new Set([
  "text",
  "image_url",
  "image",
  "json",
  "search_result",
  "custom_tool_call_output",
  "document",
  "file",
]);

/** Return the messages inside a LangGraph Overwrite or ordinary message value. */
export function unwrapMessageValues(value: unknown): unknown[] {
  if (Overwrite.isInstance(value) && "__overwrite__" in value) {
    const overwritten = value.__overwrite__;
    return Array.isArray(overwritten) ? [...overwritten] : [overwritten];
  }
  return Array.isArray(value) ? [...value] : [value];
}

function copyMessageWithId(message: BaseMessage, id: string): BaseMessage {
  const MessageType = message.constructor as new (
    fields: Record<string, unknown>,
  ) => BaseMessage;
  return new MessageType({ ...message, id });
}

/** Normalize graph-input messages and assign IDs where the caller requires them. */
export function assignMissingGraphInputMessageIds(
  graphInput: unknown,
  missingId: (index: number) => string,
): unknown {
  if (
    graphInput === null ||
    typeof graphInput !== "object" ||
    Array.isArray(graphInput)
  ) {
    return graphInput;
  }

  const input = graphInput as Record<string, unknown>;
  const rawMessages = input["messages"];
  if (rawMessages === undefined || rawMessages === null) return graphInput;

  const overwritten =
    Overwrite.isInstance(rawMessages) && "__overwrite__" in rawMessages;
  const messages = unwrapMessageValues(rawMessages).map((value, index) => {
    const message = coerceMessageLikeToMessage(value as never);
    return message.id !== undefined && message.id !== null
      ? message
      : copyMessageWithId(message, missingId(index));
  });

  return {
    ...input,
    messages: overwritten
      ? rawMessages instanceof Overwrite
        ? new Overwrite(messages)
        : {
            ...(rawMessages as Record<string, unknown>),
            __overwrite__: messages,
          }
      : messages,
  };
}

/** Normalize a platform tool result for LangChain's ToolMessage content field. */
export function toToolMessageContent(content: unknown): MessageContent {
  if (typeof content === "string") return content;
  if (
    Array.isArray(content) &&
    content.every((item) => {
      if (typeof item !== "object" || item === null) return false;
      const blockType = (item as Record<string, unknown>)["type"];
      return (
        typeof blockType === "string" && TOOL_MESSAGE_BLOCK_TYPES.has(blockType)
      );
    })
  ) {
    return content as MessageContent;
  }
  try {
    return JSON.stringify(content) ?? String(content);
  } catch {
    return String(content);
  }
}

// ---------------------------------------------------------------------------
// Internal helpers — content conversion
// ---------------------------------------------------------------------------

/**
 * Convert LangChain message content to platform format.
 *
 * LangChain messages can have multimodal content as a list of content blocks.
 * This function preserves the structure for multimodal content while keeping
 * simple string content as-is.
 */
function convertLcContentToPlatform(
  content: unknown,
): string | AnyContentBlock[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content);

  const blocks: AnyContentBlock[] = [];
  for (const block of content) {
    if (block !== null && typeof block === "object") {
      const b = block as Record<string, unknown>;
      const blockType = typeof b["type"] === "string" ? b["type"] : "text";

      if (blockType === "text") {
        const text = typeof b["text"] === "string" ? b["text"] : "";
        blocks.push({ type: "text", text } satisfies TextBlock);
      } else if (blockType === "image" || blockType === "image_url") {
        const url = extractImageUrl(b);
        if (url) {
          const imageBlock: ImageBlock =
            typeof b["mime_type"] === "string"
              ? { type: "image", url, mime_type: b["mime_type"] }
              : { type: "image", url };
          blocks.push(imageBlock);
        } else {
          logger.warn("Image block missing url; skipping");
        }
      } else if (blockType === "file" || blockType === "document") {
        const url = typeof b["url"] === "string" ? b["url"] : undefined;
        if (url) {
          const mimeType =
            typeof b["mime_type"] === "string" ? b["mime_type"] : undefined;
          const filename =
            typeof b["filename"] === "string" ? b["filename"] : undefined;
          const doc: DocumentBlock = {
            type: "document",
            url,
            ...(mimeType !== undefined && { mime_type: mimeType }),
            ...(filename !== undefined && { filename }),
          };
          blocks.push(doc);
        } else {
          logger.warn("Document block missing url; skipping");
        }
      } else {
        logger.warn(
          `Unknown content block type=${blockType}; serializing to TextBlock`,
        );
        blocks.push({
          type: "text",
          text: JSON.stringify(b),
        } satisfies TextBlock);
      }
    } else {
      blocks.push({ type: "text", text: String(block) } satisfies TextBlock);
    }
  }

  if (blocks.length === 1 && blocks[0]?.type === "text") return blocks[0].text;
  return blocks.length > 0 ? blocks : "";
}

/** Extract the URL from LangChain's image_url shape (either string or `{url, ...}`). */
function extractImageUrl(block: Record<string, unknown>): string | undefined {
  const imageData = block["image_url"];
  if (imageData !== undefined) {
    if (typeof imageData === "string") return imageData;
    if (imageData !== null && typeof imageData === "object") {
      const url = (imageData as Record<string, unknown>)["url"];
      return typeof url === "string" ? url : undefined;
    }
    return undefined;
  }
  const url = block["url"];
  return typeof url === "string" ? url : undefined;
}

/**
 * Serialize LangChain tool calls.
 *
 * LangChain's ToolCall is a TypedDict — at runtime, tool_calls on an
 * AIMessage are always plain objects with `name`, `args`, `id`, and
 * `type` keys (enforced by LangChain's model validator).
 */
function serializeToolCalls(toolCalls: readonly ToolCall[]): LLMToolCall[] {
  const requiredKeys = ["name", "args", "id"] as const;
  const result: LLMToolCall[] = [];
  for (const tc of toolCalls) {
    if (tc === null || typeof tc !== "object") {
      throw new TypeError(`Expected object tool call, got ${typeof tc}`);
    }
    const tcObj = tc as unknown as Record<string, unknown>;
    const missing = requiredKeys.filter((k) => !(k in tcObj));
    if (missing.length > 0) {
      logger.warn(
        `Tool call missing keys ${missing.join(", ")}: ${JSON.stringify(tc)}`,
      );
    }
    result.push(
      new LLMToolCall(tcObj as ConstructorParameters<typeof LLMToolCall>[0]),
    );
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API — single message conversion
// ---------------------------------------------------------------------------

const ROLE_MAP: Readonly<Record<string, Role>> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool",
};

/** Convert a LangChain message to a platform Message. */
export function lcToPlatformMessage(lcMessage: BaseMessage): Message {
  const role: Role = ROLE_MAP[lcMessage.type] ?? "user";

  const content = convertLcContentToPlatform(lcMessage.content);

  let toolCalls: LLMToolCall[] | undefined;
  // LangChain JS's `AIMessageChunk` is NOT a subclass of `AIMessage` (unlike
  // Python), so `instanceof AIMessage` misses chunks aggregated by
  // `BaseChatModel.invoke` from `_streamResponseChunks`. Accept both, and also
  // fall through on `type === "ai"` for already-deserialized messages from
  // the LangGraph checkpointer.
  if (
    (lcMessage instanceof AIMessage ||
      lcMessage instanceof AIMessageChunk ||
      lcMessage.type === "ai") &&
    Array.isArray(
      (lcMessage as unknown as { tool_calls?: unknown[] }).tool_calls,
    ) &&
    (lcMessage as unknown as { tool_calls: unknown[] }).tool_calls.length > 0
  ) {
    const rawCalls = (lcMessage as unknown as { tool_calls: ToolCall[] })
      .tool_calls;
    toolCalls = serializeToolCalls(rawCalls);
  }

  let toolCallId: string | undefined;
  let isError: boolean | undefined;
  if (lcMessage instanceof ToolMessage) {
    toolCallId = lcMessage.tool_call_id;
    isError = lcMessage.status === "error";
  }

  const rawAdditionalKwargs = nonEmptyDict(lcMessage.additional_kwargs);
  let additionalKwargs = rawAdditionalKwargs;
  if (
    toolCalls !== undefined &&
    rawAdditionalKwargs?.["tool_calls"] !== undefined
  ) {
    // LangChain JS assembles provider stream fragments under
    // additional_kwargs.tool_calls. The completed semantic tool_calls above
    // are the portable source of truth and are what durable replay rebuilds.
    const { tool_calls: _providerFragments, ...portableAdditionalKwargs } =
      rawAdditionalKwargs;
    additionalKwargs = nonEmptyDict(portableAdditionalKwargs);
  }
  const responseMetadata = nonEmptyDict(lcMessage.response_metadata);

  return {
    role,
    content,
    ...(toolCalls !== undefined && { toolCalls }),
    ...(toolCallId !== undefined && { toolCallId }),
    ...(isError !== undefined && { isError }),
    ...(lcMessage.name !== undefined &&
      lcMessage.name !== null && { name: lcMessage.name }),
    ...(lcMessage.id !== undefined &&
      lcMessage.id !== null && { id: lcMessage.id }),
    ...(additionalKwargs !== undefined && {
      additionalKwargs,
    }),
    ...(responseMetadata !== undefined && {
      responseMetadata,
    }),
  };
}

function nonEmptyDict(
  data: Record<string, unknown> | undefined,
): Readonly<Record<string, JsonValue>> | undefined {
  if (data === undefined || Object.keys(data).length === 0) return undefined;
  // LangChain typings allow `unknown` values but in practice they are JsonValue-compatible.
  // The cast is documented here rather than narrowing every field at every call site.
  return data as Readonly<Record<string, JsonValue>>;
}

/**
 * Convert platform content to LangChain's `MessageContent` type.
 *
 * LangChain 1.x's `ContentBlock` interface is more strict than our internal
 * shape (it expects provider-specific image/document variants). We emit the
 * historical LangChain dict form (`{type, url, mime_type, ...}`) that all
 * upstream chat models still accept at runtime, and cast to `MessageContent`
 * at this single boundary.
 */
function convertPlatformContentToLc(
  content: string | readonly AnyContentBlock[],
): MessageContent {
  if (typeof content === "string") return content;

  const lcBlocks: Array<Record<string, unknown>> = [];
  for (const block of content) {
    if (block.type === "text") {
      lcBlocks.push({ type: "text", text: block.text });
    } else if (block.type === "image") {
      lcBlocks.push({
        type: "image",
        url: block.url,
        ...(block.mime_type !== undefined && { mime_type: block.mime_type }),
      });
    } else if (block.type === "document") {
      lcBlocks.push({
        type: "document",
        url: block.url,
        ...(block.mime_type !== undefined && { mime_type: block.mime_type }),
        ...(block.filename !== undefined && { filename: block.filename }),
      });
    }
  }
  if (lcBlocks.length === 0) return "";
  return lcBlocks as unknown as MessageContent;
}

/**
 * Convert a platform Message to a LangChain message.
 *
 * Returns `null` if the message cannot be converted (e.g. a tool message with
 * missing tool_call_id).
 */
export function platformToLcMessage(message: Message): BaseMessage | null {
  const role = message.role;
  const lcContent = convertPlatformContentToLc(message.content);

  // Messages may arrive as camelCase (Message interface) or snake_case (MessageSchema
  // wire output from Zod). Check both so the Tool Pod /invoke_llm path works correctly.
  const msgAny = message as unknown as Record<string, unknown>;
  const resolvedToolCallId =
    message.toolCallId ?? (msgAny["tool_call_id"] as string | undefined);
  const resolvedToolCalls =
    message.toolCalls ?? (msgAny["tool_calls"] as LLMToolCall[] | undefined);
  const resolvedIsError =
    message.isError ?? (msgAny["is_error"] as boolean | undefined) ?? false;
  const resolvedAdditionalKwargs =
    message.additionalKwargs ??
    (msgAny["additional_kwargs"] as Record<string, unknown> | undefined);
  const resolvedResponseMetadata =
    message.responseMetadata ??
    (msgAny["response_metadata"] as Record<string, unknown> | undefined);

  // LangChain's `additional_kwargs` / `response_metadata` are typed `Record<string, unknown>`;
  // ours are `Record<string, JsonValue>`. `JsonValue` is a strict subset of `unknown`, so the
  // bridge is safe and documented once via this helper rather than every call site.
  const additionalKwargs: Record<string, unknown> = {
    ...resolvedAdditionalKwargs,
  };
  const responseMetadata: Record<string, unknown> = {
    ...resolvedResponseMetadata,
  };
  const baseFields = {
    additional_kwargs: additionalKwargs,
    response_metadata: responseMetadata,
    ...(message.name !== undefined && { name: message.name }),
    ...(message.id !== undefined && { id: message.id }),
  };

  if (role === "tool") {
    if (resolvedToolCallId === undefined || resolvedToolCallId === "") {
      const preview =
        typeof message.content === "string"
          ? message.content.slice(0, 50)
          : "...";
      logger.warn(
        `Skipping tool message: missing tool_call_id (content=${JSON.stringify(preview)})`,
      );
      return null;
    }
    const strContent =
      typeof lcContent === "string" ? lcContent : JSON.stringify(lcContent);
    return new ToolMessage({
      ...baseFields,
      content: strContent,
      tool_call_id: resolvedToolCallId,
      status: resolvedIsError ? "error" : "success",
    });
  }

  if (role === "assistant") {
    // The wire JSON roundtrip strips `LLMToolCall` class identity, so tool
    // calls arrive here as plain objects after Zod parse. Handle both shapes.
    const mappedToolCalls = (resolvedToolCalls ?? []).map((tc) => {
      if (
        typeof (tc as unknown as { toLangchainDict?: unknown })
          .toLangchainDict === "function"
      ) {
        return (tc as LLMToolCall).toLangchainDict() as unknown as ToolCall;
      }
      const obj = tc as unknown as Record<string, unknown>;
      return {
        id: obj["id"] as string,
        name: obj["name"] as string,
        args: (obj["args"] ?? obj["arguments"] ?? {}) as Record<
          string,
          unknown
        >,
        type: (obj["type"] as string) ?? "tool_call",
      } as unknown as ToolCall;
    });
    return new AIMessage({
      ...baseFields,
      content: lcContent,
      tool_calls: mappedToolCalls,
    });
  }

  if (role === "system") {
    return new SystemMessage({ ...baseFields, content: lcContent });
  }

  return new HumanMessage({ ...baseFields, content: lcContent });
}

// ---------------------------------------------------------------------------
// Public API — bulk conversion
// ---------------------------------------------------------------------------

/** Convert a list of LangChain messages to platform Messages. */
export function lcMessagesToPlatform(
  messages: readonly BaseMessage[],
): Message[] {
  return messages.map(lcToPlatformMessage);
}

/**
 * Convert a list of platform Messages to LangChain messages.
 *
 * Invalid messages (e.g. tool messages with missing tool_call_id) are
 * skipped with a warning logged.
 */
export function platformMessagesToLc(
  messages: readonly Message[],
): BaseMessage[] {
  const result: BaseMessage[] = [];
  for (const msg of messages) {
    const lc = platformToLcMessage(msg);
    if (lc !== null) result.push(lc);
  }
  return result;
}

// ---------------------------------------------------------------------------
// OE message serialization (LangChain ↔ dict)
// ---------------------------------------------------------------------------

/**
 * Serialize a LangChain message to the platform wire format.
 *
 * Stamps every dict with `framework: "langchain"` so downstream consumers can
 * identify which framework produced the message.
 */
export function messageToDict(message: BaseMessage): Record<string, unknown> {
  // Emit a FLAT dict matching Python's `message_to_dict`
  // (`model_dump(exclude_none=True)`) — NOT LangChain JS's nested `toDict()`
  // envelope (`{type, data:{…}}`), which is also lossy (drops id /
  // response_metadata / usage_metadata). Building the flat shape explicitly
  // keeps this the exact inverse of `dictToAiMessage` (`new AIMessage(flat)`)
  // so the pair round-trips, mirroring Python's model_dump/model_validate.
  const m = message as unknown as Record<string, unknown>;
  const flat = stripUndefined({
    type: message.type,
    content: message.content as unknown,
    additional_kwargs: m["additional_kwargs"],
    response_metadata: m["response_metadata"],
    name: m["name"],
    id: m["id"],
    tool_calls: m["tool_calls"],
    invalid_tool_calls: m["invalid_tool_calls"],
    usage_metadata: m["usage_metadata"],
    tool_call_id: m["tool_call_id"],
    artifact: m["artifact"],
    status: m["status"],
  });
  return { ...flat, framework: FRAMEWORK_LANGCHAIN };
}

/** Throw if the wire-format dict came from an unsupported framework. */
function checkFramework(data: Record<string, unknown>): void {
  const fw = data["framework"];
  // Match Python's `fw is not None` — skip the check if framework is missing or null.
  if (fw != null && fw !== FRAMEWORK_LANGCHAIN) {
    throw new Error(
      `Cannot deserialize message from framework ${JSON.stringify(fw)} with the LangChain deserializer`,
    );
  }
}

/** Deserialize a platform wire-format dict to a LangChain AIMessage. */
export function dictToAiMessage(data: unknown): AIMessage {
  if (data instanceof AIMessage) return data;
  if (data !== null && typeof data === "object") {
    const obj = data as Record<string, unknown>;
    checkFramework(obj);
    return new AIMessage(obj as ConstructorParameters<typeof AIMessage>[0]);
  }
  return new AIMessage({ content: String(data) });
}

/**
 * Deserialize a platform wire-format dict to a LangChain AIMessageChunk.
 *
 * Used for streaming replay where ChatGenerationChunk expects a chunk type.
 */
export function dictToAiMessageChunk(data: unknown): AIMessageChunk {
  if (data instanceof AIMessageChunk) return data;
  if (data !== null && typeof data === "object") {
    const obj = data as Record<string, unknown>;
    checkFramework(obj);
    return new AIMessageChunk({
      ...obj,
      type: "AIMessageChunk",
    } as ConstructorParameters<typeof AIMessageChunk>[0]);
  }
  return new AIMessageChunk({ content: String(data) });
}

// ---------------------------------------------------------------------------
// LLM Response conversion (sdk-core ↔ LangChain)
// ---------------------------------------------------------------------------

const TOKEN_KEYS = new Set([
  "input_tokens",
  "output_tokens",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "model",
]);

function usageFromResponse(response: LLMResponse): LLMTokenUsage | undefined {
  if (response.usage !== undefined) return response.usage;

  const metadata = { ...response.metadata };
  const nestedUsage = metadata["usage"];
  if (
    nestedUsage !== null &&
    typeof nestedUsage === "object" &&
    !Array.isArray(nestedUsage)
  ) {
    return newLLMTokenUsageFromDict(nestedUsage as Record<string, unknown>);
  }

  const hasTokenKey = Object.keys(metadata).some((k) => TOKEN_KEYS.has(k));
  if (hasTokenKey) return newLLMTokenUsageFromDict(metadata);
  return undefined;
}

/** Convert sdk-core LLMResponse to LangChain ChatResult. */
export function llmResponseToChatResult(response: LLMResponse): ChatResult {
  const responseMetadata: Record<string, unknown> = { ...response.metadata };
  const rawMetadata = responseMetadata["_raw_response_metadata"];
  delete responseMetadata["_raw_response_metadata"];
  const rawResponseMetadata =
    rawMetadata !== null &&
    typeof rawMetadata === "object" &&
    !Array.isArray(rawMetadata)
      ? (rawMetadata as Record<string, unknown>)
      : {};
  delete responseMetadata["usage"];

  const explicitResponseMetadata = response.responseMetadata ?? {};
  const usage = usageFromResponse(response);
  const usageMetadata = usage !== undefined ? llmTokenUsageToDict(usage) : {};

  const mergedMetadata: Record<string, unknown> = {
    ...responseMetadata,
    ...usageMetadata,
    ...rawResponseMetadata,
    ...explicitResponseMetadata,
  };

  const aiMessage = new AIMessage({
    content: response.content,
    tool_calls: (response.toolCalls ?? []).map(
      (tc) => tc.toLangchainDict() as unknown as ToolCall,
    ),
    ...(response.id !== undefined ? { id: response.id } : {}),
    ...(response.name !== undefined ? { name: response.name } : {}),
    additional_kwargs: (response.additionalKwargs ?? {}) as Record<
      string,
      unknown
    >,
    response_metadata: mergedMetadata,
  });

  // ChatGeneration is an interface in LangChain 1.x — construct via object literal.
  const generation: ChatGeneration = {
    message: aiMessage,
    text: response.content,
  };
  return { generations: [generation] };
}

/** Convert sdk-core LLMStreamChunk to LangChain ChatGenerationChunk. */
export function llmStreamChunkToGenerationChunk(
  chunk: LLMStreamChunk,
): ChatGenerationChunk {
  const toolCallChunks: LCToolCallChunk[] = (chunk.toolCalls ?? []).map(
    (tc) => ({
      type: "tool_call_chunk",
      ...(tc.id !== undefined ? { id: tc.id } : {}),
      ...(tc.name !== undefined ? { name: tc.name } : {}),
      ...(tc.args !== undefined ? { args: tc.args } : {}),
      ...(tc.index !== undefined ? { index: tc.index } : {}),
    }),
  );

  const usageMetadata = chunk.usage?.toLangchainUsageMetadata();

  // LangChain 1.x's AIMessageChunkFields generic uses inferred response_metadata /
  // usage_metadata shapes that don't unify with our plain Record types. The cast
  // is the single bridging point — runtime shape matches AIMessageChunkFields.
  const aiChunk = new AIMessageChunk({
    content: chunk.content ?? "",
    tool_call_chunks: toolCallChunks,
    ...(chunk.id !== undefined ? { id: chunk.id } : {}),
    ...(chunk.name !== undefined ? { name: chunk.name } : {}),
    additional_kwargs: (chunk.additionalKwargs ?? {}) as Record<
      string,
      unknown
    >,
    response_metadata: (chunk.responseMetadata ?? {}) as Record<
      string,
      unknown
    >,
    ...(usageMetadata !== undefined ? { usage_metadata: usageMetadata } : {}),
  } as unknown as ConstructorParameters<typeof AIMessageChunk>[0]);

  return new ChatGenerationChunk({
    message: aiChunk,
    text: chunk.content ?? "",
  });
}
