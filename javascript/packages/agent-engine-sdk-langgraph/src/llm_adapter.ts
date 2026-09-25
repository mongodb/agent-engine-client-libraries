/**
 * Lightweight adapter wrapping a LangChain `BaseChatModel` as `BaseLLM`.
 *
 * Used by the Tool Pod (`server/tool.ts`) to invoke LLMs through the
 * framework-neutral `BaseLLM` protocol while the underlying provider is
 * still a LangChain ChatModel.
 *
 * Eventually runtime `createLlm` will return `BaseLLM` directly,
 * and this adapter will only be used internally by agent-engine-sdk-langgraph.
 *
 * Port of `agent_engine_sdk_langgraph/llm_adapter.py`.
 *
 * **Sync vs async note:** Python's `BaseLLM` exposes both `invoke` (sync) and
 * `ainvoke` (async). JavaScript has no synchronous network I/O, so the TS
 * port collapses both into a single async `invoke`.
 */

import type { BaseMessage, BaseMessageChunk } from "@langchain/core/messages";

import {
  type BaseLLM,
  type JsonValue,
  type LLMResponse,
  type LLMStreamChunk,
  type LLMTokenUsage,
  type LLMToolSchema,
  type Message,
  type ToolCallChunk,
} from "@mongodb-js/agent-engine-sdk";
import {
  LLMResult,
  accumulateStreamUsage,
  jsonSafeMetadata,
  normalizeContent,
  normalizeToolCallArgs,
} from "@mongodb-js/agent-engine-runner-shared";
import { platformMessagesToLc } from "./messages.js";

// ---------------------------------------------------------------------------
// Minimal contract for the underlying LangChain chat model
// ---------------------------------------------------------------------------

/**
 * Structural contract for the LangChain chat model wrapped by the adapter.
 *
 * Avoids tying the public adapter to a specific LangChain class — any object
 * with `invoke`/`stream`/`bindTools` (or a subset of those) will satisfy it,
 * which keeps tests trivial to write.
 */
export interface LangChainChatModelLike {
  invoke(
    input: readonly BaseMessage[],
    config?: Record<string, unknown>,
  ): Promise<BaseMessage>;
  stream?(
    input: readonly BaseMessage[],
    config?: Record<string, unknown>,
  ): AsyncIterable<BaseMessageChunk> | Promise<AsyncIterable<BaseMessageChunk>>;
  bindTools?(
    tools: ReadonlyArray<Record<string, JsonValue>>,
    kwargs?: { tool_choice?: unknown },
  ): LangChainChatModelLike;
}

// ---------------------------------------------------------------------------
// LangChainLLMAdapter
// ---------------------------------------------------------------------------

export class LangChainLLMAdapter implements BaseLLM {
  private readonly llm: LangChainChatModelLike;

  constructor(
    llm: LangChainChatModelLike,
    tools?: readonly LLMToolSchema[],
    toolChoice?: unknown,
  ) {
    const serializedTools =
      tools && tools.length > 0
        ? tools.map((tool) => tool.toLangchainDict())
        : undefined;
    if (serializedTools) {
      if (typeof llm.bindTools !== "function") {
        // Match Python's behavior: raise loudly when tools are supplied but
        // the underlying model has no `bind_tools` / `bindTools` method.
        throw new Error(
          "LLM does not support tool binding (missing `bindTools` method).",
        );
      }
      // tool_choice (e.g. a function name from withStructuredOutput) is the
      // force point: LangChain translates it against the bound tools to
      // require the call. A choice without tools cannot bind to anything.
      this.llm = llm.bindTools(
        serializedTools,
        toolChoice != null ? { tool_choice: toolChoice } : undefined,
      );
    } else {
      this.llm = llm;
    }
  }

  /** LLM invocation. */
  async invoke(
    messages: readonly Message[],
    kwargs: Readonly<Record<string, unknown>> = {},
  ): Promise<LLMResponse> {
    const lcMessages = platformMessagesToLc(messages);
    const response = await this.llm.invoke(lcMessages, kwargs);
    return responseToLlmResponse(response);
  }

  /** Streaming LLM invocation. Matches `BaseLLM.stream` from agent-engine-sdk. */
  async *stream(
    messages: readonly Message[],
    kwargs: Readonly<Record<string, unknown>> = {},
  ): AsyncIterable<LLMStreamChunk> {
    const lcMessages = platformMessagesToLc(messages);

    if (typeof this.llm.stream !== "function") {
      throw new Error(
        "Underlying LLM does not support streaming (no `stream` method).",
      );
    }

    const streamResult = this.llm.stream(lcMessages, kwargs);
    // LangChain.js streams are AsyncIterable directly; some adapters return a Promise<AsyncIterable>.
    const stream = isPromise(streamResult) ? await streamResult : streamResult;
    let running: LLMTokenUsage | undefined;
    for await (const chunk of stream) {
      const streamChunk = chunkToLlmStreamChunk(chunk);
      running = accumulateStreamUsage(running, streamChunk.usage);
      yield running !== undefined
        ? { ...streamChunk, usage: running }
        : streamChunk;
    }
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/** Convert a LangChain response to sdk-core `LLMResponse`. */
function responseToLlmResponse(response: BaseMessage): LLMResponse {
  return LLMResult.fromResponse(response).toResponse();
}

function optionalStrAttr(
  message: BaseMessage,
  attr: string,
): string | undefined {
  const value = (message as unknown as Record<string, unknown>)[attr];
  return typeof value === "string" ? value : undefined;
}

function optionalJsonDictAttr(
  message: BaseMessage,
  attr: string,
): Readonly<Record<string, JsonValue>> | undefined {
  const value = (message as unknown as Record<string, unknown>)[attr];
  return jsonSafeMetadata(value);
}

/** Convert a LangChain stream chunk to sdk-core `LLMStreamChunk`. */
function chunkToLlmStreamChunk(chunk: BaseMessageChunk): LLMStreamChunk {
  const rawContent = chunk.content;
  const content = normalizeContent(rawContent);

  let toolCalls: ToolCallChunk[] | undefined;

  const rawChunks = (chunk as unknown as Record<string, unknown>)[
    "tool_call_chunks"
  ];
  if (Array.isArray(rawChunks) && rawChunks.length > 0) {
    toolCalls = [];
    for (const tc of rawChunks) {
      const tcDict = coerceToDict(tc);
      if (tcDict === undefined) continue;
      toolCalls.push(buildToolCallChunk(tcDict, /* includeIndex */ true));
    }
  }

  // Fall back to merged tool_calls when no tool_call_chunks are present.
  if (!toolCalls || toolCalls.length === 0) {
    const rawCalls = (chunk as unknown as Record<string, unknown>)[
      "tool_calls"
    ];
    if (Array.isArray(rawCalls) && rawCalls.length > 0) {
      toolCalls = [];
      for (const tc of rawCalls) {
        if (isPlainObject(tc)) {
          toolCalls.push(
            buildToolCallChunk(
              tc as Record<string, unknown>,
              /* includeIndex */ false,
            ),
          );
        }
      }
    }
  }

  const usage = LLMResult.extractUsage(chunk);
  const id = optionalStrAttr(chunk, "id");
  const name = optionalStrAttr(chunk, "name");
  const responseMetadata = optionalJsonDictAttr(chunk, "response_metadata");
  const additionalKwargs = optionalJsonDictAttr(chunk, "additional_kwargs");

  // Emit camelCase keys to match the sdk-core `LLMStreamChunk` contract.
  // (Python's LLMStreamChunk uses snake_case fields, but the TS model is
  // camelCase — the port must translate, not copy Python's key names.)
  return {
    ...(content !== "" ? { content } : {}),
    ...(toolCalls && toolCalls.length > 0 ? { toolCalls } : {}),
    ...(usage !== undefined ? { usage } : {}),
    ...(id !== undefined ? { id } : {}),
    ...(name !== undefined ? { name } : {}),
    ...(responseMetadata !== undefined ? { responseMetadata } : {}),
    ...(additionalKwargs !== undefined ? { additionalKwargs } : {}),
  };
}

function buildToolCallChunk(
  data: Record<string, unknown>,
  includeIndex: boolean,
): ToolCallChunk {
  const id = data["id"];
  const name = data["name"];
  const type = data["type"];
  const index = data["index"];
  const argsRaw = data["args"];
  const args =
    argsRaw !== undefined && argsRaw !== null
      ? normalizeToolCallArgs(argsRaw)
      : null;

  return {
    ...(typeof id === "string" ? { id } : {}),
    ...(typeof name === "string" ? { name } : {}),
    ...(args !== null ? { args } : {}),
    ...(typeof type === "string" ? { type } : {}),
    ...(includeIndex && typeof index === "number" ? { index } : {}),
  };
}

function coerceToDict(tc: unknown): Record<string, unknown> | undefined {
  if (isPlainObject(tc)) return tc as Record<string, unknown>;
  // LangChain.js sometimes ships Pydantic-style class instances; honour them
  // when they expose a `toDict()` / `toJSON()` accessor like the Python original.
  if (tc !== null && typeof tc === "object") {
    const obj = tc as { toDict?: () => unknown; toJSON?: () => unknown };
    if (typeof obj.toDict === "function") {
      const result = obj.toDict();
      return isPlainObject(result)
        ? (result as Record<string, unknown>)
        : undefined;
    }
    if (typeof obj.toJSON === "function") {
      const result = obj.toJSON();
      return isPlainObject(result)
        ? (result as Record<string, unknown>)
        : undefined;
    }
  }
  return undefined;
}

function isPlainObject(value: unknown): boolean {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isPromise<T>(value: T | Promise<T>): value is Promise<T> {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof (value as Promise<T>).then === "function"
  );
}
