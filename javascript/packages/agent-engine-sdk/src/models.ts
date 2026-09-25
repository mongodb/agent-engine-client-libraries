/** Framework-neutral types and models for the Atlas Agent Engine SDK. */

import { z } from "zod";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** Runtime schema for the recursive `JsonValue` type. */
export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema),
  ]),
);

export type Role = "user" | "assistant" | "tool" | "system";

/** Text content block. */
export const TextBlockSchema = z.object({
  type: z.literal("text").default("text"),
  text: z.string(),
});
export type TextBlock = z.infer<typeof TextBlockSchema>;

/** Image content block. */
export const ImageBlockSchema = z.object({
  type: z.literal("image").default("image"),
  url: z.string(),
  mime_type: z.string().optional(),
});
export type ImageBlock = z.infer<typeof ImageBlockSchema>;

/** Document/file content block. */
export const DocumentBlockSchema = z.object({
  type: z.literal("document").default("document"),
  url: z.string(),
  mime_type: z.string().optional(),
  filename: z.string().optional(),
});
export type DocumentBlock = z.infer<typeof DocumentBlockSchema>;

/**
 * Discriminated on `type` so a document block is never misparsed as an image
 * (a plain union would match `{ url }` against `ImageBlockSchema` first and
 * silently drop `filename`). Inputs must carry an explicit `type`.
 */
export const AnyContentBlockSchema = z.discriminatedUnion("type", [
  TextBlockSchema,
  ImageBlockSchema,
  DocumentBlockSchema,
]);
export type AnyContentBlock = z.infer<typeof AnyContentBlockSchema>;

/** Framework-neutral tool definition. */
export const ToolDefinitionSchema = z.object({
  name: z.string(),
  description: z.string(),
  args_schema: z.record(z.string(), z.unknown()),
  callable: z.custom<(...args: unknown[]) => unknown>(
    (v) => typeof v === "function",
  ),
  remote: z.boolean().default(true),
  provider_type: z.string().optional(),
  scopes: z.array(z.string()).default([]),
  network: z.array(z.string()).default([]),
  timeout_seconds: z.number().int().default(30),
  redact_fields: z.array(z.string()).default([]),
});
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;
export type ToolDefinitionInput = z.input<typeof ToolDefinitionSchema>;

/** Constructs a ToolDefinition, applying Python-parity defaults (remote=true, network=[], timeout_seconds=30, redact_fields=[]). */
export function createToolDefinition(
  input: ToolDefinitionInput,
): ToolDefinition {
  return ToolDefinitionSchema.parse(input);
}

/** Typed token-usage metadata for LLM calls. */
export class LLMTokenUsage {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly promptTokens?: number;
  readonly completionTokens?: number;
  totalTokens?: number;
  readonly model?: string;

  constructor(data: {
    input_tokens?: number;
    output_tokens?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    model?: string;
  }) {
    this.inputTokens = data.input_tokens;
    this.outputTokens = data.output_tokens;
    this.promptTokens = data.prompt_tokens;
    this.completionTokens = data.completion_tokens;
    this.totalTokens = data.total_tokens;
    this.model = data.model;

    if (this.totalTokens === undefined) {
      const prompt = this.inputTokens ?? this.promptTokens;
      const completion = this.outputTokens ?? this.completionTokens;
      if (prompt !== undefined && completion !== undefined) {
        this.totalTokens = prompt + completion;
      }
    }
  }

  /** Serializes to snake_case wire format so JSON.stringify round-trips through LLMTokenUsageSchema. */
  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    if (this.inputTokens !== undefined)
      result["input_tokens"] = this.inputTokens;
    if (this.outputTokens !== undefined)
      result["output_tokens"] = this.outputTokens;
    if (this.promptTokens !== undefined)
      result["prompt_tokens"] = this.promptTokens;
    if (this.completionTokens !== undefined)
      result["completion_tokens"] = this.completionTokens;
    if (this.totalTokens !== undefined)
      result["total_tokens"] = this.totalTokens;
    if (this.model !== undefined) result["model"] = this.model;
    return result;
  }

  /** Returns LangChain's expected usage-metadata keys. */
  toLangchainUsageMetadata(): {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } {
    const inputTokens = this.inputTokens ?? this.promptTokens ?? 0;
    const outputTokens = this.outputTokens ?? this.completionTokens ?? 0;
    const totalTokens = this.totalTokens ?? inputTokens + outputTokens;
    return {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: totalTokens,
    };
  }
}

/** Runtime validation schema for LLMTokenUsage wire data. All fields optional, matching the class constructor. */
export const LLMTokenUsageSchema = z.preprocess(
  (val) => (val instanceof LLMTokenUsage ? val.toJSON() : val),
  z
    .object({
      input_tokens: z.number().optional(),
      output_tokens: z.number().optional(),
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
      model: z.string().optional(),
    })
    .transform((data) => new LLMTokenUsage(data)),
);

/** Typed final tool call requested by an LLM. */
export class LLMToolCall {
  readonly id?: string;
  readonly name?: string;
  readonly args?: JsonValue;
  readonly type?: string;
  readonly index?: number;

  constructor(data: {
    id?: string;
    name?: string;
    args?: JsonValue;
    arguments?: JsonValue;
    type?: string;
    index?: number;
  }) {
    this.id = data.id;
    this.name = data.name;
    this.args = data.args ?? data.arguments; // accepts both 'args' and 'arguments' for cross-provider compatibility
    this.type = data.type;
    this.index = data.index;
  }

  /** Returns a LangChain-compatible tool-call dict (excludes index). */
  toLangchainDict(): Record<string, JsonValue> {
    const result: Record<string, JsonValue> = {};
    if (this.id !== undefined) result["id"] = this.id;
    if (this.name !== undefined) result["name"] = this.name;
    if (this.args !== undefined) result["args"] = this.args;
    if (this.type !== undefined) result["type"] = this.type;
    return result;
  }
}

/** Runtime validation schema for LLMToolCall wire data. */
export const LLMToolCallSchema = z
  .object({
    id: z.string().optional(),
    name: z.string().optional(),
    args: JsonValueSchema.optional(),
    arguments: JsonValueSchema.optional(),
    type: z.string().optional(),
    index: z.number().optional(),
  })
  .transform((data) => new LLMToolCall(data));

/** Serializable bound-tool schema forwarded with an LLM call. */
export class LLMToolSchema {
  readonly name?: string;
  readonly description?: string;
  readonly parameters?: JsonValue;
  readonly type?: string;
  readonly strict?: boolean;
  readonly function?: JsonValue;

  constructor(data: {
    name?: string;
    description?: string;
    parameters?: JsonValue;
    type?: string;
    strict?: boolean;
    function?: JsonValue;
  }) {
    this.name = data.name;
    this.description = data.description;
    this.parameters = data.parameters;
    this.type = data.type;
    this.strict = data.strict;
    this.function = data.function;
  }

  /** Returns the schema in the shape LangChain bind_tools expects. */
  toLangchainDict(): Record<string, JsonValue> {
    const result: Record<string, JsonValue> = {};
    if (this.name !== undefined) result["name"] = this.name;
    if (this.description !== undefined)
      result["description"] = this.description;
    if (this.parameters !== undefined) result["parameters"] = this.parameters;
    if (this.type !== undefined) result["type"] = this.type;
    if (this.strict !== undefined) result["strict"] = this.strict;
    if (this.function !== undefined) result["function"] = this.function;
    return result;
  }
}

/** Runtime validation schema for LLMToolSchema wire data. */
export const LLMToolSchemaValidator = z
  .object({
    name: z.string().optional(),
    description: z.string().optional(),
    parameters: JsonValueSchema.optional(),
    type: z.string().optional(),
    strict: z.boolean().optional(),
    function: JsonValueSchema.optional(),
  })
  .transform((data) => new LLMToolSchema(data));

const INVOCATION_KNOWN_KEYS = new Set([
  "max_tokens",
  "response_format",
  "top_p",
  "top_k",
  "frequency_penalty",
  "presence_penalty",
  "seed",
  "timeout",
  "parallel_tool_calls",
  "reasoning_effort",
]);

/** Explicit provider/model options passed with an LLM invocation. */
export class LLMInvocationOptions {
  readonly maxTokens?: number;
  readonly responseFormat?: JsonValue;
  readonly topP?: number;
  readonly topK?: number;
  readonly frequencyPenalty?: number;
  readonly presencePenalty?: number;
  readonly seed?: number;
  readonly timeout?: number;
  readonly parallelToolCalls?: boolean;
  readonly reasoningEffort?: string;
  /** Provider-specific kwargs not in the standard field set (e.g. temperature). Mirrors Python's extra="allow". */
  readonly extras: Record<string, unknown>;

  constructor(data: {
    max_tokens?: number;
    response_format?: JsonValue;
    top_p?: number;
    top_k?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    seed?: number;
    timeout?: number;
    parallel_tool_calls?: boolean;
    reasoning_effort?: string;
    [key: string]: unknown;
  }) {
    this.maxTokens = data.max_tokens;
    this.responseFormat = data.response_format;
    this.topP = data.top_p;
    this.topK = data.top_k;
    this.frequencyPenalty = data.frequency_penalty;
    this.presencePenalty = data.presence_penalty;
    this.seed = data.seed;
    this.timeout = data.timeout;
    this.parallelToolCalls = data.parallel_tool_calls;
    this.reasoningEffort = data.reasoning_effort;
    this.extras = Object.fromEntries(
      Object.entries(data).filter(
        ([k, v]) => !INVOCATION_KNOWN_KEYS.has(k) && v !== undefined,
      ),
    );
  }

  /** Serializes to snake_case wire format so JSON.stringify round-trips through LLMInvocationOptionsSchema. */
  toJSON(): Record<string, unknown> {
    return this.toModelKwargs();
  }

  /** Converts to kwargs for underlying model invocation. Mirrors Python's model_dump(exclude_none=True) with extra="allow". */
  toModelKwargs(): Record<string, unknown> {
    const result: Record<string, unknown> = { ...this.extras };
    if (this.maxTokens !== undefined) result["max_tokens"] = this.maxTokens;
    if (this.responseFormat !== undefined)
      result["response_format"] = this.responseFormat;
    if (this.topP !== undefined) result["top_p"] = this.topP;
    if (this.topK !== undefined) result["top_k"] = this.topK;
    if (this.frequencyPenalty !== undefined)
      result["frequency_penalty"] = this.frequencyPenalty;
    if (this.presencePenalty !== undefined)
      result["presence_penalty"] = this.presencePenalty;
    if (this.seed !== undefined) result["seed"] = this.seed;
    if (this.timeout !== undefined) result["timeout"] = this.timeout;
    if (this.parallelToolCalls !== undefined)
      result["parallel_tool_calls"] = this.parallelToolCalls;
    if (this.reasoningEffort !== undefined)
      result["reasoning_effort"] = this.reasoningEffort;
    return result;
  }
}

/** Runtime validation schema for LLMInvocationOptions wire data. */
export const LLMInvocationOptionsSchema = z.preprocess(
  (val) => (val instanceof LLMInvocationOptions ? val.toJSON() : val),
  z
    .looseObject({
      max_tokens: z.number().optional(),
      response_format: JsonValueSchema.optional(),
      top_p: z.number().optional(),
      top_k: z.number().optional(),
      frequency_penalty: z.number().optional(),
      presence_penalty: z.number().optional(),
      seed: z.number().optional(),
      timeout: z.number().optional(),
      parallel_tool_calls: z.boolean().optional(),
      reasoning_effort: z.string().optional(),
    })
    .transform((data) => new LLMInvocationOptions(data)),
);

/**
 * Framework-neutral message.
 *
 * Content can be a simple string for text-only messages, or a list of
 * content blocks for multimodal content (images, documents, etc.).
 */
export interface Message {
  role: Role;
  content: string | AnyContentBlock[];
  toolCalls?: LLMToolCall[];
  toolCallId?: string;
  isError?: boolean;
  name?: string;
  id?: string;
  additionalKwargs?: Record<string, JsonValue>;
  responseMetadata?: Record<string, JsonValue>;
}

/**
 * Runtime validation schema for Message wire data. Parses snake_case wire
 * fields and transforms them to the camelCase Message interface shape,
 * matching how LLMToolCallSchema/LLMTokenUsageSchema handle the same mapping.
 */
export const MessageSchema = z
  .looseObject({
    role: z.enum(["user", "assistant", "tool", "system"]),
    content: z.union([z.string(), z.array(AnyContentBlockSchema)]),
    tool_calls: z.array(LLMToolCallSchema).optional(),
    tool_call_id: z.string().optional(),
    is_error: z.boolean().optional(),
    name: z.string().optional(),
    id: z.string().optional(),
    additional_kwargs: z.record(z.string(), JsonValueSchema).optional(),
    response_metadata: z.record(z.string(), JsonValueSchema).optional(),
  })
  .transform(
    (data): Message => ({
      role: data.role,
      content: data.content,
      toolCalls: data.tool_calls,
      toolCallId: data.tool_call_id,
      isError: data.is_error,
      name: data.name,
      id: data.id,
      additionalKwargs: data.additional_kwargs,
      responseMetadata: data.response_metadata,
    }),
  );

/**
 * Serialize a Message to its snake_case wire shape — the camel→snake "dump"
 * half that mirrors Python's `model_dump(by_alias=True)`. `MessageSchema` is
 * the inverse snake→camel "parse" half.
 *
 * Keeping these two symmetric is what makes the representation safe: messages
 * are camelCase internally (the `Message` interface) and snake_case on the
 * wire, and conversion happens exactly once at each boundary. Use this when
 * dumping an already-internal `Message`; never re-run an internal Message
 * through `MessageSchema`, whose transform would blank the camelCase fields.
 */
export function serializeMessage(message: Message): Record<string, unknown> {
  const out: Record<string, unknown> = {
    role: message.role,
    content: message.content,
  };
  if (message.toolCalls !== undefined) out["tool_calls"] = message.toolCalls;
  if (message.toolCallId !== undefined)
    out["tool_call_id"] = message.toolCallId;
  if (message.isError !== undefined) out["is_error"] = message.isError;
  if (message.name !== undefined) out["name"] = message.name;
  if (message.id !== undefined) out["id"] = message.id;
  if (message.additionalKwargs !== undefined)
    out["additional_kwargs"] = message.additionalKwargs;
  if (message.responseMetadata !== undefined)
    out["response_metadata"] = message.responseMetadata;
  return out;
}

const TOKEN_KEYS = new Set([
  "input_tokens",
  "output_tokens",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "model",
]);

/** Framework-neutral LLM response. */
export class LLMResponse {
  readonly content: string;
  readonly toolCalls?: LLMToolCall[];
  readonly metadata: Record<string, JsonValue>;
  readonly usage?: LLMTokenUsage;
  readonly id?: string;
  readonly name?: string;
  readonly additionalKwargs?: Record<string, JsonValue>;
  readonly responseMetadata?: Record<string, JsonValue>;

  constructor(data: {
    content: string;
    toolCalls?: LLMToolCall[];
    metadata?: Record<string, JsonValue>;
    usage?: LLMTokenUsage;
    id?: string;
    name?: string;
    additionalKwargs?: Record<string, JsonValue>;
    responseMetadata?: Record<string, JsonValue>;
  }) {
    this.content = data.content;
    this.toolCalls = data.toolCalls;
    this.metadata = data.metadata ?? {};
    this.usage = data.usage;
    this.id = data.id;
    this.name = data.name;
    this.additionalKwargs = data.additionalKwargs;
    this.responseMetadata = data.responseMetadata;
  }

  /**
   * Constructs an LLMResponse from a raw provider dict, normalizing usage
   * from either a top-level 'usage' key or token keys embedded in 'metadata'.
   */
  static fromRaw(data: Record<string, unknown>): LLMResponse {
    let usage: LLMTokenUsage | undefined;

    if (data["usage"] != null) {
      usage = new LLMTokenUsage(
        data["usage"] as ConstructorParameters<typeof LLMTokenUsage>[0],
      );
    } else {
      const metadata = data["metadata"];
      if (
        metadata != null &&
        typeof metadata === "object" &&
        !Array.isArray(metadata)
      ) {
        const meta = metadata as Record<string, unknown>;
        const metaUsage = meta["usage"];
        let usageData: Record<string, unknown> | undefined;

        if (
          metaUsage != null &&
          typeof metaUsage === "object" &&
          !Array.isArray(metaUsage)
        ) {
          usageData = metaUsage as Record<string, unknown>;
        } else if (Object.keys(meta).some((k) => TOKEN_KEYS.has(k))) {
          usageData = meta;
        }

        if (usageData != null) {
          usage = new LLMTokenUsage(
            usageData as ConstructorParameters<typeof LLMTokenUsage>[0],
          );
        }
      }
    }

    const rawToolCalls = data["tool_calls"] ?? data["toolCalls"];
    return new LLMResponse({
      content: (data["content"] as string) ?? "",
      toolCalls: Array.isArray(rawToolCalls)
        ? (rawToolCalls as unknown[]).map((tc) => LLMToolCallSchema.parse(tc))
        : undefined,
      metadata: (data["metadata"] as Record<string, JsonValue>) ?? {},
      usage,
      id: data["id"] as string | undefined,
      name: data["name"] as string | undefined,
      additionalKwargs: (data["additional_kwargs"] ??
        data["additionalKwargs"]) as Record<string, JsonValue> | undefined,
      responseMetadata: (data["response_metadata"] ??
        data["responseMetadata"]) as Record<string, JsonValue> | undefined,
    });
  }
}

/** Runtime validation schema for LLMResponse wire data. `content` is the only required field. */
export const LLMResponseSchema = z.looseObject({
  content: z.string(),
  tool_calls: z.array(z.unknown()).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
  usage: z.record(z.string(), z.unknown()).optional(),
  id: z.string().optional(),
  name: z.string().optional(),
  additional_kwargs: z.record(z.string(), z.unknown()).optional(),
  response_metadata: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Partial tool call during streaming.
 *
 * All fields are optional since chunks may contain partial data
 * (e.g., just the start of arguments, or just the tool name).
 */
export interface ToolCallChunk {
  id?: string;
  name?: string;
  args?: string;
  type?: string;
  index?: number;
}

/** Runtime validation schema for ToolCallChunk wire data. */
export const ToolCallChunkSchema = z.object({
  id: z.string().optional(),
  name: z.string().optional(),
  args: z.string().optional(),
  type: z.string().optional(),
  index: z.number().optional(),
});

/**
 * Rich streaming chunk following OpenAI/Anthropic patterns.
 *
 * All fields are optional since different chunks contain different data:
 * - Content chunks: text deltas during generation
 * - Tool call chunks: partial tool calls being built up
 * - Usage chunks: token counts at end of stream
 */
export interface LLMStreamChunk {
  content?: string;
  toolCalls?: ToolCallChunk[];
  usage?: LLMTokenUsage;
  id?: string;
  name?: string;
  additionalKwargs?: Record<string, JsonValue>;
  responseMetadata?: Record<string, JsonValue>;
}

/** Input for an agent execution. Payload is an opaque JSON value. */
export interface AgentInput {
  payload: JsonValue;
}

/** Output from an agent execution. Response is an opaque JSON value. */
export interface AgentOutput {
  response: JsonValue;
}

/** Event streamed back to the caller during execution. */
export interface StreamEvent {
  data: JsonValue;
  event?: string;
}

/**
 * Refines `z.unknown()` to reject the absent/`undefined` case, matching Pydantic's
 * "required field" semantics for JsonValue-typed fields (where `null` is allowed
 * but a missing key is not).
 */
const RequiredJsonValueSchema = z
  .unknown()
  .refine((v) => v !== undefined, { message: "Required" });

/** Runtime validation schema for AgentInput wire data. */
export const AgentInputSchema = z.looseObject({
  payload: RequiredJsonValueSchema,
});

/** Runtime validation schema for AgentOutput wire data. */
export const AgentOutputSchema = z.looseObject({
  response: RequiredJsonValueSchema,
});

// An SSE `event:` line is terminated by CR/LF, so a name carrying those bytes
// could forge extra frames in any server-sent-event serializer. NUL is rejected
// too as it can truncate the name in C-based parsers/log sinks. Forbid them at
// the schema boundary regardless of which transport later renders the name.
const StreamEventNameSchema = z
  .string()
  // Checked via includes() rather than a regex so the NUL literal doesn't
  // trip eslint's no-control-regex rule.
  .refine(
    (v) => !v.includes("\r") && !v.includes("\n") && !v.includes("\u0000"),
    {
      message: "Stream event name must not contain CR, LF, or NUL",
    },
  );

/** Runtime validation schema for StreamEvent wire data. */
export const StreamEventSchema = z.looseObject({
  data: RequiredJsonValueSchema,
  event: StreamEventNameSchema.optional(),
});

/**
 * Request-scoped context passed to every agent execution.
 *
 * Carries framework-neutral execution identity and platform plumbing. The
 * caller's invocation input lives in `AgentInput.payload`, not here.
 * Suspend/resume state is opaque to the platform: `metadata` is a
 * framework-owned blob that the platform forwards unchanged on resume.
 */
export interface RequestContext {
  executionId?: string;
  sessionId?: string;
  userId?: string;
  /**
   * Tenant scope of the executing agent. Framework adapters use it to scope
   * state stores keyed only by sessionId (e.g. the LangGraph checkpoint
   * thread_id) so agents in the same project DB cannot collide. Mirrors
   * Python's `RequestContext.workspace_id`.
   */
  workspaceId?: string;
  requestHeaders?: Record<string, string>;
  /** Platform suspend/resume plumbing (framework-neutral). */
  resume?: boolean;
  resumeData?: JsonValue | null;
  /**
   * True when this session's most recent prior execution was cancelled (its
   * pod torn down mid-run). Framework adapters that persist per-step state
   * use it to fence off the killed step's partial writes before running this
   * turn. Mirrors Python's `previous_execution_cancelled`.
   */
  previousExecutionCancelled?: boolean;
  /**
   * Opaque framework-owned state passed through by the SDK and AER. Only the
   * selected framework adapter may interpret its keys and values.
   */
  metadata?: Record<string, JsonValue> | null;
  /**
   * Cancellation signal for execution timeout or forced abort. The runtime
   * (and any agent implementation that honours it) should unwind pending
   * awaits and stop emitting events when this signal fires. Python relies on
   * `asyncio.wait_for` for the same semantics; JS has no implicit
   * cancellation, so the signal must be plumbed explicitly.
   */
  signal?: AbortSignal;
}

/**
 * Per-session summary sourced from framework-specific persistence.
 *
 * Returned by the AER's session-enrichment endpoint. The platform layer
 * (OE) merges this with its own tenant fields (user_id, project_id,
 * workspace_id, visibility) when assembling the public sessions list.
 *
 * Mirrors Python's `SessionSummary` in `agent_engine_sdk/models.py`.
 */
export const SessionSummarySchema = z.object({
  /** Identifier of the session. */
  session_id: z.string(),
  /** ISO 8601 timestamp of the most recent activity; empty when unavailable. */
  last_activity: z.string(),
  /** ISO 8601 timestamp when the session was first persisted; empty when unavailable. */
  created_at: z.string(),
  /**
   * Number of messages in the session — the same count the session messages
   * endpoint returns.
   */
  message_count: z.number().int(),
  /**
   * Truncated first human-authored message (up to 80 chars). Empty when no
   * human message has been persisted.
   */
  first_message_preview: z.string().default(""),
});
export type SessionSummary = z.infer<typeof SessionSummarySchema>;

/** Response for the AER's session-summary enrichment endpoint. */
export const SessionsSummaryResponseSchema = z.object({
  /**
   * One summary per requested session_id that has framework-side
   * persistence. Sessions without persistence are omitted.
   */
  sessions: z.array(SessionSummarySchema),
});
export type SessionsSummaryResponse = z.infer<
  typeof SessionsSummaryResponseSchema
>;

/**
 * A single message in a session's conversation history.
 *
 * `role` uses the same vocabulary as `Message` ("user", "assistant",
 * "tool", "system"). `name` carries the tool name on tool messages.
 *
 * Mirrors Python's `SessionMessage` in `agent_engine_sdk/models.py`.
 */
export const SessionMessageSchema = z.object({
  /**
   * Stable identifier — the framework's id when available, otherwise a
   * synthetic id derived from session_id + position.
   */
  id: z.string(),
  /**
   * Platform role: "user", "assistant", "tool", or "system".
   * Unknown framework types pass through unchanged.
   */
  role: z.string(),
  /** Flat-string content. Multimodal content is collapsed to its text portions only. */
  content: z.string(),
  /**
   * ISO 8601 persistence-boundary timestamp. Native checkpoint projections use
   * the first checkpoint where the message appeared; durable workflow projections
   * use the committed snapshot boundary; empty when unavailable.
   */
  timestamp: z.string(),
  /** Identifier of the session this message belongs to. */
  session_id: z.string(),
  /** Tool name on tool messages, empty otherwise. */
  name: z.string().default(""),
  /**
   * Tool calls requested by an assistant message. Each carries a stable `id`
   * that the matching tool result echoes back via `tool_call_id`.
   */
  tool_calls: z.array(LLMToolCallSchema).nullable().default(null),
  /**
   * On a tool message, the id of the assistant tool call this message is the
   * result of. Empty/absent on non-tool messages.
   */
  tool_call_id: z.string().nullable().default(null),
  /**
   * Framework message metadata that should remain attached to the message,
   * including generic UI artifact metadata.
   */
  additional_kwargs: z.record(z.string(), z.unknown()).nullable().default(null),
});
export type SessionMessage = z.infer<typeof SessionMessageSchema>;

/** Response for the AER's session-messages endpoint. */
export const SessionMessagesResponseSchema = z.object({
  /**
   * Decoded conversation history in chronological order. Empty when the
   * session has no persisted messages.
   */
  messages: z.array(SessionMessageSchema),
});
export type SessionMessagesResponse = z.infer<
  typeof SessionMessagesResponseSchema
>;

/** Parses an ISO datetime string into a Date. Rejects unparseable input. */
export const DateFromStringSchema = z.string().transform((s, ctx) => {
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) {
    ctx.addIssue({ code: "custom", message: `Invalid datetime string: ${s}` });
    return z.NEVER;
  }
  return d;
});

/** Reference to a branch and its fork point. */
export const BranchRefSchema = z.looseObject({
  name: z.string(),
  root_event_id: z.string(),
});
export type BranchRef = z.infer<typeof BranchRefSchema>;

/** Platform event — the storage envelope. Payloads are opaque. */
export const EventSchema = z.looseObject({
  event_id: z.string(),
  session_id: z.string(),
  parent_event_id: z.string().optional(),
  actor_id: z.string(),
  payload: z.unknown().transform((v) => v as JsonValue),
  metadata: z.record(z.string(), z.string()).optional(),
  branch: BranchRefSchema.optional(),
  timestamp: DateFromStringSchema,
});
export type Event = z.infer<typeof EventSchema>;
