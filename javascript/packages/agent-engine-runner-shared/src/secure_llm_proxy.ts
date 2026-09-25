/**
 * SecureLLMProxy — framework-neutral proxy for secure LLM calls through OE.
 *
 * Packages intercepted invoke_llm requests for the Orchestration Engine and
 * unwraps the streamed OE relay back into agent-engine-sdk models. OE owns approval,
 * routing, live SSE relay, and final audit/result recording.
 */

import type {
  LLMStreamChunk,
  LLMTokenUsage,
  LLMToolSchema,
  LLMInvocationOptions,
} from "@mongodb-js/agent-engine-sdk";
import {
  LLMResponse,
  LLMToolCall,
  LLMToolSchemaValidator,
  JsonValueSchema,
  type Message,
  type ToolCallChunk,
  type JsonValue,
} from "@mongodb-js/agent-engine-sdk";
import { createParser } from "eventsource-parser";
import { getLogger } from "./logger.js";
import { getCurrentUserId, withExecutionSignal } from "./context.js";
import { getSuspendHandler } from "./hooks.js";
import {
  LLMPodStreamEventSchema,
  serializeInvokeLLMRequestArguments,
  type InvokeLLMRequestArguments,
  type LLMPodStreamEvent,
  type ToolExecuteResponse,
} from "./models.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "./tls_client.js";
import {
  LLM_READ_TIMEOUT,
  OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS,
  OE_RETRYABLE_MAX_ATTEMPTS,
  oeStreamRetry,
  oeStreamRetryDelayMs,
} from "./utils.js";
import {
  LLMInvocationError,
  OperationalStepAllocator,
  PolicyDeniedException,
  raiseForOeRejection,
  requestOeApprovalRetryable,
  type OperationalStepSource,
} from "./secure_wrapper.js";
import {
  ActivityKind,
  DurableActivityDeniedError,
  type DurableMemoryState,
  ReplayedActivityFailedError,
  WorkflowClient,
  allocateActivityOrdinal,
  currentAttemptContext,
  preallocateActivityOrdinals,
  runStreamingActivity,
  toolActivityKey,
} from "./workflow/index.js";

const logger = getLogger("agent_engine_runner_shared.secure_llm_proxy");

function usageToWireDict(usage: LLMTokenUsage): Record<string, JsonValue> {
  const result: Record<string, JsonValue> = {};
  if (usage.inputTokens != null) result["input_tokens"] = usage.inputTokens;
  if (usage.outputTokens != null) result["output_tokens"] = usage.outputTokens;
  if (usage.totalTokens != null) result["total_tokens"] = usage.totalTokens;
  if (usage.promptTokens != null) result["prompt_tokens"] = usage.promptTokens;
  if (usage.completionTokens != null)
    result["completion_tokens"] = usage.completionTokens;
  return result;
}

// =============================================================================
// Internal SSE streaming helper
// =============================================================================

/**
 * Async-iterate SSE events from a fetch Response.
 *
 * Uses eventsource-parser to handle multi-line data and edge cases. Cancels
 * the reader on exit (consumer break or throw) to release the HTTP connection.
 *
 * Note: AbortSignal.timeout() starts from the moment the signal is created, not
 * from the last received byte. Python's httpx `read=LLM_READ_TIMEOUT` is a
 * per-read idle timeout. The idle timer below resets after each raw chunk from
 * reader.read(), matching httpx's per-recv semantics exactly.
 */
async function* parseSseStream(
  response: Response,
  options: { onChunk?: () => void; signal?: AbortSignal } = {},
): AsyncGenerator<string> {
  const { onChunk, signal } = options;
  if (!response.body) {
    // 204/205/304 or a polyfill without a body — surface a legible error that
    // the caller converts into an LLMInvocationError, rather than a cryptic
    // "Cannot read properties of null" TypeError from .getReader().
    throw new Error("SSE response has no body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const pending: string[] = [];

  const parser = createParser({
    onEvent: (event) => {
      if (event.data) pending.push(event.data);
    },
  });

  // Belt-and-suspenders: when the caller's signal fires, cancel the reader so
  // reader.read() unblocks immediately. In Node.js/undici the fetch signal
  // already propagates to the body, but this covers edge cases.
  const abortHandler = () => {
    reader.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", abortHandler, { once: true });

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      onChunk?.(); // reset idle timer on every raw recv — matches httpx read= semantics
      parser.feed(decoder.decode(value, { stream: true }));
      while (pending.length > 0) {
        const chunk = pending.shift();
        if (chunk !== undefined) yield chunk;
      }
    }
  } finally {
    signal?.removeEventListener("abort", abortHandler);
    try {
      await reader.cancel();
    } catch {
      /* ignore cancel errors */
    }
  }
}

function isOeStreamTransportDisconnect(exc: unknown): boolean {
  // fetch network failures are TypeError. AbortError is idle timeout,
  // execution cancel, or consumer abort — none of those are owner-death.
  if (exc instanceof DOMException && exc.name === "AbortError") return false;
  if (exc instanceof Error && exc.name === "AbortError") return false;
  return exc instanceof TypeError;
}

// =============================================================================
// SecureLLMProxy
// =============================================================================

export class SecureLLMProxy {
  readonly oeUrl: string;
  readonly executionId: string;
  readonly modelName: string;
  readonly llmId: string;
  readonly boundTools: unknown[] | null;
  // Forwarded to the tool pod's bind_tools call so a forced tool choice
  // (e.g. withStructuredOutput) survives the OE round-trip.
  readonly boundToolChoice: unknown;
  /** Prefer the wrapper's allocator so tool + LLM share one sequence. */
  readonly operationalSteps: OperationalStepSource;
  readonly durableMemory: DurableMemoryState | null;

  lastDurationMs: number;
  lastFromCache: boolean;
  lastLatestStepNumber: number | null;
  lastPodName: string | null;

  constructor(args: {
    oeUrl: string;
    executionId: string;
    modelName?: string;
    llmId?: string;
    boundTools?: unknown[] | null;
    boundToolChoice?: unknown;
    operationalSteps?: OperationalStepSource;
    durableMemory?: DurableMemoryState | null;
  }) {
    this.oeUrl = args.oeUrl.replace(/\/$/, "");
    this.executionId = args.executionId;
    this.modelName = args.modelName ?? "unknown";
    this.llmId = args.llmId ?? "__default__";
    this.boundTools = args.boundTools ?? null;
    this.boundToolChoice = args.boundToolChoice ?? null;
    this.operationalSteps =
      args.operationalSteps ?? new OperationalStepAllocator();
    this.durableMemory = args.durableMemory ?? null;
    this.lastDurationMs = 0.0;
    this.lastFromCache = false;
    this.lastLatestStepNumber = null;
    this.lastPodName = null;
  }

  /** Current operational-step watermark for compatibility readers. */
  get stepCounter(): number {
    return this.operationalSteps.current();
  }

  private allocateStep(step?: number | null): number {
    if (step == null) {
      return this.operationalSteps.next();
    }
    this.operationalSteps.observeAtLeast(step);
    return step;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Invoke LLM by collecting the stream-oriented execution path.
   * Equivalent to Python's `invoke()` which calls `list(self.stream(...))`.
   */
  async invoke(
    messages: Message[],
    step?: number | null,
    stop?: string[] | null,
    options?: LLMInvocationOptions | null,
  ): Promise<LLMResponse> {
    const chunks: LLMStreamChunk[] = [];
    for await (const chunk of this.stream(messages, step, stop, options)) {
      chunks.push(chunk);
    }
    return SecureLLMProxy.responseFromStreamChunks(chunks);
  }

  /**
   * Stream invoke_llm chunks through OE approval and OE-owned SSE relay.
   *
   * Step auto-increments when not supplied. If OE returns a cached/sync
   * result, a synthetic LLMStreamChunk is yielded instead of opening an
   * SSE connection.
   */
  async *stream(
    messages: Message[],
    step?: number | null,
    stop?: string[] | null,
    options?: LLMInvocationOptions | null,
  ): AsyncGenerator<LLMStreamChunk> {
    const resolvedStep = this.allocateStep(step);
    const invokeRequest = this.buildInvokeRequest(
      messages,
      stop ?? null,
      options ?? null,
      true,
    );

    if (currentAttemptContext() === null) {
      yield* this.streamNative(invokeRequest, resolvedStep);
      return;
    }
    yield* this.streamDurably(invokeRequest, resolvedStep);
  }

  private async *streamDurably(
    invokeRequest: InvokeLLMRequestArguments,
    resolvedStep: number,
  ): AsyncGenerator<LLMStreamChunk> {
    const durableMemory = this.durableMemory;
    try {
      yield* runStreamingActivity({
        client: new WorkflowClient(this.oeUrl),
        kind: ActivityKind.LLM,
        name: this.modelName,
        activityOrdinal: allocateActivityOrdinal(`llm:${resolvedStep}`),
        semanticInput: serializeInvokeLLMRequestArguments(invokeRequest),
        execute: () => this.streamActivityEffect(invokeRequest, resolvedStep),
        replay: (result) => this.replayActivityResult(result),
        fold: (chunks) => this.completeActivityStream(chunks),
        onActivityResolved:
          durableMemory === null
            ? undefined
            : (client, context, result) =>
                durableMemory.synchronizeLlm(
                  client as WorkflowClient,
                  context,
                  result,
                  getCurrentUserId(),
                ),
        // The platform-owned operational step is the LLM call's stable
        // call-order identity. Frameworks may consume tool-call chunks before
        // the provider stream has fully closed.
        exclusive: false,
      });
    } catch (error) {
      if (error instanceof DurableActivityDeniedError) {
        if (error.cause instanceof PolicyDeniedException) throw error.cause;
        throw new PolicyDeniedException(error.message, null, {
          cause: error,
        });
      }
      if (error instanceof ReplayedActivityFailedError) {
        throw new LLMInvocationError(error.message, { cause: error });
      }
      throw error;
    }
  }

  private async *streamActivityEffect(
    invokeRequest: InvokeLLMRequestArguments,
    resolvedStep: number,
  ): AsyncGenerator<LLMStreamChunk> {
    try {
      yield* this.streamNative(invokeRequest, resolvedStep, "reject");
    } catch (error) {
      // The activity kernel records this error class as DENIED, not FAILED.
      if (error instanceof PolicyDeniedException) {
        throw new DurableActivityDeniedError(error.reason, { cause: error });
      }
      throw error;
    }
  }

  private async *streamNative(
    invokeRequest: InvokeLLMRequestArguments,
    resolvedStep: number,
    interruptMode: "framework" | "reject" = "framework",
  ): AsyncGenerator<LLMStreamChunk> {
    const response = await this.requestOeExecution(invokeRequest, resolvedStep);

    // Guardrail non-proceed responses: require_review suspends for a human;
    // a genuine block returns the substitute *string* in result, which we yield
    // as a clean chunk so the agent gets a response rather than an exception.
    // Any other non-proceed shape is a hard denial.
    if (!response.proceed) {
      if (response.status === "require_review") {
        if (interruptMode === "reject") {
          throw new LLMInvocationError(
            "durable LLM activities support terminal outcomes only; " +
              "the framework adapter must own review interrupts",
          );
        }
        yield* this.handleRequireReview(response);
        return;
      }
      if (typeof response.result === "string") {
        yield { content: response.result };
        return;
      }
      raiseForOeRejection(response.reason, response.guardrail_meta ?? null);
    }

    const result = response.result ?? response.cached_result;
    const status =
      response.status ??
      (this.lastFromCache && result != null ? "success" : null);

    if (status === "error") {
      throw new LLMInvocationError(response.error ?? "LLM streaming failed", {
        source: "llm",
        ...(response.error_code ? { error_code: response.error_code } : {}),
      });
    }

    if (status === "interrupted") {
      // OE stopped this call on interrupt (inline or replayed from a durable
      // interrupted step): yield the same clean marker as the live SSE path so
      // the agent continues instead of throwing on an unexpected status.
      yield { responseMetadata: { interrupted: true } };
      return;
    }

    if (
      response.route_to &&
      response.route_to !== "callback" &&
      result == null
    ) {
      yield* this.streamFromOe(response.route_to, resolvedStep);
      return;
    }

    if (status !== "success" && result == null) {
      throw new LLMInvocationError(
        `Unexpected OE invoke_llm status: ${status}`,
      );
    }

    yield* SecureLLMProxy.chunksFromResponse(
      SecureLLMProxy.convertResultToResponse(result),
    );
  }

  private *replayActivityResult(result: unknown): Generator<LLMStreamChunk> {
    this.lastFromCache = true;
    if (typeof result === "string") {
      yield { content: result };
      return;
    }
    this.preallocateToolCallsFromResult(result);
    if (
      typeof result !== "object" ||
      result === null ||
      Array.isArray(result)
    ) {
      throw new LLMInvocationError(
        `Recorded LLM result has unexpected type: ${typeof result}`,
      );
    }
    yield SecureLLMProxy.completeChunkFromResponse(
      SecureLLMProxy.convertResultToResponse(result as Record<string, unknown>),
    );
  }

  private completeActivityStream(
    chunks: LLMStreamChunk[],
  ): Record<string, unknown> {
    const payload = SecureLLMProxy.responsePayloadFromChunks(chunks);
    this.preallocateToolCallsFromResult(payload);
    return payload;
  }

  private preallocateToolCallsFromResult(result: unknown): void {
    if (
      typeof result !== "object" ||
      result === null ||
      Array.isArray(result)
    ) {
      return;
    }
    const toolCalls = (result as Record<string, unknown>)["tool_calls"];
    if (!Array.isArray(toolCalls)) return;
    const keys = toolCalls.flatMap((toolCall) => {
      if (
        typeof toolCall !== "object" ||
        toolCall === null ||
        Array.isArray(toolCall)
      ) {
        return [];
      }
      const id = (toolCall as Record<string, unknown>)["id"];
      return typeof id === "string" && id ? [toolActivityKey(id)] : [];
    });
    if (keys.length > 0) preallocateActivityOrdinals(keys);
  }

  private static chunksFromResponse(response: LLMResponse): LLMStreamChunk[] {
    const chunks: LLMStreamChunk[] = [];
    const { usage, ...responseChunk } =
      SecureLLMProxy.completeChunkFromResponse(response);
    if (
      responseChunk.content ||
      responseChunk.toolCalls != null ||
      responseChunk.id != null ||
      responseChunk.name != null ||
      responseChunk.responseMetadata != null ||
      responseChunk.additionalKwargs != null
    ) {
      chunks.push(responseChunk);
    }
    if (usage != null) chunks.push({ usage });
    return chunks;
  }

  private static completeChunkFromResponse(
    response: LLMResponse,
  ): LLMStreamChunk {
    return {
      content: response.content || undefined,
      toolCalls:
        SecureLLMProxy.convertToolCallsToStreamChunks(
          response.toolCalls ?? null,
        ) ?? undefined,
      usage: response.usage ?? undefined,
      id: response.id,
      name: response.name,
      responseMetadata: response.responseMetadata,
      additionalKwargs: response.additionalKwargs,
    };
  }

  private static responsePayloadFromChunks(
    chunks: LLMStreamChunk[],
  ): Record<string, unknown> {
    const response = SecureLLMProxy.responseFromStreamChunks(chunks);
    const result: Record<string, unknown> = {
      content: response.content,
      metadata: response.metadata,
    };
    if (response.toolCalls !== undefined) {
      result["tool_calls"] = response.toolCalls.map((toolCall) => {
        const value: Record<string, unknown> = {};
        if (toolCall.id !== undefined) value["id"] = toolCall.id;
        if (toolCall.name !== undefined) value["name"] = toolCall.name;
        if (toolCall.args !== undefined) value["args"] = toolCall.args;
        if (toolCall.type !== undefined) value["type"] = toolCall.type;
        if (toolCall.index !== undefined) value["index"] = toolCall.index;
        return value;
      });
    }
    if (response.usage !== undefined) {
      result["usage"] = response.usage.toJSON();
    }
    if (response.id !== undefined) result["id"] = response.id;
    if (response.name !== undefined) result["name"] = response.name;
    if (response.responseMetadata !== undefined) {
      result["response_metadata"] = response.responseMetadata;
    }
    if (response.additionalKwargs !== undefined) {
      result["additional_kwargs"] = response.additionalKwargs;
    }
    return result;
  }

  /**
   * Handle a guardrail require_review response by suspending via the framework
   * suspend handler (LangGraph `interrupt`). On first call the handler suspends
   * the node and never returns; on resume it returns the OE-dispatched decision
   * — an object with a top-level `guardrail_review` key. Approve yields the
   * pending LLM content; deny (or an unrecognised decision) denies the call.
   * Mirrors Python's `_handle_require_review`.
   */
  private *handleRequireReview(
    response: ToolExecuteResponse,
  ): Generator<LLMStreamChunk> {
    const interrupt = getSuspendHandler();
    if (interrupt == null) {
      throw new PolicyDeniedException(
        "Guardrail require_review: no suspend handler registered. " +
          "Ensure the framework SDK calls registerSuspendHandler() before run().",
        response.guardrail_meta ?? null,
      );
    }

    const suspendPayload: Record<string, unknown> = {
      suspend_reason: "guardrail_require_review",
      allowed_decisions: ["approve", "deny"],
      reason: response.reason ?? "Guardrail required human review",
      guardrail_meta: response.guardrail_meta ?? null,
    };
    logger.info(
      { executionId: this.executionId, step: this.stepCounter },
      "Guardrail require_review: suspending for human review",
    );

    // First call suspends via interrupt() and never returns; on resume it
    // returns the stored decision.
    const humanDecision = interrupt(suspendPayload);
    if (humanDecision == null || typeof humanDecision !== "object") {
      throw new LLMInvocationError(
        `Guardrail require_review: expected object from interrupt, got ${typeof humanDecision}`,
      );
    }

    const guardrailReview = (humanDecision as Record<string, unknown>)
      .guardrail_review;
    if (guardrailReview == null || typeof guardrailReview !== "object") {
      throw new LLMInvocationError(
        "Guardrail require_review: resume data missing or invalid " +
          `'guardrail_review' key (got keys: ${Object.keys(humanDecision).join(", ")})`,
      );
    }

    const review = guardrailReview as Record<string, unknown>;
    const decision = review.decision ?? "";

    if (decision === "approve") {
      const pendingContent = review.pending_llm_content;
      if (pendingContent == null) {
        logger.error(
          { executionId: this.executionId },
          "Guardrail require_review: approved but pending_llm_content missing from resume data",
        );
        throw new LLMInvocationError(
          "Guardrail require_review: approved but OE did not include pending LLM content",
        );
      }
      logger.info(
        { executionId: this.executionId },
        "Guardrail require_review: approved — yielding original LLM content",
      );
      yield* SecureLLMProxy.chunksFromResponse(
        SecureLLMProxy.convertResultToResponse(pendingContent),
      );
      return;
    }

    if (decision !== "deny") {
      logger.warn(
        { executionId: this.executionId, decision },
        "Guardrail require_review: unrecognised decision value — treating as deny",
      );
    } else {
      logger.info(
        { executionId: this.executionId, decision },
        "Guardrail require_review: denied by reviewer",
      );
    }
    throw new PolicyDeniedException(
      `Guardrail require_review: denied by reviewer (decision=${JSON.stringify(decision)})`,
      response.guardrail_meta ?? null,
    );
  }

  /**
   * Collect streamed sdk-core chunks into a final sdk-core LLMResponse.
   * Static — can be called without a proxy instance (e.g. after parallel streaming).
   */
  static responseFromStreamChunks(chunks: LLMStreamChunk[]): LLMResponse {
    const contentParts: string[] = [];
    const streamedToolCalls: ToolCallChunk[] = [];
    let usage: LLMTokenUsage | undefined;
    let messageId: string | undefined;
    let messageName: string | undefined;
    let responseMetadata: Record<string, JsonValue> | undefined;
    let additionalKwargs: Record<string, JsonValue> | undefined;

    for (const chunk of chunks) {
      if (chunk.content) contentParts.push(chunk.content);
      if (chunk.toolCalls) streamedToolCalls.push(...chunk.toolCalls);
      if (chunk.usage != null) {
        usage = chunk.usage;
      }
      // LangChain JS keeps the first message ID while concatenating streamed
      // AIMessageChunks. Preserve the same ID in the recorded terminal result
      // so a synthetic replay produces the identical checkpoint message.
      if (messageId === undefined && chunk.id != null) messageId = chunk.id;
      if (chunk.name != null) messageName = chunk.name;
      if (chunk.responseMetadata != null) {
        responseMetadata = {
          ...(responseMetadata ?? {}),
          ...chunk.responseMetadata,
        };
      }
      if (chunk.additionalKwargs != null) {
        additionalKwargs = {
          ...(additionalKwargs ?? {}),
          ...chunk.additionalKwargs,
        };
      }
    }

    const toolCalls =
      SecureLLMProxy.convertStreamChunksToToolCalls(streamedToolCalls);
    if (toolCalls != null && additionalKwargs?.["tool_calls"] !== undefined) {
      // Provider stream fragments are redundant once semantic tool calls
      // exist; persisting them would smuggle non-portable data into the
      // recorded result and back into replay. Same rule as
      // lcToPlatformMessage.
      const { tool_calls: _providerFragments, ...portableKwargs } =
        additionalKwargs;
      additionalKwargs =
        Object.keys(portableKwargs).length > 0 ? portableKwargs : undefined;
    }
    const metadata: Record<string, JsonValue> =
      usage != null ? usageToWireDict(usage) : {};

    return new LLMResponse({
      content: contentParts.join(""),
      toolCalls: toolCalls ?? undefined,
      metadata,
      usage,
      id: messageId,
      name: messageName,
      responseMetadata,
      additionalKwargs,
    });
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Serialize bound tools to LLMToolSchema for transmission to OE.
   *
   * TS divergence from Python's `_serialize_bound_tools`: LangChain JS has no
   * `tool_call_schema` attribute on tools (it's a Python-only API). Tools
   * carrying Zod input schemas are normalized to OpenAI-canonical dicts
   * upstream by `_normalizeBoundTool` in agent-engine-sdk-langgraph-ts via
   * `convertToOpenAITool`, so every entry that reaches us here is already
   * either a plain dict or an `LLMToolSchema` instance — both go through
   * `LLMToolSchemaValidator`.
   */
  private serializeBoundTools(): LLMToolSchema[] | null {
    if (!this.boundTools || this.boundTools.length === 0) return null;

    const serialized: LLMToolSchema[] = [];
    for (const tool of this.boundTools) {
      if (tool === null || typeof tool !== "object" || Array.isArray(tool)) {
        logger.warn(
          { toolType: typeof tool },
          `SecureLLMProxy: Skipping unrecognized tool type ${typeof tool} during serialization`,
        );
        continue;
      }

      const parsed = this.safeParseToolSchema(tool);
      if (parsed) serialized.push(parsed);
    }
    return serialized.length > 0 ? serialized : null;
  }

  private safeParseToolSchema(value: unknown): LLMToolSchema | null {
    const result = LLMToolSchemaValidator.safeParse(value);
    if (result.success) return result.data;
    logger.warn(
      { issues: result.error.issues },
      "SecureLLMProxy: Skipping tool that failed LLMToolSchema validation",
    );
    return null;
  }

  /**
   * Validate `boundToolChoice` (typed `unknown` at the proxy boundary) as a
   * `JsonValue` for the wire. Rejects non-JSON values rather than letting them
   * break `JSON.stringify` or be silently dropped. `null`/`undefined` → omit
   * the field; `false` is preserved to disable forced tool use.
   */
  private resolveToolChoice(): JsonValue | undefined {
    if (this.boundToolChoice == null) return undefined;
    const result = JsonValueSchema.safeParse(this.boundToolChoice);
    if (!result.success) {
      throw new LLMInvocationError(
        `Invalid tool_choice: expected a JSON-serializable value, got ${typeof this.boundToolChoice}`,
      );
    }
    return result.data;
  }

  private buildInvokeRequest(
    messages: Message[],
    stop: string[] | null,
    options: LLMInvocationOptions | null,
    stream: boolean,
  ): InvokeLLMRequestArguments {
    const tools = this.serializeBoundTools();
    // `messages` are already validated internal `Message[]` (built by
    // lcMessagesToPlatform). Do NOT re-run them through
    // InvokeLLMRequestArgumentsSchema.parse — its `messages: z.array(MessageSchema)`
    // applies the snake→camel "parse" transform, which blanks the camelCase
    // fields (toolCallId/toolCalls) of already-internal messages. Parse is the
    // wire→internal boundary; build the typed envelope directly here, and let
    // serializeInvokeLLMRequestArguments do the camel→snake dump on the way out.
    return {
      messages,
      model: this.modelName,
      llm_id: this.llmId,
      stop_sequences: stop ?? undefined,
      tools: tools ?? undefined,
      tool_choice: this.resolveToolChoice(),
      options: options ?? undefined,
      stream,
    };
  }

  private async requestOeExecution(
    invokeRequest: InvokeLLMRequestArguments,
    step: number,
  ): Promise<ToolExecuteResponse> {
    // The sole caller (`stream()`) always resolves the step before calling in,
    // so step is required and non-null here. Keep the counter monotonic.
    this.operationalSteps.observeAtLeast(step);

    const response = await requestOeApprovalRetryable({
      oeUrl: this.oeUrl,
      executionId: this.executionId,
      toolName: "invoke_llm",
      // serializeInvokeLLMRequestArguments is the camel→snake "dump" boundary,
      // the equivalent of Python's model_dump(by_alias=True): stop_sequences →
      // stop and each Message dumped to snake_case wire fields.
      arguments: serializeInvokeLLMRequestArguments(invokeRequest) as Record<
        string,
        unknown
      >,
      step,
      // OE holds /tool/execute open while the (non-streaming) LLM call
      // completes, so use the long LLM read timeout rather than the generic
      // request timeout. Mirrors Python's
      // `httpx.Timeout(get_request_timeout(), read=LLM_READ_TIMEOUT)`.
      timeoutMs: LLM_READ_TIMEOUT * 1000,
    });

    // Let require_review and block-substitute (proceed=false with a result
    // string) responses through — stream() handles them. Only a genuine block
    // with no substitute content is a hard denial here.
    if (
      !response.proceed &&
      response.result == null &&
      response.status !== "require_review"
    ) {
      raiseForOeRejection(response.reason, response.guardrail_meta ?? null);
    }

    this.lastDurationMs = response.duration_ms ?? 0.0;
    this.lastFromCache = response.from_cache ?? response.cached_result != null;
    this.lastLatestStepNumber = response.latest_step_number ?? null;
    this.lastPodName = response.pod_name ?? null;

    if (response.latest_step_number != null) {
      this.operationalSteps.observeAtLeast(response.latest_step_number);
    }

    return response;
  }

  private static convertResultToResponse(result: unknown): LLMResponse {
    if (result == null)
      throw new LLMInvocationError("OE returned no invoke_llm result");

    let response: LLMResponse;
    try {
      response = LLMResponse.fromRaw(result as Record<string, unknown>);
    } catch (exc) {
      throw new LLMInvocationError(
        `OE returned unexpected invoke_llm result type: ${typeof result}`,
        { cause: exc },
      );
    }

    // Python parity: if usage is present but metadata is empty, copy usage fields into metadata.
    // Python's model_validator does this as a post-parse mutation; LLMResponse.fromRaw does not.
    if (response.usage != null && Object.keys(response.metadata).length === 0) {
      const usageDict = usageToWireDict(response.usage);
      return new LLMResponse({
        content: response.content,
        toolCalls: response.toolCalls,
        metadata: usageDict,
        usage: response.usage,
        id: response.id,
        name: response.name,
        additionalKwargs: response.additionalKwargs,
        responseMetadata: response.responseMetadata,
      });
    }
    return response;
  }

  private static convertToolCallsToStreamChunks(
    toolCalls: LLMToolCall[] | null,
  ): ToolCallChunk[] | null {
    if (!toolCalls || toolCalls.length === 0) return null;

    return toolCalls.map((tc, fallbackIndex) => ({
      id: tc.id,
      name: tc.name,
      args: SecureLLMProxy.normalizeToolCallArgs(tc.args),
      type: tc.type,
      index: tc.index ?? (tc.id == null ? fallbackIndex : undefined),
    }));
  }

  private static normalizeToolCallArgs(args: unknown): string | undefined {
    if (args == null) return undefined;
    if (typeof args === "string") return args;
    try {
      return JSON.stringify(args);
    } catch {
      return undefined;
    }
  }

  private static parseToolCallArgs(rawArgs: string): JsonValue {
    try {
      return JSON.parse(rawArgs) as JsonValue;
    } catch {
      return rawArgs;
    }
  }

  private static convertStreamChunksToToolCalls(
    chunks: ToolCallChunk[],
  ): LLMToolCall[] | null {
    if (chunks.length === 0) return null;

    const accumulated = new Map<number, Record<string, unknown>>();
    const idToIndex = new Map<string, number>();
    let currentIndex: number | null = null;
    let nextIndex = 0;

    for (const chunk of chunks) {
      let index: number;
      if (chunk.index != null) {
        index = chunk.index;
        currentIndex = index;
        nextIndex = Math.max(nextIndex, index + 1);
      } else if (chunk.id != null) {
        const existing = idToIndex.get(chunk.id);
        if (existing != null) {
          index = existing;
        } else {
          index = nextIndex++;
          idToIndex.set(chunk.id, index);
        }
        currentIndex = index;
      } else if (currentIndex != null) {
        index = currentIndex;
      } else {
        index = nextIndex++;
        currentIndex = index;
      }

      const entry = accumulated.get(index) ?? {};
      accumulated.set(index, entry);

      if (chunk.index != null) entry["index"] = chunk.index;
      if (chunk.id != null) {
        const prevIndex = idToIndex.get(chunk.id);
        if (prevIndex != null && prevIndex !== index) {
          logger.warn(
            { id: chunk.id, prevIndex, index },
            `LLM: Tool call id ${chunk.id} changed stream index from ${prevIndex} to ${index}`,
          );
        }
        entry["id"] = chunk.id;
        idToIndex.set(chunk.id, index);
      }
      if (chunk.name != null) entry["name"] = chunk.name;
      if (chunk.type != null) entry["type"] = chunk.type;
      if (chunk.args != null) {
        const existing = entry["args"];
        entry["args"] =
          typeof existing === "string" ? existing + chunk.args : chunk.args;
      }
    }

    const toolCalls: LLMToolCall[] = [];
    for (const index of [...accumulated.keys()].sort((a, b) => a - b)) {
      const accumulatedEntry = accumulated.get(index);
      if (accumulatedEntry === undefined) continue;
      const payload = { ...accumulatedEntry };
      const rawArgs = payload["args"];
      if (typeof rawArgs === "string") {
        payload["args"] = SecureLLMProxy.parseToolCallArgs(rawArgs);
      }
      toolCalls.push(
        new LLMToolCall(
          payload as ConstructorParameters<typeof LLMToolCall>[0],
        ),
      );
    }
    return toolCalls.length > 0 ? toolCalls : null;
  }

  private static streamChunkFromEvent(
    event: LLMPodStreamEvent,
  ): LLMStreamChunk | null {
    const content = event.content ?? undefined;

    const toolCalls: ToolCallChunk[] = [];
    if (event.tool_call_chunks) toolCalls.push(...event.tool_call_chunks);
    for (const [i, tc] of (event.tool_calls ?? []).entries()) {
      toolCalls.push({
        id: tc["id"] as string | undefined,
        name: tc["name"] as string | undefined,
        args: SecureLLMProxy.normalizeToolCallArgs(tc["args"]),
        type: tc["type"] as string | undefined,
        index: (tc["index"] as number | undefined) ?? i,
      });
    }

    if (
      content == null &&
      toolCalls.length === 0 &&
      event.id == null &&
      event.name == null &&
      event.response_metadata == null &&
      event.additional_kwargs == null &&
      event.usage == null
    ) {
      return null;
    }

    return {
      content,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      usage: event.usage ?? undefined,
      id: event.id ?? undefined,
      name: event.name ?? undefined,
      responseMetadata: event.response_metadata ?? undefined,
      additionalKwargs: event.additional_kwargs ?? undefined,
    };
  }

  /**
   * Stream real-time LLM chunks from OE's audited SSE relay.
   *
   * Timeout: a single idle AbortController whose timer resets on every raw
   * recv (via parseSseStream's onChunk callback). This matches httpx's
   * `read=LLM_READ_TIMEOUT` behavior — the timeout only fires when the
   * connection goes *silent* for LLM_READ_TIMEOUT seconds, not after that
   * many seconds of wall-clock time. Active streams are never cut short.
   * The timer also covers the connect phase (server not responding at all).
   */
  private async *streamFromOe(
    streamUrl: string,
    step: number,
  ): AsyncGenerator<LLMStreamChunk> {
    let error: string | null = null;
    let podName: string | null = null;
    let durationMs = 0.0;
    const startTime = Date.now();
    let exposedOutput = false;

    for (let attempt = 0; attempt < OE_RETRYABLE_MAX_ATTEMPTS; attempt++) {
      let status = "success";
      error = null;
      let errorCode: string | undefined;
      let caughtException: Error | null = null;
      let doneReceived = false;
      let idleTimedOut = false;
      let retrySameUrl = false;
      let retryDelayMs = 0;
      let providerOwned = false;

      const idleController = new AbortController();
      let idleTimer: ReturnType<typeof setTimeout> | null = null;
      const resetIdle = () => {
        if (idleTimer != null) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
          idleTimedOut = true;
          idleController.abort();
        }, LLM_READ_TIMEOUT * 1000);
      };

      resetIdle(); // starts counting before fetch — covers connect phase

      // Combine the idle-timeout signal with the execution-wide abort signal so
      // an AER execution timeout cancels the in-flight SSE stream (not just the
      // idle window). Mirrors Python's wait_for cancelling the inner coroutine.
      const streamSignal = withExecutionSignal(idleController.signal);

      try {
        const tlsOptions = getFetchOptionsWithTLS(this.oeUrl);
        const resp = await fetchPlatform(streamUrl, {
          method: "POST",
          signal: streamSignal,
          ...tlsOptions,
        });

        resetIdle(); // fresh window after headers received — body read phase starts now

        if (!resp.ok) {
          throw new Error(
            `SSE stream returned HTTP ${resp.status}: ${resp.statusText}`,
          );
        }

        for await (const data of parseSseStream(resp, {
          onChunk: resetIdle,
          signal: streamSignal,
        })) {
          let streamEvent: LLMPodStreamEvent;
          try {
            streamEvent = LLMPodStreamEventSchema.parse(JSON.parse(data));
          } catch {
            logger.warn(
              { step, data: data.slice(0, 120) },
              `LLM: Step ${step} - Skipping malformed SSE payload`,
            );
            continue;
          }

          if (streamEvent.interrupted) {
            // OE stopped this call on interrupt: not error, not truncation.
            // Yield a clean marker so the agent continues instead of raising.
            doneReceived = true;
            podName = streamEvent.pod_name ?? podName;
            if (streamEvent.duration_ms != null)
              durationMs = streamEvent.duration_ms;
            yield { responseMetadata: { interrupted: true } };
            break;
          }

          if (streamEvent.error) {
            error = streamEvent.error;
            if (
              streamEvent.retryable &&
              attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1 &&
              !exposedOutput
            ) {
              retrySameUrl = true;
              retryDelayMs = oeStreamRetryDelayMs(streamEvent.retry_after_ms);
            } else {
              status = "error";
              providerOwned = true;
              errorCode = streamEvent.error_code;
            }
            break;
          }

          if (streamEvent.done) {
            doneReceived = true;
            podName = streamEvent.pod_name ?? null;
            if (streamEvent.duration_ms != null)
              durationMs = streamEvent.duration_ms;
            if (streamEvent.usage != null) {
              yield { usage: streamEvent.usage };
            }
            break;
          }

          const chunk = SecureLLMProxy.streamChunkFromEvent(streamEvent);
          if (chunk != null) {
            exposedOutput = true;
            yield chunk;
          }
        }
      } catch (exc) {
        if (status !== "error") {
          status = "error";
          error = exc instanceof Error ? exc.message : String(exc);
          caughtException = exc instanceof Error ? exc : null;
        }
        if (
          attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1 &&
          !exposedOutput &&
          !idleTimedOut &&
          isOeStreamTransportDisconnect(exc)
        ) {
          retrySameUrl = true;
          retryDelayMs = OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS;
        }
      } finally {
        if (idleTimer != null) clearTimeout(idleTimer);
        if (durationMs === 0.0) durationMs = Date.now() - startTime;
        if (idleTimedOut) {
          // Idle timer fired — override whatever error landed in the catch block
          // (usually "This operation was aborted" from the AbortError) with a
          // human-readable message that matches Python's read-timeout semantics.
          status = "error";
          error = `LLM read idle timeout: no data received for ${LLM_READ_TIMEOUT}s`;
          caughtException = null;
          retrySameUrl = false;
          providerOwned = false;
        } else if (!retrySameUrl && status === "success" && !doneReceived) {
          error = "SSE stream ended without done signal (truncated response)";
          if (attempt < OE_RETRYABLE_MAX_ATTEMPTS - 1 && !exposedOutput) {
            retrySameUrl = true;
            retryDelayMs = OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS;
          } else {
            status = "error";
          }
        }
        this.lastDurationMs = durationMs;
        this.lastPodName = podName;
      }

      if (retrySameUrl) {
        await oeStreamRetry.sleep(retryDelayMs);
        continue;
      }
      if (status === "error") {
        const msg = error ?? "LLM streaming failed";
        throw new LLMInvocationError(msg, {
          ...(caughtException ? { cause: caughtException } : {}),
          ...(providerOwned ? { source: "llm" } : {}),
          ...(errorCode ? { error_code: errorCode } : {}),
        });
      }
      return;
    }

    throw new LLMInvocationError(error ?? "LLM streaming failed");
  }
}
