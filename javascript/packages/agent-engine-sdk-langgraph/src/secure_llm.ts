/**
 * SecureWrappedLLM — LangChain BaseChatModel adapter for secure LLM calls.
 *
 * Thin adapter that provides the LangChain `BaseChatModel` interface while
 * delegating `invoke_llm` orchestration to `SecureLLMProxy` in runner-shared.
 *
 * Port of `agent_engine_sdk_langgraph/secure_llm.py`.
 */

import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  type BaseChatModel,
  BaseChatModel as _BaseChatModel,
} from "@langchain/core/language_models/chat_models";
import type { BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import {
  convertToOpenAITool,
  isLangChainTool,
} from "@langchain/core/utils/function_calling";

import {
  LLMInvocationOptions,
  LLMToolSchema,
} from "@mongodb-js/agent-engine-sdk";
import {
  type DurableMemoryState,
  getLogger,
  getTracer,
  LLMInvocationError,
  logCachedResult,
  logLLMMessages,
  logLLMResponse,
  logPolicyBlocked,
  logSeparator,
  Metrics,
  MODEL_REQUEST_PREPARE,
  MODEL_RESPONSE_PROCESS,
  OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind,
  PolicyDeniedException,
  SecureLLMProxy,
} from "@mongodb-js/agent-engine-runner-shared";
import {
  lcMessagesToPlatform,
  llmResponseToChatResult,
  llmStreamChunkToGenerationChunk,
} from "./messages.js";
import { stampDeepAgentMessageIds } from "./durable_deep_agent.js";

const logger = getLogger("agent_engine_sdk_langgraph.secure_llm");

// Re-export for backward compatibility (tests import from here)
export { LLMInvocationError, PolicyDeniedException };

const MISSING_TOOL_RESULT_TEMPLATE = (toolName: string): string =>
  `Tool execution failed before the platform recorded a result for ${toolName}. ` +
  `Treat this as a failed tool call and continue with the user request.`;

/**
 * Normalize a bound tool into a shape that `SecureLLMProxy.serializeBoundTools`
 * can handle without losing the input schema.
 *
 * - LangChain JS tools → `convertToOpenAITool` → OpenAI-canonical dict.
 * - `LLMToolSchema` instances and plain dicts → returned as-is.
 * - Anything else → returned as-is (the proxy will log and skip if it can't
 *   make sense of it, matching pre-existing behavior).
 */
function normalizeBoundTool(tool: unknown): unknown {
  if (tool instanceof LLMToolSchema) return tool;
  if (isLangChainTool(tool)) {
    try {
      return convertToOpenAITool(tool);
    } catch (err) {
      logger.warn(
        { err: err instanceof Error ? err.message : String(err) },
        "SecureWrappedLLM.bindTools: convertToOpenAITool failed; passing tool through unchanged",
      );
      return tool;
    }
  }
  return tool;
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * Structural shape of the `SecureToolWrapper` retrieved via `getCurrentWrapper()`.
 * Matches the public fields of agent-engine-runner-shared's `SecureToolWrapper`.
 */
export interface SecureLLMWrapper {
  readonly stepCounter: number;
  oeUrl: string;
  executionId: string;
  durableMemory: DurableMemoryState | null;
  operationalSteps: {
    next(): number;
    observeAtLeast(n: number): void;
    current(): number;
  };
  nextOperationalStep(): number;
  observeOperationalStep(n: number): void;
}

/** Callable that returns the active wrapper, or `null` if no execution context. */
export type GetWrapperFn = () => SecureLLMWrapper | null;

// ---------------------------------------------------------------------------
// Internal helper — repair missing ToolMessages
// ---------------------------------------------------------------------------

/**
 * Insert missing `ToolMessage`s after assistant tool calls before LLM invocation.
 *
 * A tool node failure can leave a checkpoint with an assistant `tool_calls`
 * message but no matching `ToolMessage`. Providers reject that history on the
 * next LLM call, so repair the prompt sent to OE while preserving the
 * original checkpoint state.
 *
 * @internal Exported for unit testing only — not part of the public API.
 */
export function repairMissingToolMessages(
  messages: readonly BaseMessage[],
): BaseMessage[] {
  const repaired: BaseMessage[] = [];
  let inserted = 0;
  let index = 0;

  while (index < messages.length) {
    const message = messages[index] as BaseMessage;
    repaired.push(message);
    index += 1;

    const isAiMessage =
      message instanceof AIMessage ||
      message instanceof AIMessageChunk ||
      message.type === "ai";
    const toolCalls = (message as AIMessage).tool_calls;
    if (!isAiMessage || !toolCalls || toolCalls.length === 0) {
      continue;
    }

    const expected: Array<[string, string | undefined]> = [];
    for (const toolCall of toolCalls) {
      if (toolCall === null || typeof toolCall !== "object") continue;
      const tc = toolCall as Record<string, unknown>;
      const toolCallId = tc["id"];
      if (typeof toolCallId !== "string" || toolCallId === "") continue;
      const toolName = tc["name"];
      expected.push([
        toolCallId,
        typeof toolName === "string" ? toolName : undefined,
      ]);
    }

    if (expected.length === 0) continue;

    const seen = new Set<string>();
    while (index < messages.length && messages[index] instanceof ToolMessage) {
      const toolMessage = messages[index] as ToolMessage;
      if (toolMessage.tool_call_id) seen.add(toolMessage.tool_call_id);
      repaired.push(toolMessage);
      index += 1;
    }

    for (const [toolCallId, toolName] of expected) {
      if (seen.has(toolCallId)) continue;
      const displayName = toolName ?? "the requested tool";
      repaired.push(
        new ToolMessage({
          content: MISSING_TOOL_RESULT_TEMPLATE(displayName),
          tool_call_id: toolCallId,
          ...(toolName !== undefined && { name: toolName }),
          status: "error",
        }),
      );
      inserted += 1;
    }
  }

  if (inserted > 0) {
    logger.warn(
      `Inserted ${inserted} missing tool message(s) before LLM invocation`,
    );
    return repaired;
  }
  // No repair needed — return the input as-is (matches Python's `return messages`).
  // Callers only read the result (passed straight to lcMessagesToPlatform, which
  // maps into a fresh array), so a defensive copy here is redundant.
  return messages as BaseMessage[];
}

// ---------------------------------------------------------------------------
// SecureWrappedLLM
// ---------------------------------------------------------------------------

/**
 * LangChain LLM wrapper that routes all calls through OE.
 *
 * Wraps a real LLM (Gemini, OpenAI, etc.) and delegates OE orchestration to
 * `SecureLLMProxy`. The adapter handles:
 *  - LangChain ↔ sdk-core message conversion
 *  - Translating OE-owned invoke_llm results back into LangChain types
 *  - Logging and metrics (LangChain-specific concerns)
 *
 * The `SecureToolWrapper` is obtained via a callable at invocation time,
 * allowing the graph to be built before execution context exists.
 */
export class SecureWrappedLLM extends _BaseChatModel {
  modelName: string;
  private readonly llm: BaseChatModel;
  private readonly getWrapper: GetWrapperFn;
  private boundTools: ReadonlyArray<unknown>;
  private boundToolChoice: unknown;
  private readonly llmId: string;

  constructor(
    llm: BaseChatModel,
    getWrapper: GetWrapperFn,
    llmId: string = "__default__",
    modelName?: string,
  ) {
    super({});
    this.llm = llm;
    this.getWrapper = getWrapper;
    this.boundTools = [];
    this.boundToolChoice = null;
    this.llmId = llmId;

    const llmAny = llm as unknown as {
      model?: unknown;
      modelName?: unknown;
      temperature?: unknown;
    };
    this.modelName =
      modelName ??
      (typeof llmAny.model === "string" ? llmAny.model : undefined) ??
      (typeof llmAny.modelName === "string" ? llmAny.modelName : undefined) ??
      "unknown";
  }

  override _llmType(): string {
    return "secure-wrapped-llm";
  }

  override _identifyingParams(): Record<string, unknown> {
    return { model_name: this.modelName, llm_id: this.llmId };
  }

  /**
   * Bind tools to the LLM — returns a new `SecureWrappedLLM` instance.
   *
   * LangChain JS tools (`tool()`, `DynamicTool`, `DynamicStructuredTool`,
   * `RunnableToolLike`) carry their input schema as a Zod object on
   * `.schema`, not as a JSON Schema dict. We normalize them here via
   * `convertToOpenAITool` so the dict that reaches `SecureLLMProxy` (and
   * crosses the wire to OE / the tool pod) is the canonical OpenAI shape
   * `{type:"function", function:{name, description, parameters}}`. Without
   * this step, the agent-engine-runner-shared validator strips the Zod schema and
   * the eventual provider call fails with `tools[0].type` missing.
   *
   * The inner LLM still receives the original tools so its own bindTools
   * pipeline is unchanged.
   *
   * `tool_choice` (e.g. a forced function name) is captured and forwarded to
   * the tool pod's bind_tools call: invocations route through SecureLLMProxy
   * and never touch the inner LangChain binding, so a choice left only on the
   * inner LLM would otherwise be lost.
   */
  override bindTools(
    tools: ReadonlyArray<unknown>,
    kwargs?: Record<string, unknown>,
  ): SecureWrappedLLM {
    const inner = this.llm as unknown as {
      bindTools?: (
        t: ReadonlyArray<unknown>,
        k?: Record<string, unknown>,
      ) => BaseChatModel;
    };
    const newInnerLlm =
      typeof inner.bindTools === "function"
        ? inner.bindTools(tools, kwargs)
        : this.llm;
    const next = new SecureWrappedLLM(
      newInnerLlm,
      this.getWrapper,
      this.llmId,
      this.modelName,
    );
    next.boundTools = tools.map((t) => normalizeBoundTool(t));
    // `?? null` normalizes an absent choice to null while preserving
    // tool_choice=false (which disables forced tool use).
    next.boundToolChoice = kwargs?.["tool_choice"] ?? null;
    return next;
  }

  private getWrapperOrRaise(): SecureLLMWrapper {
    const wrapper = this.getWrapper();
    if (wrapper === null) {
      throw new Error(
        "SecureToolWrapper is not initialized. This means the LLM is being invoked " +
          "outside of an execution context. Ensure the graph is invoked via AER's /execute endpoint.",
      );
    }
    return wrapper;
  }

  private static buildInvocationOptions(
    kwargs: Record<string, unknown>,
  ): LLMInvocationOptions | undefined {
    if (Object.keys(kwargs).length === 0) return undefined;
    return new LLMInvocationOptions(
      kwargs as ConstructorParameters<typeof LLMInvocationOptions>[0],
    );
  }

  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const wrapper = this.getWrapperOrRaise();
    const tracer = getTracer("agent-engine-sdk-langgraph.secure_llm");

    const { step, platformMessages, stop, invocationOptions, proxy } =
      tracer.startActiveSpan(
        MODEL_REQUEST_PREPARE,
        {
          attributes: {
            [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
          },
        },
        (span) => {
          try {
            const step = wrapper.nextOperationalStep();
            stampDeepAgentMessageIds(messages);
            const repairedMessages = repairMissingToolMessages(messages);

            logSeparator();
            logLLMMessages(repairedMessages, "LLM");

            const platformMessages = lcMessagesToPlatform(repairedMessages);
            const stop = extractStop(options);
            const kwargs = extractKwargs(options);
            const invocationOptions =
              SecureWrappedLLM.buildInvocationOptions(kwargs);

            const proxy = new SecureLLMProxy({
              oeUrl: wrapper.oeUrl,
              executionId: wrapper.executionId,
              modelName: this.modelName,
              llmId: this.llmId,
              boundTools:
                this.boundTools.length > 0 ? [...this.boundTools] : null,
              boundToolChoice: this.boundToolChoice,
              operationalSteps: wrapper.operationalSteps,
              durableMemory: wrapper.durableMemory,
            });

            return { step, platformMessages, stop, invocationOptions, proxy };
          } finally {
            span.end();
          }
        },
      );

    // proxy.stream() (the provider round trip via OE) is left
    // uninstrumented: LangChain's auto-instrumentation already wraps this
    // whole method in its own LLM-kind span, so the gap between
    // request.prepare and response.process is the provider call.
    let response: unknown;
    try {
      const chunks: Parameters<
        typeof SecureLLMProxy.responseFromStreamChunks
      >[0] = [];
      for await (const chunk of proxy.stream(
        [...platformMessages],
        step,
        stop !== undefined ? [...stop] : null,
        invocationOptions ?? null,
      )) {
        chunks.push(chunk);
      }
      response = SecureLLMProxy.responseFromStreamChunks(chunks);
    } catch (exc: unknown) {
      if (exc instanceof PolicyDeniedException) {
        logPolicyBlocked("invoke_llm", step, exc.message, "LLM");
        throw exc;
      }
      if (exc instanceof LLMInvocationError) {
        Metrics.recordLatency("llm_call", proxy.lastDurationMs, {
          model_name: this.modelName,
        });
        Metrics.recordError("llm_call", { model_name: this.modelName });
        throw exc;
      }
      throw exc;
    }

    return tracer.startActiveSpan(
      MODEL_RESPONSE_PROCESS,
      {
        attributes: { [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN },
      },
      (span) => {
        try {
          if (proxy.lastLatestStepNumber !== null) {
            wrapper.observeOperationalStep(proxy.lastLatestStepNumber);
          }

          const proxyAny = proxy as unknown as { lastFromCache?: boolean };
          if (proxyAny.lastFromCache === true) {
            logCachedResult("invoke_llm", step, "LLM");
          }

          const result = llmResponseToChatResult(
            response as Parameters<typeof llmResponseToChatResult>[0],
          );

          const aiMessage = result.generations[0]?.message;
          logLLMResponse(aiMessage, step, "LLM");
          Metrics.recordLatency("llm_call", proxy.lastDurationMs, {
            model_name: this.modelName,
          });

          return result;
        } finally {
          span.end();
        }
      },
    );
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const wrapper = this.getWrapperOrRaise();

    // Check if underlying LLM supports streaming. If not, delegate to _generate.
    const innerLlm = this.llm as unknown as {
      _streamResponseChunks?: unknown;
      stream?: unknown;
    };
    const hasStream =
      typeof innerLlm._streamResponseChunks === "function" ||
      typeof innerLlm.stream === "function";
    if (!hasStream) {
      const result = await this._generate(messages, options, runManager);
      const msg = result.generations[0]?.message;
      if (msg !== undefined) {
        const msgData = msg.toDict() as unknown as Record<string, unknown>;
        msgData["type"] = "AIMessageChunk";
        const chunk = new AIMessageChunk(
          msgData as ConstructorParameters<typeof AIMessageChunk>[0],
        );
        yield new ChatGenerationChunk({
          message: chunk,
          text: typeof msg.content === "string" ? msg.content : "",
        });
      }
      return;
    }

    const tracer = getTracer("agent-engine-sdk-langgraph.secure_llm");

    const { step, platformMessages, stop, invocationOptions, proxy } =
      tracer.startActiveSpan(
        MODEL_REQUEST_PREPARE,
        {
          attributes: {
            [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
          },
        },
        (span) => {
          try {
            const step = wrapper.nextOperationalStep();
            stampDeepAgentMessageIds(messages);
            const repairedMessages = repairMissingToolMessages(messages);

            logSeparator();
            logLLMMessages(repairedMessages, "LLM");

            const platformMessages = lcMessagesToPlatform(repairedMessages);
            const stop = extractStop(options);
            const kwargs = extractKwargs(options);
            const invocationOptions =
              SecureWrappedLLM.buildInvocationOptions(kwargs);

            const proxy = new SecureLLMProxy({
              oeUrl: wrapper.oeUrl,
              executionId: wrapper.executionId,
              modelName: this.modelName,
              llmId: this.llmId,
              boundTools:
                this.boundTools.length > 0 ? [...this.boundTools] : null,
              boundToolChoice: this.boundToolChoice,
              operationalSteps: wrapper.operationalSteps,
              durableMemory: wrapper.durableMemory,
            });

            return { step, platformMessages, stop, invocationOptions, proxy };
          } finally {
            span.end();
          }
        },
      );

    // Gap between request.prepare and response.process is the provider
    // round trip, same reasoning as _generate above.
    //
    // Spans the whole consumption loop, not just the first chunk: a
    // streaming response isn't "processed" until the stream ends. Opened
    // with startSpan (not startActiveSpan) because it must stay open across
    // the yield points below.
    const responseProcessSpan = tracer.startSpan(MODEL_RESPONSE_PROCESS, {
      attributes: { [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN },
    });
    try {
      for await (const sdkChunk of proxy.stream(
        [...platformMessages],
        step,
        stop !== undefined ? [...stop] : null,
        invocationOptions ?? null,
      )) {
        yield llmStreamChunkToGenerationChunk(
          sdkChunk as Parameters<typeof llmStreamChunkToGenerationChunk>[0],
        );
      }
    } catch (exc: unknown) {
      if (exc instanceof PolicyDeniedException) {
        logPolicyBlocked("invoke_llm", step, exc.message, "LLM");
        throw exc;
      }
      if (exc instanceof LLMInvocationError) {
        Metrics.recordLatency("llm_call", proxy.lastDurationMs, {
          model_name: this.modelName,
        });
        Metrics.recordError("llm_call", { model_name: this.modelName });
        throw exc;
      }
      throw exc;
    } finally {
      responseProcessSpan.end();
    }

    if (proxy.lastLatestStepNumber !== null) {
      wrapper.observeOperationalStep(proxy.lastLatestStepNumber);
    }

    const proxyAny = proxy as unknown as { lastFromCache?: boolean };
    if (proxyAny.lastFromCache === true) {
      logCachedResult("invoke_llm", step, "LLM");
    }

    Metrics.recordLatency("llm_call", proxy.lastDurationMs, {
      model_name: this.modelName,
    });
  }
}

// ---------------------------------------------------------------------------
// Local helpers — extract Python-style `stop` and `**kwargs` from LC.js options
// ---------------------------------------------------------------------------

function extractStop(
  options: Partial<BaseChatModelCallOptions> | undefined,
): readonly string[] | undefined {
  if (options === undefined) return undefined;
  const stop = (options as Record<string, unknown>)["stop"];
  return Array.isArray(stop) ? (stop as readonly string[]) : undefined;
}

function extractKwargs(
  options: Partial<BaseChatModelCallOptions> | undefined,
): Record<string, unknown> {
  if (options === undefined) return {};
  const known = new Set([
    "stop",
    "signal",
    "timeout",
    "maxConcurrency",
    "tags",
    "metadata",
    "callbacks",
    "runName",
    "configurable",
    // LangGraph runtime state injected into task configs. It is per-attempt
    // bookkeeping (checkpoint/task UUIDs, nodeFirstAttemptTime), never a
    // provider option; letting it reach the OE semantic input makes every
    // durable replay re-admit the LLM activity with a different input hash.
    "control",
    "durability",
    "executionInfo",
  ]);
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(
    options as Record<string, unknown>,
  )) {
    if (!known.has(key) && value !== undefined) result[key] = value;
  }
  return result;
}
