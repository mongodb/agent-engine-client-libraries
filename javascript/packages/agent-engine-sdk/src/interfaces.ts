/** Framework-neutral protocols for the Atlas Agent Engine SDK. */

import type {
  AgentInput,
  AgentOutput,
  LLMResponse,
  LLMStreamChunk,
  Message,
  RequestContext,
  StreamEvent,
} from "./models.js";

/**
 * Result of an agent execution.
 *
 * Await for the final result, or async-iterate for streaming events.
 *
 * ```ts
 * // Non-streaming
 * const output = await agent.execute(ctx, input)
 *
 * // Streaming
 * for await (const event of agent.execute(ctx, input)) {
 *   handle(event)
 * }
 * ```
 */
export interface ExecutionResult
  extends PromiseLike<AgentOutput>, AsyncIterable<StreamEvent> {}

/**
 * The contract between agent code and runtimes.
 *
 * Agents expose a single `execute()` method that returns an `ExecutionResult`.
 * Callers choose the execution mode:
 * - `await agent.execute(ctx, input)` for a final result (JSON-style)
 * - `for await (const event of agent.execute(ctx, input))` for streaming (SSE-style)
 */
export interface BaseAgent {
  execute(ctx: RequestContext, input: AgentInput): ExecutionResult;
}

/**
 * Framework-neutral LLM protocol.
 *
 * Used at runtime by the ToolPod. In Python, this protocol exposes both `invoke`
 * (sync) and `ainvoke` (async) since Python supports both call styles. JS has no
 * sync HTTP, so this interface has a single async `invoke` plus `stream`.
 */
export interface BaseLLM {
  invoke(
    messages: Message[],
    kwargs?: Record<string, unknown>,
  ): Promise<LLMResponse>;
  stream(
    messages: Message[],
    kwargs?: Record<string, unknown>,
  ): AsyncIterable<LLMStreamChunk>;
}

/**
 * Framework-neutral callback for observability during agent execution.
 *
 * Replaces LangChain's `BaseCallbackHandler`. The AER injects an
 * implementation that forwards node events to the OE for logging.
 * Framework adapters (e.g. sdk-langgraph) bridge from the framework's
 * native callback system to this interface.
 *
 * Prefer subclassing `NullExecutionCallback` over implementing this
 * interface directly — it provides no-op defaults for all methods including
 * optional extensions (e.g. `onNodeSuspend`) so new methods never break
 * existing implementations.
 */
export interface BaseExecutionCallback {
  onNodeStart(
    nodeName: string,
    inputs: Record<string, unknown>,
    opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void;

  onNodeEnd(
    nodeName: string,
    outputs: Record<string, unknown>,
    opts: {
      runId: string;
      parentRunId?: string;
      durationMs?: number;
      metadata?: Record<string, unknown>;
    },
  ): void;

  onNodeError(
    nodeName: string,
    error: string,
    opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void;
}

/**
 * Concrete base class with no-op defaults for all `BaseExecutionCallback` methods.
 *
 * Subclass this instead of implementing `BaseExecutionCallback` directly so that
 * new methods added to the interface are automatically satisfied — you only
 * override the events you care about.
 *
 * ```ts
 * class MyCallback extends NullExecutionCallback {
 *   onNodeStart(nodeName: string, _inputs, _opts) {
 *     console.log(`starting ${nodeName}`)
 *   }
 * }
 * ```
 */
export class NullExecutionCallback implements BaseExecutionCallback {
  onNodeStart(
    _nodeName: string,
    _inputs: Record<string, unknown>,
    _opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void {}

  onNodeEnd(
    _nodeName: string,
    _outputs: Record<string, unknown>,
    _opts: {
      runId: string;
      parentRunId?: string;
      durationMs?: number;
      metadata?: Record<string, unknown>;
    },
  ): void {}

  onNodeError(
    _nodeName: string,
    _error: string,
    _opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void {}

  onNodeSuspend(
    _nodeName: string,
    _opts: {
      runId: string;
      parentRunId?: string;
      metadata?: Record<string, unknown>;
    },
  ): void {}
}
