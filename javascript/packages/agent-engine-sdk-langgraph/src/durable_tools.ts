/** Durable replay identity for platform-executed LangGraph tools. */

import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";

import { ToolMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  entrypoint,
  getConfig,
  isGraphBubbleUp,
  task,
} from "@langchain/langgraph";
import {
  ToolExecutionError,
  ToolPlatformError,
  currentAttemptContext,
  decodeToolFailure,
  encodeToolFailure,
  currentOperationBoundaries,
  getCurrentWrapper,
  currentOperationPath,
  currentStepOrdinal,
  inSpawnedTask,
  runWithOperationPathResolver,
  type ChildOperationBoundary,
  type SecureToolWrapper,
  type createSecureToolFunction,
} from "@mongodb-js/agent-engine-runner-shared";

import { toToolMessageContent } from "./messages.js";

type SecureToolFunction = ReturnType<typeof createSecureToolFunction>;

function durableToolResultMessageId(toolCallId: string): string | null {
  const attempt = currentAttemptContext();
  if (attempt === null) return null;

  const { executionId } = attempt.workflowIdentity as { executionId: string };
  const messageId =
    `durable-tool-result:${executionId}:` +
    `${currentStepOrdinal()}:${toolCallId}`;
  if (!inSpawnedTask()) return messageId;
  // Send workers in one step may see the same ToolCall ID, so their results
  // also carry the worker's path. Other IDs keep their recorded form.
  const path = JSON.stringify(
    currentOperationPath().segments.map((segment) => [
      segment.name,
      Number(segment.ordinal),
    ]),
  );
  return `${messageId}:${createHash("sha256").update(path).digest("hex")}`;
}

/**
 * Give a replayed platform tool result the same LangGraph message ID as its
 * first attempt, preventing the messages reducer from appending it twice.
 * The model-provided tool-call ID remains unchanged and continues to correlate
 * the ToolMessage with the AIMessage that requested it.
 */
export function withDurableToolResultIdentity(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  return async (kwargs, config) => {
    const result = await wrapped(kwargs, config);
    const toolCallId =
      config?.toolCall?.id ?? config?.configurable?.["tool_call_id"];
    if (typeof toolCallId !== "string" || !toolCallId) return result;

    const messageId = durableToolResultMessageId(toolCallId);
    if (messageId === null) return result;

    const [content, artifact] = result as [unknown, unknown];
    const message = new ToolMessage({
      content: toToolMessageContent(content),
      artifact,
      tool_call_id: toolCallId,
      name: toolName,
      id: messageId,
    });
    // content_and_artifact unwraps this tuple, then returns the ToolMessage
    // unchanged because the artifact is already attached to the message.
    return [message, null];
  };
}

interface OAuthTaskCall {
  readonly kwargs: Record<string, unknown>;
  readonly toolCallId: string;
  readonly boundaries: readonly ChildOperationBoundary[] | undefined;
}

interface OAuthTaskResult {
  readonly kwargs: Record<string, unknown>;
  readonly toolCallId: string;
  readonly result?: unknown;
  /**
   * The call's own failure, which the caller rethrows for ToolNode. A
   * checkpoint keeps it as an `Error` with its message.
   */
  readonly error?: unknown;
}

function inGraphTask(): boolean {
  const configurable = getConfig()?.configurable;
  return typeof configurable?.["__pregel_call"] === "function";
}

/**
 * Run each call of an OAuth tool in its own LangGraph task during a durable
 * attempt.
 *
 * The call's consent wait pauses with `interrupt()`. LangGraph numbers the
 * pauses of one task in the order they happen, and the calls a ToolNode runs
 * in parallel share its task, so their waits would take each other's slots.
 * A task per call gives each wait its own.
 *
 * The task's activities keep the operation position of the graph node that
 * called the tool. The task's cached result is bound to the call it ran for:
 * when parallel calls reach their tasks in another order on a rerun, a call
 * that receives another call's cached slot runs again outside the task, and
 * OE replays its recorded activities.
 *
 * The call's own failure is the task's result, rethrown to the caller: a
 * failed task would fail the graph, while ToolNode reports a tool's failure
 * to the model. Only LangGraph's pauses leave the task as errors.
 */
function withDurableOAuthTask(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  const dispatch = task(
    `delegated_oauth_${toolName}`,
    async (call: OAuthTaskCall): Promise<OAuthTaskResult> => {
      const run = () =>
        wrapped(call.kwargs, { toolCall: { id: call.toolCallId } });
      const boundaries = call.boundaries;
      const identity = { kwargs: call.kwargs, toolCallId: call.toolCallId };
      try {
        const result =
          boundaries === undefined
            ? await run()
            : await runWithOperationPathResolver(() => boundaries, run);
        return { ...identity, result };
      } catch (error) {
        if (isGraphBubbleUp(error)) throw error;
        return { ...identity, error };
      }
    },
  );
  return async (kwargs = {}, config) => {
    if (!inGraphTask()) return wrapped(kwargs, config);
    const rawToolCallId =
      config?.toolCall?.id ?? config?.configurable?.["tool_call_id"];
    const toolCallId =
      typeof rawToolCallId === "string" && rawToolCallId ? rawToolCallId : null;
    // Without a call id, a rerun from the checkpoint gives the call a new
    // position, so OE would not match it to the consent the user gave.
    if (toolCallId === null) {
      throw new ToolExecutionError(
        `OAuth tool '${toolName}' must be invoked with a tool call during a durable run`,
      );
    }
    // Captured here because inside the task the graph path resolver sees the
    // task's namespace, which is not a graph node.
    const outcome = await dispatch({
      kwargs,
      toolCallId,
      boundaries: currentOperationBoundaries(),
    });
    if (outcome.toolCallId !== toolCallId) {
      return wrapped(kwargs, config);
    }
    if (!isDeepStrictEqual(outcome.kwargs, kwargs)) {
      throw new ToolExecutionError(
        "OAuth tool arguments changed during checkpoint replay",
      );
    }
    if ("error" in outcome) throw outcome.error;
    return outcome.result;
  };
}

const NATIVE_CALL_KEY = "__agent_engine_oauth_call";

/** The caller's current call, read by the child instead of its saved input. */
interface NativeOAuthCall {
  readonly kwargs: Record<string, unknown>;
  readonly toolCallId: string | null;
}

/**
 * What a call's child checkpoint keeps: the call it ran for and its result,
 * or the call's own failure as {@link encodeToolFailure} records it.
 */
interface NativeOAuthRecord {
  readonly toolName: string;
  readonly kwargs: Record<string, unknown>;
  readonly result?: unknown;
  readonly failure?: string;
}

/** Positions of ID-less calls, counted per run of the calling graph task. */
const nativeCallPositions = new WeakMap<object, number>();

/** Compare arguments as the checkpoint stores them, which drops `undefined`. */
function sameArguments(
  recorded: Record<string, unknown>,
  current: Record<string, unknown>,
): boolean {
  const stored = (value: unknown): unknown =>
    JSON.parse(JSON.stringify(value ?? {})) as unknown;
  return isDeepStrictEqual(stored(recorded), stored(current));
}

/** Reject a checkpoint that belongs to another tool or other arguments. */
function assertSameCall(
  recorded: NativeOAuthRecord,
  toolName: string,
  kwargs: Record<string, unknown>,
): void {
  if (recorded.toolName !== toolName) {
    throw new ToolExecutionError(
      "OAuth tool identity changed during checkpoint replay",
    );
  }
  if (!sameArguments(recorded.kwargs, kwargs)) {
    throw new ToolExecutionError(
      "OAuth tool arguments changed during checkpoint replay",
    );
  }
}

/**
 * The calling graph task's config when it runs a platform session with a
 * checkpointer. Stateless graphs and calls outside a graph have nothing to
 * resume and run directly.
 */
function nativeCheckpointConfig(): RunnableConfig | null {
  const config = getConfig();
  const configurable = config?.configurable;
  const requestContext = configurable?.["request_context"];
  if (
    config === undefined ||
    configurable?.["__pregel_checkpointer"] == null ||
    typeof configurable["__pregel_scratchpad"] !== "object" ||
    typeof requestContext !== "object" ||
    requestContext === null
  ) {
    return null;
  }
  return config;
}

/**
 * The parent's scratchpad, except that this child adds no automatic subgraph
 * ordinal to its namespace. The ordinal follows the order in which calls
 * reach their children, which parallel calls do not keep across runs.
 */
function withoutSubgraphOrdinal(scratchpad: object): object {
  return new Proxy(scratchpad, {
    get: (target, key, receiver) =>
      key === "subgraphCounter" ? 0 : Reflect.get(target, key, receiver),
    set: (target, key, value, receiver) =>
      key === "subgraphCounter" || Reflect.set(target, key, value, receiver),
  });
}

/** Errors the tool call itself threw inside its child checkpoint. */
const nativeToolErrors = new WeakSet<object>();

/**
 * The error to throw for a failure around a call's child checkpoint.
 *
 * The call's own errors, pauses and replay-identity mismatches are thrown as
 * they are. Anything else — the saver failing to read or write the child,
 * possibly after the provider answered — ends the turn as a platform failure:
 * ToolNode would hand it to the model as a tool error, and the model could
 * then repeat a call that already reached the provider.
 */
function failNativeCheckpoint(error: unknown): unknown {
  if (
    isGraphBubbleUp(error) ||
    (typeof error === "object" &&
      error !== null &&
      nativeToolErrors.has(error)) ||
    (error instanceof ToolExecutionError &&
      !(error instanceof ToolPlatformError))
  ) {
    return error;
  }
  const failure =
    error instanceof ToolPlatformError
      ? error
      : new ToolPlatformError(
          `OAuth tool checkpoint failed: ${error instanceof Error ? error.message : String(error)}`,
          { cause: error },
        );
  (getCurrentWrapper() as SecureToolWrapper | null)?.recordPlatformFailure(
    failure,
  );
  return failure;
}

/**
 * Run each call of an OAuth tool on a native checkpoint execution in its own
 * child checkpoint.
 *
 * The child sits under the calling graph task's namespace, at a slot named
 * by the tool-call ID, or by the call's position among the task's ID-less
 * OAuth calls. It records the tool and arguments before the call can pause
 * or reach the provider, then the call's result or its own failure. Its consent wait is its own
 * interrupt, so the answer to one call's wait cannot wake another call or a
 * later wait in the same node. When the graph task runs again, including in
 * a new process over the same checkpointer, a completed child returns its
 * stored result or failure without calling OE or the provider, and a slot recorded for
 * another tool or other arguments fails before either is called.
 */
function withNativeOAuthCheckpoint(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  const identity = task(
    "delegated_oauth_identity",
    async (name: string, kwargs: Record<string, unknown>) => ({
      toolName: name,
      kwargs,
    }),
  );
  // Every tool's child has the same name, so its identity task resumes from
  // the slot's record even when another tool now holds the slot.
  const child = entrypoint(
    "delegated_oauth",
    async (): Promise<NativeOAuthRecord> => {
      // A resumed child re-reads its saved input; the caller's current call
      // travels in the config so a changed call is still detected.
      const call = getConfig()?.configurable?.[NATIVE_CALL_KEY] as
        | NativeOAuthCall
        | undefined;
      if (call === undefined) {
        throw new ToolPlatformError("OAuth checkpoint call is missing");
      }
      assertSameCall(
        await identity(toolName, call.kwargs),
        toolName,
        call.kwargs,
      );
      const record = { toolName, kwargs: call.kwargs };
      try {
        const result = await wrapped(
          call.kwargs,
          call.toolCallId === null
            ? undefined
            : { toolCall: { id: call.toolCallId } },
        );
        return { ...record, result };
      } catch (error) {
        if (typeof error === "object" && error !== null) {
          nativeToolErrors.add(error);
        }
        // The call's own failure completes the child, so a rerun of the node
        // reports it again instead of dispatching the call once more. A
        // platform failure has already ended the turn and stays unrecorded,
        // as do pauses and policy denials.
        if (
          error instanceof ToolExecutionError &&
          !(error instanceof ToolPlatformError)
        ) {
          return { ...record, failure: encodeToolFailure(error) };
        }
        throw error;
      }
    },
  );
  return async (kwargs = {}, config) => {
    const parent = nativeCheckpointConfig();
    if (parent === null) return wrapped(kwargs, config);
    const rawToolCallId =
      config?.toolCall?.id ?? config?.configurable?.["tool_call_id"];
    const toolCallId =
      typeof rawToolCallId === "string" && rawToolCallId ? rawToolCallId : null;
    const {
      checkpoint_id: _checkpointId,
      __pregel_scratchpad: scratchpad,
      ...configurable
    } = parent.configurable as Record<string, unknown> & {
      __pregel_scratchpad: object;
    };
    let slot: string;
    if (toolCallId !== null) {
      // ToolNode calls reach their children in any order; the model's call
      // ID does not change. Hashing keeps the namespace delimiter-safe.
      slot = `call-${createHash("sha256").update(toolCallId).digest("hex")}`;
    } else {
      const position = nativeCallPositions.get(scratchpad) ?? 0;
      nativeCallPositions.set(scratchpad, position + 1);
      slot = `position-${position}`;
    }
    const parentNamespace = configurable["checkpoint_ns"];
    const checkpointNamespace = `${typeof parentNamespace === "string" ? parentNamespace : ""}|delegated_oauth:${slot}`;
    const call: NativeOAuthCall = { kwargs, toolCallId };
    const childConfig: RunnableConfig = {
      ...parent,
      configurable: {
        ...configurable,
        checkpoint_ns: checkpointNamespace,
        __pregel_scratchpad: withoutSubgraphOrdinal(scratchpad),
        [NATIVE_CALL_KEY]: call,
      },
    };
    const saver = configurable["__pregel_checkpointer"] as {
      getTuple(config: RunnableConfig): Promise<unknown>;
    };
    let outcome: NativeOAuthRecord;
    try {
      const saved = await saver.getTuple({
        configurable: {
          thread_id: configurable["thread_id"],
          checkpoint_ns: checkpointNamespace,
        },
      });
      // Null input continues a saved child from its checkpoint; a new child
      // needs its input to start.
      outcome = (await child.invoke(
        saved == null ? { kwargs } : null,
        childConfig,
      )) as NativeOAuthRecord;
      assertSameCall(outcome, toolName, kwargs);
    } catch (error) {
      throw failNativeCheckpoint(error);
    }
    if (outcome.failure !== undefined) {
      throw decodeToolFailure(outcome.failure);
    }
    return outcome.result;
  };
}

/**
 * Give each call of an OAuth tool its own LangGraph boundary, so a consent
 * wait pauses only its own call and a continued run neither repeats nor
 * exchanges a completed call: a task per call during a durable attempt, a
 * child checkpoint per call on a native checkpoint execution. Other calls run
 * directly.
 */
export function withOAuthTask(
  wrapped: SecureToolFunction,
  toolName: string,
): SecureToolFunction {
  const durable = withDurableOAuthTask(wrapped, toolName);
  const native = withNativeOAuthCheckpoint(wrapped, toolName);
  return async (kwargs = {}, config) =>
    currentAttemptContext() === null
      ? native(kwargs, config)
      : durable(kwargs, config);
}
