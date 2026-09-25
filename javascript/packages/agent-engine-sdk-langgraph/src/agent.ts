/**
 * LangGraph BaseAgent adapter.
 *
 * Wraps a LangGraph compiled graph to implement the framework-neutral
 * `BaseAgent` protocol from agent-engine-sdk. Execution-session setup lives in
 * `execution_session.ts`; this adapter owns graph invocation and event/result
 * translation.
 *
 * Port of `agent_engine_sdk_langgraph/agent.py`.
 */

import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import { Command, type Interrupt } from "@langchain/langgraph";
import type { CompiledStateGraph } from "@langchain/langgraph";
import { isDeepStrictEqual } from "node:util";

import {
  type AgentInput,
  type AgentOutput,
  type BaseAgent,
  type ExecutionResult,
  type RequestContext,
  type StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";
import {
  type ExecutionSession,
  type PrepareAgentInput,
  type ResolveThreadId,
} from "./execution_session.js";
import { installDurableMessageIdentity } from "./durable_message_identity.js";
import { DurableSubgraphResolver } from "./durable_subgraphs.js";
import { iterTaskCalls } from "./deep_agent_task.js";
import type { DeepAgentTaskCall } from "./deep_agent_task.js";
import { lcMessagesToPlatform, unwrapMessageValues } from "./messages.js";
import { wrapSessionForkUpdateState } from "./session_fork.js";
import { executionSession } from "./session_factory.js";

const logger = getLogger("agent_engine_sdk_langgraph.agent");

/**
 * Tenant hook registered via `App.prepareAgentInput`. Receives the
 * framework-neutral `AgentInput` (the opaque caller payload) and the
 * `RequestContext`, and returns a value LangGraph can ingest directly — a
 * graph-state object or a `Command`. Lets an agent author control how a fresh
 * invocation becomes the graph's starting input. Resume stays platform-managed.
 *
 * Port of Python's `PrepareAgentInput` (agent_engine_sdk_langgraph/agent.py).
 */
export type { PrepareAgentInput } from "./execution_session.js";

/**
 * Tenant hook registered via `App.resolveThreadId`. Maps `RequestContext` to
 * the LangGraph checkpoint `thread_id`. When set, the return value is used
 * verbatim (no workspace suffix). Port of Python's `ResolveThreadId`.
 */
export type { ResolveThreadId } from "./execution_session.js";

// LangGraph namespace segment shape: ``"<node_name>:<task_id>"``.
const NS_TASK_SEP = ":";
// LangChain/LangGraph pregel node names; never identify a real subagent.
const NS_SENTINEL_NAMES = new Set(["agent", "model", "tools"]);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Extract subagent attribution from a LangGraph subgraph namespace tuple. */
function sourceFromNamespace(namespace: readonly string[]): string {
  for (let i = namespace.length - 1; i >= 0; i--) {
    const segment = namespace[i] as string;
    const name = segment.split(NS_TASK_SEP, 1)[0] ?? "";
    if (name !== "" && !NS_SENTINEL_NAMES.has(name)) return name;
  }
  return "";
}

interface SubagentTask {
  name: string;
  tcId: string;
  description: string;
}

function startEvent(task: SubagentTask): StreamEvent {
  return {
    data: {
      source: task.name,
      subagent_name: task.name,
      tool_call_id: task.tcId,
      description: task.description,
    },
    event: "subagent_start",
  };
}

function endEvent(task: SubagentTask, summary: string = ""): StreamEvent {
  return {
    data: {
      source: task.name,
      subagent_name: task.name,
      tool_call_id: task.tcId,
      summary,
    },
    event: "subagent_end",
  };
}

function tcIdForSource(
  source: string,
  tasksById: Map<string, SubagentTask>,
): string {
  if (source === "") return "";
  for (const task of tasksById.values()) {
    if (task.name === source) return task.tcId;
  }
  return "";
}

async function* onTaskCall(
  call: DeepAgentTaskCall,
  tasksById: Map<string, SubagentTask>,
  pending: Map<string, SubagentTask>,
): AsyncIterable<StreamEvent> {
  const placeholder = pending.get(call.subagentName);
  if (placeholder !== undefined) {
    pending.delete(call.subagentName);
    placeholder.tcId = call.toolCallId;
    if (call.description !== "" && placeholder.description === "") {
      placeholder.description = call.description;
    }
    tasksById.set(call.toolCallId, placeholder);
    yield startEvent(placeholder);
    return;
  }

  const existing = tasksById.get(call.toolCallId);
  if (existing !== undefined) {
    if (call.description !== "" && existing.description === "") {
      existing.description = call.description;
    }
    return;
  }

  const task: SubagentTask = {
    name: call.subagentName,
    tcId: call.toolCallId,
    description: call.description,
  };
  tasksById.set(call.toolCallId, task);
  yield startEvent(task);
}

function onSourcedToken(
  source: string,
  tasksById: Map<string, SubagentTask>,
  pending: Map<string, SubagentTask>,
): void {
  // First sourced token signal — insert a synthetic placeholder in `pending`
  // if no entry exists. Strategy A: do NOT emit subagent_start here.
  for (const task of tasksById.values()) {
    if (task.name === source) return;
  }
  if (pending.has(source)) return;
  pending.set(source, { name: source, tcId: "", description: "" });
}

async function* drainOpenTasks(
  tasksById: Map<string, SubagentTask>,
  pending: Map<string, SubagentTask>,
): AsyncIterable<StreamEvent> {
  for (const task of [...tasksById.values()]) {
    yield endEvent(task);
  }
  for (const task of [...pending.values()]) {
    yield startEvent(task);
    yield endEvent(task);
  }
  tasksById.clear();
  pending.clear();
}

function normalizeContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (block !== null && typeof block === "object") {
        const b = block as Record<string, unknown>;
        if (b["type"] === "text" && typeof b["text"] === "string")
          parts.push(b["text"]);
      } else if (typeof block === "string") {
        parts.push(block);
      }
    }
    return parts.join("");
  }
  if (content === null || content === undefined) return "";
  return String(content);
}

function findLastAiContent(messages: readonly BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (
      msg &&
      (msg instanceof AIMessage ||
        msg instanceof AIMessageChunk ||
        msg.type === "ai")
    ) {
      const content = normalizeContent(msg.content);
      if (content !== "") return content;
    }
  }
  return "";
}

/**
 * Detect cancellation / abort errors. The TS equivalents of Python's
 * `asyncio.CancelledError` / `GeneratorExit` are `AbortError` (DOMException)
 * and any error whose name signals cancellation by upstream cancellation
 * sources.
 */
function isCancellationError(exc: unknown): boolean {
  if (exc instanceof Error) {
    return exc.name === "AbortError" || exc.name === "CancelledError";
  }
  return false;
}

// ---------------------------------------------------------------------------
// ExecutionResult wrapper
// ---------------------------------------------------------------------------

/**
 * Concrete `ExecutionResult` — dual-mode handle.
 *
 * Implements `PromiseLike<AgentOutput>` (so callers can `await`) and
 * `AsyncIterable<StreamEvent>` (so callers can `for await`). Mirrors the
 * Python dual-shape via `__await__` + `__aiter__`.
 */
class _ExecutionResult implements ExecutionResult {
  constructor(
    private readonly invokeFn: () => Promise<AgentOutput>,
    private readonly streamFn: () => AsyncIterable<StreamEvent>,
  ) {}

  // PromiseLike — `await execResult` resolves to the final AgentOutput.
  then<TResult1 = AgentOutput, TResult2 = never>(
    onfulfilled?:
      | ((value: AgentOutput) => TResult1 | PromiseLike<TResult1>)
      | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ): PromiseLike<TResult1 | TResult2> {
    return this.invokeFn().then(onfulfilled, onrejected);
  }

  // AsyncIterable — `for await (const e of execResult)` streams StreamEvents.
  [Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
    return this.streamFn()[Symbol.asyncIterator]();
  }
}

// ---------------------------------------------------------------------------
// LangGraphBaseAgent
// ---------------------------------------------------------------------------

/** LangGraph adapter implementing the BaseAgent protocol. */
export class LangGraphBaseAgent implements BaseAgent {
  private readonly graph: CompiledStateGraph<unknown, unknown>;
  private readonly callbacks: readonly unknown[];
  private readonly prepareInput: PrepareAgentInput | null;
  private readonly resolveThreadIdFn: ResolveThreadId | null;
  private readonly durableSubgraphs: DurableSubgraphResolver | null;

  constructor(
    graph: CompiledStateGraph<unknown, unknown>,
    callbacks?: readonly unknown[],
    prepareInput?: PrepareAgentInput | null,
    resolveThreadId?: ResolveThreadId | null,
  ) {
    this.graph = graph;
    installDurableMessageIdentity(this.graph);
    const durableSubgraphs = new DurableSubgraphResolver(this.graph);
    this.durableSubgraphs = durableSubgraphs.hasCompiledChildren
      ? durableSubgraphs
      : null;
    this.callbacks = callbacks ?? [];
    this.prepareInput = prepareInput ?? null;
    this.resolveThreadIdFn = resolveThreadId ?? null;
    // The platform maps live updateState to a new session+thread, not LangGraph
    // same-thread time-travel. See docs/session-fork.md.
    wrapSessionForkUpdateState(this.graph, {
      resolveThreadId: this.resolveThreadIdFn,
    });
  }

  /** The wrapped compiled graph (Python `compiled_graph` parity). */
  get compiledGraph(): CompiledStateGraph<unknown, unknown> {
    return this.graph;
  }

  /** The tenant thread-id hook, or null (Python `_resolve_thread_id` parity). */
  get resolveThreadId(): ResolveThreadId | null {
    return this.resolveThreadIdFn;
  }

  execute(ctx: RequestContext, input: AgentInput): ExecutionResult {
    return new _ExecutionResult(
      () => this.invoke(ctx, input),
      () => this.stream(ctx, input),
    );
  }

  private session(ctx: RequestContext, input: AgentInput): ExecutionSession {
    return executionSession(this.graph, ctx, input, {
      callbacks: this.callbacks,
      prepareInput: this.prepareInput,
      resolveThreadId: this.resolveThreadIdFn,
      durableSubgraphs: this.durableSubgraphs,
    });
  }

  async invoke(ctx: RequestContext, input: AgentInput): Promise<AgentOutput> {
    const session = this.session(ctx, input);
    try {
      return await session.runInGraphScope(() => this.invokeGraph(session));
    } finally {
      await session.close();
    }
  }

  private async invokeGraph(session: ExecutionSession): Promise<AgentOutput> {
    const { graphInput, config } = await session.prepareRun();
    let nextInput = graphInput;

    const graphAny = this.graph as unknown as {
      invoke(input: unknown, config: RunnableConfig): Promise<unknown>;
    };

    while (true) {
      const result = (await graphAny.invoke(nextInput, config)) as Record<
        string,
        unknown
      >;
      const messages = Array.isArray(result["messages"])
        ? (result["messages"] as BaseMessage[])
        : [];
      const response = findLastAiContent(messages);

      const interrupt = await session.invokeInterrupt(
        config,
        response,
        messages,
      );
      if (interrupt instanceof Command) {
        nextInput = interrupt;
        continue;
      }
      if (interrupt !== null) return interrupt;

      const metadata = await session.complete();
      return {
        response: {
          response,
          execution_id: session.threadId,
          status: "completed",
          message_count: messages.length,
          resumed: session.resumed,
          ...(metadata === undefined ? {} : { metadata }),
        },
      };
    }
  }

  async resume(
    ctx: RequestContext,
    input: AgentInput,
    decision: string,
  ): Promise<AgentOutput> {
    // Resume state lives on the context, not the caller payload.
    const resumeCtx: RequestContext = {
      ...ctx,
      resume: true,
      resumeData: decision,
    };
    return this.invoke(resumeCtx, input);
  }

  // ------------------------------------------------------------------------
  // Streaming
  // ------------------------------------------------------------------------

  private async *processMessagesChunk(
    chunk: unknown,
    source: string,
    tasksById: Map<string, SubagentTask>,
    pending: Map<string, SubagentTask>,
  ): AsyncIterable<StreamEvent> {
    if (!(chunk instanceof AIMessageChunk)) return;

    if (source === "") {
      for (const call of iterTaskCalls([chunk])) {
        if (call.subagentName === "") continue;
        for await (const ev of onTaskCall(call, tasksById, pending)) yield ev;
      }
    } else {
      onSourcedToken(source, tasksById, pending);
    }

    const token = normalizeContent(chunk.content);
    if (token !== "") {
      yield {
        data: {
          content: token,
          source,
          tool_call_id: tcIdForSource(source, tasksById),
        },
        event: "token",
      };
    }
  }

  private async *processAddMessagesUpdate(
    msgs: unknown,
    tasksById: Map<string, SubagentTask>,
    pending: Map<string, SubagentTask>,
    allMessages: BaseMessage[],
  ): AsyncIterable<StreamEvent> {
    const extracted = unwrapMessageValues(msgs);
    allMessages.push(...(extracted as BaseMessage[]));

    for (const call of iterTaskCalls(extracted)) {
      if (call.subagentName === "") continue;
      for await (const ev of onTaskCall(call, tasksById, pending)) yield ev;
    }

    for (const msg of extracted) {
      if (!(msg instanceof ToolMessage)) continue;
      const tcId = msg.tool_call_id;
      if (typeof tcId !== "string" || tcId === "") continue;
      const task = tasksById.get(tcId);
      if (task === undefined) continue;
      tasksById.delete(tcId);
      const summary = normalizeContent(msg.content ?? "");
      yield endEvent(task, summary);
    }
  }

  private async *processCommandUpdate(
    cmd: Command,
    tasksById: Map<string, SubagentTask>,
    allMessages: BaseMessage[],
  ): AsyncIterable<StreamEvent> {
    const cmdAny = cmd as unknown as { update?: unknown };
    const cmdUpdate = cmdAny.update;
    if (
      cmdUpdate === null ||
      typeof cmdUpdate !== "object" ||
      Array.isArray(cmdUpdate)
    )
      return;
    const cmdDict = cmdUpdate as Record<string, unknown>;
    if (!("messages" in cmdDict)) return;
    const extracted = unwrapMessageValues(cmdDict["messages"]);
    allMessages.push(...(extracted as BaseMessage[]));
    for (const msg of extracted) {
      const tcId = (msg as { tool_call_id?: unknown }).tool_call_id;
      if (typeof tcId !== "string" || tcId === "") continue;
      const task = tasksById.get(tcId);
      if (task === undefined) continue;
      tasksById.delete(tcId);
      const summary = normalizeContent(
        (msg as { content?: unknown }).content ?? "",
      );
      yield endEvent(task, summary);
    }
  }

  async *stream(
    ctx: RequestContext,
    input: AgentInput,
  ): AsyncIterable<StreamEvent> {
    const session = this.session(ctx, input);
    try {
      yield* session.iterateInGraphScope(this.streamGraph(session));
    } finally {
      await session.close();
    }
  }

  private async *streamGraph(
    session: ExecutionSession,
  ): AsyncIterable<StreamEvent> {
    const { graphInput, config } = await session.prepareRun();
    let nextInput = graphInput;

    const allMessages: BaseMessage[] = [];
    const tasksById = new Map<string, SubagentTask>();
    const pending = new Map<string, SubagentTask>();
    const graphAny = this.graph as unknown as {
      stream(
        input: unknown,
        config: RunnableConfig & {
          streamMode?: readonly string[];
          subgraphs?: boolean;
        },
      ): Promise<AsyncIterable<unknown>>;
    };

    while (true) {
      const capturedInterrupts: Interrupt[] = [];
      const capturedInterruptValues = new Map<string, unknown>();
      let streamError: unknown;
      try {
        const streamOpts = {
          ...config,
          streamMode: ["messages", "updates"] as const,
          subgraphs: true,
        };
        for await (const event of await graphAny.stream(
          nextInput,
          streamOpts,
        )) {
          // LangGraph.js emits `[namespace, streamMode, payload]` when subgraphs=true.
          if (!Array.isArray(event) || event.length < 3) continue;
          const namespace = Array.isArray(event[0])
            ? (event[0] as string[])
            : [];
          const streamMode = event[1] as string;
          const eventPayload = event[2];

          if (streamMode === "messages") {
            const tuple = Array.isArray(eventPayload)
              ? eventPayload
              : [eventPayload];
            const chunk = tuple[0];
            const source = sourceFromNamespace(namespace);
            for await (const ev of this.processMessagesChunk(
              chunk,
              source,
              tasksById,
              pending,
            )) {
              yield ev;
            }
          } else if (streamMode === "updates") {
            if (
              eventPayload === null ||
              typeof eventPayload !== "object" ||
              Array.isArray(eventPayload)
            ) {
              logger.warn(
                `Unexpected non-dict updates payload: ${typeof eventPayload}`,
              );
              continue;
            }
            const updateDict = eventPayload as Record<string, unknown>;
            const interrupts = updateDict["__interrupt__"];
            if (Array.isArray(interrupts)) {
              // A nested subgraph's interrupt projects onto its parent, so the
              // same native id legitimately streams twice (child + root) with
              // an identical value. Keep one copy. A repeated id with a
              // different value stays in the list for interruptSnapshot to
              // reject as conflicting framework state.
              for (const interrupt of interrupts as Interrupt[]) {
                const id = (interrupt as { id?: unknown }).id;
                if (typeof id === "string") {
                  const previous = capturedInterruptValues.get(id);
                  const value = (interrupt as { value?: unknown }).value;
                  if (
                    previous !== undefined &&
                    isDeepStrictEqual(previous, value)
                  ) {
                    continue;
                  }
                  capturedInterruptValues.set(id, value);
                }
                capturedInterrupts.push(interrupt);
              }
            }
            for (const nodeOutput of Object.values(updateDict)) {
              // `Command` IS an object, so it must be checked BEFORE the generic
              // object branch below — otherwise Command instances fall into the
              // `'messages' in node` check (which is false for Command, whose
              // payload lives under `.update`) and their messages are silently
              // dropped. See tests/agent.test.ts (Command.update coverage).
              if (nodeOutput instanceof Command) {
                for await (const ev of this.processCommandUpdate(
                  nodeOutput,
                  tasksById,
                  allMessages,
                )) {
                  yield ev;
                }
              } else if (
                nodeOutput !== null &&
                typeof nodeOutput === "object" &&
                !Array.isArray(nodeOutput)
              ) {
                const node = nodeOutput as Record<string, unknown>;
                if ("messages" in node) {
                  for await (const ev of this.processAddMessagesUpdate(
                    node["messages"],
                    tasksById,
                    pending,
                    allMessages,
                  )) {
                    yield ev;
                  }
                }
              }
            }
          }
        }
      } catch (exc: unknown) {
        streamError = exc;
      }

      if (streamError !== undefined) {
        // Mirror Python's `except (GeneratorExit, asyncio.CancelledError): raise`:
        // skip the drain on cancellation so we don't try to yield while the
        // surrounding task is being torn down.
        if (!isCancellationError(streamError)) {
          for await (const ev of drainOpenTasks(tasksById, pending)) yield ev;
        }
        throw streamError;
      }

      const interrupt = await session.streamInterrupt(
        config,
        capturedInterrupts,
        allMessages,
      );
      if (interrupt instanceof Command) {
        nextInput = interrupt;
        continue;
      }
      if (interrupt !== null) {
        yield interrupt;
        return;
      }
      break;
    }

    for await (const ev of drainOpenTasks(tasksById, pending)) yield ev;

    if (allMessages.length === 0) {
      throw new Error("Graph completed without producing messages");
    }

    const response = findLastAiContent(allMessages);
    const metadata = await session.complete();

    yield {
      data: {
        response,
        // Convert to platform Messages ({role, content, ...}) so the runner-shared
        // AER path receives the expected shape — matches Python's
        // `lc_messages_to_platform(...)` before emitting the result (agent.py).
        messages: lcMessagesToPlatform(allMessages) as never,
        message_count: allMessages.length,
        resumed: session.resumed,
        ...(metadata === undefined ? {} : { metadata }),
      },
      event: "result",
    };
  }
}
