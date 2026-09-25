/** Shared execution-session contract and native LangGraph implementation. */

import { HumanMessage, type BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  Command,
  type CompiledStateGraph,
  type Interrupt,
} from "@langchain/langgraph";
import { isDeepStrictEqual } from "node:util";
import type {
  AgentInput,
  AgentOutput,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "@mongodb-js/agent-engine-runner-shared";

import {
  initializeCheckpointBranch,
  branchPointFromMetadata,
  checkpointMetadata,
  type LangGraphCheckpoint,
} from "./checkpoint_branch.js";
import type { DurableSubgraphResolver } from "./durable_subgraphs.js";
import { PlatformCheckpointer } from "./platform_checkpointer.js";
import { takeContinueWithoutUserMessage } from "./session_fork.js";
import {
  interruptSnapshot,
  invokeSuspendOutput,
  streamHitlSuspendEvent,
  validateInterruptValue,
} from "./suspend.js";
import { checkpointThreadId } from "./thread_id.js";

const logger = getLogger("agent_engine_sdk_langgraph.execution_session");

export type PrepareAgentInput = (
  input: AgentInput,
  ctx: RequestContext,
) => unknown;

export type ResolveThreadId = (ctx: RequestContext) => string;

export interface PreparedRun {
  readonly graphInput: unknown;
  readonly config: RunnableConfig;
}

type ExecutionMetadata = ReturnType<typeof checkpointMetadata>;

export interface SessionOptions {
  readonly callbacks: readonly unknown[];
  readonly prepareInput: PrepareAgentInput | null;
  readonly resolveThreadId: ResolveThreadId | null;
  readonly durableSubgraphs: DurableSubgraphResolver | null;
}

interface MongoSaverLike {
  readonly db?: {
    collection?: (name: string) => {
      distinct(
        key: string,
        filter: Record<string, unknown>,
      ): Promise<unknown[]>;
      findOne(
        filter: Record<string, unknown>,
        options: Record<string, unknown>,
      ): Promise<Record<string, unknown> | null>;
      deleteMany(
        filter: Record<string, unknown>,
      ): Promise<{ deletedCount?: number }>;
    };
  };
  readonly checkpointCollectionName?: string;
  readonly checkpointWritesCollectionName?: string;
}

interface NativeStateSnapshot {
  readonly next?: readonly unknown[];
  readonly tasks?: readonly {
    readonly interrupts?: readonly {
      readonly id?: string;
      readonly value: unknown;
    }[];
  }[];
}

function interruptValuesFromState(state: NativeStateSnapshot): unknown[] {
  const values: unknown[] = [];
  const seen = new Map<string, unknown>();
  for (const task of state.tasks ?? []) {
    for (const interrupt of task.interrupts ?? []) {
      validateInterruptValue(interrupt.id ?? "<unknown>", interrupt.value);
      const id = typeof interrupt.id === "string" ? interrupt.id : "";
      // A nested subgraph's interrupt can repeat under its parent task with
      // an identical value; project each native id once. A repeated id with
      // a different value is conflicting framework state — fail closed the
      // same way the stream and durable paths do.
      if (id) {
        const previous = seen.get(id);
        if (previous !== undefined) {
          if (isDeepStrictEqual(previous, interrupt.value)) continue;
          throw new Error("LangGraph returned duplicate interrupt ids");
        }
        seen.set(id, interrupt.value);
      }
      values.push(interrupt.value);
    }
  }
  return values;
}

function payloadDict(input: AgentInput): Record<string, unknown> {
  if (
    input.payload === null ||
    typeof input.payload !== "object" ||
    Array.isArray(input.payload)
  ) {
    throw new TypeError(`Expected dict payload, got ${typeof input.payload}`);
  }
  return input.payload as Record<string, unknown>;
}

function resolveCheckpointThreadId(
  ctx: RequestContext,
  resolveThreadId: ResolveThreadId | null,
): string {
  return checkpointThreadId(ctx, resolveThreadId);
}

function normalizeCheckpointId(value: unknown): string | undefined {
  if (value === null || value === undefined) return undefined;
  const checkpointId = String(value).trim();
  if (checkpointId === "" || checkpointId.toLowerCase() === "none") {
    return undefined;
  }
  return checkpointId;
}

export function explicitCheckpointId(ctx: RequestContext): string | undefined {
  const checkpointId = normalizeCheckpointId(ctx.metadata?.["checkpoint_id"]);
  return checkpointId?.startsWith("thread:") ? undefined : checkpointId;
}

export abstract class ExecutionSession {
  abstract readonly resumed: boolean;
  readonly threadId: string;
  protected readonly graph: CompiledStateGraph<unknown, unknown>;
  protected readonly ctx: RequestContext;
  protected readonly input: AgentInput;
  protected readonly callbacks: readonly unknown[];
  protected readonly prepareInput: PrepareAgentInput | null;
  protected readonly durableSubgraphs: DurableSubgraphResolver | null;

  protected constructor(
    graph: CompiledStateGraph<unknown, unknown>,
    ctx: RequestContext,
    input: AgentInput,
    options: SessionOptions,
  ) {
    this.graph = graph;
    this.ctx = ctx;
    this.input = input;
    this.callbacks = options.callbacks;
    this.prepareInput = options.prepareInput;
    this.durableSubgraphs = options.durableSubgraphs;
    this.threadId = resolveCheckpointThreadId(ctx, options.resolveThreadId);
  }

  protected baseConfig(): RunnableConfig {
    const config: RunnableConfig = {
      configurable: {
        thread_id: this.threadId,
        request_context: this.ctx,
      },
    };
    if (this.callbacks.length > 0) {
      (config as { callbacks?: unknown }).callbacks = [...this.callbacks];
    }
    return config;
  }

  protected freshGraphInput(ctx: RequestContext = this.ctx): unknown {
    if (this.prepareInput !== null) {
      return this.prepareInput(this.input, ctx);
    }
    const payload = payloadDict(this.input);
    const graphInput: Record<string, unknown> = {
      messages: [
        new HumanMessage({ content: String(payload["message"] ?? "") }),
      ],
    };
    if (ctx.userId) graphInput["user_id"] = ctx.userId;
    if (ctx.sessionId) graphInput["session_id"] = ctx.sessionId;
    return graphInput;
  }

  abstract prepareRun(): Promise<PreparedRun>;

  abstract invokeInterrupt(
    config: RunnableConfig,
    response: string,
    messages: readonly unknown[],
  ): Promise<Command | AgentOutput | null>;

  abstract streamInterrupt(
    config: RunnableConfig,
    captured: readonly Interrupt[],
    messages: readonly BaseMessage[],
  ): Promise<Command | StreamEvent | null>;

  abstract complete(): Promise<ExecutionMetadata | void>;

  abstract close(): Promise<void>;

  runInGraphScope<T>(fn: () => T): T {
    return fn();
  }

  async *iterateInGraphScope<T>(iterable: AsyncIterable<T>): AsyncIterable<T> {
    const iterator = iterable[Symbol.asyncIterator]();
    try {
      while (true) {
        const result = await this.runInGraphScope(() => iterator.next());
        if (result.done) return;
        yield result.value;
      }
    } finally {
      const close = iterator.return;
      if (close !== undefined) {
        await this.runInGraphScope(() => close.call(iterator));
      }
    }
  }
}

export class NativeSession extends ExecutionSession {
  readonly resumed: boolean;
  private readonly config: RunnableConfig;
  private readonly branchSource: LangGraphCheckpoint | undefined;
  private readonly resumeCommand: Command | null;

  constructor(
    graph: CompiledStateGraph<unknown, unknown>,
    ctx: RequestContext,
    input: AgentInput,
    options: SessionOptions,
  ) {
    super(graph, ctx, input, options);
    this.resumed = ctx.resume === true;
    const resumeData = ctx.resumeData;
    if (
      ctx.resume === true &&
      (resumeData === undefined || resumeData === null)
    ) {
      throw new Error(
        "ctx.resume is true but resume_data is missing. " +
          "Provide resume_data to resume a suspended graph.",
      );
    }
    this.resumeCommand =
      ctx.resume === true && resumeData !== undefined && resumeData !== null
        ? new Command({ resume: resumeData })
        : null;
    this.branchSource = this.resumed
      ? undefined
      : branchPointFromMetadata(ctx.metadata);
    const config = this.baseConfig();
    const checkpointId = explicitCheckpointId(ctx);
    if (checkpointId !== undefined) {
      config.configurable = {
        ...config.configurable,
        checkpoint_id: checkpointId,
      };
    }
    this.config = config;
  }

  async prepareRun(): Promise<PreparedRun> {
    // A null graph input is meaningful (continue the copied checkpoint without
    // a user turn), so the branches stay explicit: `??` would fall through it.
    let graphInput: unknown;
    if (this.resumeCommand !== null) {
      graphInput = this.resumeCommand;
    } else {
      graphInput =
        this.firstDestContinueInput() === null ? null : this.freshGraphInput();
    }
    // HITL resume continues a thread that already received any branch copy;
    // clients and OE dispatch may still send the source branch point.
    if (!this.resumed && this.branchSource !== undefined) {
      await initializeCheckpointBranch(
        this.graph,
        this.config,
        this.branchSource,
      );
    }
    if (this.ctx.previousExecutionCancelled === true && !this.resumed) {
      await this.fenceCancelledPredecessorWrites();
    }
    return { graphInput, config: this.config };
  }

  /**
   * After an in-app fork, the destination's first continue carries no user
   * message; invoke(null) so LangGraph resumes from the copied checkpoint
   * instead of appending an empty HumanMessage (docs/session-fork.md).
   */
  private firstDestContinueInput(): null | undefined {
    if (this.resumed) return undefined;
    if (!takeContinueWithoutUserMessage(this.threadId)) return undefined;
    const raw = this.input.payload;
    const payload =
      raw !== null && typeof raw === "object" && !Array.isArray(raw)
        ? (raw as Record<string, unknown>)
        : {};
    if (String(payload["message"] ?? "").trim() !== "") return undefined;
    return null;
  }

  async invokeInterrupt(
    config: RunnableConfig,
    response: string,
    messages: readonly unknown[],
  ): Promise<AgentOutput | null> {
    const graph = this.graph as unknown as {
      getState(config: RunnableConfig): Promise<NativeStateSnapshot>;
    };
    const state = await graph.getState(config);
    if (!state.next || state.next.length === 0) return null;
    const interruptValues = interruptValuesFromState(state);
    const checkpointId = await this.getCheckpointId();
    return invokeSuspendOutput({
      response,
      threadId: this.threadId,
      checkpointId,
      interruptValues,
      messageCount: messages.length,
      resumed: this.resumed,
    });
  }

  async streamInterrupt(
    _config: RunnableConfig,
    captured: readonly Interrupt[],
    messages: readonly BaseMessage[],
  ): Promise<StreamEvent | null> {
    if (captured.length === 0) return null;
    const interrupts = interruptSnapshot(captured);
    const checkpointId = await this.getCheckpointId();
    return streamHitlSuspendEvent({
      interrupts,
      checkpointId,
      resumed: this.resumed,
      messages,
    });
  }

  async complete(): Promise<ExecutionMetadata> {
    const checkpointId = await this.getCheckpointId();
    return checkpointMetadata(
      checkpointId === undefined
        ? undefined
        : { threadId: this.threadId, checkpointId },
    );
  }

  close(): Promise<void> {
    return Promise.resolve();
  }

  private async getCheckpointId(): Promise<string | undefined> {
    const checkpointer = (this.graph as { checkpointer?: unknown })
      .checkpointer;
    if (checkpointer === null || checkpointer === undefined) return undefined;
    const saver = checkpointer as {
      getTuple?: (config: RunnableConfig) => Promise<unknown>;
    };
    if (typeof saver.getTuple !== "function") return undefined;
    try {
      const tuple = (await saver.getTuple({
        configurable: { thread_id: this.threadId },
      })) as { checkpoint?: unknown } | null;
      const checkpoint = tuple?.checkpoint;
      if (
        checkpoint !== null &&
        typeof checkpoint === "object" &&
        !Array.isArray(checkpoint)
      ) {
        const id = (checkpoint as Record<string, unknown>)["id"];
        return typeof id === "string" ? id : undefined;
      }
    } catch (exc: unknown) {
      logger.warn(
        `Failed to get checkpoint ID for thread ${this.threadId} — degrading to undefined ` +
          `without changing the agent execution outcome: ${exc instanceof Error ? exc.message : String(exc)}`,
      );
    }
    return undefined;
  }

  private async fenceCancelledPredecessorWrites(): Promise<void> {
    // A cancelled pod can leave half-finished writes on the latest checkpoint.
    // Remove only those writes before a fresh turn can fold them into its state.
    const graphSaver = (this.graph as { checkpointer?: unknown }).checkpointer;
    const saver = (
      graphSaver instanceof PlatformCheckpointer
        ? graphSaver.native
        : graphSaver
    ) as MongoSaverLike | null;
    if (
      !saver?.db?.collection ||
      typeof saver.checkpointCollectionName !== "string" ||
      typeof saver.checkpointWritesCollectionName !== "string"
    ) {
      return;
    }
    try {
      const checkpoints = saver.db.collection(saver.checkpointCollectionName);
      const writes = saver.db.collection(saver.checkpointWritesCollectionName);
      const namespaces = await checkpoints.distinct("checkpoint_ns", {
        thread_id: this.threadId,
      });
      let removed = 0;
      for (const namespace of namespaces) {
        const latest = await checkpoints.findOne(
          { thread_id: this.threadId, checkpoint_ns: namespace },
          { sort: { checkpoint_id: -1 }, projection: { checkpoint_id: 1 } },
        );
        if (!latest) continue;
        const result = await writes.deleteMany({
          thread_id: this.threadId,
          checkpoint_ns: namespace,
          checkpoint_id: latest["checkpoint_id"],
        });
        removed += result.deletedCount ?? 0;
      }
      if (removed > 0) {
        logger.info(
          `Discarded ${removed} pending checkpoint writes left by a cancelled run (thread ${this.threadId})`,
        );
      }
    } catch (exc: unknown) {
      logger.warn(
        `Failed to fence a cancelled run's pending writes (thread ${this.threadId}): ` +
          `${exc instanceof Error ? exc.message : String(exc)}`,
      );
    }
  }
}
