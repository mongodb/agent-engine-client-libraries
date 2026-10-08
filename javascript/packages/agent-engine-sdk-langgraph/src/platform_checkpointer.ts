/** Request-scoped native versus OE-durable LangGraph checkpoint routing. */

import type { RunnableConfig } from "@langchain/core/runnables";
import {
  BaseCheckpointSaver,
  MemorySaver,
  emptyCheckpoint,
  Send,
} from "@langchain/langgraph";
import type {
  ChannelVersions,
  Checkpoint,
  CheckpointListOptions,
  CheckpointMetadata,
  CheckpointTuple,
  PendingWrite,
} from "@langchain/langgraph-checkpoint";
import {
  advanceStepOrdinal,
  completeExecutionCommand,
  currentAttemptContext,
  currentStepOrdinal,
  finalizeCurrentStepCommand,
  getCurrentOeUrl,
  WorkflowClient,
  type AttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";

import { noteRaisedInterrupts } from "./raised_interrupts.js";
import {
  channelValuesToStateSnapshot,
  stateSnapshotToChannelValues,
} from "./workflow_state.js";

type PlatformWorkflowClient = Pick<
  WorkflowClient,
  "completeExecution" | "finalizeStep"
>;
type WorkflowClientFactory = (oeUrl: string) => PlatformWorkflowClient;

export class UnsupportedDurableGraphError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedDurableGraphError";
  }
}

export function requireDurableExecutionId(attempt: AttemptContext): string {
  const executionId = attempt.workflowIdentity?.executionId ?? "";
  if (!executionId || !attempt.attemptId || attempt.fencingToken <= 0n) {
    throw new UnsupportedDurableGraphError(
      "durable scratch requires complete OE attempt identity",
    );
  }
  return executionId;
}

export function scratchThreadId(attempt: AttemptContext): string {
  const executionId = requireDurableExecutionId(attempt);
  return `${executionId}:${attempt.attemptId}:${attempt.fencingToken.toString()}`;
}

/** LangGraph's dynamic-fan-out channel; Send packets on it are unsupported. */
const PREGEL_TASKS_CHANNEL = "__pregel_tasks";

/**
 * Absolute OE step ordinal recorded on committed root-loop scratch
 * checkpoints so session-fork target selection never reconstructs LangGraph's
 * seeded relative `step` offset (Python `_durable_metadata` parity). The
 * string is part of the stored checkpoint format and shared with the Python
 * SDK, so changing it invalidates forks of older checkpoints.
 */
export const OE_STEP_ORDINAL_METADATA_KEY = "agent_engine_oe_step_ordinal";

/**
 * Topic.checkpoint() stores this channel as `[seen, values]`, so Send packets
 * sit one array deeper than the checkpoint value itself. Scan arrays to a
 * bounded depth instead of matching only the top level.
 *
 * Recognize a packet by LangGraph's stable SendInterface shape
 * (`{node, args}`) in addition to `instanceof Send`: a tenant application can
 * load its own copy of `@langchain/langgraph`, and a Send from that copy
 * fails a same-module instanceof check while behaving identically here.
 */
function containsSendPacket(value: unknown, depth = 0): boolean {
  if (depth > 4) return false;
  if (Array.isArray(value)) {
    return value.some((item) => containsSendPacket(item, depth + 1));
  }
  if (value instanceof Send) return true;
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { node?: unknown }).node === "string" &&
    (value as { args?: unknown }).args !== undefined
  );
}

function rejectUnsupportedSendPackets(channel: string, value: unknown): void {
  if (channel !== PREGEL_TASKS_CHANNEL) return;
  if (containsSendPacket(value)) {
    throw new UnsupportedDurableGraphError(
      "LangGraph Send is not supported on durable_workflow sessions; " +
        "use fixed graph edges, compiled subgraphs, or Deep Agent task delegation",
    );
  }
}

export class PlatformCheckpointer extends BaseCheckpointSaver<string | number> {
  readonly native: BaseCheckpointSaver<string | number> | null;
  private readonly scratch: MemorySaver;
  private readonly clientFactory: WorkflowClientFactory;
  /**
   * Deep-agent graphs route pending ToolCalls through LangGraph Send packets
   * the adapter owns; when set, the Send rejection does not apply. The flag
   * keeps the wrapper sharing the same platform scratch and native savers.
   */
  private readonly allowSendPackets: boolean;

  constructor(args: {
    native: BaseCheckpointSaver<string | number> | null;
    scratch?: MemorySaver;
    clientFactory?: WorkflowClientFactory;
    allowSendPackets?: boolean;
  }) {
    super(args.native?.serde);
    this.native = args.native;
    this.scratch = args.scratch ?? new MemorySaver();
    this.allowSendPackets = args.allowSendPackets ?? false;
    this.clientFactory =
      args.clientFactory ?? ((oeUrl) => new WorkflowClient(oeUrl));
  }

  async releaseScratch(attempt: AttemptContext): Promise<void> {
    await this.scratch.deleteThread(scratchThreadId(attempt));
  }

  /**
   * A checkpointer view that shares this instance's platform state and test
   * seams but tolerates the adapter-owned Send routing Deep Agent graphs use
   * (see `checkpointerForDeepAgent`).
   */
  deepAgentView(): PlatformCheckpointer {
    return new PlatformCheckpointer({
      native: this.native,
      scratch: this.scratch,
      clientFactory: this.clientFactory,
      allowSendPackets: true,
    });
  }

  async seedPreviousState(
    attempt: AttemptContext,
    config: RunnableConfig,
  ): Promise<void> {
    if (attempt.previousState === undefined) return;
    const durableConfig = this.durableConfig(config, attempt);
    if ((await this.scratch.getTuple(durableConfig)) !== undefined) return;

    const checkpoint = emptyCheckpoint();
    checkpoint.channel_values = stateSnapshotToChannelValues(
      attempt.previousState,
    );
    const version = this.scratch.getNextVersion(undefined);
    checkpoint.channel_versions = Object.fromEntries(
      Object.keys(checkpoint.channel_values).map((name) => [name, version]),
    );
    await this.scratch.put(durableConfig, checkpoint, {
      source: "input",
      step: -1,
      parents: {},
    });
  }

  async completeExecution(attempt: AttemptContext): Promise<void> {
    const tuple = await this.scratch.getTuple({
      configurable: { thread_id: scratchThreadId(attempt) },
    });
    if (tuple === undefined) {
      throw new UnsupportedDurableGraphError(
        "durable execution completed without final application state",
      );
    }
    await this.workflowClient().completeExecution(
      completeExecutionCommand(
        attempt,
        channelValuesToStateSnapshot(
          tuple.checkpoint.channel_values,
          attempt.replayMode,
        ),
      ),
    );
  }

  override async getTuple(
    config: RunnableConfig,
  ): Promise<CheckpointTuple | undefined> {
    const attempt = currentAttemptContext();
    if (attempt === null) return this.requireNative().getTuple(config);
    return this.scratch.getTuple(this.durableConfig(config, attempt));
  }

  override async *list(
    config: RunnableConfig,
    options?: CheckpointListOptions,
  ): AsyncGenerator<CheckpointTuple> {
    const attempt = currentAttemptContext();
    if (attempt === null) {
      yield* this.requireNative().list(config, options);
      return;
    }
    // Read-only inspection scoped to the fenced attempt's scratch: OE
    // activity history is the durable replay authority, but the current
    // attempt's own scratch history stays inspectable (Python `alist` parity).
    if (config === undefined || config === null) {
      throw new UnsupportedDurableGraphError(
        "durable scratch history requires the current attempt config",
      );
    }
    const durableOptions =
      options?.before !== undefined
        ? { ...options, before: this.durableConfig(options.before, attempt) }
        : options;
    yield* this.scratch.list(
      this.durableConfig(config, attempt),
      durableOptions,
    );
  }

  override async put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    newVersions: ChannelVersions,
  ): Promise<RunnableConfig> {
    const attempt = currentAttemptContext();
    if (attempt === null) {
      return this.requireNative().put(
        config,
        checkpoint,
        metadata,
        newVersions,
      );
    }
    if (metadata.source === "update" || metadata.source === "fork") {
      throw new UnsupportedDurableGraphError(
        `checkpoint ${metadata.source} is not supported on durable_workflow sessions`,
      );
    }

    const durableConfig = this.durableConfig(config, attempt);
    // Reject where LangGraph synchronously commits the producer step: the
    // fanned-out workers must never start on a durable attempt.
    for (const [channel, value] of Object.entries(
      checkpoint.channel_values as Record<string, unknown>,
    )) {
      this.rejectUnsupportedSendPackets(channel, value);
    }
    const durableMetadata = this.durableMetadata(metadata, durableConfig);
    const result = await this.scratch.put(
      durableConfig,
      checkpoint,
      durableMetadata,
    );
    if (this.shouldFinalize(metadata, durableConfig)) {
      const command = finalizeCurrentStepCommand(
        attempt,
        channelValuesToStateSnapshot(
          checkpoint.channel_values,
          attempt.replayMode,
        ),
      );
      const entries = await this.workflowClient().finalizeStep(command);
      if (entries.length > 0) {
        throw new UnsupportedDurableGraphError(
          "settled step finalization returned suspension entries",
        );
      }
      advanceStepOrdinal(Number(command.stepOrdinal));
    }
    return result;
  }

  override async putWrites(
    config: RunnableConfig,
    writes: PendingWrite[],
    taskId: string,
  ): Promise<void> {
    const attempt = currentAttemptContext();
    if (attempt === null) {
      await this.requireNative().putWrites(config, writes, taskId);
      return;
    }
    // Reject dynamic fan-out where LangGraph backgrounds the write: a PUSH
    // worker could start before this callback fails.
    for (const [channel, value] of writes) {
      this.rejectUnsupportedSendPackets(channel, value);
    }
    noteRaisedInterrupts(attempt, writes);
    await this.scratch.putWrites(
      this.durableConfig(config, attempt),
      writes,
      taskId,
    );
  }

  override async deleteThread(threadId: string): Promise<void> {
    const attempt = currentAttemptContext();
    if (attempt === null) {
      await this.requireNative().deleteThread(threadId);
      return;
    }
    await this.scratch.deleteThread(scratchThreadId(attempt));
  }

  override getNextVersion(
    current: string | number | undefined,
  ): string | number {
    const attempt = currentAttemptContext();
    if (attempt === null) return this.requireNative().getNextVersion(current);
    if (typeof current === "string") {
      throw new UnsupportedDurableGraphError(
        "durable scratch requires numeric channel versions",
      );
    }
    return this.scratch.getNextVersion(current);
  }

  /**
   * Reject application-authored dynamic fan-out in durable graphs. Deep-agent
   * graphs route their own adapter-owned Send packets through the same
   * platform state, so they opt out via `allowSendPackets`.
   */
  protected rejectUnsupportedSendPackets(
    channel: string,
    value: unknown,
  ): void {
    if (this.allowSendPackets) return;
    rejectUnsupportedSendPackets(channel, value);
  }

  private requireNative(): BaseCheckpointSaver<string | number> {
    if (this.native === null) {
      throw new UnsupportedDurableGraphError(
        "native_checkpoint session requires a MongoDB-backed checkpointer",
      );
    }
    return this.native;
  }

  private durableConfig(
    config: RunnableConfig,
    attempt: AttemptContext,
  ): RunnableConfig {
    return {
      ...config,
      configurable: {
        ...(config.configurable ?? {}),
        thread_id: scratchThreadId(attempt),
        checkpoint_ns: config.configurable?.["checkpoint_ns"] ?? "",
      },
    };
  }

  private shouldFinalize(
    metadata: CheckpointMetadata,
    config: RunnableConfig,
  ): boolean {
    return (
      metadata.source === "loop" &&
      (config.configurable?.["checkpoint_ns"] ?? "") === ""
    );
  }

  /**
   * Stamp the absolute ordinal FinalizeStep uses onto the same root-loop
   * scratch checkpoint. LangGraph's relative `step` can shift when OE seeds
   * previous state, so fork target selection reads this key instead.
   */
  private durableMetadata(
    metadata: CheckpointMetadata,
    config: RunnableConfig,
  ): CheckpointMetadata {
    if (!this.shouldFinalize(metadata, config)) return metadata;
    return {
      ...metadata,
      [OE_STEP_ORDINAL_METADATA_KEY]: currentStepOrdinal(),
    } as CheckpointMetadata;
  }

  private workflowClient(): PlatformWorkflowClient {
    const oeUrl = getCurrentOeUrl();
    if (!oeUrl) {
      throw new UnsupportedDurableGraphError(
        "durable workflow state requires the OE callback URL",
      );
    }
    return this.clientFactory(oeUrl);
  }
}
