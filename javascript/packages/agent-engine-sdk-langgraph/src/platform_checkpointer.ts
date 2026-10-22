/** Request-scoped native versus OE-durable LangGraph checkpoint routing. */

import { clone } from "@bufbuild/protobuf";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  BaseCheckpointSaver,
  MemorySaver,
  emptyCheckpoint,
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
  currentCancelledHistory,
  currentStepOrdinal,
  finalizeCurrentStepCommand,
  getCurrentOeUrl,
  getCurrentWrapper,
  WorkflowClient,
  StateSnapshotSchema,
  type AttemptContext,
  type SecureToolWrapper,
} from "@mongodb-js/agent-engine-runner-shared";

import { noteRaisedInterrupts } from "./raised_interrupts.js";
import {
  channelValuesToStateSnapshot,
  stateSnapshotToChannelValues,
} from "./workflow_state.js";

type PlatformWorkflowClient = Pick<
  WorkflowClient,
  "completeExecution" | "finalizeStep"
> &
  Partial<Pick<WorkflowClient, "finalizeStepWithAcknowledgement">>;
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

/**
 * Absolute OE step ordinal recorded on committed root-loop scratch
 * checkpoints so session-fork target selection never reconstructs LangGraph's
 * seeded relative `step` offset (Python `_durable_metadata` parity). The
 * string is part of the stored checkpoint format and shared with the Python
 * SDK, so changing it invalidates forks of older checkpoints.
 */
export const OE_STEP_ORDINAL_METADATA_KEY = "agent_engine_oe_step_ordinal";

/**
 * Refuse to save anything once a tool call in this execution hit a platform
 * failure. ToolNode turns that failure into a message for the model; a saved
 * checkpoint or pending write would hand it to the model on a retry or the
 * next turn, since LangGraph restores a finished task's writes without
 * running it again.
 */
function throwIfPlatformFailed(): void {
  (getCurrentWrapper() as SecureToolWrapper | null)?.throwIfPlatformFailed();
}

export class PlatformCheckpointer extends BaseCheckpointSaver<string | number> {
  readonly native: BaseCheckpointSaver<string | number> | null;
  private readonly scratch: MemorySaver;
  private readonly clientFactory: WorkflowClientFactory;
  /**
   * Marks the view a Deep Agent graph uses. Deep Agent routes each ToolCall
   * through an internal Send, and its recorded activity paths predate Send
   * task segments, so the adapter keeps those tasks on their parent path.
   */
  readonly isDeepAgentView: boolean;

  constructor(args: {
    native: BaseCheckpointSaver<string | number> | null;
    scratch?: MemorySaver;
    clientFactory?: WorkflowClientFactory;
    deepAgentView?: boolean;
  }) {
    super(args.native?.serde);
    this.native = args.native;
    this.scratch = args.scratch ?? new MemorySaver();
    this.isDeepAgentView = args.deepAgentView ?? false;
    this.clientFactory =
      args.clientFactory ?? ((oeUrl) => new WorkflowClient(oeUrl));
  }

  async releaseScratch(attempt: AttemptContext): Promise<void> {
    await this.scratch.deleteThread(scratchThreadId(attempt));
  }

  /**
   * A checkpointer view that shares this instance's platform state and test
   * seams and marks the graph as Deep Agent-owned
   * (see `checkpointerForDeepAgent`).
   */
  deepAgentView(): PlatformCheckpointer {
    return new PlatformCheckpointer({
      native: this.native,
      scratch: this.scratch,
      clientFactory: this.clientFactory,
      deepAgentView: true,
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
    throwIfPlatformFailed();
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
      // Save the exact submitted state before awaiting transport or reducer
      // mutations, but promote it only once OE acknowledges the root step.
      if (command.state === undefined) {
        throw new Error("root finalization requires state");
      }
      const submittedState = clone(StateSnapshotSchema, command.state);
      const client = this.workflowClient();
      const acknowledgement =
        client.finalizeStepWithAcknowledgement === undefined
          ? {
              entries: await client.finalizeStep(command),
              sourceStateHash: undefined,
            }
          : await client.finalizeStepWithAcknowledgement(command);
      const entries = acknowledgement.entries;
      if (entries.length > 0) {
        throw new UnsupportedDurableGraphError(
          "settled step finalization returned suspension entries",
        );
      }
      currentCancelledHistory()?.acknowledge(
        submittedState,
        command.stepOrdinal,
        acknowledgement.sourceStateHash,
      );
      advanceStepOrdinal(Number(command.stepOrdinal));
    }
    return result;
  }

  override async putWrites(
    config: RunnableConfig,
    writes: PendingWrite[],
    taskId: string,
  ): Promise<void> {
    throwIfPlatformFailed();
    const attempt = currentAttemptContext();
    if (attempt === null) {
      await this.requireNative().putWrites(config, writes, taskId);
      return;
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
