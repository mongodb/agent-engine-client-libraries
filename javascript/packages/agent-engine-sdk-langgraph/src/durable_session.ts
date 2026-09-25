/** OE-attempt-scoped LangGraph session behavior. */

import { create, equals, type JsonObject } from "@bufbuild/protobuf";
import { isDeepStrictEqual } from "node:util";
import type { BaseMessage } from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  Command,
  GraphInterrupt,
  isCommand,
  isGraphInterrupt,
  type CompiledStateGraph,
  type Interrupt,
} from "@langchain/langgraph";
import type {
  AgentInput,
  AgentOutput,
  StreamEvent,
  RequestContext,
} from "@mongodb-js/agent-engine-sdk";
import {
  runWithOperationPathResolver,
  ActivityOutcomeKind,
  ActivityKind,
  ActivityContextSchema,
  ActivitySuspensionSchema,
  OperationPathSchema,
  OperationPathSegmentSchema,
  StepSuspensionEntrySchema,
  WorkflowClient,
  WorkflowIdentitySchema,
  currentStepOrdinal,
  buildActivityCommand,
  finalizeCurrentStepSuspensionsCommand,
  getCurrentOeUrl,
  getCurrentUserId,
  getCurrentWrapper,
  interruptedActivities,
  recordInterruptedActivity,
  recordObservedActivity,
  setActivityReconstructionIds,
  unwrapActivityOutcome,
  type ActivityPosition,
  type AttemptContext,
  type InterruptedActivity,
  type StepActivityEntry,
  type StepSuspensionEntry,
  type SecureToolWrapper,
} from "@mongodb-js/agent-engine-runner-shared";

import {
  ExecutionSession,
  type PreparedRun,
  type SessionOptions,
} from "./execution_session.js";
import {
  interruptSnapshot,
  invokeSuspendOutput,
  streamHitlSuspendEvent,
  type PublicInterrupt,
} from "./suspend.js";
import { assignMissingGraphInputMessageIds } from "./messages.js";
import type { PlatformCheckpointer } from "./platform_checkpointer.js";
import { UnsupportedDurableGraphError } from "./platform_checkpointer.js";

const FRAMEWORK_INTERRUPT_REASON = "agent_interrupt";
const DIRECT_INTERRUPT_ACTIVITY_NAME = "langgraph.interrupt";
interface StateSnapshotLike {
  readonly next?: readonly unknown[];
  readonly tasks?: readonly {
    readonly path?: readonly unknown[];
    readonly interrupts?: readonly Interrupt[];
    readonly state?: StateSnapshotLike | RunnableConfig;
  }[];
}

function validateDurableInterruptNumbers(value: unknown): void {
  if (
    typeof value === "number" &&
    Number.isInteger(value) &&
    !Number.isSafeInteger(value)
  ) {
    throw new UnsupportedDurableGraphError(
      "durable interrupt integer exceeds the exact ProtoJSON range",
    );
  }
  if (Array.isArray(value)) {
    for (const item of value) validateDurableInterruptNumbers(item);
  } else if (value !== null && typeof value === "object") {
    for (const item of Object.values(value))
      validateDurableInterruptNumbers(item);
  }
}

function uniqueInterrupts(captured: readonly Interrupt[]): PublicInterrupt[] {
  const byId = new Map<string, PublicInterrupt>();
  for (const raw of captured) {
    for (const interrupt of interruptSnapshot([raw])) {
      validateDurableInterruptNumbers(interrupt.value);
      const previous = byId.get(interrupt.id);
      if (previous && !isDeepStrictEqual(previous.value, interrupt.value)) {
        throw new UnsupportedDurableGraphError(
          "LangGraph returned conflicting values for an interrupt id",
        );
      }
      byId.set(interrupt.id, interrupt);
    }
  }
  return [...byId.values()];
}

function pendingInterrupts(state: StateSnapshotLike): PublicInterrupt[] {
  // Parent snapshots can repeat a subgraph interrupt, so visit children first
  // and keep each native id only once.
  const pending: PublicInterrupt[] = [];
  const valueById = new Map<string, PublicInterrupt["value"]>();
  const visit = (snapshot: StateSnapshotLike): void => {
    for (const task of snapshot.tasks ?? []) {
      const taskState = task.state;
      if (
        taskState !== undefined &&
        taskState !== null &&
        typeof taskState === "object" &&
        "tasks" in taskState
      ) {
        visit(taskState as StateSnapshotLike);
      }
      for (const interrupt of uniqueInterrupts(task.interrupts ?? [])) {
        const previous = valueById.get(interrupt.id);
        if (previous === undefined) {
          valueById.set(interrupt.id, interrupt.value);
          pending.push(interrupt);
        } else if (!isDeepStrictEqual(previous, interrupt.value)) {
          throw new UnsupportedDurableGraphError(
            "LangGraph returned conflicting values for an interrupt id",
          );
        }
      }
    }
  };
  visit(state);
  return pending;
}

function isDirectInterruptActivity(activity: InterruptedActivity): boolean {
  const segments = activity.command.position?.operationPath?.segments ?? [];
  return (
    activity.command.activityName === DIRECT_INTERRUPT_ACTIVITY_NAME &&
    segments.length === 1 &&
    segments[0]?.name.startsWith("langgraph.task:") === true
  );
}

function recordDirectInterrupts(
  attempt: AttemptContext,
  state: StateSnapshotLike,
): void {
  const currentActivities = interruptedActivities(currentStepOrdinal());
  const localIds = new Set(
    currentActivities
      .filter((activity) => !isDirectInterruptActivity(activity))
      .flatMap((activity) =>
        isGraphInterrupt(activity.controlFlow)
          ? activity.controlFlow.interrupts.map((interrupt) => interrupt.id)
          : [],
      ),
  );
  // Registered local tools already carry their admitted ActivityCommand.
  // Do not synthesize a second root interrupt activity for the checkpoint view.
  const existing = new Map(
    currentActivities
      .filter(isDirectInterruptActivity)
      .map((activity) => [positionKey(activity.command.position), activity]),
  );
  const previouslySettled = new Set(existing.keys());
  for (const task of state.tasks ?? []) {
    if (!task.interrupts?.length) continue;
    const path = task.path;
    if (
      !path?.length ||
      path.some(
        (part) =>
          typeof part !== "string" &&
          !(typeof part === "number" && Number.isSafeInteger(part)),
      )
    ) {
      throw new UnsupportedDurableGraphError(
        "durable direct interrupt requires a stable LangGraph task path",
      );
    }
    // JS putWrites receives only the regenerated task ID. The quiescent
    // snapshot exposes the stable path Python receives at its write boundary.
    const direct = uniqueInterrupts(task.interrupts).filter(
      (interrupt) => !localIds.has(interrupt.id),
    );
    for (const [index, interrupt] of direct.entries()) {
      const command = buildActivityCommand({
        attempt,
        kind: ActivityKind.TOOL,
        name: DIRECT_INTERRUPT_ACTIVITY_NAME,
        activityOrdinal: index + 1,
        operationPath: create(OperationPathSchema, {
          segments: [
            create(OperationPathSegmentSchema, {
              name: `langgraph.task:${JSON.stringify(path)}`,
              ordinal: 1n,
            }),
          ],
        }),
        semanticInput: { value: interrupt.value },
      });
      const key = positionKey(command.position);
      if (previouslySettled.has(key)) {
        // A second graph iteration reached another interrupt before this task
        // finished. Native IDs and even payloads may be identical at both calls.
        throw new UnsupportedDurableGraphError(
          "durable workflow does not support multiple sequential direct interrupts in one graph task",
        );
      }
      const previous = existing.get(key);
      if (previous) {
        if (
          !isGraphInterrupt(previous.controlFlow) ||
          previous.controlFlow.interrupts[0]?.id !== interrupt.id ||
          !isDeepStrictEqual(
            previous.controlFlow.interrupts[0]?.value,
            interrupt.value,
          )
        ) {
          throw new UnsupportedDurableGraphError(
            "durable workflow does not support multiple sequential direct interrupts in one graph task",
          );
        }
        continue;
      }
      const controlFlow = new GraphInterrupt([interrupt]);
      if (command.position === undefined) {
        throw new UnsupportedDurableGraphError(
          "durable interrupt requires an activity position",
        );
      }
      recordObservedActivity(command.position);
      recordInterruptedActivity(command, controlFlow);
      existing.set(key, { command, controlFlow });
    }
  }
}

export interface DurableExecution {
  readonly attempt: AttemptContext;
  readonly checkpointer: PlatformCheckpointer;
  readonly executionId: string;
}

function positionKey(position: ActivityPosition | undefined): string {
  if (position === undefined) {
    throw new UnsupportedDurableGraphError(
      "durable interrupt settlement returned an entry without a position",
    );
  }
  return JSON.stringify([
    position.stepOrdinal.toString(),
    (position.operationPath?.segments ?? []).map((segment) => [
      segment.name,
      segment.ordinal.toString(),
    ]),
    position.activityOrdinal.toString(),
  ]);
}

function suspensionEntry(
  activity: InterruptedActivity,
  interrupt: PublicInterrupt,
): StepSuspensionEntry {
  const command = activity.command;
  return create(StepSuspensionEntrySchema, {
    position: command.position,
    activityKind: command.activityKind,
    activityName: command.activityName,
    semanticInput: command.semanticInput,
    suspension: create(ActivitySuspensionSchema, {
      reason: FRAMEWORK_INTERRUPT_REASON,
      context: {
        value: interrupt.value,
      } as JsonObject,
    }),
  });
}

function validateSettledEntries(
  attempt: AttemptContext,
  requested: readonly StepSuspensionEntry[],
  entries: readonly StepActivityEntry[],
): Map<string, NonNullable<StepActivityEntry["outcome"]>> {
  if (entries.length !== requested.length) {
    throw new UnsupportedDurableGraphError(
      "durable interrupt settlement returned an incomplete frontier",
    );
  }
  const requestedPositions = new Set(
    requested.map((entry) => positionKey(entry.position)),
  );
  const outcomes = new Map<string, NonNullable<StepActivityEntry["outcome"]>>();
  for (const entry of entries) {
    const key = positionKey(entry.position);
    if (!requestedPositions.has(key) || outcomes.has(key)) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt settlement returned an unknown or duplicate position",
      );
    }
    if (entry.outcome === undefined || entry.outcome.activityId.length === 0) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt settlement returned an invalid activity outcome",
      );
    }
    if (
      !attempt.workflowIdentity ||
      !entry.outcome.workflowIdentity ||
      !equals(
        WorkflowIdentitySchema,
        attempt.workflowIdentity,
        entry.outcome.workflowIdentity,
      )
    ) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt settlement returned a cross-workflow outcome",
      );
    }
    if (
      entry.outcome.attemptId !== attempt.attemptId ||
      entry.outcome.fencingToken !== attempt.fencingToken
    ) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt settlement returned a cross-attempt outcome",
      );
    }
    outcomes.set(key, entry.outcome);
  }
  const suspended = [...outcomes.values()].map(
    (outcome) => outcome.outcomeKind === ActivityOutcomeKind.SUSPENDED,
  );
  if (suspended.some(Boolean) && !suspended.every(Boolean)) {
    throw new UnsupportedDurableGraphError(
      "durable interrupt settlement mixed suspended and resolved outcomes",
    );
  }
  return outcomes;
}

export class DurableSession extends ExecutionSession {
  readonly resumed: boolean;
  private readonly attempt: AttemptContext;
  private readonly checkpointer: PlatformCheckpointer;
  private readonly executionId: string;
  private readonly resumeActivityIds: ReadonlySet<string>;
  private resumeFrontierSettled = false;

  constructor(
    graph: CompiledStateGraph<unknown, unknown>,
    ctx: RequestContext,
    input: AgentInput,
    options: SessionOptions,
    execution: DurableExecution,
  ) {
    super(graph, ctx, input, options);
    this.attempt = execution.attempt;
    this.checkpointer = execution.checkpointer;
    this.executionId = execution.executionId;
    this.durableSubgraphs?.validate();
    this.resumed = ctx.resume === true;
    if (this.resumed) {
      if (
        ctx.resumeData === null ||
        typeof ctx.resumeData !== "object" ||
        Array.isArray(ctx.resumeData)
      ) {
        throw new UnsupportedDurableGraphError(
          "durable resume requires an activity-id-keyed resume map",
        );
      }
      const activityIds = Object.keys(ctx.resumeData);
      if (
        activityIds.length === 0 ||
        activityIds.some((id) => id.length === 0)
      ) {
        throw new UnsupportedDurableGraphError(
          "durable resume requires a non-empty activity-id-keyed resume map",
        );
      }
      this.resumeActivityIds = new Set(activityIds);
    } else {
      this.resumeActivityIds = new Set();
    }
  }

  async prepareRun(): Promise<PreparedRun> {
    setActivityReconstructionIds([...this.resumeActivityIds]);
    // Replacement attempts rebuild the original turn, including tenant hooks.
    const prepareContext =
      this.attempt.replayMode === true && this.prepareInput !== null
        ? {
            ...this.ctx,
            resume: false,
            resumeData: null,
            metadata: null,
          }
        : this.ctx;
    const config = {
      ...this.baseConfig(),
      durability: "sync",
    } as RunnableConfig;
    const freshInput = this.freshGraphInput(prepareContext);
    if (isCommand(freshInput)) {
      throw new UnsupportedDurableGraphError(
        "Command input is not supported on durable_workflow sessions",
      );
    }
    // Preserve Python's unindexed default message and indexed hook messages.
    const graphInput = assignMissingGraphInputMessageIds(
      freshInput,
      this.prepareInput === null
        ? () => `durable-input:${this.executionId}`
        : (index) => `durable-input:${this.executionId}:${index}`,
    );
    await this.checkpointer.seedPreviousState(this.attempt, config);
    return { graphInput, config };
  }

  async invokeInterrupt(
    config: RunnableConfig,
    response: string,
    messages: readonly unknown[],
  ): Promise<Command | AgentOutput | null> {
    const result = await this.settlePendingInterrupts(config, []);
    if (result === null || result instanceof Command) return result;
    return invokeSuspendOutput({
      response,
      threadId: this.threadId,
      checkpointId: undefined,
      interruptValues: result.map((item) => item.value),
      interrupts: result,
      messageCount: messages.length,
      resumed: this.resumed,
    });
  }

  async streamInterrupt(
    config: RunnableConfig,
    captured: readonly Interrupt[],
    messages: readonly BaseMessage[],
  ): Promise<Command | StreamEvent | null> {
    const result = await this.settlePendingInterrupts(config, captured);
    if (result === null || result instanceof Command) return result;
    return streamHitlSuspendEvent({
      interrupts: result,
      checkpointId: undefined,
      resumed: this.resumed,
      messages,
    });
  }

  private async settlePendingInterrupts(
    config: RunnableConfig,
    captured: readonly Interrupt[],
  ): Promise<Command | PublicInterrupt[] | null> {
    const graph = this.graph as unknown as {
      getState(
        config: RunnableConfig,
        options?: { subgraphs?: boolean },
      ): Promise<StateSnapshotLike>;
    };
    const state = await graph.getState(config, { subgraphs: true });
    const pending = pendingInterrupts(state);
    if (pending.length === 0) {
      if ((state.next?.length ?? 0) > 0 || captured.length > 0) {
        throw new UnsupportedDurableGraphError(
          "durable workflow does not support LangGraph " +
            "interrupt_before or interrupt_after pauses",
        );
      }
      return null;
    }

    recordDirectInterrupts(this.attempt, state);
    const activitiesByNativeId = new Map<string, InterruptedActivity>();
    for (const activity of interruptedActivities(currentStepOrdinal())) {
      if (!isGraphInterrupt(activity.controlFlow)) continue;
      const native = activity.controlFlow.interrupts;
      if (native.length !== 1) {
        throw new UnsupportedDurableGraphError(
          "a durable interrupt activity must contain exactly one native interrupt",
        );
      }
      const id = native[0]?.id;
      if (!id || activitiesByNativeId.has(id)) {
        throw new UnsupportedDurableGraphError(
          "durable native interrupt has an invalid or duplicate identity",
        );
      }
      activitiesByNativeId.set(id, activity);
    }

    const interruptByPosition = new Map<string, PublicInterrupt>();
    const suspensions = pending.map((interrupt) => {
      const activity = activitiesByNativeId.get(interrupt.id);
      if (activity === undefined) {
        throw new UnsupportedDurableGraphError(
          "durable native interrupt checkpoint did not register an activity",
        );
      }
      const entry = suspensionEntry(activity, interrupt);
      const key = positionKey(entry.position);
      if (interruptByPosition.has(key)) {
        throw new UnsupportedDurableGraphError(
          "durable native interrupts resolved to a duplicate activity position",
        );
      }
      interruptByPosition.set(key, interrupt);
      return entry;
    });

    const oeUrl = getCurrentOeUrl();
    if (!oeUrl) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt requires an orchestration engine URL",
      );
    }
    const entries = await new WorkflowClient(oeUrl).finalizeStep(
      finalizeCurrentStepSuspensionsCommand(this.attempt, suspensions),
    );
    const outcomes = validateSettledEntries(this.attempt, suspensions, entries);
    const settled = entries.map((entry) => {
      const key = positionKey(entry.position);
      const interrupt = interruptByPosition.get(key);
      const outcome = outcomes.get(key);
      if (interrupt === undefined || outcome === undefined) {
        throw new UnsupportedDurableGraphError(
          "durable interrupt settlement could not join its activity position",
        );
      }
      return { interrupt, outcome };
    });

    const settledIds = new Set(
      settled.map(({ outcome }) => outcome.activityId),
    );
    if (settledIds.size !== settled.length) {
      throw new UnsupportedDurableGraphError(
        "durable interrupt settlement returned duplicate activity ids",
      );
    }

    if (settled[0]?.outcome.outcomeKind === ActivityOutcomeKind.SUSPENDED) {
      return interruptSnapshot(
        settled.map(({ interrupt, outcome }) => ({
          id: outcome.activityId,
          value: interrupt.value,
        })),
      );
    }

    const wrapper = getCurrentWrapper() as SecureToolWrapper | null;
    const durableMemory = wrapper?.durableMemory;

    // Earlier waits replay before the frontier named by this resume request.
    const targetsRequestedFrontier =
      this.resumed &&
      [...settledIds].some((id) => this.resumeActivityIds.has(id));
    if (
      targetsRequestedFrontier &&
      (settledIds.size !== this.resumeActivityIds.size ||
        [...settledIds].some((id) => !this.resumeActivityIds.has(id)))
    ) {
      throw new UnsupportedDurableGraphError(
        "durable resume map does not match the resolved suspension frontier",
      );
    }
    if (targetsRequestedFrontier) {
      this.resumeFrontierSettled = true;
    }

    // Sync only after the frontier is accepted: a rejected resume must not
    // leave durable Memory side effects behind.
    if (durableMemory) {
      const memoryClient = new WorkflowClient(oeUrl);
      for (const { interrupt, outcome } of settled) {
        const activity = activitiesByNativeId.get(interrupt.id);
        if (activity === undefined || !isDirectInterruptActivity(activity)) {
          continue;
        }
        await durableMemory.synchronizeTool(
          memoryClient,
          create(ActivityContextSchema, {
            workflowIdentity: outcome.workflowIdentity,
            activityId: outcome.activityId,
            attemptId: outcome.attemptId,
            fencingToken: outcome.fencingToken,
          }),
          unwrapActivityOutcome(outcome),
          getCurrentUserId(),
          `${DIRECT_INTERRUPT_ACTIVITY_NAME}:${outcome.activityId}`,
          DIRECT_INTERRUPT_ACTIVITY_NAME,
        );
      }
    }
    return new Command({
      resume: Object.fromEntries(
        settled.map(({ interrupt, outcome }) => [
          interrupt.id,
          unwrapActivityOutcome(outcome),
        ]),
      ),
    });
  }

  complete(): Promise<void> {
    if (this.resumed && !this.resumeFrontierSettled) {
      throw new UnsupportedDurableGraphError(
        "durable resume did not reconstruct its suspension frontier",
      );
    }
    return this.checkpointer.completeExecution(this.attempt);
  }

  async close(): Promise<void> {
    await this.checkpointer.releaseScratch(this.attempt);
  }

  override runInGraphScope<T>(fn: () => T): T {
    return this.durableSubgraphs === null
      ? fn()
      : runWithOperationPathResolver(this.durableSubgraphs.resolve, fn);
  }
}
