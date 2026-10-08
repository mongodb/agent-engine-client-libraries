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
  guardrailReviewWaitId,
  interruptedActivities,
  markInterruptedActivitiesAnswered,
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

import { rejectNodeRetryPolicies } from "./durable_subgraphs.js";
import { takeRaisedInterruptIds } from "./raised_interrupts.js";
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

type SnapshotTask = NonNullable<StateSnapshotLike["tasks"]>[number];

/**
 * The direct interrupts one root task is waiting on, including those raised
 * by tasks of a compiled child it runs. LangGraph JS projects only one of a
 * child's parallel interrupts to the parent task, so the child's own snapshot
 * is the complete source. Children come first, in a path order that is the
 * same on every attempt, and each native id appears once.
 */
function taskInterrupts(task: SnapshotTask): PublicInterrupt[] {
  const collected: Interrupt[] = [];
  const visit = (current: SnapshotTask): void => {
    const state = current.state;
    if (
      state !== undefined &&
      state !== null &&
      typeof state === "object" &&
      "tasks" in state
    ) {
      // Ordinals need the same order on every attempt, not a meaningful one,
      // so a string comparison of the paths is enough.
      const children = [...((state as StateSnapshotLike).tasks ?? [])].sort(
        (left, right) => {
          const leftPath = JSON.stringify(left.path ?? []);
          const rightPath = JSON.stringify(right.path ?? []);
          return leftPath < rightPath ? -1 : leftPath > rightPath ? 1 : 0;
        },
      );
      for (const child of children) visit(child);
    }
    collected.push(...(current.interrupts ?? []));
  };
  visit(task);
  return uniqueInterrupts(collected);
}

function pendingInterrupts(state: StateSnapshotLike): PublicInterrupt[] {
  // A parent snapshot can repeat a subgraph interrupt, so keep each native id
  // once across the root tasks.
  return uniqueInterrupts(
    (state.tasks ?? []).flatMap((task) => taskInterrupts(task)),
  );
}

function isDirectInterruptActivity(activity: InterruptedActivity): boolean {
  const segments = activity.command.position?.operationPath?.segments ?? [];
  return (
    activity.command.activityName === DIRECT_INTERRUPT_ACTIVITY_NAME &&
    segments.length === 1 &&
    segments[0]?.name.startsWith("langgraph.task:") === true
  );
}

function directInterruptId(activity: InterruptedActivity): string | undefined {
  return isGraphInterrupt(activity.controlFlow)
    ? activity.controlFlow.interrupts[0]?.id
    : undefined;
}

function recordDirectInterrupts(
  attempt: AttemptContext,
  state: StateSnapshotLike,
  isLeftover: (interrupt: PublicInterrupt) => boolean,
): void {
  const stepOrdinal = currentStepOrdinal();
  const waiting = interruptedActivities(stepOrdinal);
  const waitingKeys = new Set(
    waiting.map((activity) => positionKey(activity.command.position)),
  );
  const localIds = new Set(
    waiting
      .filter((activity) => !isDirectInterruptActivity(activity))
      .flatMap((activity) =>
        isGraphInterrupt(activity.controlFlow)
          ? activity.controlFlow.interrupts.map((interrupt) => interrupt.id)
          : [],
      ),
  );
  // Registered local tools already carry their admitted ActivityCommand.
  // Do not synthesize a second root interrupt activity for the checkpoint view.
  // Answered pauses stay listed so a task that pauses again takes the next
  // ordinal: every pause in one task shares one native id.
  const recordedByTask = new Map<string, InterruptedActivity[]>();
  for (const activity of interruptedActivities(stepOrdinal, {
    includeAnswered: true,
  }).filter(isDirectInterruptActivity)) {
    const segment =
      activity.command.position?.operationPath?.segments[0]?.name ?? "";
    recordedByTask.set(segment, [
      ...(recordedByTask.get(segment) ?? []),
      activity,
    ]);
  }
  for (const task of state.tasks ?? []) {
    const owned = taskInterrupts(task);
    if (owned.length === 0) continue;
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
    const direct = owned.filter(
      (interrupt) => !localIds.has(interrupt.id) && !isLeftover(interrupt),
    );
    const segmentName = `langgraph.task:${JSON.stringify(path)}`;
    const recorded = recordedByTask.get(segmentName) ?? [];
    for (const interrupt of direct) {
      const waitingHere = recorded.filter((activity) =>
        waitingKeys.has(positionKey(activity.command.position)),
      );
      const same = waitingHere.find(
        (activity) => directInterruptId(activity) === interrupt.id,
      );
      if (same !== undefined) {
        if (
          !isGraphInterrupt(same.controlFlow) ||
          !isDeepStrictEqual(
            same.controlFlow.interrupts[0]?.value,
            interrupt.value,
          )
        ) {
          throw new UnsupportedDurableGraphError(
            "LangGraph returned conflicting values for a waiting durable interrupt",
          );
        }
        continue;
      }
      const command = buildActivityCommand({
        attempt,
        kind: ActivityKind.TOOL,
        name: DIRECT_INTERRUPT_ACTIVITY_NAME,
        // `recorded` gains this activity below, so the next pause takes the
        // next ordinal.
        activityOrdinal: recorded.length + 1,
        operationPath: create(OperationPathSchema, {
          segments: [
            create(OperationPathSegmentSchema, {
              name: segmentName,
              ordinal: 1n,
            }),
          ],
        }),
        semanticInput: { value: interrupt.value },
      });
      const controlFlow = new GraphInterrupt([interrupt]);
      if (command.position === undefined) {
        throw new UnsupportedDurableGraphError(
          "durable interrupt requires an activity position",
        );
      }
      recordObservedActivity(command.position);
      recordInterruptedActivity(command, controlFlow);
      const activity = { command, controlFlow };
      recorded.push(activity);
      recordedByTask.set(segmentName, recorded);
      waitingKeys.add(positionKey(command.position));
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
    rejectNodeRetryPolicies(graph);
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
    raised: readonly Interrupt[],
  ): Promise<Command | AgentOutput | null> {
    const result = await this.settlePendingInterrupts(config, raised);
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
    // LangGraph's state keeps the interrupt of a task that has since received
    // its answer and finished, until the whole superstep completes. Such a
    // leftover carries only an answered pause's id and did not fire in the
    // latest run; a task that paused again did.
    const raisedIds = new Set([
      ...captured.map((interrupt) => interrupt.id),
      ...takeRaisedInterruptIds(this.attempt),
    ]);
    const waitingIds = new Set(
      interruptedActivities(currentStepOrdinal())
        .map(directInterruptId)
        .filter((id): id is string => id !== undefined),
    );
    const answeredIds = new Set(
      interruptedActivities(currentStepOrdinal(), { includeAnswered: true })
        .map(directInterruptId)
        .filter((id) => id !== undefined && !waitingIds.has(id)),
    );
    const isLeftover = (interrupt: PublicInterrupt): boolean =>
      answeredIds.has(interrupt.id) && !raisedIds.has(interrupt.id);
    const reported = pendingInterrupts(state);
    const pending = reported.filter((interrupt) => !isLeftover(interrupt));
    if (reported.length > 0 && pending.length === 0) {
      throw new UnsupportedDurableGraphError(
        "durable graph reported only already-answered interrupts",
      );
    }
    if (pending.length === 0) {
      if ((state.next?.length ?? 0) > 0 || captured.length > 0) {
        throw new UnsupportedDurableGraphError(
          "durable workflow does not support LangGraph " +
            "interrupt_before or interrupt_after pauses",
        );
      }
      return null;
    }

    recordDirectInterrupts(this.attempt, state, isLeftover);
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
        const context = create(ActivityContextSchema, {
          workflowIdentity: outcome.workflowIdentity,
          activityId: outcome.activityId,
          attemptId: outcome.attemptId,
          fencingToken: outcome.fencingToken,
        });
        if (guardrailReviewWaitId(interrupt.value) !== null) {
          // The answer is the platform's review decision, not something the
          // user or a tool said.
          await durableMemory.acknowledge(memoryClient, context);
          continue;
        }
        await durableMemory.synchronizeTool(
          memoryClient,
          context,
          unwrapActivityOutcome(outcome),
          getCurrentUserId(),
          `${DIRECT_INTERRUPT_ACTIVITY_NAME}:${outcome.activityId}`,
          DIRECT_INTERRUPT_ACTIVITY_NAME,
        );
      }
    }
    markInterruptedActivitiesAnswered(
      settled.flatMap(({ interrupt }) => {
        const position = activitiesByNativeId.get(interrupt.id)?.command
          .position;
        return position === undefined ? [] : [position];
      }),
    );
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
