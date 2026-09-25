/** Serial activity helpers for OE-owned durable workflow identity. */

import { create, fromJson, toJson, type JsonValue } from "@bufbuild/protobuf";
import { ValueSchema, type Value } from "@bufbuild/protobuf/wkt";

import {
  ActivityCommandSchema,
  ActivityContextSchema,
  ActivityOutcomeKind,
  ActivityOutcomeSchema,
  type ActivityCommand,
  type ActivityContext,
  type ActivityKind,
  type ActivityOutcome,
} from "../generated/workflow/v1/activity_pb.js";
import {
  ActivityPositionSchema,
  OperationPathSchema,
  WorkflowErrorCode,
  WorkflowErrorSchema,
  type OperationPath,
} from "../generated/workflow/v1/common_pb.js";
import type { AttemptContext } from "../generated/workflow/v1/runtime_pb.js";
import { getLogger } from "../logger.js";
import {
  ActivityReplay,
  WorkflowClientError,
  type StartActivityResult,
  type WorkflowClient,
} from "./client.js";
import {
  currentAttemptContext,
  currentOperationPath,
  currentStepOrdinal,
  activityRequiresReconstruction,
  recordInterruptedActivity,
  recordObservedActivity,
  recordReconstructedActivityInterrupt,
} from "./context.js";

type ActivityRuntimeClient = Pick<
  WorkflowClient,
  "startActivity" | "reportOutcome"
>;

// Post-terminal hook fired after an activity resolves on fresh, replay, and
// reconstructed paths. Mirrors Python's ActivityResolvedHook.
export type ActivityResolvedHook<T = unknown> = (
  client: ActivityRuntimeClient,
  context: ActivityContext,
  result: T,
) => void | Promise<void>;

const logger = getLogger("agent_engine_runner_shared.workflow.activity");

export class DurableActivityDeniedError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "DurableActivityDeniedError";
  }
}

export class DurableActivityInterrupted extends Error {
  constructor(readonly controlFlow: unknown) {
    super(
      controlFlow instanceof Error ? controlFlow.message : String(controlFlow),
    );
    this.name = "DurableActivityInterrupted";
  }
}

export class ReplayedActivityFailedError extends WorkflowClientError {
  constructor(code: WorkflowErrorCode, message: string) {
    super(code, message);
    this.name = "ReplayedActivityFailedError";
  }
}

export function semanticInputFromJson(value: unknown): Value {
  const encoded = JSON.stringify(value === undefined ? null : value);
  if (encoded === undefined) {
    throw new TypeError("semantic input is not JSON-serializable");
  }
  return fromJson(ValueSchema, JSON.parse(encoded) as JsonValue);
}

export function valueToJson(value: Value | undefined): unknown {
  if (value === undefined) return null;
  return toJson(ValueSchema, value);
}

export function buildActivityCommand(args: {
  attempt: AttemptContext;
  kind: ActivityKind;
  name: string;
  activityOrdinal: number;
  semanticInput: unknown;
  stepOrdinal?: number;
  operationPath?: OperationPath;
}): ActivityCommand {
  if (!args.attempt.workflowIdentity?.executionId) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      "ActivityCommand requires AttemptContext.workflow_identity",
    );
  }
  if (args.activityOrdinal <= 0) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      "activity_ordinal must be positive",
    );
  }
  const resolvedStep =
    args.stepOrdinal === undefined ? currentStepOrdinal() : args.stepOrdinal;
  if (resolvedStep <= 0) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      "step_ordinal must be positive",
    );
  }
  const operationPath = args.operationPath ?? currentOperationPath();
  return create(ActivityCommandSchema, {
    workflowIdentity: args.attempt.workflowIdentity,
    attemptId: args.attempt.attemptId,
    fencingToken: args.attempt.fencingToken,
    position: create(ActivityPositionSchema, {
      operationPath: create(OperationPathSchema, {
        segments: operationPath.segments,
      }),
      activityOrdinal: BigInt(args.activityOrdinal),
      stepOrdinal: BigInt(resolvedStep),
    }),
    activityKind: args.kind,
    activityName: args.name,
    semanticInput: semanticInputFromJson(args.semanticInput),
  });
}

export function completedOutcome(
  context: ActivityContext,
  result: unknown,
): ActivityOutcome {
  return create(ActivityOutcomeSchema, {
    workflowIdentity: context.workflowIdentity,
    activityId: context.activityId,
    attemptId: context.attemptId,
    fencingToken: context.fencingToken,
    outcomeKind: ActivityOutcomeKind.COMPLETED,
    result: semanticInputFromJson(result),
  });
}

export function failedOutcome(
  context: ActivityContext,
  message: string,
  code: WorkflowErrorCode = WorkflowErrorCode.OUTCOME_UNKNOWN,
): ActivityOutcome {
  return create(ActivityOutcomeSchema, {
    workflowIdentity: context.workflowIdentity,
    activityId: context.activityId,
    attemptId: context.attemptId,
    fencingToken: context.fencingToken,
    outcomeKind: ActivityOutcomeKind.FAILED,
    error: create(WorkflowErrorSchema, { code, message }),
  });
}

function contextFromOutcome(outcome: ActivityOutcome): ActivityContext {
  return create(ActivityContextSchema, {
    workflowIdentity: outcome.workflowIdentity,
    activityId: outcome.activityId,
    attemptId: outcome.attemptId,
    fencingToken: outcome.fencingToken,
  });
}

function deniedOutcome(
  context: ActivityContext,
  message: string,
): ActivityOutcome {
  return create(ActivityOutcomeSchema, {
    workflowIdentity: context.workflowIdentity,
    activityId: context.activityId,
    attemptId: context.attemptId,
    fencingToken: context.fencingToken,
    outcomeKind: ActivityOutcomeKind.DENIED,
    // OE WorkflowError.Validate rejects UNSPECIFIED; denied fixtures use UNAUTHORIZED.
    error: create(WorkflowErrorSchema, {
      code: WorkflowErrorCode.UNAUTHORIZED,
      message,
    }),
  });
}

export function unwrapActivityOutcome(outcome: ActivityOutcome): unknown {
  switch (outcome.outcomeKind) {
    case ActivityOutcomeKind.COMPLETED:
      return valueToJson(outcome.result);
    case ActivityOutcomeKind.DENIED:
      throw new DurableActivityDeniedError(
        outcome.error?.message || "durable activity denied by policy",
      );
    case ActivityOutcomeKind.FAILED:
      throw new ReplayedActivityFailedError(
        outcome.error?.code || WorkflowErrorCode.OUTCOME_UNKNOWN,
        outcome.error?.message || "durable activity failed",
      );
    case ActivityOutcomeKind.SUSPENDED:
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "shared durable activities support terminal outcomes only; " +
          "framework adapters must own suspension and resume",
      );
    case ActivityOutcomeKind.UNSPECIFIED:
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "activity outcome kind is unspecified",
      );
    default: {
      const _exhaustive: never = outcome.outcomeKind;
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        `activity outcome kind is unspecified: ${_exhaustive}`,
      );
    }
  }
}

interface AttemptActivityGate {
  exclusive: boolean;
  shared: number;
}

const attemptActivityGates = new Map<string, AttemptActivityGate>();

const CONFLICT_MESSAGE = "this attempt already has an in-flight activity";

/**
 * Admit one activity under the attempt-local concurrency gate. Activities with
 * stable position identity may share the attempt; unkeyed effects are exclusive.
 */
function acquireActivitySlot(
  attemptId: string,
  exclusive: boolean,
): () => void {
  const gate = attemptActivityGates.get(attemptId) ?? {
    exclusive: false,
    shared: 0,
  };
  if (exclusive ? gate.exclusive || gate.shared > 0 : gate.exclusive) {
    throw new WorkflowClientError(WorkflowErrorCode.CONFLICT, CONFLICT_MESSAGE);
  }
  if (exclusive) gate.exclusive = true;
  else gate.shared += 1;
  attemptActivityGates.set(attemptId, gate);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    const current = attemptActivityGates.get(attemptId);
    if (current === undefined) return;
    if (exclusive) current.exclusive = false;
    else current.shared = Math.max(0, current.shared - 1);
    if (!current.exclusive && current.shared === 0) {
      attemptActivityGates.delete(attemptId);
    }
  };
}

async function withActivitySlot<T>(
  attemptId: string,
  exclusive: boolean,
  fn: () => Promise<T>,
): Promise<T> {
  const release = acquireActivitySlot(attemptId, exclusive);
  try {
    return await fn();
  } finally {
    release();
  }
}

async function reportDenied(
  client: ActivityRuntimeClient,
  context: ActivityContext,
  error: DurableActivityDeniedError,
): Promise<void> {
  try {
    await client.reportOutcome(deniedOutcome(context, error.message));
  } catch (reportError) {
    if (reportError instanceof WorkflowClientError) {
      logger.error(
        "Failed to report denied activity outcome",
        error,
        reportError,
      );
      throw reportError;
    }
    logger.error("Failed to report denied activity outcome", reportError);
  }
}

async function reportFailure(
  client: ActivityRuntimeClient,
  context: ActivityContext,
  error: unknown,
): Promise<void> {
  const message = error instanceof Error ? error.message : String(error);
  try {
    await client.reportOutcome(failedOutcome(context, message));
  } catch (reportError) {
    if (reportError instanceof WorkflowClientError) {
      logger.error(
        "Failed to report failed activity outcome",
        error,
        reportError,
      );
      throw reportError;
    }
    logger.error("Failed to report failed activity outcome", reportError);
  }
}

function requireAttempt(
  attempt: AttemptContext | undefined,
  caller: string,
): AttemptContext {
  const resolved = attempt ?? currentAttemptContext() ?? undefined;
  if (resolved === undefined) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      `${caller} requires an OE-issued AttemptContext`,
    );
  }
  return resolved;
}

export async function runSerialActivity<T>(args: {
  client: ActivityRuntimeClient;
  kind: ActivityKind;
  name: string;
  activityOrdinal: number;
  semanticInput: unknown;
  execute: (context: ActivityContext) => T | Promise<T>;
  onActivityResolved?: ActivityResolvedHook<T>;
  attempt?: AttemptContext;
  stepOrdinal?: number;
  /** False only when the framework has assigned a stable activity identity. */
  exclusive?: boolean;
}): Promise<T> {
  const attempt = requireAttempt(args.attempt, "runSerialActivity");
  return withActivitySlot(
    attempt.attemptId,
    args.exclusive ?? true,
    async () => {
      const command = buildActivityCommand({
        attempt,
        kind: args.kind,
        name: args.name,
        activityOrdinal: args.activityOrdinal,
        stepOrdinal: args.stepOrdinal,
        semanticInput: args.semanticInput,
      });
      return runDispatchedActivity(
        args.client,
        command,
        args.execute,
        args.onActivityResolved,
      );
    },
  );
}

type StreamingItems<T> = Iterable<T> | AsyncIterable<T>;

/**
 * Run one generator-shaped durable activity.
 *
 * Replay expands a recorded result back into stream items without executing
 * the effect. A fresh dispatch yields live items immediately, then folds them
 * into the single terminal result recorded by OE. Closing the stream early
 * terminal-fails the dispatch instead of leaving it in flight.
 */
export async function* runStreamingActivity<T>(args: {
  client: ActivityRuntimeClient;
  kind: ActivityKind;
  name: string;
  activityOrdinal: number;
  semanticInput: unknown;
  execute: () => StreamingItems<T>;
  replay: (result: unknown) => StreamingItems<T>;
  fold: (items: T[]) => unknown | Promise<unknown>;
  onActivityResolved?: ActivityResolvedHook;
  attempt?: AttemptContext;
  stepOrdinal?: number;
  /** False when the caller has assigned a stable activity identity. */
  exclusive?: boolean;
}): AsyncGenerator<T> {
  const attempt = requireAttempt(args.attempt, "runStreamingActivity");
  const release = acquireActivitySlot(
    attempt.attemptId,
    args.exclusive ?? true,
  );
  try {
    const command = buildActivityCommand({
      attempt,
      kind: args.kind,
      name: args.name,
      activityOrdinal: args.activityOrdinal,
      stepOrdinal: args.stepOrdinal,
      semanticInput: args.semanticInput,
    });
    const started = await args.client.startActivity(command);
    if (command.position !== undefined) {
      recordObservedActivity(command.position);
    }
    if (started instanceof ActivityReplay) {
      const result = unwrapActivityOutcome(started.outcome);
      await args.onActivityResolved?.(
        args.client,
        contextFromOutcome(started.outcome),
        result,
      );
      for await (const item of args.replay(result)) {
        yield item;
      }
      return;
    }

    yield* runDispatchedActivityStream(
      args.client,
      started.context,
      args.execute,
      args.fold,
      args.onActivityResolved,
    );
  } finally {
    release();
  }
}

async function* runDispatchedActivityStream<T>(
  client: ActivityRuntimeClient,
  context: ActivityContext,
  execute: () => StreamingItems<T>,
  fold: (items: T[]) => unknown | Promise<unknown>,
  onActivityResolved?: ActivityResolvedHook,
): AsyncGenerator<T> {
  let terminalReportStarted = false;
  try {
    const items: T[] = [];
    try {
      for await (const item of execute()) {
        items.push(item);
        yield item;
      }
    } catch (error) {
      terminalReportStarted = true;
      if (error instanceof DurableActivityDeniedError) {
        await reportDenied(client, context, error);
      } else {
        await reportFailure(client, context, error);
      }
      throw error;
    }

    let outcome: ActivityOutcome;
    let result: unknown;
    try {
      result = await fold(items);
      outcome = completedOutcome(context, result);
    } catch (error) {
      terminalReportStarted = true;
      await reportFailure(client, context, error);
      throw error;
    }

    terminalReportStarted = true;
    await client.reportOutcome(outcome);
    await onActivityResolved?.(client, context, result);
  } finally {
    // Closing an async generator stops its producer at the current yield. The
    // dispatched effect must still receive a terminal outcome in that case.
    if (!terminalReportStarted) {
      await client.reportOutcome(
        failedOutcome(context, "stream abandoned before completion"),
      );
    }
  }
}

async function runDispatchedActivity<T>(
  client: ActivityRuntimeClient,
  command: ActivityCommand,
  execute: (context: ActivityContext) => T | Promise<T>,
  onActivityResolved?: ActivityResolvedHook<T>,
): Promise<T> {
  const started: StartActivityResult = await client.startActivity(command);
  if (command.position !== undefined) {
    recordObservedActivity(command.position);
  }
  if (started instanceof ActivityReplay) {
    if (!activityRequiresReconstruction(started.outcome.activityId)) {
      const result = unwrapActivityOutcome(started.outcome) as T;
      await onActivityResolved?.(
        client,
        contextFromOutcome(started.outcome),
        result,
      );
      return result;
    }
  }
  const reconstructing = started instanceof ActivityReplay;
  const context = reconstructing
    ? contextFromOutcome(started.outcome)
    : started.context;

  let result: T;
  try {
    result = await execute(context);
  } catch (error) {
    if (error instanceof DurableActivityDeniedError) {
      if (!reconstructing) await reportDenied(client, context, error);
      throw error;
    }
    if (error instanceof DurableActivityInterrupted) {
      if (
        reconstructing &&
        recordReconstructedActivityInterrupt(context.activityId) > 1
      ) {
        throw new Error(
          "a durable local tool cannot raise another native interrupt " +
            "after its recorded answer is resumed",
          { cause: error },
        );
      }
      recordInterruptedActivity(command, error.controlFlow);
      throw error.controlFlow;
    }
    if (!reconstructing) await reportFailure(client, context, error);
    throw error;
  }

  if (reconstructing) {
    await onActivityResolved?.(client, context, result);
    return result;
  }

  let outcome: ActivityOutcome;
  try {
    outcome = completedOutcome(context, result);
  } catch (error) {
    await reportFailure(client, context, error);
    throw error;
  }
  await client.reportOutcome(outcome);
  await onActivityResolved?.(client, context, result);
  return result;
}
