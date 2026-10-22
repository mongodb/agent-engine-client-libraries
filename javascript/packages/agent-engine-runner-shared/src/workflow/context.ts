/** Request-local durable attempt identity, positions, and observed activities. */

import { AsyncLocalStorage } from "node:async_hooks";
import { create, fromJson, toJson } from "@bufbuild/protobuf";

import {
  ActivityCommandSchema,
  type ActivityCommand,
} from "../generated/workflow/v1/activity_pb.js";
import {
  ActivityPositionSchema,
  OperationPathSchema,
  OperationPathSegmentSchema,
  type ActivityPosition,
  type OperationPath,
} from "../generated/workflow/v1/common_pb.js";
import type { AttemptContext } from "../generated/workflow/v1/runtime_pb.js";
import { CancelledHistoryState } from "./cancelled_history.js";

/**
 * Mutable holder so detached work created inside the scoped callback cannot
 * keep using the OE-issued attempt after the request settles. AsyncLocalStorage
 * restores the surrounding caller when `fn` settles, but timers and floating
 * promises retain the stored object. Closing the holder is the Node equivalent
 * of Python's `reset_attempt_context` in `finally`.
 */
interface AttemptContextHolder {
  attempt: AttemptContext | null;
  step: { value: number };
  activityOrdinals: ActivityOrdinalAllocator;
  childPaths: ChildOperationPathAllocator;
  pendingChildBatch: { boundaries: ChildOperationBoundary[] };
  observed: ObservedActivityPositions;
  interrupted: InterruptedActivities;
  reconstruction: ActivityReconstruction;
  cancelledHistory: CancelledHistoryState;
  guardrailReviewWaits: GuardrailReviewWaits;
}

const storage = new AsyncLocalStorage<AttemptContextHolder>();

/**
 * Framework-neutral identity for one nested operation.
 *
 * `spawnedTask` marks a task the framework started at runtime on behalf of its
 * enclosing graph, such as a LangGraph `Send` task. A keyed activity inside
 * one may take the position its key was preallocated at elsewhere in that
 * enclosing graph; see `allocateActivityPosition`.
 */
export interface ChildOperationBoundary {
  name: string;
  occurrenceKey: string;
  spawnedTask?: boolean;
}

export type OperationPathResolver = () => readonly ChildOperationBoundary[];

/**
 * Nested operation-path resolvers are a ContextVar analog: they must follow
 * the async chain, not a mutable field on the shared attempt holder. Sibling
 * `resolverStorage.run` scopes then keep distinct paths after `await`.
 */
const resolverStorage = new AsyncLocalStorage<OperationPathResolver>();

export class UnsupportedChildOperationFanOutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UnsupportedChildOperationFanOutError";
  }
}

/** One key was preallocated at two positions a spawned task could take. */
export class AmbiguousActivityKeyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AmbiguousActivityKeyError";
  }
}

type PathKey = string;

function pathKeyFromSegments(
  segments: readonly { name: string; ordinal: bigint | number }[],
): PathKey {
  return JSON.stringify(
    segments.map((segment) => [segment.name, Number(segment.ordinal)]),
  );
}

function pathSegments(path: PathKey): [string, number][] {
  return JSON.parse(path) as [string, number][];
}

/** Whether `path` is `scope` or lies inside it. */
function within(path: PathKey, scope: PathKey): boolean {
  const scopeSegments = pathSegments(scope);
  return (
    JSON.stringify(pathSegments(path).slice(0, scopeSegments.length)) === scope
  );
}

function clonePosition(position: ActivityPosition): ActivityPosition {
  return fromJson(
    ActivityPositionSchema,
    toJson(ActivityPositionSchema, position),
  );
}

function activityPositionKey(
  position: ActivityPosition,
): [number, PathKey, number] {
  return [
    Number(position.stepOrdinal),
    pathKeyFromSegments(position.operationPath?.segments ?? []),
    Number(position.activityOrdinal),
  ];
}

function compareObservedPositions(
  left: ActivityPosition,
  right: ActivityPosition,
): number {
  const leftSegments = left.operationPath?.segments ?? [];
  const rightSegments = right.operationPath?.segments ?? [];
  const shared = Math.min(leftSegments.length, rightSegments.length);
  for (let index = 0; index < shared; index += 1) {
    const leftSegment = leftSegments[index];
    const rightSegment = rightSegments[index];
    if (leftSegment === undefined || rightSegment === undefined) {
      break;
    }
    if (leftSegment.name < rightSegment.name) return -1;
    if (leftSegment.name > rightSegment.name) return 1;
    const ordinalCmp =
      Number(leftSegment.ordinal) - Number(rightSegment.ordinal);
    if (ordinalCmp !== 0) return ordinalCmp;
  }
  if (leftSegments.length !== rightSegments.length) {
    return leftSegments.length - rightSegments.length;
  }
  return Number(left.activityOrdinal) - Number(right.activityOrdinal);
}

class ObservedActivityPositions {
  private readonly positions = new Map<string, ActivityPosition>();

  record(position: ActivityPosition): void {
    const key = JSON.stringify(activityPositionKey(position));
    this.positions.set(key, clonePosition(position));
  }

  forStep(stepOrdinal: number): ActivityPosition[] {
    return [...this.positions.values()]
      .filter((position) => Number(position.stepOrdinal) === stepOrdinal)
      .sort(compareObservedPositions);
  }
}

export interface InterruptedActivity {
  readonly command: ActivityCommand;
  readonly controlFlow: unknown;
}

class InterruptedActivities {
  private readonly items: InterruptedActivity[] = [];
  private readonly answered = new Set<string>();

  record(command: ActivityCommand, controlFlow: unknown): void {
    // Settlement needs the command as admitted, even if the caller mutates it.
    this.items.push({
      command: fromJson(
        ActivityCommandSchema,
        toJson(ActivityCommandSchema, command),
      ),
      controlFlow,
    });
  }

  markAnswered(positions: readonly ActivityPosition[]): void {
    for (const position of positions) {
      this.answered.add(JSON.stringify(activityPositionKey(position)));
    }
  }

  forStep(
    stepOrdinal: number,
    includeAnswered: boolean,
  ): InterruptedActivity[] {
    return this.items.filter(
      (item) =>
        item.command.position?.stepOrdinal === BigInt(stepOrdinal) &&
        (includeAnswered ||
          !this.answered.has(
            JSON.stringify(activityPositionKey(item.command.position)),
          )),
    );
  }
}

/**
 * The reviews this attempt paused for, as recorded by the platform's own
 * halts. A pause is a review wait because the platform recorded the halt
 * behind it, never because of what its value looks like: an application may
 * pause with a value of any shape.
 */
class GuardrailReviewWaits {
  private readonly reviewIds = new Set<string>();

  note(reviewId: string): void {
    this.reviewIds.add(reviewId);
  }

  has(reviewId: string): boolean {
    return this.reviewIds.has(reviewId);
  }
}

class ActivityReconstruction {
  private activityIds = new Set<string>();
  private readonly interruptCounts = new Map<string, number>();

  replace(activityIds: readonly string[]): void {
    this.activityIds = new Set(activityIds);
  }

  contains(activityId: string): boolean {
    return this.activityIds.has(activityId);
  }

  recordInterrupt(activityId: string): number {
    if (!this.activityIds.has(activityId)) {
      throw new Error(`activity "${activityId}" is not being reconstructed`);
    }
    const count = (this.interruptCounts.get(activityId) ?? 0) + 1;
    this.interruptCounts.set(activityId, count);
    return count;
  }
}

class ActivityOrdinalAllocator {
  private readonly nextByPath = new Map<PathKey, number>();
  private readonly byPathAndKey = new Map<string, number>();
  private readonly preallocatedPathsByKey = new Map<string, PathKey[]>();
  // Key -> spawning scope -> owner, for keys a spawned task took, so a later
  // preallocation of the same key in that scope fails instead of having lost a
  // race it would win on another attempt. Kept per scope so a take in one
  // graph cannot hide a take in another.
  private readonly takenBySpawnedTask = new Map<
    string,
    Map<PathKey, PathKey>
  >();
  private readonly callsByScope = new WeakMap<object, number>();

  /** Count calls within one scope object, starting at 1. */
  nextInScope(scope: object): number {
    const count = (this.callsByScope.get(scope) ?? 0) + 1;
    this.callsByScope.set(scope, count);
    return count;
  }

  preallocate(path: PathKey, keys: readonly string[]): number[] {
    return keys.map((key) => {
      if (key === "") throw new Error("activity ordinal key must be non-empty");
      for (const [scope, owner] of this.takenBySpawnedTask.get(key) ?? []) {
        if (path !== owner && within(path, scope)) {
          throw new AmbiguousActivityKeyError(
            `activity key ${JSON.stringify(key)} was preallocated at two ` +
              "positions in one graph; tool call ids must be unique within a turn",
          );
        }
      }
      if (!this.byPathAndKey.has(JSON.stringify([path, key]))) {
        const paths = this.preallocatedPathsByKey.get(key) ?? [];
        paths.push(path);
        this.preallocatedPathsByKey.set(key, paths);
      }
      return this.allocate(path, key);
    });
  }

  /**
   * Return the path and ordinal for `key`, preferring where it was preallocated.
   *
   * Only an activity inside a spawned task (`spawningScope` is the path of the
   * graph that spawned it) may move: a tool a model requested and that then
   * runs in its own LangGraph `Send` task keeps the model's position. The key
   * must have been preallocated exactly once within that scope, even when the
   * tool runs in the task that preallocated it. Any other activity is
   * allocated in `path`, so paths that contain no spawned task record exactly
   * as they did before spawned tasks existed.
   */
  allocatePosition(
    path: PathKey,
    key: string | undefined,
    spawningScope: PathKey | null,
  ): [PathKey, number] {
    if (key !== undefined && key !== "" && spawningScope !== null) {
      // A key may legitimately be preallocated once per spawning graph, so
      // only owners inside this task's spawning graph compete for it.
      const owners = (this.preallocatedPathsByKey.get(key) ?? []).filter(
        (candidate) => within(candidate, spawningScope),
      );
      if (owners.length > 1) {
        throw new AmbiguousActivityKeyError(
          `activity key ${JSON.stringify(key)} was preallocated at ` +
            `${owners.length} positions in one graph; tool call ids must be ` +
            "unique within a turn",
        );
      }
      const owner = owners[0];
      if (owner !== undefined) {
        const takes = this.takenBySpawnedTask.get(key) ?? new Map();
        takes.set(spawningScope, owner);
        this.takenBySpawnedTask.set(key, takes);
        return [owner, this.allocate(owner, key)];
      }
    }
    return [path, this.allocate(path, key)];
  }

  allocate(path: PathKey, key?: string): number {
    if (key !== undefined && key === "") {
      throw new Error("activity ordinal key must be non-empty");
    }
    if (key !== undefined) {
      const pathKey = JSON.stringify([path, key]);
      const existing = this.byPathAndKey.get(pathKey);
      if (existing !== undefined) return existing;
      const ordinal = this.nextByPath.get(path) ?? 1;
      this.nextByPath.set(path, ordinal + 1);
      this.byPathAndKey.set(pathKey, ordinal);
      return ordinal;
    }
    const ordinal = this.nextByPath.get(path) ?? 1;
    this.nextByPath.set(path, ordinal + 1);
    return ordinal;
  }
}

class ChildOperationPathAllocator {
  private readonly nextOrdinalByChild = new Map<string, number>();
  private readonly ordinalByOccurrence = new Map<string, number>();
  private readonly keysByStepChild = new Map<string, Set<string>>();
  private readonly spawnedTaskPaths = new Set<PathKey>();

  /**
   * Assign ordinals to a complete sibling batch in list order, before
   * concurrent same-name dispatch. Idempotent per occurrence key. The batch is
   * validated before any assignment so a malformed or duplicate sibling
   * identity cannot leave a partial reservation.
   */
  preallocate(
    path: PathKey,
    stepOrdinal: number,
    boundaries: readonly ChildOperationBoundary[],
  ): number[] {
    if (boundaries.length === 0) {
      throw new Error(
        "child operation name and occurrence key must be non-empty",
      );
    }
    const seen = new Set<string>();
    for (const boundary of boundaries) {
      if (!boundary.name || !boundary.occurrenceKey) {
        throw new Error(
          "child operation name and occurrence key must be non-empty",
        );
      }
      const identity = JSON.stringify([boundary.name, boundary.occurrenceKey]);
      if (seen.has(identity)) {
        throw new UnsupportedChildOperationFanOutError(
          `child '${boundary.name}' has duplicate occurrence ` +
            `'${boundary.occurrenceKey}' in the same batch`,
        );
      }
      seen.add(identity);
    }
    return boundaries.map((boundary) =>
      this.assign(path, stepOrdinal, boundary, true),
    );
  }

  resolve(
    boundaries: readonly ChildOperationBoundary[],
    stepOrdinal: number,
  ): OperationPath {
    const segments = [
      create(OperationPathSegmentSchema, { name: "agent", ordinal: 1n }),
    ];
    for (const boundary of boundaries) {
      const parent = pathKeyFromSegments(segments);
      const ordinal = this.assign(parent, stepOrdinal, boundary, false);
      segments.push(
        create(OperationPathSegmentSchema, {
          name: boundary.name,
          ordinal: BigInt(ordinal),
        }),
      );
      if (boundary.spawnedTask === true) {
        this.spawnedTaskPaths.add(pathKeyFromSegments(segments));
      }
    }
    return create(OperationPathSchema, { segments });
  }

  /** Return the path of the graph that spawned the innermost task in `path`. */
  spawningScope(path: PathKey): PathKey | null {
    const segments = pathSegments(path);
    for (let end = segments.length; end > 0; end -= 1) {
      if (this.spawnedTaskPaths.has(JSON.stringify(segments.slice(0, end)))) {
        return JSON.stringify(segments.slice(0, end - 1));
      }
    }
    return null;
  }

  private assign(
    parent: PathKey,
    stepOrdinal: number,
    boundary: ChildOperationBoundary,
    preallocating: boolean,
  ): number {
    if (!boundary.name || !boundary.occurrenceKey) {
      throw new Error(
        "child operation name and occurrence key must be non-empty",
      );
    }
    const childKey = JSON.stringify([parent, boundary.name]);
    const stepChildKey = JSON.stringify([stepOrdinal, childKey]);
    const occurrenceKey = JSON.stringify([
      stepChildKey,
      boundary.occurrenceKey,
    ]);
    const existing = this.ordinalByOccurrence.get(occurrenceKey);
    if (existing === undefined) {
      const seenKeys = this.keysByStepChild.get(stepChildKey);
      if (seenKeys && seenKeys.size > 0 && !preallocating) {
        throw new UnsupportedChildOperationFanOutError(
          `child '${boundary.name}' has multiple occurrences in root step ` +
            `${stepOrdinal}; deterministic preallocation is required`,
        );
      }
      const ordinal = this.nextOrdinalByChild.get(childKey) ?? 1;
      this.nextOrdinalByChild.set(childKey, ordinal + 1);
      this.ordinalByOccurrence.set(occurrenceKey, ordinal);
    }
    const keys = this.keysByStepChild.get(stepChildKey) ?? new Set<string>();
    keys.add(boundary.occurrenceKey);
    this.keysByStepChild.set(stepChildKey, keys);
    return this.ordinalByOccurrence.get(occurrenceKey) as number;
  }
}

function rootOperationPath(): OperationPath {
  return create(OperationPathSchema, {
    segments: [
      create(OperationPathSegmentSchema, { name: "agent", ordinal: 1n }),
    ],
  });
}

function initialStepOrdinal(context: AttemptContext): number {
  const lineage = context.branchLineage;
  if (lineage === undefined) return 1;
  const source = lineage.sourceWorkflowIdentity;
  if (
    source === undefined ||
    !source.sessionId ||
    !source.executionId ||
    lineage.sourceStepOrdinal <= 0n ||
    !lineage.sourceStateHash
  ) {
    throw new Error(
      "branch_lineage must identify a positive immutable source cutoff",
    );
  }
  return Number(lineage.sourceStepOrdinal) + 1;
}

function bindHolder(attempt: AttemptContext): AttemptContextHolder {
  return {
    attempt,
    step: { value: initialStepOrdinal(attempt) },
    activityOrdinals: new ActivityOrdinalAllocator(),
    childPaths: new ChildOperationPathAllocator(),
    pendingChildBatch: { boundaries: [] },
    observed: new ObservedActivityPositions(),
    interrupted: new InterruptedActivities(),
    reconstruction: new ActivityReconstruction(),
    cancelledHistory: new CancelledHistoryState(),
    guardrailReviewWaits: new GuardrailReviewWaits(),
  };
}

function activeHolder(): AttemptContextHolder | undefined {
  const holder = storage.getStore();
  if (holder === undefined || holder.attempt === null) return undefined;
  return holder;
}

function requireHolder(what: string): AttemptContextHolder {
  const holder = activeHolder();
  if (holder === undefined) {
    throw new Error(
      `${what} requires an active durable attempt context; ` +
        "bind AttemptContext before admitting activities or committing steps",
    );
  }
  return holder;
}

export function runWithAttemptContext<T>(
  attempt: AttemptContext,
  fn: () => T,
): T {
  const holder = bindHolder(attempt);
  const close = (): void => {
    holder.cancelledHistory.close();
    holder.attempt = null;
  };
  try {
    const result = storage.run(holder, fn);
    if (result instanceof Promise) {
      return result.finally(close) as T;
    }
    close();
    return result;
  } catch (error) {
    close();
    throw error;
  }
}

export function currentAttemptContext(): AttemptContext | null {
  return storage.getStore()?.attempt ?? null;
}

/** @internal Executor adapter capture scope. */
export function currentCancelledHistory(): CancelledHistoryState | null {
  return activeHolder()?.cancelledHistory ?? null;
}

/** @internal Executor lifecycle callback before adapter scratch cleanup. */
export async function reportCancelledHistoryBeforeCleanup(): Promise<void> {
  await currentCancelledHistory()?.reportBeforeCleanup();
}

export function currentStepOrdinal(): number {
  const holder = requireHolder("step_ordinal");
  if (holder.step.value <= 0) {
    throw new Error(
      "step_ordinal requires an active durable attempt context; " +
        "bind AttemptContext before admitting activities or committing steps",
    );
  }
  return holder.step.value;
}

export function currentOperationPath(): OperationPath {
  const holder = activeHolder();
  const resolver = resolverStorage.getStore();
  if (holder === undefined || resolver === undefined) {
    return rootOperationPath();
  }
  return holder.childPaths.resolve(resolver(), currentStepOrdinal());
}

export function runWithOperationPathResolver<T>(
  resolver: OperationPathResolver,
  fn: () => T,
): T {
  return resolverStorage.run(resolver, fn);
}

export function childOperationBoundaryScope<T>(
  boundary: ChildOperationBoundary,
  fn: () => T,
): T {
  // Append one child boundary to the active framework operation path.
  //
  // Framework adapters use this around dynamic child execution whose boundary
  // is not visible in the parent graph topology. If a parent resolver is
  // active, its boundaries are captured before the child changes framework
  // runtime context and remain the prefix; otherwise the supplied boundary
  // starts the nested path below the fixed root agent segment.
  const parentResolver = resolverStorage.getStore();
  const parentBoundaries = parentResolver?.() ?? [];

  return runWithOperationPathResolver(
    () => [...parentBoundaries, boundary],
    fn,
  );
}

export function recordObservedActivity(position: ActivityPosition): void {
  activeHolder()?.observed.record(position);
}

export function observedActivityPositions(
  stepOrdinal: number,
): ActivityPosition[] {
  return activeHolder()?.observed.forStep(stepOrdinal) ?? [];
}

export function recordInterruptedActivity(
  command: ActivityCommand,
  controlFlow: unknown,
): void {
  activeHolder()?.interrupted.record(command, controlFlow);
}

/**
 * Framework-interrupted activities observed in one workflow step.
 *
 * Answered activities are excluded unless requested. One LangGraph task can
 * pause several times and every pause carries the same native interrupt id,
 * so only the pauses still waiting may claim that id.
 */
export function interruptedActivities(
  stepOrdinal: number,
  options: { readonly includeAnswered?: boolean } = {},
): InterruptedActivity[] {
  return (
    activeHolder()?.interrupted.forStep(
      stepOrdinal,
      options.includeAnswered === true,
    ) ?? []
  );
}

/** Record that these interrupted activities received their answer in this attempt. */
export function markInterruptedActivitiesAnswered(
  positions: readonly ActivityPosition[],
): void {
  activeHolder()?.interrupted.markAnswered(positions);
}

/** Record that this attempt pauses for a review the platform halted a call for. */
export function noteGuardrailReviewWait(reviewId: string): void {
  activeHolder()?.guardrailReviewWaits.note(reviewId);
}

/** Whether this attempt paused for that review. */
export function isGuardrailReviewWait(reviewId: unknown): boolean {
  if (typeof reviewId !== "string") return false;
  return activeHolder()?.guardrailReviewWaits.has(reviewId) ?? false;
}

export function setActivityReconstructionIds(
  activityIds: readonly string[],
): void {
  activeHolder()?.reconstruction.replace(activityIds);
}

export function activityRequiresReconstruction(activityId: string): boolean {
  return activeHolder()?.reconstruction.contains(activityId) ?? false;
}

export function recordReconstructedActivityInterrupt(
  activityId: string,
): number {
  return requireHolder(
    "activity reconstruction",
  ).reconstruction.recordInterrupt(activityId);
}

export function advanceStepOrdinal(committedStepOrdinal: number): number {
  if (committedStepOrdinal <= 0) {
    throw new Error("committed step_ordinal must be positive");
  }
  const holder = requireHolder("step_ordinal");
  holder.step.value = committedStepOrdinal + 1;
  return holder.step.value;
}

export function toolActivityKey(toolCallId: string): string {
  if (!toolCallId) {
    throw new Error("tool_call_id must be non-empty");
  }
  return `tool:${toolCallId}`;
}

/**
 * The 1-based index of this call within `scope` for the current attempt.
 *
 * An adapter passes the framework object that lives for one run of a unit the
 * framework may re-run, such as one run of a graph task. The index restarts
 * for each run, so a key built from it names the same call on every run and
 * its recorded activity replays.
 */
export function nextScopedCallIndex(scope: object): number {
  return requireHolder("activity_ordinal").activityOrdinals.nextInScope(scope);
}

export function allocateActivityOrdinal(key?: string): number {
  const holder = requireHolder("activity_ordinal");
  return holder.activityOrdinals.allocate(
    pathKeyFromSegments(currentOperationPath().segments),
    key,
  );
}

/**
 * Return the operation path and ordinal for a possibly preallocated `key`.
 *
 * Use this where the activity may run in a spawned task other than the one
 * that preallocated its key; pass the returned path to the activity command.
 */
export function allocateActivityPosition(key?: string): {
  operationPath: OperationPath;
  activityOrdinal: number;
} {
  const holder = requireHolder("activity_ordinal");
  const current = pathKeyFromSegments(currentOperationPath().segments);
  const [path, activityOrdinal] = holder.activityOrdinals.allocatePosition(
    current,
    key,
    holder.childPaths.spawningScope(current),
  );
  return {
    operationPath: create(OperationPathSchema, {
      segments: pathSegments(path).map(([name, ordinal]) =>
        create(OperationPathSegmentSchema, { name, ordinal: BigInt(ordinal) }),
      ),
    }),
    activityOrdinal,
  };
}

/** Whether the current operation path runs inside a spawned task. */
export function inSpawnedTask(): boolean {
  const holder = activeHolder();
  if (holder === undefined) return false;
  return (
    holder.childPaths.spawningScope(
      pathKeyFromSegments(currentOperationPath().segments),
    ) !== null
  );
}

export function preallocateActivityOrdinals(keys: readonly string[]): number[] {
  const holder = requireHolder("activity_ordinal");
  return holder.activityOrdinals.preallocate(
    pathKeyFromSegments(currentOperationPath().segments),
    keys,
  );
}

export function preallocateChildOperationOrdinals(
  boundaries: readonly ChildOperationBoundary[],
): number[] {
  const holder = requireHolder("child operation paths");
  return holder.childPaths.preallocate(
    pathKeyFromSegments(currentOperationPath().segments),
    currentStepOrdinal(),
    boundaries,
  );
}

export function currentPendingChildOperationBatch(): readonly ChildOperationBoundary[] {
  return activeHolder()?.pendingChildBatch.boundaries ?? [];
}

export function setPendingChildOperationBatch(
  boundaries: readonly ChildOperationBoundary[],
): void {
  // Replace in the shared attempt holder so detached worker copies observe
  // the batch without a process-global map. The active-holder gate keeps
  // detached callbacks from touching the batch after the attempt closes,
  // matching every other attempt-scoped accessor.
  const holder = activeHolder();
  if (holder === undefined) return;
  holder.pendingChildBatch.boundaries = [...boundaries];
}
