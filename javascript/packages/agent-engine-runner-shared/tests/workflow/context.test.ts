import { create } from "@bufbuild/protobuf";
import { describe, expect, test } from "vitest";

import {
  ActivityPositionSchema,
  OperationPathSchema,
  OperationPathSegmentSchema,
  WorkflowIdentitySchema,
} from "../../src/generated/workflow/v1/common_pb.js";
import { AttemptContextSchema } from "../../src/generated/workflow/v1/runtime_pb.js";
import { ActivityCommandSchema } from "../../src/generated/workflow/v1/activity_pb.js";
import {
  AmbiguousActivityKeyError,
  allocateActivityOrdinal,
  allocateActivityPosition,
  activityRequiresReconstruction,
  currentOperationPath,
  currentPendingChildOperationBatch,
  currentStepOrdinal,
  interruptedActivities,
  markInterruptedActivitiesAnswered,
  nextScopedCallIndex,
  observedActivityPositions,
  preallocateActivityOrdinals,
  preallocateChildOperationOrdinals,
  inSpawnedTask,
  recordObservedActivity,
  recordInterruptedActivity,
  recordReconstructedActivityInterrupt,
  runWithAttemptContext,
  runWithOperationPathResolver,
  setActivityReconstructionIds,
  setPendingChildOperationBatch,
  toolActivityKey,
  UnsupportedChildOperationFanOutError,
  type ChildOperationBoundary,
} from "../../src/workflow/index.js";

function attempt() {
  return create(AttemptContextSchema, {
    attemptId: "attempt-1",
    fencingToken: 1n,
    ownerId: "aer-1",
  });
}

function position(activityOrdinal: number, step = 1) {
  return create(ActivityPositionSchema, {
    operationPath: create(OperationPathSchema, {
      segments: [
        create(OperationPathSegmentSchema, { name: "agent", ordinal: 1n }),
      ],
    }),
    activityOrdinal: BigInt(activityOrdinal),
    stepOrdinal: BigInt(step),
  });
}

function branchAttempt() {
  return create(AttemptContextSchema, {
    attemptId: "attempt-branch",
    fencingToken: 1n,
    ownerId: "aer-1",
    branchLineage: {
      sourceWorkflowIdentity: create(WorkflowIdentitySchema, {
        sessionId: "source-session",
        executionId: "source-execution",
      }),
      sourceStepOrdinal: 4n,
      sourceStateHash: "sha256:source",
    },
  });
}

function nestedPosition(childName: string, activityOrdinal = 1) {
  return create(ActivityPositionSchema, {
    operationPath: create(OperationPathSchema, {
      segments: [
        create(OperationPathSegmentSchema, { name: "agent", ordinal: 1n }),
        create(OperationPathSegmentSchema, {
          name: childName,
          ordinal: 1n,
        }),
      ],
    }),
    activityOrdinal: BigInt(activityOrdinal),
    stepOrdinal: 1n,
  });
}

describe("observed activity positions", () => {
  test("are unique and reported in order by step", () => {
    runWithAttemptContext(attempt(), () => {
      recordObservedActivity(position(2));
      recordObservedActivity(position(1, 2));
      recordObservedActivity(position(1));
      recordObservedActivity(position(1));
      expect(
        observedActivityPositions(1).map((item) =>
          Number(item.activityOrdinal),
        ),
      ).toEqual([1, 2]);
      expect(
        observedActivityPositions(2).map((item) =>
          Number(item.activityOrdinal),
        ),
      ).toEqual([1]);
    });
  });

  test("sorts multi-digit activity ordinals numerically", () => {
    runWithAttemptContext(attempt(), () => {
      recordObservedActivity(position(10));
      recordObservedActivity(position(2));
      recordObservedActivity(position(1));
      expect(
        observedActivityPositions(1).map((item) =>
          Number(item.activityOrdinal),
        ),
      ).toEqual([1, 2, 10]);
    });
  });

  test("sorts child names by code units, matching Python tuple order", () => {
    runWithAttemptContext(attempt(), () => {
      recordObservedActivity(nestedPosition("a"));
      recordObservedActivity(nestedPosition("Z"));
      expect(
        observedActivityPositions(1).map(
          (item) => item.operationPath?.segments[1]?.name,
        ),
      ).toEqual(["Z", "a"]);
    });
  });
});

describe("interrupted activity snapshots", () => {
  test("copies the command at capture and retains control flow by identity", () => {
    runWithAttemptContext(attempt(), () => {
      const command = create(ActivityCommandSchema, {
        activityName: "review",
        position: position(2),
        semanticInput: { kind: { case: "stringValue", value: "original" } },
      });
      const controlFlow = new Error("framework interrupt");
      recordInterruptedActivity(command, controlFlow);
      recordInterruptedActivity(
        create(ActivityCommandSchema, { position: position(1, 2) }),
        new Error("next step"),
      );
      command.activityName = "changed";
      if (
        command.position === undefined ||
        command.semanticInput === undefined
      ) {
        throw new Error("missing test command fields");
      }
      command.position.activityOrdinal = 99n;
      command.semanticInput.kind = { case: "stringValue", value: "changed" };

      const captured = interruptedActivities(1);
      expect(captured).toHaveLength(1);
      expect(captured[0]?.command).toMatchObject({
        activityName: "review",
        position: { activityOrdinal: 2n, stepOrdinal: 1n },
        semanticInput: { kind: { case: "stringValue", value: "original" } },
      });
      expect(captured[0]?.controlFlow).toBe(controlFlow);
      expect(interruptedActivities(2)).toHaveLength(1);
    });
  });

  test("drops answered activities from the waiting view within one attempt", () => {
    const record = (ordinal: number) => {
      const command = create(ActivityCommandSchema, {
        activityName: "langgraph.interrupt",
        position: position(ordinal),
      });
      recordInterruptedActivity(command, new Error(`pause ${ordinal}`));
      if (command.position === undefined) throw new Error("missing position");
      return command.position;
    };
    runWithAttemptContext(attempt(), () => {
      markInterruptedActivitiesAnswered([record(1)]);
      record(2);

      expect(
        interruptedActivities(1).map((item) => item.command.position),
      ).toMatchObject([{ activityOrdinal: 2n }]);
      expect(
        interruptedActivities(1, { includeAnswered: true }).map(
          (item) => item.command.position?.activityOrdinal,
        ),
      ).toEqual([1n, 2n]);
    });
    // Answers are attempt-local: a new attempt starts with nothing answered.
    runWithAttemptContext(attempt(), () => {
      record(1);
      expect(interruptedActivities(1)).toHaveLength(1);
    });
  });

  test("shares capture across child tasks but isolates overlapping attempts", async () => {
    const ready = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    const left = runWithAttemptContext(attempt(), async () => {
      await Promise.resolve().then(() => {
        recordInterruptedActivity(
          create(ActivityCommandSchema, {
            activityName: "left",
            position: position(1),
          }),
          new Error("left"),
        );
      });
      ready.resolve();
      await release.promise;
      return interruptedActivities(1).map((item) => item.command.activityName);
    });
    const right = runWithAttemptContext(attempt(), async () => {
      await ready.promise;
      recordInterruptedActivity(
        create(ActivityCommandSchema, {
          activityName: "right",
          position: position(1),
        }),
        new Error("right"),
      );
      release.resolve();
      return interruptedActivities(1).map((item) => item.command.activityName);
    });
    await expect(Promise.all([left, right])).resolves.toEqual([
      ["left"],
      ["right"],
    ]);
  });

  test.each([false, true])(
    "detached work cannot retain capture after exit (failure=%s)",
    async (fail) => {
      const release = Promise.withResolvers<void>();
      const command = create(ActivityCommandSchema, { position: position(1) });
      let detached: Promise<unknown> | undefined;
      const scoped = runWithAttemptContext(attempt(), async () => {
        recordInterruptedActivity(command, new Error("before exit"));
        detached = release.promise.then(() => {
          recordInterruptedActivity(command, new Error("after exit"));
          return interruptedActivities(1);
        });
        if (fail) throw new Error("attempt failed");
      });
      if (fail) await expect(scoped).rejects.toThrow("attempt failed");
      else await scoped;
      release.resolve();
      await expect(detached).resolves.toEqual([]);
      recordInterruptedActivity(command, new Error("outside attempt"));
      expect(interruptedActivities(1)).toEqual([]);
      runWithAttemptContext(attempt(), () => {
        expect(interruptedActivities(1)).toEqual([]);
      });
    },
  );
});

describe("activity reconstruction", () => {
  test("selects activity ids within one attempt context", () => {
    runWithAttemptContext(attempt(), () => {
      setActivityReconstructionIds(["activity-1"]);
      expect(activityRequiresReconstruction("activity-1")).toBe(true);
      expect(activityRequiresReconstruction("activity-2")).toBe(false);
      expect(recordReconstructedActivityInterrupt("activity-1")).toBe(1);
      expect(recordReconstructedActivityInterrupt("activity-1")).toBe(2);
      expect(() => recordReconstructedActivityInterrupt("activity-2")).toThrow(
        'activity "activity-2" is not being reconstructed',
      );
    });

    expect(activityRequiresReconstruction("activity-1")).toBe(false);
  });

  test("isolates overlapping attempts", async () => {
    const ready = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    const left = runWithAttemptContext(attempt(), async () => {
      setActivityReconstructionIds(["left"]);
      ready.resolve();
      await release.promise;
      return [
        activityRequiresReconstruction("left"),
        activityRequiresReconstruction("right"),
      ];
    });
    const right = runWithAttemptContext(attempt(), async () => {
      await ready.promise;
      setActivityReconstructionIds(["right"]);
      release.resolve();
      return [
        activityRequiresReconstruction("left"),
        activityRequiresReconstruction("right"),
      ];
    });

    await expect(Promise.all([left, right])).resolves.toEqual([
      [true, false],
      [false, true],
    ]);
  });
});

describe("branch lineage", () => {
  test("starts the step counter after the source cutoff", () => {
    runWithAttemptContext(branchAttempt(), () => {
      expect(currentStepOrdinal()).toBe(5);
    });
  });

  test("rejects lineage without a source identity", () => {
    const malformed = branchAttempt();
    const source = malformed.branchLineage?.sourceWorkflowIdentity;
    if (source !== undefined) source.executionId = "";
    expect(() => runWithAttemptContext(malformed, () => undefined)).toThrow(
      /immutable source cutoff/,
    );
  });

  test("rejects lineage with a non-positive source step", () => {
    const malformed = create(AttemptContextSchema, {
      attemptId: "attempt-branch",
      fencingToken: 1n,
      ownerId: "aer-1",
      branchLineage: {
        sourceWorkflowIdentity: create(WorkflowIdentitySchema, {
          sessionId: "source-session",
          executionId: "source-execution",
        }),
        sourceStepOrdinal: 0n,
        sourceStateHash: "sha256:source",
      },
    });
    expect(() => runWithAttemptContext(malformed, () => undefined)).toThrow(
      /immutable source cutoff/,
    );
  });

  test("rejects lineage missing the source identity", () => {
    const malformed = create(AttemptContextSchema, {
      attemptId: "attempt-branch",
      fencingToken: 1n,
      ownerId: "aer-1",
      branchLineage: {
        sourceStepOrdinal: 4n,
        sourceStateHash: "sha256:source",
      },
    });
    expect(() => runWithAttemptContext(malformed, () => undefined)).toThrow(
      /immutable source cutoff/,
    );
  });
});

describe("scoped call index", () => {
  test("restarts for a new scope and a new attempt", () => {
    const firstRun = {};
    const secondRun = {};
    runWithAttemptContext(attempt(), () => {
      expect([1, 2, 3].map(() => nextScopedCallIndex(firstRun))).toEqual([
        1, 2, 3,
      ]);
      // A re-run of the same unit is a new scope: its calls count from 1
      // again, so they name the calls the first run already made.
      expect([1, 2].map(() => nextScopedCallIndex(secondRun))).toEqual([1, 2]);
      expect(nextScopedCallIndex(firstRun)).toBe(4);
    });
    runWithAttemptContext(attempt(), () => {
      expect(nextScopedCallIndex(firstRun)).toBe(1);
    });
  });
});

describe("activity ordinal allocation", () => {
  test("preallocates stable keys and reuses them", () => {
    runWithAttemptContext(attempt(), () => {
      const keys = [toolActivityKey("call-a"), toolActivityKey("call-b")];
      expect(preallocateActivityOrdinals(keys)).toEqual([1, 2]);
      expect(allocateActivityOrdinal(toolActivityKey("call-b"))).toBe(2);
      expect(allocateActivityOrdinal(toolActivityKey("call-a"))).toBe(1);
      expect(allocateActivityOrdinal()).toBe(3);
      expect(allocateActivityOrdinal("stable")).toBe(4);
      expect(allocateActivityOrdinal("stable")).toBe(4);
    });
  });
});

describe("spawned-task tool positions", () => {
  type Boundary = ChildOperationBoundary;
  const spawned = (name: string): Boundary => ({
    name,
    occurrenceKey: name,
    spawnedTask: true,
  });
  const child = (name: string): Boundary => ({ name, occurrenceKey: name });
  const preallocate = (...keys: string[]): void => {
    preallocateActivityOrdinals(keys.map(toolActivityKey));
  };
  const position = (key: string): [string[], number] => {
    const { operationPath, activityOrdinal } = allocateActivityPosition(
      toolActivityKey(key),
    );
    return [pathNames(operationPath), activityOrdinal];
  };
  /** Run `fn` in a fresh attempt whose resolver returns `boundaries`. */
  const inAttempt = (fn: (boundaries: Boundary[]) => void): void => {
    const boundaries: Boundary[] = [];
    runWithAttemptContext(attempt(), () =>
      runWithOperationPathResolver(
        () => boundaries,
        () => fn(boundaries),
      ),
    );
  };
  const set = (boundaries: Boundary[], next: Boundary[]): void => {
    boundaries.splice(0, boundaries.length, ...next);
  };

  test("a tool in a spawned task keeps its model's position", () => {
    inAttempt((boundaries) => {
      // A model in Send worker 0 requests two tools.
      set(boundaries, [spawned("worker-0")]);
      preallocate("call-a", "call-b");
      expect(inSpawnedTask()).toBe(true);

      // Each tool runs in its own Send task spawned by the root graph.
      set(boundaries, [spawned("tool-1")]);
      expect(position("call-b")).toEqual([["agent", "worker-0"], 2]);
      set(boundaries, [spawned("tool-0")]);
      expect(position("call-a")).toEqual([["agent", "worker-0"], 1]);

      // A key that was never preallocated keeps today's allocation.
      expect(position("call-c")).toEqual([["agent", "tool-0"], 1]);
    });
  });

  test("a tool outside a spawned task allocates in its own path", () => {
    inAttempt((boundaries) => {
      // A planner subgraph requests a tool that a sibling tools subgraph
      // runs, and a root model's tool runs inside a subgraph. Neither path
      // has a spawned task, so both record where they did before.
      set(boundaries, [child("planner")]);
      preallocate("call-a");
      set(boundaries, []);
      preallocate("call-b");

      set(boundaries, [child("tools")]);
      expect(inSpawnedTask()).toBe(false);
      expect(position("call-a")).toEqual([["agent", "tools"], 1]);
      expect(position("call-b")).toEqual([["agent", "tools"], 2]);
    });
  });

  test("spawned tools of parallel graphs stay in their own graph", () => {
    inAttempt((boundaries) => {
      // Two coordinator workers reuse one tool call id.
      for (const worker of ["worker-0", "worker-1"]) {
        set(boundaries, [spawned(worker), child("coordinator")]);
        preallocate("call-x", "call-a");
      }
      // Each coordinator's tool runs in a Send task it spawned.
      for (const worker of ["worker-1", "worker-0"]) {
        set(boundaries, [
          spawned(worker),
          child("coordinator"),
          spawned("tool"),
        ]);
        expect(position("call-a")).toEqual([
          ["agent", worker, "coordinator"],
          2,
        ]);
      }
    });
  });

  test("a tool call id reused within one graph fails in either order", () => {
    // Both models preallocate before the tool runs, in its own task or in
    // the task of a model that requested it.
    for (const toolTask of [[spawned("tool")], [spawned("worker-1")]]) {
      inAttempt((boundaries) => {
        for (const worker of ["worker-0", "worker-1"]) {
          set(boundaries, [spawned(worker)]);
          preallocate("call-a");
        }
        set(boundaries, toolTask);
        expect(() => position("call-a")).toThrow(AmbiguousActivityKeyError);
      });
    }
    // The tool runs before the second model preallocates.
    for (const toolTask of [[spawned("tool")], [spawned("worker-0")]]) {
      inAttempt((boundaries) => {
        set(boundaries, [spawned("worker-0")]);
        preallocate("call-a");
        set(boundaries, toolTask);
        expect(position("call-a")).toEqual([["agent", "worker-0"], 1]);
        set(boundaries, [spawned("worker-1")]);
        expect(() => preallocate("call-a")).toThrow(/call-a/);
      });
    }
  });

  test("a take in one graph does not hide a take in another", () => {
    inAttempt((boundaries) => {
      const inCoordinator = (worker: string, inner: Boundary): void =>
        set(boundaries, [spawned(worker), child("coordinator"), inner]);
      // Two parallel coordinators reuse one tool call id; each tool takes
      // its own coordinator's reservation.
      for (const worker of ["worker-0", "worker-1"]) {
        inCoordinator(worker, spawned("model-0"));
        preallocate("call-a");
        inCoordinator(worker, spawned("tool"));
        position("call-a");
      }
      // A second reservation in the first coordinator still fails.
      inCoordinator("worker-0", spawned("model-1"));
      expect(() => preallocate("call-a")).toThrow(AmbiguousActivityKeyError);
    });
  });
});

function pathNames(path = currentOperationPath()): string[] {
  return path.segments.map((segment) => segment.name);
}

describe("operation path resolver scope", () => {
  test("overlapping sibling scopes keep distinct nested paths after await", async () => {
    await runWithAttemptContext(attempt(), async () => {
      const leftReady = Promise.withResolvers<void>();
      const rightReady = Promise.withResolvers<void>();
      const left = runWithOperationPathResolver(
        () => [{ name: "left_child", occurrenceKey: "L" }],
        async () => {
          await Promise.resolve();
          const nested = pathNames();
          leftReady.resolve();
          await rightReady.promise;
          return nested;
        },
      );
      const right = runWithOperationPathResolver(
        () => [{ name: "right_child", occurrenceKey: "R" }],
        async () => {
          await Promise.resolve();
          const nested = pathNames();
          rightReady.resolve();
          await leftReady.promise;
          return nested;
        },
      );
      await expect(Promise.all([left, right])).resolves.toEqual([
        ["agent", "left_child"],
        ["agent", "right_child"],
      ]);
    });
  });

  test("rejects a second unpreallocated occurrence of the same child", () => {
    runWithAttemptContext(attempt(), () => {
      runWithOperationPathResolver(
        () => [{ name: "case_investigation", occurrenceKey: "task-a" }],
        () => {
          expect(pathNames()).toEqual(["agent", "case_investigation"]);
        },
      );
      expect(() =>
        runWithOperationPathResolver(
          () => [{ name: "case_investigation", occurrenceKey: "task-b" }],
          () => pathNames(),
        ),
      ).toThrow(UnsupportedChildOperationFanOutError);
      runWithOperationPathResolver(
        () => [{ name: "case_investigation", occurrenceKey: "task-a" }],
        () => {
          expect(pathNames()).toEqual(["agent", "case_investigation"]);
        },
      );
    });
  });

  test("a failed preallocation batch consumes no ordinals", () => {
    runWithAttemptContext(attempt(), () => {
      expect(() =>
        preallocateChildOperationOrdinals([
          { name: "researcher", occurrenceKey: "task-a" },
          { name: "researcher", occurrenceKey: "" },
        ]),
      ).toThrow(/must be non-empty/);

      // The rejected batch left no partial reservation: the first valid
      // sibling after it still starts the child's ordinal sequence at 1.
      runWithOperationPathResolver(
        () => [{ name: "researcher", occurrenceKey: "task-b" }],
        () => {
          expect(
            currentOperationPath().segments.map((segment) => [
              segment.name,
              Number(segment.ordinal),
            ]),
          ).toEqual([
            ["agent", 1],
            ["researcher", 1],
          ]);
        },
      );
    });
  });
});

describe("pending child operation batch", () => {
  test("is visible inside the attempt and absent outside it", () => {
    const boundaries = [{ name: "researcher", occurrenceKey: "task-a" }];
    runWithAttemptContext(attempt(), () => {
      setPendingChildOperationBatch(boundaries);
      expect(currentPendingChildOperationBatch()).toEqual(boundaries);
    });
    expect(currentPendingChildOperationBatch()).toEqual([]);
    runWithAttemptContext(attempt(), () => {
      expect(currentPendingChildOperationBatch()).toEqual([]);
    });
  });

  test("detached work cannot retain the batch after exit", async () => {
    const release = Promise.withResolvers<void>();
    const boundaries = [{ name: "researcher", occurrenceKey: "task-a" }];
    let detached: Promise<readonly unknown[]> | undefined;
    runWithAttemptContext(attempt(), () => {
      setPendingChildOperationBatch(boundaries);
      detached = release.promise.then(() => {
        // A detached callback that kept the holder after close must neither
        // read the stale batch nor repopulate it.
        const stale = currentPendingChildOperationBatch();
        setPendingChildOperationBatch([{ name: "late", occurrenceKey: "k" }]);
        return stale;
      });
    });
    release.resolve();
    await expect(detached).resolves.toEqual([]);
    runWithAttemptContext(attempt(), () => {
      expect(currentPendingChildOperationBatch()).toEqual([]);
    });
  });
});
