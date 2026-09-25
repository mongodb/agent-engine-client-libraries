/**
 * Tests for the session-finish request primitive.
 *
 * Mirrors Python's tests/unit/test_session_finish_context.py.
 */

import { describe, test, expect } from "vitest";
import {
  runWithExecutionContext,
  requestSessionFinish,
  isSessionFinishRequested,
  closeSessionFinish,
} from "../../src/index.js";

const EXEC_ID = "test-exec-001";
const WRAPPER = {};
const OE_URL = "http://localhost:8000";
const SESSION_ID = "sess-1";

describe("session finish context", () => {
  test("returns unavailable outside an execution context", () => {
    expect(requestSessionFinish()).toBe("unavailable");
    expect(isSessionFinishRequested()).toBe(false);
  });

  test("is idempotent inside a run and visible across awaits", async () => {
    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        expect(requestSessionFinish()).toBe("requested");
        await Promise.resolve();
        expect(requestSessionFinish()).toBe("already_requested");
        expect(isSessionFinishRequested()).toBe(true);
      },
    );
  });

  test("does not leak between contexts", async () => {
    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        expect(requestSessionFinish()).toBe("requested");
      },
    );

    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        expect(isSessionFinishRequested()).toBe(false);
        expect(requestSessionFinish()).toBe("requested");
      },
    );

    expect(isSessionFinishRequested()).toBe(false);
  });

  test("setTimeout callback firing after the run reports unavailable", async () => {
    let afterRunResult: string | undefined;
    let notifyFired: () => void;
    const fired = new Promise<void>((resolve) => {
      notifyFired = resolve;
    });

    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        // Scheduled during the turn; fires after this callback (and the
        // AER's closing `finally`) has already returned.
        setTimeout(() => {
          afterRunResult = requestSessionFinish();
          notifyFired();
        }, 0);
        // Mirrors the AER's finally, which runs as soon as the execute
        // frame returns - before the timer above has any chance to fire.
        closeSessionFinish();
      },
    );
    await fired;
    expect(afterRunResult).toBe("unavailable");
  });

  test("a floating promise resolving after the run reports unavailable", async () => {
    let afterRunResult: string | undefined;
    let capture: (() => void) | undefined;
    let settled: Promise<void> | undefined;

    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        // Created during the turn (not awaited inside the run), so its
        // continuation inherits the run's ALS context the same way a
        // fire-and-forget write does.
        const floating = new Promise<void>((resolve) => {
          capture = resolve;
        });
        settled = floating.then(() => {
          afterRunResult = requestSessionFinish();
        });
        // Mirrors the AER's finally, which runs as soon as the execute
        // frame returns - before the floating promise above has resolved.
        closeSessionFinish();
      },
    );
    capture?.();
    await settled;
    expect(afterRunResult).toBe("unavailable");
  });

  test("in-run first/second calls are requested / already_requested", async () => {
    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        expect(requestSessionFinish()).toBe("requested");
        expect(requestSessionFinish()).toBe("already_requested");
      },
    );
  });

  test("a finish latched during the turn is still drained after close", async () => {
    let requestedDuringRun = "";
    await runWithExecutionContext(
      {
        executionId: EXEC_ID,
        wrapper: WRAPPER,
        oeUrl: OE_URL,
        sessionId: SESSION_ID,
      },
      async () => {
        requestedDuringRun = requestSessionFinish();
        expect(isSessionFinishRequested()).toBe(true);
        closeSessionFinish();
        // Drained before teardown, same as the AER reading it right after
        // the callback fires and before `finally` closes the latch.
        expect(isSessionFinishRequested()).toBe(true);
      },
    );
    expect(requestedDuringRun).toBe("requested");
  });
});
