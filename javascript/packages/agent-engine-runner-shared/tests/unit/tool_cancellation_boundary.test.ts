/**
 * What cancelling a tool call does and does not reach, in TypeScript.
 *
 * Mirrors Python's tests/unit/test_tool_cancellation_boundary.py. These are
 * characterization tests, not aspirations: stopping a run aborts the platform's
 * own outbound request to the tool pod, and does not reach work the handler
 * started. Pinning the boundary here keeps the gap visible in the suite rather
 * than rediscovered from a customer report.
 *
 * These drive `ToolServer.doHandleExecute`, so the behaviour asserted is the
 * server's real dispatch for a registered tool rather than a restatement of it.
 * Private members are reached via cast-through-unknown, matching
 * aer_policy_denied.test.ts.
 *
 * TS-vs-Python adaptations, and why this side is weaker:
 *   - Python's `asyncio.to_thread` offload has no TS equivalent, and a promise
 *     has no cancellation mechanism at all. There is no `task.cancel()` to
 *     call: the platform can only stop awaiting, which the handler cannot
 *     observe.
 *   - Python's async handler at least receives `CancelledError` and can clean
 *     up. Nothing equivalent reaches a TypeScript tool.
 *   - The only cooperative mechanism available is an `AbortSignal` threaded to
 *     the handler by hand. The server does not pass one, which the last test
 *     records.
 */

import { describe, expect, test } from "vitest";

import { AsyncMutex, ToolServer } from "../../src/server/tool.js";
import type { ToolPodExecuteRequest } from "../../src/models.js";

interface ToolServerPrivates {
  runtime: {
    tools: Record<string, (args: Record<string, unknown>) => unknown>;
    toolDefinitions: Record<string, unknown>;
    getAgentConfig: () => { mcp?: unknown };
  };
  executeGate: AsyncMutex;
  restrictionDisabled: boolean;
  doHandleExecute: (
    request: ToolPodExecuteRequest,
  ) => Promise<{ status: string; result?: string }>;
}

function makeServer(
  tool: (args: Record<string, unknown>) => unknown,
): ToolServerPrivates {
  const server = Object.create(
    ToolServer.prototype,
  ) as unknown as ToolServerPrivates;
  server.runtime = {
    tools: { tool },
    toolDefinitions: { tool: {} },
    getAgentConfig: () => ({}),
  };
  // Object.create skips the constructor, so seed the members the execute path
  // reads: the mutex it serializes tool calls on, and the restriction flag that
  // decides whether metadata env is applied around them.
  server.executeGate = new AsyncMutex();
  server.restrictionDisabled = true;
  return server;
}

function makeRequest(): ToolPodExecuteRequest {
  return {
    execution_id: "exec-1",
    tool_name: "tool",
    arguments: {},
    session_id: "session-1",
    metadata: {},
  } as unknown as ToolPodExecuteRequest;
}

/** Resolves after `ms`, used to let an abandoned handler make progress. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("an abandoned tool execute keeps running", () => {
  test("the tool body completes after the platform stops awaiting it", async () => {
    const observed: string[] = [];
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const server = makeServer(async () => {
      observed.push("job-1 submitted");
      await gate;
      observed.push("job-1 completed");
      return "result nobody reads";
    });

    // The platform starts the call and then gives up on it. There is no
    // cancellation to perform: dropping interest is all it can do.
    const abandoned = server.doHandleExecute(makeRequest());
    await sleep(10);
    expect(observed).toEqual(["job-1 submitted"]);

    release?.();
    await abandoned;
    expect(observed).toEqual(["job-1 submitted", "job-1 completed"]);
  });

  test("the tool cannot observe that nobody is waiting", async () => {
    const cleanedUp: string[] = [];
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const server = makeServer(async () => {
      try {
        await gate;
        return "done";
      } finally {
        // Runs when the tool itself finishes — never as a result of the caller
        // losing interest, because that is not an observable event.
        cleanedUp.push("finally reached");
      }
    });

    const abandoned = server.doHandleExecute(makeRequest());
    await sleep(10);
    expect(cleanedUp).toEqual([]);

    release?.();
    await abandoned;
    expect(cleanedUp).toEqual(["finally reached"]);
  });

  test("the side effect lands even though its result is unreachable", async () => {
    const externalJobs: string[] = [];
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const server = makeServer(async () => {
      await gate;
      externalJobs.push("payment captured");
      return "receipt-1";
    });

    // Floating on purpose: this is the shape a stopped run leaves behind.
    const floating = server.doHandleExecute(makeRequest());
    release?.();
    await floating;

    expect(externalJobs).toEqual(["payment captured"]);
  });
});

describe("the cooperative seam that exists but is unused", () => {
  test("a tool given an AbortSignal by hand can clean up its external work", async () => {
    // Not something the server does — this is what a cooperative-cancellation
    // change would have to introduce, shown working so the gap is concrete.
    const cleanedUp: string[] = [];

    const cooperativeTool = async (signal: AbortSignal): Promise<string> => {
      await new Promise<void>((resolve) => {
        if (signal.aborted) resolve();
        else signal.addEventListener("abort", () => resolve());
      });
      cleanedUp.push("job-1 cancelled with the provider");
      return "stopped early";
    };

    const controller = new AbortController();
    const running = cooperativeTool(controller.signal);
    controller.abort();
    await running;

    expect(cleanedUp).toEqual(["job-1 cancelled with the provider"]);
  });

  test("the server passes a tool no signal to cooperate with", async () => {
    // The tool records every argument it was handed. Nothing resembling a
    // cancellation signal is among them, so there is nothing a well-behaved
    // tool could react to even if it wanted to.
    let receivedArgs: unknown[] = [];
    const server = makeServer((...args: unknown[]) => {
      receivedArgs = args;
      return "done";
    });

    const response = await server.doHandleExecute(makeRequest());

    expect(response.status).toBe("success");
    const hasSignal = receivedArgs.some(
      (arg) =>
        arg instanceof AbortSignal ||
        (typeof arg === "object" &&
          arg !== null &&
          Object.values(arg).some((v) => v instanceof AbortSignal)),
    );
    expect(hasSignal).toBe(false);
  });
});
