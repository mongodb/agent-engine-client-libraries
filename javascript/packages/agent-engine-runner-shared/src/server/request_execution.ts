/**
 * A route binds the execution ID it parsed for its whole lifetime, so setup,
 * callback delivery, cleanup and the work they start log with that ID, not
 * only the agent or tool body inside `runWithExecutionContext`. Mirrors
 * Python's `execution_output_scope` / `bind_log_execution_id`.
 */

import type { FastifyRequest } from "fastify";

import {
  runWithLogCorrelation,
  runWithLogOutputScope,
} from "../log_correlation.js";
import { flushCapturedOutput } from "../structured_logging/stdio_capture.js";

// Fastify runs the error handler outside the route's async context, so it
// looks the ID up here. Keyed by request so the entry dies with the request.
const requestExecutionIds = new WeakMap<FastifyRequest, string>();

/**
 * For routes that run agent, tool or LLM code. `flushOutput` is for work that
 * outlives the route, such as a stream generator whose cleanup settles after
 * the response ends.
 */
export async function runWithRequestExecution<T>(
  request: FastifyRequest,
  executionId: string,
  fn: (flushOutput: () => void) => Promise<T>,
): Promise<T> {
  recordRequestExecution(request, executionId);
  return runWithExecutionOutput(executionId, fn);
}

/**
 * Also for work that runs after its request ended, which must not write into
 * that request's already-flushed output scope.
 */
export async function runWithExecutionOutput<T>(
  executionId: string,
  fn: (flushOutput: () => void) => Promise<T>,
): Promise<T> {
  const scope = {};
  const flushOutput = (): void => flushCapturedOutput(scope);
  return runWithLogCorrelation(executionId, () =>
    runWithLogOutputScope(scope, async () => {
      try {
        return await fn(flushOutput);
      } finally {
        flushOutput();
      }
    }),
  );
}

/**
 * For side requests (drain, interrupt, guardrails). Async work they start (a
 * drain's finalize) outlives them, so an output scope they owned would hold
 * that work's later partial lines with nothing left to flush them.
 */
export function bindRequestExecution<T>(
  request: FastifyRequest,
  executionId: string,
  fn: () => T,
): T {
  recordRequestExecution(request, executionId);
  return runWithLogCorrelation(executionId, fn);
}

function recordRequestExecution(
  request: FastifyRequest,
  executionId: string,
): void {
  if (executionId.trim()) requestExecutionIds.set(request, executionId);
}

export function getRequestExecutionId(request: FastifyRequest): string | null {
  return requestExecutionIds.get(request) ?? null;
}
