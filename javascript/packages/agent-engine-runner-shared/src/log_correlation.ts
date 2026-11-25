/**
 * Log-only execution identity, separate from the execution store in
 * `context.ts`: the AER keys owner-URL failure handling and other runtime
 * decisions off `getCurrentExecutionId()`, so the window before and after an
 * execution's frame must not gain a store, only log attribution. Not
 * re-exported from the package index.
 */

import { AsyncLocalStorage } from "node:async_hooks";

const logCorrelationStorage = new AsyncLocalStorage<string | null>();

/**
 * Runtime accessors such as `getCurrentExecutionId` are unaffected. A blank
 * ID clears an inherited one, so work with no originating execution cannot
 * take its caller's.
 */
export function runWithLogCorrelation<T>(
  executionId: string | null | undefined,
  fn: () => T,
): T {
  return logCorrelationStorage.run(
    executionId && executionId.trim() ? executionId : null,
    fn,
  );
}

export function getLogCorrelation(): string | null {
  return logCorrelationStorage.getStore() ?? null;
}

// Identifies the workload request that owns flushing a writer's pending
// captured output; see structured_logging/stdio_capture.ts.
const logOutputScopeStorage = new AsyncLocalStorage<object | null>();

export function getLogOutputScope(): object | null {
  return logOutputScopeStorage.getStore() ?? null;
}

export function runWithLogOutputScope<T>(scope: object | null, fn: () => T): T {
  return logOutputScopeStorage.run(scope, fn);
}
