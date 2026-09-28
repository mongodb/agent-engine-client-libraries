/**
 * Per-call Stop opt-in for local (callback-routed) tools.
 *
 * Node has no preemptive cancel: the runtime can stop *waiting* on a tool
 * body, but only the body can stop its own work. A tool branded with
 * {@link withCallInterruptSupport} declares that its body consumes the
 * per-call signal (via `getCallAbortSignal()` from
 * `@mongodb-js/agent-engine-runner-shared`) and stops at its checkpoints.
 * Only branded tools answer a per-call Stop with `interrupted`; unbranded
 * tools honestly answer `not_cancellable` and run out.
 */

/**
 * Brand a tool as supporting per-call interrupt. The body must honor the
 * signal from `getCallAbortSignal()` — check `signal.aborted` between steps
 * or pass the signal to APIs that accept one (`fetch`, streams).
 *
 * Returns the same tool for chaining at definition time.
 */
export function withCallInterruptSupport<T>(tool: T): T {
  (tool as { supportsCallInterrupt?: boolean }).supportsCallInterrupt = true;
  return tool;
}
