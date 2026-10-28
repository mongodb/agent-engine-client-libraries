/**
 * Deadline helper for awaiting work that must not outlive its budget.
 *
 * Not re-exported from the package index: it is an internal building block, not
 * part of the SDK surface. `TimeoutError` in particular must stay internal,
 * because its name is the DOM and Undici convention that the tool-call path
 * already classifies by name.
 */

/**
 * The work did not settle within its budget.
 *
 * Distinct from the work's own rejection so a caller can tell "the deadline
 * passed while this was still running" from "this failed", which need different
 * handling: the former leaves the work live.
 */
export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TimeoutError";
  }
}

/**
 * Reject with a {@link TimeoutError} when `work` has not settled within `ms`.
 *
 * The work is not cancelled. Callers use this for author code that owns its own
 * I/O, where abandoning the promise would leave the work running against
 * process-global state rather than stopping it. Failing the caller is the point.
 */
export function withTimeout<T>(
  work: Promise<T>,
  ms: number,
  message: () => string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const handle = setTimeout(() => reject(new TimeoutError(message())), ms);
    work.then(
      (value) => {
        clearTimeout(handle);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(handle);
        reject(error);
      },
    );
  });
}
