/**
 * Promise-chain mutex serializing async work within one process.
 *
 * Mirrors Python's asyncio.Lock on ToolServer._execute_gate. Lives in its own
 * module rather than in server/tool.ts so hooks.ts can take the same lock:
 * hooks.ts is imported by server/tool.ts, so importing back into it would be a
 * cycle. server/tool.ts re-exports this for the callers that already import it
 * from there.
 */
export class AsyncMutex {
  private locked: Promise<void> = Promise.resolve();

  private async acquire(): Promise<() => void> {
    const previous = this.locked;
    let release!: () => void;
    this.locked = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    return release;
  }

  async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }

  /**
   * Generator form of runExclusive: the lock is held for the lifetime of the
   * iteration and released when it completes, throws, or is closed early.
   */
  async *runExclusiveGen<T>(
    genFactory: () => AsyncGenerator<T>,
  ): AsyncGenerator<T> {
    const release = await this.acquire();
    try {
      yield* genFactory();
    } finally {
      release();
    }
  }
}
