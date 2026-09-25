/**
 * Hijacked /invoke_llm/stream must not crash the tool
 * pod on client disconnect, must stop consuming on close, and must not buffer
 * unboundedly for never-reading clients.
 */

import { EventEmitter } from "node:events";
import { describe, test, expect, afterEach } from "vitest";
import { pipeSseLinesToResponse } from "../../src/server/tool.js";

type MockRes = EventEmitter & {
  write: (chunk: string) => boolean;
  end: () => void;
  destroy: () => void;
  destroyed: boolean;
  writableEnded: boolean;
};

function makeMockRes(
  opts: {
    writeReturnsFalseUntilDrain?: boolean;
  } = {},
): {
  res: MockRes;
  written: string[];
  endCalls: { n: number };
  destroyCalls: { n: number };
} {
  const written: string[] = [];
  const endCalls = { n: 0 };
  const destroyCalls = { n: 0 };
  const res = new EventEmitter() as MockRes;
  res.destroyed = false;
  res.writableEnded = false;
  let backpressure = opts.writeReturnsFalseUntilDrain === true;

  res.write = (chunk: string): boolean => {
    if (res.destroyed || res.writableEnded) {
      const err = new Error("write after end");
      queueMicrotask(() => res.emit("error", err));
      return false;
    }
    written.push(chunk);
    if (backpressure) {
      // First write reports backpressure; subsequent writes after drain succeed.
      backpressure = false;
      return false;
    }
    return true;
  };

  res.end = (): void => {
    endCalls.n += 1;
    res.writableEnded = true;
    res.emit("finish");
    // Mirror Node: close follows a successful end.
    queueMicrotask(() => res.emit("close"));
  };

  res.destroy = (): void => {
    destroyCalls.n += 1;
    res.destroyed = true;
    queueMicrotask(() => res.emit("close"));
  };

  return { res, written, endCalls, destroyCalls };
}

async function* linesOf(
  items: string[],
  onPull?: () => void,
): AsyncGenerator<string> {
  for (const item of items) {
    onPull?.();
    yield item;
  }
}

describe("pipeSseLinesToResponse", () => {
  const uncaught: Error[] = [];
  const onUncaught = (err: Error): void => {
    uncaught.push(err);
  };

  afterEach(() => {
    process.off("uncaughtException", onUncaught);
    uncaught.length = 0;
  });

  test("writes all lines and ends the response on the happy path", async () => {
    const { res, written, endCalls, destroyCalls } = makeMockRes();

    await pipeSseLinesToResponse(
      res as never,
      linesOf(["data: a\n\n", "data: b\n\n"]),
    );

    expect(written).toEqual(["data: a\n\n", "data: b\n\n"]);
    expect(endCalls.n).toBe(1);
    expect(destroyCalls.n).toBe(0);
    expect(res.writableEnded).toBe(true);
  });

  test("an external abort signal unblocks a parked iterator", async () => {
    // An execution drain reaches the pipe via opts.signal: even when the
    // upstream generator never yields and ignores every signal it was handed,
    // the pipe must stop waiting and force-close.
    const { res, destroyCalls } = makeMockRes();
    const external = new AbortController();
    const parked = () =>
      (async function* (): AsyncGenerator<string> {
        await new Promise(() => {});
        yield "data: unreachable\n\n";
      })();

    const pipe = pipeSseLinesToResponse(res as never, parked, {
      signal: external.signal,
    });
    external.abort();
    await pipe;

    expect(destroyCalls.n).toBe(1);
  });

  test("onGeneratorSettled fires on the graceful path before the pipe resolves", async () => {
    const { res } = makeMockRes();
    let settled = false;
    await pipeSseLinesToResponse(res as never, linesOf(["data: a\n\n"]), {
      onGeneratorSettled: () => {
        settled = true;
      },
    });
    expect(settled).toBe(true);
  });

  test("onGeneratorSettled waits out a signal-ignoring generator", async () => {
    // Force-close destroys the socket promptly, but the settlement callback
    // must not fire while the generator is still parked — a drain keyed off
    // it would otherwise report completed with provider work in flight. It
    // fires when the generator's pending await resolves and return() is
    // actually processed.
    const { res, destroyCalls } = makeMockRes();
    const external = new AbortController();
    let releaseGenerator!: () => void;
    const parked = new Promise<void>((r) => {
      releaseGenerator = r;
    });
    let unwound = false;
    const ignoring = () =>
      (async function* (): AsyncGenerator<string> {
        try {
          await parked; // never observes the abort signal
        } finally {
          unwound = true;
        }
        yield "data: late\n\n";
      })();

    let settled = false;
    const pipe = pipeSseLinesToResponse(res as never, ignoring, {
      signal: external.signal,
      onGeneratorSettled: () => {
        settled = true;
      },
    });
    external.abort();
    await pipe;
    expect(destroyCalls.n).toBe(1);
    for (let i = 0; i < 20 && !settled; i++) {
      await new Promise((r) => setImmediate(r));
    }
    expect(settled).toBe(false);
    expect(unwound).toBe(false);

    releaseGenerator();
    for (let i = 0; i < 50 && !settled; i++) {
      await new Promise((r) => setImmediate(r));
    }
    expect(unwound).toBe(true);
    expect(settled).toBe(true);
  });

  test("response close before end aborts further pulls", async () => {
    const { res, written, destroyCalls } = makeMockRes();
    const pulls: number[] = [];

    async function* slowLines(): AsyncGenerator<string> {
      for (let i = 0; i < 10; i++) {
        pulls.push(i);
        yield `data: ${i}\n\n`;
        if (i === 0) {
          // Peer dropped the connection mid-stream (writableEnded still false).
          res.emit("close");
        }
        await new Promise<void>((resolve) => queueMicrotask(resolve));
      }
    }

    await pipeSseLinesToResponse(res as never, slowLines());

    expect(written.length).toBeGreaterThanOrEqual(1);
    expect(pulls.length).toBeLessThan(10);
    expect(destroyCalls.n).toBe(1);
  });

  test("request-body close must not abort a healthy stream", async () => {
    // Regression for Augment: IncomingMessage `close` fires after Fastify has
    // already consumed the POST body; aborting on that would send zero chunks.
    const { res, written } = makeMockRes();
    const req = new EventEmitter();

    async function* slowLines(): AsyncGenerator<string> {
      yield "data: a\n\n";
      req.emit("close");
      await new Promise<void>((resolve) => queueMicrotask(resolve));
      yield "data: b\n\n";
    }

    await pipeSseLinesToResponse(res as never, slowLines());

    expect(written).toEqual(["data: a\n\n", "data: b\n\n"]);
    expect(res.writableEnded).toBe(true);
  });

  test("socket error after write does not crash the process", async () => {
    process.on("uncaughtException", onUncaught);
    const { res, written } = makeMockRes();

    async function* noisyLines(): AsyncGenerator<string> {
      yield "data: first\n\n";
      queueMicrotask(() => {
        res.destroyed = true;
        res.emit("error", new Error("socket hang up"));
      });
      await new Promise<void>((resolve) => queueMicrotask(resolve));
      yield "data: after-error\n\n";
      await new Promise<void>((resolve) => queueMicrotask(resolve));
      yield "data: should-not-write\n\n";
    }

    await expect(
      pipeSseLinesToResponse(res as never, noisyLines()),
    ).resolves.toBeUndefined();

    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(written[0]).toBe("data: first\n\n");
    expect(written).not.toContain("data: should-not-write\n\n");
    expect(uncaught).toEqual([]);
  });

  test("error emitted asynchronously during end does not crash", async () => {
    process.on("uncaughtException", onUncaught);
    const { res, written } = makeMockRes();
    const originalEnd = res.end.bind(res);
    res.end = (): void => {
      originalEnd();
      // Race Augment flagged: error after listeners were previously detached.
      queueMicrotask(() => res.emit("error", new Error("EPIPE during end")));
    };

    await pipeSseLinesToResponse(res as never, linesOf(["data: a\n\n"]));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(written).toEqual(["data: a\n\n"]);
    expect(uncaught).toEqual([]);
  });

  test("backpressure waits for drain then continues", async () => {
    const { res, written } = makeMockRes({ writeReturnsFalseUntilDrain: true });

    const pipePromise = pipeSseLinesToResponse(
      res as never,
      linesOf(["data: a\n\n", "data: b\n\n"]),
      { drainTimeoutMs: 500 },
    );

    // Wait until the pipe is parked on drain — a single microtask is not
    // enough once next() is raced via promises.
    const started = Date.now();
    while (res.listenerCount("drain") === 0) {
      if (Date.now() - started > 1000) {
        throw new Error("timed out waiting for drain listener");
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 5));
    }
    res.emit("drain");
    await pipePromise;

    expect(written).toEqual(["data: a\n\n", "data: b\n\n"]);
    expect(res.writableEnded).toBe(true);
  });

  test("drain timeout destroys the socket instead of end()", async () => {
    const { res, written, endCalls, destroyCalls } = makeMockRes({
      writeReturnsFalseUntilDrain: true,
    });
    const pulls: string[] = [];

    const started = Date.now();
    await pipeSseLinesToResponse(
      res as never,
      linesOf(["data: a\n\n", "data: b\n\n", "data: c\n\n"], () => {
        pulls.push("pull");
      }),
      { drainTimeoutMs: 50 },
    );
    const elapsed = Date.now() - started;

    expect(written).toEqual(["data: a\n\n"]);
    expect(pulls.length).toBe(1);
    expect(elapsed).toBeLessThan(1000);
    expect(endCalls.n).toBe(0);
    expect(destroyCalls.n).toBe(1);
    expect(res.destroyed).toBe(true);
  });

  test("abort while awaiting next() returns promptly and destroys", async () => {
    const { res, written, destroyCalls, endCalls } = makeMockRes();

    async function* hungAfterFirst(
      signal: AbortSignal,
    ): AsyncGenerator<string> {
      yield "data: first\n\n";
      // Simulate a hung provider / backoff that honors the pipe signal.
      await new Promise<void>((resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException("This operation was aborted", "AbortError"));
          return;
        }
        const timer = setTimeout(resolve, 60_000);
        signal.addEventListener(
          "abort",
          () => {
            clearTimeout(timer);
            reject(
              new DOMException("This operation was aborted", "AbortError"),
            );
          },
          { once: true },
        );
      });
      yield "data: should-not-emit\n\n";
    }

    const pipePromise = pipeSseLinesToResponse(res as never, (signal) =>
      hungAfterFirst(signal),
    );
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    // Let the generator park on the hung await, then drop the peer.
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    res.emit("close");

    const started = Date.now();
    await pipePromise;
    expect(Date.now() - started).toBeLessThan(500);

    expect(written).toEqual(["data: first\n\n"]);
    expect(destroyCalls.n).toBe(1);
    expect(endCalls.n).toBe(0);
  });

  test("signal factory receives the pipe AbortSignal", async () => {
    const { res, written } = makeMockRes();
    let seenSignal: AbortSignal | undefined;

    await pipeSseLinesToResponse(res as never, (signal) => {
      seenSignal = signal;
      return linesOf(["data: a\n\n"]);
    });

    expect(seenSignal).toBeInstanceOf(AbortSignal);
    expect(written).toEqual(["data: a\n\n"]);
    expect(res.writableEnded).toBe(true);
  });

  test("generator errors are swallowed and the response is force-closed", async () => {
    const { res, destroyCalls } = makeMockRes();

    async function* failing(): AsyncGenerator<string> {
      yield "data: ok\n\n";
      throw new Error("upstream boom");
    }

    await expect(
      pipeSseLinesToResponse(res as never, failing()),
    ).resolves.toBeUndefined();
    expect(destroyCalls.n).toBe(1);
    expect(res.destroyed).toBe(true);
  });
});
