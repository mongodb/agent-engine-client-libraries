/**
 * Port of the per-invocation build coverage in
 * `agent-engine-sdk-langgraph/tests/test_per_invocation_build.py`.
 *
 * Covers the `loadStrategy` opt-in, the per-execution build path, and the
 * registered-LLM-id invariant the Tool Pod relies on.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetHooks } from "@mongodb-js/agent-engine-runner-shared";

import { App, LLMRegistrationDriftError } from "../src/runtime.js";

const fakeLlm = { invoke: vi.fn() } as never;

let savedMode: string | undefined;

beforeEach(() => {
  savedMode = process.env["RUNNER_MODE"];
  process.env["RUNNER_MODE"] = "aer";
});

afterEach(() => {
  if (savedMode === undefined) delete process.env["RUNNER_MODE"];
  else process.env["RUNNER_MODE"] = savedMode;
  resetHooks();
});

describe("app.entrypoint loadStrategy", () => {
  it("defaults to global", () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(() => ({ graph: true }));
    expect(app.loadStrategy).toBe("global");
  });

  it("accepts per-execution with a two-argument builder", () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => ({ graph: true }),
      { loadStrategy: "per-execution" },
    );
    expect(app.loadStrategy).toBe("per-execution");
  });

  it("accepts a builder with an optional parameter on the default path", () => {
    const app = new App({ appName: "test-agent" });
    // TypeScript erases `?`, so emitted JavaScript cannot tell an optional
    // parameter from a required one. This builder was valid before the opt-in
    // existed and has to stay valid: the overloads are the arity gate, and a
    // runtime check would reject it.
    const withOptional = (opts?: { model?: string }) => ({ graph: opts });
    app.entrypoint(withOptional);

    expect(app.loadStrategy).toBe("global");
  });

  it("rejects an unknown strategy", () => {
    const app = new App({ appName: "test-agent" });
    expect(() =>
      app.entrypoint(() => ({ graph: true }), {
        loadStrategy: "sometimes" as never,
      }),
    ).toThrow(/loadStrategy must be one of/);
  });
});

describe("per-execution build", () => {
  it("builds a distinct graph per invocation", async () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => ({ graph: (builds += 1) }),
      { loadStrategy: "per-execution" },
    );

    await expect(app.buildGraph({ payload: {} }, {})).resolves.toEqual({
      graph: 1,
    });
    await expect(app.buildGraph({ payload: {} }, {})).resolves.toEqual({
      graph: 2,
    });
  });

  it("passes the invocation input and context to the builder", async () => {
    const app = new App({ appName: "test-agent" });
    const seen: unknown[] = [];
    app.entrypoint(
      async (input: unknown, ctx: unknown) => {
        seen.push(input, ctx);
        return { graph: true };
      },
      { loadStrategy: "per-execution" },
    );

    const input = { payload: { message: "hi" } };
    const ctx = { userId: "u1" };
    await app.buildGraph(input as never, ctx as never);
    expect(seen).toEqual([input, ctx]);
  });

  it("returns undefined for a global agent so the cached path is used", async () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(() => ({ graph: true }));
    await expect(app.buildGraph({ payload: {} }, {})).resolves.toBeUndefined();
  });

  it("does not build during warm-up", () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => ({ graph: (builds += 1) }),
      { loadStrategy: "per-execution" },
    );

    app.warmUp();
    expect(builds).toBe(0);
  });

  it("still builds during warm-up for a global agent", () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(() => ({ graph: (builds += 1) }));
    app.warmUp();
    expect(builds).toBe(1);
  });

  it("serializes overlapping builds so neither sees the other's registry", async () => {
    const app = new App({ appName: "test-agent" });
    const entered: string[] = [];
    let releaseFirst: () => void = () => {};
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });

    app.entrypoint(
      async (_input: unknown, ctx: unknown) => {
        const id = (ctx as { id: string }).id;
        entered.push(id);
        app.llm(fakeLlm, "primary");
        if (id === "a") await firstGate;
        // Registered after an await, so this also covers the scope surviving the
        // builder's own I/O.
        app.llm(fakeLlm, "secondary");
        return { graph: id };
      },
      { loadStrategy: "per-execution" },
    );

    const first = app.buildGraph(
      { payload: {} } as never,
      {
        id: "a",
      } as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    const second = app.buildGraph(
      { payload: {} } as never,
      {
        id: "b",
      } as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    // Serialized: the second builder has not entered while the first is held.
    // Without the lock it would have run, reset the registry, and made the
    // first build's second registration a duplicate-id throw.
    expect(entered).toEqual(["a"]);

    releaseFirst();
    const [a, b] = await Promise.all([first, second]);

    expect(entered).toEqual(["a", "b"]);
    expect([a, b]).toEqual([{ graph: "a" }, { graph: "b" }]);
  });
});

describe("registered LLM id invariant", () => {
  it("accepts a build that registers the same ids each time", async () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => {
        app.llm(fakeLlm, "primary");
        return { graph: true };
      },
      { loadStrategy: "per-execution" },
    );

    await app.buildGraph({ payload: {} }, {});
    await expect(app.buildGraph({ payload: {} }, {})).resolves.toEqual({
      graph: true,
    });
  });

  it("rejects a build that registers a different id set", async () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => {
        builds += 1;
        app.llm(fakeLlm, builds === 1 ? "primary" : "other");
        return { graph: true };
      },
      { loadStrategy: "per-execution" },
    );

    await app.buildGraph({ payload: {} }, {});
    await expect(app.buildGraph({ payload: {} }, {})).rejects.toThrow(
      /must register the same LLM ids on every invocation/,
    );
  });

  it("marks the drift error so runner-shared can detect it without importing this package", async () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => {
        builds += 1;
        app.llm(fakeLlm, builds === 1 ? "primary" : "other");
        return { graph: true };
      },
      { loadStrategy: "per-execution" },
    );

    await app.buildGraph({ payload: {} }, {});
    await expect(app.buildGraph({ payload: {} }, {})).rejects.toBeInstanceOf(
      LLMRegistrationDriftError,
    );
    await expect(
      app.buildGraph({ payload: {} }, {}).catch((e: unknown) => e),
    ).resolves.toMatchObject({ isLlmRegistrationDrift: true });
  });

  it("lets a provisional discovery baseline grow, since the pod only gains ids", async () => {
    const app = new App({ appName: "test-agent" });
    let builds = 0;
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => {
        builds += 1;
        app.llm(fakeLlm, "primary");
        if (builds >= 2) app.llm(fakeLlm, "secondary");
        throw new Error("needs invocation context");
      },
      { loadStrategy: "per-execution" },
    );

    // A context-free discovery build: it registers, then raises.
    await expect(app.buildGraph()).rejects.toThrow(
      /context-free discovery path/,
    );
    // The second discovery registers strictly more, which is a recovery.
    await expect(app.buildGraph()).rejects.toThrow(
      /context-free discovery path/,
    );
  });

  it("reports a context-free build failure that cannot reach its registrations", async () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => {
        throw new Error("needs invocation context");
      },
      { loadStrategy: "per-execution" },
    );

    await expect(app.buildGraph()).rejects.toThrow(
      /context-free discovery path/,
    );
  });
});

describe("materializing an opted-in agent", () => {
  it("refuses to build through the cached path, which would store a pending Promise", () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => ({ graph: true }),
      {
        loadStrategy: "per-execution",
      },
    );

    expect(() => app.getAgent()).toThrow(/builds its graph per invocation/);
  });
});

describe("a build whose caller gives up", () => {
  it("refuses a queued build whose deadline passed, then resumes after the first settles", async () => {
    const app = new App({ appName: "test-agent" });
    const entered: string[] = [];
    let releaseFirst: () => void = () => {};
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });

    app.entrypoint(
      async (_input: unknown, ctx: unknown) => {
        const id = (ctx as { id: string }).id;
        entered.push(id);
        if (id === "a") await firstGate;
        return { graph: id };
      },
      { loadStrategy: "per-execution" },
    );

    // A holds the build lock and is still running.
    const first = app.buildGraph(
      { payload: {} } as never,
      {
        id: "a",
      } as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(entered).toEqual(["a"]);

    // B queues behind A and its own invocation deadline passes while it waits.
    const bController = new AbortController();
    const second = app.buildGraph(
      { payload: {} } as never,
      {
        id: "b",
        signal: bController.signal,
      } as never,
    );
    // Capture the outcome now: a queued build rejects as soon as its signal
    // aborts, which is well before the assertion below, and an unhandled
    // window in between would be flagged.
    const secondOutcome = second.then(
      () => null,
      (error: unknown) => error,
    );
    bController.abort();
    await new Promise((resolve) => setTimeout(resolve, 0));

    // C queues with a live deadline.
    const third = app.buildGraph(
      { payload: {} } as never,
      {
        id: "c",
      } as never,
    );

    releaseFirst();

    await expect(first).resolves.toEqual({ graph: "a" });
    // B never entered the builder, even though the lock came free for it.
    const secondError = await secondOutcome;
    expect(secondError).toBeInstanceOf(Error);
    expect((secondError as Error).message).toMatch(
      /abandoned before it started/,
    );
    // C resumes normally once the abandoned build released the lock.
    await expect(third).resolves.toEqual({ graph: "c" });
    expect(entered).toEqual(["a", "c"]);
  });

  it("leaves no unhandled rejection when the signal is already aborted", async () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, _ctx: unknown) => ({ graph: true }),
      {
        loadStrategy: "per-execution",
      },
    );
    const controller = new AbortController();
    controller.abort();

    const rejections: unknown[] = [];
    const onRejection = (reason: unknown): void => {
      rejections.push(reason);
    };
    process.on("unhandledRejection", onRejection);
    try {
      await expect(
        app.buildGraph(
          { payload: {} } as never,
          {
            signal: controller.signal,
          } as never,
        ),
      ).rejects.toThrow(/abandoned before it started/);
      // Let the queued callback run and reject, and the turn after it. An
      // unobserved rejection there is fatal to the runner process.
      await new Promise((resolve) => setTimeout(resolve, 20));
    } finally {
      process.removeListener("unhandledRejection", onRejection);
    }

    expect(rejections).toEqual([]);
  });

  it("leaves a global agent unaffected", async () => {
    const app = new App({ appName: "test-agent" });
    app.entrypoint(() => ({ graph: true }));

    await expect(app.buildGraph({ payload: {} }, {})).resolves.toBeUndefined();
  });
});
