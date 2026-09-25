/**
 * Tests for emit / emitStep.
 *
 * Mirrors Python's tests/unit/test_emit_progress.py. Two Python tests are not
 * ported:
 *   - `test_emit_reuses_module_level_client` — TS has no module-level client;
 *     connection pooling / cert rotation live in `tls_client.ts`
 *     (`getFetchOptionsWithTLS`), out of scope for this module.
 *   - the httpx-specific `InvalidURL` case folds into the generic
 *     "swallows network error" test — `fetch` rejects the same way.
 *
 * Context is established with `runWithExecutionContext` (the TS equivalent of
 * Python's `set_execution_context` + try/finally `clear_execution_context`).
 * `fetch` is stubbed via `vi.stubGlobal`.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { describe, test, expect, vi, afterEach } from "vitest";

import { emit, emitStep } from "../../src/progress.js";
import { runWithExecutionContext } from "../../src/context.js";
import {
  DONE,
  ERROR,
  STEP,
  SUBAGENT_END,
  SUBAGENT_START,
  TEXT,
} from "../../src/server/chunk_types.js";

function withCtx(
  executionId: string,
  oeUrl: string,
  fn: () => Promise<void>,
): Promise<void> {
  return runWithExecutionContext({ executionId, wrapper: null, oeUrl }, fn);
}

function withOwnerCtx(
  executionId: string,
  oeUrl: string,
  oeOwnerUrl: string,
  fn: () => Promise<void>,
): Promise<void> {
  return runWithExecutionContext(
    { executionId, wrapper: null, oeUrl, oeOwnerUrl },
    fn,
  );
}

/** A fetch stub that records calls and returns a 200 OK. */
function okFetch(): ReturnType<typeof vi.fn> {
  return vi.fn(async () => new Response(null, { status: 200 }));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Happy path — emitStep (the common case)
// ---------------------------------------------------------------------------

test("emitStep posts a step chunk to /stream/chunk", async () => {
  const mock = okFetch();
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-1", "http://oe:8000", async () => {
    await emitStep("Fetching page...");
  });

  expect(mock).toHaveBeenCalledTimes(1);
  const [url, init] = mock.mock.calls[0] as [string, RequestInit];
  expect(url).toBe("http://oe:8000/stream/chunk");
  expect(init.method).toBe("POST");
  expect(JSON.parse(init.body as string)).toEqual({
    execution_id: "exec-1",
    chunk_type: STEP,
    content: "Fetching page...",
    metadata: {},
  });
});

test("emitStep strips a trailing slash from oeUrl", async () => {
  const mock = okFetch();
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-2", "http://oe:8000/", async () => {
    await emitStep("Parsing...");
  });

  const [url] = mock.mock.calls[0] as [string, RequestInit];
  expect(url).toBe("http://oe:8000/stream/chunk");
});

// ---------------------------------------------------------------------------
// General emit() with a non-step event type
// ---------------------------------------------------------------------------

test("emit accepts an arbitrary (non-reserved) event type", async () => {
  const mock = okFetch();
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-emit", "http://oe:8000", async () => {
    await emit("custom_event", "payload");
  });

  const [, init] = mock.mock.calls[0] as [string, RequestInit];
  expect(JSON.parse(init.body as string)).toEqual({
    execution_id: "exec-emit",
    chunk_type: "custom_event",
    content: "payload",
    metadata: {},
  });
});

// ---------------------------------------------------------------------------
// Owner-callback preference
// ---------------------------------------------------------------------------

describe("emit owner-callback preference", () => {
  const OWNER = "http://10-1-2-3.oe-headless.ns:8000";

  test("prefers the owner URL when one is set on the context", async () => {
    const mock = okFetch();
    vi.stubGlobal("fetch", mock);

    await withOwnerCtx("exec-owner", "http://oe:8000", OWNER, async () => {
      await emitStep("hello");
    });

    expect(mock).toHaveBeenCalledTimes(1);
    const [url] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${OWNER}/stream/chunk`);
  });

  test("falls back to the service URL on an owner transport error with an identical body", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const mock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse((init?.body as string) ?? "null") });
      if (url.startsWith(OWNER)) throw new Error("connection refused");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await withOwnerCtx("exec-fb", "http://oe:8000", OWNER, async () => {
      await emitStep("hi");
    });

    expect(calls.map((c) => c.url)).toEqual([
      `${OWNER}/stream/chunk`,
      "http://oe:8000/stream/chunk",
    ]);
    // Same chunk body delivered to the fallback target.
    expect(calls[0].body).toEqual(calls[1].body);
  });

  test("falls back to the service URL on an owner HTTP error; owner not retried", async () => {
    const ownerStatus = 500;
    const calls: Array<{
      url: string;
      body: unknown;
      redirect: RequestRedirect | undefined;
    }> = [];
    const mock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({
        url,
        body: JSON.parse((init?.body as string) ?? "null"),
        redirect: init?.redirect,
      });
      if (url.startsWith(OWNER))
        return new Response(null, { status: ownerStatus });
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await withOwnerCtx("exec-owner-http", "http://oe:8000", OWNER, async () => {
      await emitStep("hi");
    });

    expect(calls.map((c) => c.url)).toEqual([
      `${OWNER}/stream/chunk`,
      "http://oe:8000/stream/chunk",
    ]);
    expect(calls.filter((c) => c.url.startsWith(OWNER))).toHaveLength(1);
    expect(calls[0]?.redirect).toBe("manual");
    expect(calls[1]?.redirect).toBeUndefined();
    expect(calls[0]?.body).toEqual(calls[1]?.body);
  });

  test("owner timeout does not pre-abort the service fallback", async () => {
    const ownerTimeout = new AbortController();
    const serviceTimeout = new AbortController();
    const timeoutSpy = vi
      .spyOn(AbortSignal, "timeout")
      .mockImplementationOnce(() => ownerTimeout.signal)
      .mockImplementationOnce(() => serviceTimeout.signal);
    const calls: Array<{ url: string; signal: AbortSignal | undefined }> = [];
    const mock = vi.fn(
      async (url: string, init?: RequestInit): Promise<Response> => {
        const signal = init?.signal as AbortSignal | undefined;
        calls.push({ url, signal });
        if (url.startsWith(OWNER)) {
          ownerTimeout.abort(new Error("owner timed out"));
          throw signal?.reason ?? new Error("owner timed out");
        }
        if (signal?.aborted) {
          throw signal.reason ?? new Error("service received aborted signal");
        }
        return new Response(null, { status: 200 });
      },
    );
    vi.stubGlobal("fetch", mock);

    await withOwnerCtx(
      "exec-owner-timeout",
      "http://oe:8000",
      OWNER,
      async () => {
        await emitStep("hi");
      },
    );

    expect(timeoutSpy).toHaveBeenCalledTimes(2);
    expect(calls.map((c) => c.url)).toEqual([
      `${OWNER}/stream/chunk`,
      "http://oe:8000/stream/chunk",
    ]);
    expect(calls[0]?.signal).not.toBe(calls[1]?.signal);
    expect(calls[0]?.signal?.aborted).toBe(true);
    expect(calls[1]?.signal?.aborted).toBe(false);
  });

  test("a failed owner pre-attempt latches: later emits skip the owner", async () => {
    const urls: string[] = [];
    const mock = vi.fn(async (url: string) => {
      urls.push(url);
      if (url.startsWith(OWNER)) throw new Error("connection refused");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await withOwnerCtx("exec-latch", "http://oe:8000", OWNER, async () => {
      await emitStep("first");
      await emitStep("second");
    });

    // The owner is paid for exactly once; the second emit goes straight to
    // the service instead of re-paying the pre-attempt timeout.
    expect(urls).toEqual([
      `${OWNER}/stream/chunk`,
      "http://oe:8000/stream/chunk",
      "http://oe:8000/stream/chunk",
    ]);
  });

  test("the owner-failure latch is per-execution", async () => {
    const runOnce = async (executionId: string): Promise<string[]> => {
      const urls: string[] = [];
      const mock = vi.fn(async (url: string) => {
        urls.push(url);
        if (url.startsWith(OWNER)) throw new Error("connection refused");
        return new Response(null, { status: 200 });
      });
      vi.stubGlobal("fetch", mock);
      await withOwnerCtx(executionId, "http://oe:8000", OWNER, async () => {
        await emitStep("hello");
      });
      return urls;
    };

    const expected = [`${OWNER}/stream/chunk`, "http://oe:8000/stream/chunk"];
    expect(await runOnce("exec-latch-a")).toEqual(expected);
    // A fresh execution context offers the owner again.
    expect(await runOnce("exec-latch-b")).toEqual(expected);
  });

  test("uses only the service URL when no owner is set", async () => {
    const mock = okFetch();
    vi.stubGlobal("fetch", mock);

    await withCtx("exec-no-owner", "http://oe:8000", async () => {
      await emitStep("hi");
    });

    expect(mock).toHaveBeenCalledTimes(1);
    const [url] = mock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://oe:8000/stream/chunk");
  });
});

// ---------------------------------------------------------------------------
// No-op outside execution context
// ---------------------------------------------------------------------------

test("emit is a no-op outside an execution context", async () => {
  const mock = okFetch();
  vi.stubGlobal("fetch", mock);

  await emitStep("should be ignored");
  await emit("step", "also ignored");

  expect(mock).not.toHaveBeenCalled();
});

// ---------------------------------------------------------------------------
// Failures are swallowed — a bad emit must never abort the tool
// ---------------------------------------------------------------------------

test("emit swallows a network error", async () => {
  const mock = vi.fn(async () => {
    throw new Error("connection refused");
  });
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-3", "http://oe:8000", async () => {
    await expect(emitStep("Running...")).resolves.toBeUndefined();
  });
});

test("emit swallows a non-2xx response", async () => {
  const mock = vi.fn(async () => new Response(null, { status: 500 }));
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-4", "http://oe:8000", async () => {
    await expect(emitStep("Running...")).resolves.toBeUndefined();
  });
  expect(mock).toHaveBeenCalledTimes(1);
});

test("emit cancels the response body so undici can reuse the connection", async () => {
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    cancel() {
      cancelled = true;
    },
  });
  const mock = vi.fn(async () => new Response(body, { status: 200 }));
  vi.stubGlobal("fetch", mock);

  await withCtx("exec-body", "http://oe:8000", async () => {
    await emitStep("x");
  });

  expect(cancelled).toBe(true);
});

// ---------------------------------------------------------------------------
// Reserved event types — tool code must not close or spoof the stream
// ---------------------------------------------------------------------------

describe("reserved event types are rejected and never POST", () => {
  for (const reserved of [DONE, ERROR, TEXT, SUBAGENT_START, SUBAGENT_END]) {
    test(`emit('${reserved}') rejects`, async () => {
      const mock = okFetch();
      vi.stubGlobal("fetch", mock);

      await withCtx("exec-reserved", "http://oe:8000", async () => {
        await expect(emit(reserved, "should not be sent")).rejects.toThrow(
          /reserved/,
        );
      });
      expect(mock).not.toHaveBeenCalled();
    });
  }
});

test("reserved-event check fires even outside an execution context", async () => {
  const mock = okFetch();
  vi.stubGlobal("fetch", mock);

  await expect(emit(DONE, "x")).rejects.toThrow(/reserved/);
  expect(mock).not.toHaveBeenCalled();
});

// ---------------------------------------------------------------------------
// Integration: real HTTP server — validates wire format without mocking fetch
// ---------------------------------------------------------------------------

test("emitStep sends a correctly-formed JSON body to a real TCP server", async () => {
  const received: unknown[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      received.push(JSON.parse(Buffer.concat(chunks).toString("utf-8")));
      res.statusCode = 200;
      res.end();
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;

  try {
    await withCtx("exec-wire", `http://127.0.0.1:${port}`, async () => {
      await emitStep("Integration check");
    });
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  expect(received).toHaveLength(1);
  expect(received[0]).toEqual({
    execution_id: "exec-wire",
    chunk_type: STEP,
    content: "Integration check",
    metadata: {},
  });
});
