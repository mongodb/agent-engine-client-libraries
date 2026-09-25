/**
 * Tests for the structured JSON logging formatter and stdout/stderr
 * capture. Mirrors `runner-shared/tests/unit/test_structured_logging.py`.
 *
 * 18 of the Python file's 20 tests are ported here. The other two are parked
 * for fundamental design reasons (not internals-leak):
 *
 *   - `test_logging_stream_is_thread_safe` — Node's event loop is single-
 *     threaded; `LoggingStream.write` is synchronous so concurrent writes
 *     cannot interleave bytes. The Python `RLock` has no TS analog and the
 *     property the test guards is trivially true here.
 *   - `test_unwrap_logging_stream_is_recursive` — TS uses a `Symbol.for(...)`
 *     slot on the stream itself to stash the original `.write`, not a
 *     recursive `LoggingStream`-wrapping-`LoggingStream` chain. The
 *     equivalent invariant (re-install doesn't recurse) is already covered
 *     by `install is idempotent` below.
 *
 * Fixture design — `setupCapture()`:
 *   Python tests inject a `StringIO` as the `stream=` argument to
 *   `install_structured_logging` so JSON lines accumulate in a buffer. TS
 *   `installStructuredLogging` has no `stream=` arg — the appender always
 *   writes through the snapshotted real `process.stdout.write` (see
 *   `structured_logging.ts:420-432`). To capture, we replace
 *   `process.stdout.write` (and stderr) BEFORE install, clear the
 *   `Symbol.for("agent_engine_runner_shared.original_write")` slot so the appender
 *   re-snapshots, and read records from the captureWrite's parsed buffer.
 *   `patchStdio` then overwrites `process.stdout.write` again (with
 *   `LoggingStream.write`), but the appender still holds a closure ref to
 *   our captureWrite via its earlier snapshot — so logger emits AND
 *   console.log/console.error all flow through captureWrite. `afterEach`
 *   restores the original writes, clears the Symbol slot, and shuts down
 *   log4js so the next test installs cleanly.
 */

import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import log4js, { type Logger } from "log4js";

import {
  installStructuredLogging,
  LoggingStream,
  MAX_BUFFER_BYTES,
  MAX_TRACEBACK_BYTES,
  FALLBACK_EXIT_MS,
  type InstallStructuredLoggingArgs,
} from "../../src/structured_logging.js";
import { setupLogging, getLogger } from "../../src/logger.js";
import {
  runWithCustomerOrigin,
  runWithExecutionContext,
} from "../../src/context.js";

// Tests that call `fx.install()` directly (bypassing setupLogging) MUST use
// log4js.getLogger rather than agent_engine_runner_shared/logger.ts's getLogger
// wrapper. The wrapper's `ensureDefaultConfig()` side effect fires on first
// call and overrides our just-installed structured config with the human-
// pattern config — which routes records back through process.stdout.write
// (now LoggingStream.write after patchStdio), causing infinite recursion.
// Going straight to log4js sidesteps that entirely.
const lg = (name: string): Logger => log4js.getLogger(name);

const ORIGINAL_WRITE_SLOT = Symbol.for(
  "agent_engine_runner_shared.original_write",
);

interface JsonRecord {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  service: string;
  tenantId: string | null;
  projectId: string | null;
  workspaceId: string | null;
  executionId: string | null;
  sessionId: string | null;
  bootId: string | null;
  podName: string | null;
  source: string;
  origin?: string;
  traceId?: string;
  fields: Record<string, unknown>;
}

interface Fixture {
  records: JsonRecord[];
  rawLines: string[];
  /** The capture function written to process.stdout.write before install. */
  captureWrite: NodeJS.WriteStream["write"];
  /** Re-invokes installStructuredLogging with merged options. */
  install: (opts?: Partial<InstallStructuredLoggingArgs>) => void;
}

function makeCaptureWrite(
  records: JsonRecord[],
  rawLines: string[],
): NodeJS.WriteStream["write"] {
  let buffer = "";
  const writer = ((chunk: unknown, ...rest: unknown[]): boolean => {
    const text =
      typeof chunk === "string"
        ? chunk
        : Buffer.isBuffer(chunk)
          ? chunk.toString("utf-8")
          : String(chunk);
    buffer += text;
    let nl = buffer.indexOf("\n");
    while (nl !== -1) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (line.trim()) {
        rawLines.push(line);
        try {
          records.push(JSON.parse(line) as JsonRecord);
        } catch {
          // Non-JSON line (legacy human-readable formatter, debug noise) —
          // intentionally swallowed; rawLines still has the original text.
        }
      }
      nl = buffer.indexOf("\n");
    }
    // Honor write(chunk, cb) / write(chunk, encoding, cb) so callers
    // waiting on drain don't hang.
    const maybeCb = rest.find((a) => typeof a === "function");
    if (typeof maybeCb === "function")
      (maybeCb as (err?: Error | null) => void)();
    return true;
  }) as NodeJS.WriteStream["write"];
  return writer;
}

let realStdoutWrite: NodeJS.WriteStream["write"];
let realStderrWrite: NodeJS.WriteStream["write"];

function setupCapture(): Fixture {
  // Clear any prior Symbol slot so the upcoming install re-snapshots the
  // patched write below (rather than reusing a slot left behind by a
  // previous test that escaped afterEach cleanup).
  delete (process.stdout as unknown as Record<symbol, unknown>)[
    ORIGINAL_WRITE_SLOT
  ];
  delete (process.stderr as unknown as Record<symbol, unknown>)[
    ORIGINAL_WRITE_SLOT
  ];

  const records: JsonRecord[] = [];
  const rawLines: string[] = [];
  const captureWrite = makeCaptureWrite(records, rawLines);
  process.stdout.write = captureWrite;
  process.stderr.write = captureWrite;

  const install = (opts: Partial<InstallStructuredLoggingArgs> = {}): void => {
    installStructuredLogging({ level: "info", mode: "aer", ...opts });
  };

  return { records, rawLines, captureWrite, install };
}

const UNCAUGHT_HANDLERS_REGISTERED = Symbol.for(
  "agent_engine_runner_shared.structured_logging_uncaught_handlers_registered",
);
const CRASH_EVENTS = [
  "uncaughtExceptionMonitor",
  "unhandledRejection",
] as const;

// Crash listeners present before the current test, so cleanup can remove
// exactly what the test added. Snapshotted per test rather than once, because
// the framework rotates its own listener between tests — a stale global
// baseline would make cleanup delete the framework's live listener.
type CrashListener = (...args: never[]) => void;
const listenersBeforeTest = new Map<string, CrashListener[]>();

// Detach what this test added, and only then clear the registration guard.
// Clearing the guard while a listener is still attached is what lets a second
// runner listener install alongside the first.
function detachRunnerCrashListeners(): void {
  for (const event of CRASH_EVENTS) {
    const keep = listenersBeforeTest.get(event) ?? [];
    for (const listener of process.listeners(event) as CrashListener[]) {
      if (!keep.includes(listener)) {
        process.removeListener(event, listener as never);
      }
    }
  }
  delete (globalThis as unknown as Record<symbol, unknown>)[
    UNCAUGHT_HANDLERS_REGISTERED
  ];
}

beforeEach(() => {
  for (const event of CRASH_EVENTS) {
    listenersBeforeTest.set(event, process.listeners(event) as CrashListener[]);
  }
  realStdoutWrite = process.stdout.write.bind(process.stdout);
  realStderrWrite = process.stderr.write.bind(process.stderr);
  // Default env mirrors Python `install_to_buffer`'s monkeypatch block.
  // `envOrNull` treats empty string as null, so stubbing WORKSPACE_ID to ""
  // matches Python's `monkeypatch.delenv("WORKSPACE_ID")`.
  vi.stubEnv("RUNNER_MODE", "aer");
  vi.stubEnv("ORG_ID", "tenant-acme");
  vi.stubEnv("PROJECT_ID", "project-zeta");
  vi.stubEnv("POD_NAME", "test-pod-xyz");
  vi.stubEnv("AGENTIC_BOOT_ID", "11111111-1111-4111-8111-111111111111");
  vi.stubEnv("WORKSPACE_ID", "");
});

afterEach(() => {
  process.stdout.write = realStdoutWrite;
  process.stderr.write = realStderrWrite;
  delete (process.stdout as unknown as Record<symbol, unknown>)[
    ORIGINAL_WRITE_SLOT
  ];
  delete (process.stderr as unknown as Record<symbol, unknown>)[
    ORIGINAL_WRITE_SLOT
  ];
  // Flush pending appenders and clear log4js's internal config so the next
  // test starts fresh. Without this, the previous test's appender (which
  // still holds a closure ref to the previous captureWrite) could fire if
  // any code emits between tests.
  log4js.shutdown(() => {});
  detachRunnerCrashListeners();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------
// structured-logging envelope shape + context propagation
// ---------------------------------------------------------------------------

test("formatter emits design-doc contract for every field", () => {
  const fx = setupCapture();
  fx.install();

  lg("runner_test").info("hello world");

  const msgs = fx.records.filter((r) => r.message === "hello world");
  expect(msgs).toHaveLength(1);

  const rec = msgs[0];
  for (const key of [
    "timestamp",
    "level",
    "logger",
    "message",
    "service",
    "tenantId",
    "projectId",
    "workspaceId",
    "executionId",
    "sessionId",
    "bootId",
    "podName",
    "source",
    "fields",
  ] as const) {
    expect(rec, `missing ${key}`).toHaveProperty(key);
  }

  expect(rec.level).toBe("INFO");
  expect(rec.logger).toBe("runner_test");
  expect(rec.service).toBe("agent-execution-runtime");
  expect(rec.tenantId).toBe("tenant-acme");
  expect(rec.projectId).toBe("project-zeta");
  expect(rec.bootId).toBe("11111111-1111-4111-8111-111111111111");
  expect(rec.podName).toBe("test-pod-xyz");
  expect(rec.source).toBe("node-logging");
  expect(rec).not.toHaveProperty("traceId");
  expect(rec.fields["component"]).toBe("aer");
  // filename/lineno/funcName populated for logger-sourced records — the
  // parity-pass fix that landed today (mirrors structured_logging.py:125-128).
  expect(rec.fields).toHaveProperty("filename");
  expect(typeof rec.fields["lineno"]).toBe("number");
});

test("context vars stay isolated across concurrent logging and captured output", async () => {
  const fx = setupCapture();
  fx.install();
  const ids = [
    "0123456789abcdef0123456789abcdef",
    "abcdef0123456789abcdef0123456789",
  ];
  await Promise.all(
    ids.map((traceId) =>
      runWithExecutionContext(
        {
          executionId: `exec-${traceId}`,
          traceId,
          requestId: "legacy-request",
          sessionId: "sess-test-002",
          workspaceId: "ws-test-003",
          wrapper: null,
          oeUrl: "http://oe",
        },
        async () => {
          await Promise.resolve();
          lg("trace_test").info("logged");
          process.stdout.write("printed\n");
          process.stderr.write("stderr\n");
        },
      ),
    ),
  );
  for (const id of ids) {
    const records = fx.records.filter((r) => r.executionId === `exec-${id}`);
    expect(records).toHaveLength(3);
    expect(new Set(records.map((r) => r.source))).toEqual(
      new Set(["node-logging", "stdout", "stderr"]),
    );
    for (const rec of records) {
      expect(rec).toMatchObject({
        traceId: id,
        sessionId: "sess-test-002",
        workspaceId: "ws-test-003",
      });
    }
  }
  lg("trace_test").info("background");
  expect(fx.records.at(-1)).not.toHaveProperty("traceId");
});

test.each([
  undefined,
  null,
  "",
  "A".repeat(32),
  "0".repeat(32),
  "abc",
  "g".repeat(32),
])("missing or invalid platform trace %s is omitted", (traceId) => {
  const fx = setupCapture();
  fx.install();
  runWithExecutionContext(
    {
      executionId: "e",
      wrapper: null,
      oeUrl: "http://oe",
      traceId,
      requestId: "0123456789abcdef0123456789abcdef",
    },
    () => lg("trace_test").info("logged"),
  );
  expect(fx.records.at(-1)?.message).toBe("logged");
  expect(fx.records.at(-1)).not.toHaveProperty("traceId");
});

// ---------------------------------------------------------------------------
// stdout/stderr capture
// ---------------------------------------------------------------------------

test("console.log captured as source=stdout, level=INFO", () => {
  const fx = setupCapture();
  fx.install();

  // After install, process.stdout.write is LoggingStream.write — a direct
  // call through it exercises the capture path the same way console.log does.
  process.stdout.write("printed line\n");

  const rec = fx.records.find((r) => r.message === "printed line");
  expect(rec).toBeDefined();
  expect(rec?.source).toBe("stdout");
  // Captured records carry only `component` in fields — no filename/lineno
  // because the call site of a captured line is LoggingStream.emit, which
  // is noise; the layout guards on `source === DEFAULT_RECORD_SOURCE`.
  expect(rec?.fields).toEqual({ component: "aer" });
});

test("stderr writes captured as source=stderr, level=WARNING-equivalent", () => {
  const fx = setupCapture();
  fx.install();

  process.stderr.write("stderr line\n");

  const rec = fx.records.find((r) => r.message === "stderr line");
  expect(rec).toBeDefined();
  expect(rec?.source).toBe("stderr");
  // log4js spells the level "WARN"; the layout normalizes it to the Python
  // stdlib name "WARNING" so TS and Python agents ship identical level
  // strings for the Splunk dashboards / playground UI logs tab.
  expect(rec?.level).toBe("WARNING");
});

// ---------------------------------------------------------------------------
// Exception capture
// ---------------------------------------------------------------------------

test("Error-as-arg populates fields.exc_type/exc_message/exc_traceback", () => {
  const fx = setupCapture();
  fx.install();

  const err = new Error("kaboom");
  // Python uses `logger.exception()` which pulls sys.exc_info; TS doesn't
  // have that — the convention is `logger.error(err, "message")` or
  // `logger.error({ err }, "message")`. parseEventData (TS 163-192) handles
  // both shapes.
  lg("runner_test").error(err, "caught");

  const rec = fx.records.find((r) => r.message === "caught");
  expect(rec).toBeDefined();
  expect(rec?.level).toBe("ERROR");
  expect(rec?.fields["exc_type"]).toBe("Error");
  expect(rec?.fields["exc_message"]).toBe("kaboom");
  expect(rec?.fields["exc_traceback"]).toContain("kaboom");
});

// ---------------------------------------------------------------------------
// uncaughtException / unhandledRejection handling
// ---------------------------------------------------------------------------

describe("uncaughtException/unhandledRejection handling", () => {
  let drainBaseline: ((...args: unknown[]) => void)[];

  beforeEach(() => {
    // handleFatal() schedules a real (unref'd) setTimeout fallback exit. Fake
    // timers keep that from firing seconds later, mid-way through an unrelated
    // later test.
    vi.useFakeTimers();
    drainBaseline = process.stdout.listeners("drain") as ((
      ...args: unknown[]
    ) => void)[];
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    // The sole-listener test leaves a once("drain") that would call the real
    // process.exit once restoreAllMocks() un-stubs it.
    process.stdout.removeAllListeners("drain");
    for (const l of drainBaseline) {
      process.stdout.on("drain", l as never);
    }
    // handleFatal() sets process.exitCode = 1 as a belt-and-suspenders
    // guard against a natural drain silently reporting success — reset it
    // so it doesn't leak into the overall test-run's real exit code.
    process.exitCode = 0;
  });

  test("uncaught exception produces one ERROR record with exc_* fields and leaves termination to Node", () => {
    const exitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
    const fx = setupCapture();
    fx.install();

    process.emit("uncaughtExceptionMonitor", new Error("kaboom"));

    const errorRecs = fx.records.filter(
      (r) => r.level === "ERROR" && r.fields["exc_message"] === "kaboom",
    );
    expect(errorRecs).toHaveLength(1);
    const errorRec = errorRecs[0];
    expect(errorRec?.fields["exc_type"]).toBe("Error");
    expect(errorRec?.message).toBe("Uncaught exception: Error: kaboom");

    // A monitor observes only. Touching exitCode here would report failure
    // for an agent whose own handler recovered.
    expect(exitSpy).not.toHaveBeenCalled();
    expect(process.exitCode).not.toBe(1);
  });

  test("unhandledRejection with an Error reason produces one ERROR record", () => {
    vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const fx = setupCapture();
    fx.install();

    process.emit(
      "unhandledRejection",
      new Error("rejected"),
      Promise.resolve(),
    );

    const recs = fx.records.filter(
      (r) => r.fields["exc_message"] === "rejected",
    );
    expect(recs).toHaveLength(1);
    const rec = recs[0];
    expect(rec?.level).toBe("ERROR");
    expect(rec?.fields["exc_type"]).toBe("Error");
  });

  test("unhandledRejection with a non-Error reason preserves the original value instead of always reporting exc_type=Error", () => {
    vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const fx = setupCapture();
    fx.install();

    process.emit("unhandledRejection", "timeout", Promise.resolve());

    const recs = fx.records.filter(
      (r) => r.level === "ERROR" && r.fields["exc_message"] === "timeout",
    );
    expect(recs).toHaveLength(1);
    const rec = recs[0];
    // Not the generic "Error" default — a plain string reason must stay
    // distinguishable from a genuine Error object.
    expect(rec?.fields["exc_type"]).toBe("NonErrorRejection");
  });

  // Both go through toError(), which tags any non-Error value
  // "NonErrorRejection" so it stays distinguishable from a real Error.
  test.each([
    ["null", null],
    ["a string", "just-a-string"],
  ])(
    "uncaught exception carrying %s still produces an ERROR record",
    (_label, thrown) => {
      vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
      const fx = setupCapture();
      fx.install();

      // JS permits throwing any value. Without normalization the handler reads
      // `.name`/`.stack` off it and dies before logging anything.
      expect(() =>
        process.emit("uncaughtExceptionMonitor", thrown as unknown as Error),
      ).not.toThrow();

      const errorRecs = fx.records.filter((r) => r.level === "ERROR");
      expect(errorRecs).toHaveLength(1);
      const errorRec = errorRecs[0];
      expect(errorRec?.fields["exc_type"]).toBe("NonErrorRejection");
      expect(errorRec?.fields["exc_message"]).toBe(String(thrown));
      expect(errorRec?.message).toContain("Uncaught exception:");
    },
  );

  test("never force-exits a rejection another listener is handling, not even after the fallback timeout", () => {
    const exitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
    const fx = setupCapture();
    // An agent's own listener that deliberately swallows and keeps serving.
    process.on("unhandledRejection", () => {});
    fx.install();

    process.emit(
      "unhandledRejection",
      new Error("shared-crash"),
      Promise.resolve(),
    );

    expect(exitSpy).not.toHaveBeenCalled();
    // exitCode must stay clean too, or a later graceful shutdown reports
    // failure for an agent that recovered.
    expect(process.exitCode).not.toBe(1);

    vi.advanceTimersByTime(FALLBACK_EXIT_MS * 2);
    expect(exitSpy).not.toHaveBeenCalled();
  });

  test("still exits a sole-listener rejection via the fallback timeout when stdout never drains", () => {
    const exitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation(() => undefined as never);
    const baseline = process.listeners("unhandledRejection");
    process.removeAllListeners("unhandledRejection");
    try {
      const fx = setupCapture();
      fx.install();
      vi.spyOn(process.stdout, "writableLength", "get").mockReturnValue(1);

      process.emit(
        "unhandledRejection",
        new Error("stuck-drain"),
        Promise.resolve(),
      );

      expect(exitSpy).not.toHaveBeenCalled();
      vi.advanceTimersByTime(FALLBACK_EXIT_MS);
      expect(exitSpy).toHaveBeenCalledWith(1);
    } finally {
      for (const l of baseline) {
        process.on("unhandledRejection", l as never);
      }
    }
  });

  test("repeat installStructuredLogging does not stack crash listeners", () => {
    vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const fx = setupCapture();
    const counts = () => CRASH_EVENTS.map((e) => process.listenerCount(e));
    // Absolute against the framework baseline, not a delta: a delta stays
    // green when a listener has already leaked in from an earlier test.
    const expected = CRASH_EVENTS.map(
      (e) => (listenersBeforeTest.get(e) ?? []).length + 1,
    );

    fx.install();
    expect(counts()).toEqual(expected);
    fx.install();
    expect(counts()).toEqual(expected);
  });

  test("a rejection is still logged when the agent has its own rejection listener", () => {
    vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const fx = setupCapture();
    // The reason the rejection path keeps a real listener instead of relying on
    // the monitor: this listener suppresses Node's default, so a monitor would
    // never see the rejection and the crash would go unlogged entirely.
    process.on("unhandledRejection", () => {});
    fx.install();

    process.emit(
      "unhandledRejection",
      new Error("agent-handled"),
      Promise.resolve(),
    );

    const recs = fx.records.filter(
      (r) => r.level === "ERROR" && r.fields["exc_message"] === "agent-handled",
    );
    expect(recs).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Real crash in a subprocess: termination is Node's job now, so assert it
// ---------------------------------------------------------------------------

describe("uncaught exception in a real process", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "uncaught-test-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  test("exits non-zero, prints the stack, and emits exactly one ERROR record", () => {
    const entry = join(dir, "crash.mjs");
    // the built output — a spawned plain `node` cannot import TypeScript
    const barrel = new URL("../../dist/structured_logging.js", import.meta.url)
      .href;
    writeFileSync(
      entry,
      [
        `const { installStructuredLogging } = await import(${JSON.stringify(barrel)});`,
        `installStructuredLogging();`,
        `setTimeout(() => { throw new Error("subprocess-crash"); }, 0);`,
      ].join("\n"),
    );

    const r = spawnSync(process.execPath, [entry], {
      env: { ...process.env, RUNNER_MODE: "aer", ORG_ID: "o", PROJECT_ID: "p" },
      encoding: "utf-8",
    });

    expect(r.status).toBe(1);
    // Node prints the stack itself now; we no longer route it through capture.
    expect(r.stderr).toContain("subprocess-crash");

    const errorRecs = (r.stdout ?? "")
      .split("\n")
      .filter((l) => l.trim().startsWith("{"))
      .map((l) => JSON.parse(l) as Record<string, unknown>)
      .filter((rec) => rec["level"] === "ERROR");
    expect(errorRecs).toHaveLength(1);
    expect(
      (errorRecs[0]?.["fields"] as Record<string, unknown>)["exc_message"],
    ).toBe("subprocess-crash");
  });
});

// ---------------------------------------------------------------------------
// Idempotency / level handling
// ---------------------------------------------------------------------------

test("install is idempotent — re-install does not recurse", () => {
  const fx = setupCapture();
  fx.install();
  // Second install — the Symbol.for slot snapshot from the first install
  // protects against wrapping the already-patched write.
  fx.install();

  lg("runner_test").info("after-reinstall");

  const rec = fx.records.find((r) => r.message === "after-reinstall");
  expect(rec).toBeDefined();
  expect(rec?.level).toBe("INFO");
});

test("install honors level arg — DEBUG flips through", () => {
  const fx = setupCapture();
  fx.install({ level: "DEBUG" });

  lg("runner_test").debug("dbg");

  expect(fx.records.some((r) => r.message === "dbg")).toBe(true);
});

test("install diagnostic is DEBUG-only", () => {
  const fx = setupCapture();
  fx.install({ level: "INFO" });

  expect(
    fx.records.some((r) => r.message === "structured logging installed"),
  ).toBe(false);

  fx.install({ level: "DEBUG" });
  const installRecord = fx.records.find(
    (r) => r.message === "structured logging installed",
  );
  expect(installRecord?.level).toBe("DEBUG");
});

test("captures bypass LOG_LEVEL filter — console.log emerges even at LOG_LEVEL=ERROR", () => {
  const fx = setupCapture();
  fx.install({ level: "ERROR" });

  // logger.info() is filtered at the default-category level filter.
  lg("runner_test").info("filtered-info");
  // But console.log routes through the stdio-capture category which is
  // pinned to TRACE regardless of LOG_LEVEL — Python parity via the
  // dedicated category instead of Python's `Logger.handle()` bypass.
  process.stdout.write("captured-print\n");
  process.stderr.write("captured-stderr\n");

  const msgs = fx.records.map((r) => r.message);
  expect(msgs).not.toContain("filtered-info");
  expect(msgs).toContain("captured-print");
  expect(msgs).toContain("captured-stderr");
});

// ---------------------------------------------------------------------------
// Truncation contracts (need internal exports)
// ---------------------------------------------------------------------------

test("traceback truncation honors UTF-8 byte budget, not char count", () => {
  const fx = setupCapture();
  fx.install();

  // Each "字" is 3 bytes in UTF-8 — char-count truncation would let the
  // encoded payload reach ~3x the budget. Build a string that will produce
  // a stack longer than MAX_TRACEBACK_BYTES once formatted.
  const bigMsg = "字".repeat(4000);
  const err = new Error(bigMsg);
  // err.stack starts with "Error: <message>\n    at ..." so the message
  // bytes alone (4000 * 3 = 12000) exceed MAX_TRACEBACK_BYTES (8192).
  lg("runner_test").error(err, "caught");

  const rec = fx.records.find((r) => r.message === "caught");
  expect(rec).toBeDefined();
  const tb = rec?.fields["exc_traceback"];
  expect(typeof tb).toBe("string");
  expect(tb as string).toMatch(/\.\.\.\(truncated\)$/);
  const body = (tb as string).replace(/\.\.\.\(truncated\)$/, "");
  expect(Buffer.byteLength(body, "utf-8")).toBeLessThanOrEqual(
    MAX_TRACEBACK_BYTES,
  );
});

test("LoggingStream emits exactly one record per newline-terminated line", () => {
  // Direct unit test — bypasses installStructuredLogging entirely. Mocks
  // the Logger interface (LoggingStream only uses .info / .warn).
  const captured: string[] = [];
  const mockLogger = {
    info(_extra: object, message: string) {
      captured.push(message);
    },
    warn(_extra: object, message: string) {
      captured.push(message);
    },
  } as unknown as Logger;

  const stream = new LoggingStream(mockLogger, "info", "stdout");
  stream.write("partial ");
  expect(captured).toEqual([]);
  stream.write("line\nsecond line\n");
  expect(captured).toEqual(["partial line", "second line"]);
});

test("LoggingStream.flush emits a trailing line that never got a newline", () => {
  // Parity with Python's LoggingStream.flush() (invoked at interpreter
  // shutdown): a final write without a `\n` must not be lost. The install
  // path wires flush() to `process.on("exit")`.
  const captured: string[] = [];
  const mockLogger = {
    info(_extra: object, message: string) {
      captured.push(message);
    },
    warn(_extra: object, message: string) {
      captured.push(message);
    },
  } as unknown as Logger;

  const stream = new LoggingStream(mockLogger, "info", "stdout");
  stream.write("tail without newline");
  expect(captured).toEqual([]); // buffered, no newline yet
  stream.flush();
  expect(captured).toEqual(["tail without newline"]);

  // A second flush is a no-op (buffer already drained) and a whitespace-only
  // tail is skipped — same rule as the per-line filter in write().
  stream.flush();
  stream.write("   ");
  stream.flush();
  expect(captured).toEqual(["tail without newline"]);
});

test("re-install flushes the superseded stream's buffered tail", () => {
  // Regression: a re-install replaces process.stdout.write with a fresh
  // LoggingStream. A partial (newline-less) line buffered in the previous
  // stream must be flushed as it is superseded — otherwise it is lost (the new
  // stream doesn't carry it and the old stream is dropped).
  const fx = setupCapture();
  fx.install(); // install #1

  process.stdout.write("tail-before-reinstall"); // no newline → buffered in #1
  expect(
    fx.records.find((r) => r.message === "tail-before-reinstall"),
  ).toBeUndefined();

  fx.install(); // install #2 must flush #1's buffered tail first

  const rec = fx.records.find((r) => r.message === "tail-before-reinstall");
  expect(rec).toBeDefined();
  expect(rec?.source).toBe("stdout");
});

test("exit-flush handler flushes the currently-installed capture", () => {
  // The one-shot exit handler must flush whatever streams are current in the
  // shared global slot. Capture the handler off `process.on` rather than
  // firing the real "exit" event (which would invoke every listener in the
  // Vitest worker). Reset the once-per-process guard so this install actually
  // (re-)registers, letting the spy observe it.
  const g = globalThis as unknown as Record<symbol, unknown>;
  delete g[
    Symbol.for("agent_engine_runner_shared.structured_logging_exit_flush")
  ];
  const onSpy = vi.spyOn(process, "on");

  const fx = setupCapture();
  fx.install();

  const exitCall = onSpy.mock.calls.find(([event]) => event === "exit");
  expect(exitCall).toBeDefined();
  const exitHandler = exitCall?.[1] as (() => void) | undefined;

  process.stdout.write("tail-at-exit"); // no newline → buffered in current
  expect(fx.records.find((r) => r.message === "tail-at-exit")).toBeUndefined();

  exitHandler?.();

  const rec = fx.records.find((r) => r.message === "tail-at-exit");
  expect(rec).toBeDefined();
  expect(rec?.source).toBe("stdout");
});

test("LoggingStream caps unbounded buffer at MAX_BUFFER_BYTES", () => {
  const captured: string[] = [];
  const mockLogger = {
    info(_extra: object, message: string) {
      captured.push(message);
    },
    warn(_extra: object, message: string) {
      captured.push(message);
    },
  } as unknown as Logger;

  const stream = new LoggingStream(mockLogger, "info", "stdout");
  // No `\n` — must force-emit a truncated record once the buffer crosses
  // MAX_BUFFER_BYTES and reset internal state.
  stream.write("x".repeat(MAX_BUFFER_BYTES + 4096));

  expect(captured).toHaveLength(1);
  expect(captured[0]).toMatch(/\.\.\.\(truncated\)$/);
  expect(Buffer.byteLength(captured[0], "utf-8")).toBeLessThanOrEqual(
    MAX_BUFFER_BYTES + "...(truncated)".length,
  );
});

// ---------------------------------------------------------------------------
// Stream identity preserved — TS analog of Python's fileno() delegation test
// ---------------------------------------------------------------------------

test("process.stdout/stderr identity preserved after install (subprocess piping regression guard)", () => {
  // Python's `test_fileno_delegates_to_wrapped_stream` verifies that
  // `LoggingStream.fileno()` delegates to the wrapped stream so
  // `subprocess.run(stdout=sys.stdout)` keeps working. TS achieves the
  // same property via a *different mechanism*: we never replace
  // process.stdout/stderr — only patch their `.write` method — so all
  // fd/isTTY/columns introspection still hits the real stream by virtue of
  // identity preservation. Locking this in at the test level prevents a
  // future refactor from swapping the stream objects.
  const beforeStdout = process.stdout;
  const beforeStderr = process.stderr;

  const fx = setupCapture();
  fx.install();

  expect(process.stdout).toBe(beforeStdout);
  expect(process.stderr).toBe(beforeStderr);
});

// ---------------------------------------------------------------------------
// setupLogging integration — env-gated structured vs legacy path
// ---------------------------------------------------------------------------

test("setupLogging activates structured layout when STRUCTURED_LOGGING=true", () => {
  vi.stubEnv("STRUCTURED_LOGGING", "true");
  const fx = setupCapture();
  // setupLogging delegates to installStructuredLogging when env=true.
  setupLogging({ appName: "test", mode: "aer" });

  // patchStdio ran → process.stdout.write is now LoggingStream.write,
  // not our captureWrite. captureWrite is still alive via the appender's
  // closure, so it still receives JSON.
  expect(process.stdout.write).not.toBe(fx.captureWrite);
  lg("runner_test").info("structured-active");
  const record = fx.records.find((r) => r.message === "structured-active");
  expect(record?.service).toBe("agent-execution-runtime");
});

test("setupLogging uses legacy formatter when STRUCTURED_LOGGING is unset", () => {
  vi.stubEnv("STRUCTURED_LOGGING", ""); // empty string → not "true"
  const fx = setupCapture();
  setupLogging({ appName: "test", mode: "aer" });

  // patchStdio did NOT run — process.stdout.write is still our captureWrite.
  expect(process.stdout.write).toBe(fx.captureWrite);
  // No structured JSON records — the human pattern layout writes plain text
  // that fails JSON.parse and lives only in rawLines.
  expect(fx.records).toHaveLength(0);
  expect(fx.rawLines.some((l) => l.includes("Logging initialized"))).toBe(true);
});

// ---------------------------------------------------------------------------
// Env-var resolution edge cases
// ---------------------------------------------------------------------------

test("projectId is null when PROJECT_ID env unset", () => {
  vi.stubEnv("PROJECT_ID", ""); // envOrNull treats "" as null
  const fx = setupCapture();
  fx.install();

  lg("runner_test").info("hello");

  const rec = fx.records.find((r) => r.message === "hello");
  expect(rec).toBeDefined();
  expect(rec?.projectId).toBeNull();
});

test("mode arg overrides RUNNER_MODE env at install time", () => {
  vi.stubEnv("RUNNER_MODE", "orchestrator");
  const fx = setupCapture();
  // Explicit mode wins — even when env disagrees.
  fx.install({ mode: "tool" });

  lg("runner_test").info("x");

  const rec = fx.records.find((r) => r.message === "x");
  expect(rec).toBeDefined();
  expect(rec?.service).toBe("tool-executor");
  expect(rec?.fields["component"]).toBe("tool");
});

test("setupLogging threads mode through to structured install when RUNNER_MODE env is unset", () => {
  vi.stubEnv("STRUCTURED_LOGGING", "true");
  vi.stubEnv("RUNNER_MODE", "");
  const fx = setupCapture();
  setupLogging({ appName: "test", mode: "aer" });

  lg("runner_test").info("mode-check");
  const record = fx.records.find((r) => r.message === "mode-check");
  expect(record?.service).toBe("agent-execution-runtime");
});

// ---------------------------------------------------------------------------
// setupLogging + AGENTIC_DEV_MODES — dev-up file logging gate
//
// Mirrors the four Python `dev_modes` tests in test_structured_logging.py:
// file logging is a dev-up-only convenience (gated on AGENTIC_DEV_MODES,
// which only the CLI compose templates set), never active in production.
// ---------------------------------------------------------------------------

test("setupLogging writes human log file when AGENTIC_DEV_MODES is set", async () => {
  vi.stubEnv("STRUCTURED_LOGGING", ""); // legacy human path
  vi.stubEnv("AGENTIC_DEV_MODES", "aer,tool");
  const dir = mkdtempSync(join(tmpdir(), "runner-log-"));
  vi.stubEnv("LOG_DIR", dir);
  const fx = setupCapture();
  try {
    setupLogging({ appName: "test", mode: "aer" });
    log4js.getLogger("runner_test").info("dev file handler hello");
    await new Promise<void>((resolve) => log4js.shutdown(() => resolve())); // flush the file appender (shutdown returns void — only the callback form waits)

    const content = readFileSync(join(dir, "test-aer.log"), "utf8");
    expect(content).toContain("dev file handler hello");
    // Human text on disk — not the structured JSON envelope.
    expect(content).not.toContain('"message"');
    // Console output still flows — the same line reached stdout.
    expect(fx.rawLines.some((l) => l.includes("dev file handler hello"))).toBe(
      true,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("setupLogging structured + AGENTIC_DEV_MODES writes file alongside JSON stdout", async () => {
  vi.stubEnv("STRUCTURED_LOGGING", "true");
  vi.stubEnv("AGENTIC_DEV_MODES", "aer,tool");
  const dir = mkdtempSync(join(tmpdir(), "runner-log-"));
  vi.stubEnv("LOG_DIR", dir);
  const fx = setupCapture();
  try {
    setupLogging({ appName: "test", mode: "aer" });
    log4js.getLogger("runner_test").info("structured dev hello");
    await new Promise<void>((resolve) => log4js.shutdown(() => resolve())); // flush the file appender (shutdown returns void — only the callback form waits)

    // File gets the human-readable pattern, not JSON.
    const content = readFileSync(join(dir, "test-aer.log"), "utf8");
    expect(content).toContain("structured dev hello");
    expect(content).not.toContain('"message"');
    // Structured JSON still flows to stdout untouched.
    expect(fx.records.some((r) => r.message === "structured dev hello")).toBe(
      true,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("setupLogging writes no log file when AGENTIC_DEV_MODES is unset", async () => {
  vi.stubEnv("STRUCTURED_LOGGING", ""); // empty string → falsy gate
  vi.stubEnv("AGENTIC_DEV_MODES", "");
  const dir = mkdtempSync(join(tmpdir(), "runner-log-"));
  vi.stubEnv("LOG_DIR", dir);
  try {
    setupLogging({ appName: "test", mode: "aer" });
    log4js.getLogger("runner_test").info("no dev file");
    await new Promise<void>((resolve) => log4js.shutdown(() => resolve()));
    expect(existsSync(join(dir, "test-aer.log"))).toBe(false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("setupLogging structured without AGENTIC_DEV_MODES writes no file", async () => {
  vi.stubEnv("STRUCTURED_LOGGING", "true");
  vi.stubEnv("AGENTIC_DEV_MODES", "");
  const dir = mkdtempSync(join(tmpdir(), "runner-log-"));
  vi.stubEnv("LOG_DIR", dir);
  const fx = setupCapture();
  try {
    setupLogging({ appName: "test", mode: "aer" });
    log4js.getLogger("runner_test").info("no structured dev file");
    await new Promise<void>((resolve) => log4js.shutdown(() => resolve()));
    expect(existsSync(join(dir, "test-aer.log"))).toBe(false);
    // Structured stdout path still works.
    expect(fx.records.some((r) => r.message === "no structured dev file")).toBe(
      true,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("setupLogging dev mode with unusable LOG_DIR continues console-only", async () => {
  // Parity with Python's unwritable-log_dir test: an unusable LOG_DIR
  // (parent path is a regular file) must disable file logging without
  // crashing — file logging is a dev convenience, never a correctness
  // requirement.
  vi.stubEnv("STRUCTURED_LOGGING", "");
  vi.stubEnv("AGENTIC_DEV_MODES", "aer,tool");
  const blocker = join(tmpdir(), "runner-log-blocker");
  writeFileSync(blocker, "not a dir");
  vi.stubEnv("LOG_DIR", join(blocker, "logs"));
  const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  try {
    const root = setupLogging({ appName: "test", mode: "aer" });
    root.info("still logging to console");
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0]?.[0])).toContain(
      "File logging disabled",
    );
  } finally {
    warnSpy.mockRestore();
    rmSync(blocker, { force: true });
  }
});

test("setupLogging keeps the dev log file inside LOG_DIR when appName has separators", async () => {
  // appName is caller-supplied; without sanitizing, a value containing a
  // path separator could redirect the write outside LOG_DIR.
  vi.stubEnv("STRUCTURED_LOGGING", "");
  vi.stubEnv("AGENTIC_DEV_MODES", "aer,tool");
  const dir = mkdtempSync(join(tmpdir(), "runner-log-"));
  vi.stubEnv("LOG_DIR", dir);
  setupCapture();
  try {
    setupLogging({ appName: "../escape", mode: "aer" });
    log4js.getLogger("runner_test").info("sanitized hello");
    await new Promise<void>((resolve) => log4js.shutdown(() => resolve()));

    // Separators are replaced, so the file lands inside LOG_DIR...
    const inside = join(dir, ".._escape-aer.log");
    expect(existsSync(inside)).toBe(true);
    expect(readFileSync(inside, "utf8")).toContain("sanitized hello");
    // ...and nothing is written to the parent directory.
    expect(existsSync(join(dir, "..", "escape-aer.log"))).toBe(false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// RUNNER_MODE → service/component mapping table
// ---------------------------------------------------------------------------

describe("RUNNER_MODE maps to service identifier and component", () => {
  test.each([
    ["orchestrator", "orchestration-engine"],
    ["aer", "agent-execution-runtime"],
    ["tool", "tool-executor"],
    ["tool_function", "tool-executor"],
    ["memory-server", "memory-server"],
  ])("mode=%s → service=%s", (mode, expectedService) => {
    vi.stubEnv("RUNNER_MODE", mode);
    const fx = setupCapture();
    fx.install({ mode });

    lg("runner_test").info("hi");

    const rec = fx.records.find((r) => r.message === "hi");
    expect(rec).toBeDefined();
    expect(rec?.service).toBe(expectedService);
    expect(rec?.fields["component"]).toBe(mode);
  });
});

// ---------------------------------------------------------------------------
// logger.ts ↔ structured_logging.ts state sync
// ---------------------------------------------------------------------------

describe("getLogger after installStructuredLogging keeps structured-logging config", () => {
  test("wrapper getLogger emits JSON and does not tear down structured config", () => {
    const fx = setupCapture();
    fx.install();

    // The logger.ts `getLogger` wrapper (not log4js.getLogger): before the
    // markConfigured() sync this could re-run ensureDefaultConfig() and
    // replace the just-installed structured layout with the human stdout
    // appender, dropping structured output (and risking recursion through
    // patched stdio). It must now be a no-op on configuration.
    getLogger("wrapper_test").info("via wrapper");

    const rec = fx.records.find((r) => r.message === "via wrapper");
    expect(rec).toBeDefined();
    // Proves the structured layout is still active — a human-pattern reconfigure
    // would have produced a non-JSON line (no parsed record) instead.
    expect(rec?.logger).toBe("wrapper_test");
    expect(rec?.service).toBe("agent-execution-runtime");
  });
});

// ---------------------------------------------------------------------------
// Safe serialization — cycles, DAGs, BigInt
// ---------------------------------------------------------------------------

describe("structured record serialization is crash-safe", () => {
  test("true reference cycle in extra renders as [Circular], not a crash", () => {
    const fx = setupCapture();
    fx.install();

    const cyclic: Record<string, unknown> = { name: "node" };
    cyclic["self"] = cyclic;
    lg("runner_test").info({ payload: cyclic }, "cycle");

    const rec = fx.records.find((r) => r.message === "cycle");
    expect(rec).toBeDefined();
    const payload = rec?.fields["payload"] as Record<string, unknown>;
    expect(payload["name"]).toBe("node");
    expect(payload["self"]).toBe("[Circular]");
  });

  test("DAG (object referenced from two siblings) is rendered in full", () => {
    const fx = setupCapture();
    fx.install();

    // shared is NOT a cycle — a flat seen-set would falsely flag the second
    // occurrence as [Circular]. Ancestor-path tracking renders both.
    const shared = { id: 7 };
    lg("runner_test").info({ a: shared, b: shared }, "dag");

    const rec = fx.records.find((r) => r.message === "dag");
    expect(rec).toBeDefined();
    expect(rec?.fields["a"]).toEqual({ id: 7 });
    expect(rec?.fields["b"]).toEqual({ id: 7 });
  });

  test("BigInt in a trailing message arg does not throw and is serialized", () => {
    const fx = setupCapture();
    fx.install();

    // Non-leading arg → flows through parseEventData's serializer, which must
    // not throw on a BigInt (plain JSON.stringify would).
    lg("runner_test").info("id is", 9007199254740993n);

    const rec = fx.records.find((r) => r.message.startsWith("id is"));
    expect(rec).toBeDefined();
    expect(rec?.message).toContain("9007199254740993");
  });
});

test("origin is absent outside customer scope", () => {
  const fx = setupCapture();
  fx.install();
  lg("runner_test").info("platform line");
  const rec = fx.records.find((r) => r.message === "platform line");
  expect(rec).toBeDefined();
  expect(rec).not.toHaveProperty("origin");
});

test("origin is customer inside scope", () => {
  const fx = setupCapture();
  fx.install();
  runWithCustomerOrigin(() => {
    lg("runner_test").info("customer line");
    process.stdout.write("printed customer\n");
  });
  lg("runner_test").info("after scope");
  const customer = fx.records.find((r) => r.message === "customer line");
  const printed = fx.records.find((r) => r.message === "printed customer");
  const after = fx.records.find((r) => r.message === "after scope");
  expect(customer?.origin).toBe("customer");
  expect(printed?.origin).toBe("customer");
  expect(after).not.toHaveProperty("origin");
});
