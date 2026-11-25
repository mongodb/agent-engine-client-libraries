/**
 * Capture of `process.stdout` / `process.stderr` writes so stray
 * `console.log` / direct writes emerge as structured-logging records tagged
 * `source: "stdout"` / `"stderr"`.
 */

import log4js, { type Logger } from "log4js";

import { getLogOutputScope } from "../log_correlation.js";
import {
  ACTIVE_CAPTURES,
  EXIT_FLUSH_REGISTERED,
  MAX_BUFFER_BYTES,
  MAX_PENDING_BYTES,
  MAX_PENDING_WRITERS,
  ORIGINAL_WRITE_SLOT,
  RECORD_ATTRIBUTION,
  STDIO_CAPTURE_CATEGORY,
  TRUNCATION_MARKER,
} from "./constants.js";
import {
  currentLogAttribution,
  getLogExecutionId,
  type LogAttribution,
} from "./layout.js";
import { truncateUtf8 } from "./serialize.js";

function withinJsonDepth(value: unknown, depth = 0): boolean {
  if (!value || typeof value !== "object") return true;
  // Keep hostile payloads from exhausting the stack, with the same bound in Python.
  if (depth >= 64) return false;
  return Object.values(value).every((child: unknown) =>
    withinJsonDepth(child, depth + 1),
  );
}

type WriteFn = NodeJS.WriteStream["write"];
type PatchableStream = NodeJS.WriteStream & { [ORIGINAL_WRITE_SLOT]?: WriteFn };

/**
 * Snapshot the original `write` on the first install and stash it on the
 * stream under `ORIGINAL_WRITE_SLOT`. Subsequent calls return the same
 * snapshot — matches the Python `_unwrap_logging_stream` recursion guard:
 * the appender always writes via the snapshot, never via the patched
 * `write` we've installed on top.
 */
export function snapshotOriginalWrite(stream: PatchableStream): WriteFn {
  const existing = stream[ORIGINAL_WRITE_SLOT];
  if (existing) return existing;
  const original = stream.write.bind(stream) as WriteFn;
  stream[ORIGINAL_WRITE_SLOT] = original;
  return original;
}

interface PendingLine {
  key: string;
  scope: object | null;
  attribution: LogAttribution;
  text: string;
  size: number;
}

// Map keys must be primitives to compose, so each scope object gets a number.
const scopeIds = new WeakMap<object, number>();
let nextScopeId = 1;

function writerKey(scope: object | null, executionId: string | null): string {
  let scopeId = 0;
  if (scope !== null) {
    scopeId = scopeIds.get(scope) ?? nextScopeId++;
    scopeIds.set(scope, scopeId);
  }
  return JSON.stringify([scopeId, executionId]);
}

/**
 * Replacement for `process.stdout.write` / `process.stderr.write` that
 * routes writes through a logger.
 *
 * Bare `console.log` / `process.stdout.write` calls (and any other code
 * writing directly to the captured stream) are rewritten as log records
 * tagged with `source: "stdout"` (or `"stderr"`) so the layout can mark
 * them as such on the wire. Lines are buffered until newline, one pending
 * line per writer (workload scope + execution) so concurrent executions'
 * partial writes neither merge nor take each other's IDs. Each pending line
 * carries the attribution its writer had when the line started. Hard byte
 * and writer caps stop a producer writing without `\n` from growing the
 * buffers until OOM (see `MAX_BUFFER_BYTES`, `MAX_PENDING_BYTES` and
 * `MAX_PENDING_WRITERS`).
 *
 * Unlike the Python `LoggingStream` we do not need an `RLock` — Node's
 * event loop is single-threaded, so one `write()` call can't interleave
 * bytes with another. We also don't reproduce Python's
 * `fileno()`/`buffer`/`encoding` delegation: we only patch the `write`
 * method, leaving `process.stdout`'s fd / TTY introspection intact for
 * subprocess piping, progress-bar libs, and grpc-internal logging.
 */
/** @internal — exported only for tests; not part of the public API. */
export class LoggingStream {
  // Re-inserted on every write, so budget eviction drops the writer that
  // wrote least recently.
  private readonly pending = new Map<string, PendingLine>();
  private pendingBytes = 0;

  constructor(
    private readonly logger: Logger,
    private readonly level: "info" | "warn",
    private readonly source: "stdout" | "stderr",
  ) {}

  write(chunk: unknown, encodingOrCb?: unknown, maybeCb?: unknown): boolean {
    const text =
      typeof chunk === "string"
        ? chunk
        : Buffer.isBuffer(chunk)
          ? chunk.toString(
              typeof encodingOrCb === "string"
                ? (encodingOrCb as BufferEncoding)
                : "utf-8",
            )
          : String(chunk ?? "");

    if (text) {
      const scope = getLogOutputScope();
      const key = writerKey(scope, getLogExecutionId());
      const pending = this.pending.get(key);
      this.pending.delete(key);
      this.pendingBytes -= pending?.size ?? 0;
      // Settle the pending state before emitting: an appender may flush this
      // stream while a line is emitted.
      const emits: Array<[string, LogAttribution]> = [];
      const lines = ((pending?.text ?? "") + text).split("\n");
      let tail = lines.pop() ?? "";
      let attribution = pending?.attribution ?? currentLogAttribution();
      for (const line of lines) {
        // Whitespace-only lines are uninteresting — emitting a JSON
        // record with `message: "   "` adds noise and makes downstream
        // filters work harder. Mirrors the `if line.strip()` filter in
        // the Python LoggingStream; do not "fix" by removing it.
        if (line.trim()) emits.push([line, attribution]);
        // Only the line that was pending began under the stored attribution.
        attribution = currentLogAttribution();
      }
      // Force-emit a truncated record if a producer is writing without
      // newlines — mirrors the exc_traceback truncation pattern. Slice
      // by UTF-8 byte count via `truncateUtf8` so the cap reflects what
      // Fluent Bit ships AND a partial multi-byte sequence at the cut
      // point is dropped rather than turned into U+FFFD replacement bytes
      // that would push the encoded payload back over the budget.
      let size = Buffer.byteLength(tail, "utf-8");
      if (size > MAX_BUFFER_BYTES) {
        emits.push([truncateUtf8(tail, MAX_BUFFER_BYTES), attribution]);
        tail = "";
        size = 0;
      }
      if (tail) {
        this.pending.set(key, { key, scope, attribution, text: tail, size });
        this.pendingBytes += size;
      }
      for (const oldest of this.pending.values()) {
        if (
          this.pendingBytes <= MAX_PENDING_BYTES &&
          this.pending.size <= MAX_PENDING_WRITERS
        ) {
          break;
        }
        this.pending.delete(oldest.key);
        this.pendingBytes -= oldest.size;
        if (oldest.text.trim()) {
          emits.push([oldest.text + TRUNCATION_MARKER, oldest.attribution]);
        }
      }
      for (const [line, lineAttribution] of emits) {
        this.emit(line, lineAttribution);
      }
    }

    // Honor `write(chunk, cb)` and `write(chunk, encoding, cb)` callback
    // signatures so callers waiting on drain semantics don't hang.
    const cb = typeof encodingOrCb === "function" ? encodingOrCb : maybeCb;
    if (typeof cb === "function") (cb as (err?: Error | null) => void)();
    return true;
  }

  /**
   * Emit every writer's buffered bytes that never got a trailing newline,
   * each under its own attribution, then clear them. Called on process exit
   * so a final `console.log`/`process.stdout.write` without a `\n` right
   * before shutdown is not lost — mirrors Python's `LoggingStream.flush()`
   * (which the interpreter invokes when it flushes `sys.stdout` at
   * shutdown). The same whitespace-only skip as `write()` applies: a blank
   * tail is not worth a record.
   */
  flush(): void {
    this.flushMatching(() => true);
  }

  flushScope(scope: object): void {
    this.flushMatching((line) => line.scope === scope);
  }

  private flushMatching(matches: (line: PendingLine) => boolean): void {
    const flushed = [...this.pending.values()].filter(matches);
    for (const line of flushed) {
      this.pending.delete(line.key);
      this.pendingBytes -= line.size;
    }
    for (const line of flushed) {
      if (line.text.trim()) this.emit(line.text, line.attribution);
    }
  }

  private emit(line: string, attribution: LogAttribution): void {
    let level: "debug" | "info" | "warn" | "error" = this.level;
    if (this.source === "stdout") {
      try {
        const payload: unknown = JSON.parse(line);
        if (
          payload &&
          typeof payload === "object" &&
          !Array.isArray(payload) &&
          withinJsonDepth(payload)
        ) {
          const value = (payload as Record<string, unknown>)["level"];
          if (typeof value === "string") {
            switch (value.trim().toLowerCase()) {
              case "debug":
                level = "debug";
                break;
              case "info":
                level = "info";
                break;
              case "warning":
              case "warn":
                level = "warn";
                break;
              case "error":
              case "err":
                level = "error";
                break;
            }
          }
        }
      } catch {
        // Non-JSON output retains the stream's default severity.
      }
    }
    this.logger[level](
      { source: this.source, [RECORD_ATTRIBUTION]: attribution },
      line,
    );
  }
}

type ActiveCaptures = { stdout: LoggingStream; stderr: LoggingStream };

/** Lets a request's partial lines ship when it ends, not at a next newline. */
export function flushCapturedOutput(scope: object): void {
  const g = globalThis as unknown as Record<symbol, unknown>;
  const active = g[ACTIVE_CAPTURES] as ActiveCaptures | undefined;
  for (const stream of [active?.stdout, active?.stderr]) {
    // Best-effort: a throwing appender must not fail the request or skip the
    // other stream.
    try {
      stream?.flushScope(scope);
    } catch {
      // Dropped, as in Python's `suppress(Exception)`.
    }
  }
}

export function patchStdio(): void {
  const stdoutStream = process.stdout as PatchableStream;
  const stderrStream = process.stderr as PatchableStream;
  // Snapshot before patching even though we don't reference the result
  // here — ensures the slot is populated for the appender below, and
  // protects against a re-install wrapping our already-patched `write`.
  snapshotOriginalWrite(stdoutStream);
  snapshotOriginalWrite(stderrStream);

  // Pull from the dedicated capture category (configured at `trace` in
  // installStructuredLogging) instead of the root logger — see the
  // STDIO_CAPTURE_CATEGORY comment for why. Routing through the root would
  // inherit the operator-set LOG_LEVEL and silently drop stray
  // `console.log` lines when LOG_LEVEL=WARN/ERROR, which is the exact
  // behaviour Python's `Logger.handle()` bypass was added to prevent.
  const captureLogger = log4js.getLogger(STDIO_CAPTURE_CATEGORY);
  const stdoutCapture = new LoggingStream(captureLogger, "info", "stdout");
  const stderrCapture = new LoggingStream(captureLogger, "warn", "stderr");

  stdoutStream.write = stdoutCapture.write.bind(stdoutCapture) as WriteFn;
  stderrStream.write = stderrCapture.write.bind(stderrCapture) as WriteFn;

  // Flush any buffered tail (a final write without a trailing newline) on
  // process exit so it isn't lost — the parity counterpart to Python's stdout
  // flush at interpreter shutdown. The handler is registered once per process
  // (`Symbol.for` guards against stacking one on every re-install), but it
  // must flush the *currently* installed capture streams, not the ones from
  // the first install it happened to close over. So the streams live in a
  // shared global slot that every patchStdio call overwrites, and the handler
  // reads that slot at exit time.
  const g = globalThis as unknown as Record<symbol, unknown>;
  // Flush the outgoing streams before dropping the reference to them: a
  // re-install replaces process.stdout.write, so a partial (newline-less) line
  // buffered in the previous capture would otherwise never be emitted anywhere
  // — it's not carried over to the new stream, and the exit handler only sees
  // whatever is current in ACTIVE_CAPTURES.
  const prev = g[ACTIVE_CAPTURES] as
    | { stdout: LoggingStream; stderr: LoggingStream }
    | undefined;
  if (prev) {
    prev.stdout.flush();
    prev.stderr.flush();
  }
  g[ACTIVE_CAPTURES] = { stdout: stdoutCapture, stderr: stderrCapture };
  if (!g[EXIT_FLUSH_REGISTERED]) {
    g[EXIT_FLUSH_REGISTERED] = true;
    process.on("exit", () => {
      const active = g[ACTIVE_CAPTURES] as
        | { stdout: LoggingStream; stderr: LoggingStream }
        | undefined;
      if (active) {
        active.stdout.flush();
        active.stderr.flush();
      }
    });
  }
}
