/**
 * Capture of `process.stdout` / `process.stderr` writes so stray
 * `console.log` / direct writes emerge as structured-logging records tagged
 * `source: "stdout"` / `"stderr"`.
 */

import log4js, { type Logger } from "log4js";

import {
  ACTIVE_CAPTURES,
  EXIT_FLUSH_REGISTERED,
  MAX_BUFFER_BYTES,
  ORIGINAL_WRITE_SLOT,
  STDIO_CAPTURE_CATEGORY,
} from "./constants.js";
import { truncateUtf8 } from "./serialize.js";

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

/**
 * Replacement for `process.stdout.write` / `process.stderr.write` that
 * routes writes through a logger.
 *
 * Bare `console.log` / `process.stdout.write` calls (and any other code
 * writing directly to the captured stream) are rewritten as log records
 * tagged with `source: "stdout"` (or `"stderr"`) so the layout can mark
 * them as such on the wire. Lines are buffered until newline, with a hard
 * byte cap so a producer writing without `\n` cannot grow the buffer
 * until OOM (see `MAX_BUFFER_BYTES`).
 *
 * Unlike the Python `LoggingStream` we do not need an `RLock` — Node's
 * event loop is single-threaded, so concurrent `write()` calls can't
 * interleave bytes mid-line. We also don't reproduce Python's
 * `fileno()`/`buffer`/`encoding` delegation: we only patch the `write`
 * method, leaving `process.stdout`'s fd / TTY introspection intact for
 * subprocess piping, progress-bar libs, and grpc-internal logging.
 */
/** @internal — exported only for tests; not part of the public API. */
export class LoggingStream {
  private buffer = "";

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
      this.buffer += text;
      let newlineIdx = this.buffer.indexOf("\n");
      while (newlineIdx !== -1) {
        const line = this.buffer.slice(0, newlineIdx);
        this.buffer = this.buffer.slice(newlineIdx + 1);
        // Whitespace-only lines are uninteresting — emitting a JSON
        // record with `message: "   "` adds noise and makes downstream
        // filters work harder. Mirrors the `if line.strip()` filter in
        // the Python LoggingStream; do not "fix" by removing it.
        if (line.trim()) this.emit(line);
        newlineIdx = this.buffer.indexOf("\n");
      }
      // Force-emit a truncated record if a producer is writing without
      // newlines — mirrors the exc_traceback truncation pattern. Slice
      // by UTF-8 byte count via `truncateUtf8` so the cap reflects what
      // Fluent Bit ships AND a partial multi-byte sequence at the cut
      // point is dropped rather than turned into U+FFFD replacement bytes
      // that would push the encoded payload back over the budget.
      if (Buffer.byteLength(this.buffer, "utf-8") > MAX_BUFFER_BYTES) {
        this.emit(truncateUtf8(this.buffer, MAX_BUFFER_BYTES));
        this.buffer = "";
      }
    }

    // Honor `write(chunk, cb)` and `write(chunk, encoding, cb)` callback
    // signatures so callers waiting on drain semantics don't hang.
    const cb = typeof encodingOrCb === "function" ? encodingOrCb : maybeCb;
    if (typeof cb === "function") (cb as (err?: Error | null) => void)();
    return true;
  }

  /**
   * Emit any buffered bytes that never got a trailing newline, then clear the
   * buffer. Called on process exit so a final `console.log`/`process.stdout.
   * write` without a `\n` right before shutdown is not lost — mirrors Python's
   * `LoggingStream.flush()` (which the interpreter invokes when it flushes
   * `sys.stdout` at shutdown). The same whitespace-only skip as `write()`
   * applies: a blank tail is not worth a record.
   */
  flush(): void {
    if (this.buffer.trim()) this.emit(this.buffer);
    this.buffer = "";
  }

  private emit(line: string): void {
    if (this.level === "warn") {
      this.logger.warn({ source: this.source }, line);
    } else {
      this.logger.info({ source: this.source }, line);
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
