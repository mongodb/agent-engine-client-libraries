/**
 * Shared constants for the structured-logging module.
 */

// Map RUNNER_MODE values to the canonical service identifier on the log line.
export const SERVICE_BY_MODE: Record<string, string> = {
  orchestrator: "orchestration-engine",
  aer: "agent-execution-runtime",
  tool: "tool-executor",
  tool_function: "tool-executor",
  "memory-server": "memory-server",
};

// Truncation for the formatted stack embedded in fields.exc_traceback.
// Long traces explode the JSON line size and choke Fluent Bit's batch
// pipeline; 8 KiB is enough to retain the call site and outer frames.
/** @internal — exported only for tests; not part of the public API. */
export const MAX_TRACEBACK_BYTES = 8192;

// Cap on LoggingStream's per-stream pending buffer. A library that writes a
// huge payload without `\n` (binary blob accidentally hitting stdout, a
// multi-MB stack from a native module, a misbehaving progress-bar lib)
// would otherwise grow the buffer until process OOM. 64 KiB is well above
// any reasonable log line and far below a runaway producer.
/** @internal — exported only for tests; not part of the public API. */
export const MAX_BUFFER_BYTES = 65536;

// Source value attached to log records by LoggingStream. For records that
// come straight from `logger.*` (i.e. not stdout/stderr capture) we default
// to `node-logging`. Mirrors the Python `_RECORD_SOURCE_ATTR` /
// `_DEFAULT_RECORD_SOURCE` constants; the value travels on the `source`
// field of the extras object passed to `logger.info({source}, msg)`.
export const DEFAULT_RECORD_SOURCE = "node-logging";

// Dedicated log4js category for stdout/stderr capture. Pinned to `trace` in
// the configuration (see install.ts) so captured records emerge regardless of
// the operator-set `LOG_LEVEL`. Python achieves the same effect by calling
// `Logger.handle(record)` from its `LoggingStream._emit` — which bypasses
// `isEnabledFor` (structured_logging.py:269-280 + 347-350 comment). log4js
// has no per-call bypass, but it does have per-category levels, so we route
// captures through a separate category whose level is held below the lowest
// LoggingStream emit level (info). Result: a stray `console.log` still ships
// as a JSON record even when an operator flips `LOG_LEVEL=ERROR` to chase an
// issue — same property Python guarantees.
export const STDIO_CAPTURE_CATEGORY =
  "agent_engine_runner_shared.stdio_capture";

// Mirrors Python's `for noisy in ("httpx", "httpcore", "urllib3")` silencing
// block. Node-native names kept too.
export const NOISY_LOGGER_NAMES: ReadonlyArray<string> = [
  "undici",
  "fastify.access",
  "httpx",
  "httpcore",
  "urllib3",
  "uvicorn.access",
];

// Symbol slot used to stash the original `write` function on
// `process.stdout` / `process.stderr` before we patch them. `Symbol.for`
// shares the slot across module instances, so a double-install (or a tester
// reloading the module) finds the prior snapshot instead of wrapping the
// already-patched write — which would recurse to RangeError.
export const ORIGINAL_WRITE_SLOT = Symbol.for(
  "agent_engine_runner_shared.original_write",
);

// Guard so the process-exit buffer-flush handler is registered only once per
// process, even across re-installs / a module reload (see patchStdio).
export const EXIT_FLUSH_REGISTERED = Symbol.for(
  "agent_engine_runner_shared.structured_logging_exit_flush",
);

// Slot holding the currently-installed capture streams. The exit handler is
// registered once and reads this slot at exit time (rather than closing over
// the first install's streams), so a re-install's trailing buffer is the one
// that gets flushed. Shared across module reloads via `Symbol.for`, matching
// the one-shot exit handler it feeds.
export const ACTIVE_CAPTURES = Symbol.for(
  "agent_engine_runner_shared.structured_logging_active_captures",
);

// Registers the crash listeners once — `process.on` is additive, so a repeat
// install would log the same crash twice.
export const UNCAUGHT_HANDLERS_REGISTERED = Symbol.for(
  "agent_engine_runner_shared.structured_logging_uncaught_handlers_registered",
);

// Read by those listeners at fire time, so a re-install logs through the
// current logger rather than the first install's.
export const ACTIVE_ROOT_LOGGER = Symbol.for(
  "agent_engine_runner_shared.structured_logging_active_root_logger",
);
