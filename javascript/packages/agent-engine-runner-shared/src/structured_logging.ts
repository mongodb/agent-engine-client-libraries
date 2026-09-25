/**
 * Structured JSON logging for the agent runtime log capture pipeline.
 *
 * Emits single-line JSON records on container stdout. Fluent Bit tails the
 * container log, the Lua filter derives tenant/workspace keys from pod
 * metadata, and writes gzipped batches to S3. The API Gateway
 * `GET /admin/api/v1/agent-logs` endpoint reads them back.
 *
 * Activated by calling `installStructuredLogging()` at runner startup —
 * typically gated by `STRUCTURED_LOGGING=true`. When inactive, the default
 * log4js root in `logger.ts` is unaffected.
 *
 * This module is split into focused files under `structured_logging/`:
 *
 * - `constants.ts` — tunables and shared identifiers (service map, byte caps,
 *   capture category, the `Symbol.for` write slot).
 * - `env.ts` — the install-time env snapshot (`FormatterEnv`, `snapshotEnv`).
 * - `serialize.ts` — the crash-proof `jsonSafeStringify` and UTF-8-aware
 *   `truncateUtf8` helpers.
 * - `layout.ts` — `buildStructuredLayout`, the log4js layout that renders the JSON
 *   envelope (`StructuredJSONFormatter` in the Python port).
 * - `stdio_capture.ts` — `LoggingStream` + `patchStdio`, which replace
 *   `process.stdout.write` / `process.stderr.write` so stray `console.log` /
 *   direct writes also emerge as structured-logging records tagged `source: "stdout"`
 *   (INFO) or `"stderr"` (WARNING). Forcing stderr to ERROR was tried in
 *   Python and backed out — every deprecation warning would have paged
 *   on-call; WARNING is more honest.
 * - `install.ts` — `installStructuredLogging`, the public entry point.
 * - `uncaught.ts` — logs one ERROR summary per crash: an
 *   `uncaughtExceptionMonitor` for exceptions, a real listener for rejections.
 *   Installed by `installStructuredLogging`, not exported.
 *
 * Mirrors `agent_engine_runner_shared/structured_logging.py`. `_unwrap_logging_stream` maps
 * to the snapshot of the original `write` stashed on a `Symbol.for` slot on the
 * stream itself, so re-installing routes the custom appender back to the *real*
 * stdout instead of recursing through the patched write.
 *
 * Unlike the Python port we do not replace `process.stdout` itself — Node
 * code probes `process.stdout.fd` / `.isTTY` / `.columns` aggressively and a
 * full replacement breaks subprocess piping, tqdm-style progress bars, and
 * grpc-internal logging. We only patch `write`, which is enough to capture
 * every byte that would otherwise hit the console.
 */

export {
  MAX_TRACEBACK_BYTES,
  MAX_BUFFER_BYTES,
} from "./structured_logging/constants.js";
export { LoggingStream } from "./structured_logging/stdio_capture.js";
export {
  installStructuredLogging,
  type InstallStructuredLoggingArgs,
} from "./structured_logging/install.js";
export { FALLBACK_EXIT_MS } from "./structured_logging/uncaught.js";
