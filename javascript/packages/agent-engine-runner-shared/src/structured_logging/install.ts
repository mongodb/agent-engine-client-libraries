/**
 * Public entry point: `installStructuredLogging()` wires the structured-logging layout
 * into log4js as a custom appender, silences noisy HTTP loggers, and (unless
 * disabled) patches stdout/stderr capture.
 */

import log4js, {
  type Configuration,
  type Logger,
  type LoggingEvent,
} from "log4js";

import { markConfigured } from "../logger.js";
import { NOISY_LOGGER_NAMES, STDIO_CAPTURE_CATEGORY } from "./constants.js";
import { resolveLevel, snapshotEnv } from "./env.js";
import { buildStructuredLayout } from "./layout.js";
import { patchStdio, snapshotOriginalWrite } from "./stdio_capture.js";
import { installUncaughtExceptionHandlers } from "./uncaught.js";

// Custom log4js appender — writes via the snapshotted original stdout write so
// it doesn't recurse through our patched `process.stdout.write`.
function buildAppenderModule(layout: (event: LoggingEvent) => string): {
  configure: () => (event: LoggingEvent) => void;
} {
  return {
    configure: () => {
      const originalWrite = snapshotOriginalWrite(process.stdout);
      return (event: LoggingEvent): void => {
        originalWrite(layout(event) + "\n");
      };
    },
  };
}

export interface InstallStructuredLoggingArgs {
  /**
   * Log level — string name (`"DEBUG"`, `"info"`). Defaults to `LOG_LEVEL`
   * env var, then INFO. Must be honored on rollout because operators
   * continue to flip `LOG_LEVEL=DEBUG` to chase issues.
   */
  level?: string | null;
  /**
   * Runner mode (`"orchestrator"`, `"aer"`, `"tool"`, `"memory-server"`)
   * used to derive `service` and `fields.component` on the wire. Takes
   * precedence over `RUNNER_MODE` env var when set; falls back to env
   * when `null`. Lets `setupLogging(mode=...)` callers stay authoritative
   * even if pod env hasn't been stamped.
   */
  mode?: string | null;
  /**
   * Set to `false` to skip `process.stdout` / `process.stderr` patching.
   * Useful in tests that share a Node process across cases, or in callers
   * that own their own stdio routing. Defaults to `true` (matches Python).
   */
  captureStdio?: boolean;
  /**
   * Absolute path for a dev-mode `dateFile` appender alongside the
   * structured stdout appender. When set, a second log4js appender writes
   * human-readable records to this path, rotating daily and keeping the
   * last 5 files (Python `TimedRotatingFileHandler` parity). Only used by
   * `setupLogging` when `AGENTIC_DEV_MODES` is set; left unset in
   * production, where Fluent Bit is the only sink.
   */
  fileLogPath?: string | null;
}

/**
 * Install structured logging on the log4js root.
 *
 * Behaves like the Python SDK's `install_structured_logging`:
 *
 * - Idempotent: re-installing routes the appender through the snapshotted
 *   original `stdout.write` (stashed under `ORIGINAL_WRITE_SLOT`), so it
 *   keeps writing to the real stdout instead of recursing through a
 *   patched write.
 * - Silences noisy HTTP libraries (see `NOISY_LOGGER_NAMES`) — equivalent
 *   to Python's `logging.getLogger("httpx").setLevel(WARNING)` block.
 * - Patches `process.stdout.write` / `process.stderr.write` (unless
 *   `captureStdio: false`) so stray writes emerge as structured-logging records
 *   tagged with `source: "stdout"` (INFO) or `source: "stderr"` (WARNING).
 *   stderr was deliberately *not* mapped to ERROR — Python had to back that
 *   out because every deprecation warning would have paged on-call.
 */
export function installStructuredLogging(
  args: InstallStructuredLoggingArgs = {},
): Logger {
  const env = snapshotEnv(args.mode ?? null);
  const level = resolveLevel(args.level);
  const layout = buildStructuredLayout(env);
  const appenderModule = buildAppenderModule(layout);

  type CategoryConfig = Configuration["categories"][string];
  // `enableCallStack: true` makes log4js attach `fileName`/`lineNumber`/
  // `functionName` to each LoggingEvent — required for the layout to emit
  // `fields.filename` / `fields.lineno` / `fields.funcName` for logger-
  // sourced records (Python parity, see layout above). It's set per-
  // category in log4js (no global toggle), so enable on every category we
  // configure including the noisy ones — a noisy library that does fire a
  // WARN is exactly the case where the call site is useful to keep.
  const noisyCategories: Record<string, CategoryConfig> = Object.fromEntries(
    NOISY_LOGGER_NAMES.map((name) => [
      name,
      { appenders: ["structured"], level: "warn", enableCallStack: true },
    ]),
  );

  const appenders: Configuration["appenders"] = {
    // log4js accepts an inline module object for `type` — its `configure`
    // function is invoked once at `log4js.configure(...)` time and
    // returns the appender function used for every record.
    structured: { type: appenderModule },
  };

  if (args.fileLogPath) {
    appenders.devFile = {
      type: "dateFile",
      filename: args.fileLogPath,
      layout: {
        type: "pattern",
        pattern: "%d{yyyy-MM-dd hh:mm:ss} | %p | %c | %m",
      },
      // Daily rotation, 5 kept files — Python TimedRotatingFileHandler parity.
      pattern: "yyyy-MM-dd",
      numToKeep: 5,
      keepFileExt: true,
    };
  }

  const defaultAppenders = args.fileLogPath
    ? ["structured", "devFile"]
    : ["structured"];

  const config: Configuration = {
    appenders,
    categories: {
      default: { appenders: defaultAppenders, level, enableCallStack: true },
      // Stdio capture category pinned to `trace` so emits never get
      // filtered by the operator-set `LOG_LEVEL`. Mirrors Python's
      // `Logger.handle()` bypass; see STDIO_CAPTURE_CATEGORY comment.
      // `enableCallStack` deliberately off — the call site for a captured
      // line is always `LoggingStream.emit`, which is noise, and the
      // layout already suppresses fields.filename/lineno/funcName when
      // source !== DEFAULT_RECORD_SOURCE.
      [STDIO_CAPTURE_CATEGORY]: { appenders: defaultAppenders, level: "trace" },
      ...noisyCategories,
    },
  };
  log4js.configure(config);
  // Keep `logger.ts` state in sync: a later `getLogger()` must not re-run
  // `ensureDefaultConfig()` and replace this structured-logging config with the human
  // stdout appender (which would also re-`configure` over patched stdio).
  markConfigured();

  const rootLogger = log4js.getLogger();
  if (args.captureStdio !== false) patchStdio();
  installUncaughtExceptionHandlers(rootLogger);

  rootLogger.debug({ source: "node-logging" }, "structured logging installed");
  return rootLogger;
}
