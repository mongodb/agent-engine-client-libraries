/**
 * Module logger factory for the Runner SDK.
 *
 * Wraps log4js to provide a Python-equivalent logging surface. log4js was
 * chosen over pino because its named-category registry mirrors Python's
 * `logging.getLogger(name)` semantics one-for-one: a global, hierarchical
 * registry of named loggers with per-name level overrides — none of which
 * pino offers natively. Reconfiguring the root via `setupLogging` is
 * automatically visible to every previously-issued `getLogger()` reference,
 * matching Python's behaviour without the Proxy-rebind dance pino required.
 *
 * Surface:
 *
 * - `getLogger("agent_engine_runner_shared.utils")` — named logger from the global
 *   registry, equivalent to Python `logging.getLogger(__name__)`. The same
 *   instance is returned on every call for a given name.
 * - `setupLogging({mode, level, ...})` — configure the root. When
 *   `STRUCTURED_LOGGING=true` delegates to `installStructuredLogging` so all
 *   output emerges as single-line JSON matching the agent-log
 *   contract; otherwise installs a human-readable pattern layout to stdout.
 *   On-disk logging is not part of the production model — container stdout
 *   is shipped to S3 by Fluent Bit, so a duplicate copy adds no value. When
 *   `AGENTIC_DEV_MODES` is set (local dev-up compose stacks only), a
 *   best-effort `dateFile` appender is added so the Local Dev UI log viewer
 *   can read agent output from disk; it rotates daily and keeps 5 files,
 *   mirroring Python's `TimedRotatingFileHandler`.
 * - Noisy HTTP libraries (`undici`, `httpx`, `httpcore`, `urllib3`,
 *   `fastify.access`, `uvicorn.access`) are silenced to `warn` — equivalent
 *   to Python's `logging.getLogger("httpx").setLevel(WARNING)` block.
 */

import log4js, { type Configuration, type Logger } from "log4js";
import { accessSync, constants, mkdirSync } from "node:fs";
import { join } from "node:path";

import { installStructuredLogging } from "./structured_logging.js";

const NOISY_LOGGERS: ReadonlyArray<string> = ["undici", "fastify.access"];

// Pattern matches Python's `setup_logging` console formatter shape:
// `%(asctime)s | %(levelname)-8s | %(message)s`. The file variant adds the
// category (mirroring Python's detailed file formatter).
const HUMAN_PATTERN = "%d{yyyy-MM-dd hh:mm:ss} | %p | %c | %m";
const FILE_PATTERN = "%d{yyyy-MM-dd hh:mm:ss} | %p | %c{1} | %m";
// Daily rotation with 5 kept files, matching Python's TimedRotatingFileHandler
// (which="midnight", backupCount=5). Keeps the dev log file bounded so a noisy
// local agent cannot fill the host disk.
const DEV_FILE_ROTATION = {
  pattern: "yyyy-MM-dd",
  numToKeep: 5,
  keepFileExt: true,
} as const;

function envLevel(): string {
  return (process.env["LOG_LEVEL"] ?? "info").toLowerCase();
}

type CategoryConfig = Configuration["categories"][string];

function noisyCategories(): Record<string, CategoryConfig> {
  return Object.fromEntries(
    NOISY_LOGGERS.map((name) => [
      name,
      { appenders: ["console"], level: "warn" },
    ]),
  );
}

function buildHumanConfig(
  level: string,
  fileOpts?: { logPath: string },
): Configuration {
  const appenders: Configuration["appenders"] = {
    console: {
      type: "stdout",
      layout: { type: "pattern", pattern: HUMAN_PATTERN },
    },
  };
  const defaultAppenders: string[] = ["console"];

  if (fileOpts) {
    appenders.devFile = {
      type: "dateFile",
      filename: fileOpts.logPath,
      layout: { type: "pattern", pattern: FILE_PATTERN },
      ...DEV_FILE_ROTATION,
    };
    defaultAppenders.push("devFile");
  }

  return {
    appenders,
    categories: {
      default: { appenders: defaultAppenders, level },
      ...noisyCategories(),
    },
  };
}

let configured = false;

function ensureDefaultConfig(): void {
  if (configured) return;
  log4js.configure(buildHumanConfig(envLevel()));
  configured = true;
}

/**
 * Mark the log4js root as already configured.
 *
 * Called by `installStructuredLogging` so that a direct `installStructuredLogging()`
 * (the documented startup entrypoint) followed by `getLogger()` does NOT let
 * `ensureDefaultConfig()` clobber the structured config with the
 * human stdout appender. Without this the structured pipeline could be torn
 * down — and re-running `log4js.configure` over patched stdio risks recursion.
 */
export function markConfigured(): void {
  configured = true;
}

/**
 * Return a logger named after the caller's module.
 *
 * Mirrors Python `logging.getLogger(__name__)`. log4js maintains a global
 * registry, so a later `setupLogging` reconfiguration is automatically
 * visible to every previously-issued reference. First call performs a
 * default configuration so module-scope `const logger = getLogger(__name__)`
 * patterns work without an explicit `setupLogging` at startup.
 */
export function getLogger(name?: string): Logger {
  ensureDefaultConfig();
  return log4js.getLogger(name);
}

export interface SetupLoggingArgs {
  /**
   * Application name — surfaced in the install log line and used to name
   * the dev-mode log file (`<appName>-<mode>.log`) when AGENTIC_DEV_MODES
   * is set. Matches Python `app_name`.
   */
  appName?: string;
  /** Runner mode (`aer` / `tool` / `orchestrator` / `memory-server`). */
  mode?: string;
  /** Log level — string ("debug", "info", ...) or `LOG_LEVEL` env when omitted. */
  logLevel?: string | null;
  /**
   * Log directory — used only when `AGENTIC_DEV_MODES` is set to write
   * a dev-mode file log alongside console output.
   */
  logDir?: string | null;
  /** Unused — parity placeholder. */
  backupCount?: number;
}

/**
 * Keep the dev log file name inside the log directory. `appName`/`mode` are
 * caller-supplied (`RuntimeOpts.appName`) and this is the one place the
 * package writes to disk, so a value containing a path separator could
 * otherwise redirect the write outside `logDir`. Separators and NUL are
 * replaced; an empty result falls back to a default so the name stays
 * deterministic.
 */
function sanitizeFileComponent(part: string, fallback: string): string {
  const cleaned = part.replace(/[/\\\0]/g, "_").trim();
  return cleaned.length > 0 ? cleaned : fallback;
}

/**
 * Best-effort mkdir + writability probe for the dev-mode log directory.
 * Returns false (and warns) when the directory cannot be created or written
 * so both logging paths skip the file appender instead of installing one
 * that errors on every write. The write probe matters because
 * `mkdirSync(recursive)` is a no-op success on an existing directory that
 * happens to be read-only. Mirrors Python's warn-and-continue in
 * `_install_file_handler`.
 */
function ensureLogDir(logDir: string): boolean {
  try {
    mkdirSync(logDir, { recursive: true });
    accessSync(logDir, constants.W_OK);
    return true;
  } catch (err) {
    // Pre-configuration diagnostic — the logger itself isn't set up yet,
    // so warn on stderr rather than pulling in a default log4js config.
    console.warn(
      `File logging disabled (logDir=${logDir} not writable: ${err instanceof Error ? err.message : String(err)}). Continuing with console output only.`,
    );
    return false;
  }
}

/**
 * Configure the root logger. Mirrors `agent_engine_runner_shared.utils.setup_logging`.
 *
 * When `STRUCTURED_LOGGING=true` is set, delegates to
 * `installStructuredLogging` so all output emerges as single-line JSON
 * matching the agent-log contract. On-disk logging is not part of the
 * production model — Fluent Bit ships container stdout to S3, so a
 * duplicate copy adds no value. The exception is local dev: when
 * `AGENTIC_DEV_MODES` is set (dev-up compose stacks only), a rotating file
 * sink is attached alongside the console/structured output.
 *
 * Unlike Python's `setup_logging` we don't need to manually drop existing
 * handlers before re-adding the console handler: `log4js.configure` fully
 * replaces the prior `appenders`/`categories` config on each call, so a
 * repeat call never accumulates duplicate writers. There's also no
 * stdout/stderr handler-close hazard to guard against — log4js's `stdout`
 * appender owns its own write path.
 *
 * @param args.appName     Application name — install line + dev log file name.
 * @param args.mode        Runtime mode (aer, tool, orchestrator, memory-server).
 * @param args.logLevel    Log level (default: from `LOG_LEVEL` env or INFO).
 * @param args.logDir      Log directory for dev-mode file logging (only when AGENTIC_DEV_MODES is set).
 * @param args.backupCount Unused — parity placeholder.
 */
export function setupLogging(args: SetupLoggingArgs = {}): Logger {
  const mode = args.mode ?? "aer";
  const logLevel = args.logLevel ?? null;
  const appName = args.appName ?? "runner";
  const logDir = args.logDir ?? process.env["LOG_DIR"] ?? "./logs";
  // Dev-only file logging: gated on AGENTIC_DEV_MODES (set exclusively by
  // the CLI's dev-up compose templates) AND a usable log directory.
  const devFileEnabled =
    !!process.env["AGENTIC_DEV_MODES"] && ensureLogDir(logDir);
  const devLogPath = devFileEnabled
    ? join(
        logDir,
        `${sanitizeFileComponent(appName, "runner")}-${sanitizeFileComponent(mode, "aer")}.log`,
      )
    : null;

  if ((process.env["STRUCTURED_LOGGING"] ?? "").toLowerCase() === "true") {
    // Pass `mode` through explicitly so the layout's `service` field
    // reflects the caller's intent even if `RUNNER_MODE` env happens to
    // be unset — without this, a `setupLogging({mode: "aer"})` call with
    // no env would silently produce `service="agent-execution-runtime"`
    // (the default), lying about which component emitted the line.
    installStructuredLogging({
      level: logLevel,
      mode,
      fileLogPath: devLogPath,
    });
    configured = true;
    return log4js.getLogger();
  }

  const level = (logLevel ?? envLevel()).toLowerCase();
  log4js.configure(
    buildHumanConfig(level, devLogPath ? { logPath: devLogPath } : undefined),
  );
  configured = true;
  const root = log4js.getLogger();
  root.info(
    `Logging initialized: level=${level.toUpperCase()}, file=${devLogPath ?? "<disabled>"}, mode=${mode}, app=${appName}`,
  );
  return root;
}
