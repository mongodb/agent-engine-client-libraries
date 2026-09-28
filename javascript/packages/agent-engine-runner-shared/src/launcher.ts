/**
 * Launcher module for pipeline-built agent containers (TS port of
 * `agent_engine_runner_shared/launcher.py`).
 *
 * Reads `AGENT_ENTRYPOINT` and starts the user's agent code. The generated
 * Dockerfile CMD invokes this file directly:
 *
 *     CMD ["node", "/app/node_modules/@mongodb-js/agent-engine-runner-shared/dist/launcher.js"]
 *
 * Environment variables:
 *   AGENT_ENTRYPOINT  Startup target in one of these forms:
 *     1) `<module-specifier>:<export-name>` — module is dynamically
 *        imported and the named export is invoked.
 *     2) `<module-specifier>:<export-name>` where the export is not a
 *        function but has a callable `run()` method — `.run()` is invoked.
 *     3) `<module-specifier>` (no colon) — defaults to `:main`.
 *
 * Structured logging is installed at the start of `runLauncher()` (before
 * import) so import-time and entrypoint failures emit `ERROR` records
 * instead of raw stderr stacks. `TenantRuntime` re-installs later; that
 * call is idempotent. After install, `console.error` is captured as
 * WARNING, so failure paths must use the logger.
 *
 * Security note: `AGENT_ENTRYPOINT` is baked into the Dockerfile ENV at
 * image-build time by the platform pipeline, not supplied by the user at
 * container runtime. The dynamic `import()` here serves the same role as
 * `node -e "import('module')"` — bootstrapping a known, pre-installed
 * module — and is an intentional exception to the "no dynamic code
 * execution in platform code" guideline.
 */

import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { getLogger, setupLogging } from "./logger.js";
import { runWithCustomerOrigin } from "./context.js";
import { redactText } from "./error_reporting.js";
import { materializeMcpOauthSecretCache } from "./mcp_oauth_secret.js";
import { getRuntimeMode } from "./utils.js";

const logger = getLogger("agent_engine_runner_shared.launcher");

/**
 * Kubernetes' default termination-message path (the "File"
 * TerminationMessagePolicy Kubelet always checks first, before falling back
 * to log-tail capture). Nothing in the deployed pod spec sets a custom
 * terminationMessagePath, so the default applies. A mutable binding so tests
 * can redirect it to a temp file.
 */
export let terminationLogPath = "/dev/termination-log";

/** Test-only override, mirroring `setHomeDirForTest` in error_reporting.ts. */
export function setTerminationLogPathForTest(path: string): void {
  terminationLogPath = path;
}

/**
 * 4 KiB / 40 lines, whichever is reached first — the same bound the deploy
 * event pipeline applies (see pkg/logredaction.DefaultMaxBytes/DefaultMaxLines
 * in the Go services, and `_bound_text` in the Python launcher; kept in sync
 * by convention across all three, since there's no shared package for this
 * across languages).
 */
const TERMINATION_MESSAGE_MAX_BYTES = 4 * 1024;
const TERMINATION_MESSAGE_MAX_LINES = 40;

/**
 * Truncate `text` to at most `maxLines` lines or `maxBytes` UTF-8 bytes,
 * whichever limit is hit first, appending a truncation marker when either
 * limit was hit.
 */
export function boundText(
  text: string,
  maxBytes: number = TERMINATION_MESSAGE_MAX_BYTES,
  maxLines: number = TERMINATION_MESSAGE_MAX_LINES,
): string {
  const marker = "\n[truncated]";
  let truncated = false;

  const lines = text.split("\n");
  if (lines.length > maxLines) {
    text = lines.slice(0, maxLines).join("\n");
    truncated = true;
  }

  const encoded = Buffer.from(text, "utf-8");
  if (encoded.length > maxBytes) {
    // toString("utf-8") on a boundary-split multi-byte sequence replaces the
    // partial tail with U+FFFD rather than throwing, which is an acceptable
    // (rare, cosmetic) outcome for a diagnostic message under a hard cap.
    text = encoded.subarray(0, maxBytes).toString("utf-8");
    truncated = true;
  }

  if (!truncated) return text;

  // Reserve headroom for `marker` so appending it below can't push the
  // final write past maxBytes — kubelet's own read of /dev/termination-log
  // is a blind byte-level cut that would otherwise mangle the tail,
  // potentially the marker itself.
  const budget = Math.max(0, maxBytes - marker.length);
  const finalEncoded = Buffer.from(text, "utf-8");
  if (finalEncoded.length > budget) {
    text = finalEncoded.subarray(0, budget).toString("utf-8");
  }
  return `${text}${marker}`;
}

/**
 * Record a bounded, redacted summary of a fatal startup error to the
 * container's termination message before the process exits, so the real
 * cause of the crash survives past this process's own stdout into
 * `ContainerStatus.LastTerminationState.Terminated.Message` — the field the
 * platform's crash diagnostics read, and from there into the
 * customer-facing deploy timeline.
 *
 * This is customer code (container mode), so unlike the platform's own
 * components the design intentionally keeps the full exception name,
 * message, and stack — redacted, not summarized away — since that detail is
 * what the customer needs to fix their own agent. Best-effort: if the write
 * fails, nothing is lost beyond what
 * `TerminationMessagePolicy: FallbackToLogsOnError` already provides.
 */
export function writeTerminationMessage(summary: string, err?: Error): void {
  let text = summary;
  if (err !== undefined) {
    text = `${summary}\n${err.stack ?? `${err.name}: ${err.message}`}`;
  }
  text = redactText(text);
  text = boundText(text);
  try {
    writeFileSync(terminationLogPath, text, { encoding: "utf-8" });
  } catch {
    // Best-effort — see doc comment above.
  }
}

/**
 * Log an ERROR, write a termination message via `writeTerminationMessage`,
 * and exit. Routes every deployment-breaking launcher exit through one path
 * — rather than instrumenting a hand-picked few — so the container's
 * termination message always carries the real cause forward to
 * `ContainerStatus.LastTerminationState.Terminated.Message`.
 */
function fatalWithTermination(summary: string, err?: Error, code = 1): never {
  if (err !== undefined) {
    logger.error(err, `${summary}: ${err.message}`);
  } else {
    logger.error(summary);
  }
  writeTerminationMessage(summary, err);
  process.exit(code);
}

/**
 * Startup-failure exit codes. OE's readiness wait can observe a
 * workload's real exit code once it exits (via fctr's Wait RPC) but not the
 * exception that caused it, so the exit code itself is the only signal that
 * reliably survives a startup crash to reach OE. Chosen to avoid every range
 * fctr's own const.go already claims: 0/1 (generic), 64-78 (sysexits.h), and
 * 128+signal (signal deaths, e.g. 137=SIGKILL, 143=SIGTERM).
 *
 * Frozen wire contract: OE's classification of a startup failure depends on
 * these exact values, and they are duplicated in
 * runner-shared/src/agent_engine_runner_shared/launcher.py. Changing either file breaks
 * OE's ability to distinguish failure causes and/or cross-language parity —
 * keep the two in lockstep.
 */
export const EXIT_IMPORT_ERROR = 82;
export const EXIT_NO_ENTRYPOINT = 83;
export const EXIT_STARTUP_CRASH = 84;

export interface ResolvedEntrypoint {
  modulePath: string;
  exportName: string;
}

/**
 * Resolve RUNNER_MODE for the early structured-logging install, falling back
 * to "aer" when unset or unrecognized. An unrecognized value would reach the
 * record's `service` field verbatim, and `/agent-logs` drops anything outside
 * `{agent-execution-runtime, tool-executor}` — so the crash log would vanish.
 * AGENT_ENTRYPOINT pods only ever run aer/tool.
 */
function resolveRunnerMode(): string {
  try {
    return getRuntimeMode();
  } catch {
    return "aer";
  }
}

/**
 * Normalize a thrown value to an `Error`: `layout.ts` only populates
 * `exc_type`/`exc_message`/`exc_traceback` for a genuine `Error`, and casting
 * a non-Error would make `.name`/`.message` read `undefined`.
 */
function toError(value: unknown): Error {
  return value instanceof Error
    ? value
    : new Error(String(value), { cause: value });
}

/**
 * Resolve an AGENT_ENTRYPOINT module path to a value Node's dynamic `import()`
 * can load.
 *
 * The platform bakes the `agent.yaml` `entrypoint` verbatim into
 * AGENT_ENTRYPOINT (same contract as the Python launcher), so the common form
 * is a dotted, Python-style module path — e.g. `agent_pkg.main` — that maps to
 * the agent's *compiled* output `<agentRoot>/dist/agent_pkg/main.js`. Node ESM
 * `import()` treats a dotted string as a bare package specifier and cannot
 * resolve it, so we translate it here: dots → path separators, under the
 * compiled `dist/` directory, with a `.js` suffix, returned as a `file://` URL
 * (the portable form for importing an absolute path across platforms).
 *
 * `agentRoot` defaults to `process.cwd()`, which at runtime is the agent
 * package root: the generated Dockerfile sets `WORKDIR` to the install target
 * and the `CMD` runs the launcher from there. It is injectable for tests.
 *
 * A value that is already directly importable — a relative path, an absolute
 * path, or any path-bearing specifier (one that contains a `/`) — is returned
 * unchanged, so a pre-resolved entrypoint (or a test passing an absolute file
 * path) still works and is never double-translated. A path-shape signal (not a
 * file extension) is used deliberately: a dotted module path whose final
 * segment happens to be `js`/`mjs`/`cjs` (e.g. `agent_pkg.cjs`) must still be
 * translated, not mistaken for an already-importable file.
 */
export function resolveImportTarget(
  modulePath: string,
  agentRoot: string = process.cwd(),
): string {
  const alreadyImportable =
    modulePath.startsWith(".") ||
    modulePath.startsWith("/") ||
    modulePath.includes("/");
  if (alreadyImportable) {
    return modulePath;
  }
  const relPath = modulePath.split(".").join("/");
  const absPath = resolve(agentRoot, "dist", `${relPath}.js`);
  return pathToFileURL(absPath).href;
}

/**
 * Parse `AGENT_ENTRYPOINT` into `{ modulePath, exportName }`.
 *
 * Exits the process with `EXIT_NO_ENTRYPOINT` if the env var is unset or
 * malformed — matches Python `_resolve_entrypoint`'s equivalent exit code.
 * Exported so tests can spawn this in a subprocess and assert on exit
 * code / stderr.
 */
export function resolveEntrypoint(): ResolvedEntrypoint {
  // Surrounding whitespace is stripped because the build pipeline extracts this
  // value from agent.yaml with a shell pipeline: a file authored on Windows
  // leaves a trailing CR that would otherwise become part of the export name
  // looked up below.
  const raw = (process.env["AGENT_ENTRYPOINT"] ?? "").trim();
  if (!raw) {
    fatalWithTermination(
      "AGENT_ENTRYPOINT is not set",
      undefined,
      EXIT_NO_ENTRYPOINT,
    );
  }

  let modulePath: string;
  let exportName: string;
  if (raw.includes(":")) {
    const idx = raw.lastIndexOf(":");
    modulePath = raw.slice(0, idx).trim();
    exportName = raw.slice(idx + 1).trim();
  } else {
    modulePath = raw;
    exportName = "main";
  }

  if (!modulePath || !exportName) {
    fatalWithTermination(
      `Invalid AGENT_ENTRYPOINT '${raw}': both module path and export name must be non-empty`,
      undefined,
      EXIT_NO_ENTRYPOINT,
    );
  }

  return { modulePath, exportName };
}

/**
 * Entry point invoked when this file is run directly:
 *
 *   `node /app/node_modules/@mongodb-js/agent-engine-runner-shared/dist/launcher.js`
 *
 * Exits the process with the exit code matching the failure category
 * (see `EXIT_IMPORT_ERROR`/`EXIT_NO_ENTRYPOINT`/`EXIT_STARTUP_CRASH` above)
 * on any error path; resolves normally after the user's target function or
 * `.run()` returns.
 */
export async function runLauncher(): Promise<void> {
  // Install before any import/entrypoint work. After this, console.error is
  // captured as WARNING (not ERROR), so failure paths must use the logger.
  setupLogging({ appName: "launcher", mode: resolveRunnerMode() });

  // Decode platform-injected MCP OAuth secrets into the file cache before the
  // agent module is imported, so any OAuth MCP client it constructs finds its
  // token. Sets AGENTIC_MCP_OAUTH_DIR before mcp_oauth.ts resolves its cache
  // dir on first (lazy) import.
  try {
    await materializeMcpOauthSecretCache();
  } catch (e) {
    fatalWithTermination(
      "Cannot materialize MCP OAuth credentials",
      toError(e),
    );
  }

  const { modulePath, exportName } = resolveEntrypoint();

  logger.info(`Launcher: importing ${modulePath}:${exportName}`);

  const importTarget = resolveImportTarget(modulePath);

  let mod: Record<string, unknown>;
  try {
    mod = (await runWithCustomerOrigin(() => import(importTarget))) as Record<
      string,
      unknown
    >;
  } catch (e) {
    // Include the resolved target so a cwd/WORKDIR mismatch or missing
    // compiled file is visible during on-call debugging, not just the dotted
    // module name the user wrote.
    fatalWithTermination(
      `Cannot import module '${modulePath}' (resolved to '${importTarget}')`,
      toError(e),
      EXIT_IMPORT_ERROR,
    );
  }

  const target = mod[exportName];
  if (target === undefined) {
    fatalWithTermination(
      `Module '${modulePath}' has no export '${exportName}'`,
      undefined,
      EXIT_NO_ENTRYPOINT,
    );
  }

  // Preferred form module:function; compatibility form module:appObject with a
  // callable .run(). Function first, so a callable export that also carries a
  // .run property still invokes the export itself.
  const run =
    target !== null && typeof target === "object"
      ? (target as Record<string, unknown>)["run"]
      : undefined;
  const invoke: (() => unknown | Promise<unknown>) | null =
    typeof target === "function"
      ? (target as () => unknown | Promise<unknown>)
      : typeof run === "function"
        ? (run as () => unknown | Promise<unknown>).bind(target)
        : null;

  if (invoke !== null) {
    try {
      // Scope covers the entrypoint's execution, not just its import —
      // AsyncLocalStorage propagates across awaits inside the callback.
      await runWithCustomerOrigin(() => invoke());
    } catch (e) {
      fatalWithTermination(
        `Unhandled exception from agent entrypoint '${modulePath}:${exportName}'`,
        toError(e),
        EXIT_STARTUP_CRASH,
      );
    }
    return;
  }

  fatalWithTermination(
    `'${modulePath}:${exportName}' is not callable and has no callable 'run()' method`,
    undefined,
    EXIT_NO_ENTRYPOINT,
  );
}

// CLI entrypoint guard: only auto-run when invoked as `node launcher.js`,
// not when imported as a library (e.g., from tests or barrel re-exports).
const isCliEntry =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (isCliEntry) {
  runLauncher().catch((e) => {
    fatalWithTermination("Launcher failed", toError(e));
  });
}
