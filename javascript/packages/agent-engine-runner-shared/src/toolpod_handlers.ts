/**
 * Built-in Tool Pod handler functions (TypeScript port of Python's
 * `toolpod_handlers.py`).
 *
 * These handlers execute filesystem and shell operations inside the Tool Pod.
 * They are registered at startup via `registerBuiltinTools` (gated on
 * `features.deep_agent`) and dispatched by `ToolServer` when requests arrive
 * from `AgentEngineToolPodBackend` through the OE secure path.
 *
 * All handlers accept a single named-argument object (the `ServerToolFn`
 * contract), return a JSON-serializable object, and catch their own
 * exceptions (returning `{error: ...}` on failure). `shell_execute` and
 * `filesystem_glob` are async (the latter streams matches via
 * `fs.promises.glob` so a broad pattern never materializes a huge list); the
 * rest are synchronous.
 *
 * Return formats match what `AgentEngineToolPodBackend` expects to parse:
 * - filesystem_ls       -> {entries: [{path, is_dir}], truncated}
 * - filesystem_read     -> {content, encoding: "utf-8"}
 * - filesystem_write    -> {path}
 * - filesystem_edit     -> {occurrences}
 * - filesystem_glob     -> {matches: [{path, is_dir}], truncated}
 * - filesystem_grep     -> {matches: [{path, line, text}], truncated}
 * - filesystem_download -> {path, content_base64, encoding: "base64"}
 * - shell_execute       -> {output, exit_code, truncated}
 *
 * Read-only skill roots resolve lazily at first use / startup, not at
 * import, so a static SDK import before the agent root is set still works.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";

import { getCurrentSessionId } from "./context.js";
import { getLogger } from "./logger.js";
import type { ITenantRuntime, ServerToolFn } from "./server/base.js";

const logger = getLogger("agent_engine_runner_shared.toolpod_handlers");

// ---------------------------------------------------------------------------
// realpath helpers
// ---------------------------------------------------------------------------

/**
 * Python's `os.path.realpath` resolves symlinks for the components that
 * exist and leaves the non-existent tail lexical; Node's `fs.realpathSync`
 * throws on any missing component. Resolve the longest existing ancestor and
 * re-join the remainder so a workspace path that doesn't exist yet (and a
 * symlinked mount like macOS `/tmp` -> `/private/tmp`) both normalize the way
 * the Python handler expects.
 */
function realpathBestEffort(p: string): string {
  let current = path.resolve(p);
  const tail: string[] = [];
  // Walk up until an existing ancestor is found, then realpath it.
  for (;;) {
    try {
      const real = fs.realpathSync(current);
      return tail.length ? path.join(real, ...tail.reverse()) : real;
    } catch {
      const parent = path.dirname(current);
      if (parent === current) {
        // Reached the filesystem root without finding an existing component.
        return path.resolve(p);
      }
      tail.push(path.basename(current));
      current = parent;
    }
  }
}

function realAbs(p: string): string {
  return realpathBestEffort(path.resolve(p));
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

// Default the workspace to a path *outside* the typical project tree. Putting
// it under the project root causes dev-mode file watchers to restart the
// AER/OE processes on every write, killing in-flight LLM streams. Explicit
// `WORKSPACE_DIR` env wins so production containers can pin any path.
// `realpath` (not just `resolve`) so a symlinked mount and its target compare
// equal when callers pass either spelling.
export const WORKSPACE_DIR = realAbs(
  process.env["WORKSPACE_DIR"] ?? "/tmp/agent-workspace",
);

function agentSourceRootFromEnv(): string | null {
  const configPath = process.env["AGENTIC_AGENT_CONFIG_PATH"];
  if (configPath) return path.dirname(realAbs(configPath));
  const workdir = process.env["AGENTIC_AGENT_WORKDIR"];
  if (workdir) return realAbs(workdir);
  return null;
}

function readonlySkillsRootFromEnv(): string | null {
  const sourceRoot = agentSourceRootFromEnv();
  const explicit = process.env["AGENTIC_SKILLS_DIR"];
  const skillsDirSetting = explicit || "skills";
  if (sourceRoot === null) {
    if (explicit) {
      logger.warn(
        `Ignoring AGENTIC_SKILLS_DIR=${JSON.stringify(explicit)} because the agent source root is unknown`,
      );
    }
    return null;
  }
  if (path.isAbsolute(skillsDirSetting)) {
    logger.warn(
      `Ignoring AGENTIC_SKILLS_DIR=${JSON.stringify(skillsDirSetting)} because it must be relative to the agent source root ${sourceRoot}`,
    );
    return null;
  }
  const skillsDir = realAbs(path.join(sourceRoot, skillsDirSetting));
  if (!isWithinRoot(skillsDir, sourceRoot)) {
    logger.warn(
      `Ignoring AGENTIC_SKILLS_DIR=${JSON.stringify(skillsDirSetting)} because it resolves outside the agent source root ${sourceRoot}`,
    );
    return null;
  }
  return skillsDir;
}

function isReadableDir(candidate: string): boolean {
  try {
    if (!fs.statSync(candidate).isDirectory()) return false;
    // X_OK too: opening files under a directory requires search permission.
    fs.accessSync(candidate, fs.constants.R_OK | fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

// Resolved lazily, not at import: a static import before the app sets
// AGENTIC_AGENT_WORKDIR/AGENTIC_AGENT_CONFIG_PATH must still see the right
// root. Memoized — the skill root is process-global and must not
// shift between sessions. Roots that aren't readable directories are excluded
// (agents that bundle no skills); a configured-but-unusable root throws at
// startup in validateReadonlyResourceRootsAtStartup.
let readonlyResourceRootsMemo: readonly string[] | undefined;

function getReadonlyResourceRoots(): readonly string[] {
  if (readonlyResourceRootsMemo === undefined) {
    readonlyResourceRootsMemo = [readonlySkillsRootFromEnv()]
      .filter((r): r is string => r !== null)
      .filter(isReadableDir);
  }
  return readonlyResourceRootsMemo;
}

const SHELL_OUTPUT_MAX_BYTES = envInt("SHELL_OUTPUT_MAX_BYTES", 65536);

// `timeout=null` means "use the backend default", not "run forever".
const SHELL_DEFAULT_TIMEOUT_SECONDS = envInt(
  "SHELL_DEFAULT_TIMEOUT_SECONDS",
  30,
);

// Cap filesystem_read / filesystem_edit raw bytes so broad reads on giant
// files don't exhaust pod memory. filesystem_edit loads the whole file, so the
// same cap applies.
const FILESYSTEM_READ_MAX_BYTES = envInt(
  "FILESYSTEM_READ_MAX_BYTES",
  10 * 1024 * 1024,
);

// Cap filesystem_grep matches + wall-clock budget. Deepagents' grep contract is
// literal substring matching (not regex), so classic ReDoS doesn't apply — but
// the time budget still protects against deep tree walks pegging a worker.
const MAX_GREP_MATCHES = envInt("MAX_GREP_MATCHES", 1000);
const MAX_GREP_SECONDS =
  Number.parseFloat(process.env["MAX_GREP_SECONDS"] ?? "") || 5;

const MAX_GLOB_MATCHES = envInt("MAX_GLOB_MATCHES", 1000);
const MAX_LS_ENTRIES = envInt("MAX_LS_ENTRIES", 1000);
const DOWNLOAD_MAX_BYTES = envInt("DOWNLOAD_MAX_BYTES", 10 * 1024 * 1024);

// Warn (but do not reject) when filesystem_write writes a file larger than
// this. Tool Pod workspace storage is RAM-backed; large writes reduce pod
// memory without the usual page-cache eviction safety net.
const FILESYSTEM_WRITE_WARN_BYTES = envInt(
  "FILESYSTEM_WRITE_WARN_BYTES",
  1 * 1024 * 1024,
);

// Known writable RAM-backed mounts inside a Tool Pod. Validated at startup.
const KNOWN_WORKSPACE_PREFIXES: readonly string[] = ["/tmp", "/scratch"];

// Env vars shell_execute may propagate to the child. Everything else (tenant
// API keys, DB URIs, service URLs) is stripped — a prompt-injected agent
// running `env` must not see them (OWASP LLM01). Per-call scrub; the pod-level
// mount may be revisited in the future.
const SHELL_ENV_ALLOWLIST: ReadonlySet<string> = new Set([
  "PATH",
  "HOME",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TZ",
]);

// Wall-clock budget for shell cleanup: SIGKILL reaping after a timeout.
const SHELL_CLEANUP_TIMEOUT_MS = 5000;

// Sentinel exit codes. A child killed by a signal reports a negative code on
// Unix; these small reserved values flag framework-level states instead.
const EXIT_CODE_TIMEOUT = -1;
const EXIT_CODE_FRAMEWORK_ERROR = -2;

/**
 * Canonical set of built-in tool names registered by `registerBuiltinTools`.
 * Consumed by `ToolServer.onStartup`'s completeness assertion and by tests so
 * the callsites can't drift silently.
 */
export const BUILTIN_TOOL_NAMES: ReadonlySet<string> = new Set([
  "filesystem_ls",
  "filesystem_read",
  "filesystem_write",
  "filesystem_edit",
  "filesystem_glob",
  "filesystem_grep",
  "filesystem_download",
  "shell_execute",
]);

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

function isWithinRoot(resolved: string, root: string): boolean {
  const realResolved = realpathBestEffort(resolved);
  return realResolved === root || realResolved.startsWith(root + path.sep);
}

// Reserved subdirectory under WORKSPACE_DIR holding per-session subtrees.
const SESSIONS_NAMESPACE = ".sessions";

/**
 * Return a filesystem-safe directory name derived from the current session id,
 * or `""` when no session is active (dev scripts, low-level tests fall back to
 * the base WORKSPACE_DIR).
 *
 * The session id is hashed so any caller-provided string (including forms like
 * `team/thread-1` or unicode) maps to a fixed-charset slot name that cannot
 * contain path separators or collapse to `.`/`..`. The slot is not a security
 * boundary — cross-session isolation is enforced by `isWithinWorkspace`.
 */
function sessionSubdir(): string {
  const sid = getCurrentSessionId();
  if (!sid) return "";
  const normalized = Buffer.from(sid.normalize("NFC"), "utf-8");
  return crypto.createHash("sha256").update(normalized).digest("hex");
}

function sessionsRoot(): string {
  return path.normalize(path.join(WORKSPACE_DIR, SESSIONS_NAMESPACE));
}

/**
 * The workspace directory for the current session: WORKSPACE_DIR when no
 * session is active, else `WORKSPACE_DIR/.sessions/<slot>`.
 */
function effectiveWorkspaceDir(): string {
  const sub = sessionSubdir();
  return sub ? path.normalize(path.join(sessionsRoot(), sub)) : WORKSPACE_DIR;
}

/**
 * True if *resolved* is the caller's allowed workspace or a path under it.
 * When a session is active this narrows to the session's own subtree so
 * concurrent sessions on the same Tool Pod can't read/write each other's
 * files. Uses realpath so symlinks can't escape the boundary.
 */
function isWithinWorkspace(resolved: string): boolean {
  return isWithinRoot(resolved, effectiveWorkspaceDir());
}

function isWithinReadonlyResourceRoot(resolved: string): boolean {
  return getReadonlyResourceRoots().some((root) =>
    isWithinRoot(resolved, root),
  );
}

function isWithinReadableRoot(resolved: string): boolean {
  return isWithinWorkspace(resolved) || isWithinReadonlyResourceRoot(resolved);
}

/**
 * Resolve *path* to an absolute path under WORKSPACE_DIR (or the per-session
 * subtree). Relative paths join the effective workspace; absolute paths already
 * under WORKSPACE_DIR are kept; other absolute paths have their leading `/`
 * stripped and are rebased. Read-only ops may pass absolute paths under bundled
 * resource roots (e.g. `<agent-dir>/skills`). Escapes via `..` or symlinks
 * throw so the agent gets a clear signal instead of silent clamping.
 */
function resolvePath(
  inputPath: string,
  opts: { allowReadonlyRoots?: boolean } = {},
): string {
  const allowReadonlyRoots = opts.allowReadonlyRoots ?? false;
  const base = effectiveWorkspaceDir();
  let resolved: string;

  if (path.isAbsolute(inputPath)) {
    const normalized = path.normalize(inputPath);
    // Accept the input as-is or after symlink resolution, so a symlinked
    // workspace mount (macOS /tmp -> /private/tmp) is recognized.
    const realNormalized = realpathBestEffort(normalized);
    if (
      normalized === WORKSPACE_DIR ||
      normalized.startsWith(WORKSPACE_DIR + path.sep) ||
      realNormalized === WORKSPACE_DIR ||
      realNormalized.startsWith(WORKSPACE_DIR + path.sep)
    ) {
      resolved = normalized;
    } else if (isWithinReadonlyResourceRoot(normalized)) {
      if (allowReadonlyRoots) return normalized;
      throw new Error(
        `Path is inside a read-only resource root: ${JSON.stringify(inputPath)}. ` +
          "Use read-only filesystem tools or write to the workspace.",
      );
    } else {
      fs.mkdirSync(base, { recursive: true });
      resolved = path.normalize(
        path.join(base, inputPath.replace(/^[/\\]+/, "")),
      );
    }
  } else {
    // Auto-create the session subdir so a brand-new session's first `ls('.')`
    // doesn't ENOENT before any write.
    fs.mkdirSync(base, { recursive: true });
    resolved = path.normalize(path.join(base, inputPath));
  }

  if (!isWithinWorkspace(resolved)) {
    throw new Error(
      `Path escapes workspace sandbox: ${JSON.stringify(inputPath)}. ` +
        "Use a path inside the workspace (relative paths recommended).",
    );
  }
  return resolved;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function errResult(e: unknown): Record<string, unknown> {
  return { error: e instanceof Error ? e.message : String(e) };
}

/** True if `file`'s realpath resolves to a directory (follows symlinks, like
 * Python's `os.DirEntry.is_dir()`); false on any error (e.g. broken symlink). */
function isDir(file: string): boolean {
  try {
    return fs.statSync(file).isDirectory();
  } catch {
    return false;
  }
}

/** Translate an fnmatch-style glob (`*`, `?`, `[...]`) to a RegExp anchored to
 * the whole filename — mirrors Python's `fnmatch.fnmatch` for grep's filter. */
function fnmatchToRegExp(pattern: string): RegExp {
  let re = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i] ?? "";
    if (c === "*") re += ".*";
    else if (c === "?") re += ".";
    else if (c === "[") {
      let j = i + 1;
      if (pattern[j] === "!") j++;
      if (pattern[j] === "]") j++;
      while (j < pattern.length && pattern[j] !== "]") j++;
      if (j >= pattern.length) {
        re += "\\[";
      } else {
        let inner = pattern.slice(i + 1, j);
        if (inner.startsWith("!")) inner = "^" + inner.slice(1);
        re += "[" + inner.replace(/\\/g, "\\\\") + "]";
        i = j;
      }
    } else re += c.replace(/[.+^${}()|\\]/g, "\\$&");
  }
  return new RegExp(`^${re}$`);
}

// ---------------------------------------------------------------------------
// Filesystem handlers
// ---------------------------------------------------------------------------

/**
 * List directory contents, sorted by name. When the directory has more than
 * MAX_LS_ENTRIES entries, `truncated=true` and the returned subset is the
 * readdir-order first MAX_LS_ENTRIES entries (filesystem-defined), then sorted
 * by path. The early break is intentional — sorting all entries first would
 * defeat the memory cap on directories with hundreds of thousands of entries.
 */
export function filesystemLs(
  args: Record<string, unknown>,
): Record<string, unknown> {
  try {
    const resolved = resolvePath(String(args["path"] ?? ""), {
      allowReadonlyRoots: true,
    });
    const entries: Array<{ path: string; is_dir: boolean }> = [];
    let truncated = false;
    // opendir + early break so a runaway directory never materializes every
    // entry into memory (readdirSync would read them all at once).
    const dir = fs.opendirSync(resolved);
    try {
      for (;;) {
        const dirent = dir.readSync();
        if (dirent === null) break;
        if (entries.length >= MAX_LS_ENTRIES) {
          truncated = true;
          break;
        }
        const full = path.join(resolved, dirent.name);
        entries.push({ path: full, is_dir: isDir(full) });
      }
    } finally {
      dir.closeSync();
    }
    entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    return { entries, truncated };
  } catch (e) {
    return errResult(e);
  }
}

/**
 * Read file content as text with line-based slicing (lines
 * `[offset, offset+limit)`). Rejects files larger than
 * FILESYSTEM_READ_MAX_BYTES upfront so pathologically large files cannot
 * exhaust pod memory.
 */
export function filesystemRead(
  args: Record<string, unknown>,
): Record<string, unknown> {
  try {
    const filePath = String(args["file_path"] ?? "");
    const offset = Number(args["offset"] ?? 0);
    const limit = Number(args["limit"] ?? 2000);
    const resolved = resolvePath(filePath, { allowReadonlyRoots: true });
    const size = fs.statSync(resolved).size;
    if (size > FILESYSTEM_READ_MAX_BYTES) {
      return {
        error: `File exceeds FILESYSTEM_READ_MAX_BYTES (${size} > ${FILESYSTEM_READ_MAX_BYTES})`,
      };
    }
    const text = fs.readFileSync(resolved, "utf-8");
    // Split keeping newlines so re-joined content is byte-identical to Python's
    // line-iterator slice. A trailing empty element (from a final newline) is
    // dropped so it isn't counted as a line.
    const lines = text.split(/(?<=\n)/);
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    const selected = lines.slice(offset, offset + limit);
    return { content: selected.join(""), encoding: "utf-8" };
  } catch (e) {
    return errResult(e);
  }
}

/**
 * Write *content* to a new file, creating parent directories. Per
 * `BackendProtocol.write` this is create-only: if the file exists the call
 * fails. Agents modify existing files via filesystem_edit (which has a
 * unique-match guard).
 */
export function filesystemWrite(
  args: Record<string, unknown>,
): Record<string, unknown> {
  const filePath = String(args["file_path"] ?? "");
  const content = String(args["content"] ?? "");
  let resolved: string;
  try {
    resolved = resolvePath(filePath);
  } catch (e) {
    return errResult(e);
  }
  try {
    const contentBytes = Buffer.byteLength(content, "utf-8");
    if (contentBytes > FILESYSTEM_WRITE_WARN_BYTES) {
      logger.warn(
        `filesystem_write: writing large file ${JSON.stringify(resolved)} (${contentBytes} bytes). ` +
          "Tool Pod workspace is RAM-backed — large files reduce available pod memory.",
      );
    }
    const parent = path.dirname(resolved);
    if (parent) fs.mkdirSync(parent, { recursive: true });
    // "wx" == O_CREAT | O_EXCL: fail if the file already exists.
    fs.writeFileSync(resolved, content, { encoding: "utf-8", flag: "wx" });
    return { path: resolved };
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "EEXIST") {
      return {
        error: `File already exists at ${JSON.stringify(filePath)}; use filesystem_edit to modify existing files.`,
      };
    }
    return errResult(e);
  }
}

/**
 * Find and replace text in a file, returning the number of occurrences
 * replaced. When *replace_all* is false the match must be unique — a non-unique
 * `old_string` is rejected so a caller cannot silently corrupt the wrong
 * region. Files larger than FILESYSTEM_READ_MAX_BYTES (before or after the
 * edit) are rejected to mirror filesystem_read's memory guard.
 */
export function filesystemEdit(
  args: Record<string, unknown>,
): Record<string, unknown> {
  try {
    const filePath = String(args["file_path"] ?? "");
    const oldString = String(args["old_string"] ?? "");
    const newString = String(args["new_string"] ?? "");
    const replaceAll = Boolean(args["replace_all"] ?? false);
    // Empty old_string: count("") is len+1 (fails unique-match) and a
    // replace-all interleaves new_string between every character.
    if (oldString === "") return { error: "old_string must be non-empty" };
    const resolved = resolvePath(filePath);
    const size = fs.statSync(resolved).size;
    if (size > FILESYSTEM_READ_MAX_BYTES) {
      return {
        error: `File exceeds FILESYSTEM_READ_MAX_BYTES (${size} > ${FILESYSTEM_READ_MAX_BYTES})`,
      };
    }
    const text = fs.readFileSync(resolved, "utf-8");
    const matches = countOccurrences(text, oldString);
    if (matches === 0) {
      return { error: `old_string not found in ${JSON.stringify(filePath)}` };
    }

    let count: number;
    let newText: string;
    if (replaceAll) {
      count = matches;
      newText = text.split(oldString).join(newString);
    } else {
      if (matches > 1) {
        return {
          error:
            `old_string is not unique (${matches} matches in ${JSON.stringify(filePath)}); ` +
            "set replace_all=true or supply a longer anchor",
        };
      }
      count = 1;
      newText = text.replace(oldString, newString);
    }

    // Re-check the cap on the post-replacement text so a large new_string can't
    // grow the file past the guard the up-front check enforces.
    const newSize = Buffer.byteLength(newText, "utf-8");
    if (newSize > FILESYSTEM_READ_MAX_BYTES) {
      return {
        error:
          `Edit would exceed FILESYSTEM_READ_MAX_BYTES (${newSize} > ${FILESYSTEM_READ_MAX_BYTES}); ` +
          "reduce new_string size or split the edit into smaller pieces",
      };
    }

    fs.writeFileSync(resolved, newText, "utf-8");
    return { occurrences: count };
  } catch (e) {
    return errResult(e);
  }
}

/** Count non-overlapping occurrences of *needle* in *haystack* (Python
 * `str.count` semantics; needle is guaranteed non-empty by the caller). */
function countOccurrences(haystack: string, needle: string): number {
  let count = 0;
  let idx = haystack.indexOf(needle);
  while (idx !== -1) {
    count++;
    idx = haystack.indexOf(needle, idx + needle.length);
  }
  return count;
}

/**
 * Match files using a glob pattern under *path* (workspace-relative, default
 * `"."`). Supports recursive `**` patterns. Bounded by MAX_GLOB_MATCHES.
 * Absolute patterns and `..` traversal segments are rejected so they cannot
 * bypass the sandbox; returned matches are also filtered through
 * `isWithinReadableRoot` for defense-in-depth against symlinks.
 */
export async function filesystemGlob(
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  try {
    const pattern = String(args["pattern"] ?? "");
    const inputPath = String(args["path"] ?? ".");
    if (path.isAbsolute(pattern)) {
      return {
        error: `Absolute pattern ${JSON.stringify(pattern)} is not allowed; use a workspace-relative pattern.`,
      };
    }
    // Reject traversal up front. glob enumerates the filesystem while expanding
    // a pattern like `../../etc/*`; even though matches are filtered, the
    // enumeration itself can reveal structure outside the readable roots.
    const patternParts = pattern.replace(/\\/g, "/").split("/");
    if (patternParts.includes("..")) {
      return {
        error: `Pattern ${JSON.stringify(pattern)} contains traversal segments; use a workspace-relative pattern.`,
      };
    }
    const resolved = resolvePath(inputPath, { allowReadonlyRoots: true });
    const matches: Array<{ path: string; is_dir: boolean }> = [];
    let truncated = false;
    // `fs.glob` yields matches lazily (unlike `fs.globSync`, which materializes
    // the full list first), so the MAX_GLOB_MATCHES break below stops the walk
    // before a broad pattern allocates an unbounded array — parity with the
    // Python handler's `glob.iglob` generator. Paths are relative to `cwd`;
    // re-join to absolute for the readable-root filter and returned `path`.
    for await (const rel of fs.promises.glob(pattern, { cwd: resolved })) {
      const full = path.resolve(resolved, rel);
      if (!isWithinReadableRoot(full)) continue;
      matches.push({ path: full, is_dir: isDir(full) });
      if (matches.length >= MAX_GLOB_MATCHES) {
        truncated = true;
        break;
      }
    }
    matches.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
    return { matches, truncated };
  } catch (e) {
    return errResult(e);
  }
}

/**
 * Search file contents for a literal substring (per `BackendProtocol.grep`,
 * not a regex). Walks the tree under *path* (default `"."`); when *glob* is
 * given only files whose names match are searched. Binary files are skipped.
 * Bounded by MAX_GREP_MATCHES and MAX_GREP_SECONDS so a deep-tree walk cannot
 * peg a worker indefinitely.
 */
export function filesystemGrep(
  args: Record<string, unknown>,
): Record<string, unknown> {
  try {
    const pattern = String(args["pattern"] ?? "");
    const inputPath = args["path"] == null ? "." : String(args["path"]);
    const glob = args["glob"] == null ? null : String(args["glob"]);
    const globRe = glob ? fnmatchToRegExp(glob) : null;
    const searchPath = resolvePath(inputPath, { allowReadonlyRoots: true });
    const matches: Array<{ path: string; line: number; text: string }> = [];
    let truncated = false;
    const deadline = Date.now() + MAX_GREP_SECONDS * 1000;

    // Iterative directory walk (avoids recursion depth limits on deep trees).
    const stack: string[] = [searchPath];
    walk: while (stack.length > 0) {
      const dirpath = stack.pop();
      if (dirpath === undefined) continue;
      let dirents: fs.Dirent[];
      try {
        dirents = fs.readdirSync(dirpath, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const dirent of dirents) {
        const full = path.join(dirpath, dirent.name);
        if (isDir(full)) {
          stack.push(full);
          continue;
        }
        if (Date.now() > deadline) {
          truncated = true;
          break walk;
        }
        if (globRe && !globRe.test(dirent.name)) continue;
        // A symlinked file inside a readable root can point outside the
        // sandbox; resolve before reading follows the link.
        if (!isWithinReadableRoot(full)) continue;
        if (
          grepFileStreaming(full, pattern, matches, deadline) === "truncated"
        ) {
          truncated = true;
          break walk;
        }
      }
    }
    return { matches, truncated };
  } catch (e) {
    return errResult(e);
  }
}

/**
 * Grep a single file for a literal substring, reading it in bounded chunks so
 * a large file is never loaded fully into memory — parity with Python's
 * line-by-line `for line in fh`. Hits are appended to *matches*; returns
 * `"truncated"` when the deadline or MAX_GREP_MATCHES cap is reached, else
 * `"ok"`. Binary files (first NUL byte) and unreadable files are skipped.
 * A `StringDecoder` reassembles multi-byte UTF-8 sequences split across chunks.
 */
function grepFileStreaming(
  full: string,
  pattern: string,
  matches: Array<{ path: string; line: number; text: string }>,
  deadline: number,
): "ok" | "truncated" {
  let fd: number;
  try {
    fd = fs.openSync(full, "r");
  } catch {
    return "ok";
  }
  const CHUNK_BYTES = 65536;
  const buf = Buffer.allocUnsafe(CHUNK_BYTES);
  const decoder = new StringDecoder("utf-8");
  let leftover = "";
  let lineNo = 0;
  const check = (text: string): "ok" | "truncated" | "no" => {
    lineNo += 1;
    if (Date.now() > deadline) return "truncated";
    if (text.includes(pattern)) {
      matches.push({ path: full, line: lineNo, text });
      if (matches.length >= MAX_GREP_MATCHES) return "truncated";
    }
    return "no";
  };
  try {
    for (;;) {
      const bytesRead = fs.readSync(fd, buf, 0, CHUNK_BYTES, null);
      if (bytesRead === 0) break;
      const chunk = buf.subarray(0, bytesRead);
      if (chunk.includes(0)) return "ok"; // binary file — skip
      leftover += decoder.write(chunk);
      let nl: number;
      while ((nl = leftover.indexOf("\n")) !== -1) {
        const line = leftover.slice(0, nl);
        leftover = leftover.slice(nl + 1);
        if (check(line) === "truncated") return "truncated";
      }
    }
    leftover += decoder.end();
    if (leftover.length > 0 && check(leftover) === "truncated") {
      return "truncated";
    }
    return "ok";
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Download a file's raw bytes base64-encoded so the JSON transport can carry
 * arbitrary binary content. `AgentEngineToolPodBackend.downloadFiles` base64-decodes
 * this to produce the response bytes.
 */
export function filesystemDownload(
  args: Record<string, unknown>,
): Record<string, unknown> {
  try {
    const filePath = String(args["file_path"] ?? "");
    const resolved = resolvePath(filePath, { allowReadonlyRoots: true });
    const size = fs.statSync(resolved).size;
    if (size > DOWNLOAD_MAX_BYTES) {
      return {
        error: `File exceeds DOWNLOAD_MAX_BYTES (${size} > ${DOWNLOAD_MAX_BYTES})`,
      };
    }
    const raw = fs.readFileSync(resolved);
    return {
      path: resolved,
      content_base64: raw.toString("base64"),
      encoding: "base64",
    };
  } catch (e) {
    return errResult(e);
  }
}

// ---------------------------------------------------------------------------
// Shell handler
// ---------------------------------------------------------------------------

function truncateOutput(text: string): [string, boolean] {
  const encoded = Buffer.from(text, "utf-8");
  if (encoded.length <= SHELL_OUTPUT_MAX_BYTES) return [text, false];
  return [encoded.subarray(0, SHELL_OUTPUT_MAX_BYTES).toString("utf-8"), true];
}

/**
 * The minimal env the shell child inherits. The Tool-Pod container currently
 * mounts tenant secrets via `envFrom.secretRef`; without scrubbing, a
 * prompt-injected agent running `shell_execute("env")` would exfiltrate them
 * (OWASP LLM01). Until the pod-level mount is removed, reconstruct the
 * child env from an explicit allowlist.
 *
 * Scope limit: this scrubs only the child's *own* environment. The child runs
 * as the same UID as the tool server (PID 1), so it can still read the
 * parent's full environment via `/proc/1/environ` regardless of whether the
 * image runs as root or a dedicated user. The scrub blunts casual
 * exfiltration; it is not an OS-enforced secret boundary.
 */
function buildShellEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const key of SHELL_ENV_ALLOWLIST) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

/**
 * Run a shell command and capture its output. Per
 * `SandboxBackendProtocol.execute`, *timeout* is `null` = "use the backend
 * default" (SHELL_DEFAULT_TIMEOUT_SECONDS), not "no timeout".
 *
 * Threat model: `shell: true` is intentional — agents need pipes, redirects,
 * globs. The sandbox is the Tool Pod itself.
 *
 * Per-session isolation gap: unlike the filesystem handlers (which enforce the
 * workspace boundary via `resolvePath()` + `isWithinWorkspace()` realpath
 * checks), `cwd` here is only the spawned shell's default working directory —
 * the process is not OS-confined, so `..`/absolute paths can reach sibling
 * sessions or arbitrary pod paths. Treat as session-shared until the
 * bubblewrap sandbox lands. There is no upstream approval gate on the command
 * string; treat the content as agent-authored. This means shell_execute does
 * NOT match the per-session filesystem isolation the fs handlers enforce —
 * an accepted, tracked risk, not local path validation.
 *
 * Output handling: stdout/stderr are drained as they stream. Each stream stops
 * *appending* once SHELL_OUTPUT_MAX_BYTES is captured but the stream is never
 * paused, so the child never blocks on a full pipe — a runaway `yes` runs to
 * its timeout without OOMing the pod. Partial output is preserved on timeout.
 * Framework errors (e.g. missing `/bin/sh`) return
 * `exit_code = EXIT_CODE_FRAMEWORK_ERROR`, distinct from the timeout sentinel.
 */
export async function shellExecute(
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const command = String(args["command"] ?? "");
  const rawTimeout = args["timeout"];
  const effectiveTimeout =
    rawTimeout == null ? SHELL_DEFAULT_TIMEOUT_SECONDS : Number(rawTimeout);

  let cwd: string;
  try {
    cwd = effectiveWorkspaceDir();
    fs.mkdirSync(cwd, { recursive: true });
  } catch (e) {
    return {
      output: "",
      exit_code: EXIT_CODE_FRAMEWORK_ERROR,
      truncated: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }

  return new Promise((resolve) => {
    const cap = SHELL_OUTPUT_MAX_BYTES;
    const stdoutBuf: Buffer[] = [];
    const stderrBuf: Buffer[] = [];
    let stdoutCaptured = 0;
    let stderrCaptured = 0;
    let overflow = false;
    let settled = false;

    const capture = (
      buf: Buffer[],
      captured: number,
      chunk: Buffer,
    ): number => {
      // Keep consuming past the cap (drop the excess) so the child never blocks
      // on a full pipe; flag overflow so the result is marked truncated.
      if (captured < cap) {
        const take = Math.min(chunk.length, cap - captured);
        buf.push(chunk.subarray(0, take));
        if (chunk.length > take) overflow = true;
        return captured + take;
      }
      overflow = true;
      return captured;
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(command, {
        shell: true,
        cwd,
        env: buildShellEnv(),
        // New process group so a timeout can SIGKILL the whole tree (the shell
        // *and* its children, e.g. a `sleep`). Killing only the shell leaves
        // grandchildren holding the stdout pipe open, delaying `close`.
        detached: true,
      });
    } catch (e) {
      resolve({
        output: "",
        exit_code: EXIT_CODE_FRAMEWORK_ERROR,
        truncated: false,
        error: e instanceof Error ? e.message : String(e),
      });
      return;
    }

    // Kill the child's whole process group (negative pid); fall back to the
    // child alone if the group is already gone (ESRCH).
    const killTree = (signal: NodeJS.Signals): void => {
      try {
        if (child.pid !== undefined) process.kill(-child.pid, signal);
        else child.kill(signal);
      } catch {
        try {
          child.kill(signal);
        } catch {
          /* already reaped */
        }
      }
    };

    let timeoutFired = false;
    let killTimer: NodeJS.Timeout | null = null;
    const timer = setTimeout(() => {
      timeoutFired = true;
      killTree("SIGKILL");
      // Guard against a child wedged in uninterruptible sleep never emitting
      // `close`: settle after the cleanup budget. `code=null, signal=SIGKILL`
      // marks it as our kill so `settle` reports a timeout.
      killTimer = setTimeout(
        () => settle(null, "SIGKILL"),
        SHELL_CLEANUP_TIMEOUT_MS,
      );
    }, effectiveTimeout * 1000);

    // `code`/`signal` come from the `close` event. A timeout is only real when
    // we fired the timer AND the process was actually terminated by our
    // SIGKILL (signal-killed → `code === null`). If the command exits on its
    // own right at the deadline, `close` still delivers a numeric exit code, so
    // we report that instead of a spurious timeout — closing the timer-vs-close
    // race the old time-based `timedOut` flag lost (Node runs the timers phase
    // before the poll phase, so the timer could win by an event-loop tick).
    const settle = (
      code: number | null,
      signal: NodeJS.Signals | null,
    ): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);

      const wasTimeout = timeoutFired && code === null && signal === "SIGKILL";

      const stdoutStr = Buffer.concat(stdoutBuf).toString("utf-8");
      const stderrStr = Buffer.concat(stderrBuf).toString("utf-8");
      let combined = stdoutStr + stderrStr;
      if (wasTimeout) {
        combined = `Command timed out after ${effectiveTimeout}s\n` + combined;
      }
      const [output, postTruncated] = truncateOutput(combined);
      resolve({
        output,
        exit_code: wasTimeout
          ? EXIT_CODE_TIMEOUT
          : (code ?? EXIT_CODE_FRAMEWORK_ERROR),
        truncated: overflow || postTruncated,
      });
    };

    child.stdout?.on("data", (chunk: Buffer) => {
      stdoutCaptured = capture(stdoutBuf, stdoutCaptured, chunk);
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderrCaptured = capture(stderrBuf, stderrCaptured, chunk);
    });
    child.on("error", (e) => {
      // Framework-level spawn failure (e.g. /bin/sh missing).
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (killTimer) clearTimeout(killTimer);
      resolve({
        output: "",
        exit_code: EXIT_CODE_FRAMEWORK_ERROR,
        truncated: false,
        error: e instanceof Error ? e.message : String(e),
      });
    });
    // `close` (not `exit`) fires after stdio streams are fully drained, so all
    // buffered output is captured before we settle. `signal` lets `settle`
    // distinguish our timeout SIGKILL from a clean exit at the deadline.
    child.on("close", (code, signal) => settle(code, signal));
  });
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const HANDLERS: Record<string, ServerToolFn> = {
  filesystem_ls: filesystemLs,
  filesystem_read: filesystemRead,
  filesystem_write: filesystemWrite,
  filesystem_edit: filesystemEdit,
  filesystem_glob: filesystemGlob,
  filesystem_grep: filesystemGrep,
  filesystem_download: filesystemDownload,
  shell_execute: shellExecute,
};

// Invariant: BUILTIN_TOOL_NAMES (the public contract) must match the actual
// handler set. Checked at import so drift fails fast.
{
  const declared = [...BUILTIN_TOOL_NAMES].sort();
  const registered = Object.keys(HANDLERS).sort();
  if (JSON.stringify(declared) !== JSON.stringify(registered)) {
    throw new Error(
      `BUILTIN_TOOL_NAMES drift: declared=${JSON.stringify(declared)} vs registered=${JSON.stringify(registered)}`,
    );
  }
}

/**
 * Fail fast at startup when an explicit AGENTIC_SKILLS_DIR resolves to no
 * usable skills root. A default "skills" dir that doesn't exist is not an
 * error (agents may bundle no skills); one that exists but isn't a readable
 * directory was previously a silent no-op, so it only warns for one release
 * to avoid crash-looping already-deployed agents — upgrade to a throw after.
 */
function validateReadonlyResourceRootsAtStartup(): void {
  const roots = getReadonlyResourceRoots();
  const explicit = process.env["AGENTIC_SKILLS_DIR"];
  if (explicit && roots.length === 0) {
    throw new Error(
      `AGENTIC_SKILLS_DIR=${JSON.stringify(explicit)} was set but no readable skills root ` +
        `could be resolved. Ensure AGENTIC_AGENT_CONFIG_PATH or AGENTIC_AGENT_WORKDIR is set ` +
        `before the Tool Pod starts and that the directory exists under the agent source root.`,
    );
  }
  // The resolver filters unusable roots, so re-derive the candidate: a
  // default skills dir that exists but is not a readable directory is a
  // misconfiguration, not "no skills bundled".
  const candidate = readonlySkillsRootFromEnv();
  if (
    !explicit &&
    candidate !== null &&
    roots.length === 0 &&
    fs.existsSync(candidate)
  ) {
    logger.warn(
      `Configured read-only skills root ${JSON.stringify(candidate)} is not a ` +
        `readable directory; bundled skills will be unavailable. Check the ` +
        `bundled skills path and AGENTIC_SKILLS_DIR.`,
    );
  }
}

/**
 * Register all built-in tool handlers on *runtime* (callable lookup +
 * metadata for all 8 handlers). Called by `ToolServer.onStartup` when
 * `features.deep_agent` is on.
 *
 * If a user `@app.tool()` registered a tool with a reserved built-in name,
 * this logs a WARNING and overrides it with the built-in — without the warning
 * the collision was silent and surfaced only when the built-in was invoked.
 */
export function registerBuiltinTools(runtime: ITenantRuntime): void {
  // Warn before side effects if WORKSPACE_DIR is outside the known writable
  // mounts. On production Tool Pods only /tmp and /scratch are RAM-backed;
  // any other path will likely fail or write to a read-only layer. Warning
  // rather than hard error so local dev with non-standard mounts still starts.
  const inKnownMount =
    KNOWN_WORKSPACE_PREFIXES.includes(WORKSPACE_DIR) ||
    KNOWN_WORKSPACE_PREFIXES.some((p) =>
      WORKSPACE_DIR.startsWith(p + path.sep),
    );
  if (!inKnownMount) {
    logger.warn(
      `WORKSPACE_DIR=${JSON.stringify(WORKSPACE_DIR)} is outside the known writable mounts (/tmp, /scratch). ` +
        "On production Tool Pods only /tmp and /scratch are writable; " +
        "filesystem and shell handlers may fail at runtime.",
    );
  }

  fs.mkdirSync(WORKSPACE_DIR, { recursive: true });
  logger.info(`Tool Pod workspace: ${WORKSPACE_DIR}`);

  validateReadonlyResourceRootsAtStartup();

  for (const [name, func] of Object.entries(HANDLERS)) {
    if (runtime.tools[name] !== undefined && runtime.tools[name] !== func) {
      logger.warn(
        `registerBuiltinTools: overriding user-registered tool ${JSON.stringify(name)} ` +
          "with built-in handler. Rename the @app.tool() or remove it to avoid " +
          "surprising runtime behavior.",
      );
    }
    runtime.tools[name] = func;
    runtime.toolDefinitions[name] = {
      name,
      description: "",
      is_local: false,
      network: [],
      timeout_seconds: 30,
      redact_fields: [],
    };
  }
}
