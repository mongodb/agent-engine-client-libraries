/**
 * AgentEngineToolPodBackend — routes deep-agent filesystem/shell ops through the
 * secure path (SecureToolWrapper → Orchestration Engine → Tool Pod handlers).
 *
 * Port of Python `agent_engine_sdk_langgraph/backends/toolpod.py`, targeting
 * deepagents' current V2 backend protocol (`SandboxBackendProtocolV2`): every
 * method returns a structured Result object with an optional `error`, so a
 * failure is surfaced through the same channel as success — never thrown into
 * the graph and never leaking internal topology.
 *
 * The active `SecureToolWrapper` is read from the async-context store on every
 * call (not captured at construction): LangGraph builds the backend on one task
 * and invokes it on another, so a captured reference would be stale.
 *
 * Batch downloads (`downloadFiles`) fan out concurrently in native mode, with
 * one `filesystem_download` call per file so OE policy and audit stay
 * independent. Under a durable `AttemptContext` they stay sequential: unkeyed
 * activities are exclusive on the attempt gate, and overlapping them raises
 * CONFLICT.
 */

import type {
  EditResult,
  ExecuteResponse,
  FileData,
  FileDownloadResponse,
  FileInfo,
  FileUploadResponse,
  GlobResult,
  GrepMatch,
  GrepResult,
  LsResult,
  ReadRawResult,
  ReadResult,
  SandboxBackendProtocolV2,
  WriteResult,
} from "deepagents";
import {
  currentAttemptContext,
  getCurrentWrapper,
  getLogger,
  PolicyDeniedException,
  TerminalExecutionError,
  ToolCallTimeoutError,
  type SecureToolWrapper,
} from "@mongodb-js/agent-engine-runner-shared";

const logger = getLogger("agent_engine_sdk_langgraph.backends.toolpod");

const RETRYABLE = "[RETRYABLE]";
const NON_RETRYABLE = "[NON-RETRYABLE]";
const MAX_DOWNLOAD_WORKERS = 16;

// Cluster-internal DNS suffixes (Kubernetes, consul, .local). Catches things
// like `db.foo.svc.cluster.local` before the host:port pattern below does.
const INTERNAL_DNS_RE =
  /\b[a-z0-9][a-z0-9.-]*\.(?:svc\.cluster\.local|cluster\.local|internal|local)\b/gi;
// Bracketed IPv6 host:port (`[::1]:8080`, `[fd00::1]:27017`). Runs before the
// alphanumeric host:port regex — that one's leading `[a-z0-9]` class can't
// match `[`.
const IPV6_HOST_PORT_RE = /\[[0-9a-fA-F:]+\]:\d{2,5}/g;
// `host:port` where port is 2–5 digits. Runs after the DNS replacement so
// `db.internal:27017` loses its port here rather than being orphaned.
const HOST_PORT_RE = /\b[a-z0-9][a-z0-9.-]*:\d{2,5}\b/gi;
// Any remaining multi-segment absolute path.
const ABS_PATH_RE = /\/[^/\s"'()]+(?:\/[^/\s"'()]*)*/g;

/**
 * Strip internal topology (workspace path, cluster hostnames/ports, absolute
 * paths) from a handler-supplied error string before it reaches the agent
 * graph. Mirrors Python `_sanitize_handler_error`: the "no internal
 * hostnames/ports/paths" invariant applies to handler errors, not only
 * exception-derived ones. Order matters — host:port pairs are consumed before
 * bare DNS names so a port is never left orphaned.
 */
function sanitizeHandlerError(message: string): string {
  let out = message;
  const workspace = process.env["WORKSPACE_DIR"]?.replace(/\/+$/, "");
  if (workspace) out = out.split(workspace).join("<workspace>");
  out = out.replace(IPV6_HOST_PORT_RE, "<host>");
  out = out.replace(HOST_PORT_RE, "<host>");
  out = out.replace(INTERNAL_DNS_RE, "<host>");
  return out.replace(ABS_PATH_RE, "<path>");
}

/**
 * Classify a thrown error as retryable or non-retryable and return a generic,
 * topology-free description. Full details are logged by the caller; only this
 * category string reaches the agent. Mirrors Python `_classify_error`.
 *
 * PolicyDenied and programming errors (TypeError/RangeError etc., and the
 * "no wrapper in context" RuntimeError) are non-retryable — retrying with the
 * same inputs can never succeed. Everything else defaults to retryable so the
 * agent doesn't give up on a transient.
 */
function classifyError(exc: unknown): string {
  if (exc instanceof PolicyDeniedException) {
    return `${NON_RETRYABLE} Policy denied this operation`;
  }
  if (exc instanceof TerminalExecutionError) {
    return `${NON_RETRYABLE} This execution already ended`;
  }
  if (exc instanceof ToolCallTimeoutError) {
    // Non-retryable: the call already consumed its whole deadline, so an
    // automatic retry spends another full one to most likely fail the same way.
    // The agent decides whether the work is worth re-attempting.
    return `${NON_RETRYABLE} Operation timed out`;
  }
  if (
    exc instanceof TypeError ||
    exc instanceof RangeError ||
    exc instanceof ReferenceError ||
    exc instanceof SyntaxError ||
    exc instanceof MissingWrapperError
  ) {
    return `${NON_RETRYABLE} Programming error — retry will not help`;
  }
  return `${RETRYABLE} Operation failed`;
}

/** Raised when the backend is used outside an AER execution context. */
class MissingWrapperError extends Error {}

function getWrapper(): SecureToolWrapper {
  const wrapper = getCurrentWrapper() as SecureToolWrapper | null;
  if (wrapper == null) {
    throw new MissingWrapperError(
      "No SecureToolWrapper in context — " +
        "AgentEngineToolPodBackend must run inside AER execution",
    );
  }
  return wrapper;
}

/** A handler wire result is always a JSON object. */
type WireResult = Record<string, unknown>;

/**
 * Result of a wire call — a discriminated union so callers narrow `result`
 * without a non-null assertion (`@typescript-eslint/no-non-null-assertion`).
 */
type CallOutcome =
  | { ok: true; result: WireResult }
  | { ok: false; error: string };

/** Truthy `error` field on a handler result, sanitized; else undefined. */
function handlerError(result: WireResult): string | undefined {
  const err = result["error"];
  return err ? sanitizeHandlerError(String(err)) : undefined;
}

function isObject(v: unknown): v is WireResult {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Human-readable type for protocol-violation messages (`typeof [] === "object"`). */
function describeType(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}

/**
 * Backend that routes every operation through Atlas Agent Engine's secure path. Each
 * method delegates to `SecureToolWrapper.executeTool`, which sends the request
 * to the OE for policy approval and audit logging before it runs in the Tool
 * Pod.
 */
export class AgentEngineToolPodBackend implements SandboxBackendProtocolV2 {
  readonly id = "agent-engine-toolpod";

  /**
   * Execute a Tool Pod handler through the secure wrapper. Returns the wire
   * result on success, or a classified `[RETRYABLE]`/`[NON-RETRYABLE]` error
   * string on failure (exceptions are logged and translated, never rethrown).
   */
  private async call(
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<CallOutcome> {
    try {
      const wrapper = getWrapper();
      const raw = await wrapper.executeTool(toolName, args, { isLocal: false });
      if (isObject(raw)) return { ok: true, result: raw };
      return {
        ok: false,
        error:
          `${NON_RETRYABLE} ${toolName} protocol violation: ` +
          `expected object, got ${describeType(raw)}`,
      };
    } catch (exc) {
      const msg = classifyError(exc);
      logger.warn(`Tool call ${toolName} failed: ${String(exc)}`);
      return { ok: false, error: msg };
    }
  }

  // ------------------------------------------------------------------
  // Read operations
  // ------------------------------------------------------------------

  async ls(path: string): Promise<LsResult> {
    const outcome = await this.call("filesystem_ls", { path });
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const entries = result["entries"];
    if (!Array.isArray(entries)) {
      return {
        error: `${NON_RETRYABLE} filesystem_ls protocol violation: 'entries' not an array`,
      };
    }
    return mapFileInfos(entries, "filesystem_ls", "entry");
  }

  async read(filePath: string, offset = 0, limit = 2000): Promise<ReadResult> {
    const outcome = await this.call("filesystem_read", {
      file_path: filePath,
      offset,
      limit,
    });
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const content = result["content"];
    if (typeof content !== "string") {
      return {
        error: `${NON_RETRYABLE} filesystem_read protocol violation: 'content' not a string`,
      };
    }
    return { content, mimeType: "text/plain" };
  }

  /**
   * Raw read is not backed by a dedicated wire handler; the Tool Pod exposes
   * only line-oriented `filesystem_read`. We wrap its text content in a v2
   * FileData so binary-unaware callers still work. Genuine binary reads should
   * use `downloadFiles`, which carries raw bytes base64-encoded.
   */
  async readRaw(filePath: string): Promise<ReadRawResult> {
    const res = await this.read(filePath);
    if (res.error !== undefined) return { error: res.error };
    const now = new Date().toISOString();
    const data: FileData = {
      content: res.content ?? "",
      mimeType: res.mimeType ?? "text/plain",
      created_at: now,
      modified_at: now,
    };
    return { data };
  }

  async grep(
    pattern: string,
    path?: string | null,
    glob?: string | null,
    maxCount?: number | null,
  ): Promise<GrepResult> {
    const args: Record<string, unknown> = { pattern };
    if (path != null) args["path"] = path;
    if (glob != null) args["glob"] = glob;

    const outcome = await this.call("filesystem_grep", args);
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const raw = result["matches"];
    if (!Array.isArray(raw)) {
      return {
        error: `${NON_RETRYABLE} filesystem_grep protocol violation: 'matches' not an array`,
      };
    }
    let matches: GrepMatch[] = [];
    for (const m of raw) {
      // Validate each element rather than coercing — an unchecked
      // `Number(o["line"])` / `String(o["path"])` would silently yield `NaN` /
      // `"undefined"` on a malformed element instead of surfacing the
      // protocol violation, matching the array-shape checks above.
      if (
        !isObject(m) ||
        typeof m["path"] !== "string" ||
        typeof m["line"] !== "number" ||
        typeof m["text"] !== "string"
      ) {
        return {
          error: `${NON_RETRYABLE} filesystem_grep protocol violation: malformed match (expected {path: string, line: number, text: string})`,
        };
      }
      matches.push({ path: m["path"], line: m["line"], text: m["text"] });
    }
    // The handler caps its own walk and reports truncated; maxCount is the
    // protocol's caller-side total cap, enforced after the call.
    let truncated = result["truncated"] === true;
    if (maxCount != null && matches.length > maxCount) {
      matches = matches.slice(0, maxCount);
      truncated = true;
    }
    return { matches, truncated };
  }

  async glob(pattern: string, path?: string): Promise<GlobResult> {
    const args: Record<string, unknown> = { pattern };
    if (path !== undefined) args["path"] = path;
    const outcome = await this.call("filesystem_glob", args);
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const raw = result["matches"];
    if (!Array.isArray(raw)) {
      return {
        error: `${NON_RETRYABLE} filesystem_glob protocol violation: 'matches' not an array`,
      };
    }
    const mapped = mapFileInfos(raw, "filesystem_glob", "match");
    if ("error" in mapped) return mapped;
    return { files: mapped.files, truncated: result["truncated"] === true };
  }

  // ------------------------------------------------------------------
  // Write operations
  // ------------------------------------------------------------------

  /**
   * Create a new file. The Tool Pod handler is create-only — it opens the path
   * with `wx` and returns an already-exists error rather than overwriting — so
   * use {@link edit} to modify an existing file.
   */
  async write(filePath: string, content: string): Promise<WriteResult> {
    const outcome = await this.call("filesystem_write", {
      file_path: filePath,
      content,
    });
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const path = result["path"];
    if (typeof path !== "string") {
      return {
        error: `${NON_RETRYABLE} filesystem_write protocol violation: success response missing 'path' string`,
      };
    }
    return { path };
  }

  async edit(
    filePath: string,
    oldString: string,
    newString: string,
    replaceAll = false,
  ): Promise<EditResult> {
    const outcome = await this.call("filesystem_edit", {
      file_path: filePath,
      old_string: oldString,
      new_string: newString,
      replace_all: replaceAll,
    });
    if (!outcome.ok) return { error: outcome.error };
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) return { error: handlerErr };

    const occurrences = result["occurrences"];
    if (typeof occurrences !== "number") {
      return {
        error: `${NON_RETRYABLE} filesystem_edit protocol violation: success response missing 'occurrences' int`,
      };
    }
    return { path: filePath, occurrences };
  }

  // ------------------------------------------------------------------
  // Shell execution
  // ------------------------------------------------------------------

  async execute(command: string): Promise<ExecuteResponse> {
    const outcome = await this.call("shell_execute", { command });
    if (!outcome.ok) {
      return { output: outcome.error, exitCode: 1, truncated: false };
    }
    const { result } = outcome;
    // A framework error (e.g. missing /bin/sh) returns an `error` field
    // alongside a non-zero exit_code and empty output. Surface the sanitized
    // message — otherwise the agent sees only an empty output + exit code and
    // loses the cause — and preserve the handler's exit code.
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) {
      const code = result["exit_code"];
      return {
        output: handlerErr,
        exitCode: typeof code === "number" ? code : 1,
        truncated: false,
      };
    }
    const output = result["output"];
    const exitCode = result["exit_code"];
    if (typeof output !== "string" || typeof exitCode !== "number") {
      return {
        output: `${NON_RETRYABLE} shell_execute protocol violation: response must contain str 'output' and int 'exit_code'`,
        exitCode: 1,
        truncated: false,
      };
    }
    return {
      output,
      exitCode,
      truncated: Boolean(result["truncated"]),
    };
  }

  // ------------------------------------------------------------------
  // Download / upload
  // ------------------------------------------------------------------

  /**
   * Download a single path. Never throws: failures become error responses.
   */
  private async downloadOne(path: string): Promise<FileDownloadResponse> {
    const outcome = await this.call("filesystem_download", {
      file_path: path,
    });
    if (!outcome.ok) {
      return downloadError(path, outcome.error);
    }
    const { result } = outcome;
    const handlerErr = handlerError(result);
    if (handlerErr !== undefined) {
      return downloadError(path, handlerErr);
    }
    const b64 = result["content_base64"];
    if (typeof b64 !== "string") {
      return downloadError(
        path,
        `${NON_RETRYABLE} filesystem_download protocol violation: success response missing 'content_base64' field`,
      );
    }
    return {
      path,
      content: new Uint8Array(Buffer.from(b64, "base64")),
      error: null,
    };
  }

  /**
   * Download files via the audited `filesystem_download` handler — one call per
   * path so each is policy-checked and logged independently — then base64-decode
   * the `content_base64` field into raw bytes. Responses preserve input order;
   * on failure `content` is null and `error` carries a classified message.
   *
   * Native-mode batches fan out on a bounded worker pool so skills loading
   * pays one round-trip of latency instead of a sum. Durable attempts stay
   * sequential: unkeyed activities are exclusive under the attempt gate, and
   * overlapping them raises CONFLICT.
   */
  async downloadFiles(paths: string[]): Promise<FileDownloadResponse[]> {
    if (paths.length <= 1 || currentAttemptContext() != null) {
      // Single path skips pool overhead. Durable attempts stay sequential:
      // unkeyed activities are exclusive under the attempt gate, and
      // overlapping them raises CONFLICT.
      const responses: FileDownloadResponse[] = [];
      for (const path of paths) {
        responses.push(await this.downloadOne(path));
      }
      return responses;
    }
    return mapLimited(paths, MAX_DOWNLOAD_WORKERS, (path) =>
      this.downloadOne(path),
    );
  }

  async uploadFiles(
    files: Array<[string, Uint8Array]>,
  ): Promise<FileUploadResponse[]> {
    // Return per-file classified errors rather than throwing — a raw throw
    // would bypass the [RETRYABLE]/[NON-RETRYABLE] contract every other method
    // honours. No wire handler exists for upload yet.
    const err = `${NON_RETRYABLE} uploadFiles is not implemented`;
    return files.map(([path]) => uploadError(path, err));
  }
}

/**
 * Map raw wire entries to `FileInfo[]`, validating each element. A malformed
 * element (not an object, or a non-string `path`) fails the whole result with a
 * protocol-violation error rather than coercing to `String(undefined)` — the
 * same all-or-nothing contract the array-shape checks in `ls`/`glob` use.
 * `handler`/`elem` name the source for the error message.
 */
function mapFileInfos(
  entries: unknown[],
  handler: string,
  elem: string,
): { files: FileInfo[] } | { error: string } {
  const files: FileInfo[] = [];
  for (const e of entries) {
    if (!isObject(e) || typeof e["path"] !== "string") {
      return {
        error: `${NON_RETRYABLE} ${handler} protocol violation: malformed ${elem} (expected object with string 'path')`,
      };
    }
    files.push({ path: e["path"], is_dir: Boolean(e["is_dir"]) });
  }
  return { files };
}

// `FileDownloadResponse.error` / `FileUploadResponse.error` are typed as a
// `FileOperationError` enum in the deepagents protocol. We deliberately ship
// richer strings — sanitized handler messages and [RETRYABLE]/[NON-RETRYABLE]
// prefixes — so the agent can tell transients from contract bugs. The cast is
// parked once on each helper instead of at every call site.
function downloadError(path: string, err: string): FileDownloadResponse {
  return { path, content: null, error: err as never };
}

/**
 * Run `fn` over `items` with at most `limit` in-flight promises. Result[i]
 * corresponds to items[i] even when completions finish out of order.
 */
async function mapLimited<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array<R>(items.length);
  let cursor = 0;
  async function worker(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      const item = items[index];
      if (item === undefined) return;
      results[index] = await fn(item);
    }
  }
  const workers = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: workers }, () => worker()));
  return results;
}

function uploadError(path: string, err: string): FileUploadResponse {
  return { path, error: err as never };
}
