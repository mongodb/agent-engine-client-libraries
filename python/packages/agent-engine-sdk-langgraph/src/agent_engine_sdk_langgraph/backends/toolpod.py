"""AgentEngineToolPodBackend — routes deep agent operations through SecureToolWrapper.

All filesystem and shell operations are forwarded to the Orchestration Engine
for approval and logging, then executed in the Tool Pod.  The backend reads the
active ``SecureToolWrapper`` from a ``ContextVar`` on every call so it works
correctly inside LangGraph's async execution model where construction and
invocation may happen on different tasks.

Batch downloads (``download_files`` / ``adownload_files``) fan out concurrently
in native mode, with one ``filesystem_download`` call per file so OE policy
and audit stay independent. Under a durable ``AttemptContext`` they stay
sequential: unkeyed activities are exclusive on the attempt gate, and
overlapping them raises CONFLICT.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from typing import Any

from deepagents.backends.protocol import (
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
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)

from agent_engine_runner_shared.context import current_wrapper
from agent_engine_runner_shared.secure_wrapper import (
    PolicyDeniedException,
    TerminalExecutionError,
    ToolCallTimeoutError,
)
from agent_engine_runner_shared.workflow import current_attempt_context

logger = logging.getLogger(__name__)

_RETRYABLE = "[RETRYABLE]"
_NON_RETRYABLE = "[NON-RETRYABLE]"
_MAX_DOWNLOAD_WORKERS = 16


# Exceptions that indicate a programming bug or contract mismatch — retrying
# them with the same inputs will never succeed, so classify as non-retryable
# to surface the real defect instead of burning tokens on loops. Protocol
# violations are not raised as exceptions in this module — the read/write/exec
# methods return result dataclasses with ``[NON-RETRYABLE] ... protocol
# violation`` error strings instead, so there is no dedicated BackendProtocolError.
_NON_RETRYABLE_EXCS: tuple[type[BaseException], ...] = (
    ValueError,
    KeyError,
    AttributeError,
    TypeError,
    NotImplementedError,
    json.JSONDecodeError,
    # RuntimeError is raised by ``_get_wrapper`` when the backend is used
    # outside an AER execution context — a structural programming error
    # that will never succeed on retry. Without this entry the default
    # ``[RETRYABLE] Operation failed`` classification would put the agent
    # in an infinite retry loop on misconfiguration.
    RuntimeError,
)


# Patterns used by ``_sanitize_handler_error`` to strip internal topology
# (filesystem paths, hostnames, ports) out of handler-returned error strings
# before surfacing them to the agent graph. ``_classify_error``'s invariant —
# "Internal hostnames, ports, and paths MUST NOT appear" — applies to handler
# errors too, not only exception-derived strings.
_ABS_PATH_RE = re.compile(r"/[^/\s\"'()]+(?:/[^/\s\"'()]*)*")
# Cluster-internal DNS suffixes (Kubernetes, consul, .local). Catches things
# like ``db.foo.svc.cluster.local`` before the host:port pattern below does.
_INTERNAL_DNS_RE = re.compile(
    r"\b[a-z0-9][a-z0-9.-]*\.(?:svc\.cluster\.local|cluster\.local|internal|local)\b",
    re.IGNORECASE,
)
# Bracketed IPv6 host:port (``[::1]:8080``, ``[fd00::1]:27017``). Runs
# before the alphanumeric host:port regex below — that one's word-boundary
# anchor and ``[a-z0-9]`` leading class don't match ``[``.
_IPV6_HOST_PORT_RE = re.compile(r"\[[0-9a-fA-F:]+\]:\d{2,5}")
# ``host:port`` where port is 2–5 digits. Runs after the DNS replacement so
# ``db.internal:27017`` becomes ``<host>`` first, then any bare IPv4:port or
# short-hostname:port left in the string is caught here.
_HOST_PORT_RE = re.compile(r"\b[a-z0-9][a-z0-9.-]*:\d{2,5}\b", re.IGNORECASE)


def _sanitize_handler_error(message: str) -> str:
    """Strip internal topology from a handler-supplied error string.

    Tool-Pod handlers frequently surface ``str(OSError)`` / ``str(ConnectionError)``
    which embed the resolved workspace path or downstream service coordinates
    (``db.internal.svc.cluster.local:27017``). Surfacing those to the agent
    leaks internal layout — the "no internal hostnames/ports/paths" invariant
    ``_classify_error`` enforces for exception-based failures must apply here
    too.

    We rewrite five patterns in order (IPv6 and IPv4/hostname host:port
    before bare DNS so a ``hostname:port`` pair doesn't lose its port when
    the hostname gets replaced in isolation):

    * The active ``WORKSPACE_DIR`` prefix → ``<workspace>``.
    * Bracketed IPv6 ``host:port`` pairs (``[::1]:8080``,
      ``[fd00::1]:27017``) → ``<host>``. Runs first because the
      alphanumeric host:port regex below can't match ``[``.
    * ``host:port`` pairs → ``<host>``. Consumes both host and port in one
      match so the DNS pass below can't leave the port orphaned.
    * Cluster-internal DNS names (``*.svc.cluster.local``, ``*.internal``,
      ``*.local``) → ``<host>``. Catches bare hostnames not already eaten
      by the host:port pass.
    * Any remaining multi-segment absolute path → ``<path>``.

    The agent still learns *which* operation failed (message prefix) and
    *why* (category, permission/not-found/etc.), just not where on disk
    or which downstream service.
    """
    workspace = os.environ.get("WORKSPACE_DIR")
    if workspace:
        workspace = workspace.rstrip("/")
        if workspace:
            message = message.replace(workspace, "<workspace>")
    message = _IPV6_HOST_PORT_RE.sub("<host>", message)
    message = _HOST_PORT_RE.sub("<host>", message)
    message = _INTERNAL_DNS_RE.sub("<host>", message)
    return _ABS_PATH_RE.sub("<path>", message)


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Classify an exception as retryable or non-retryable.

    Returns only the error category and a generic description — never
    raw exception internals.  Internal hostnames, ports, and paths MUST
    NOT appear in strings returned to the agent graph.  Full exception
    details are logged at WARNING level by ``_safe_call``.
    """
    if isinstance(exc, PolicyDeniedException):
        return f"{_NON_RETRYABLE} Policy denied this operation", False
    if isinstance(exc, TerminalExecutionError):
        return f"{_NON_RETRYABLE} This execution already ended", False
    if isinstance(exc, ToolCallTimeoutError):
        # Non-retryable: the call already consumed its whole deadline, so an
        # automatic retry spends another full one to most likely fail the same
        # way. The agent decides whether the work is worth re-attempting.
        return f"{_NON_RETRYABLE} Operation timed out", False
    if isinstance(exc, _NON_RETRYABLE_EXCS):
        return f"{_NON_RETRYABLE} Programming error — retry will not help", False
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return f"{_RETRYABLE} Connection failed", True
    # Default to retryable so agents don't give up prematurely on transients.
    return f"{_RETRYABLE} Operation failed", True


def _get_wrapper() -> Any:
    """Return the active SecureToolWrapper or raise if absent."""
    wrapper = current_wrapper.get()
    if wrapper is None:
        raise RuntimeError(
            "No SecureToolWrapper in context — "
            "AgentEngineToolPodBackend must run inside AER execution"
        )
    return wrapper


def _build_search_result(
    result_cls: type[Any], *, matches: list[Any], truncated: bool
) -> Any:
    """Build a grep/glob result across deepagents protocol versions.

    ``truncated`` exists from deepagents 0.7; the SDK leaves the framework
    version to the agent, so an installed 0.5.x result type accepts only
    ``error``/``matches`` and must not receive the keyword.
    """
    if "truncated" in getattr(result_cls, "__dataclass_fields__", {}):
        return result_cls(matches=matches, truncated=truncated)
    return result_cls(matches=matches)


class AgentEngineToolPodBackend(SandboxBackendProtocol):
    """Backend that routes all operations through Atlas Agent Engine's secure path.

    Each protocol method delegates to ``SecureToolWrapper.execute_tool`` which
    sends the request to the Orchestration Engine for policy approval, logging,
    and routing before execution occurs in the Tool Pod.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:  # noqa: A003
        return "agent-engine-toolpod"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool via SecureToolWrapper and JSON-parse string results."""
        wrapper = _get_wrapper()
        result = wrapper.execute_tool(tool_name, arguments, is_local=False)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, ValueError):
                return result
        return result

    def _safe_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[Any, str | None]:
        """Call ``_call_tool`` and translate exceptions into classified error strings.

        Error strings are prefixed with ``[RETRYABLE]`` or ``[NON-RETRYABLE]``
        so the agent can decide whether to retry.
        """
        try:
            result = self._call_tool(tool_name, arguments)
            return result, None
        except Exception as exc:
            msg, retryable = _classify_error(exc)
            logger.warning(
                "Tool call %s failed (retryable=%s): %s", tool_name, retryable, exc
            )
            return None, msg

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        result, error = self._safe_call("filesystem_ls", {"path": path})
        if error is not None:
            return LsResult(error=error)

        # Handler returns {"error": ...} on failure (bad path, sandbox violation,
        # etc.). Surface it — empty "entries" would be indistinguishable from a
        # genuine empty directory.
        if isinstance(result, dict) and result.get("error"):
            return LsResult(error=_sanitize_handler_error(str(result["error"])))

        entries: list[Any] = []
        if isinstance(result, list):
            entries = result
        elif isinstance(result, dict) and "entries" in result:
            entries = result["entries"]
        else:
            return LsResult(
                error=f"{_NON_RETRYABLE} filesystem_ls protocol violation: "
                f"expected list or dict with 'entries', got {type(result).__name__}"
            )

        try:
            file_infos: list[FileInfo] = [
                FileInfo(path=e["path"], is_dir=e.get("is_dir", False)) for e in entries
            ]
        except (KeyError, TypeError) as exc:
            return LsResult(
                error=f"{_NON_RETRYABLE} filesystem_ls protocol violation: "
                f"entry missing required field — {exc}"
            )
        return LsResult(entries=file_infos)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        result, error = self._safe_call(
            "filesystem_read",
            {"file_path": file_path, "offset": offset, "limit": limit},
        )
        if error is not None:
            return ReadResult(error=error)

        if isinstance(result, dict):
            if "error" in result and result["error"]:
                return ReadResult(error=_sanitize_handler_error(str(result["error"])))
            if "content" in result:
                file_data = FileData(
                    content=result["content"],
                    encoding=result.get("encoding", "utf-8"),
                )
                return ReadResult(file_data=file_data)

        if isinstance(result, str):
            return ReadResult(file_data=FileData(content=result, encoding="utf-8"))

        return ReadResult(
            error=f"{_NON_RETRYABLE} filesystem_read protocol violation: "
            f"unexpected result format — got {type(result).__name__}"
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        args: dict[str, Any] = {"pattern": pattern}
        if path is not None:
            args["path"] = path
        if glob is not None:
            args["glob"] = glob

        result, error = self._safe_call("filesystem_grep", args)
        if error is not None:
            return GrepResult(error=error)

        # Handler returns {"error": ...} on failure (bad path, sandbox
        # violation). Surface it — empty "matches" would be
        # indistinguishable from "no matches".
        if isinstance(result, dict) and result.get("error"):
            return GrepResult(error=_sanitize_handler_error(str(result["error"])))

        raw_matches: list[Any] = []
        if isinstance(result, list):
            raw_matches = result
        elif isinstance(result, dict) and "matches" in result:
            raw_matches = result["matches"]
        else:
            return GrepResult(
                error=f"{_NON_RETRYABLE} filesystem_grep protocol violation: "
                f"expected list or dict with 'matches', got {type(result).__name__}"
            )

        try:
            matches: list[GrepMatch] = [
                GrepMatch(path=m["path"], line=m["line"], text=m["text"])
                for m in raw_matches
            ]
        except (KeyError, TypeError) as exc:
            return GrepResult(
                error=f"{_NON_RETRYABLE} filesystem_grep protocol violation: "
                f"match missing required field — {exc}"
            )
        # The handler caps its own walk and reports truncated; the protocol's
        # max_count is a caller-side total cap, so enforce it after the call.
        truncated = bool(result.get("truncated")) if isinstance(result, dict) else False
        if max_count is not None and len(matches) > max_count:
            matches = matches[:max_count]
            truncated = True
        return _build_search_result(GrepResult, matches=matches, truncated=truncated)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        args: dict[str, Any] = {"pattern": pattern}
        if path is not None:
            args["path"] = path
        result, error = self._safe_call("filesystem_glob", args)
        if error is not None:
            return GlobResult(error=error)

        # Handler returns {"error": ...} on failure (bad path, sandbox
        # violation). Surface it — empty "matches" would be indistinguishable
        # from "no files matched the glob".
        if isinstance(result, dict) and result.get("error"):
            return GlobResult(error=_sanitize_handler_error(str(result["error"])))

        raw_matches: list[Any] = []
        if isinstance(result, list):
            raw_matches = result
        elif isinstance(result, dict) and "matches" in result:
            raw_matches = result["matches"]
        else:
            return GlobResult(
                error=f"{_NON_RETRYABLE} filesystem_glob protocol violation: "
                f"expected list or dict with 'matches', got {type(result).__name__}"
            )

        try:
            file_infos: list[FileInfo] = [
                FileInfo(path=m["path"], is_dir=m.get("is_dir", False))
                for m in raw_matches
            ]
        except (KeyError, TypeError) as exc:
            return GlobResult(
                error=f"{_NON_RETRYABLE} filesystem_glob protocol violation: "
                f"match missing required field — {exc}"
            )
        truncated = bool(result.get("truncated")) if isinstance(result, dict) else False
        return _build_search_result(GlobResult, matches=file_infos, truncated=truncated)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        result, error = self._safe_call(
            "filesystem_write", {"file_path": file_path, "content": content}
        )
        if error is not None:
            return WriteResult(error=error)

        if not isinstance(result, dict):
            return WriteResult(
                error=f"{_NON_RETRYABLE} filesystem_write protocol violation: "
                f"expected dict, got {type(result).__name__}"
            )
        if result.get("error"):
            return WriteResult(error=_sanitize_handler_error(str(result["error"])))
        path = result.get("path")
        if not isinstance(path, str):
            return WriteResult(
                error=f"{_NON_RETRYABLE} filesystem_write protocol violation: "
                "success response missing 'path' string"
            )
        return WriteResult(path=path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        result, error = self._safe_call(
            "filesystem_edit",
            {
                "file_path": file_path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
        )
        if error is not None:
            return EditResult(error=error)

        if not isinstance(result, dict):
            return EditResult(
                error=f"{_NON_RETRYABLE} filesystem_edit protocol violation: "
                f"expected dict, got {type(result).__name__}"
            )
        if result.get("error"):
            return EditResult(error=_sanitize_handler_error(str(result["error"])))

        occurrences = result.get("occurrences", result.get("count"))
        if not isinstance(occurrences, int):
            return EditResult(
                error=f"{_NON_RETRYABLE} filesystem_edit protocol violation: "
                "success response missing 'occurrences' int"
            )
        return EditResult(path=file_path, occurrences=occurrences)

    # ------------------------------------------------------------------
    # Shell execution
    # ------------------------------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        args: dict[str, Any] = {"command": command}
        if timeout is not None:
            args["timeout"] = timeout

        result, error = self._safe_call("shell_execute", args)
        if error is not None:
            return ExecuteResponse(output=error, exit_code=1)

        if not isinstance(result, dict):
            return ExecuteResponse(
                output=f"{_NON_RETRYABLE} shell_execute protocol violation: "
                f"expected dict, got {type(result).__name__}",
                exit_code=1,
            )
        output = result.get("output")
        exit_code = result.get("exit_code")
        if not isinstance(output, str) or not isinstance(exit_code, int):
            return ExecuteResponse(
                output=f"{_NON_RETRYABLE} shell_execute protocol violation: "
                "response must contain str 'output' and int 'exit_code'",
                exit_code=1,
            )
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=bool(result.get("truncated", False)),
        )

    # ------------------------------------------------------------------
    # Download files
    # ------------------------------------------------------------------

    def _download_one(self, path: str) -> FileDownloadResponse:
        """Download a single path. Never raises: failures become error responses."""
        result, error = self._safe_call("filesystem_download", {"file_path": path})
        if error is not None:
            return _download_error(path, error)

        if isinstance(result, dict):
            if "error" in result and result["error"]:
                return _download_error(
                    path, _sanitize_handler_error(str(result["error"]))
                )

            content_b64 = result.get("content_base64")
            if content_b64 is None:
                return _download_error(
                    path,
                    f"{_NON_RETRYABLE} filesystem_download protocol violation: "
                    "success response missing 'content_base64' field",
                )

            try:
                content_bytes = base64.b64decode(content_b64, validate=True)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "filesystem_download returned invalid base64 for %s: %s",
                    path,
                    exc,
                )
                return _download_error(
                    path,
                    f"{_NON_RETRYABLE} filesystem_download: "
                    "backend returned invalid base64 content",
                )

            return FileDownloadResponse(path=path, content=content_bytes, error=None)

        return _download_error(
            path, "Unexpected result format from filesystem_download"
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files via the audited filesystem_download handler.

        Routes one ``filesystem_download`` call per path through
        ``SecureToolWrapper`` (so each download is policy-checked and
        logged independently by the OE), then base64-decodes the
        ``content_base64`` field the handler returns into raw ``bytes``.

        Native-mode batches fan out on a bounded thread pool so skills
        loading pays one round-trip of latency instead of a sum. Durable
        attempts stay sequential: unkeyed activities are exclusive under
        the attempt gate, and overlapping them raises CONFLICT.

        Returns a list of ``FileDownloadResponse`` in the same order as
        *paths*; on failure ``content`` is ``None`` and ``error`` carries
        a classified message (either the handler's ``{"error": ...}``
        string or a ``[RETRYABLE]``/``[NON-RETRYABLE]`` prefix from
        ``_classify_error``).
        """
        if len(paths) <= 1:
            return [self._download_one(p) for p in paths]
        # Unkeyed activities are exclusive under the durable attempt gate;
        # overlapping them raises CONFLICT.
        if current_attempt_context() is not None:
            return [self._download_one(p) for p in paths]

        # One Context copy per task: a Context must not be entered from
        # multiple threads at once, and ThreadPoolExecutor workers do not
        # inherit ContextVars (unlike asyncio.to_thread).
        contexts = [copy_context() for _ in paths]

        def _run(ctx: Context, path: str) -> FileDownloadResponse:
            return ctx.run(self._download_one, path)

        with ThreadPoolExecutor(
            max_workers=min(len(paths), _MAX_DOWNLOAD_WORKERS)
        ) as pool:
            return list(pool.map(_run, contexts, paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async entry point; parallelism lives in ``download_files``."""
        return await asyncio.to_thread(self.download_files, paths)

    # ------------------------------------------------------------------
    # Not yet implemented
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        # Return per-file classified errors instead of raising. A raw
        # NotImplementedError bypasses ``_safe_call`` and reaches the agent
        # graph un-prefixed, breaking the [RETRYABLE]/[NON-RETRYABLE]
        # contract every other protocol method honours.
        err = f"{_NON_RETRYABLE} upload_files is not implemented"
        return [_upload_error(path, err) for path, _ in files]


# ``FileDownloadResponse.error`` and ``FileUploadResponse.error`` are typed as
# ``Optional[Literal['file_not_found', 'permission_denied', 'is_directory',
# 'invalid_path']]`` in the deepagents protocol. We deliberately ship richer
# strings instead — handler-supplied messages, sanitized topology, and our
# ``[RETRYABLE]``/``[NON-RETRYABLE]`` prefixes — so the agent can distinguish
# transients from contract bugs. The ignore is parked once on each helper
# instead of repeated at every call site.


def _download_error(path: str, err: str) -> FileDownloadResponse:
    return FileDownloadResponse(path=path, content=None, error=err)  # type: ignore[arg-type]


def _upload_error(path: str, err: str) -> FileUploadResponse:
    return FileUploadResponse(path=path, error=err)  # type: ignore[arg-type]
