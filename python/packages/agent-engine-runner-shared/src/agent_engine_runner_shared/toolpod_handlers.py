"""Built-in Tool Pod handler functions.

These handlers execute filesystem and shell operations inside the Tool Pod.
They are registered at startup via ``register_builtin_tools`` and dispatched
by ``ToolServer._handle_execute`` when requests arrive from AgentEngineToolPodBackend
through the OE secure path.

All handlers are synchronous, return JSON-serializable dicts, and catch their
own exceptions (returning ``{"error": ...}`` on failure).

Return formats match what ``AgentEngineToolPodBackend`` expects to parse:
- filesystem_ls       -> {"entries": [{"path": str, "is_dir": bool}, ...], "truncated": bool}
- filesystem_read     -> {"content": str, "encoding": "utf-8"}
- filesystem_write    -> {"path": str}
- filesystem_edit     -> {"occurrences": int}
- filesystem_glob     -> {"matches": [{"path": str, "is_dir": bool}, ...]}
- filesystem_grep     -> {"matches": [{"path": str, "line": int, "text": str}, ...]}
- filesystem_download -> {"path": str, "content_base64": str, "encoding": "base64"}
- shell_execute       -> {"output": str, "exit_code": int, "truncated": bool}
"""

from __future__ import annotations

import base64
import fnmatch
import glob as glob_mod
import hashlib
import logging
import os
import subprocess
import threading
import time
import unicodedata
from typing import IO, TYPE_CHECKING, Any, Callable

from agent_engine_runner_shared.context import get_current_session_id

if TYPE_CHECKING:
    from agent_engine_runner_shared.runtime import TenantRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default the workspace to a path *outside* the typical project tree.
# Putting it under the project root (e.g. ``./workspace``) causes dev-mode
# file watchers to restart the AER/OE processes every time the agent writes
# a file, killing in-flight LLM streams with a mid-read peer-close. Explicit
# ``WORKSPACE_DIR`` env wins so production containers can pin any path.
# ``realpath`` (not just ``abspath``) so a symlinked mount and its target
# compare equal when callers pass either spelling. Without this, the
# fast-path in ``_resolve`` rebases legitimate in-workspace absolute paths
# into a nonsense nested location.
WORKSPACE_DIR = os.path.realpath(
    os.path.abspath(os.environ.get("WORKSPACE_DIR", "/tmp/agent-workspace"))
)


def _real_abs(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _agent_source_root_from_env() -> str | None:
    agent_config_path = os.environ.get("AGENTIC_AGENT_CONFIG_PATH")
    if agent_config_path:
        return os.path.dirname(_real_abs(agent_config_path))
    agent_workdir = os.environ.get("AGENTIC_AGENT_WORKDIR")
    if agent_workdir:
        return _real_abs(agent_workdir)
    return None


def _readonly_skills_root_from_env() -> str | None:
    source_root = _agent_source_root_from_env()
    explicit = os.environ.get("AGENTIC_SKILLS_DIR")
    skills_dir_setting = explicit or "skills"
    if source_root is None:
        if explicit:
            logger.warning(
                "Ignoring AGENTIC_SKILLS_DIR=%r because the agent source root is unknown",
                explicit,
            )
        return None
    if os.path.isabs(skills_dir_setting):
        logger.warning(
            "Ignoring AGENTIC_SKILLS_DIR=%r because it must be relative to "
            "the agent source root %s",
            skills_dir_setting,
            source_root,
        )
        return None

    skills_dir = _real_abs(os.path.join(source_root, skills_dir_setting))
    if os.path.commonpath([source_root, skills_dir]) != source_root:
        logger.warning(
            "Ignoring AGENTIC_SKILLS_DIR=%r because it resolves outside the agent source root %s",
            skills_dir_setting,
            source_root,
        )
        return None
    return skills_dir


def _is_readable_dir(candidate: str) -> bool:
    # X_OK too: opening files under a directory requires search permission.
    return os.path.isdir(candidate) and os.access(candidate, os.R_OK | os.X_OK)


# Resolved lazily, not at import: a normal import before the app sets
# AGENTIC_AGENT_WORKDIR/AGENTIC_AGENT_CONFIG_PATH must still resolve the
# right root. Memoized — the skill root is a process-global
# bundled resource and must not shift between sessions. Roots that aren't
# readable directories are excluded (agents may bundle no skills); a
# configured-but-unusable root raises at startup in register_builtin_tools.
_readonly_resource_roots_memo: tuple[str, ...] | None = None


def get_readonly_resource_roots() -> tuple[str, ...]:
    global _readonly_resource_roots_memo
    if _readonly_resource_roots_memo is None:
        root = _readonly_skills_root_from_env()
        _readonly_resource_roots_memo = (root,) if root and _is_readable_dir(root) else ()
    return _readonly_resource_roots_memo


SHELL_OUTPUT_MAX_BYTES = int(os.environ.get("SHELL_OUTPUT_MAX_BYTES", "65536"))

# Default shell_execute timeout when the caller passes ``timeout=None``.
# Matches the protocol: ``timeout=None`` means "use the backend default",
# not "run forever".
SHELL_DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SHELL_DEFAULT_TIMEOUT_SECONDS", "30"))

# Cap filesystem_read / filesystem_edit raw bytes so broad reads on giant
# files don't exhaust pod memory. ``filesystem_edit`` loads the whole file
# to do in-place find/replace, so the same cap applies.
FILESYSTEM_READ_MAX_BYTES = int(os.environ.get("FILESYSTEM_READ_MAX_BYTES", str(10 * 1024 * 1024)))

# Cap filesystem_grep matches + wall-clock budget to bound memory *and*
# compute when walking large directory trees. Deepagents' grep contract
# is literal substring matching (not regex), so classic ReDoS doesn't
# apply — but the time budget still protects against deep tree walks
# pegging a thread-pool worker indefinitely.
MAX_GREP_MATCHES = int(os.environ.get("MAX_GREP_MATCHES", "1000"))
MAX_GREP_SECONDS = float(os.environ.get("MAX_GREP_SECONDS", "5"))

# Cap filesystem_glob matches with the same shape as grep. Recursive
# ``**/*`` against a cloned monorepo can materialize millions of paths.
MAX_GLOB_MATCHES = int(os.environ.get("MAX_GLOB_MATCHES", "1000"))

# Cap filesystem_ls entries. A directory with hundreds of thousands of
# entries (e.g. an agent-driven ``git clone`` of a large repo, or a
# runaway write loop) would otherwise materialize the full list into
# memory and ship it through the OE path.
MAX_LS_ENTRIES = int(os.environ.get("MAX_LS_ENTRIES", "1000"))

# Cap filesystem_download file size. base64 expansion is ~33%; keep the raw
# limit well below typical pod memory.
DOWNLOAD_MAX_BYTES = int(os.environ.get("DOWNLOAD_MAX_BYTES", str(10 * 1024 * 1024)))

# Warn (but do not reject) when filesystem_write is asked to write a file
# larger than this threshold. Tool Pod workspace storage is RAM-backed (/tmp
# or /scratch); large individual writes reduce available pod memory without
# the usual OS page-cache eviction safety net. Override via env var.
FILESYSTEM_WRITE_WARN_BYTES = int(
    os.environ.get("FILESYSTEM_WRITE_WARN_BYTES", str(1 * 1024 * 1024))
)

# Prefixes that correspond to the known writable RAM-backed mounts inside a
# Tool Pod. Validated at startup in ``register_builtin_tools``.
_KNOWN_WORKSPACE_PREFIXES: tuple[str, ...] = ("/tmp", "/scratch")

# Env vars that ``shell_execute`` is allowed to propagate to the child process.
# Everything else (tenant API keys, DB URIs, internal service URLs, pod identity
# hints) is stripped — a prompt-injected agent running ``env`` or
# ``cat /proc/self/environ`` must not see them. This is a per-call scrub; the
# pod-level mount is slated for removal under the tenant-secret boundary ADR.
_SHELL_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}
)

# ``_drain_bounded`` reads from the child's stdout/stderr pipe in fixed-size
# chunks. 4 KiB matches the typical Linux pipe buffer page size — small enough
# that we react quickly to closes/timeouts, large enough to avoid syscall churn
# on chatty children.
_PIPE_READ_CHUNK_BYTES = 4096

# Wall-clock budget for cleanup waits in ``shell_execute``: SIGKILL reaping and
# drain-thread joins. Same value for both since they happen back-to-back on the
# cleanup path; a single knob keeps total worst-case cleanup bounded and easy
# to reason about.
_SHELL_CLEANUP_TIMEOUT_SECONDS = 5

# Canonical set of built-in tool names registered by ``register_builtin_tools``.
# Imported by ``agent_engine_runner_shared.server.tool`` for its startup assertion and by
# tests, so the three callsites can't drift silently.
BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "filesystem_ls",
        "filesystem_read",
        "filesystem_write",
        "filesystem_edit",
        "filesystem_glob",
        "filesystem_grep",
        "filesystem_download",
        "shell_execute",
    }
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _is_within_root(resolved: str, root: str) -> bool:
    real_resolved = os.path.realpath(resolved)
    return real_resolved == root or real_resolved.startswith(root + os.sep)


# Reserved subdirectory under WORKSPACE_DIR that holds per-session subtrees.
# Sessions live in their own namespace so ``_is_within_workspace`` can
# distinguish a caller's own session subtree from somebody else's.
_SESSIONS_NAMESPACE = ".sessions"


def _session_subdir() -> str:
    """Return a filesystem-safe directory name derived from the current
    session id, or ``""`` when no session is active.

    Direct invocations outside an AER request (dev scripts, low-level tests)
    have no session in context and fall back to the base ``WORKSPACE_DIR``.
    Inside AER every request sets ``current_session_id`` via
    ``set_execution_context``.

    The session id is hashed so any caller-provided string (including forms
    like ``team/thread-1`` or unicode content) maps to a fixed-charset slot
    name that cannot contain path separators or collapse to ``.`` / ``..``.
    The slot is not a security boundary — cross-session isolation is
    enforced by ``_is_within_workspace``.
    """
    sid = get_current_session_id()
    if not sid:
        return ""
    normalized = unicodedata.normalize("NFC", sid).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _sessions_root() -> str:
    """Return ``WORKSPACE_DIR/<_SESSIONS_NAMESPACE>``."""
    return os.path.normpath(os.path.join(WORKSPACE_DIR, _SESSIONS_NAMESPACE))


def _effective_workspace_dir() -> str:
    """Return the workspace directory for the current session.

    Equal to ``WORKSPACE_DIR`` when no session is active; otherwise
    ``WORKSPACE_DIR/<_SESSIONS_NAMESPACE>/<slot>``, where ``<slot>`` is
    derived from the current session id by :func:`_session_subdir`.
    """
    sub = _session_subdir()
    return os.path.normpath(os.path.join(_sessions_root(), sub)) if sub else WORKSPACE_DIR


def _is_within_workspace(resolved: str) -> bool:
    """Return True if *resolved* is the current caller's allowed workspace
    or a path under it.

    When a session is active, this narrows to the session's own subtree
    under ``.sessions/<slot>/`` so concurrent sessions on the same Tool
    Pod cannot read or write each other's files. When no session is
    active (dev scripts, low-level tests), this is the full
    ``WORKSPACE_DIR``.

    Uses ``os.path.realpath`` so symlinks along the path cannot escape
    the boundary.
    """
    return _is_within_root(resolved, _effective_workspace_dir())


def _is_within_readonly_resource_root(resolved: str) -> bool:
    return any(_is_within_root(resolved, root) for root in get_readonly_resource_roots())


def _is_within_readable_root(resolved: str) -> bool:
    return _is_within_workspace(resolved) or _is_within_readonly_resource_root(resolved)


def _resolve(path: str, *, allow_readonly_roots: bool = False) -> str:
    """Resolve *path* to an absolute path under :data:`WORKSPACE_DIR`.

    Every path the agent supplies is interpreted as being inside the sandboxed
    workspace. Within an AER session, the rebase target and the boundary check
    both narrow to a ``.sessions/<slot>/`` subtree of ``WORKSPACE_DIR``.

    * Relative paths are joined to the effective per-session workspace.
    * Absolute paths already under ``WORKSPACE_DIR`` are returned as-is
      (supports environments that mount the workspace at a real path, e.g.
      a micro-VM where ``WORKSPACE_DIR=/home/agent``); they are still
      rejected later by the boundary check if they fall outside the
      caller's session subtree.
    * Read-only operations may pass absolute paths under bundled resource roots
      such as ``<agent-dir>/skills``.
    * Other absolute paths have their leading ``/`` stripped and are rebased
      under the effective per-session workspace.

    Paths that escape via ``..`` or via symlinks raise :class:`ValueError` so
    the agent gets a clear signal instead of silent clamping.
    """
    base = _effective_workspace_dir()

    if os.path.isabs(path):
        normalized = os.path.normpath(path)
        # Accept the input either as-is or after symlink resolution, so a
        # symlinked workspace mount (macOS ``/tmp`` -> ``/private/tmp``,
        # Linux bind mounts, etc.) is recognized as the same workspace.
        real_normalized = os.path.realpath(normalized)
        if (
            normalized == WORKSPACE_DIR
            or normalized.startswith(WORKSPACE_DIR + os.sep)
            or real_normalized == WORKSPACE_DIR
            or real_normalized.startswith(WORKSPACE_DIR + os.sep)
        ):
            resolved = normalized
        elif _is_within_readonly_resource_root(normalized):
            if allow_readonly_roots:
                return normalized
            raise ValueError(
                f"Path is inside a read-only resource root: {path!r}. "
                "Use read-only filesystem tools or write to the workspace."
            )
        else:
            os.makedirs(base, exist_ok=True)
            resolved = os.path.normpath(os.path.join(base, path.lstrip(os.sep)))
    else:
        # Auto-create the session subdir so a brand-new session's first
        # ``ls('.')`` doesn't ENOENT before any write has happened.
        os.makedirs(base, exist_ok=True)
        resolved = os.path.normpath(os.path.join(base, path))

    if not _is_within_workspace(resolved):
        raise ValueError(
            f"Path escapes workspace sandbox: {path!r}. "
            f"Use a path inside the workspace (relative paths recommended)."
        )
    return resolved


# ---------------------------------------------------------------------------
# Filesystem handlers
# ---------------------------------------------------------------------------


def filesystem_ls(path: str) -> dict[str, Any]:
    """List directory contents, sorted by name.

    Returns entries with full paths and ``is_dir`` flags. When the directory
    contains more than :data:`MAX_LS_ENTRIES` entries, ``truncated=True`` and
    the returned subset is the **scandir-order** first ``MAX_LS_ENTRIES``
    entries (filesystem-defined and not deterministic across systems), then
    sorted by path for display. The early break is intentional — sorting all
    entries first would defeat the memory cap on directories with hundreds
    of thousands of entries.
    """
    try:
        resolved = _resolve(path, allow_readonly_roots=True)
        # scandir + early break so a runaway directory (e.g. an agent-driven
        # ``git clone`` of a large repo) never materializes all entries into
        # memory. Per-entry ``is_dir`` avoids a second syscall per name vs
        # ``os.listdir`` + ``os.path.isdir``.
        entries: list[dict[str, Any]] = []
        truncated = False
        with os.scandir(resolved) as it:
            for entry in it:
                if len(entries) >= MAX_LS_ENTRIES:
                    truncated = True
                    break
                entries.append({"path": entry.path, "is_dir": entry.is_dir()})
        entries.sort(key=lambda e: e["path"])
        return {"entries": entries, "truncated": truncated}
    except Exception as e:
        return {"error": str(e)}


def filesystem_read(file_path: str, offset: int = 0, limit: int = 2000) -> dict[str, Any]:
    """Read file content as text with line-based slicing.

    Reads lines ``[offset : offset + limit]`` from the file. Rejects files
    larger than :data:`FILESYSTEM_READ_MAX_BYTES` upfront so pathologically
    large files cannot exhaust pod memory.
    """
    try:
        resolved = _resolve(file_path, allow_readonly_roots=True)
        size = os.path.getsize(resolved)
        if size > FILESYSTEM_READ_MAX_BYTES:
            return {
                "error": (
                    f"File exceeds FILESYSTEM_READ_MAX_BYTES ({size} > {FILESYSTEM_READ_MAX_BYTES})"
                )
            }
        selected: list[str] = []
        end = offset + limit
        with open(resolved, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx >= end:
                    break
                if idx >= offset:
                    selected.append(line)
        return {"content": "".join(selected), "encoding": "utf-8"}
    except Exception as e:
        return {"error": str(e)}


def filesystem_write(file_path: str, content: str) -> dict[str, Any]:
    """Write *content* to a new file, creating parent directories as needed.

    Per ``deepagents.BackendProtocol.write``, this is **create-only**: if
    the file already exists the call fails with an error. Agents that want
    to modify an existing file must use :func:`filesystem_edit`, which
    enforces a unique-match guard to prevent silent wrong-region rewrites.
    """
    try:
        resolved = _resolve(file_path)
    except Exception as e:
        return {"error": str(e)}
    try:
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > FILESYSTEM_WRITE_WARN_BYTES:
            logger.warning(
                "filesystem_write: writing large file %r (%d bytes). "
                "Tool Pod workspace is RAM-backed — large files reduce available pod memory.",
                resolved,
                content_bytes,
            )
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved, "x", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": resolved}
    except FileExistsError:
        return {
            "error": (
                f"File already exists at {file_path!r}; "
                "use filesystem_edit to modify existing files."
            )
        }
    except Exception as e:
        return {"error": str(e)}


def filesystem_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> dict[str, Any]:
    """Find and replace text in a file.

    Returns the number of occurrences that were replaced. When *replace_all*
    is ``False`` the match must be unique — a non-unique ``old_string`` is
    rejected with an error so an LLM caller cannot silently corrupt the
    wrong region. Files larger than :data:`FILESYSTEM_READ_MAX_BYTES` are
    rejected upfront to mirror ``filesystem_read``'s memory guard.
    """
    try:
        # Reject empty old_string up front: ``str.count("")`` is ``len(text)+1``
        # (fails unique-match) and ``str.replace("", x, replace_all=True)``
        # interleaves ``x`` between every character, exploding file size.
        if old_string == "":
            return {"error": "old_string must be non-empty"}
        resolved = _resolve(file_path)
        size = os.path.getsize(resolved)
        if size > FILESYSTEM_READ_MAX_BYTES:
            return {
                "error": (
                    f"File exceeds FILESYSTEM_READ_MAX_BYTES ({size} > {FILESYSTEM_READ_MAX_BYTES})"
                )
            }
        with open(resolved, "r", encoding="utf-8") as fh:
            text = fh.read()

        matches = text.count(old_string)
        if matches == 0:
            return {"error": f"old_string not found in {file_path!r}"}

        if replace_all:
            count = matches
            new_text = text.replace(old_string, new_string)
        else:
            if matches > 1:
                return {
                    "error": (
                        f"old_string is not unique ({matches} matches in "
                        f"{file_path!r}); set replace_all=true or supply a "
                        "longer anchor"
                    )
                }
            count = 1
            new_text = text.replace(old_string, new_string, 1)

        # Re-check the size cap on the post-replacement text. Without this a
        # large ``new_string`` (or ``replace_all`` against many matches) could
        # grow the file past FILESYSTEM_READ_MAX_BYTES, defeating the
        # pod-memory guard the up-front size check enforces.
        new_size = len(new_text.encode("utf-8"))
        if new_size > FILESYSTEM_READ_MAX_BYTES:
            return {
                "error": (
                    f"Edit would exceed FILESYSTEM_READ_MAX_BYTES "
                    f"({new_size} > {FILESYSTEM_READ_MAX_BYTES}); reduce "
                    "new_string size or split the edit into smaller pieces"
                )
            }

        with open(resolved, "w", encoding="utf-8") as fh:
            fh.write(new_text)

        return {"occurrences": count}
    except Exception as e:
        return {"error": str(e)}


def filesystem_glob(pattern: str, path: str = ".") -> dict[str, Any]:
    """Match files using a glob pattern under *path*.

    *path* is interpreted as workspace-relative (default ``"."`` = workspace
    root), matching :func:`filesystem_grep`'s convention. Supports recursive
    patterns like ``**/*.py``. Bounded by :data:`MAX_GLOB_MATCHES` to
    prevent unbounded memory on broad recursive patterns (e.g. ``**/*``
    against a large monorepo); when hit, returns ``truncated=True``.
    Absolute patterns are rejected so they cannot bypass the sandbox via
    ``os.path.join``'s left-discard behavior — matches are also filtered
    through ``_is_within_readable_root`` for defense-in-depth against
    symlinks inside the workspace or bundled read-only roots.
    """
    try:
        if os.path.isabs(pattern):
            return {
                "error": (
                    f"Absolute pattern {pattern!r} is not allowed; use a "
                    "workspace-relative pattern."
                )
            }
        # Reject traversal segments up front. glob enumerates the filesystem
        # while expanding a pattern like ``../../etc/*``; even though
        # _is_within_readable_root() filters the returned matches, the
        # enumeration itself can reveal directory structure outside the
        # readable roots (e.g. via permission-denied timing on hidden subtrees).
        # Split on both separators so Windows-style inputs can't slip past.
        pattern_parts = pattern.replace("\\", "/").split("/")
        if ".." in pattern_parts:
            return {
                "error": (
                    f"Pattern {pattern!r} contains traversal segments; "
                    "use a workspace-relative pattern."
                )
            }
        resolved = _resolve(path, allow_readonly_roots=True)
        full_pattern = os.path.join(resolved, pattern)
        matches: list[dict[str, Any]] = []
        truncated = False
        for m in glob_mod.iglob(full_pattern, recursive=True):
            if not _is_within_readable_root(m):
                continue
            matches.append({"path": m, "is_dir": os.path.isdir(m)})
            if len(matches) >= MAX_GLOB_MATCHES:
                truncated = True
                break
        matches.sort(key=lambda e: e["path"])
        return {"matches": matches, "truncated": truncated}
    except Exception as e:
        return {"error": str(e)}


def filesystem_grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> dict[str, Any]:
    """Search file contents for a **literal substring**.

    Per ``deepagents.BackendProtocol.grep``, *pattern* is an exact string
    match (not a regex). Walks the directory tree under *path* (defaults
    to ``"."``). When *glob* is provided only files whose names match
    that glob are searched. Binary files are silently skipped.

    Bounded by both :data:`MAX_GREP_MATCHES` (result count) and
    :data:`MAX_GREP_SECONDS` (wall-clock) so a deep-tree walk cannot
    peg a Tool-Pod worker thread indefinitely.
    """
    try:
        search_path = _resolve(path or ".", allow_readonly_roots=True)
        matches: list[dict[str, Any]] = []
        truncated = False
        deadline = time.monotonic() + MAX_GREP_SECONDS

        for dirpath, _dirnames, filenames in os.walk(search_path):
            if truncated:
                break
            for filename in filenames:
                if time.monotonic() > deadline:
                    truncated = True
                    break
                if glob and not fnmatch.fnmatch(filename, glob):
                    continue
                file_path = os.path.join(dirpath, filename)
                # A symlinked file inside the workspace or a bundled read-only
                # root can point outside the sandbox. Resolve before ``open``
                # follows the link.
                if not _is_within_readable_root(file_path):
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        for line_no, line_text in enumerate(fh, start=1):
                            if time.monotonic() > deadline:
                                truncated = True
                                break
                            if pattern in line_text:
                                matches.append(
                                    {
                                        "path": file_path,
                                        "line": line_no,
                                        "text": line_text.rstrip("\n"),
                                    }
                                )
                                if len(matches) >= MAX_GREP_MATCHES:
                                    truncated = True
                                    break
                except (UnicodeDecodeError, PermissionError):
                    continue
                if truncated:
                    break

        return {"matches": matches, "truncated": truncated}
    except Exception as e:
        return {"error": str(e)}


def filesystem_download(file_path: str) -> dict[str, Any]:
    """Download a file's raw bytes base64-encoded.

    Reads the file in binary mode, base64-encodes the bytes, and returns
    them under ``content_base64`` so the JSON transport can carry arbitrary
    binary content.  ``AgentEngineToolPodBackend.download_files`` base64-decodes
    this to produce the ``FileDownloadResponse.content`` bytes.
    """
    try:
        resolved = _resolve(file_path, allow_readonly_roots=True)
        size = os.path.getsize(resolved)
        if size > DOWNLOAD_MAX_BYTES:
            return {"error": (f"File exceeds DOWNLOAD_MAX_BYTES ({size} > {DOWNLOAD_MAX_BYTES})")}
        with open(resolved, "rb") as fh:
            raw = fh.read()
        return {
            "path": resolved,
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Shell handler
# ---------------------------------------------------------------------------


def _truncate_output(text: str) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= SHELL_OUTPUT_MAX_BYTES:
        return text, False
    return encoded[:SHELL_OUTPUT_MAX_BYTES].decode("utf-8", errors="replace"), True


# Sentinel exit codes. subprocess.run's returncode is never negative on
# Unix except when the child is killed by a signal (returncode = -signal);
# we reserve these small values for framework-level states instead.
EXIT_CODE_TIMEOUT = -1
EXIT_CODE_FRAMEWORK_ERROR = -2


def _drain_bounded(
    pipe: IO[bytes],
    bucket: list[bytes],
    cap: int,
    overflow: list[bool],
) -> None:
    """Drain *pipe* into *bucket*, stopping appends once ``cap`` bytes
    have been captured. Sets ``overflow[0] = True`` if the child emits
    more than ``cap`` bytes so the caller can flag the output as
    truncated regardless of whether post-decode ``_truncate_output``
    trips (the drain already discarded the excess, so the final string
    is always under the cap).

    Critically, this keeps **reading** past ``cap`` — it just discards the
    rest — so the child process never blocks on a full pipe. That's what
    lets us bound memory while still letting the child run to completion
    (or to its timeout).
    """
    captured = sum(len(chunk) for chunk in bucket)
    while True:
        try:
            chunk = pipe.read(_PIPE_READ_CHUNK_BYTES)
        except (ValueError, OSError):
            # Main thread closed the pipe (e.g. on framework-error cleanup)
            # while we were still draining. Python raises
            # ``ValueError("I/O operation on closed file")`` here; on some
            # platforms an OSError is possible. Either way, stop cleanly so
            # the daemon thread doesn't die with an unhandled exception.
            return
        if not chunk:
            return
        if captured < cap:
            take = min(len(chunk), cap - captured)
            bucket.append(chunk[:take])
            captured += take
            if len(chunk) > take:
                overflow[0] = True
        else:
            # Still read to keep the child unblocked, but drop the bytes
            # and flag the overflow.
            overflow[0] = True


def _build_shell_env() -> dict[str, str]:
    """Return the minimal env the shell child inherits.

    The Tool-Pod container currently mounts tenant secrets via
    ``envFrom.secretRef`` (org-secrets + app-secrets). Without scrubbing,
    a prompt-injected agent running ``shell_execute("env")`` would exfiltrate
    every one of them into LLM output — a known prompt-injection attack class
    (OWASP LLM01). Until the tenant-secret boundary ADR removes the
    pod-level mount, we reconstruct the child's env from an explicit
    allowlist.

    Scope limit: this scrubs only the child's *own* environment. The child
    runs as the same UID as the tool server (PID 1), so it can still read
    the parent's full environment via ``/proc/1/environ`` regardless of
    whether the image runs as root or a dedicated user. The scrub blunts
    casual exfiltration; it is not an OS-enforced secret boundary.
    """
    return {k: v for k, v in os.environ.items() if k in _SHELL_ENV_ALLOWLIST}


def shell_execute(command: str, *, timeout: int | None = None) -> dict[str, Any]:
    """Run a shell command and capture its output.

    Per ``deepagents.SandboxBackendProtocol.execute``, *timeout* is
    keyword-only. ``None`` means "use the backend default"
    (:data:`SHELL_DEFAULT_TIMEOUT_SECONDS`), not "no timeout".

    Threat model
    ------------
    ``shell=True`` is intentional: agents routinely need pipes, redirects,
    globs, and compound commands. The sandbox is the Tool Pod itself —
    network egress, filesystem scope, and process privileges are enforced at
    the pod boundary, not by argument parsing here. There is no upstream
    approval gate on the command string; treat the content as
    agent-authored.

    Per-session isolation gap: ``cwd`` is set to the caller's per-session
    subtree, but the spawned shell process is not constrained at the OS
    level. ``..`` traversal and absolute paths can therefore reach sibling
    sessions on the same Tool Pod. Multi-tenant deployments must treat
    ``shell_execute`` as session-shared until the bubblewrap-based sandbox
    lands.

    The child process does **not** inherit the Tool-Pod's full env. See
    :func:`_build_shell_env` — only an allowlist of non-sensitive vars
    (``PATH``, ``HOME``, locale) is passed through. The scrub covers only
    the child's own environment: a same-UID child can still read the
    parent's env via ``/proc/1/environ``, so tenant secrets mounted via
    ``envFrom`` remain reachable from a user-authored shell command until
    the pod-level mount is removed.

    Output handling
    ---------------
    stdout and stderr are drained concurrently by two background threads.
    Each thread stops appending once :data:`SHELL_OUTPUT_MAX_BYTES` have
    been captured from its pipe, but keeps reading (and discarding) so the
    child never blocks on a full pipe. That's what lets a runaway ``yes``
    or ``cat /dev/urandom`` run to its timeout without OOMing the pod.
    Partial output is preserved on timeout. Framework-level errors (e.g.
    missing ``/bin/sh``) return ``exit_code = EXIT_CODE_FRAMEWORK_ERROR``
    with the error message in ``"error"`` — distinct from the timeout
    sentinel so callers can tell the two apart.
    """
    effective_timeout = timeout if timeout is not None else SHELL_DEFAULT_TIMEOUT_SECONDS

    try:
        cwd = _effective_workspace_dir()
        os.makedirs(cwd, exist_ok=True)
    except Exception as e:
        return {
            "output": "",
            "exit_code": EXIT_CODE_FRAMEWORK_ERROR,
            "truncated": False,
            "error": str(e),
        }

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_build_shell_env(),
        )
    except Exception as e:
        return {
            "output": "",
            "exit_code": EXIT_CODE_FRAMEWORK_ERROR,
            "truncated": False,
            "error": str(e),
        }

    stdout_buf: list[bytes] = []
    stderr_buf: list[bytes] = []
    stdout_overflow: list[bool] = [False]
    stderr_overflow: list[bool] = [False]
    assert proc.stdout is not None and proc.stderr is not None
    cap = SHELL_OUTPUT_MAX_BYTES
    out_thread = threading.Thread(
        target=_drain_bounded,
        args=(proc.stdout, stdout_buf, cap, stdout_overflow),
        daemon=True,
    )
    err_thread = threading.Thread(
        target=_drain_bounded,
        args=(proc.stderr, stderr_buf, cap, stderr_overflow),
        daemon=True,
    )
    out_thread.start()
    err_thread.start()

    try:
        timed_out = False
        try:
            proc.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                proc.wait(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                pass

        # Join drain threads BEFORE reading the buffers so any bytes the
        # child wrote just before exit (or before SIGKILL) make it out of
        # the OS pipe buffer and into ``stdout_buf``/``stderr_buf``. Closing
        # the pipes happens in ``finally``; if we close before joining, the
        # drain threads exit via ``_drain_bounded``'s (ValueError, OSError)
        # handler with the in-flight chunk lost — which surfaces as missing
        # partial output on timeout (e.g. ``echo EARLY; sleep 5`` losing
        # ``EARLY`` when killed at 1s).
        out_thread.join(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)
        err_thread.join(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)

        stdout_bytes = b"".join(stdout_buf)
        stderr_bytes = b"".join(stderr_buf)
        combined = stdout_bytes.decode("utf-8", errors="replace") + stderr_bytes.decode(
            "utf-8", errors="replace"
        )
        drain_truncated = stdout_overflow[0] or stderr_overflow[0]

        if timed_out:
            combined = f"Command timed out after {effective_timeout}s\n" + combined
            output, post_truncated = _truncate_output(combined)
            return {
                "output": output,
                "exit_code": EXIT_CODE_TIMEOUT,
                "truncated": drain_truncated or post_truncated,
            }

        output, post_truncated = _truncate_output(combined)
        return {
            "output": output,
            "exit_code": proc.returncode,
            "truncated": drain_truncated or post_truncated,
        }
    except Exception as e:
        return {
            "output": "",
            "exit_code": EXIT_CODE_FRAMEWORK_ERROR,
            "truncated": False,
            "error": str(e),
        }
    finally:
        # Cleanup must run on both the success and framework-error paths.
        # Without this, a framework error after Popen + thread start would
        # leak the subprocess (zombie until reaped), the two pipe FDs, and
        # the two daemon drain threads — all of which accumulate across
        # calls in a long-lived Tool Pod.
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                # SIGKILL didn't reap within the cleanup budget — extremely
                # unusual (kernel zombie / uninterruptible sleep). Log so
                # operators can see it; we still proceed to close FDs and
                # join threads.
                logger.warning(
                    "shell_execute: SIGKILL did not reap pid=%d within %ds; "
                    "subprocess may be in uninterruptible sleep",
                    proc.pid,
                    _SHELL_CLEANUP_TIMEOUT_SECONDS,
                )

        # Explicitly close the pipe FDs — Popen.__del__ would eventually do
        # this, but relying on GC in a long-lived Tool Pod leaks FDs across
        # many calls. Each close in its own try/except so a failure on
        # stdout doesn't prevent stderr cleanup.
        for pipe_name, pipe in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            if pipe is None:
                continue
            try:
                pipe.close()
            except Exception:
                logger.debug(
                    "shell_execute: failed to close %s pipe for pid=%d",
                    pipe_name,
                    proc.pid,
                    exc_info=True,
                )

        # Belt-and-suspenders join: the happy path already joined inside
        # ``try``. On the framework-error path the threads may still be
        # running; closing the pipes above caused them to exit via
        # ``_drain_bounded``'s (ValueError, OSError) handler, so this join
        # returns quickly. ``Thread.join`` is idempotent on already-joined
        # threads.
        out_thread.join(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)
        err_thread.join(timeout=_SHELL_CLEANUP_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "filesystem_ls": filesystem_ls,
    "filesystem_read": filesystem_read,
    "filesystem_write": filesystem_write,
    "filesystem_edit": filesystem_edit,
    "filesystem_glob": filesystem_glob,
    "filesystem_grep": filesystem_grep,
    "filesystem_download": filesystem_download,
    "shell_execute": shell_execute,
}

# Invariant: BUILTIN_TOOL_NAMES (the public contract) must match the actual
# registered handler set. Locked at import so drift fails fast rather than
# surfacing as a confusing startup-assertion failure.
assert frozenset(_HANDLERS.keys()) == BUILTIN_TOOL_NAMES, (
    "BUILTIN_TOOL_NAMES drift: "
    f"declared={sorted(BUILTIN_TOOL_NAMES)} vs registered={sorted(_HANDLERS)}"
)


def register_builtin_tools(runtime: "TenantRuntime") -> None:
    """Register all built-in tool handlers on *runtime*.

    Populates ``runtime._tools`` (callable lookup) and
    ``runtime._tool_definitions`` (metadata) for all 8 handlers.
    Called by ``ToolServer.on_startup`` before the server starts
    accepting requests.

    If a user ``@app.tool()`` registered a tool with one of the reserved
    built-in names at import time, this function logs a WARNING and
    overrides it with the built-in. Without the warning the collision
    was silent and would surface as confusing runtime behavior only when
    the built-in was actually invoked.
    """
    # Warn before any side effects if WORKSPACE_DIR is outside the known writable
    # mounts. On production Tool Pods only /tmp and /scratch are RAM-backed writable
    # mounts; any other path will likely fail or write to a read-only layer.
    # Warning rather than hard error so local dev with non-standard mounts still
    # starts — upgrade to raise RuntimeError once the allowlist is confirmed against
    # all prod pod configurations.
    _sep = os.sep
    if WORKSPACE_DIR not in _KNOWN_WORKSPACE_PREFIXES and not any(
        WORKSPACE_DIR.startswith(p + _sep) for p in _KNOWN_WORKSPACE_PREFIXES
    ):
        logger.warning(
            "WORKSPACE_DIR=%r is outside the known writable mounts (/tmp, /scratch). "
            "On production Tool Pods only /tmp and /scratch are writable; "
            "filesystem and shell handlers may fail at runtime.",
            WORKSPACE_DIR,
        )

    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    logger.info("Tool Pod workspace: %s", WORKSPACE_DIR)

    # Fail fast when an explicit AGENTIC_SKILLS_DIR resolves to no usable
    # skills root. A default "skills" dir that doesn't exist is not an error
    # (agents may bundle no skills); one that exists but isn't a readable
    # directory was previously a silent no-op, so it only warns for one
    # release to avoid crash-looping already-deployed agents — upgrade to a
    # raise after.
    roots = get_readonly_resource_roots()
    explicit_skills_dir = os.environ.get("AGENTIC_SKILLS_DIR")
    if explicit_skills_dir and not roots:
        raise RuntimeError(
            f"AGENTIC_SKILLS_DIR={explicit_skills_dir!r} was set but no readable skills root "
            "could be resolved. Ensure AGENTIC_AGENT_CONFIG_PATH or AGENTIC_AGENT_WORKDIR is "
            "set before the Tool Pod starts and that the directory exists under the agent "
            "source root."
        )
    # The resolver filters unusable roots, so re-derive the candidate: a
    # default skills dir that exists but is not a readable directory is a
    # misconfiguration, not "no skills bundled".
    candidate = _readonly_skills_root_from_env()
    if not explicit_skills_dir and candidate and not roots and os.path.exists(candidate):
        logger.warning(
            "Configured read-only skills root %r is not a readable directory; "
            "bundled skills will be unavailable. Check the bundled skills path "
            "and AGENTIC_SKILLS_DIR.",
            candidate,
        )

    for name, func in _HANDLERS.items():
        if name in runtime._tools and runtime._tools[name] is not func:
            logger.warning(
                "register_builtin_tools: overriding user-registered tool %r "
                "with built-in handler. Rename the @app.tool() or remove it "
                "to avoid surprising runtime behavior.",
                name,
            )
        runtime._tools[name] = func
        runtime._tool_definitions[name] = {
            "name": name,
            "description": (func.__doc__ or "").strip(),
            "is_local": False,
            "network": [],
            "timeout_seconds": 30,
            "redact_fields": [],
        }
