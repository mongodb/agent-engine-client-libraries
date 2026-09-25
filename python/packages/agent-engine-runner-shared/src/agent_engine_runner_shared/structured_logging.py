"""Structured JSON logging for the agent runtime log capture pipeline.

Emits single-line JSON records on container stdout. Fluent Bit tails the
container log, the Lua filter derives tenant/workspace keys from pod
metadata, and writes gzipped batches to S3. The API Gateway
``GET /api/v1/agent-logs`` endpoint reads them back.

Activated by calling ``install_structured_logging()`` at runner startup —
typically gated by ``STRUCTURED_LOGGING=true``. When inactive, the existing
human-readable logging in ``agent_engine_runner_shared.utils.setup_logging`` is
unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback as tb_mod
from datetime import datetime, timezone
from io import IOBase
from types import TracebackType
from typing import IO, Any, Callable, Dict, Optional, cast

from agent_engine_runner_shared.context import (
    get_current_execution_id,
    get_current_log_origin,
    get_current_session_id,
    get_current_trace_id,
    get_current_workspace_id,
)

# Map RUNNER_MODE values to the canonical service identifier on the log line.
# The mode strings are the same ones produced by agent_engine_runner_shared.utils.RuntimeMode.
_SERVICE_BY_MODE: Dict[str, str] = {
    "orchestrator": "orchestration-engine",
    "aer": "agent-execution-runtime",
    "tool": "tool-executor",
    # Function mode is a per-call tool executor, so it logs under the same
    # service as the long-running tool server for routing and alerting.
    "tool_function": "tool-executor",
}

# Source value attached to LogRecord by LoggingStream. For records that come
# straight from logging.* (i.e. not stdout/stderr capture) we default to
# python-logging. The attribute name is namespaced (no leading underscore)
# to avoid collision risk with future stdlib LogRecord attributes — the
# stdlib uses leading-underscore for its internal attrs.
_RECORD_SOURCE_ATTR = "agentic_log_source"
_DEFAULT_RECORD_SOURCE = "python-logging"

# Truncation for the formatted traceback embedded in fields.exc_traceback.
# Long traces explode the JSON line size and choke Fluent Bit's batch
# pipeline; 8 KiB is enough to retain the call site and outer frames.
_MAX_TRACEBACK_BYTES = 8192

# Cap on LoggingStream's per-stream pending buffer. A library that writes a
# huge payload without ``\n`` (binary blob accidentally hitting stdout, a
# multi-MB stack from a C extension, a misbehaving tqdm) would otherwise
# grow the buffer until process OOM. 64 KiB is well above any reasonable
# log line and far below a runaway producer.
_MAX_BUFFER_BYTES = 65536


def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    return val or None


def _service_for(mode: Optional[str]) -> str:
    """Map a RUNNER_MODE value to a service identifier."""
    if not mode:
        return "agent-execution-runtime"
    return _SERVICE_BY_MODE.get(mode.lower(), mode.lower())


def _resolve_level(level: Optional[str]) -> int:
    """Resolve a log level from arg → ``LOG_LEVEL`` env → INFO default.

    Accepts both string names ("DEBUG", "info") and callers passing in
    explicit ``logging.LEVEL`` ints (which we leave unchanged).
    """
    if isinstance(level, int):
        return level
    raw = level or _env("LOG_LEVEL") or "INFO"
    return getattr(logging, raw.upper(), logging.INFO)


class StructuredJSONFormatter(logging.Formatter):
    """Formats LogRecord instances as single-line JSON.

    Tenant context is read from agent-engine-runner-shared contextvars at format time so
    that records emitted from inside an execution carry executionId/sessionId
    automatically; records emitted outside execution (startup, idle) leave
    those fields ``null``.
    """

    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        # Env vars (RUNNER_MODE, ORG_ID, PROJECT_ID, WORKSPACE_ID,
        # AGENTIC_BOOT_ID, POD_NAME) are snapshotted here at construction time,
        # not re-read per record
        # — every LogRecord goes through this formatter and re-resolving env
        # on each call would noticeably load the hot path. Callers must
        # ensure the pod env is set before install_structured_logging()
        # runs; if the install runs before the entrypoint stamps env, the
        # formatter caches stale (or missing) values for the process
        # lifetime. The ``mode`` arg lets callers override RUNNER_MODE
        # explicitly — useful when the pod env happens to be unset but the
        # caller knows its mode (e.g. setup_logging passes its mode= arg).
        resolved_mode = mode or _env("RUNNER_MODE")
        self._service = _service_for(resolved_mode)
        self._component = (resolved_mode or "agent").lower()
        self._tenant_id = _env("ORG_ID")
        self._project_id = _env("PROJECT_ID")
        self._workspace_id_env = _env("WORKSPACE_ID")
        self._boot_id = _env("AGENTIC_BOOT_ID")
        self._pod_name = _env("POD_NAME") or _env("HOSTNAME")

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        source = getattr(record, _RECORD_SOURCE_ATTR, _DEFAULT_RECORD_SOURCE)

        fields: Dict[str, Any] = {"component": self._component}
        if source == _DEFAULT_RECORD_SOURCE:
            fields["filename"] = record.filename
            fields["lineno"] = record.lineno
            fields["funcName"] = record.funcName
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            fields["exc_type"] = exc_type.__name__
            fields["exc_message"] = str(exc_value) if exc_value else ""
            # Capture the formatted traceback so on-call can debug from the
            # archived JSON without reproducing. Truncated to keep S3 batch
            # sizes manageable; truncation is marked so consumers know the
            # tail was cut.
            tb_text = "".join(tb_mod.format_exception(exc_type, exc_value, exc_tb))
            # Encode to bytes for the size check so the budget reflects
            # what Fluent Bit actually ships — slicing the str by char
            # count would let non-ASCII tracebacks blow past the limit.
            # ``errors="ignore"`` discards a partial UTF-8 sequence at the
            # cut point so the resulting str round-trips cleanly to JSON.
            encoded = tb_text.encode("utf-8")
            if len(encoded) > _MAX_TRACEBACK_BYTES:
                tb_text = (
                    encoded[:_MAX_TRACEBACK_BYTES].decode("utf-8", errors="ignore")
                    + "...(truncated)"
                )
            fields["exc_traceback"] = tb_text

        entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
            "tenantId": self._tenant_id,
            "projectId": self._project_id,
            "workspaceId": get_current_workspace_id() or self._workspace_id_env,
            "executionId": get_current_execution_id(),
            "sessionId": get_current_session_id(),
            "bootId": self._boot_id,
            "podName": self._pod_name,
            "source": source,
            "fields": fields,
        }
        origin = get_current_log_origin()
        if origin is not None:
            entry["origin"] = origin
        trace_id = get_current_trace_id()
        if trace_id and re.fullmatch(r"[0-9a-f]{32}", trace_id) and trace_id != "0" * 32:
            entry["traceId"] = trace_id

        return json.dumps(entry, default=str, separators=(",", ":"))


class LoggingStream(IOBase):
    """Replacement for ``sys.stdout``/``sys.stderr`` that routes writes through logging.

    Bare ``print()`` calls (and any other code writing directly to the
    captured stream) are rewritten as LogRecord instances tagged with
    ``agentic_log_source = "stdout"`` (or ``"stderr"``) so the formatter
    can mark them as such on the wire. Lines are buffered until newline,
    with a hard byte cap so a producer writing without ``\\n`` cannot
    grow the buffer until OOM (see ``_MAX_BUFFER_BYTES``).

    Holds a reference to the wrapped *original* stream so that a re-install
    can recover and ``fileno()``/``buffer``/``encoding`` requests from
    third-party libraries (subprocess, tqdm, gRPC) can be delegated through
    instead of raising. ``write()``/``flush()`` are guarded by an
    ``RLock`` so concurrent producers don't interleave bytes mid-line.

    Inherits from ``IOBase`` rather than ``TextIOBase``: ``TextIOBase``
    declares ``encoding`` as a read-only C descriptor and we need a writable
    instance attribute to surface the wrapped stream's encoding to callers
    that probe ``sys.stdout.encoding`` (CLI helpers, click, rich).
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        source: str,
        wrapped: Optional[IO[str]] = None,
    ) -> None:
        super().__init__()
        self._logger = logger
        self._level = level
        self._source = source
        self._buffer = ""
        self._lock = threading.RLock()
        self._wrapped = wrapped
        wrapped_encoding = getattr(wrapped, "encoding", None) if wrapped is not None else None
        self.encoding = wrapped_encoding or "utf-8"

    def writable(self) -> bool:  # pragma: no cover - trivial
        return True

    def fileno(self) -> int:
        # Subprocess piping (stdout=sys.stdout), tqdm progress bars, and
        # grpc-internal logging all probe fileno(). Delegate to the
        # wrapped stream so they keep working; raise if we have no
        # underlying fd to point at, matching TextIOBase's default.
        wrapped_fileno = (
            getattr(self._wrapped, "fileno", None) if self._wrapped is not None else None
        )
        if wrapped_fileno is not None:
            return wrapped_fileno()
        raise OSError("LoggingStream has no underlying file descriptor")

    @property
    def buffer(self) -> Any:
        # Some libraries write bytes via ``sys.stdout.buffer``. Delegate
        # to the wrapped stream's buffer when one exists; raise so callers
        # fall back to text writes when none does. ``getattr`` keeps
        # pyright happy since IO[str] doesn't declare ``buffer``.
        wrapped_buffer = (
            getattr(self._wrapped, "buffer", None) if self._wrapped is not None else None
        )
        if wrapped_buffer is None:
            raise AttributeError("LoggingStream has no underlying binary buffer")
        return wrapped_buffer

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self._emit(line)
            # Force-emit a truncated record if a producer is writing without
            # newlines — mirrors the exc_traceback truncation pattern. Slice
            # by UTF-8 byte count so the cap reflects what Fluent Bit ships.
            encoded = self._buffer.encode("utf-8")
            if len(encoded) > _MAX_BUFFER_BYTES:
                truncated = (
                    encoded[:_MAX_BUFFER_BYTES].decode("utf-8", errors="ignore") + "...(truncated)"
                )
                self._emit(truncated)
                self._buffer = ""
            return len(text)

    def flush(self) -> None:
        with self._lock:
            # Whitespace-only buffer content is uninteresting — emitting a
            # JSON record with ``message: "   "`` adds noise and makes
            # downstream filters work harder. Same rule as write()'s
            # per-line ``if line.strip()`` filter; do not 'fix' by
            # removing the strip.
            if self._buffer.strip():
                self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        record = self._logger.makeRecord(
            self._logger.name,
            self._level,
            "(stdout-capture)",
            0,
            line,
            (),
            None,
        )
        setattr(record, _RECORD_SOURCE_ATTR, self._source)
        self._logger.handle(record)


def _unwrap_logging_stream(stream: IO[str]) -> IO[str]:
    """Return the original stream a ``LoggingStream`` was wrapping, or
    ``stream`` itself if it isn't one. Used on re-install so the new
    handler writes to the real stdout/stderr instead of the previous
    install's ``LoggingStream`` (which would feed logging back into itself
    and recurse to ``RecursionError``).

    Recursive: today ``install_structured_logging`` always passes the
    unwrapped stream as ``wrapped=``, so one level is enough. The loop
    is defensive — a future caller that hands us a ``LoggingStream``
    *as* the wrapped argument would otherwise re-introduce recursion.
    """
    while isinstance(stream, LoggingStream) and stream._wrapped is not None:
        stream = stream._wrapped
    return stream


# Thread-local, not global: both excepthooks run on the crashing thread, so a
# shared flag would let one thread's crash silence another's.
_crash_state = threading.local()


def _log_crash_and_chain(message: str, exc_info: Any, chain: Callable[[], object]) -> None:
    """Log one ERROR per crash, then delegate. Only the outermost of our
    wrappers on this thread logs; inner ones chain silently."""
    outermost = not getattr(_crash_state, "logging", False)
    _crash_state.logging = True
    try:
        if outermost:
            logging.getLogger().error(message, exc_info=exc_info)
        chain()
    finally:
        if outermost:
            _crash_state.logging = False


def _install_uncaught_exception_hooks() -> None:
    """Install ``sys.excepthook``/``threading.excepthook`` overrides that log
    one ERROR summary per uncaught exception before chaining to whatever
    hook was previously installed.

    Severity should reflect outcome, not stream: an uncaught, process-
    terminating exception writes its traceback to ``sys.stderr``, which
    ``install_structured_logging()`` tags WARNING like any other stderr
    write. Chaining onto the previous hook (by default
    ``sys.__excepthook__``) preserves that existing line-by-line WARNING
    trace unchanged — this only adds one additional, filterable ERROR
    record summarizing the crash.

    Guarded by a marker attribute on our own wrapper so repeat
    ``install_structured_logging()`` calls (``setup_logging`` runs on every
    call, plus the launcher's own early install, plus
    ``TenantRuntime.__init__``) don't chain onto themselves and log the
    same crash once per install.
    """
    if not getattr(sys.excepthook, "_agent_engine_runner_shared_installed", False):
        _previous_excepthook = sys.excepthook

        def _handle_uncaught_exception(exc_type, exc_value, exc_tb):
            if exc_type is KeyboardInterrupt:
                return _previous_excepthook(exc_type, exc_value, exc_tb)
            _log_crash_and_chain(
                f"{exc_type.__name__}: {exc_value}",
                (exc_type, exc_value, exc_tb),
                lambda: _previous_excepthook(exc_type, exc_value, exc_tb),
            )

        # setattr, not plain assignment: pyright rejects an undeclared
        # attribute on a FunctionType value. Same marker idiom this module
        # already uses for LogRecord in LoggingStream.write.
        setattr(_handle_uncaught_exception, "_agent_engine_runner_shared_installed", True)
        sys.excepthook = _handle_uncaught_exception

    if not getattr(threading.excepthook, "_agent_engine_runner_shared_installed", False):
        _previous_thread_excepthook = threading.excepthook

        def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
            # SystemExit in a thread means the thread wants to end, not a
            # crash — threading's own default hook ignores it too. Still
            # delegate: a custom monitoring hook may track thread exits, and
            # swallowing the call would break the chaining guarantee above.
            if args.exc_type is SystemExit:
                _previous_thread_excepthook(args)
                return
            thread_name = args.thread.name if args.thread is not None else "<unknown>"
            _log_crash_and_chain(
                f"Uncaught exception in thread {thread_name}: "
                f"{args.exc_type.__name__}: {args.exc_value}",
                # threading only ever calls this with a live sys.exc_info(),
                # so this asserts what the caller already guarantees rather
                # than adding a runtime branch that would change the record.
                cast(
                    tuple[type[BaseException], BaseException, TracebackType | None],
                    (args.exc_type, args.exc_value, args.exc_traceback),
                ),
                lambda: _previous_thread_excepthook(args),
            )

        setattr(_handle_thread_exception, "_agent_engine_runner_shared_installed", True)
        threading.excepthook = _handle_thread_exception


def install_structured_logging(
    stream: Optional[IO[str]] = None,
    level: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Install the structured JSON formatter on the root logger and capture
    stdout/stderr.

    Idempotent: if already installed, recovers the previously-wrapped
    original stream so a second call does not feed the new handler back
    into its own ``LoggingStream`` (which would recurse to RecursionError).

    Args:
        stream: Output stream for the JSON handler. Defaults to the
            *current* (or previously-wrapped) ``sys.stdout`` at install
            time, captured before stdout is replaced by ``LoggingStream``.
            Useful in tests to inject a buffer.
        level: Log level — either a string name (``"DEBUG"``, ``"info"``)
            or a ``logging.LEVEL`` int. Defaults to ``LOG_LEVEL`` env var,
            then INFO. Must be honored on rollout because operators
            continue to flip ``LOG_LEVEL=DEBUG`` to chase issues.

            Applies to ``logging.*`` calls only — stdout/stderr writes
            captured by ``LoggingStream`` always emit, so a stray
            ``print()`` does not silently disappear when an operator sets
            ``LOG_LEVEL=WARNING``. Captures are routed via
            ``Logger.handle()`` directly, bypassing ``isEnabledFor`` and
            the handler's level check.
        mode: Runner mode (``"orchestrator"``, ``"aer"``, ``"tool"``) used
            to derive ``service`` and
            ``fields.component`` on the wire. Takes precedence over
            ``RUNNER_MODE`` env var when set; falls back to env when
            ``None``. Lets ``setup_logging(mode=...)`` callers stay
            authoritative even if pod env hasn't been stamped.
    """
    # On re-install, sys.stdout/stderr is already a LoggingStream. Recover
    # the original so the new StreamHandler writes to a real stream instead
    # of feeding logging back into itself.
    original_stdout = _unwrap_logging_stream(sys.stdout)
    original_stderr = _unwrap_logging_stream(sys.stderr)

    target = stream if stream is not None else original_stdout
    resolved_level = _resolve_level(level)

    formatter = StructuredJSONFormatter(mode=mode)
    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)
    # Intentionally leave handler.level at NOTSET so stdout/stderr
    # capture records — which reach the handler via Logger.handle()
    # rather than logger.info()/logger.warning() — are not filtered
    # by LOG_LEVEL. Filtering happens at root.level via isEnabledFor.

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved_level)

    # Silence chatty HTTP/SDK libraries that would otherwise drown the log
    # volume under typical agent execution. botocore/boto3/s3transfer log
    # full request/response bodies at DEBUG (e.g. secretsmanager.GetSecretValue
    # responses), so this floor also prevents credentials fetched by agent
    # code from leaking into logs when a customer sets a verbose LOG_LEVEL.
    # This list is necessarily incomplete for third-party libraries an agent
    # may install on its own; it covers only known offenders.
    for noisy in ("httpx", "httpcore", "urllib3", "botocore", "boto3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # stdout captures at INFO; stderr at WARNING. Forcing stderr to ERROR
    # (an earlier choice) made every deprecation warning, pip retry banner,
    # and warnings.warn() call emerge as level=ERROR — alerting that fires
    # on level=="ERROR" would fire constantly. WARNING is more honest.
    sys.stdout = LoggingStream(root, logging.INFO, "stdout", wrapped=original_stdout)
    sys.stderr = LoggingStream(root, logging.WARNING, "stderr", wrapped=original_stderr)

    _install_uncaught_exception_hooks()

    root.debug("structured logging installed")
