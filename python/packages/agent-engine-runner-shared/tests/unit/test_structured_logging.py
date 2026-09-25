"""Tests for the structured JSON logging formatter and stdout/stderr capture.

The formatter emits one JSON record per line on the configured output stream.
Tests assert (a) every record is valid JSON matching the design-doc contract,
(b) tenant context propagates from agent_engine_runner_shared contextvars, (c) print() and
sys.stderr writes are captured with the correct ``source`` value, and (d)
records emitted outside an execution leave executionId/sessionId null.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from agent_engine_runner_shared.context import (
    clear_execution_context,
    customer_origin_scope,
    get_current_trace_id,
    set_execution_context,
)
from agent_engine_runner_shared.structured_logging import (
    LoggingStream,
    StructuredJSONFormatter,
    install_structured_logging,
)

_EXEC_ID = "exec-test-001"
_SESSION_ID = "sess-test-002"
_WORKSPACE_ID = "ws-test-003"
_OE_URL = "http://localhost:8000"


@pytest.mark.parametrize("trace_id", ["", "A" * 32, "0" * 32])
def test_invalid_platform_trace_is_not_emitted(trace_id):
    tokens = set_execution_context(
        "e", None, _OE_URL, trace_id=trace_id, request_id="0123456789abcdef0123456789abcdef"
    )
    try:
        record = logging.LogRecord("trace_test", logging.INFO, __file__, 1, "hi", (), None)
        assert "traceId" not in json.loads(StructuredJSONFormatter().format(record))
    finally:
        clear_execution_context(tokens)


# Loggers whose levels install_structured_logging() and setup_logging()
# mutate. Snapshot and restore in tests so leaked levels don't make later
# tests order-dependent.
_TOUCHED_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "uvicorn.access",
    "botocore",
    "boto3",
    "s3transfer",
)


@dataclass
class _LoggingSnapshot:
    root_level: int
    root_handlers: list[logging.Handler]
    logger_levels: dict[str, int]
    excepthook: Any
    thread_excepthook: Any


def _snapshot_logging_state() -> _LoggingSnapshot:
    """Capture root logger level + handlers and the levels of loggers that
    install/setup mutate. Clearing ``root.handlers = []`` without restoring
    drops handlers a parent conftest set up; the noisy levels persist
    silently and can change other tests' assertions about library output.

    Also snapshots ``sys.excepthook``/``threading.excepthook``
    ``install_structured_logging()`` installs guarded wrappers on these that
    only re-install once per process, so a test that installs must restore
    them or it leaks a patched hook into every later test in the process.
    """
    root = logging.getLogger()
    return _LoggingSnapshot(
        root_level=root.level,
        root_handlers=list(root.handlers),
        logger_levels={n: logging.getLogger(n).level for n in _TOUCHED_LOGGERS},
        excepthook=sys.excepthook,
        thread_excepthook=threading.excepthook,
    )


def _restore_logging_state(snap: _LoggingSnapshot) -> None:
    root = logging.getLogger()
    root.handlers = list(snap.root_handlers)
    root.setLevel(snap.root_level)
    for name, level in snap.logger_levels.items():
        logging.getLogger(name).setLevel(level)
    sys.excepthook = snap.excepthook
    threading.excepthook = snap.thread_excepthook


@pytest.fixture
def install_to_buffer(monkeypatch):
    """Set log-context env vars, then install structured logging on demand.

    Pytest's stdout-capture autouse fixture replaces ``sys.stdout`` after
    user fixtures finish setup, so the install must happen *inside* the
    test body (after pytest's capture is in place) — otherwise our
    LoggingStream gets clobbered. The fixture returns a callable that
    performs the install and yields the captured buffer.
    """
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("ORG_ID", "tenant-acme")
    monkeypatch.setenv("PROJECT_ID", "project-zeta")
    monkeypatch.setenv("POD_NAME", "test-pod-xyz")
    monkeypatch.setenv("AGENTIC_BOOT_ID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.delenv("WORKSPACE_ID", raising=False)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    buf = io.StringIO()

    def _install() -> io.StringIO:
        install_structured_logging(stream=buf)
        return buf

    try:
        yield _install
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def _records(buf: io.StringIO) -> List[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_formatter_emits_design_doc_contract(install_to_buffer):
    """Every required field is present and the record validates as JSON."""
    buf = install_to_buffer()
    logging.getLogger("runner_test").info("hello world")

    records = _records(buf)
    msgs = [r for r in records if r["message"] == "hello world"]
    assert len(msgs) == 1, records

    rec = msgs[0]
    for key in (
        "timestamp",
        "level",
        "logger",
        "message",
        "service",
        "tenantId",
        "projectId",
        "workspaceId",
        "executionId",
        "sessionId",
        "bootId",
        "podName",
        "source",
        "fields",
    ):
        assert key in rec, f"missing {key} in {rec}"

    assert rec["level"] == "INFO"
    assert rec["logger"] == "runner_test"
    assert rec["service"] == "agent-execution-runtime"
    assert rec["tenantId"] == "tenant-acme"
    assert rec["projectId"] == "project-zeta"
    assert rec["bootId"] == "11111111-1111-4111-8111-111111111111"
    assert rec["podName"] == "test-pod-xyz"
    assert rec["source"] == "python-logging"
    assert rec["fields"]["component"] == "aer"
    # filename/lineno/funcName populated for python-logging records
    assert "filename" in rec["fields"]
    assert isinstance(rec["fields"]["lineno"], int)
    assert "traceId" not in rec


@pytest.mark.asyncio
async def test_context_vars_populate_when_inside_execution(install_to_buffer):
    """Concurrent executions keep their log context independent of the OTel span."""
    from opentelemetry import trace

    buf = install_to_buffer()
    ids = ["0123456789abcdef0123456789abcdef", "abcdef0123456789abcdef0123456789"]

    async def emit(trace_id):
        tokens = set_execution_context(
            _EXEC_ID + trace_id,
            MagicMock(),
            _OE_URL,
            session_id=_SESSION_ID,
            workspace_id=_WORKSPACE_ID,
            trace_id=trace_id,
            request_id="legacy-request",
        )
        try:
            await asyncio.sleep(0)
            logging.getLogger("runner_test").info("inside execution")
            print("printed")
            sys.stderr.write("stderr\n")
        finally:
            clear_execution_context(tokens)

    span = trace.NonRecordingSpan(trace.SpanContext(trace_id=1, span_id=2, is_remote=False))
    with trace.use_span(span):
        await asyncio.gather(*(emit(trace_id) for trace_id in ids))
        tokens = set_execution_context("background", None, _OE_URL)
        try:
            assert get_current_trace_id() is None
            logging.getLogger("runner_test").info("background")
        finally:
            clear_execution_context(tokens)
    for trace_id in ids:
        records = [r for r in _records(buf) if r["executionId"] == _EXEC_ID + trace_id]
        assert {r["source"] for r in records} == {"python-logging", "stdout", "stderr"}
        for rec in records:
            assert rec["traceId"] == trace_id
            assert rec["sessionId"] == _SESSION_ID
            assert rec["workspaceId"] == _WORKSPACE_ID
    assert "traceId" not in _records(buf)[-1]


def test_print_captured_as_stdout_source(install_to_buffer):
    """sys.stdout writes go through LoggingStream and emerge with source=stdout."""
    buf = install_to_buffer()
    assert isinstance(sys.stdout, LoggingStream), type(sys.stdout)
    sys.stdout.write("printed line\n")
    sys.stdout.flush()

    rec = next(r for r in _records(buf) if r["message"] == "printed line")
    assert rec["source"] == "stdout"
    # source-tagged records do not carry filename/lineno (those are python-
    # logging only); only component is required in fields.
    assert rec["fields"] == {"component": "aer"}


def test_stderr_writes_captured_with_stderr_source(install_to_buffer):
    """Writes to sys.stderr are tagged source=stderr at level=WARNING.

    WARNING (not ERROR) because routine library noise — deprecation
    warnings, urllib retry banners, pip progress — goes to stderr; labeling
    all of it ERROR would fire every alert configured on level==ERROR.
    """
    buf = install_to_buffer()
    assert isinstance(sys.stderr, LoggingStream), type(sys.stderr)
    sys.stderr.write("stderr line\n")
    sys.stderr.flush()

    rec = next(r for r in _records(buf) if r["message"] == "stderr line")
    assert rec["source"] == "stderr"
    assert rec["level"] == "WARNING"


def test_exception_fields_populated_when_logged(install_to_buffer):
    """logger.exception() populates fields.exc_type, .exc_message, and
    .exc_traceback so on-call can debug from the archived JSON without
    reproducing."""
    buf = install_to_buffer()
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        logging.getLogger("runner_test").exception("caught")

    rec = next(r for r in _records(buf) if r["message"] == "caught")
    assert rec["level"] == "ERROR"
    assert rec["fields"]["exc_type"] == "RuntimeError"
    assert rec["fields"]["exc_message"] == "kaboom"
    assert "kaboom" in rec["fields"]["exc_traceback"]


def test_excepthook_emits_error_summary_with_type_and_message(install_to_buffer):
    """An uncaught exception fired through sys.excepthook logs one ERROR
    record whose top-level message is a searchable "Type: message" summary,
    with fields.exc_type/.exc_message/.exc_traceback also populated —
    matching the searchable/visible/copyable convention already used by
    AER's own error logging."""
    buf = install_to_buffer()
    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    error_recs = [r for r in _records(buf) if r["level"] == "ERROR"]
    assert len(error_recs) == 1, error_recs
    rec = error_recs[0]
    assert rec["message"] == "ValueError: boom"
    assert rec["fields"]["exc_type"] == "ValueError"
    assert rec["fields"]["exc_message"] == "boom"
    assert "boom" in rec["fields"]["exc_traceback"]


def test_excepthook_does_not_remove_default_warning_trace(install_to_buffer):
    """Chaining to the default excepthook must still print the traceback
    through the stderr capture at WARNING, exactly as it does today — the
    ERROR summary is purely additive; nothing that worked before is
    removed."""
    buf = install_to_buffer()
    try:
        raise ValueError("boom-trace")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    warning_recs = [r for r in _records(buf) if r["level"] == "WARNING" and r["source"] == "stderr"]
    assert warning_recs, (
        "expected the chained default excepthook to still emit stderr WARNING lines"
    )
    assert any("boom-trace" in r["message"] or "ValueError" in r["message"] for r in warning_recs)


def test_excepthook_idempotent_install_does_not_duplicate(install_to_buffer):
    """A second install_structured_logging() call must not wrap its own
    prior hook — otherwise a crash would log one ERROR record per install
    instead of exactly one."""
    buf = install_to_buffer()
    install_structured_logging(stream=buf)

    try:
        raise ValueError("boom-twice")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    error_recs = [
        r for r in _records(buf) if r["level"] == "ERROR" and "boom-twice" in r["message"]
    ]
    assert len(error_recs) == 1, error_recs


def test_excepthook_chains_to_non_default_previous_hook(install_to_buffer, monkeypatch):
    """A pre-existing custom hook (e.g. a customer's own crash-reporting
    integration) must still fire after our ERROR summary — chain, don't
    overwrite, so cooperation works regardless of install order."""
    calls: list[tuple] = []

    def custom_previous_hook(exc_type, exc_value, exc_tb):
        calls.append((exc_type, exc_value, exc_tb))

    monkeypatch.setattr(sys, "excepthook", custom_previous_hook)
    buf = install_to_buffer()

    try:
        raise ValueError("boom-chain")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    assert len(calls) == 1, calls
    assert calls[0][0] is ValueError
    error_recs = [
        r for r in _records(buf) if r["level"] == "ERROR" and "boom-chain" in r["message"]
    ]
    assert len(error_recs) == 1, error_recs


def test_excepthook_skips_keyboard_interrupt(install_to_buffer):
    """Ctrl-C in local dev is not a crash — delegate without an ERROR
    record."""
    seen: list[type[BaseException]] = []

    def spy(exc_type, exc_value, exc_tb):
        seen.append(exc_type)

    # Interposed before our install, so our wrapper chains to this.
    sys.excepthook = spy
    buf = install_to_buffer()

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    assert not any(r["level"] == "ERROR" for r in _records(buf))
    assert seen == [KeyboardInterrupt], seen


def test_excepthook_logs_once_when_a_chaining_hook_interposes(install_to_buffer):
    """The real two-install lifecycle: the launcher installs (wrapper A),
    agent code installs its own reporter that chains to A, then
    ``TenantRuntime.__init__`` installs again — leaving two of our wrappers
    in one chain. The install guard only inspects the *current* hook, so it
    cannot see A behind the interposer; the re-entrancy guard is what keeps
    the crash to exactly one ERROR record."""
    buf = install_to_buffer()
    wrapper_a = sys.excepthook

    interposed: list[str] = []

    def chaining_hook(exc_type, exc_value, exc_tb):
        interposed.append(exc_type.__name__)
        wrapper_a(exc_type, exc_value, exc_tb)

    sys.excepthook = chaining_hook
    install_structured_logging(stream=buf)

    try:
        raise ValueError("boom-interposed")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    sys.excepthook(exc_type, exc_value, exc_tb)

    error_recs = [
        r for r in _records(buf) if r["level"] == "ERROR" and "boom-interposed" in r["message"]
    ]
    assert len(error_recs) == 1, error_recs
    # The interposer must still run — the fix must not achieve one record by
    # breaking the chain.
    assert interposed == ["ValueError"], interposed


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_excepthook_logs_every_concurrently_crashing_thread(install_to_buffer):
    """Two threads crashing at once must both get an ERROR record. The guard
    that suppresses a nested wrapper is per-thread, so one thread's crash must
    not silence another's."""
    # Chained hook holds the guard open long enough for the threads to overlap;
    # the real default hook does the same by rendering a traceback under a lock.
    threading.excepthook = lambda args: time.sleep(0.3)
    buf = install_to_buffer()

    gate = threading.Barrier(2)

    def boom(n: int) -> None:
        gate.wait()
        raise RuntimeError(f"concurrent-crash-{n}")

    threads = [threading.Thread(target=boom, args=(n,), name=f"worker-{n}") for n in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    crashes = sorted(
        r["fields"]["exc_message"]
        for r in _records(buf)
        if r["level"] == "ERROR" and "concurrent-crash-" in r["fields"].get("exc_message", "")
    )
    assert crashes == ["concurrent-crash-1", "concurrent-crash-2"], crashes


def test_thread_excepthook_skips_system_exit(install_to_buffer):
    """SystemExit in a thread means the thread wants to end, not a crash, so
    no ERROR record — but the previously installed hook must still see it.
    A custom monitoring hook may track thread termination, and swallowing the
    call would break the chaining guarantee."""
    seen: list[threading.ExceptHookArgs] = []
    # Interposed before our install, so our wrapper chains to this rather
    # than to pytest's own thread hook.
    threading.excepthook = seen.append
    buf = install_to_buffer()

    try:
        raise SystemExit(0)
    except SystemExit:
        exc_type, exc_value, exc_tb = sys.exc_info()

    args = threading.ExceptHookArgs((exc_type, exc_value, exc_tb, threading.current_thread()))
    threading.excepthook(args)

    assert not any(r["level"] == "ERROR" for r in _records(buf))
    assert len(seen) == 1, seen
    assert seen[0].exc_type is SystemExit


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_excepthook_logs_real_exception(install_to_buffer):
    """A genuine uncaught exception on a background thread yields one ERROR
    record naming the thread.

    Chaining to the previous hook means pytest's own thread-exception
    hook (installed ahead of ours) also fires here, which is exactly the
    chaining behavior under test — hence the filtered warning above."""
    buf = install_to_buffer()
    try:
        raise RuntimeError("thread-boom")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()

    fake_thread = threading.Thread(name="worker-1")
    args = threading.ExceptHookArgs((exc_type, exc_value, exc_tb, fake_thread))
    threading.excepthook(args)

    error_recs = [r for r in _records(buf) if r["level"] == "ERROR"]
    assert len(error_recs) == 1, error_recs
    assert "worker-1" in error_recs[0]["message"]
    assert error_recs[0]["fields"]["exc_type"] == "RuntimeError"


def test_asyncio_task_exception_already_resolves_to_error(install_to_buffer):
    """A realistic LangGraph-agent failure shape: an exception inside a
    fire-and-forget asyncio.Task reaches neither sys.excepthook nor
    threading.excepthook — asyncio routes it through its own
    default_exception_handler instead, which ends in a plain
    logger.error(..., exc_info=...) call on the "asyncio" logger. That
    should already resolve to level=ERROR via the existing, unmodified
    formatter with no new code from this fix involved. Confirms the path
    rather than assuming it — none of the excepthook work above would
    catch a regression here. Drives
    loop.call_exception_handler() directly (mirroring what
    asyncio.Task.__del__ does for an exception nobody ever retrieved) for
    a deterministic test, rather than relying on garbage-collection
    timing."""
    buf = install_to_buffer()

    async def _drive() -> None:
        loop = asyncio.get_running_loop()
        try:
            raise RuntimeError("orphaned-task-boom")
        except RuntimeError as exc:
            loop.call_exception_handler(
                {"message": "Task exception was never retrieved", "exception": exc}
            )

    asyncio.run(_drive())

    error_recs = [r for r in _records(buf) if r["level"] == "ERROR"]
    matches = [r for r in error_recs if r["fields"].get("exc_type") == "RuntimeError"]
    assert matches, error_recs
    assert "orphaned-task-boom" in matches[0]["fields"]["exc_message"]


def test_install_is_idempotent_does_not_recurse(install_to_buffer):
    """Calling install_structured_logging twice does not feed the handler
    back into its own LoggingStream (which would RecursionError)."""
    buf = install_to_buffer()
    install_structured_logging(stream=buf)
    logging.getLogger("runner_test").info("after-reinstall")

    rec = next(r for r in _records(buf) if r["message"] == "after-reinstall")
    assert rec["level"] == "INFO"


def test_logging_stream_is_thread_safe(install_to_buffer):
    """Concurrent writes do not interleave bytes mid-line."""
    import threading as th

    buf = install_to_buffer()
    barrier = th.Barrier(2)
    line_a = "thread-a-AAAAAAAAA"
    line_b = "thread-b-BBBBBBBBB"

    def writer(text: str) -> None:
        barrier.wait()
        for _ in range(50):
            sys.stdout.write(text + "\n")

    ta = th.Thread(target=writer, args=(line_a,))
    tb = th.Thread(target=writer, args=(line_b,))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    sys.stdout.flush()

    msgs = [r["message"] for r in _records(buf) if r["message"].startswith("thread-")]
    for m in msgs:
        assert m in (line_a, line_b), f"interleaved write produced {m!r}"


def test_aws_sdk_debug_wire_logs_suppressed_even_at_debug_level(monkeypatch):
    """botocore/boto3/s3transfer log full request/response bodies at DEBUG
    (e.g. a secretsmanager.GetSecretValue response containing plaintext
    credentials). A customer setting LOG_LEVEL=DEBUG for their own
    diagnostics must not cause those bodies to reach the log stream, since
    they flow straight into the platform's log pipeline and long-term
    storage otherwise."""
    monkeypatch.setenv("RUNNER_MODE", "aer")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    buf = io.StringIO()
    try:
        install_structured_logging(stream=buf, level="DEBUG")

        secret_payload = '{"SecretString": "{\\"voyage_api_key\\": \\"al-super-secret-value\\"}"}'
        logging.getLogger("botocore.endpoint").debug("Response body:\n%s", secret_payload)
        logging.getLogger("boto3.resources.action").debug("noisy boto3 debug line")
        logging.getLogger("s3transfer.utils").debug("noisy s3transfer debug line")

        text = buf.getvalue()
        assert "al-super-secret-value" not in text, text
        assert "SecretString" not in text, text
        assert not any(
            r["logger"].startswith(("botocore", "boto3", "s3transfer")) for r in _records(buf)
        )
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_install_honors_log_level_arg(monkeypatch):
    """`level` arg is honored so operators can flip LOG_LEVEL=DEBUG."""
    monkeypatch.setenv("RUNNER_MODE", "aer")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    buf = io.StringIO()
    try:
        install_structured_logging(stream=buf, level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        logging.getLogger("runner_test").debug("dbg")
        assert any(r["message"] == "dbg" for r in _records(buf)), (
            "DEBUG record should emit when level=DEBUG"
        )
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_install_diagnostic_is_debug_only(monkeypatch):
    """Framework setup noise stays hidden at the customer-facing INFO default."""
    monkeypatch.setenv("RUNNER_MODE", "aer")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    buf = io.StringIO()
    try:
        install_structured_logging(stream=buf, level="INFO")
        assert not any(r["message"] == "structured logging installed" for r in _records(buf))

        install_structured_logging(stream=buf, level="DEBUG")
        install_record = next(
            r for r in _records(buf) if r["message"] == "structured logging installed"
        )
        assert install_record["level"] == "DEBUG"
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_traceback_truncation_uses_byte_budget(install_to_buffer):
    """Truncation honors the byte budget, not str length, so non-ASCII
    tracebacks don't blow past the Fluent Bit batch ceiling."""
    from agent_engine_runner_shared import structured_logging as sl

    buf = install_to_buffer()
    # Each "字" is 3 bytes in UTF-8 — char-count truncation would let the
    # encoded payload reach ~3x the budget. Force a small budget so the
    # test runs fast, then assert the encoded length stays within it (plus
    # the truncation marker).
    big_msg = "字" * 4000
    try:
        raise RuntimeError(big_msg)
    except RuntimeError:
        logging.getLogger("runner_test").exception("caught")

    rec = next(r for r in _records(buf) if r["message"] == "caught")
    tb = rec["fields"]["exc_traceback"]
    assert tb.endswith("...(truncated)"), tb[-40:]
    body_bytes = tb.removesuffix("...(truncated)").encode("utf-8")
    assert len(body_bytes) <= sl._MAX_TRACEBACK_BYTES, len(body_bytes)


def test_captures_bypass_log_level_filter(monkeypatch):
    """stdout/stderr writes must emerge even when LOG_LEVEL is high.

    print() and writes to sys.stderr would normally bypass logging's level
    filtering entirely, since they don't call logger.* at all. After
    install they flow through a LoggingStream; if root.level/handler.level
    silently dropped them, an operator setting LOG_LEVEL=ERROR would lose
    every print() statement, which is a surprising behavior change.
    """
    monkeypatch.setenv("RUNNER_MODE", "aer")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    buf = io.StringIO()
    try:
        install_structured_logging(stream=buf, level="ERROR")
        # logger.info() should be filtered (operator asked for ERROR).
        logging.getLogger("runner_test").info("filtered-info")
        # but print() to stdout (captured at INFO) must still emit.
        sys.stdout.write("captured-print\n")
        sys.stdout.flush()
        sys.stderr.write("captured-stderr\n")
        sys.stderr.flush()

        msgs = [r["message"] for r in _records(buf)]
        assert "filtered-info" not in msgs, msgs
        assert "captured-print" in msgs, msgs
        assert "captured-stderr" in msgs, msgs
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_fileno_delegates_to_wrapped_stream(install_to_buffer):
    """fileno() must delegate so subprocess.run(stdout=sys.stdout) works."""
    install_to_buffer()
    # The fixture wraps the original sys.stdout; fileno() should reach it.
    fd = sys.stdout.fileno()
    assert isinstance(fd, int) and fd >= 0


def test_logging_stream_buffers_until_newline():
    """A LoggingStream emits exactly one record per newline-terminated line."""
    logger = logging.getLogger("stream_test")
    logger.handlers = []
    handler = logging.StreamHandler(io.StringIO())
    captured: List[logging.LogRecord] = []
    handler.emit = captured.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    stream = LoggingStream(logger, logging.INFO, "stdout")
    stream.write("partial ")
    assert captured == []
    stream.write("line\nsecond line\n")
    assert [r.getMessage() for r in captured] == ["partial line", "second line"]


def test_setup_logging_activates_structured_when_env_var_set(monkeypatch):
    """``setup_logging()`` delegates to ``install_structured_logging`` when
    ``STRUCTURED_LOGGING=true`` is set.

    Without this wiring the structured logging is dead code: ``install_structured_logging``
    has no caller in the platform, so the operator's pod templates can stamp
    ``STRUCTURED_LOGGING=true`` all day and still get the legacy human-
    readable formatter.
    """
    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        assert isinstance(sys.stdout, LoggingStream), type(sys.stdout)
        root = logging.getLogger()
        assert any(isinstance(h.formatter, StructuredJSONFormatter) for h in root.handlers), (
            root.handlers
        )
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_uses_legacy_formatter_when_env_var_unset(monkeypatch):
    """``setup_logging()`` keeps the legacy text formatter when the env var
    is unset, so existing services are unaffected."""
    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.delenv("STRUCTURED_LOGGING", raising=False)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        assert not isinstance(sys.stdout, LoggingStream), type(sys.stdout)
        root = logging.getLogger()
        assert not any(isinstance(h.formatter, StructuredJSONFormatter) for h in root.handlers)
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def _close_file_handlers() -> None:
    """Close rotating file handlers so tmp_path cleanup never hits an open fd."""
    from logging.handlers import TimedRotatingFileHandler

    for handler in logging.getLogger().handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            handler.close()


def test_setup_logging_structured_dev_modes_writes_log_file(monkeypatch, tmp_path):
    """``setup_logging()`` attaches a rotating file handler when
    ``STRUCTURED_LOGGING=true`` and ``AGENTIC_DEV_MODES`` is set.

    Without this, dev-up compose stacks running structured logging produce
    no on-disk copy, so the Local Dev UI log viewer shows nothing for those
    agents.
    """
    from logging.handlers import TimedRotatingFileHandler

    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1, root.handlers

        logging.getLogger("runner_test").info("dev file handler hello")

        log_file = tmp_path / "runner-aer.log"
        assert log_file.exists()
        # Human-readable text on disk, not the structured JSON contract.
        content = log_file.read_text()
        assert "dev file handler hello" in content
        assert "message" not in content  # JSON envelope would carry this key
    finally:
        _close_file_handlers()
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_structured_without_dev_modes_no_file_handler(monkeypatch, tmp_path):
    """Production regression guard: structured logging writes no disk copy
    when ``AGENTIC_DEV_MODES`` is unset — Fluent Bit owns persistence there,
    and a duplicate on-disk copy would waste IO."""
    from logging.handlers import TimedRotatingFileHandler

    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.delenv("AGENTIC_DEV_MODES", raising=False)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        root = logging.getLogger()
        assert not any(isinstance(h, TimedRotatingFileHandler) for h in root.handlers)
        assert not (tmp_path / "runner-aer.log").exists()
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_default_path_dev_modes_writes_log_file(monkeypatch, tmp_path):
    """The legacy (non-structured) path keeps writing the log file when dev
    modes are set — behavior that predates the dev-mode gate and must not
    regress."""
    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.delenv("STRUCTURED_LOGGING", raising=False)
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    snap = _snapshot_logging_state()
    try:
        setup_logging()
        logging.getLogger("runner_test").info("legacy path hello")
        log_file = tmp_path / "runner-aer.log"
        assert log_file.exists()
        assert "legacy path hello" in log_file.read_text()
    finally:
        _close_file_handlers()
        _restore_logging_state(snap)


def test_setup_logging_dev_modes_unwritable_log_dir_continues_stdout_only(monkeypatch, tmp_path):
    """An unusable LOG_DIR (parent path is a regular file) disables file
    logging with a warning instead of crashing the runner — file logging is
    a dev convenience, never a correctness requirement."""
    from logging.handlers import TimedRotatingFileHandler

    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    # A path under an existing regular file can never be mkdir'd as a dir.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("LOG_DIR", str(blocker / "logs"))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        root = logging.getLogger()
        assert not any(isinstance(h, TimedRotatingFileHandler) for h in root.handlers)
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_structured_dev_modes_repeat_call_keeps_one_file_handler(
    monkeypatch, tmp_path
):
    """Repeated ``setup_logging`` in structured dev mode must not stack file
    handlers.

    ``install_structured_logging`` owns the stdout handlers, but the dev file
    sink is ours — without dropping the prior one, every repeat call would
    duplicate each record in the file.
    """
    from logging.handlers import TimedRotatingFileHandler

    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging()
        setup_logging()

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1, root.handlers

        logging.getLogger("runner_test").info("repeated setup hello")
        content = (tmp_path / "runner-aer.log").read_text()
        assert content.count("repeated setup hello") == 1
    finally:
        _close_file_handlers()
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_structured_dev_modes_normalizes_lowercase_level(monkeypatch, tmp_path):
    """A lowercase ``log_level`` must still reach the dev file handler.

    The structured stdout path resolves levels case-insensitively; a
    ``getattr(logging, "debug")`` miss here would silently attach the file
    handler at INFO and filter records stdout still shows.
    """
    from logging.handlers import TimedRotatingFileHandler

    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging(log_level="debug")
        file_handlers = [
            h for h in logging.getLogger().handlers if isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG
    finally:
        _close_file_handlers()
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_setup_logging_dev_modes_sanitizes_app_name_paths(monkeypatch, tmp_path):
    """A caller-supplied app_name with path separators must not escape LOG_DIR.

    ``setup_logging(app_name=...)`` is public, so a ``../`` in the name would
    otherwise drop the log file outside the log directory — and in a
    ``dev up --all`` stack the sibling containers share the ``/app`` bind
    mount. Parity with the TypeScript sanitizeFileComponent test.
    """
    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENTIC_DEV_MODES", "aer,tool")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging(app_name="../escape")
        logging.getLogger("runner_test").info("sanitized hello")

        inside = tmp_path / ".._escape-aer.log"
        assert inside.exists()
        assert "sanitized hello" in inside.read_text()
        assert not (tmp_path.parent / "escape-aer.log").exists()
    finally:
        _close_file_handlers()
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_project_id_null_when_env_unset(monkeypatch):
    """projectId is null when PROJECT_ID env var is unset.

    The env-set path is covered by test_formatter_emits_design_doc_contract.
    This guards the negative-path so a future refactor that hard-codes a
    fallback ("unknown", empty string, etc.) can't slip through.
    """
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.delenv("PROJECT_ID", raising=False)
    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="m",
        args=None,
        exc_info=None,
    )
    assert json.loads(formatter.format(record))["projectId"] is None


def test_mode_arg_overrides_runner_mode_env(monkeypatch):
    """Explicit ``mode=`` arg wins over ``RUNNER_MODE`` env so callers like
    setup_logging(mode=...) stay authoritative when pod env is unset or
    misconfigured."""
    monkeypatch.delenv("RUNNER_MODE", raising=False)
    formatter = StructuredJSONFormatter(mode="aer")
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="m",
        args=None,
        exc_info=None,
    )
    rec = json.loads(formatter.format(record))
    assert rec["service"] == "agent-execution-runtime"
    assert rec["fields"]["component"] == "aer"

    # Explicit arg also wins when env *is* set but disagrees.
    monkeypatch.setenv("RUNNER_MODE", "orchestrator")
    formatter_override = StructuredJSONFormatter(mode="tool")
    rec = json.loads(formatter_override.format(record))
    assert rec["service"] == "tool-executor"


def test_setup_logging_threads_mode_to_structured_install(monkeypatch):
    """``setup_logging(mode="aer")`` produces ``service=agent-execution-runtime``
    even when ``RUNNER_MODE`` env is unset — guards against a silent
    pod-template misconfiguration where the env var is missing."""
    from agent_engine_runner_shared.utils import setup_logging

    monkeypatch.setenv("STRUCTURED_LOGGING", "true")
    monkeypatch.delenv("RUNNER_MODE", raising=False)

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    snap = _snapshot_logging_state()
    try:
        setup_logging(mode="aer")
        formatter = next(
            h.formatter
            for h in logging.getLogger().handlers
            if isinstance(h.formatter, StructuredJSONFormatter)
        )
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="m",
            args=None,
            exc_info=None,
        )
        rec = json.loads(formatter.format(record))
        assert rec["service"] == "agent-execution-runtime"
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        _restore_logging_state(snap)


def test_logging_stream_caps_unbounded_buffer():
    """A producer writing without ``\\n`` cannot grow ``_buffer`` past the
    cap — it gets force-emitted with a truncation marker and the buffer
    is reset."""
    from agent_engine_runner_shared import structured_logging as sl

    logger = logging.getLogger("buffer_cap_test")
    logger.handlers = []
    captured: List[logging.LogRecord] = []
    handler = logging.StreamHandler(io.StringIO())
    handler.emit = captured.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    stream = LoggingStream(logger, logging.INFO, "stdout")
    stream.write("x" * (sl._MAX_BUFFER_BYTES + 4096))

    assert len(captured) == 1, captured
    msg = captured[0].getMessage()
    assert msg.endswith("...(truncated)"), msg[-40:]
    assert len(msg.encode("utf-8")) <= sl._MAX_BUFFER_BYTES + len("...(truncated)")
    assert stream._buffer == ""


def test_unwrap_logging_stream_is_recursive():
    """Multiple wrap layers all get unwrapped — defensive against a future
    caller that hands a LoggingStream as the ``wrapped=`` arg."""
    from agent_engine_runner_shared.structured_logging import _unwrap_logging_stream

    real = io.StringIO()
    inner = LoggingStream(logging.getLogger(), logging.INFO, "stdout", wrapped=real)
    outer = LoggingStream(logging.getLogger(), logging.INFO, "stdout", wrapped=inner)
    assert _unwrap_logging_stream(outer) is real


def test_runner_mode_maps_to_service_and_component(monkeypatch):
    """Each RUNNER_MODE value maps to its design-doc service identifier."""
    cases = {
        "orchestrator": "orchestration-engine",
        "aer": "agent-execution-runtime",
        "tool": "tool-executor",
        "tool_function": "tool-executor",
    }
    for mode, expected_service in cases.items():
        monkeypatch.setenv("RUNNER_MODE", mode)
        formatter = StructuredJSONFormatter()
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hi",
            args=None,
            exc_info=None,
        )
        rec = json.loads(formatter.format(record))
        assert rec["service"] == expected_service
        assert rec["fields"]["component"] == mode


def test_origin_absent_outside_customer_scope(install_to_buffer):
    buf = install_to_buffer()
    logging.getLogger("runner_test").info("platform line")
    rec = next(r for r in _records(buf) if r["message"] == "platform line")
    assert "origin" not in rec


def test_origin_customer_inside_scope(install_to_buffer):
    buf = install_to_buffer()
    with customer_origin_scope():
        logging.getLogger("runner_test").info("customer line")
        sys.stdout.write("printed customer\n")
        sys.stdout.flush()
    logging.getLogger("runner_test").info("after scope")

    recs = {r["message"]: r for r in _records(buf)}
    assert recs["customer line"]["origin"] == "customer"
    assert recs["printed customer"]["origin"] == "customer"
    assert "origin" not in recs["after scope"]


def test_aclose_stream_iter_attributes_customer_cleanup_logs(install_to_buffer):
    """Generator cleanup runs during aclose and must carry origin=customer."""
    from agent_engine_runner_shared.server.aer import _aclose_stream_iter

    buf = install_to_buffer()

    async def _events():
        try:
            yield "event"
        finally:
            logging.getLogger("runner_test").info("cleanup line")

    async def _run() -> None:
        gen = _events()
        await gen.__anext__()
        await _aclose_stream_iter(gen, "exec-1")

    asyncio.run(_run())
    cleanup = next(r for r in _records(buf) if r["message"] == "cleanup line")
    assert cleanup["origin"] == "customer"
