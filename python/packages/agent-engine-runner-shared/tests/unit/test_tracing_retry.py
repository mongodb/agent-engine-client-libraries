"""
Unit tests for the trace-store retry / attach / status lifecycle.

When the tenant store DB is unreachable at AER pod startup, a background retry
re-connects and attaches the MongoDB span processor once the store recovers —
without a pod restart. The failure
surfaces via tracing_status() (-> /health DEGRADED) rather than a single
swallowed warning.
"""

import logging
import threading
from unittest.mock import MagicMock

import pytest

import agent_engine_runner_shared.tracing.setup as tracing_setup
from agent_engine_runner_shared.tracing import scrub_credentials


@pytest.fixture
def isolated_tracer_provider(monkeypatch):
    """Ensure setup_tracing()'s module-level singletons don't leak across tests."""
    monkeypatch.setattr(tracing_setup, "_tracer_provider", None)
    monkeypatch.setattr(tracing_setup, "_mongodb_exporter_attached", False)
    monkeypatch.setattr(tracing_setup, "_mongodb_degraded", False)
    monkeypatch.setattr(tracing_setup, "_retry_stop_event", None)
    monkeypatch.setattr(tracing_setup, "_retry_thread", None)
    monkeypatch.setattr(tracing_setup, "_tracing_mongo_client", None)
    yield tracing_setup
    # shutdown_tracing() signals and joins the retry thread itself.
    if tracing_setup._tracer_provider is not None or tracing_setup._retry_thread is not None:
        tracing_setup.shutdown_tracing()


# scrub_credentials runs on Mongo driver error messages, which echo the raw
# connection string. Every credential-bearing case must end with the whole
# userinfo masked — a password containing an unescaped '@' or whitespace
# defeated the old first-'@' regex and leaked to the centralized log sink.
# Cases mirror the TypeScript scrubCredentials tests (cross-language parity).
SCRUB_CASES = [
    (
        "connect to mongodb://user:password@host:27017/db failed: TLS",
        "connect to mongodb://***@host:27017/db failed: TLS",
    ),
    (
        "parse error: mongodb://user@ssword@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pass word@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pa/ss@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pass\nword@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        '{"error":"mongodb://u:p@h1/db","email":"a@b.com"}',
        '{"error":"mongodb://***@h1/db","email":"a@b.com"}',
    ),
    (
        "parse error: mongodb://user:p@ ss word@host/db tail",
        "parse error: mongodb://***@host/db tail",
    ),
    (
        "mongodb+srv://user@ss@host/db failed",
        "mongodb+srv://***@host/db failed",
    ),
    (
        "mongodb://host/db unreachable",
        "mongodb://host/db unreachable",
    ),
    (
        "a mongodb://u:p@h1/db b mongodb://u2:p2@h2/db2 c",
        "a mongodb://***@h1/db b mongodb://***@h2/db2 c",
    ),
    (
        "mongodb://u:p@h1:27017,h2:27017/db?replicaSet=rs",
        "mongodb://***@h1:27017,h2:27017/db?replicaSet=rs",
    ),
    (
        'Invalid connection string "mongodb://user:pass@host/db" foo',
        'Invalid connection string "mongodb://***@host/db" foo',
    ),
    (
        "MongoDB://User:P@ss@Host/db",
        "MongoDB://***@Host/db",
    ),
    (
        "something else entirely",
        "something else entirely",
    ),
]


@pytest.mark.parametrize("raw,expected", SCRUB_CASES)
def test_scrub_credentials_masks_uri_userinfo(raw, expected):
    assert scrub_credentials(raw) == expected


@pytest.mark.parametrize(
    "password", ["p@ss", "pass word", "pa/ss", "p@ ss word", "p:a@s/s w", "pass\nword"]
)
def test_scrub_credentials_no_fragment_survives(password):
    out = scrub_credentials(f"err mongodb://user:{password}@host/db")
    assert out == "err mongodb://***@host/db"
    assert password not in out
    assert "user" not in out


def test_retry_loop_log_masks_credentials_in_error(monkeypatch, isolated_tracer_provider, tmp_path):
    """Regression: a retry failure whose message echoes a URI with '@'/space in
    the password must log only the masked form."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-scrub-retry")

    # Emission-driven sync (no wall-clock polling): the handler signals
    # exactly when the retry's failure record is emitted, so the test can't
    # miss the retry thread under load. The timeout bounds only the
    # "never logged" failure case — it is not a timing assumption.
    record_seen = threading.Event()
    seen_messages: list[str] = []

    class SignalHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Trace store retry failed" in record.getMessage():
                seen_messages.append(record.getMessage())
                record_seen.set()

    def fail_with_uri():
        raise RuntimeError(
            "Invalid connection string: mongodb://admin:p@ss word@atlas-host.mongodb.net/db"
        )

    module_logger = tracing_setup.logger
    previous_level = module_logger.level
    handler = SignalHandler(level=logging.DEBUG)
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.DEBUG)
    try:
        isolated_tracer_provider.start_mongodb_tracing_retry(fail_with_uri, interval=0.01)
        assert record_seen.wait(timeout=5.0), "retry failure was not logged"
    finally:
        module_logger.removeHandler(handler)
        module_logger.setLevel(previous_level)

    assert seen_messages
    for msg in seen_messages:
        assert "mongodb://***@atlas-host.mongodb.net/db" in msg
        assert "p@ss" not in msg
        assert "admin:" not in msg


def test_tracing_status_disabled_by_default(monkeypatch, isolated_tracer_provider, tmp_path):
    """No store URI configured -> database_exporter is "disabled", not degraded."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-no-store")
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "disabled"}


def test_tracing_status_attached_when_collection_provided(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """Store reachable at startup -> status "attached" (behavior unchanged)."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    collection = MagicMock()
    isolated_tracer_provider.setup_tracing(
        service_name="svc-attached", mongodb_collection=collection
    )
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "attached"}


def test_tracing_status_degraded_after_retry_starts(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """start_mongodb_tracing_retry marks the store degraded until attached."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-degraded")
    isolated_tracer_provider.start_mongodb_tracing_retry(lambda: None, interval=999)
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "degraded"}


def test_attach_mongodb_tracing_routes_new_spans_to_store(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """A late attach (store recovered after startup failure) routes new spans to Mongo."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-late-attach")
    isolated_tracer_provider.start_mongodb_tracing_retry(lambda: None, interval=999)
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "degraded"}

    collection = MagicMock()
    isolated_tracer_provider.attach_mongodb_tracing(collection)
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "attached"}

    provider = isolated_tracer_provider._tracer_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("late-attach-span"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert collection.insert_many.called
    span_names = {
        doc["name"] for call in collection.insert_many.call_args_list for doc in call[0][0]
    }
    assert "late-attach-span" in span_names


def test_attach_mongodb_tracing_is_idempotent(monkeypatch, isolated_tracer_provider, tmp_path):
    """A second attach is a no-op — the first collection stays the export target."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    collection = MagicMock()
    isolated_tracer_provider.setup_tracing(
        service_name="svc-idempotent", mongodb_collection=collection
    )
    second_collection = MagicMock()
    isolated_tracer_provider.attach_mongodb_tracing(second_collection)

    provider = isolated_tracer_provider._tracer_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("idempotent-span"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert collection.insert_many.called
    assert not second_collection.insert_many.called


def test_start_mongodb_tracing_retry_attaches_on_recovery(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """The background retry connects once the store recovers and attaches the exporter."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-retry")

    collection = MagicMock()
    connect = MagicMock(return_value=collection)
    # Event-driven sync instead of wall-clock polling: the retry thread
    # signals when its attach runs.
    attached = threading.Event()
    real_attach = isolated_tracer_provider.attach_mongodb_tracing

    def attach_and_signal(coll):
        result = real_attach(coll)
        attached.set()
        return result

    monkeypatch.setattr(isolated_tracer_provider, "attach_mongodb_tracing", attach_and_signal)
    isolated_tracer_provider.start_mongodb_tracing_retry(connect, interval=0.01)
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "degraded"}

    assert attached.wait(timeout=2.0), "retry did not attach within 2s"
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "attached"}
    assert connect.call_count >= 1
    # A span emitted after recovery routes to the store.
    provider = isolated_tracer_provider._tracer_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("recovered-span"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert collection.insert_many.called


def test_start_mongodb_tracing_retry_keeps_degraded_when_store_stays_down(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """If the store never recovers, status stays "degraded" and no exporter attaches."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-stays-down")

    attempt_count = 0
    third_attempt = threading.Event()

    def always_fail():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count >= 3:
            third_attempt.set()
        raise RuntimeError("still unreachable")

    isolated_tracer_provider.start_mongodb_tracing_retry(always_fail, interval=0.01)
    # Sync on the retry attempts themselves, not wall time: several attempts
    # firing proves the loop is live while the status stays degraded.
    assert third_attempt.wait(timeout=2.0), "retry loop did not fire within 2s"

    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "degraded"}
    # shutdown_tracing() (in the fixture) signals the retry thread to stop.


def test_retry_interval_defaults_to_30s(monkeypatch):
    """Unset AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS -> 30s default."""
    monkeypatch.delenv("AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS", raising=False)
    assert tracing_setup._get_retry_interval_seconds() == 30.0


@pytest.mark.parametrize("value", ["", "abc", "0", "-5", "inf", "nan"])
def test_retry_interval_rejects_invalid_env(monkeypatch, value):
    """Unparsable, non-positive, and non-finite values fall back to 30s."""
    monkeypatch.setenv("AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS", value)
    assert tracing_setup._get_retry_interval_seconds() == 30.0


def test_retry_interval_uses_env_when_positive(monkeypatch):
    monkeypatch.setenv("AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS", "0.5")
    assert tracing_setup._get_retry_interval_seconds() == 0.5


def test_env_interval_drives_retry_cadence(monkeypatch, isolated_tracer_provider, tmp_path):
    """start_mongodb_tracing_retry without an explicit interval honors the env var.

    At the 0.01s env cadence several attempts fire within the window; with
    the 30s default the first attempt would not have fired at all.
    """
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS", "0.01")
    isolated_tracer_provider.setup_tracing(service_name="svc-env-cadence")

    attempt_count = 0
    third_attempt = threading.Event()

    def always_fail():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count >= 3:
            third_attempt.set()
        raise RuntimeError("still unreachable")

    isolated_tracer_provider.start_mongodb_tracing_retry(always_fail)  # no interval arg
    assert third_attempt.wait(timeout=2.0), "retry loop ignored the env cadence"
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "degraded"}


def test_shutdown_during_in_flight_retry_stops_further_retries(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """A retry released as failed after shutdown must not reschedule.

    Mirrors the TypeScript test of the same name: the retry's connect suspends
    on the loop's own stop event (deterministic — no wall-clock settle), so the
    attempt is in-flight when shutdown runs.
    """
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-race-fail")

    attempt_count = 0
    connect_started = threading.Event()

    def blocking_connect():
        nonlocal attempt_count
        attempt_count += 1
        # Capture the stop event BEFORE signaling: shutdown_tracing() sets the
        # event and then clears the module global, so reading the global after
        # connect_started would race None.wait().
        stop_event = isolated_tracer_provider._retry_stop_event
        connect_started.set()
        # Block until shutdown sets the loop's stop event, then fail.
        stop_event.wait(timeout=2.0)
        raise RuntimeError("still down")

    isolated_tracer_provider.start_mongodb_tracing_retry(blocking_connect, interval=0.01)
    assert connect_started.wait(timeout=2.0), "retry attempt did not start"

    isolated_tracer_provider.shutdown_tracing()  # signals and joins the in-flight thread

    # The thread exited with shutdown's join — no reschedule is possible.
    assert attempt_count == 1
    assert isolated_tracer_provider._retry_thread is None


def test_shutdown_during_in_flight_retry_closes_client_it_opened(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    """A retry that connects as shutdown runs closes the client, never attaches.

    Same race as above, but the in-flight attempt SUCCEEDS after shutdown.
    Attaching now would target a torn-down (or later, reinitialized) provider —
    and skipping the close would leak the MongoClient the attempt opened.
    """
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-race-success")

    connect_started = threading.Event()
    collection = MagicMock()

    def blocking_connect():
        # Capture the stop event BEFORE signaling: shutdown_tracing() sets the
        # event and then clears the module global, so reading the global after
        # connect_started would race None.wait().
        stop_event = isolated_tracer_provider._retry_stop_event
        connect_started.set()
        # Return a live collection only after shutdown sets the stop event.
        stop_event.wait(timeout=2.0)
        return collection

    isolated_tracer_provider.start_mongodb_tracing_retry(blocking_connect, interval=0.01)
    assert connect_started.wait(timeout=2.0), "retry attempt did not start"

    isolated_tracer_provider.shutdown_tracing()

    collection.database.client.close.assert_called_once()
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "disabled"}


def test_shutdown_closes_retry_recovered_client(monkeypatch, isolated_tracer_provider, tmp_path):
    """A client the retry connected and attached is closed by shutdown_tracing.

    Successful recovery transfers client ownership to the tracing module
    (mirroring the TypeScript tracingMongoClient lifecycle): the provider is
    flushed first — its final export still needs the pool — then the client
    closes.
    """
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-recovered-close")

    collection = MagicMock()
    attached = threading.Event()
    real_attach = isolated_tracer_provider.attach_mongodb_tracing

    def attach_and_signal(coll):
        result = real_attach(coll)
        attached.set()
        return result

    monkeypatch.setattr(isolated_tracer_provider, "attach_mongodb_tracing", attach_and_signal)
    isolated_tracer_provider.start_mongodb_tracing_retry(
        MagicMock(return_value=collection), interval=0.01
    )
    assert attached.wait(timeout=2.0), "retry did not attach within 2s"
    assert isolated_tracer_provider.tracing_status() == {"database_exporter": "attached"}

    isolated_tracer_provider.shutdown_tracing()

    collection.database.client.close.assert_called_once()
    assert isolated_tracer_provider._tracing_mongo_client is None


def test_shutdown_leaves_caller_owned_client_open(monkeypatch, isolated_tracer_provider, tmp_path):
    """setup_tracing is non-owning by default: shutdown_tracing flushes the
    provider but never closes the client behind a caller-provided collection —
    it may be shared with application code."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    collection = MagicMock()
    isolated_tracer_provider.setup_tracing(
        service_name="svc-caller-owned", mongodb_collection=collection
    )

    isolated_tracer_provider.shutdown_tracing()

    collection.database.client.close.assert_not_called()
    assert isolated_tracer_provider._tracing_mongo_client is None


def test_shutdown_closes_owned_startup_client(monkeypatch, isolated_tracer_provider, tmp_path):
    """take_client_ownership=True (the runtime's dedicated trace-store client)
    is closed by shutdown_tracing, after the provider flush."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    collection = MagicMock()
    isolated_tracer_provider.setup_tracing(
        service_name="svc-startup-close", mongodb_collection=collection, take_client_ownership=True
    )

    isolated_tracer_provider.shutdown_tracing()

    collection.database.client.close.assert_called_once()
    assert isolated_tracer_provider._tracing_mongo_client is None
