"""Tests for the process-wide shared OE callback client in tls_client."""

from __future__ import annotations

import datetime
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from agent_engine_runner_shared import tls_client
from agent_engine_runner_shared.owner_callback import (
    MAX_DISCARD_BYTES,
    owner_delivered,
    post_without_reading_response_body,
)
from agent_engine_runner_shared.tls_client import (
    _MAX_SHARED_CLIENTS,
    _SHARED_CLIENT_LIMITS,
    _get_cert_mtimes,
    _shared_clients,
    _shared_clients_lock,
    close_shared_clients,
    create_httpx_client_with_tls,
    get_shared_client,
)

_TLS_ENV_VARS = [
    f"TLS_{kind}_{suffix}" for kind in ("CERT", "KEY", "CA_CERT") for suffix in ("PATH", "PEM")
]


@pytest.fixture(autouse=True)
def _clear_tls_env(monkeypatch):
    for name in _TLS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_real_certs(directory: Path) -> dict[str, Path]:
    """Write a self-signed CA plus a client cert/key signed by it."""
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "runner")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(ca_key, hashes.SHA256())
    )
    paths = {
        "cert": directory / "tls.crt",
        "key": directory / "tls.key",
        "ca": directory / "ca.crt",
    }
    paths["cert"].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    paths["ca"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    paths["key"].write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return paths


@pytest.fixture
def real_cert_env(tmp_path, monkeypatch) -> dict[str, Path]:
    paths = _write_real_certs(tmp_path)
    monkeypatch.setenv("TLS_CERT_PATH", str(paths["cert"]))
    monkeypatch.setenv("TLS_KEY_PATH", str(paths["key"]))
    monkeypatch.setenv("TLS_CA_CERT_PATH", str(paths["ca"]))
    return paths


@pytest.fixture
def counting_factory(monkeypatch):
    """Wrap the real factory so creations can be counted and inspected."""
    calls: list[tuple[str, dict]] = []
    real = create_httpx_client_with_tls

    def wrapper(base_url, timeout, **kwargs):
        calls.append((base_url, kwargs))
        return real(base_url, timeout, **kwargs)

    monkeypatch.setattr(tls_client, "create_httpx_client_with_tls", wrapper)
    return calls


class TestOriginKeying:
    def test_same_origin_variants_share_one_client(self, counting_factory):
        a = get_shared_client("http://oe:8000")
        b = get_shared_client("http://oe:8000/")
        c = get_shared_client("http://oe:8000/tool/execute")
        assert a is b is c
        assert len(counting_factory) == 1
        assert counting_factory[0][1]["limits"] is _SHARED_CLIENT_LIMITS

    def test_different_host_or_port_gets_different_client(self, counting_factory):
        base = get_shared_client("http://oe:8000")
        other_host = get_shared_client("http://oe-0.oe:8000")
        other_port = get_shared_client("http://oe:9000")
        assert len({id(base), id(other_host), id(other_port)}) == 3
        assert len(counting_factory) == 3
        assert all(kwargs["limits"] is _SHARED_CLIENT_LIMITS for _, kwargs in counting_factory)

    def test_verify_hostname_partition_is_cached_separately(self, real_cert_env, counting_factory):
        # The cache key partitions on verify_hostname, so a hostname-verification
        # -disabled client is never confused with a verifying one for the same
        # origin — and vice versa.
        verifying = get_shared_client("https://oe:8443", verify_hostname=True)
        noverify = get_shared_client("https://oe:8443", verify_hostname=False)

        assert verifying is not noverify
        assert len(counting_factory) == 2
        assert set(_shared_clients) == {"https://oe:8443", "https://oe:8443#noverify"}

        # Each mode is cached independently under its own key.
        assert get_shared_client("https://oe:8443", verify_hostname=True) is verifying
        assert get_shared_client("https://oe:8443", verify_hostname=False) is noverify
        assert len(counting_factory) == 2

    @pytest.mark.parametrize("url", ["/tool/execute", "oe:8000", "", "not a url"])
    def test_relative_or_invalid_url_raises(self, url):
        with pytest.raises(ValueError):
            get_shared_client(url)
        assert not _shared_clients

    @pytest.mark.parametrize("url", ["ws://oe:8000", "ftp://oe:8000", "grpc://oe:8000"])
    def test_non_http_scheme_raises(self, url):
        # The default-port mapping assumes http(s); another scheme would be keyed
        # under port 80 and could collide with a distinct origin.
        with pytest.raises(ValueError, match=r"http\(s\)"):
            get_shared_client(url)
        assert not _shared_clients


class TestRotationAndFork:
    def test_cert_mtime_change_rebuilds_and_closes_old(self, real_cert_env):
        old = get_shared_client("https://oe:8443")
        assert get_shared_client("https://oe:8443") is old

        st = os.stat(real_cert_env["cert"])
        os.utime(real_cert_env["cert"], (st.st_atime + 10, st.st_mtime + 10))

        new = get_shared_client("https://oe:8443")
        assert new is not old
        assert old.is_closed
        assert not new.is_closed
        assert get_shared_client("https://oe:8443") is new

    def test_cert_mtimes_are_read_under_the_lock(self, real_cert_env, monkeypatch):
        """Regression: the rotation check must read mtimes while holding the lock.

        A snapshot taken before acquiring the lock can be stale relative to an
        entry another thread has already rebuilt; comparing that stale snapshot
        against the fresh entry closes the client the other thread is actively
        using. ``_shared_clients_lock`` is a plain (non-reentrant) ``Lock``, so a
        non-blocking acquire succeeds only when the current thread does not
        already hold it — which makes "read under the lock" observable.
        """
        read_under_lock: list[bool] = []
        original = tls_client._get_cert_mtimes

        def spy():
            acquired = _shared_clients_lock.acquire(blocking=False)
            read_under_lock.append(not acquired)
            if acquired:
                _shared_clients_lock.release()
            return original()

        monkeypatch.setattr(tls_client, "_get_cert_mtimes", spy)
        get_shared_client("https://oe:8443")

        assert read_under_lock == [True]

    def test_pem_mode_has_no_mtimes_and_is_stable(self, monkeypatch, tmp_path):
        paths = _write_real_certs(tmp_path)
        monkeypatch.setattr(tls_client, "_decoded_cert_paths", {})
        monkeypatch.setenv("TLS_CERT_PEM", paths["cert"].read_text())
        monkeypatch.setenv("TLS_KEY_PEM", paths["key"].read_text())
        monkeypatch.setenv("TLS_CA_CERT_PEM", paths["ca"].read_text())

        assert _get_cert_mtimes() is None
        first = get_shared_client("https://oe:8443")
        assert get_shared_client("https://oe:8443") is first
        assert not first.is_closed

    def test_pid_change_rebuilds_without_closing_parent_client(self, monkeypatch):
        parent = get_shared_client("http://oe:8000")
        real_pid = os.getpid()
        monkeypatch.setattr(tls_client.os, "getpid", lambda: real_pid + 1)

        child = get_shared_client("http://oe:8000")
        assert child is not parent
        assert parent.is_closed is False
        parent.close()

    def test_https_without_certs_fails_closed_every_call(self):
        for _ in range(3):
            with pytest.raises(ValueError, match="fail-closed"):
                get_shared_client("https://oe:8443")
        assert not _shared_clients

    def test_uppercase_https_scheme_fails_closed_without_certs(self):
        # The key normalizes the scheme, so the TLS decision must too; otherwise
        # a certless client would be cached under the https key (fail-open) and
        # served to a later canonical https call.
        with pytest.raises(ValueError, match="fail-closed"):
            get_shared_client("HTTPS://oe:8443")
        assert not _shared_clients

    def test_vanished_cert_file_fails_closed_instead_of_reusing_cache(self, real_cert_env):
        client = get_shared_client("https://oe:8443")
        assert get_shared_client("https://oe:8443") is client

        # A configured cert file disappearing must drop the cached client and
        # rebuild (which fails closed), not silently reuse stale TLS material.
        os.remove(real_cert_env["cert"])

        with pytest.raises(ValueError, match="mTLS"):
            get_shared_client("https://oe:8443")
        assert client.is_closed
        assert not _shared_clients

    def test_transient_cert_stat_failure_keeps_cached_client(self, real_cert_env, monkeypatch):
        # A momentary stat failure (a kubelet symlink swap, a mount flip) must not
        # drop a working client: the stat is retried before failing closed.
        client = get_shared_client("https://oe:8443")
        assert get_shared_client("https://oe:8443") is client

        real_getmtime = os.path.getmtime
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 1:  # the first stat attempt fails; the retry succeeds
                raise OSError("transient stat failure")
            return real_getmtime(path)

        monkeypatch.setattr(os.path, "getmtime", flaky)

        assert get_shared_client("https://oe:8443") is client
        assert not client.is_closed
        assert calls["n"] > 1  # the retry actually ran


# The at-fork re-arm can only be observed across a real ``fork()``. Run the whole
# scenario in a subprocess so the fork never happens inside the pytest process,
# where it can collide with coverage/plugin fork handlers and at-fork ordering.
_FORK_REARM_SCRIPT = """
import os
import signal
import sys

from agent_engine_runner_shared import tls_client

# Seed the cache so the child has an inherited entry to drop.
tls_client.get_shared_client("http://oe:8000")

tls_client._shared_clients_lock.acquire()
try:
    pid = os.fork()
    if pid == 0:
        # Bound the child: without the at-fork handler the inherited lock is
        # held forever (its owner thread does not exist here), so the child
        # would hang until this alarm kills it. The child deliberately does no
        # httpx work -- building a client in a forked child is not fork-safe.
        signal.alarm(5)
        try:
            # Resolve the module attribute: the at-fork handler rebinds it to a
            # fresh lock and clears the inherited entries.
            tls_client._shared_clients_lock.acquire()
            tls_client._shared_clients_lock.release()
            os._exit(0 if not tls_client._shared_clients else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
finally:
    tls_client._shared_clients_lock.release()
sys.exit(os.waitstatus_to_exitcode(status))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_fork_while_lock_held_re_arms_lock_and_cache():
    result = subprocess.run(
        [sys.executable, "-c", _FORK_REARM_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_close_shared_clients_closes_and_empties():
    clients = [get_shared_client("http://oe:8000"), get_shared_client("http://oe:9000")]
    assert len(_shared_clients) == 2
    close_shared_clients()
    assert not _shared_clients
    assert all(c.is_closed for c in clients)


class TestBoundedCache:
    def test_cache_evicts_and_closes_least_recently_used(self):
        urls = [f"http://oe-{i}:8000" for i in range(_MAX_SHARED_CLIENTS + 1)]
        clients = [get_shared_client(u) for u in urls]

        assert len(_shared_clients) == _MAX_SHARED_CLIENTS
        # The first origin (least recently used) was evicted and closed.
        assert urls[0] not in _shared_clients
        assert clients[0].is_closed
        assert not clients[-1].is_closed
        # Re-resolving the evicted origin builds a fresh client, never the closed one.
        assert get_shared_client(urls[0]) is not clients[0]

    def test_recently_used_origin_survives_eviction(self):
        urls = [f"http://oe-{i}:8000" for i in range(_MAX_SHARED_CLIENTS)]
        clients = [get_shared_client(u) for u in urls]

        # Touch the oldest so it becomes most-recently-used, then overflow the cap.
        assert get_shared_client(urls[0]) is clients[0]
        get_shared_client("http://oe-extra:8000")

        assert urls[0] in _shared_clients
        assert not clients[0].is_closed
        # The next-least-recently-used origin is evicted instead.
        assert urls[1] not in _shared_clients
        assert clients[1].is_closed


def test_concurrent_first_use_creates_one_client(counting_factory):
    barrier = threading.Barrier(16)
    results: list[httpx.Client] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        client = get_shared_client("http://oe:8000")
        with lock:
            results.append(client)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(counting_factory) == 1
    assert len(results) == 16
    assert all(c is results[0] for c in results)


# ---------------------------------------------------------------------------
# End-to-end connection reuse against a real HTTP/1.1 server
# ---------------------------------------------------------------------------


class _CountingServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.connections = 0
        self.requests = 0
        self.large_body = False
        self.counter_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _CountingHandler)


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _CountingServer

    def setup(self):
        super().setup()
        with self.server.counter_lock:
            self.server.connections += 1

    def do_POST(self):  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        with self.server.counter_lock:
            self.server.requests += 1
        if self.server.large_body:
            body = b"x" * (MAX_DISCARD_BYTES * 2)
        else:
            body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):  # noqa: A002 - silence test output
        pass


@pytest.fixture
def counting_server() -> Iterator[_CountingServer]:
    server = _CountingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _url(server: _CountingServer, path: str = "/tool/result") -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


class TestConnectionReuse:
    def test_shared_client_reuses_one_connection(self, counting_server):
        url = _url(counting_server)
        client = get_shared_client(url)
        for i in range(4):
            response = post_without_reading_response_body(client, url, {"i": i}, timeout=5.0)
            assert response.status_code == 200
        for i in range(2):
            assert owner_delivered(client, url, {"o": i}, lambda _m: None, timeout=5.0)

        assert counting_server.requests == 6
        assert counting_server.connections == 1

    def test_per_call_client_opens_connection_per_request(self, counting_server):
        url = _url(counting_server)
        for i in range(4):
            with create_httpx_client_with_tls(url, 5.0) as client:
                post_without_reading_response_body(client, url, {"i": i}, timeout=5.0)

        assert counting_server.requests == 4
        assert counting_server.connections == 4

    def test_oversized_body_closes_connection_instead_of_pooling(self, counting_server):
        url = _url(counting_server)
        client = get_shared_client(url)

        post_without_reading_response_body(client, url, {}, timeout=5.0)
        assert counting_server.connections == 1

        counting_server.large_body = True
        response = post_without_reading_response_body(client, url, {}, timeout=5.0)
        assert response.status_code == 200

        counting_server.large_body = False
        post_without_reading_response_body(client, url, {}, timeout=5.0)
        assert counting_server.requests == 3
        assert counting_server.connections == 2


class TestPerRequestTimeout:
    def test_explicit_timeout_applied_to_request(self):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, json={})

        with httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0) as client:
            post_without_reading_response_body(client, "http://oe:8000/x", {}, timeout=7.0)

        assert seen[0]["read"] == 7.0

    def test_default_keeps_client_timeout(self):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, json={})

        with httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0) as client:
            post_without_reading_response_body(client, "http://oe:8000/x", {})

        assert seen[0]["read"] == 30.0
        assert seen[0]["read"] is not None
