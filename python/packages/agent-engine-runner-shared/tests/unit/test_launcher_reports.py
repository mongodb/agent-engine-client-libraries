"""Exercise startup reporting through the launcher and real HTTP transport."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import ssl
import sys
import threading
import time
import types
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from agent_engine_runner_shared import launcher
from agent_engine_runner_shared.startup_state import mark_startup_complete

BOOT_ID = "6f1c0000-1234-4567-8901-000000000001"


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler], tls: ssl.SSLContext | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    if tls is not None:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def configure_launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, oe_url: str) -> None:
    monkeypatch.setenv("OE_URL", oe_url)
    monkeypatch.setenv("AGENTIC_BOOT_ID", BOOT_ID)
    monkeypatch.setenv("AGENTIC_MCP_OAUTH_DIR", str(tmp_path / "oauth"))
    monkeypatch.setattr(launcher, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "_TERMINATION_LOG_PATH", str(tmp_path / "termination"))


@pytest.fixture
def report_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    reports: list[tuple[str, dict]] = []
    behavior = {"status": 201, "delay": 0.0, "trickle": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reports.append((self.path, body))
            time.sleep(behavior["delay"])
            self.send_response(behavior["status"])
            self.send_header("Location", "/redirected")
            self.send_header("Content-Length", "100000")
            self.end_headers()
            if behavior["trickle"]:
                try:
                    for _ in range(30):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_args) -> None:
            pass

    with serve(Handler) as server:
        configure_launcher(monkeypatch, tmp_path, f"http://127.0.0.1:{server.server_port}")
        yield reports, behavior


def install_target(monkeypatch: pytest.MonkeyPatch, target) -> None:
    module = types.ModuleType("report_agent")
    module.main = target
    monkeypatch.setitem(sys.modules, "report_agent", module)
    monkeypatch.setenv("AGENT_ENTRYPOINT", "report_agent:main")


@pytest.mark.parametrize("failure,code", [("import", 82), ("entrypoint", 83), ("startup", 84)])
def test_one_report_before_classified_exit(report_server, monkeypatch, failure, code):
    reports, _ = report_server

    def crash():
        raise RuntimeError("MCP discovery rejected")

    install_target(monkeypatch, crash)
    if failure == "import":
        monkeypatch.setenv("AGENT_ENTRYPOINT", "missing_report_agent:main")
    elif failure == "entrypoint":
        monkeypatch.delenv("AGENT_ENTRYPOINT")
    with pytest.raises(SystemExit) as exit_info:
        launcher.main()
    assert exit_info.value.code == code
    assert len(reports) == 1
    path, body = reports[0]
    assert path == f"/boots/{BOOT_ID}/errors"
    assert body["exitCode"] == code
    assert body["sandbox"] == "agent"
    assert "Traceback" not in body["message"]


@pytest.mark.parametrize("missing", ["OE_URL", "AGENTIC_BOOT_ID"])
def test_skip_without_boot_configuration(report_server, monkeypatch, missing):
    reports, _ = report_server
    monkeypatch.delenv(missing)
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    with pytest.raises(SystemExit):
        launcher.main()
    assert reports == []


@pytest.mark.parametrize("status", [201, 307, 404, 409, 500])
def test_status_only_no_redirect_or_body_read(report_server, monkeypatch, status):
    reports, behavior = report_server
    behavior.update(status=status, trickle=True)
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main()
    assert exit_info.value.code == 83
    assert len(reports) == 1


def test_delayed_headers_respect_total_deadline(report_server, monkeypatch):
    reports, behavior = report_server
    behavior["delay"] = 10.0
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    started = time.monotonic()
    with pytest.raises(SystemExit):
        launcher.main()
    assert 1.8 < time.monotonic() - started < 6.0
    assert len(reports) == 1


def test_snapshot_scrubs_removed_oauth_and_tenant_values(report_server, monkeypatch):
    reports, _ = report_server
    token = "opaque-oauth-value"
    encoded = base64.b64encode(json.dumps({"access_token": token}).encode()).decode()
    monkeypatch.setenv("AGENTIC_MCP_OAUTH_B64_TEST", encoded)
    monkeypatch.setenv("TENANT_API_KEY", "opaque-tenant-value")
    monkeypatch.setenv("TENANT_DEBUG", "1")
    monkeypatch.setenv("TLS_KEY_PEM", "private-key-material")

    def materialize():
        monkeypatch.delenv("AGENTIC_MCP_OAUTH_B64_TEST")

    monkeypatch.setattr(launcher, "materialize_mcp_oauth_secret_cache", materialize)

    def crash():
        raise RuntimeError(
            f"port 1: {token} opaque-tenant-value private-key-material\n\x1b[31m" + "界" * 1000
        )

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit):
        launcher.main()
    body = reports[0][1]
    assert len(body["message"].encode()) <= 512
    for secret in [token, encoded, "opaque-tenant-value", "private-key-material"]:
        assert secret not in json.dumps(body)
    assert "port 1: " in body["message"]
    assert "\n" not in body["message"]
    assert "\x1b" not in body["message"]


def test_post_startup_crash_is_not_reported(report_server, monkeypatch):
    reports, _ = report_server

    def crash():
        mark_startup_complete()
        raise RuntimeError("serving crash")

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main()
    assert exit_info.value.code == 84
    assert reports == []


@pytest.mark.parametrize(
    "message",
    [
        "https://alice:quiet-pass@db.example",
        "https://alice:quiet-pass" + "x" * 5000 + "@db.example",
    ],
)
def test_scrubbing_preserves_uri_defenses(report_server, monkeypatch, message):
    reports, _ = report_server
    monkeypatch.setenv("TENANT_PROTOCOL", "https")

    def crash():
        raise RuntimeError(message)

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit):
        launcher.main()
    assert "quiet-pass" not in json.dumps(reports[0][1])


def test_partially_pattern_matched_bearer_value_is_scrubbed(report_server, monkeypatch):
    reports, _ = report_server
    monkeypatch.setenv("TENANT_API_KEY", "abc+tail/xyz==")

    def crash():
        raise RuntimeError("Authorization: Bearer abc+tail/xyz== rejected")

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit):
        launcher.main()
    body = json.dumps(reports[0][1])
    assert "tail" not in body
    assert "xyz" not in body


def test_tenant_password_inside_uri_is_fully_scrubbed(report_server, monkeypatch):
    reports, _ = report_server
    monkeypatch.setenv("TENANT_DB_PASSWORD", "quiet-pass")

    def crash():
        raise RuntimeError("connect mongodb://alice:quiet-pass@db.example failed")

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit):
        launcher.main()
    message = reports[0][1]["message"]
    assert "alice" not in message
    assert "quiet-pass" not in message


def test_empty_oauth_fields_do_not_garble_the_message(report_server, monkeypatch):
    reports, _ = report_server
    secret = {"server_url": "https://mcp.example", "access_token": "opaque", "scope": ""}
    encoded = base64.b64encode(json.dumps(secret).encode()).decode()
    monkeypatch.setenv("AGENTIC_MCP_OAUTH_B64_TEST", encoded)
    monkeypatch.setattr(launcher, "materialize_mcp_oauth_secret_cache", lambda: None)

    def crash():
        raise RuntimeError("MCP discovery rejected")

    install_target(monkeypatch, crash)
    with pytest.raises(SystemExit):
        launcher.main()
    assert "MCP discovery rejected" in reports[0][1]["message"]


@pytest.mark.parametrize(
    "variable,value",
    [
        ("OE_URL", "http://[::1"),
        ("AGENTIC_BOOT_ID", "urn:uuid:" + BOOT_ID),
        ("AGENTIC_BOOT_ID", BOOT_ID.replace("-", "")),
    ],
)
def test_invalid_report_target_keeps_classified_exit(report_server, monkeypatch, variable, value):
    reports, _ = report_server
    monkeypatch.setenv(variable, value)
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main()
    assert exit_info.value.code == 83
    assert reports == []


def test_password_only_userinfo_is_not_sent(report_server, monkeypatch):
    reports, _ = report_server
    monkeypatch.setenv("OE_URL", os.environ["OE_URL"].replace("http://", "http://:pass@"))
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main()
    assert exit_info.value.code == 83
    assert reports == []


@pytest.mark.parametrize("phase", ["hook", "bind", "serving"])
def test_real_listener_boundary(report_server, monkeypatch, phase):
    import socket
    from types import SimpleNamespace

    import uvicorn

    from agent_engine_runner_shared.server.base import BaseServer

    class StartupTestServer(BaseServer):
        @property
        def mode_name(self):
            return "aer"

        def register_routes(self, app):
            pass

        async def on_startup(self):
            if phase == "hook":
                raise RuntimeError("hook failure")

    reports, _ = report_server
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)

    async def serving_crash(_server):
        raise RuntimeError("serving failure")

    monkeypatch.setattr(uvicorn.Server, "main_loop", serving_crash)
    server = StartupTestServer(SimpleNamespace(app_name="startup-boundary"))
    port = held.getsockname()[1] if phase == "bind" else 0

    def target():
        asyncio.run(server.run(host="127.0.0.1", port=port))

    install_target(monkeypatch, target)
    try:
        with pytest.raises(SystemExit) as exit_info:
            launcher.main()
        assert exit_info.value.code == 84
        assert len(reports) == (0 if phase == "serving" else 1)
    finally:
        held.close()


def test_tool_sandbox(report_server, monkeypatch):
    reports, _ = report_server
    monkeypatch.setenv("RUNNER_MODE", "tool")
    monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
    with pytest.raises(SystemExit):
        launcher.main()
    assert reports[0][1]["sandbox"] == "tool"


def test_https_exchanges_a_real_client_certificate(monkeypatch, tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    peers = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            peers.append(self.connection.getpeercert())
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert_path, key_path)
    tls.load_verify_locations(cert_path)
    tls.verify_mode = ssl.CERT_REQUIRED
    with serve(Handler, tls) as server:
        configure_launcher(monkeypatch, tmp_path, f"https://localhost:{server.server_port}")
        monkeypatch.setenv("TLS_CERT_PATH", str(cert_path))
        monkeypatch.setenv("TLS_KEY_PATH", str(key_path))
        monkeypatch.setenv("TLS_CA_CERT_PATH", str(cert_path))
        monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
        with pytest.raises(SystemExit):
            launcher.main()
    assert len(peers) == 1
    assert peers[0]["subject"] == ((("commonName", "localhost"),),)
