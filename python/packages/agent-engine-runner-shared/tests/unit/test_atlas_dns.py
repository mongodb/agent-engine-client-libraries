"""Atlas SRV DNS classification: 503 + error_code, no hostname leak."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError

from agent_engine_runner_shared.server.atlas_dns import (
    ATLAS_DNS_ERROR_CODE,
    is_atlas_dns_failure,
)
from agent_engine_runner_shared.server.base import BaseServer

_HOST = "cluster0.abc.mongodb-dev.net"
_URI = f"mongodb+srv://user:pass@{_HOST}/?retryWrites=true"
_DNS_MESSAGE = (
    "All nameservers failed to answer the query "
    f"_mongodb._tcp.{_HOST}. IN SRV: Server "
    "Do53:fc00:fee1:dead::53@53 answered REFUSED"
)


@pytest.fixture(autouse=True)
def _platform_atlas_uri(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", _URI)


class _RaisingServer(BaseServer):
    def __init__(self, exc: Exception):
        super().__init__(SimpleNamespace(app_name="test-app"))
        self._exc = exc

    @property
    def mode_name(self) -> str:
        return "aer"

    def register_routes(self, app) -> None:
        exc = self._exc

        @app.get("/boom")
        def boom() -> None:
            raise exc


def _client(exc: Exception) -> TestClient:
    return TestClient(_RaisingServer(exc).create_app(), raise_server_exceptions=False)


def test_is_atlas_dns_failure_matches_refused_srv():
    assert is_atlas_dns_failure(ConfigurationError(_DNS_MESSAGE))


def test_is_atlas_dns_failure_rejects_unrelated_error():
    assert not is_atlas_dns_failure(ConfigurationError("bad option: tls"))


def test_is_atlas_dns_failure_rejects_other_host(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://user:pass@other.example.net/")
    assert not is_atlas_dns_failure(ConfigurationError(_DNS_MESSAGE))


def test_is_atlas_dns_failure_rejects_when_uri_unset(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert not is_atlas_dns_failure(ConfigurationError(_DNS_MESSAGE))


_TIMEOUT_MESSAGE = (
    "The resolution lifetime expired after 5.016 seconds: Server "
    "Do53:fc00:fee1:dead::53@53 answered The DNS operation timed out"
)


def test_is_atlas_dns_failure_matches_resolution_timeout():
    assert is_atlas_dns_failure(ConfigurationError(_TIMEOUT_MESSAGE))


def test_resolution_timeout_returns_503():
    resp = _client(ConfigurationError(_TIMEOUT_MESSAGE)).get("/boom")

    assert resp.status_code == 503
    assert resp.json()["error_code"] == ATLAS_DNS_ERROR_CODE


def test_atlas_dns_failure_returns_503_with_code():
    resp = _client(ConfigurationError(_DNS_MESSAGE)).get("/boom")

    assert resp.status_code == 503
    body = resp.json()
    assert body["error_code"] == ATLAS_DNS_ERROR_CODE
    # The hostname must not leak to the caller.
    assert _HOST not in body["detail"]


def test_server_selection_timeout_with_dns_signature_is_classified():
    exc = ServerSelectionTimeoutError(f"{_HOST}:27017: _mongodb._tcp lookup failed")

    resp = _client(exc).get("/boom")

    assert resp.status_code == 503
    assert resp.json()["error_code"] == ATLAS_DNS_ERROR_CODE


def test_customer_own_srv_host_stays_500(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://user:pass@platform.example.net/")
    exc = ConfigurationError(
        "All nameservers failed to answer the query "
        "_mongodb._tcp.customer-own.example.com. IN SRV: answered REFUSED"
    )

    resp = _client(exc).get("/boom")

    assert resp.status_code == 500


def test_unrelated_pymongo_error_stays_500():
    resp = _client(ConfigurationError("bad option: retryWrites")).get("/boom")

    assert resp.status_code == 500
    assert resp.text == "Internal Server Error"


def test_unrelated_pymongo_error_is_logged(caplog):
    # Handling the exception suppresses Uvicorn's default traceback logging, so
    # the handler must log it itself.
    with caplog.at_level("ERROR"):
        _client(ConfigurationError("bad option: retryWrites")).get("/boom")

    assert any("Unclassified pymongo failure" in record.message for record in caplog.records)
