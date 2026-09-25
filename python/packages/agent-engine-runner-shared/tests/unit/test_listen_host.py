"""Unit tests for ``_resolve_listen_host``.

The resolver is now a thin wrapper around ``APP_HOST`` with a safe default
— ECP stamps ``APP_HOST`` on every runner component (AER, Tool,
memory-server) at deploy time, and the runner-base image bakes in
``APP_HOST=0.0.0.0`` as a baseline. The fallback only matters for ad-hoc
Python invocations that bypass both.
"""

import pytest

from agent_engine_runner_shared.runtime import _resolve_listen_host


@pytest.fixture(autouse=True)
def _clear_app_host(monkeypatch):
    monkeypatch.delenv("APP_HOST", raising=False)


def test_returns_app_host_when_set(monkeypatch):
    monkeypatch.setenv("APP_HOST", "::")
    assert _resolve_listen_host() == "::"


def test_returns_arbitrary_app_host_value(monkeypatch):
    """The resolver passes APP_HOST through verbatim — ECP can stamp any
    bindable host (e.g. a specific interface IP for tests)."""
    monkeypatch.setenv("APP_HOST", "192.0.2.42")
    assert _resolve_listen_host() == "192.0.2.42"


def test_defaults_to_v4_when_app_host_unset():
    assert _resolve_listen_host() == "0.0.0.0"


def test_empty_app_host_treated_as_unset(monkeypatch):
    """A shell that exports APP_HOST="" must not produce a bind to ""
    (which uvicorn would reject). Empty falls through to the default."""
    monkeypatch.setenv("APP_HOST", "")
    assert _resolve_listen_host() == "0.0.0.0"


def test_default_value_is_bind_compatible():
    """The default-path return (APP_HOST unset) must be directly usable
    as the host arg to ``socket.bind`` / ``uvicorn.Config`` — bare IP
    literal, never the bracketed URI form (``getaddrinfo`` rejects
    ``[::]`` on at least macOS). Operator-set values are passed through
    verbatim and are the operator's responsibility to keep
    bind-compatible."""
    host = _resolve_listen_host()
    assert not host.startswith("["), (
        f"_resolve_listen_host default returned URI-bracketed {host!r}; "
        "the default must be the bare IP literal so getaddrinfo accepts it"
    )
