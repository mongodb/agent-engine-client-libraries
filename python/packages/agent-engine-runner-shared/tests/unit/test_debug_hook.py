"""Unit tests for the local-debug-mode debugpy hook."""

import sys
from unittest.mock import MagicMock

from agent_engine_runner_shared.runtime import _maybe_start_debugger


def test_debug_mode_off_is_a_noop(monkeypatch):
    monkeypatch.delenv("MDBAE_LOCAL_MODE", raising=False)
    fake_debugpy = MagicMock()
    monkeypatch.setitem(sys.modules, "debugpy", fake_debugpy)

    _maybe_start_debugger()

    fake_debugpy.listen.assert_not_called()


def test_debug_mode_on_listens_without_blocking(monkeypatch):
    """Deliberately non-blocking: listen() only, no
    wait_for_client() -- the server must start and serve traffic whether or
    not a debugger ever attaches, the Java/JDWP suspend=n model."""
    monkeypatch.setenv("MDBAE_LOCAL_MODE", "true")
    monkeypatch.delenv("MDBAE_LOCAL_PORT", raising=False)
    fake_debugpy = MagicMock()
    monkeypatch.setitem(sys.modules, "debugpy", fake_debugpy)

    _maybe_start_debugger()

    fake_debugpy.listen.assert_called_once_with(("127.0.0.1", 5678))
    fake_debugpy.wait_for_client.assert_not_called()


def test_debug_port_env_var_is_honored(monkeypatch):
    monkeypatch.setenv("MDBAE_LOCAL_MODE", "true")
    monkeypatch.setenv("MDBAE_LOCAL_PORT", "9999")
    fake_debugpy = MagicMock()
    monkeypatch.setitem(sys.modules, "debugpy", fake_debugpy)

    _maybe_start_debugger()

    fake_debugpy.listen.assert_called_once_with(("127.0.0.1", 9999))


def test_debug_mode_off_never_imports_debugpy(monkeypatch):
    """debugpy is not a production dependency; importing it must be deferred
    until MDBAE_LOCAL_MODE=true actually needs it."""
    monkeypatch.delenv("MDBAE_LOCAL_MODE", raising=False)
    monkeypatch.delitem(sys.modules, "debugpy", raising=False)

    _maybe_start_debugger()

    assert "debugpy" not in sys.modules
