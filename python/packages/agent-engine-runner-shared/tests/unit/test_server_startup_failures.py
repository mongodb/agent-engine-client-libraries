"""Startup failures must reach the launcher's fatal path.

uvicorn exits with STARTUP_FAILURE on a lifespan startup failure or a bind
error; both must propagate to the launcher as a normal error.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn

from agent_engine_runner_shared.server import base


class _StartupTestServer(base.BaseServer):
    def __init__(self, on_startup_error: BaseException | None = None) -> None:
        super().__init__(SimpleNamespace(app_name="test-app"))
        self._on_startup_error = on_startup_error

    @property
    def mode_name(self) -> str:
        return "aer"

    def register_routes(self, app: Any) -> None:
        pass

    async def on_startup(self) -> None:
        if self._on_startup_error is not None:
            raise self._on_startup_error


class TestLifespanRecordsStartupError:
    """The lifespan records the cause so ``run`` can re-raise it after uvicorn exits."""

    @pytest.mark.asyncio
    async def test_startup_hook_failure_is_recorded_and_reraised(self) -> None:
        server = _StartupTestServer(RuntimeError("capability registration rejected"))

        with pytest.raises(RuntimeError, match="capability registration rejected"):
            async with server._create_lifespan()(None):
                pass

        assert isinstance(server._startup_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_successful_startup_records_no_error(self) -> None:
        server = _StartupTestServer()

        async with server._create_lifespan()(None):
            pass

        assert server._startup_error is None


class TestRunConvertsStartupFailureExit:
    """``run`` turns uvicorn's startup-failure exit into a normal error."""

    @pytest.fixture
    def fake_uvicorn(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {"exit_code": uvicorn.config.STARTUP_FAILURE}

        class _FakeServer:
            def __init__(self, config: uvicorn.Config) -> None:
                captured["config"] = config

            async def serve(self) -> None:
                raise SystemExit(captured["exit_code"])

        monkeypatch.setattr(uvicorn, "Server", _FakeServer)
        return captured

    @pytest.mark.asyncio
    async def test_recorded_startup_error_is_preferred(self) -> None:
        server = _StartupTestServer()
        recorded = RuntimeError("capability registration rejected")
        server._startup_error = recorded

        assert server._startup_failure_error() is recorded

    @pytest.mark.asyncio
    async def test_bind_failure_is_the_default(self) -> None:
        server = _StartupTestServer()

        assert isinstance(server._startup_failure_error(), RuntimeError)

    @pytest.mark.asyncio
    async def test_bind_failure_becomes_an_error(self, fake_uvicorn: dict[str, Any]) -> None:
        server = _StartupTestServer()

        with pytest.raises(RuntimeError, match="listen socket could not be bound"):
            await server.run()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_code", [1, 7])
    async def test_a_non_startup_system_exit_is_preserved(
        self, fake_uvicorn: dict[str, Any], exit_code: int
    ) -> None:
        fake_uvicorn["exit_code"] = exit_code
        server = _StartupTestServer()

        with pytest.raises(SystemExit) as exc_info:
            await server.run()

        assert exc_info.value.code == exit_code

    @pytest.mark.asyncio
    async def test_startup_error_is_reset_between_runs(self, fake_uvicorn: dict[str, Any]) -> None:
        server = _StartupTestServer()
        server._startup_error = RuntimeError("stale")

        with pytest.raises(RuntimeError, match="listen socket could not be bound"):
            await server.run()


class TestRealUvicornStartupFailure:
    """End to end against the real uvicorn server."""

    @pytest.mark.asyncio
    async def test_normal_startup_and_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _stop_after_startup(self: Any) -> None:
            return None

        monkeypatch.setattr(uvicorn.Server, "main_loop", _stop_after_startup)
        server = _StartupTestServer()

        await server.run(host="127.0.0.1", port=_free_port())

        assert server._startup_error is None

    @pytest.mark.asyncio
    async def test_startup_hook_failure_raises_the_original_error(self) -> None:
        server = _StartupTestServer(RuntimeError("startup hook exploded"))

        with pytest.raises(RuntimeError, match="startup hook exploded"):
            await server.run(host="127.0.0.1", port=_free_port())

    @pytest.mark.asyncio
    async def test_bind_failure_raises_a_startup_error(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            server = _StartupTestServer()

            with pytest.raises(RuntimeError, match="listen socket could not be bound"):
                await server.run(host="127.0.0.1", port=port)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestLauncherIntegration:
    """A server startup-hook failure reaches the launcher's classified fatal path."""

    def test_startup_hook_failure_writes_termination_message(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from agent_engine_runner_shared import launcher

        path = tmp_path / "termination-log"
        monkeypatch.setattr(launcher, "_TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(launcher, "setup_logging", lambda **_kwargs: None)
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_agent:main")

        port = _free_port()
        server = _StartupTestServer(RuntimeError("capability registration rejected"))

        def target() -> None:
            asyncio.run(server.run(host="127.0.0.1", port=port))

        fake_module = types.ModuleType("fake_agent")
        fake_module.main = target  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_agent", fake_module)

        with (
            caplog.at_level(logging.ERROR, logger=launcher.__name__),
            pytest.raises(SystemExit) as exc_info,
        ):
            launcher.main()

        assert exc_info.value.code == launcher.EXIT_STARTUP_CRASH
        assert "capability registration rejected" in caplog.text
        assert "Unhandled exception from agent entrypoint 'fake_agent:main'" in path.read_text()
