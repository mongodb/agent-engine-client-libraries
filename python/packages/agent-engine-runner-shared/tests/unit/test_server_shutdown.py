from types import SimpleNamespace

import pytest
import uvicorn

from agent_engine_runner_shared.server.base import BaseServer


class _ShutdownTestServer(BaseServer):
    @property
    def mode_name(self) -> str:
        return "aer"

    def register_routes(self, app) -> None:
        pass


class _ToolShutdownTestServer(BaseServer):
    @property
    def mode_name(self) -> str:
        return "tool"

    def register_routes(self, app) -> None:
        pass


@pytest.fixture
def captured_uvicorn_config(monkeypatch):
    captured = {}

    class FakeUvicornServer:
        def __init__(self, config):
            captured["config"] = config

        async def serve(self):
            return None

    monkeypatch.setattr(uvicorn, "Server", FakeUvicornServer)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected_seconds"),
    [("1800000", 1800), ("1500", 2)],
)
async def test_run_applies_platform_shutdown_grace_period(
    monkeypatch, captured_uvicorn_config, raw, expected_seconds
):
    monkeypatch.setenv("SHUTDOWN_GRACE_PERIOD_MS", raw)

    server = _ShutdownTestServer(SimpleNamespace(app_name="test-app"))
    await server.run()

    assert captured_uvicorn_config["config"].timeout_graceful_shutdown == expected_seconds


@pytest.mark.asyncio
async def test_run_preserves_uvicorn_default_without_platform_budget(
    monkeypatch, captured_uvicorn_config
):
    monkeypatch.delenv("SHUTDOWN_GRACE_PERIOD_MS", raising=False)

    server = _ShutdownTestServer(SimpleNamespace(app_name="test-app"))
    await server.run()

    assert captured_uvicorn_config["config"].timeout_graceful_shutdown is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["invalid", "0", "-1"])
async def test_run_rejects_invalid_platform_shutdown_budget(monkeypatch, raw):
    monkeypatch.setenv("SHUTDOWN_GRACE_PERIOD_MS", raw)
    server = _ShutdownTestServer(SimpleNamespace(app_name="test-app"))

    with pytest.raises(
        ValueError,
        match="SHUTDOWN_GRACE_PERIOD_MS must be a positive integer in milliseconds",
    ):
        await server.run()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1800000", "invalid", "0", "-1"])
async def test_tool_run_ignores_aer_shutdown_budget(monkeypatch, captured_uvicorn_config, raw):
    monkeypatch.setenv("SHUTDOWN_GRACE_PERIOD_MS", raw)

    server = _ToolShutdownTestServer(SimpleNamespace(app_name="test-tool"))
    await server.run()

    assert captured_uvicorn_config["config"].timeout_graceful_shutdown is None
