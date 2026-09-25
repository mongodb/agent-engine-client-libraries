"""Turn-level ADK rewind client tests."""

from __future__ import annotations

import traceback
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from agent_engine_sdk import SessionForkResponse

from agent_engine_sdk_adk.rewind import RewindBranchError, create_rewind_branch
from agent_engine_sdk_adk.runner import DurableADKRunner


@pytest.mark.asyncio
async def test_platform_runner_rewind_delegates_google_invocation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_execution_id", lambda: "execution-4"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_session_id", lambda: "session-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_user_id", lambda: "user-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_oe_url", lambda: "http://oe:8000"
    )

    async def rewind_client(
        *,
        execution_id: str,
        oe_url: str,
        rewind_before_invocation_id: str,
    ) -> SessionForkResponse:
        captured.update(
            execution_id=execution_id,
            oe_url=oe_url,
            target=rewind_before_invocation_id,
        )
        return SessionForkResponse(
            session_id="branch-session", execution_id="branch-execution"
        )

    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.create_rewind_branch", rewind_client
    )
    runner = DurableADKRunner(app_name="app")

    result = await runner.rewind_async(
        user_id="user-1",
        session_id="session-1",
        rewind_before_invocation_id="execution-3",
    )

    assert captured == {
        "execution_id": "execution-4",
        "oe_url": "http://oe:8000",
        "target": "execution-3",
    }
    assert result.session_id == "branch-session"
    assert result.execution_id == "branch-execution"


@pytest.mark.asyncio
async def test_platform_runner_rewind_requires_active_durable_invocation() -> None:
    runner = DurableADKRunner(app_name="app")

    with pytest.raises(RewindBranchError, match="active durable invocation"):
        await runner.rewind_async(
            user_id="user-1",
            session_id="session-2",
            rewind_before_invocation_id="execution-3",
        )


@pytest.mark.asyncio
async def test_platform_runner_rewind_rejects_another_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_execution_id", lambda: "execution-4"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_session_id", lambda: "session-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_user_id", lambda: "user-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_oe_url", lambda: "http://oe:8000"
    )

    with pytest.raises(RewindBranchError, match="match the active invocation"):
        await DurableADKRunner(app_name="app").rewind_async(
            user_id="user-1",
            session_id="another-session",
            rewind_before_invocation_id="execution-3",
        )


@pytest.mark.asyncio
async def test_platform_runner_rewind_rejects_another_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewind_client = AsyncMock()
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_execution_id", lambda: "execution-4"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_session_id", lambda: "session-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_user_id", lambda: "user-1"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.get_current_oe_url", lambda: "http://oe:8000"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.runner.create_rewind_branch", rewind_client
    )

    with pytest.raises(RewindBranchError, match="match the active invocation"):
        await DurableADKRunner(app_name="app").rewind_async(
            user_id="another-user",
            session_id="session-1",
            rewind_before_invocation_id="execution-3",
        )
    rewind_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewind_sends_only_server_resolvable_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def post_branch(oe_url: str, body: dict[str, object]) -> tuple[int, object]:
        captured.update(oe_url=oe_url, body=body)
        return 201, {
            "session_id": "branch-session",
            "execution_id": "branch-execution",
        }

    monkeypatch.setattr("agent_engine_sdk_adk.rewind._post_branch", post_branch)

    branch = await create_rewind_branch(
        execution_id="execution-4",
        oe_url="http://oe:8000",
        rewind_before_invocation_id="execution-3",
    )

    assert captured == {
        "oe_url": "http://oe:8000",
        "body": {
            "execution_id": "execution-4",
            "before_execution_id": "execution-3",
        },
    }
    assert branch.session_id == "branch-session"
    assert branch.execution_id == "branch-execution"


@pytest.mark.asyncio
async def test_rewind_reports_status_without_exposing_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def post_branch(oe_url: str, body: dict[str, object]) -> tuple[int, object]:
        return 409, {"error": "sensitive backend detail"}

    monkeypatch.setattr("agent_engine_sdk_adk.rewind._post_branch", post_branch)

    with pytest.raises(RewindBranchError, match="HTTP 409") as caught:
        await create_rewind_branch(
            execution_id="execution-4",
            oe_url="http://oe:8000",
            rewind_before_invocation_id="execution-3",
        )

    assert "sensitive" not in str(caught.value)
    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_rewind_normalizes_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FailingClient:
        async def post(self, url: str, **kwargs: object) -> None:
            captured.update(url=url, body=kwargs["json"])
            raise httpx.ConnectError("backend address")

        async def aclose(self) -> None:
            return None

    async def client_factory(*_args: object, **_kwargs: object) -> FailingClient:
        return FailingClient()

    monkeypatch.setattr(
        "agent_engine_sdk_adk.rewind.create_async_httpx_client_with_tls", client_factory
    )
    monkeypatch.delenv("OE_URL", raising=False)

    with pytest.raises(
        RewindBranchError, match="could not reach orchestration engine"
    ) as caught:
        await create_rewind_branch(
            execution_id="execution-4",
            oe_url="http://callback-oe:8000",
            rewind_before_invocation_id="execution-3",
        )

    assert "backend address" not in str(caught.value)

    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "backend address" not in rendered
    assert captured == {
        "url": "http://callback-oe:8000/workflow/branches",
        "body": {
            "execution_id": "execution-4",
            "before_execution_id": "execution-3",
        },
    }


@pytest.mark.asyncio
async def test_rewind_normalizes_tls_client_setup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def client_factory(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("could not load /var/run/secrets/client.key")

    monkeypatch.setattr(
        "agent_engine_sdk_adk.rewind.create_async_httpx_client_with_tls", client_factory
    )

    with pytest.raises(
        RewindBranchError, match="could not initialize orchestration engine client"
    ) as caught:
        await create_rewind_branch(
            execution_id="execution-4",
            oe_url="http://oe:8000",
            rewind_before_invocation_id="execution-3",
        )

    assert "/var/run/secrets/client.key" not in str(caught.value)
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "/var/run/secrets/client.key" not in rendered
