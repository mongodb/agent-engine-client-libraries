"""Tests for delegated tool authorization models."""

from agent_engine_runner_shared.models import ToolAuthorization, ToolPodExecuteRequest


def test_tool_pod_execute_request_parses_authorization_payload() -> None:
    req = ToolPodExecuteRequest(
        execution_id="exec-1",
        tool_name="list_repos",
        arguments={"org": "mongodb"},
        session_id="session-1",
        authorization={"token": "secret-token", "expires_at": 1_735_689_600},
    )

    assert req.authorization is not None
    assert req.authorization.token == "secret-token"
    assert req.authorization.expires_at == 1_735_689_600


def test_tool_authorization_repr_redacts_token_value() -> None:
    auth = ToolAuthorization(token="very-secret-token", expires_at=1_735_689_600)

    # repr=False on the token field should prevent token values from being rendered.
    assert "very-secret-token" not in repr(auth)
