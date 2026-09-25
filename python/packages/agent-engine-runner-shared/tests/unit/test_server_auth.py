"""
Unit tests for runner request authentication.

Mirrors TypeScript's tests/unit/server_auth.test.ts — the two runner runtimes
are parallel implementations, so the twins must stay behaviourally identical.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server.auth import (
    MissingRunnerAuthTokenError,
    bearer_token,
    is_unauthenticated_path,
    register_auth_middleware,
    tokens_equal,
)


def _app_with_auth(env: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/execute")
    async def execute() -> dict:
        return {"ok": True}

    register_auth_middleware(app, "aer", env)
    return app


def test_bearer_token_parsing() -> None:
    assert bearer_token("Bearer s3cret") == "s3cret"
    # RFC 7235 makes the scheme case-insensitive.
    assert bearer_token("bearer s3cret") == "s3cret"
    assert bearer_token("s3cret") is None
    assert bearer_token("Basic s3cret") is None
    assert bearer_token("Bearer ") is None
    assert bearer_token("") is None
    assert bearer_token(None) is None


def test_tokens_equal() -> None:
    assert tokens_equal("s3cret", "s3cret")
    assert not tokens_equal("s3cre", "s3cret")
    assert not tokens_equal("s3cretX", "s3cret")
    assert not tokens_equal("", "s3cret")


def test_health_paths_are_unauthenticated() -> None:
    # Kubelet probes cannot present a token and these carry no tenant data.
    for path in ("/", "/health", "/healthz", "/ready", "/readyz"):
        assert is_unauthenticated_path(path)
    assert not is_unauthenticated_path("/execute")
    assert not is_unauthenticated_path("/invoke_llm")
    assert not is_unauthenticated_path("/query/sessions")
    # /metrics carries tenant-derived labels (tool_name, model_name) and is
    # not a scrape target, so it is gated with everything else.
    assert not is_unauthenticated_path("/metrics")


def test_execute_requires_token_when_configured() -> None:
    client = TestClient(_app_with_auth({"RUNNER_AUTH_TOKEN": "s3cret"}))

    assert client.post("/execute").status_code == 401
    assert client.post("/execute", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/execute", headers={"Authorization": "s3cret"}).status_code == 401

    resp = client.post("/execute", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_health_reachable_without_token_when_configured() -> None:
    client = TestClient(_app_with_auth({"RUNNER_AUTH_TOKEN": "s3cret"}))
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "host",
    ["x/?", "x/health?", "x/", "x?", "x/#", "x/?a", "1.2.3.4:80/?", "evil"],
)
def test_crafted_host_cannot_reach_unauthenticated_paths(host: str) -> None:
    """A Host header that would corrupt request.url.path must not bypass auth.

    Regression guard: on older starlette, a Host carrying path
    separators (``/?``, ``?``, ``#``) makes ``request.url.path`` resolve to an
    allowlisted path while the router still dispatches on ``scope["path"]``.
    The middleware now reads ``scope["path"]`` directly, and ``starlette>=1.4.1``
    validates the Host header, so these are double-defended.

    This test *passes today* — it does not reproduce a present bug. It locks
    the current posture against a future starlette downgrade or a loosened
    dependency floor, so do not read a passing run as evidence the fix is
    unnecessary.
    """
    client = TestClient(_app_with_auth({"RUNNER_AUTH_TOKEN": "s3cret"}))

    resp = client.post("/execute", headers={"Host": host})
    # No Authorization header presented: the crafted Host must not let the
    # request through as if it hit an unauthenticated path.
    assert resp.status_code == 401

    # The same crafted Host with a valid token still authenticates normally —
    # the fix must not regress legitimate callers.
    resp = client.post("/execute", headers={"Host": host, "Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


@pytest.mark.parametrize("env", [{}, {"RUNNER_AUTH_TOKEN": ""}, {"RUNNER_AUTH_TOKEN": "   "}])
def test_unconfigured_server_does_not_enforce(env: dict) -> None:
    # The runner image also runs under `agentengine dev up`, where no
    # deploy-time secret exists; a startup warning covers that case.
    assert register_auth_middleware(FastAPI(), "aer", env) is False
    client = TestClient(_app_with_auth(env))
    assert client.post("/execute").status_code == 200


@pytest.mark.parametrize("required", ["true", "TRUE", "1"])
def test_required_without_token_refuses_to_start(required: str) -> None:
    # A deployment that mandates auth must fail loudly on a provisioning
    # gap instead of silently serving unauthenticated routes.
    with pytest.raises(MissingRunnerAuthTokenError):
        register_auth_middleware(FastAPI(), "aer", {"RUNNER_AUTH_REQUIRED": required})


@pytest.mark.parametrize("required", ["", "false", "0", "yes"])
def test_non_affirmative_required_does_not_enforce(required: str) -> None:
    env = {"RUNNER_AUTH_REQUIRED": required}
    assert register_auth_middleware(FastAPI(), "aer", env) is False
    client = TestClient(_app_with_auth(env))
    assert client.post("/execute").status_code == 200


def test_required_with_token_enforces() -> None:
    env = {"RUNNER_AUTH_TOKEN": "s3cret", "RUNNER_AUTH_REQUIRED": "true"}
    client = TestClient(_app_with_auth(env))
    assert client.post("/execute").status_code == 401
    assert client.post("/execute", headers={"Authorization": "Bearer s3cret"}).status_code == 200
