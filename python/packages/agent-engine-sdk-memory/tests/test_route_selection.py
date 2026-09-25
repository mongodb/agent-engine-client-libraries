"""Contract matrix for route selection.

One application code path; only configuration changes. The spec's configs table
maps (token, base_url, project_id) to a route shape and auth. These assert the
resolved profile, base path, and auth header for each row, plus the two
misconfiguration errors. Wire-level path assertions live in
``test_http_runtime.py``; here we pin the facade's input -> routing decision.
"""

from __future__ import annotations

import httpx
import pytest
from agent_engine_sdk_memory import Memory
from agent_engine_sdk_memory._http_runtime import (
    _GATEWAY_PROFILE,
    _OE_PROFILE,
    _HttpMemoryRuntime,
)
from agent_engine_sdk_memory.errors import MemoryRouteNotFoundError

GATEWAY_URL = "https://gw.example.com"
OE_URL = "http://oe.local"
HOSTED_OE_URL = "https://hosted-oe.example.com"
API_KEY = "k"

# Clear the AGENTIC_MEMORY_* env so kwargs alone drive the decision.
_ENV_VARS = (
    "AGENTIC_MEMORY_API_KEY",
    "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN",
    "AGENTIC_MEMORY_BASE_URL",
    "AGENTIC_MEMORY_PROJECT_ID",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _auth(memory: Memory) -> str | None:
    return memory._runtime._transport._client.headers.get("authorization")


def test_hosted_via_gateway():
    # token set + Gateway URL + project_id set -> project-scoped, bearer auth.
    m = Memory(service_account_token=API_KEY, base_url=GATEWAY_URL, project_id="p123")
    assert m._runtime._profile is _GATEWAY_PROFILE
    assert m._runtime._base_path == "/api/v1/projects/p123/memory"
    assert _auth(m) == f"Bearer {API_KEY}"


def test_local_direct_oe():
    # no token + local OE URL + no project_id -> flat, no auth.
    m = Memory(base_url=OE_URL)
    assert m._runtime._profile is _OE_PROFILE
    assert m._runtime._base_path == "/api/v1/memory"
    assert _auth(m) is None


def test_platform_direct_oe():
    # Platform runs in-cluster against its project OE: same config shape as local
    # (no token, no project_id), just a different in-cluster URL -> flat, no auth.
    m = Memory(base_url="http://memory-server.tenant.svc")
    assert m._runtime._profile is _OE_PROFILE
    assert m._runtime._base_path == "/api/v1/memory"
    assert _auth(m) is None


def test_future_hosted_oe_with_key():
    # token set + hosted OE URL + NO project_id -> flat, bearer. Proves auth
    # is not the route-shape discriminator; project_id presence is.
    m = Memory(service_account_token=API_KEY, base_url=HOSTED_OE_URL)
    assert m._runtime._profile is _OE_PROFILE
    assert m._runtime._base_path == "/api/v1/memory"
    assert _auth(m) == f"Bearer {API_KEY}"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_project_id_kwarg_selects_flat(blank):
    # A blank/whitespace project_id is "unset": presence is the discriminator, and
    # an empty path segment would be a silent footgun. Resolves to the flat OE.
    m = Memory(base_url=OE_URL, project_id=blank)
    assert m._runtime._profile is _OE_PROFILE
    assert m._runtime._base_path == "/api/v1/memory"


def test_blank_project_id_env_selects_flat(monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_PROJECT_ID", "")
    m = Memory(base_url=OE_URL)
    assert m._runtime._profile is _OE_PROFILE


@pytest.mark.parametrize("source", ["kwarg", "env"])
def test_surrounding_whitespace_project_id_is_stripped(source, monkeypatch):
    # " p1 " is a present project_id, stripped before it reaches the path.
    if source == "kwarg":
        m = Memory(
            service_account_token=API_KEY, base_url=GATEWAY_URL, project_id="  p1  "
        )
    else:
        monkeypatch.setenv("AGENTIC_MEMORY_PROJECT_ID", "  p1  ")
        m = Memory(service_account_token=API_KEY, base_url=GATEWAY_URL)
    assert m._runtime._profile is _GATEWAY_PROFILE
    assert m._runtime._base_path == "/api/v1/projects/p1/memory"


def _runtime_returning_404(*, project_id: str | None) -> _HttpMemoryRuntime:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    profile = _GATEWAY_PROFILE if project_id else _OE_PROFILE
    return _HttpMemoryRuntime(
        base_url="http://backend",
        profile=profile,
        project_id=project_id,
        transport=httpx.MockTransport(handler),
    )


def test_stale_project_id_against_oe_is_actionable():
    # project_id set but backend has no project-scoped route (local OE) -> 404 ->
    # hint to unset project_id.
    rt = _runtime_returning_404(project_id="p1")
    with pytest.raises(MemoryRouteNotFoundError) as ei:
        rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    assert "unset project_id" in str(ei.value).lower()


def test_missing_project_id_against_gateway_is_actionable():
    # project_id empty but backend only serves project-scoped routes (Gateway) ->
    # 404 -> hint to set project_id.
    rt = _runtime_returning_404(project_id=None)
    with pytest.raises(MemoryRouteNotFoundError) as ei:
        rt.build_context(query="q", user_id="u")
    msg = str(ei.value).lower()
    assert "set project_id" in msg and "unset" not in msg
