"""Facade construction, URL resolution, and resource lifecycle.

The route shape comes from project_id presence, and auth comes from the token,
independently. Behaviors that hold across constructions (lazy/no-network
construction, kwarg>env precedence, owned-runtime cleanup) are parametrized over
the token and base-url constructors. The per-shape specifics live in the
sections below.
"""

import json

import httpx
import pytest
from agent_engine_sdk_memory import Memory, MemoryRequestContext
from agent_engine_sdk_memory._direct_crud import _EmptyTenancyCrudClient
from agent_engine_sdk_memory._http_runtime import (
    _GATEWAY_PROFILE,
    _OE_PROFILE,
    _HttpMemoryRuntime,
)
from agent_engine_sdk_memory.errors import MemoryNotSupportedError
from agent_engine_sdk_memory.protocol import MemoryCrudClient

API_KEY = "test-api-key"
ENV_VAR = "AGENTIC_MEMORY_BASE_URL"
GATEWAY_DEFAULT_URL = "https://agentengine.mongodb.com"


def _base_url(memory: Memory) -> str:
    return str(memory._runtime._transport._client.base_url)


def _gateway(**kwargs) -> Memory:
    return Memory(service_account_token=API_KEY, **kwargs)


def _direct(**kwargs) -> Memory:
    return Memory(base_url=kwargs.pop("base_url", "http://oe.local"), **kwargs)


_MODES = [pytest.param(_gateway, id="gateway"), pytest.param(_direct, id="direct")]


# --- behaviors identical across both construction modes ---


@pytest.mark.parametrize("make", _MODES)
def test_constructs_lazily_with_no_network(make, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("construction must not make a network call")

    # Client.send is the lowest chokepoint every verb funnels through, so this
    # catches a construction-time request regardless of HTTP method.
    monkeypatch.setattr(httpx.Client, "send", _boom)
    assert isinstance(make(), Memory)


@pytest.mark.parametrize("make", _MODES)
def test_base_url_kwarg_takes_precedence(make, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "http://from-env")
    memory = make(base_url="http://from-kwarg")
    assert _base_url(memory).rstrip("/") == "http://from-kwarg"


@pytest.mark.parametrize("make", _MODES)
def test_close_releases_owned_runtime(make):
    memory = make()
    client = memory._runtime._transport._client
    assert client.is_closed is False
    memory.close()
    assert client.is_closed is True


@pytest.mark.parametrize("make", _MODES)
def test_context_manager_closes_owned_runtime(make):
    with make() as memory:
        client = memory._runtime._transport._client
        assert client.is_closed is False
    assert client.is_closed is True


# --- project-scoped (api_key + project_id) ---


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_legacy_api_key_rejected_without_warning(blank, recwarn):
    # The blank check fires before the deprecation warning, so a rejected
    # legacy value never warns.
    with pytest.raises(ValueError, match="api_key"):
        Memory(api_key=blank)
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_service_account_token_kwarg_constructs_hosted(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_API_KEY", raising=False)
    memory = Memory(service_account_token=API_KEY)
    assert _base_url(memory).rstrip("/") == GATEWAY_DEFAULT_URL
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == f"Bearer {API_KEY}"
    )


def test_explicit_token_wins_over_env(monkeypatch):
    # The kwarg>env invariant holds for the renamed input.
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_API_KEY", raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", "from-env")
    memory = Memory(service_account_token="from-kwarg")
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == "Bearer from-kwarg"
    )


def test_legacy_api_key_kwarg_warns_but_works(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", raising=False)
    with pytest.warns(DeprecationWarning, match="service_account_token"):
        memory = Memory(api_key=API_KEY)
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == f"Bearer {API_KEY}"
    )
    assert _base_url(memory).rstrip("/") == GATEWAY_DEFAULT_URL


def test_legacy_api_key_env_warns_but_works(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_API_KEY", "legacy-from-env")
    with pytest.warns(DeprecationWarning, match="AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN"):
        memory = Memory()
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == "Bearer legacy-from-env"
    )


def test_both_auth_args_rejected():
    with pytest.raises(ValueError, match="only one"):
        Memory(service_account_token="new", api_key="legacy")


def test_new_arg_with_legacy_env_rejected(monkeypatch):
    monkeypatch.setenv("AGENTIC_MEMORY_API_KEY", "legacy-from-env")
    with pytest.raises(ValueError, match="only one"):
        Memory(service_account_token="new")


def test_blank_service_account_token_rejected(monkeypatch):
    monkeypatch.delenv("AGENTIC_MEMORY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="service_account_token"):
        Memory(service_account_token="   ")


def test_blank_legacy_env_var_does_not_conflict(monkeypatch, recwarn):
    # An empty-exported AGENTIC_MEMORY_API_KEY (a common CI/Helm placeholder)
    # is unset, not a conflicting second input — and triggers no warning.
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_API_KEY", "")
    memory = Memory(service_account_token=API_KEY)
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == f"Bearer {API_KEY}"
    )
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_blank_new_env_var_does_not_conflict(monkeypatch):
    # Symmetric: an empty-exported AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN is unset,
    # so the legacy argument alone drives auth (with its deprecation warning).
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", "")
    with pytest.warns(DeprecationWarning, match="AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN"):
        memory = Memory(api_key=API_KEY)
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == f"Bearer {API_KEY}"
    )


def test_blank_auth_env_vars_alone_treated_as_unset(monkeypatch):
    # Both auth env vars exported blank: nothing is provided, so construction
    # falls through to the no-config ValueError, same as fully unset.
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_API_KEY", "")
    monkeypatch.setenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", "  ")
    with pytest.raises(ValueError, match="api_key"):
        Memory()


def test_env_var_used_when_no_kwarg(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "http://from-env")
    memory = Memory(service_account_token=API_KEY)
    assert _base_url(memory).rstrip("/") == "http://from-env"


def test_default_url_when_neither(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    memory = Memory(service_account_token=API_KEY)
    assert _base_url(memory).rstrip("/") == GATEWAY_DEFAULT_URL


def test_service_account_token_read_from_env(monkeypatch):
    # The renamed auth env var resolves with no kwarg; with a token and no
    # base_url it falls back to the hosted default and attaches the bearer.
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_PROJECT_ID", raising=False)
    monkeypatch.delenv("AGENTIC_MEMORY_API_KEY", raising=False)
    monkeypatch.setenv("AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN", "token-from-env")
    memory = Memory()
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == "Bearer token-from-env"
    )
    assert _base_url(memory).rstrip("/") == GATEWAY_DEFAULT_URL


def test_gateway_crud_is_enabled():
    # Gateway CRUD is available under token auth on the project-scoped routes;
    # the Gateway infers org/project from the token and the caller supplies
    # user_id.
    memory = Memory(service_account_token=API_KEY, project_id="proj-1")
    assert memory._runtime._profile is _GATEWAY_PROFILE
    assert memory._client is not None
    assert isinstance(memory._client, MemoryCrudClient)
    assert (
        memory._runtime._transport._client.headers.get("authorization")
        == f"Bearer {API_KEY}"
    )


def test_gateway_crud_request_shape():
    """CRUD calls in Gateway mode send the bearer token and empty tenancy strings."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            201,
            json={
                "id": "s1",
                "label": "fact",
                "text": "t",
                "created_at": "2024-01-01T00:00:00Z",
                "has_embedding": True,
            },
        )

    runtime = _HttpMemoryRuntime(
        base_url="http://gw",
        profile=_GATEWAY_PROFILE,
        project_id="proj-1",
        headers={"Authorization": f"Bearer {API_KEY}"},
        transport=httpx.MockTransport(handler),
    )
    memory = Memory(
        runtime=runtime,
        client=_EmptyTenancyCrudClient(runtime.transport, runtime.base_path),
    )
    memory.save_semantic(text="t", label="fact", user_id="u")

    assert len(captured) == 1
    req = captured[0]
    assert req.headers.get("authorization") == f"Bearer {API_KEY}"
    # CRUD hangs off the project-scoped base path, same route shape as the core loop.
    assert req.url.path == "/api/v1/projects/proj-1/memory/semantic"
    body = json.loads(req.content)
    # SDK sends empty tenancy; the Gateway stamps org_id/project_id from the key.
    assert body["org_id"] == ""
    assert body["project_id"] == ""
    # SDK sends caller-supplied user_id; the Gateway does not overwrite it.
    assert body["user_id"] == "u"


def test_close_on_bound_handle_releases_owned_runtime():
    memory = Memory(service_account_token=API_KEY)
    client = memory._runtime._transport._client
    bound = memory.bind(MemoryRequestContext(user_id="u"))
    bound.close()
    assert client.is_closed is True


def test_context_manager_on_bound_handle_closes_owned_runtime():
    memory = Memory(service_account_token=API_KEY)
    client = memory._runtime._transport._client
    with memory.bind(MemoryRequestContext(user_id="u")):
        assert client.is_closed is False
    assert client.is_closed is True


def test_injected_runtime_is_not_closed_by_facade():
    class _SpyRuntime:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    spy = _SpyRuntime()
    Memory(runtime=spy).close()  # type: ignore[arg-type]
    assert spy.closed is False


def test_record_turn_accepts_tool_metadata_in_gateway_mode():
    # Gateway profile (project_id set) now forwards tool-call turn metadata.
    memory = Memory(service_account_token=API_KEY, project_id="proj-1")
    with pytest.raises(Exception) as exc_info:
        memory.record_turn(role="tool", content="result", tool_call_id="call_1")
    # Should reach the network (auth error), not be blocked client-side.
    assert not isinstance(exc_info.value, MemoryNotSupportedError)


# --- direct (base_url, no project_id) ---


def test_constructs_direct_mode_without_raising(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    memory = Memory(base_url="http://oe.local")
    assert memory._runtime._profile is _OE_PROFILE
    assert memory._client is not None
    assert isinstance(memory._client, MemoryCrudClient)


def test_project_id_presence_selects_route_shape():
    # project_id presence — not auth — picks the route shape.
    # Set => Gateway/project-scoped profile; empty => flat OE. The
    # project-scoped Gateway supports CRUD, so the adapter is wired in both shapes.
    gateway = Memory(
        service_account_token="k", base_url="http://gw", project_id="proj-1"
    )
    assert gateway._runtime._profile is _GATEWAY_PROFILE
    assert gateway._client is not None

    # token + base_url but NO project_id => flat OE profile (authed direct-OE),
    # with the bearer still attached. Auth does not pick the route shape.
    flat = Memory(service_account_token="k", base_url="http://hosted-oe")
    assert flat._runtime._profile is _OE_PROFILE
    assert flat._client is not None
    assert flat._runtime._transport._client.headers.get("authorization") == "Bearer k"


def test_no_url_resolves_to_none(monkeypatch):
    # With neither a kwarg, the env var, nor an auth token, resolution returns
    # None; the constructor then raises ValueError (no config at all).
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert Memory._resolve_url(None, auth_token=None) is None


def test_env_var_alone_selects_direct_mode(monkeypatch):
    # No auth token and no base_url kwarg, but the env var is set: the facade
    # must enter OE mode, not raise ValueError.
    monkeypatch.setenv(ENV_VAR, "http://oe.from-env")
    memory = Memory()
    assert memory._runtime._profile is _OE_PROFILE
    assert memory._client is not None
    assert _base_url(memory).rstrip("/") == "http://oe.from-env"


def test_blank_base_url_raises(monkeypatch):
    # A whitespace-only base_url is rejected up front (mirrors the auth-token
    # non-blank check) rather than surfacing later as a confusing URL error.
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="base_url"):
        Memory(base_url="   ")


def test_blank_env_var_treated_as_unset(monkeypatch):
    # A whitespace-only env var is not a real URL; no config resolves, so ValueError.
    monkeypatch.setenv(ENV_VAR, "   ")
    with pytest.raises(ValueError, match="api_key"):
        Memory()


def test_no_config_raises_value_error(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="api_key"):
        Memory()


def test_direct_mode_crud_reaches_adapter():
    # In direct mode CRUD conveniences must NOT raise MemoryNotSupportedError
    # at the _require_client boundary; they reach the adapter (which would then
    # attempt the wrapped client call). We assert the boundary is crossed by
    # confirming _require_client returns the adapter rather than raising.
    memory = Memory(base_url="http://oe.local")
    try:
        client = memory._require_client()
    except MemoryNotSupportedError:  # pragma: no cover - guards the regression
        pytest.fail("direct mode must expose a CRUD client")
    assert client is memory._client


def test_direct_crud_adapter_shares_runtime_transport():
    # CRUD adapter shares the runtime's transport — one httpx.Client for both,
    # so the parametrized close test above releases the adapter too.
    memory = Memory(base_url="http://oe.local")
    assert memory._client._t is memory._runtime._transport
