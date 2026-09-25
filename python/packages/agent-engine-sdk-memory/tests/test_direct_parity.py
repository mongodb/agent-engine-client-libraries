"""Endpoint-profile call-site parity.

The facade calls the runtime through one set of keyword-only expressions; this
suite proves those expressions back the same facade calls whichever endpoint
profile the runtime carries, with no endpoint branching in the facade. There is
one runtime class, so signature parity is structural; what remains worth pinning
is that its methods are keyword-only and that it satisfies the Protocol under
either profile.

The per-endpoint wire differences (auth header, empty tenancy, taxonomic route)
are pinned in test_http_runtime.py; identity-resolution precedence in
test_identity.py and test_memory_facade.py; and the shared retry/idempotency
loop in test_transport.py.
"""

import inspect

import httpx
from agent_engine_sdk_memory import Memory, MemoryRequestContext
from agent_engine_sdk_memory._http_runtime import (
    _GATEWAY_PROFILE,
    _OE_PROFILE,
    _HttpMemoryRuntime,
)
from agent_engine_sdk_memory.models import (
    ContextResponse,
    MemoryChunk,
    WriteTurnResult,
)
from agent_engine_sdk_memory.protocol import MemoryRuntime

_RUNTIME_METHODS = (
    "record_turn",
    "build_context",
    "search_semantic",
    "search_episodes",
    "search_taxonomic",
    "discover_procedures",
)


# Exact routes each mode is expected to hit. The Gateway (project_id set) is
# project-scoped (/api/v1/projects/{id}/memory/*); direct OE is flat
# (/api/v1/memory/*). Both share the same /turns,/context,/search operation
# surface; all searches (including taxonomic) dispatch through
# /search. Matching on the full path means a wrong route 404s here and fails
# the test.
_GW_PREFIX = "/api/v1/projects/proj-1/memory"
_OE_PREFIX = "/api/v1/memory"
_TURN_PATHS = {f"{_OE_PREFIX}/turns", f"{_GW_PREFIX}/turns"}
_CONTEXT_PATHS = {f"{_OE_PREFIX}/context", f"{_GW_PREFIX}/context"}
_MEMORY_LIST_PATHS = {f"{_OE_PREFIX}/search", f"{_GW_PREFIX}/search"}


def _handler(request: httpx.Request) -> httpx.Response:
    """Serve both the api-key and direct route shapes off one handler.

    Both: POST /api/v1/memory/{turns,context,search}
    """
    path = request.url.path
    if path in _TURN_PATHS:
        return httpx.Response(200, json={"id": "t1", "session_id": "s", "turn_seq": 1})
    if path in _CONTEXT_PATHS:
        return httpx.Response(200, json={"formatted_context": "", "metadata": {}})
    if path in _MEMORY_LIST_PATHS:
        return httpx.Response(200, json={"memories": []})
    return httpx.Response(404, json={"error": f"unhandled route {path}"})


def _gateway_memory() -> Memory:
    return Memory(
        runtime=_HttpMemoryRuntime(
            base_url="http://gw",
            profile=_GATEWAY_PROFILE,
            project_id="proj-1",
            headers={"Authorization": "Bearer k"},
            transport=httpx.MockTransport(_handler),
        ),
    )


def _oe_memory() -> Memory:
    return Memory(
        runtime=_HttpMemoryRuntime(
            base_url="http://oe",
            profile=_OE_PROFILE,
            headers=None,
            transport=httpx.MockTransport(_handler),
        )
    )


def test_runtime_satisfies_protocol_under_both_profiles():
    assert isinstance(
        _HttpMemoryRuntime(base_url="http://x", profile=_OE_PROFILE), MemoryRuntime
    )
    assert isinstance(
        _HttpMemoryRuntime(
            base_url="http://x", profile=_GATEWAY_PROFILE, project_id="p"
        ),
        MemoryRuntime,
    )


def test_runtime_methods_are_keyword_only():
    # The facade calls the runtime with one keyword expression regardless of
    # profile, so every runtime arg (beyond self) must be keyword-only.
    for name in _RUNTIME_METHODS:
        sig = inspect.signature(getattr(_HttpMemoryRuntime, name))
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{name}.{pname}"


def test_facade_record_turn_identical_call_site():
    ctx = MemoryRequestContext(user_id="u", session_id="s")
    for memory in (_gateway_memory(), _oe_memory()):
        result = memory.bind(ctx).record_turn(role="user", content="hi")
        assert isinstance(result, WriteTurnResult)


def test_facade_search_episodes_identical_call_site():
    # Episodic search works under both profiles (semantic is gated in OE mode),
    # so it is the cross-profile call-site parity check.
    for memory in (_gateway_memory(), _oe_memory()):
        result = memory.search_episodes("q", user_id="u")
        assert isinstance(result, list)
        assert all(isinstance(c, MemoryChunk) for c in result)


def test_facade_build_context_identical_call_site():
    for memory in (_gateway_memory(), _oe_memory()):
        result = memory.build_context("q", user_id="u")
        assert isinstance(result, ContextResponse)
