"""The shared ``_HttpMemoryRuntime`` under both endpoint profiles.

Gateway and OE are the same runtime with different inputs. This file covers
three layers against ``httpx.MockTransport``:

* request-body building, response parsing, and typed-error mapping — shared
  code, exercised through the gateway profile;
* OE-only behavior — deterministic ``enabled_sources`` ordering;
* the endpoint-profile axes — tool-turn policy, ranked search availability
  (including taxonomic), and CRUD availability; plus auth being an independent
  header rather than a runtime subtype.

The transport's retry loop and idempotency handling live in test_transport.py.
"""

import json

import httpx
import pytest
from agent_engine_sdk_memory._http_runtime import (
    _GATEWAY_PROFILE,
    _OE_PROFILE,
    _HttpMemoryRuntime,
)
from agent_engine_sdk_memory.errors import (
    MemoryBadRequestError,
    MemoryNotProvisionedError,
    MemoryRouteNotFoundError,
    MemoryServerError,
)
from agent_engine_sdk_memory.models import (
    ContextResponse,
    MemoryChunk,
    SourceSpec,
    WriteTurnResult,
)

API_KEY = "test-api-key"


class _Recorder:
    """Captures the requests a MockTransport handler saw."""

    def __init__(self):
        self.requests: list[httpx.Request] = []


def _gateway_runtime(handler, project_id="proj-1"):
    return _HttpMemoryRuntime(
        base_url="https://gateway.example.com",
        profile=_GATEWAY_PROFILE,
        project_id=project_id,
        headers={"Authorization": f"Bearer {API_KEY}"},
        transport=httpx.MockTransport(handler),
    )


def _oe_runtime(handler):
    return _HttpMemoryRuntime(
        base_url="http://oe.local",
        profile=_OE_PROFILE,
        headers=None,
        transport=httpx.MockTransport(handler),
    )


# =====================================================================
# Shared request/response/error mapping (exercised via the gateway profile)
# =====================================================================


def test_record_turn_path_body_and_auth():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(201, json={"id": "t1", "session_id": "s1", "turn_seq": 3})

    rt = _gateway_runtime(handler)
    result = rt.record_turn(
        role="user",
        content="hi",
        session_id="s1",
        user_id="u1",
        agent_id="a1",
        idempotency_key="idem-1",
    )
    assert isinstance(result, WriteTurnResult)
    assert result.id == "t1"
    assert result.turn_seq == 3

    req = rec.requests[-1]
    assert req.url.path == "/api/v1/projects/proj-1/memory/turns"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(req.content)
    assert body == {
        "session_id": "s1",
        "user_id": "u1",
        "role": "user",
        "content": "hi",
        "agent_id": "a1",
        "idempotency_key": "idem-1",
    }
    assert "org_id" not in body
    assert "project_id" not in body


def test_record_turn_omits_none_optionals():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt = _gateway_runtime(handler)
    rt.record_turn(role="user", content="hi", session_id="s", user_id="u")

    body = json.loads(rec.requests[-1].content)
    assert "agent_id" not in body
    assert "idempotency_key" not in body
    assert "metadata" not in body


def test_record_turn_forwards_metadata():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt = _gateway_runtime(handler)
    rt.record_turn(
        role="user",
        content="hi",
        session_id="s",
        user_id="u",
        metadata={"channel": "slack"},
    )

    body = json.loads(rec.requests[-1].content)
    assert body["metadata"] == {"channel": "slack"}


def test_build_context_path_and_parse():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(
            200,
            json={
                "formatted_context": "ctx text",
                "metadata": {"token_count": 5, "memory_counts": {}, "timing": {}},
                "selected_memories": None,
            },
        )

    rt = _gateway_runtime(handler)
    resp = rt.build_context(query="q", user_id="u1", session_id="s1", top_k=10)
    assert isinstance(resp, ContextResponse)
    assert resp.formatted_context == "ctx text"

    req = rec.requests[-1]
    assert req.url.path == "/api/v1/projects/proj-1/memory/context"
    body = json.loads(req.content)
    assert body["query"] == "q"
    assert body["user_id"] == "u1"
    assert body["session_id"] == "s1"
    assert body["top_k"] == 10
    assert "org_id" not in body and "project_id" not in body


def _context_ok_handler(recorder: _Recorder):
    def handler(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        return httpx.Response(
            200,
            json={
                "formatted_context": "c",
                "metadata": {"token_count": 0, "memory_counts": {}, "timing": {}},
                "selected_memories": None,
            },
        )

    return handler


@pytest.mark.parametrize(
    "runtime_factory",
    [_gateway_runtime, _oe_runtime],
    ids=["gateway", "oe"],
)
def test_build_context_serializes_max_tokens_when_explicit(runtime_factory):
    rec = _Recorder()
    rt = runtime_factory(_context_ok_handler(rec))
    rt.build_context(query="q", user_id="u1", top_k=10, max_tokens=2048)
    body = json.loads(rec.requests[-1].content)
    assert body["max_tokens"] == 2048
    assert body["top_k"] == 10


@pytest.mark.parametrize(
    "runtime_factory",
    [_gateway_runtime, _oe_runtime],
    ids=["gateway", "oe"],
)
def test_build_context_omits_max_tokens_when_absent(runtime_factory):
    rec = _Recorder()
    rt = runtime_factory(_context_ok_handler(rec))
    rt.build_context(query="q", user_id="u1", top_k=10)
    body = json.loads(rec.requests[-1].content)
    assert "max_tokens" not in body
    assert body["top_k"] == 10


@pytest.mark.parametrize(
    "runtime_factory",
    [_gateway_runtime, _oe_runtime],
    ids=["gateway", "oe"],
)
def test_build_context_serializes_format_style_and_include_memories(runtime_factory):
    rec = _Recorder()
    rt = runtime_factory(_context_ok_handler(rec))
    rt.build_context(
        query="q", user_id="u1", format_style="jinja2", include_memories=True
    )
    body = json.loads(rec.requests[-1].content)
    assert body["format_style"] == "jinja2"
    assert body["include_memories"] is True


@pytest.mark.parametrize(
    "runtime_factory",
    [_gateway_runtime, _oe_runtime],
    ids=["gateway", "oe"],
)
def test_build_context_omits_format_kwargs_when_unset(runtime_factory):
    rec = _Recorder()
    rt = runtime_factory(_context_ok_handler(rec))
    rt.build_context(query="q", user_id="u1")
    body = json.loads(rec.requests[-1].content)
    assert "format_style" not in body
    assert "include_memories" not in body


def test_build_context_from_sources_path_body_and_parse():
    rec = _Recorder()
    rt = _gateway_runtime(_context_ok_handler(rec))
    resp = rt.build_context_from_sources(
        query="q",
        sources=[
            SourceSpec(source="stm", top_k=5),
            SourceSpec(source="semantic", metadata_filter={"channel": "web"}),
        ],
        user_id="u1",
        session_id="s1",
        rerank=True,
        max_tokens=2048,
    )
    assert isinstance(resp, ContextResponse)

    req = rec.requests[-1]
    assert (
        req.url.path == "/api/v1/projects/proj-1/memory/retrieval/context-from-sources"
    )
    body = json.loads(req.content)
    assert body["query"] == "q"
    assert body["user_id"] == "u1"
    assert body["session_id"] == "s1"
    assert body["rerank"] is True
    assert body["max_tokens"] == 2048
    # Sources serialize via model_dump(exclude_none=True): the unset
    # metadata_filter drops out of the stm spec, the set one stays.
    assert body["sources"] == [
        {"source": "stm", "mode": "semantic", "top_k": 5},
        {
            "source": "semantic",
            "mode": "semantic",
            "metadata_filter": {"channel": "web"},
            "top_k": 20,
        },
    ]
    # Tenancy comes from the route/headers, never the body.
    assert "org_id" not in body and "project_id" not in body


@pytest.mark.parametrize(
    ("runtime_factory", "expected_path"),
    [
        (
            _gateway_runtime,
            "/api/v1/projects/proj-1/memory/retrieval/context-from-sources",
        ),
        (_oe_runtime, "/api/v1/memory/retrieval/context-from-sources"),
    ],
    ids=["gateway", "oe"],
)
def test_build_context_from_sources_omits_absent_optionals(
    runtime_factory, expected_path
):
    rec = _Recorder()
    rt = runtime_factory(_context_ok_handler(rec))
    rt.build_context_from_sources(query="q", sources=[SourceSpec(source="semantic")])
    req = rec.requests[-1]
    assert req.url.path == expected_path
    body = json.loads(req.content)
    assert body["rerank"] is False
    for absent in (
        "user_id",
        "session_id",
        "visibility",
        "max_tokens",
        "format_style",
        "include_memories",
    ):
        assert absent not in body


def test_build_context_from_sources_serializes_format_style_and_include_memories():
    rec = _Recorder()
    rt = _gateway_runtime(_context_ok_handler(rec))
    rt.build_context_from_sources(
        query="q",
        sources=[SourceSpec(source="semantic")],
        format_style="openai",
        include_memories=True,
    )
    body = json.loads(rec.requests[-1].content)
    assert body["format_style"] == "openai"
    assert body["include_memories"] is True


def _memories_response(handler_recorder):
    def handler(request: httpx.Request) -> httpx.Response:
        handler_recorder.requests.append(request)
        return httpx.Response(
            200,
            json={
                "memories": [
                    {
                        "id": "m1",
                        "content": "remembered",
                        "source": "semantic",
                        "timestamp": "2026-06-11T00:00:00Z",
                        "similarity_score": 0.9,
                        "metadata": {"procedure": "deploy"},
                    }
                ]
            },
        )

    return handler


def test_search_semantic():
    rec = _Recorder()
    rt = _gateway_runtime(_memories_response(rec))
    chunks = rt.search_semantic(query="q", user_id="u1", top_k=5)
    assert len(chunks) == 1
    assert isinstance(chunks[0], MemoryChunk)
    assert chunks[0].id == "m1"

    req = rec.requests[-1]
    assert req.url.path == "/api/v1/projects/proj-1/memory/search"
    body = json.loads(req.content)
    assert body["type"] == "semantic"
    assert body["query"] == "q"
    assert body["top_k"] == 5
    assert "session_id" not in body


def test_search_tolerates_null_memories():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"memories": None})

    rt = _gateway_runtime(handler)
    assert rt.search_semantic(query="q", user_id="u1") == []


def test_search_non_dict_payload_maps_to_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "shape"])

    rt = _gateway_runtime(handler)
    with pytest.raises(MemoryServerError):
        rt.search_semantic(query="q", user_id="u1")


def test_search_episodes_sends_session_id():
    rec = _Recorder()
    rt = _gateway_runtime(_memories_response(rec))
    rt.search_episodes(query="q", user_id="u1", session_id="s1")
    body = json.loads(rec.requests[-1].content)
    assert body["type"] == "episodic"
    assert body["session_id"] == "s1"


def test_discover_procedures_reshapes():
    rec = _Recorder()
    rt = _gateway_runtime(_memories_response(rec))
    results = rt.discover_procedures(query="q", user_id="u1", tags=["t"])
    assert results == [
        {
            "procedure": "deploy",
            "content": "remembered",
            "score": 0.9,
            "id": "m1",
            "timestamp": "2026-06-11T00:00:00Z",
        }
    ]
    body = json.loads(rec.requests[-1].content)
    assert body["type"] == "procedural"
    assert body["tags"] == ["t"]


def test_discover_procedures_tolerates_null_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "memories": [
                    {
                        "id": "m1",
                        "content": "remembered",
                        "source": "procedural",
                        "similarity_score": 0.5,
                        "metadata": None,
                    }
                ]
            },
        )

    rt = _gateway_runtime(handler)
    results = rt.discover_procedures(query="q", user_id="u1")
    assert results == [
        {
            "procedure": "",
            "content": "remembered",
            "score": 0.5,
            "id": "m1",
            "timestamp": None,
        }
    ]


def test_discover_procedures_tolerates_non_dict_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "memories": [
                    {
                        "id": "m1",
                        "content": "remembered",
                        "similarity_score": 0.5,
                        "metadata": "not-a-dict",
                    }
                ]
            },
        )

    rt = _gateway_runtime(handler)
    results = rt.discover_procedures(query="q", user_id="u1")
    assert results == [
        {
            "procedure": "",
            "content": "remembered",
            "score": 0.5,
            "id": "m1",
            "timestamp": None,
        }
    ]


def test_discover_procedures_curated_keys_win_over_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "memories": [
                    {
                        "id": "m1",
                        "content": "real content",
                        "similarity_score": 0.7,
                        "metadata": {
                            "procedure": "deploy",
                            "content": "META CLOBBER",
                            "score": 999,
                            "extra": "kept",
                        },
                    }
                ]
            },
        )

    rt = _gateway_runtime(handler)
    results = rt.discover_procedures(query="q", user_id="u1")
    assert results == [
        {
            "procedure": "deploy",
            "content": "real content",
            "score": 0.7,
            "extra": "kept",
            "id": "m1",
            "timestamp": None,
        }
    ]


def test_non_dict_json_error_body_maps_to_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        # An intermediary proxy can answer a 5xx with a JSON scalar/array body.
        return httpx.Response(500, json="service unavailable")

    rt = _gateway_runtime(handler)
    with pytest.raises(MemoryServerError) as exc:
        rt.search_semantic(query="q", user_id="u1")
    assert exc.value.status == 500


def test_not_provisioned_400():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "success": False,
                "error": "not deployed",
                "code": "AGENT_NOT_DEPLOYED",
            },
        )

    rt = _gateway_runtime(handler)
    with pytest.raises(MemoryNotProvisionedError) as exc:
        rt.search_semantic(query="q", user_id="u1")
    assert exc.value.code == "AGENT_NOT_DEPLOYED"


def test_non_json_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="plain text crash")

    rt = _gateway_runtime(handler)
    with pytest.raises(MemoryServerError) as exc:
        rt.search_semantic(query="q", user_id="u1")
    assert "plain text crash" in exc.value.message


def test_context_manager_closes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"memories": []})

    with _gateway_runtime(handler) as rt:
        assert rt.search_semantic(query="q", user_id="u1") == []


# =====================================================================
# Tool-call metadata and OE-specific behavior
# =====================================================================


def test_record_turn_forwards_tool_call_metadata():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt = _oe_runtime(handler)
    rt.record_turn(
        role="tool",
        content="tool output",
        session_id="s",
        user_id="u",
        tool_calls=[{"id": "c1", "name": "lookup"}],
        tool_call_id="c1",
        tool_name="lookup",
        is_error=True,
        model_name="gpt-test",
    )

    body = json.loads(rec.requests[-1].content)
    assert body["tool_calls"] == [{"id": "c1", "name": "lookup"}]
    assert body["tool_call_id"] == "c1"
    assert body["tool_name"] == "lookup"
    assert body["is_error"] is True
    assert body["model_name"] == "gpt-test"

    # Gateway profile also forwards tool-call metadata now that the Gateway
    # accepts these fields.
    rec_gw = _Recorder()

    def gw_handler(request: httpx.Request) -> httpx.Response:
        rec_gw.requests.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt_gw = _gateway_runtime(gw_handler)
    rt_gw.record_turn(
        role="tool",
        content="tool output",
        session_id="s",
        user_id="u",
        tool_calls=[{"id": "c1", "name": "lookup"}],
        tool_call_id="c1",
        tool_name="lookup",
        is_error=True,
        model_name="gpt-test",
    )
    gw_body = json.loads(rec_gw.requests[-1].content)
    assert gw_body["tool_calls"] == [{"id": "c1", "name": "lookup"}]
    assert gw_body["tool_call_id"] == "c1"
    assert gw_body["tool_name"] == "lookup"
    assert gw_body["is_error"] is True
    assert gw_body["model_name"] == "gpt-test"


def test_build_context_sorts_enabled_sources():
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(
            200,
            json={
                "formatted_context": "c",
                "metadata": {"token_count": 0, "memory_counts": {}, "timing": {}},
                "selected_memories": None,
            },
        )

    rt = _oe_runtime(handler)
    rt.build_context(query="q", user_id="u1", enabled_sources={"semantic", "episodic"})
    body = json.loads(rec.requests[-1].content)
    assert body["enabled_sources"] == ["episodic", "semantic"]


def test_search_taxonomic_uses_ranked_search_route():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    result = _oe_runtime(handler).search_taxonomic(
        query="q", user_id="u", domain="finance"
    )
    assert result == []
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == "/api/v1/memory/search"
    body = json.loads(req.content)
    assert body["type"] == "taxonomic"
    assert body["query"] == "q"
    assert body["domain"] == "finance"
    assert "org_id" not in body
    assert "project_id" not in body


def test_search_taxonomic_is_ranked_not_get_list_regression_guard():
    # Taxonomic search must dispatch through the ranked /search route, never the
    # unranked GET /taxonomic list route it once used.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    _oe_runtime(handler).search_taxonomic(query="q", user_id="u")
    req = seen[-1]
    assert req.method == "POST"
    assert req.url.path == "/api/v1/memory/search"
    assert not str(req.url).rstrip("/").endswith("/taxonomic")


def test_search_taxonomic_omits_domain_when_absent():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    _oe_runtime(handler).search_taxonomic(query="q", user_id="u")
    body = json.loads(seen[-1].content)
    assert "domain" not in body


def test_oe_semantic_search_makes_http_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    result = _oe_runtime(handler).search_semantic(query="q", user_id="u")
    assert result == []
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["type"] == "semantic"
    assert body["query"] == "q"
    assert "org_id" not in body
    assert "project_id" not in body


def test_oe_procedural_search_makes_http_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    result = _oe_runtime(handler).discover_procedures(query="q", user_id="u")
    assert result == []
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["type"] == "procedural"
    assert body["query"] == "q"
    assert "org_id" not in body
    assert "project_id" not in body


# =====================================================================
# Endpoint-profile axes — what flips between profiles, and what doesn't
# =====================================================================


def _capture():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"entries": [], "count": 0})
        if request.url.path.endswith("/turns"):
            return httpx.Response(
                201, json={"id": "t", "session_id": "s", "turn_seq": 1}
            )
        if request.url.path.endswith("/context"):
            return httpx.Response(200, json={"formatted_context": "", "metadata": {}})
        return httpx.Response(200, json={"memories": []})

    return seen, handler


def test_endpoint_profiles_flip_the_axes():
    assert _GATEWAY_PROFILE.tool_turn_unsupported_reason is None
    assert _OE_PROFILE.tool_turn_unsupported_reason is None

    assert _GATEWAY_PROFILE.taxonomic_unsupported_reason is None
    assert _OE_PROFILE.taxonomic_unsupported_reason is None

    assert _GATEWAY_PROFILE.ranked_search_unsupported_reason is None
    assert _OE_PROFILE.ranked_search_unsupported_reason is None

    assert _GATEWAY_PROFILE.crud_unsupported_reason is None
    assert _OE_PROFILE.crud_unsupported_reason is None


def test_oe_profile_uses_flat_core_loop_paths():
    # project_id-empty backends (local + platform OE) keep the flat surface.
    seen, handler = _capture()
    rt = _oe_runtime(handler)
    rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    assert seen[-1].url.path == "/api/v1/memory/turns"
    rt.build_context(query="q", user_id="u")
    assert seen[-1].url.path == "/api/v1/memory/context"
    rt.search_semantic(query="q", user_id="u")
    assert seen[-1].url.path == "/api/v1/memory/search"


def test_gateway_profile_uses_project_scoped_core_loop_paths():
    # project_id present => the Gateway base path carries the project segment.
    seen, handler = _capture()
    rt = _gateway_runtime(handler, project_id="p123")
    rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    assert seen[-1].url.path == "/api/v1/projects/p123/memory/turns"
    rt.build_context(query="q", user_id="u")
    assert seen[-1].url.path == "/api/v1/projects/p123/memory/context"
    rt.search_semantic(query="q", user_id="u")
    assert seen[-1].url.path == "/api/v1/projects/p123/memory/search"


def test_gateway_profile_requires_project_id():
    # The Gateway base path has a {project_id} placeholder; constructing it
    # without one is a programming error, surfaced at construction not on the wire.
    with pytest.raises(ValueError, match="project_id"):
        _HttpMemoryRuntime(
            base_url="https://gateway.example.com",
            profile=_GATEWAY_PROFILE,
            headers={"Authorization": f"Bearer {API_KEY}"},
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        )


def _not_found_handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404, json={"error": "not found"})

    return handler


def test_project_scoped_404_raises_actionable_route_error():
    # project_id set + a backend that has no project-scoped route (e.g. local OE)
    # => 404 => directional hint to UNSET project_id. No silent flat retry.
    seen: list[httpx.Request] = []
    rt = _gateway_runtime(_not_found_handler(seen), project_id="p123")
    with pytest.raises(MemoryRouteNotFoundError) as ei:
        rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    msg = str(ei.value)
    assert "project_id" in msg
    assert "unset project_id" in msg.lower()
    assert ei.value.status == 404
    assert isinstance(ei.value, MemoryBadRequestError)  # back-compat 4xx handling
    assert len(seen) == 1  # reactive, no second (flat) attempt


def test_flat_404_raises_actionable_route_error():
    # project_id empty + a backend that only serves project-scoped routes (e.g.
    # the hosted Gateway) => 404 => directional hint to SET project_id.
    seen: list[httpx.Request] = []
    rt = _oe_runtime(_not_found_handler(seen))
    with pytest.raises(MemoryRouteNotFoundError) as ei:
        rt.build_context(query="q", user_id="u")
    msg = str(ei.value)
    assert "set project_id" in msg.lower()
    assert "unset" not in msg.lower()
    assert len(seen) == 1


def test_project_id_is_url_encoded_in_path():
    # A clean id passes through unchanged; an id with path/query metacharacters is
    # percent-encoded so it cannot inject segments or a query string (misroute).
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt = _gateway_runtime(handler, project_id="507f1f77bcf86cd799439011")
    rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    assert seen[-1].url.path == "/api/v1/projects/507f1f77bcf86cd799439011/memory/turns"

    seen.clear()
    rt2 = _gateway_runtime(handler, project_id="a/b?c#d")
    rt2.record_turn(role="user", content="hi", session_id="s", user_id="u")
    # Assert on the wire bytes (raw_path): the id is one percent-encoded segment,
    # so no raw '/', '?', or '#' leaks in to inject a segment or query string.
    assert seen[-1].url.raw_path == b"/api/v1/projects/a%2Fb%3Fc%23d/memory/turns"


def test_project_id_surrounding_whitespace_is_stripped():
    # " p1 " must not be encoded into the path as "%20p1%20".
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "t", "session_id": "s", "turn_seq": 1})

    rt = _gateway_runtime(handler, project_id="  p1  ")
    rt.record_turn(role="user", content="hi", session_id="s", user_id="u")
    assert seen[-1].url.path == "/api/v1/projects/p1/memory/turns"


def test_taxonomic_404_raises_actionable_route_error():
    # A flat-profile taxonomic search against a backend that only serves
    # project-scoped routes 404s, and gets the same directional hint as POSTs.
    seen: list[httpx.Request] = []
    rt = _oe_runtime(_not_found_handler(seen))
    with pytest.raises(MemoryRouteNotFoundError):
        rt.search_taxonomic(query="", user_id="u")
    assert len(seen) == 1


def test_not_provisioned_404_is_not_wrapped_as_route_error():
    # A 404 carrying a not-provisioned code must stay MemoryNotProvisionedError —
    # the route-shape hint must not mask a genuine provisioning failure.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": "PROJECT_RUNTIME_FAILED"})

    rt = _gateway_runtime(handler, project_id="p1")
    with pytest.raises(MemoryNotProvisionedError):
        rt.record_turn(role="user", content="hi", session_id="s", user_id="u")


def test_search_404_also_raises_route_error():
    # search_* routes through the same _post wrapper as turns/context.
    seen: list[httpx.Request] = []
    rt = _oe_runtime(_not_found_handler(seen))
    with pytest.raises(MemoryRouteNotFoundError):
        rt.search_semantic(query="q", user_id="u")
    assert len(seen) == 1


def test_tenancy_axis_on_the_wire():
    seen, handler = _capture()
    _gateway_runtime(handler).search_episodes(query="q", user_id="u")
    body = json.loads(seen[-1].content)
    assert "org_id" not in body and "project_id" not in body

    seen, handler = _capture()
    _oe_runtime(handler).search_episodes(query="q", user_id="u")
    body = json.loads(seen[-1].content)
    assert "org_id" not in body and "project_id" not in body


def test_auth_header_axis_on_the_wire():
    seen, handler = _capture()
    _gateway_runtime(handler).search_episodes(query="q", user_id="u")
    assert seen[-1].headers["Authorization"] == f"Bearer {API_KEY}"

    seen, handler = _capture()
    _oe_runtime(handler).search_episodes(query="q", user_id="u")
    assert "Authorization" not in seen[-1].headers


def test_auth_is_independent_of_profile():
    # Auth is a header, not a runtime subtype: an endpoint profile can carry an
    # auth header (the future "hosted OE with an api key") without changing its
    # capability behavior.
    seen, handler = _capture()
    runtime = _HttpMemoryRuntime(
        base_url="http://oe",
        profile=_OE_PROFILE,
        headers={"Authorization": "Bearer k"},
        transport=httpx.MockTransport(handler),
    )
    runtime.search_episodes(query="q", user_id="u")
    assert seen[-1].headers["Authorization"] == "Bearer k"
    body = json.loads(seen[-1].content)
    assert "org_id" not in body and "project_id" not in body


def test_tool_turn_axis_forwards_in_both_gateway_and_oe():
    seen, handler = _capture()
    _gateway_runtime(handler).record_turn(
        role="tool", content="x", session_id="s", user_id="u", tool_call_id="c1"
    )
    body = json.loads(seen[-1].content)
    assert body["tool_call_id"] == "c1"

    seen, handler = _capture()
    _oe_runtime(handler).record_turn(role="tool", content="x", tool_call_id="c1")
    body = json.loads(seen[-1].content)
    assert body["tool_call_id"] == "c1"


def test_taxonomic_searches_via_search_route_in_both_profiles():
    # Taxonomic search is ranked on both profiles: it dispatches
    # through POST /search like the other long-term types, no longer rejected on
    # the Gateway nor a GET-list on the OE.
    seen, handler = _capture()
    _gateway_runtime(handler).search_taxonomic(query="q", user_id="u")
    assert seen[-1].method == "POST"
    assert seen[-1].url.path == "/api/v1/projects/proj-1/memory/search"
    assert json.loads(seen[-1].content)["type"] == "taxonomic"

    seen, handler = _capture()
    _oe_runtime(handler).search_taxonomic(query="q", user_id="u")
    assert seen[-1].method == "POST"
    assert seen[-1].url.path == "/api/v1/memory/search"
    assert json.loads(seen[-1].content)["type"] == "taxonomic"
