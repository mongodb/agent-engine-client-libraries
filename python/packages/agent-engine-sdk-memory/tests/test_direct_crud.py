"""Empty-tenancy CRUD adapter: MockTransport assertions on path + empty tenancy.

`_EmptyTenancyCrudClient` is a thin forwarder whose one job is to inject
`org_id=""`/`project_id=""` on every call; one parametrized test covers that
across all methods. The remaining tests pin the adapter's own logic: the
`summary_text -> ""` transform and the `update_existing` rejection. (Error
mapping is the shared transport's job, covered in test_transport.py.)
"""

import json

import httpx
import pytest
from agent_engine_sdk_memory._direct_crud import _EmptyTenancyCrudClient
from agent_engine_sdk_memory._transport import _HttpTransport
from agent_engine_sdk_memory.errors import MemoryNotSupportedError, MemoryServerError

BASE_URL = "http://oe.local"


class _Recorder:
    """Captures every request a MockTransport handler saw."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []


def _adapter(handler):
    transport = _HttpTransport(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return _EmptyTenancyCrudClient(transport, "/api/v1/memory")


def _route_handler(rec: _Recorder):
    """Branch on method + path, returning a minimal valid body per op."""

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        path = request.url.path
        method = request.method

        if method == "POST" and path == "/api/v1/memory/semantic":
            return httpx.Response(
                201, json={"id": "1", "label": "l", "has_embedding": False}
            )
        if method == "GET" and path == "/api/v1/memory/semantic":
            return httpx.Response(200, json={"entries": []})
        if method == "POST" and path == "/api/v1/memory/episodic":
            return httpx.Response(
                201, json={"id": "1", "title": "t", "has_embedding": False}
            )
        if method == "GET" and path == "/api/v1/memory/episodic":
            return httpx.Response(200, json={"entries": []})
        if method == "POST" and path == "/api/v1/memory/taxonomic":
            return httpx.Response(
                201,
                json={"id": "1", "domain": "d", "term": "x", "has_embedding": False},
            )
        if method == "GET" and path == "/api/v1/memory/taxonomic":
            return httpx.Response(200, json={"entries": []})
        if method == "GET" and path == "/api/v1/memory/taxonomic/domains":
            return httpx.Response(200, json={"domains": ["d"]})
        if method == "POST" and path == "/api/v1/memory/procedural":
            return httpx.Response(
                201, json={"id": "1", "procedure": "p", "has_embedding": False}
            )
        if method == "GET" and path == "/api/v1/memory/procedural":
            return httpx.Response(200, json={"entries": []})
        return httpx.Response(404, json={})

    return handler


# (call, expected method, expected path) for every adapter method. Empty tenancy
# rides the body on writes and the query string on reads, so the assertion below
# branches on method, not per-test.
_CRUD_CALLS = [
    (
        "create_semantic",
        lambda a: a.create_semantic(label="l", text="t", user_id="u"),
        "POST",
        "/api/v1/memory/semantic",
    ),
    (
        "get_semantic",
        lambda a: a.get_semantic(label="l", user_id="u"),
        "GET",
        "/api/v1/memory/semantic",
    ),
    (
        "create_episodic",
        lambda a: a.create_episodic(
            title="t", content="c", user_id="u", session_id="s"
        ),
        "POST",
        "/api/v1/memory/episodic",
    ),
    (
        "list_episodic",
        lambda a: a.list_episodic(user_id="u"),
        "GET",
        "/api/v1/memory/episodic",
    ),
    (
        "create_taxonomic",
        lambda a: a.create_taxonomic(
            domain="d", term="x", definition="def", user_id="u"
        ),
        "POST",
        "/api/v1/memory/taxonomic",
    ),
    (
        "get_taxonomic",
        lambda a: a.get_taxonomic(domain="d", term="x"),
        "GET",
        "/api/v1/memory/taxonomic",
    ),
    (
        "get_distinct_domains",
        lambda a: a.get_distinct_domains(),
        "GET",
        "/api/v1/memory/taxonomic/domains",
    ),
    (
        "create_procedural",
        lambda a: a.create_procedural(
            procedure="p", description="d", content="c", user_id="u"
        ),
        "POST",
        "/api/v1/memory/procedural",
    ),
    (
        "get_procedural",
        lambda a: a.get_procedural(procedure="p", user_id="u"),
        "GET",
        "/api/v1/memory/procedural",
    ),
]


@pytest.mark.parametrize(
    "call, method, path",
    [(c, m, p) for _, c, m, p in _CRUD_CALLS],
    ids=[name for name, *_ in _CRUD_CALLS],
)
def test_empty_tenancy_on_every_forwarded_call(call, method, path):
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))

    call(adapter)

    req = rec.requests[-1]
    assert req.method == method
    assert req.url.path == path
    if method == "POST":
        body = json.loads(req.content)
        assert body["org_id"] == ""
        assert body["project_id"] == ""
    else:
        assert req.url.params.get("org_id") == ""
        assert req.url.params.get("project_id") == ""


def test_create_episodic_passes_empty_summary():
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))

    adapter.create_episodic(
        title="t", content="c", user_id="u", session_id="s", summary_text=None
    )

    body = json.loads(rec.requests[-1].content)
    assert body["summary_text"] == ""


def test_create_does_not_retry_on_transient_5xx():
    # Creates are non-idempotent (no idempotency key), so a 503 must surface
    # immediately rather than risk a duplicate record via the retry loop.
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    adapter = _adapter(handler)
    with pytest.raises(MemoryServerError):
        adapter.create_semantic(label="l", text="t", user_id="u")
    assert len(rec.requests) == 1


def test_create_maps_malformed_2xx_body_to_typed_error():
    # A 2xx with a non-JSON (or non-object) body must surface as a typed
    # MemoryServerError, not a raw JSON/validation exception.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, text="not json")

    adapter = _adapter(handler)
    with pytest.raises(MemoryServerError):
        adapter.create_semantic(label="l", text="t", user_id="u")


def test_create_procedural_rejects_update_existing():
    # The HTTP create route has no update-existing semantics; the adapter must
    # surface a typed error rather than forward a flag the server ignores.
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    with pytest.raises(MemoryNotSupportedError, match="update_existing"):
        adapter.create_procedural(
            procedure="p",
            description="d",
            content="c",
            user_id="u",
            update_existing=True,
        )
    assert rec.requests == []  # no request issued


def test_create_procedural_does_not_send_update_existing_field():
    # Default path: the no-op flag is not put on the wire at all.
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    adapter.create_procedural(procedure="p", description="d", content="c", user_id="u")
    body = json.loads(rec.requests[-1].content)
    assert "update_existing" not in body


def test_create_taxonomic_forwards_metadata():
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    adapter.create_taxonomic(
        domain="d", term="x", definition="def", user_id="u", metadata={"k": "v"}
    )
    body = json.loads(rec.requests[-1].content)
    assert body["metadata"] == {"k": "v"}


def test_create_taxonomic_omits_metadata_when_absent():
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    adapter.create_taxonomic(domain="d", term="x", definition="def", user_id="u")
    body = json.loads(rec.requests[-1].content)
    assert "metadata" not in body


def test_create_procedural_forwards_metadata():
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    adapter.create_procedural(
        procedure="p", description="d", content="c", user_id="u", metadata={"k": "v"}
    )
    body = json.loads(rec.requests[-1].content)
    assert body["metadata"] == {"k": "v"}


def test_create_procedural_omits_metadata_when_absent():
    rec = _Recorder()
    adapter = _adapter(_route_handler(rec))
    adapter.create_procedural(procedure="p", description="d", content="c", user_id="u")
    body = json.loads(rec.requests[-1].content)
    assert "metadata" not in body
