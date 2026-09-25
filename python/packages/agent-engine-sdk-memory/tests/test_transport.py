"""The shared HTTP transport: error mapping, extraction, retry, idempotency.

Two layers, both against ``httpx.MockTransport`` with backoff patched to a
no-op so the suite stays sub-second:

* the transport in isolation — ``_HttpTransport`` driven through a tiny
  throwaway subclass that posts to one path, proving error mapping, ``memories``
  extraction, and header injection independently of any runtime;
* retry + idempotency through ``_HttpMemoryRuntime`` (gateway profile), which is
  where the decision of *which* calls carry an idempotency key is made.
"""

import json

import httpx
import pytest
from agent_engine_sdk_memory import _transport
from agent_engine_sdk_memory._http_runtime import _GATEWAY_PROFILE, _HttpMemoryRuntime
from agent_engine_sdk_memory._transport import _HttpTransport
from agent_engine_sdk_memory.errors import (
    MemoryAuthError,
    MemoryBadRequestError,
    MemoryConnectionError,
    MemoryServerError,
)

API_KEY = "test-api-key"
BASE_URL = "https://gateway.example.com"
PATH = "/api/v1/memory/search"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make backoff instantaneous and record per-attempt delays."""
    delays: list[float] = []
    monkeypatch.setattr(_transport.time, "sleep", lambda d: delays.append(d))
    return delays


# =====================================================================
# Transport in isolation (_HttpTransport via a throwaway subclass)
# =====================================================================


def _mk(handler, headers=None):
    return _HttpTransport(
        base_url="http://oe.local",
        headers=headers or {},
        transport=httpx.MockTransport(handler),
    )


def test_retry_resends_same_body_on_503_then_succeeds():
    sent_bodies: list[dict] = []
    responses = iter(
        [
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(200, json={"memories": []}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content))
        return next(responses)

    rt = _mk(handler)
    body = {"query": "q", "idempotency_key": "idem-1"}
    resp = rt._post(PATH, body)

    assert resp.status_code == 200
    assert len(sent_bodies) == 2
    # The identical body (idempotency_key included) is resent on every attempt.
    assert sent_bodies[0] == sent_bodies[1] == body


def test_4xx_maps_to_bad_request():
    rt = _mk(lambda _req: httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(MemoryBadRequestError):
        rt._post(PATH, {})


def test_5xx_maps_to_server_error():
    rt = _mk(lambda _req: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(MemoryServerError):
        rt._post(PATH, {})


def test_401_maps_to_auth_error():
    rt = _mk(lambda _req: httpx.Response(401, json={"error": "nope"}))
    with pytest.raises(MemoryAuthError):
        rt._post(PATH, {})


def test_memories_extraction_handles_missing_and_bad_shape():
    rt_missing = _mk(lambda _req: httpx.Response(200, json={}))
    assert rt_missing._memories(rt_missing._post(PATH, {})) == []

    rt_bad = _mk(lambda _req: httpx.Response(200, json={"memories": "nope"}))
    with pytest.raises(MemoryServerError):
        rt_bad._memories(rt_bad._post(PATH, {}))


def test_no_auth_header_when_headers_empty():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memories": []})

    rt = _mk(handler, headers={})
    rt._post(PATH, {})
    assert "Authorization" not in seen[-1].headers


def test_post_with_retry_false_makes_a_single_attempt_on_503():
    # Non-idempotent writes opt out of retry so a transient 5xx after the server
    # committed cannot duplicate the record.
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, json={"error": "x"})

    rt = _mk(handler)
    with pytest.raises(MemoryServerError):
        rt._post(PATH, {}, retry=False)
    assert len(attempts) == 1


# =====================================================================
# Retry + idempotency through the runtime (gateway profile)
# =====================================================================


class _Recorder:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []


def _runtime(handler):
    return _HttpMemoryRuntime(
        base_url=BASE_URL,
        profile=_GATEWAY_PROFILE,
        project_id="proj-1",
        headers={"Authorization": f"Bearer {API_KEY}"},
        transport=httpx.MockTransport(handler),
    )


def _capturing_handler(rec, responses):
    """Return a handler that replays ``responses`` (status or exc) per attempt."""
    seq = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        rec.bodies.append(json.loads(request.content) if request.content else {})
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    return handler


def test_record_turn_retries_and_resends_same_idempotency_key():
    rec = _Recorder()
    ok = httpx.Response(201, json={"id": "t1", "session_id": "s1", "turn_seq": 1})
    handler = _capturing_handler(
        rec,
        [
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(502, json={"error": "bad gateway"}),
            ok,
        ],
    )
    rt = _runtime(handler)
    result = rt.record_turn(
        role="user",
        content="hi",
        session_id="s1",
        user_id="u1",
        idempotency_key="idem-xyz",
    )
    assert result.id == "t1"
    assert len(rec.requests) == 3
    keys = {b.get("idempotency_key") for b in rec.bodies}
    assert keys == {"idem-xyz"}


def test_read_calls_never_carry_idempotency_key():
    rec = _Recorder()
    memories = httpx.Response(200, json={"memories": []})
    ctx = httpx.Response(
        200,
        json={
            "formatted_context": "c",
            "metadata": {"token_count": 0, "memory_counts": {}, "timing": {}},
            "selected_memories": None,
        },
    )

    def run(call, response):
        rec.requests.clear()
        rec.bodies.clear()
        handler = _capturing_handler(
            rec, [httpx.Response(503, json={"error": "x"}), response]
        )
        rt = _runtime(handler)
        call(rt)
        for body in rec.bodies:
            assert "idempotency_key" not in body

    run(lambda rt: rt.build_context(query="q", user_id="u1"), ctx)
    run(lambda rt: rt.search_semantic(query="q", user_id="u1"), memories)
    run(lambda rt: rt.search_episodes(query="q", user_id="u1"), memories)
    run(lambda rt: rt.discover_procedures(query="q", user_id="u1"), memories)


@pytest.mark.parametrize("status", [502, 503, 504])
def test_retryable_status_then_success(status):
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.Response(status, json={"error": "transient"}),
            httpx.Response(200, json={"memories": []}),
        ],
    )
    rt = _runtime(handler)
    assert rt.search_semantic(query="q", user_id="u1") == []
    assert len(rec.requests) == 2


def test_three_attempts_then_exhausted_raises_typed_server_error():
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.Response(503, json={"error": "x"}),
            httpx.Response(503, json={"error": "x"}),
            httpx.Response(503, json={"error": "x"}),
        ],
    )
    rt = _runtime(handler)
    with pytest.raises(MemoryServerError):
        rt.search_semantic(query="q", user_id="u1")
    assert len(rec.requests) == 3


def test_connect_error_retries_then_raises_connection_error():
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
        ],
    )
    rt = _runtime(handler)
    with pytest.raises(MemoryConnectionError):
        rt.search_semantic(query="q", user_id="u1")
    assert len(rec.requests) == 3


def test_read_error_retries_then_raises_connection_error():
    # A non-connect/timeout transport error (read/write/protocol) must still
    # map to a typed MemoryConnectionError rather than escape raw.
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.ReadError("reset"),
            httpx.ReadError("reset"),
            httpx.ReadError("reset"),
        ],
    )
    rt = _runtime(handler)
    with pytest.raises(MemoryConnectionError):
        rt.search_semantic(query="q", user_id="u1")
    assert len(rec.requests) == 3


def test_timeout_then_success():
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.TimeoutException("slow"),
            httpx.Response(200, json={"memories": []}),
        ],
    )
    rt = _runtime(handler)
    assert rt.search_semantic(query="q", user_id="u1") == []
    assert len(rec.requests) == 2


@pytest.mark.parametrize(
    "status,exc",
    [
        (400, MemoryBadRequestError),
        (401, MemoryAuthError),
        (403, MemoryAuthError),
        (500, MemoryServerError),
    ],
)
def test_non_retryable_status_raises_immediately(status, exc):
    rec = _Recorder()
    handler = _capturing_handler(
        rec, [httpx.Response(status, json={"error": "no", "code": "C"})]
    )
    rt = _runtime(handler)
    with pytest.raises(exc):
        rt.search_semantic(query="q", user_id="u1")
    assert len(rec.requests) == 1


def test_backoff_grows_exponentially(_no_sleep, monkeypatch):
    # Pin jitter to 0 so the assertion tests the deterministic exponential
    # base growth rather than a probabilistic ordering (jitter ranges overlap).
    monkeypatch.setattr(_transport.random, "uniform", lambda _a, _b: 0.0)
    rec = _Recorder()
    handler = _capturing_handler(
        rec,
        [
            httpx.Response(503, json={"error": "x"}),
            httpx.Response(503, json={"error": "x"}),
            httpx.Response(200, json={"memories": []}),
        ],
    )
    rt = _runtime(handler)
    rt.search_semantic(query="q", user_id="u1")
    # Two retries -> two recorded sleeps; the second is exactly double the first.
    assert len(_no_sleep) == 2
    assert _no_sleep[1] == 2 * _no_sleep[0]
