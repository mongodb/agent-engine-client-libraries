"""Explicit TenantRuntime memory calls raise the memory SDK's typed errors.

An agent must be able to tell a failed memory call from a record that does not
exist, as it can in TypeScript. Not-found reads still return None and empty
searches still return []. The failure tests run through the real MemoryClient
against a mocked HTTP transport, so they cover the client's error mapping too.
"""

import logging
from typing import Any, Callable
from unittest.mock import Mock

import httpx
import pytest
from agent_engine_sdk_memory._client import MemoryClient
from agent_engine_sdk_memory.errors import (
    MemoryAPIError,
    MemoryAuthError,
    MemoryConnectionError,
    MemoryNotProvisionedError,
    MemoryServerError,
)

from agent_engine_runner_shared.memory import build_context, write_turn_to_memory
from agent_engine_runner_shared.runtime import TenantRuntime

# (runtime method, its arguments)
EXPLICIT_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("save_semantic", {"text": "t", "label": "l", "user_id": "u"}),
    ("search_semantic", {"query": "q"}),
    ("get_semantic", {"label": "l"}),
    ("create_taxonomic", {"domain": "d", "term": "t", "definition": "x", "user_id": "u"}),
    ("search_taxonomic", {"query": "q"}),
    ("get_taxonomic_term", {"domain": "d", "term": "t"}),
    ("list_taxonomic_domains", {}),
    ("save_episode", {"title": "t", "content": "c", "user_id": "u", "session_id": "s"}),
    ("search_episodes", {"query": "q"}),
    ("list_episodes", {}),
    ("discover_procedures", {"query": "q"}),
    ("get_procedure", {"procedure_name": "p"}),
    ("save_procedure", {"procedure": "p", "description": "d", "content": "c", "user_id": "u"}),
]


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


# (what the memory server does, the error the agent sees)
FAILURES: list[tuple[str, Callable[[httpx.Request], httpx.Response], type[MemoryAPIError]]] = [
    ("server error", lambda _: httpx.Response(503, json={"detail": "down"}), MemoryServerError),
    ("unauthorized", lambda _: httpx.Response(401, json={"error": "no"}), MemoryAuthError),
    (
        "not provisioned",
        lambda _: httpx.Response(503, json={"code": "AGENT_NOT_DEPLOYED", "error": "starting"}),
        MemoryNotProvisionedError,
    ),
    ("unreachable", _unreachable, MemoryConnectionError),
    ("malformed body", lambda _: httpx.Response(200, text="<html>proxy</html>"), MemoryServerError),
    ("wrong-shape body", lambda _: httpx.Response(200, json=["unexpected"]), MemoryServerError),
]

# (runtime method, its arguments, the memory client method that finds nothing)
EMPTY_RESULTS: list[tuple[str, dict[str, Any], str]] = [
    ("search_semantic", {"query": "q"}, "fetch_semantic_memories"),
    ("search_taxonomic", {"query": "q"}, "fetch_taxonomic_memories"),
    ("list_taxonomic_domains", {}, "get_distinct_domains"),
    ("search_episodes", {"query": "q"}, "fetch_episodic_memories"),
    ("list_episodes", {}, "list_episodic"),
    ("discover_procedures", {"query": "q"}, "discover_procedures"),
]


def _runtime(engine: Any) -> TenantRuntime:
    rt = TenantRuntime.__new__(TenantRuntime)
    rt._memory_engine = engine
    rt.org_id = "o"
    rt.project_id = "p"
    return rt


def _client_runtime(handler: Callable[[httpx.Request], httpx.Response]) -> TenantRuntime:
    return _runtime(MemoryClient("http://memory", transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(("method", "kwargs"), EXPLICIT_CALLS)
@pytest.mark.parametrize(("failure", "handler", "error"), FAILURES)
def test_failed_call_raises_the_typed_error(
    method: str,
    kwargs: dict[str, Any],
    failure: str,
    handler: Callable[[httpx.Request], httpx.Response],
    error: type[MemoryAPIError],
) -> None:
    with pytest.raises(error):
        getattr(_client_runtime(handler), method)(**kwargs)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_semantic", {"label": "l"}),
        ("get_taxonomic_term", {"domain": "d", "term": "t"}),
        ("get_procedure", {"procedure_name": "p"}),
    ],
)
def test_lookup_of_a_missing_record_returns_none(method: str, kwargs: dict[str, Any]) -> None:
    runtime = _client_runtime(lambda _: httpx.Response(404, json={"detail": "not found"}))

    assert getattr(runtime, method)(**kwargs) is None


@pytest.mark.parametrize(("method", "kwargs", "engine_method"), EMPTY_RESULTS)
def test_search_or_list_with_no_match_returns_empty(
    method: str, kwargs: dict[str, Any], engine_method: str
) -> None:
    engine = Mock()
    getattr(engine, engine_method).return_value = []

    assert getattr(_runtime(engine), method)(**kwargs) == []


def _leaky_server_error(_: httpx.Request) -> httpx.Response:
    # The memory server puts str(exception) into a 5xx detail.
    return httpx.Response(500, json={"detail": "write failed: mongodb://svc:hunter2@db/x"})


def test_failed_context_read_logs_the_status_but_not_the_server_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MemoryClient("http://memory", transport=httpx.MockTransport(_leaky_server_error))

    with caplog.at_level(logging.WARNING):
        context = build_context(client, "q", "s", "o", "u", project_id="p")

    assert context == ""
    assert "MemoryServerError (HTTP 500)" in caplog.text
    assert "hunter2" not in caplog.text
    # An expected memory failure is named by its status; no traceback.
    assert not any(record.exc_info for record in caplog.records)


def test_failed_turn_write_logs_the_status_but_not_the_server_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MemoryClient("http://memory", transport=httpx.MockTransport(_leaky_server_error))

    with caplog.at_level(logging.WARNING):
        write_turn_to_memory(client, "hi", [], "s", "o", "u", "p")

    assert "MemoryServerError (HTTP 500)" in caplog.text
    assert "hunter2" not in caplog.text


def test_unexpected_failure_logs_its_redacted_message() -> None:
    from agent_engine_runner_shared.memory import describe_memory_error

    described = describe_memory_error(RuntimeError("dial mongodb://svc:hunter2@db/x"))

    assert described.startswith("RuntimeError: ")
    assert "hunter2" not in described
