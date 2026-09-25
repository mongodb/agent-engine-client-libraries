"""Custom-type SDK surface: models, client ops, and facade behavior."""

from unittest.mock import Mock

import httpx
import pytest
from agent_engine_sdk_memory import (
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
    Memory,
    RetrievedCustomMemory,
)
from agent_engine_sdk_memory._client import MemoryClient
from agent_engine_sdk_memory.errors import (
    MemoryBadRequestError,
    MemoryClientError,
    MemoryNotSupportedError,
)


class TestCustomTypeModels:
    def test_save_result_parses_server_echo(self) -> None:
        r = CustomMemorySaveResult.model_validate(
            {
                "id": "m1",
                "type": "tickets",
                "tags": {"queue": "billing", "priority": 3},
                "has_embedding": True,
            }
        )
        assert r.tags["priority"] == 3
        assert r.tags["queue"] == "billing"

    def test_retrieve_result_parses_typed_tags(self) -> None:
        r = CustomMemoryRetrieveResult.model_validate(
            {
                "results": [
                    {
                        "id": "m1",
                        "type": "tickets",
                        "content": "c",
                        "tags": {"open": True, "score": 1.5},
                        "score": 0.9,
                        "user_id": "u1",
                    }
                ],
                "count": 1,
            }
        )
        assert isinstance(r.results[0], RetrievedCustomMemory)
        assert r.results[0].tags == {"open": True, "score": 1.5}
        assert r.results[0].contextual_metadata is None
        assert r.results[0].agent_id is None


class TestMemoryClientCustomOps:
    def test_create_custom_posts_to_types_route(self, mock_httpx_client) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "m1",
            "type": "tickets",
            "tags": {"queue": "billing"},
            "has_embedding": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.create_custom(
            memory_type="tickets",
            content="c",
            tags={"queue": "billing"},
            contextual_metadata={"k": "v"},
        )

        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == ("http://localhost:8081/api/v1/memory/types/tickets")
        assert call_args[1]["json"] == {
            "content": "c",
            "tags": {"queue": "billing"},
            "contextual_metadata": {"k": "v"},
        }
        assert result.id == "m1"

    def test_create_custom_omits_none_fields(self, mock_httpx_client) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "m1",
            "type": "tickets",
            "tags": {},
            "has_embedding": True,
        }
        mock_httpx_client.post.return_value = mock_response

        MemoryClient("http://x").create_custom(memory_type="tickets", content="c")
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body == {"content": "c"}

    def test_retrieve_custom_posts_query_and_top_k(self, mock_httpx_client) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {"results": [], "count": 0}
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.retrieve_custom(memory_type="tickets", query="q", top_k=5)

        call_args = mock_httpx_client.post.call_args
        assert (
            call_args[0][0]
            == "http://localhost:8081/api/v1/memory/types/tickets/retrieve"
        )
        assert call_args[1]["json"] == {"query": "q", "top_k": 5}
        assert result.count == 0


class FakeCustomClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: dict[str, dict] = {}
        self._error = error

    def create_custom(self, **kwargs):
        self.calls["create_custom"] = kwargs
        if self._error:
            raise self._error
        return CustomMemorySaveResult(
            id="m1", type=kwargs["memory_type"], tags={}, has_embedding=True
        )

    def retrieve_custom(self, **kwargs):
        self.calls["retrieve_custom"] = kwargs
        if self._error:
            raise self._error
        return CustomMemoryRetrieveResult(results=[], count=0)


class _NullRuntime:
    """Facade requires a runtime arg; custom ops never touch it."""


def _mem(client: FakeCustomClient) -> Memory:
    return Memory(runtime=_NullRuntime(), client=client)  # type: ignore[arg-type]


class TestFacadeSave:
    def test_save_delegates_to_client(self) -> None:
        client = FakeCustomClient()
        r = _mem(client).save("tickets", "c", tags={"queue": "billing"})
        assert client.calls["create_custom"] == {
            "memory_type": "tickets",
            "content": "c",
            "tags": {"queue": "billing"},
            "contextual_metadata": None,
        }
        assert r.id == "m1"

    def test_save_rejects_builtin_type_before_http(self) -> None:
        client = FakeCustomClient()
        with pytest.raises(MemoryClientError, match="built-in memory type"):
            _mem(client).save("semantic", "c")
        assert client.calls == {}

    def test_save_rejects_bad_tag_syntax_before_http(self) -> None:
        client = FakeCustomClient()
        with pytest.raises(
            MemoryClientError, match="nesting is at most one level deep"
        ):
            _mem(client).save("tickets", "c", tags={"a.b.c": "x"})
        assert client.calls == {}

    def test_save_maps_missing_route_to_not_supported(self) -> None:
        err = MemoryBadRequestError("Not Found", status=404, code=None)
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="does not support custom memory types"
        ):
            _mem(client).save("tickets", "c")

    def test_save_passes_through_unknown_type_404(self) -> None:
        err = MemoryBadRequestError(
            "unknown custom memory type 'tickets'", status=404, code=None
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(MemoryBadRequestError, match="unknown custom memory type"):
            _mem(client).save("tickets", "c")


class TestFacadeRetrieve:
    def test_retrieve_delegates_with_default_top_k(self) -> None:
        client = FakeCustomClient()
        _mem(client).retrieve("tickets", "q")
        assert client.calls["retrieve_custom"] == {
            "memory_type": "tickets",
            "query": "q",
            "tags": None,
            "top_k": 10,
        }

    def test_retrieve_validates_type_and_tags(self) -> None:
        client = FakeCustomClient()
        with pytest.raises(MemoryClientError, match="built-in memory type"):
            _mem(client).retrieve("episodic", "q")
        with pytest.raises(MemoryClientError, match="non-empty path"):
            _mem(client).retrieve("tickets", "q", tags={"": "x"})
        assert client.calls == {}

    def test_retrieve_maps_missing_route_to_not_supported(self) -> None:
        err = MemoryBadRequestError("Not Found", status=404, code=None)
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="does not support custom memory types"
        ):
            _mem(client).retrieve("tickets", "q")


def _real_client(handler) -> MemoryClient:
    """A real MemoryClient wired to an httpx.MockTransport, no mocked internals."""
    return MemoryClient("http://localhost:8081", transport=httpx.MockTransport(handler))


class TestFacadeOverRealMemoryClient:
    """Exercises the real MemoryClient (no mocked internals) through the facade,
    so a raw httpx error can never masquerade as a typed one."""

    def test_bare_404_maps_to_not_supported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        mem = Memory(runtime=_NullRuntime(), client=_real_client(handler))  # type: ignore[arg-type]
        with pytest.raises(
            MemoryNotSupportedError, match="does not support custom memory types"
        ):
            mem.save("tickets", "c")

    def test_structured_404_body_passes_through_as_bad_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"detail": "unknown custom memory type 'tickets'"}
            )

        mem = Memory(runtime=_NullRuntime(), client=_real_client(handler))  # type: ignore[arg-type]
        with pytest.raises(MemoryBadRequestError, match="unknown custom memory type"):
            mem.retrieve("tickets", "q")

    def test_400_with_tag_error_body_surfaces_server_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": "tag 'urgent' is not declared for custom type 'tickets'"
                },
            )

        mem = Memory(runtime=_NullRuntime(), client=_real_client(handler))  # type: ignore[arg-type]
        with pytest.raises(MemoryBadRequestError) as exc_info:
            mem.save("tickets", "c", tags={"urgent": "true"})
        assert "tag 'urgent' is not declared" in str(exc_info.value)


class TestFastAPIDetailMessages:
    """A structured server message must surface as the message, not raw JSON.

    The memory server returns FastAPI's {"detail": ...} shape, which the
    transport previously did not read — callers saw the whole JSON body.
    """

    def test_detail_becomes_the_error_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "detail": "tag 'assignee' is not declared for custom type 'tickets'"
                },
            )

        client = MemoryClient(
            "http://memory.test",
            api_prefix="/api/v1/memory",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MemoryBadRequestError) as exc:
            client.create_custom(
                memory_type="tickets", content="c", tags={"assignee": "sam"}
            )
        assert (
            str(exc.value) == "tag 'assignee' is not declared for custom type 'tickets'"
        )
        assert "{" not in str(exc.value)

    def test_non_string_detail_falls_back_to_raw_text(self) -> None:
        # FastAPI validation errors put a list under "detail"; it must not be
        # coerced into the message.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422, json={"detail": [{"loc": ["body"], "msg": "bad"}]}
            )

        client = MemoryClient(
            "http://memory.test",
            api_prefix="/api/v1/memory",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(MemoryBadRequestError) as exc:
            client.create_custom(memory_type="tickets", content="c")
        assert "loc" in str(exc.value)


def test_retrieval_ignores_the_server_score_field() -> None:
    """The server still projects `score`; the SDK deliberately does not expose it.

    Parsing must tolerate the extra key so the two can diverge without the
    client breaking.
    """
    result = CustomMemoryRetrieveResult.model_validate(
        {
            "results": [
                {
                    "id": "m1",
                    "type": "tickets",
                    "content": "c",
                    "tags": {"queue": "billing"},
                    "score": 0.87,
                    "user_id": "u1",
                }
            ],
            "count": 1,
        }
    )
    assert result.count == 1
    assert not hasattr(result.results[0], "score")
    assert result.results[0].tags == {"queue": "billing"}


class TestMethodNotAllowedMapping:
    """405 is treated the same as 404: a platform without these operations.

    A gateway that does not know the route may reject the method rather than the
    path, so both statuses must reach the same clear error.
    """

    def test_405_maps_to_not_supported(self) -> None:
        err = MemoryBadRequestError("Method Not Allowed", status=405, code=None)
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="does not support custom memory types"
        ):
            _mem(client).save("tickets", "c")

    def test_405_maps_to_not_supported_on_retrieve(self) -> None:
        err = MemoryBadRequestError("Method Not Allowed", status=405, code=None)
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="does not support custom memory types"
        ):
            _mem(client).retrieve("tickets", "q")

    def test_405_naming_an_unknown_type_still_passes_through(self) -> None:
        # The structured unknown-type body wins over the status-based mapping.
        err = MemoryBadRequestError(
            "unknown custom memory type 'tickets'", status=405, code=None
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(MemoryBadRequestError, match="unknown custom memory type"):
            _mem(client).save("tickets", "c")


class TestFeatureFlagOffMapping:
    """The gateway rejects flag-off deployments with a 400 whose code is the
    generic INVALID_REQUEST, so the message substring is the only stable
    discriminator; that 400 must surface as MemoryNotSupportedError."""

    def test_flag_off_400_maps_to_not_supported_on_save(self) -> None:
        err = MemoryBadRequestError(
            "custom_memory_types is not enabled on this deployment",
            status=400,
            code="INVALID_REQUEST",
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="disabled on this deployment"
        ) as info:
            _mem(client).save("tickets", "c")
        assert info.value.__cause__ is err

    def test_flag_off_400_maps_to_not_supported_on_retrieve(self) -> None:
        err = MemoryBadRequestError(
            "custom_memory_types is not enabled on this deployment",
            status=400,
            code="INVALID_REQUEST",
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(
            MemoryNotSupportedError, match="disabled on this deployment"
        ):
            _mem(client).retrieve("tickets", "q")

    def test_other_400s_pass_through_untouched(self) -> None:
        err = MemoryBadRequestError(
            "tag 'urgent' is not declared for custom type 'tickets'",
            status=400,
            code="INVALID_REQUEST",
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(MemoryBadRequestError) as info:
            _mem(client).save("tickets", "c", tags={"urgent": "true"})
        assert info.value is err


class TestExceptionChaining:
    """A pass-through must not be chained to itself.

    `raise exc from exc` sets a self-referential __cause__ and suppresses
    __context__, which misleads error reporters that walk the cause chain.
    """

    def test_passthrough_has_no_self_cause(self) -> None:
        err = MemoryBadRequestError(
            "unknown custom memory type 'tickets'", status=404, code=None
        )
        client = FakeCustomClient(error=err)
        with pytest.raises(MemoryBadRequestError) as info:
            _mem(client).save("tickets", "c")
        assert info.value is err
        assert info.value.__cause__ is not info.value
        assert not info.value.__suppress_context__

    def test_mapped_error_keeps_the_original_as_its_cause(self) -> None:
        err = MemoryBadRequestError("Not Found", status=404, code=None)
        client = FakeCustomClient(error=err)
        with pytest.raises(MemoryNotSupportedError) as info:
            _mem(client).retrieve("tickets", "q")
        assert info.value.__cause__ is err
