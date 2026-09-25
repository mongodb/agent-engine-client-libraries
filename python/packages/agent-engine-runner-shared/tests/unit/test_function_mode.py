"""Tests for function mode (agent_engine_runner_shared.server.function)."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import httpx
import pytest

from agent_engine_runner_shared import hooks
from agent_engine_runner_shared.server.function import ToolFunctionRunner
from agent_engine_runner_shared.utils import RuntimeMode, get_runtime_mode


async def _run(runtime: Any, *, client: Any, metadata_dir: str) -> None:
    """Drive one function-mode execution via the ToolFunctionRunner."""
    await ToolFunctionRunner(runtime, metadata_dir=metadata_dir, client=client).run()


class _FakeResponse:
    def __init__(self, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            # response=self keeps the fake aligned with the production error shape:
            # post_json_with_retries() reads e.response.status_code on a 5xx.
            raise httpx.HTTPStatusError("error", request=Mock(), response=self)


class _AsyncPostStream:
    def __init__(self, post: Any, url: str, payload: Dict[str, Any]) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        return await self._post(self._url, self._payload)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _SyncPostStream:
    def __init__(self, post: Any, url: str, payload: Dict[str, Any]) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    def __enter__(self) -> Any:
        return self._post(self._url, json=self._payload)

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _CaptureClient:
    """Records POSTed result reports; always succeeds."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def post(self, url: str, json: Dict[str, Any]) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse(200)

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Dict[str, Any],
        follow_redirects: bool,
    ) -> _AsyncPostStream:
        return _AsyncPostStream(self.post, url, json)

    async def aclose(self) -> None:
        pass


class _FailingClient:
    """Raises a transient transport error on every POST."""

    async def post(self, url: str, json: Dict[str, Any]) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Dict[str, Any],
        follow_redirects: bool,
    ) -> _AsyncPostStream:
        return _AsyncPostStream(self.post, url, json)

    async def aclose(self) -> None:
        pass


def _make_runtime(tools: Dict[str, Any]) -> Mock:
    runtime = Mock()
    runtime._tools = tools
    runtime._tool_definitions = {name: {} for name in tools}
    runtime._graph_builder = None
    runtime.agent_config.feature_enabled = Mock(return_value=False)
    return runtime


def _write_contract(
    meta_dir: Path,
    *,
    tool_name: str = "greet",
    arguments: Optional[Dict[str, Any]] = None,
    execution_id: str = "exec-1",
    session_id: str = "sess-1",
    oe_url: Optional[str] = "http://oe",
    step: Any = 3,
    include_step: bool = True,
    request_json: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    write_request: bool = True,
) -> str:
    """Write the function-mode metadata into ``meta_dir``; return its path.

    Mirrors what fctr materializes from ``RunRequest.metadata``: a single
    ``request`` file holding the ``ToolFunctionRequest`` envelope
    ``{request, step}``.
    """
    if request_json is None:
        tool_request: Dict[str, Any] = {
            "execution_id": execution_id,
            "tool_name": tool_name,
            "arguments": arguments if arguments is not None else {"name": "ada"},
            "session_id": session_id,
        }
        if oe_url is not None:
            tool_request["oe_url"] = oe_url
        if extra:
            tool_request.update(extra)
        envelope: Dict[str, Any] = {"request": tool_request}
        if include_step:
            envelope["step"] = step
        request_json = json.dumps(envelope)
    if write_request:
        (meta_dir / "request").write_text(request_json)
    return str(meta_dir)


def test_function_is_a_valid_runtime_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RuntimeMode.TOOL_FUNCTION.value == "tool_function"
    monkeypatch.setenv("RUNNER_MODE", "tool_function")
    assert get_runtime_mode() is RuntimeMode.TOOL_FUNCTION


@pytest.fixture(autouse=True)
def _reset_hooks():
    hooks.reset_hooks()
    yield
    hooks.reset_hooks()


@pytest.mark.asyncio
async def test_run_populates_named_llm_registry_from_entrypoint(tmp_path: Path) -> None:
    """Function preparation populates named LLMs before executing the tool."""

    seen_llm: Dict[str, Any] = {}

    def use_named_llm(**_kwargs: Any) -> str:
        seen_llm["llm"] = hooks.get_named_llm("primary")
        return "ok"

    def fake_entrypoint():
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", "the-llm")
        return Mock()

    mock_app = Mock()
    mock_app.get_agent = Mock(side_effect=fake_entrypoint)

    meta_dir = _write_contract(tmp_path, tool_name="use_named_llm")
    client = _CaptureClient()
    runtime = _make_runtime({"use_named_llm": use_named_llm})
    runtime._graph_builder = mock_app

    await _run(runtime, client=client, metadata_dir=meta_dir)

    assert seen_llm["llm"] == "the-llm"
    _, body = client.calls[0]
    assert body["status"] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("construction_fails", [False, True])
async def test_reports_success_result_to_tool_result(
    tmp_path: Path, construction_fails: bool
) -> None:
    def greet(name: str) -> str:
        return f"hello {name}"

    meta_dir = _write_contract(tmp_path)
    client = _CaptureClient()
    runtime = _make_runtime({"greet": greet})
    builder = (
        Mock(side_effect=RuntimeError("constructor unavailable")) if construction_fails else Mock()
    )
    runtime._graph_builder = Mock(get_agent=builder)
    await _run(runtime, client=client, metadata_dir=meta_dir)
    builder.assert_called_once()

    assert len(client.calls) == 1
    url, body = client.calls[0]
    assert url == "http://oe/tool/result"
    assert body["execution_id"] == "exec-1"
    assert body["step_number"] == 3
    assert body["tool_name"] == "greet"
    assert body["status"] == "success"
    assert body["result"] == "hello ada"
    assert body["error"] is None
    assert isinstance(body["duration_ms"], (int, float))


@pytest.mark.asyncio
async def test_reports_no_trace_context_without_active_span(tmp_path: Path) -> None:
    """No active OTel span: trace_id/span_id stay None (opt-in no-op)."""

    def greet(name: str) -> str:
        return f"hello {name}"

    meta_dir = _write_contract(tmp_path)
    client = _CaptureClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["trace_id"] is None
    assert body["span_id"] is None


@pytest.mark.asyncio
async def test_reports_trace_context_from_active_span(tmp_path: Path) -> None:
    """trace_id/span_id on the /tool/result body mirror the active OTel span."""
    from opentelemetry.sdk.trace import TracerProvider

    def greet(name: str) -> str:
        return f"hello {name}"

    meta_dir = _write_contract(tmp_path)
    client = _CaptureClient()
    tracer = TracerProvider().get_tracer("test")

    with tracer.start_as_current_span("tool-pod-result") as span:
        ctx = span.get_span_context()
        await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["trace_id"] == format(ctx.trace_id, "032x")
    assert body["span_id"] == format(ctx.span_id, "016x")


@pytest.mark.asyncio
async def test_reports_no_trace_context_when_tracing_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``tracing`` extra must not break function mode (walter's review,
    PR #3001): the lazy import in ``_current_trace_context()`` should fail closed
    to (None, None) instead of raising ImportError out of ``_result_request()``."""
    import sys

    monkeypatch.setitem(sys.modules, "agent_engine_runner_shared.tracing.setup", None)

    def greet(name: str) -> str:
        return f"hello {name}"

    meta_dir = _write_contract(tmp_path)
    client = _CaptureClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["status"] == "success"
    assert body["trace_id"] is None
    assert body["span_id"] is None


@pytest.mark.asyncio
async def test_supports_async_tools(tmp_path: Path) -> None:
    async def greet(name: str) -> str:
        return f"hi {name}"

    meta_dir = _write_contract(tmp_path)
    client = _CaptureClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["status"] == "success"
    assert body["result"] == "hi ada"


@pytest.mark.asyncio
async def test_full_request_fields_reach_tool_context(tmp_path: Path) -> None:
    """The whole ToolPodExecuteRequest is delivered, so context is populated."""
    captured: Dict[str, Any] = {}

    def greet(name: str) -> str:
        from agent_engine_runner_shared.context import get_current_payload, get_current_user_id

        captured["user_id"] = get_current_user_id()
        captured["payload"] = get_current_payload()
        return "ok"

    meta_dir = _write_contract(
        tmp_path,
        extra={"user_id": "user-7", "payload": {"k": "v"}, "custom_headers": {"x-trace": "abc"}},
    )
    client = _CaptureClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    assert captured["user_id"] == "user-7"
    assert captured["payload"] == {"k": "v"}


@pytest.mark.asyncio
async def test_tool_exception_reported_as_error(tmp_path: Path) -> None:
    def boom(name: str) -> str:
        raise RuntimeError("kaboom")

    meta_dir = _write_contract(tmp_path, tool_name="boom")
    client = _CaptureClient()
    await _run(_make_runtime({"boom": boom}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["status"] == "error"
    assert "kaboom" in body["error"]


@pytest.mark.asyncio
async def test_unknown_tool_reported_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_engine_runner_shared.mcp_tools as mcp_tools

    monkeypatch.setattr(mcp_tools, "is_configured_mcp_sdk_tool_name", lambda *a, **k: False)
    meta_dir = _write_contract(tmp_path, tool_name="missing")
    client = _CaptureClient()
    await _run(_make_runtime({"greet": Mock()}), client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["status"] == "error"
    assert "missing" in body["error"]


@pytest.mark.asyncio
async def test_unserializable_result_downgraded_to_error(tmp_path: Path) -> None:
    """A tool result the JSON encoder can't handle is reported as an error, not dropped."""

    def weird(name: str) -> object:
        return object()

    meta_dir = _write_contract(tmp_path, tool_name="weird")
    client = _CaptureClient()
    await _run(_make_runtime({"weird": weird}), client=client, metadata_dir=meta_dir)

    assert len(client.calls) == 1
    _, body = client.calls[0]
    assert body["status"] == "error"
    assert body["result"] is None
    assert "serializable" in body["error"]


def test_result_payload_downgrade_clears_unserializable_metadata() -> None:
    """If the encode failure is in metadata (not result), the downgrade still encodes."""
    from agent_engine_runner_shared.models import ToolResultRequest
    from agent_engine_runner_shared.server.function import _result_payload

    result = ToolResultRequest(
        execution_id="exec-1",
        step_number=1,
        tool_name="t",
        status="success",
        result="ok",
        duration_ms=1.0,
        metadata={"bad": object()},
    )
    payload = _result_payload(result, "exec-1")

    assert payload["status"] == "error"
    assert payload["result"] is None
    assert payload["metadata"] == {}
    assert "serializable" in payload["error"]
    json.dumps(payload)  # the downgraded report must itself be serializable


def test_nan_and_inf_results_serialize_to_strict_json() -> None:
    """Non-finite floats in a result must still produce strict JSON the OE accepts.

    Python's default ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens that the
    OE's Go ``encoding/json`` parser rejects. ``_result_payload`` runs
    ``model_dump(mode="json")`` first, and Pydantic renders non-finite floats as
    ``null`` at every depth, so the reported payload stays strict-JSON clean and the
    terminal report is not dropped. Locks that invariant against a serializer-default
    change.
    """
    from agent_engine_runner_shared.models import ToolResultRequest
    from agent_engine_runner_shared.server.function import _result_payload

    result = ToolResultRequest(
        execution_id="exec-1",
        step_number=1,
        tool_name="t",
        status="success",
        result={"score": float("nan"), "bounds": [float("inf"), float("-inf")]},
        duration_ms=1.0,
        metadata={"telemetry": {"ratio": float("nan")}},
    )
    payload = _result_payload(result, "exec-1")

    # Reported as success: non-finite floats are valid Python, just not valid JSON,
    # so they are rendered as null rather than failing to encode (no downgrade).
    assert payload["status"] == "success"
    assert payload["result"] == {"score": None, "bounds": [None, None]}
    assert payload["metadata"] == {"telemetry": {"ratio": None}}
    json.dumps(payload, allow_nan=False)  # strict JSON, as Go's parser requires


@pytest.mark.asyncio
async def test_missing_request_raises(tmp_path: Path) -> None:
    meta_dir = _write_contract(tmp_path, write_request=False)
    with pytest.raises(ValueError, match="request"):
        await _run(_make_runtime({}), client=_CaptureClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_missing_step_raises(tmp_path: Path) -> None:
    # Envelope without a step is an invalid ToolFunctionRequest.
    meta_dir = _write_contract(tmp_path, include_step=False)
    with pytest.raises(ValueError, match="ToolFunctionRequest"):
        await _run(_make_runtime({"greet": Mock()}), client=_CaptureClient(), metadata_dir=meta_dir)


def test_invalid_request_error_hides_rejected_arguments(tmp_path: Path, monkeypatch) -> None:
    import traceback

    import agent_engine_runner_shared.models as models
    from agent_engine_runner_shared.server.function import _read_function_request

    monkeypatch.setattr(models, "MAX_TOOL_ARGUMENT_BYTES", 20)
    secret = "distinctive-sensitive-argument"
    meta_dir = _write_contract(
        tmp_path,
        tool_name="write",
        arguments={"content": secret},
    )

    with pytest.raises(ValueError, match="tool arguments exceed 20 bytes") as caught:
        _read_function_request(meta_dir)

    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert secret not in str(caught.value)
    assert secret not in rendered


@pytest.mark.asyncio
async def test_function_mode_emits_custom_event_via_installed_transport(
    tmp_path: Path,
) -> None:
    """Function-mode installs the Tool Pod transport for emit_custom_event."""
    from unittest.mock import MagicMock, patch

    from agent_engine_runner_shared.custom_events import get_custom_event_transport
    from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT

    posts: list[dict[str, Any]] = []

    async def greet_async(name: str) -> str:
        from agent_engine_runner_shared.custom_events import emit_custom_event

        await emit_custom_event({"event": "step", "data": f"hi {name}"})
        assert get_custom_event_transport() is not None
        return f"hello {name}"

    meta_dir = _write_contract(tmp_path, tool_name="greet_async", step=7)
    client = _CaptureClient()
    runtime = _make_runtime({"greet_async": greet_async})
    runtime.agent_config.feature_enabled = Mock(
        side_effect=lambda name, default=False: name == "use_custom_parser"
    )

    with patch("agent_engine_runner_shared.custom_events._get_client") as mock_get_client:
        mock_client = MagicMock()

        def _post(url: str, json: dict[str, Any]) -> MagicMock:
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        mock_client.post.side_effect = _post
        mock_client.stream.side_effect = lambda method, url, *, json, follow_redirects: (
            _SyncPostStream(mock_client.post, url, json)
        )
        mock_get_client.return_value = mock_client
        await _run(
            runtime,
            client=client,
            metadata_dir=meta_dir,
        )

    assert len(posts) == 1
    assert posts[0]["chunk_type"] == CUSTOM_EVENT
    assert posts[0]["custom_event"] == {"event": "step", "data": "hi ada"}
    assert "metadata" not in posts[0]
    assert get_custom_event_transport() is None


@pytest.mark.asyncio
async def test_non_integer_step_raises(tmp_path: Path) -> None:
    meta_dir = _write_contract(tmp_path, step="notanint")
    with pytest.raises(ValueError, match="ToolFunctionRequest"):
        await _run(_make_runtime({"greet": Mock()}), client=_CaptureClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_missing_oe_url_raises(tmp_path: Path) -> None:
    meta_dir = _write_contract(tmp_path, oe_url=None)
    with pytest.raises(ValueError, match="oe_url"):
        await _run(_make_runtime({"greet": Mock()}), client=_CaptureClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_invalid_request_json_raises(tmp_path: Path) -> None:
    meta_dir = _write_contract(tmp_path, request_json="not json")
    with pytest.raises(ValueError, match="ToolFunctionRequest"):
        await _run(_make_runtime({"greet": Mock()}), client=_CaptureClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_inner_request_missing_required_field_raises(tmp_path: Path) -> None:
    # Inner request without session_id (required, min_length=1) -> schema violation.
    bad = json.dumps(
        {
            "request": {
                "execution_id": "e",
                "tool_name": "greet",
                "arguments": {},
                "oe_url": "http://oe",
            },
            "step": 1,
        }
    )
    meta_dir = _write_contract(tmp_path, request_json=bad)
    with pytest.raises(ValueError, match="ToolFunctionRequest"):
        await _run(_make_runtime({"greet": Mock()}), client=_CaptureClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_startup_failure_includes_platform_trace_id(tmp_path: Path) -> None:
    from agent_engine_runner_shared.context import get_current_trace_id

    trace_id = "0123456789abcdef0123456789abcdef"
    seen: List[Optional[str]] = []
    meta_dir = _write_contract(tmp_path, extra={"platform_trace_id": trace_id})
    runner = ToolFunctionRunner(
        _make_runtime({"greet": Mock()}),
        client=_CaptureClient(),
        metadata_dir=meta_dir,
    )

    async def prepare() -> None:
        seen.append(get_current_trace_id())
        raise RuntimeError("registry failed")

    runner.prepare = prepare  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="registry failed"):
        await runner.run()

    assert seen == [trace_id]
    assert get_current_trace_id() is None


@pytest.mark.asyncio
async def test_result_delivery_failure_raises(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(tmp_path)
    with pytest.raises(httpx.ConnectError):
        await _run(_make_runtime({"greet": greet}), client=_FailingClient(), metadata_dir=meta_dir)


@pytest.mark.asyncio
async def test_result_delivery_failure_includes_platform_trace_id(tmp_path: Path) -> None:
    from agent_engine_runner_shared.context import get_current_trace_id

    trace_id = "0123456789abcdef0123456789abcdef"
    seen: List[Optional[str]] = []

    class _FailingTraceClient(_FailingClient):
        async def post(self, url: str, json: Dict[str, Any]) -> _FakeResponse:
            seen.append(get_current_trace_id())
            return await super().post(url, json)

    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(tmp_path, extra={"platform_trace_id": trace_id})
    with pytest.raises(httpx.ConnectError):
        await _run(
            _make_runtime({"greet": greet}),
            client=_FailingTraceClient(),
            metadata_dir=meta_dir,
        )

    assert seen
    assert all(item == trace_id for item in seen)
    assert get_current_trace_id() is None


@pytest.mark.asyncio
async def test_cancellation_closes_connector_runtime_before_result_delivery(
    tmp_path: Path,
) -> None:
    meta_dir = _write_contract(tmp_path)
    runner = ToolFunctionRunner(
        _make_runtime({"greet": Mock()}),
        client=_CaptureClient(),
        metadata_dir=meta_dir,
    )
    connector_runtime = Mock()

    async def prepare() -> None:
        runner._connector_runtime = connector_runtime

    async def cancel_invocation(*_args: Any, **_kwargs: Any) -> None:
        raise asyncio.CancelledError

    runner.prepare = prepare
    runner.invoke_tool = cancel_invocation

    with pytest.raises(asyncio.CancelledError):
        await runner.run()

    connector_runtime.close.assert_called_once_with()


# ---------------------------------------------------------------------------
# Owner-callback fallback: /tool/result runs one best-effort owner pre-attempt
# and falls back to the service URL on ANY owner failure (a transport error OR
# any non-2xx response). The owner is never retried and never fails the report;
# the service URL keeps its full retry budget.
# ---------------------------------------------------------------------------

_SERVICE = "https://oe.ns.svc.cluster.local:8443"
_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"


class _OwnerScriptedClient:
    """Records POSTs. The first (owner) call is scripted; service calls succeed.

    ``owner_action`` is either an int status code returned to the owner, or an
    exception raised on the owner attempt. ``None`` means the owner succeeds.
    """

    def __init__(self, owner_action: Any = None) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._owner_action = owner_action

    async def post(self, url: str, json: Dict[str, Any]) -> _FakeResponse:
        first = not self.calls
        self.calls.append((url, json))
        if first and self._owner_action is not None:
            if isinstance(self._owner_action, BaseException):
                raise self._owner_action
            return _FakeResponse(self._owner_action)
        return _FakeResponse(200)

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Dict[str, Any],
        follow_redirects: bool,
    ) -> Any:
        assert method == "POST"
        assert follow_redirects is False

        client = self

        class _StreamCtx:
            async def __aenter__(self_inner) -> _FakeResponse:
                return await client.post(url, json)

            async def __aexit__(self_inner, exc_type, exc, tb) -> bool:
                return False

        return _StreamCtx()

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_tool_result_prefers_owner_url(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(tmp_path, oe_url=_SERVICE, extra={"oe_owner_url": _OWNER})
    client = _OwnerScriptedClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    assert [url for url, _ in client.calls] == [f"{_OWNER}/tool/result"]


@pytest.mark.asyncio
async def test_tool_result_falls_back_to_service_on_owner_transport_error(tmp_path: Path) -> None:
    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(tmp_path, oe_url=_SERVICE, extra={"oe_owner_url": _OWNER})
    client = _OwnerScriptedClient(owner_action=httpx.ConnectError("owner replica refused"))
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    urls = [url for url, _ in client.calls]
    assert urls == [f"{_OWNER}/tool/result", f"{_SERVICE}/tool/result"]
    # The fallback carries the identical body the owner attempt sent.
    assert client.calls[0][1] == client.calls[1][1]


@pytest.mark.asyncio
async def test_emit_owner_failure_suppresses_final_result_owner_attempt(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from agent_engine_runner_shared.progress import emit_step

    def greet(name: str) -> str:
        emit_step("working")
        return "ok"

    meta_dir = _write_contract(tmp_path, oe_url=_SERVICE, extra={"oe_owner_url": _OWNER})
    result_client = _OwnerScriptedClient()
    progress_client = MagicMock()

    def progress_post(url, json):
        if url == f"{_OWNER}/stream/chunk":
            raise httpx.ConnectError("owner refused")
        return MagicMock()

    progress_client.post.side_effect = progress_post
    progress_client.stream.side_effect = lambda method, url, *, json, follow_redirects: (
        _SyncPostStream(progress_client.post, url, json)
    )

    with patch("agent_engine_runner_shared.progress._get_client", return_value=progress_client):
        await _run(_make_runtime({"greet": greet}), client=result_client, metadata_dir=meta_dir)

    assert [url for url, _ in result_client.calls] == [f"{_SERVICE}/tool/result"]


@pytest.mark.asyncio
async def test_tool_result_falls_back_to_service_on_owner_http_error(tmp_path: Path) -> None:
    """ANY non-2xx owner response (4xx or 5xx) falls back to the service URL and
    the owner is never retried."""

    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(tmp_path, oe_url=_SERVICE, extra={"oe_owner_url": _OWNER})
    client = _OwnerScriptedClient(owner_action=500)
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    urls = [url for url, _ in client.calls]
    assert urls == [f"{_OWNER}/tool/result", f"{_SERVICE}/tool/result"]
    assert urls.count(f"{_OWNER}/tool/result") == 1


@pytest.mark.asyncio
async def test_tool_result_forged_owner_url_uses_service_only(tmp_path: Path) -> None:
    """Security-negative: a forged owner URL is rejected; only the service URL is used."""

    def greet(name: str) -> str:
        return "ok"

    meta_dir = _write_contract(
        tmp_path,
        oe_url=_SERVICE,
        extra={"oe_owner_url": "https://attacker.example:8443"},
    )
    client = _OwnerScriptedClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    assert [url for url, _ in client.calls] == [f"{_SERVICE}/tool/result"]


@pytest.mark.asyncio
async def test_forged_oe_url_and_owner_pair_rejected_when_env_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Security-negative: with OE_URL stamped, a self-consistent forged
    (oe_url, oe_owner_url) pair cannot redirect the callback. The trust anchor
    is the env-resolved OE, so the forged owner is rejected AND the result is
    reported to the configured service, never the forged oe_url."""

    def greet(name: str) -> str:
        return "ok"

    monkeypatch.setenv("OE_URL", _SERVICE)
    forged_service = "https://oe.attacker.svc.cluster.local:8443"
    forged_owner = "https://10-9-9-9.oe-headless.attacker.svc.cluster.local:8443"
    meta_dir = _write_contract(
        tmp_path,
        oe_url=forged_service,
        extra={"oe_owner_url": forged_owner},
    )
    client = _OwnerScriptedClient()
    await _run(_make_runtime({"greet": greet}), client=client, metadata_dir=meta_dir)

    # Only the configured service is contacted; the forged owner and forged
    # service host are never touched.
    assert [url for url, _ in client.calls] == [f"{_SERVICE}/tool/result"]


@pytest.mark.asyncio
async def test_tool_api_error_propagated_in_result_request(tmp_path: Path) -> None:
    """A classified external API failure reaches OE via tool_api_error on the
    /tool/result POST body, not just as an error string."""

    def call_api(name: str) -> str:
        resp = httpx.Response(503, request=httpx.Request("GET", "https://api.example.com/x"))
        raise httpx.HTTPStatusError("unavailable", request=resp.request, response=resp)

    meta_dir = _write_contract(tmp_path, tool_name="call_api")
    client = _CaptureClient()
    runtime = _make_runtime({"call_api": call_api})
    runtime._tool_definitions = {"call_api": {"provider_type": "atlas"}}

    await _run(runtime, client=client, metadata_dir=meta_dir)

    _, body = client.calls[0]
    assert body["status"] == "error"
    assert body["tool_api_error"] is not None
    assert body["tool_api_error"]["provider_type"] == "atlas"
    assert body["tool_api_error"]["classification"] == "PROVIDER_UNAVAILABLE"
    assert body["tool_api_error"]["http_status"] == 503
    assert body["tool_api_error"]["retryable"] is True
