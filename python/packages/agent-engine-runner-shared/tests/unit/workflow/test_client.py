"""Contract tests for the workflow ProtoJSON HTTP client."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from opentelemetry import baggage
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import attach, detach
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ActivityCommand,
    ActivityContext,
    ActivityMemoryCommand,
    ActivityOutcome,
    MemoryWrite,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ActivityHeartbeatRequest as ActivityHeartbeatRequestPb,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    WORKFLOW_ERROR_CODE_NOT_FOUND,
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
    WORKFLOW_ERROR_CODE_STALE_FENCE,
    WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    ActivityPosition,
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptHeartbeatRequest,
    AttemptStartRequest,
    WorkflowDeclaration,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    CompleteExecutionCommand,
    FinalizeStepCommand,
    StateSnapshot,
)
from agent_engine_runner_shared.workflow import (
    AsyncWorkflowClient,
    WorkflowClient,
    WorkflowClientError,
)
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay
from agent_engine_runner_shared.workflow.protojson import encode_protojson

_ATTEMPT_START_PATH = "/executor/attempt/start"
_ATTEMPT_HEARTBEAT_PATH = "/executor/attempt/heartbeat"
_STEP_FINALIZE_PATH = "/executor/step/finalize"
_EXECUTION_COMPLETE_PATH = "/executor/complete"


@dataclass
class _RecordedRequest:
    path: str
    content_type: str
    body: dict
    traceparent: str | None
    baggage: str | None


@dataclass
class _ServerScript:
    """One scripted response plus the requests the server actually received."""

    status: int = 200
    body: str = "{}"
    content_type: str = "application/json"
    delay_seconds: float = 0.0
    requests: list[_RecordedRequest] = field(default_factory=list)


@pytest.fixture()
def oe_server() -> Iterator[tuple[str, _ServerScript]]:
    script = _ServerScript()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            script.requests.append(
                _RecordedRequest(
                    path=self.path,
                    content_type=self.headers.get("Content-Type", ""),
                    body=json.loads(raw) if raw else {},
                    traceparent=self.headers.get("traceparent"),
                    baggage=self.headers.get("baggage"),
                )
            )
            if script.delay_seconds:
                time.sleep(script.delay_seconds)
            payload = script.body.encode()
            self.send_response(script.status)
            self.send_header("Content-Type", script.content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", script
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@pytest.fixture()
def instrumented_httpx_tracing() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=provider)
    try:
        yield provider, exporter
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


def _start_request() -> AttemptStartRequest:
    return AttemptStartRequest(
        workflow_identity=WorkflowIdentity(
            tenant_scope=TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="execution-1",
        ),
        owner_id="aer-1",
        declaration=WorkflowDeclaration(
            workflow_name="insurance",
            workflow_version="1",
            adapter_name="langgraph",
            adapter_version="0.1",
        ),
    )


def _heartbeat_request() -> AttemptHeartbeatRequest:
    return AttemptHeartbeatRequest(
        workflow_identity=WorkflowIdentity(
            tenant_scope=TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="execution-1",
        ),
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
    )


def _accepted_attempt_body(request: AttemptStartRequest | None = None) -> str:
    request_body = json.loads(encode_protojson(request or _start_request()))
    return json.dumps(
        {
            "attempt_context": {
                "attempt_id": "attempt-1",
                "fencing_token": "7",
                "owner_id": "aer-1",
                "replay_mode": True,
                "workflow_identity": request_body.get("workflow_identity"),
                "declaration": request_body.get("declaration"),
                "heartbeat_interval_ms": "1000",
            }
        }
    )


def test_start_attempt_posts_snake_case_protojson_to_exact_path(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()

    with WorkflowClient(base_url) as client:
        context = client.start_attempt(_start_request())

    assert context is not None
    assert context.attempt_id == "attempt-1"
    assert context.fencing_token == 7
    assert context.replay_mode is True
    assert context.heartbeat_interval_ms == 1000

    (request,) = script.requests
    assert request.path == _ATTEMPT_START_PATH
    assert request.content_type == "application/json"
    assert request.body == {
        "workflow_identity": {
            "tenant_scope": {
                "org_id": "org-1",
                "project_id": "project-1",
                "workspace_id": "workspace-1",
            },
            "session_id": "session-1",
            "execution_id": "execution-1",
        },
        "owner_id": "aer-1",
        "declaration": {
            "workflow_name": "insurance",
            "workflow_version": "1",
            "adapter_name": "langgraph",
            "adapter_version": "0.1",
        },
    }


def test_workflow_request_propagates_trace_context_without_exporting_http_span(
    oe_server: tuple[str, _ServerScript],
    instrumented_httpx_tracing: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()
    provider, exporter = instrumented_httpx_tracing
    tracer = provider.get_tracer("workflow-client-test")
    previous_propagator = get_global_textmap()
    set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
    token = attach(baggage.set_baggage("tenant.secret", "should-not-leak"))
    try:
        with tracer.start_as_current_span("agent") as agent_span:
            context = agent_span.get_span_context()
            with WorkflowClient(base_url) as client:
                client.start_attempt(_start_request())
    finally:
        detach(token)
        set_global_textmap(previous_propagator)

    assert [span.name for span in exporter.get_finished_spans()] == ["agent"]
    (request,) = script.requests
    assert request.traceparent == (
        f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
    )
    assert request.baggage is None


async def test_async_workflow_request_suppresses_http_span_across_await(
    oe_server: tuple[str, _ServerScript],
    instrumented_httpx_tracing: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()
    provider, exporter = instrumented_httpx_tracing
    tracer = provider.get_tracer("async-workflow-client-test")
    previous_propagator = get_global_textmap()
    set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
    token = attach(baggage.set_baggage("tenant.secret", "should-not-leak"))
    try:
        with tracer.start_as_current_span("agent") as agent_span:
            context = agent_span.get_span_context()
            async with AsyncWorkflowClient(base_url) as client:
                await client.start_attempt(_start_request())
    finally:
        detach(token)
        set_global_textmap(previous_propagator)

    assert [span.name for span in exporter.get_finished_spans()] == ["agent"]
    (request,) = script.requests
    assert request.traceparent == (
        f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
    )
    assert request.baggage is None


def test_start_attempt_omits_default_and_empty_fields(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()

    request = AttemptStartRequest(
        workflow_identity=WorkflowIdentity(session_id="session-1"),
        owner_id="aer-1",
    )
    script.body = _accepted_attempt_body(request)
    with WorkflowClient(base_url) as client:
        client.start_attempt(request)

    (recorded,) = script.requests
    assert recorded.body == {
        "workflow_identity": {"session_id": "session-1"},
        "owner_id": "aer-1",
    }


def test_start_attempt_read_response_tolerates_unknown_fields(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    response = json.loads(_accepted_attempt_body())
    response["future_response_field"] = {"nested": True}
    script.body = json.dumps(response)

    with WorkflowClient(base_url) as client:
        context = client.start_attempt(_start_request())

    assert context is not None
    assert context.attempt_id == "attempt-1"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attempt_id",), ""),
        (("fencing_token",), "0"),
        (("heartbeat_interval_ms",), "0"),
        (("workflow_identity", "session_id"), "other-session"),
        (("declaration", "adapter_name"), "other-adapter"),
        (("owner_id",), "other-owner"),
    ],
    ids=[
        "missing-attempt-id",
        "invalid-fence",
        "invalid-heartbeat",
        "mismatched-identity",
        "mismatched-declaration",
        "mismatched-owner",
    ],
)
def test_start_attempt_rejects_unqualified_attempt_context(
    oe_server: tuple[str, _ServerScript],
    path: tuple[str, ...],
    value: str,
) -> None:
    base_url, script = oe_server
    response = json.loads(_accepted_attempt_body())
    target = response["attempt_context"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    script.body = json.dumps(response)

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT


def test_start_attempt_bare_404_leaves_invocation_native(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 404
    script.body = "404 page not found"
    script.content_type = "text/plain"

    with WorkflowClient(base_url) as client:
        assert client.start_attempt(_start_request()) is None


def test_start_attempt_404_with_bare_workflow_error_body_raises_that_code(
    oe_server: tuple[str, _ServerScript],
) -> None:
    """A bare (unwrapped) coded error parses as an empty AttemptStartResponse
    under tolerant reads; it must still win over the 404-means-old-OE
    fallback, or a workflow-aware OE's rejection silently pins native."""
    base_url, script = oe_server
    script.status = 404
    script.body = json.dumps(
        {"code": "WORKFLOW_ERROR_CODE_NOT_FOUND", "message": "unknown execution"}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_NOT_FOUND


def test_start_attempt_404_with_workflow_error_body_raises_that_code(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 404
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_NOT_FOUND", "message": "unknown execution"}}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_NOT_FOUND


def test_start_attempt_body_error_wins_over_http_status(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 409
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "stale fence"}}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, WORKFLOW_ERROR_CODE_INVALID_ARGUMENT),
        (401, WORKFLOW_ERROR_CODE_UNAUTHORIZED),
        (403, WORKFLOW_ERROR_CODE_UNAUTHORIZED),
        (409, WORKFLOW_ERROR_CODE_CONFLICT),
        (412, WORKFLOW_ERROR_CODE_STALE_FENCE),
        (500, WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN),
        (503, WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN),
    ],
)
def test_start_attempt_maps_http_status_to_stable_codes(
    oe_server: tuple[str, _ServerScript],
    status: int,
    expected_code: int,
) -> None:
    base_url, script = oe_server
    script.status = status
    script.body = "unstructured failure"
    script.content_type = "text/plain"

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == expected_code


def test_start_attempt_success_without_attempt_context_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "{}"

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT


def test_start_attempt_malformed_response_body_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "not json"

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT


def test_start_attempt_timeout_maps_to_outcome_unknown(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()
    script.delay_seconds = 1.0

    with WorkflowClient(base_url, timeout=0.2) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN


def test_start_attempt_connection_failure_maps_to_outcome_unknown() -> None:
    with WorkflowClient("http://127.0.0.1:9", timeout=0.2) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_attempt(_start_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN


def test_heartbeat_posts_to_exact_path_and_returns_none(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server

    with WorkflowClient(base_url) as client:
        assert client.heartbeat(_heartbeat_request()) is None

    (request,) = script.requests
    assert request.path == _ATTEMPT_HEARTBEAT_PATH
    assert request.body == {
        "workflow_identity": {
            "tenant_scope": {
                "org_id": "org-1",
                "project_id": "project-1",
                "workspace_id": "workspace-1",
            },
            "session_id": "session-1",
            "execution_id": "execution-1",
        },
        "attempt_id": "attempt-1",
        "fencing_token": "7",
        "owner_id": "aer-1",
    }


def test_heartbeat_bare_404_maps_to_not_found(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 404
    script.body = "404 page not found"
    script.content_type = "text/plain"

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.heartbeat(_heartbeat_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_NOT_FOUND


def test_heartbeat_stale_fence_maps_to_stable_code(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 412
    script.body = json.dumps(
        {"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "attempt superseded"}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.heartbeat(_heartbeat_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


def test_heartbeat_2xx_with_stale_fence_body_raises(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 200
    script.body = json.dumps({"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "superseded"})

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.heartbeat(_heartbeat_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


def test_report_outcome_2xx_with_error_envelope_raises(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 200
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "superseded"}}
    )
    outcome = ActivityOutcome(
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
        outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.report_outcome(outcome)

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


def test_client_strips_trailing_slash_from_base_url(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()

    with WorkflowClient(base_url + "/") as client:
        client.start_attempt(_start_request())

    (request,) = script.requests
    assert request.path == _ATTEMPT_START_PATH


def _activity_identity() -> WorkflowIdentity:
    return WorkflowIdentity(
        tenant_scope=TenantScope(
            org_id="org-1",
            project_id="project-1",
            workspace_id="workspace-1",
        ),
        session_id="session-1",
        execution_id="execution-1",
    )


def _activity_command() -> ActivityCommand:
    return ActivityCommand(
        workflow_identity=_activity_identity(),
        attempt_id="attempt-1",
        fencing_token=7,
        position=ActivityPosition(activity_ordinal=1),
        activity_kind=ACTIVITY_KIND_TOOL,
        activity_name="lookup",
    )


def _matching_activity_context(**overrides: object) -> dict[str, object]:
    body = json.loads(
        encode_protojson(
            ActivityContext(
                workflow_identity=_activity_identity(),
                activity_id="activity-1",
                attempt_id="attempt-1",
                fencing_token=7,
            )
        )
    )
    body.update(overrides)
    return body


def test_start_activity_dispatch_posts_to_exact_path(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps({"activity_context": _matching_activity_context()})

    with WorkflowClient(base_url) as client:
        started = client.start_activity(_activity_command())

    assert isinstance(started, ActivityDispatch)
    assert started.context.activity_id == "activity-1"
    assert started.context.workflow_identity.execution_id == "execution-1"
    (request,) = script.requests
    assert request.path == "/executor/activity/start"
    assert request.body["activity_name"] == "lookup"
    assert request.body["fencing_token"] == "7"


def test_start_activity_replays_recorded_outcome(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps(
        {
            "outcome": {
                "activity_id": "activity-1",
                "outcome_kind": "ACTIVITY_OUTCOME_KIND_COMPLETED",
                "result": {"cached": True},
            }
        }
    )

    with WorkflowClient(base_url) as client:
        started = client.start_activity(_activity_command())

    assert isinstance(started, ActivityReplay)
    assert started.outcome.activity_id == "activity-1"


def test_start_activity_error_envelope_raises_that_code(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 409
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "superseded"}}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


def test_start_activity_success_without_context_or_outcome_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "{}"

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT


def test_start_activity_empty_context_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps({"activity_context": {}})

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
    assert "missing identity fields" in raised.value.message


def test_start_activity_context_missing_workflow_identity_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps(
        {
            "activity_context": {
                "activity_id": "activity-1",
                "attempt_id": "attempt-1",
                "fencing_token": "7",
            }
        }
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
    assert "missing identity fields" in raised.value.message


def test_start_activity_context_mismatched_attempt_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps(
        {"activity_context": _matching_activity_context(attempt_id="attempt-other")}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
    assert "does not match the submitted command" in raised.value.message


def test_start_activity_context_mismatched_fence_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps({"activity_context": _matching_activity_context(fencing_token="8")})

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
    assert "does not match the submitted command" in raised.value.message


def test_start_activity_context_mismatched_execution_is_invalid(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    mismatched = _activity_identity()
    mismatched.execution_id = "execution-other"
    script.body = json.dumps(
        {
            "activity_context": _matching_activity_context(
                workflow_identity=json.loads(encode_protojson(mismatched))
            )
        }
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.start_activity(_activity_command())

    assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
    assert "does not match the submitted command" in raised.value.message


def test_report_outcome_posts_to_exact_path(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    outcome = ActivityOutcome(
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
        outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
    )

    with WorkflowClient(base_url) as client:
        assert client.report_outcome(outcome) is None

    (request,) = script.requests
    assert request.path == "/executor/activity/outcome"
    assert request.body["outcome_kind"] == "ACTIVITY_OUTCOME_KIND_COMPLETED"


def test_ensure_memory_written_posts_to_exact_path(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    command = ActivityMemoryCommand(
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
        memory_writes=[MemoryWrite(id="write-1", payload_json=b'{"content":"hi"}')],
    )

    with WorkflowClient(base_url) as client:
        assert client.ensure_memory_written(command) is None

    (request,) = script.requests
    assert request.path == "/executor/activity/memory"
    assert request.body["activity_id"] == "activity-1"
    assert request.body["memory_writes"] == [
        {"id": "write-1", "payload_json": "eyJjb250ZW50IjoiaGkifQ=="}
    ]


def test_heartbeat_activity_maps_stale_fence(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 412
    script.body = json.dumps({"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "superseded"})

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.heartbeat_activity(
                ActivityHeartbeatRequestPb(activity_id="activity-1", fencing_token=7)
            )

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE
    (request,) = script.requests
    assert request.path == "/executor/activity/heartbeat"


def test_complete_execution_acknowledges_final_state(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "{}"
    command = CompleteExecutionCommand(
        workflow_identity=WorkflowIdentity(session_id="session-1", execution_id="execution-1"),
        attempt_id="attempt-1",
        fencing_token=7,
        state=StateSnapshot(properties={"quote_count": 2}),
    )

    with WorkflowClient(base_url) as client:
        response = client.complete_execution(command)

    assert response is None
    (request,) = script.requests
    assert request.path == _EXECUTION_COMPLETE_PATH
    assert request.body["fencing_token"] == "7"
    assert request.body["state"]["properties"] == {"quote_count": 2}


def test_finalize_step_acknowledges_settled_step(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "{}"
    command = FinalizeStepCommand(
        workflow_identity=WorkflowIdentity(session_id="session-1", execution_id="execution-1"),
        attempt_id="attempt-1",
        fencing_token=7,
        step_ordinal=1,
        state=StateSnapshot(properties={"quote_count": 2}),
    )

    with WorkflowClient(base_url) as client:
        response = client.finalize_step(command)

    assert response == []
    (request,) = script.requests
    assert request.path == _STEP_FINALIZE_PATH
    assert request.body["fencing_token"] == "7"
    assert request.body["step_ordinal"] == "1"
    assert request.body["state"]["properties"] == {"quote_count": 2}


def test_finalize_step_returns_positioned_activity_entries(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps(
        {
            "entries": [
                {
                    "position": {
                        "step_ordinal": "1",
                        "operation_path": {"segments": [{"name": "review", "ordinal": "1"}]},
                        "activity_ordinal": "2",
                    },
                    "outcome": {
                        "activity_id": "activity-2",
                        "attempt_id": "attempt-1",
                        "fencing_token": "7",
                        "outcome_kind": "ACTIVITY_OUTCOME_KIND_SUSPENDED",
                        "suspension": {"reason": "agent_interrupt"},
                    },
                }
            ]
        }
    )

    with WorkflowClient(base_url) as client:
        entries = client.finalize_step(
            FinalizeStepCommand(
                workflow_identity=WorkflowIdentity(
                    session_id="session-1", execution_id="execution-1"
                ),
                attempt_id="attempt-1",
                fencing_token=7,
                step_ordinal=1,
            )
        )

    assert len(entries) == 1
    assert entries[0].position.activity_ordinal == 2
    assert entries[0].outcome.activity_id == "activity-2"


def test_finalize_step_surfaces_stable_workflow_errors(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 409
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_CONFLICT", "message": "unsettled"}}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.finalize_step(
                FinalizeStepCommand(
                    workflow_identity=WorkflowIdentity(
                        session_id="session-1", execution_id="execution-1"
                    ),
                    step_ordinal=1,
                )
            )

    assert raised.value.code == WORKFLOW_ERROR_CODE_CONFLICT


def test_complete_execution_surfaces_stable_workflow_errors(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 412
    script.body = json.dumps(
        {"error": {"code": "WORKFLOW_ERROR_CODE_STALE_FENCE", "message": "superseded"}}
    )

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.complete_execution(
                CompleteExecutionCommand(
                    workflow_identity=WorkflowIdentity(
                        session_id="session-1", execution_id="execution-1"
                    )
                )
            )

    assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE


async def test_async_activity_endpoints_match_sync_semantics(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = json.dumps({"activity_context": _matching_activity_context()})

    async with AsyncWorkflowClient(base_url) as client:
        started = await client.start_activity(_activity_command())

    assert isinstance(started, ActivityDispatch)
    (request,) = script.requests
    assert request.path == "/executor/activity/start"


async def test_async_memory_synchronization_matches_sync_semantics(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server

    async with AsyncWorkflowClient(base_url) as client:
        await client.ensure_memory_written(ActivityMemoryCommand(activity_id="activity-1"))

    (request,) = script.requests
    assert request.path == "/executor/activity/memory"
    assert request.body["activity_id"] == "activity-1"


async def test_async_execution_completion_matches_sync_semantics(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = "{}"

    async with AsyncWorkflowClient(base_url) as client:
        completed = await client.complete_execution(CompleteExecutionCommand())

    assert completed is None
    assert [request.path for request in script.requests] == [_EXECUTION_COMPLETE_PATH]


async def test_async_start_attempt_matches_sync_semantics(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.body = _accepted_attempt_body()

    async with AsyncWorkflowClient(base_url) as client:
        context = await client.start_attempt(_start_request())

    assert context is not None
    assert context.attempt_id == "attempt-1"
    (request,) = script.requests
    assert request.path == _ATTEMPT_START_PATH
    assert request.body["owner_id"] == "aer-1"


async def test_async_start_attempt_bare_404_leaves_invocation_native(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 404
    script.body = "404 page not found"
    script.content_type = "text/plain"

    async with AsyncWorkflowClient(base_url) as client:
        assert await client.start_attempt(_start_request()) is None


def test_heartbeat_preserves_server_message_when_code_is_unset(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 409
    script.body = json.dumps({"message": "execution already owned"})

    with WorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            client.heartbeat(_heartbeat_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_CONFLICT
    assert raised.value.message == "execution already owned"


async def test_async_concurrent_first_calls_share_one_owned_client(
    oe_server: tuple[str, _ServerScript],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from agent_engine_runner_shared.workflow import client as client_module

    base_url, script = oe_server
    script.body = _accepted_attempt_body()
    created: list[httpx.AsyncClient] = []

    async def _counting_factory(base: str, timeout: float) -> httpx.AsyncClient:
        await asyncio.sleep(0.01)
        instance = httpx.AsyncClient(timeout=timeout)
        created.append(instance)
        return instance

    monkeypatch.setattr(client_module, "create_async_httpx_client_with_tls", _counting_factory)

    async with AsyncWorkflowClient(base_url) as client:
        results = await asyncio.gather(
            client.start_attempt(_start_request()),
            client.start_attempt(_start_request()),
        )

    assert all(context is not None for context in results)
    assert len(created) == 1


async def test_async_close_racing_first_call_still_closes_owned_client(
    oe_server: tuple[str, _ServerScript],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from agent_engine_runner_shared.workflow import client as client_module

    base_url, script = oe_server
    script.body = _accepted_attempt_body()
    created: list[httpx.AsyncClient] = []

    async def _slow_factory(base: str, timeout: float) -> httpx.AsyncClient:
        await asyncio.sleep(0.05)
        instance = httpx.AsyncClient(timeout=timeout)
        created.append(instance)
        return instance

    monkeypatch.setattr(client_module, "create_async_httpx_client_with_tls", _slow_factory)

    client = AsyncWorkflowClient(base_url)
    start_task = asyncio.create_task(client.start_attempt(_start_request()))
    await asyncio.sleep(0.01)
    await client.close()

    # The in-flight request may finish or fail depending on who wins the
    # post/aclose race; the invariant is that the owned client never leaks.
    await asyncio.gather(start_task, return_exceptions=True)
    assert len(created) == 1
    assert created[0].is_closed


async def test_async_client_rejects_requests_after_close(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, _ = oe_server
    client = AsyncWorkflowClient(base_url)
    await client.close()

    with pytest.raises(RuntimeError):
        await client.start_attempt(_start_request())


async def test_async_heartbeat_maps_conflict(
    oe_server: tuple[str, _ServerScript],
) -> None:
    base_url, script = oe_server
    script.status = 409
    script.body = "unstructured failure"
    script.content_type = "text/plain"

    async with AsyncWorkflowClient(base_url) as client:
        with pytest.raises(WorkflowClientError) as raised:
            await client.heartbeat(_heartbeat_request())

    assert raised.value.code == WORKFLOW_ERROR_CODE_CONFLICT
