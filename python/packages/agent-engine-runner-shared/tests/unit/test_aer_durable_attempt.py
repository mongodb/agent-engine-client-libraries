"""AER execute lifecycle wiring for durable workflow attempts."""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import threading
from contextlib import suppress
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from agent_engine_runner_shared.context import get_current_execution_id, get_current_wrapper
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ActivityContext,
    ActivityOutcome,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import ExecuteRequest, InterruptResult, StreamingResult
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay
from agent_engine_runner_shared.workflow.protojson import json_to_proto_value

pytestmark = pytest.mark.asyncio


def _make_server() -> Any:
    from agent_engine_runner_shared.server.aer import AERServer

    runtime = Mock()
    runtime.app_name = "insurance"
    runtime.app_version = "1"
    runtime.memory_enabled = False
    server = AERServer(runtime)
    server._send_stream_chunk = AsyncMock()
    server._send_callback_body = AsyncMock()
    return server


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        replay_mode=False,
        workflow_identity=WorkflowIdentity(
            tenant_scope=TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="execution-1",
        ),
        heartbeat_interval_ms=1000,
    )


class _FakeHeartbeat:
    instances: list["_FakeHeartbeat"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        _FakeHeartbeat.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


async def test_native_request_skips_attempt_start() -> None:
    server = _make_server()

    result = await server._start_durable_attempt(
        ExecuteRequest(
            execution_id="execution-1",
            message="hi",
            platform_api_url="http://oe:8000",
        ),
        "session-1",
    )

    assert result is None


async def test_manifest_feature_does_not_override_oe_session_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module
    from agent_engine_runner_shared.hooks import clear_workflow_adapter, register_workflow_adapter

    server = _make_server()
    server.runtime.agent_config.feature_enabled.return_value = False
    workflow_client = AsyncMock()
    workflow_client.__aenter__.return_value = workflow_client
    workflow_client.__aexit__.return_value = None
    workflow_client.start_attempt.return_value = None
    monkeypatch.setattr(
        aer_module,
        "AsyncWorkflowClient",
        Mock(return_value=workflow_client),
    )
    register_workflow_adapter("langgraph", "1")
    try:
        result = await server._start_durable_attempt(
            ExecuteRequest(
                execution_id="execution-1",
                session_id="session-1",
                workspace_id="workspace-1",
                org_id="org-1",
                project_id="project-1",
                user_id="user-1",
                message="hi",
                platform_api_url="http://oe:8000",
            ),
            "session-1",
        )
    finally:
        clear_workflow_adapter()

    assert result is None
    workflow_client.start_attempt.assert_awaited_once()


async def test_native_execute_cleans_up_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _close_wrapper(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")

    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)

    server = _make_server()
    server.runtime._memory_writer = None
    server._start_durable_attempt = AsyncMock(return_value=None)
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )

    async def _capture_callback(_oe_url: str, callback: dict[str, Any], **_kwargs: Any) -> None:
        assert get_current_execution_id() is None
        assert current_attempt_context() is None
        assert callback["suspend_generation"] == 7
        settlement_order.append("callback")

    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    response = await server._handle_execute(
        ExecuteRequest(
            execution_id="execution-1",
            message="hi",
            platform_api_url="http://oe:8000",
            suspend_generation=7,
        )
    )

    assert response.status == "completed"
    assert settlement_order == ["wrapper-close", "callback"]


async def test_durable_execute_binds_context_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    server.runtime.memory_enabled = True
    attempt = _attempt()
    server._start_durable_attempt = AsyncMock(return_value=attempt)

    seen: dict[str, Any] = {}

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        seen["callback_attempt"] = current_attempt_context()

    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    async def _capture(
        _agent: Any, _ctx: Any, _agent_input: Any, _oe_url: Any, _execution_id: Any
    ) -> Any:
        seen["attempt"] = current_attempt_context()
        seen["durable_memory"] = get_current_wrapper().durable_memory
        raise RuntimeError("stop after capture")

    server._execute_via_agent_stream = _capture

    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
                user_id="user-1",
            )
        )

    captured = seen["attempt"]
    assert captured is not None and captured.attempt_id == "attempt-1"
    assert seen["durable_memory"] is not None
    # The attempt context does not leak past the request.
    assert current_attempt_context() is None

    (heartbeat,) = _FakeHeartbeat.instances
    assert heartbeat.started is True
    assert heartbeat.stopped is True
    assert heartbeat.kwargs["interval_seconds"] == 1.0
    assert seen["callback_attempt"] is None


async def test_durable_wrapper_cleanup_failure_still_reports_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _close_wrapper(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")
        raise RuntimeError("wrapper-close")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )

    async def _capture_callback(_oe_url: str, callback: dict[str, Any], **_kwargs: Any) -> None:
        assert get_current_execution_id() is None
        assert current_attempt_context() is None
        assert callback["suspend_generation"] == 7
        settlement_order.append("callback")

    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with pytest.raises(RuntimeError, match="wrapper-close"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
                suspend_generation=7,
            )
        )

    assert settlement_order == ["wrapper-close", "callback"]
    server._send_callback_body.assert_awaited_once()


async def test_durable_heartbeat_stop_failure_still_reports_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    def _stop_heartbeat(_heartbeat: Any) -> None:
        settlement_order.append("heartbeat-stop")
        raise RuntimeError("heartbeat-stop")

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        assert get_current_execution_id() is None
        assert current_attempt_context() is None
        settlement_order.append("callback")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr(_FakeHeartbeat, "stop", _stop_heartbeat)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )
    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with pytest.raises(RuntimeError, match="heartbeat-stop"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert settlement_order == ["heartbeat-stop", "callback"]
    server._send_callback_body.assert_awaited_once()


async def test_native_cancellation_during_cleanup_still_reports_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _cancel_during_wrapper_close(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")
        raise asyncio.CancelledError

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        assert get_current_execution_id() is None
        assert current_attempt_context() is None
        settlement_order.append("callback")

    monkeypatch.setattr(
        aer_module.SecureToolWrapper,
        "close",
        _cancel_during_wrapper_close,
    )

    server = _make_server()
    server.runtime._memory_writer = None
    server._start_durable_attempt = AsyncMock(return_value=None)
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )
    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with pytest.raises(asyncio.CancelledError):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert settlement_order == ["wrapper-close", "callback"]
    server._send_callback_body.assert_awaited_once()


async def test_durable_callback_waits_for_cancelled_heartbeat_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    stop_started = threading.Event()
    release_stop = threading.Event()
    callback_started = asyncio.Event()

    class BlockingHeartbeat(_FakeHeartbeat):
        def stop(self) -> None:
            stop_started.set()
            release_stop.wait(timeout=5)
            self.stopped = True

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", BlockingHeartbeat)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        callback_started.set()

    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    task = asyncio.create_task(
        server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )
    )
    try:
        assert await asyncio.to_thread(stop_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.05)
        assert callback_started.is_set() is False
        release_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_stop.set()
        if not task.done():
            with suppress(BaseException):
                await task

    assert _FakeHeartbeat.instances[0].stopped is True
    assert callback_started.is_set() is True


async def test_suspended_execution_propagates_cleanup_failure_after_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _close_wrapper(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")
        raise RuntimeError("wrapper-close")

    async def _capture_callback(_oe_url: str, callback: dict[str, Any], **_kwargs: Any) -> None:
        assert callback["status"] == "SUSPENDED"
        assert callback["suspend_generation"] == 7
        assert callback["suspend_reason"] == "awaiting_human_review"
        assert get_current_execution_id() is None
        assert current_attempt_context() is None
        settlement_order.append("callback")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(
        return_value=InterruptResult(
            suspend_payload={
                "suspend_reason": "awaiting_human_review",
                "suspend_context": {},
            },
            messages=[],
        )
    )
    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with pytest.raises(RuntimeError, match="wrapper-close"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
                suspend_generation=7,
            )
        )

    assert settlement_order == ["wrapper-close", "callback"]


async def test_callback_delivery_uses_the_staged_metadata_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    outcome = StreamingResult(
        content="done",
        messages=[],
        metadata={"nested": {"value": "before-cleanup"}},
    )

    async def _close_wrapper(_wrapper: Any) -> None:
        assert outcome.metadata is not None
        outcome.metadata["nested"]["value"] = "after-cleanup"

    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)

    delivered: list[dict[str, Any]] = []
    server = _make_server()
    server.runtime._memory_writer = None
    server._start_durable_attempt = AsyncMock(return_value=None)
    server._execute_via_agent_stream = AsyncMock(return_value=outcome)

    async def _capture_callback_body(
        _oe_url: str,
        callback_body: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        delivered.append(callback_body)

    server._send_callback_body = AsyncMock(side_effect=_capture_callback_body)

    await server._handle_execute(
        ExecuteRequest(
            execution_id="execution-1",
            message="hi",
            platform_api_url="http://oe:8000",
        )
    )

    assert delivered[0]["metadata"] == {"nested": {"value": "before-cleanup"}}


async def test_secondary_cleanup_failure_is_logged_without_replacing_the_first(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    async def _close_wrapper(_wrapper: Any) -> None:
        raise RuntimeError("wrapper-close")

    original_reset_attempt_context = aer_module.reset_attempt_context

    def _reset_attempt_context_then_fail(token: Any) -> None:
        original_reset_attempt_context(token)
        raise RuntimeError("attempt-reset")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)
    monkeypatch.setattr(
        aer_module,
        "reset_attempt_context",
        _reset_attempt_context_then_fail,
    )

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )

    with caplog.at_level(logging.ERROR, logger=aer_module.__name__):
        with pytest.raises(RuntimeError, match="wrapper-close"):
            await server._handle_execute(
                ExecuteRequest(
                    execution_id="execution-1",
                    message="hi",
                    platform_api_url="http://oe:8000",
                )
            )

    assert any(
        "attempt context reset" in record.message
        and "execution-1" in record.message
        and "attempt-reset" in record.message
        for record in caplog.records
    )


async def test_execution_failure_is_preserved_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _close_wrapper(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")
        raise RuntimeError("wrapper-close")

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        settlement_order.append("callback")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)
    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(side_effect=RuntimeError("execution-failed"))
    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with caplog.at_level(logging.ERROR, logger=aer_module.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(
                ExecuteRequest(
                    execution_id="execution-1",
                    message="hi",
                    platform_api_url="http://oe:8000",
                )
            )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "execution-failed"
    assert settlement_order == ["wrapper-close", "callback"]
    assert any(
        "wrapper close" in record.message
        and "execution-1" in record.message
        and "wrapper-close" in record.message
        for record in caplog.records
    )


async def test_tracing_import_fallback_does_not_hide_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    settlement_order: list[str] = []

    async def _close_wrapper(_wrapper: Any) -> None:
        settlement_order.append("wrapper-close")
        raise RuntimeError("wrapper-close")

    async def _capture_callback(*_args: Any, **_kwargs: Any) -> None:
        settlement_order.append("callback")

    original_import = builtins.__import__

    def _fail_tracing_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "opentelemetry.trace.propagation.tracecontext":
            raise ImportError("tracing unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(aer_module.SecureToolWrapper, "close", _close_wrapper)
    monkeypatch.setattr(builtins, "__import__", _fail_tracing_import)

    server = _make_server()
    server.runtime._memory_writer = None
    server._start_durable_attempt = AsyncMock(return_value=None)
    server._execute_via_agent_stream = AsyncMock(
        return_value=StreamingResult(content="done", messages=[])
    )
    server._send_callback_body = AsyncMock(side_effect=_capture_callback)

    with pytest.raises(RuntimeError, match="wrapper-close"):
        await server._execute_with_inbound_trace_context(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            ),
            Headers(),
        )

    assert settlement_order == ["wrapper-close", "callback"]


async def test_durable_replay_restarts_at_the_original_activity_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable replay returns recorded work at the original activity position."""
    from fastapi import HTTPException

    import agent_engine_runner_shared.server.aer as aer_module
    from agent_engine_runner_shared.context import get_current_wrapper

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    attempt = _attempt()
    attempt.replay_mode = True
    server._start_durable_attempt = AsyncMock(return_value=attempt)

    seen: dict[str, Any] = {}

    async def _capture(
        _agent: Any, _ctx: Any, _agent_input: Any, _oe_url: Any, _execution_id: Any
    ) -> Any:
        wrapper = get_current_wrapper()
        seen["durable_memory"] = wrapper.durable_memory
        workflow = Mock()
        workflow.start_activity.return_value = ActivityReplay(
            outcome=ActivityOutcome(
                outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
                result=json_to_proto_value("recorded"),
            )
        )
        wrapper._workflow = workflow
        seen["result"] = wrapper.execute_tool("precheck", {"claim_id": "claim-1"})
        seen["command"] = workflow.start_activity.call_args.args[0]
        raise RuntimeError("stop after capture")

    server._execute_via_agent_stream = _capture

    with pytest.raises(HTTPException):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="resume",
                platform_api_url="http://oe:8000",
                resume=True,
                resume_from_step=7,
            )
        )

    assert seen["result"] == "recorded"
    assert seen["command"].position.activity_ordinal == 1
    assert seen["durable_memory"] is None


async def test_durable_memory_identity_fails_before_heartbeat_and_tenant_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    server.runtime.memory_enabled = True
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock()

    with pytest.raises(ValueError, match="durable Memory identity is incomplete"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert _FakeHeartbeat.instances == []
    server._execute_via_agent_stream.assert_not_awaited()


async def test_payload_only_durable_tool_omits_empty_user_memory_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    server.runtime.memory_enabled = True
    attempt = _attempt()
    server._start_durable_attempt = AsyncMock(return_value=attempt)
    seen: dict[str, Any] = {}

    async def _execute(
        _agent: Any, _ctx: Any, _agent_input: Any, _oe_url: Any, _execution_id: Any
    ) -> StreamingResult:
        wrapper = get_current_wrapper()
        seen["durable_memory"] = wrapper.durable_memory
        workflow = Mock()
        workflow.start_activity.return_value = ActivityDispatch(
            context=ActivityContext(
                workflow_identity=attempt.workflow_identity,
                activity_id="activity-1",
                attempt_id=attempt.attempt_id,
                fencing_token=attempt.fencing_token,
            )
        )
        wrapper._workflow = workflow
        wrapper._execute_tool_native = Mock(return_value={"ok": True})

        assert wrapper.execute_tool("lookup", {}, tool_call_id="call-1") == {"ok": True}
        memory_command = workflow.ensure_memory_written.call_args.args[0]
        seen["memory_payloads"] = [
            json.loads(write.payload_json) for write in memory_command.memory_writes
        ]
        return StreamingResult(content="done", messages=[])

    server._execute_via_agent_stream = _execute
    response = await server._handle_execute(
        ExecuteRequest(
            execution_id="execution-1",
            platform_api_url="http://oe:8000",
            user_id="user-1",
            payload={"query": "structured input"},
        )
    )

    assert response.status == "completed"
    assert seen["durable_memory"] is not None
    assert [payload["role"] for payload in seen["memory_payloads"]] == ["tool"]


async def test_durable_resume_before_first_completed_activity_retains_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    server.runtime.memory_enabled = True
    attempt = _attempt()
    attempt.replay_mode = True
    server._start_durable_attempt = AsyncMock(return_value=attempt)
    seen: dict[str, Any] = {}

    async def _execute(
        _agent: Any, _ctx: Any, _agent_input: Any, _oe_url: Any, _execution_id: Any
    ) -> StreamingResult:
        wrapper = get_current_wrapper()
        workflow = Mock()
        workflow.start_activity.return_value = ActivityDispatch(
            context=ActivityContext(
                workflow_identity=attempt.workflow_identity,
                activity_id="activity-1",
                attempt_id=attempt.attempt_id,
                fencing_token=attempt.fencing_token,
            )
        )
        wrapper._workflow = workflow
        wrapper._execute_tool_native = Mock(return_value={"approved": True})

        assert wrapper.execute_tool("approve_claim", {}, tool_call_id="call-1") == {
            "approved": True
        }
        memory_command = workflow.ensure_memory_written.call_args.args[0]
        seen["memory_payloads"] = [
            json.loads(write.payload_json) for write in memory_command.memory_writes
        ]
        return StreamingResult(content="done", messages=[])

    server._execute_via_agent_stream = _execute
    response = await server._handle_execute(
        ExecuteRequest(
            execution_id="execution-1",
            message="file this claim",
            platform_api_url="http://oe:8000",
            user_id="user-1",
            resume=True,
            resume_data={"approved": True},
        )
    )

    assert response.status == "completed"
    assert [(payload["role"], payload["content"]) for payload in seen["memory_payloads"]] == [
        ("user", "file this claim"),
        ("tool", "{'approved': True}"),
    ]


async def test_failed_heartbeat_start_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    class _FailingHeartbeat(_FakeHeartbeat):
        def start(self) -> None:
            raise RuntimeError("initial attempt heartbeat failed")

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FailingHeartbeat)

    server = _make_server()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock()

    with pytest.raises(RuntimeError, match="heartbeat"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    server._execute_via_agent_stream.assert_not_awaited()


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (
            StreamingResult(
                content="done",
                messages=[{"role": "assistant", "content": "done"}],
            ),
            "completed",
        ),
        (
            InterruptResult(
                suspend_payload={"suspend_reason": "approval", "suspend_context": {}},
                messages=[{"role": "assistant", "content": "waiting"}],
            ),
            "suspended",
        ),
    ],
)
async def test_durable_attempt_does_not_use_native_memory_writer(
    monkeypatch: pytest.MonkeyPatch,
    outcome: StreamingResult | InterruptResult,
    expected_status: str,
) -> None:
    import agent_engine_runner_shared.server.aer as aer_module

    _FakeHeartbeat.instances = []
    monkeypatch.setattr(aer_module, "AttemptHeartbeat", _FakeHeartbeat)

    server = _make_server()
    server.runtime._memory_writer = Mock()
    server.runtime.write_turn_async = Mock()
    server._start_durable_attempt = AsyncMock(return_value=_attempt())
    server._execute_via_agent_stream = AsyncMock(return_value=outcome)

    result = await server._handle_execute(
        ExecuteRequest(
            execution_id="execution-1",
            message="hello",
            platform_api_url="http://oe:8000",
            user_id="user-1",
            session_id="session-1",
        )
    )

    assert result.status == expected_status
    server.runtime.write_turn_async.assert_not_called()


async def test_agent_is_built_before_the_durable_routing_decision() -> None:
    """Graph materialization registers the workflow adapter, so it must
    precede the attempt gate — otherwise the first request per process
    would route native before eligibility exists."""
    server = _make_server()
    order: list[str] = []

    def _get_agent(callbacks: Any = None) -> Any:
        order.append("get_agent")
        raise RuntimeError("stop after agent build")

    server.runtime.get_agent = _get_agent

    async def _start(request: Any, session_id: Any) -> Any:
        order.append("start_attempt")
        return None

    server._start_durable_attempt = _start

    with pytest.raises(RuntimeError, match="stop after agent build"):
        await server._handle_execute(
            ExecuteRequest(
                execution_id="execution-1",
                message="hi",
                platform_api_url="http://oe:8000",
            )
        )

    assert order == ["get_agent"]
