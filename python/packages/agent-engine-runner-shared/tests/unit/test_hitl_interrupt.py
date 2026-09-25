"""Tests for HITL interrupt migration: interrupt() / Command(resume=...) flow."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest
from agent_engine_sdk import RequestContext
from fastapi import HTTPException
from pydantic import ValidationError

from agent_engine_runner_shared import hooks
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import (
    InterruptResult,
    PendingInterrupt,
    StreamingResult,
    SuspendPayload,
    ToolExecuteResponse,
)
from agent_engine_runner_shared.secure_wrapper import (
    PolicyDeniedException,
    SecureToolWrapper,
    ToolExecutionError,
)
from agent_engine_runner_shared.workflow import attempt_context_scope


class TestSuspendPayload:
    """Tests for the SuspendPayload model."""

    def test_requires_suspend_reason(self):
        with pytest.raises(ValidationError):
            SuspendPayload()

    def test_defaults_context_to_empty_dict(self):
        p = SuspendPayload(suspend_reason="review")
        assert p.suspend_context == {}

    def test_to_json_includes_suspend_flag(self):
        p = SuspendPayload(
            suspend_reason="awaiting_human_review",
            suspend_context={"claim_id": "C-123"},
        )
        data = json.loads(p.to_json())
        assert data["__suspend__"] is True
        assert data["suspend_reason"] == "awaiting_human_review"
        assert data["suspend_context"] == {"claim_id": "C-123"}

    def test_round_trip_through_model_validate(self):
        original = SuspendPayload(
            suspend_reason="review",
            suspend_context={"task_id": "T-1"},
        )
        data = json.loads(original.to_json())
        restored = SuspendPayload.model_validate(data)
        assert restored.suspend_reason == original.suspend_reason
        assert restored.suspend_context == original.suspend_context


class TestHandleSuspend:
    """Tests for SecureToolWrapper._handle_suspend, which fires the framework
    interrupt for an OE-confirmed suspend. Provenance (only the author suspend
    API triggers it, never relayed tool-result content) is enforced at the
    tool-pod layer (test_tool.py) and the AER status gate.
    """

    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        yield
        hooks.reset_hooks()

    def _make_wrapper(self) -> SecureToolWrapper:
        return SecureToolWrapper(oe_url="http://oe:8000", execution_id="test-exec-123")

    def test_suspend_calls_interrupt_with_validated_payload(self):
        wrapper = self._make_wrapper()
        suspend_result = SuspendPayload(
            suspend_reason="awaiting_human_review",
            suspend_context={"task_id": "t1", "claim_id": "c1"},
        ).to_json()

        mock_interrupt = Mock(return_value={"decision": "approved"})
        hooks.register_suspend_handler(mock_interrupt)
        result = wrapper._handle_suspend(suspend_result)

        mock_interrupt.assert_called_once_with(
            {
                "suspend_reason": "awaiting_human_review",
                "suspend_context": {"task_id": "t1", "claim_id": "c1"},
            }
        )
        assert json.loads(result) == {"decision": "approved"}

    def test_suspend_returns_human_decision_as_json(self):
        wrapper = self._make_wrapper()
        suspend_result = SuspendPayload(
            suspend_reason="review",
            suspend_context={},
        ).to_json()

        human_decision = {"decision": "rejected", "notes": "insufficient docs"}
        hooks.register_suspend_handler(Mock(return_value=human_decision))
        result = wrapper._handle_suspend(suspend_result)

        assert json.loads(result) == human_decision

    def test_durable_suspend_uses_activity_handler(self):
        wrapper = self._make_wrapper()
        suspend_result = SuspendPayload(
            suspend_reason="review",
            suspend_context={"claim_id": "claim-1"},
        ).to_json()
        suspension = RuntimeError("durable suspension")
        durable_handler = Mock(side_effect=suspension)
        native_handler = Mock()
        hooks.register_durable_activity_suspend_handler(durable_handler)
        hooks.register_suspend_handler(native_handler)

        with (
            attempt_context_scope(AttemptContext(attempt_id="attempt-1")),
            pytest.raises(RuntimeError, match="durable suspension"),
        ):
            wrapper._handle_suspend(suspend_result)

        durable_handler.assert_called_once_with(
            {
                "suspend_reason": "review",
                "suspend_context": {"claim_id": "claim-1"},
            }
        )
        native_handler.assert_not_called()

    def test_suspend_without_reason_raises_validation_error(self):
        """Missing suspend_reason is caught by SuspendPayload validation."""
        wrapper = self._make_wrapper()
        hooks.register_suspend_handler(Mock())
        bad_payload = json.dumps({"__suspend__": True})

        with pytest.raises(ValidationError):
            wrapper._handle_suspend(bad_payload)

    def test_suspend_without_registered_handler_raises_runtime_error(self):
        """A suspend payload with no registered handler raises RuntimeError."""
        wrapper = self._make_wrapper()
        suspend_result = SuspendPayload(
            suspend_reason="awaiting_human_review",
            suspend_context={"claim_id": "c1"},
        ).to_json()

        with pytest.raises(RuntimeError, match="No suspend handler registered"):
            wrapper._handle_suspend(suspend_result)

    def test_suspend_with_non_payload_result_raises(self):
        """An OE suspend status whose result is not a payload is a contract error."""
        wrapper = self._make_wrapper()
        hooks.register_suspend_handler(Mock())

        with pytest.raises(ToolExecutionError, match="not a suspend payload"):
            wrapper._handle_suspend("plain text")


class TestAuthorizationRequiredInterrupt:
    """Tests for broker authorization-required responses in execute_tool."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        yield
        hooks.reset_hooks()

    def _make_wrapper(self) -> SecureToolWrapper:
        return SecureToolWrapper(oe_url="http://oe:8000", execution_id="test-exec-123")

    def test_model_deserializes_elicitation_payload(self):
        response = ToolExecuteResponse.model_validate(
            {
                "proceed": False,
                "reason": "authorization required for credential broker resource",
                "status": "suspend",
                "elicitation": {
                    "elicitation_id": "el-123",
                    "authorization_url": "https://auth.example.com/device",
                    "message": "Connect your account to continue",
                    "created": True,
                },
            }
        )

        assert response.elicitation is not None
        assert response.elicitation.authorization_url == "https://auth.example.com/device"
        assert response.elicitation.created is True

    def test_execute_tool_auth_required_triggers_interrupt(self):
        wrapper = self._make_wrapper()
        interrupt_signal = RuntimeError("interrupt called")
        mock_interrupt = Mock(side_effect=interrupt_signal)
        hooks.register_suspend_handler(mock_interrupt)
        response = ToolExecuteResponse.model_validate(
            {
                "proceed": False,
                "reason": "authorization required for credential broker resource",
                "status": "suspend",
                "elicitation": {
                    "elicitation_id": "el-123",
                    "authorization_url": "https://auth.example.com/device",
                    "message": "Connect your account to continue",
                    "created": False,
                },
            }
        )

        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval", return_value=response
        ):
            with pytest.raises(RuntimeError, match="interrupt called"):
                wrapper.execute_tool("demo_tool", {"value": 1})

        mock_interrupt.assert_called_once_with(
            {
                "suspend_reason": "authorization_required",
                "suspend_context": {
                    "authorization_url": "https://auth.example.com/device",
                    "elicitation_id": "el-123",
                    "message": "Connect your account to continue",
                    "created": False,
                },
            }
        )

    def test_execute_tool_auth_required_without_handler_raises_runtime_error(self):
        wrapper = self._make_wrapper()
        response = ToolExecuteResponse.model_validate(
            {
                "proceed": False,
                "reason": "Connect your account to continue",
                "status": "suspend",
                "elicitation": {
                    "elicitation_id": "el-123",
                    "authorization_url": "https://auth.example.com/device",
                    "message": "Connect your account to continue",
                    "created": True,
                },
            }
        )

        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval", return_value=response
        ):
            with pytest.raises(RuntimeError, match="No suspend handler registered"):
                wrapper.execute_tool("demo_tool", {"value": 1})

    def test_execute_tool_proceed_false_without_elicitation_still_raises_policy_denied(self):
        wrapper = self._make_wrapper()
        mock_interrupt = Mock(side_effect=RuntimeError("unexpected"))
        hooks.register_suspend_handler(mock_interrupt)
        response = ToolExecuteResponse(proceed=False, reason="Policy denied")

        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval", return_value=response
        ):
            with pytest.raises(PolicyDeniedException, match="Policy denied"):
                wrapper.execute_tool("demo_tool", {"value": 1})

        mock_interrupt.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers for AERServer streaming tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeInterrupt:
    """Mimics ``langgraph.types.Interrupt`` which carries the interrupt payload."""

    value: dict


def _make_aer_server():
    """Build a minimal AERServer with mocked runtime."""
    from agent_engine_runner_shared.server.aer import AERServer

    runtime = Mock()
    return AERServer(runtime)


async def _async_agent_events(events):
    """Turn an iterable of StreamEvent objects into an async generator."""
    for event in events:
        yield event


class _MockRunResult:
    """Concrete RunResult for tests (the protocol is no longer constructible)."""

    def __init__(self, stream_fn):
        self._stream = stream_fn

    def __await__(self):
        async def _noop():
            return None

        return _noop().__await__()

    async def __aiter__(self):
        async for event in self._stream():
            yield event


def _mock_agent_with_events(events):
    """Create a mock agent whose run() returns a RunResult streaming the events."""
    agent = Mock()
    agent.execute = Mock(
        return_value=_MockRunResult(
            stream_fn=lambda: _async_agent_events(events),
        )
    )
    return agent


class TestExecuteViaAgentStream:
    """Tests for _execute_via_agent_stream with BaseAgent protocol.

    Covers:
    - Suspend: agent yields a "suspend" event.
    - Result: agent yields a "result" event with messages.
    - Token streaming with thinking token filtering.
    """

    @pytest.mark.asyncio
    async def test_suspend_returns_interrupt(self):
        """When agent yields a suspend event, the method returns InterruptResult.

        The adapter's opaque ``metadata`` dict is carried through verbatim; the
        AER does not inspect or transform it.
        """
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        suspend_payload = {
            "suspend_reason": "awaiting_human_review",
            "suspend_context": {"claim_id": "CLM-100"},
        }
        adapter_metadata = {"checkpoint_id": "ckpt-123", "function_call_id": "fc-1"}

        events = [
            StreamEvent(data={"content": "Processing"}, event="token"),
            StreamEvent(
                data={"suspend_payload": suspend_payload, "metadata": adapter_metadata},
                event="suspend",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "File a claim",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-suspend-001",
        )

        assert isinstance(result, InterruptResult)
        # Opaque metadata is forwarded verbatim — no AER inspection.
        assert result.metadata == adapter_metadata
        # suspend_payload stored directly, not wrapped in Interrupt objects
        assert result.suspend_payload == suspend_payload

    @pytest.mark.asyncio
    async def test_suspend_without_metadata_yields_empty_dict(self):
        """When the adapter provides no metadata, the AER stores an empty dict.

        The AER never fabricates resume state (e.g. ``thread:{id}``) — that is
        entirely the framework adapter's responsibility.
        """
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        suspend_payload = {"suspend_reason": "review", "suspend_context": {}}

        events = [
            StreamEvent(
                data={"suspend_payload": suspend_payload},  # No metadata
                event="suspend",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(payload={"message": "Do something"})

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(session_id="my-thread-id"),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-fallback-001",
        )

        assert isinstance(result, InterruptResult)
        assert result.metadata == {}
        assert result.suspend_payload == suspend_payload

    @pytest.mark.asyncio
    async def test_result_returns_streaming_result_with_messages(self):
        """When agent yields a result event, messages are returned in StreamingResult."""
        from agent_engine_sdk import AgentInput, Message, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        # Simulate platform messages returned by the agent
        platform_messages = [
            Message(role="user", content="File a claim"),
            Message(role="assistant", content="Processing your claim..."),
            Message(role="tool", content='{"status":"approved"}', tool_call_id="tc1"),
            Message(role="assistant", content="Your claim CLM-100 has been approved!"),
        ]

        events = [
            StreamEvent(
                data={
                    "response": "Your claim CLM-100 has been approved!",
                    "messages": [m.model_dump() for m in platform_messages],
                },
                event="result",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "File a claim",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-resume-001",
        )

        assert isinstance(result, StreamingResult)
        # Messages are converted to LangChain format
        assert len(result.messages) == 4

    @pytest.mark.asyncio
    async def test_result_content_preserved(self):
        """StreamingResult.content comes from the agent's result event."""
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        events = [
            StreamEvent(
                data={
                    "response": "Claim approved — $14,500 payout.",
                    "messages": [
                        {"role": "assistant", "content": "Claim approved — $14,500 payout."}
                    ],
                },
                event="result",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "Resolve my claim",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-resume-002",
        )

        assert isinstance(result, StreamingResult)
        assert "Claim approved" in result.content
        assert "$14,500" in result.content

    @pytest.mark.asyncio
    async def test_no_result_raises_runtime_error(self):
        """If agent stream ends without result or suspend, a RuntimeError is raised."""
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        # Only token events, no result or suspend
        events = [
            StreamEvent(data={"content": "Processing..."}),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "Do something",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        with pytest.raises(RuntimeError, match="without result or suspend"):
            await server._execute_via_agent_stream(
                agent,
                RequestContext(),
                agent_input,
                oe_url="http://oe:8000",
                execution_id="exec-empty-001",
            )

    @pytest.mark.asyncio
    async def test_thinking_tokens_not_streamed(self):
        """<think>...</think> tokens are suppressed during real-time streaming."""
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        events = [
            StreamEvent(data={"content": "<think>"}),
            StreamEvent(data={"content": "internal reasoning"}),
            StreamEvent(data={"content": "</think>"}),
            StreamEvent(data={"content": "Hello!"}),
            StreamEvent(
                data={
                    "response": "<think>internal reasoning</think>Hello!",
                    "messages": [
                        {"role": "assistant", "content": "<think>internal reasoning</think>Hello!"}
                    ],
                },
                event="result",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "Say hello",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-think-001",
        )

        assert isinstance(result, StreamingResult)
        assert result.content == "Hello!"

        streamed_contents = [
            call.kwargs.get("content", "") for call in server._send_stream_chunk.call_args_list
        ]
        for content in streamed_contents:
            assert "<think>" not in content
            assert "internal reasoning" not in content

    @pytest.mark.asyncio
    async def test_thinking_tokens_stripped_from_final_content(self):
        """Final content has <think> blocks removed even without streaming."""
        from agent_engine_sdk import AgentInput, StreamEvent

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()

        events = [
            StreamEvent(
                data={
                    "response": "<think>Let me think about this</think>The answer is 42.",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<think>Let me think about this</think>The answer is 42.",
                        }
                    ],
                },
                event="result",
            ),
        ]

        agent = _mock_agent_with_events(events)

        agent_input = AgentInput(
            payload={
                "message": "What is the answer?",
                "user_id": "user-1",
                "session_id": "sess-1",
                "thread_id": "thread-1",
            },
        )

        result = await server._execute_via_agent_stream(
            agent,
            RequestContext(),
            agent_input,
            oe_url="http://oe:8000",
            execution_id="exec-think-002",
        )

        assert isinstance(result, StreamingResult)
        assert result.content == "The answer is 42."
        assert "<think>" not in result.content


class TestPreSuspendMemoryWrite:
    """Tests that a suspend persists the pre-suspend turn to STM.

    The suspend branch otherwise writes nothing, so the user prompt and any
    assistant/tool output produced before the interrupt would be lost from
    memory. On a fresh invoke the user turn is included; on a resume leg there
    is no new user prompt, so it is omitted (include_user_turn=not resume).
    """

    def _server_with_suspend(self):
        """An AER server wired to produce a suspend outcome with a memory writer."""
        from agent_engine_runner_shared.models import InterruptResult

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()
        server._send_callback_body = AsyncMock()
        server.runtime._graph_builder = Mock()
        server.runtime.org_id = None
        # Truthy memory writer so the suspend branch attempts the STM write,
        # and a mocked write_turn_async to capture the call.
        server.runtime._memory_writer = Mock()
        server.runtime.write_turn_async = Mock()

        presuspend_messages = [
            {"role": "user", "content": "File a claim"},
            {"role": "assistant", "content": "Checking your policy..."},
        ]

        async def _suspend_stream(agent, ctx, agent_input, oe_url, execution_id):
            return InterruptResult(
                suspend_payload={"suspend_reason": "Need approval", "suspend_context": {}},
                messages=presuspend_messages,
                metadata={"checkpoint_id": "ckpt-1"},
            )

        server._execute_via_agent_stream = _suspend_stream
        return server, presuspend_messages

    @pytest.mark.asyncio
    async def test_fresh_suspend_writes_turn_including_user(self):
        """A fresh (non-resume) suspend queues the pre-suspend turn with the user prompt."""
        from agent_engine_runner_shared.models import ExecuteRequest

        server, presuspend_messages = self._server_with_suspend()

        request = ExecuteRequest(
            execution_id="exec-suspend-fresh",
            message="File a claim",
            platform_api_url="http://oe:8000",
            user_id="user-1",
            session_id="session-suspend-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "suspended"
        server.runtime.write_turn_async.assert_called_once()
        kwargs = server.runtime.write_turn_async.call_args.kwargs
        assert kwargs["message"] == "File a claim"
        assert kwargs["result_messages"] == presuspend_messages
        assert kwargs["user_id"] == "user-1"
        assert kwargs["session_id"] == "session-suspend-1"
        # Fresh invoke → include the user turn.
        assert kwargs["include_user_turn"] is True

    @pytest.mark.asyncio
    async def test_legacy_interrupt_envelope_forwards_suspend_payload(self):
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()
        interrupts = [
            PendingInterrupt(
                id="int-1",
                value={
                    "suspend_reason": "awaiting_human_review",
                    "suspend_context": {"allowed_decisions": ["approve", "reject"]},
                },
            )
        ]
        resume_schema = {"type": "object", "required": ["message"]}

        async def _suspend_stream(agent, ctx, agent_input, oe_url, execution_id):
            return InterruptResult(
                suspend_payload=interrupts[0].value,
                interrupts=interrupts,
                resume_schema=resume_schema,
                metadata={"checkpoint_id": "ckpt-1"},
            )

        server._execute_via_agent_stream = _suspend_stream
        request = ExecuteRequest(
            execution_id="exec-legacy-envelope",
            message="File a claim",
            platform_api_url="http://oe:8000",
            session_id="session-suspend-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "suspended"
        callback = server._send_callback_body.call_args.args[1]
        assert callback["suspend_reason"] == "awaiting_human_review"
        assert callback["suspend_context"] == {"allowed_decisions": ["approve", "reject"]}
        assert callback["interrupts"] == [interrupt.model_dump() for interrupt in interrupts]
        assert callback["resume_schema"] == resume_schema

    @pytest.mark.asyncio
    async def test_nonlegacy_interrupt_envelope_uses_generic_suspend_reason(self):
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()
        interrupts = [PendingInterrupt(id="int-1", value={"question": "Approve?"})]
        resume_schema = {"type": "object", "required": ["message"]}

        async def _suspend_stream(agent, ctx, agent_input, oe_url, execution_id):
            return InterruptResult(
                suspend_payload=interrupts[0].value,
                interrupts=interrupts,
                resume_schema=resume_schema,
                metadata={"checkpoint_id": "ckpt-1"},
            )

        server._execute_via_agent_stream = _suspend_stream
        request = ExecuteRequest(
            execution_id="exec-nonlegacy-envelope",
            message="File a claim",
            platform_api_url="http://oe:8000",
            session_id="session-suspend-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "suspended"
        callback = server._send_callback_body.call_args.args[1]
        assert callback["suspend_reason"] == "agent_interrupt"
        assert callback["suspend_context"] is None
        assert callback["interrupts"] == [interrupt.model_dump() for interrupt in interrupts]
        assert callback["resume_schema"] == resume_schema

    @pytest.mark.asyncio
    async def test_multi_interrupt_envelope_is_forwarded_unchanged(self):
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()
        interrupts = [
            PendingInterrupt(id="int-1", value={"question": "Approve?"}),
            PendingInterrupt(id="int-2", value=["a", "b"]),
        ]
        resume_schema = {"type": "object", "required": ["resume_map"]}

        async def _suspend_stream(agent, ctx, agent_input, oe_url, execution_id):
            return InterruptResult(
                suspend_payload=interrupts[0].value,
                interrupts=interrupts,
                resume_schema=resume_schema,
                metadata={"checkpoint_id": "ckpt-1"},
            )

        server._execute_via_agent_stream = _suspend_stream
        request = ExecuteRequest(
            execution_id="exec-multi-envelope",
            message="File a claim",
            platform_api_url="http://oe:8000",
            session_id="session-suspend-1",
        )

        await server._handle_execute(request)

        callback = server._send_callback_body.call_args.args[1]
        assert callback["interrupts"] == [interrupt.model_dump() for interrupt in interrupts]
        assert callback["resume_schema"] == resume_schema

    @pytest.mark.asyncio
    async def test_non_serializable_interrupt_error_reports_terminal_callback(self):
        """Adapter validation failures must terminate the turn instead of hanging."""
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()
        error_message = (
            'LangGraph interrupt "approval" has a non-JSON-serializable value of type set'
        )

        async def _invalid_interrupt(agent, ctx, agent_input, oe_url, execution_id):
            raise ValueError(error_message)

        server._execute_via_agent_stream = _invalid_interrupt
        request = ExecuteRequest(
            execution_id="exec-non-serializable",
            message="review this",
            platform_api_url="http://oe:8000",
            session_id="session-suspend-1",
        )

        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(request)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == error_message
        server._send_callback_body.assert_awaited_once()
        assert server._send_callback_body.await_args.args[0] == "http://oe:8000"
        assert server._send_callback_body.await_args.args[1] == {
            "execution_id": "exec-non-serializable",
            "status": "ERROR",
            "error": error_message,
            "result": None,
            "suspend_reason": None,
            "suspend_context": None,
            "interrupts": None,
            "resume_schema": None,
            "metadata": None,
        }

    @pytest.mark.asyncio
    async def test_report_callback_rejects_nan_before_posting(self):
        """NaN survives json.dumps by default but is invalid JSON on the wire; the
        pre-serialization guard must reject it before the callback is posted."""
        server = _make_aer_server()
        client = AsyncMock()
        server._get_client = AsyncMock(return_value=client)

        with pytest.raises(ValueError, match="not JSON compliant"):
            await server._report_callback(
                "http://oe:8000",
                "exec-nan",
                status="SUSPENDED",
                suspend_reason="agent_interrupt",
                suspend_context={"interrupt_values": [float("nan")]},
            )

        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resume_suspend_omits_user_turn(self):
        """A resume-leg suspend queues the pre-suspend turn WITHOUT re-recording the user prompt."""
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()

        request = ExecuteRequest(
            execution_id="exec-suspend-resume",
            message="",  # resume legs carry no new user prompt
            platform_api_url="http://oe:8000",
            user_id="user-1",
            session_id="session-suspend-1",
            resume=True,
        )

        result = await server._handle_execute(request)

        assert result.status == "suspended"
        server.runtime.write_turn_async.assert_called_once()
        # Resume leg → no new user prompt, so it must be omitted.
        assert server.runtime.write_turn_async.call_args.kwargs["include_user_turn"] is False

    @pytest.mark.asyncio
    async def test_suspend_without_memory_writer_skips_write(self):
        """With no memory writer configured, the suspend branch attempts no STM write."""
        from agent_engine_runner_shared.models import ExecuteRequest

        server, _ = self._server_with_suspend()
        server.runtime._memory_writer = None  # disable memory

        request = ExecuteRequest(
            execution_id="exec-suspend-nomem",
            message="File a claim",
            platform_api_url="http://oe:8000",
            user_id="user-1",
            session_id="session-suspend-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "suspended"
        server.runtime.write_turn_async.assert_not_called()


class TestResumeFromStep:
    """Tests for resume_from_step advancing wrapper.step_counter in AER."""

    @pytest.mark.asyncio
    async def test_resume_from_step_sets_wrapper_step_counter(self):
        """When resume_from_step is set, the wrapper's step_counter is advanced
        before the agent executes, so OE does not serve stale cached results."""

        from agent_engine_sdk import StreamEvent

        from agent_engine_runner_shared.context import get_current_wrapper
        from agent_engine_runner_shared.models import ExecuteRequest

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()
        server._send_callback_body = AsyncMock()
        server.runtime._graph_builder = Mock()
        server.runtime._memory_writer = None
        server.runtime.org_id = None

        captured_step_counter = {}

        original_stream = server._execute_via_agent_stream

        async def capture_wrapper_then_stream(agent, ctx, agent_input, oe_url, execution_id):
            wrapper = get_current_wrapper()
            captured_step_counter["value"] = wrapper.step_counter
            events = [
                StreamEvent(
                    data={
                        "response": "Resumed.",
                        "messages": [{"role": "assistant", "content": "Resumed."}],
                    },
                    event="result",
                ),
            ]
            mock_agent = _mock_agent_with_events(events)
            return await original_stream(mock_agent, ctx, agent_input, oe_url, execution_id)

        server._execute_via_agent_stream = capture_wrapper_then_stream

        request = ExecuteRequest(
            execution_id="exec-resume-step-001",
            message="Continue after suspend",
            platform_api_url="http://oe:8000",
            resume=True,
            resume_from_step=7,
            session_id="session-resume-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "completed"
        assert captured_step_counter["value"] == 7

    @pytest.mark.asyncio
    async def test_fresh_execution_starts_at_step_zero(self):
        """Without resume_from_step, the wrapper starts at step_counter=0."""

        from agent_engine_sdk import StreamEvent

        from agent_engine_runner_shared.context import get_current_wrapper
        from agent_engine_runner_shared.models import ExecuteRequest

        server = _make_aer_server()
        server._send_stream_chunk = AsyncMock()
        server._send_callback_body = AsyncMock()
        server.runtime._graph_builder = Mock()
        server.runtime._memory_writer = None
        server.runtime.org_id = None

        captured_step_counter = {}

        original_stream = server._execute_via_agent_stream

        async def capture_wrapper_then_stream(agent, ctx, agent_input, oe_url, execution_id):
            wrapper = get_current_wrapper()
            captured_step_counter["value"] = wrapper.step_counter
            events = [
                StreamEvent(
                    data={
                        "response": "Fresh run.",
                        "messages": [{"role": "assistant", "content": "Fresh run."}],
                    },
                    event="result",
                ),
            ]
            mock_agent = _mock_agent_with_events(events)
            return await original_stream(mock_agent, ctx, agent_input, oe_url, execution_id)

        server._execute_via_agent_stream = capture_wrapper_then_stream

        request = ExecuteRequest(
            execution_id="exec-fresh-001",
            message="Hello",
            platform_api_url="http://oe:8000",
            session_id="session-fresh-1",
        )

        result = await server._handle_execute(request)

        assert result.status == "completed"
        assert captured_step_counter["value"] == 0
