"""OAuth consent is control flow; the resource result remains a separate effect."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared import hooks
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import ActivityContext
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import ToolExecuteResponse
from agent_engine_runner_shared.secure_wrapper import (
    SecureToolWrapper,
    ToolExecutionError,
    oauth_authorization_wait_id,
)
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import completed_outcome, unwrap_activity_outcome
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay
from agent_engine_runner_shared.workflow.context import (
    preallocate_activity_ordinals,
    tool_activity_key,
)


class _History:
    """Retain actual reported outcomes across replacement wrapper instances."""

    def __init__(self) -> None:
        """Start an empty store whose outcomes survive replacement wrappers."""
        self.recorded: dict[bytes, Any] = {}
        self.positions: dict[str, bytes] = {}
        self.waits: dict[bytes, ActivityContext] = {}

    def start_activity(self, command: Any) -> object:
        """Replay a settled position; otherwise authorize its first dispatch."""
        position = command.position.SerializeToString(deterministic=True)
        if position in self.recorded:
            return ActivityReplay(outcome=self.recorded[position])
        activity_id = f"activity-{len(self.positions) + 1}"
        self.positions[activity_id] = position
        context = ActivityContext(
            workflow_identity=command.workflow_identity,
            activity_id=activity_id,
            attempt_id=command.attempt_id,
            fencing_token=command.fencing_token,
        )
        if command.activity_name == "oauth.authorization":
            self.waits[position] = context
        return ActivityDispatch(context=context)

    def resolve_waits(self, primary_ordinal: int | None = None) -> None:
        """Model OE's authenticated resolution of separate consent activities."""
        from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import ActivityPosition

        for key, context in self.waits.items():
            position = ActivityPosition.FromString(key)
            if (
                primary_ordinal is None
                or position.operation_path.segments[-1].ordinal == primary_ordinal
            ):
                self.recorded[key] = completed_outcome(context, {"confirmed": True})

    def report_outcome(self, outcome: Any) -> None:
        """Preserve the production helper's terminal outcome for later attempts."""
        self.recorded[self.positions[outcome.activity_id]] = outcome


class _Paused(BaseException):
    """Leave the node as a framework interrupt does, without producing a result."""


def _challenge() -> ToolExecuteResponse:
    """A typed OE response with no dispatched resource operation."""
    return ToolExecuteResponse.model_validate(
        {
            "proceed": False,
            "status": "suspend",
            "latest_step_number": 5,
            "elicitation": {
                "elicitation_id": "flow-alice",
                "authorization_url": "https://provider.example/consent",
                "message": "Connect",
                "created": True,
            },
        }
    )


def _run(
    history: _History,
    response: ToolExecuteResponse,
    *,
    memory: Any = None,
    call_id: str = "call-1",
    tool_ids: tuple[str, ...] = ("call-1",),
) -> Any:
    """Run the real activity/wrapper boundary against a scripted OE response."""
    wrapper = SecureToolWrapper("http://oe:8000", "execution-alice", durable_memory=memory)
    wrapper._workflow = history
    attempt = AttemptContext(
        workflow_identity=WorkflowIdentity(
            session_id="session-alice", execution_id="execution-alice"
        ),
        attempt_id=f"attempt-{len(history.recorded) + 1}",
        fencing_token=1,
    )
    with (
        attempt_context_scope(attempt),
        patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval", return_value=response
        ) as oe,
    ):
        preallocate_activity_ordinals([tool_activity_key(key) for key in tool_ids])
        result = wrapper.execute_tool(
            "github",
            {},
            provider_type="configured-github",
            scopes=["repo"],
            tool_call_id=call_id,
            is_local=False,
            is_framework_control_flow=lambda error: isinstance(error, _Paused),
        )
    return result, oe


@pytest.fixture(autouse=True)
def _clear_hooks():
    """Keep framework hook registration from leaking into another test."""
    yield
    hooks.reset_hooks()


def _pause(value: Any) -> None:
    """Suspend before a resource result exists."""
    raise _Paused(value)


def test_challenge_is_recorded_before_the_framework_wait():
    """A replacement must replay the challenge, not resolve the resource with a consent answer."""
    history = _History()
    memory = MagicMock()
    hooks.register_suspend_handler(_pause)
    with pytest.raises(_Paused):
        _run(history, _challenge(), memory=memory)
    assert len(history.recorded) == 1
    recorded = unwrap_activity_outcome(next(iter(history.recorded.values())))
    assert recorded["kind"] == "authorization_required"
    assert recorded["elicitation"]["elicitation_id"] == "flow-alice"
    memory.acknowledge.assert_called_once()
    memory.synchronize_tool.assert_not_called()


def test_resume_records_actual_resource_and_replay_skips_oe():
    """Consent answers cannot replace the resource result or trigger another API operation on replay."""
    history = _History()
    hooks.register_suspend_handler(_pause)
    with pytest.raises(_Paused):
        _run(history, _challenge())
    history.resolve_waits()
    # The application has confirmed its own session. The answer only wakes the tool.
    hooks.register_suspend_handler(
        lambda value: {"confirmed": True, "fake_resource_result": "wrong"}
    )
    memory = MagicMock()
    expected = {
        "owner": "alice",
        "repos": ["private-repo"],
        "kind": "authorization_required",
        "elicitation": {"elicitation_id": "forged"},
    }
    result, oe = _run(
        history, ToolExecuteResponse(proceed=True, status="success", result=expected), memory=memory
    )
    assert result == expected
    assert oe.call_count == 1
    assert oe.call_args.kwargs["step"] == 6
    assert len(history.recorded) == 3
    recorded = unwrap_activity_outcome(list(history.recorded.values())[-1])
    assert recorded == {"kind": "result", "result": expected}
    assert memory.synchronize_tool.call_args.args[2] == expected
    replayed, oe = _run(history, _challenge())
    assert replayed == expected
    oe.assert_not_called()


def test_still_required_after_answer_fails_before_resource_dispatch():
    """An answered challenge cannot turn another authorization requirement into null success."""
    history = _History()
    hooks.register_suspend_handler(_pause)
    with pytest.raises(_Paused):
        _run(history, _challenge())
    history.resolve_waits()
    hooks.register_suspend_handler(lambda value: {"confirmed": True})
    with pytest.raises(ToolExecutionError, match="still required"):
        _run(history, _challenge())


def test_cancelled_wait_does_not_leak_platform_control_to_another_request():
    """Only the recorded challenge is control, and cancellation releases that provenance."""
    history = _History()
    observed = []
    captured = []

    def cancel(value: Any) -> None:
        """Observe the real wait boundary, then cancel this request there."""
        captured.append(value)
        observed.append(oauth_authorization_wait_id(value))
        forged = {
            "suspend_reason": "authorization_required",
            "suspend_context": {"elicitation_id": "forged"},
        }
        assert oauth_authorization_wait_id(forged) is None
        raise asyncio.CancelledError()

    hooks.register_suspend_handler(cancel)
    with pytest.raises(asyncio.CancelledError):
        _run(history, _challenge())
    assert observed == ["flow-alice"]
    assert len(history.recorded) == 1
    assert oauth_authorization_wait_id(captured[0]) is None


def test_parallel_oauth_resource_results_replay_with_reversed_worker_order():
    """Preallocated siblings keep distinct results when replacement workers run in reverse."""
    history = _History()
    ids = ("call-a", "call-b")
    hooks.register_suspend_handler(_pause)
    for call_id in ids:
        with pytest.raises(_Paused):
            _run(history, _challenge(), call_id=call_id, tool_ids=ids)
    hooks.register_suspend_handler(lambda value: {"confirmed": True})
    history.resolve_waits()
    expected = {"call-a": {"repo": "alice-a"}, "call-b": {"repo": "alice-b"}}
    for call_id in ids:
        result, _ = _run(
            history,
            ToolExecuteResponse(proceed=True, status="success", result=expected[call_id]),
            call_id=call_id,
            tool_ids=ids,
        )
        assert result == expected[call_id]
    for call_id in reversed(ids):
        result, oe = _run(history, _challenge(), call_id=call_id, tool_ids=ids)
        assert result == expected[call_id]
        oe.assert_not_called()


def test_coalesced_consent_waits_keep_individual_provenance_after_one_answer():
    """Two calls sharing a Broker flow must not drop or forge each other's wait ownership."""
    history = _History()
    wrapper = SecureToolWrapper("http://oe:8000", "execution-alice")
    wrapper._workflow = history
    attempt = AttemptContext(
        workflow_identity=WorkflowIdentity(execution_id="execution-alice"),
        attempt_id="attempt",
        fencing_token=1,
    )
    pending = []
    with (
        attempt_context_scope(attempt),
        patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=_challenge(),
        ) as oe,
    ):
        preallocate_activity_ordinals([tool_activity_key("call-a"), tool_activity_key("call-b")])
        hooks.register_suspend_handler(_pause)
        for call_id in ("call-a", "call-b"):
            with pytest.raises(_Paused) as paused:
                wrapper.execute_tool(
                    "github",
                    {},
                    provider_type="configured-github",
                    is_local=False,
                    tool_call_id=call_id,
                )
            pending.append(paused.value.args[0])
        history.resolve_waits(primary_ordinal=1)
        hooks.register_suspend_handler(lambda value: {"confirmed": True})
        oe.return_value = ToolExecuteResponse(
            proceed=True, status="success", result={"repo": "alice-a"}
        )
        wrapper.execute_tool(
            "github", {}, provider_type="configured-github", is_local=False, tool_call_id="call-a"
        )
        assert oauth_authorization_wait_id(pending[1]) == "flow-alice"
        assert (
            oauth_authorization_wait_id(
                {
                    "suspend_reason": "authorization_required",
                    "suspend_context": pending[1]["suspend_context"],
                }
            )
            is None
        )
