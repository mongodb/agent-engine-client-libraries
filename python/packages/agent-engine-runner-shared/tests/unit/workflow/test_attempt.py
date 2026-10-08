"""Tests for AttemptStart request assembly and reconnect identity."""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import StepSuspensionEntry
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    ActivityPosition,
    OperationPath,
    OperationPathSegment,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import ExecuteRequest
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.attempt import (
    attempt_start_request_from_execute,
    finalize_current_step_suspensions_command,
    heartbeat_interval_seconds_from_attempt,
    new_durability_owner_id,
    workflow_identity_from_execute_request,
)
from agent_engine_runner_shared.workflow.context import record_observed_activity


def _execute_request(**overrides: object) -> ExecuteRequest:
    fields: dict = {
        "execution_id": "execution-1",
        "message": "hi",
        "platform_api_url": "http://oe:8000",
        "org_id": "org-1",
        "project_id": "project-1",
        "workspace_id": "workspace-1",
    }
    fields.update(overrides)
    return ExecuteRequest(**fields)


@pytest.fixture()
def registered_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_engine_runner_shared.hooks as hooks

    monkeypatch.setattr(hooks, "_workflow_adapter", ("langgraph", "0.1.0"))


class TestAttemptStartRequest:
    def test_builds_complete_identity_and_declaration(self, registered_adapter: None) -> None:
        request = attempt_start_request_from_execute(
            _execute_request(),
            "session-1",
            "owner-1",
            "insurance",
            memory_enabled=True,
        )

        assert request is not None
        identity = request.workflow_identity
        assert identity.tenant_scope.org_id == "org-1"
        assert identity.tenant_scope.project_id == "project-1"
        assert identity.tenant_scope.workspace_id == "workspace-1"
        assert identity.session_id == "session-1"
        assert identity.execution_id == "execution-1"
        assert request.owner_id == "owner-1"
        assert request.declaration.workflow_name == "insurance"
        assert request.declaration.workflow_version == "1"  # default when unspecified
        assert request.declaration.adapter_name == "langgraph"
        assert request.declaration.adapter_version
        assert request.declaration.memory_enabled is True

    @pytest.mark.parametrize(
        "overrides",
        [
            {"org_id": None},
            {"org_id": " "},
            {"project_id": None},
            {"workspace_id": None},
        ],
    )
    def test_incomplete_tenant_identity_stays_native(self, overrides: dict) -> None:
        assert (
            attempt_start_request_from_execute(
                _execute_request(**overrides), "session-1", "owner-1", "app"
            )
            is None
        )

    def test_configured_application_version_reaches_the_declaration(
        self, registered_adapter: None
    ) -> None:
        request = attempt_start_request_from_execute(
            _execute_request(), "session-1", "owner-1", "insurance", workflow_version="2.3.4"
        )
        assert request is not None
        assert request.declaration.workflow_version == "2.3.4"

    def test_blank_session_stays_native(self) -> None:
        assert workflow_identity_from_execute_request(_execute_request(), "  ") is None

    def test_framework_without_registered_adapter_stays_native(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agent_engine_runner_shared.hooks as hooks

        monkeypatch.setattr(hooks, "_workflow_adapter", None)
        # Complete tenant identity is not enough: an unregistered framework
        # cannot qualify a durable attempt.
        assert (
            attempt_start_request_from_execute(_execute_request(), "session-1", "owner-1", "app")
            is None
        )


def test_owner_id_is_unique_per_call() -> None:
    first, second = new_durability_owner_id(), new_durability_owner_id()
    assert first != second
    assert len(first.split(":")) >= 3


class TestHeartbeatInterval:
    def test_converts_milliseconds_to_seconds(self) -> None:
        assert heartbeat_interval_seconds_from_attempt(1500) == 1.5

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_interval(self, value: int) -> None:
        with pytest.raises(ValueError):
            heartbeat_interval_seconds_from_attempt(value)


class TestFinalizeStepSuspensionsCommand:
    def test_builds_the_current_positioned_frontier(self) -> None:
        attempt = AttemptContext(
            attempt_id="attempt-1",
            fencing_token=3,
            workflow_identity=WorkflowIdentity(execution_id="execution-1"),
        )
        position = ActivityPosition(
            step_ordinal=1,
            operation_path=OperationPath(segments=[OperationPathSegment(name="agent", ordinal=1)]),
            activity_ordinal=1,
        )
        entry = StepSuspensionEntry(position=position)

        with attempt_context_scope(attempt):
            record_observed_activity(position)
            command = finalize_current_step_suspensions_command(attempt, [entry])

        assert command.attempt_id == "attempt-1"
        assert command.fencing_token == 3
        assert command.step_ordinal == 1
        assert list(command.observed_activity_positions) == [position]
        assert list(command.suspensions) == [entry]
        assert not command.HasField("state")

    def test_rejects_empty_or_wrong_step_frontiers(self) -> None:
        attempt = AttemptContext(attempt_id="attempt-1")
        with attempt_context_scope(attempt):
            with pytest.raises(ValueError, match="at least one"):
                finalize_current_step_suspensions_command(attempt, [])
            with pytest.raises(ValueError, match="another workflow step"):
                finalize_current_step_suspensions_command(
                    attempt,
                    [StepSuspensionEntry(position=ActivityPosition(step_ordinal=2))],
                )
