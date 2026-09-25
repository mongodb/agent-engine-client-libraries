"""Tests for serial activity helpers and the activity client endpoints."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_LLM,
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_DENIED,
    ACTIVITY_OUTCOME_KIND_FAILED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityContext,
    ActivityOutcome,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
    WORKFLOW_ERROR_CODE_STALE_FENCE,
    WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    WorkflowError,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import WorkflowClientError
from agent_engine_runner_shared.workflow.activity import (
    DurableActivityControlFlow,
    DurableActivityDeniedError,
    DurableActivityInterrupted,
    DurableActivitySuspended,
    ReplayedActivityFailedError,
    build_activity_command,
    completed_outcome,
    failed_outcome,
    run_serial_activity,
    semantic_input_from_json,
    suspended_outcome,
    unwrap_activity_outcome,
    value_to_json,
)
from agent_engine_runner_shared.workflow.protojson import proto_struct_to_json


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


def _context() -> ActivityContext:
    return ActivityContext(
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
    )


class TestBuildActivityCommand:
    def test_builds_from_attempt_with_root_serial_path(self) -> None:
        command = build_activity_command(
            attempt=_attempt(),
            kind=ACTIVITY_KIND_LLM,
            name="gpt-test",
            activity_ordinal=3,
            step_ordinal=1,
            semantic_input={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert command.workflow_identity.execution_id == "execution-1"
        assert command.attempt_id == "attempt-1"
        assert command.fencing_token == 7
        assert command.activity_kind == ACTIVITY_KIND_LLM
        assert command.activity_name == "gpt-test"
        segments = command.position.operation_path.segments
        assert [(s.name, s.ordinal) for s in segments] == [("agent", 1)]
        assert command.position.activity_ordinal == 3
        assert command.position.step_ordinal == 1
        assert value_to_json(command.semantic_input) == {
            "messages": [{"role": "user", "content": "hi"}]
        }

    def test_requires_workflow_identity(self) -> None:
        with pytest.raises(WorkflowClientError) as raised:
            build_activity_command(
                attempt=AttemptContext(attempt_id="attempt-1", fencing_token=1),
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
            )
        assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT

    def test_requires_positive_ordinal(self) -> None:
        with pytest.raises(WorkflowClientError) as raised:
            build_activity_command(
                attempt=_attempt(),
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=0,
                step_ordinal=1,
                semantic_input=None,
            )
        assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT

    def test_requires_positive_step_ordinal(self) -> None:
        with pytest.raises(WorkflowClientError) as raised:
            build_activity_command(
                attempt=_attempt(),
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=0,
                semantic_input=None,
            )
        assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT

    def test_omitted_step_ordinal_uses_advanced_request_counter(self) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.context import advance_step_ordinal

        with attempt_context_scope(_attempt()):
            advance_step_ordinal(1)
            command = build_activity_command(
                attempt=_attempt(),
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                semantic_input=None,
            )
        assert command.position.step_ordinal == 2

    def test_omitted_operation_path_snapshots_current_nested_path(self) -> None:
        from agent_engine_runner_shared.workflow import (
            ChildOperationBoundary,
            attempt_context_scope,
            operation_path_resolver_scope,
        )

        with attempt_context_scope(_attempt()):
            with operation_path_resolver_scope(
                lambda: (
                    ChildOperationBoundary("case_investigation", "task-a"),
                    ChildOperationBoundary("policy_analysis", "task-b"),
                )
            ):
                command = build_activity_command(
                    attempt=_attempt(),
                    kind=ACTIVITY_KIND_TOOL,
                    name="lookup_policy",
                    activity_ordinal=1,
                    semantic_input={"policy": "refund"},
                )

        assert [
            (segment.name, segment.ordinal) for segment in command.position.operation_path.segments
        ] == [
            ("agent", 1),
            ("case_investigation", 1),
            ("policy_analysis", 1),
        ]


class TestSemanticInputRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [None, "text", 3.5, True, {"nested": {"list": [1, 2]}}, ["a", {"b": None}]],
    )
    def test_round_trips_json_values(self, value: object) -> None:
        assert value_to_json(semantic_input_from_json(value)) == value


class TestUnwrapActivityOutcome:
    def test_completed_returns_result_json(self) -> None:
        outcome = completed_outcome(_context(), {"claim_id": "CLM-1"})
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_COMPLETED
        assert unwrap_activity_outcome(outcome) == {"claim_id": "CLM-1"}

    def test_denied_raises_stable_denial(self) -> None:
        outcome = ActivityOutcome(
            outcome_kind=ACTIVITY_OUTCOME_KIND_DENIED,
            error=WorkflowError(message="policy denied"),
        )
        with pytest.raises(DurableActivityDeniedError, match="policy denied"):
            unwrap_activity_outcome(outcome)

    def test_failed_raises_with_stable_code(self) -> None:
        outcome = failed_outcome(_context(), "boom", WORKFLOW_ERROR_CODE_CONFLICT)
        with pytest.raises(ReplayedActivityFailedError) as raised:
            unwrap_activity_outcome(outcome)
        assert raised.value.code == WORKFLOW_ERROR_CODE_CONFLICT
        assert raised.value.message == "boom"

    def test_failed_without_code_maps_to_outcome_unknown(self) -> None:
        outcome = ActivityOutcome(
            outcome_kind=ACTIVITY_OUTCOME_KIND_FAILED,
            error=WorkflowError(message="boom"),
        )
        with pytest.raises(ReplayedActivityFailedError) as raised:
            unwrap_activity_outcome(outcome)
        assert raised.value.code == WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN

    def test_suspended_reconstructs_the_recorded_wait(self) -> None:
        outcome = suspended_outcome(
            _context(),
            reason="awaiting_human_review",
            context={"ticket": "T-1"},
        )
        with pytest.raises(DurableActivitySuspended) as raised:
            unwrap_activity_outcome(outcome)
        assert raised.value.reason == "awaiting_human_review"
        assert raised.value.context == {"ticket": "T-1"}

    def test_unspecified_kind_is_invalid(self) -> None:
        with pytest.raises(WorkflowClientError) as raised:
            unwrap_activity_outcome(ActivityOutcome())
        assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT


class _FakeClient:
    """Scripted stand-in capturing the start/report exchange."""

    def __init__(self, started: object) -> None:
        self.started = started
        self.commands: list[object] = []
        self.outcomes: list[ActivityOutcome] = []
        self.report_error: WorkflowClientError | None = None

    def start_activity(self, command: object) -> object:
        self.commands.append(command)
        return self.started

    def report_outcome(self, outcome: ActivityOutcome) -> None:
        self.outcomes.append(outcome)
        if self.report_error is not None:
            raise self.report_error


class TestRunSerialActivity:
    def test_replay_returns_recorded_outcome_without_execution(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = completed_outcome(_context(), {"cached": True})
        recorded.suspension.reason = "awaiting_human_review"
        client = _FakeClient(ActivityReplay(outcome=recorded))
        calls: list[str] = []

        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="lookup",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input={"q": 1},
            execute=lambda context: calls.append("ran") or {"fresh": True},
            attempt=_attempt(),
        )

        assert result == {"cached": True}
        assert calls == []
        assert client.outcomes == []

    def test_framework_interrupt_replay_reenters_the_activity(self) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityReplay
        from agent_engine_runner_shared.workflow.context import set_activity_reconstruction_ids

        recorded = completed_outcome(_context(), {"decision": "approve"})
        client = _FakeClient(ActivityReplay(outcome=recorded))
        calls: list[ActivityContext] = []
        resolved: list[tuple[ActivityContext, object]] = []

        with attempt_context_scope(_attempt()):
            set_activity_reconstruction_ids(("activity-1",))
            result = run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="review",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"claim": "CLM-1"},
                execute=lambda context: calls.append(context) or {"tool_result": "continued"},
                on_activity_resolved=lambda _client, context, value: resolved.append(
                    (context, value)
                ),
                attempt=_attempt(),
            )

        assert result == {"tool_result": "continued"}
        assert calls == [_context()]
        assert resolved == [(_context(), {"tool_result": "continued"})]
        assert client.outcomes == []

    def test_framework_interrupt_reconstruction_preserves_callback_failure(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityReplay
        from agent_engine_runner_shared.workflow.context import set_activity_reconstruction_ids

        recorded = completed_outcome(_context(), {"decision": "approve"})
        client = _FakeClient(ActivityReplay(outcome=recorded))
        failure = RuntimeError("callback failed")

        def fail(_context: ActivityContext) -> None:
            raise failure

        with attempt_context_scope(_attempt()):
            set_activity_reconstruction_ids(("activity-1",))
            with pytest.raises(RuntimeError) as raised:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="review",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input={"claim": "CLM-1"},
                    execute=fail,
                    attempt=_attempt(),
                )

        assert raised.value is failure
        assert client.outcomes == []

    def test_framework_interrupt_reconstruction_preserves_callback_denial(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityReplay
        from agent_engine_runner_shared.workflow.context import set_activity_reconstruction_ids

        recorded = completed_outcome(_context(), {"decision": "approve"})
        client = _FakeClient(ActivityReplay(outcome=recorded))
        denial = DurableActivityDeniedError("callback denied")

        def deny(_context: ActivityContext) -> None:
            raise denial

        with attempt_context_scope(_attempt()):
            set_activity_reconstruction_ids(("activity-1",))
            with pytest.raises(DurableActivityDeniedError) as raised:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="review",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input={"claim": "CLM-1"},
                    execute=deny,
                    attempt=_attempt(),
                )

        assert raised.value is denial
        assert client.outcomes == []

    def test_framework_interrupt_reconstruction_rejects_a_second_interrupt(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityReplay
        from agent_engine_runner_shared.workflow.context import (
            interrupted_activities,
            set_activity_reconstruction_ids,
        )

        recorded = completed_outcome(_context(), {"decision": "approve"})
        client = _FakeClient(ActivityReplay(outcome=recorded))
        first = RuntimeError("reconstructed interrupt")
        second = RuntimeError("second interrupt")

        def interrupt(error: RuntimeError):
            def execute(_context: ActivityContext) -> None:
                raise DurableActivityInterrupted(error)

            return execute

        with attempt_context_scope(_attempt()):
            set_activity_reconstruction_ids(("activity-1",))
            with pytest.raises(RuntimeError) as raised:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="review",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input={"claim": "CLM-1"},
                    execute=interrupt(first),
                    attempt=_attempt(),
                )
            assert raised.value is first

            with pytest.raises(RuntimeError, match="cannot raise another"):
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="review",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input={"claim": "CLM-1"},
                    execute=interrupt(second),
                    attempt=_attempt(),
                )

            assert len(interrupted_activities(1)) == 1

        assert client.outcomes == []

    def test_framework_interrupt_records_the_activity_command(self) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityDispatch
        from agent_engine_runner_shared.workflow.context import interrupted_activities

        control_flow = RuntimeError("native interrupt")
        client = _FakeClient(ActivityDispatch(context=_context()))

        def interrupt(_context: ActivityContext) -> None:
            raise DurableActivityInterrupted(control_flow)

        with attempt_context_scope(_attempt()):
            with pytest.raises(RuntimeError) as raised:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="review",
                    activity_ordinal=1,
                    semantic_input={"claim": "CLM-1"},
                    execute=interrupt,
                )
            (activity,) = interrupted_activities(1)

        assert raised.value is control_flow
        assert activity.command.activity_name == "review"
        assert activity.command.position.activity_ordinal == 1
        assert activity.control_flow is control_flow
        assert client.outcomes == []

    def test_replay_runs_the_same_post_outcome_hook_without_execution(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = completed_outcome(_context(), {"cached": True})
        client = _FakeClient(ActivityReplay(outcome=recorded))
        resolved: list[tuple[ActivityContext, object]] = []

        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="lookup",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input=None,
            execute=lambda _context: pytest.fail("replay executed the tool"),
            on_activity_resolved=lambda _client, context, value: resolved.append((context, value)),
            attempt=_attempt(),
        )

        assert result == {"cached": True}
        assert resolved == [(_context(), {"cached": True})]
        assert client.outcomes == []

    def test_replay_propagates_post_outcome_failure(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = completed_outcome(_context(), {"cached": True})
        client = _FakeClient(ActivityReplay(outcome=recorded))

        def fail_on_activity_resolved(
            _client: object, _context: ActivityContext, _result: object
        ) -> None:
            raise WorkflowClientError(
                WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
                "synchronization response was lost",
            )

        with pytest.raises(WorkflowClientError):
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="lookup",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda _context: pytest.fail("replay executed the tool"),
                on_activity_resolved=fail_on_activity_resolved,
                attempt=_attempt(),
            )

        assert client.outcomes == []

    def test_replay_raises_recorded_suspension_without_execution(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = suspended_outcome(
            _context(),
            reason="awaiting_human_review",
            context={"ticket": "T-1"},
        )
        client = _FakeClient(ActivityReplay(outcome=recorded))
        calls: list[str] = []

        with pytest.raises(DurableActivitySuspended) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="lookup",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"q": 1},
                execute=lambda context: calls.append("ran") or {"fresh": True},
                attempt=_attempt(),
            )

        assert raised.value.reason == "awaiting_human_review"
        assert raised.value.context == {"ticket": "T-1"}
        assert calls == []
        assert client.outcomes == []

    def test_dispatch_executes_and_reports_completion(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))

        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_LLM,
            name="gpt-test",
            activity_ordinal=2,
            step_ordinal=1,
            semantic_input={"prompt": "hi"},
            execute=lambda context: {"answer": 42},
            attempt=_attempt(),
        )

        assert result == {"answer": 42}
        (outcome,) = client.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_COMPLETED
        assert outcome.activity_id == "activity-1"
        assert unwrap_activity_outcome(outcome) == {"answer": 42}

    def test_dispatch_runs_post_outcome_hook_after_completion_is_accepted(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        resolved: list[tuple[ActivityContext, object, int]] = []

        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="lookup",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input={"city": "Portland"},
            execute=lambda _context: {"temperature": 72},
            on_activity_resolved=lambda _client, context, value: resolved.append(
                (context, value, len(client.outcomes))
            ),
            attempt=_attempt(),
        )

        assert result == {"temperature": 72}
        (outcome,) = client.outcomes
        assert outcome == completed_outcome(_context(), {"temperature": 72})
        assert resolved == [(_context(), {"temperature": 72}, 1)]

    def test_post_outcome_hook_is_not_run_when_completion_is_not_accepted(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        client.report_error = WorkflowClientError(
            WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
            "outcome response was lost",
        )
        resolved: list[bool] = []

        with pytest.raises(WorkflowClientError):
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="lookup",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda _context: "answer",
                on_activity_resolved=lambda _client, _context, _result: resolved.append(True),
                attempt=_attempt(),
            )

        assert resolved == []

    def test_worker_failure_reports_failed_outcome_and_reraises(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))

        def explode(context: ActivityContext) -> None:
            raise ValueError("worker blew up")

        with pytest.raises(ValueError, match="worker blew up"):
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=explode,
                attempt=_attempt(),
            )

        (outcome,) = client.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_FAILED
        assert outcome.error.message == "worker blew up"

    def test_framework_control_flow_leaves_activity_started_and_remains_live(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        class NativeInterrupt(BaseException):
            pass

        interrupt = NativeInterrupt("pause")
        client = _FakeClient(ActivityDispatch(context=_context()))

        def pause(_context: ActivityContext) -> None:
            raise DurableActivityInterrupted(interrupt)

        with pytest.raises(NativeInterrupt) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="approval",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"claim": "CLM-1"},
                execute=pause,
                attempt=_attempt(),
            )

        assert raised.value is interrupt
        assert client.outcomes == []

    def test_suspension_is_recorded_without_becoming_a_failure(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        suspension = DurableActivitySuspended(
            "awaiting_human_review",
            {
                "allowed_decisions": ["approve", "reject"],
                "claim_id": "claim-1",
            },
        )

        def suspend(context: ActivityContext) -> None:
            raise suspension

        with pytest.raises(DurableActivitySuspended) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="request_claim_review",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"claim_id": "claim-1"},
                execute=suspend,
                attempt=_attempt(),
            )

        assert raised.value is suspension
        (outcome,) = client.outcomes
        assert outcome.workflow_identity == _context().workflow_identity
        assert outcome.activity_id == "activity-1"
        assert outcome.attempt_id == "attempt-1"
        assert outcome.fencing_token == 7
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED
        assert outcome.suspension.reason == "awaiting_human_review"
        assert proto_struct_to_json(outcome.suspension.context) == {
            "allowed_decisions": ["approve", "reject"],
            "claim_id": "claim-1",
        }

    def test_rejected_suspension_terminal_fails_the_sibling(self) -> None:
        """OE may reject a second suspension while another wait is open.

        The rejected sibling must still reach a terminal FAILED outcome so the
        open wait is not stranded beside an unfinished dispatched activity.
        """
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        conflict = WorkflowClientError(
            WORKFLOW_ERROR_CODE_CONFLICT,
            "execution already has a suspended activity",
        )
        original_report = client.report_outcome

        def reject_suspension(outcome: ActivityOutcome) -> None:
            if outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED:
                client.outcomes.append(outcome)
                raise conflict
            original_report(outcome)

        client.report_outcome = reject_suspension  # type: ignore[method-assign]
        suspension = DurableActivitySuspended("awaiting_human_review", {})

        def suspend(context: ActivityContext) -> None:
            raise suspension

        with pytest.raises(DurableActivitySuspended) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="request_claim_review",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"claim_id": "claim-1"},
                execute=suspend,
                attempt=_attempt(),
            )

        assert raised.value is suspension
        assert raised.value.__cause__ is conflict
        assert [outcome.outcome_kind for outcome in client.outcomes] == [
            ACTIVITY_OUTCOME_KIND_SUSPENDED,
            ACTIVITY_OUTCOME_KIND_FAILED,
        ]
        assert client.outcomes[1].error.message == str(conflict)

    def test_ambiguous_suspension_delivery_does_not_invent_a_failure(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        ambiguous = WorkflowClientError(
            WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
            "activity outcome commit is uncertain",
        )
        client.report_error = ambiguous
        suspension = DurableActivitySuspended("awaiting_human_review", {})

        def suspend(context: ActivityContext) -> None:
            raise suspension

        with pytest.raises(WorkflowClientError) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="request_claim_review",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"claim_id": "claim-1"},
                execute=suspend,
                attempt=_attempt(),
            )

        assert raised.value is ambiguous
        assert [outcome.outcome_kind for outcome in client.outcomes] == [
            ACTIVITY_OUTCOME_KIND_SUSPENDED,
        ]

    def test_unsupported_control_flow_remains_the_explicit_failure(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        report_error = WorkflowClientError(
            WORKFLOW_ERROR_CODE_STALE_FENCE,
            "attempt no longer owns the execution",
        )
        client.report_error = report_error
        control_flow = DurableActivityControlFlow("unsupported durable wait")

        def fail(context: ActivityContext) -> None:
            raise control_flow

        with pytest.raises(DurableActivityControlFlow) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="request_authorization",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input={"provider": "atlas"},
                execute=fail,
                attempt=_attempt(),
            )

        assert raised.value is control_flow

    def test_stale_fence_on_report_wins_over_worker_failure(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        client.report_error = WorkflowClientError(WORKFLOW_ERROR_CODE_STALE_FENCE, "superseded")

        def explode(context: ActivityContext) -> None:
            raise ValueError("worker blew up")

        with pytest.raises(WorkflowClientError) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=explode,
                attempt=_attempt(),
            )
        assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE

    def test_requires_attempt_context(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        with pytest.raises(WorkflowClientError) as raised:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda context: None,
            )
        assert raised.value.code == WORKFLOW_ERROR_CODE_INVALID_ARGUMENT

    def test_uses_current_attempt_context_when_not_passed(self) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.client import ActivityReplay
        from agent_engine_runner_shared.workflow.context import advance_step_ordinal

        recorded = completed_outcome(_context(), "ok")
        client = _FakeClient(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            advance_step_ordinal(1)
            result = run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                semantic_input=None,
                execute=lambda context: None,
            )
        assert result == "ok"
        command = client.commands[0]
        assert command.attempt_id == "attempt-1"  # type: ignore[attr-defined]
        assert command.position.step_ordinal == 2  # type: ignore[attr-defined]


class TestDeniedAndExclusive:
    def test_live_denial_is_recorded_as_denied_outcome(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))

        def deny(context: ActivityContext) -> None:
            raise DurableActivityDeniedError("policy denied")

        with pytest.raises(DurableActivityDeniedError):
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=deny,
                attempt=_attempt(),
            )

        (outcome,) = client.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_DENIED
        assert outcome.error.code == WORKFLOW_ERROR_CODE_UNAUTHORIZED
        # Replaying the recorded outcome reproduces the denial semantics.
        with pytest.raises(DurableActivityDeniedError, match="policy denied"):
            unwrap_activity_outcome(outcome)

    def test_concurrent_activities_on_one_attempt_dispatch_in_parallel(self) -> None:
        import threading as _threading

        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        started_commands: list[Any] = []
        lock = _threading.Lock()

        class _RecordingClient(_FakeClient):
            def start_activity(self, command: Any) -> Any:  # type: ignore[override]
                with lock:
                    started_commands.append(command)
                return super().start_activity(command)

        client = _RecordingClient(ActivityDispatch(context=_context()))
        first_running = _threading.Event()
        second_started = _threading.Event()
        release_first = _threading.Event()

        def slow(context: ActivityContext) -> str:
            first_running.set()
            assert second_started.wait(timeout=5.0)
            release_first.wait(timeout=5.0)
            return "done"

        def run_second() -> None:
            first_running.wait(timeout=5.0)
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool-b",
                activity_ordinal=2,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda context: "second",
                attempt=_attempt(),
            )
            second_started.set()
            release_first.set()

        second = _threading.Thread(target=run_second)
        second.start()
        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="tool-a",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input=None,
            execute=slow,
            attempt=_attempt(),
        )
        second.join(timeout=5.0)

        assert result == "done"
        assert [command.position.activity_ordinal for command in started_commands] == [1, 2]

    def test_exclusive_activities_reject_concurrent_overlap(self) -> None:
        import threading as _threading

        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        first_running = _threading.Event()
        second_rejected = _threading.Event()
        release_first = _threading.Event()
        second_error: list[BaseException] = []

        def slow(context: ActivityContext) -> str:
            first_running.set()
            assert second_rejected.wait(timeout=5.0)
            release_first.wait(timeout=5.0)
            return "done"

        def run_second(*, exclusive: bool) -> None:
            first_running.wait(timeout=5.0)
            try:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name="tool-b",
                    activity_ordinal=2,
                    step_ordinal=1,
                    semantic_input=None,
                    execute=lambda context: "second",
                    attempt=_attempt(),
                    exclusive=exclusive,
                )
            except BaseException as error:  # noqa: BLE001 - captured for assertion
                second_error.append(error)
            finally:
                second_rejected.set()
                release_first.set()

        second = _threading.Thread(target=run_second, kwargs={"exclusive": True})
        second.start()
        result = run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="tool-a",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input=None,
            execute=slow,
            attempt=_attempt(),
            exclusive=True,
        )
        second.join(timeout=5.0)

        assert result == "done"
        assert len(second_error) == 1
        assert isinstance(second_error[0], WorkflowClientError)
        assert second_error[0].code == WORKFLOW_ERROR_CODE_CONFLICT
        assert len(client.commands) == 1

    def test_exclusive_conflicts_with_active_keyed_sibling_both_orderings(self) -> None:
        import threading as _threading

        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        def _overlap(*, first_exclusive: bool, second_exclusive: bool) -> WorkflowClientError:
            client = _FakeClient(ActivityDispatch(context=_context()))
            first_running = _threading.Event()
            second_done = _threading.Event()
            release_first = _threading.Event()
            second_error: list[BaseException] = []

            def slow(context: ActivityContext) -> str:
                first_running.set()
                assert second_done.wait(timeout=5.0)
                release_first.wait(timeout=5.0)
                return "done"

            def run_second() -> None:
                first_running.wait(timeout=5.0)
                try:
                    run_serial_activity(
                        client=client,  # type: ignore[arg-type]
                        kind=ACTIVITY_KIND_TOOL,
                        name="tool-b",
                        activity_ordinal=2,
                        step_ordinal=1,
                        semantic_input=None,
                        execute=lambda context: "second",
                        attempt=_attempt(),
                        exclusive=second_exclusive,
                    )
                except BaseException as error:  # noqa: BLE001 - captured for assertion
                    second_error.append(error)
                finally:
                    second_done.set()
                    release_first.set()

            second = _threading.Thread(target=run_second)
            second.start()
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool-a",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=slow,
                attempt=_attempt(),
                exclusive=first_exclusive,
            )
            second.join(timeout=5.0)
            assert len(second_error) == 1
            assert isinstance(second_error[0], WorkflowClientError)
            return second_error[0]

        exclusive_then_keyed = _overlap(first_exclusive=True, second_exclusive=False)
        keyed_then_exclusive = _overlap(first_exclusive=False, second_exclusive=True)
        assert exclusive_then_keyed.code == WORKFLOW_ERROR_CODE_CONFLICT
        assert keyed_then_exclusive.code == WORKFLOW_ERROR_CODE_CONFLICT


class TestCompletionPathFailures:
    def test_unserializable_result_reports_failed_before_raising(self) -> None:
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))

        with pytest.raises(TypeError):
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda context: object(),  # not JSON-serializable
                attempt=_attempt(),
            )

        # The dispatched activity still reached a terminal state.
        (outcome,) = client.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_FAILED

    def test_streaming_fold_failure_reports_failed_before_raising(self) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))

        def bad_fold(collected: list[str]) -> object:
            raise ValueError("fold blew up")

        with pytest.raises(ValueError, match="fold blew up"):
            list(
                run_streaming_activity(
                    kind=ACTIVITY_KIND_LLM,
                    name="gpt-test",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input=None,
                    execute=lambda: iter(["a"]),
                    replay=lambda result: iter([result]),
                    fold=bad_fold,
                    client=client,  # type: ignore[arg-type]
                    attempt=_attempt(),
                )
            )

        (outcome,) = client.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_FAILED


class TestRunStreamingActivity:
    def test_identified_stream_may_overlap_identified_sibling(self) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        stream = run_streaming_activity(
            kind=ACTIVITY_KIND_LLM,
            name="gpt-test",
            activity_ordinal=1,
            step_ordinal=1,
            semantic_input=None,
            execute=lambda: iter(["first", "second"]),
            replay=lambda result: iter([result]),
            fold=lambda collected: collected,
            client=client,  # type: ignore[arg-type]
            attempt=_attempt(),
            exclusive=False,
        )

        assert next(stream) == "first"
        assert (
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name="lookup",
                activity_ordinal=2,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda context: "found",
                attempt=_attempt(),
                exclusive=False,
            )
            == "found"
        )

        stream.close()
        assert len(client.commands) == 2

    def test_dispatch_runs_post_outcome_hook_after_completion_is_accepted(self) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        resolved: list[tuple[ActivityContext, object, int]] = []

        items = list(
            run_streaming_activity(
                kind=ACTIVITY_KIND_LLM,
                name="gpt-test",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda: iter(["answer"]),
                replay=lambda result: iter([result]),
                fold=lambda collected: "".join(collected),
                on_activity_resolved=lambda _client, context, value: resolved.append(
                    (context, value, len(client.outcomes))
                ),
                client=client,  # type: ignore[arg-type]
                attempt=_attempt(),
            )
        )

        assert items == ["answer"]
        (outcome,) = client.outcomes
        assert outcome == completed_outcome(_context(), "answer")
        assert resolved == [(_context(), "answer", 1)]

    def test_replay_runs_the_same_post_outcome_hook_without_execution(self) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = completed_outcome(_context(), "cached")
        client = _FakeClient(ActivityReplay(outcome=recorded))
        resolved: list[tuple[ActivityContext, object]] = []

        items = list(
            run_streaming_activity(
                kind=ACTIVITY_KIND_LLM,
                name="gpt-test",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda: pytest.fail("replay executed the LLM"),
                replay=lambda result: iter([result]),
                fold=lambda collected: collected,
                on_activity_resolved=lambda _client, context, value: resolved.append(
                    (context, value)
                ),
                client=client,  # type: ignore[arg-type]
                attempt=_attempt(),
            )
        )

        assert items == ["cached"]
        assert resolved == [(_context(), "cached")]
        assert client.outcomes == []

    def test_post_outcome_hook_is_not_run_when_completion_is_not_accepted(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = _FakeClient(ActivityDispatch(context=_context()))
        client.report_error = WorkflowClientError(
            WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
            "outcome response was lost",
        )
        resolved: list[bool] = []

        with pytest.raises(WorkflowClientError):
            list(
                run_streaming_activity(
                    kind=ACTIVITY_KIND_LLM,
                    name="gpt-test",
                    activity_ordinal=1,
                    step_ordinal=1,
                    semantic_input=None,
                    execute=lambda: iter(["answer"]),
                    replay=lambda result: iter([result]),
                    fold=lambda collected: "".join(collected),
                    on_activity_resolved=lambda _client, _context, _result: resolved.append(True),
                    client=client,  # type: ignore[arg-type]
                    attempt=_attempt(),
                )
            )

        assert resolved == []

    def test_owned_client_is_created_and_closed_per_activity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock

        from agent_engine_runner_shared.utils import get_request_timeout
        from agent_engine_runner_shared.workflow import activity as activity_module
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch

        client = MagicMock()
        client.start_activity.return_value = ActivityDispatch(context=_context())
        factory = MagicMock(return_value=client)
        monkeypatch.setattr(activity_module, "WorkflowClient", factory)

        items = list(
            run_streaming_activity(
                kind=ACTIVITY_KIND_LLM,
                name="gpt-test",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda: iter(["a", "b"]),
                replay=lambda result: iter([result]),
                fold=lambda collected: {"joined": "".join(collected)},
                oe_url="http://oe:8000",
                attempt=_attempt(),
            )
        )

        assert items == ["a", "b"]
        factory.assert_called_once_with("http://oe:8000", timeout=get_request_timeout())
        client.close.assert_called_once()
        (outcome,) = client.report_outcome.call_args.args
        assert unwrap_activity_outcome(outcome) == {"joined": "ab"}

    def test_injected_client_is_reused_and_not_closed(self) -> None:
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityReplay

        recorded = completed_outcome(_context(), {"cached": True})
        client = _FakeClient(ActivityReplay(outcome=recorded))

        items = list(
            run_streaming_activity(
                kind=ACTIVITY_KIND_TOOL,
                name="tool",
                activity_ordinal=1,
                step_ordinal=1,
                semantic_input=None,
                execute=lambda: iter([]),
                replay=lambda result: iter([result]),
                fold=lambda collected: collected,
                client=client,  # type: ignore[arg-type]
                attempt=_attempt(),
            )
        )

        # Replay translated the recorded outcome; nothing executed or reported.
        assert items == [{"cached": True}]
        assert client.outcomes == []

    def test_omitted_step_ordinal_uses_advanced_request_counter(self) -> None:
        from agent_engine_runner_shared.workflow import attempt_context_scope
        from agent_engine_runner_shared.workflow.activity import run_streaming_activity
        from agent_engine_runner_shared.workflow.client import ActivityDispatch
        from agent_engine_runner_shared.workflow.context import advance_step_ordinal

        client = _FakeClient(ActivityDispatch(context=_context()))

        with attempt_context_scope(_attempt()):
            advance_step_ordinal(1)
            items = list(
                run_streaming_activity(
                    kind=ACTIVITY_KIND_LLM,
                    name="gpt-test",
                    activity_ordinal=1,
                    semantic_input=None,
                    execute=lambda: iter(["a"]),
                    replay=lambda result: iter([result]),
                    fold=lambda collected: {"joined": "".join(collected)},
                    client=client,  # type: ignore[arg-type]
                )
            )

        assert items == ["a"]
        command = client.commands[0]
        assert command.position.step_ordinal == 2  # type: ignore[attr-defined]


def test_activity_command_protojson_matches_wire_expectations() -> None:
    from agent_engine_runner_shared.workflow.protojson import encode_protojson

    command = build_activity_command(
        attempt=_attempt(),
        kind=ACTIVITY_KIND_LLM,
        name="gpt-test",
        activity_ordinal=1,
        step_ordinal=1,
        semantic_input={"prompt": "hi"},
    )

    assert json.loads(encode_protojson(command)) == {
        "workflow_identity": {
            "session_id": "session-1",
            "execution_id": "execution-1",
        },
        "attempt_id": "attempt-1",
        "fencing_token": "7",
        "position": {
            "operation_path": {"segments": [{"name": "agent", "ordinal": "1"}]},
            "activity_ordinal": "1",
            "step_ordinal": "1",
        },
        "activity_kind": "ACTIVITY_KIND_LLM",
        "activity_name": "gpt-test",
        "semantic_input": {"prompt": "hi"},
    }
