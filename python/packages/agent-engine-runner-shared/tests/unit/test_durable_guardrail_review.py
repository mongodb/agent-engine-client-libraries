"""Guardrail review of a model call on a durable session.

OE halts the call and names a review. The halt is the call's recorded result,
the pause happens after it, and a second call that names the review gets the
decided outcome from OE.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from agent_engine_sdk.models import Message

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_DENIED,
    ACTIVITY_OUTCOME_KIND_FAILED,
    ActivityContext,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.hooks import register_suspend_handler
from agent_engine_runner_shared.models import ToolExecuteResponse
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import (
    GUARDRAIL_REVIEW_INVALID_CODE,
    LLMInvocationError,
    OperationalStepAllocator,
    PolicyDeniedException,
    _request_oe_approval,
)
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import unwrap_activity_outcome
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay

REVIEW_ID = "review-1"
MESSAGES = [Message(role="user", content="summarize the claim")]
HELD = {"content": "the held answer"}


def _attempt(attempt_id: str) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
            tenant_scope=TenantScope(
                org_id="org-1", project_id="project-1", workspace_id="workspace-1"
            ),
        ),
    )


class _History:
    """A workflow history shared by attempts: a recorded position replays."""

    def __init__(self) -> None:
        self.recorded: dict[int, Any] = {}
        self.memory_commands: list[Any] = []
        self._dispatched: dict[str, int] = {}

    def start_activity(self, command: Any) -> object:
        ordinal = command.position.activity_ordinal
        if ordinal in self.recorded:
            return ActivityReplay(outcome=self.recorded[ordinal])
        activity_id = f"activity-{ordinal}"
        self._dispatched[activity_id] = ordinal
        return ActivityDispatch(
            context=ActivityContext(
                workflow_identity=command.workflow_identity,
                activity_id=activity_id,
                attempt_id=command.attempt_id,
                fencing_token=command.fencing_token,
            )
        )

    def report_outcome(self, outcome: Any) -> None:
        self.recorded[self._dispatched[outcome.activity_id]] = outcome

    def ensure_memory_written(self, command: Any) -> None:
        self.memory_commands.append(command)

    def kinds(self) -> list[int]:
        return [self.recorded[ordinal].outcome_kind for ordinal in sorted(self.recorded)]


class _Paused(Exception):
    """What a framework interrupt does the first time: it leaves the node."""

    def __init__(self, value: Any) -> None:
        super().__init__("paused")
        self.value = value


def _halt_response(**overrides: Any) -> ToolExecuteResponse:
    fields: dict[str, Any] = {
        "proceed": False,
        "status": "require_review",
        "reason": "output needs human review",
        "latest_step_number": 1,
        "guardrail_review": {
            "review_id": REVIEW_ID,
            "allowed_decisions": ["approve", "deny"],
            "reason": "output needs human review",
            "guardrails": [{"id": "policy-1", "name": "pii", "category": "output_validation"}],
        },
    }
    fields.update(overrides)
    return ToolExecuteResponse(**fields)


class _Run:
    """One attempt of the node: a fresh proxy over the shared history and OE."""

    def __init__(self, history: _History, oe: MagicMock, memory: MagicMock) -> None:
        self.history = history
        self.oe = oe
        self.memory = memory

    def stream(
        self,
        attempt_id: str,
        *,
        protocol: bool = True,
        steps_before: int = 0,
        steps: OperationalStepAllocator | None = None,
    ) -> list[Any]:
        if steps is None:
            steps = OperationalStepAllocator()
            steps.observe_at_least(steps_before)
        proxy = SecureLLMProxy(
            "http://oe:8000",
            "execution-1",
            model_name="gpt-test",
            operational_steps=steps,
            durable_memory=self.memory,
            guardrail_review_protocol=protocol,
        )
        proxy._workflow = self.history  # type: ignore[assignment]
        with attempt_context_scope(_attempt(attempt_id)):
            with patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval", self.oe):
                return list(proxy.stream(MESSAGES, activity_key="llm:task-1:1"))


@pytest.fixture(autouse=True)
def _reset_suspend_handler() -> Iterator[None]:
    register_suspend_handler(None)
    yield
    register_suspend_handler(None)


@pytest.fixture
def run() -> _Run:
    return _Run(_History(), MagicMock(), MagicMock())


def _pause_on_review() -> list[Any]:
    """Register a handler that pauses, and return the values it paused with."""
    paused: list[Any] = []

    def handler(value: Any) -> Any:
        paused.append(value)
        raise _Paused(value)

    register_suspend_handler(handler)
    return paused


def _answer_review(decision: str, review_id: str = REVIEW_ID) -> None:
    register_suspend_handler(
        lambda value: {"guardrail_review": {"review_id": review_id, "decision": decision}}
    )


def _halt(run: _Run) -> None:
    """Attempt 1: the call is halted for review and the node pauses."""
    run.oe.return_value = _halt_response()
    _pause_on_review()
    with pytest.raises(_Paused):
        run.stream("attempt-1")
    run.oe.reset_mock()


def test_halt_is_the_calls_recorded_result_and_the_pause_follows_it(run: _Run) -> None:
    run.oe.return_value = _halt_response()
    paused = _pause_on_review()

    with pytest.raises(_Paused):
        run.stream("attempt-1")

    # The halted call opted in to the protocol and named no review.
    (request,) = run.oe.call_args_list
    assert request.kwargs["review_protocol"] == 1
    assert "review_id" not in request.kwargs
    # The activity completed with the halt; the pause names only the review.
    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_COMPLETED]
    recorded = unwrap_activity_outcome(run.history.recorded[1])
    assert recorded["guardrail_review_halt"]["review_id"] == REVIEW_ID
    assert recorded["guardrail_review_halt_step"] == 1
    assert paused == [{"guardrail_review": {"review_id": REVIEW_ID}}]
    # A halt is not a model response: its activity is closed in memory with
    # nothing written.
    run.memory.synchronize_llm.assert_not_called()
    run.memory.acknowledge.assert_called_once()


def test_approve_resolves_with_a_second_call_that_names_the_review(run: _Run) -> None:
    _halt(run)
    run.oe.return_value = ToolExecuteResponse(
        proceed=True, status="success", result=HELD, latest_step_number=2
    )
    _answer_review("approve")

    chunks = run.stream("attempt-2")

    assert "".join(chunk.content or "" for chunk in chunks) == "the held answer"
    # The halted call replayed from history; only the resolving call reached OE,
    # at a step after the halted one.
    (request,) = run.oe.call_args_list
    assert request.kwargs["review_protocol"] == 1
    assert request.kwargs["review_id"] == REVIEW_ID
    assert request.kwargs["step"] == 2
    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_COMPLETED, ACTIVITY_OUTCOME_KIND_COMPLETED]
    assert unwrap_activity_outcome(run.history.recorded[2])["content"] == "the held answer"
    # The released response is the turn's model response.
    run.memory.synchronize_llm.assert_called_once()
    assert run.memory.synchronize_llm.call_args.args[2]["content"] == "the held answer"


def test_resolving_call_takes_a_step_after_the_halted_one_when_the_attempt_counts_fewer(
    run: _Run,
) -> None:
    # The attempt that halted re-ran an earlier node to rebuild its interrupt,
    # so it counted four steps before this call. The attempt that answers the
    # review replays that interrupt without the re-run and counts two, then
    # replays this halted call. OE resolves a review only at a later step.
    run.oe.return_value = _halt_response()
    _pause_on_review()
    with pytest.raises(_Paused):
        run.stream("attempt-1", steps_before=4)
    assert unwrap_activity_outcome(run.history.recorded[1])["guardrail_review_halt_step"] == 5
    run.oe.reset_mock()
    run.oe.return_value = ToolExecuteResponse(proceed=True, status="success", result=HELD)
    _answer_review("approve")

    chunks = run.stream("attempt-2", steps_before=2)

    assert "".join(chunk.content or "" for chunk in chunks) == "the held answer"
    (request,) = run.oe.call_args_list
    assert request.kwargs["review_id"] == REVIEW_ID
    assert request.kwargs["step"] == 6


def test_parallel_halted_calls_resolve_at_steps_neither_of_them_halted_at() -> None:
    # Two branches of one superstep share the step counter and each halts.
    first, second = (_Run(_History(), MagicMock(), MagicMock()) for _ in range(2))
    halting = OperationalStepAllocator()
    halting.observe_at_least(4)
    _pause_on_review()
    for branch, review_id in ((first, "review-a"), (second, "review-b")):
        branch.oe.return_value = _halt_response(guardrail_review={"review_id": review_id})
        with pytest.raises(_Paused):
            branch.stream("attempt-1", steps=halting)
        branch.oe.reset_mock()
        branch.oe.return_value = ToolExecuteResponse(proceed=True, status="success", result=HELD)

    # The answering attempt counted fewer steps. Both branches replay their
    # halt and pause before the framework re-runs them with the answers.
    answering = OperationalStepAllocator()
    answering.observe_at_least(2)
    _pause_on_review()
    for branch in (first, second):
        with pytest.raises(_Paused):
            branch.stream("attempt-2", steps=answering)
    register_suspend_handler(
        lambda value: {"guardrail_review": {**value["guardrail_review"], "decision": "approve"}}
    )
    first.stream("attempt-2", steps=answering)
    second.stream("attempt-2", steps=answering)

    resolved = [branch.oe.call_args.kwargs["step"] for branch in (first, second)]
    assert len(set(resolved)) == 2
    assert min(resolved) > 6  # the branches halted at steps 5 and 6


def test_a_halt_with_no_suspend_handler_is_an_error_not_a_denial(run: _Run) -> None:
    run.oe.return_value = _halt_response()

    with pytest.raises(LLMInvocationError, match="no suspend handler registered"):
        run.stream("attempt-1")


def test_deny_ends_the_call_as_a_policy_denial(run: _Run) -> None:
    _halt(run)
    run.oe.return_value = ToolExecuteResponse(
        proceed=False,
        status="blocked",
        reason="guardrail review denied by reviewer",
        guardrail_id="policy-1",
        guardrail_category="output_validation",
    )
    _answer_review("deny")

    with pytest.raises(
        PolicyDeniedException, match="guardrail review denied by reviewer"
    ) as denied:
        run.stream("attempt-2")

    assert denied.value.guardrail_meta is not None
    assert denied.value.guardrail_meta.guardrail_id == "policy-1"
    assert run.oe.call_args.kwargs["review_id"] == REVIEW_ID
    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_COMPLETED, ACTIVITY_OUTCOME_KIND_DENIED]
    run.memory.synchronize_llm.assert_not_called()


def test_a_resolved_review_replays_without_reaching_oe(run: _Run) -> None:
    _halt(run)
    run.oe.return_value = ToolExecuteResponse(proceed=True, status="success", result=HELD)
    _answer_review("approve")
    run.stream("attempt-2")
    run.oe.reset_mock()

    chunks = run.stream("attempt-3")

    assert "".join(chunk.content or "" for chunk in chunks) == "the held answer"
    run.oe.assert_not_called()


def test_an_answer_for_another_review_resolves_nothing(run: _Run) -> None:
    _halt(run)
    _answer_review("approve", review_id="review-2")

    with pytest.raises(LLMInvocationError, match="different review"):
        run.stream("attempt-2")

    run.oe.assert_not_called()
    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_COMPLETED]


@pytest.mark.parametrize("answer", ["approve", None, {"decision": "approve"}])
def test_an_answer_that_names_no_review_resolves_nothing(run: _Run, answer: Any) -> None:
    _halt(run)
    register_suspend_handler(lambda value: answer)

    with pytest.raises(LLMInvocationError):
        run.stream("attempt-2")

    run.oe.assert_not_called()


def test_a_halt_that_names_no_review_fails_closed(run: _Run) -> None:
    run.oe.return_value = _halt_response(guardrail_review=None)
    paused = _pause_on_review()

    with pytest.raises(LLMInvocationError, match="named no review"):
        run.stream("attempt-1")

    assert paused == []
    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_FAILED]


def test_a_halt_in_place_of_a_resolution_fails_closed(run: _Run) -> None:
    _halt(run)
    run.oe.return_value = _halt_response()
    _answer_review("approve")

    with pytest.raises(LLMInvocationError, match="named no review"):
        run.stream("attempt-2")

    assert run.history.kinds() == [ACTIVITY_OUTCOME_KIND_COMPLETED, ACTIVITY_OUTCOME_KIND_FAILED]
    run.memory.synchronize_llm.assert_not_called()


def test_without_the_adapter_opt_in_the_call_does_not_use_the_protocol(run: _Run) -> None:
    run.oe.return_value = ToolExecuteResponse(proceed=True, status="success", result=HELD)

    run.stream("attempt-1", protocol=False)

    assert "review_protocol" not in run.oe.call_args.kwargs


def test_outside_a_durable_attempt_the_call_does_not_use_the_protocol() -> None:
    oe = MagicMock(return_value=ToolExecuteResponse(proceed=True, status="success", result=HELD))
    proxy = SecureLLMProxy(
        "http://oe:8000", "execution-1", model_name="gpt-test", guardrail_review_protocol=True
    )

    with patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval", oe):
        list(proxy.stream(MESSAGES))

    assert "review_protocol" not in oe.call_args.kwargs


class TestReviewFieldsOnTheWire:
    def _post(self, response: httpx.Response, **review: Any) -> tuple[Any, MagicMock]:
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value = response
        with patch(
            "agent_engine_runner_shared.secure_wrapper.create_httpx_client_with_tls",
            return_value=client,
        ):
            result = _request_oe_approval(
                "http://oe:8000", "execution-1", "invoke_llm", {"messages": []}, 3, **review
            )
        return result, client

    @staticmethod
    def _response(status: int, body: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            status, json=body, request=httpx.Request("POST", "http://oe:8000/tool/execute")
        )

    def test_review_fields_are_sent_only_when_set(self) -> None:
        ok = self._response(200, {"proceed": True})

        _, client = self._post(ok)
        assert "review_protocol" not in client.post.call_args.kwargs["json"]
        assert "review_id" not in client.post.call_args.kwargs["json"]

        _, client = self._post(ok, review_protocol=1, review_id=REVIEW_ID)
        assert client.post.call_args.kwargs["json"]["review_protocol"] == 1
        assert client.post.call_args.kwargs["json"]["review_id"] == REVIEW_ID

    def test_a_halt_response_carries_the_review(self) -> None:
        result, _ = self._post(self._response(200, _halt_response().model_dump(mode="json")))

        assert result.guardrail_review is not None
        assert result.guardrail_review.review_id == REVIEW_ID
        assert result.guardrail_review.guardrails[0].id == "policy-1"

    def test_a_refused_resolution_is_a_failure_not_an_unreachable_oe(self) -> None:
        refused = self._response(
            409,
            {
                "error": "guardrail review cannot resolve this call",
                "code": "GUARDRAIL_REVIEW_INVALID",
            },
        )

        with pytest.raises(LLMInvocationError) as error:
            self._post(refused, review_protocol=1, review_id=REVIEW_ID)

        assert error.value.error_code == GUARDRAIL_REVIEW_INVALID_CODE

    def test_another_conflict_still_blocks_as_before(self) -> None:
        with pytest.raises(PolicyDeniedException, match="OE unreachable"):
            self._post(self._response(409, {"error": "something else"}))


class TestConversationMemory:
    """What reaches durable memory, with the real memory state and its pending input."""

    @staticmethod
    def _run() -> _Run:
        from agent_engine_runner_shared.workflow.memory import DurableMemoryState

        return _Run(_History(), MagicMock(), DurableMemoryState("summarize the claim"))  # type: ignore[arg-type]

    @staticmethod
    def _writes(run: _Run) -> list[list[tuple[str, str]]]:
        import json

        decoded = [
            [json.loads(write.payload_json) for write in command.memory_writes]
            for command in run.history.memory_commands
        ]
        return [[(turn["role"], turn["content"]) for turn in batch] for batch in decoded]

    @pytest.fixture(autouse=True)
    def _user(self) -> Iterator[None]:
        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.get_current_user_id",
            return_value="user-1",
        ):
            yield

    def test_the_halt_is_closed_with_nothing_written_and_the_input_stays_pending(self) -> None:
        run = self._run()
        _halt(run)

        # Every completed activity needs an acknowledged batch before its step
        # can commit; the halt's batch is empty.
        assert self._writes(run) == [[]]

    def test_the_released_response_is_written_with_the_users_input(self) -> None:
        run = self._run()
        _halt(run)
        run.oe.return_value = ToolExecuteResponse(proceed=True, status="success", result=HELD)
        _answer_review("approve")

        run.stream("attempt-2")

        halt_live, halt_replayed, released = self._writes(run)
        assert halt_live == [] and halt_replayed == []
        assert released == [("user", "summarize the claim"), ("assistant", "the held answer")]

    def test_a_denied_response_is_never_written(self) -> None:
        run = self._run()
        _halt(run)
        run.oe.return_value = ToolExecuteResponse(
            proceed=False, status="blocked", reason="guardrail review denied by reviewer"
        )
        _answer_review("deny")

        with pytest.raises(PolicyDeniedException):
            run.stream("attempt-2")

        assert all(writes == [] for writes in self._writes(run))


def test_a_pause_is_a_review_wait_only_when_this_attempt_recorded_its_halt(run: _Run) -> None:
    from agent_engine_runner_shared.secure_llm_proxy import guardrail_review_wait_id
    from agent_engine_runner_shared.workflow.context import note_guardrail_review_wait

    value = {"guardrail_review": {"review_id": REVIEW_ID}}

    # Outside an attempt, and for a value an application wrote, it is not one.
    assert guardrail_review_wait_id(value) is None
    with attempt_context_scope(_attempt("attempt-1")):
        assert guardrail_review_wait_id(value) is None
        note_guardrail_review_wait(REVIEW_ID)
        assert guardrail_review_wait_id(value) == REVIEW_ID
        assert guardrail_review_wait_id({"guardrail_review": {"review_id": "another"}}) is None
        assert guardrail_review_wait_id("approve") is None
    # A later attempt starts with nothing recorded.
    with attempt_context_scope(_attempt("attempt-2")):
        assert guardrail_review_wait_id(value) is None
