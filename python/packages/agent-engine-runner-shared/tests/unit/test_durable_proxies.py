"""Durable-session behavior of the secure LLM and tool proxies."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_LLM,
    ACTIVITY_KIND_TOOL,
    ActivityContext,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException, SecureToolWrapper
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import (
    DurableActivityDeniedError,
    completed_outcome,
)
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay


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


def _activity_context() -> ActivityContext:
    return ActivityContext(
        workflow_identity=WorkflowIdentity(session_id="session-1", execution_id="execution-1"),
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
    )


class _FakeWorkflow:
    def __init__(self, started: object) -> None:
        self.started = started
        self.commands: list[Any] = []
        self.outcomes: list[Any] = []

    def start_activity(self, command: Any) -> object:
        self.commands.append(command)
        if isinstance(self.started, Exception):
            raise self.started
        return self.started

    def report_outcome(self, outcome: Any) -> None:
        self.outcomes.append(outcome)


class TestDurableToolWrapper:
    def test_native_sessions_do_not_touch_the_workflow_client(self) -> None:
        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        with patch.object(wrapper, "_execute_tool_native", return_value="ok") as native:
            assert wrapper.execute_tool("lookup", {"q": 1}) == "ok"
        native.assert_called_once()
        assert wrapper._workflow is None

    def test_durable_replay_skips_native_execution(self) -> None:
        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        recorded = completed_outcome(_activity_context(), {"cached": True})
        wrapper._workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            with patch.object(wrapper, "_execute_tool_native") as native:
                result = wrapper.execute_tool("lookup", {"q": 1}, tool_call_id="tc-1")

        native.assert_not_called()
        assert result == {"cached": True}
        (command,) = wrapper._workflow.commands
        assert command.activity_kind == ACTIVITY_KIND_TOOL
        assert command.activity_name == "lookup"
        assert command.fencing_token == 7

    def test_durable_dispatch_runs_native_and_reports(self) -> None:
        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        wrapper._workflow = _FakeWorkflow(ActivityDispatch(context=_activity_context()))

        with attempt_context_scope(_attempt()):
            with patch.object(
                wrapper, "_execute_tool_native", return_value={"fresh": True}
            ) as native:
                result = wrapper.execute_tool("lookup", {"q": 1})

        native.assert_called_once()
        assert result == {"fresh": True}
        (outcome,) = wrapper._workflow.outcomes
        assert outcome.activity_id == "activity-1"

    def test_durable_denial_maps_to_policy_denied(self) -> None:
        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        wrapper._workflow = _FakeWorkflow(DurableActivityDeniedError("blocked by policy"))

        with attempt_context_scope(_attempt()):
            with pytest.raises(PolicyDeniedException, match="blocked by policy"):
                wrapper.execute_tool("lookup", {"q": 1})

    def test_dispatch_then_native_denial_records_denied_and_keeps_live_semantics(
        self,
    ) -> None:
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_DENIED,
        )
        from agent_engine_runner_shared.models import GuardrailMeta

        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        wrapper._workflow = _FakeWorkflow(ActivityDispatch(context=_activity_context()))
        denial = PolicyDeniedException(
            "input blocked",
            guardrail_meta=GuardrailMeta(guardrail_id="gr-1", guardrail_category="pii"),
        )

        with attempt_context_scope(_attempt()):
            with patch.object(wrapper, "_execute_tool_native", side_effect=denial):
                with pytest.raises(PolicyDeniedException) as raised:
                    wrapper.execute_tool("lookup", {"q": 1})

        # The live call preserves the original denial (guardrail identity intact)…
        assert raised.value is denial
        assert raised.value.guardrail_meta is not None
        # …and the recorded outcome is DENIED, so replay reproduces a denial.
        (outcome,) = wrapper._workflow.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_DENIED

    def test_platform_execution_configuration_is_not_part_of_activity_identity(
        self,
    ) -> None:
        from agent_engine_runner_shared.workflow.activity import value_to_json

        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        recorded = completed_outcome(_activity_context(), {"cached": True})
        wrapper._workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            result = wrapper.execute_tool(
                "lookup",
                {"q": 1},
                metadata={"trace": "platform-value"},
                provider_type="github",
                scopes=["repo", "issues"],
                tool_call_id="call-1",
                is_local=True,
                local_executor=lambda: {"must": "not run"},
            )

        assert result == {"cached": True}
        (command,) = wrapper._workflow.commands
        assert value_to_json(command.semantic_input) == {"arguments": {"q": 1}}

    def test_preallocated_tool_call_ids_keep_stable_ordinals(self) -> None:
        from agent_engine_runner_shared.workflow.context import (
            preallocate_activity_ordinals,
            tool_activity_key,
        )

        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        recorded = completed_outcome(_activity_context(), {"ok": True})
        wrapper._workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            preallocate_activity_ordinals(
                [tool_activity_key("call-b"), tool_activity_key("call-a")]
            )
            # Inverted start order still uses the preallocated ordinals.
            wrapper.execute_tool("search_a", {}, tool_call_id="call-a")
            wrapper.execute_tool("search_b", {}, tool_call_id="call-b")

        ordinals = [command.position.activity_ordinal for command in wrapper._workflow.commands]
        assert ordinals == [2, 1]


class TestDurableLLMProxy:
    def _proxy(self, started: object) -> tuple[SecureLLMProxy, _FakeWorkflow]:
        proxy = SecureLLMProxy("http://oe:8000", "execution-1", model_name="gpt-test")
        fake = _FakeWorkflow(started)
        proxy._workflow = fake  # type: ignore[assignment]
        return proxy, fake

    def test_native_stream_bypasses_workflow(self) -> None:
        proxy = SecureLLMProxy("http://oe:8000", "execution-1", model_name="gpt-test")
        with patch.object(proxy, "_stream_llm", return_value=iter([])) as native:
            list(proxy.stream([]))
        native.assert_called_once()
        assert proxy._workflow is None

    def test_durable_llm_admission_uses_operational_step_identity(self) -> None:
        proxy = SecureLLMProxy("http://oe:8000", "execution-1", model_name="gpt-test")

        with (
            attempt_context_scope(_attempt()),
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.run_streaming_activity",
                return_value=iter([]),
            ) as run_activity,
        ):
            list(proxy.stream([], step=1))

        assert run_activity.call_args.kwargs["exclusive"] is False

    def test_stable_non_root_llm_admission_is_non_exclusive(self) -> None:
        from agent_engine_runner_shared.workflow import (
            ChildOperationBoundary,
            child_operation_boundary_scope,
        )

        proxy = SecureLLMProxy(
            "http://oe:8000",
            "execution-1",
            model_name="gpt-test",
            stable_operation_identity=True,
        )

        with (
            attempt_context_scope(_attempt()),
            child_operation_boundary_scope(
                ChildOperationBoundary(name="left", occurrence_key="parallel.left")
            ),
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.run_streaming_activity",
                return_value=iter([]),
            ) as run_activity,
        ):
            list(proxy.stream([], step=1))

        assert run_activity.call_args.kwargs["exclusive"] is False

    def test_stable_identity_hint_is_allowed_at_the_root(self) -> None:
        proxy = SecureLLMProxy(
            "http://oe:8000",
            "execution-1",
            model_name="gpt-test",
            stable_operation_identity=True,
        )

        with (
            attempt_context_scope(_attempt()),
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.run_streaming_activity",
                return_value=iter([]),
            ) as run_activity,
        ):
            list(proxy.stream([], step=1))

        assert run_activity.call_args.kwargs["exclusive"] is False

    def test_durable_replay_yields_recorded_response_without_model_call(self) -> None:
        recorded = completed_outcome(
            _activity_context(),
            {
                "content": "cached answer",
                "tool_calls": None,
                "id": "message-1",
                "name": "assistant",
                "response_metadata": {"finish_reason": "stop"},
                "additional_kwargs": {"provider": "recorded"},
            },
        )
        proxy, fake = self._proxy(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm") as native:
                chunks = list(proxy.stream([], step=3))

        native.assert_not_called()
        assert proxy.last_from_cache is True
        assert chunks[0].content == "cached answer"
        assert chunks[0].id == "message-1"
        assert chunks[0].name == "assistant"
        assert chunks[0].response_metadata == {"finish_reason": "stop"}
        assert chunks[0].additional_kwargs == {"provider": "recorded"}
        (command,) = fake.commands
        assert command.activity_kind == ACTIVITY_KIND_LLM
        assert command.position.activity_ordinal == 1

    def test_durable_dispatch_streams_native_and_reports_folded_response(self) -> None:
        from agent_engine_sdk.models import LLMStreamChunk

        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))
        native_chunks = [
            LLMStreamChunk(content="hel"),
            LLMStreamChunk(content="lo"),
        ]

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", return_value=iter(native_chunks)):
                streamed = list(proxy.stream([], step=1))

        assert [chunk.content for chunk in streamed] == ["hel", "lo"]
        (outcome,) = fake.outcomes
        from agent_engine_runner_shared.workflow.activity import unwrap_activity_outcome

        recorded = unwrap_activity_outcome(outcome)
        assert recorded["content"] == "hello"

    def test_durable_dispatch_failure_reports_failed_outcome(self) -> None:
        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", side_effect=RuntimeError("model unavailable")):
                with pytest.raises(RuntimeError, match="model unavailable"):
                    list(proxy.stream([], step=1))

        (outcome,) = fake.outcomes
        assert outcome.error.message == "model unavailable"

    def test_dispatch_then_native_denial_records_denied_outcome(self) -> None:
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_DENIED,
        )

        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))
        denial = PolicyDeniedException("model blocked by policy")

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", side_effect=denial):
                with pytest.raises(PolicyDeniedException) as raised:
                    list(proxy.stream([], step=1))

        assert raised.value is denial
        (outcome,) = fake.outcomes
        assert outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_DENIED

    def test_durable_denial_maps_to_policy_denied(self) -> None:
        proxy, _ = self._proxy(DurableActivityDeniedError("model blocked"))

        with attempt_context_scope(_attempt()):
            with pytest.raises(PolicyDeniedException, match="model blocked"):
                list(proxy.stream([], step=1))

    def test_abandoned_durable_stream_reports_terminal_outcome(self) -> None:
        from agent_engine_sdk.models import LLMStreamChunk

        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))
        native_chunks = [LLMStreamChunk(content="a"), LLMStreamChunk(content="b")]

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", return_value=iter(native_chunks)):
                stream = proxy.stream([], step=1)
                next(stream)
                stream.close()  # consumer abandons the stream mid-way

        # The dispatched activity still reaches a terminal outcome instead of
        # dangling until OE lease expiry.
        (outcome,) = fake.outcomes
        assert "abandoned" in outcome.error.message

    def test_durable_dispatch_folds_tool_call_fragments_before_recording(self) -> None:
        from agent_engine_sdk.models import LLMStreamChunk, ToolCallChunk

        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))
        # A single streamed tool call arriving as two partial-args fragments.
        native_chunks = [
            LLMStreamChunk(
                tool_calls=[ToolCallChunk(id="tc-1", name="lookup", args='{"q": ', index=0)]
            ),
            LLMStreamChunk(tool_calls=[ToolCallChunk(args='"x"}', index=0)]),
        ]

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", return_value=iter(native_chunks)):
                list(proxy.stream([], step=1))

        from agent_engine_runner_shared.workflow.activity import unwrap_activity_outcome

        recorded = unwrap_activity_outcome(fake.outcomes[0])
        (tool_call,) = recorded["tool_calls"]
        # Fragments are merged the same way live consumers fold them.
        assert tool_call["name"] == "lookup"
        assert tool_call["args"] == {"q": "x"}

    def test_live_llm_fold_preallocates_tool_ordinals_in_list_order(self) -> None:
        """Live fold must pin sibling ToolCall ordinals before inverted fan-out."""
        from agent_engine_sdk.models import LLMStreamChunk, ToolCallChunk

        proxy, llm_fake = self._proxy(ActivityDispatch(context=_activity_context()))
        native_chunks = [
            LLMStreamChunk(
                tool_calls=[
                    ToolCallChunk(id="call-a", name="search_a", args="{}", index=0),
                    ToolCallChunk(id="call-b", name="search_b", args="{}", index=1),
                ]
            )
        ]
        wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
        recorded = completed_outcome(_activity_context(), {"ok": True})
        wrapper._workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", return_value=iter(native_chunks)):
                list(proxy.stream([], step=1))
            # Inverted worker start order must still use LLM list-order ordinals.
            wrapper.execute_tool("search_b", {}, tool_call_id="call-b")
            wrapper.execute_tool("search_a", {}, tool_call_id="call-a")

        assert llm_fake.commands[0].position.activity_ordinal == 1
        tool_ordinals = [
            command.position.activity_ordinal for command in wrapper._workflow.commands
        ]
        assert tool_ordinals == [3, 2]

    def test_stale_fence_on_llm_report_supersedes_worker_failure(self) -> None:
        from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
            WORKFLOW_ERROR_CODE_STALE_FENCE,
        )
        from agent_engine_runner_shared.workflow import WorkflowClientError

        proxy, fake = self._proxy(ActivityDispatch(context=_activity_context()))

        def _raise_stale(outcome: Any) -> None:
            raise WorkflowClientError(WORKFLOW_ERROR_CODE_STALE_FENCE, "superseded")

        fake.report_outcome = _raise_stale  # type: ignore[method-assign]

        with attempt_context_scope(_attempt()):
            with patch.object(proxy, "_stream_llm", side_effect=RuntimeError("model unavailable")):
                with pytest.raises(WorkflowClientError) as raised:
                    list(proxy.stream([], step=1))

        assert raised.value.code == WORKFLOW_ERROR_CODE_STALE_FENCE
