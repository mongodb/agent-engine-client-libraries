"""Tests for the request-scoped workflow attempt context."""

from __future__ import annotations

import asyncio

import pytest

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    ActivityPosition,
    OperationPath,
    OperationPathSegment,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.workflow import attempt_context_scope, current_attempt_context
from agent_engine_runner_shared.workflow.context import (
    ChildOperationBoundary,
    UnsupportedChildOperationFanOutError,
    advance_step_ordinal,
    child_operation_boundary_scope,
    current_operation_path,
    current_step_ordinal,
    observed_activity_positions,
    operation_path_resolver_scope,
    preallocate_child_operation_ordinals,
    record_observed_activity,
    record_reconstructed_activity_interrupt,
    reset_attempt_context,
    set_activity_reconstruction_ids,
    set_attempt_context,
)


def _attempt(attempt_id: str) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=1,
        owner_id="aer-1",
    )


def _position(ordinal: int, *, step: int = 1) -> ActivityPosition:
    return ActivityPosition(
        step_ordinal=step,
        operation_path=OperationPath(segments=[OperationPathSegment(name="agent", ordinal=1)]),
        activity_ordinal=ordinal,
    )


def test_observed_activity_positions_are_unique_and_reported_in_order_by_step() -> None:
    with attempt_context_scope(_attempt("attempt-replay")):
        record_observed_activity(_position(2))
        record_observed_activity(_position(1, step=2))
        record_observed_activity(_position(1))
        record_observed_activity(_position(1))

        assert observed_activity_positions(1) == [_position(1), _position(2)]
        assert observed_activity_positions(2) == [_position(1, step=2)]


def test_reconstructed_interrupt_count_requires_the_selected_activity() -> None:
    with attempt_context_scope(_attempt("attempt-replay")):
        set_activity_reconstruction_ids(["activity-1"])

        assert record_reconstructed_activity_interrupt("activity-1") == 1
        assert record_reconstructed_activity_interrupt("activity-1") == 2
        with pytest.raises(RuntimeError, match='activity "activity-2" is not being reconstructed'):
            record_reconstructed_activity_interrupt("activity-2")


def _branch_attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-branch",
        fencing_token=1,
        owner_id="aer-1",
        branch_lineage=BranchLineage(
            source_workflow_identity=WorkflowIdentity(
                session_id="source-session",
                execution_id="source-execution",
            ),
            source_step_ordinal=4,
            source_state_hash="sha256:source",
        ),
    )


def test_no_attempt_context_by_default() -> None:
    assert current_attempt_context() is None


def test_step_ordinal_requires_active_attempt_context() -> None:
    with pytest.raises(RuntimeError, match="active durable attempt context"):
        current_step_ordinal()


def _operation_segments() -> list[tuple[str, int]]:
    return [(segment.name, segment.ordinal) for segment in current_operation_path().segments]


def test_operation_path_defaults_to_root() -> None:
    assert _operation_segments() == [("agent", 1)]


def test_operation_path_reuses_nested_occurrences() -> None:
    boundaries = (
        ChildOperationBoundary("case_investigation", "task-a"),
        ChildOperationBoundary("policy_analysis", "task-b"),
    )

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: boundaries):
            expected = [
                ("agent", 1),
                ("case_investigation", 1),
                ("policy_analysis", 1),
            ]
            assert _operation_segments() == expected
            assert _operation_segments() == expected


def test_child_boundary_composes_with_and_restores_parent_resolver() -> None:
    parent = ChildOperationBoundary("case_investigation", "task-a")
    child = ChildOperationBoundary("policy_analysis", "task-b")

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: (parent,)):
            with child_operation_boundary_scope(child):
                assert _operation_segments() == [
                    ("agent", 1),
                    ("case_investigation", 1),
                    ("policy_analysis", 1),
                ]
            assert _operation_segments() == [
                ("agent", 1),
                ("case_investigation", 1),
            ]


@pytest.mark.asyncio
async def test_child_boundary_propagates_across_await() -> None:
    boundary = ChildOperationBoundary("case_investigation", "task-a")

    with attempt_context_scope(_attempt("attempt-1")):
        with child_operation_boundary_scope(boundary):
            await asyncio.sleep(0)
            assert _operation_segments() == [
                ("agent", 1),
                ("case_investigation", 1),
            ]
        assert _operation_segments() == [("agent", 1)]


def test_child_boundary_restores_parent_after_failure() -> None:
    boundary = ChildOperationBoundary("case_investigation", "task-a")

    with attempt_context_scope(_attempt("attempt-1")):
        with pytest.raises(RuntimeError, match="boom"):
            with child_operation_boundary_scope(boundary):
                raise RuntimeError("boom")
        assert _operation_segments() == [("agent", 1)]


def test_later_step_gets_next_child_occurrence() -> None:
    current = [ChildOperationBoundary("case_investigation", "task-a")]

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: current):
            assert _operation_segments()[-1] == ("case_investigation", 1)
            advance_step_ordinal(1)
            current[:] = [ChildOperationBoundary("case_investigation", "task-b")]
            assert _operation_segments()[-1] == ("case_investigation", 2)


def test_same_child_has_only_one_occurrence_per_step() -> None:
    current = [ChildOperationBoundary("case_investigation", "task-a")]

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: current):
            assert _operation_segments()[-1] == ("case_investigation", 1)
            current[:] = [ChildOperationBoundary("case_investigation", "task-b")]
            with pytest.raises(
                UnsupportedChildOperationFanOutError,
                match="deterministic preallocation",
            ):
                current_operation_path()


def test_preallocate_child_paths_preserve_identity_under_inverted_start_order() -> None:
    current: list[ChildOperationBoundary] = []
    siblings = (
        ChildOperationBoundary("research", "task-a"),
        ChildOperationBoundary("research", "task-b"),
    )

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: current):
            assert preallocate_child_operation_ordinals(siblings) == [1, 2]
            current[:] = [siblings[1]]
            assert _operation_segments()[-1] == ("research", 2)
            current[:] = [siblings[0]]
            assert _operation_segments()[-1] == ("research", 1)
            assert preallocate_child_operation_ordinals(siblings) == [1, 2]


def test_preallocate_child_paths_reuse_ordinals_on_replacement_attempt() -> None:
    siblings = (
        ChildOperationBoundary("research", "task-a"),
        ChildOperationBoundary("research", "task-b"),
    )

    def ordinals_by_key(attempt_id: str, start_order: tuple[int, ...]) -> dict[str, int]:
        current: list[ChildOperationBoundary] = []
        seen: dict[str, int] = {}
        with attempt_context_scope(_attempt(attempt_id)):
            with operation_path_resolver_scope(lambda: current):
                preallocate_child_operation_ordinals(siblings)
                for index in start_order:
                    current[:] = [siblings[index]]
                    seen[siblings[index].occurrence_key] = _operation_segments()[-1][1]
        return seen

    first = ordinals_by_key("attempt-1", (1, 0))
    replay = ordinals_by_key("attempt-2", (0, 1))
    assert first == replay == {"task-a": 1, "task-b": 2}


def test_later_step_after_preallocated_siblings_advances() -> None:
    current: list[ChildOperationBoundary] = []
    siblings = (
        ChildOperationBoundary("research", "task-a"),
        ChildOperationBoundary("research", "task-b"),
    )

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: current):
            preallocate_child_operation_ordinals(siblings)
            current[:] = [siblings[0]]
            assert _operation_segments()[-1] == ("research", 1)
            advance_step_ordinal(1)
            current[:] = [ChildOperationBoundary("research", "task-c")]
            assert _operation_segments()[-1] == ("research", 3)


def test_preallocate_child_paths_rejects_empty_keys_before_resolve() -> None:
    with attempt_context_scope(_attempt("attempt-1")):
        with pytest.raises(ValueError, match="non-empty"):
            preallocate_child_operation_ordinals((ChildOperationBoundary("research", ""),))
        with pytest.raises(ValueError, match="non-empty"):
            preallocate_child_operation_ordinals(())
        current = [ChildOperationBoundary("research", "task-a")]
        with operation_path_resolver_scope(lambda: current):
            assert _operation_segments()[-1] == ("research", 1)


def test_preallocate_child_paths_rejects_duplicate_identities_before_assigning() -> None:
    duplicates = (
        ChildOperationBoundary("research", "task-a"),
        ChildOperationBoundary("research", "task-a"),
    )
    later = [ChildOperationBoundary("research", "task-b")]

    with attempt_context_scope(_attempt("attempt-1")):
        with pytest.raises(
            UnsupportedChildOperationFanOutError,
            match="duplicate occurrence",
        ):
            preallocate_child_operation_ordinals(duplicates)
        with operation_path_resolver_scope(lambda: later):
            assert _operation_segments()[-1] == ("research", 1)


@pytest.mark.asyncio
async def test_operation_path_resolver_propagates_and_restores() -> None:
    boundary = ChildOperationBoundary("case_investigation", "task-a")

    with attempt_context_scope(_attempt("attempt-1")):
        with operation_path_resolver_scope(lambda: (boundary,)):
            assert await asyncio.create_task(asyncio.to_thread(_operation_segments)) == [
                ("agent", 1),
                ("case_investigation", 1),
            ]
        assert _operation_segments() == [("agent", 1)]


def test_scope_sets_and_restores_context() -> None:
    attempt = _attempt("attempt-1")
    with attempt_context_scope(attempt):
        current = current_attempt_context()
        assert current is not None
        assert current.attempt_id == "attempt-1"
        assert current_step_ordinal() == 1
    assert current_attempt_context() is None
    with pytest.raises(RuntimeError, match="active durable attempt context"):
        current_step_ordinal()


def test_branch_scope_starts_after_source_cutoff() -> None:
    with attempt_context_scope(_branch_attempt()):
        assert current_step_ordinal() == 5


def test_branch_token_binding_starts_after_source_cutoff() -> None:
    binding = set_attempt_context(_branch_attempt())
    try:
        assert current_step_ordinal() == 5
    finally:
        reset_attempt_context(binding)


def test_malformed_branch_lineage_is_rejected() -> None:
    malformed = _attempt("attempt-branch")
    malformed.branch_lineage.source_step_ordinal = 4
    malformed.branch_lineage.source_state_hash = "sha256:source"
    with pytest.raises(ValueError, match="immutable source cutoff"):
        with attempt_context_scope(malformed):
            pass


def test_branch_lineage_requires_nonempty_source_identity() -> None:
    malformed = _branch_attempt()
    malformed.branch_lineage.source_workflow_identity.execution_id = ""
    with pytest.raises(ValueError, match="immutable source cutoff"):
        set_attempt_context(malformed)


def test_nested_scopes_restore_outer_context() -> None:
    outer = _attempt("attempt-outer")
    inner = _attempt("attempt-inner")
    with attempt_context_scope(outer):
        advance_step_ordinal(1)
        assert current_step_ordinal() == 2
        with attempt_context_scope(inner):
            current = current_attempt_context()
            assert current is not None
            assert current.attempt_id == "attempt-inner"
            assert current_step_ordinal() == 1
        current = current_attempt_context()
        assert current is not None
        assert current.attempt_id == "attempt-outer"
        assert current_step_ordinal() == 2


def test_scope_restores_context_when_body_raises() -> None:
    try:
        with attempt_context_scope(_attempt("attempt-1")):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_attempt_context() is None


@pytest.mark.asyncio
async def test_concurrent_tasks_do_not_share_context() -> None:
    seen: dict[str, str | None] = {}
    entered = asyncio.Event()
    release = asyncio.Event()

    async def durable_task() -> None:
        with attempt_context_scope(_attempt("attempt-durable")):
            entered.set()
            await release.wait()
            current = current_attempt_context()
            seen["durable"] = current.attempt_id if current else None

    async def native_task() -> None:
        await entered.wait()
        current = current_attempt_context()
        seen["native"] = current.attempt_id if current else None
        release.set()

    await asyncio.gather(durable_task(), native_task())

    assert seen == {"durable": "attempt-durable", "native": None}


@pytest.mark.asyncio
async def test_step_ordinal_advances_across_asyncio_tasks() -> None:
    """LangGraph schedules checkpointer puts on child tasks; the counter must share."""

    async def advance_in_child() -> None:
        advance_step_ordinal(current_step_ordinal())

    with attempt_context_scope(_attempt("attempt-1")):
        assert current_step_ordinal() == 1
        await asyncio.create_task(advance_in_child())
        assert current_step_ordinal() == 2
        await asyncio.create_task(advance_in_child())
        assert current_step_ordinal() == 3


def test_set_attempt_context_nested_binding_restores_outer_step() -> None:
    outer = set_attempt_context(_attempt("attempt-outer"))
    try:
        advance_step_ordinal(1)
        assert current_step_ordinal() == 2
        inner = set_attempt_context(_attempt("attempt-inner"))
        try:
            assert current_attempt_context() is not None
            assert current_attempt_context().attempt_id == "attempt-inner"
            assert current_step_ordinal() == 1
        finally:
            reset_attempt_context(inner)
        assert current_attempt_context() is not None
        assert current_attempt_context().attempt_id == "attempt-outer"
        assert current_step_ordinal() == 2
    finally:
        reset_attempt_context(outer)


def test_preallocate_preserves_identity_under_inverted_start_order() -> None:
    import threading
    from contextvars import copy_context

    from agent_engine_runner_shared.workflow.context import (
        allocate_activity_ordinal,
        preallocate_activity_ordinals,
        tool_activity_key,
    )

    with attempt_context_scope(_attempt("attempt-1")):
        keys = [tool_activity_key("call-a"), tool_activity_key("call-b")]
        assert preallocate_activity_ordinals(keys) == [1, 2]

        # Inverted lookup order still returns the preallocated ordinals.
        assert allocate_activity_ordinal(tool_activity_key("call-b")) == 2
        assert allocate_activity_ordinal(tool_activity_key("call-a")) == 1

        seen: dict[str, int] = {}
        parent = copy_context()

        def lookup(tool_call_id: str) -> None:
            seen[tool_call_id] = allocate_activity_ordinal(tool_activity_key(tool_call_id))

        threads = [
            threading.Thread(target=lambda: parent.run(lookup, "call-b")),
            threading.Thread(target=lambda: parent.run(lookup, "call-a")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        assert seen == {"call-a": 1, "call-b": 2}
        assert allocate_activity_ordinal() == 3


def test_allocate_without_key_is_monotonic() -> None:
    from agent_engine_runner_shared.workflow.context import allocate_activity_ordinal

    with attempt_context_scope(_attempt("attempt-1")):
        assert allocate_activity_ordinal() == 1
        assert allocate_activity_ordinal("stable") == 2
        assert allocate_activity_ordinal("stable") == 2
        assert allocate_activity_ordinal() == 3
    assert current_attempt_context() is None


def test_activity_ordinals_are_stable_when_nested_paths_start_in_different_orders() -> None:
    from agent_engine_runner_shared.workflow.context import allocate_activity_ordinal

    def allocate(order: list[tuple[str, str]], attempt_id: str) -> dict[str, int]:
        current = [ChildOperationBoundary("first_child", "task-a")]
        ordinals: dict[str, int] = {}
        with attempt_context_scope(_attempt(attempt_id)):
            with operation_path_resolver_scope(lambda: current):
                for child_name, llm_key in order:
                    current[:] = [ChildOperationBoundary(child_name, f"task-{child_name}")]
                    ordinals[child_name] = allocate_activity_ordinal(llm_key)
        return ordinals

    first_attempt = allocate(
        [("first_child", "llm:1"), ("second_child", "llm:2")],
        "attempt-1",
    )
    replay_attempt = allocate(
        [("second_child", "llm:1"), ("first_child", "llm:2")],
        "attempt-2",
    )

    assert (
        first_attempt
        == replay_attempt
        == {
            "first_child": 1,
            "second_child": 1,
        }
    )
