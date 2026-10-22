"""ADK bugs patched by ``adk_workarounds.replay_order``.

Each ``test_stock_adk_*`` test reproduces a bug on stock ADK, with no Agent
Engine code. When one fails on a newer ADK, that release fixed the bug: delete
its half of the workaround and the test.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest
from google.adk.agents.context import Context
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, BaseNode, FunctionNode, Workflow

from agent_engine_sdk_adk.adk_workarounds import replay_order
from agent_engine_sdk_adk.adk_workarounds.replay_order import (
    _release_from_replay_order,
)

from ._support import (
    APPROVAL,
    PIN_HINT,
    after_recorded,
    executing_siblings,
    inactive_workaround_warnings,
    joined_inputs,
    recorded_first,
    run_and_resume,
    shorten_replay_barrier,
    slow_rerun_workflow,
    slow_review_runs,
)


@pytest.mark.asyncio
async def test_stock_adk_times_out_behind_a_slow_rerun_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shorten_replay_barrier(monkeypatch)

    # The replayed audit waits for the slowly re-executing review's turn.
    with pytest.raises(Exception, match="Replay divergence detected"):
        await run_and_resume(slow_rerun_workflow(), durable=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrency", [None, 2])
async def test_slow_resumed_branch_does_not_fail_replayed_siblings(
    monkeypatch: pytest.MonkeyPatch, max_concurrency: int | None
) -> None:
    shorten_replay_barrier(monkeypatch)

    waits = await run_and_resume(slow_rerun_workflow(max_concurrency), durable=True)

    assert waits == ["review-input"]
    assert slow_review_runs == [APPROVAL]
    assert joined_inputs == [
        {"audit": "audited", "review": "reviewed", "cached": "cached"}
    ]
    assert executing_siblings[1] <= (max_concurrency or 3)


@pytest.mark.asyncio
async def test_releasing_a_rerun_node_keeps_replayed_order_enforced() -> None:
    from google.adk.workflow.utils._replay_sequence_barrier import (
        ReplaySequenceBarrier,
    )

    barrier = ReplaySequenceBarrier(["review@1", "audit@1", "verify@1"], 0.05)

    # A later re-executing node leaves the replayed nodes ahead of it gated.
    _release_from_replay_order(barrier, "audit@1")
    with pytest.raises(Exception, match="Replay divergence detected"):
        await barrier.wait("verify@1")

    # Releasing the node currently expected unblocks the next replayed node.
    _release_from_replay_order(barrier, "review@1")
    await barrier.wait("verify@1")


def test_barrier_without_the_next_event_leaves_replay_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    barrier = SimpleNamespace(
        sequence=["review@1", "audit@1"], events={}, current_index=0
    )

    _release_from_replay_order(barrier, "review@1")

    assert barrier.sequence == ["review@1", "audit@1"]
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "replay order on resume" in warnings[0]


_paired_reviews: list[tuple[object, object]] = []


def _forward(node_input: object) -> object:
    return node_input


def _inner_lookup() -> str:
    recorded_first["inner"].set()
    return "from-inner"


async def _cached_check() -> str:
    # History records inner, cached, (audit,) then review's interrupt.
    await after_recorded("inner")
    recorded_first["cached"].set()
    return "from-cached"


async def _audit_check() -> str:
    await after_recorded("cached")
    recorded_first["audit"].set()
    return "from-audit"


async def _paired_review(ctx: Context, node_input: object) -> object:
    answer = (ctx.resume_inputs or {}).get("review-input")
    if answer is None:
        await after_recorded("last_sibling")
        return RequestInput(interrupt_id="review-input", message="Review")
    _paired_reviews.append((node_input, answer))
    return "reviewed"


def _nested_siblings_workflow(siblings_in_workflows: bool) -> Workflow:
    """A completed nested Workflow and its siblings, all feeding one review.

    ADK re-enters the nested Workflow on resume even though it completed, and
    it replays more children than its siblings, so its replay finishes last.
    With ``siblings_in_workflows`` every sibling is a nested Workflow; a third
    one checks that each held sibling releases the next.
    """
    _paired_reviews.clear()
    recorded_first.clear()
    recorded_first.update(
        inner=asyncio.Event(), cached=asyncio.Event(), audit=asyncio.Event()
    )
    step = FunctionNode(name="step", func=_inner_lookup)
    forward = FunctionNode(name="forward", func=_forward)
    last = FunctionNode(name="last", func=_forward)
    inner = Workflow(
        name="inner", edges=[(START, step), (step, forward), (forward, last)]
    )
    cached: BaseNode = FunctionNode(name="cached", func=_cached_check)
    siblings: tuple[BaseNode, ...] = (inner, cached)
    recorded_first["last_sibling"] = recorded_first["cached"]
    if siblings_in_workflows:
        audit = FunctionNode(name="audit", func=_audit_check)
        siblings = (
            inner,
            Workflow(name="cached", edges=[(START, cached)]),
            Workflow(name="audit", edges=[(START, audit)]),
        )
        recorded_first["last_sibling"] = recorded_first["audit"]
    review = FunctionNode(name="review", func=_paired_review, rerun_on_resume=True)
    return Workflow(
        name="outer",
        edges=[(START, siblings), *((sibling, review) for sibling in siblings)],
    )


@pytest.mark.asyncio
async def test_stock_adk_pairs_the_review_with_the_wrong_sibling() -> None:
    # Stock ADK's barrier still orders a plain sibling; it orders no re-entered
    # nested Workflow, so the bug needs every sibling to be one.
    await run_and_resume(_nested_siblings_workflow(True), durable=False)

    # The answer requested for the nested workflow's output reaches
    # a sibling's, because the nested workflow finished replaying last.
    assert _paired_reviews[0][0] != "from-inner"


@pytest.mark.parametrize("siblings_in_workflows", [False, True])
@pytest.mark.asyncio
async def test_completed_nested_workflow_keeps_its_replay_order(
    siblings_in_workflows: bool,
) -> None:
    # With a plain sibling this guards the re-run child release: it must not let
    # the re-entered nested Workflow leave the order as a re-run child would.
    await run_and_resume(_nested_siblings_workflow(siblings_in_workflows), durable=True)

    # The answer was requested for the review triggered by the nested workflow.
    assert _paired_reviews[0] == ("from-inner", APPROVAL)


def test_renamed_start_hook_leaves_replay_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ReshapedWorkflow:
        def _start_node_task(
            self, state: object, ctx: object, name: str, trigger: object
        ) -> bool:
            del state, ctx, name, trigger
            return True

    original_start = ReshapedWorkflow._start_node_task

    replay_order.install(ReshapedWorkflow, lambda: True)
    # Every new session retries the install; the warning stays one per process.
    replay_order.install(ReshapedWorkflow, lambda: True)

    assert ReshapedWorkflow._start_node_task is original_start
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "replay order on resume" in warnings[0]
    assert re.search(PIN_HINT, warnings[0])


def test_defaulted_start_hook_argument_leaves_replay_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started: list[str] = []

    class DefaultedWorkflow:
        def _start_node_task(
            self,
            loop_state: object,
            ctx: object,
            node_name: str = "review",
            trigger: object = None,
        ) -> bool:
            del loop_state, ctx, trigger
            started.append(node_name)
            return True

    original_start = DefaultedWorkflow._start_node_task

    replay_order.install(DefaultedWorkflow, lambda: True)
    replay_order.install(DefaultedWorkflow, lambda: True)
    # A caller may omit the defaulted name, which a wrapper could not bind.
    results = [
        DefaultedWorkflow()._start_node_task(SimpleNamespace(), None) for _ in range(2)
    ]

    assert DefaultedWorkflow._start_node_task is original_start
    assert results == [True, True]
    assert started == ["review", "review"]
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "replay order on resume" in warnings[0]
    assert re.search(PIN_HINT, warnings[0])


def test_reshaped_loop_state_leaves_the_resumed_turn_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CurrentWorkflow:
        def _start_node_task(
            self, loop_state: object, ctx: object, node_name: str, trigger: object
        ) -> bool:
            del loop_state, ctx, node_name, trigger
            return True

    replay_order.install(CurrentWorkflow, lambda: True)

    started = CurrentWorkflow()._start_node_task(
        SimpleNamespace(), None, "review", None
    )

    assert started is True
    assert len(inactive_workaround_warnings(caplog)) == 1
