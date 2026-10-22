"""ADK bug left as a known limitation: a nested Workflow's early output.

A nested Workflow whose output is recorded before its parallel branches finish
replays at that earlier time on resume, so a downstream node can resume with a
sibling's input. The fix belongs in ADK's replay sequencing, so the README
documents it with its avoidance instead of a workaround. When the stock test
fails on a newer ADK, that release fixed the bug: drop the limitation from the
README and delete this file.
"""

from __future__ import annotations

import asyncio

import pytest
from google.adk.agents.context import Context
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, FunctionNode, JoinNode, Workflow

from ._support import after_recorded, recorded_first, run_and_resume

# (node path, input) for each review run, first turn and resumed turn.
_asked: dict[str, object] = {}
_resumed: dict[str, object] = {}


def _step() -> str:
    # The nested Workflow's output is recorded here, first.
    recorded_first["step"].set()
    return "from-inner"


async def _side() -> None:
    # The nested Workflow completes only after its sibling.
    await after_recorded("cached")
    recorded_first["side"].set()


async def _cached() -> str:
    await after_recorded("step")
    recorded_first["cached"].set()
    return "from-cached"


async def _review(ctx: Context, node_input: object) -> object:
    if (ctx.resume_inputs or {}).get("review-input") is None:
        _asked[ctx.node_path] = node_input
        await after_recorded("side")
        return RequestInput(interrupt_id="review-input", message="Review")
    _resumed[ctx.node_path] = node_input
    return "reviewed"


def _workflow(*, join_branches: bool) -> Workflow:
    _asked.clear()
    _resumed.clear()
    recorded_first.clear()
    recorded_first.update(
        step=asyncio.Event(), side=asyncio.Event(), cached=asyncio.Event()
    )
    step = FunctionNode(name="step", func=_step)
    side = FunctionNode(name="side", func=_side)
    edges: list = [(START, (step, side))]
    if join_branches:
        join = JoinNode(name="join")
        edges += [(step, join), (side, join)]
    inner = Workflow(name="inner", edges=edges)
    cached = FunctionNode(name="cached", func=_cached)
    review = FunctionNode(name="review", func=_review, rerun_on_resume=True)
    return Workflow(
        name="outer",
        edges=[(START, (inner, cached)), (inner, review), (cached, review)],
    )


def _mismatched_resumes() -> dict[str, tuple[object, object]]:
    """Reviews that resumed with an input other than the one they asked with."""
    assert _resumed
    return {
        path: (_asked.get(path), node_input)
        for path, node_input in _resumed.items()
        if _asked.get(path) != node_input
    }


@pytest.mark.asyncio
async def test_stock_adk_resumes_a_review_with_a_sibling_input() -> None:
    await run_and_resume(_workflow(join_branches=False), durable=False)

    assert _mismatched_resumes()


@pytest.mark.asyncio
async def test_join_node_keeps_each_review_paired_with_its_input() -> None:
    await run_and_resume(_workflow(join_branches=True), durable=True)

    assert not _mismatched_resumes()
