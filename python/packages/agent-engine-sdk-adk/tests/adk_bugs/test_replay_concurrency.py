"""ADK bug patched by ``adk_workarounds.replay_concurrency``.

ADK counts a replayed child toward ``max_concurrency`` while it waits for its
recorded turn. That shows only once a re-run child is allowed to be slow,
which stock ADK fails first (the slow re-run bug), so the reproduction keeps the
``replay_order`` workaround and gives only the cap back to ADK. When it fails
on a newer ADK, that release fixed the bug: delete the workaround and this
file.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest
from google.adk.workflow import START, FunctionNode, Workflow

from agent_engine_sdk_adk.adk_workarounds import replay_concurrency
from agent_engine_sdk_adk.route import _CURRENT_ROUTE_ADAPTER, DurableRouteAdapter

from ._support import (
    PIN_HINT,
    inactive_workaround_warnings,
    run_and_resume,
    shorten_replay_barrier,
    slow_rerun_workflow,
)


@pytest.mark.asyncio
async def test_stock_adk_cap_times_out_a_replay_sharing_its_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shorten_replay_barrier(monkeypatch)
    workflow = slow_rerun_workflow(max_concurrency=2)
    DurableRouteAdapter(workflow)
    monkeypatch.setattr(
        Workflow,
        "_at_concurrency_limit",
        inspect.unwrap(getattr(Workflow, "_at_concurrency_limit")),
    )

    # The replayed audit holds a slot waiting for cached, which has none.
    with pytest.raises(Exception, match="Replay divergence detected"):
        await run_and_resume(workflow, durable=True)


@pytest.mark.parametrize(
    ("max_concurrency", "replayed", "at_limit"),
    [
        (2, {"cached"}, True),
        (2, {"audit", "cached"}, False),
        (-1, set(), None),
        (None, set(), None),
    ],
)
def test_concurrency_cap_counts_only_real_executions(
    max_concurrency: int | None, replayed: set[str], at_limit: bool | None
) -> None:
    finish = FunctionNode(name="finish", func=lambda: "done")
    workflow = Workflow(
        name="outer", edges=[(START, finish)], max_concurrency=max_concurrency
    )
    DurableRouteAdapter(workflow)
    loop_state = SimpleNamespace(
        pending_tasks=dict.fromkeys(("audit", "review", "cached")),
        replayed_nodes=replayed,
    )
    at_concurrency_limit = getattr(Workflow, "_at_concurrency_limit")
    adapter_token = _CURRENT_ROUTE_ADAPTER.set(object())
    try:
        actual = at_concurrency_limit(workflow, loop_state)
    finally:
        _CURRENT_ROUTE_ADAPTER.reset(adapter_token)

    # Unset and nonpositive caps keep ADK's own answer.
    expected = (
        inspect.unwrap(at_concurrency_limit)(workflow, loop_state)
        if at_limit is None
        else at_limit
    )
    assert actual == expected


def test_defaulted_cap_hook_leaves_the_cap_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ReshapedWorkflow:
        def _at_concurrency_limit(self, loop_state: object = None) -> bool:
            del loop_state
            return False

    original_limit = ReshapedWorkflow._at_concurrency_limit

    replay_concurrency.install(ReshapedWorkflow, lambda: True)

    assert ReshapedWorkflow._at_concurrency_limit is original_limit
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "replay under max_concurrency" in warnings[0]
    assert re.search(PIN_HINT, warnings[0])


def test_reshaped_loop_state_leaves_the_cap_to_stock_adk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CurrentWorkflow:
        max_concurrency = 1

        def _at_concurrency_limit(self, loop_state: object) -> str:
            del loop_state
            return "stock answer"

    replay_concurrency.install(CurrentWorkflow, lambda: True)

    at_limit = CurrentWorkflow()._at_concurrency_limit(SimpleNamespace())

    assert at_limit == "stock answer"
    assert len(inactive_workaround_warnings(caplog)) == 1
