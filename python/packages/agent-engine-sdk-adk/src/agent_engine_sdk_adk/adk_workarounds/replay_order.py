"""Keep children that ADK re-executes on resume in the recorded replay order.

On resume, ADK releases each child it replays from history only after the
child recorded before it completes, and fails the turn after a fixed 15
seconds. Two bugs come from children that run for real on resume:

- A child whose history holds only an interrupt or a failure re-executes, and
  a sibling replayed behind it times out whenever that work is slow. Such a
  child triggered nothing downstream, so it leaves the order.
- ADK re-enters a completed nested Workflow to replay its children without
  waiting for its turn, so several of them complete out of order and a
  downstream node pairs with the wrong input. Such a child completes
  only in its recorded turn.

Delete this module once a released ADK keeps both kinds of children in order.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from agent_engine_sdk_adk.adk_compat import warn_workaround_inactive

if TYPE_CHECKING:
    from google.adk.agents.context import Context

INSTALLED_ATTRIBUTE = "_agent_engine_rerun_replay_order_installed"
WORKAROUND = "replay order on resume"
CONSEQUENCE = (
    "a resumed Workflow can time out while a re-run child is slow, and a "
    "resumed node can receive another sibling's input when several completed "
    "nested Workflows replay"
)


def _release_from_replay_order(barrier: Any, key: str) -> None:
    """Remove a re-executing node from ADK's recorded replay order.

    Its history holds only an interrupt or a failure, so it triggered nothing
    downstream and its recorded position orders nothing. When it was the
    current key, the next replayed child is released. When ADK has no event
    for that next child, the order is left to stock ADK with one warning.
    """
    if barrier is None or key not in barrier.sequence:
        return
    index = barrier.sequence.index(key)
    if index < barrier.current_index:
        return
    next_event = None
    if index == barrier.current_index and index + 1 < len(barrier.sequence):
        next_event = barrier.events.get(barrier.sequence[index + 1])
        if next_event is None:
            warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
            return
    del barrier.sequence[index]
    barrier.events.pop(key, None)
    if next_event is not None:
        next_event.set()


def _hold_for_recorded_turn(
    barrier: Any, pending_tasks: dict[str, Any], node_name: str, key: str
) -> None:
    """Complete a re-entered child only in its recorded replay turn.

    The child's own replay still runs at once; only its completion waits for
    its turn, after which the next recorded child is released.
    """
    rerun = pending_tasks[node_name]

    async def complete_in_recorded_turn() -> Context:
        child_ctx = await rerun
        await barrier.wait(key)
        # ADK's own advance for this child already ran (2.10) or no-ops later
        # (2.9) because the key was not yet current, so release the next child.
        barrier.check_and_advance(key)
        return child_ctx

    pending_tasks[node_name] = asyncio.create_task(complete_in_recorded_turn())


def _has_recorded_outcome(recovered: Any) -> bool:
    """Whether history holds a child's successful result, which may have
    triggered downstream nodes, rather than only an interrupt or a failure."""
    return (
        recovered.output is not None
        or recovered.route is not None
        or recovered.transfer_to_agent is not None
    )


def _has_replay_barrier_shape(barrier: Any) -> bool:
    """Whether ADK's replay barrier still has the members the workaround uses."""
    return (
        isinstance(getattr(barrier, "sequence", None), list)
        and isinstance(getattr(barrier, "events", None), dict)
        and isinstance(getattr(barrier, "current_index", None), int)
        and callable(getattr(barrier, "wait", None))
        and callable(getattr(barrier, "check_and_advance", None))
    )


def _keep_recorded_replay_order(loop_state: Any, node_name: str) -> None:
    """Place a child ADK just started for real into the recorded replay order.

    ADK also re-enters a nested Workflow that completed, to replay its
    children; its recorded outcome triggered later siblings, so it keeps its
    place. Only a child with no recorded outcome is new. When ADK's loop
    state or barrier no longer has the expected shape, this leaves the turn to
    stock ADK and warns once.
    """
    try:
        key = f"{node_name}@{loop_state.nodes[node_name].run_id}"
        recovered = loop_state.recovered_executions.get(key)
        if recovered is None:
            return
        barrier = loop_state.replay_manager.sequence_barrier
        pending_tasks = loop_state.pending_tasks
        recorded = _has_recorded_outcome(recovered)
    except (AttributeError, KeyError, TypeError):
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return
    if barrier is None:
        return
    if not _has_replay_barrier_shape(barrier) or node_name not in pending_tasks:
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return
    if recorded:
        _hold_for_recorded_turn(barrier, pending_tasks, node_name, key)
    else:
        _release_from_replay_order(barrier, key)


def install(workflow_type: type, in_durable_turn: Callable[[], bool]) -> None:
    """Wrap ADK's private ``Workflow._start_node_task`` once.

    If ADK renamed it or changed its parameters, the workaround is skipped
    with one warning.
    """
    if getattr(workflow_type, INSTALLED_ATTRIBUTE, False):
        return

    original_start_node_task = getattr(workflow_type, "_start_node_task", None)
    try:
        signature = inspect.signature(cast(Any, original_start_node_task))
    except (TypeError, ValueError):
        signature = None
    # Both arguments must be required, so every call binds them.
    if signature is None or any(
        name not in signature.parameters
        or signature.parameters[name].default is not inspect.Parameter.empty
        for name in ("loop_state", "node_name")
    ):
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return
    original = cast(Any, original_start_node_task)

    @wraps(original)
    def start_node_task(*args: Any, **kwargs: Any) -> Any:
        runs_for_real = original(*args, **kwargs)
        if runs_for_real and in_durable_turn():
            arguments = signature.bind(*args, **kwargs).arguments
            _keep_recorded_replay_order(arguments["loop_state"], arguments["node_name"])
        return runs_for_real

    setattr(workflow_type, "_start_node_task", start_node_task)
    setattr(workflow_type, INSTALLED_ATTRIBUTE, True)
