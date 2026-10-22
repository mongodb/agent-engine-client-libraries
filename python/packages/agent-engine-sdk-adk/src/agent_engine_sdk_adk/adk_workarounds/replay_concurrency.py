"""Stop replayed children from counting toward a Workflow's ``max_concurrency``.

On resume, a child replayed from history holds a concurrency slot while it
waits for its recorded turn. When that turn belongs to a child still waiting
for a slot, the replay can only time out, so during a durable turn the cap
counts real executions only. Delete this module once a released ADK stops
counting replayed children toward the cap.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from agent_engine_sdk_adk.adk_compat import warn_workaround_inactive

INSTALLED_ATTRIBUTE = "_agent_engine_replay_concurrency_installed"
WORKAROUND = "replay under max_concurrency"
CONSEQUENCE = (
    "a resumed Workflow that sets max_concurrency can time out while replayed "
    "children hold its slots"
)


def install(workflow_type: type, in_durable_turn: Callable[[], bool]) -> None:
    """Wrap ADK's private ``Workflow._at_concurrency_limit`` once.

    If ADK renamed it or reshaped its loop state, stock ADK decides, with one
    warning.
    """
    if getattr(workflow_type, INSTALLED_ATTRIBUTE, False):
        return

    original_at_concurrency_limit = getattr(
        workflow_type, "_at_concurrency_limit", None
    )
    try:
        signature = inspect.signature(cast(Any, original_at_concurrency_limit))
    except (TypeError, ValueError):
        signature = None
    # Wrap only ADK's (self, loop_state) shape, so delegation never changes a call.
    if signature is None or [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in signature.parameters.values()
    ] != [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        (
            "loop_state",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
        ),
    ]:
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return
    original = cast(Any, original_at_concurrency_limit)

    @wraps(original)
    def at_concurrency_limit(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs).arguments
        workflow, loop_state = arguments["self"], arguments["loop_state"]
        # ADK owns unset and nonpositive caps, which newer releases treat as
        # unlimited.
        limit = getattr(workflow, "max_concurrency", None)
        if not in_durable_turn() or limit is None or limit <= 0:
            return original(*args, **kwargs)
        try:
            executing = loop_state.pending_tasks.keys() - loop_state.replayed_nodes
        except (AttributeError, TypeError):
            warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
            return original(*args, **kwargs)
        return len(executing) >= limit

    setattr(workflow_type, "_at_concurrency_limit", at_concurrency_limit)
    setattr(workflow_type, INSTALLED_ATTRIBUTE, True)
