"""Follow a recorded sub-agent transfer inside a Workflow on ADK 2.9.

ADK 2.9 fast-forwards a Workflow child whose history holds a transfer, and the
Workflow loop does not follow it. ADK 2.10 fixed this, so the patch installs
only on older releases. Delete this module when the package's google-adk floor
reaches 2.10.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any

from agent_engine_sdk_adk.adk_compat import (
    adk_release_before,
    warn_workaround_inactive,
)

if TYPE_CHECKING:
    from google.adk.workflow import BaseNode

INSTALLED_ATTRIBUTE = "_agent_engine_workflow_transfer_replay_installed"
WORKAROUND = "transfer replay inside a Workflow"
CONSEQUENCE = "a resumed sub-agent transfer inside a Workflow is not followed"


def install(in_durable_turn: Callable[[], bool]) -> None:
    """Wrap ADK's private ``_workflow.check_interception`` on ADK 2.9.

    If it is missing, or the installed ADK version is unknown, the workaround
    is skipped with one warning.
    """
    release_before_fix = adk_release_before(2, 10)
    if release_before_fix is None:
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return
    if not release_before_fix:
        return
    try:
        from google.adk.workflow.utils._replay_interceptor import InterceptionResult

        adk_workflow = import_module("google.adk.workflow._workflow")
        if getattr(adk_workflow, INSTALLED_ATTRIBUTE, False):
            return
        original_interception = adk_workflow.check_interception
    except (ImportError, AttributeError):
        warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
        return

    @wraps(original_interception)
    def check_recorded_transfer(
        *, node: BaseNode, recovered: Any, **kwargs: Any
    ) -> Any:
        if in_durable_turn() and recovered is not None:
            try:
                transfer_to_agent = recovered.transfer_to_agent
            except AttributeError:
                warn_workaround_inactive(WORKAROUND, CONSEQUENCE)
            else:
                if transfer_to_agent is not None:
                    # Only a configured transfer agent records a transfer.
                    # Context replays the node before following its target.
                    return InterceptionResult(should_run=True)
        return original_interception(node=node, recovered=recovered, **kwargs)

    setattr(adk_workflow, "check_interception", check_recorded_transfer)
    setattr(adk_workflow, INSTALLED_ATTRIBUTE, True)
