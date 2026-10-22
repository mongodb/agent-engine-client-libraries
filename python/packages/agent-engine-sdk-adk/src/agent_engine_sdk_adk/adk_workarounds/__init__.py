"""Temporary patches for google-adk bugs on rare durable-resume paths.

Durable execution needs only a few ADK internals; those patches live with the
adapter in ``route.py``. This package holds the other kind: patches for ADK
bugs, each one in its own module, named after the ADK internal it patches.

- ``replay_order`` patches ``Workflow._start_node_task`` (ADK 2.9 and newer).
  A child re-run on resume stalls siblings replayed behind it past the 15s
  barrier timeout, and several completed nested Workflows replay out of
  recorded order, so a resumed node pairs with the wrong input.
- ``replay_concurrency`` patches ``Workflow._at_concurrency_limit`` (ADK 2.9
  and newer). Replayed children count toward ``max_concurrency`` while they
  wait for their recorded turn.
- ``workflow_transfer_replay`` patches ``_workflow.check_interception`` (ADK
  2.9 only). A recorded sub-agent transfer inside a Workflow is fast-forwarded
  instead of followed.

An ADK bug gets a patch here only when the fix is small and cannot affect other
behavior; a hard-to-fix bug is a known limitation in the README instead. Every
patch acts only during a durable adapter turn, checks the ADK shape it patches
and otherwise steps aside with one warning, and is deleted once a released ADK
fixes its bug. ``tests/adk_bugs`` reproduces each bug the locked ADK still
has and fails when an ADK release fixes it, which is the signal to delete the
module.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_engine_sdk_adk.adk_workarounds import (
    replay_concurrency,
    replay_order,
    workflow_transfer_replay,
)

__all__ = ["install_adk_workarounds"]


def install_adk_workarounds(in_durable_turn: Callable[[], bool]) -> None:
    """Install every ADK bug patch once per process.

    ``in_durable_turn`` reports whether a durable adapter turn is running; the
    patches leave every other ADK run to stock behavior. Repeated calls are
    no-ops.
    """
    from google.adk.workflow import Workflow

    workflow_transfer_replay.install(in_durable_turn)
    replay_order.install(Workflow, in_durable_turn)
    replay_concurrency.install(Workflow, in_durable_turn)
