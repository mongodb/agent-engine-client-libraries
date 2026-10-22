"""How the adapter behaves when an installed ADK reshapes internals it patches."""

from __future__ import annotations

import re
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import tomllib
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.workflow import START, BaseNode, Node, Workflow, node

from agent_engine_sdk_adk import adk_compat
from agent_engine_sdk_adk.agent_route import AgentRouteAdapter
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.execution_session import _reject_unsupported_collaboration
from agent_engine_sdk_adk.route import (
    _CURRENT_WORKFLOW_PATH,
    _PARALLEL_WORKER_RUN_WRAPPER_ATTRIBUTE,
    DurableRouteAdapter,
)

_PACKAGE = Path(__file__).resolve().parents[1]
_PIN_HINT = f"pin google-adk<={re.escape(adk_compat.TESTED_ADK_MAX)}"


class _ItemNode(Node):
    async def run_node_impl(
        self, *, ctx: Context, node_input: object
    ) -> AsyncGenerator[object, None]:
        del ctx
        yield node_input


def _echo(node_input: str) -> str:
    return node_input


def test_tested_range_matches_the_floor_and_the_lock() -> None:
    pyproject = tomllib.loads((_PACKAGE / "pyproject.toml").read_text())
    floor = next(
        requirement.split(">=")[1].split(",")[0]
        for requirement in pyproject["project"]["dependencies"]
        if requirement.startswith("google-adk>=")
    )
    lock = tomllib.loads((_PACKAGE.parents[1] / "uv.lock").read_text())
    locked = next(
        package["version"]
        for package in lock["package"]
        if package["name"] == "google-adk"
    )

    # The messages promise this range, so it must be what CI actually tests:
    # the minimum-version suite installs the floor and the main suite the lock.
    assert (adk_compat.TESTED_ADK_MIN, adk_compat.TESTED_ADK_MAX) == (floor, locked)


def test_moved_parallel_worker_module_refuses_only_worker_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = Workflow(name="plain", edges=[(START, _ItemNode(name="step"))])
    workers = Workflow(
        name="fanout",
        edges=[(START, node(_echo, name="worker", parallel_worker=True))],
    )
    monkeypatch.setitem(sys.modules, "google.adk.workflow._parallel_worker", None)

    _reject_unsupported_collaboration(plain)
    DurableRouteAdapter(plain)

    for check in (_reject_unsupported_collaboration, DurableRouteAdapter):
        with pytest.raises(
            UnsupportedDurableADKError,
            match=rf"parallel_worker nodes.*{_PIN_HINT}.*parallel Workflow branches",
        ):
            check(workers)


@pytest.mark.parametrize("renamed", ["worker template", "worker wrapper"])
def test_renamed_parallel_worker_internals_refuse_before_running(
    monkeypatch: pytest.MonkeyPatch, renamed: str
) -> None:
    worker: BaseNode
    if renamed == "worker template":
        worker = node(_echo, name="worker", parallel_worker=True)
        monkeypatch.setattr(worker, "_node", None)
    else:
        worker = _ItemNode(name="worker", parallel_worker=True)
        monkeypatch.setattr(worker, "_inner_node", None)
    workflow = Workflow(name="fanout", edges=[(START, worker)])

    with pytest.raises(UnsupportedDurableADKError, match=_PIN_HINT):
        _reject_unsupported_collaboration(workflow)


async def _reshaped_run_impl(
    self: object, *, context: Context, node_input: object
) -> AsyncGenerator[object, None]:
    del self, context
    yield node_input


@pytest.mark.parametrize("reshaped", ["inherited stub", "renamed parameters"])
def test_reshaped_worker_run_implementation_refuses_before_running(
    monkeypatch: pytest.MonkeyPatch, reshaped: str
) -> None:
    worker = node(_echo, name="worker", parallel_worker=True)
    workflow = Workflow(name="fanout", edges=[(START, worker)])
    worker_type = type(worker)
    # Earlier adapters may already have wrapped the class; model a fresh
    # process on an ADK whose worker reshaped its run implementation.
    monkeypatch.delattr(
        worker_type, _PARALLEL_WORKER_RUN_WRAPPER_ATTRIBUTE, raising=False
    )
    if reshaped == "inherited stub":
        monkeypatch.delattr(worker_type, "_run_impl")
    else:
        monkeypatch.setattr(worker_type, "_run_impl", _reshaped_run_impl)

    with pytest.raises(UnsupportedDurableADKError, match=_PIN_HINT):
        DurableRouteAdapter(workflow)


def test_transfer_agents_rerun_on_resume_unless_the_developer_chose() -> None:
    specialist = LlmAgent(name="specialist", model="gemini-2.0-flash")
    pinned = LlmAgent(name="pinned", model="gemini-2.0-flash", rerun_on_resume=False)
    router = LlmAgent(
        name="router", model="gemini-2.0-flash", sub_agents=[specialist, pinned]
    )

    AgentRouteAdapter(router, _CURRENT_WORKFLOW_PATH)

    assert (router.rerun_on_resume, specialist.rerun_on_resume) == (True, True)
    assert pinned.rerun_on_resume is False
