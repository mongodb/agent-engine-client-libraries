"""ADK 2.9 bug patched by ``adk_workarounds.workflow_transfer_replay``.

ADK 2.10 fixed the bug, so the patch installs only on 2.9 and needs no stock
reproduction to signal a fix; this file covers how the patch steps aside.
"""

from __future__ import annotations

import re
from importlib import import_module
from types import SimpleNamespace

import pytest

from agent_engine_sdk_adk import adk_compat
from agent_engine_sdk_adk.adk_workarounds import workflow_transfer_replay

from ._support import PIN_HINT, inactive_workaround_warnings


def test_reshaped_transfer_record_leaves_interception_to_stock_adk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    adk_workflow = import_module("google.adk.workflow._workflow")
    stock_calls: list[object] = []

    def stock_interception(*, node: object, recovered: object) -> str:
        del node
        stock_calls.append(recovered)
        return "stock answer"

    # Install the ADK 2.9 workaround over a stub, whatever ADK is installed.
    monkeypatch.setattr(
        workflow_transfer_replay, "adk_release_before", lambda major, minor: True
    )
    monkeypatch.setattr(adk_workflow, "check_interception", stock_interception)
    monkeypatch.setattr(
        adk_workflow,
        workflow_transfer_replay.INSTALLED_ATTRIBUTE,
        False,
        raising=False,
    )
    workflow_transfer_replay.install(lambda: True)
    reshaped = SimpleNamespace()
    without_transfer = SimpleNamespace(transfer_to_agent=None)

    answers = [
        adk_workflow.check_interception(node=None, recovered=recovered)
        for recovered in (reshaped, reshaped, None, without_transfer)
    ]

    assert answers == ["stock answer"] * 4
    assert stock_calls == [reshaped, reshaped, None, without_transfer]
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "transfer replay inside a Workflow" in warnings[0]
    assert re.search(PIN_HINT, warnings[0])


def test_unknown_adk_version_warns_that_the_transfer_workaround_is_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    adk_workflow = import_module("google.adk.workflow._workflow")
    stock_interception = adk_workflow.check_interception
    # A vendored ADK has no package metadata, so it might be 2.9.
    monkeypatch.setattr(adk_compat, "_installed_adk_version", lambda: "unknown")
    monkeypatch.setattr(
        adk_workflow,
        workflow_transfer_replay.INSTALLED_ATTRIBUTE,
        False,
        raising=False,
    )

    workflow_transfer_replay.install(lambda: True)

    assert adk_workflow.check_interception is stock_interception
    warnings = inactive_workaround_warnings(caplog)
    assert len(warnings) == 1
    assert "transfer replay inside a Workflow" in warnings[0]
