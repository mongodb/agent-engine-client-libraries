"""Run an ADK workflow through a human-input wait, on stock ADK or durably."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
from collections.abc import Iterator

import pytest
from google.adk.agents.context import Context
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, FunctionNode, JoinNode, Workflow
from google.genai import types

from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_sdk_adk import adk_compat
from agent_engine_sdk_adk.route import DurableRouteAdapter
from agent_engine_sdk_adk.stream import AdkTurn
from agent_engine_sdk_adk.suspend import function_response_frontier_content

from ..test_route import _attempt

APPROVAL = {"decision": "approved"}
PIN_HINT = f"pin google-adk<={re.escape(adk_compat.TESTED_ADK_MAX)}"
_IDS = {"app_name": "app", "user_id": "user-1", "session_id": "session-1"}


async def run_and_resume(workflow: Workflow, *, durable: bool) -> list[str]:
    """Run until the workflow waits for input, answer each wait once, resume.

    ``durable=True`` runs both turns through the Agent Engine adapter, as a
    deployed agent does. ``durable=False`` runs stock ADK with no Agent Engine
    code, so a test can show that an ADK bug exists without us. Returns the
    interrupt IDs the first turn waited on; errors from the resumed turn
    propagate.
    """
    service = InMemorySessionService()
    await service.create_session(**_IDS)
    runner = Runner(node=workflow, app_name="app", session_service=service)
    start = types.Content(role="user", parts=[types.Part.from_text(text="go")])
    ids = {"user_id": _IDS["user_id"], "session_id": _IDS["session_id"]}
    if durable:
        adapter = DurableRouteAdapter(workflow)
        with attempt_context_scope(_attempt()):
            first = AdkTurn(runner, **ids, new_message=start, route_adapter=adapter)
            [event async for event in first.stream(emit_tokens=True)]
            frontier = first.wait_frontier()
            answers = function_response_frontier_content(
                frontier, [APPROVAL] * len(frontier)
            )
            second = AdkTurn(runner, **ids, new_message=answers, route_adapter=adapter)
            [event async for event in second.stream(emit_tokens=True)]
        return [wait.function_call_id for wait in frontier]

    [event async for event in runner.run_async(**ids, new_message=start)]
    session = await service.get_session(**_IDS)
    assert session is not None
    waits = [
        (event.invocation_id, part.function_call)
        for event in session.events
        if event.content
        for part in event.content.parts or []
        if part.function_call and part.function_call.name == "adk_request_input"
    ]
    assert waits, "the workflow finished without requesting human input"
    answers = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response=APPROVAL
                )
            )
            for _, call in waits
        ],
    )
    [
        event
        async for event in runner.run_async(
            **ids, new_message=answers, invocation_id=waits[0][0]
        )
    ]
    return [call.id or "" for _, call in waits]


def inactive_workaround_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The compatibility warnings logged when a workaround steps aside."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == adk_compat.__name__ and record.levelno == logging.WARNING
    ]


# Set by the sibling history must record first; created per test on its loop.
recorded_first: dict[str, asyncio.Event] = {}


async def after_recorded(name: str) -> None:
    """Wait until ``name`` ran, then long enough for ADK to record its result."""
    await recorded_first[name].wait()
    await asyncio.sleep(0.05)


def shorten_replay_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADK's replay barrier gives up after 15s; make a stall fail in 0.2s."""
    from google.adk.workflow.utils import _replay_manager

    monkeypatch.setattr(
        _replay_manager,
        "ReplaySequenceBarrier",
        functools.partial(_replay_manager.ReplaySequenceBarrier, timeout_sec=0.2),
    )


slow_review_runs: list[object] = []
joined_inputs: list[object] = []
# Siblings running at once now, and the most seen; checked against the cap.
executing_siblings: list[int] = [0, 0]


@contextlib.contextmanager
def _count_executing_sibling() -> Iterator[None]:
    executing_siblings[0] += 1
    executing_siblings[1] = max(executing_siblings)
    try:
        yield
    finally:
        executing_siblings[0] -= 1


async def _slow_review(ctx: Context) -> object:
    with _count_executing_sibling():
        answer = (ctx.resume_inputs or {}).get("review-input")
        if answer is None:
            return RequestInput(interrupt_id="review-input", message="Review")
        slow_review_runs.append(answer)
        # Slower than the shortened replay barrier.
        await asyncio.sleep(0.5)
        return "reviewed"


async def _quick_audit() -> str:
    with _count_executing_sibling():
        # Finishes last, so history orders it behind both siblings.
        await after_recorded("cached")
        return "audited"


async def _cached_lookup() -> str:
    with _count_executing_sibling():
        recorded_first["cached"].set()
        return "cached"


def _record_joined(node_input: object) -> str:
    joined_inputs.append(node_input)
    return "joined"


def slow_rerun_workflow(max_concurrency: int | None = None) -> Workflow:
    """A ``rerun_on_resume`` review that is slow on resume, beside two siblings.

    History records review, cached, audit. On resume the review re-executes
    slowly while audit replays from history behind it; with a cap of 2, audit
    shares the slots with the re-executed review while cached waits for one.
    """
    slow_review_runs.clear()
    joined_inputs.clear()
    executing_siblings[:] = [0, 0]
    recorded_first.clear()
    recorded_first["cached"] = asyncio.Event()
    review = FunctionNode(name="review", func=_slow_review, rerun_on_resume=True)
    audit = FunctionNode(name="audit", func=_quick_audit)
    cached = FunctionNode(name="cached", func=_cached_lookup)
    join = JoinNode(name="join")
    finish = FunctionNode(name="finish", func=_record_joined)
    return Workflow(
        name="outer",
        edges=[
            (START, (audit, review, cached)),
            ((audit, review, cached), join),
            (join, finish),
        ],
        max_concurrency=max_concurrency,
    )
