"""Real ADK coverage for durable identity across nested Workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import ClassVar

import pytest
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import (
    START,
    BaseNode,
    FunctionNode,
    JoinNode,
    Node,
    Workflow,
    node,
)
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import (
    attempt_context_scope,
    current_operation_path,
)
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.route import (
    DurableRouteAdapter,
    adk_operation_boundaries_from_node_path,
    adk_suspension_operation_path_scope,
    has_stable_operation_identity,
)
from agent_engine_sdk_adk.stream import AdkTurn, _append_frontier_response_events
from agent_engine_sdk_adk.suspend import function_response_frontier_content


class _RecordingLlm(BaseLlm):
    observations: ClassVar[list[tuple[str, bool, list[str]]]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        await asyncio.sleep(0.02 if self.model == "left-model" else 0)
        self.observations.append(
            (
                self.model,
                has_stable_operation_identity(),
                [segment.name for segment in current_operation_path().segments],
            )
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=self.model)],
            ),
            partial=False,
        )


class _JoinLlm(BaseLlm):
    observations: ClassVar[list[str]] = []
    operation_paths: ClassVar[list[list[str]]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.observations.append(self.model)
        self.operation_paths.append(
            [segment.name for segment in current_operation_path().segments]
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="frontier joined")],
            ),
            partial=False,
        )


class _ToolCallingLlm(BaseLlm):
    calls: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        type(self).calls += 1
        if type(self).calls == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    id="lookup-1",
                    name="lookup",
                    args={},
                )
            )
        else:
            part = types.Part.from_text(text="done")
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
        )


class _ParallelResumeLlm(BaseLlm):
    observed_response_ids: ClassVar[dict[str, list[set[str]]]] = {}

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        response_ids = {
            part.function_response.id
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response is not None
            and part.function_response.id is not None
        }
        self.observed_response_ids.setdefault(self.model, []).append(response_ids)
        expected_id = f"{self.model}-review"
        if expected_id in response_ids:
            part = types.Part.from_text(text=f"{self.model} resumed")
        else:
            part = types.Part(
                function_call=types.FunctionCall(
                    id=expected_id,
                    name=f"request_{self.model}_review",
                    args={},
                )
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
        )


class _MixedParallelToolsLlm(BaseLlm):
    calls: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        type(self).calls += 1
        if type(self).calls == 1:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(
                        id="lookup-left", name="lookup_left", args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="lookup-right", name="lookup_right", args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="left-review", name="request_left_review", args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="right-review", name="request_right_review", args={}
                    )
                ),
            ]
        else:
            parts = [types.Part.from_text(text="done")]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
        )


class _DynamicNodeCaller(BaseNode):
    children: tuple[BaseNode, ...]

    async def _run_impl(
        self, *, ctx: Context, node_input: object
    ) -> AsyncGenerator[object, None]:
        yield await asyncio.gather(
            *(ctx.run_node(child, node_input) for child in self.children)
        )


_parallel_worker_paths: list[list[str]] = []
_parallel_worker_ordinals: list[list[int]] = []


async def _parallel_function() -> str:
    path = current_operation_path()
    _parallel_worker_paths.append([segment.name for segment in path.segments])
    _parallel_worker_ordinals.append([segment.ordinal for segment in path.segments])
    return "function-worker"


class _ParallelNode(Node):
    async def run_node_impl(
        self, *, ctx: Context, node_input: object
    ) -> AsyncGenerator[object, None]:
        del ctx, node_input
        _parallel_worker_paths.append(
            [segment.name for segment in current_operation_path().segments]
        )
        yield "node-worker"


_replacement_worker_observations: list[tuple[str, int]] = []


class _ReplacementParallelNode(Node):
    async def run_node_impl(
        self, *, ctx: Context, node_input: object
    ) -> AsyncGenerator[object, None]:
        del ctx
        assert isinstance(node_input, str)
        ordinal = current_operation_path().segments[-1].ordinal
        _replacement_worker_observations.append((node_input, ordinal))
        yield node_input


def left_wait() -> RequestInput:
    return RequestInput(
        interrupt_id="left-input",
        message="Review left",
        payload={"route": "left"},
    )


def right_wait() -> RequestInput:
    return RequestInput(
        interrupt_id="right-input",
        message="Review right",
        payload={"route": "right"},
    )


def request_left_review() -> None:
    pass


def request_right_review() -> None:
    pass


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


async def _run_configured_node(configured_node: BaseNode) -> None:
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    workflow = Workflow(name="outer", edges=[(START, configured_node)])
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        async for _event in runner.run_async(
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
        ):
            pass


@pytest.mark.asyncio
async def test_parallel_routes_keep_identity_when_completion_inverts() -> None:
    _RecordingLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    left = LlmAgent(name="left", model=_RecordingLlm(model="left-model"))
    right = LlmAgent(name="right", model=_RecordingLlm(model="right-model"))
    join = JoinNode(name="join")
    workflow = Workflow(
        name="parallel",
        edges=[(START, (left, right)), ((left, right), join)],
    )
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(
        node=workflow,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        events = [
            event
            async for event in runner.run_async(
                user_id="user-1",
                session_id="session-1",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="go")],
                ),
            )
        ]

    assert _RecordingLlm.observations == [
        ("right-model", True, ["agent", "right"]),
        ("left-model", True, ["agent", "left"]),
    ]
    assert {event.branch for event in events if event.branch} == {
        "left@1",
        "right@1",
    }


def test_nested_node_path_installs_every_operation_boundary() -> None:
    assert [
        (boundary.name, boundary.occurrence_key)
        for boundary in adk_operation_boundaries_from_node_path(
            "outer@1/inner@1/review@1"
        )
    ] == [
        ("inner", "outer@1/inner@1"),
        ("review", "outer@1/inner@1/review@1"),
    ]


@pytest.mark.parametrize(
    "node_path",
    ["outer", "@1", "outer@", "outer@1/", "/outer@1"],
)
def test_invalid_workflow_node_path_fails_explicitly(node_path: str) -> None:
    with pytest.raises(RuntimeError, match="invalid workflow node path"):
        adk_operation_boundaries_from_node_path(node_path)


@pytest.mark.asyncio
async def test_concurrent_runtime_created_nodes_fail_before_model_activity() -> None:
    _RecordingLlm.observations = []
    dynamic_child = LlmAgent(
        name="dynamic_child",
        model=_RecordingLlm(model="dynamic-model"),
    )
    caller = _DynamicNodeCaller(
        name="caller",
        children=(dynamic_child, dynamic_child),
    )
    workflow = Workflow(name="outer", edges=[(START, caller)])
    adapter = DurableRouteAdapter(workflow)
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with (
        attempt_context_scope(_attempt()),
        adapter.operation_path_scope(),
        pytest.raises(UnsupportedDurableADKError, match="statically configured"),
    ):
        async for _event in runner.run_async(
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
        ):
            pass

    assert _RecordingLlm.observations == []


@pytest.mark.asyncio
async def test_fresh_tool_context_rejects_runtime_created_node() -> None:
    configured = FunctionNode(name="configured", func=lambda: "configured")
    adapter = DurableRouteAdapter(configured)
    dynamic = LlmAgent(
        name="dynamic",
        model=_RecordingLlm(model="dynamic-model"),
    )
    fresh_tool_context = object.__new__(ToolContext)

    with (
        adapter.operation_path_scope(),
        pytest.raises(UnsupportedDurableADKError, match="statically configured"),
    ):
        await fresh_tool_context.run_node(dynamic)


@pytest.mark.asyncio
async def test_runtime_created_node_cannot_impersonate_configured_worker() -> None:
    configured = _ParallelNode(name="node_worker", parallel_worker=True)
    workflow = Workflow(name="outer", edges=[(START, configured)])
    adapter = DurableRouteAdapter(workflow)
    lookalike = _ParallelNode(name="node_worker")
    fresh_tool_context = object.__new__(ToolContext)
    object.__setattr__(fresh_tool_context, "_node_path", "outer@1/node_worker@1")

    with (
        adapter.operation_path_scope(),
        pytest.raises(UnsupportedDurableADKError, match="statically configured"),
    ):
        await fresh_tool_context.run_node(lookalike)


@pytest.mark.asyncio
async def test_closing_instrumented_node_closes_adk_generator() -> None:
    closed = asyncio.Event()
    observed_stable_identity: list[bool] = []

    class ClosingFunctionNode(FunctionNode):
        async def _run_impl(
            self, *, ctx: Context, node_input: object
        ) -> AsyncGenerator[Event, None]:
            del self, ctx, node_input
            try:
                observed_stable_identity.append(has_stable_operation_identity())
                yield Event(author="configured")
                await asyncio.Event().wait()
            finally:
                closed.set()

    configured = ClosingFunctionNode(name="configured", func=lambda: "configured")
    adapter = DurableRouteAdapter(configured)
    context = SimpleNamespace(node_path="outer@1/configured@1")

    with adapter.operation_path_scope():
        events = configured.run(ctx=context, node_input=None)  # type: ignore[arg-type]
        await events.__anext__()
        await events.aclose()
        assert not has_stable_operation_identity()

    assert closed.is_set()
    assert observed_stable_identity == [True]


@pytest.mark.asyncio
async def test_configured_function_parallel_worker_is_instrumented() -> None:
    _parallel_worker_paths.clear()
    _parallel_worker_ordinals.clear()
    worker = node(
        _parallel_function,
        name="function_worker",
        parallel_worker=True,
    )

    await _run_configured_node(worker)

    assert _parallel_worker_paths == [["agent", "function_worker", "function_worker"]]


@pytest.mark.asyncio
async def test_configured_parallel_worker_preallocates_repeated_child_paths() -> None:
    _parallel_worker_paths.clear()
    _parallel_worker_ordinals.clear()
    inputs = node(lambda: ["one", "two", "three"], name="worker_inputs")
    worker = node(
        _parallel_function,
        name="function_worker",
        parallel_worker=True,
        max_parallel_workers=2,
    )
    workflow = Workflow(
        name="outer",
        edges=[(START, inputs), (inputs, worker)],
    )
    adapter = DurableRouteAdapter(workflow)
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        async for _event in runner.run_async(
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
        ):
            pass

    assert _parallel_worker_paths == [
        ["agent", "function_worker", "function_worker"],
        ["agent", "function_worker", "function_worker"],
        ["agent", "function_worker", "function_worker"],
    ]
    assert sorted(path[-1] for path in _parallel_worker_ordinals) == [1, 2, 3]


@pytest.mark.asyncio
async def test_parallel_worker_preserves_input_identity_when_start_order_inverts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_node = FunctionNode(name="bootstrap", func=lambda: None)
    DurableRouteAdapter(Workflow(name="bootstrap", edges=[(START, bootstrap_node)]))
    original_run_node = Context.run_node
    delayed_input = "one"
    other_input_reached = asyncio.Event()
    allocation_order: list[str] = []

    async def run_node_with_inverted_start(
        context: Context, target_node: object, *args: object, **kwargs: object
    ) -> object:
        node_input = kwargs.get("node_input", args[0] if args else None)
        if (
            isinstance(target_node, _ReplacementParallelNode)
            and node_input == delayed_input
        ):
            await other_input_reached.wait()
        if isinstance(target_node, _ReplacementParallelNode):
            assert isinstance(node_input, str)
            allocation_order.append(node_input)
            other_input_reached.set()
        return await original_run_node(context, target_node, *args, **kwargs)

    monkeypatch.setattr(Context, "run_node", run_node_with_inverted_start)

    async def run_attempt(attempt_id: str) -> tuple[dict[str, int], list[str]]:
        nonlocal other_input_reached
        _replacement_worker_observations.clear()
        allocation_order.clear()
        other_input_reached = asyncio.Event()
        inputs = FunctionNode(
            name="worker_inputs",
            func=lambda: ["one", "two", "three"],
        )
        worker = _ReplacementParallelNode(
            name="claim_worker",
            parallel_worker=True,
            max_parallel_workers=3,
        )
        workflow = Workflow(
            name="outer",
            edges=[(START, inputs), (inputs, worker)],
        )
        adapter = DurableRouteAdapter(workflow)
        service = InMemorySessionService()
        await service.create_session(
            app_name="app",
            user_id="user-1",
            session_id="session-1",
        )
        attempt = _attempt()
        attempt.attempt_id = attempt_id
        runner = Runner(node=workflow, app_name="app", session_service=service)

        with attempt_context_scope(attempt), adapter.operation_path_scope():
            async for _event in runner.run_async(
                user_id="user-1",
                session_id="session-1",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="go")],
                ),
            ):
                pass

        return dict(_replacement_worker_observations), allocation_order.copy()

    first, first_order = await run_attempt("attempt-1")
    delayed_input = "three"
    replacement, replacement_order = await run_attempt("attempt-2")

    assert first == replacement == {"one": 1, "two": 2, "three": 3}
    assert first_order.index("one") > 0
    assert replacement_order.index("three") > 0


@pytest.mark.asyncio
async def test_configured_node_parallel_worker_is_instrumented() -> None:
    _parallel_worker_paths.clear()
    worker = _ParallelNode(name="node_worker", parallel_worker=True)

    await _run_configured_node(worker)

    assert _parallel_worker_paths == [["agent", "node_worker", "node_worker"]]


@pytest.mark.asyncio
async def test_configured_llm_parallel_worker_is_instrumented() -> None:
    _RecordingLlm.observations = []
    worker = LlmAgent(
        name="llm_worker",
        model=_RecordingLlm(model="worker-model"),
        parallel_worker=True,
    )

    await _run_configured_node(worker)

    assert _RecordingLlm.observations == [
        (
            "worker-model",
            True,
            ["agent", "llm_worker", "llm_worker"],
        )
    ]


@pytest.mark.asyncio
async def test_nested_workflow_includes_configured_serial_ancestor() -> None:
    _RecordingLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    left = LlmAgent(name="left", model=_RecordingLlm(model="left-model"))
    right = LlmAgent(name="right", model=_RecordingLlm(model="right-model"))
    inner = Workflow(
        name="inner",
        edges=[
            (START, (left, right)),
            ((left, right), JoinNode(name="inner_join")),
        ],
    )
    outer = Workflow(name="outer", edges=[(START, inner)])
    adapter = DurableRouteAdapter(outer)
    runner = Runner(
        node=outer,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        events = [
            event
            async for event in runner.run_async(
                user_id="user-1",
                session_id="session-1",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="go")],
                ),
            )
        ]

    assert _RecordingLlm.observations == [
        ("right-model", True, ["agent", "inner", "right"]),
        ("left-model", True, ["agent", "inner", "left"]),
    ]
    right_event = next(event for event in events if event.author == "right")
    right_boundaries = adapter.operation_boundaries_for_event(right_event)
    assert right_boundaries is not None
    assert [
        (boundary.name, boundary.occurrence_key) for boundary in right_boundaries
    ] == [
        ("inner", "outer@1/inner@1"),
        ("right", "outer@1/inner@1/right@1"),
    ]


@pytest.mark.asyncio
async def test_nested_tool_inherits_canonical_node_path() -> None:
    _ToolCallingLlm.calls = 0
    tool_paths: list[list[str]] = []

    def lookup() -> str:
        tool_paths.append(
            [segment.name for segment in current_operation_path().segments]
        )
        return "found"

    review = LlmAgent(
        name="review",
        model=_ToolCallingLlm(model="tool-calling-model"),
        tools=[lookup],
    )
    inner = Workflow(name="inner", edges=[(START, review)])
    outer = Workflow(name="outer", edges=[(START, inner)])
    adapter = DurableRouteAdapter(outer)
    # A new durable attempt may construct another adapter over the same ADK
    # objects; instrumentation must remain idempotent.
    DurableRouteAdapter(outer)
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    runner = Runner(node=outer, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        async for _event in runner.run_async(
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
        ):
            pass

    assert tool_paths == [["agent", "inner", "review"]]


def test_equal_leaf_names_under_different_workflows_are_not_ambiguous() -> None:
    first = Workflow(
        name="first",
        edges=[
            (
                START,
                LlmAgent(name="review", model=_RecordingLlm(model="first-model")),
            )
        ],
    )
    second = Workflow(
        name="second",
        edges=[
            (
                START,
                LlmAgent(name="review", model=_RecordingLlm(model="second-model")),
            )
        ],
    )
    root = Workflow(
        name="root",
        edges=[
            (START, (first, second)),
            ((first, second), JoinNode(name="root_join")),
        ],
    )

    DurableRouteAdapter(root)
    first_review = adk_operation_boundaries_from_node_path("root@1/first@1/review@1")[
        -1
    ]
    second_review = adk_operation_boundaries_from_node_path("root@1/second@1/review@1")[
        -1
    ]

    assert first_review.name == second_review.name == "review"
    assert first_review.occurrence_key != second_review.occurrence_key


def test_wait_event_uses_canonical_node_path() -> None:
    left = LlmAgent(name="left", model=_RecordingLlm(model="left-model"))
    inner = Workflow(name="inner", edges=[(START, left)])
    outer = Workflow(name="outer", edges=[(START, inner)])
    adapter = DurableRouteAdapter(outer)
    event = Event(
        author="inner",
        node_info=NodeInfo(path="outer@1/inner@2/left@3"),
    )

    boundaries = adapter.operation_boundaries_for_event(event)
    assert boundaries is not None
    assert [(boundary.name, boundary.occurrence_key) for boundary in boundaries] == [
        ("inner", "outer@1/inner@2"),
        ("left", "outer@1/inner@2/left@3"),
    ]


@pytest.mark.asyncio
async def test_real_workflow_resumes_parallel_frontier_and_reaches_join() -> None:
    _JoinLlm.observations = []
    _JoinLlm.operation_paths = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    left = FunctionNode(name="left", func=left_wait)
    right = FunctionNode(name="right", func=right_wait)
    join = JoinNode(name="join")
    joined = LlmAgent(name="joined", model=_JoinLlm(model="joined-model"))
    inner = Workflow(
        name="parallel",
        edges=[
            (START, (left, right)),
            ((left, right), join),
        ],
    )
    workflow = Workflow(name="outer", edges=[(START, inner), (inner, joined)])
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(
        node=workflow,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()):
        first = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
            route_adapter=adapter,
        )
        assert [event async for event in first.stream(emit_tokens=True)] == []
        frontier = first.wait_frontier()
        with adk_suspension_operation_path_scope(
            frontier[0].provenance.operation_boundaries
        ):
            assert [segment.name for segment in current_operation_path().segments] == [
                "agent",
                "parallel",
                "left",
            ]
        second = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=function_response_frontier_content(
                frontier,
                [{"answer": "L"}, {"answer": "R"}],
            ),
            route_adapter=adapter,
        )
        events = [event async for event in second.stream(emit_tokens=True)]

    assert [wait.function_call_id for wait in frontier] == [
        "left-input",
        "right-input",
    ]
    assert [
        [boundary.name for boundary in wait.provenance.operation_boundaries]
        for wait in frontier
    ] == [["parallel", "left"], ["parallel", "right"]]
    assert second.wait_frontier() == ()
    assert _JoinLlm.observations == ["joined-model"]
    assert _JoinLlm.operation_paths == [["agent", "joined"]]
    assert [event.data["content"] for event in events] == [
        "frontier joined",
    ]


@pytest.mark.asyncio
async def test_real_parallel_llm_frontier_routes_each_resume_response() -> None:
    _ParallelResumeLlm.observed_response_ids = {}
    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    left = LlmAgent(
        name="left",
        model=_ParallelResumeLlm(model="left"),
        tools=[LongRunningFunctionTool(request_left_review)],
    )
    right = LlmAgent(
        name="right",
        model=_ParallelResumeLlm(model="right"),
        tools=[LongRunningFunctionTool(request_right_review)],
    )
    join = JoinNode(name="join")
    workflow = Workflow(
        name="parallel",
        edges=[(START, (left, right)), ((left, right), join)],
    )
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        first = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
            route_adapter=adapter,
        )
        assert [event async for event in first.stream(emit_tokens=True)] == []
        frontier = first.wait_frontier()
        session = await service.get_session(
            app_name="app", user_id="user-1", session_id="session-1"
        )
        assert session is not None
        invocation_id = await _append_frontier_response_events(
            session,
            session_service=service,
            suspends=frontier,
            results=({"decision": "approved"}, {"decision": "approved"}),
        )
        second = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=None,
            invocation_id=invocation_id,
            route_adapter=adapter,
        )
        assert [event async for event in second.stream(emit_tokens=True)]

    assert _ParallelResumeLlm.observed_response_ids["left"][-1] == {"left-review"}
    assert _ParallelResumeLlm.observed_response_ids["right"][-1] == {"right-review"}


@pytest.mark.asyncio
async def test_real_llm_frontier_with_regular_and_long_running_tools_resumes() -> None:
    _MixedParallelToolsLlm.calls = 0

    def lookup_left() -> str:
        return "left"

    def lookup_right() -> str:
        return "right"

    service = InMemorySessionService()
    await service.create_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )
    agent = LlmAgent(
        name="review",
        model=_MixedParallelToolsLlm(model="mixed-tools"),
        tools=[
            lookup_left,
            lookup_right,
            LongRunningFunctionTool(request_left_review),
            LongRunningFunctionTool(request_right_review),
        ],
    )
    workflow = Workflow(name="outer", edges=[(START, agent)])
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        first = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="go")],
            ),
            route_adapter=adapter,
        )
        assert [event async for event in first.stream(emit_tokens=True)] == []
        frontier = first.wait_frontier()
        session = await service.get_session(
            app_name="app", user_id="user-1", session_id="session-1"
        )
        assert session is not None
        invocation_id = await _append_frontier_response_events(
            session,
            session_service=service,
            suspends=frontier,
            results=({"decision": "approved"}, {"decision": "approved"}),
        )
        second = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=None,
            invocation_id=invocation_id,
            route_adapter=adapter,
        )
        events = [event async for event in second.stream(emit_tokens=True)]

    assert [wait.function_call_id for wait in frontier] == [
        "left-review",
        "right-review",
    ]
    assert second.wait_frontier() == ()
    assert [event.data["content"] for event in events] == ["done"]
