"""
In-process graph testing helpers (AP-2701, extended in AP-4343).

These let a tenant test their compiled LangGraph graph directly — no AER, no
Orchestration Engine, no real LLM provider — by running under
``RUNNER_MODE=tool`` and swapping in fake LLMs before the graph is built.
``FakeChatModel`` scripts each model's responses, including tool calls, and
``assert_tool_calls`` verifies the names, arguments, and order of the tools
the graph actually invoked; multi-model agents give each named ``llm_id``
its own test model via ``build_test_graph(..., llms={...})``. Two build
helpers cover the two levels tests care about: ``build_test_graph`` returns
the raw compiled graph (routing/prompt/tool-contract assertions on raw
messages), while ``build_test_agent`` returns the framework adapter the AER
serves — its ``stream()`` applies the app's custom output parser, so the
user-facing event format (``StreamEvent.custom_event``) is assertable too.

This is deliberately low-fidelity: tools and the LLM run in-process with no
policy/guardrails/streaming, matching TOOL mode's existing raw-object
contract (``App.llm()``/``App.get_tools()`` return unwrapped objects,
``App.checkpointer()`` is ``None``).

This is "Layer 1" of a two-layer testing strategy (AP-2701): fast, per-PR,
in-process tests here, catching routing/prompt/tool-contract/state
regressions; a small number of "Layer 2" tests against a real deployed
staging workspace validate what Layer 1 cannot see — policy/guardrails,
typed streaming events, checkpointer persistence, interrupt/HITL flows,
secure-LLM proxying. A higher-fidelity fake-OE emulator was considered and
rejected: it would mean maintaining a second OE implementation that drifts
from the real one, for coverage the two-layer split already gets more
cheaply and more honestly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from agent_engine_sdk_langgraph.runtime import App
from agent_engine_runner_shared.context import (
    ContextTokens,
    clear_execution_context,
    set_execution_context,
)
from agent_engine_runner_shared.hooks import entrypoint_scope, reset_llm_registry
from agent_engine_runner_shared.utils import RuntimeMode

if TYPE_CHECKING:
    from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent

__all__ = [
    "ExpectedToolCall",
    "FakeChatModel",
    "assert_tool_calls",
    "build_test_agent",
    "build_test_graph",
    "execution_context",
]

#: Sentinel id ``App.llm()`` registers unnamed LLMs under (runtime.py).
_DEFAULT_LLM_ID = "__default__"


#: A scripted ``FakeChatModel`` response. Three forms:
#:
#: - ``str`` — a plain-text reply, wrapped in ``AIMessage(content=...)``.
#: - ``AIMessage`` — returned verbatim; script tool calls natively via
#:   ``AIMessage(content="", tool_calls=[{"name": ..., "args": ..., "id": ...}])``.
#: - ``dict`` — shorthand for a tool-calling reply:
#:   ``{"tool_calls": [{"name": "get_weather", "args": {"city": "Boston"}}]}``,
#:   with optional ``"content"`` text and optional per-call ``"id"`` (ids are
#:   auto-generated deterministically — ``call_1``, ``call_2``, ... — when
#:   omitted, which ``ToolNode`` requires).
ResponseSpec = str | AIMessage | dict[str, Any]


class FakeChatModel(BaseChatModel):
    """A canned-response ``BaseChatModel`` for in-process graph tests.

    The base class's ``bind_tools`` raises ``NotImplementedError``; tenant
    agent code that calls it (to bind tool schemas before invoking the LLM)
    would otherwise crash before the graph even runs. This override returns
    ``self`` so binding is a no-op — the fake ignores tool schemas and always
    returns the next scripted response.

    Responses are consumed one per invocation, sticking on the last once the
    script runs out — so a tool-calling script reads naturally::

        FakeChatModel(responses=[
            {"tool_calls": [{"name": "get_weather", "args": {"city": "Boston"}}]},
            "The weather in Boston is sunny.",  # after ToolNode loops back
        ])

    Each response may be a ``str``, an ``AIMessage`` (returned verbatim), or a
    ``{"tool_calls": [...]}`` dict — see ``ResponseSpec``. Tool calls emitted
    this way are executed by the graph's real ``ToolNode`` and recorded in the
    graph's message history, where ``assert_tool_calls`` can verify their
    names, arguments, and order.
    """

    responses: list[ResponseSpec] = ["ok"]
    _call_count: int = 0
    _tool_call_seq: int = 0
    _used_tool_call_ids: set[str]

    def __init__(self, responses: list[ResponseSpec] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if responses is not None:
            if not responses:
                raise ValueError("FakeChatModel responses must be a non-empty list")
            for spec in responses:
                _validate_response_spec(spec)
            _validate_script_tool_call_ids(responses)
            self.responses = responses
        self._call_count = 0
        self._tool_call_seq = 0
        self._used_tool_call_ids = set()

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        return self

    def _next_tool_call_id(self) -> str:
        # Skip any id already emitted or pinned by the test author: duplicate
        # tool-call ids would make the resulting ToolMessages impossible to
        # correlate unambiguously with their originating calls.
        while True:
            self._tool_call_seq += 1
            candidate = f"call_{self._tool_call_seq}"
            if candidate not in self._used_tool_call_ids:
                self._used_tool_call_ids.add(candidate)
                return candidate

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        spec = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=self._to_message(spec))])

    def _to_message(self, spec: ResponseSpec) -> AIMessage:
        if isinstance(spec, AIMessage):
            # Register passthrough ids so later auto-generated ids skip them
            # (uniqueness itself is validated at construction).
            for call in spec.tool_calls:
                call_id = call.get("id")
                if call_id:
                    self._used_tool_call_ids.add(call_id)
            return spec
        if isinstance(spec, str):
            return AIMessage(content=spec)
        # Register this response's pinned ids BEFORE generating any auto ids,
        # so an auto id can never collide with a pinned one regardless of the
        # order the calls appear in within the response.
        for call in spec["tool_calls"]:
            if call.get("id") is not None:
                self._used_tool_call_ids.add(call["id"])
        tool_calls = []
        for call in spec["tool_calls"]:
            tool_calls.append(
                {
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "id": call.get("id") or self._next_tool_call_id(),
                }
            )
        return AIMessage(content=spec.get("content", ""), tool_calls=tool_calls)


def _validate_response_spec(spec: ResponseSpec) -> None:
    """Fail fast at construction on a malformed scripted response.

    A bad entry would otherwise surface as a ``KeyError``/``TypeError`` deep
    inside a graph run, far from the test that wrote it.
    """
    if isinstance(spec, (str, AIMessage)):
        return
    if not isinstance(spec, dict):
        raise TypeError(
            "FakeChatModel responses must be str, AIMessage, or "
            f"{{'tool_calls': [...]}} dicts; got {type(spec).__name__}: {spec!r}"
        )
    calls = spec.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(
            "FakeChatModel dict responses must contain a non-empty "
            f"'tool_calls' list; got: {spec!r}"
        )
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError(
                "Each scripted tool call must be a dict with a string 'name' "
                f"(optional 'args' dict and 'id' string); got: {call!r}"
            )
        if "args" in call and not isinstance(call["args"], dict):
            raise ValueError(f"Scripted tool call 'args' must be a dict; got: {call!r}")
        if "id" in call and not isinstance(call["id"], str):
            raise ValueError(f"Scripted tool call 'id' must be a string; got: {call!r}")


def _validate_script_tool_call_ids(responses: list[ResponseSpec]) -> None:
    """Fail at construction if any tool-call id would be used twice across the
    whole script — pinned vs pinned, or pinned vs an auto-generated id.

    Auto ids are deterministic (``call_1``, ``call_2``, ... in script order,
    skipping used ones), so the full assignment can be simulated here with the
    same algorithm the runtime uses — a collision surfaces as a clear
    construction error instead of two AIMessages silently sharing an id at
    runtime, which would make the graph's ToolMessages impossible to correlate
    with their originating calls. Covers dict specs and AIMessage passthroughs
    alike.
    """
    used: set[str] = set()
    seq = 0

    def scripted_calls(spec: ResponseSpec) -> list[Mapping[str, Any]]:
        if isinstance(spec, dict):
            return spec["tool_calls"]
        if isinstance(spec, AIMessage):
            # ToolCall is a TypedDict; pyright doesn't treat it as a Mapping
            # subtype, but it supports the .get access used below.
            return cast("list[Mapping[str, Any]]", spec.tool_calls)
        return []

    for index, spec in enumerate(responses):
        calls = scripted_calls(spec)
        # Pinned ids register before any auto id is generated within a
        # response — mirroring _to_message, so the simulation and the runtime
        # can never disagree.
        for call in calls:
            pinned = call.get("id")
            if pinned is not None:
                if pinned in used:
                    raise ValueError(
                        f"FakeChatModel script reuses tool-call id {pinned!r} "
                        f"(response {index}); ids must be unique across the "
                        "whole script for ToolMessages to correlate with "
                        "their calls."
                    )
                used.add(pinned)
        for call in calls:
            if call.get("id") is None:
                while True:
                    seq += 1
                    if f"call_{seq}" not in used:
                        used.add(f"call_{seq}")
                        break


@dataclass(frozen=True)
class ExpectedToolCall:
    """One expected tool invocation, in order, for ``assert_tool_calls``.

    ``args=None`` asserts only the tool name; a dict asserts a *subset* —
    every listed key must be present in the actual call with an equal value,
    and unlisted actual keys are ignored (so tests can pin the arguments they
    care about without breaking on incidental ones).
    """

    name: str
    args: dict[str, Any] | None = None


def _recorded_tool_calls(
    result: Mapping[str, Any] | Sequence[BaseMessage],
) -> list[tuple[str, dict[str, Any]]]:
    """Flatten a graph result (or message list) into the ordered sequence of
    ``(tool_name, args)`` the model requested — message order, then
    within-message order."""
    messages: Sequence[BaseMessage]
    if isinstance(result, Mapping):
        raw = result.get("messages")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(
                "assert_tool_calls expected a graph result with a 'messages' "
                f"list or a message sequence; got mapping keys {list(result)}"
            )
        messages = raw
    else:
        messages = result
    # Reject non-message elements rather than silently filtering them out:
    # passing e.g. a list of plain dicts would otherwise report "the graph
    # made 0 tool calls", misdirecting the reader away from the type error.
    if isinstance(messages, (str, bytes)) or any(
        not isinstance(message, BaseMessage) for message in messages
    ):
        raise TypeError(
            "assert_tool_calls expected a sequence of LangChain BaseMessage "
            f"objects; got {type(messages).__name__} with a non-message "
            "element"
        )
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(
                (call["name"], dict(call["args"])) for call in message.tool_calls
            )
    return calls


def assert_tool_calls(
    result: Mapping[str, Any] | Sequence[BaseMessage],
    expected: Sequence[ExpectedToolCall],
) -> None:
    """Assert the exact ordered sequence of tool calls a graph run made.

    ``result`` is the dict ``graph.invoke(...)`` returned (or its ``messages``
    list). ``expected`` is matched in order against every tool call recorded
    in the run's ``AIMessage`` history — one entry per call, including calls
    made on later loop iterations after tool results come back. An empty
    ``expected`` asserts no tools were called at all.

    Raises:
        AssertionError: With a side-by-side expected/actual listing that
            pinpoints the first divergence (wrong count, wrong tool, or the
            specific argument keys that differ).
    """
    actual = _recorded_tool_calls(result)

    def _fmt_expected(i: int, call: ExpectedToolCall) -> str:
        args = "any args" if call.args is None else f"args={call.args!r}"
        return f"  {i}. {call.name}({args})"

    def _fmt_actual(i: int, call: tuple[str, dict[str, Any]]) -> str:
        return f"  {i}. {call[0]}(args={call[1]!r})"

    def _fail(reason: str) -> None:
        listing = [
            f"assert_tool_calls failed: {reason}",
            "",
            f"expected ({len(expected)} call(s)):",
            *(_fmt_expected(i, c) for i, c in enumerate(expected)),
            f"actual ({len(actual)} call(s)):",
            *(_fmt_actual(i, c) for i, c in enumerate(actual)),
        ]
        raise AssertionError("\n".join(listing))

    if len(actual) != len(expected):
        _fail(f"expected {len(expected)} tool call(s), the graph made {len(actual)}")

    for i, (exp, (name, args)) in enumerate(zip(expected, actual, strict=True)):
        if name != exp.name:
            _fail(
                f"call {i} was {name!r}, expected {exp.name!r} "
                f"(call order differs at position {i})"
            )
        if exp.args is None:
            continue
        diff = {
            key: (value, args.get(key, "<missing>"))
            for key, value in exp.args.items()
            if key not in args or args[key] != value
        }
        if diff:
            detail = ", ".join(
                f"{key}: expected {want!r}, got {got!r}"
                for key, (want, got) in diff.items()
            )
            _fail(f"call {i} to {name!r} had mismatched args — {detail}")


def build_test_graph(
    app: App,
    llm: BaseChatModel | None = None,
    *,
    llms: dict[str, BaseChatModel] | None = None,
) -> Any:
    """Build the raw compiled graph behind ``app``, substituting test models
    for the real LLMs the tenant's entrypoint constructs.

    Single-model agents pass ``llm``; every ``app.llm()`` call returns it::

        graph = build_test_graph(app, FakeChatModel(responses=["ok"]))

    Multi-model agents pass ``llms`` — a map from the ``llm_id`` each
    ``app.llm(..., llm_id=...)`` call registers under to the test model that
    node should run with (``"__default__"`` is the id for an unnamed
    ``app.llm()`` call). Mapping is strict: if the graph registers an id with
    no entry in ``llms``, the build raises immediately listing the missing
    and provided ids, rather than silently running a node with the wrong
    model::

        graph = build_test_graph(app, llms={
            "primary": FakeChatModel(responses=[...]),
            "fast": FakeChatModel(responses=[...]),
        })

    Note the map only guarantees every registered ``llm_id`` has an entry —
    it does not itself verify a node received its *intended* model (a
    swapped entry goes undetected); assert on each model's scripted
    responses to confirm the wiring.

    Requires ``RUNNER_MODE=tool`` to already be set (before the tenant's app
    module was imported — ``TenantRuntime`` reads it at construction, so
    setting it here would be too late) and ``app.entrypoint(...)`` to have
    already run, i.e. ``app._builder_fn`` is populated. Resets the LLM
    registry and wraps the builder call in ``entrypoint_scope()`` — the same
    two steps ``App.get_agent()`` takes before evaluating the entrypoint, so
    repeated calls (e.g. once per test) don't collide on a stale ``llm_id``
    from a previous build.

    Args:
        app: The tenant's App instance.
        llm: A ``BaseChatModel`` to substitute for every real LLM the
            tenant's entrypoint constructs — typically a ``FakeChatModel``.
            Mutually exclusive with ``llms``.
        llms: Per-``llm_id`` test models for multi-model agents. Mutually
            exclusive with ``llm``.

    Returns:
        The raw compiled graph (``app._builder_fn()``'s return value), not
        wrapped in a framework agent class.

    Raises:
        ValueError: If neither or both of ``llm``/``llms`` are given.
        RuntimeError: If ``app`` was not constructed in TOOL mode, the app
            has no entrypoint registered, or (with ``llms``) the graph
            registers an ``llm_id`` with no matching entry.
    """
    with _test_llm_substitution(app, llm, llms, helper="build_test_graph"):
        builder_fn = app._builder_fn
        if builder_fn is None:
            raise RuntimeError(
                "app has no entrypoint registered — call @app.entrypoint before "
                "build_test_graph."
            )
        # Mirrors get_agent()'s own fail-fast: the parser is inert unless
        # features.use_custom_parser is true, but opting in without a registered
        # parser is a misconfiguration that should surface here too — otherwise a
        # Layer 1 test would build and run the raw graph silently (the parser is
        # never consulted for the raw graph regardless), masking a config error
        # that fails immediately under real AER execution.
        use_custom_parser = app.agent_config.feature_enabled("use_custom_parser")
        if use_custom_parser and app._output_parser_cls is None:
            raise RuntimeError(
                "agent.yaml features.use_custom_parser is true but no output "
                "parser is registered; decorate one with @app.output_parser."
            )
        reset_llm_registry()  # mirrors get_agent(): a fresh registry per build
        with entrypoint_scope():
            return builder_fn()


def build_test_agent(
    app: App,
    llm: BaseChatModel | None = None,
    *,
    llms: dict[str, BaseChatModel] | None = None,
) -> LangGraphBaseAgent:
    """Build the framework-wrapped agent behind ``app`` — what
    ``app.get_agent()`` returns — with test models substituted, so tests can
    exercise the same invoke/stream surface the AER serves in production.

    Where ``build_test_graph`` returns the raw compiled graph (the parser is
    never consulted), this returns the ``LangGraphBaseAgent`` adapter, whose
    ``stream()`` dual-runs the app's custom output parser: parser yields land
    on ``StreamEvent.custom_event`` exactly as the Gateway would relay them.
    That makes the parser's output format — not just the graph's raw
    messages — assertable in a headless test::

        agent = build_test_agent(app, FakeChatModel(responses=["Hello!"]))
        events = [
            event
            async for event in agent.stream(
                RequestContext(session_id="s1", user_id="u1"),
                AgentInput(payload={"message": "hi"}),
            )
        ]
        custom = [e.custom_event for e in events if e.custom_event is not None]
        assert custom == [...]

    Two substitutions make the adapter work without the platform:

    - ``app.llm(...)`` returns the test model(s), exactly as in
      ``build_test_graph`` (same ``llm``/``llms`` arguments and strict
      per-``llm_id`` semantics).
    - ``app.checkpointer()`` returns a fresh in-memory saver instead of
      ``None``. AER always runs with a real checkpointer, and the adapter's
      ``invoke()`` path calls ``aget_state`` (which raises without one) for
      suspend detection — in-memory keeps that faithful with no Mongo
      dependency. Durable-execution plumbing stays inert (no OE attempt
      context exists in a test).

    Everything else about TOOL mode is unchanged: no policy/guardrails,
    tools run raw and in-process, and there is no OE — durable
    suspend/resume and interrupt flows remain Layer 2 territory.

    Args:
        app: The tenant's App instance.
        llm: A ``BaseChatModel`` to substitute for every real LLM —
            typically a ``FakeChatModel``. Mutually exclusive with ``llms``.
        llms: Per-``llm_id`` test models for multi-model agents. Mutually
            exclusive with ``llm``. Guarantees every registered ``llm_id``
            has an entry; it does not verify a node received its intended
            model — assert on each model's scripted responses to confirm
            the wiring.

    Returns:
        The ``LangGraphBaseAgent`` produced by ``app.get_agent()``.

    Raises:
        ValueError: If neither or both of ``llm``/``llms`` are given.
        RuntimeError: If ``app`` was not constructed in TOOL mode, the app
            has no entrypoint registered, or (with ``llms``) the graph
            registers an ``llm_id`` with no matching entry.
    """
    with _test_llm_substitution(app, llm, llms, helper="build_test_agent"):
        original_checkpointer = app.checkpointer

        def _memory_checkpointer() -> Any:
            return InMemorySaver()

        app.checkpointer = _memory_checkpointer
        try:
            # get_agent() does its own entrypoint/use_custom_parser fail-fasts
            # plus reset_llm_registry() + entrypoint_scope() around the build.
            return app.get_agent()
        finally:
            app.checkpointer = original_checkpointer


@contextmanager
def _test_llm_substitution(
    app: App,
    llm: BaseChatModel | None,
    llms: dict[str, BaseChatModel] | None,
    helper: str,
) -> Iterator[None]:
    """Shared guard + ``app.llm`` monkeypatch for ``build_test_graph`` and
    ``build_test_agent``: validates the exactly-one-of argument contract and
    the app's constructed TOOL mode, then swaps in the test model(s) for the
    duration of the build, restoring ``app.llm`` afterwards."""
    if (llm is None) == (llms is None):
        raise ValueError(
            f"{helper} requires exactly one of llm= (substitute one model "
            "everywhere) or llms= (a per-llm_id model map)."
        )
    # Check the app's own constructed mode, not just the RUNNER_MODE env var:
    # TenantRuntime reads RUNNER_MODE once at App construction, so a test that
    # mutates the env var after creating the app would pass an env-only guard
    # while app.llm() still returns SecureWrappedLLM (AER behaviour). Guard on
    # the mode app.llm() itself branches on (runtime.py: App.llm).
    if app._runtime.mode != RuntimeMode.TOOL:
        raise RuntimeError(
            f"{helper} requires an App constructed in TOOL mode "
            f"(got {app._runtime.mode.value}). Set RUNNER_MODE=tool before the "
            "tenant's app module is imported — App reads it once at "
            "construction, so changing it afterwards has no effect."
        )

    original_llm = app.llm
    # Captured under a different name to avoid shadowing it inside the
    # closure: _fake_llm's first parameter must be named `llm` to match
    # App.llm's real signature — pyright checks the `app.llm = _fake_llm`
    # assignment against the original method's parameter names
    # (reportAttributeAccessIssue), and naming both `llm` would collide.
    single_llm = llm

    def _fake_llm(llm: BaseChatModel, llm_id: str | None = None) -> BaseChatModel:
        del llm  # tenant's real LLM is discarded; substitute the fake
        if llms is None:
            # The exactly-one-of guard above guarantees single_llm is not
            # None whenever llms is None.
            assert single_llm is not None
            return original_llm(single_llm, llm_id=llm_id)
        resolved_id = llm_id if llm_id is not None else _DEFAULT_LLM_ID
        if resolved_id not in llms:
            raise RuntimeError(
                f"{helper}: the graph registered llm_id {resolved_id!r} "
                f"but llms= has no model for it. Provided ids: "
                f"{sorted(llms)}. Add an entry for {resolved_id!r} (use "
                f"{_DEFAULT_LLM_ID!r} for an unnamed app.llm() call) or pass "
                "llm= to substitute one model everywhere."
            )
        return original_llm(llms[resolved_id], llm_id=llm_id)

    app.llm = _fake_llm
    try:
        yield
    finally:
        app.llm = original_llm


@contextmanager
def execution_context(
    execution_id: str | None = None,
    oe_url: str = "http://test-oe.invalid",
    **kwargs: Any,
) -> Iterator[None]:
    """Optional convenience wrapper around
    ``agent_engine_runner_shared.context.set_execution_context``/``clear_execution_context``
    for tests that need context vars populated (e.g. user_id/session_id) but
    have no real SecureToolWrapper or OE — appropriate only for TOOL-mode
    tests, where ``get_tools()``/``llm()`` never dereference ``wrapper``.
    """
    tokens: ContextTokens = set_execution_context(
        execution_id=execution_id or f"test-{uuid4().hex[:12]}",
        wrapper=None,
        oe_url=oe_url,
        **kwargs,
    )
    try:
        yield
    finally:
        clear_execution_context(tokens)
