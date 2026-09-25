"""Unit tests for agent_engine_sdk_langgraph.testing (in-process graph tests,
extended with scripted tool calls, tool-call assertions, and
per-llm_id multi-model substitution)."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_engine_sdk_langgraph.runtime import App
from agent_engine_sdk_langgraph.testing import (
    ExpectedToolCall,
    FakeChatModel,
    assert_tool_calls,
    build_test_agent,
    build_test_graph,
    execution_context,
)

# Real tenant code passes an actual BaseChatModel to app.llm(); build_test_graph
# discards it before it's ever invoked, so a plain sentinel object stands in for
# it here — cast rather than typed for real, since nothing about it is ever used
# as an LLM.
_UNUSED_REAL_LLM = cast(BaseChatModel, object())


class TestFakeChatModel:
    def test_bind_tools_is_a_noop_returning_self(self):
        fake = FakeChatModel()
        bound = fake.bind_tools([{"name": "get_weather"}])
        assert bound is fake

    def test_responses_cycle_then_stick_on_last(self):
        fake = FakeChatModel(responses=["first", "second"])
        msgs = [HumanMessage(content="hi")]
        assert fake.invoke(msgs).content == "first"
        assert fake.invoke(msgs).content == "second"
        assert fake.invoke(msgs).content == "second"

    def test_default_response(self):
        fake = FakeChatModel()
        assert fake.invoke([HumanMessage(content="hi")]).content == "ok"

    def test_empty_responses_rejected(self):
        # A [] response list would otherwise make _generate index responses[-1]
        # and silently return the "last" of nothing / IndexError deep in a
        # graph run; fail fast and clearly at construction instead.
        with pytest.raises(ValueError, match="non-empty"):
            FakeChatModel(responses=[])

    def test_ai_message_passthrough_returned_verbatim(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "Boston"}, "id": "pinned"}
            ],
        )
        fake = FakeChatModel(responses=[msg])
        assert fake.invoke([HumanMessage(content="hi")]) is msg

    def test_dict_response_produces_tool_calls_with_deterministic_auto_ids(self):
        fake = FakeChatModel(
            responses=[
                {"tool_calls": [{"name": "a", "args": {"x": 1}}, {"name": "b"}]},
                {"content": "thinking", "tool_calls": [{"name": "c"}]},
            ]
        )
        first = fake.invoke([HumanMessage(content="hi")])
        assert first.content == ""
        # ToolCall dicts also carry a "type" key; compare the fields we script.
        assert [(c["name"], c["args"], c["id"]) for c in first.tool_calls] == [
            ("a", {"x": 1}, "call_1"),
            ("b", {}, "call_2"),
        ]
        second = fake.invoke([HumanMessage(content="and?")])
        # Ids stay unique and deterministic across responses within one model.
        assert [(c["name"], c["args"], c["id"]) for c in second.tool_calls] == [
            ("c", {}, "call_3")
        ]
        assert second.content == "thinking"

    def test_dict_response_honors_explicit_tool_call_id(self):
        fake = FakeChatModel(
            responses=[{"tool_calls": [{"name": "a", "id": "custom"}]}]
        )
        assert fake.invoke([HumanMessage(content="hi")]).tool_calls[0]["id"] == "custom"

    def test_mixed_pinned_and_auto_ids_never_collide(self):
        # Regression: an id pinned to call_1 next to an id-less call must not
        # assign call_1 twice — duplicate tool-call ids make ToolMessages
        # impossible to correlate. Both orders must be safe.
        pinned_first = FakeChatModel(
            responses=[{"tool_calls": [{"name": "a", "id": "call_1"}, {"name": "b"}]}]
        )
        ids = [
            c["id"]
            for c in pinned_first.invoke([HumanMessage(content="hi")]).tool_calls
        ]
        assert ids == ["call_1", "call_2"]

        pinned_second = FakeChatModel(
            responses=[{"tool_calls": [{"name": "a"}, {"name": "b", "id": "call_2"}]}]
        )
        ids = [
            c["id"]
            for c in pinned_second.invoke([HumanMessage(content="hi")]).tool_calls
        ]
        assert ids == ["call_1", "call_2"]

    def test_duplicate_pinned_ids_rejected_at_construction(self):
        with pytest.raises(ValueError, match="reuses tool-call id 'dup'"):
            FakeChatModel(
                responses=[
                    {"tool_calls": [{"name": "a", "id": "dup"}]},
                    {"tool_calls": [{"name": "b", "id": "dup"}]},
                ]
            )

    def test_pinned_id_colliding_with_earlier_auto_id_rejected_at_construction(self):
        # Response 1 auto-generates call_1 at runtime; response 2 pins
        # call_1. Deterministic auto ids mean this is knowable — and
        # rejected — at construction, not silently duplicated at runtime.
        with pytest.raises(
            ValueError, match="reuses tool-call id 'call_1'.*response 1"
        ):
            FakeChatModel(
                responses=[
                    {"tool_calls": [{"name": "a"}]},
                    {"tool_calls": [{"name": "b", "id": "call_1"}]},
                ]
            )

    def test_ai_message_passthrough_ids_join_the_uniqueness_check(self):
        # Pinned ids inside AIMessage passthroughs count too — colliding with
        # a dict spec's pinned id fails at construction...
        with pytest.raises(ValueError, match="reuses tool-call id 'shared'"):
            FakeChatModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "a", "args": {}, "id": "shared"}],
                    ),
                    {"tool_calls": [{"name": "b", "id": "shared"}]},
                ]
            )
        # ...and a passthrough's pinned id is skipped by later auto
        # generation, so this script assigns call_2, not a duplicate call_1.
        fake = FakeChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "a", "args": {}, "id": "call_1"}]
                ),
                {"tool_calls": [{"name": "b"}]},
            ]
        )
        fake.invoke([HumanMessage(content="hi")])
        second = fake.invoke([HumanMessage(content="again")])
        assert second.tool_calls[0]["id"] == "call_2"

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ([42], "str, AIMessage"),  # wrong entry type
            ([{"content": "no calls"}], "tool_calls"),  # dict without tool_calls
            ([{"tool_calls": []}], "non-empty"),  # empty tool_calls
            ([{"tool_calls": [{"args": {}}]}], "name"),  # call without a name
            ([{"tool_calls": [{"name": "a", "args": "x"}]}], "args.*dict"),  # bad args
            ([{"tool_calls": [{"name": "a", "id": []}]}], "id.*string"),  # bad id type
        ],
    )
    def test_malformed_response_specs_rejected_at_construction(self, bad, match):
        with pytest.raises((TypeError, ValueError), match=match):
            FakeChatModel(responses=bad)


class TestAssertToolCalls:
    def _result(self, *calls: tuple[str, dict]) -> dict:
        return {
            "messages": [
                HumanMessage(content="go"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": f"call_{i}"}
                        for i, (name, args) in enumerate(calls)
                    ],
                ),
            ]
        }

    def test_passes_on_exact_ordered_match(self):
        result = self._result(("get_weather", {"city": "Boston"}), ("get_time", {}))
        assert_tool_calls(
            result,
            [
                ExpectedToolCall("get_weather", {"city": "Boston"}),
                ExpectedToolCall("get_time"),
            ],
        )

    def test_args_are_subset_matched(self):
        result = self._result(("get_weather", {"city": "Boston", "units": "F"}))
        assert_tool_calls(result, [ExpectedToolCall("get_weather", {"city": "Boston"})])
        # Explicit args={} is a deliberate no-op diff over zero keys —
        # equivalent to args=None. Pinned by test so the equivalence can't
        # silently diverge.
        assert_tool_calls(result, [ExpectedToolCall("get_weather", {})])

    def test_empty_expected_asserts_no_tool_calls(self):
        assert_tool_calls({"messages": [AIMessage(content="just text")]}, [])
        with pytest.raises(AssertionError, match="expected 0 tool call"):
            assert_tool_calls(self._result(("get_weather", {})), [])

    def test_fails_on_wrong_order_with_position(self):
        result = self._result(("b", {}), ("a", {}))
        with pytest.raises(AssertionError, match="call 0 was 'b', expected 'a'"):
            assert_tool_calls(result, [ExpectedToolCall("a"), ExpectedToolCall("b")])

    def test_fails_on_arg_value_mismatch_with_detail(self):
        result = self._result(("get_weather", {"city": "Cambridge"}))
        with pytest.raises(
            AssertionError, match="city: expected 'Boston', got 'Cambridge'"
        ):
            assert_tool_calls(
                result, [ExpectedToolCall("get_weather", {"city": "Boston"})]
            )

    def test_fails_on_missing_arg_key(self):
        result = self._result(("get_weather", {"city": "Boston"}))
        with pytest.raises(
            AssertionError, match="units: expected 'F', got '<missing>'"
        ):
            assert_tool_calls(
                result,
                [ExpectedToolCall("get_weather", {"city": "Boston", "units": "F"})],
            )

    def test_fails_on_count_mismatch_listing_both_sides(self):
        result = self._result(("a", {}))
        with pytest.raises(AssertionError, match="expected 2 tool call.*made 1"):
            assert_tool_calls(result, [ExpectedToolCall("a"), ExpectedToolCall("b")])

    def test_accepts_a_raw_message_list(self):
        assert_tool_calls(
            [AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}])],
            [ExpectedToolCall("a")],
        )

    def test_rejects_non_message_sequences_instead_of_silent_zero_calls(self):
        # A list of plain dicts would otherwise be filtered to zero calls and
        # reported as "the graph made 0 tool calls" — misdirecting the reader
        # away from the actual bug (wrong input type).
        with pytest.raises(TypeError, match="BaseMessage"):
            assert_tool_calls([{"role": "ai", "content": "hi"}], [])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="BaseMessage"):
            assert_tool_calls("not a message list", [])  # type: ignore[arg-type]

    def test_collects_calls_across_loop_iterations_in_order(self):
        # A tool-calling graph's history holds one AIMessage per loop
        # iteration; the recorded sequence must span all of them, in order.
        result = {
            "messages": [
                HumanMessage(content="go"),
                AIMessage(
                    content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}]
                ),
                ToolMessage(content="ra", tool_call_id="1"),
                AIMessage(
                    content="", tool_calls=[{"name": "b", "args": {}, "id": "2"}]
                ),
                ToolMessage(content="rb", tool_call_id="2"),
                AIMessage(content="done"),
            ]
        }
        assert_tool_calls(result, [ExpectedToolCall("a"), ExpectedToolCall("b")])

    def test_rejects_result_without_messages(self):
        with pytest.raises(TypeError, match="messages"):
            assert_tool_calls({"not_messages": []}, [])


class TestBuildTestGraph:
    @pytest.fixture(autouse=True)
    def _tool_mode(self, monkeypatch):
        monkeypatch.setenv("RUNNER_MODE", "tool")

    def test_requires_tool_mode(self, monkeypatch):
        # App is constructed under aer; build_test_graph must reject it based
        # on the app's constructed mode, even if RUNNER_MODE is later flipped
        # to tool (which the autouse fixture does on the next test).
        monkeypatch.setenv("RUNNER_MODE", "aer")
        app = App(app_name="AER-mode Agent")

        @app.entrypoint
        def build():
            return "graph"

        # Flip the env var after construction to prove the guard checks the
        # app's own mode, not the current env — this would pass an env-only
        # guard but must still be rejected.
        monkeypatch.setenv("RUNNER_MODE", "tool")
        with pytest.raises(RuntimeError, match="TOOL mode"):
            build_test_graph(app, FakeChatModel())

    def test_requires_registered_entrypoint(self):
        app = App(app_name="No Entrypoint Agent")
        with pytest.raises(RuntimeError, match="entrypoint"):
            build_test_graph(app, FakeChatModel())

    def test_raises_when_use_custom_parser_true_without_registered_parser(self):
        # Mirrors get_agent()'s own fail-fast (runtime.py, get_agent): opting
        # into features.use_custom_parser without registering a parser is a
        # misconfiguration that must surface here too, not just under real
        # AER execution — build_test_graph invokes the raw graph directly, so
        # nothing else would ever consult (or complain about) the parser.
        from agent_engine_runner_shared.agent_config import (
            AgentFeatureConfig,
            RuntimeAgentConfig,
        )

        app = App(app_name="Custom-parser Agent")
        app._runtime._agent_config = RuntimeAgentConfig(
            features=AgentFeatureConfig(use_custom_parser=True)
        )

        @app.entrypoint
        def build():
            return "graph"

        with pytest.raises(RuntimeError, match="use_custom_parser"):
            build_test_graph(app, FakeChatModel())

    def test_repeated_calls_do_not_collide_on_default_llm_id(self):
        # Regression guard: get_agent() calls reset_llm_registry() before
        # evaluating the entrypoint; build_test_graph must too, or a second
        # call (e.g. a second test in the same process) that registers under
        # the same default "__default__" id raises a stale duplicate-id error
        # from the *previous* call, not a fresh build. Found by porting these
        # helpers onto a real agent (insurance-agent) that calls app.llm()
        # with no explicit llm_id, at all — the in-repo unit tests above only
        # ever used explicit ids or never called app.llm() at all.
        app = App(app_name="Repeated-build Agent")

        @app.entrypoint
        def build():
            return app.llm(_UNUSED_REAL_LLM)

        build_test_graph(app, FakeChatModel())
        build_test_graph(app, FakeChatModel())  # must not raise duplicate-id

    def test_substitutes_fake_llm_for_every_call_and_binds_tools(self):
        app = App(app_name="Multi-LLM Agent")
        fake = FakeChatModel(responses=["The weather in Boston is sunny."])
        seen = {}

        @app.tool()
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Sunny in {city}"

        @app.entrypoint
        def build():
            # Mirrors real tenant code: construct "real" LLMs and register
            # two of them under distinct ids, then bind tool schemas —
            # exactly the pattern that requires FakeChatModel.bind_tools()
            # to be a working no-op.
            primary = app.llm(_UNUSED_REAL_LLM, llm_id="primary")
            secondary = app.llm(_UNUSED_REAL_LLM, llm_id="secondary")
            seen["primary"] = primary
            seen["secondary"] = secondary
            bound = secondary.bind_tools(app.get_tool_schemas())
            seen["bound"] = bound

            from typing import TypedDict

            from langgraph.graph import END, StateGraph

            class State(TypedDict):
                messages: list

            def call_llm(state: State) -> State:
                response = bound.invoke(state["messages"])
                return {"messages": state["messages"] + [response]}

            graph = StateGraph(State)
            graph.add_node("call_llm", call_llm)
            graph.set_entry_point("call_llm")
            graph.add_edge("call_llm", END)
            return graph.compile()

        compiled = build_test_graph(app, fake)
        assert seen["primary"] is fake
        assert seen["secondary"] is fake
        assert seen["bound"] is fake

        result = compiled.invoke(
            {"messages": [HumanMessage(content="weather in Boston?")]}
        )
        assert result["messages"][-1].content == "The weather in Boston is sunny."

    def test_requires_exactly_one_of_llm_or_llms(self):
        app = App(app_name="Arg-validation Agent")

        @app.entrypoint
        def build():
            return "graph"

        with pytest.raises(ValueError, match="exactly one of"):
            build_test_graph(app)
        with pytest.raises(ValueError, match="exactly one of"):
            build_test_graph(app, FakeChatModel(), llms={"primary": FakeChatModel()})

    def test_llms_map_routes_each_llm_id_to_its_own_model(self):
        app = App(app_name="Per-id Agent")
        fake_primary = FakeChatModel()
        fake_fast = FakeChatModel()
        seen = {}

        @app.entrypoint
        def build():
            seen["primary"] = app.llm(_UNUSED_REAL_LLM, llm_id="primary")
            seen["fast"] = app.llm(_UNUSED_REAL_LLM, llm_id="fast")
            seen["default"] = app.llm(_UNUSED_REAL_LLM)
            return "graph"

        build_test_graph(
            app,
            llms={
                "primary": fake_primary,
                "fast": fake_fast,
                "__default__": fake_primary,
            },
        )
        assert seen["primary"] is fake_primary
        assert seen["fast"] is fake_fast
        assert seen["default"] is fake_primary

    def test_llms_map_raises_on_unmapped_llm_id(self):
        # Strict by design: silently falling back to some default model would
        # run a node with the wrong LLM and mask exactly the misconfiguration
        # a multi-model test is trying to pin down.
        app = App(app_name="Unmapped-id Agent")

        @app.entrypoint
        def build():
            app.llm(_UNUSED_REAL_LLM, llm_id="summarizer")
            return "graph"

        with pytest.raises(
            RuntimeError, match="'summarizer'.*Provided ids: \\['primary'\\]"
        ):
            build_test_graph(app, llms={"primary": FakeChatModel()})

    def test_llms_map_reports_the_first_unmapped_id_when_several_exist(self):
        # Locks in current behavior: the build fails on the FIRST unmapped
        # id it hits (registration order), not a collected list of all of
        # them — a test author fixing iteratively should rely on that.
        app = App(app_name="Two-unmapped Agent")

        @app.entrypoint
        def build():
            app.llm(_UNUSED_REAL_LLM, llm_id="first_unmapped")
            app.llm(_UNUSED_REAL_LLM, llm_id="second_unmapped")
            return "graph"

        with pytest.raises(RuntimeError, match="'first_unmapped'") as exc_info:
            build_test_graph(app, llms={"primary": FakeChatModel()})
        assert "second_unmapped" not in str(exc_info.value)

    def test_llms_map_names_unnamed_calls_default(self):
        app = App(app_name="Unmapped-default Agent")

        @app.entrypoint
        def build():
            app.llm(_UNUSED_REAL_LLM)  # registers under "__default__"
            return "graph"

        with pytest.raises(RuntimeError, match="'__default__'"):
            build_test_graph(app, llms={"primary": FakeChatModel()})

    def test_end_to_end_scripted_tool_call_with_real_tool_execution(self):
        # The headline flow: script a tool call, let the graph's real
        # ToolNode execute the real @app.tool, loop back for the final answer,
        # then assert tool name/args/order and the final output.
        app = App(app_name="Tool-calling Agent")

        @app.tool()
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Sunny in {city}"

        @app.entrypoint
        def build():
            from langgraph.graph import END, MessagesState, StateGraph
            from langgraph.prebuilt import ToolNode

            llm = app.llm(_UNUSED_REAL_LLM).bind_tools(app.get_tool_schemas())

            def call_llm(state: MessagesState) -> dict:
                return {"messages": [llm.invoke(state["messages"])]}

            # Annotated `dict`, not MessagesState: add_conditional_edges resolves
            # the router's annotations against module globals at runtime.
            def should_continue(state: dict) -> str:
                last = state["messages"][-1]
                return "tools" if getattr(last, "tool_calls", None) else END

            graph = StateGraph(MessagesState)
            graph.add_node("call_llm", call_llm)
            graph.add_node("tools", ToolNode(app.get_tools()))
            graph.set_entry_point("call_llm")
            graph.add_conditional_edges(
                "call_llm", should_continue, {"tools": "tools", END: END}
            )
            graph.add_edge("tools", "call_llm")
            return graph.compile()

        fake = FakeChatModel(
            responses=[
                {"tool_calls": [{"name": "get_weather", "args": {"city": "Boston"}}]},
                "The weather in Boston is sunny.",
            ]
        )
        result = build_test_graph(app, fake).invoke(
            {"messages": [HumanMessage(content="weather in Boston?")]}
        )

        assert_tool_calls(result, [ExpectedToolCall("get_weather", {"city": "Boston"})])
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert [m.content for m in tool_messages] == ["Sunny in Boston"]
        assert result["messages"][-1].content == "The weather in Boston is sunny."

    def test_restores_original_llm_method_on_success(self):
        app = App(app_name="Restore-on-success Agent")

        @app.entrypoint
        def build():
            return "ok"

        original = app.llm
        build_test_graph(app, FakeChatModel())
        # Bound methods aren't singletons (`f.bar is f.bar` is False even
        # though both wrap the same underlying function on the same
        # instance) — `==` is the correct identity check here.
        assert app.llm == original

    def test_restores_original_llm_method_on_exception(self):
        app = App(app_name="Restore-on-exception Agent")

        @app.entrypoint
        def build():
            raise ValueError("boom")

        original = app.llm
        with pytest.raises(ValueError, match="boom"):
            build_test_graph(app, FakeChatModel())
        assert app.llm == original


class TestBuildTestAgent:
    """build_test_agent exercises the LangGraphBaseAgent adapter — the layer
    where the custom output parser runs — with a real App-built graph, not
    the mocked graphs of test_agent.py's adapter unit tests."""

    @pytest.fixture(autouse=True)
    def _tool_mode(self, monkeypatch):
        monkeypatch.setenv("RUNNER_MODE", "tool")

    def _parser_app(self) -> App:
        from agent_engine_runner_shared.agent_config import (
            AgentFeatureConfig,
            RuntimeAgentConfig,
        )

        from agent_engine_sdk_langgraph.output_parser import LangGraphOutputParser

        app = App(app_name="Parser Agent")
        app._runtime._agent_config = RuntimeAgentConfig(
            features=AgentFeatureConfig(use_custom_parser=True)
        )

        @app.output_parser
        class BriefParser(LangGraphOutputParser):
            stream_modes = ("messages",)

            async def parse(self, item, ctx):
                if item.stream_mode != "messages":
                    return
                chunk, _metadata = item.payload
                content = getattr(chunk, "content", "")
                if content:
                    yield {"kind": "brief", "text": content}

            async def on_stream_error(self, ctx, error):
                return None

        @app.entrypoint
        def build():
            from langgraph.graph import END, MessagesState, StateGraph

            llm = app.llm(_UNUSED_REAL_LLM)

            def call_llm(state: MessagesState) -> dict:
                return {"messages": [llm.invoke(state["messages"])]}

            graph = StateGraph(MessagesState)
            graph.add_node("call_llm", call_llm)
            graph.set_entry_point("call_llm")
            graph.add_edge("call_llm", END)
            return graph.compile(checkpointer=app.checkpointer())

        return app

    @pytest.mark.anyio
    async def test_stream_applies_the_custom_parser(self):
        # The headline capability: the parser's output format — what the
        # Gateway would relay as custom_event — is assertable headlessly.
        from agent_engine_sdk import AgentInput, RequestContext

        app = self._parser_app()
        agent = build_test_agent(app, FakeChatModel(responses=["hello world"]))

        ctx = RequestContext(execution_id="e1", session_id="s1", user_id="u1")
        events = [
            event
            async for event in agent.stream(ctx, AgentInput(payload={"message": "hi"}))
        ]

        customs = [e.custom_event for e in events if e.custom_event is not None]
        assert customs == [{"kind": "brief", "text": "hello world"}]
        # The platform pipeline still runs alongside (dual-run), ending in a
        # result event carrying the final response.
        assert events[-1].event == "result"
        result_data = events[-1].data
        assert isinstance(result_data, dict)
        assert result_data["response"] == "hello world"

    @pytest.mark.anyio
    async def test_invoke_completes_with_substituted_in_memory_checkpointer(self):
        # invoke()'s suspend detection calls aget_state, which raises without
        # a checkpointer — TOOL mode's app.checkpointer() is None, so
        # build_test_agent substitutes an in-memory saver to keep the
        # production-shaped code path working.
        from agent_engine_sdk import AgentInput, RequestContext

        app = self._parser_app()
        agent = build_test_agent(app, FakeChatModel(responses=["hello world"]))

        output = await agent.invoke(
            RequestContext(execution_id="e1", session_id="s1", user_id="u1"),
            AgentInput(payload={"message": "hi"}),
        )

        response = output.response
        assert isinstance(response, dict)
        assert response["status"] == "completed"
        assert response["response"] == "hello world"

    @pytest.mark.anyio
    async def test_checkpointer_persists_state_across_invocations(self):
        # The InMemorySaver substitution must actually persist: two invokes
        # on the same session share one thread, so the second run's state
        # carries the first run's messages (message_count grows 2 -> 4).
        from agent_engine_sdk import AgentInput, RequestContext

        app = self._parser_app()
        agent = build_test_agent(app, FakeChatModel(responses=["hello world"]))
        ctx = RequestContext(execution_id="e1", session_id="s1", user_id="u1")

        first = await agent.invoke(ctx, AgentInput(payload={"message": "hi"}))
        second = await agent.invoke(ctx, AgentInput(payload={"message": "again"}))

        first_response = first.response
        second_response = second.response
        assert isinstance(first_response, dict) and isinstance(second_response, dict)
        assert first_response["message_count"] == 2
        assert second_response["message_count"] == 4

    @pytest.mark.anyio
    async def test_llms_map_routes_per_llm_id(self):
        from agent_engine_sdk import AgentInput, RequestContext

        app = App(app_name="Multi-model Adapter Agent")

        @app.entrypoint
        def build():
            from langgraph.graph import END, MessagesState, StateGraph

            fast = app.llm(_UNUSED_REAL_LLM, llm_id="fast")
            primary = app.llm(_UNUSED_REAL_LLM, llm_id="primary")

            def route(state: MessagesState) -> dict:
                return {"messages": [fast.invoke(state["messages"])]}

            def answer(state: MessagesState) -> dict:
                return {"messages": [primary.invoke(state["messages"])]}

            graph = StateGraph(MessagesState)
            graph.add_node("route", route)
            graph.add_node("answer", answer)
            graph.set_entry_point("route")
            graph.add_edge("route", "answer")
            graph.add_edge("answer", END)
            return graph.compile(checkpointer=app.checkpointer())

        agent = build_test_agent(
            app,
            llms={
                "fast": FakeChatModel(responses=["routing note"]),
                "primary": FakeChatModel(responses=["final answer"]),
            },
        )
        output = await agent.invoke(
            RequestContext(session_id="s1", user_id="u1"),
            AgentInput(payload={"message": "hi"}),
        )
        response = output.response
        assert isinstance(response, dict)
        assert response["response"] == "final answer"

    def test_requires_exactly_one_of_llm_or_llms(self):
        app = App(app_name="Agent Arg-validation")

        @app.entrypoint
        def build():
            return "graph"

        with pytest.raises(
            ValueError, match="build_test_agent requires exactly one of"
        ):
            build_test_agent(app)

    def test_requires_tool_mode(self, monkeypatch):
        monkeypatch.setenv("RUNNER_MODE", "aer")
        app = App(app_name="AER-mode Adapter Agent")

        @app.entrypoint
        def build():
            return "graph"

        monkeypatch.setenv("RUNNER_MODE", "tool")
        with pytest.raises(RuntimeError, match="TOOL mode"):
            build_test_agent(app, FakeChatModel())

    def test_restores_llm_and_checkpointer_methods(self):
        app = self._parser_app()
        original_llm = app.llm
        original_checkpointer = app.checkpointer
        build_test_agent(app, FakeChatModel())
        # Bound methods aren't singletons — `==` is the correct identity check.
        assert app.llm == original_llm
        assert app.checkpointer == original_checkpointer


class TestExecutionContext:
    def test_sets_and_clears_context(self):
        from agent_engine_runner_shared.context import (
            get_current_execution_id,
            get_current_user_id,
        )

        assert get_current_execution_id() is None
        with execution_context(user_id="user-123"):
            assert get_current_execution_id() is not None
            assert get_current_user_id() == "user-123"
        assert get_current_execution_id() is None
        assert get_current_user_id() is None

    def test_generates_execution_id_when_not_given(self):
        from agent_engine_runner_shared.context import get_current_execution_id

        with execution_context():
            execution_id = get_current_execution_id()
            assert execution_id is not None
            assert execution_id.startswith("test-")
