"""Opt-in per-invocation agent building: ``load_strategy`` on ``@app.entrypoint``.

An agent that opts into ``load_strategy="per-execution"`` gets a fresh agent
built per invocation from the invocation's ``(input, ctx)``.
``load_strategy="global"`` (the default) keeps the existing behavior.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext

from agent_engine_sdk_adk import App, LLMRegistrationDriftError


def _llm() -> MagicMock:
    """A stand-in ADK LLM.

    ``model`` has to be a string: in AER mode ``app.llm()`` wraps the argument
    in ``SecureLlm``, whose pydantic validation rejects a mock's attribute.
    """
    llm = MagicMock()
    llm.model = "gemini-3-flash-preview"
    return llm


class TestEntrypointLoadStrategy:
    def test_bare_entrypoint_defaults_to_global(self):
        app = App(app_name="Test Agent")

        @app.entrypoint
        def build():
            return MagicMock()

        assert app._builder_fn is build
        assert app._load_strategy == "global"

    def test_entrypoint_accepts_per_execution_strategy(self):
        app = App(app_name="Test Agent")

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            return MagicMock()

        assert app._builder_fn is build
        assert app._load_strategy == "per-execution"

    def test_entrypoint_rejects_unknown_strategy(self):
        app = App(app_name="Test Agent")

        with pytest.raises(ValueError, match="load_strategy"):

            @app.entrypoint(load_strategy="sometimes")  # type: ignore[call-overload, arg-type]
            def build(input, ctx):
                return MagicMock()

    def test_a_global_builder_taking_arguments_is_rejected_at_decoration(self):
        app = App(app_name="Test Agent")

        with pytest.raises(TypeError, match="take no arguments"):
            # Deliberately mistyped: the arity check is what rejects it.
            @app.entrypoint  # type: ignore[arg-type]
            def build(input, ctx):
                return MagicMock()

    def test_a_per_execution_builder_taking_no_arguments_is_rejected(self):
        app = App(app_name="Test Agent")

        with pytest.raises(TypeError, match="input, context"):
            # Deliberately mistyped: the arity check is what rejects it.
            @app.entrypoint(load_strategy="per-execution")  # type: ignore[arg-type]
            def build():
                return MagicMock()


class TestPerExecutionBuild:
    @staticmethod
    def _per_execution_app(build: Any) -> App:
        app = App(app_name="Test Agent")
        app.entrypoint(load_strategy="per-execution")(build)
        return app

    @pytest.mark.asyncio
    async def test_a_global_app_builds_nothing(self):
        app = App(app_name="Test Agent")

        @app.entrypoint
        def build():
            return MagicMock()

        assert await app.build_graph() is None

    @pytest.mark.asyncio
    async def test_the_builder_receives_the_invocation_input_and_context(self):
        seen = []

        async def build(input, ctx):
            seen.append((input, ctx))
            return MagicMock()

        app = self._per_execution_app(build)
        agent_input, ctx = AgentInput(payload={"message": "hi"}), RequestContext()

        built = await app.build_graph(agent_input, ctx)

        assert built is not None
        assert seen == [(agent_input, ctx)]

    @pytest.mark.asyncio
    async def test_each_call_rebuilds(self):
        calls = []

        async def build(input, ctx):
            calls.append(1)
            return MagicMock()

        app = self._per_execution_app(build)

        first = await app.build_graph()
        second = await app.build_graph()

        assert first is not second
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_a_context_free_failure_names_the_discovery_path(self):
        async def build(input, ctx):
            ctx.user_id  # the discovery path supplies no context
            return MagicMock()

        app = self._per_execution_app(build)

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

    @pytest.mark.asyncio
    async def test_a_builder_failure_keeps_its_own_error(self):
        async def build(input, ctx):
            raise KeyError("boom")

        app = self._per_execution_app(build)

        with pytest.raises(KeyError):
            await app.build_graph(AgentInput(payload={}), RequestContext())

    @pytest.mark.asyncio
    async def test_get_agent_wraps_a_graph_it_was_given(self):
        """Wrapping the supplied agent must keep the ids its build registered.

        The AER awaits the builder and then materializes the adapter through
        get_agent(graph=...). The Tool Pod resolves a model by ``llm_id`` against
        the registry, so resetting it on this path would leave a model the build
        just registered unresolvable.
        """
        from agent_engine_runner_shared import hooks

        built = MagicMock()
        llm = _llm()

        async def build(input, ctx):
            app.llm(llm, llm_id="primary")
            return built

        app = self._per_execution_app(build)
        graph = await app.build_graph(AgentInput(payload={}), RequestContext())

        agent = app.get_agent(graph=graph)

        assert agent._adk_agent is built
        assert hooks.get_named_llm("primary") is llm


class TestInvariantLlmRegistration:
    """A per-execution builder must register the same LLM ids every invocation.

    The Tool Pod resolves a model by ``llm_id`` against a registry built once per
    pod from a context-free build, so every id an invocation could use has to be
    registered there.
    """

    @staticmethod
    def _app_registering(ids_by_build: list[list[str]]) -> tuple[App, list[int]]:
        app = App(app_name="Test Agent")
        calls = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            index = min(calls[0], len(ids_by_build) - 1)
            calls[0] += 1
            for llm_id in ids_by_build[index]:
                app.llm(_llm(), llm_id=llm_id)
            return MagicMock()

        return app, calls

    @pytest.mark.asyncio
    async def test_a_build_that_changes_the_registered_ids_is_rejected(self):
        app, _ = self._app_registering([["premium"], ["free"]])
        agent_input, ctx = AgentInput(payload={}), RequestContext()

        await app.build_graph(agent_input, ctx)

        with pytest.raises(RuntimeError, match="same LLM ids"):
            await app.build_graph(agent_input, ctx)

    @pytest.mark.asyncio
    async def test_an_invariant_builder_is_accepted_across_invocations(self):
        app, calls = self._app_registering([["primary", "fast"]])

        await app.build_graph()
        await app.build_graph()

        assert calls[0] == 2

    @pytest.mark.asyncio
    async def test_a_failed_discovery_establishes_the_baseline(self):
        app = App(app_name="Test Agent")
        builds = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            index = min(builds[0], 1)
            builds[0] += 1
            app.llm(_llm(), llm_id="primary")
            if index == 1:
                app.llm(_llm(), llm_id="extra")
            if input is None:  # the context-free discovery build
                raise KeyError("no invocation context")
            return MagicMock()

        # The discovery build fails after registering `primary`; its retained
        # ids become the baseline.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # A later build registering `primary` plus `extra` is rejected.
        with pytest.raises(LLMRegistrationDriftError, match="same LLM ids"):
            await app.build_graph(AgentInput(payload={}), RequestContext())

    @pytest.mark.asyncio
    async def test_a_failed_discovery_that_registers_more_recovers(self):
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = _llm(), _llm()

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(primary, llm_id="primary")
            if attempts[0] > 1:
                app.llm(fast, llm_id="fast")
            ctx.user_id  # the discovery path cannot supply context
            return MagicMock()

        # The first discovery fails constructing `fast`, after `primary` only.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # The retry registers both, then fails on the absent context, as this
        # builder always does on the discovery path. That failure must not
        # reject the model it just recovered.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        assert hooks.get_named_llm("primary") is primary
        assert hooks.get_named_llm("fast") is fast

    @pytest.mark.asyncio
    async def test_a_failed_retry_that_drops_a_baseline_id_is_rejected(self):
        app = App(app_name="Test Agent")
        builds = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            index = min(builds[0], 1)
            builds[0] += 1
            app.llm(_llm(), llm_id="primary" if index == 0 else "extra")
            raise KeyError("no invocation context")  # fails every discovery

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        with pytest.raises(LLMRegistrationDriftError, match="same LLM ids"):
            await app.build_graph()

    @pytest.mark.asyncio
    async def test_a_failed_retry_that_registers_nothing_is_not_drift(self):
        app = App(app_name="Test Agent")
        attempts = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            if attempts[0] > 1:
                raise RuntimeError("temporary failure before registering anything")
            app.llm(_llm(), llm_id="primary")
            raise KeyError("no invocation context")

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()


class TestToolPodReadsTheOptIn:
    """runner-shared decides partial-registry retention by reading the App.

    ``TenantRuntime.uses_per_invocation_build`` looks the opt-in up on the
    registered App by name across the package boundary. Nothing else exercises
    the real lookup, so a rename on either side would silently disable the
    retention instead of failing.
    """

    @staticmethod
    def _sees_the_opt_in(app: App) -> bool:
        # The App's own runtime, with the App registered as its builder: exactly
        # the pairing the Tool Pod runs with.
        app._runtime._graph_builder = app
        return app._runtime.uses_per_invocation_build()

    def test_a_per_execution_app_is_seen_as_opting_in(self):
        app = App(app_name="Test Agent")

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            return MagicMock()

        assert self._sees_the_opt_in(app) is True

    def test_a_global_app_is_not(self):
        app = App(app_name="Test Agent")

        @app.entrypoint
        def build():
            return MagicMock()

        assert self._sees_the_opt_in(app) is False


class TestToolPodRegistryRecovery:
    """The Tool Pod's registry load, driven by a real App.

    The Tool Pod populates its per-pod registry by building the agent with no
    invocation context, and re-runs that load inside each request's credential
    isolation. A discovery that fails partway through registration must not
    stop a later load from completing.
    """

    @staticmethod
    def _pod_serving(app: App) -> Any:
        from agent_engine_runner_shared.server.tool import ToolServer

        # The App's own runtime, so the pod's opt-in check is the real one.
        app._runtime._graph_builder = app
        return ToolServer(app._runtime)

    @pytest.mark.asyncio
    async def test_a_partial_discovery_failure_recovers_on_a_later_load(self):
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = _llm(), _llm()

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(primary, llm_id="primary")
            if attempts[0] > 1:
                app.llm(fast, llm_id="fast")
            if attempts[0] == 1:
                raise RuntimeError("credential unavailable outside request isolation")
            return MagicMock()

        server = self._pod_serving(app)

        # Startup discovery fails after registering `primary` only. The pod
        # keeps that registration so it still has a model to serve.
        await server._ensure_llm_registry_loaded()
        assert hooks.get_named_llm("primary") is primary

        # The next request's load runs inside credential isolation, completes,
        # and the pod ends up with the model the failure had cut short.
        await server._ensure_llm_registry_loaded()
        assert hooks.get_named_llm("fast") is fast
        assert server._llm_registry_load_error is None

    @pytest.mark.asyncio
    async def test_a_context_requiring_builder_recovers_the_model_it_registered(self):
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = _llm(), _llm()

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(primary, llm_id="primary")
            if attempts[0] > 1:
                app.llm(fast, llm_id="fast")
            ctx.user_id  # discovery supplies no context, so this always raises
            return MagicMock()

        server = self._pod_serving(app)

        await server._ensure_llm_registry_loaded()
        assert hooks.get_named_llm("primary") is primary

        # The retry registers `fast` and then fails on the absent context, as
        # this builder always does here. The pod still ends up able to resolve it.
        await server._ensure_llm_registry_loaded()
        assert hooks.get_named_llm("fast") is fast

        # The failure is still recorded, so a later request retries the build.
        assert server._llm_registry_load_error is not None
