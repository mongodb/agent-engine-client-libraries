"""Opt-in per-invocation graph building: ``load_strategy`` on ``@app.entrypoint``.

An agent that opts into ``load_strategy="per-execution"`` gets a fresh graph per
invocation, built by running the entrypoint builder with the invocation's
``(input, ctx)``. ``load_strategy="global"`` (the default) keeps today's cached,
build-once behavior untouched.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext

from agent_engine_sdk_langgraph import App, LLMRegistrationDriftError


def _input(payload: dict | None = None) -> AgentInput:
    return AgentInput(payload=payload if payload is not None else {"message": "hi"})


def _ctx() -> RequestContext:
    return RequestContext(execution_id="exec-1", session_id="sess-1")


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
            # Deliberately invalid: the decorator must reject it at decoration.
            @app.entrypoint(load_strategy="sometimes")  # type: ignore[call-overload, arg-type]
            def build(input, ctx):
                return MagicMock()


class TestPerExecutionBuild:
    @pytest.mark.anyio
    async def test_build_graph_passes_input_and_ctx_to_the_builder(self):
        app = App(app_name="Test Agent")
        received: list[object] = []

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            received.append((input, ctx))
            return MagicMock()

        agent_input, ctx = _input(), _ctx()
        await app.build_graph(agent_input, ctx)

        assert received == [(agent_input, ctx)]

    @pytest.mark.anyio
    async def test_build_graph_awaits_an_async_builder(self):
        app = App(app_name="Test Agent")
        received: list[object] = []

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            received.append((input, ctx))
            return MagicMock()

        agent_input, ctx = _input(), _ctx()
        graph = await app.build_graph(agent_input, ctx)

        assert received == [(agent_input, ctx)]
        assert graph is not None

    @pytest.mark.anyio
    async def test_build_graph_returns_none_for_global(self):
        app = App(app_name="Test Agent")

        @app.entrypoint
        def build():
            return MagicMock()

        assert await app.build_graph(_input(), _ctx()) is None

    @pytest.mark.anyio
    async def test_sequential_invocations_build_distinct_graphs(self):
        app = App(app_name="Test Agent")
        graphs: list[object] = []

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            graph = MagicMock()
            graphs.append(graph)
            return graph

        first = await app.build_graph(_input({"n": 1}), _ctx())
        second = await app.build_graph(_input({"n": 2}), _ctx())

        assert len(graphs) == 2, "per-execution must rebuild on every invocation"
        assert first is not second
        assert app._graph_cache is None, "per-execution must not populate the cache"

    @pytest.mark.anyio
    async def test_each_caller_gets_its_own_input(self):
        app = App(app_name="Test Agent")
        received: list[Any] = []

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            received.append(input)
            return MagicMock()

        await app.build_graph(_input({"user": "a"}), _ctx())
        await app.build_graph(_input({"user": "b"}), _ctx())

        assert [i.payload for i in received] == [{"user": "a"}, {"user": "b"}]

    @pytest.mark.anyio
    async def test_concurrent_async_builds_do_not_collide_on_the_registry(self):
        """Two overlapping builds must not fail each other's ``app.llm()``.

        The LLM registry is process-global, so without a serialization gate one
        build's reset/register lifetime could interleave with the other's and
        reject an otherwise-valid duplicate id.
        """
        app = App(app_name="Test Agent")

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            await asyncio.sleep(0)  # yield, so the two builds interleave
            app.llm(MagicMock(), llm_id="primary")
            return MagicMock()

        first, second = await asyncio.gather(
            app.build_graph(_input(), _ctx()),
            app.build_graph(_input(), _ctx()),
        )

        assert first is not second

    def test_global_strategy_still_caches(self):
        app = App(app_name="Test Agent")
        calls: list[int] = []

        @app.entrypoint
        def build():
            calls.append(1)
            return MagicMock()

        first = app.get_agent()
        second = app.get_agent()

        assert len(calls) == 1
        assert first._graph is second._graph

    def test_get_agent_uses_a_prebuilt_graph(self):
        app = App(app_name="Test Agent")
        graph = MagicMock()

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            raise AssertionError("get_agent must not rebuild when given a graph")

        agent = app.get_agent(graph=graph)

        assert agent._graph is graph

    def test_per_execution_builder_arity_mismatch_errors_at_decoration(self):
        app = App(app_name="Test Agent")

        with pytest.raises(TypeError, match="context"):
            # Deliberately wrong arity: the decorator must reject it.
            @app.entrypoint(load_strategy="per-execution")  # type: ignore[arg-type]
            def build():  # zero-arg builder under per-execution
                return MagicMock()

    def test_global_builder_arity_mismatch_errors_at_decoration(self):
        app = App(app_name="Test Agent")

        with pytest.raises(TypeError, match="context"):
            # Deliberately wrong arity: the decorator must reject it.
            @app.entrypoint  # type: ignore[call-overload, arg-type]
            def build(input, ctx):  # two-arg builder under global
                return MagicMock()

    def test_variadic_builder_is_accepted_under_either_strategy(self):
        app = App(app_name="Test Agent")

        @app.entrypoint(load_strategy="per-execution")
        def build(*args):
            return MagicMock()

        assert app._builder_fn is build

    def test_global_builder_with_a_defaulted_parameter_is_accepted(self):
        """The maintained example agents take an optional model override."""
        app = App(app_name="Test Agent")

        @app.entrypoint
        def build(llm=None):
            return MagicMock()

        assert app._builder_fn is build
        assert app._get_or_build_graph() is not None

    def test_per_execution_builder_with_a_required_keyword_only_parameter_is_rejected(
        self,
    ):
        """It cannot be called as ``build(input, ctx)``, so it must not decorate."""
        app = App(app_name="Test Agent")

        with pytest.raises(TypeError, match="context"):
            # Deliberately wrong arity: the decorator must reject it.
            @app.entrypoint(load_strategy="per-execution")  # type: ignore[arg-type]
            def build(input, ctx, *, extra):
                return MagicMock()

    def test_per_execution_builder_with_an_optional_parameter_is_accepted(self):
        app = App(app_name="Test Agent")

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx, llm=None):
            return MagicMock()

        assert app._builder_fn is build


class TestWarmUp:
    def test_warm_up_is_noop_for_per_execution(self):
        app = App(app_name="Test Agent")
        calls: list[int] = []

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            calls.append(1)
            return MagicMock()

        assert app.warm_up() is True
        assert calls == [], "warm-up must not build for an opted-in agent"
        assert app._graph_cache is None

    def test_warm_up_still_builds_for_global(self):
        app = App(app_name="Test Agent")
        calls: list[int] = []

        @app.entrypoint
        def build():
            calls.append(1)
            return MagicMock()

        assert app.warm_up() is True
        assert len(calls) == 1


class TestContextFreeDiscoveryDiagnostic:
    """A context-free build of a context-requiring builder must say so.

    The platform discovers an agent's LLM ids by building the graph with no
    invocation context. That cannot work for a builder that needs context, and
    the failure has to name the discovery rather than look like a bug in the
    builder's own code.
    """

    @staticmethod
    def _app_needing_context() -> App:
        app = App(app_name="Test Agent")
        app.checkpointer = lambda: None  # type: ignore[method-assign]

        @app.entrypoint(load_strategy="per-execution")
        def build(input, ctx):
            # Reads the input, the shape the feature-flag rollout has.
            rollout = input.payload["rollout_enabled"]
            del rollout
            return MagicMock()

        return app

    @pytest.mark.anyio
    async def test_omitted_invocation_names_the_context_free_discovery(self):
        app = self._app_needing_context()

        with pytest.raises(RuntimeError, match="context-free discovery") as raised:
            await app.build_graph()

        assert isinstance(raised.value.__cause__, AttributeError)

    @pytest.mark.anyio
    async def test_an_explicit_invocation_is_not_wrapped(self):
        """Only the discovery path is annotated; a caller's own input keeps its
        original error."""
        app = self._app_needing_context()

        with pytest.raises(KeyError):
            await app.build_graph(_input({}), _ctx())


class TestInvariantLlmRegistration:
    """A per-execution builder must register the same LLM ids every invocation.

    The Tool Pod resolves a model by ``llm_id`` against a registry built once
    per pod from a context-free build, so every id an invocation could use has
    to be registered there. Per-caller variation would have the platform ask for
    a model the pod never registered, which fails mid-execution; the build is
    where it should fail instead.
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
                app.llm(MagicMock(), llm_id=llm_id)
            return MagicMock()

        return app, calls

    @pytest.mark.anyio
    async def test_a_build_that_changes_the_registered_ids_is_rejected(self):
        app, _ = self._app_registering([["premium"], ["free"]])

        await app.build_graph(_input(), _ctx())  # establishes the baseline

        with pytest.raises(RuntimeError, match="same LLM ids"):
            await app.build_graph(_input(), _ctx())

    @pytest.mark.anyio
    async def test_an_extra_registered_id_is_rejected(self):
        app, _ = self._app_registering([["primary"], ["primary", "extra"]])

        await app.build_graph(_input(), _ctx())

        with pytest.raises(RuntimeError, match="same LLM ids"):
            await app.build_graph(_input(), _ctx())

    @pytest.mark.anyio
    async def test_an_invariant_builder_is_accepted_across_invocations(self):
        app, calls = self._app_registering([["primary", "fast"]])

        first = await app.build_graph(_input(), _ctx())
        second = await app.build_graph(_input(), _ctx())

        assert first is not second, "per-execution still rebuilds every invocation"
        assert calls[0] == 2

    @pytest.mark.anyio
    async def test_a_failed_discovery_establishes_the_baseline(self):
        """The ids a failed context-free discovery retained become the baseline.

        The Tool Pod keeps those registrations, so a later build that registers a
        different set must be rejected against them, not accepted for lack of a
        baseline.
        """
        app = App(app_name="Test Agent")
        builds = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            index = min(builds[0], 1)
            builds[0] += 1
            app.llm(MagicMock(), llm_id="primary")
            if index == 1:
                app.llm(MagicMock(), llm_id="extra")
            if input is None:  # the context-free discovery build
                raise KeyError("no invocation context")
            return MagicMock()

        # The discovery build fails after registering `primary`; its retained
        # ids become the baseline.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # A later build registering `primary` plus `extra` is rejected.
        with pytest.raises(LLMRegistrationDriftError, match="same LLM ids"):
            await app.build_graph(_input(), _ctx())

    @pytest.mark.anyio
    async def test_a_failed_retry_that_drops_a_baseline_id_is_rejected(self):
        """A retry that stops registering a baseline id is drift.

        A builder that needs invocation context fails on every context-free
        discovery, so the retry never reaches the success-path check. The failure
        path has to validate it. Registering *more* than the baseline is a
        recovery and is accepted; registering a set that no longer contains a
        baseline id would leave the pod unable to resolve a model it served.
        """
        app = App(app_name="Test Agent")
        builds = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            index = min(builds[0], 1)
            builds[0] += 1
            app.llm(MagicMock(), llm_id="primary" if index == 0 else "extra")
            raise KeyError("no invocation context")  # fails every discovery

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        with pytest.raises(LLMRegistrationDriftError, match="same LLM ids"):
            await app.build_graph()

    @pytest.mark.anyio
    async def test_a_transient_failure_before_registering_does_not_freeze_a_baseline(
        self,
    ):
        """A failure that registered nothing is transient, so a retry recovers.

        Freezing an empty baseline would reject the next successful discovery
        against it and leave the pod permanently mismatched.
        """
        app = App(app_name="Test Agent")
        attempts = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            if attempts[0] == 1:
                raise RuntimeError("temporary configuration failure")
            app.llm(MagicMock(), llm_id="primary")
            if input is None:
                raise KeyError("no invocation context")
            return MagicMock()

        # First discovery fails before registering anything.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # A later successful build is accepted, not rejected against {}.
        assert await app.build_graph(_input(), _ctx()) is not None

    @pytest.mark.anyio
    async def test_a_partial_failure_does_not_freeze_an_incomplete_baseline(self):
        """A build that completes may supersede the partial set a failure left.

        A discovery that fails partway through registration leaves a nonempty
        but incomplete set. Treating it as authoritative would reject the
        recovered build that registers the rest, so those models would stay
        unavailable for the pod's lifetime.
        """
        app = App(app_name="Test Agent")
        attempts = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(MagicMock(), llm_id="primary")
            if attempts[0] > 1:
                app.llm(MagicMock(), llm_id="fast")
            if attempts[0] == 1:
                raise RuntimeError("credential unavailable outside request isolation")
            return MagicMock()

        # The first discovery fails after registering `primary` only.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # The retry completes with the full set, which supersedes the partial one.
        assert await app.build_graph() is not None

    @pytest.mark.anyio
    async def test_a_failed_discovery_that_registers_more_recovers(self):
        """A failed discovery that registered more is a recovery, not drift.

        The motivating builder registers its models and then reads invocation
        context, so it fails on *every* context-free discovery and never reaches
        the success path. A model whose construction failed on the first attempt
        is registered on the next; the pod has to keep it, or that model stays
        unavailable for the pod's lifetime.
        """
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = MagicMock(), MagicMock()

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(primary, llm_id="primary")
            if attempts[0] > 1:
                app.llm(fast, llm_id="fast")
            ctx.user_id  # the context read the discovery path cannot supply
            return MagicMock()

        # The first discovery fails constructing `fast`, after registering
        # `primary` only.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # The retry registers both, then fails on the absent context, as this
        # builder always does on the discovery path. That failure must not
        # reject the model it just recovered.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        assert hooks.get_named_llm("primary") is primary
        assert hooks.get_named_llm("fast") is fast

    @pytest.mark.anyio
    async def test_a_failed_retry_that_registers_nothing_is_not_drift(self):
        """A retry that registered nothing is transient, not drift.

        The pod keeps the registry it already held, so there is no set to
        reject. Reporting it as drift misdiagnoses a transient failure, which
        is the one case the discovery path is supposed to tolerate.
        """
        app = App(app_name="Test Agent")
        attempts = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            if attempts[0] > 1:
                raise RuntimeError("temporary failure before registering anything")
            app.llm(MagicMock(), llm_id="primary")
            raise KeyError("no invocation context")

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

    @pytest.mark.anyio
    async def test_a_completed_discovery_fixes_the_set_a_failure_left(self):
        """A superseded set becomes authoritative, so drift is enforced again.

        Superseding the provisional set is a one-way correction, not a licence to
        keep re-baselining: once a discovery completes, a later build that
        registers a different set is drift.
        """
        app = App(app_name="Test Agent")
        attempts = [0]

        @app.entrypoint(load_strategy="per-execution")
        async def build(input, ctx):
            attempts[0] += 1
            app.llm(MagicMock(), llm_id="primary")
            if attempts[0] > 1:
                app.llm(MagicMock(), llm_id="fast")
            if attempts[0] > 2:
                app.llm(MagicMock(), llm_id="extra")
            if attempts[0] == 1:
                raise RuntimeError("credential unavailable outside request isolation")
            return MagicMock()

        # Fails after registering `primary` only.
        with pytest.raises(RuntimeError, match="context-free discovery"):
            await app.build_graph()

        # Completes with the full set, which supersedes the partial one.
        assert await app.build_graph() is not None

        # A third set is drift against the completed build, not a new baseline.
        with pytest.raises(LLMRegistrationDriftError, match="same LLM ids"):
            await app.build_graph()


class TestToolPodReadsTheOptIn:
    """runner-shared decides partial-registry retention by reading the App.

    ``TenantRuntime.uses_per_invocation_build`` looks the opt-in up on the
    registered App by name across the package boundary. Every other test stubs
    or overrides that method, so nothing else exercises the real lookup, and a
    rename on either side would silently disable the retention rather than
    fail.
    """

    @staticmethod
    def _sees_the_opt_in(app: Any) -> bool:
        from types import SimpleNamespace

        from agent_engine_runner_shared.runtime import TenantRuntime

        # The real method against a real App: the only thing supplied is the
        # builder it reads, so this pins the attribute it reads by name.
        return TenantRuntime.uses_per_invocation_build(
            SimpleNamespace(_graph_builder=app)
        )

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

    The Tool Pod populates its per-pod registry by building the graph with no
    invocation context, and re-runs that load inside each request's credential
    isolation. A discovery that fails partway through registration must not
    stop a later load from completing, or the models the failure cut short
    would never resolve on that pod.
    """

    @staticmethod
    def _pod_serving(app: App) -> Any:
        from agent_engine_runner_shared.server.tool import ToolServer

        runtime = MagicMock()
        runtime._tools = {}
        runtime._tool_definitions = {}
        runtime._graph_builder = app
        runtime.uses_per_invocation_build = lambda: True
        return ToolServer(runtime)

    @pytest.mark.anyio
    async def test_a_partial_discovery_failure_recovers_on_a_later_load(self):
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = MagicMock(), MagicMock()

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

    @pytest.mark.anyio
    async def test_a_context_requiring_builder_recovers_the_model_it_registered(self):
        """Recovery also holds when the builder never completes on this path.

        An opted-in builder registers its models and then reads invocation
        context, so every context-free discovery fails after the registrations.
        A model whose construction failed on the first attempt must still join
        the pod's registry once it succeeds, even though the discovery that
        registers it goes on to fail.
        """
        from agent_engine_runner_shared import hooks

        app = App(app_name="Test Agent")
        attempts = [0]
        primary, fast = MagicMock(), MagicMock()

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
