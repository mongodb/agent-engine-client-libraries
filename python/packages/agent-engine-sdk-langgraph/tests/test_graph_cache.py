"""App._get_or_build_graph()/warm_up() caching.

get_agent() used to call self._builder_fn() (the user's @app.entrypoint
function) on every /execute -- real, measurable first-invoke latency. This
caches the built graph, while still re-running the per-call side effects
(workflow-adapter registration) that must reflect the latest materialization
regardless of cache state.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_engine_sdk_langgraph import App
from agent_engine_runner_shared.span_names import ATTR_CACHE_HIT


@pytest.fixture
def in_memory_tracer(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    return exporter


def _make_app_with_counting_builder(delay: float = 0.0):
    app = App(app_name="Test Agent")
    calls: list[int] = []

    def build():
        calls.append(1)
        if delay:
            time.sleep(delay)
        return object()

    app._builder_fn = build
    return app, calls


class TestGraphCache:
    def test_cache_miss_builds_once(self, in_memory_tracer):
        app, calls = _make_app_with_counting_builder()

        graph = app._get_or_build_graph()

        assert graph is not None
        assert len(calls) == 1

    def test_cache_hit_skips_rebuild(self, in_memory_tracer):
        app, calls = _make_app_with_counting_builder()

        first = app._get_or_build_graph()
        second = app._get_or_build_graph()

        assert first is second
        assert len(calls) == 1

    def test_version_change_invalidates_cache(self, in_memory_tracer):
        app, calls = _make_app_with_counting_builder()

        with patch(
            "agent_engine_sdk_langgraph.runtime._adapter_version", return_value="v1"
        ):
            app._get_or_build_graph()
        with patch(
            "agent_engine_sdk_langgraph.runtime._adapter_version", return_value="v2"
        ):
            app._get_or_build_graph()

        assert len(calls) == 2, (
            "a runtime/adapter version change must rebuild the graph"
        )

    def test_concurrent_first_invocations_dedupe(self, in_memory_tracer):
        app, calls = _make_app_with_counting_builder(delay=0.05)

        results: list[object] = []

        def call():
            results.append(app._get_or_build_graph())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 1, "concurrent first invocations must build exactly once"
        assert len({id(r) for r in results}) == 1

    def test_warm_up_failure_does_not_raise(self, in_memory_tracer):
        app = App(app_name="Test Agent")

        def failing_build():
            raise RuntimeError("entrypoint blew up")

        app._builder_fn = failing_build

        assert app.warm_up() is False

        assert app._graph_cache is None

    def test_warm_up_failure_falls_back_to_lazy_build(self, in_memory_tracer):
        app = App(app_name="Test Agent")
        state = {"should_fail": True}

        def build():
            if state["should_fail"]:
                raise RuntimeError("entrypoint blew up")
            return object()

        app._builder_fn = build

        assert app.warm_up() is False
        assert app._graph_cache is None

        state["should_fail"] = False
        graph = app._get_or_build_graph()  # lazy build on first real call
        assert graph is not None

    def test_warm_up_populates_cache_for_later_get_or_build(self, in_memory_tracer):
        app, calls = _make_app_with_counting_builder()

        assert app.warm_up() is True
        app._get_or_build_graph()

        assert len(calls) == 1, (
            "get_or_build_graph after a successful warm_up is a cache hit"
        )

    def test_cache_hit_span_records_true(self, in_memory_tracer):
        app, _ = _make_app_with_counting_builder()

        app._get_or_build_graph()
        in_memory_tracer.clear()
        app._get_or_build_graph()

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == ["graph.build"]
        assert spans[0].attributes[ATTR_CACHE_HIT] is True

    def test_cache_miss_span_records_false(self, in_memory_tracer):
        app, _ = _make_app_with_counting_builder()

        app._get_or_build_graph()

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == ["graph.build"]
        assert spans[0].attributes[ATTR_CACHE_HIT] is False
