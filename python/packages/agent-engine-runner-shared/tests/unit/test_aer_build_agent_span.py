"""AER's aer.build_agent span around runtime.get_agent().

`_traced_step` is the helper `_handle_execute` uses to attribute the
`get_agent()` call: without it, the time this call takes (materializing the
graph on every /execute) is invisible inside the parent aer.execute span.
"""

from __future__ import annotations

import sys

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.span_kinds import OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind
from agent_engine_runner_shared.span_names import AER_BUILD_AGENT


@pytest.fixture
def in_memory_tracer(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    return exporter


class TestTracedStep:
    def test_opens_named_span_with_chain_kind_by_default(self, in_memory_tracer):
        with AERServer._traced_step(AER_BUILD_AGENT):
            pass

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == [AER_BUILD_AGENT]
        assert spans[0].attributes[OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKind.CHAIN.value

    def test_span_closes_even_when_body_raises(self, in_memory_tracer):
        with pytest.raises(RuntimeError):
            with AERServer._traced_step(AER_BUILD_AGENT):
                raise RuntimeError("graph build failed")

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == [AER_BUILD_AGENT]
        assert spans[0].status.status_code == trace.StatusCode.ERROR

    def test_falls_back_to_untraced_when_opentelemetry_unavailable(self, monkeypatch):
        # Same optional-tracing-extra simulation as test_aer_trace_context.py:
        # force the import inside _traced_step's try block to fail.
        monkeypatch.setitem(sys.modules, "agent_engine_runner_shared.tracing", None)

        ran = False
        with AERServer._traced_step(AER_BUILD_AGENT) as span:
            ran = True
            assert span is None

        assert ran, "the wrapped body must still execute when tracing is unavailable"
