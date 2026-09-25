"""skills_tracing.apply() -- patched SkillsMiddleware.before_agent/abefore_agent."""

from __future__ import annotations

from typing import Any, cast

import pytest
from deepagents.middleware.skills import SkillsMiddleware
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_engine_sdk_langgraph import skills_tracing
from agent_engine_runner_shared.span_names import (
    ATTR_SKILLS_LOADED_COUNT,
    ATTR_SKILLS_SOURCE_COUNT,
    SKILLS_MIDDLEWARE_BEFORE_AGENT,
)


@pytest.fixture
def in_memory_tracer(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    return exporter


class _FakeBackend:
    """Minimal backend stub -- ls()/download_files() are stubbed out per test.

    Cast to ``Any`` at call sites below: it doesn't implement the full
    ``BackendProtocol``, which is fine since ``before_agent``/``abefore_agent``
    are monkeypatched in every test here and never actually call the backend.
    """


@pytest.fixture(autouse=True)
def apply_patch():
    skills_tracing.apply()
    yield


def test_before_agent_emits_wrapper_span(in_memory_tracer, monkeypatch):
    middleware = SkillsMiddleware(
        backend=cast(Any, _FakeBackend()), sources=["/skills/a/"]
    )
    monkeypatch.setattr(
        skills_tracing,
        "_original_before_agent",
        lambda self, state, runtime, config: {
            "skills_metadata": [{"name": "s1"}, {"name": "s2"}]
        },
    )

    result = middleware.before_agent(cast(Any, {}), cast(Any, None), cast(Any, {}))

    assert result == {"skills_metadata": [{"name": "s1"}, {"name": "s2"}]}
    spans = in_memory_tracer.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == SKILLS_MIDDLEWARE_BEFORE_AGENT
    assert span.attributes[ATTR_SKILLS_SOURCE_COUNT] == 1
    assert span.attributes[ATTR_SKILLS_LOADED_COUNT] == 2


def test_before_agent_cache_hit_records_zero_loaded(in_memory_tracer, monkeypatch):
    middleware = SkillsMiddleware(
        backend=cast(Any, _FakeBackend()), sources=["/skills/a/"]
    )
    monkeypatch.setattr(
        skills_tracing,
        "_original_before_agent",
        lambda self, state, runtime, config: None,
    )

    result = middleware.before_agent(
        cast(Any, {"skills_metadata": []}), cast(Any, None), cast(Any, {})
    )

    assert result is None
    span = in_memory_tracer.get_finished_spans()[0]
    assert span.attributes[ATTR_SKILLS_LOADED_COUNT] == 0


@pytest.mark.anyio
async def test_abefore_agent_emits_wrapper_span(in_memory_tracer, monkeypatch):
    middleware = SkillsMiddleware(
        backend=cast(Any, _FakeBackend()), sources=["/skills/a/", "/skills/b/"]
    )

    async def fake_original(self, state, runtime, config):
        return {"skills_metadata": [{"name": "s1"}]}

    monkeypatch.setattr(skills_tracing, "_original_abefore_agent", fake_original)

    result = await middleware.abefore_agent(
        cast(Any, {}), cast(Any, None), cast(Any, {})
    )

    assert result == {"skills_metadata": [{"name": "s1"}]}
    span = in_memory_tracer.get_finished_spans()[0]
    assert span.name == SKILLS_MIDDLEWARE_BEFORE_AGENT
    assert span.attributes[ATTR_SKILLS_SOURCE_COUNT] == 2
    assert span.attributes[ATTR_SKILLS_LOADED_COUNT] == 1


def test_before_agent_falls_back_when_tracing_unavailable(monkeypatch):
    """Mirrors AERServer._traced_step's untraced fallback: the optional
    ``tracing`` extra may not be installed.
    """
    middleware = SkillsMiddleware(
        backend=cast(Any, _FakeBackend()), sources=["/skills/a/"]
    )
    monkeypatch.setattr(
        skills_tracing,
        "_original_before_agent",
        lambda self, state, runtime, config: {"skills_metadata": [{"name": "s1"}]},
    )

    import sys

    monkeypatch.setitem(sys.modules, "agent_engine_runner_shared.tracing", None)

    result = middleware.before_agent(cast(Any, {}), cast(Any, None), cast(Any, {}))

    assert result == {"skills_metadata": [{"name": "s1"}]}


def test_apply_is_idempotent():
    """Calling apply() twice must not double-wrap (and thus double-span)."""
    skills_tracing.apply()
    skills_tracing.apply()

    assert SkillsMiddleware.before_agent is skills_tracing._traced_before_agent
    assert SkillsMiddleware.abefore_agent is skills_tracing._traced_abefore_agent
