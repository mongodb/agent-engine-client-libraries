"""Traces ``SkillsMiddleware.before_agent``/``abefore_agent`` (deepagents).

Its ``ls()``/``download_files()`` calls happen entirely inside a third-party
package. This wraps the hook in a single span rather than inventing internal
phase boundaries (request/download/parse/register) that don't exist as call
sites we own; splitting further would require forking deepagents, which is
out of scope.

Patches the class methods rather than subclassing: ``create_deep_agent``
constructs ``SkillsMiddleware`` itself in three places (the main agent, the
auto-injected general-purpose subagent, and any subagent with its own
``skills=``), none of which we can intercept to substitute a subclass
without also disabling deepagents' own ``skills=`` handling for those other
two paths -- and disabling it silently drops skills the caller asked for on
the general-purpose subagent. Patching the shared class instruments every
instance no matter which of those three paths constructed it, while leaving
deepagents' own wiring untouched. Safe here because Python classes are
mutable; the same trick isn't available in the TS port, since deepagents-js
exports its middleware factory as a frozen ESM binding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware.skills import SkillsMiddleware

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

from agent_engine_runner_shared.span_kinds import (
    OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind,
)
from agent_engine_runner_shared.span_names import (
    ATTR_SKILLS_LOADED_COUNT,
    ATTR_SKILLS_SOURCE_COUNT,
    SKILLS_MIDDLEWARE_BEFORE_AGENT,
)

_original_before_agent = SkillsMiddleware.before_agent
_original_abefore_agent = SkillsMiddleware.abefore_agent


def _loaded_count(cache_hit: bool, result: Any) -> int:
    if cache_hit or not result:
        return 0
    return len(result.get("skills_metadata", []))


def _traced_before_agent(
    self: SkillsMiddleware, state: Any, runtime: Runtime, config: RunnableConfig
) -> Any:
    try:
        from agent_engine_runner_shared.tracing import get_tracer
    except Exception:
        return _original_before_agent(self, state, runtime, config)

    cache_hit = "skills_metadata" in state
    tracer = get_tracer("agent-engine-sdk-langgraph.skills")
    with tracer.start_as_current_span(
        SKILLS_MIDDLEWARE_BEFORE_AGENT,
        attributes={
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value,
            ATTR_SKILLS_SOURCE_COUNT: len(self.sources),
        },
    ) as span:
        result = _original_before_agent(self, state, runtime, config)
        span.set_attribute(ATTR_SKILLS_LOADED_COUNT, _loaded_count(cache_hit, result))
        return result


async def _traced_abefore_agent(
    self: SkillsMiddleware, state: Any, runtime: Runtime, config: RunnableConfig
) -> Any:
    try:
        from agent_engine_runner_shared.tracing import get_tracer
    except Exception:
        return await _original_abefore_agent(self, state, runtime, config)

    cache_hit = "skills_metadata" in state
    tracer = get_tracer("agent-engine-sdk-langgraph.skills")
    with tracer.start_as_current_span(
        SKILLS_MIDDLEWARE_BEFORE_AGENT,
        attributes={
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value,
            ATTR_SKILLS_SOURCE_COUNT: len(self.sources),
        },
    ) as span:
        result = await _original_abefore_agent(self, state, runtime, config)
        span.set_attribute(ATTR_SKILLS_LOADED_COUNT, _loaded_count(cache_hit, result))
        return result


def apply() -> None:
    """Idempotently patch ``SkillsMiddleware`` to emit the wrapper span.

    Safe to call more than once (e.g. from multiple import sites): re-patches
    from the same saved originals rather than wrapping an already-wrapped
    method.
    """
    SkillsMiddleware.before_agent = _traced_before_agent
    SkillsMiddleware.abefore_agent = _traced_abefore_agent


__all__ = ["apply"]
