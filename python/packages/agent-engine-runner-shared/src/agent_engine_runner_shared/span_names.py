"""Stable span names for the first-invoke lifecycle.

Wire contract between the Runner SDK (this module) and the orchestration
engine's ``inferKind`` (Go, mirrors ``events.py`` the same way
``observability.go`` does). Renaming a value here breaks any dashboard or
alert filtering on span name.

No manual ``llm.call`` / ``LLM``-kind span: ``LangChainInstrumentor``
already wraps ``SecureWrappedLLM._generate``/``_stream`` in its own
auto-instrumented span (see ``span_kinds.py``). ``MODEL_REQUEST_PREPARE``
and ``MODEL_RESPONSE_PROCESS`` are children of that span, not duplicates.

Dependency-free (no OTel imports), like ``span_kinds.py``.
"""

from __future__ import annotations

AER_BUILD_AGENT = "aer.build_agent"
GRAPH_BUILD = "graph.build"

# Children of the existing LangChain auto-instrumented span.
MODEL_REQUEST_PREPARE = "request.prepare"
MODEL_RESPONSE_PROCESS = "response.process"

# Wraps deepagents' SkillsMiddleware.before_agent as a single span. Its
# ls()/download_files() calls happen inside a third-party package, so this
# attributes total time to the hook without inventing internal phases that
# don't exist in code we own.
SKILLS_MIDDLEWARE_BEFORE_AGENT = "skills.middleware"

# Low-cardinality only; never prompts/completions/secrets.
ATTR_COLD_START = "cold_start"
ATTR_CACHE_HIT = "cache_hit"
ATTR_SKILLS_SOURCE_COUNT = "skills.source_count"
ATTR_SKILLS_LOADED_COUNT = "skills.loaded_count"

__all__ = [
    "AER_BUILD_AGENT",
    "GRAPH_BUILD",
    "MODEL_REQUEST_PREPARE",
    "MODEL_RESPONSE_PROCESS",
    "SKILLS_MIDDLEWARE_BEFORE_AGENT",
    "ATTR_COLD_START",
    "ATTR_CACHE_HIT",
    "ATTR_SKILLS_SOURCE_COUNT",
    "ATTR_SKILLS_LOADED_COUNT",
]
