"""OpenInference span-kind annotation for platform-created spans.

Framework auto-instrumentation annotates the spans it creates with
``openinference.span.kind``, which is how a tracing backend decides whether a
span is an agent step, an LLM call, a retrieval, and so on. The spans the Runner
SDK creates itself — the AER and tool-pod request spans — are built by hand and
so must declare their own kind; without it they reach a customer's configured
backend with no kind to display or filter on.

Kinds are restricted to the values the platform's own span-kind inference
understands (``_infer_kind`` in ``events.py``, mirrored by ``inferKind`` in the
orchestration engine). That inference reads this attribute first and falls back
to matching the span *name*, so an unrecognised value would leave the platform
showing a name-derived kind while the customer's backend shows the declared one.
``GUARDRAIL`` for the guardrails-check span is the notable casualty: a real
kind, but not one the platform recognises yet.

No span here declares ``LLM``, deliberately. The tool pod's ``/invoke_llm``
routes are a transport hop, not a model call: the hop runs ``llm.astream()`` on
a real LangChain model, which ``LangChainInstrumentor`` already wraps in its own
``LLM`` span carrying ``llm.model_name`` and ``llm.token_count.*`` — and the
caller's ``SecureWrappedLLM`` is itself a ``BaseChatModel``, so it is wrapped
too. A third ``LLM`` span with none of those attributes would inflate LLM-call
counts and dilute per-call token and cost averages in backends that aggregate by
kind. ``CHAIN`` describes what the hop actually is and leaves the real model
spans to carry the model semantics.

The kind survives metadata-only export: it is a fixed enum value rather than
request data, and ``ContentPolicyOTLPSpanExporter``'s denylist does not match
this key.

This module deliberately sits outside ``tracing/``, which imports the
OpenTelemetry SDK at module scope and is therefore unimportable without the
optional ``tracing`` extra. Keeping it dependency-free lets the AER and tool-pod
servers name a kind at module scope, the way ``events.py`` reads span attributes
without depending on the SDK.
"""

from __future__ import annotations

from enum import Enum

# The memory server carries its own copy of this literal
# (``mongomem_core/observability/span_kinds.py``). The two packages have
# separate dependency graphs, so the constant is duplicated rather than shared,
# the same way ``inferKind`` duplicates ``_infer_kind``.
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"


class OpenInferenceSpanKind(str, Enum):
    """Span kinds the Runner SDK declares on the spans it creates itself."""

    AGENT = "AGENT"
    CHAIN = "CHAIN"
    TOOL = "TOOL"


__all__ = ["OPENINFERENCE_SPAN_KIND", "OpenInferenceSpanKind"]
