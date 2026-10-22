"""Distinguish an inbound caller trace from a root this runner created.

AER always starts ``aer.execute``. With no inbound traceparent that span is a
root, and ``tls_client`` injects it on the call to the OE. The OE forwards
that header, so the tool process sees a remote parent even though no caller
trace was bound. W3C traceparent cannot say who created the root, so the
outgoing tracestate carries ``agentengine=root`` for a runner-created trace.
Guardrail content is recorded only when that marker is absent.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

# Simple W3C tracestate member. A value of "root" means a runner process
# created this trace; an inbound caller does not set it.
RUNNER_ROOT_TRACESTATE_KEY = "agentengine"
RUNNER_ROOT_TRACESTATE_VALUE = "root"

_RUNNER_ROOTED_KEY: Any = None


def _runner_rooted_key() -> Any:
    global _RUNNER_ROOTED_KEY
    if _RUNNER_ROOTED_KEY is None:
        from opentelemetry import context

        _RUNNER_ROOTED_KEY = context.create_key("agentengine.runner_rooted_trace")
    return _RUNNER_ROOTED_KEY


def trace_state_marks_runner_root(trace_state: Any) -> bool:
    """True when tracestate says a runner created this trace's root."""
    if trace_state is None:
        return False
    try:
        marked = RUNNER_ROOT_TRACESTATE_KEY in trace_state
    except TypeError:
        return False
    if not marked:
        return False
    return trace_state[RUNNER_ROOT_TRACESTATE_KEY] == RUNNER_ROOT_TRACESTATE_VALUE


def merge_runner_root_trace_state(existing: str | None) -> str:
    """Return tracestate with the runner-root member first, without duplicates."""
    kept: list[str] = []
    prefix = f"{RUNNER_ROOT_TRACESTATE_KEY}="
    for part in (existing or "").split(","):
        item = part.strip()
        if not item or item.startswith(prefix):
            continue
        kept.append(item)
    member = f"{RUNNER_ROOT_TRACESTATE_KEY}={RUNNER_ROOT_TRACESTATE_VALUE}"
    return ",".join([member, *kept])


def parent_context_is_runner_rooted(parent_context: Any) -> bool:
    """True when ``parent_context`` is not an inbound caller trace.

    An invalid extract means this process is about to create the root. A
    remote parent without the runner-root marker was already bound by a caller.
    """
    from opentelemetry import trace

    span_context = trace.get_current_span(parent_context).get_span_context()
    if span_context is None or not span_context.is_valid:
        return True
    if trace_state_marks_runner_root(getattr(span_context, "trace_state", None)):
        return True
    if getattr(span_context, "is_remote", False):
        return False
    return True


def bind_runner_rooted_trace(ctx: Any, rooted: bool) -> Any:
    """Return ``ctx`` with the runner-rooted flag set for descendant spans."""
    from opentelemetry import context

    return context.set_value(_runner_rooted_key(), rooted, ctx)


def runner_rooted_trace_active() -> bool:
    """True when the active span's trace was created by a runner, not a caller.

    The flag set at span entry wins. Without it, a remote parent that carries
    the marker — or any local root — is treated as runner-created so guardrail
    content stays off unless a caller trace was already bound.
    """
    try:
        from opentelemetry import context, trace
    except ImportError:
        return False

    explicit = context.get_value(_runner_rooted_key())
    if explicit is True:
        return True
    if explicit is False:
        return False

    span = trace.get_current_span()
    span_context = span.get_span_context()
    if trace_state_marks_runner_root(getattr(span_context, "trace_state", None)):
        return True
    parent = getattr(span, "parent", None)
    if parent is not None and getattr(parent, "is_valid", False):
        if trace_state_marks_runner_root(getattr(parent, "trace_state", None)):
            return True
        if getattr(parent, "is_remote", False):
            return False
        # Parent was created in this process and nobody bound a caller trace.
        return True
    if span_context is not None and span_context.is_valid:
        if getattr(span_context, "is_remote", False):
            return trace_state_marks_runner_root(getattr(span_context, "trace_state", None))
        return True
    return False


@contextmanager
def runner_trace_scope(parent_context: Any) -> Iterator[None]:
    """Attach the runner-rooted flag for the span about to be started.

    ``start_as_current_span`` copies the ambient context, so the flag has to
    be attached before the span or nested ``tls_client`` calls will not see it.
    """
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        yield
        return
    rooted = parent_context_is_runner_rooted(parent_context)
    token = otel_context.attach(bind_runner_rooted_trace(otel_context.get_current(), rooted))
    try:
        yield
    finally:
        otel_context.detach(token)
