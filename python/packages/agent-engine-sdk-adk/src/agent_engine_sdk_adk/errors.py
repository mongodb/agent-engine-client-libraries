"""Errors for the durable ADK execution contract."""

__all__ = ["UnsupportedDurableADKError"]


class UnsupportedDurableADKError(RuntimeError):
    """The invocation cannot run under the durable ADK execution contract."""
