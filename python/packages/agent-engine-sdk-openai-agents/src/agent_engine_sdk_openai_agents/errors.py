"""Errors for the durable OpenAI Agents execution contract."""

__all__ = ["DurableOpenAIAgentsStateError", "UnsupportedDurableOpenAIAgentsError"]


class UnsupportedDurableOpenAIAgentsError(RuntimeError):
    """The requested OpenAI Agents surface is outside the durable contract."""


class DurableOpenAIAgentsStateError(ValueError):
    """Native OpenAI Agents items cannot be carried as durable state."""
