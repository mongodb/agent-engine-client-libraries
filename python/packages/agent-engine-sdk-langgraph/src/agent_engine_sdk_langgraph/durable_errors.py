"""Errors shared by durable LangGraph adapter components."""

from agent_engine_runner_shared.workflow.activity import DurableActivityControlFlow


class UnsupportedDurableGraphError(DurableActivityControlFlow):
    """A LangGraph API with no durable_workflow meaning was used durably."""
