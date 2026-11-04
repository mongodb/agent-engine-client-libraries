"""Helpers for reaching the provider client behind a LangChain runnable."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableBinding


def unwrap_chat_model(llm: Any) -> Any:
    """Return the provider client behind a ``bind_tools`` binding.

    The adapter binds tools before invoking, so the object it holds is a
    ``RunnableBinding`` around the client the agent registered. Provider
    detection and feature probes must inspect the client, not the binding.
    """
    return llm.bound if isinstance(llm, RunnableBinding) else llm
