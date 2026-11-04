"""Apply the provider-specific shape of one LLM request.

The adapter's normal work is provider-neutral: convert platform messages, call
the client, convert the response back. A provider can still require a request
shape that no earlier stage can express — a marker that has to sit on one
specific block, say, which a wrapped model loses long before the provider call.

Those rules live in one module per provider. The adapter calls
:func:`prepare_provider_call` and never names a provider itself, so this
dispatcher is also the place a second provider would be wired in.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from langchain_core.messages import BaseMessage

from agent_engine_sdk import Message
from agent_engine_sdk_langgraph import anthropic_cache
from agent_engine_sdk_langgraph.messages import platform_messages_to_lc


class PreparedCall(NamedTuple):
    """Messages and call options ready to hand to the provider client."""

    messages: list[BaseMessage]
    options: dict[str, Any]


def prepare_provider_call(
    llm: Any, messages: list[Message], options: dict[str, Any]
) -> PreparedCall:
    """Return the provider's request shape for one call.

    The caller's ``options`` are never mutated, and a provider with no rules
    here gets them back unchanged.
    """
    lc_messages = platform_messages_to_lc(messages)
    if anthropic_cache.matches(llm):
        prepared_messages, prepared_options = anthropic_cache.prepare(
            llm, lc_messages, options
        )
        return PreparedCall(prepared_messages, prepared_options)
    return PreparedCall(lc_messages, dict(options))
