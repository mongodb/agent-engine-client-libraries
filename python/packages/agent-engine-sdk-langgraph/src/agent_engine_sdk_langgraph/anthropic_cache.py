"""Anthropic prompt caching for the tool-pod LLM adapter.

Anthropic caches a *prefix* of the request, in the order tools -> system ->
messages, and only where the request carries an explicit ``cache_control``
breakpoint. Where the breakpoint sits decides what is cached:

* on the last block of the system prompt it covers the tools and the system
  prompt only, so every conversation built from the same agent can read that
  prefix instead of writing it again;
* on the last message it covers the whole conversation prefix, which the next
  turn of a tool loop extends and re-reads.

Nothing earlier in the pipeline can place one. The framework's prompt-caching
middleware is gated on ``isinstance(model, ChatAnthropic)`` and the platform's
``SecureWrappedLLM`` is not a provider client, so the middleware skips every
call. The agent pod also rebuilds each content block as ``TextBlock(type,
text)``, so a breakpoint an agent writes on a block is dropped before the tool
pod sees it.

The adapter is the last stop before the provider call, so the breakpoints are
placed here: after OE approval, immediately before the provider formats the
request, and out of reach of agent input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage

from agent_engine_sdk_langgraph.chat_model import unwrap_chat_model

_CACHE_CONTROL_OPTION = "cache_control"
_EPHEMERAL_BREAKPOINT: Mapping[str, str] = {"type": "ephemeral"}
# Anthropic rejects a breakpoint on thinking blocks, and an empty text block has
# nothing to cache. Every other block type the platform produces is eligible.
_NON_CACHEABLE_BLOCK_TYPES = frozenset(
    {"thinking", "redacted_thinking", "reasoning", "reasoning_content"}
)


def matches(llm: Any) -> bool:
    """True when the client speaks the Anthropic wire format.

    ``langchain_anthropic`` is optional: a tool pod only has it when the agent
    registered an Anthropic client, and every other provider must keep its own
    request shape untouched.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return False
    return isinstance(unwrap_chat_model(llm), ChatAnthropic)


def prepare(
    llm: Any, messages: list[BaseMessage], options: dict[str, Any]
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Add the breakpoints Anthropic needs and return the shaped request.

    Two breakpoints, deliberately placed:

    * the system prompt's last block, so a new conversation reads tools + system
      from cache instead of rewriting them (``_mark_anthropic_system_prompt``);
    * the conversation tail, as the request-level option, because that is the
      only form that can mark a ``tool_result`` block: Anthropic rejects a
      breakpoint nested inside the block's content. A translated transport
      expands the option into a block marker itself.

    The option is consumed here rather than forwarded, because only an Anthropic
    client accepts it. ``options`` is copied, never mutated.
    """
    shaped_options = dict(options)
    cache_control = _take_anthropic_cache_control(llm, shaped_options)
    if cache_control is None:
        return messages, shaped_options
    if any(getattr(message, "type", None) != "system" for message in messages):
        shaped_options[_CACHE_CONTROL_OPTION] = cache_control
    return _mark_anthropic_system_prompt(messages, cache_control), shaped_options


def _take_anthropic_cache_control(
    llm: Any, kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """Consume the request-level cache option and return the breakpoint value.

    This value is used for *both* breakpoints: Anthropic rejects a request whose
    longer cache TTL follows a shorter one, so the system marker and the tail
    must share one TTL.

    Precedence is the invocation option, then the default ephemeral breakpoint.
    A falsy value disables placement for the call. A value configured on the
    client through ``model_kwargs`` is deliberately not consulted: the provider
    spreads those defaults into the payload after consuming this option, which
    would restore a request-level field for a translated transport and defeat a
    per-call disable.
    """
    if not matches(llm):
        return None
    value = kwargs.pop(_CACHE_CONTROL_OPTION, None)
    if value is None:
        value = _EPHEMERAL_BREAKPOINT
    if value is True:
        return dict(_EPHEMERAL_BREAKPOINT)
    if isinstance(value, dict) and value:
        return value
    return None


def _mark_anthropic_system_prompt(
    messages: list[BaseMessage], cache_control: Mapping[str, Any]
) -> list[BaseMessage]:
    """Mark the last block of the first system message, or return messages as-is.

    The first message is the one Anthropic hoists into the request's ``system``
    field; the cache entry it creates covers the tools and the system prompt, so
    a conversation that starts later can read it without writing it.
    """
    tagged = list(messages)
    if not tagged:
        return tagged
    if getattr(tagged[0], "type", None) == "system":
        tagged[0] = _attach_anthropic_breakpoint(tagged[0], cache_control)
    return tagged


def _attach_anthropic_breakpoint(
    message: BaseMessage, cache_control: Mapping[str, Any]
) -> BaseMessage:
    """Return ``message`` with a breakpoint on its last cacheable block.

    The message is rebuilt rather than mutated: the caller's messages may be
    reused by the agent pod's own bookkeeping.
    """
    content = message.content
    if isinstance(content, str):
        if not content:
            return message
        return message.model_copy(
            update={
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(cache_control),
                    }
                ]
            }
        )
    if not isinstance(content, list):
        return message
    blocks = _attach_breakpoint_to_last_block(content, cache_control)
    if blocks is None:
        return message
    return message.model_copy(update={"content": blocks})


def _attach_breakpoint_to_last_block(
    blocks: list[Any], cache_control: Mapping[str, Any]
) -> list[Any] | None:
    """Copy ``blocks`` with a breakpoint on the last cacheable one, or None.

    None means leave the message untouched: no block can carry a breakpoint, or
    the caller already placed one, in which case the caller's choice wins.
    """
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str):
            continue
        if block_type in _NON_CACHEABLE_BLOCK_TYPES:
            continue
        if block.get(_CACHE_CONTROL_OPTION) is not None:
            return None
        if block_type == "text" and not block.get("text"):
            continue
        tagged = list(blocks)
        tagged[index] = {**block, _CACHE_CONTROL_OPTION: dict(cache_control)}
        return tagged
    return None
