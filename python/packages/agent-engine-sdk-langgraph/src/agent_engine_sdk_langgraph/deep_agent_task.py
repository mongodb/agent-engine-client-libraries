"""Parse Deep Agent ``task`` tool calls for adapter-owned behavior."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

_DISPATCH_TOOL_NAME = "task"


def _tool_call_value(tool_call: Any, key: str) -> Any:
    """Read a field from dict and pydantic-like ToolCall representations."""
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


@dataclass(frozen=True)
class DeepAgentTaskCall:
    """Normalized identity and stream metadata for one ``task`` call."""

    tool_call_id: str
    subagent_name: str
    description: str


def parse_task_call(tool_call: Any) -> DeepAgentTaskCall | None:
    """Normalize one Deep Agent task call, retaining malformed identity as empty."""
    if _tool_call_value(tool_call, "name") != _DISPATCH_TOOL_NAME:
        return None

    raw_id = _tool_call_value(tool_call, "id")
    args = _tool_call_value(tool_call, "args")
    args_dict = args if isinstance(args, dict) else {}
    return DeepAgentTaskCall(
        tool_call_id=raw_id if isinstance(raw_id, str) else "",
        subagent_name=str(args_dict.get("subagent_type", "") or ""),
        description=str(args_dict.get("description", "") or ""),
    )


def iter_task_calls(messages_or_chunks: Iterable[Any]) -> Iterator[DeepAgentTaskCall]:
    """Yield task calls with IDs from LangChain messages or chunks.

    Stream attribution historically ignores incomplete tool-call chunks until
    their stable ID is assembled. Durable dispatch performs its stricter name
    and ID validation directly through :func:`parse_task_call`.
    """
    for message in messages_or_chunks:
        for tool_call in getattr(message, "tool_calls", None) or []:
            parsed = parse_task_call(tool_call)
            if parsed is not None and parsed.tool_call_id:
                yield parsed
