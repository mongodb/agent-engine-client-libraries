"""Subagent stream attribution for LangGraph ``astream(subgraphs=True)``.

Owns the ``subagent_start`` / ``subagent_end`` / sourced-token state machine
that ``LangGraphBaseAgent.stream`` asks on each item. LangGraph ``interrupt()``
and custom-event framing live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command, Interrupt, Overwrite
from agent_engine_sdk import StreamEvent

from agent_engine_sdk_langgraph.deep_agent_task import (
    DeepAgentTaskCall,
    iter_task_calls,
)
from agent_engine_sdk_langgraph.messages import normalize_content

# LangGraph namespace segment shape: ``"<node_name>:<task_id>"``.
_NS_TASK_SEP = ":"
# LangChain/LangGraph pregel node names; never identify a real subagent.
_NS_SENTINEL_NAMES = frozenset({"agent", "model", "tools"})


def source_from_namespace(namespace: tuple[str, ...]) -> str:
    """Extract subagent attribution from a LangGraph subgraph namespace tuple.

    ``astream(..., subgraphs=True)`` yields a tuple of ``"<node>:<task_id>"``
    segments identifying the subgraph path; the root graph yields ``()``.
    Walk right-to-left, strip ``:task_id``, return the first non-sentinel
    name. Matches the deepagents streaming guide:
    https://docs.langchain.com/oss/python/deepagents/streaming.

    ``lc_agent_name`` is deliberately not used: ``create_agent`` sets it on
    the root too (commonly to ``"model"``), so any non-empty value would fire
    spurious ``subagent_start`` events on every root LLM turn.
    """
    for segment in reversed(namespace):
        name = segment.split(_NS_TASK_SEP, 1)[0]
        if name and name not in _NS_SENTINEL_NAMES:
            return name
    return ""


def unwrap_overwrite(x: Any) -> list[Any]:
    """Unwrap ``Overwrite``-wrapped messages from ``PatchToolCallsMiddleware``.

    Materializes to a list so callers can iterate it twice (extend +
    extract) without worrying about generator exhaustion.
    """
    return list(x.value) if isinstance(x, Overwrite) else list(x)


@dataclass
class SubagentTask:
    """Per-subagent state for stream() boundary tracking.

    ``tc_id == ""`` means tokens arrived before the parent's ``task``
    tool_call assembled; the entry lives in ``pending`` until the real
    tool_call arrives and promotes it to ``tasks_by_id``.
    """

    name: str
    tc_id: str
    description: str


def _start_event(task: SubagentTask) -> StreamEvent:
    return StreamEvent(
        data={
            "source": task.name,
            "subagent_name": task.name,
            "tool_call_id": task.tc_id,
            "description": task.description,
        },
        event="subagent_start",
    )


def _end_event(task: SubagentTask, summary: str | None = None) -> StreamEvent:
    return StreamEvent(
        data={
            "source": task.name,
            "subagent_name": task.name,
            "tool_call_id": task.tc_id,
            "summary": summary or "",
        },
        event="subagent_end",
    )


@dataclass
class SubagentStreamTracker:
    """Closed state machine for subagent start/end and sourced tokens."""

    tasks_by_id: dict[str, SubagentTask] = field(default_factory=dict)
    pending: dict[str, SubagentTask] = field(default_factory=dict)

    @property
    def has_open(self) -> bool:
        return bool(self.tasks_by_id or self.pending)

    def _tc_id_for_source(self, source: str) -> str:
        """Return the tc_id of the first matching task for token attribution."""
        if not source:
            return ""
        for task in self.tasks_by_id.values():
            if task.name == source:
                return task.tc_id
        return ""

    def on_task_call(self, call: DeepAgentTaskCall) -> list[StreamEvent]:
        """Handle a real ``task`` tool_call. Promote any pending placeholder,
        insert otherwise. Emit ``subagent_start`` once per task entry."""
        placeholder = self.pending.pop(call.subagent_name, None)
        if placeholder is not None:
            # Strategy A: ``on_sourced_token`` deferred the start event so we
            # could emit it here with the real ``tool_call_id`` and ``description``.
            # Promote the placeholder into ``tasks_by_id`` and yield the
            # canonical start exactly once.
            placeholder.tc_id = call.tool_call_id
            if call.description and not placeholder.description:
                placeholder.description = call.description
            self.tasks_by_id[call.tool_call_id] = placeholder
            return [_start_event(placeholder)]

        existing = self.tasks_by_id.get(call.tool_call_id)
        if existing is not None:
            if call.description and not existing.description:
                existing.description = call.description
            return []

        task = SubagentTask(
            name=call.subagent_name,
            tc_id=call.tool_call_id,
            description=call.description,
        )
        self.tasks_by_id[call.tool_call_id] = task
        return [_start_event(task)]

    def on_sourced_token(self, source: str) -> list[StreamEvent]:
        """First sourced token signal. Insert a synthetic placeholder into
        ``pending`` if no entry exists yet for this name. Strategy A: do NOT
        emit ``subagent_start`` here — defer until ``on_task_call`` promotes
        the placeholder (so the start carries the real ``tool_call_id`` and
        ``description``) or until the drain path emits start+end for a
        never-promoted orphan."""
        for task in self.tasks_by_id.values():
            if task.name == source:
                return []
        if source in self.pending:
            return []
        self.pending[source] = SubagentTask(name=source, tc_id="", description="")
        return []

    def drain_open_tasks(self) -> list[StreamEvent]:
        """Emit a defensive ``subagent_end`` for every open task, then clear.

        Tasks already in ``tasks_by_id`` had their ``subagent_start`` emitted by
        ``on_task_call`` (placeholder-promotion or new-task branch); only the
        end is needed.

        Tasks still in ``pending`` are never-promoted synthetic-only orphans
        under Strategy A — their start was deferred. To preserve a balanced
        lifecycle on the wire, emit ``subagent_start`` immediately followed by
        ``subagent_end`` (both carrying ``tool_call_id == ""``).
        """
        events: list[StreamEvent] = []
        for task in list(self.tasks_by_id.values()):
            events.append(_end_event(task))
        for task in list(self.pending.values()):
            events.append(_start_event(task))
            events.append(_end_event(task))
        self.tasks_by_id.clear()
        self.pending.clear()
        return events

    def is_terminal_agent_update(
        self,
        namespace: tuple[str, ...],
        payload: dict[str, Any],
        all_messages: list[Any],
        captured_interrupts: list[Interrupt],
    ) -> bool:
        """Return true when an updates payload may be the final ReAct response.

        LangGraph 1.2.x can leave the async stream iterator open after the
        terminal ReAct ``agent`` update. The update already contains the final
        AI message, so the adapter treats this as a completion candidate and
        waits briefly for a subsequent event before finishing. Non-root
        subgraph updates and graphs with open subagent tasks are not terminal
        for the root run.
        """
        if (
            namespace != ()
            or captured_interrupts
            or self.has_open
            or "agent" not in payload
            or not all_messages
        ):
            return False
        last_message = all_messages[-1]
        return isinstance(last_message, AIMessage) and not (
            getattr(last_message, "tool_calls", []) or []
        )

    def process_messages_chunk(self, chunk: Any, source: str) -> list[StreamEvent]:
        """Handle one messages-mode chunk from LangGraph astream."""
        if not isinstance(chunk, AIMessageChunk):
            return []

        events: list[StreamEvent] = []
        if not source:
            for call in iter_task_calls([chunk]):
                if not call.subagent_name:
                    continue
                events.extend(self.on_task_call(call))
        else:
            events.extend(self.on_sourced_token(source))

        token = normalize_content(chunk.content)
        if token:
            events.append(
                StreamEvent(
                    data={
                        "content": token,
                        "source": source,
                        "tool_call_id": self._tc_id_for_source(source),
                    },
                    event="token",
                )
            )
        return events

    def process_add_messages_update(
        self, msgs: Any, all_messages: list[Any]
    ) -> list[StreamEvent]:
        """Update path — subagent open (parent ``task`` tool_calls) and close.

        deepagents' ``task`` tool returns ``Command(update={"messages":
        [ToolMessage(..., tool_call_id=<parent_tc>)]})``, but the LangGraph
        pregel runtime unwraps the Command before streaming. So on the updates
        stream we see a plain ``{"<node>": {"messages": [ToolMessage(...)]}}``
        dict — never a raw Command. The close arrives here, not via
        ``process_command_update`` (which is kept as a defensive fallback).
        """
        extracted = unwrap_overwrite(msgs)
        all_messages.extend(extracted)
        events: list[StreamEvent] = []

        for call in iter_task_calls(extracted):
            if not call.subagent_name:
                continue
            events.extend(self.on_task_call(call))

        for msg in extracted:
            if not isinstance(msg, ToolMessage):
                continue
            tc_id = getattr(msg, "tool_call_id", None)
            if not isinstance(tc_id, str) or not tc_id:
                continue
            task = self.tasks_by_id.pop(tc_id, None)
            if task is None:
                continue
            summary = normalize_content(getattr(msg, "content", None) or "")
            events.append(_end_event(task, summary=summary))
        return events

    def process_command_update(
        self, cmd: Command, all_messages: list[Any]
    ) -> list[StreamEvent]:
        """Defensive fallback — handle a raw ``Command(update=...)`` if a future
        LangGraph version ever leaks one onto the updates stream. In practice
        the pregel runtime unwraps Commands inside the tools node, so closes
        arrive via ``process_add_messages_update`` instead.
        """
        cmd_update = cmd.update
        if not isinstance(cmd_update, dict) or "messages" not in cmd_update:
            return []
        extracted = unwrap_overwrite(cmd_update["messages"])
        all_messages.extend(extracted)
        events: list[StreamEvent] = []
        for msg in extracted:
            tc_id = getattr(msg, "tool_call_id", None)
            if not isinstance(tc_id, str) or not tc_id:
                continue
            task = self.tasks_by_id.pop(tc_id, None)
            if task is None:
                continue
            summary = normalize_content(getattr(msg, "content", None) or "")
            events.append(_end_event(task, summary=summary))
        return events
