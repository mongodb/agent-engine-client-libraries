"""
Event reading and streaming for observability.

Reads traces from JSONL files (written by tracing/exporters.py) and
execution logs (written by logging.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_STREAM_POLL_INTERVAL_SECONDS = 2.0


class SpanKind(str, Enum):
    """Span kinds for agent operations."""

    AGENT = "agent"
    A2A = "a2a"
    TOOL = "tool"
    LLM = "llm"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"


class SpanStatus(str, Enum):
    """Span status codes."""

    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class ObservabilityEvent(BaseModel):
    """
    Unified event format for the UI.

    Represents an OpenInference span in a format the TraceViewer can display.
    """

    id: str
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None

    name: str
    kind: SpanKind = SpanKind.TOOL
    status: SpanStatus = SpanStatus.UNSET

    timestamp: datetime
    duration_ms: Optional[float] = None

    session_id: Optional[str] = None
    execution_id: Optional[str] = None

    input_value: Optional[str] = None
    output_value: Optional[str] = None
    error_message: Optional[str] = None

    attributes: Dict[str, Any] = Field(default_factory=dict)


class EventsResponse(BaseModel):
    """Response model for events endpoint."""

    events: List[ObservabilityEvent]
    count: int
    session_id: Optional[str] = None
    execution_id: Optional[str] = None


def _get_trace_path() -> Path:
    """Get the trace file path from env vars or default."""
    import os

    if file_env := os.getenv("AGENTIC_TRACES_FILE"):
        return Path(file_env)
    if base_dir := os.getenv("AGENTIC_OBSERVABILITY_DIR"):
        return Path(base_dir) / "traces.jsonl"
    return Path("observability") / "traces.jsonl"


def _get_executions_path() -> Path:
    """Get the executions log file path from env vars or default."""
    import os

    if file_env := os.getenv("AGENTIC_EXECUTIONS_FILE"):
        return Path(file_env)
    if base_dir := os.getenv("AGENTIC_OBSERVABILITY_DIR"):
        return Path(base_dir) / "executions.jsonl"
    return Path("observability") / "executions.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, returning list of parsed objects."""
    if not path.exists():
        return []

    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSONL line: {e}")
    return entries


def _parse_span(span: Dict[str, Any]) -> ObservabilityEvent:
    """Convert a span to ObservabilityEvent."""
    attrs = span.get("attributes", {})

    # Timing.
    start_ns = span.get("start_time_ns", 0)
    timestamp = (
        datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
        if start_ns
        else datetime.now(timezone.utc)
    )
    duration_ns = span.get("duration_ns", 0)
    duration_ms = duration_ns / 1e6 if duration_ns else None

    # Status.
    status_info = span.get("status", {})
    status_code = status_info.get("code", "UNSET")
    status = (
        SpanStatus.OK
        if status_code == "OK"
        else SpanStatus.ERROR
        if status_code == "ERROR"
        else SpanStatus.UNSET
    )

    # Infer kind from OpenInference attributes or span name.
    kind = _infer_kind(span.get("name", ""), attrs)

    # Session context (OpenInference uses session.id).
    session_id = attrs.get("session.id") or attrs.get("session_id")
    execution_id = attrs.get("execution.id") or attrs.get("execution_id")

    # Content (OpenInference uses input.value / output.value).
    input_val = attrs.get("input.value")
    output_val = attrs.get("output.value")

    return ObservabilityEvent(
        id=span.get("span_id", ""),
        trace_id=span.get("trace_id"),
        parent_id=span.get("parent_span_id"),
        name=span.get("name", "unknown"),
        kind=kind,
        status=status,
        timestamp=timestamp,
        duration_ms=duration_ms,
        session_id=session_id,
        execution_id=execution_id,
        input_value=input_val[:500] if input_val else None,
        output_value=output_val[:500] if output_val else None,
        error_message=status_info.get("description"),
        attributes=attrs,
    )


def _parse_execution_log(log: Dict[str, Any]) -> ObservabilityEvent:
    """Convert an execution log entry to ObservabilityEvent."""
    # Parse timestamp
    timestamp_str = log.get("timestamp")
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)

    # Status mapping
    status_str = log.get("status", "started")
    status = (
        SpanStatus.OK
        if status_str in ("success", "cached", "suspend")
        else SpanStatus.ERROR
        if status_str in ("error", "blocked")
        else SpanStatus.UNSET
    )

    # Build input/output
    inputs = log.get("inputs")
    output = log.get("output")
    error = log.get("error")

    input_value = json.dumps(inputs)[:500] if inputs else None
    output_value = str(output)[:500] if output else None

    return ObservabilityEvent(
        id=log.get("id", ""),
        trace_id=log.get("trace_id"),
        parent_id=None,
        name=log.get("tool") or log.get("node", "unknown"),
        kind=SpanKind.TOOL,
        status=status,
        timestamp=timestamp,
        duration_ms=log.get("duration_ms"),
        session_id=log.get("session_id"),
        execution_id=log.get("execution_id"),
        input_value=input_value,
        output_value=output_value,
        error_message=error,
        attributes={
            "step_number": log.get("step_number"),
            "status": status_str,
        },
    )


def _infer_kind(name: str, attrs: Dict[str, Any]) -> SpanKind:
    """Infer span kind from attributes or name."""
    oi_kind = attrs.get("openinference.span.kind", "").upper()
    if oi_kind == "LLM":
        return SpanKind.LLM
    if oi_kind == "CHAIN":
        return SpanKind.CHAIN
    if oi_kind == "RETRIEVER":
        return SpanKind.RETRIEVER
    if oi_kind == "EMBEDDING":
        return SpanKind.EMBEDDING
    if oi_kind == "AGENT":
        return SpanKind.AGENT
    if oi_kind == "TOOL":
        return SpanKind.TOOL

    if attrs.get("agentic.span.kind", "").lower() == "a2a":
        return SpanKind.A2A

    # Fallback to name-based inference.
    name_lower = name.lower()
    if "llm" in name_lower or "chat" in name_lower:
        return SpanKind.LLM
    if "chain" in name_lower:
        return SpanKind.CHAIN
    if "retriev" in name_lower:
        return SpanKind.RETRIEVER
    if "embed" in name_lower:
        return SpanKind.EMBEDDING
    if "agent" in name_lower:
        return SpanKind.AGENT

    return SpanKind.TOOL


def _matches_filter(
    event: ObservabilityEvent,
    session_id: Optional[str],
    execution_id: Optional[str],
) -> bool:
    """Check if event matches the filter criteria."""
    if not session_id and not execution_id:
        return True
    if session_id and event.session_id == session_id:
        return True
    if execution_id and event.execution_id == execution_id:
        return True
    return False


def get_events(
    session_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    limit: int = 100,
    include_traces: bool = True,
    include_executions: bool = True,
) -> List[ObservabilityEvent]:
    """
    Get events from traces.jsonl and executions.jsonl.

    Args:
        session_id: Filter by session ID
        execution_id: Filter by execution ID
        limit: Maximum events to return
        include_traces: Include OpenTelemetry traces
        include_executions: Include execution logs
    """
    events = []

    # Read traces
    if include_traces:
        raw_spans = _read_jsonl(_get_trace_path())
        for span in raw_spans:
            event = _parse_span(span)
            if _matches_filter(event, session_id, execution_id):
                events.append(event)

    # Read execution logs
    if include_executions:
        raw_logs = _read_jsonl(_get_executions_path())
        for log in raw_logs:
            event = _parse_execution_log(log)
            if _matches_filter(event, session_id, execution_id):
                events.append(event)

    events.sort(key=lambda e: e.timestamp)
    return events[-limit:]


# =============================================================================
# SSE Streaming
# =============================================================================


@dataclass
class _StreamState:
    """Tracks seen event IDs for deduplication."""

    seen_ids: set = field(default_factory=set)


async def stream_events(
    session_id: Optional[str] = None,
    execution_id: Optional[str] = None,
    poll_interval: float = DEFAULT_STREAM_POLL_INTERVAL_SECONDS,
) -> AsyncGenerator[str, None]:
    """
    Stream events as Server-Sent Events.

    Polls traces.jsonl and executions.jsonl for new events and yields them in SSE format.

    Args:
        session_id: Filter by session ID
        execution_id: Filter by execution ID
        poll_interval: Seconds between polls

    Yields:
        SSE-formatted event strings (JSON)
    """
    state = _StreamState()

    # Send initial events
    for event in get_events(session_id=session_id, execution_id=execution_id):
        state.seen_ids.add(event.id)
        yield event.model_dump_json()

    # Poll for new events
    while True:
        await asyncio.sleep(poll_interval)

        for event in get_events(session_id=session_id, execution_id=execution_id):
            if event.id not in state.seen_ids:
                state.seen_ids.add(event.id)
                yield event.model_dump_json()
