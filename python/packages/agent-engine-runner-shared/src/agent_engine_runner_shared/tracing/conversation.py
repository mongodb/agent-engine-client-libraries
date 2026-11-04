"""Trim an LLM span's input messages to the messages added since the last call.

Every LLM call receives the whole conversation, so an OpenInference span
carries the entire transcript under ``llm.input_messages.*``. A run with N
calls therefore stores the conversation N times and the trace grows
quadratically in the number of steps.

The tracker rewrites each span's input messages to the messages added since
the previous call in the same trace and stream, and records where that slice
starts. A reader replays the slices to rebuild the full context. Spans
written before this existed carry no offset and read as a full list, so old
and new traces read the same way.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

# The trace UI reads these; keep the names in sync with
# `agent-engine-runner-shared/src/tracing/conversation.ts`.
ATTR_INPUT_MESSAGES_OFFSET = "agent_engine.internal.input_messages.offset"
ATTR_INPUT_MESSAGES_TOTAL = "agent_engine.internal.input_messages.total"
ATTR_INPUT_MESSAGES_STREAM = "agent_engine.internal.input_messages.stream"

# Resource attribute carrying the pod's role. The name predates the product
# rename and is the contract the runner already emits on every span, not
# branding: it is how a reader tells the agent process from the tool process.
ATTR_RUNTIME_MODE = "agentic_platform.runtime_mode"  # open-source-refs:ignore — data contract

_INPUT_MESSAGES_PREFIX = "llm.input_messages."
_INPUT_MESSAGES_KEY = re.compile(r"^llm\.input_messages\.(\d+)\.(.+)$")

# One entry per in-flight conversation. Bounded because a long-lived pod
# serves many traces; eviction only costs compression on a later span, never
# correctness, since a missing entry reads as "store the full list".
MAX_TRACKED_CONVERSATIONS = 256


def _fingerprint(message: Mapping[str, Any]) -> bytes:
    """Stable digest of one message's fields, for prefix comparison.

    Hashing keeps the tracker's memory proportional to message count rather
    than transcript size, which is the whole point of trimming the span.
    """
    payload = "\u0000".join(f"{field}\u0001{value!r}" for field, value in sorted(message.items()))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()


def has_input_messages(attributes: Mapping[str, Any]) -> bool:
    """Whether a span carries an OpenInference input-message list."""
    return any(name.startswith(_INPUT_MESSAGES_PREFIX) for name in attributes)


def trace_stream(resource: Mapping[str, Any]) -> str:
    """Identify which process wrote a span.

    The agent process and the tool process both record the conversation for
    the same call. Their message lists must not share a prefix tracker, or
    each would treat the other's progress as its own.
    """
    parts = [
        str(resource.get("service.name") or ""),
        # The resource attribute is the durable form; RUNNER_MODE covers the
        # case where the resource was built without platform attributes.
        str(resource.get(ATTR_RUNTIME_MODE) or os.getenv("RUNNER_MODE") or ""),
    ]
    return "|".join(part for part in parts if part) or "unknown"


def _collect_messages(
    attributes: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Group ``llm.input_messages.<index>.<field>`` attributes by message."""
    messages: Dict[int, Dict[str, Any]] = {}
    for key, value in attributes.items():
        match = _INPUT_MESSAGES_KEY.match(key)
        if not match:
            continue
        messages.setdefault(int(match.group(1)), {})[match.group(2)] = value
    return messages


class ConversationDeltaTracker:
    """Remembers, per trace and stream, the input messages already stored.

    Stream separates the agent-side span from the tool-pod span for the same
    call: both carry the conversation, they are exported by different
    processes, and their message indices must not be compared.
    """

    def __init__(self, max_conversations: int = MAX_TRACKED_CONVERSATIONS):
        self._max_conversations = max_conversations
        self._stored: OrderedDict[tuple[str, str], List[bytes]] = OrderedDict()

    def copy(self) -> "ConversationDeltaTracker":
        """Independent tracker holding the same progress.

        The exporter trims a batch on a copy and adopts it only after the write
        succeeds, so a failed write cannot advance progress the store never
        recorded.
        """
        clone = ConversationDeltaTracker(self._max_conversations)
        clone._stored = OrderedDict(self._stored)
        return clone

    def trim(
        self,
        attributes: Mapping[str, Any],
        *,
        trace_id: str,
        stream_id: str,
    ) -> Dict[str, Any]:
        """Return ``attributes`` with only the new input messages kept."""
        messages = _collect_messages(attributes)
        if not messages:
            return dict(attributes)

        total = max(messages) + 1
        fingerprints = [_fingerprint(messages.get(index, {})) for index in range(total)]

        key = (stream_id, trace_id)
        stored = self._stored.get(key, [])
        # Only a strict extension can be expressed as an offset. A retry or
        # a branch that rewrote history stores its full list instead.
        extends = len(stored) <= total and fingerprints[: len(stored)] == stored
        offset = len(stored) if extends else 0
        self._remember(key, fingerprints)

        trimmed = {
            name: value
            for name, value in attributes.items()
            if not name.startswith(_INPUT_MESSAGES_PREFIX)
        }
        # OpenInference serializes the same prompt into input.value, so leaving
        # it keeps the trace quadratic; readers replay the trimmed message list
        # instead, and GET /events therefore previews no input for LLM spans.
        trimmed.pop("input.value", None)
        trimmed.pop("input.mime_type", None)
        for index in range(offset, total):
            for field, value in messages.get(index, {}).items():
                trimmed[f"{_INPUT_MESSAGES_PREFIX}{index - offset}.{field}"] = value
        trimmed[ATTR_INPUT_MESSAGES_OFFSET] = offset
        trimmed[ATTR_INPUT_MESSAGES_TOTAL] = total
        trimmed[ATTR_INPUT_MESSAGES_STREAM] = stream_id
        return trimmed

    def _remember(self, key: tuple[str, str], fingerprints: List[bytes]) -> None:
        self._stored[key] = fingerprints
        self._stored.move_to_end(key)
        while len(self._stored) > self._max_conversations:
            self._stored.popitem(last=False)
