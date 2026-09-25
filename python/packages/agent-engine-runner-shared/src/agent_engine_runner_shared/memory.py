"""
Memory utilities for Runner SDK.

Provides modular functions for working with MemoryEngine instances,
including message extraction, context building, and turn writing.

Architecture:
    - Pure functions for message extraction and context building
    - MemoryWriter class for thread-safe, non-blocking writes
    - All threading/locking concerns encapsulated here
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_engine_runner_shared.context import (
    get_current_execution_id,
    record_current_memory_metadata,
)
from agent_engine_runner_shared.types import MemoryContextProtocol, MemoryEngineProtocol
from agent_engine_runner_shared.utils import normalize_content

if TYPE_CHECKING:
    from agent_engine_sdk_memory.models import ContextResponse, SourceSpec

logger = logging.getLogger(__name__)


class MemoryToolCall(BaseModel):
    """Tool call stored on an assistant STM turn."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str
    arguments: Union[Dict[str, Any], str] = Field(default_factory=dict)


class _MemoryTurn(BaseModel):
    """Fields shared by every Memory ``write_turn`` request."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    org_id: str
    user_id: str
    project_id: str
    agent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def write_turn_arguments(self) -> Dict[str, Any]:
        """Return keyword arguments accepted by ``MemoryEngine.write_turn``."""
        return self.model_dump(exclude_none=True)

    def payload_json(self) -> bytes:
        """Return the exact JSON body OE forwards to the Memory server."""
        return self.model_dump_json(exclude_none=True).encode()


class UserMemoryTurn(_MemoryTurn):
    role: Literal["user"] = "user"
    content: str = Field(min_length=1)


class AssistantMemoryTurn(_MemoryTurn):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: Optional[List[MemoryToolCall]] = None

    @model_validator(mode="after")
    def validate_message_content(self) -> "AssistantMemoryTurn":
        if not self.content and not self.tool_calls:
            raise ValueError("assistant messages must have either content or tool_calls")
        return self


class ToolMemoryTurn(_MemoryTurn):
    role: Literal["tool"] = "tool"
    content: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    tool_name: str
    is_error: bool = False


MemoryTurn = Union[UserMemoryTurn, AssistantMemoryTurn, ToolMemoryTurn]


def _require_positive_max_tokens(max_tokens: object | None) -> None:
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("max_tokens must be a positive integer")


def extract_turn_messages(
    message: str,
    result_messages: List[Any],
    user_id: Optional[str],
    include_user_turn: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract messages from a turn for memory storage.

    Converts sdk-core Message objects to plain dicts suitable for MemoryEngine.

    The user message is graph *input* and frequently never appears in
    ``result_messages`` (many LangGraph agents only emit AIMessages as node
    output, so the seeded HumanMessage never surfaces in the updates stream).
    To guarantee the user turn is recorded, it is prepended here from
    the original ``message`` string rather than discovered in ``result_messages``.
    If the user message *is* present in ``result_messages`` (agents that echo it),
    that copy is skipped so it is not written twice.

    Args:
        message: The original user message. Prepended as the user turn when
            ``include_user_turn`` is True and ``message`` is non-empty.
        result_messages: All messages (sdk-core Message objects)
        user_id: User ID to attach to user messages
        include_user_turn: When True (default), the user message is recorded.
            Set False on resume legs, where there is no new user prompt and the
            user turn was already written when the turn first started.

    Returns:
        List of message dicts with role, content, and optional tool_calls/tool_call_id
    """
    turn_messages: List[Dict[str, Any]] = []

    # Normalize once. The dedup check below and the content we write must use
    # the same normalized form so the comparison is apples-to-apples — comparing
    # raw msg.content against a raw message string would miss when content is a
    # multimodal list (e.g. [{"type": "text", "text": "hi"}]) or differs only in
    # formatting, causing the echoed user turn to be written twice.
    normalized_message = normalize_content(message)

    # Record the user turn from the original message string, independent of
    # whether it appears in result_messages.
    if include_user_turn and message:
        turn_messages.append(
            {
                "role": "user",
                "content": normalized_message,
                "user_id": user_id,
            }
        )

    # Locate the assistant/tool tail. If the user message is present in
    # result_messages, advance past it (start_idx = i + 1) so the explicit
    # prepend above is not duplicated. Compare on the normalized content so the
    # dedup matches what would actually be written.
    start_idx = 0
    for i in range(len(result_messages) - 1, -1, -1):
        msg = result_messages[i]
        if msg.role == "user" and normalize_content(msg.content) == normalized_message:
            start_idx = i + 1
            break

    for msg in result_messages[start_idx:]:
        if msg.role == "user":
            # Honor include_user_turn for user messages discovered in
            # result_messages too, not just the prepended one. On a resume leg
            # (include_user_turn=False) there is no new user prompt, so any
            # user message surfaced here (e.g. replayed history from a future
            # adapter or a values-mode stream) must not be written.
            if not include_user_turn:
                continue
            turn_messages.append(
                {
                    "role": "user",
                    "content": normalize_content(msg.content),
                    "user_id": user_id,
                }
            )
        elif msg.role == "assistant":
            content = normalize_content(msg.content)
            has_tool_calls = msg.tool_calls
            if not content and not has_tool_calls:
                continue
            entry: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if has_tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }
                    for tc in msg.tool_calls
                ]
            turn_messages.append(entry)
        elif msg.role == "tool":
            turn_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(msg, "tool_call_id", "") or "",
                    "name": getattr(msg, "name", "") or "",
                    "content": normalize_content(msg.content),
                    "is_error": bool(getattr(msg, "is_error", False)),
                }
            )

    return turn_messages


def build_memory_turns(
    *,
    message: str,
    result_messages: List[Any],
    session_id: str,
    org_id: str,
    user_id: str,
    project_id: str,
    include_user_turn: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
) -> Iterator[MemoryTurn]:
    """Yield typed Memory turns for incremental native-checkpoint delivery.

    ``idempotency_key`` is stamped on every yielded turn; callers must only
    supply it when exactly one turn will be emitted (see
    :func:`write_turn_to_memory` for the single-turn guard).
    """
    messages = extract_turn_messages(
        message=message,
        result_messages=result_messages,
        user_id=user_id,
        include_user_turn=include_user_turn,
    )
    for msg in messages:
        role = msg.get("role", "user")
        if role == "user":
            yield UserMemoryTurn(
                session_id=session_id,
                org_id=org_id,
                user_id=user_id,
                project_id=project_id,
                content=msg.get("content", ""),
                metadata=metadata,
                idempotency_key=idempotency_key,
            )
        elif role == "assistant":
            tool_calls = None
            if msg.get("tool_calls"):
                tool_calls = [
                    MemoryToolCall(
                        id=tool_call.get("id", ""),
                        name=tool_call.get("name", ""),
                        arguments=tool_call.get("args", {}),
                    )
                    for tool_call in msg["tool_calls"]
                ]
            yield AssistantMemoryTurn(
                session_id=session_id,
                org_id=org_id,
                user_id=user_id,
                project_id=project_id,
                content=msg.get("content", ""),
                tool_calls=tool_calls,
                metadata=metadata,
                idempotency_key=idempotency_key,
            )
        elif role == "tool":
            yield ToolMemoryTurn(
                session_id=session_id,
                org_id=org_id,
                user_id=user_id,
                project_id=project_id,
                content=msg.get("content", ""),
                tool_call_id=msg.get("tool_call_id", ""),
                tool_name=msg.get("name", ""),
                is_error=msg.get("is_error", False),
                metadata=metadata,
                idempotency_key=idempotency_key,
            )


def build_context(
    memory_engine: Optional[MemoryEngineProtocol],
    query: str,
    session_id: Optional[str],
    org_id: str,
    user_id: str,
    *,
    project_id: Optional[str] = None,
    visibility: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    enabled_sources: Optional[set[str]] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Build memory context for a query using the MemoryEngine.

    Returns empty string if:
    - No memory_engine configured
    - memory_engine.build_context() fails
    - No context available

    Args:
        memory_engine: The MemoryEngine instance (or None)
        query: The user's query/message
        session_id: Session ID
        org_id: Organization ID for multi-tenant isolation
        user_id: User ID
        visibility: Visibility scope filter (private, shared, org)
        enabled_sources: Memory sources to include. ``None`` uses the engine
            default (episodic, semantic); include "stm" for recent turns.
        max_tokens: Optional gross context-construction budget. After retrieval
            and ranking, the server subtracts a 500-token formatting reserve,
            then greedily selects whole memory chunks that fit in the
            remainder. Positive values at or below 500 leave no budget for
            memories. Values above 500 can still yield empty context when no
            chunk fits. Omit to use the server default.

    Returns:
        Formatted context string, or empty string
    """
    if memory_engine is None:
        return ""
    if not session_id:
        logger.warning("session_id required for memory context; skipping context build")
        return ""
    _require_positive_max_tokens(max_tokens)

    try:
        kwargs: Dict[str, Any] = dict(
            query=query,
            session_id=session_id,
            org_id=org_id,
            user_id=user_id,
            project_id=project_id,
            visibility=visibility,
            enabled_sources=enabled_sources,
            format_style="jinja2",
        )
        if metadata_filter is not None:
            kwargs["metadata_filter"] = metadata_filter
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        context_response = memory_engine.build_context(**kwargs)

        if isinstance(context_response, MemoryContextProtocol) or hasattr(
            context_response, "formatted_context"
        ):
            try:
                result = context_response.formatted_context
            except AttributeError:
                logger.warning(
                    "context_response has formatted_context attribute but property access failed"
                )
                result = ""
        else:
            logger.warning(
                f"build_context returned unexpected type: {type(context_response)}. "
                f"Expected MemoryContextProtocol with formatted_context property."
            )
            result = ""

        if result:
            logger.debug(f"Built memory context: {len(result)} chars")
        return result

    except Exception as e:
        logger.warning(f"Failed to build memory context: {e}", exc_info=True)
        return ""


def build_context_from_sources(
    memory_engine: Optional[Any],
    query: str,
    sources: "list[SourceSpec]",
    org_id: str,
    user_id: str,
    *,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    visibility: Optional[str] = None,
    rerank: bool = False,
    max_tokens: Optional[int] = None,
) -> "ContextResponse":
    """Build memory context from an explicit, per-source-configured source set.

    Unlike :func:`build_context`, which applies one filter and one mode to every
    source and flattens the result to a string, this preserves the full
    ``ContextResponse`` so per-source metadata (``ranking_strategy``,
    ``source_outcomes``) survives to the caller. Each ``SourceSpec`` carries its
    own retrieval mode, metadata filter, and candidate count.

    Returns an empty ``ContextResponse`` (blank ``formatted_context``, default
    metadata) when memory is disabled or the read fails, degrading gracefully
    like :func:`build_context` rather than raising into the agent turn.
    """
    from agent_engine_sdk_memory.models import ContextMetadata, ContextResponse

    def _empty() -> "ContextResponse":
        return ContextResponse(formatted_context="", metadata=ContextMetadata())

    if memory_engine is None:
        return _empty()
    _require_positive_max_tokens(max_tokens)

    try:
        kwargs: Dict[str, Any] = dict(
            query=query,
            sources=sources,
            org_id=org_id,
            user_id=user_id,
            project_id=project_id,
            visibility=visibility,
            rerank=rerank,
        )
        if session_id is not None:
            kwargs["session_id"] = session_id
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return memory_engine.build_context2(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to build per-source memory context: {e}", exc_info=True)
        return _empty()


def write_turn_to_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    message: str,
    result_messages: List[Any],
    session_id: Optional[str],
    org_id: str,
    user_id: str,
    project_id: str,
    include_user_turn: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
) -> None:
    """
    Write the current turn to memory for future retrieval.

    Writes each message as a separate write_turn call following the
    MemoryEngine API:
    - User messages: role="user", content=...
    - Assistant messages: role="assistant", content=..., tool_calls=...
    - Tool messages: role="tool", content=..., tool_call_id=..., tool_name=...

    No-op if:
    - No memory_engine configured
    - No messages to write
    - write_turn() fails (logs warning)

    Args:
        memory_engine: The MemoryEngine instance (or None)
        message: Original user message
        result_messages: All messages from graph execution
        session_id: Session ID
        org_id: Organization ID for multi-tenant isolation
        user_id: User ID
        project_id: Project ID for multi-tenant isolation
        include_user_turn: When True (default), the user turn is recorded. Set
            False on resume legs, where there is no new user prompt and the user
            turn was already written when the turn first started; this prevents
            re-recording the user message on resume.
        metadata: Arbitrary key-value metadata stamped on every turn written for
            this call. Omitting it preserves prior behavior.
        idempotency_key: Deduplication key stamped on the written turn so the
            server can collapse retried writes. Only meaningful when the call
            emits exactly one turn (the app-bound record_turn shape); ignored
            for multi-message writes, where one key cannot represent the whole
            set.
    """
    if memory_engine is None:
        return
    if not session_id:
        logger.warning("session_id required for memory write; skipping turn write")
        return

    try:
        # A caller-supplied idempotency key deduplicates one logical write.
        # Stamp it only when exactly one turn will be written — one key cannot
        # represent a multi-message set, and stamping every turn would let
        # key-based dedupe collapse distinct history. Count via the pure
        # message extraction (no turn construction) so writes stay lazy and a
        # malformed later message cannot discard earlier writes.
        turn_messages = extract_turn_messages(
            message=message,
            result_messages=result_messages,
            user_id=user_id,
            include_user_turn=include_user_turn,
        )
        turn_idempotency_key = (
            idempotency_key if idempotency_key and len(turn_messages) == 1 else None
        )
        turns = build_memory_turns(
            message=message,
            result_messages=result_messages,
            session_id=session_id,
            org_id=org_id,
            user_id=user_id,
            project_id=project_id,
            include_user_turn=include_user_turn,
            metadata=metadata,
            idempotency_key=turn_idempotency_key,
        )

        written = 0
        for turn in turns:
            memory_engine.write_turn(**turn.write_turn_arguments())
            written += 1

        if written:
            logger.debug(f"Wrote {written} messages to memory")
        else:
            logger.debug("No messages to write to memory")

    except Exception as e:
        logger.warning(f"Failed to write turn to memory: {e}")


# =============================================================================
# MemoryWriter - Thread-safe, Non-blocking Memory Operations
# =============================================================================


class MemoryWriter:
    """
    Thread-safe, non-blocking memory writer.

    Encapsulates all threading concerns for memory operations, providing:
    - Non-blocking writes via background thread pool
    - Thread-safe access to memory engine
    - Retry with exponential backoff
    - Graceful shutdown with pending write tracking

    Usage:
        writer = MemoryWriter(memory_engine)

        # Non-blocking write (returns immediately)
        writer.write_turn_async(message, result_messages, session_id, org_id, user_id, project_id)

        # Graceful shutdown
        writer.shutdown()

    Note:
        The runtime should create ONE MemoryWriter instance and reuse it.
        Thread safety is guaranteed for concurrent calls.
    """

    def __init__(
        self,
        memory_engine: MemoryEngineProtocol,
        max_workers: int = 2,
        max_retries: int = 3,
    ):
        """
        Initialize the memory writer.

        Args:
            memory_engine: The MemoryEngine instance to write to
            max_workers: Maximum concurrent write threads (default: 2)
            max_retries: Maximum retry attempts on failure (default: 3)
        """
        self._memory_engine = memory_engine
        self._max_retries = max_retries

        # Thread pool for background writes
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="memory_writer",
        )

        # Thread safety
        self._write_lock = threading.Lock()
        self._pending_writes = 0
        self._pending_lock = threading.Lock()
        self._shutdown = False

        logger.info(f"MemoryWriter initialized (workers={max_workers}, retries={max_retries})")

    @property
    def memory_engine(self) -> MemoryEngineProtocol:
        """Get the underlying memory engine."""
        return self._memory_engine

    @property
    def pending_writes(self) -> int:
        """Get the number of pending background writes."""
        with self._pending_lock:
            return self._pending_writes

    def write_turn_sync(
        self,
        message: str,
        result_messages: List[Any],
        session_id: Optional[str],
        org_id: str,
        user_id: str,
        project_id: str,
        include_user_turn: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        """
        Write turn to memory synchronously (blocking).

        Use this when you need guaranteed write completion before continuing.

        Args:
            message: Original user message
            result_messages: All messages from graph execution
            session_id: Session ID
            org_id: Organization ID
            user_id: User ID
            project_id: Project ID
            include_user_turn: Record the user message (default True). Pass False
                on resume legs where there is no new user prompt.
            metadata: Arbitrary key-value metadata stamped on every turn written.
            idempotency_key: Deduplication key for single-turn writes; ignored
                for multi-message writes (see :func:`write_turn_to_memory`).
        """
        with self._write_lock:
            write_turn_to_memory(
                memory_engine=self._memory_engine,
                message=message,
                result_messages=result_messages,
                session_id=session_id,
                org_id=org_id,
                user_id=user_id,
                project_id=project_id,
                include_user_turn=include_user_turn,
                metadata=metadata,
                idempotency_key=idempotency_key,
            )

    def write_turn_async(
        self,
        message: str,
        result_messages: List[Any],
        session_id: Optional[str],
        org_id: str,
        user_id: str,
        project_id: str,
        include_user_turn: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        """
        Write turn to memory asynchronously (non-blocking).

        Returns immediately while write happens in background thread.
        Includes automatic retry with exponential backoff on failure.

        Args:
            message: Original user message
            result_messages: All messages from graph execution
            session_id: Session ID
            org_id: Organization ID (required)
            user_id: User ID (required)
            project_id: Project ID (required)
            include_user_turn: Record the user message (default True). Pass False
                on resume legs where there is no new user prompt.
            metadata: Arbitrary key-value metadata stamped on every turn written.
            idempotency_key: Deduplication key for single-turn writes; ignored
                for multi-message writes (see :func:`write_turn_to_memory`).
        """
        if self._shutdown:
            logger.warning("MemoryWriter is shut down, falling back to sync write")
            self.write_turn_sync(
                message,
                result_messages,
                session_id,
                org_id,
                user_id,
                project_id,
                include_user_turn=include_user_turn,
                metadata=metadata,
                idempotency_key=idempotency_key,
            )
            return

        # Track pending writes
        with self._pending_lock:
            self._pending_writes += 1

        # Capture the current execution context so that ContextVars (e.g.
        # current_execution_id used by MemoryClient._request_headers) are
        # available inside the background thread. Python does not propagate
        # ContextVars to manually submitted ThreadPoolExecutor tasks by default.
        import contextvars

        ctx = contextvars.copy_context()

        def _do_write() -> None:
            try:
                ctx.run(
                    self._write_with_retry,
                    message=message,
                    result_messages=result_messages,
                    session_id=session_id,
                    org_id=org_id,
                    user_id=user_id,
                    project_id=project_id,
                    include_user_turn=include_user_turn,
                    metadata=metadata,
                    idempotency_key=idempotency_key,
                )
            finally:
                with self._pending_lock:
                    self._pending_writes -= 1

        self._executor.submit(_do_write)
        logger.debug(f"Memory write queued for session {session_id}")

    def _write_with_retry(
        self,
        message: str,
        result_messages: List[Any],
        session_id: Optional[str],
        org_id: str,
        user_id: str,
        project_id: str,
        include_user_turn: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        """
        Write with retry and exponential backoff.

        Internal method - use write_turn_async or write_turn_sync instead.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                with self._write_lock:
                    write_turn_to_memory(
                        memory_engine=self._memory_engine,
                        message=message,
                        result_messages=result_messages,
                        session_id=session_id,
                        org_id=org_id,
                        user_id=user_id,
                        project_id=project_id,
                        include_user_turn=include_user_turn,
                        metadata=metadata,
                        idempotency_key=idempotency_key,
                    )
                logger.debug(f"Memory write completed for session {session_id}")
                return  # Success

            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    backoff = 0.1 * (2**attempt)  # 0.1s, 0.2s, 0.4s
                    logger.warning(
                        f"Memory write attempt {attempt + 1}/{self._max_retries} failed "
                        f"for session {session_id}: {e}. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"Memory write failed after {self._max_retries} attempts "
                        f"for session {session_id}: {last_error}",
                        exc_info=True,
                    )

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the memory writer.

        Args:
            wait: If True, wait for pending writes to complete (default: True)
        """
        self._shutdown = True

        pending = self.pending_writes
        if pending > 0:
            logger.info(f"Waiting for {pending} pending memory writes to complete...")

        self._executor.shutdown(wait=wait)
        logger.info("MemoryWriter shutdown complete")


# =============================================================================
# Semantic Memory Operations
# =============================================================================


def create_semantic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    text: str,
    label: str,
    org_id: str,
    user_id: str,
    source: str = "agent",
    visibility: str = "private",
    metadata: Optional[Dict[str, Any]] = None,
    upsert: bool = True,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """
    Create a semantic memory entry (long-lived facts and knowledge).

    Semantic memories are used for:
    - Storing user profiles and preferences
    - Recording facts and knowledge
    - Persisting structured information

    Args:
        memory_engine: The MemoryEngine instance (or None)
        text: The memory content to store
        label: A unique label/identifier for this memory
        org_id: Organization ID for multi-tenancy
        user_id: User ID who owns this memory
        source: Source of the memory (default: "agent")
        visibility: Visibility scope (private, shared, org)
        metadata: Optional caller-supplied metadata dict stored with the memory
        upsert: If True, update existing memory with same label (default: True)

    Returns:
        Created document ID, or None if creation fails
    """
    if memory_engine is None:
        logger.warning("Memory not enabled, cannot create semantic memory")
        return None

    logger.info(
        f"Creating semantic memory: label={label}, org_id={org_id}, "
        f"user_id={user_id}, visibility={visibility}, text='{text[:50]}...'"
    )

    try:
        result = memory_engine.create_semantic(
            label=label,
            text=text,
            org_id=org_id,
            user_id=user_id,
            source=source,
            visibility=visibility,
            project_id=project_id,
            agent_id=agent_id,
            embedding=None,
            metadata=metadata or {},
            upsert=upsert,
        )
    except Exception as create_error:
        logger.warning(f"Failed to create semantic memory: {create_error}")
        return None

    logger.info(f"Created semantic memory: label={label}, user_id={user_id}")
    record_current_memory_metadata(action="write", memory_type="semantic", content=text)
    return str(result.id) if hasattr(result, "id") else str(result)


def search_semantic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    query: str,
    org_id: str,
    project_id: str,
    user_id: Optional[str] = None,
    visibility: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search semantic memories by query.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        query: Search query text
        org_id: Organization ID for multi-tenancy
        user_id: Optional user ID filter
        top_k: Maximum number of results

    Returns:
        List of matching semantic entries with label, text, and metadata
    """
    if memory_engine is None:
        return []

    try:
        chunks = memory_engine.fetch_semantic_memories(
            query=query,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            visibility=visibility,
            top_k=top_k,
        )

        results = []
        for chunk in chunks:
            chunk_meta = getattr(chunk, "metadata", {}) or {}
            results.append(
                {
                    "label": chunk_meta.get("label", ""),
                    "text": getattr(chunk, "text", getattr(chunk, "content", "")),
                    "source": chunk_meta.get("source", ""),
                    "visibility": getattr(chunk, "visibility", "private"),
                    # Prefer caller-supplied metadata; fall back to the internal
                    # contextual_metadata blob for extraction-created memories
                    # written before caller metadata existed. The fallback
                    # to read contextual_metadata is transitional and will be
                    # dropped after a few releases.
                    "metadata": chunk_meta.get("metadata")
                    or chunk_meta.get("contextual_metadata")
                    or {},
                    "similarity_score": getattr(chunk, "similarity_score", 0.0),
                }
            )

        logger.debug(f"Found {len(results)} semantic memories for query: {query[:50]}...")
        first = results[0] if results else {}
        record_current_memory_metadata(
            action="recall",
            memory_type="semantic",
            content=str(first.get("text") or query),
            relevance_score=first.get("similarity_score"),
            query=query,
        )
        return results

    except Exception as e:
        logger.warning(f"Failed to search semantic memory: {e}")
        return []


def get_semantic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    org_id: str,
    project_id: str,
    label: str,
    user_id: Optional[str] = None,
    visibility: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific semantic memory by label.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        org_id: Organization ID
        label: Memory label

    Returns:
        Dict with memory details, or None if not found
    """
    if memory_engine is None:
        return None

    try:
        result = memory_engine.get_semantic(
            org_id=org_id,
            project_id=project_id,
            label=label,
            user_id=user_id,
            visibility=visibility,
        )
        if result:
            content = result.get("text") or result.get("content") or ""
            record_current_memory_metadata(
                action="recall",
                memory_type="semantic",
                content=content or label,
            )
            return {
                "id": str(result.get("id") or result.get("_id", "")),
                "label": result.get("label", label),
                "text": content,
                "source": result.get("source", ""),
                "visibility": result.get("visibility", ""),
                # Prefer caller-supplied metadata; fall back to the internal
                # contextual_metadata blob for extraction-created memories.
                "metadata": result.get("metadata") or result.get("contextual_metadata") or {},
            }
        record_current_memory_metadata(action="recall", memory_type="semantic", content=label)
        return None
    except Exception as e:
        logger.warning(f"Failed to get semantic memory: {e}")
        return None


# =============================================================================
# Taxonomic Memory Operations
# =============================================================================


def create_taxonomic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    domain: str,
    term: str,
    definition: str,
    org_id: str,
    user_id: str,
    project_id: str,
    related_terms: Optional[List[str]] = None,
    visibility: str = "org",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Create a taxonomic memory entry (domain-specific term with definition).

    Taxonomic memories are used for:
    - Domain terminology definitions
    - Query expansion (improving search relevance)
    - Semantic linking via related terms

    Args:
        memory_engine: The MemoryEngine instance (or None)
        domain: Domain/category for this term (e.g., "insurance", "coverage_types")
        term: The term being defined (e.g., "deductible")
        definition: Definition of the term
        org_id: Organization ID for multi-tenancy
        user_id: User ID who owns this memory
        related_terms: Related terms for semantic linking
        visibility: Visibility scope (default: "org" for organization-wide)
        metadata: Arbitrary caller-supplied key-value metadata stored on the entry

    Returns:
        Created document ID, or None if creation fails
    """
    if memory_engine is None:
        logger.warning("Memory not enabled, cannot create taxonomic memory")
        return None

    try:
        result = memory_engine.create_taxonomic(
            domain=domain,
            term=term,
            definition=definition,
            org_id=org_id,
            user_id=user_id,
            project_id=project_id,
            related_terms=related_terms or [],
            query_expansion=True,
            visibility=visibility,
            metadata=metadata,
        )
        logger.info(f"Created taxonomic memory: domain={domain}, term={term}")
        record_current_memory_metadata(
            action="write",
            memory_type="taxonomic",
            content=f"{term}: {definition}",
        )
        return result.id
    except Exception as e:
        logger.warning(f"Failed to create taxonomic memory: {e}")
        return None


def search_taxonomic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    query: str,
    org_id: str,
    project_id: str,
    user_id: Optional[str] = None,
    domain: Optional[str] = None,
    visibility: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search taxonomic memories by query.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        query: Search query text
        org_id: Organization ID for multi-tenancy
        domain: Optional domain filter (e.g., "insurance")
        top_k: Maximum number of results

    Returns:
        List of matching taxonomic entries with term, definition, and related_terms
    """
    if memory_engine is None:
        return []

    try:
        chunks = memory_engine.fetch_taxonomic_memories(
            query=query,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            domain=domain,
            visibility=visibility,
            top_k=top_k,
        )

        results = []
        for chunk in chunks:
            results.append(
                {
                    "term": getattr(
                        chunk, "term", chunk.content[:50] if hasattr(chunk, "content") else ""
                    ),
                    "definition": getattr(chunk, "definition", getattr(chunk, "content", "")),
                    "domain": getattr(chunk, "domain", domain or "unknown"),
                    "related_terms": getattr(chunk, "related_terms", []),
                    "similarity_score": getattr(chunk, "similarity_score", 0.0),
                }
            )

        logger.debug(f"Found {len(results)} taxonomic memories for query: {query[:50]}...")
        first = results[0] if results else {}
        term = first.get("term")
        definition = first.get("definition")
        content = f"{term}: {definition}" if term and definition else query
        record_current_memory_metadata(
            action="recall",
            memory_type="taxonomic",
            content=content,
            relevance_score=first.get("similarity_score"),
            query=query,
        )
        return results

    except Exception as e:
        logger.warning(f"Failed to search taxonomic memory: {e}")
        return []


def list_taxonomic_domains(
    memory_engine: Optional[MemoryEngineProtocol],
    org_id: str,
    project_id: str,
    visibility: Optional[str] = None,
) -> List[str]:
    """
    List all distinct taxonomic domains for an organization.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        org_id: Organization ID

    Returns:
        List of domain names (e.g., ["insurance", "coverage_types", "discounts"])
    """
    if memory_engine is None:
        return []

    try:
        domains = memory_engine.get_distinct_domains(
            org_id=org_id, project_id=project_id, visibility=visibility
        )
        record_current_memory_metadata(
            action="recall",
            memory_type="taxonomic",
            content=", ".join(domains) if domains else "taxonomic domains",
        )
        return domains
    except Exception as e:
        logger.warning(f"Failed to list taxonomic domains: {e}")
        return []


def get_taxonomic_term(
    memory_engine: Optional[MemoryEngineProtocol],
    org_id: str,
    project_id: str,
    domain: str,
    term: str,
    user_id: Optional[str] = None,
    visibility: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific taxonomic term by domain and term name.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        org_id: Organization ID
        domain: Domain name
        term: Term name

    Returns:
        Dict with term details, or None if not found
    """
    if memory_engine is None:
        return None

    try:
        result = memory_engine.get_taxonomic(
            org_id=org_id,
            project_id=project_id,
            domain=domain,
            term=term,
            user_id=user_id,
            visibility=visibility,
        )
        if result:
            term_val = result.get("term", term)
            definition_val = result.get("definition", "")
            record_current_memory_metadata(
                action="recall",
                memory_type="taxonomic",
                content=f"{term_val}: {definition_val}",
            )
            return {
                "id": str(result.get("id") or result.get("_id", "")),
                "domain": result.get("domain", domain),
                "term": term_val,
                "definition": definition_val,
                "related_terms": result.get("related_terms", []),
            }
        record_current_memory_metadata(action="recall", memory_type="taxonomic", content=term)
        return None
    except Exception as e:
        logger.warning(f"Failed to get taxonomic term: {e}")
        return None


# =============================================================================
# Episodic Memory Operations
# =============================================================================


def create_episodic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    title: str,
    content: str,
    org_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    summary_text: Optional[str] = None,
    participants: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    visibility: str = "private",
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Create an episodic memory entry (conversation summary or event).

    Episodic memories are used for:
    - Storing conversation summaries
    - Recording significant events/interactions
    - Enabling context retrieval across sessions

    Args:
        memory_engine: The MemoryEngine instance (or None)
        title: Title/label for this episode (e.g., "Customer inquiry about coverage")
        content: Full content of the episode
        org_id: Organization ID for multi-tenancy
        user_id: User ID who owns this memory
        session_id: Session ID this episode originated from
        summary_text: Brief summary of the episode (for search/display)
        participants: List of participants (e.g., ["Customer", "Agent"])
        tags: Tags for categorization (e.g., ["quote", "auto_insurance"])
        visibility: Visibility scope (default: "private")
        metadata: Optional caller-supplied metadata dictionary stored with the
            episode (mirrors create_semantic_memory's metadata)

    Returns:
        Created document ID, or None if creation fails
    """
    if memory_engine is None:
        logger.warning("Memory not enabled, cannot create episodic memory")
        return None

    if not session_id:
        logger.warning("session_id required for episodic memory; skipping episodic write")
        return None

    logger.info(
        f"Creating episodic memory: title='{title[:50]}...', org_id={org_id}, "
        f"user_id={user_id}, visibility={visibility}, tags={tags}"
    )

    try:
        import uuid

        result = memory_engine.create_episodic(
            title=title,
            content=content,
            summary_text=summary_text or title,
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            snapshot_ref_id=f"snapshot_{uuid.uuid4().hex[:8]}",
            summary_type="llm",
            source_agent="agent",
            participants=participants or [],
            tags=tags or [],
            visibility=visibility,
            project_id=project_id,
            agent_id=agent_id,
            metadata=metadata,
        )
        logger.info(f"Created episodic memory: title={title[:50]}..., user_id={user_id}")
        record_current_memory_metadata(
            action="write",
            memory_type="episodic",
            content=summary_text or title or content,
        )
        return result.id
    except Exception as e:
        logger.warning(f"Failed to create episodic memory: {e}")
        return None


def search_episodic_memory(
    memory_engine: Optional[MemoryEngineProtocol],
    query: str,
    org_id: str,
    project_id: str,
    user_id: Optional[str] = None,
    visibility: Optional[str] = None,
    session_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search episodic memories by query.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        query: Search query text
        org_id: Organization ID for multi-tenancy
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        top_k: Maximum number of results

    Returns:
        List of matching episodic entries with title, content, and summary
    """
    if memory_engine is None:
        return []

    try:
        chunks = memory_engine.fetch_episodic_memories(
            query=query,
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            visibility=visibility,
            session_id=session_id,
            top_k=top_k,
        )

        results = []
        for chunk in chunks:
            results.append(
                {
                    "title": getattr(chunk, "title", ""),
                    "content": getattr(chunk, "content", ""),
                    "summary": getattr(chunk, "summary_text", ""),
                    "session_id": getattr(chunk, "session_id", ""),
                    "participants": getattr(chunk, "participants", []),
                    "tags": getattr(chunk, "tags", []),
                    "similarity_score": getattr(chunk, "similarity_score", 0.0),
                }
            )

        logger.debug(f"Found {len(results)} episodic memories for query: {query[:50]}...")
        first = results[0] if results else {}
        record_current_memory_metadata(
            action="recall",
            memory_type="episodic",
            content=str(first.get("summary") or first.get("content") or query),
            relevance_score=first.get("similarity_score"),
            query=query,
        )
        return results

    except Exception as e:
        logger.warning(f"Failed to search episodic memory: {e}")
        return []


def list_episodic_memories(
    memory_engine: Optional[MemoryEngineProtocol],
    org_id: str,
    project_id: str,
    user_id: Optional[str] = None,
    visibility: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    List episodic memories with optional filters.

    Args:
        memory_engine: The MemoryEngine instance (or None)
        org_id: Organization ID
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        limit: Maximum number of results

    Returns:
        List of episodic memory entries
    """
    if memory_engine is None:
        return []

    try:
        episodes = memory_engine.list_episodic(
            org_id=org_id,
            project_id=project_id,
            user_id=user_id,
            visibility=visibility,
            session_id=session_id,
            limit=limit,
        )

        results = []
        for ep in episodes:
            results.append(
                {
                    "id": str(ep.get("id") or ep.get("_id", "")),
                    "title": ep.get("title", ""),
                    "summary": ep.get("summary_text") or ep.get("summary", ""),
                    "session_id": ep.get("session_id"),
                    "tags": ep.get("tags", []),
                }
            )

        first = results[0] if results else {}
        record_current_memory_metadata(
            action="recall",
            memory_type="episodic",
            content=str(first.get("summary") or first.get("title") or "episodic memories"),
        )
        return results

    except Exception as e:
        logger.warning(f"Failed to list episodic memories: {e}")
        return []


# =============================================================================
# Memory Client Factory
# =============================================================================


def create_memory_client(
    server_url: str,
    timeout: float = 30.0,
) -> Optional[MemoryEngineProtocol]:
    """
    Create MemoryClient instance that connects to the memory HTTP server.

    This is the recommended approach - uses the HTTP client to connect to
    the mongomem HTTP server instead of using the engine directly.

    Environment Variables:
        MEMORY_SERVER_URL: URL of the memory HTTP server. ECP wires this
            based on the project's deployment topology, not on whether
            memory-server runs in fctr (it never does — memory-server is
            always a plain container). When the per-project topology is on
            (executor_type_vm FF on) the URL points at the per-project
            Service memory-server-<projectID>, reconciled by
            TenantEnvironment; otherwise it points at the per-workspace
            Service <workspaceID>-memory-server, reconciled by AgentStack.
            For local dev, "http://localhost:8081" is typical.
        APP_ID: Agent ID injected by ECP at pod deploy time. Sent as
            X-Agent-Engine-Agent-Id (and the legacy X-Agentic-Agent-Id during
            the rename transition) on every request so the memory server can
            identify the calling agent.

    Args:
        server_url: Base URL of the memory server
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        MemoryClient instance (compatible with MemoryEngineProtocol), or None if creation fails
    """
    import os

    # Header names defined locally — importing private symbols from the memory
    # package would cause ImportError to incorrectly disable memory on a rename.
    _H_AGENT_ID = "X-Agent-Engine-Agent-Id"
    _H_AGENT_ID_LEGACY = "X-Agentic-Agent-Id"

    try:
        from agent_engine_sdk_memory._client import MemoryClient
    except ImportError:
        logger.warning("agent_engine_sdk_memory not installed - memory client creation skipped")
        return None

    try:
        static_headers: Dict[str, str] = {}
        if agent_id := os.environ.get("APP_ID", "").strip():
            static_headers[_H_AGENT_ID] = agent_id
            static_headers[_H_AGENT_ID_LEGACY] = agent_id

        # Use the platform client for both HTTP and HTTPS so trace context is
        # propagated without recording the proxy request as another operation.
        # HTTPS additionally configures the mTLS credentials required by OE.
        from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls

        http_client = create_httpx_client_with_tls(server_url, timeout)

        client = MemoryClient(
            base_url=server_url,
            timeout=timeout,
            static_headers=static_headers,
            execution_id_provider=get_current_execution_id,
            http_client=http_client,
        )
        logger.info(f"MemoryClient initialized (server: {server_url})")
        return client  # type: ignore[return-value]  # MemoryClient implements MemoryEngineProtocol

    except Exception as e:
        logger.warning(f"MemoryClient initialization failed: {e}")
        return None
