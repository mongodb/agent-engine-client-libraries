"""
Execution logging for Runner SDK.

Provides durable logging of tool/LLM executions to:
- JSONL files (always enabled)
- MongoDB collection (when configured)

This mirrors the agent-runtime's tool/logger.py for feature parity.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_engine_runner_shared.tool_api_error import ToolAPIError

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection

    from agent_engine_runner_shared.models import (
        Execution,
        NodeExecutionRequest,
        ToolExecuteRequest,
        ToolResultRequest,
    )

logger = logging.getLogger(__name__)

_file_lock = threading.Lock()
_tool_logs_collection: Optional["Collection"] = None  # execution_logs collection
_node_logs_collection: Optional["Collection"] = None  # node_executions collection
_executions_collection: Optional["Collection"] = None  # For execution state persistence


# =============================================================================
# Types
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of a tool execution."""

    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"
    CACHED = "cached"
    SUSPENDED = "suspend"


class ExecutionLog(BaseModel):
    """
    Log entry for a tool execution.

    Each tool call produces log entries for:
    - status="started" with inputs
    - status="success" or "error" with output/error and duration
    """

    id: str
    execution_id: str
    tool: str
    # Stable tool-call id from the LLM (LangChain tool_calls[].id / tool_call_id,
    # ADK FunctionCall.id). Joins a tool start to its result and to the session
    # message. None for invoke_llm and other non-tool-call events.
    tool_call_id: Optional[str] = None
    kind: Optional[str] = None
    status: ExecutionStatus
    timestamp: datetime

    # Request data
    inputs: Optional[Dict[str, Any]] = None

    # Response data
    output: Optional[Any] = None
    error: Optional[str] = None
    tool_api_error: Optional[ToolAPIError] = None
    duration_ms: Optional[float] = None

    # Context (session/user info)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    project_id: Optional[str] = None

    # Step tracking
    step_number: Optional[int] = None

    # Execution environment
    pod_name: Optional[str] = None  # Hostname/pod name where tool was executed

    # Tracing context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Token usage (populated for invoke_llm calls)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    workspace_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeExecutionStatus(str, Enum):
    """Status of a node execution.

    Deliberately omits BLOCKED and CACHED: those statuses apply to tool
    executions only and have no node-level equivalent.
    """

    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"
    SUSPENDED = "suspend"


class NodeExecutionLog(BaseModel):
    """
    Log entry for a LangGraph node execution.

    Each node traversal produces log entries for:
    - status="started" with inputs
    - status="success" with outputs and duration
    - status="error" with error message
    """

    id: str
    execution_id: str
    node: str
    status: NodeExecutionStatus
    timestamp: datetime

    # LangChain run tracking
    run_id: str
    parent_run_id: Optional[str] = None

    # Request/Response data
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None

    # Context (session/user info)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    project_id: Optional[str] = None

    # Tracing context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


# =============================================================================
# Path Resolution
# =============================================================================


def _resolve_log_path() -> Path:
    """Resolve the JSONL log file path."""
    file_env = os.getenv("AGENTIC_EXECUTIONS_FILE")
    base_dir_env = os.getenv("AGENTIC_OBSERVABILITY_DIR")

    if file_env:
        path = Path(file_env)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    base_dir = Path(base_dir_env) if base_dir_env else Path("observability")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "executions.jsonl"


def get_log_path() -> Path:
    """Get the current log file path."""
    return _resolve_log_path()


# =============================================================================
# MongoDB Configuration
# =============================================================================


def set_tool_logs_collection(collection: Optional["Collection"]) -> None:
    """
    Set the MongoDB collection for tool execution logging.

    When set, tool logs will be written to both JSONL and MongoDB.
    """
    global _tool_logs_collection
    _tool_logs_collection = collection

    if collection is not None:
        logger.info(f"MongoDB tool logging enabled: {collection.database.name}.{collection.name}")


def get_tool_logs_collection() -> Optional["Collection"]:
    """Get the current MongoDB collection for tool logging."""
    return _tool_logs_collection


def set_node_logs_collection(collection: Optional["Collection"]) -> None:
    """
    Set the MongoDB collection for node execution logging.

    When set, node logs will be written to both JSONL and MongoDB.
    """
    global _node_logs_collection
    _node_logs_collection = collection

    if collection is not None:
        logger.info(f"MongoDB node logging enabled: {collection.database.name}.{collection.name}")


def get_node_logs_collection() -> Optional["Collection"]:
    """Get the current MongoDB collection for node logging."""
    return _node_logs_collection


def set_executions_collection(collection: Optional["Collection"]) -> None:
    """
    Set the MongoDB collection for execution state persistence.

    When set, execution state will be persisted to MongoDB for
    human-in-the-loop review queue and cross-session tracking.
    """
    global _executions_collection
    _executions_collection = collection

    if collection is not None:
        logger.info(f"Executions persistence enabled: {collection.database.name}.{collection.name}")


def get_executions_collection() -> Optional["Collection"]:
    """Get the current MongoDB collection for execution state."""
    return _executions_collection


# =============================================================================
# Logging Functions
# =============================================================================


def log_entry(log: ExecutionLog | NodeExecutionLog) -> None:
    """
    Log an execution event (tool or node).

    Writes to:
    1. JSONL file (always)
    2. MongoDB collection (if configured) - uses appropriate collection based on log type
    """
    # JSONL needs JSON-serializable types (datetimes as strings)
    json_dict = log.model_dump(mode="json")
    line = json.dumps(json_dict, separators=(",", ":"))
    path = get_log_path()

    with _file_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Write to MongoDB if configured - use appropriate collection based on log type
    if isinstance(log, NodeExecutionLog):
        collection = _node_logs_collection
    else:
        collection = _tool_logs_collection

    if collection is not None:
        try:
            # Use JSON mode so Any-typed fields (inputs, output, metadata) are
            # coerced to BSON-safe primitives, then restore datetime fields that
            # MongoDB needs as native objects for $gte/$lte queries.
            mongo_dict = log.model_dump(mode="json")
            mongo_dict["timestamp"] = log.timestamp
            collection.insert_one(mongo_dict)
        except Exception as e:
            logger.error(f"Failed to write log to MongoDB: {e}", exc_info=True)


def redact_fields(data: Dict[str, Any], fields_to_redact: List[str]) -> Dict[str, Any]:
    """
    Redact sensitive fields from a dictionary.

    Args:
        data: The dictionary to redact
        fields_to_redact: List of field names to redact

    Returns:
        A copy of the dictionary with redacted fields
    """
    if not fields_to_redact:
        return data

    result = dict(data)
    for field in fields_to_redact:
        if field in result:
            result[field] = "[REDACTED]"
    return result


# =============================================================================
# Execution Logger Class
# =============================================================================


class ExecutionLogger:
    """
    High-level execution logger for the OE.

    Provides convenient methods for logging tool/LLM executions
    and node executions with automatic step tracking and context injection.

    MongoDB is required - the logger will fail fast if connection fails.
    """

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str,
        tool_logs_collection_name: str,
        node_logs_collection_name: str,
    ):
        """
        Initialize the execution logger and connect to MongoDB.

        Args:
            mongodb_uri: MongoDB connection string (required)
            database_name: Database name for MongoDB logging (required)
            tool_logs_collection_name: Collection name for tool execution logging (required)
            node_logs_collection_name: Collection name for node execution logging (required)

        Raises:
            RuntimeError: If MongoDB connection fails.
                This ensures the container fails fast and Kubernetes restarts it.
        """
        from pymongo import MongoClient

        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test connection - raises exception if MongoDB is not reachable
        client.admin.command("ping")

        # Store client reference so other code (e.g. session query endpoints)
        # can access additional databases via get_mongo_client().
        self._client: "MongoClient" = client

        # Initialize tool execution logs collection
        tool_collection = client[database_name][tool_logs_collection_name]
        set_tool_logs_collection(tool_collection)
        logger.info(f"Tool logging initialized: {database_name}.{tool_logs_collection_name}")

        # Initialize node execution logs collection
        node_collection = client[database_name][node_logs_collection_name]
        set_node_logs_collection(node_collection)
        logger.info(f"Node logging initialized: {database_name}.{node_logs_collection_name}")

        # Initialize executions collection (for human-in-the-loop state)
        executions_collection = client[database_name]["executions"]
        set_executions_collection(executions_collection)
        logger.info(f"Executions persistence initialized: {database_name}.executions")

    def get_client(self) -> "MongoClient":
        """Return the underlying MongoClient instance.

        Used by session query endpoints to access the checkpoint database.
        """
        return self._client

    def log_tool_start(
        self,
        request: "ToolExecuteRequest",
        execution: "Execution",
    ) -> str:
        """
        Log the start of a tool execution.

        Args:
            request: The tool execute request from AER
            execution: The parent execution context

        Returns:
            The log ID used
        """
        log = request.to_start_log(execution)
        log_entry(log)
        return log.id

    def log_tool_result(
        self,
        request: "ToolResultRequest",
        execution: Optional["Execution"] = None,
    ) -> str:
        """
        Log the result of a tool execution.

        Args:
            request: The tool result request from AER
            execution: The parent execution context (optional)

        Returns:
            The log ID used
        """
        log = request.to_log(execution)
        log_entry(log)
        return log.id

    def log_cached(
        self,
        request: "ToolExecuteRequest",
        execution: "Execution",
        cached_result: Any,
    ) -> str:
        """
        Log a cached result (replay scenario).

        Args:
            request: The tool execute request from AER
            execution: The parent execution context
            cached_result: The cached result being returned

        Returns:
            The log ID used
        """
        log = request.to_cached_log(execution, cached_result)
        log_entry(log)
        return log.id

    def log_node(self, request: "NodeExecutionRequest") -> str:
        """
        Log a node execution event.

        Args:
            request: The node execution request from AER

        Returns:
            The log ID used
        """
        log = request.to_log()
        log_entry(log)
        return log.id

    def persist_execution(self, execution: "Execution") -> None:
        """
        Persist execution state to MongoDB for human-in-the-loop review queue.

        This enables admins to see pending reviews across all users.
        Only called when execution becomes SUSPENDED (needs human review).
        """
        collection = get_executions_collection()
        if collection is None:
            logger.debug("Executions collection not configured, skipping persistence")
            return

        now = datetime.now(timezone.utc)
        doc = execution.to_persistence_doc(updated_at=now)

        try:
            # Upsert: insert if new, update if exists
            # created_at is only set on insert, not on update
            collection.update_one(
                {"execution_id": execution.id},
                {"$set": doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            logger.debug(f"Persisted execution {execution.id} with status {execution.status.value}")
        except Exception as e:
            logger.error(f"Failed to persist execution {execution.id}: {e}", exc_info=True)

    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        *,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        suspend_reason: Optional[str] = None,
        suspend_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update execution status in MongoDB.

        Called when execution transitions between states:
        pending -> running -> completed/suspended/error
        """
        collection = get_executions_collection()
        if collection is None:
            raise RuntimeError("Executions collection not initialized")

        update_doc: Dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }

        # Only set fields that are provided
        if result is not None:
            update_doc["result"] = result
        if error is not None:
            update_doc["error"] = error
        if suspend_reason is not None:
            update_doc["suspend_reason"] = suspend_reason
        if suspend_context is not None:
            update_doc["suspend_context"] = suspend_context

        # Clear suspend fields on completion
        if status in ("completed", "error"):
            update_doc["suspend_reason"] = None
            update_doc["suspend_context"] = None

        try:
            collection.update_one(
                {"execution_id": execution_id},
                {"$set": update_doc},
            )
            logger.debug(f"Updated execution {execution_id} to status {status}")
        except Exception as e:
            logger.error(f"Failed to update execution {execution_id}: {e}", exc_info=True)


# =============================================================================
# Global Logger Instance
# =============================================================================

_execution_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """Get the global execution logger. Must be initialized first via init_execution_logging()."""
    if _execution_logger is None:
        raise RuntimeError(
            "ExecutionLogger not initialized. Call init_execution_logging() at startup."
        )
    return _execution_logger


def get_mongo_client() -> Optional["MongoClient"]:
    """Get the MongoClient used by the execution logger.

    Returns None if the execution logger has not been initialized.
    Used by session query endpoints to access the checkpoint database.
    """
    if _execution_logger is None:
        return None
    return _execution_logger.get_client()


def init_execution_logging(
    mongodb_uri: str,
    database_name: str,
    tool_logs_collection_name: str,
    node_logs_collection_name: str,
) -> ExecutionLogger:
    """
    Initialize execution logging. Call this at application startup.

    Args:
        mongodb_uri: MongoDB connection string (required)
        database_name: Database name (required)
        tool_logs_collection_name: Collection name for tool logs (required)
        node_logs_collection_name: Collection name for node logs (required)

    Returns:
        The configured ExecutionLogger instance
    """
    global _execution_logger

    _execution_logger = ExecutionLogger(
        mongodb_uri=mongodb_uri,
        database_name=database_name,
        tool_logs_collection_name=tool_logs_collection_name,
        node_logs_collection_name=node_logs_collection_name,
    )

    return _execution_logger
