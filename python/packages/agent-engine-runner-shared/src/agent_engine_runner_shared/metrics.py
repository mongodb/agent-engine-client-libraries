"""
Metrics and observability for the Runner SDK.

Provides metrics collection and structured logging for observability.

Usage:
    from agent_engine_runner_shared.metrics import Metrics, record_latency, record_error

    # Record tool latency
    with record_latency("tool_execution", tool_name="lookup_policy"):
        result = tool.invoke(args)

    # Record error
    record_error("tool_execution", tool_name="lookup_policy", error="Connection timeout")

    # Get all metrics
    metrics = Metrics.get_all()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics for a metric."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, duration_ms: float) -> None:
        """Record a latency measurement."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    @property
    def avg_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0.0,
            "max_ms": round(self.max_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
        }


class Metrics:
    """
    Thread-safe metrics collector for the Runner SDK.

    Collects:
    - Tool execution latency (per tool)
    - LLM call latency
    - Error counts (per operation type)
    - Request counts
    """

    _lock = threading.Lock()
    _latencies: Dict[str, LatencyStats] = defaultdict(LatencyStats)
    _errors: Dict[str, int] = defaultdict(int)
    _requests: Dict[str, int] = defaultdict(int)

    @classmethod
    def record_latency(
        cls,
        operation: str,
        duration_ms: float,
        **labels: str,
    ) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Operation name (e.g., "tool_execution", "llm_call")
            duration_ms: Duration in milliseconds
            **labels: Additional labels (e.g., tool_name="lookup_policy")
        """
        # Create a key including labels for per-operation-per-label tracking
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        key = f"{operation}:{label_str}" if label_str else operation

        with cls._lock:
            cls._latencies[key].record(duration_ms)
            cls._requests[operation] = cls._requests.get(operation, 0) + 1

        # Structured log
        logger.debug(
            "METRIC latency",
            extra={
                "metric_type": "latency",
                "operation": operation,
                "duration_ms": round(duration_ms, 2),
                **labels,
            },
        )

    @classmethod
    def record_error(cls, operation: str, **labels: str) -> None:
        """
        Record an error occurrence.

        Args:
            operation: Operation name
            **labels: Additional labels (e.g., error_type="timeout")
        """
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        key = f"{operation}:{label_str}" if label_str else operation

        with cls._lock:
            cls._errors[key] = cls._errors.get(key, 0) + 1

        # Structured log
        logger.warning(
            "METRIC error",
            extra={
                "metric_type": "error",
                "operation": operation,
                **labels,
            },
        )

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all collected metrics."""
        with cls._lock:
            return {
                "latencies": {k: v.to_dict() for k, v in cls._latencies.items()},
                "errors": dict(cls._errors),
                "requests": dict(cls._requests),
            }

    @classmethod
    def reset(cls) -> None:
        """Reset all metrics (useful for testing)."""
        with cls._lock:
            cls._latencies.clear()
            cls._errors.clear()
            cls._requests.clear()


@contextmanager
def record_latency(
    operation: str,
    **labels: str,
) -> Generator[None, None, None]:
    """
    Context manager to record latency of an operation.

    Usage:
        with record_latency("tool_execution", tool_name="lookup_policy"):
            result = tool.invoke(args)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        Metrics.record_latency(operation, duration_ms, **labels)


def record_error(operation: str, **labels: str) -> None:
    """Record an error occurrence."""
    Metrics.record_error(operation, **labels)


def with_metrics(operation: str, record_errors: bool = True):
    """
    Decorator to record latency and errors for a function.

    Works with both sync and async functions.

    Usage:
        @with_metrics("oe_agent_start")
        async def _handle_start(self, request):
            ...

        @with_metrics("tool_execution", record_errors=False)
        def execute_tool(self, ...):
            ...
    """
    import asyncio
    import functools
    from typing import Callable, TypeVar

    F = TypeVar("F", bound=Callable[..., Any])

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                if record_errors:
                    Metrics.record_error(operation)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                Metrics.record_latency(operation, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                if record_errors:
                    Metrics.record_error(operation)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                Metrics.record_latency(operation, duration_ms)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Structured Logging Helpers
# =============================================================================


def log_tool_call(
    execution_id: str,
    step_number: int,
    tool_name: str,
    arguments: Dict[str, Any],
    is_local: bool,
) -> None:
    """Log a tool call with structured data."""
    logger.info(
        f"TOOL_CALL step={step_number} tool={tool_name}",
        extra={
            "event_type": "tool_call",
            "execution_id": execution_id,
            "step_number": step_number,
            "tool_name": tool_name,
            "is_local": is_local,
            "argument_keys": list(arguments.keys()),
        },
    )


def log_tool_result(
    execution_id: str,
    step_number: int,
    tool_name: str,
    status: str,
    duration_ms: float,
    error: Optional[str] = None,
) -> None:
    """Log a tool result with structured data."""
    log_level = logging.INFO if status in ("success", "suspend", "cached") else logging.ERROR
    logger.log(
        log_level,
        f"TOOL_RESULT step={step_number} tool={tool_name} status={status}",
        extra={
            "event_type": "tool_result",
            "execution_id": execution_id,
            "step_number": step_number,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "error": error,
        },
    )

    # Record metrics
    Metrics.record_latency("tool_execution", duration_ms, tool_name=tool_name)
    if status == "error":
        Metrics.record_error("tool_execution", tool_name=tool_name)


def log_llm_call(
    execution_id: str,
    step_number: int,
    model_name: str,
    message_count: int,
) -> None:
    """Log an LLM call with structured data."""
    logger.info(
        f"LLM_CALL step={step_number} model={model_name}",
        extra={
            "event_type": "llm_call",
            "execution_id": execution_id,
            "step_number": step_number,
            "model_name": model_name,
            "message_count": message_count,
        },
    )


def log_llm_result(
    execution_id: str,
    step_number: int,
    model_name: str,
    status: str,
    duration_ms: float,
    tool_calls: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """Log an LLM result with structured data."""
    log_level = logging.INFO if status == "success" else logging.ERROR
    logger.log(
        log_level,
        f"LLM_RESULT step={step_number} model={model_name} status={status}",
        extra={
            "event_type": "llm_result",
            "execution_id": execution_id,
            "step_number": step_number,
            "model_name": model_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "tool_calls": tool_calls,
            "error": error,
        },
    )

    # Record metrics
    Metrics.record_latency("llm_call", duration_ms, model_name=model_name)
    if status == "error":
        Metrics.record_error("llm_call", model_name=model_name)


def log_execution_event(
    execution_id: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an execution lifecycle event."""
    logger.info(
        f"EXECUTION_{event} execution_id={execution_id[:8]}...",
        extra={
            "event_type": f"execution_{event.lower()}",
            "execution_id": execution_id,
            **(details or {}),
        },
    )
