"""
Node execution logger — framework-neutral.

Logs graph node traversals to OE for observability.
The OE then writes to MongoDB, keeping database writes centralized.

Implements the BaseExecutionCallback protocol from agent-engine-sdk.
Framework adapters (e.g. LangGraphCallbackAdapter in agent-engine-sdk-langgraph)
bridge from framework-specific callback systems to this class.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent_engine_sdk.interfaces import BaseExecutionCallback

from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls

logger = logging.getLogger(__name__)


class NodeExecutionLogger(BaseExecutionCallback):
    """
    Callback handler that logs node executions to OE.

    Implements the BaseExecutionCallback protocol.
    Node execution events are sent to OE's /node/execution endpoint,
    which handles the database writes.
    """

    def __init__(
        self,
        oe_url: str,
        execution_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self._oe_url = oe_url
        self._execution_id = execution_id
        self._session_id = session_id
        self._user_id = user_id
        self._org_id = org_id
        self._project_id = project_id
        self._node_start_times: Dict[str, datetime] = {}

    def on_node_start(
        self,
        node_name: str,
        inputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        start_time = datetime.now(timezone.utc)
        self._node_start_times[run_id] = start_time
        self._send_node_event(
            node_name=node_name,
            status="started",
            timestamp=start_time,
            run_id=run_id,
            parent_run_id=parent_run_id,
            inputs=inputs,
        )

    def on_node_end(
        self,
        node_name: str,
        outputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        start_time = self._node_start_times.pop(run_id, None)
        end_time = datetime.now(timezone.utc)
        if duration_ms is None and start_time is not None:
            duration_ms = (end_time - start_time).total_seconds() * 1000
        self._send_node_event(
            node_name=node_name,
            status="success",
            timestamp=end_time,
            run_id=run_id,
            parent_run_id=parent_run_id,
            outputs=outputs,
            duration_ms=duration_ms,
        )

    def on_node_error(
        self,
        node_name: str,
        error: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._node_start_times.pop(run_id, None)
        self._send_node_event(
            node_name=node_name,
            status="error",
            timestamp=datetime.now(timezone.utc),
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=error,
        )

    def on_node_suspend(
        self,
        node_name: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        start_time = self._node_start_times.pop(run_id, None)
        end_time = datetime.now(timezone.utc)
        duration_ms = (
            (end_time - start_time).total_seconds() * 1000 if start_time is not None else None
        )
        self._send_node_event(
            node_name=node_name,
            status="suspend",
            timestamp=end_time,
            run_id=run_id,
            parent_run_id=parent_run_id,
            duration_ms=duration_ms,
        )

    def _serialize_for_json(self, obj: Any) -> Any:
        """Serialize an object for JSON transmission."""
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        if isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)

    def _send_node_event(
        self,
        node_name: str,
        status: str,
        timestamp: datetime,
        run_id: str,
        parent_run_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Send a node execution event to OE (fire-and-forget)."""
        payload: Dict[str, Any] = {
            "execution_id": self._execution_id,
            "node_name": node_name,
            "status": status,
            "timestamp": timestamp.isoformat(),
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "session_id": self._session_id,
            "user_id": self._user_id,
            "org_id": self._org_id,
            "project_id": self._project_id,
        }
        if inputs is not None:
            payload["inputs"] = self._serialize_for_json(inputs)
        if outputs is not None:
            payload["outputs"] = self._serialize_for_json(outputs)
        if error is not None:
            payload["error"] = error
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms

        try:
            with create_httpx_client_with_tls(self._oe_url, 2.0) as client:
                client.post(f"{self._oe_url}/node/execution", json=payload)
        except Exception as e:
            # Best-effort: swallow ALL errors including ValueError from missing TLS
            # certs (fail-closed policy). A missing cert must not abort node execution.
            logger.debug(f"Failed to send node event to OE: {e}")
