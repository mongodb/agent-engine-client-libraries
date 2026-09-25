"""Durable activity synchronization with short-term Memory."""

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, TypedDict

from agent_engine_sdk.models import LLMResponse

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ActivityContext,
    ActivityMemoryCommand,
    MemoryWrite,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.memory import (
    AssistantMemoryTurn,
    MemoryToolCall,
    MemoryTurn,
    ToolMemoryTurn,
    UserMemoryTurn,
)
from agent_engine_runner_shared.secure_wrapper import CALL_INTERRUPTED_ARTIFACT_KEY
from agent_engine_runner_shared.utils import normalize_content

logger = logging.getLogger(__name__)


class _TurnIdentity(TypedDict):
    session_id: str
    org_id: str
    user_id: str
    project_id: str
    agent_id: str
    idempotency_key: str


class _WorkflowMemoryClient(Protocol):
    def ensure_memory_written(self, command: ActivityMemoryCommand) -> None: ...


def validate_durable_memory_identity(
    identity: WorkflowIdentity,
    user_id: Optional[str],
) -> None:
    """Reject incomplete durable Memory attribution before tenant work starts."""
    scope = identity.tenant_scope
    if not user_id or not all(
        (
            identity.execution_id,
            identity.session_id,
            scope.org_id,
            scope.project_id,
            scope.workspace_id,
        )
    ):
        raise ValueError("durable Memory identity is incomplete")


class DurableMemoryState:
    """Own durable Memory projection, delivery, and pending user input."""

    def __init__(self, pending_user_message: Optional[str] = None) -> None:
        self._pending_user_message = pending_user_message
        self._lock = threading.Lock()

    def synchronize_llm(
        self,
        client: _WorkflowMemoryClient,
        context: ActivityContext,
        result: Any,
        *,
        user_id: Optional[str],
    ) -> None:
        """Synchronize one live or replayed LLM outcome."""
        projection = DurableMemoryProjection.from_activity(context, user_id)
        self._synchronize(client, context, projection, self._llm_writes(projection, result))

    def synchronize_tool(
        self,
        client: _WorkflowMemoryClient,
        context: ActivityContext,
        result: Any,
        *,
        user_id: Optional[str],
        tool_call_id: Optional[str],
        tool_name: str,
    ) -> None:
        """Synchronize one live or replayed tool outcome."""
        projection = DurableMemoryProjection.from_activity(context, user_id)
        writes: List[MemoryWrite] = []
        if result != {CALL_INTERRUPTED_ARTIFACT_KEY: True}:
            content = normalize_content(result)
            if tool_call_id and content:
                writes.append(projection.tool(content, tool_call_id, tool_name))
            elif tool_call_id:
                logger.warning(
                    "Skipping incompatible durable Memory tool result",
                    extra={"activity_id": context.activity_id},
                )
        self._synchronize(client, context, projection, writes)

    @staticmethod
    def _llm_writes(
        projection: "DurableMemoryProjection",
        result: Any,
    ) -> List[MemoryWrite]:
        try:
            response = LLMResponse.model_validate(result)
            tool_calls = None
            if response.tool_calls:
                tool_calls = []
                for tool_call in response.tool_calls:
                    arguments = tool_call.args if tool_call.args is not None else {}
                    if not isinstance(arguments, (dict, str)):
                        raise ValueError("tool-call arguments must be an object or string")
                    tool_calls.append(
                        MemoryToolCall(
                            id=tool_call.id or "",
                            name=tool_call.name or "",
                            arguments=arguments,
                        )
                    )
            if not response.content and not tool_calls:
                raise ValueError("assistant result has no content or tool calls")
            return [projection.assistant(response.content, tool_calls)]
        except (TypeError, ValueError):
            logger.warning(
                "Skipping incompatible durable Memory LLM result",
                extra={"activity_id": projection.activity_id},
            )
            return []

    def _synchronize(
        self,
        client: _WorkflowMemoryClient,
        context: ActivityContext,
        projection: "DurableMemoryProjection",
        result_writes: List[MemoryWrite],
    ) -> None:
        with self._lock:
            pending_user_message = self._pending_user_message
        writes = result_writes
        if pending_user_message is not None:
            writes = [projection.user_input(pending_user_message), *writes]
        client.ensure_memory_written(
            ActivityMemoryCommand(
                workflow_identity=context.workflow_identity,
                activity_id=context.activity_id,
                attempt_id=context.attempt_id,
                fencing_token=context.fencing_token,
                memory_writes=writes,
            )
        )
        with self._lock:
            self._pending_user_message = None


@dataclass(frozen=True)
class DurableMemoryProjection:
    """Build exact STM writes for one durable activity."""

    execution_id: str
    activity_id: str
    session_id: str
    org_id: str
    project_id: str
    workspace_id: str
    user_id: str

    @classmethod
    def from_activity(
        cls,
        context: ActivityContext,
        user_id: Optional[str],
    ) -> "DurableMemoryProjection":
        identity = context.workflow_identity
        scope = identity.tenant_scope
        validate_durable_memory_identity(identity, user_id)
        if user_id is None:
            raise ValueError("durable Memory identity is incomplete")
        if not context.activity_id:
            raise ValueError("durable Memory identity is incomplete")
        return cls(
            execution_id=identity.execution_id,
            activity_id=context.activity_id,
            session_id=identity.session_id,
            org_id=scope.org_id,
            project_id=scope.project_id,
            workspace_id=scope.workspace_id,
            user_id=user_id,
        )

    def user_input(self, content: str) -> MemoryWrite:
        write_id = f"workflow:{self.execution_id}:input"
        return self._write(
            write_id,
            UserMemoryTurn(content=content, **self._turn_identity(write_id)),
        )

    def assistant(
        self,
        content: str,
        tool_calls: Optional[List[MemoryToolCall]],
    ) -> MemoryWrite:
        write_id = f"workflow:{self.execution_id}:{self.activity_id}:assistant:0"
        return self._write(
            write_id,
            AssistantMemoryTurn(
                content=content,
                tool_calls=tool_calls,
                **self._turn_identity(write_id),
            ),
        )

    def tool(
        self,
        content: str,
        tool_call_id: str,
        tool_name: str,
    ) -> MemoryWrite:
        write_id = f"workflow:{self.execution_id}:{self.activity_id}:tool:0"
        return self._write(
            write_id,
            ToolMemoryTurn(
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                **self._turn_identity(write_id),
            ),
        )

    def _turn_identity(self, write_id: str) -> _TurnIdentity:
        return {
            "session_id": self.session_id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "agent_id": self.workspace_id,
            "idempotency_key": write_id,
        }

    @staticmethod
    def _write(write_id: str, turn: MemoryTurn) -> MemoryWrite:
        return MemoryWrite(id=write_id, payload_json=turn.payload_json())
