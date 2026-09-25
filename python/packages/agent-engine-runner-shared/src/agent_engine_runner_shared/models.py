"""
Shared Pydantic models for Runner SDK components.

These models define the API contracts between:
- Orchestration Engine (OE)
- Agent Execution Runtime (AER)
- Tool Executor Pod
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_engine_sdk.models import (
    LLMInvocationOptions,
    LLMResponse,
    LLMTokenUsage,
    LLMToolCall,
    LLMToolSchema,
    Message,
    ToolCallChunk,
)
from pydantic import AliasChoices, BaseModel, Field, JsonValue, field_validator, model_validator

from agent_engine_runner_shared.tool_api_error import ToolAPIError

if TYPE_CHECKING:
    from agent_engine_runner_shared.logging import ExecutionLog, NodeExecutionLog

# =============================================================================
# Execution Status
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of an agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


# =============================================================================
# Suspend Payload (Agent → Platform)
# =============================================================================


class SuspendPayload(BaseModel):
    """Payload returned by an agent tool to trigger human-in-the-loop suspension.

    Agent tools signal a suspend by returning a JSON string containing these
    fields.  The ``__suspend__`` flag is stripped before the payload is passed
    to LangGraph's ``interrupt()``.

    Example usage in an agent tool::

        from agent_engine_runner_shared.models import SuspendPayload

        payload = SuspendPayload(
            suspend_reason="awaiting_human_review",
            suspend_context={"claim_id": "C-123", "task_id": "T-456"},
        )
        return payload.to_json()
    """

    suspend_reason: str = Field(
        ..., description="Why the agent is suspending (e.g. 'awaiting_human_review')"
    )
    suspend_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context the human reviewer needs to make a decision",
    )

    def to_json(self) -> str:
        """Serialize to the suspend wire marker, and record an out-of-band
        suspend request on the current execution frame.

        The Tool Pod honors suspend from that recorded signal — set only here,
        in the tool author's own code — not by sniffing tool-result content, so
        untrusted data a tool relays can no longer forge a HITL suspend. The
        marker string is still returned unchanged for wire/replay compatibility.
        """
        from agent_engine_runner_shared.context import record_suspend_request

        data = self.model_dump()
        data["__suspend__"] = True
        record_suspend_request(data)
        return json.dumps(data)


# =============================================================================
# Streaming / Interrupt Results (AER internal)
# =============================================================================


class PendingInterrupt(BaseModel):
    """Framework-native interrupt lifted into the platform envelope."""

    id: str = Field(description="Framework-generated interrupt identifier")
    value: Any = Field(description="Opaque framework-native interrupt payload")


class InterruptResult(BaseModel):
    """Result from agent execution when the agent is suspended.

    Contains the suspend payload directly (framework-agnostic) rather than
    wrapping LangGraph-specific Interrupt objects. Any framework-specific
    state needed to resume (LangGraph checkpoint id, ADK function-call
    correlation, etc.) is carried opaquely in ``metadata`` — the AER never
    inspects it; the framework adapter writes it on suspend and reads it back
    from ``RequestContext.metadata`` on resume.
    """

    suspend_payload: Any = Field(
        description="Legacy first-interrupt payload from the agent's suspend event",
    )
    interrupts: Optional[List[PendingInterrupt]] = Field(
        default=None, description="Complete framework-native pending interrupt snapshot"
    )
    resume_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for the Atlas Agent Engine continue request"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Opaque framework-specific resume state, round-tripped to resume",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description=(
            "Messages produced before the suspend (assistant/tool node outputs), "
            "for writing the pre-suspend portion of the turn to memory"
        ),
    )


class StreamingResult(BaseModel):
    """Result from agent execution for normal completion.

    Returned by _execute_via_agent_stream when the agent finishes without
    suspend. The caller uses content for the final response and messages
    for memory writing.
    """

    content: str = Field(
        default="",
        description="Sanitized final AI response (thinking stripped, trailing empties skipped)",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="All messages from agent execution (sdk-core Message objects), for memory writer",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Opaque framework-owned state produced by the completed execution",
    )


# =============================================================================
# Agent Start (Client → OE)
# =============================================================================


class InvokeRequest(BaseModel):
    """Request to invoke the agent (compatible with agent-runtime)."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for trace correlation")
    # Multi-tenant context
    org_id: Optional[str] = Field(
        default=None, description="Organization ID for multi-tenant isolation"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace identifier for cost tracking"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project ID for project-level scoping"
    )
    # Per-agent AER routing
    aer_url: Optional[str] = Field(
        default=None, description="AER HTTP endpoint for this agent; overrides default AER_URL"
    )
    tool_url: Optional[str] = Field(
        default=None, description="Tool HTTP endpoint for this agent; overrides default TOOL_URL"
    )
    # Sync mode (default: wait for completion)
    wait: bool = Field(default=True, description="If true, wait for completion and return result")
    # Custom headers forwarded from caller (X-Mdb-Agent-Engine-Custom-* HTTP headers, prefix-stripped and lowercased)
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Caller-provided custom headers"
    )


class InvokeResponse(BaseModel):
    """Response from invoking the agent (compatible with agent-runtime)."""

    result: Any = Field(default=None, description="The agent's response")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    user_id: Optional[str] = Field(default=None, description="User ID used")
    # Runner-specific fields (for async mode and tracking)
    execution_id: Optional[str] = Field(default=None, description="Unique execution identifier")
    status: Optional[str] = Field(default=None, description="Execution status")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# =============================================================================
# Execute Request (OE → AER)
# =============================================================================


class ExecuteRequest(BaseModel):
    """Request to execute an agent in AER."""

    platform_trace_id: Optional[str] = Field(
        default=None, description="Platform request trace ID for log correlation"
    )

    execution_id: str = Field(..., description="Unique execution identifier")
    message: str = Field(default="", description="User message")
    platform_api_url: str = Field(..., description="URL of the OE for callbacks")
    suspend_generation: Optional[int] = Field(
        default=None,
        ge=0,
        description="Suspension generation expected by this dispatch, when provided by OE",
    )
    platform_api_owner_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional OE-supplied replica-specific owner callback URL. "
            "Runners use it as a best-effort optimization when present and "
            "otherwise fall back to platform_api_url."
        ),
    )
    resume: bool = Field(default=False, description="Whether this is a resume after SUSPEND")
    resume_from_step: Optional[int] = Field(default=None, description="Step to resume from")
    resume_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Data to inject on resume"
    )
    # Multi-tenant context
    org_id: Optional[str] = Field(
        default=None, description="Organization ID for multi-tenant isolation"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    session_id: Optional[str] = Field(
        default=None, description="Session ID; the AER adapter uses this as the LangGraph thread_id"
    )
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace identifier for cost tracking"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project ID for project-level scoping"
    )
    root_session_id: Optional[str] = Field(
        default=None, description="Root user-facing session ID for cross-agent trace correlation"
    )
    root_execution_id: Optional[str] = Field(
        default=None, description="Root execution ID for cross-agent trace correlation"
    )
    # Custom headers forwarded from caller (X-Mdb-Agent-Engine-Custom-* HTTP headers, prefix-stripped and lowercased)
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Caller-provided custom headers"
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Opaque caller-provided input forwarded unchanged on every dispatch; payload['message'] fills top-level message when empty",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Opaque framework-owned state passed through to the selected adapter",
    )
    previous_execution_cancelled: bool = Field(
        default=False,
        description=(
            "True when this session's most recent prior execution was cancelled "
            "(pod torn down mid-run); the framework adapter fences off that "
            "run's partial checkpoint writes before running this turn"
        ),
    )
    # `message` resolution (promote payload["message"] when the top level is
    # empty; reject when both are set) lives in the AER, the only component that
    # unpacks the opaque payload — see agent_engine_runner_shared.server.aer. Keeping it out
    # of a model validator means the conflict surfaces as a clean 400 instead of
    # a 422 that would echo the payload back to the caller and into logs.


# =============================================================================
# Tool Execution (AER → OE → AER/ToolPod)
# =============================================================================


class ToolExecuteRequest(BaseModel):
    """Request to execute a tool (AER → OE for approval)."""

    execution_id: str = Field(..., description="Execution identifier")
    tool_name: str = Field(..., description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    step_number: int = Field(..., description="Step number in execution sequence")
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Stable tool-call id from the LLM, joining this call to its result and session message",
    )
    kind: Optional[str] = Field(default=None, description="Explicit observability kind")
    is_local: bool = Field(
        default=True,
        description="Whether OE should return the approved call to the current AER stack",
    )
    redact_fields: List[str] = Field(
        default_factory=list,
        description="Top-level tool argument names to redact from execution logs",
    )
    provider_type: Optional[str] = Field(
        default=None, description="Credential provider type for broker token exchange"
    )
    scopes: List[str] = Field(
        default_factory=list, description="Requested OAuth scopes for delegated auth"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Observability metadata")
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Caller-provided headers carried only for this active request",
    )
    trace_id: Optional[str] = Field(
        default=None, description="Active OTel trace ID, for OE execution-log correlation"
    )
    span_id: Optional[str] = Field(
        default=None, description="Active OTel span ID, for OE execution-log correlation"
    )

    def to_start_log(self, execution: "Execution") -> "ExecutionLog":
        """Convert this request to an ExecutionLog for tool start."""
        from uuid import uuid4

        from agent_engine_runner_shared.logging import ExecutionLog, ExecutionStatus, redact_fields

        inputs = redact_fields(self.arguments, self.redact_fields)

        return ExecutionLog(
            id=str(uuid4()),
            execution_id=self.execution_id,
            tool=self.tool_name,
            tool_call_id=self.tool_call_id,
            kind=self.kind,
            status=ExecutionStatus.STARTED,
            timestamp=datetime.now(timezone.utc),
            inputs=inputs,
            step_number=self.step_number,
            session_id=execution.session_id,
            user_id=execution.user_id,
            org_id=execution.org_id,
            project_id=execution.project_id,
            workspace_id=execution.workspace_id,
            metadata=self.metadata,
            trace_id=self.trace_id,
            span_id=self.span_id,
        )

    def to_cached_log(
        self,
        execution: "Execution",
        cached_result: Any,
    ) -> "ExecutionLog":
        """
        Convert this request to an ExecutionLog for cached result.

        Args:
            execution: The parent execution context
            cached_result: The cached result being returned
        """
        from uuid import uuid4

        from agent_engine_runner_shared.logging import ExecutionLog, ExecutionStatus

        return ExecutionLog(
            id=str(uuid4()),
            execution_id=self.execution_id,
            tool=self.tool_name,
            tool_call_id=self.tool_call_id,
            kind=self.kind,
            status=ExecutionStatus.CACHED,
            timestamp=datetime.now(timezone.utc),
            output=cached_result,
            step_number=self.step_number,
            session_id=execution.session_id,
            user_id=execution.user_id,
            org_id=execution.org_id,
            project_id=execution.project_id,
            workspace_id=execution.workspace_id,
            metadata={"replay": True, **self.metadata},
            trace_id=self.trace_id,
            span_id=self.span_id,
        )


class ElicitationInfo(BaseModel):
    """Authorization details returned when broker consent is required.

    Its presence in a ToolExecuteResponse signals that the user must authorize
    via the provided URL before the tool invocation can proceed.
    """

    elicitation_id: str = Field(..., description="Credential broker elicitation identifier")
    authorization_url: str = Field(
        ...,
        description="URL the user must visit to grant authorization",
    )
    message: str = Field(default="", description="Human-readable authorization guidance")
    created: bool = Field(
        default=False,
        description="Whether this response created a new elicitation",
    )


class GuardrailMeta(BaseModel):
    """Identity of the policy that caused a guardrail block or require_review halt."""

    guardrail_id: str = Field(..., description="ID of the policy that caused the halt")
    guardrail_category: str = Field(..., description="Category of the policy that caused the halt")


class ToolExecuteResponse(BaseModel):
    """Response from OE for tool execution request."""

    proceed: bool = Field(..., description="Whether to proceed with execution")
    cached_result: Optional[Any] = Field(default=None, description="Cached result if replaying")
    route_to: Optional[str] = Field(
        default=None, description="URL to route to (callback or OE-owned stream relay)"
    )
    reason: Optional[str] = Field(default=None, description="Reason if blocked")
    elicitation: Optional[ElicitationInfo] = Field(
        default=None,
        description="Authorization elicitation details when broker consent is required",
    )
    status: Optional[str] = Field(
        default=None,
        description="Final status when OE directly executes the intercepted tool",
    )
    result: Optional[Any] = Field(default=None, description="Final result from OE-owned execution")
    error: Optional[str] = Field(default=None, description="Execution error message")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable failure classification for an invoke_llm error",
    )
    retryable: bool = Field(default=False, description="Whether retry is safe")
    latest_step_number: Optional[int] = Field(
        default=None,
        description="Highest step number observed during nested execution",
    )
    from_cache: bool = Field(default=False, description="Whether replay cache answered")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration")
    pod_name: Optional[str] = Field(default=None, description="Execution pod name")
    guardrail_meta: Optional[GuardrailMeta] = Field(
        default=None, description="Policy identity when a guardrail halt fires"
    )
    tool_api_error: Optional[ToolAPIError] = Field(
        default=None, description="Structured external API failure classification"
    )

    @model_validator(mode="before")
    @classmethod
    def _assemble_guardrail_meta(cls, data: Any) -> Any:
        """Assemble guardrail_meta from the flat guardrail_id/guardrail_category fields
        that the OE wire format sends, so callers work with a single structured object."""
        if isinstance(data, dict):
            gid = data.get("guardrail_id")
            gcat = data.get("guardrail_category")
            if gid and gcat and data.get("guardrail_meta") is None:
                data["guardrail_meta"] = {"guardrail_id": gid, "guardrail_category": gcat}
        return data


class ToolResultRequest(BaseModel):
    """Report tool execution result (AER → OE)."""

    execution_id: str = Field(..., description="Execution identifier")
    step_number: int = Field(..., description="Step number")
    tool_name: str = Field(..., description="Tool name")
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Stable tool-call id from the LLM, joining this result to its call and session message",
    )
    status: str = Field(..., description="Execution status: success, error, suspend, interrupted")
    result: Optional[Any] = Field(default=None, description="Tool result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    pod_name: Optional[str] = Field(
        default=None, description="Hostname/pod name where tool was executed"
    )
    # Token usage fields (populated for invoke_llm calls)
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt/input tokens used")
    completion_tokens: Optional[int] = Field(
        default=None, description="Completion/output tokens used"
    )
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")
    model: Optional[str] = Field(default=None, description="LLM model name")
    workspace_id: Optional[str] = Field(default=None, description="Workspace identifier")
    kind: Optional[str] = Field(default=None, description="Explicit observability kind")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Observability metadata")
    trace_id: Optional[str] = Field(
        default=None, description="Active OTel trace ID, for OE execution-log correlation"
    )
    span_id: Optional[str] = Field(
        default=None, description="Active OTel span ID, for OE execution-log correlation"
    )
    tool_api_error: Optional[ToolAPIError] = Field(
        default=None, description="Structured external API failure classification"
    )

    def to_log(self, execution: Optional["Execution"] = None) -> "ExecutionLog":
        """
        Convert this request to an ExecutionLog for tool result.

        Args:
            execution: The parent execution context (optional)
        """
        from uuid import uuid4

        from agent_engine_runner_shared.logging import ExecutionLog, ExecutionStatus

        exec_status = (
            ExecutionStatus.SUCCESS
            if self.status == "success"
            else ExecutionStatus.SUSPENDED
            if self.status == "suspend"
            else ExecutionStatus.ERROR
            if self.status == "error"
            else ExecutionStatus.BLOCKED
            if self.status == "blocked"
            else ExecutionStatus.CACHED
            if self.status == "cached"
            else ExecutionStatus.ERROR
        )

        return ExecutionLog(
            id=str(uuid4()),
            execution_id=self.execution_id,
            tool=self.tool_name,
            tool_call_id=self.tool_call_id,
            kind=self.kind,
            status=exec_status,
            timestamp=datetime.now(timezone.utc),
            output=self.result,
            error=self.error,
            duration_ms=self.duration_ms,
            step_number=self.step_number,
            session_id=execution.session_id if execution else None,
            user_id=execution.user_id if execution else None,
            org_id=execution.org_id if execution else None,
            project_id=execution.project_id if execution else None,
            pod_name=self.pod_name,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            model=self.model,
            workspace_id=self.workspace_id or (execution.workspace_id if execution else None),
            metadata=self.metadata,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tool_api_error=self.tool_api_error,
        )


# =============================================================================
# Tool Pod Execution (OE → ToolPod)
# =============================================================================

MAX_TOOL_ARGUMENT_BYTES = 16 * 1024 * 1024


class ToolAuthorization(BaseModel):
    """Delegated credential injected by OE for tool execution."""

    token: str = Field(..., repr=False, description="Bearer token for third-party API access")
    expires_at: Optional[int] = Field(
        default=None,
        description="Token expiry as Unix seconds",
        json_schema_extra={"format": "int64"},
    )


class ToolPodExecuteRequest(BaseModel):
    """Request to execute a tool in a Tool Pod."""

    platform_trace_id: Optional[str] = Field(
        default=None, description="Platform request trace ID for log correlation"
    )

    execution_id: str = Field(..., description="Execution identifier")
    tool_name: str = Field(..., description="Name of the tool")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID from LLM")
    # Step number of this call in the execution sequence, so the per-call
    # interrupt (POST /interrupt/call) can address exactly this work.
    # Optional for backward compatibility with older OE dispatchers: absent,
    # the call is only drain-addressable.
    step_number: Optional[int] = Field(
        default=None, ge=0, description="Step number in execution sequence"
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Session ID for memory context. For LangGraph applications, this is the "
            "LangGraph thread ID."
        ),
    )
    user_id: Optional[str] = Field(default=None, description="User ID for context")
    oe_url: Optional[str] = Field(
        default=None, description="OE callback URL for memory and other operations"
    )
    oe_owner_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional OE-supplied replica-specific owner callback URL. "
            "Runners use it as a best-effort optimization when present and "
            "otherwise fall back to oe_url."
        ),
    )
    authorization: Optional[ToolAuthorization] = Field(
        default=None, description="Delegated credential injected by OE via credential broker"
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Caller-provided custom headers"
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Opaque caller-provided invocation payload, forwarded so tools can read it via get_current_payload()",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool execution metadata forwarded by OE. Recognized keys: "
            "mcp_server (str), mcp_tool (str)."
        ),
        json_schema_extra={"default": {}},
    )

    @field_validator("arguments")
    @classmethod
    def validate_argument_size(cls, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Bound the JSON payload every Tool Pod callable can receive."""
        try:
            encoded = json.dumps(
                arguments,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ValueError("tool arguments must contain JSON values") from None
        if len(encoded) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError(f"tool arguments exceed {MAX_TOOL_ARGUMENT_BYTES} bytes")
        return arguments


class ToolPodExecuteResponse(BaseModel):
    """Response from Tool Pod execution."""

    status: str = Field(..., description="Execution status: success, error")
    result: Optional[Any] = Field(default=None, description="Tool result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    pod_name: Optional[str] = Field(
        default=None, description="Hostname/pod name where tool was executed"
    )
    kind: Optional[str] = Field(default=None, description="Explicit observability kind")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Observability metadata")
    oob_suspend_supported: bool = Field(
        default=False,
        description=(
            "True when this runner reports HITL suspend out of band via status. "
            "Absent/false on older runners, which the OE treats "
            "as legacy pods that signal suspend through in-band result content."
        ),
    )
    tool_api_error: Optional[ToolAPIError] = Field(
        default=None, description="Structured external API failure classification"
    )


# =============================================================================
# Guardrails Runtime Check (OE → ToolPod)
# =============================================================================


class GuardrailRuntimeStage(str, Enum):
    """Runtime stage where OE is asking the Tool Pod to evaluate guardrails."""

    LLM_INPUT = "llm_input"
    LLM_OUTPUT = "llm_output"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"


class GuardrailCheckDecision(str, Enum):
    """Decision returned by the Tool Pod guardrails evaluator."""

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    REQUIRE_REVIEW = "require_review"
    LOG_ONLY = "log_only"


class GuardrailRuntimePolicy(BaseModel):
    """OE-selected guardrail policy sent to the Tool Pod for evaluation."""

    id: str = Field(..., description="Guardrail policy identifier")
    type: str = Field(..., description="Guardrail policy type")
    status: str = Field(default="active", description="Guardrail policy status")
    action: str = Field(..., description="Action requested when the policy triggers")
    stage_filter: List[str] = Field(
        default_factory=list,
        description="Runtime stages this policy applies to; empty means all stages",
    )
    config: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Evaluator-specific policy configuration",
    )

    def applies_to_stage(self, stage: GuardrailRuntimeStage) -> bool:
        if self.status.strip().lower() != "active":
            return False
        if not self.stage_filter:
            return True

        return any(value.strip().lower() == stage.value for value in self.stage_filter)

    def check_decision(self) -> GuardrailCheckDecision:
        action = self.action.strip().lower()
        if action in {"modify", "transform", "fix"}:
            return GuardrailCheckDecision.MODIFY
        if action in {"noop", "no_op", "warn", "log_only"}:
            return GuardrailCheckDecision.LOG_ONLY
        if action == "require_review":
            return GuardrailCheckDecision.REQUIRE_REVIEW
        if action in {"block", "exception"}:
            return GuardrailCheckDecision.BLOCK

        on_fail = self.config.get("on_fail")
        if isinstance(on_fail, str):
            normalized_on_fail = on_fail.strip().lower()
            if normalized_on_fail == "fix":
                return GuardrailCheckDecision.MODIFY
            if normalized_on_fail in {"noop", "warn", "log_only"}:
                return GuardrailCheckDecision.LOG_ONLY
            if normalized_on_fail in {"block", "exception"}:
                return GuardrailCheckDecision.BLOCK

        return GuardrailCheckDecision.ALLOW


class GuardrailCheckInput(BaseModel):
    """Runtime content and metadata to evaluate."""

    text: str = Field(..., description="Text content to evaluate")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Runtime metadata such as model, tool name, or source",
    )


class GuardrailCheckContext(BaseModel):
    """Execution context for a guardrail check."""

    org_id: str = Field(..., description="Organization ID")
    project_id: str = Field(..., description="Project ID")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    user_id: Optional[str] = Field(default=None, description="User ID")


class GuardrailCheckRequest(BaseModel):
    """Request from OE to Tool Pod to evaluate selected guardrail policies."""

    execution_id: str = Field(..., description="Execution identifier")
    stage: GuardrailRuntimeStage = Field(..., description="Runtime stage being evaluated")
    input: GuardrailCheckInput = Field(..., description="Content to evaluate")
    context: GuardrailCheckContext = Field(..., description="Execution context")
    policies: List[GuardrailRuntimePolicy] = Field(
        default_factory=list,
        description="OE-selected policies to evaluate",
    )


class GuardrailCheckEvidence(BaseModel):
    """Evidence explaining why a guardrail policy triggered."""

    policy_id: str = Field(..., description="Triggered policy identifier")
    message: str = Field(default="", description="Human-readable evidence summary")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Evaluator-specific evidence metadata",
    )


class GuardrailCheckResponse(BaseModel):
    """Decision returned by the Tool Pod guardrails evaluator."""

    decision: GuardrailCheckDecision = Field(..., description="Guardrails decision")
    allowed: bool = Field(..., description="Whether execution may continue")
    transformed_text: Optional[str] = Field(
        default=None,
        description="Modified text when decision is modify, or original text for allow/no-op",
    )
    triggered_policy_ids: List[str] = Field(
        default_factory=list,
        description="Policy IDs that triggered",
    )
    evidence: List[GuardrailCheckEvidence] = Field(
        default_factory=list,
        description="Policy trigger evidence",
    )
    reason: Optional[str] = Field(default=None, description="Decision reason")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Evaluator-specific response metadata",
    )


# =============================================================================
# LLM Pod Execution (AER → ToolPod)
# =============================================================================


class InvokeLLMRequestArguments(BaseModel):
    """Typed invoke_llm arguments forwarded through OE and tool pods."""

    model: str = Field(
        ..., description="Model name (e.g. 'gpt-5.4-mini', 'gemini-3-flash-preview')"
    )
    messages: List[Message] = Field(
        ..., description="Conversation in agent-engine-sdk Message wire format"
    )
    llm_id: str = Field(
        default="__default__",
        description="Stable identifier for the LLM instance registered via app.llm(llm_id=...); "
        "the tool pod resolves this id against its named-LLM registry. "
        "Unnamed app.llm() calls register under the sentinel id '__default__'. "
        "Defaults to '__default__' for backward compatibility with callers that omit this field.",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("stop_sequences", "stop"),
        serialization_alias="stop",
        description="Stop sequences forwarded to the underlying LLM provider",
    )
    tools: Optional[List[LLMToolSchema]] = Field(
        default=None, description="Serialized tool schemas for bind_tools"
    )
    tool_choice: Optional[JsonValue] = Field(
        default=None,
        description="Forced tool selection forwarded to the tool pod's bind_tools "
        "call (e.g. a function name from with_structured_output). LangChain "
        "translates the value against the bound tools.",
    )
    options: Optional[LLMInvocationOptions] = Field(
        default=None,
        description="Explicit provider/model invocation options (for example max_tokens)",
    )
    stream: bool = Field(
        default=False,
        description="Whether OE should approve and route a real streaming invoke_llm call",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_options_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "options" in value or "kwargs" not in value:
            return value
        payload = dict(value)
        payload["options"] = payload.pop("kwargs")
        return payload


class LLMPodInvokeRequest(BaseModel):
    """Request to invoke LLM on a tool executor pod."""

    platform_trace_id: Optional[str] = Field(
        default=None, description="Platform request trace ID for log correlation"
    )
    execution_id: str = Field(..., description="Execution identifier")
    arguments: InvokeLLMRequestArguments = Field(..., description="Typed invoke_llm arguments")
    # Step number of this LLM call, so the per-call interrupt
    # (POST /interrupt/call) can address exactly this invocation. Optional
    # for backward compatibility with older OE dispatchers.
    step_number: Optional[int] = Field(
        default=None, ge=0, description="Step number in execution sequence"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_flat_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "arguments" in value:
            return value
        payload = dict(value)
        if "execution_id" not in payload:
            return value
        execution_id = payload.pop("execution_id")
        platform_trace_id = payload.pop("platform_trace_id", None)
        # step_number is routing metadata, not an LLM argument — keep it out
        # of the arguments bag a flat sender would otherwise pour it into.
        step_number = payload.pop("step_number", None)
        out: dict[str, Any] = {"execution_id": execution_id, "arguments": payload}
        if platform_trace_id is not None:
            out["platform_trace_id"] = platform_trace_id
        if step_number is not None:
            out["step_number"] = step_number
        return out


_USAGE_INT_KEYS = (
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
_USAGE_META_KEYS = frozenset({"usage", "token_usage"})
_JSON_SAFE_DEPTH = 8


def _usage_int(value: Any) -> int | None:
    """Return a non-negative token count when ``value`` is a real number.

    Bools are rejected (``True`` is an ``int``). NaN/infinity and negatives
    are skipped so a malformed provider blob cannot fail the LLM call.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            return None
        return int(value)
    return None


def _has_token_counts(usage: LLMTokenUsage) -> bool:
    return any(
        v is not None
        for v in (
            usage.input_tokens,
            usage.output_tokens,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    )


def _prompt_of(usage: LLMTokenUsage) -> int | None:
    if usage.input_tokens is not None:
        return usage.input_tokens
    return usage.prompt_tokens


def _completion_of(usage: LLMTokenUsage) -> int | None:
    if usage.output_tokens is not None:
        return usage.output_tokens
    return usage.completion_tokens


def _safe_attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except Exception:  # noqa: BLE001 - usage coercion is best-effort
        return None


def _validate_usage_mapping(data: Mapping[str, Any]) -> LLMTokenUsage | None:
    if not data:
        return None
    try:
        filtered: Dict[str, Any] = {}
        for key in _USAGE_INT_KEYS:
            parsed = _usage_int(data.get(key))
            if parsed is not None:
                filtered[key] = parsed
        model = data.get("model")
        if isinstance(model, str) and model:
            filtered["model"] = model
        if not filtered:
            return None
        usage = LLMTokenUsage.model_validate(filtered)
    except Exception:  # noqa: BLE001 - usage coercion is best-effort
        return None
    return usage if _has_token_counts(usage) else None


def coerce_token_usage(value: Any) -> Optional[LLMTokenUsage]:
    """Turn a provider usage blob into ``LLMTokenUsage``, or None.

    Accepts dicts, non-dict ``Mapping``s, Pydantic ``model_dump()`` objects
    (Anthropic ``Usage``), and attribute bags (``input_tokens`` /
    ``prompt_tokens``). Malformed values return None rather than raising —
    a bad usage blob must not fail the LLM call.
    """
    try:
        return _coerce_token_usage_inner(value)
    except Exception:  # noqa: BLE001 - usage coercion is best-effort
        return None


def _coerce_token_usage_inner(value: Any) -> Optional[LLMTokenUsage]:
    if value is None:
        return None
    if isinstance(value, LLMTokenUsage):
        return value if _has_token_counts(value) else None
    if isinstance(value, (str, bytes, bool, int, float)):
        return None

    if isinstance(value, Mapping):
        return _validate_usage_mapping(value)

    dump_fn = _safe_attr(value, "model_dump")
    if callable(dump_fn):
        try:
            dumped = dump_fn()
        except Exception:  # noqa: BLE001 - usage coercion is best-effort
            dumped = None
        if isinstance(dumped, Mapping):
            coerced = _validate_usage_mapping(dumped)
            if coerced is not None:
                return coerced

    data: Dict[str, Any] = {}
    for key in _USAGE_INT_KEYS:
        parsed = _usage_int(_safe_attr(value, key))
        if parsed is not None:
            data[key] = parsed
    model = _safe_attr(value, "model")
    if isinstance(model, str) and model:
        data["model"] = model
    return _validate_usage_mapping(data) if data else None


def _usage_json(usage: LLMTokenUsage) -> Dict[str, JsonValue]:
    dumped = usage.model_dump(mode="json", exclude_none=True)
    return {key: dumped[key] for key in (*_USAGE_INT_KEYS, "model") if key in dumped}


def _json_safe_value(value: Any, depth: int) -> JsonValue | None:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, LLMTokenUsage):
        return _usage_json(usage=value)
    if depth >= _JSON_SAFE_DEPTH:
        return None
    if isinstance(value, Mapping):
        return json_safe_metadata(value, _depth=depth)
    if isinstance(value, (list, tuple)):
        items: List[JsonValue] = []
        for item in value:
            if item is None:
                items.append(None)
                continue
            safe = _json_safe_value(item, depth + 1)
            if safe is not None:
                items.append(safe)
        return items
    coerced = coerce_token_usage(value)
    if coerced is not None:
        return _usage_json(coerced)
    return None


def json_safe_metadata(value: Any, *, _depth: int = 0) -> Dict[str, JsonValue] | None:
    """Copy a provider metadata mapping into JSON-safe values, or None.

    ``usage`` / ``token_usage`` are replaced with a dump of the coerced
    ``LLMTokenUsage``. Other non-JSON values are omitted so invoke/stream
    cannot fail after a successful model response.
    """
    if not isinstance(value, Mapping) or not value:
        return None
    if _depth >= _JSON_SAFE_DEPTH:
        return None
    out: Dict[str, JsonValue] = {}
    try:
        items = list(value.items())
    except Exception:  # noqa: BLE001 - metadata sanitization is best-effort
        return None
    for key, raw in items:
        if not isinstance(key, str):
            continue
        if key in _USAGE_META_KEYS:
            coerced = coerce_token_usage(raw)
            if coerced is not None:
                out[key] = _usage_json(coerced)
            continue
        if raw is None:
            out[key] = None
            continue
        safe = _json_safe_value(raw, _depth + 1)
        if safe is not None:
            out[key] = safe
    return out or None


def add_token_usage(
    existing: Optional[LLMTokenUsage],
    incoming: Any,
) -> Optional[LLMTokenUsage]:
    """Sum LangChain-style additive usage deltas into a cumulative snapshot.

    Missing components count as 0. Tool-pod snapshot merge must not use
    this helper — OpenAI-style last-chunk snapshots would double-count.
    Adapters should call ``accumulate_stream_usage`` so cumulative
    per-chunk snapshots are not added.
    """
    next_usage = incoming if isinstance(incoming, LLMTokenUsage) else coerce_token_usage(incoming)
    if next_usage is None:
        return existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    prior = existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    if prior is None:
        return next_usage

    prompt = (_prompt_of(prior) or 0) + (_prompt_of(next_usage) or 0)
    completion = (_completion_of(prior) or 0) + (_completion_of(next_usage) or 0)
    model = next_usage.model or prior.model
    if prior.total_tokens is not None or next_usage.total_tokens is not None:
        total = (prior.total_tokens or 0) + (next_usage.total_tokens or 0)
    else:
        total = prompt + completion
    return LLMTokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        model=model,
    )


def _looks_like_usage_delta(prior: LLMTokenUsage, incoming: LLMTokenUsage) -> bool:
    """True when incoming decreased a counted field (Gemini zero-filled deltas)."""
    pairs = (
        (_prompt_of(prior), _prompt_of(incoming)),
        (_completion_of(prior), _completion_of(incoming)),
        (prior.total_tokens, incoming.total_tokens),
    )
    return any(prev is not None and nxt is not None and nxt < prev for prev, nxt in pairs)


def accumulate_stream_usage(
    existing: Optional[LLMTokenUsage],
    incoming: Any,
) -> Optional[LLMTokenUsage]:
    """Fold a stream chunk's usage into a running snapshot.

    Cumulative providers repeat growing totals (``10/1`` then ``10/2``);
    last-wins merge keeps ``10/2``. Additive providers zero-fill the
    unchanged side (``18/1`` then ``0/4``); those deltas are summed.
    """
    next_usage = incoming if isinstance(incoming, LLMTokenUsage) else coerce_token_usage(incoming)
    if next_usage is None:
        return existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    prior = existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    if prior is None:
        return next_usage
    if _looks_like_usage_delta(prior, next_usage):
        return add_token_usage(prior, next_usage)
    return merge_token_usage(prior, next_usage)


def merge_token_usage(
    existing: Optional[LLMTokenUsage],
    incoming: Any,
) -> Optional[LLMTokenUsage]:
    """Merge two usage snapshots field-by-field.

    Later chunks fill missing prompt/completion/total/model rather than
    replacing the whole object. Incoming non-null fields win, so an
    Anthropic message_start (input only) plus message_delta (output only)
    yields both. An explicit ``total_tokens`` is preserved.
    """
    next_usage = incoming if isinstance(incoming, LLMTokenUsage) else coerce_token_usage(incoming)
    if next_usage is None:
        return existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    prior = existing if isinstance(existing, LLMTokenUsage) else coerce_token_usage(existing)
    if prior is None:
        return next_usage

    incoming_prompt = _prompt_of(next_usage)
    incoming_completion = _completion_of(next_usage)
    prompt = incoming_prompt if incoming_prompt is not None else _prompt_of(prior)
    completion = incoming_completion if incoming_completion is not None else _completion_of(prior)
    model = next_usage.model or prior.model
    total = next_usage.total_tokens if next_usage.total_tokens is not None else prior.total_tokens
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return LLMTokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        model=model,
    )


class LLMResult(BaseModel):
    """Normalized result from an LLM invocation.

    Provides a consistent Pydantic shape for LLM responses regardless of the
    underlying provider (OpenAI, Gemini, Anthropic, etc.).
    """

    content: str = Field(default="", description="Generated text content")
    tool_calls: List[LLMToolCall] = Field(
        default_factory=list, description="Tool calls requested by the model"
    )
    usage: Optional[LLMTokenUsage] = Field(
        default=None, description="Token usage (input_tokens, output_tokens, total_tokens)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    id: Optional[str] = Field(default=None, description="Provider response identifier")
    name: Optional[str] = Field(default=None, description="Provider response name")
    additional_kwargs: Optional[Dict[str, JsonValue]] = Field(
        default=None, description="LangChain additional message kwargs"
    )
    response_metadata: Optional[Dict[str, JsonValue]] = Field(
        default=None, description="LangChain response metadata"
    )

    @classmethod
    def from_response(cls, response: Any) -> "LLMResult":
        """Extract a normalized result from a LangChain LLM response.

        Handles AIMessage, BaseMessage, and arbitrary objects by falling back
        to str() for content extraction.
        """
        if isinstance(response, LLMResponse):
            return cls(
                content=response.content,
                tool_calls=response.tool_calls or [],
                usage=response.usage,
                metadata=json_safe_metadata(response.metadata) or {},
                id=response.id,
                name=response.name,
                additional_kwargs=json_safe_metadata(response.additional_kwargs),
                response_metadata=json_safe_metadata(response.response_metadata),
            )

        content = response.content if hasattr(response, "content") else str(response)
        tool_calls = _safe_attr(response, "tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        usage = cls.extract_usage(response)
        metadata = json_safe_metadata(_safe_attr(response, "metadata")) or {}
        response_metadata = json_safe_metadata(_safe_attr(response, "response_metadata"))
        additional_kwargs = json_safe_metadata(_safe_attr(response, "additional_kwargs"))
        response_id = _safe_attr(response, "id")
        response_name = _safe_attr(response, "name")

        return cls(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
            id=response_id if isinstance(response_id, str) else None,
            name=response_name if isinstance(response_name, str) else None,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
        )

    @staticmethod
    def extract_usage(response: Any) -> Optional[LLMTokenUsage]:
        """Extract token usage metadata from a LangChain response or chunk.

        Coerces provider-variant shapes (dicts, non-dict Mappings, Anthropic
        Usage objects, ``.usage`` attribute bags) into ``LLMTokenUsage``.
        Malformed blobs return None rather than raising.
        """
        try:
            if isinstance(response, LLMResponse) and response.usage is not None:
                return response.usage

            usage_meta = _safe_attr(response, "usage_metadata")
            coerced = coerce_token_usage(usage_meta)
            if coerced is not None:
                return coerced

            response_metadata = _safe_attr(response, "response_metadata")
            if isinstance(response_metadata, Mapping):
                for key in ("usage", "token_usage"):
                    coerced = coerce_token_usage(response_metadata.get(key))
                    if coerced is not None:
                        return coerced

            coerced = coerce_token_usage(_safe_attr(response, "usage"))
            if coerced is not None:
                return coerced

            metadata = _safe_attr(response, "metadata")
            if isinstance(metadata, Mapping):
                return coerce_token_usage(metadata.get("usage"))
            return None
        except Exception:  # noqa: BLE001 - usage extraction is best-effort
            return None

    def to_response(self) -> LLMResponse:
        """Convert the normalized runner result to sdk-core's response shape."""
        metadata = dict(self.metadata)
        if self.usage is not None and not metadata:
            metadata = self.usage.model_dump(exclude_none=True)
        return LLMResponse(
            content=self.content,
            tool_calls=self.tool_calls or None,
            metadata=metadata,
            usage=self.usage,
            id=self.id,
            name=self.name,
            additional_kwargs=self.additional_kwargs,
            response_metadata=self.response_metadata,
        )


class LLMPodInvokeResponse(BaseModel):
    """Response from LLM invocation on a tool executor pod."""

    status: str = Field(..., description="Execution status: success, error")
    result: Optional[LLMResponse] = Field(
        default=None, description="LLM result with content, tool_calls, and usage"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable failure classification (e.g. llm_credential_rejected)",
    )
    pod_name: str = Field(..., description="Hostname/pod name where LLM was executed")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    usage: Optional[LLMTokenUsage] = Field(
        default=None, description="Token usage (input_tokens, output_tokens, total_tokens)"
    )

    @model_validator(mode="after")
    def _sync_usage_with_result(self) -> "LLMPodInvokeResponse":
        if self.result is not None and self.result.usage is None and self.usage is not None:
            self.result.usage = self.usage
        if self.usage is None and self.result is not None:
            self.usage = self.result.usage
        return self


class LLMPodStreamEvent(BaseModel):
    """SSE event emitted by a tool pod during invoke_llm streaming."""

    content: Optional[str] = Field(default=None, description="Generated text delta")
    tool_call_chunks: Optional[List[ToolCallChunk]] = Field(
        default=None, description="Partial tool-call deltas"
    )
    tool_calls: Optional[List[LLMToolCall]] = Field(
        default=None, description="Complete tool calls for compatibility with older stream senders"
    )
    id: Optional[str] = Field(default=None, description="Provider message identifier")
    name: Optional[str] = Field(default=None, description="Provider message name")
    additional_kwargs: Optional[Dict[str, JsonValue]] = Field(
        default=None, description="LangChain additional message kwargs"
    )
    response_metadata: Optional[Dict[str, JsonValue]] = Field(
        default=None, description="LangChain response metadata"
    )
    done: Optional[bool] = Field(default=None, description="Whether the stream is complete")
    interrupted: Optional[bool] = Field(
        default=None,
        description="OE stopped this call on interrupt request; terminal, distinct from done/error",
    )
    error: Optional[str] = Field(default=None, description="Error message if streaming failed")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable failure classification (e.g. llm_credential_rejected)",
    )
    retryable: Optional[bool] = Field(
        default=None,
        description="When true with error, the same stream URL may be retried",
    )
    retry_after_ms: Optional[float] = Field(
        default=None,
        description="When set with retryable, wait this many milliseconds before retrying",
    )
    pod_name: Optional[str] = Field(default=None, description="Hostname/pod name")
    duration_ms: Optional[float] = Field(default=None, description="Stream duration")
    usage: Optional[LLMTokenUsage] = Field(default=None, description="Final token usage")


# =============================================================================
# Executor Callback (AER → OE)
# =============================================================================


class ExecutorCallbackRequest(BaseModel):
    """Callback from AER to OE when execution completes or suspends."""

    execution_id: str = Field(..., description="Execution identifier")
    status: str = Field(..., description="Status: COMPLETED, SUSPENDED, ERROR")
    suspend_generation: Optional[int] = Field(
        default=None,
        ge=0,
        description="Suspension generation expected by the originating dispatch, when provided",
    )
    result: Optional[Any] = Field(default=None, description="Final result if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    suspend_reason: Optional[str] = Field(default=None, description="Reason for suspension")
    suspend_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Legacy context for resume"
    )
    interrupts: Optional[List[PendingInterrupt]] = Field(
        default=None, description="Complete framework-native pending interrupt snapshot"
    )
    resume_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for the Atlas Agent Engine continue request"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Opaque framework-owned state returned by the selected adapter",
    )


# =============================================================================
# Resume (Client → OE)
# =============================================================================


class HumanReviewData(BaseModel):
    """Data provided by human reviewer when resuming a suspended execution."""

    decision: str = Field(
        ...,
        description="Review decision: 'approved', 'rejected', or custom value",
        examples=["approved", "rejected", "approved_with_conditions"],
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Optional notes from the reviewer",
    )


class AgentResumeRequest(BaseModel):
    """Request to resume a suspended execution."""

    human_review: HumanReviewData = Field(
        ...,
        description="Human review decision data (for human-in-the-loop resume)",
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Caller-provided custom headers"
    )


class AgentResumeResponse(BaseModel):
    """Response from resuming an execution."""

    execution_id: str = Field(..., description="Execution identifier")
    status: str = Field(..., description="Execution status")


# =============================================================================
# Execution Status Query
# =============================================================================


class ExecutionStatusResponse(BaseModel):
    """Response for execution status query."""

    execution_id: str = Field(..., description="Execution identifier")
    status: ExecutionStatus = Field(..., description="Current status")
    result: Optional[Any] = Field(default=None, description="Result if completed")
    error: Optional[str] = Field(default=None, description="Error if failed")
    suspend_reason: Optional[str] = Field(default=None, description="Reason if suspended")
    suspend_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Context if suspended"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# =============================================================================
# Database Models
# =============================================================================


class Execution(BaseModel):
    """Execution record stored in TenantDB."""

    id: str = Field(..., description="Unique execution identifier")
    status: ExecutionStatus = Field(..., description="Current status")
    message: str = Field(..., description="Original user message")
    result: Optional[Any] = Field(default=None, description="Final result")
    error: Optional[str] = Field(default=None, description="Error message")
    suspend_reason: Optional[str] = Field(default=None, description="Reason for suspension")
    suspend_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Context for resume"
    )
    # Multi-tenant context
    session_id: Optional[str] = Field(default=None, description="Session ID for trace correlation")
    org_id: Optional[str] = Field(
        default=None, description="Organization ID for multi-tenant isolation"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace identifier for cost tracking"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project ID for project-level scoping"
    )
    # Per-agent AER routing
    aer_url: Optional[str] = Field(
        default=None, description="AER HTTP endpoint for this agent; overrides default AER_URL"
    )
    tool_url: Optional[str] = Field(
        default=None, description="Tool HTTP endpoint for this agent; overrides default TOOL_URL"
    )
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_persistence_doc(self, updated_at: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Convert to a document for MongoDB persistence.

        Args:
            updated_at: Optional timestamp override (defaults to now)

        Returns:
            Dict suitable for MongoDB upsert
        """
        return {
            "execution_id": self.id,
            "status": self.status.value,
            "message": self.message,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "result": self.result,
            "error": self.error,
            "suspend_reason": self.suspend_reason,
            "suspend_context": self.suspend_context,
            "updated_at": updated_at or datetime.now(timezone.utc),
        }

    @staticmethod
    def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
        """Normalize a datetime to UTC-aware.

        PyMongo returns naive datetimes by default. The OE uses
        ``datetime.now(timezone.utc)`` everywhere, so mixing aware and
        naive values causes ``TypeError`` on comparison.  This helper
        stamps naive datetimes as UTC (which is what MongoDB stores).
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @classmethod
    def from_persistence_doc(cls, doc: Dict[str, Any]) -> "Execution":
        """
        Rehydrate an Execution from a MongoDB document.

        Uses .get() with defaults for all Optional fields so documents
        created before new fields were added still deserialize safely.
        """
        now = datetime.now(timezone.utc)
        return cls(
            id=doc["execution_id"],
            status=ExecutionStatus(doc["status"]),
            message=doc.get("message", ""),
            result=doc.get("result"),
            error=doc.get("error"),
            suspend_reason=doc.get("suspend_reason"),
            suspend_context=doc.get("suspend_context"),
            session_id=doc.get("session_id"),
            org_id=doc.get("org_id"),
            user_id=doc.get("user_id"),
            workspace_id=doc.get("workspace_id"),
            project_id=doc.get("project_id"),
            created_at=cls._ensure_utc(doc.get("created_at"))
            or cls._ensure_utc(doc.get("updated_at"))
            or now,
            updated_at=cls._ensure_utc(doc.get("updated_at")) or now,
        )


class ExecutionStep(BaseModel):
    """Execution step record stored in TenantDB."""

    id: str = Field(..., description="Unique step identifier")
    execution_id: str = Field(..., description="Parent execution ID")
    step_number: int = Field(..., description="Step sequence number")
    tool_name: str = Field(..., description="Tool or operation name")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed")
    status: str = Field(..., description="Step status: pending, success, error")
    result: Optional[Any] = Field(default=None, description="Step result")
    error: Optional[str] = Field(default=None, description="Error message")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_log_doc(cls, doc: Dict[str, Any]) -> "ExecutionStep":
        """
        Reconstruct an ExecutionStep from an execution_logs MongoDB document.

        The execution_logs collection stores tool start/result events with
        fields: execution_id, step_number, tool, inputs, status, output,
        error, duration_ms, timestamp. This method maps those fields back
        to the ExecutionStep model for step cache rehydration after OE restart.
        """
        from uuid import uuid4

        return cls(
            id=doc.get("id", str(uuid4())),
            execution_id=doc["execution_id"],
            step_number=doc.get("step_number", 0),
            tool_name=doc.get("tool", ""),
            arguments=doc.get("inputs") or {},
            status=doc.get("status", "error"),
            result=doc.get("output"),
            error=doc.get("error"),
            duration_ms=doc.get("duration_ms"),
            timestamp=Execution._ensure_utc(doc.get("timestamp")) or datetime.now(timezone.utc),
        )


# =============================================================================
# Health Check
# =============================================================================


class HealthStatus(str, Enum):
    """Health status of a component."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus = Field(..., description="Health status")
    component: str = Field(..., description="Component name")
    mode: str = Field(..., description="Runtime mode")
    version: str = Field(default="1.0.0", description="Component version")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


# =============================================================================
# Tool Definition
# =============================================================================


class ToolDefinition(BaseModel):
    """Definition of a registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    is_local: bool = Field(default=True, description="Whether tool runs locally")
    provider_type: Optional[str] = Field(
        default=None, description="Credential provider type for delegated auth"
    )
    scopes: List[str] = Field(
        default_factory=list, description="Requested OAuth scopes for delegated auth"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema for parameters"
    )


# =============================================================================
# AER Route Response Models
# =============================================================================


class AERExecuteResponse(BaseModel):
    """Response from AER /execute endpoint.

    Returned on both normal completion and HITL suspension.
    """

    status: str = Field(
        ...,
        description="Execution outcome: 'completed' or 'suspended'",
        examples=["completed", "suspended"],
    )
    result: Optional[str] = Field(
        default=None,
        description="Final agent response text (present when status is 'completed')",
    )
    suspend_reason: Optional[str] = Field(
        default=None,
        description="Why the agent suspended (present when status is 'suspended')",
    )


class ToolsListResponse(BaseModel):
    """Response from /tools endpoint listing registered tools."""

    tools: List[Dict[str, Any]] = Field(
        ..., description="Registered tool definitions with name and metadata"
    )
    count: Optional[int] = Field(
        default=None, description="Number of registered tools (included by Tool Pod)"
    )


# =============================================================================
# Streaming
# =============================================================================


class StreamChunk(BaseModel):
    """
    A chunk of streaming response from the agent.

    Used for real-time streaming of agent responses via SSE or gRPC.
    """

    chunk_type: str = Field(
        ...,
        description=(
            "Chunk type string. AER-emitted values are defined in "
            "agent_engine_runner_shared.server.chunk_types ('text', 'subagent_start', "
            "'subagent_end', 'done', 'error'). OE may inject additional "
            "infrastructure types (e.g. 'metadata', 'tool_call', 'tool_result') "
            "before forwarding to clients."
        ),
    )
    content: str = Field(default="", description="Content of the chunk")
    metadata: Dict[str, JsonValue] = Field(default_factory=dict, description="Optional metadata")
    error: Optional[str] = Field(default=None, description="Error message if chunk_type is 'error'")
    code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code if chunk_type is 'error' (e.g. AGENT_UNAVAILABLE)",
    )

    # Author-shaped output-parser payload (@app.output_parser). Present only on
    # parser-produced chunks; consumers gate on its presence, not on chunk_type.
    # Modeled here so typed parsing round-trips it instead of silently dropping it.
    custom_event: Optional[JsonValue] = Field(
        default=None, description="Custom output-parser payload, if present"
    )

    # Tool-specific fields
    tool_name: Optional[str] = Field(
        default=None, description="Tool name for tool_call/tool_result chunks"
    )
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID")

    # Execution context
    execution_id: Optional[str] = Field(default=None, description="Execution ID")
    step_number: Optional[int] = Field(default=None, description="Step number in execution")


class AgentStartStreamRequest(BaseModel):
    """Request to start an agent execution with streaming response."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation continuity"
    )
    org_id: Optional[str] = Field(
        default=None, description="Organization ID for multi-tenant isolation"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")


# =============================================================================
# Node Execution (AER → OE)
# =============================================================================


# =============================================================================
# Query Response Models — Execution Logs & Node Executions
# =============================================================================


class ExecutionLogsQueryResponse(BaseModel):
    """Response for execution logs query (used by API Gateway proxy)."""

    logs: List[Dict[str, Any]] = Field(default_factory=list, description="Execution log documents")
    count: int = Field(0, description="Number of logs returned")


class NodeExecutionsQueryResponse(BaseModel):
    """Response for node executions query (used by API Gateway proxy)."""

    executions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Node execution documents"
    )
    count: int = Field(0, description="Number of executions returned")


# Cost Dashboard Query Response Models
# =============================================================================


class CostSummary(BaseModel):
    """Aggregate cost metrics for the requested period."""

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_calls: int = 0
    unpriced_llm_calls: int = 0


class CostByWorkspace(BaseModel):
    """Cost breakdown for a single workspace."""

    workspace_id: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    percentage: float = 0.0


class CostByModel(BaseModel):
    """Cost breakdown for a single model."""

    model: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    percentage: float = 0.0


class DailyCostEntry(BaseModel):
    """Cost data for a single day."""

    date: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    call_count: int = 0


class CostDashboardResponse(BaseModel):
    """Response for cost dashboard aggregation (used by API Gateway proxy)."""

    summary: CostSummary = Field(default_factory=CostSummary)
    by_workspace: List[CostByWorkspace] = Field(default_factory=list)
    by_model: List[CostByModel] = Field(default_factory=list)
    daily_trend: List[DailyCostEntry] = Field(default_factory=list)


# =============================================================================
# Executions Query Response Models
# =============================================================================


class ExecutionDocument(BaseModel):
    """An execution document as stored in the platform database."""

    execution_id: str
    status: str = ""
    message: str = ""
    session_id: str = ""
    user_id: str = ""
    org_id: str = ""
    project_id: Optional[str] = ""
    workspace_id: Optional[str] = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    suspend_reason: Optional[str] = None
    suspend_context: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ExecutionsListQueryResponse(BaseModel):
    """Response for executions list query (used by API Gateway proxy)."""

    success: bool = True
    executions: List[ExecutionDocument] = Field(default_factory=list)
    count: int = 0


class ExecutionDetailQueryResponse(BaseModel):
    """Response for single execution detail query (used by API Gateway proxy)."""

    success: bool = True
    execution: Optional[ExecutionDocument] = None
    error: Optional[str] = None


# =============================================================================
# Node Execution (AER → OE)
# =============================================================================


class NodeExecutionRequest(BaseModel):
    """Report node execution event (AER → OE for logging)."""

    execution_id: str = Field(..., description="Execution identifier")
    node_name: str = Field(..., description="Name of the LangGraph node")
    status: str = Field(..., description="Node status: started, success, error, suspend")
    timestamp: datetime = Field(..., description="Event timestamp")
    run_id: str = Field(..., description="LangChain run ID for this node execution")
    parent_run_id: Optional[str] = Field(default=None, description="Parent run ID if nested")
    session_id: Optional[str] = Field(default=None, description="Session ID for correlation")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    inputs: Optional[Dict[str, Any]] = Field(
        default=None, description="Node inputs (for started status)"
    )
    outputs: Optional[Dict[str, Any]] = Field(
        default=None, description="Node outputs (for success status)"
    )
    error: Optional[str] = Field(default=None, description="Error message (for error status)")
    duration_ms: Optional[float] = Field(
        default=None, description="Execution duration in milliseconds"
    )
    org_id: Optional[str] = Field(default=None, description="Organization ID")
    project_id: Optional[str] = Field(default=None, description="Project ID")
    trace_id: Optional[str] = Field(
        default=None, description="Active OTel trace ID, for OE execution-log correlation"
    )
    span_id: Optional[str] = Field(
        default=None, description="Active OTel span ID, for OE execution-log correlation"
    )

    def to_log(self) -> "NodeExecutionLog":
        """Convert this request to a NodeExecutionLog for persistence."""
        from uuid import uuid4

        from agent_engine_runner_shared.logging import NodeExecutionLog, NodeExecutionStatus

        node_status = (
            NodeExecutionStatus.STARTED
            if self.status == "started"
            else NodeExecutionStatus.SUCCESS
            if self.status == "success"
            else NodeExecutionStatus.SUSPENDED
            if self.status == "suspend"
            else NodeExecutionStatus.ERROR
        )

        return NodeExecutionLog(
            id=str(uuid4()),
            execution_id=self.execution_id,
            node=self.node_name,
            status=node_status,
            timestamp=self.timestamp,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            inputs=self.inputs,
            outputs=self.outputs,
            error=self.error,
            duration_ms=self.duration_ms,
            session_id=self.session_id,
            user_id=self.user_id,
            org_id=self.org_id,
            project_id=self.project_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
        )
