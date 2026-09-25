"""
Runner SDK - Framework-agnostic agent runtime.

This package provides the TenantRuntime SDK for building agents that work
with the Orchestration Engine (OE), Agent Execution Runtime (AER), and Tool pods.
"""

# Copyright 2026 MongoDB, Inc.
# SPDX-License-Identifier: Apache-2.0

from agent_engine_runner_shared.agent_config import (
    A2AConfig,
    A2ASkillConfig,
    RuntimeAgentConfig,
    RuntimeMCPAuthConfig,
    RuntimeMCPConfig,
    RuntimeMCPServerConfig,
    load_runtime_agent_config,
)

# A2A client: import from agent_engine_runner_shared.a2a when needed.
# e.g.: from agent_engine_runner_shared.a2a import AgentToAgent
from agent_engine_runner_shared.context import (
    SessionFinishStatus,
    clear_execution_context,
    current_custom_headers,
    current_execution_id,
    current_execution_metadata,
    current_oe_url,
    current_request_id,
    current_trace_id,
    current_user_id,
    current_wrapper,
    get_current_custom_headers,
    get_current_execution_id,
    get_current_execution_metadata,
    get_current_oe_url,
    get_current_payload,
    get_current_request_id,
    get_current_trace_id,
    get_current_user_id,
    get_current_wrapper,
    is_session_finish_requested,
    record_current_memory_metadata,
    request_session_finish,
    set_execution_context,
)
from agent_engine_runner_shared.custom_events import emit_custom_event, emit_custom_event_sync
from agent_engine_runner_shared.guardrails_evaluator import (
    GuardrailPolicyEngine,
    GuardrailPolicyEngineResult,
    register_guardrail_policy_engine,
)
from agent_engine_runner_shared.metrics import (
    Metrics,
    log_execution_event,
    log_llm_call,
    log_llm_result,
    log_tool_call,
    log_tool_result,
    record_error,
    record_latency,
)
from agent_engine_runner_shared.models import (
    AgentResumeRequest,
    AgentResumeResponse,
    Execution,
    ExecutionStatus,
    ExecutionStatusResponse,
    ExecutionStep,
    ExecutorCallbackRequest,
    GuardrailCheckDecision,
    GuardrailCheckEvidence,
    GuardrailCheckInput,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailRuntimePolicy,
    GuardrailRuntimeStage,
    InvokeRequest,
    InvokeResponse,
    SuspendPayload,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolPodExecuteRequest,
    ToolPodExecuteResponse,
    ToolResultRequest,
)
from agent_engine_runner_shared.progress import emit, emit_step
from agent_engine_runner_shared.runtime import TenantRuntime
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import (
    ExternalAPICallError,
    LLMInvocationError,
    PolicyDeniedException,
    SecureToolWrapper,
    TerminalExecutionError,
    ToolCallTimeoutError,
    ToolExecutionError,
    extract_usage,
)
from agent_engine_runner_shared.structured_logging import (
    LoggingStream,
    StructuredJSONFormatter,
    install_structured_logging,
)
from agent_engine_runner_shared.utils import (
    RuntimeMode,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_request_timeout,
    get_runtime_mode,
    setup_logging,
)

__all__ = [
    # Main SDK
    "TenantRuntime",
    "emit",
    "emit_step",
    "emit_custom_event",
    "emit_custom_event_sync",
    "RuntimeAgentConfig",
    "A2AConfig",
    "A2ASkillConfig",
    "RuntimeMCPAuthConfig",
    "RuntimeMCPConfig",
    "RuntimeMCPServerConfig",
    "load_runtime_agent_config",
    # A2A client: import from agent_engine_runner_shared.a2a directly
    # Context (per-execution isolation, log correlation)
    "current_execution_id",
    "current_execution_metadata",
    "current_wrapper",
    "current_oe_url",
    "current_request_id",
    "current_trace_id",
    "current_user_id",
    "current_custom_headers",
    "set_execution_context",
    "clear_execution_context",
    "get_current_execution_id",
    "get_current_execution_metadata",
    "get_current_wrapper",
    "get_current_oe_url",
    "get_current_request_id",
    "get_current_trace_id",
    "get_current_user_id",
    "get_current_custom_headers",
    "get_current_payload",
    "request_session_finish",
    "is_session_finish_requested",
    "SessionFinishStatus",
    "record_current_memory_metadata",
    # Models
    "InvokeRequest",
    "InvokeResponse",
    "AgentResumeRequest",
    "AgentResumeResponse",
    "ExecutionStatus",
    "ExecutionStatusResponse",
    "Execution",
    "ExecutionStep",
    "ToolExecuteRequest",
    "ToolExecuteResponse",
    "ToolResultRequest",
    "ToolPodExecuteRequest",
    "ToolPodExecuteResponse",
    "GuardrailCheckDecision",
    "GuardrailCheckEvidence",
    "GuardrailCheckInput",
    "GuardrailCheckRequest",
    "GuardrailCheckResponse",
    "GuardrailRuntimePolicy",
    "GuardrailRuntimeStage",
    "GuardrailPolicyEngine",
    "GuardrailPolicyEngineResult",
    "register_guardrail_policy_engine",
    "ExecutorCallbackRequest",
    "SuspendPayload",
    "StructuredJSONFormatter",
    "LoggingStream",
    "install_structured_logging",
    # Wrapper classes
    "SecureLLMProxy",
    "SecureToolWrapper",
    "LLMInvocationError",
    "PolicyDeniedException",
    "TerminalExecutionError",
    "ToolCallTimeoutError",
    "ToolExecutionError",
    "ExternalAPICallError",
    # Utilities
    "extract_usage",
    # Utilities
    "RuntimeMode",
    "get_env",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "get_request_timeout",
    "get_runtime_mode",
    "setup_logging",
    # Metrics
    "Metrics",
    "record_latency",
    "record_error",
    "log_tool_call",
    "log_tool_result",
    "log_llm_call",
    "log_llm_result",
    "log_execution_event",
]
