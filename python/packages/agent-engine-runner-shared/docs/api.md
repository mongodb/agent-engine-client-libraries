# Table of Contents

* [agent\_engine\_runner\_shared.runtime](#agent_engine_runner_shared.runtime)
  * [TenantRuntime](#agent_engine_runner_shared.runtime.TenantRuntime)
    * [\_\_init\_\_](#agent_engine_runner_shared.runtime.TenantRuntime.__init__)
    * [agent\_config](#agent_engine_runner_shared.runtime.TenantRuntime.agent_config)
    * [memory\_enabled](#agent_engine_runner_shared.runtime.TenantRuntime.memory_enabled)
    * [get\_agent](#agent_engine_runner_shared.runtime.TenantRuntime.get_agent)
    * [warm\_up\_agent](#agent_engine_runner_shared.runtime.TenantRuntime.warm_up_agent)
    * [memory\_engine](#agent_engine_runner_shared.runtime.TenantRuntime.memory_engine)
    * [memory\_writer](#agent_engine_runner_shared.runtime.TenantRuntime.memory_writer)
    * [build\_context](#agent_engine_runner_shared.runtime.TenantRuntime.build_context)
    * [build\_context\_from\_sources](#agent_engine_runner_shared.runtime.TenantRuntime.build_context_from_sources)
    * [write\_turn\_async](#agent_engine_runner_shared.runtime.TenantRuntime.write_turn_async)
    * [save\_semantic](#agent_engine_runner_shared.runtime.TenantRuntime.save_semantic)
    * [search\_semantic](#agent_engine_runner_shared.runtime.TenantRuntime.search_semantic)
    * [save\_custom](#agent_engine_runner_shared.runtime.TenantRuntime.save_custom)
    * [retrieve\_custom](#agent_engine_runner_shared.runtime.TenantRuntime.retrieve_custom)
    * [get\_semantic](#agent_engine_runner_shared.runtime.TenantRuntime.get_semantic)
    * [create\_taxonomic](#agent_engine_runner_shared.runtime.TenantRuntime.create_taxonomic)
    * [search\_taxonomic](#agent_engine_runner_shared.runtime.TenantRuntime.search_taxonomic)
    * [get\_taxonomic\_term](#agent_engine_runner_shared.runtime.TenantRuntime.get_taxonomic_term)
    * [list\_taxonomic\_domains](#agent_engine_runner_shared.runtime.TenantRuntime.list_taxonomic_domains)
    * [save\_episode](#agent_engine_runner_shared.runtime.TenantRuntime.save_episode)
    * [search\_episodes](#agent_engine_runner_shared.runtime.TenantRuntime.search_episodes)
    * [list\_episodes](#agent_engine_runner_shared.runtime.TenantRuntime.list_episodes)
    * [discover\_procedures](#agent_engine_runner_shared.runtime.TenantRuntime.discover_procedures)
    * [get\_procedure](#agent_engine_runner_shared.runtime.TenantRuntime.get_procedure)
    * [save\_procedure](#agent_engine_runner_shared.runtime.TenantRuntime.save_procedure)
    * [get\_current\_user\_id](#agent_engine_runner_shared.runtime.TenantRuntime.get_current_user_id)
    * [has\_active\_guardrail\_rules](#agent_engine_runner_shared.runtime.TenantRuntime.has_active_guardrail_rules)
    * [validate\_output](#agent_engine_runner_shared.runtime.TenantRuntime.validate_output)
    * [a2a](#agent_engine_runner_shared.runtime.TenantRuntime.a2a)
    * [register\_tool](#agent_engine_runner_shared.runtime.TenantRuntime.register_tool)
    * [get\_tool\_metadata](#agent_engine_runner_shared.runtime.TenantRuntime.get_tool_metadata)
    * [register\_query\_plugin](#agent_engine_runner_shared.runtime.TenantRuntime.register_query_plugin)
    * [get\_query\_plugin](#agent_engine_runner_shared.runtime.TenantRuntime.get_query_plugin)
    * [note\_checkpoint\_wire\_workspace\_id](#agent_engine_runner_shared.runtime.TenantRuntime.note_checkpoint_wire_workspace_id)
    * [get\_checkpoint\_workspace\_id](#agent_engine_runner_shared.runtime.TenantRuntime.get_checkpoint_workspace_id)
    * [register\_and\_run](#agent_engine_runner_shared.runtime.TenantRuntime.register_and_run)
* [agent\_engine\_runner\_shared.agent\_config](#agent_engine_runner_shared.agent_config)
  * [AgentFeatureConfig](#agent_engine_runner_shared.agent_config.AgentFeatureConfig)
    * [explicit](#agent_engine_runner_shared.agent_config.AgentFeatureConfig.explicit)
  * [A2ASkillConfig](#agent_engine_runner_shared.agent_config.A2ASkillConfig)
  * [A2AConfig](#agent_engine_runner_shared.agent_config.A2AConfig)
  * [RuntimeMCPAuthConfig](#agent_engine_runner_shared.agent_config.RuntimeMCPAuthConfig)
  * [RuntimeMCPServerConfig](#agent_engine_runner_shared.agent_config.RuntimeMCPServerConfig)
  * [RuntimeMCPConfig](#agent_engine_runner_shared.agent_config.RuntimeMCPConfig)
  * [RuntimeConnectorReference](#agent_engine_runner_shared.agent_config.RuntimeConnectorReference)
  * [RuntimeAgentConfig](#agent_engine_runner_shared.agent_config.RuntimeAgentConfig)
    * [feature\_enabled](#agent_engine_runner_shared.agent_config.RuntimeAgentConfig.feature_enabled)
    * [configured\_feature](#agent_engine_runner_shared.agent_config.RuntimeAgentConfig.configured_feature)
  * [load\_runtime\_agent\_config](#agent_engine_runner_shared.agent_config.load_runtime_agent_config)
* [agent\_engine\_runner\_shared.context](#agent_engine_runner_shared.context)
  * [OwnerUrlFailureState](#agent_engine_runner_shared.context.OwnerUrlFailureState)
  * [\_SessionFinishState](#agent_engine_runner_shared.context._SessionFinishState)
    * [close](#agent_engine_runner_shared.context._SessionFinishState.close)
  * [SessionFinishStatus](#agent_engine_runner_shared.context.SessionFinishStatus)
    * [REQUESTED](#agent_engine_runner_shared.context.SessionFinishStatus.REQUESTED)
    * [ALREADY\_REQUESTED](#agent_engine_runner_shared.context.SessionFinishStatus.ALREADY_REQUESTED)
    * [UNAVAILABLE](#agent_engine_runner_shared.context.SessionFinishStatus.UNAVAILABLE)
  * [set\_execution\_context](#agent_engine_runner_shared.context.set_execution_context)
  * [clear\_execution\_context](#agent_engine_runner_shared.context.clear_execution_context)
  * [get\_current\_wrapper](#agent_engine_runner_shared.context.get_current_wrapper)
  * [get\_current\_execution\_id](#agent_engine_runner_shared.context.get_current_execution_id)
  * [get\_current\_request\_id](#agent_engine_runner_shared.context.get_current_request_id)
  * [get\_current\_trace\_id](#agent_engine_runner_shared.context.get_current_trace_id)
  * [get\_current\_oe\_url](#agent_engine_runner_shared.context.get_current_oe_url)
  * [get\_current\_oe\_owner\_url](#agent_engine_runner_shared.context.get_current_oe_owner_url)
  * [report\_oe\_owner\_url\_failure](#agent_engine_runner_shared.context.report_oe_owner_url_failure)
  * [get\_current\_user\_id](#agent_engine_runner_shared.context.get_current_user_id)
  * [get\_current\_session\_id](#agent_engine_runner_shared.context.get_current_session_id)
  * [get\_current\_workspace\_id](#agent_engine_runner_shared.context.get_current_workspace_id)
  * [get\_current\_authorization](#agent_engine_runner_shared.context.get_current_authorization)
  * [get\_current\_custom\_headers](#agent_engine_runner_shared.context.get_current_custom_headers)
  * [get\_all\_custom\_headers](#agent_engine_runner_shared.context.get_all_custom_headers)
  * [get\_current\_execution\_metadata](#agent_engine_runner_shared.context.get_current_execution_metadata)
  * [get\_current\_payload](#agent_engine_runner_shared.context.get_current_payload)
  * [request\_session\_finish](#agent_engine_runner_shared.context.request_session_finish)
  * [is\_session\_finish\_requested](#agent_engine_runner_shared.context.is_session_finish_requested)
  * [close\_session\_finish\_latch](#agent_engine_runner_shared.context.close_session_finish_latch)
  * [record\_suspend\_request](#agent_engine_runner_shared.context.record_suspend_request)
  * [get\_requested\_suspend](#agent_engine_runner_shared.context.get_requested_suspend)
  * [local\_suspend\_request\_context](#agent_engine_runner_shared.context.local_suspend_request_context)
  * [record\_current\_memory\_metadata](#agent_engine_runner_shared.context.record_current_memory_metadata)
  * [ValidationContext](#agent_engine_runner_shared.context.ValidationContext)
    * [to\_dict](#agent_engine_runner_shared.context.ValidationContext.to_dict)
  * [get\_validation\_context](#agent_engine_runner_shared.context.get_validation_context)
  * [get\_current\_log\_origin](#agent_engine_runner_shared.context.get_current_log_origin)
  * [customer\_origin\_scope](#agent_engine_runner_shared.context.customer_origin_scope)
* [agent\_engine\_runner\_shared.models](#agent_engine_runner_shared.models)
  * [ExecutionStatus](#agent_engine_runner_shared.models.ExecutionStatus)
  * [SuspendPayload](#agent_engine_runner_shared.models.SuspendPayload)
    * [to\_json](#agent_engine_runner_shared.models.SuspendPayload.to_json)
  * [PendingInterrupt](#agent_engine_runner_shared.models.PendingInterrupt)
  * [InterruptResult](#agent_engine_runner_shared.models.InterruptResult)
  * [StreamingResult](#agent_engine_runner_shared.models.StreamingResult)
  * [InvokeRequest](#agent_engine_runner_shared.models.InvokeRequest)
  * [InvokeResponse](#agent_engine_runner_shared.models.InvokeResponse)
  * [ExecuteRequest](#agent_engine_runner_shared.models.ExecuteRequest)
  * [ToolExecuteRequest](#agent_engine_runner_shared.models.ToolExecuteRequest)
    * [to\_start\_log](#agent_engine_runner_shared.models.ToolExecuteRequest.to_start_log)
    * [to\_cached\_log](#agent_engine_runner_shared.models.ToolExecuteRequest.to_cached_log)
  * [ElicitationInfo](#agent_engine_runner_shared.models.ElicitationInfo)
  * [GuardrailMeta](#agent_engine_runner_shared.models.GuardrailMeta)
  * [ToolExecuteResponse](#agent_engine_runner_shared.models.ToolExecuteResponse)
  * [ToolResultRequest](#agent_engine_runner_shared.models.ToolResultRequest)
    * [to\_log](#agent_engine_runner_shared.models.ToolResultRequest.to_log)
  * [ToolAuthorization](#agent_engine_runner_shared.models.ToolAuthorization)
  * [ToolPodExecuteRequest](#agent_engine_runner_shared.models.ToolPodExecuteRequest)
    * [validate\_argument\_size](#agent_engine_runner_shared.models.ToolPodExecuteRequest.validate_argument_size)
  * [ToolPodExecuteResponse](#agent_engine_runner_shared.models.ToolPodExecuteResponse)
  * [GuardrailRuntimeStage](#agent_engine_runner_shared.models.GuardrailRuntimeStage)
  * [GuardrailCheckDecision](#agent_engine_runner_shared.models.GuardrailCheckDecision)
  * [GuardrailRuntimePolicy](#agent_engine_runner_shared.models.GuardrailRuntimePolicy)
  * [GuardrailCheckInput](#agent_engine_runner_shared.models.GuardrailCheckInput)
  * [GuardrailCheckContext](#agent_engine_runner_shared.models.GuardrailCheckContext)
  * [GuardrailCheckRequest](#agent_engine_runner_shared.models.GuardrailCheckRequest)
  * [GuardrailCheckEvidence](#agent_engine_runner_shared.models.GuardrailCheckEvidence)
  * [GuardrailCheckResponse](#agent_engine_runner_shared.models.GuardrailCheckResponse)
  * [InvokeLLMRequestArguments](#agent_engine_runner_shared.models.InvokeLLMRequestArguments)
  * [LLMPodInvokeRequest](#agent_engine_runner_shared.models.LLMPodInvokeRequest)
  * [coerce\_token\_usage](#agent_engine_runner_shared.models.coerce_token_usage)
  * [json\_safe\_metadata](#agent_engine_runner_shared.models.json_safe_metadata)
  * [add\_token\_usage](#agent_engine_runner_shared.models.add_token_usage)
  * [accumulate\_stream\_usage](#agent_engine_runner_shared.models.accumulate_stream_usage)
  * [merge\_token\_usage](#agent_engine_runner_shared.models.merge_token_usage)
  * [LLMResult](#agent_engine_runner_shared.models.LLMResult)
    * [from\_response](#agent_engine_runner_shared.models.LLMResult.from_response)
    * [extract\_usage](#agent_engine_runner_shared.models.LLMResult.extract_usage)
    * [to\_response](#agent_engine_runner_shared.models.LLMResult.to_response)
  * [LLMPodInvokeResponse](#agent_engine_runner_shared.models.LLMPodInvokeResponse)
  * [LLMPodStreamEvent](#agent_engine_runner_shared.models.LLMPodStreamEvent)
  * [ExecutorCallbackRequest](#agent_engine_runner_shared.models.ExecutorCallbackRequest)
  * [HumanReviewData](#agent_engine_runner_shared.models.HumanReviewData)
  * [AgentResumeRequest](#agent_engine_runner_shared.models.AgentResumeRequest)
  * [AgentResumeResponse](#agent_engine_runner_shared.models.AgentResumeResponse)
  * [ExecutionStatusResponse](#agent_engine_runner_shared.models.ExecutionStatusResponse)
  * [Execution](#agent_engine_runner_shared.models.Execution)
    * [to\_persistence\_doc](#agent_engine_runner_shared.models.Execution.to_persistence_doc)
    * [from\_persistence\_doc](#agent_engine_runner_shared.models.Execution.from_persistence_doc)
  * [ExecutionStep](#agent_engine_runner_shared.models.ExecutionStep)
    * [from\_log\_doc](#agent_engine_runner_shared.models.ExecutionStep.from_log_doc)
  * [HealthStatus](#agent_engine_runner_shared.models.HealthStatus)
  * [HealthResponse](#agent_engine_runner_shared.models.HealthResponse)
  * [ToolDefinition](#agent_engine_runner_shared.models.ToolDefinition)
  * [AERExecuteResponse](#agent_engine_runner_shared.models.AERExecuteResponse)
  * [ToolsListResponse](#agent_engine_runner_shared.models.ToolsListResponse)
  * [StreamChunk](#agent_engine_runner_shared.models.StreamChunk)
  * [AgentStartStreamRequest](#agent_engine_runner_shared.models.AgentStartStreamRequest)
  * [ExecutionLogsQueryResponse](#agent_engine_runner_shared.models.ExecutionLogsQueryResponse)
  * [NodeExecutionsQueryResponse](#agent_engine_runner_shared.models.NodeExecutionsQueryResponse)
  * [CostSummary](#agent_engine_runner_shared.models.CostSummary)
  * [CostByWorkspace](#agent_engine_runner_shared.models.CostByWorkspace)
  * [CostByModel](#agent_engine_runner_shared.models.CostByModel)
  * [DailyCostEntry](#agent_engine_runner_shared.models.DailyCostEntry)
  * [CostDashboardResponse](#agent_engine_runner_shared.models.CostDashboardResponse)
  * [ExecutionDocument](#agent_engine_runner_shared.models.ExecutionDocument)
  * [ExecutionsListQueryResponse](#agent_engine_runner_shared.models.ExecutionsListQueryResponse)
  * [ExecutionDetailQueryResponse](#agent_engine_runner_shared.models.ExecutionDetailQueryResponse)
  * [NodeExecutionRequest](#agent_engine_runner_shared.models.NodeExecutionRequest)
    * [to\_log](#agent_engine_runner_shared.models.NodeExecutionRequest.to_log)
* [agent\_engine\_runner\_shared.progress](#agent_engine_runner_shared.progress)
  * [emit](#agent_engine_runner_shared.progress.emit)
  * [emit\_step](#agent_engine_runner_shared.progress.emit_step)
* [agent\_engine\_runner\_shared.custom\_events](#agent_engine_runner_shared.custom_events)
  * [CustomEventTransport](#agent_engine_runner_shared.custom_events.CustomEventTransport)
    * [emit\_sync](#agent_engine_runner_shared.custom_events.CustomEventTransport.emit_sync)
    * [emit](#agent_engine_runner_shared.custom_events.CustomEventTransport.emit)
  * [get\_custom\_event\_transport](#agent_engine_runner_shared.custom_events.get_custom_event_transport)
  * [set\_custom\_event\_transport](#agent_engine_runner_shared.custom_events.set_custom_event_transport)
  * [clear\_custom\_event\_transport](#agent_engine_runner_shared.custom_events.clear_custom_event_transport)
  * [custom\_event\_transport](#agent_engine_runner_shared.custom_events.custom_event_transport)
  * [emit\_custom\_event\_sync](#agent_engine_runner_shared.custom_events.emit_custom_event_sync)
  * [emit\_custom\_event](#agent_engine_runner_shared.custom_events.emit_custom_event)
  * [OEHTTPCustomEventTransport](#agent_engine_runner_shared.custom_events.OEHTTPCustomEventTransport)
  * [FeatureOffCustomEventTransport](#agent_engine_runner_shared.custom_events.FeatureOffCustomEventTransport)
  * [install\_tool\_custom\_event\_transport](#agent_engine_runner_shared.custom_events.install_tool_custom_event_transport)
* [agent\_engine\_runner\_shared.guardrails\_evaluator.core](#agent_engine_runner_shared.guardrails_evaluator.core)
  * [GuardrailPolicyEngineResult](#agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngineResult)
  * [GuardrailPolicyEngine](#agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngine)
    * [evaluate](#agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngine.evaluate)
  * [register\_guardrail\_policy\_engine](#agent_engine_runner_shared.guardrails_evaluator.core.register_guardrail_policy_engine)
  * [evaluate\_guardrail\_check](#agent_engine_runner_shared.guardrails_evaluator.core.evaluate_guardrail_check)
* [agent\_engine\_runner\_shared.secure\_llm\_proxy](#agent_engine_runner_shared.secure_llm_proxy)
  * [SecureLLMProxy](#agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy)
    * [step\_counter](#agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.step_counter)
    * [invoke](#agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.invoke)
    * [response\_from\_stream\_chunks](#agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.response_from_stream_chunks)
    * [stream](#agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.stream)
* [agent\_engine\_runner\_shared.secure\_wrapper](#agent_engine_runner_shared.secure_wrapper)
  * [extract\_usage](#agent_engine_runner_shared.secure_wrapper.extract_usage)
  * [PolicyDeniedException](#agent_engine_runner_shared.secure_wrapper.PolicyDeniedException)
  * [OERetryAfterError](#agent_engine_runner_shared.secure_wrapper.OERetryAfterError)
  * [ToolExecutionError](#agent_engine_runner_shared.secure_wrapper.ToolExecutionError)
  * [TerminalExecutionError](#agent_engine_runner_shared.secure_wrapper.TerminalExecutionError)
  * [raise\_for\_oe\_rejection](#agent_engine_runner_shared.secure_wrapper.raise_for_oe_rejection)
  * [ToolCallTimeoutError](#agent_engine_runner_shared.secure_wrapper.ToolCallTimeoutError)
  * [OperationalStepAllocator](#agent_engine_runner_shared.secure_wrapper.OperationalStepAllocator)
  * [ExternalAPICallError](#agent_engine_runner_shared.secure_wrapper.ExternalAPICallError)
  * [LLMInvocationError](#agent_engine_runner_shared.secure_wrapper.LLMInvocationError)
  * [request\_oe\_approval](#agent_engine_runner_shared.secure_wrapper.request_oe_approval)
  * [request\_oe\_approval\_retryable](#agent_engine_runner_shared.secure_wrapper.request_oe_approval_retryable)
  * [report\_oe\_result](#agent_engine_runner_shared.secure_wrapper.report_oe_result)
  * [SecureToolWrapper](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper)
    * [\_\_init\_\_](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.__init__)
    * [step\_counter](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.step_counter)
    * [next\_operational\_step](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.next_operational_step)
    * [observe\_operational\_step](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.observe_operational_step)
    * [workflow](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.workflow)
    * [close](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.close)
    * [execute\_tool](#agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.execute_tool)
  * [create\_secure\_tool\_function](#agent_engine_runner_shared.secure_wrapper.create_secure_tool_function)
* [agent\_engine\_runner\_shared.structured\_logging](#agent_engine_runner_shared.structured_logging)
  * [StructuredJSONFormatter](#agent_engine_runner_shared.structured_logging.StructuredJSONFormatter)
  * [LoggingStream](#agent_engine_runner_shared.structured_logging.LoggingStream)
  * [install\_structured\_logging](#agent_engine_runner_shared.structured_logging.install_structured_logging)
* [agent\_engine\_runner\_shared.metrics](#agent_engine_runner_shared.metrics)
  * [MetricPoint](#agent_engine_runner_shared.metrics.MetricPoint)
  * [LatencyStats](#agent_engine_runner_shared.metrics.LatencyStats)
    * [record](#agent_engine_runner_shared.metrics.LatencyStats.record)
    * [avg\_ms](#agent_engine_runner_shared.metrics.LatencyStats.avg_ms)
    * [to\_dict](#agent_engine_runner_shared.metrics.LatencyStats.to_dict)
  * [Metrics](#agent_engine_runner_shared.metrics.Metrics)
    * [record\_latency](#agent_engine_runner_shared.metrics.Metrics.record_latency)
    * [record\_error](#agent_engine_runner_shared.metrics.Metrics.record_error)
    * [get\_all](#agent_engine_runner_shared.metrics.Metrics.get_all)
    * [reset](#agent_engine_runner_shared.metrics.Metrics.reset)
  * [record\_latency](#agent_engine_runner_shared.metrics.record_latency)
  * [record\_error](#agent_engine_runner_shared.metrics.record_error)
  * [with\_metrics](#agent_engine_runner_shared.metrics.with_metrics)
  * [log\_tool\_call](#agent_engine_runner_shared.metrics.log_tool_call)
  * [log\_tool\_result](#agent_engine_runner_shared.metrics.log_tool_result)
  * [log\_llm\_call](#agent_engine_runner_shared.metrics.log_llm_call)
  * [log\_llm\_result](#agent_engine_runner_shared.metrics.log_llm_result)
  * [log\_execution\_event](#agent_engine_runner_shared.metrics.log_execution_event)
* [agent\_engine\_runner\_shared.utils](#agent_engine_runner_shared.utils)
  * [normalize\_content](#agent_engine_runner_shared.utils.normalize_content)
  * [normalize\_tool\_call\_args](#agent_engine_runner_shared.utils.normalize_tool_call_args)
  * [normalize\_optional\_str](#agent_engine_runner_shared.utils.normalize_optional_str)
  * [strip\_thinking](#agent_engine_runner_shared.utils.strip_thinking)
  * [filter\_thinking\_tokens](#agent_engine_runner_shared.utils.filter_thinking_tokens)
  * [RuntimeMode](#agent_engine_runner_shared.utils.RuntimeMode)
  * [get\_runtime\_mode](#agent_engine_runner_shared.utils.get_runtime_mode)
  * [get\_env](#agent_engine_runner_shared.utils.get_env)
  * [get\_env\_int](#agent_engine_runner_shared.utils.get_env_int)
  * [get\_env\_float](#agent_engine_runner_shared.utils.get_env_float)
  * [get\_env\_bool](#agent_engine_runner_shared.utils.get_env_bool)
  * [is\_platform\_env\_var](#agent_engine_runner_shared.utils.is_platform_env_var)
  * [tenant\_env\_vars](#agent_engine_runner_shared.utils.tenant_env_vars)
  * [LLM\_INITIAL\_BACKOFF](#agent_engine_runner_shared.utils.LLM_INITIAL_BACKOFF)
  * [LLM\_MAX\_BACKOFF](#agent_engine_runner_shared.utils.LLM_MAX_BACKOFF)
  * [LLM\_READ\_TIMEOUT](#agent_engine_runner_shared.utils.LLM_READ_TIMEOUT)
  * [TOOL\_READ\_TIMEOUT](#agent_engine_runner_shared.utils.TOOL_READ_TIMEOUT)
  * [oe\_stream\_retry\_delay\_s](#agent_engine_runner_shared.utils.oe_stream_retry_delay_s)
  * [sleep\_oe\_stream\_retry](#agent_engine_runner_shared.utils.sleep_oe_stream_retry)
  * [is\_retryable\_error](#agent_engine_runner_shared.utils.is_retryable_error)
  * [format\_llm\_error](#agent_engine_runner_shared.utils.format_llm_error)
  * [is\_llm\_credential\_rejection](#agent_engine_runner_shared.utils.is_llm_credential_rejection)
  * [get\_request\_timeout](#agent_engine_runner_shared.utils.get_request_timeout)
  * [setup\_logging](#agent_engine_runner_shared.utils.setup_logging)
  * [log\_separator](#agent_engine_runner_shared.utils.log_separator)
  * [log\_section](#agent_engine_runner_shared.utils.log_section)
  * [log\_llm\_messages](#agent_engine_runner_shared.utils.log_llm_messages)
  * [log\_llm\_response](#agent_engine_runner_shared.utils.log_llm_response)
  * [log\_tool\_request](#agent_engine_runner_shared.utils.log_tool_request)
  * [log\_tool\_result](#agent_engine_runner_shared.utils.log_tool_result)
  * [log\_cached\_result](#agent_engine_runner_shared.utils.log_cached_result)
  * [log\_policy\_blocked](#agent_engine_runner_shared.utils.log_policy_blocked)
  * [log\_execution\_start](#agent_engine_runner_shared.utils.log_execution_start)
  * [log\_execution\_callback](#agent_engine_runner_shared.utils.log_execution_callback)

TenantRuntime — framework-agnostic platform runtime for tenant agent applications.

The same application code runs in all supported modes:
- aer: Agent execution with secure wrappers
- tool: Tool function execution

Usage (via a framework SDK, e.g. agent-engine-sdk-langgraph):
    The framework SDK's ``App`` class wraps ``TenantRuntime`` and registers
    hooks so agent-engine-runner-shared can call framework code without importing it.
    See ``agent_engine_runner_shared.hooks`` for the hook registry.

<a id="agent_engine_runner_shared.runtime.TenantRuntime"></a>

## TenantRuntime

```python
class TenantRuntime()
```

Tenant Runtime SDK.

The runtime discovers its role from RUNNER_MODE environment variable:

- aer: Full LangGraph execution with SecureToolWrapper
- tool: Tool function execution

<a id="agent_engine_runner_shared.runtime.TenantRuntime.__init__"></a>

#### \_\_init\_\_

```python
def __init__(app_name: str = "Agent",
             app_version: str = "1.0.0",
             mongodb_uri: Optional[str] = None,
             database_name: Optional[str] = None,
             enable_tracing: bool | None = None,
             traces_collection_name: str = "traces",
             enable_memory: bool | None = None,
             org_id: Optional[str] = None,
             project_id: Optional[str] = None)
```

Initialize the runner runtime.

**Arguments**:

- `app_name` - Application name
- `app_version` - Application version
- `mongodb_uri` - MongoDB URI (for OE execution logging)
- `database_name` - Database name
- `enable_tracing` - Deprecated. Tracing is always enabled.
- `traces_collection_name` - Collection name for trace storage
- `enable_memory` - Deprecated. Use agent.yaml features.memory instead.
- `org_id` - Deprecated and ignored. The org is taken from the
  ``ORG_ID`` environment variable, which the platform injects.
  Passing this argument raises a DeprecationWarning and will be
  removed in a future release.
- `project_id` - Deprecated and ignored. The project is taken from the
  ``PROJECT_ID`` environment variable, which the platform injects.
  Passing this argument raises a DeprecationWarning and will be
  removed in a future release.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.agent_config"></a>

#### agent\_config

```python
@property
def agent_config() -> RuntimeAgentConfig
```

Return the parsed runtime view of agent.yaml.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.memory_enabled"></a>

#### memory\_enabled

```python
@property
def memory_enabled() -> bool
```

Return the resolved agent.yaml Memory feature setting.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_agent"></a>

#### get\_agent

```python
def get_agent(callbacks: List[Any] | None = None) -> BaseAgent
```

Get a BaseAgent instance from the registered App.

Delegates to the App's get_agent() method, which handles
framework-specific graph building and callback adaptation.

**Arguments**:

- `callbacks` - Framework-neutral callbacks for observability
  (e.g., NodeExecutionLogger)


**Returns**:

  BaseAgent instance


**Raises**:

- `RuntimeError` - If no App has been registered via register_and_run()

<a id="agent_engine_runner_shared.runtime.TenantRuntime.warm_up_agent"></a>

#### warm\_up\_agent

```python
def warm_up_agent() -> bool
```

Best-effort pre-build of the agent graph before the first /execute.

Delegates to the framework App's warm_up() if it declares one;
frameworks that don't opt in keep today's lazy build on first
get_agent(). A ``False`` result reports that the one-shot optimization
did not complete; ``None`` from an older or third-party implementation
remains a successful result for compatibility.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.memory_engine"></a>

#### memory\_engine

```python
@property
def memory_engine() -> Optional[Any]
```

Get the MemoryEngine instance if configured.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.memory_writer"></a>

#### memory\_writer

```python
@property
def memory_writer() -> Optional[Any]
```

Get the MemoryWriter instance if configured.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.build_context"></a>

#### build\_context

```python
def build_context(query: str,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  visibility: Optional[str] = None,
                  metadata_filter: Optional[Dict[str, Any]] = None,
                  enabled_sources: Optional[set[str]] = None,
                  max_tokens: Optional[int] = None) -> str
```

Build memory context for a query.

Works with both MemoryEngine (direct) and MemoryClient (HTTP) backends.
Backend choice is controlled by USE_MEMORY_CLIENT environment variable
at initialization time (_setup_memory).

**Arguments**:

- `query` - The user's query/message
- `user_id` - User ID (required for memory isolation)
- `session_id` - Session ID
- `visibility` - Visibility scope filter (private, shared, org)
- `max_tokens` - Optional gross context-construction budget. After
  retrieval and ranking, the server subtracts a 500-token
  formatting reserve, then greedily selects whole memory chunks
  that fit in the remainder. Positive values at or below 500
  leave no budget for memories. Values above 500 can still yield
  empty context when no chunk fits. Omit to use the server
  default.


**Returns**:

  Formatted context string, or empty string if memory is disabled

<a id="agent_engine_runner_shared.runtime.TenantRuntime.build_context_from_sources"></a>

#### build\_context\_from\_sources

```python
def build_context_from_sources(
        query: str,
        sources: "list[SourceSpec]",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        visibility: Optional[str] = None,
        rerank: bool = False,
        max_tokens: Optional[int] = None) -> "ContextResponse"
```

Build memory context from an explicit, per-source-configured source set.

The per-source counterpart to :meth:`build_context`. Where that method
applies one filter and mode to every source and returns a flattened
string, this preserves the full ``ContextResponse`` so the per-source
metadata (``ranking_strategy``, ``source_outcomes``) reaches the caller.

Tenancy (``org_id``, ``project_id``) is stamped from the runtime; the
caller passes none of it. Unlike :meth:`build_context`, each source's
``top_k`` is honored, since it is the caller's explicit per-source intent.
``session_id`` is required only when the ``stm`` source is requested;
omitting it for any other source set is fine.

Returns an empty ``ContextResponse`` when memory is unavailable or a
required identity field is unresolved, degrading like
:meth:`build_context` rather than raising into the agent turn.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.write_turn_async"></a>

#### write\_turn\_async

```python
def write_turn_async(message: str,
                     result_messages: List[Any],
                     user_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     include_user_turn: bool = True,
                     metadata: Optional[Dict[str, Any]] = None,
                     idempotency_key: Optional[str] = None) -> None
```

Write turn to memory asynchronously (non-blocking).

**Arguments**:

- `message` - Original user message
- `result_messages` - All messages from graph execution
- `user_id` - User ID (defaults to current execution context user)
- `session_id` - Session ID
- `include_user_turn` - Record the user message (default True). Pass False
  on resume legs, where there is no new user prompt and the user
  turn was already written when the turn first started.
- `metadata` - Arbitrary key-value metadata stamped on every turn written.
- `idempotency_key` - Deduplication key for single-turn writes; ignored
  for multi-message writes (see :func:`write_turn_to_memory`).

<a id="agent_engine_runner_shared.runtime.TenantRuntime.save_semantic"></a>

#### save\_semantic

```python
def save_semantic(text: str,
                  label: str,
                  user_id: Optional[str] = None,
                  source: str = "agent",
                  visibility: str = "private",
                  metadata: Optional[Dict[str, Any]] = None,
                  upsert: bool = True,
                  agent_id: Optional[str] = None) -> bool
```

Save a semantic memory to the memory engine.

Handles embedding generation and storage automatically.

**Arguments**:

- `text` - The memory content to store
- `label` - A unique label/identifier for this memory
- `user_id` - User ID (required)
- `source` - Source of the memory (default: "agent")
- `visibility` - Memory visibility (private, shared, org)
- `metadata` - Optional caller-supplied metadata dict stored with the memory
- `upsert` - If True, update existing memory with same label (default: True)


**Returns**:

  True if saved successfully, False otherwise

<a id="agent_engine_runner_shared.runtime.TenantRuntime.search_semantic"></a>

#### search\_semantic

```python
def search_semantic(query: str,
                    user_id: Optional[str] = None,
                    visibility: Optional[str] = None,
                    top_k: int = 10) -> List[Dict[str, Any]]
```

Search semantic memories by query.

**Arguments**:

- `query` - Search query text
- `user_id` - Optional user ID filter
- `top_k` - Maximum number of results


**Returns**:

  List of matching memories with label, text, and metadata

<a id="agent_engine_runner_shared.runtime.TenantRuntime.save_custom"></a>

#### save\_custom

```python
def save_custom(
    memory_type: str,
    content: str,
    tags: Optional[Dict[str, Any]] = None,
    contextual_metadata: Optional[Dict[str, Any]] = None
) -> "CustomMemorySaveResult"
```

Save a memory of a declared custom type.

Identity is stamped by the platform; the OE memory proxy forwards the
request to the project's memory server.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.retrieve_custom"></a>

#### retrieve\_custom

```python
def retrieve_custom(memory_type: str,
                    query: str,
                    tags: Optional[Dict[str, Any]] = None,
                    top_k: int = 10) -> "CustomMemoryRetrieveResult"
```

Retrieve memories of a declared custom type by semantic query.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_semantic"></a>

#### get\_semantic

```python
def get_semantic(label: str,
                 user_id: Optional[str] = None,
                 visibility: Optional[str] = None) -> Optional[Dict[str, Any]]
```

Get a specific semantic memory by label.

**Arguments**:

- `label` - Memory label


**Returns**:

  Dict with memory details, or None if not found

<a id="agent_engine_runner_shared.runtime.TenantRuntime.create_taxonomic"></a>

#### create\_taxonomic

```python
def create_taxonomic(
        domain: str,
        term: str,
        definition: str,
        related_terms: Optional[List[str]] = None,
        visibility: str = "org",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> Optional[str]
```

Create a taxonomic memory entry (domain-specific term with definition).

Taxonomic memories are used for:
- Domain terminology definitions (e.g., "deductible", "premium")
- Query expansion (improving search relevance)
- Semantic linking via related terms

**Arguments**:

- `domain` - Domain/category for this term (e.g., "insurance", "coverage_types")
- `term` - The term being defined (e.g., "deductible")
- `definition` - Definition of the term
- `related_terms` - Related terms for semantic linking
- `visibility` - Visibility scope (default: "org" for organization-wide)
- `user_id` - User ID (defaults to current execution context user)
- `metadata` - Arbitrary key-value metadata stored on the entry


**Returns**:

  Created document ID, or None if creation fails

<a id="agent_engine_runner_shared.runtime.TenantRuntime.search_taxonomic"></a>

#### search\_taxonomic

```python
def search_taxonomic(query: str,
                     user_id: Optional[str] = None,
                     domain: Optional[str] = None,
                     visibility: Optional[str] = None,
                     top_k: int = 10) -> List[Dict[str, Any]]
```

Search taxonomic memories by query.

**Arguments**:

- `query` - Search query text (e.g., "what is a deductible")
- `domain` - Optional domain filter (e.g., "insurance")
- `top_k` - Maximum number of results


**Returns**:

  List of matching terms with term, definition, and related_terms

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_taxonomic_term"></a>

#### get\_taxonomic\_term

```python
def get_taxonomic_term(
        domain: str,
        term: str,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None) -> Optional[Dict[str, Any]]
```

Get a specific taxonomic term by domain and term name.

**Arguments**:

- `domain` - Domain name (e.g., "insurance")
- `term` - Term name (e.g., "deductible")


**Returns**:

  Dict with term details, or None if not found

<a id="agent_engine_runner_shared.runtime.TenantRuntime.list_taxonomic_domains"></a>

#### list\_taxonomic\_domains

```python
def list_taxonomic_domains(visibility: Optional[str] = None) -> List[str]
```

List all distinct taxonomic domains for the organization.

**Returns**:

  List of domain names (e.g., ["insurance", "coverage_types", "discounts"])

<a id="agent_engine_runner_shared.runtime.TenantRuntime.save_episode"></a>

#### save\_episode

```python
def save_episode(title: str,
                 content: str,
                 user_id: Optional[str] = None,
                 summary: Optional[str] = None,
                 session_id: Optional[str] = None,
                 participants: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None,
                 visibility: str = "private",
                 agent_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]
```

Save an episodic memory (conversation summary or significant event).

Episodic memories are used for:
- Storing conversation summaries for future recall
- Recording significant events/interactions
- Enabling context retrieval across sessions ("last time we discussed...")

**Arguments**:

- `title` - Title/label for this episode (e.g., "Customer inquiry about coverage")
- `content` - Full content of the episode
- `user_id` - User ID who owns this memory (required)
- `summary` - Brief summary for search/display (defaults to title)
- `session_id` - Session ID this episode originated from
- `participants` - List of participants (e.g., ["Customer", "Alex"])
- `tags` - Tags for categorization (e.g., ["quote", "auto_insurance"])
- `visibility` - Visibility scope (default: "private")
- `metadata` - Optional caller-supplied metadata dictionary stored with
  the episode (mirrors save_semantic's metadata)


**Returns**:

  Created document ID, or None if creation fails

<a id="agent_engine_runner_shared.runtime.TenantRuntime.search_episodes"></a>

#### search\_episodes

```python
def search_episodes(query: str,
                    user_id: Optional[str] = None,
                    visibility: Optional[str] = None,
                    session_id: Optional[str] = None,
                    top_k: int = 10) -> List[Dict[str, Any]]
```

Search episodic memories by query.

**Arguments**:

- `query` - Search query text (e.g., "teenage driver discussion")
- `user_id` - Optional user ID filter
- `session_id` - Optional session ID filter
- `top_k` - Maximum number of results


**Returns**:

  List of matching episodes with title, content, summary

<a id="agent_engine_runner_shared.runtime.TenantRuntime.list_episodes"></a>

#### list\_episodes

```python
def list_episodes(user_id: Optional[str] = None,
                  visibility: Optional[str] = None,
                  session_id: Optional[str] = None,
                  limit: int = 20) -> List[Dict[str, Any]]
```

List episodic memories with optional filters.

**Arguments**:

- `user_id` - Optional user ID filter
- `session_id` - Optional session ID filter
- `limit` - Maximum number of results


**Returns**:

  List of episodic memory entries

<a id="agent_engine_runner_shared.runtime.TenantRuntime.discover_procedures"></a>

#### discover\_procedures

```python
def discover_procedures(
        query: str,
        user_id: Optional[str] = None,
        *,
        visibility: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str,
                                       Any]] = None) -> List[Dict[str, Any]]
```

Discover procedures matching a query (lightweight results).

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_procedure"></a>

#### get\_procedure

```python
def get_procedure(procedure_name: str,
                  *,
                  user_id: Optional[str] = None,
                  visibility: Optional[str] = None,
                  include_deleted: bool = False) -> Optional[Dict[str, Any]]
```

Get a full procedural memory by procedure name.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.save_procedure"></a>

#### save\_procedure

```python
def save_procedure(*,
                   procedure: str,
                   description: str,
                   content: str,
                   user_id: Optional[str] = None,
                   steps: Optional[List[Dict[str, Any]]] = None,
                   resources: Optional[List[Dict[str, Any]]] = None,
                   allowed_tools: Optional[List[str]] = None,
                   compatibility: Optional[str] = None,
                   license: Optional[str] = None,
                   trigger_conditions: Optional[List[str]] = None,
                   tags: Optional[List[str]] = None,
                   visibility: str = "private",
                   agent_id: Optional[str] = None,
                   extraction_source: Optional[str] = None,
                   source_format: Optional[str] = None,
                   source_path: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   update_existing: bool = False) -> Optional[Dict[str, Any]]
```

Create or update a procedural memory.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_current_user_id"></a>

#### get\_current\_user\_id

```python
def get_current_user_id() -> Optional[str]
```

Get the current user_id from execution context.

**Returns**:

  User ID from the current execution context, or None if not available

<a id="agent_engine_runner_shared.runtime.TenantRuntime.has_active_guardrail_rules"></a>

#### has\_active\_guardrail\_rules

```python
def has_active_guardrail_rules() -> bool
```

Always returns False. Guardrails enforcement is handled by OE.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.validate_output"></a>

#### validate\_output

```python
def validate_output(text: str) -> str
```

Pass-through. Guardrails enforcement is handled by OE.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.a2a"></a>

#### a2a

```python
@property
def a2a() -> Optional[Any]
```

Get an AgentToAgent client for calling other agents.

Lazily creates the client using the OE HTTP URL and A2A token
from the current execution context. Returns None when not in AER mode
or when the OE did not provide an A2A token.

The client is cached per OE URL — if the URL changes between requests
the client is recreated.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.register_tool"></a>

#### register\_tool

```python
def register_tool(name: str, func: Callable, metadata: Dict[str, Any]) -> None
```

Register a raw tool function and its metadata.

This stores the function for Tool Pod execution and the metadata
for routing decisions. It does NOT create any framework-specific
tool objects — that is the responsibility of the framework SDK.

**Arguments**:

- `name` - Tool name
- `func` - Raw Python function
- `metadata` - Tool metadata (is_local, network, timeout, etc.)

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_tool_metadata"></a>

#### get\_tool\_metadata

```python
def get_tool_metadata(name: str) -> Dict[str, Any]
```

Get metadata for a registered tool.

**Arguments**:

- `name` - Tool name


**Returns**:

  Tool metadata dict (is_local, network, timeout, etc.)
  or empty dict if tool not found.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.register_query_plugin"></a>

#### register\_query\_plugin

```python
def register_query_plugin(plugin: Any) -> None
```

Register the framework adapter's :class:`AERQueryPlugin`.

Must be called before the AER server starts — i.e. during framework
``App`` initialization, before :meth:`register_and_run`, while
``mode == RuntimeMode.AER``. Calling more than once replaces the
previous registration. When no plugin is registered, the AER's
``/query/sessions*`` routes return 501.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_query_plugin"></a>

#### get\_query\_plugin

```python
def get_query_plugin() -> Optional[Any]
```

Return the registered :class:`AERQueryPlugin`, or ``None`` if
none has been registered. Used by the AER routes to decide whether
to delegate (plugin present) or surface 501 (plugin absent).

<a id="agent_engine_runner_shared.runtime.TenantRuntime.note_checkpoint_wire_workspace_id"></a>

#### note\_checkpoint\_wire\_workspace\_id

```python
def note_checkpoint_wire_workspace_id(wire_workspace_id: str | None) -> None
```

Remember the wire ``workspace_id`` from ``/execute`` for query reads.

Query routes carry no workspace identifier; when ``APP_ID`` is unset
the read path falls back to the most recently observed wire value so
it scopes checkpoints the same way the write path does.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.get_checkpoint_workspace_id"></a>

#### get\_checkpoint\_workspace\_id

```python
def get_checkpoint_workspace_id() -> str
```

Resolved workspace scope for checkpoint reads (matches write path).

Returns ``""`` only for intentionally unscoped local runtimes.
Managed AERs carry ``REQUIRE_PROJECT_SCOPED_DB`` and require
``APP_ID``; they deliberately reject the wire fallback if it is
missing.

<a id="agent_engine_runner_shared.runtime.TenantRuntime.register_and_run"></a>

#### register\_and\_run

```python
def register_and_run(graph_builder: Optional[Callable] = None,
                     **kwargs) -> None
```

Register the graph builder and start the runtime.

Behavior depends on RUNNER_MODE:
- aer: Stores graph builder, starts AER server
- tool: Starts Tool server (no graph needed)

**Arguments**:

- `graph_builder` - Function that returns a compiled LangGraph
- `**kwargs` - Runtime startup options. Supported key is
  ``log_level``. Deprecated ``host``, ``http_port``, and
  ``grpc_port`` keys are still accepted for backwards
  compatibility but are ignored.

Structured runtime helpers for reading ``agent.yaml``.

Lookup order is intentionally small and explicit:
1. ``config_path`` argument: exact ``agent.yaml`` path, or a directory that contains it
2. ``AGENTIC_AGENT_CONFIG_PATH``: exact in-container file path baked into runtime images
3. ``AGENTIC_AGENT_WORKDIR``: agent working directory used by generated local dev stacks
4. current working directory: supports tests and ad-hoc SDK usage

<a id="agent_engine_runner_shared.agent_config.AgentFeatureConfig"></a>

## AgentFeatureConfig

```python
class AgentFeatureConfig(BaseModel)
```

Runtime feature flags from ``agent.yaml``.

Keep field names in parity with the TypeScript schema and CLI allowlist.
Add a field here when introducing a flag, then mirror it in those surfaces.

``None`` means the flag was omitted, which lets the runtime fall back to
legacy environment variables during the transition.

<a id="agent_engine_runner_shared.agent_config.AgentFeatureConfig.explicit"></a>

#### explicit

```python
def explicit() -> dict[str, bool]
```

Return only flags present in ``agent.yaml`` (omit unset / ``None``).

<a id="agent_engine_runner_shared.agent_config.A2ASkillConfig"></a>

## A2ASkillConfig

```python
class A2ASkillConfig(BaseModel)
```

A skill advertised by the agent for A2A discovery.

<a id="agent_engine_runner_shared.agent_config.A2AConfig"></a>

## A2AConfig

```python
class A2AConfig(BaseModel)
```

Agent-to-agent configuration from ``agent.yaml``.

When ``enabled`` is true, this agent is discoverable and callable
by other agents through the Orchestration Engine.

<a id="agent_engine_runner_shared.agent_config.RuntimeMCPAuthConfig"></a>

## RuntimeMCPAuthConfig

```python
class RuntimeMCPAuthConfig(BaseModel)
```

Authentication settings for a remote MCP server.

<a id="agent_engine_runner_shared.agent_config.RuntimeMCPServerConfig"></a>

## RuntimeMCPServerConfig

```python
class RuntimeMCPServerConfig(BaseModel)
```

Runtime configuration for one remote MCP server.

<a id="agent_engine_runner_shared.agent_config.RuntimeMCPConfig"></a>

## RuntimeMCPConfig

```python
class RuntimeMCPConfig(BaseModel)
```

Remote MCP server configuration from ``agent.yaml``.

<a id="agent_engine_runner_shared.agent_config.RuntimeConnectorReference"></a>

## RuntimeConnectorReference

```python
class RuntimeConnectorReference(BaseModel)
```

One ``agent.yaml`` connectors entry linking to a connector ``tool.yaml``.

<a id="agent_engine_runner_shared.agent_config.RuntimeAgentConfig"></a>

## RuntimeAgentConfig

```python
class RuntimeAgentConfig(BaseModel)
```

Validated runtime view of ``agent.yaml``.

<a id="agent_engine_runner_shared.agent_config.RuntimeAgentConfig.feature_enabled"></a>

#### feature\_enabled

```python
def feature_enabled(name: FeatureName, default: bool = False) -> bool
```

Return a feature flag value, falling back to ``default`` when omitted.

<a id="agent_engine_runner_shared.agent_config.RuntimeAgentConfig.configured_feature"></a>

#### configured\_feature

```python
def configured_feature(name: FeatureName) -> bool | None
```

Return the explicit feature value from ``agent.yaml``, if it exists.

<a id="agent_engine_runner_shared.agent_config.load_runtime_agent_config"></a>

#### load\_runtime\_agent\_config

```python
def load_runtime_agent_config(
        config_path: ConfigPath | None = None,
        *,
        env_vars: Mapping[str, str] | None = None) -> RuntimeAgentConfig
```

Load and validate ``agent.yaml`` for runtime use.

Missing files are treated as an empty config so unit tests and ad-hoc SDK
usage can still construct ``App`` / ``TenantRuntime`` outside generated
runtime environments. When a file exists, the runtime validates the fields
it owns directly (entrypoint/features) while passing the application-owned
``config`` block through as raw data.

``env_vars`` is the substitution mapping used to resolve ``${VAR}``
references at allowlisted YAML paths (currently ``mcp.servers.*.url``).
Pass the tenant-owned subset of the process environment -- never raw
``os.environ`` -- so tenant ``agent.yaml`` cannot dereference platform
secrets. ``agent_engine_runner_shared.utils.tenant_env_vars()`` builds the safe subset.
When ``env_vars`` is ``None`` and a ``${...}`` reference is present at an
allowlisted path, the loader raises so the misconfiguration is visible
instead of falling through to ``urlparse`` with the literal string.

Per-execution context management using contextvars.

This module provides thread-safe, coroutine-safe context for execution state.
Each concurrent execution gets its own isolated context, preventing cross-execution
contamination of execution_id, wrapper, and OE URL.

Best Practice: All context variables are available for log correlation.

Usage:
    from agent_engine_runner_shared.context import (
        current_execution_id,
        current_wrapper,
        current_oe_url,
        set_execution_context,
        clear_execution_context,
    )

    # Set context at start of execution
    tokens = set_execution_context(execution_id, wrapper, oe_url)

    try:
        # During execution, access via .get()
        exec_id = current_execution_id.get()
        wrapper = current_wrapper.get()
    finally:
        # Always clear context when done
        clear_execution_context(tokens)

<a id="agent_engine_runner_shared.context.OwnerUrlFailureState"></a>

## OwnerUrlFailureState

```python
@dataclass
class OwnerUrlFailureState()
```

One-way owner-failure latch: once an owner pre-attempt fails, later
owner-preferring posts in this execution skip the owner URL entirely
instead of re-paying the pre-attempt timeout on every emit. A mutable
holder for the same reason as ``_SessionFinishState``: emit() may run on a
worker thread via ``asyncio.to_thread``, whose copied context shares this
object reference while a ``ContextVar.set`` there would be invisible to
the parent. No lock: the only write is the one-way ``failed = True``,
atomic under the GIL. Mirrors the AER stream path's per-execution
owner-URL discard.

<a id="agent_engine_runner_shared.context._SessionFinishState"></a>

## \_SessionFinishState

```python
@dataclass
class _SessionFinishState()
```

Mutable holder so a request from a copied context (LangGraph node,
worker thread) is visible to the parent frame reading it later.

Shared across real OS threads (LangGraph worker threads), not just
copied contexts - request_session_finish()'s read-modify-write of
`requested` and `closed` must be adjudicated together under one lock.
Reading either field unlocked while a writer is mid-transition on the
other reintroduces the exact race the lock exists to close, so every
access (read or write, of either field) goes through `lock`.

<a id="agent_engine_runner_shared.context._SessionFinishState.close"></a>

#### close

```python
def close() -> None
```

Close the latch: nothing outside this run can finish anymore.
Called from the AER's `finally` once the execute frame ends -
keeps the lock private to the holder rather than making callers
take it themselves.

<a id="agent_engine_runner_shared.context.SessionFinishStatus"></a>

## SessionFinishStatus

```python
class SessionFinishStatus(str, Enum)
```

Outcome of a request_session_finish() call.

<a id="agent_engine_runner_shared.context.SessionFinishStatus.REQUESTED"></a>

#### REQUESTED

this call recorded the request

<a id="agent_engine_runner_shared.context.SessionFinishStatus.ALREADY_REQUESTED"></a>

#### ALREADY\_REQUESTED

a previous call in this run did

<a id="agent_engine_runner_shared.context.SessionFinishStatus.UNAVAILABLE"></a>

#### UNAVAILABLE

no execution context (local dev, tool pod)

<a id="agent_engine_runner_shared.context.set_execution_context"></a>

#### set\_execution\_context

```python
def set_execution_context(
        execution_id: str,
        wrapper: Any,
        oe_url: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        authorization: Optional["ToolAuthorization"] = None,
        payload: Optional[Dict[str, Any]] = None,
        oe_owner_url: Optional[str] = None,
        owner_url_failure: Optional[OwnerUrlFailureState] = None,
        *,
        trace_id: Optional[str] = None) -> ContextTokens
```

Set execution context for the current async task.

**Arguments**:

- `execution_id` - The execution ID for this request
- `wrapper` - The SecureToolWrapper instance
- `oe_url` - The OE callback URL
- `trace_id` - Optional originating platform trace ID for log correlation
- `request_id` - Optional request ID, auto-generated as ``req-<12 hex>`` when omitted
- `user_id` - Optional user ID for the current execution
- `session_id` - Optional session/thread ID for conversation continuity
- `workspace_id` - Optional workspace ID for guardrails/cost attribution
- `custom_headers` - Optional caller-provided custom headers (X-Mdb-Agent-Engine-Custom-* prefix-stripped and lowercased)
- `authorization` - Optional delegated credential context for tool execution
- `payload` - Optional opaque caller-provided invocation payload
- `oe_owner_url` - Optional validated replica-specific OE owner URL for callback fallback


**Returns**:

  Tuple of tokens for resetting context later

<a id="agent_engine_runner_shared.context.clear_execution_context"></a>

#### clear\_execution\_context

```python
def clear_execution_context(tokens: ContextTokens) -> None
```

Clear execution context using the tokens from set_execution_context.

**Arguments**:

- `tokens` - The tuple of tokens returned by set_execution_context

<a id="agent_engine_runner_shared.context.get_current_wrapper"></a>

#### get\_current\_wrapper

```python
def get_current_wrapper() -> Optional[Any]
```

Get the current execution's SecureToolWrapper, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_execution_id"></a>

#### get\_current\_execution\_id

```python
def get_current_execution_id() -> Optional[str]
```

Get the current execution ID, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_request_id"></a>

#### get\_current\_request\_id

```python
def get_current_request_id() -> Optional[str]
```

Get the current request ID, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_trace_id"></a>

#### get\_current\_trace\_id

```python
def get_current_trace_id() -> Optional[str]
```

Get the current platform trace ID for log correlation, or None.

<a id="agent_engine_runner_shared.context.get_current_oe_url"></a>

#### get\_current\_oe\_url

```python
def get_current_oe_url() -> Optional[str]
```

Get the current OE URL, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_oe_owner_url"></a>

#### get\_current\_oe\_owner\_url

```python
def get_current_oe_owner_url() -> Optional[str]
```

Get the validated replica-specific OE owner URL.

None if absent or if a previous owner pre-attempt in this execution
already failed (see :func:`report_oe_owner_url_failure`).

<a id="agent_engine_runner_shared.context.report_oe_owner_url_failure"></a>

#### report\_oe\_owner\_url\_failure

```python
def report_oe_owner_url_failure() -> None
```

Mark the current execution's owner URL unusable.

One-way: after this, :func:`get_current_oe_owner_url` returns None for the
rest of the execution so repeated emits stop re-paying the pre-attempt
timeout against a dead owner. No-op outside an execution context.

<a id="agent_engine_runner_shared.context.get_current_user_id"></a>

#### get\_current\_user\_id

```python
def get_current_user_id() -> Optional[str]
```

Get the current user ID, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_session_id"></a>

#### get\_current\_session\_id

```python
def get_current_session_id() -> Optional[str]
```

Get the current session/thread ID, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_workspace_id"></a>

#### get\_current\_workspace\_id

```python
def get_current_workspace_id() -> Optional[str]
```

Get the current workspace ID, or None if not in execution context.

<a id="agent_engine_runner_shared.context.get_current_authorization"></a>

#### get\_current\_authorization

```python
def get_current_authorization() -> Optional["ToolAuthorization"]
```

Get delegated authorization for the current execution, if available.

<a id="agent_engine_runner_shared.context.get_current_custom_headers"></a>

#### get\_current\_custom\_headers

```python
def get_current_custom_headers() -> Dict[str, str]
```

Get caller-provided custom headers, or empty dict if not set.

Platform-internal headers (``a2a-`` prefix) are stripped - agent code
should never see A2A tokens or routing metadata.  Internal platform
code that needs the full set (e.g. the A2A client) should call
:func:`get_all_custom_headers` instead.

<a id="agent_engine_runner_shared.context.get_all_custom_headers"></a>

#### get\_all\_custom\_headers

```python
def get_all_custom_headers() -> Dict[str, str]
```

Get all custom headers including platform-internal ``a2a-`` entries.

<a id="agent_engine_runner_shared.context.get_current_execution_metadata"></a>

#### get\_current\_execution\_metadata

```python
def get_current_execution_metadata() -> Dict[str, Any]
```

Get metadata accumulated for the current execution step.

<a id="agent_engine_runner_shared.context.get_current_payload"></a>

#### get\_current\_payload

```python
def get_current_payload() -> Dict[str, Any]
```

Get the caller-provided invocation payload, or empty dict if not set.

This is the opaque request body the caller sent alongside ``message``
(e.g. screen state, structured context). Agent code can read it from
anywhere in the execution, including tool functions.

<a id="agent_engine_runner_shared.context.request_session_finish"></a>

#### request\_session\_finish

```python
def request_session_finish() -> SessionFinishStatus
```

Record that the agent considers this session finished.

Mutates a holder shared with child contexts and worker threads on purpose:
LangGraph nodes run in copied contexts, where a ContextVar.set is invisible
to the AER frame that reads this at the end of the turn.

`current_wrapper` is None for Tool/Function contexts (see server/tool.py) —
only the AER holds the finish latch, so those contexts must also report
UNAVAILABLE rather than a misleading success.

<a id="agent_engine_runner_shared.context.is_session_finish_requested"></a>

#### is\_session\_finish\_requested

```python
def is_session_finish_requested() -> bool
```

Return whether request_session_finish() has been called during this execution.

<a id="agent_engine_runner_shared.context.close_session_finish_latch"></a>

#### close\_session\_finish\_latch

```python
def close_session_finish_latch() -> None
```

Close the current execution's session-finish latch.

Called from the AER's `finally` once the execute frame ends (covers the
success, error, policy-denied, and suspend paths alike - a finish
requested from post-turn work on a failed run is equally unactionable).
After this, request_session_finish() reports UNAVAILABLE instead of
promising a release that will never happen, because the caller (e.g. an
asyncio.create_task scheduled during the turn but running after it)
already got its answer from a latch that no longer has anyone reading
it.

<a id="agent_engine_runner_shared.context.record_suspend_request"></a>

#### record\_suspend\_request

```python
def record_suspend_request(payload: Dict[str, Any]) -> None
```

Record an author-intended HITL suspend for the current tool call.

Called only by ``SuspendPayload.to_json``, so the signal's provenance is
the tool author's code, not tool-result data. A no-op outside
an execution context (the holder is unset).

<a id="agent_engine_runner_shared.context.get_requested_suspend"></a>

#### get\_requested\_suspend

```python
def get_requested_suspend() -> Optional[Dict[str, Any]]
```

Return the suspend payload this tool call requested via
``SuspendPayload.to_json``, or None. The Tool Pod reads this after the tool
returns to decide whether to report ``status="suspend"``.

<a id="agent_engine_runner_shared.context.local_suspend_request_context"></a>

#### local\_suspend\_request\_context

```python
@contextmanager
def local_suspend_request_context() -> Iterator[None]
```

Give one in-process tool call an isolated suspend marker.

<a id="agent_engine_runner_shared.context.record_current_memory_metadata"></a>

#### record\_current\_memory\_metadata

```python
def record_current_memory_metadata(*,
                                   action: str,
                                   memory_type: str,
                                   content: Optional[str] = None,
                                   relevance_score: Optional[float] = None,
                                   query: Optional[str] = None) -> None
```

Attach explicit memory metadata to the current execution step, if one exists.

<a id="agent_engine_runner_shared.context.ValidationContext"></a>

## ValidationContext

```python
@dataclass(frozen=True)
class ValidationContext()
```

Execution identity passed to guardrails for audit logging.

<a id="agent_engine_runner_shared.context.ValidationContext.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict[str, str]
```

Serialize to a dict for inclusion in HTTP payloads.

<a id="agent_engine_runner_shared.context.get_validation_context"></a>

#### get\_validation\_context

```python
def get_validation_context() -> ValidationContext
```

Build a ValidationContext from the current execution contextvars.

<a id="agent_engine_runner_shared.context.get_current_log_origin"></a>

#### get\_current\_log\_origin

```python
def get_current_log_origin() -> Optional[str]
```

Return ``"customer"`` inside a customer-code boundary, else None.

<a id="agent_engine_runner_shared.context.customer_origin_scope"></a>

#### customer\_origin\_scope

```python
@contextmanager
def customer_origin_scope() -> Iterator[None]
```

Mark the dynamic extent of customer agent/tool code for log attribution.

Framework-internal. Nested scopes are a no-op. Missing origin means
unclassified, not a proven platform-authored record.

Shared Pydantic models for Runner SDK components.

These models define the API contracts between:
- Orchestration Engine (OE)
- Agent Execution Runtime (AER)
- Tool Executor Pod

<a id="agent_engine_runner_shared.models.ExecutionStatus"></a>

## ExecutionStatus

```python
class ExecutionStatus(str, Enum)
```

Status of an agent execution.

<a id="agent_engine_runner_shared.models.SuspendPayload"></a>

## SuspendPayload

```python
class SuspendPayload(BaseModel)
```

Payload returned by an agent tool to trigger human-in-the-loop suspension.

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

<a id="agent_engine_runner_shared.models.SuspendPayload.to_json"></a>

#### to\_json

```python
def to_json() -> str
```

Serialize to the suspend wire marker, and record an out-of-band
suspend request on the current execution frame.

The Tool Pod honors suspend from that recorded signal — set only here,
in the tool author's own code — not by sniffing tool-result content, so
untrusted data a tool relays can no longer forge a HITL suspend. The
marker string is still returned unchanged for wire/replay compatibility.

<a id="agent_engine_runner_shared.models.PendingInterrupt"></a>

## PendingInterrupt

```python
class PendingInterrupt(BaseModel)
```

Framework-native interrupt lifted into the platform envelope.

<a id="agent_engine_runner_shared.models.InterruptResult"></a>

## InterruptResult

```python
class InterruptResult(BaseModel)
```

Result from agent execution when the agent is suspended.

Contains the suspend payload directly (framework-agnostic) rather than
wrapping LangGraph-specific Interrupt objects. Any framework-specific
state needed to resume (LangGraph checkpoint id, ADK function-call
correlation, etc.) is carried opaquely in ``metadata`` — the AER never
inspects it; the framework adapter writes it on suspend and reads it back
from ``RequestContext.metadata`` on resume.

<a id="agent_engine_runner_shared.models.StreamingResult"></a>

## StreamingResult

```python
class StreamingResult(BaseModel)
```

Result from agent execution for normal completion.

Returned by _execute_via_agent_stream when the agent finishes without
suspend. The caller uses content for the final response and messages
for memory writing.

<a id="agent_engine_runner_shared.models.InvokeRequest"></a>

## InvokeRequest

```python
class InvokeRequest(BaseModel)
```

Request to invoke the agent (compatible with agent-runtime).

<a id="agent_engine_runner_shared.models.InvokeResponse"></a>

## InvokeResponse

```python
class InvokeResponse(BaseModel)
```

Response from invoking the agent (compatible with agent-runtime).

<a id="agent_engine_runner_shared.models.ExecuteRequest"></a>

## ExecuteRequest

```python
class ExecuteRequest(BaseModel)
```

Request to execute an agent in AER.

<a id="agent_engine_runner_shared.models.ToolExecuteRequest"></a>

## ToolExecuteRequest

```python
class ToolExecuteRequest(BaseModel)
```

Request to execute a tool (AER → OE for approval).

<a id="agent_engine_runner_shared.models.ToolExecuteRequest.to_start_log"></a>

#### to\_start\_log

```python
def to_start_log(execution: "Execution") -> "ExecutionLog"
```

Convert this request to an ExecutionLog for tool start.

<a id="agent_engine_runner_shared.models.ToolExecuteRequest.to_cached_log"></a>

#### to\_cached\_log

```python
def to_cached_log(execution: "Execution",
                  cached_result: Any) -> "ExecutionLog"
```

Convert this request to an ExecutionLog for cached result.

**Arguments**:

- `execution` - The parent execution context
- `cached_result` - The cached result being returned

<a id="agent_engine_runner_shared.models.ElicitationInfo"></a>

## ElicitationInfo

```python
class ElicitationInfo(BaseModel)
```

Authorization details returned when broker consent is required.

Its presence in a ToolExecuteResponse signals that the user must authorize
via the provided URL before the tool invocation can proceed.

<a id="agent_engine_runner_shared.models.GuardrailMeta"></a>

## GuardrailMeta

```python
class GuardrailMeta(BaseModel)
```

Identity of the policy that caused a guardrail block or require_review halt.

<a id="agent_engine_runner_shared.models.ToolExecuteResponse"></a>

## ToolExecuteResponse

```python
class ToolExecuteResponse(BaseModel)
```

Response from OE for tool execution request.

<a id="agent_engine_runner_shared.models.ToolResultRequest"></a>

## ToolResultRequest

```python
class ToolResultRequest(BaseModel)
```

Report tool execution result (AER → OE).

<a id="agent_engine_runner_shared.models.ToolResultRequest.to_log"></a>

#### to\_log

```python
def to_log(execution: Optional["Execution"] = None) -> "ExecutionLog"
```

Convert this request to an ExecutionLog for tool result.

**Arguments**:

- `execution` - The parent execution context (optional)

<a id="agent_engine_runner_shared.models.ToolAuthorization"></a>

## ToolAuthorization

```python
class ToolAuthorization(BaseModel)
```

Delegated credential injected by OE for tool execution.

<a id="agent_engine_runner_shared.models.ToolPodExecuteRequest"></a>

## ToolPodExecuteRequest

```python
class ToolPodExecuteRequest(BaseModel)
```

Request to execute a tool in a Tool Pod.

<a id="agent_engine_runner_shared.models.ToolPodExecuteRequest.validate_argument_size"></a>

#### validate\_argument\_size

```python
@field_validator("arguments")
@classmethod
def validate_argument_size(cls, arguments: Dict[str, Any]) -> Dict[str, Any]
```

Bound the JSON payload every Tool Pod callable can receive.

<a id="agent_engine_runner_shared.models.ToolPodExecuteResponse"></a>

## ToolPodExecuteResponse

```python
class ToolPodExecuteResponse(BaseModel)
```

Response from Tool Pod execution.

<a id="agent_engine_runner_shared.models.GuardrailRuntimeStage"></a>

## GuardrailRuntimeStage

```python
class GuardrailRuntimeStage(str, Enum)
```

Runtime stage where OE is asking the Tool Pod to evaluate guardrails.

<a id="agent_engine_runner_shared.models.GuardrailCheckDecision"></a>

## GuardrailCheckDecision

```python
class GuardrailCheckDecision(str, Enum)
```

Decision returned by the Tool Pod guardrails evaluator.

<a id="agent_engine_runner_shared.models.GuardrailRuntimePolicy"></a>

## GuardrailRuntimePolicy

```python
class GuardrailRuntimePolicy(BaseModel)
```

OE-selected guardrail policy sent to the Tool Pod for evaluation.

<a id="agent_engine_runner_shared.models.GuardrailCheckInput"></a>

## GuardrailCheckInput

```python
class GuardrailCheckInput(BaseModel)
```

Runtime content and metadata to evaluate.

<a id="agent_engine_runner_shared.models.GuardrailCheckContext"></a>

## GuardrailCheckContext

```python
class GuardrailCheckContext(BaseModel)
```

Execution context for a guardrail check.

<a id="agent_engine_runner_shared.models.GuardrailCheckRequest"></a>

## GuardrailCheckRequest

```python
class GuardrailCheckRequest(BaseModel)
```

Request from OE to Tool Pod to evaluate selected guardrail policies.

<a id="agent_engine_runner_shared.models.GuardrailCheckEvidence"></a>

## GuardrailCheckEvidence

```python
class GuardrailCheckEvidence(BaseModel)
```

Evidence explaining why a guardrail policy triggered.

<a id="agent_engine_runner_shared.models.GuardrailCheckResponse"></a>

## GuardrailCheckResponse

```python
class GuardrailCheckResponse(BaseModel)
```

Decision returned by the Tool Pod guardrails evaluator.

<a id="agent_engine_runner_shared.models.InvokeLLMRequestArguments"></a>

## InvokeLLMRequestArguments

```python
class InvokeLLMRequestArguments(BaseModel)
```

Typed invoke_llm arguments forwarded through OE and tool pods.

<a id="agent_engine_runner_shared.models.LLMPodInvokeRequest"></a>

## LLMPodInvokeRequest

```python
class LLMPodInvokeRequest(BaseModel)
```

Request to invoke LLM on a tool executor pod.

<a id="agent_engine_runner_shared.models.coerce_token_usage"></a>

#### coerce\_token\_usage

```python
def coerce_token_usage(value: Any) -> Optional[LLMTokenUsage]
```

Turn a provider usage blob into ``LLMTokenUsage``, or None.

Accepts dicts, non-dict ``Mapping``s, Pydantic ``model_dump()`` objects
(Anthropic ``Usage``), and attribute bags (``input_tokens`` /
``prompt_tokens``). Malformed values return None rather than raising —
a bad usage blob must not fail the LLM call.

<a id="agent_engine_runner_shared.models.json_safe_metadata"></a>

#### json\_safe\_metadata

```python
def json_safe_metadata(value: Any,
                       *,
                       _depth: int = 0) -> Dict[str, JsonValue] | None
```

Copy a provider metadata mapping into JSON-safe values, or None.

``usage`` / ``token_usage`` are replaced with a dump of the coerced
``LLMTokenUsage``. Other non-JSON values are omitted so invoke/stream
cannot fail after a successful model response.

<a id="agent_engine_runner_shared.models.add_token_usage"></a>

#### add\_token\_usage

```python
def add_token_usage(existing: Optional[LLMTokenUsage],
                    incoming: Any) -> Optional[LLMTokenUsage]
```

Sum LangChain-style additive usage deltas into a cumulative snapshot.

Missing components count as 0. Tool-pod snapshot merge must not use
this helper — OpenAI-style last-chunk snapshots would double-count.
Adapters should call ``accumulate_stream_usage`` so cumulative
per-chunk snapshots are not added.

<a id="agent_engine_runner_shared.models.accumulate_stream_usage"></a>

#### accumulate\_stream\_usage

```python
def accumulate_stream_usage(existing: Optional[LLMTokenUsage],
                            incoming: Any) -> Optional[LLMTokenUsage]
```

Fold a stream chunk's usage into a running snapshot.

Cumulative providers repeat growing totals (``10/1`` then ``10/2``);
last-wins merge keeps ``10/2``. Additive providers zero-fill the
unchanged side (``18/1`` then ``0/4``); those deltas are summed.

<a id="agent_engine_runner_shared.models.merge_token_usage"></a>

#### merge\_token\_usage

```python
def merge_token_usage(existing: Optional[LLMTokenUsage],
                      incoming: Any) -> Optional[LLMTokenUsage]
```

Merge two usage snapshots field-by-field.

Later chunks fill missing prompt/completion/total/model rather than
replacing the whole object. Incoming non-null fields win, so an
Anthropic message_start (input only) plus message_delta (output only)
yields both. An explicit ``total_tokens`` is preserved.

<a id="agent_engine_runner_shared.models.LLMResult"></a>

## LLMResult

```python
class LLMResult(BaseModel)
```

Normalized result from an LLM invocation.

Provides a consistent Pydantic shape for LLM responses regardless of the
underlying provider (OpenAI, Gemini, Anthropic, etc.).

<a id="agent_engine_runner_shared.models.LLMResult.from_response"></a>

#### from\_response

```python
@classmethod
def from_response(cls, response: Any) -> "LLMResult"
```

Extract a normalized result from a LangChain LLM response.

Handles AIMessage, BaseMessage, and arbitrary objects by falling back
to str() for content extraction.

<a id="agent_engine_runner_shared.models.LLMResult.extract_usage"></a>

#### extract\_usage

```python
@staticmethod
def extract_usage(response: Any) -> Optional[LLMTokenUsage]
```

Extract token usage metadata from a LangChain response or chunk.

Coerces provider-variant shapes (dicts, non-dict Mappings, Anthropic
Usage objects, ``.usage`` attribute bags) into ``LLMTokenUsage``.
Malformed blobs return None rather than raising.

<a id="agent_engine_runner_shared.models.LLMResult.to_response"></a>

#### to\_response

```python
def to_response() -> LLMResponse
```

Convert the normalized runner result to sdk-core's response shape.

<a id="agent_engine_runner_shared.models.LLMPodInvokeResponse"></a>

## LLMPodInvokeResponse

```python
class LLMPodInvokeResponse(BaseModel)
```

Response from LLM invocation on a tool executor pod.

<a id="agent_engine_runner_shared.models.LLMPodStreamEvent"></a>

## LLMPodStreamEvent

```python
class LLMPodStreamEvent(BaseModel)
```

SSE event emitted by a tool pod during invoke_llm streaming.

<a id="agent_engine_runner_shared.models.ExecutorCallbackRequest"></a>

## ExecutorCallbackRequest

```python
class ExecutorCallbackRequest(BaseModel)
```

Callback from AER to OE when execution completes or suspends.

<a id="agent_engine_runner_shared.models.HumanReviewData"></a>

## HumanReviewData

```python
class HumanReviewData(BaseModel)
```

Data provided by human reviewer when resuming a suspended execution.

<a id="agent_engine_runner_shared.models.AgentResumeRequest"></a>

## AgentResumeRequest

```python
class AgentResumeRequest(BaseModel)
```

Request to resume a suspended execution.

<a id="agent_engine_runner_shared.models.AgentResumeResponse"></a>

## AgentResumeResponse

```python
class AgentResumeResponse(BaseModel)
```

Response from resuming an execution.

<a id="agent_engine_runner_shared.models.ExecutionStatusResponse"></a>

## ExecutionStatusResponse

```python
class ExecutionStatusResponse(BaseModel)
```

Response for execution status query.

<a id="agent_engine_runner_shared.models.Execution"></a>

## Execution

```python
class Execution(BaseModel)
```

Execution record stored in TenantDB.

<a id="agent_engine_runner_shared.models.Execution.to_persistence_doc"></a>

#### to\_persistence\_doc

```python
def to_persistence_doc(
        updated_at: Optional[datetime] = None) -> Dict[str, Any]
```

Convert to a document for MongoDB persistence.

**Arguments**:

- `updated_at` - Optional timestamp override (defaults to now)


**Returns**:

  Dict suitable for MongoDB upsert

<a id="agent_engine_runner_shared.models.Execution.from_persistence_doc"></a>

#### from\_persistence\_doc

```python
@classmethod
def from_persistence_doc(cls, doc: Dict[str, Any]) -> "Execution"
```

Rehydrate an Execution from a MongoDB document.

Uses .get() with defaults for all Optional fields so documents
created before new fields were added still deserialize safely.

<a id="agent_engine_runner_shared.models.ExecutionStep"></a>

## ExecutionStep

```python
class ExecutionStep(BaseModel)
```

Execution step record stored in TenantDB.

<a id="agent_engine_runner_shared.models.ExecutionStep.from_log_doc"></a>

#### from\_log\_doc

```python
@classmethod
def from_log_doc(cls, doc: Dict[str, Any]) -> "ExecutionStep"
```

Reconstruct an ExecutionStep from an execution_logs MongoDB document.

The execution_logs collection stores tool start/result events with
fields: execution_id, step_number, tool, inputs, status, output,
error, duration_ms, timestamp. This method maps those fields back
to the ExecutionStep model for step cache rehydration after OE restart.

<a id="agent_engine_runner_shared.models.HealthStatus"></a>

## HealthStatus

```python
class HealthStatus(str, Enum)
```

Health status of a component.

<a id="agent_engine_runner_shared.models.HealthResponse"></a>

## HealthResponse

```python
class HealthResponse(BaseModel)
```

Health check response.

<a id="agent_engine_runner_shared.models.ToolDefinition"></a>

## ToolDefinition

```python
class ToolDefinition(BaseModel)
```

Definition of a registered tool.

<a id="agent_engine_runner_shared.models.AERExecuteResponse"></a>

## AERExecuteResponse

```python
class AERExecuteResponse(BaseModel)
```

Response from AER /execute endpoint.

Returned on normal completion, HITL suspension, or cancellation of the
handler task by a drain/teardown.

<a id="agent_engine_runner_shared.models.ToolsListResponse"></a>

## ToolsListResponse

```python
class ToolsListResponse(BaseModel)
```

Response from /tools endpoint listing registered tools.

<a id="agent_engine_runner_shared.models.StreamChunk"></a>

## StreamChunk

```python
class StreamChunk(BaseModel)
```

A chunk of streaming response from the agent.

Used for real-time streaming of agent responses via SSE or gRPC.

<a id="agent_engine_runner_shared.models.AgentStartStreamRequest"></a>

## AgentStartStreamRequest

```python
class AgentStartStreamRequest(BaseModel)
```

Request to start an agent execution with streaming response.

<a id="agent_engine_runner_shared.models.ExecutionLogsQueryResponse"></a>

## ExecutionLogsQueryResponse

```python
class ExecutionLogsQueryResponse(BaseModel)
```

Response for execution logs query (used by API Gateway proxy).

<a id="agent_engine_runner_shared.models.NodeExecutionsQueryResponse"></a>

## NodeExecutionsQueryResponse

```python
class NodeExecutionsQueryResponse(BaseModel)
```

Response for node executions query (used by API Gateway proxy).

<a id="agent_engine_runner_shared.models.CostSummary"></a>

## CostSummary

```python
class CostSummary(BaseModel)
```

Aggregate cost metrics for the requested period.

<a id="agent_engine_runner_shared.models.CostByWorkspace"></a>

## CostByWorkspace

```python
class CostByWorkspace(BaseModel)
```

Cost breakdown for a single workspace.

<a id="agent_engine_runner_shared.models.CostByModel"></a>

## CostByModel

```python
class CostByModel(BaseModel)
```

Cost breakdown for a single model.

<a id="agent_engine_runner_shared.models.DailyCostEntry"></a>

## DailyCostEntry

```python
class DailyCostEntry(BaseModel)
```

Cost data for a single day.

<a id="agent_engine_runner_shared.models.CostDashboardResponse"></a>

## CostDashboardResponse

```python
class CostDashboardResponse(BaseModel)
```

Response for cost dashboard aggregation (used by API Gateway proxy).

<a id="agent_engine_runner_shared.models.ExecutionDocument"></a>

## ExecutionDocument

```python
class ExecutionDocument(BaseModel)
```

An execution document as stored in the platform database.

<a id="agent_engine_runner_shared.models.ExecutionsListQueryResponse"></a>

## ExecutionsListQueryResponse

```python
class ExecutionsListQueryResponse(BaseModel)
```

Response for executions list query (used by API Gateway proxy).

<a id="agent_engine_runner_shared.models.ExecutionDetailQueryResponse"></a>

## ExecutionDetailQueryResponse

```python
class ExecutionDetailQueryResponse(BaseModel)
```

Response for single execution detail query (used by API Gateway proxy).

<a id="agent_engine_runner_shared.models.NodeExecutionRequest"></a>

## NodeExecutionRequest

```python
class NodeExecutionRequest(BaseModel)
```

Report node execution event (AER → OE for logging).

<a id="agent_engine_runner_shared.models.NodeExecutionRequest.to_log"></a>

#### to\_log

```python
def to_log() -> "NodeExecutionLog"
```

Convert this request to a NodeExecutionLog for persistence.

emit / emit_step — send a mid-execution chunk from within a tool function.

Tool functions running inside a Tool Pod call these to stream events to the
client while the tool is still executing. Both execution_id and oe_url are
injected into Python contextvars by the Tool Pod server before the tool runs.

The call is best-effort: network errors are logged and swallowed so a failed
chunk emit never aborts the tool.

<a id="agent_engine_runner_shared.progress.emit"></a>

#### emit

```python
def emit(event: str, data: str) -> None
```

Emit a chunk of the given event type to the client stream.

General-purpose API. For the common textual-step case, prefer
:func:`emit_step`.

**Arguments**:

- `event` - Chunk type identifier (e.g. ``"step"``). Reserved infrastructure
  event types (``done``, ``error``, ``text``, ``subagent_start``,
  ``subagent_end``) raise ``ValueError`` to prevent tool code from
  prematurely closing or spoofing the stream. ``"custom_event"`` is
  not this API — use :func:`agent_engine_runner_shared.emit_custom_event` and
  ``features.use_custom_parser`` in ``agent.yaml``. Gateway drops the
  ``"custom_event"`` chunk_type unless that feature is on.
- `data` - Payload string for the event.


**Raises**:

- `ValueError` - if ``event`` is a reserved infrastructure event type.

  Transport failures are logged at DEBUG and swallowed — best-effort delivery
  must never abort the tool function. No-op outside an execution context.
  Synchronous; async tool functions must wrap in ``asyncio.to_thread``.

<a id="agent_engine_runner_shared.progress.emit_step"></a>

#### emit\_step

```python
def emit_step(message: str) -> None
```

Emit a textual progress step from within a running tool function.

Convenience wrapper around ``emit(event="step", data=message)``.

Example::

from agent_engine_runner_shared import emit_step

@app.tool(is_local=False)
def crawl_website(url: str) -> str:
emit_step("Fetching page...")
html = fetch(url)
emit_step(f"Parsing links from {url}...")
return parse(html)

Runtime-transparent custom-event emission across Atlas Agent Engine runtimes.

Authors call :func:`emit_custom_event` without knowing which runtime is
hosting the code (AER agent process vs Tool Pod / function mode). The active
runtime installs exactly one :class:`CustomEventTransport` before invoking
tenant code; the helper dispatches to that transport.

Atlas Agent Engine transports an opaque author-final JSON object as ``custom_event``.
It does not interpret keys — customers who need a shaped vocabulary (e.g.
Holly ``{event, data}``) build that object in their own helper.

Framework adapters (e.g. LangGraph in ``agent-engine-sdk-langgraph``) own how AER
delivers events. This package only owns the author API, the install/dispatch
hook, and the Tool Pod OE HTTP transport.

<a id="agent_engine_runner_shared.custom_events.CustomEventTransport"></a>

## CustomEventTransport

```python
@runtime_checkable
class CustomEventTransport(Protocol)
```

Request-scoped delivery path for author custom events.

The hosting runtime installs one transport per invocation. Tool Pods use
:class:`OEHTTPCustomEventTransport`; AER framework adapters install their
own implementation.

<a id="agent_engine_runner_shared.custom_events.CustomEventTransport.emit_sync"></a>

#### emit\_sync

```python
def emit_sync(payload: Mapping[str, JsonValue]) -> None
```

Deliver one author-final JSON object from synchronous code.

<a id="agent_engine_runner_shared.custom_events.CustomEventTransport.emit"></a>

#### emit

```python
async def emit(payload: Mapping[str, JsonValue]) -> None
```

Deliver one author-final JSON object from asynchronous code.

<a id="agent_engine_runner_shared.custom_events.get_custom_event_transport"></a>

#### get\_custom\_event\_transport

```python
def get_custom_event_transport() -> Optional[CustomEventTransport]
```

Return the transport installed for this request, or None.

<a id="agent_engine_runner_shared.custom_events.set_custom_event_transport"></a>

#### set\_custom\_event\_transport

```python
def set_custom_event_transport(
    transport: Optional[CustomEventTransport]
) -> Token[Optional[CustomEventTransport]]
```

Install ``transport`` for the current async task.

Returns a :class:`~contextvars.Token` that :func:`clear_custom_event_transport`
uses to restore the previous value (nested installs / concurrent tasks).

<a id="agent_engine_runner_shared.custom_events.clear_custom_event_transport"></a>

#### clear\_custom\_event\_transport

```python
def clear_custom_event_transport(
        token: Token[Optional[CustomEventTransport]]) -> None
```

Restore the previous transport via ``ContextVar.reset(token)``.

<a id="agent_engine_runner_shared.custom_events.custom_event_transport"></a>

#### custom\_event\_transport

```python
@contextmanager
def custom_event_transport(
        transport: CustomEventTransport) -> Iterator[CustomEventTransport]
```

Install ``transport`` for the block, then restore the previous value.

<a id="agent_engine_runner_shared.custom_events.emit_custom_event_sync"></a>

#### emit\_custom\_event\_sync

```python
def emit_custom_event_sync(payload: Mapping[str, Any]) -> None
```

Emit an author-final JSON object from synchronous tenant code.

Use :func:`emit_custom_event` from asynchronous code. This synchronous
sibling is intended for ordinary synchronous Tool Pod handlers and
synchronous LangGraph nodes.

<a id="agent_engine_runner_shared.custom_events.emit_custom_event"></a>

#### emit\_custom\_event

```python
async def emit_custom_event(payload: Mapping[str, Any]) -> None
```

Emit an author-final JSON object through the installed runtime transport.

Atlas Agent Engine does not interpret keys. Customers who need a shaped vocabulary
(for example Holly ``{"event": ..., "data": ...}``) build that object
themselves before calling this helper.

Use :func:`emit_custom_event_sync` from synchronous tenant code. No-op when
no transport is installed (outside an execution context).

<a id="agent_engine_runner_shared.custom_events.OEHTTPCustomEventTransport"></a>

## OEHTTPCustomEventTransport

```python
class OEHTTPCustomEventTransport()
```

Tool Pod / function-mode transport: POST final ``custom_event`` chunks to OE.

Delivery is awaited with a bounded timeout and is non-fatal: delivery
failures are logged and swallowed so a progress event cannot fail the tool.

<a id="agent_engine_runner_shared.custom_events.FeatureOffCustomEventTransport"></a>

## FeatureOffCustomEventTransport

```python
class FeatureOffCustomEventTransport()
```

Fail closed when ``use_custom_parser`` is disabled.

Runtime-neutral rejecting transport shared by AER streaming (when Atlas Agent Engine
does not subscribe to LangGraph ``custom``) and Tool Pod / function mode
(when OE POSTs would otherwise succeed while Gateway strips client
delivery). Authors must not observe a successful emit when the workspace
cannot deliver custom events to clients.

<a id="agent_engine_runner_shared.custom_events.install_tool_custom_event_transport"></a>

#### install\_tool\_custom\_event\_transport

```python
def install_tool_custom_event_transport(
        *, enabled: bool = True) -> Token[Optional[CustomEventTransport]]
```

Install Tool Pod / function-mode custom-event transport; return reset token.

When ``enabled`` is false (workspace does not have ``use_custom_parser``),
install a rejecting transport so emit fails closed instead of POSTing an
event that Gateway will strip for clients.

Core guardrail policy evaluation orchestration.

<a id="agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngineResult"></a>

## GuardrailPolicyEngineResult

```python
@dataclass(frozen=True)
class GuardrailPolicyEngineResult()
```

Engine-owned result for a policy evaluation.

<a id="agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngine"></a>

## GuardrailPolicyEngine

```python
class GuardrailPolicyEngine(Protocol)
```

Contract implemented by native and future remote guardrail engines.

<a id="agent_engine_runner_shared.guardrails_evaluator.core.GuardrailPolicyEngine.evaluate"></a>

#### evaluate

```python
def evaluate(policy: GuardrailRuntimePolicy,
             text: str) -> GuardrailPolicyEngineResult | None
```

Return engine-owned evidence and optional transformed text when triggered.

<a id="agent_engine_runner_shared.guardrails_evaluator.core.register_guardrail_policy_engine"></a>

#### register\_guardrail\_policy\_engine

```python
def register_guardrail_policy_engine(engine: GuardrailPolicyEngine) -> None
```

Register a policy engine so ``evaluate_guardrail_check`` can dispatch to it by ``policy_type``.

<a id="agent_engine_runner_shared.guardrails_evaluator.core.evaluate_guardrail_check"></a>

#### evaluate\_guardrail\_check

```python
def evaluate_guardrail_check(
        request: GuardrailCheckRequest) -> GuardrailCheckResponse
```

Evaluate guardrail policies for one runtime boundary.

SecureLLMProxy - Framework-neutral proxy for secure LLM calls through OE.

The proxy packages intercepted invoke_llm requests for the Orchestration Engine
and unwraps the streamed OE relay back into sdk-core models. The OE owns
approval, routing, live SSE relay, and final audit/result recording.

<a id="agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy"></a>

## SecureLLMProxy

```python
class SecureLLMProxy()
```

Framework-neutral proxy for secure LLM calls through OE.

Framework callers such as ``SecureWrappedLLM`` own client-side ``llm_call``
metrics and consume the ``last_*`` fields exposed here after each call.

<a id="agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.step_counter"></a>

#### step\_counter

```python
@property
def step_counter() -> int
```

Current operational-step watermark for compatibility readers.

<a id="agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.invoke"></a>

#### invoke

```python
def invoke(messages: list[Message],
           step: int | None = None,
           stop: list[str] | None = None,
           options: LLMInvocationOptions | None = None) -> LLMResponse
```

Invoke LLM by collecting the stream-oriented execution path.

<a id="agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.response_from_stream_chunks"></a>

#### response\_from\_stream\_chunks

```python
@classmethod
def response_from_stream_chunks(cls,
                                chunks: list[LLMStreamChunk]) -> LLMResponse
```

Collect streamed sdk-core chunks into a final sdk-core LLMResponse.

<a id="agent_engine_runner_shared.secure_llm_proxy.SecureLLMProxy.stream"></a>

#### stream

```python
def stream(
        messages: list[Message],
        step: int | None = None,
        stop: list[str] | None = None,
        options: LLMInvocationOptions | None = None
) -> Iterator[LLMStreamChunk]
```

Stream invoke_llm chunks through OE approval and OE-owned SSE relay.

Sessions routed to the platform-owned workflow wrap the call in a
serial LLM activity: a recorded outcome replays without a model call;
a fresh dispatch streams the model and records the folded response
under the OE-issued fence.

SecureToolWrapper - Routes all tool/LLM calls through OE for logging and policy enforcement.

This module provides:
- SecureToolWrapper: Wraps tool calls to route through OE
- SecureWrappedLLM: Wraps LLM calls to route through OE

HITL is handled via LangGraph's native interrupt() / Command(resume=...) API.

<a id="agent_engine_runner_shared.secure_wrapper.extract_usage"></a>

#### extract\_usage

```python
def extract_usage(response: Any,
                  fallback_model: Optional[str] = None) -> Dict[str, Any]
```

Extract token usage from an LLM response's response_metadata.

Handles provider-variant key names:
- OpenAI: response_metadata.usage.{prompt_tokens, completion_tokens, total_tokens}
- Anthropic: response_metadata.usage.{input_tokens, output_tokens}
- Some providers: response_metadata.token_usage.{...}
- Anthropic Usage objects and non-dict Mappings

Returns dict with keys: prompt_tokens, completion_tokens, total_tokens, model.
All values may be None if unavailable. Malformed blobs do not raise.

<a id="agent_engine_runner_shared.secure_wrapper.PolicyDeniedException"></a>

## PolicyDeniedException

```python
class PolicyDeniedException(Exception)
```

Raised when OE policy engine denies execution.

<a id="agent_engine_runner_shared.secure_wrapper.OERetryAfterError"></a>

## OERetryAfterError

```python
class OERetryAfterError(Exception)
```

OE asked the runner to retry /tool/execute after a Retry-After delay.

<a id="agent_engine_runner_shared.secure_wrapper.ToolExecutionError"></a>

## ToolExecutionError

```python
class ToolExecutionError(Exception)
```

Raised when tool execution fails.

<a id="agent_engine_runner_shared.secure_wrapper.TerminalExecutionError"></a>

## TerminalExecutionError

```python
class TerminalExecutionError(ToolExecutionError)
```

Raised when OE rejects a tool call because the execution already ended.

<a id="agent_engine_runner_shared.secure_wrapper.raise_for_oe_rejection"></a>

#### raise\_for\_oe\_rejection

```python
def raise_for_oe_rejection(
        reason: str | None,
        *,
        guardrail_meta: Optional[GuardrailMeta] = None) -> None
```

Map a proceed=false OE response to the matching exception. Never returns.

<a id="agent_engine_runner_shared.secure_wrapper.ToolCallTimeoutError"></a>

## ToolCallTimeoutError

```python
class ToolCallTimeoutError(ToolExecutionError)
```

Raised when a tool call outlives its deadline.

Distinct from PolicyDeniedException on purpose. A timeout means the call was
permitted and ran — it just ran too long — so reporting it as a denial sends
the developer to debug governance instead of their tool.

<a id="agent_engine_runner_shared.secure_wrapper.OperationalStepAllocator"></a>

## OperationalStepAllocator

```python
class OperationalStepAllocator()
```

Process-local operational step_number mint for one AER execution.

Shared by SecureToolWrapper and SecureLLMProxy so parallel tools and LLM
calls cannot collide under a single AER. Not a multi-OE authority.

<a id="agent_engine_runner_shared.secure_wrapper.ExternalAPICallError"></a>

## ExternalAPICallError

```python
class ExternalAPICallError(ToolExecutionError)
```

Raised when a tool's external API call fails with a classified error.

<a id="agent_engine_runner_shared.secure_wrapper.LLMInvocationError"></a>

## LLMInvocationError

```python
class LLMInvocationError(Exception)
```

Raised when an LLM invocation fails after OE approval.

``source`` is optional invoke-owner attribution. Only ``"llm"`` means the
provider failed; omit it for relay, truncation, or guardrail plumbing that
uses this same exception type.

``error_code`` is the machine-readable classification the tool pod stamped
on the failure (e.g. a provider credential rejection), when it did. It
travels to the OE/UI on the ERROR chunk metadata instead of the generic
invocation code so consumers can classify without string-matching prose.

<a id="agent_engine_runner_shared.secure_wrapper.request_oe_approval"></a>

#### request\_oe\_approval

```python
def request_oe_approval(
        oe_url: str,
        execution_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        step: int,
        *,
        kind: Optional[str] = None,
        is_local: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        redact_fields: Optional[list[str]] = None,
        provider_type: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        tool_call_id: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: float | httpx.Timeout | None = None) -> ToolExecuteResponse
```

Request approval from OE before executing a tool/LLM call.

**Returns**:

  ToolExecuteResponse with proceed flag and optional cached_result


**Raises**:

- `PolicyDeniedException` - If OE is unreachable

<a id="agent_engine_runner_shared.secure_wrapper.request_oe_approval_retryable"></a>

#### request\_oe\_approval\_retryable

```python
def request_oe_approval_retryable(request: Callable[..., ToolExecuteResponse],
                                  **kwargs: Any) -> ToolExecuteResponse
```

Repeat the same /tool/execute while OE says the failure is retryable.

Reservation-loss returns status=error with retryable=true after releasing
the step stamp. Retrying the same step here keeps durable activity from
recording FAILED on a call that can still succeed.

Wrapped in a span: without one, a retried call is invisible —
the surrounding tool-node span just looks slower, with no record of how
many attempts happened. Started before ``request`` (typically
``request_oe_approval``) runs, so the trace_id/span_id it reads via
``_current_trace_context()`` reflect this span.

<a id="agent_engine_runner_shared.secure_wrapper.report_oe_result"></a>

#### report\_oe\_result

```python
def report_oe_result(oe_url: str,
                     execution_id: str,
                     tool_name: str,
                     step: int,
                     status: str,
                     result: Any,
                     error: Optional[str],
                     duration_ms: float,
                     pod_name: Optional[str] = None,
                     prompt_tokens: Optional[int] = None,
                     completion_tokens: Optional[int] = None,
                     total_tokens: Optional[int] = None,
                     model: Optional[str] = None,
                     kind: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     tool_call_id: Optional[str] = None,
                     owner_url: Optional[str] = None,
                     on_owner_failure: Optional[Callable[[], None]] = None,
                     tool_api_error: Optional[ToolAPIError] = None) -> None
```

Report an execution result and require OE to acknowledge settlement.

``owner_url`` — when given, the replica-specific OE owner base URL (already
validated against ``oe_url``). A single owner pre-attempt runs before the
service loop and does not consume the service retry budget; the owner is
best-effort, so any failure — a transport error OR any non-2xx response —
marks the replica unusable and falls through to the trusted ``oe_url`` loop.
The owner is never retried and an owner response never raises.

Wrapped in a span: without one, a slow-to-ack OE or a retried
settlement is invisible — the surrounding tool-node span just looks
slower, with no record of how many attempts happened. Started before
``_current_trace_context()`` is read below, so the trace_id/span_id put
on the wire reflect this span.

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper"></a>

## SecureToolWrapper

```python
class SecureToolWrapper()
```

Wrap tool calls so OE owns execution, replay, logging, and routing.

Flow for each call:
1. Send the intercepted tool call to OE (POST /tool/execute)
2. OE returns a final outcome or routes the call back in process
3. The wrapper logs the outcome and converts suspend payloads back into framework interrupts

This enables:
- Audit logging of all operations
- Policy enforcement (block if not allowed)
- Deterministic replay scenarios (resume after SUSPEND)

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.__init__"></a>

#### \_\_init\_\_

```python
def __init__(oe_url: str,
             execution_id: str,
             custom_headers: Optional[Dict[str, str]] = None,
             durable_memory: Optional[DurableMemoryState] = None,
             oe_owner_url: Optional[str] = None,
             call_registry: Optional["DrainRegistry"] = None)
```

Initialize the wrapper.

**Arguments**:

- `oe_url` - URL of the Orchestration Engine
- `execution_id` - Unique execution identifier
- `custom_headers` - Caller-provided custom headers forwarded from the invoke request
- `durable_memory` - Durable activity Memory state, when enabled
- `oe_owner_url` - Validated replica-specific OE owner URL for tool-result
  callback fallback, or None when the request supplied no usable one
- `call_registry` - Per-execution drain registry, enabling the per-call
  abort (POST /interrupt/call) for tools this wrapper runs locally;
  None leaves local calls unaddressable, as before

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.step_counter"></a>

#### step\_counter

```python
@property
def step_counter() -> int
```

Current operational-step watermark for compatibility readers.

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.next_operational_step"></a>

#### next\_operational\_step

```python
def next_operational_step() -> int
```

Allocate the next operational step_number for this execution.

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.observe_operational_step"></a>

#### observe\_operational\_step

```python
def observe_operational_step(n: int) -> None
```

Raise the allocator watermark (e.g. from OE latest_step_number).

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.workflow"></a>

#### workflow

```python
@property
def workflow() -> Any
```

Lazily created workflow client for durable serial tool activities.

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.close"></a>

#### close

```python
async def close() -> None
```

Clean up resources owned by this execution's wrapper.

<a id="agent_engine_runner_shared.secure_wrapper.SecureToolWrapper.execute_tool"></a>

#### execute\_tool

```python
def execute_tool(tool_name: str,
                 arguments: Dict[str, Any],
                 *,
                 metadata: Optional[Dict[str, Any]] = None,
                 provider_type: Optional[str] = None,
                 scopes: Optional[list[str]] = None,
                 tool_call_id: Optional[str] = None,
                 raw_on_interrupt: bool = False,
                 redact_fields: Optional[list[str]] = None,
                 is_local: bool = True,
                 local_executor: Optional[Callable[[], Any]] = None,
                 is_framework_control_flow: Optional[Callable[[BaseException],
                                                              bool]] = None,
                 call_channel_expected: bool = False) -> Any
```

Execute a tool call through OE.

**Arguments**:

- `tool_name` - Name of the tool
- `arguments` - Arguments to pass to the tool
- `metadata` - Optional execution metadata to send to OE
- `tool_call_id` - Stable LLM tool-call id used to join this call to its
  result and session message across the execution-log surface
- `raw_on_interrupt` - When True, return the private ``_CALL_INTERRUPTED``
  sentinel on interrupt instead of the plain ``{"interrupted": True}``
  dict. Only ``create_secure_tool_function`` should pass this; it
  coerces the sentinel immediately. On the durable workflow route
  the sentinel is recorded as a reserved wire marker (replay needs
  JSON) and restored on the caller's side of the replay boundary.
- `is_local` - Whether OE should return the approved call to this AER.
- `local_executor` - Callback that invokes the registered tool on the
  current framework stack. Required for local tools.


**Returns**:

  Result from execution or cached result


**Raises**:

- `PolicyDeniedException` - If OE blocks the call
- `ToolExecutionError` - If execution fails
- `langgraph.types.interrupt` - If result contains __suspend__ signal,
  or if OE returns an elicitation requiring user authorization
  (triggers suspend with authorization_url in context)

<a id="agent_engine_runner_shared.secure_wrapper.create_secure_tool_function"></a>

#### create\_secure\_tool\_function

```python
def create_secure_tool_function(
    original_tool: Any,
    tool_name: str,
    *,
    allow_direct: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    is_local: bool = True,
    provider_type: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    response_format: Literal["content", "content_and_artifact"] = "content",
    tool_declared_format: Optional[Literal["content",
                                           "content_and_artifact"]] = None,
    redact_fields: Optional[list[str]] = None,
    is_framework_control_flow: Optional[Callable[[BaseException], bool]] = None
) -> Callable
```

Create a wrapped tool function that routes through SecureToolWrapper.

The wrapper is looked up from context at call time, allowing the tool
to be built before execution context exists.

**Arguments**:

- `original_tool` - The original LangChain tool
- `tool_name` - Name of the tool
- `allow_direct` - Whether to allow direct execution when wrapper is missing
- `metadata` - Optional execution metadata to send to OE
- `is_local` - Whether to execute the approved tool on the current AER stack
- `response_format` - The wire format the caller's StructuredTool is built
  with. See ``_coerce_content_and_artifact``.
- `tool_declared_format` - The tool author's own original format, used only
  to decide whether a normal result gets shape-matched into a tuple.
  Defaults to ``response_format``.


**Returns**:

  Wrapped function that routes through OE via SecureToolWrapper

Structured JSON logging for the agent runtime log capture pipeline.

Emits single-line JSON records on container stdout. Fluent Bit tails the
container log, the Lua filter derives tenant/workspace keys from pod
metadata, and writes gzipped batches to S3. The API Gateway
``GET /api/v1/agent-logs`` endpoint reads them back.

Activated by calling ``install_structured_logging()`` at runner startup —
typically gated by ``STRUCTURED_LOGGING=true``. When inactive, the existing
human-readable logging in ``agent_engine_runner_shared.utils.setup_logging`` is
unaffected.

<a id="agent_engine_runner_shared.structured_logging.StructuredJSONFormatter"></a>

## StructuredJSONFormatter

```python
class StructuredJSONFormatter(logging.Formatter)
```

Formats LogRecord instances as single-line JSON.

Tenant context is read from agent-engine-runner-shared contextvars at format time so
that records emitted from inside an execution carry executionId/sessionId
automatically; records emitted outside execution (startup, idle) leave
those fields ``null``.

<a id="agent_engine_runner_shared.structured_logging.LoggingStream"></a>

## LoggingStream

```python
class LoggingStream(IOBase)
```

Replacement for ``sys.stdout``/``sys.stderr`` that routes writes through logging.

Bare ``print()`` calls (and any other code writing directly to the
captured stream) are rewritten as LogRecord instances tagged with
``agentic_log_source = "stdout"`` (or ``"stderr"``) so the formatter
can mark them as such on the wire. Lines are buffered until newline,
with a hard byte cap so a producer writing without ``\n`` cannot
grow the buffer until OOM (see ``_MAX_BUFFER_BYTES``).

Holds a reference to the wrapped *original* stream so that a re-install
can recover and ``fileno()``/``buffer``/``encoding`` requests from
third-party libraries (subprocess, tqdm, gRPC) can be delegated through
instead of raising. ``write()``/``flush()`` are guarded by an
``RLock`` so concurrent producers don't interleave bytes mid-line.

Inherits from ``IOBase`` rather than ``TextIOBase``: ``TextIOBase``
declares ``encoding`` as a read-only C descriptor and we need a writable
instance attribute to surface the wrapped stream's encoding to callers
that probe ``sys.stdout.encoding`` (CLI helpers, click, rich).

<a id="agent_engine_runner_shared.structured_logging.install_structured_logging"></a>

#### install\_structured\_logging

```python
def install_structured_logging(stream: Optional[IO[str]] = None,
                               level: Optional[str] = None,
                               mode: Optional[str] = None) -> None
```

Install the structured JSON formatter on the root logger and capture
stdout/stderr.

Idempotent: if already installed, recovers the previously-wrapped
original stream so a second call does not feed the new handler back
into its own ``LoggingStream`` (which would recurse to RecursionError).

**Arguments**:

- `stream` - Output stream for the JSON handler. Defaults to the
  *current* (or previously-wrapped) ``sys.stdout`` at install
  time, captured before stdout is replaced by ``LoggingStream``.
  Useful in tests to inject a buffer.
- `level` - Log level — either a string name (``"DEBUG"``, ``"info"``)
  or a ``logging.LEVEL`` int. Defaults to ``LOG_LEVEL`` env var,
  then INFO. Must be honored on rollout because operators
  continue to flip ``LOG_LEVEL=DEBUG`` to chase issues.

  Applies to ``logging.*`` calls only — stdout/stderr writes
  captured by ``LoggingStream`` always emit, so a stray
  ``print()`` does not silently disappear when an operator sets
  ``LOG_LEVEL=WARNING``. Captures are routed via
  ``Logger.handle()`` directly, bypassing ``isEnabledFor`` and
  the handler's level check.
- `mode` - Runner mode (``"orchestrator"``, ``"aer"``, ``"tool"``) used
  to derive ``service`` and
  ``fields.component`` on the wire. Takes precedence over
  ``RUNNER_MODE`` env var when set; falls back to env when
  ``None``. Lets ``setup_logging(mode=...)`` callers stay
  authoritative even if pod env hasn't been stamped.

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

<a id="agent_engine_runner_shared.metrics.MetricPoint"></a>

## MetricPoint

```python
@dataclass
class MetricPoint()
```

Single metric data point.

<a id="agent_engine_runner_shared.metrics.LatencyStats"></a>

## LatencyStats

```python
@dataclass
class LatencyStats()
```

Latency statistics for a metric.

<a id="agent_engine_runner_shared.metrics.LatencyStats.record"></a>

#### record

```python
def record(duration_ms: float) -> None
```

Record a latency measurement.

<a id="agent_engine_runner_shared.metrics.LatencyStats.avg_ms"></a>

#### avg\_ms

```python
@property
def avg_ms() -> float
```

Average latency in milliseconds.

<a id="agent_engine_runner_shared.metrics.LatencyStats.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Convert to dictionary.

<a id="agent_engine_runner_shared.metrics.Metrics"></a>

## Metrics

```python
class Metrics()
```

Thread-safe metrics collector for the Runner SDK.

Collects:
- Tool execution latency (per tool)
- LLM call latency
- Error counts (per operation type)
- Request counts

<a id="agent_engine_runner_shared.metrics.Metrics.record_latency"></a>

#### record\_latency

```python
@classmethod
def record_latency(cls, operation: str, duration_ms: float,
                   **labels: str) -> None
```

Record a latency measurement.

**Arguments**:

- `operation` - Operation name (e.g., "tool_execution", "llm_call")
- `duration_ms` - Duration in milliseconds
- `**labels` - Additional labels (e.g., tool_name="lookup_policy")

<a id="agent_engine_runner_shared.metrics.Metrics.record_error"></a>

#### record\_error

```python
@classmethod
def record_error(cls, operation: str, **labels: str) -> None
```

Record an error occurrence.

**Arguments**:

- `operation` - Operation name
- `**labels` - Additional labels (e.g., error_type="timeout")

<a id="agent_engine_runner_shared.metrics.Metrics.get_all"></a>

#### get\_all

```python
@classmethod
def get_all(cls) -> Dict[str, Any]
```

Get all collected metrics.

<a id="agent_engine_runner_shared.metrics.Metrics.reset"></a>

#### reset

```python
@classmethod
def reset(cls) -> None
```

Reset all metrics (useful for testing).

<a id="agent_engine_runner_shared.metrics.record_latency"></a>

#### record\_latency

```python
@contextmanager
def record_latency(operation: str,
                   **labels: str) -> Generator[None, None, None]
```

Context manager to record latency of an operation.

Usage:
    with record_latency("tool_execution", tool_name="lookup_policy"):
        result = tool.invoke(args)

<a id="agent_engine_runner_shared.metrics.record_error"></a>

#### record\_error

```python
def record_error(operation: str, **labels: str) -> None
```

Record an error occurrence.

<a id="agent_engine_runner_shared.metrics.with_metrics"></a>

#### with\_metrics

```python
def with_metrics(operation: str, record_errors: bool = True)
```

Decorator to record latency and errors for a function.

Works with both sync and async functions.

Usage:
    @with_metrics("oe_agent_start")
    async def _handle_start(self, request):
        ...

    @with_metrics("tool_execution", record_errors=False)
    def execute_tool(self, ...):
        ...

<a id="agent_engine_runner_shared.metrics.log_tool_call"></a>

#### log\_tool\_call

```python
def log_tool_call(execution_id: str, step_number: int, tool_name: str,
                  arguments: Dict[str, Any], is_local: bool) -> None
```

Log a tool call with structured data.

<a id="agent_engine_runner_shared.metrics.log_tool_result"></a>

#### log\_tool\_result

```python
def log_tool_result(execution_id: str,
                    step_number: int,
                    tool_name: str,
                    status: str,
                    duration_ms: float,
                    error: Optional[str] = None) -> None
```

Log a tool result with structured data.

<a id="agent_engine_runner_shared.metrics.log_llm_call"></a>

#### log\_llm\_call

```python
def log_llm_call(execution_id: str, step_number: int, model_name: str,
                 message_count: int) -> None
```

Log an LLM call with structured data.

<a id="agent_engine_runner_shared.metrics.log_llm_result"></a>

#### log\_llm\_result

```python
def log_llm_result(execution_id: str,
                   step_number: int,
                   model_name: str,
                   status: str,
                   duration_ms: float,
                   tool_calls: Optional[List[str]] = None,
                   error: Optional[str] = None) -> None
```

Log an LLM result with structured data.

<a id="agent_engine_runner_shared.metrics.log_execution_event"></a>

#### log\_execution\_event

```python
def log_execution_event(execution_id: str,
                        event: str,
                        details: Optional[Dict[str, Any]] = None) -> None
```

Log an execution lifecycle event.

Utility functions for Runner SDK.

<a id="agent_engine_runner_shared.utils.normalize_content"></a>

#### normalize\_content

```python
def normalize_content(content: Any) -> str
```

Normalize LLM message content to a string.

LLM content can be:
- A string (normal text)
- A list (multimodal content with text and other parts)
- None or empty

This handles cases where LangChain's AIMessageChunk.content is a list
of content blocks (e.g., from Gemini multimodal responses) rather than
a simple string.

**Arguments**:

- `content` - The content to normalize (string, list, or None)


**Returns**:

  A string representation of the content

<a id="agent_engine_runner_shared.utils.normalize_tool_call_args"></a>

#### normalize\_tool\_call\_args

```python
def normalize_tool_call_args(args: Any) -> str | None
```

Normalize streamed tool-call args into the wire-format string payload.

<a id="agent_engine_runner_shared.utils.normalize_optional_str"></a>

#### normalize\_optional\_str

```python
def normalize_optional_str(value: Any) -> str | None
```

Trim a string tool-metadata value, collapsing blank/non-str to None.

Shared by the framework SDKs' tool-wrapping code so metadata values written
as "" or "  " are treated the same as absent
rather than sent to the OE as a non-empty-looking but meaningless string.

<a id="agent_engine_runner_shared.utils.strip_thinking"></a>

#### strip\_thinking

```python
def strip_thinking(text: str) -> str
```

Remove ``<think>...</think>`` blocks and unclosed ``<think>`` tails.

<a id="agent_engine_runner_shared.utils.filter_thinking_tokens"></a>

#### filter\_thinking\_tokens

```python
def filter_thinking_tokens(token: str, buffer: str,
                           inside: bool) -> tuple[str, str, bool]
```

Filter ``<think>`` blocks from a stream of token chunks.

Accumulates text in *buffer* until we can determine whether content
is inside a thinking block.  Returns ``(streamable, new_buffer,
inside)`` where *streamable* is the text safe to send to the client.

<a id="agent_engine_runner_shared.utils.RuntimeMode"></a>

## RuntimeMode

```python
class RuntimeMode(str, Enum)
```

Runtime mode for the Runner SDK.

<a id="agent_engine_runner_shared.utils.get_runtime_mode"></a>

#### get\_runtime\_mode

```python
def get_runtime_mode() -> RuntimeMode
```

Get the current runtime mode from environment variable.

**Returns**:

  RuntimeMode based on RUNNER_MODE environment variable.


**Raises**:

- `ValueError` - If RUNNER_MODE is not set or is not a valid mode.

<a id="agent_engine_runner_shared.utils.get_env"></a>

#### get\_env

```python
def get_env(name: str, default: str = "") -> str
```

Get environment variable with default.

<a id="agent_engine_runner_shared.utils.get_env_int"></a>

#### get\_env\_int

```python
def get_env_int(name: str, default: int) -> int
```

Get integer environment variable with default.

<a id="agent_engine_runner_shared.utils.get_env_float"></a>

#### get\_env\_float

```python
def get_env_float(name: str, default: float) -> float
```

Get float environment variable with default.

<a id="agent_engine_runner_shared.utils.get_env_bool"></a>

#### get\_env\_bool

```python
def get_env_bool(name: str, default: bool = False) -> bool
```

Get boolean environment variable with default.

<a id="agent_engine_runner_shared.utils.is_platform_env_var"></a>

#### is\_platform\_env\_var

```python
def is_platform_env_var(name: str) -> bool
```

Return ``True`` if ``name`` matches a platform-owned env var.

Public-API view of the same membership check used by ``tenant_env_vars``.
Used by ``agent_config`` to validate fields that name an env var (e.g.
``mcp.servers.*.auth.token_env``) so tenant YAML cannot redirect a
secret-indirection path at a platform-owned variable.

<a id="agent_engine_runner_shared.utils.tenant_env_vars"></a>

#### tenant\_env\_vars

```python
def tenant_env_vars(source: Mapping[str, str] | None = None) -> dict[str, str]
```

Return the tenant-owned subset of environment variables.

``source`` defaults to ``os.environ``. Names listed in ``_PLATFORM_ENV_VARS``
or matching any prefix in ``_PLATFORM_ENV_VAR_PREFIXES`` are excluded, so
the result is safe to pass as the substitution mapping to
``load_runtime_agent_config(env_vars=...)``.

<a id="agent_engine_runner_shared.utils.LLM_INITIAL_BACKOFF"></a>

#### LLM\_INITIAL\_BACKOFF

seconds

<a id="agent_engine_runner_shared.utils.LLM_MAX_BACKOFF"></a>

#### LLM\_MAX\_BACKOFF

seconds

<a id="agent_engine_runner_shared.utils.LLM_READ_TIMEOUT"></a>

#### LLM\_READ\_TIMEOUT

seconds

<a id="agent_engine_runner_shared.utils.TOOL_READ_TIMEOUT"></a>

#### TOOL\_READ\_TIMEOUT

seconds

<a id="agent_engine_runner_shared.utils.oe_stream_retry_delay_s"></a>

#### oe\_stream\_retry\_delay\_s

```python
def oe_stream_retry_delay_s(retry_after_ms: float | None = None) -> float
```

Honor a server-provided SSE retry_after_ms, capped.

Absent/None means retry immediately (reservation-loss). Transport
disconnects use ``OE_DISPATCH_TAKEOVER_RETRY_DELAY_S`` instead.

<a id="agent_engine_runner_shared.utils.sleep_oe_stream_retry"></a>

#### sleep\_oe\_stream\_retry

```python
def sleep_oe_stream_retry(delay_s: float) -> None
```

Sleep before an SSE same-URL retry. Tests patch this to avoid wall-clock waits.

<a id="agent_engine_runner_shared.utils.is_retryable_error"></a>

#### is\_retryable\_error

```python
def is_retryable_error(error: Exception) -> bool
```

Classify provider failures before any LLM output has been exposed.

<a id="agent_engine_runner_shared.utils.format_llm_error"></a>

#### format\_llm\_error

```python
def format_llm_error(error: Exception) -> str
```

Render an LLM provider exception for display, without escaped-JSON text.

Some provider SDKs (e.g. google-genai) attach the raw API error body as a
dict on the exception or its ``__cause__`` (``.details``, ``.body``).
That dict's values are often themselves JSON-encoded strings containing
real newlines (e.g. a pretty-printed nested error payload), so Python's
default ``str()`` ends up repr-ing those strings and turning the
newlines into literal ``\n`` sequences. Decode any
JSON-encoded string values first and re-serialize with ``json.dumps`` so
the result renders as readable, indented JSON instead.

<a id="agent_engine_runner_shared.utils.is_llm_credential_rejection"></a>

#### is\_llm\_credential\_rejection

```python
def is_llm_credential_rejection(error: BaseException) -> bool
```

True when an LLM provider exception is an auth rejection (HTTP 401/403).

Reads only the provider SDK's own status fields — ``status_code``
(openai/anthropic SDK style), ``response.status_code`` (httpx style), and
an integer ``code`` (google-genai style) — walking the explicit
``__cause__`` chain because LangChain adapters occasionally re-raise with
``from``. ``__context__`` is deliberately not walked: an unrelated error
raised while handling an auth failure would otherwise inherit the
rejection. Text is never matched: the caller maps a positive result onto
the wire-level credential error code, and anything unrecognized returns
False so the failure keeps its existing generic classification.

<a id="agent_engine_runner_shared.utils.get_request_timeout"></a>

#### get\_request\_timeout

```python
def get_request_timeout() -> float
```

Get the HTTP request timeout from environment.

Uses RUNNER_REQUEST_TIMEOUT env var, defaults to 60.0 seconds.

<a id="agent_engine_runner_shared.utils.setup_logging"></a>

#### setup\_logging

```python
def setup_logging(*,
                  app_name: str = "runner",
                  mode: str = "aer",
                  log_dir: Optional[str] = None,
                  log_level: Optional[str] = None,
                  backup_count: int = 5) -> logging.Logger
```

Configure logging with both console and file output.

All arguments are keyword-only. Prior versions took ``log_level`` as the
first positional argument; keyword-only ensures ``setup_logging("DEBUG")``
from older call sites fails loudly instead of silently binding ``"DEBUG"``
to ``app_name`` and taking the default log level.

When ``STRUCTURED_LOGGING=true`` is set, delegates to
``agent_engine_runner_shared.structured_logging.install_structured_logging`` so all
output emerges as single-line JSON matching the agent-log
contract. Disk logging is skipped in that mode in production — Fluent
Bit ships container stdout to S3, so a duplicate on-disk copy adds no
value and just wastes IO. The one exception is local dev: when
``AGENTIC_DEV_MODES`` is set (dev-up compose stacks only), a
human-readable rotating file sink is attached alongside the structured
stdout output so the Local Dev UI can read the agent's logs.

**Arguments**:

- `app_name` - Application name for log file naming
- `mode` - Runtime mode (aer, tool, tool_function)
- `log_dir` - Directory for log files (default: ./logs)
- `log_level` - Log level (default: from LOG_LEVEL env or INFO)
- `backup_count` - Number of rotated backup files to keep


**Returns**:

  Root logger configured with handlers

<a id="agent_engine_runner_shared.utils.log_separator"></a>

#### log\_separator

```python
def log_separator() -> None
```

Log a minor section separator (for individual operations).

<a id="agent_engine_runner_shared.utils.log_section"></a>

#### log\_section

```python
def log_section() -> None
```

Log a major section separator (for execution boundaries).

<a id="agent_engine_runner_shared.utils.log_llm_messages"></a>

#### log\_llm\_messages

```python
def log_llm_messages(messages: List[Any],
                     prefix: str = "LLM",
                     count_only: bool = False) -> None
```

Log LLM conversation messages in a consistent format.

**Arguments**:

- `messages` - List of message dicts or LangChain message objects
- `prefix` - Log line prefix (e.g., "OE", "LLM", "AER")
- `count_only` - If True, only log message count (for info level)

<a id="agent_engine_runner_shared.utils.log_llm_response"></a>

#### log\_llm\_response

```python
def log_llm_response(result: Any, step: int, prefix: str = "LLM") -> None
```

Log an LLM response in a consistent format.

**Arguments**:

- `result` - LLM response (dict or AIMessage)
- `step` - Step number
- `prefix` - Log line prefix

<a id="agent_engine_runner_shared.utils.log_tool_request"></a>

#### log\_tool\_request

```python
def log_tool_request(tool_name: str,
                     arguments: Dict[str, Any],
                     step: int,
                     prefix: str = "TOOL",
                     fields_to_redact: Optional[List[str]] = None) -> None
```

Log a tool execution request.

**Arguments**:

- `tool_name` - Name of the tool
- `arguments` - Tool arguments
- `step` - Step number
- `prefix` - Log line prefix
- `fields_to_redact` - Catalog sensitive-field policy. Matching fields omit
  even their type/length metadata; no values are logged.

<a id="agent_engine_runner_shared.utils.log_tool_result"></a>

#### log\_tool\_result

```python
def log_tool_result(tool_name: str,
                    step: int,
                    status: str,
                    result: Any = None,
                    error: Optional[str] = None,
                    duration_ms: float = 0,
                    prefix: str = "TOOL") -> None
```

Log a tool execution result.

**Arguments**:

- `tool_name` - Name of the tool
- `step` - Step number
- `status` - Execution status (success, error, suspend, interrupted)
- `result` - Tool result
- `error` - Error message if failed
- `duration_ms` - Execution duration in milliseconds
- `prefix` - Log line prefix

<a id="agent_engine_runner_shared.utils.log_cached_result"></a>

#### log\_cached\_result

```python
def log_cached_result(tool_name: str, step: int, prefix: str = "TOOL") -> None
```

Log that a cached result is being used (replay scenario).

<a id="agent_engine_runner_shared.utils.log_policy_blocked"></a>

#### log\_policy\_blocked

```python
def log_policy_blocked(tool_name: str,
                       step: int,
                       reason: str,
                       prefix: str = "TOOL") -> None
```

Log that a tool call was blocked by policy.

<a id="agent_engine_runner_shared.utils.log_execution_start"></a>

#### log\_execution\_start

```python
def log_execution_start(execution_id: str,
                        input_keys: List[str],
                        prefix: str = "OE") -> None
```

Log the start of an execution.

<a id="agent_engine_runner_shared.utils.log_execution_callback"></a>

#### log\_execution\_callback

```python
def log_execution_callback(execution_id: str,
                           status: str,
                           result: Any = None,
                           error: Optional[str] = None,
                           suspend_reason: Optional[str] = None,
                           prefix: str = "OE") -> None
```

Log an execution callback (completion/suspension/error).
