# @mongodb-js/agent-engine-runner-shared

## Table of Contents

- **Namespaces**
  - [chunkTypes](#namespace-chunkTypes)
- **Enumerations**
  - [ActivityKind](#api-activitykind)
  - [ActivityOutcomeKind](#api-activityoutcomekind)
  - [MessageRole](#api-messagerole)
  - [RuntimeMode](#api-runtimemode)
  - [WorkflowErrorCode](#api-workflowerrorcode)
- **Classes**
  - [ActivityDispatch](#api-activitydispatch)
  - [ActivityReplay](#api-activityreplay)
  - [AERServer](#api-aerserver)
    - [close()](#api-close)
    - [createApp()](#api-createapp)
    - [getHealthDetails()](#api-gethealthdetails)
    - [onShutdown()](#api-onshutdown)
    - [onStartup()](#api-onstartup)
    - [registerCommonRoutes()](#api-registercommonroutes)
    - [registerRoutes()](#api-registerroutes)
    - [run()](#api-run)
  - [AppBoundCrudClient](#api-appboundcrudclient)
    - [createCustom()](#api-createcustom)
    - [createEpisodic()](#api-createepisodic)
    - [createProcedural()](#api-createprocedural)
    - [createSemantic()](#api-createsemantic)
    - [createTaxonomic()](#api-createtaxonomic)
    - [getDistinctDomains()](#api-getdistinctdomains)
    - [getProcedural()](#api-getprocedural)
    - [getSemantic()](#api-getsemantic)
    - [getTaxonomic()](#api-gettaxonomic)
    - [listEpisodic()](#api-listepisodic)
    - [retrieveCustom()](#api-retrievecustom)
  - [AppBoundRuntime](#api-appboundruntime)
    - [buildContext()](#api-buildcontext)
    - [discoverProcedures()](#api-discoverprocedures)
    - [recordTurn()](#api-recordturn)
    - [requestContext()](#api-requestcontext)
    - [searchEpisodes()](#api-searchepisodes)
    - [searchSemantic()](#api-searchsemantic)
    - [searchTaxonomic()](#api-searchtaxonomic)
  - [AttemptHeartbeat](#api-attemptheartbeat)
    - [start()](#api-start)
    - [stop()](#api-stop)
  - [ContentPolicyOTLPSpanExporter](#api-contentpolicyotlpspanexporter)
    - [export()](#api-export)
    - [forceFlush()](#api-forceflush)
    - [shutdown()](#api-shutdown)
  - [DurableActivityDeniedError](#api-durableactivitydeniederror)
  - [DurableActivityInterrupted](#api-durableactivityinterrupted)
  - [DurableMemoryState](#api-durablememorystate)
    - [synchronizeLlm()](#api-synchronizellm)
    - [synchronizeTool()](#api-synchronizetool)
  - [ExternalAPICallError](#api-externalapicallerror)
  - [JSONLSpanExporter](#api-jsonlspanexporter)
    - [export()](#api-export-1)
    - [forceFlush()](#api-forceflush-1)
    - [shutdown()](#api-shutdown-1)
  - [LLMInvocationError](#api-llminvocationerror)
  - [LLMInvocationOptions](#api-llminvocationoptions)
    - [toJSON()](#api-tojson)
    - [toModelKwargs()](#api-tomodelkwargs)
  - [LLMRegistryLoadError](#api-llmregistryloaderror)
  - [LLMResponse](#api-llmresponse)
    - [fromRaw()](#api-fromraw)
  - [LLMResult](#api-llmresult)
    - [toResponse()](#api-toresponse)
    - [extractUsage()](#api-extractusage)
    - [fromResponse()](#api-fromresponse)
  - [LLMTokenUsage](#api-llmtokenusage)
    - [toJSON()](#api-tojson-1)
    - [toLangchainUsageMetadata()](#api-tolangchainusagemetadata)
  - [LLMToolCall](#api-llmtoolcall)
    - [toLangchainDict()](#api-tolangchaindict)
  - [LLMToolSchema](#api-llmtoolschema)
    - [toLangchainDict()](#api-tolangchaindict-1)
  - [MCPConfigError](#api-mcpconfigerror)
  - [MCPToolError](#api-mcptoolerror)
  - [Metrics](#api-metrics)
    - [getAll()](#api-getall)
    - [recordError()](#api-recorderror)
    - [recordLatency()](#api-recordlatency)
    - [reset()](#api-reset)
  - [MongoDBSpanExporter](#api-mongodbspanexporter)
    - [export()](#api-export-2)
    - [forceFlush()](#api-forceflush-2)
    - [shutdown()](#api-shutdown-2)
  - [NodeExecutionLogger](#api-nodeexecutionlogger)
    - [onNodeEnd()](#api-onnodeend)
    - [onNodeError()](#api-onnodeerror)
    - [onNodeStart()](#api-onnodestart)
    - [onNodeSuspend()](#api-onnodesuspend)
  - [OperationalStepAllocator](#api-operationalstepallocator)
    - [current()](#api-current)
    - [next()](#api-next)
    - [observeAtLeast()](#api-observeatleast)
  - [OutputValidationPolicyEngine](#api-outputvalidationpolicyengine)
    - [evaluate()](#api-evaluate)
  - [PolicyDeniedException](#api-policydeniedexception)
  - [ReplayedActivityFailedError](#api-replayedactivityfailederror)
  - [RuntimeAgentConfig](#api-runtimeagentconfig)
    - [configuredFeature()](#api-configuredfeature)
    - [featureEnabled()](#api-featureenabled)
  - [SecureLLMProxy](#api-securellmproxy)
    - [invoke()](#api-invoke)
    - [stream()](#api-stream)
    - [responseFromStreamChunks()](#api-responsefromstreamchunks)
  - [SecureToolWrapper](#api-securetoolwrapper)
    - [close()](#api-close-1)
    - [executeTool()](#api-executetool)
    - [nextOperationalStep()](#api-nextoperationalstep)
    - [observeOperationalStep()](#api-observeoperationalstep)
  - [TenantRuntime](#api-tenantruntime)
    - [getAgent()](#api-getagent)
    - [getAgentConfig()](#api-getagentconfig)
    - [getCurrentSessionId()](#api-getcurrentsessionid)
    - [getCurrentUserId()](#api-getcurrentuserid)
    - [getMongodbUri()](#api-getmongodburi)
    - [getToolMetadata()](#api-gettoolmetadata)
    - [registerAndRun()](#api-registerandrun)
    - [registerTool()](#api-registertool)
    - [shutdown()](#api-shutdown-3)
    - [warmUpAgent()](#api-warmupagent)
  - [TerminalExecutionError](#api-terminalexecutionerror)
  - [ToolCallTimeoutError](#api-toolcalltimeouterror)
  - [ToolExecutionError](#api-toolexecutionerror)
  - [ToolFunctionRunner](#api-toolfunctionrunner)
    - [run()](#api-run-1)
  - [ToolServer](#api-toolserver)
    - [close()](#api-close-2)
    - [createApp()](#api-createapp-1)
    - [getHealthDetails()](#api-gethealthdetails-1)
    - [onShutdown()](#api-onshutdown-1)
    - [onStartup()](#api-onstartup-1)
    - [registerCommonRoutes()](#api-registercommonroutes-1)
    - [registerRoutes()](#api-registerroutes-1)
    - [run()](#api-run-2)
  - [UnsupportedChildOperationFanOutError](#api-unsupportedchildoperationfanouterror)
  - [WorkflowClient](#api-workflowclient)
    - [completeExecution()](#api-completeexecution)
    - [ensureMemoryWritten()](#api-ensurememorywritten)
    - [finalizeStep()](#api-finalizestep)
    - [heartbeat()](#api-heartbeat)
    - [reportOutcome()](#api-reportoutcome)
    - [startActivity()](#api-startactivity)
    - [startAttempt()](#api-startattempt)
  - [WorkflowClientError](#api-workflowclienterror)
- **Interfaces**
  - [AERQueryPlugin](#api-aerqueryplugin)
    - [getMessagesForSession()](#api-getmessagesforsession)
    - [getSummariesForSessions()](#api-getsummariesforsessions)
  - [CaptureExceptionArgs](#api-captureexceptionargs)
  - [ChildOperationBoundary](#api-childoperationboundary)
  - [ExecutionStore](#api-executionstore)
  - [ExtractedUsage](#api-extractedusage)
  - [GraphBuilderLike](#api-graphbuilderlike)
    - [getAgent()?](#api-getagent-1)
    - [ready()?](#api-ready)
    - [warmUp()?](#api-warmup)
  - [GuardrailPolicyEngine](#api-guardrailpolicyengine)
    - [evaluate()](#api-evaluate-1)
  - [GuardrailPolicyEngineResult](#api-guardrailpolicyengineresult)
  - [InitErrorReportingArgs](#api-initerrorreportingargs)
  - [InstallStructuredLoggingArgs](#api-installstructuredloggingargs)
  - [InterruptedActivity](#api-interruptedactivity)
  - [ITenantRuntime](#api-itenantruntime)
    - [getAgent()](#api-getagent-2)
    - [getAgentConfig()](#api-getagentconfig-1)
    - [getMongodbUri()](#api-getmongodburi-1)
    - [warmUpAgent()?](#api-warmupagent-1)
  - [LatencyStatsJson](#api-latencystatsjson)
  - [MCPToolBinding](#api-mcptoolbinding)
  - [MCPToolResult](#api-mcptoolresult)
  - [Message](#api-message)
  - [NodeExecutionLoggerOpts](#api-nodeexecutionloggeropts)
  - [OperationalStepSource](#api-operationalstepsource)
    - [current()](#api-current-1)
    - [next()](#api-next-1)
    - [observeAtLeast()](#api-observeatleast-1)
  - [OwnerUrlFailureState](#api-ownerurlfailurestate)
  - [RecordMemoryMetadataArgs](#api-recordmemorymetadataargs)
  - [RegisterAndRunOptions](#api-registerandrunoptions)
  - [ReportOeResultArgs](#api-reportoeresultargs)
  - [RequestOeApprovalArgs](#api-requestoeapprovalargs)
  - [ResolvedEntrypoint](#api-resolvedentrypoint)
  - [SetExecutionContextArgs](#api-setexecutioncontextargs)
  - [SetupLoggingArgs](#api-setuploggingargs)
  - [SetupTracingArgs](#api-setuptracingargs)
  - [SubprocessFailure](#api-subprocessfailure)
  - [TenantRuntimeOptions](#api-tenantruntimeoptions)
  - [ToolCallChunk](#api-toolcallchunk)
  - [WorkflowAdapter](#api-workflowadapter)
- **Type Aliases**
  - [ActivityCommand](#api-activitycommand)
  - [ActivityContext](#api-activitycontext)
  - [ActivityHeartbeatRequest](#api-activityheartbeatrequest)
  - [ActivityMemoryCommand](#api-activitymemorycommand)
  - [ActivityOutcome](#api-activityoutcome)
  - [ActivityPosition](#api-activityposition)
  - [ActivityResolvedHook](#api-activityresolvedhook)
  - [ActivitySuspension](#api-activitysuspension)
  - [AERExecuteResponse](#api-aerexecuteresponse)
  - [AgentFeatureConfig](#api-agentfeatureconfig)
  - [AgentResumeRequest](#api-agentresumerequest)
  - [AgentResumeResponse](#api-agentresumeresponse)
  - [AgentStartStreamRequest](#api-agentstartstreamrequest)
  - [AttemptContext](#api-attemptcontext)
  - [AttemptHeartbeatRequest](#api-attemptheartbeatrequest)
  - [AttemptStartRequest](#api-attemptstartrequest)
  - [AttemptStartResponse](#api-attemptstartresponse)
  - [BranchLineage](#api-branchlineage)
  - [CompleteExecutionCommand](#api-completeexecutioncommand)
  - [CostByModel](#api-costbymodel)
  - [CostByWorkspace](#api-costbyworkspace)
  - [CostDashboardResponse](#api-costdashboardresponse)
  - [CostSummary](#api-costsummary)
  - [DailyCostEntry](#api-dailycostentry)
  - [ElicitationInfo](#api-elicitationinfo)
  - [ExecuteRequest](#api-executerequest)
  - [Execution](#api-execution)
  - [ExecutionDetailQueryResponse](#api-executiondetailqueryresponse)
  - [ExecutionDocument](#api-executiondocument)
  - [ExecutionLogsQueryResponse](#api-executionlogsqueryresponse)
  - [ExecutionsListQueryResponse](#api-executionslistqueryresponse)
  - [ExecutionStatus](#api-executionstatus)
  - [ExecutionStatusResponse](#api-executionstatusresponse)
  - [ExecutionStep](#api-executionstep)
  - [ExecutorCallbackRequest](#api-executorcallbackrequest)
  - [FeatureName](#api-featurename)
  - [FinalizeStepCommand](#api-finalizestepcommand)
  - [FinalizeStepResponse](#api-finalizestepresponse)
  - [GuardrailCheckContext](#api-guardrailcheckcontext)
  - [GuardrailCheckDecision](#api-guardrailcheckdecision)
  - [GuardrailCheckEvidence](#api-guardrailcheckevidence)
  - [GuardrailCheckInput](#api-guardrailcheckinput)
  - [GuardrailCheckRequest](#api-guardrailcheckrequest)
  - [GuardrailCheckResponse](#api-guardrailcheckresponse)
  - [GuardrailMeta](#api-guardrailmeta)
  - [GuardrailRuntimePolicy](#api-guardrailruntimepolicy)
  - [GuardrailRuntimeStage](#api-guardrailruntimestage)
  - [HealthResponse](#api-healthresponse)
  - [HealthStatus](#api-healthstatus)
  - [HumanReviewData](#api-humanreviewdata)
  - [Instrumentor](#api-instrumentor)
  - [InterruptResult](#api-interruptresult)
  - [InvokeLLMRequestArguments](#api-invokellmrequestarguments)
  - [InvokeRequest](#api-invokerequest)
  - [InvokeResponse](#api-invokeresponse)
  - [JsonValue](#api-jsonvalue)
  - [LLMAdapterFactory](#api-llmadapterfactory)
  - [LLMPodInvokeRequest](#api-llmpodinvokerequest)
  - [LLMPodInvokeResponse](#api-llmpodinvokeresponse)
  - [LLMPodStreamEvent](#api-llmpodstreamevent)
  - [MemoryWrite](#api-memorywrite)
  - [NodeExecutionRequest](#api-nodeexecutionrequest)
  - [NodeExecutionsQueryResponse](#api-nodeexecutionsqueryresponse)
  - [OperationPath](#api-operationpath)
  - [OperationPathResolver](#api-operationpathresolver)
  - [OperationPathSegment](#api-operationpathsegment)
  - [PendingInterrupt](#api-pendinginterrupt)
  - [RuntimeMCPAuthConfig](#api-runtimemcpauthconfig)
  - [RuntimeMCPConfig](#api-runtimemcpconfig)
  - [RuntimeMCPServerConfig](#api-runtimemcpserverconfig)
  - [SecretsConfig](#api-secretsconfig)
  - [SecretsConfigInput](#api-secretsconfiginput)
  - [ServerToolFn](#api-servertoolfn)
  - [SessionFinishStatus](#api-sessionfinishstatus)
  - [SessionInfo](#api-sessioninfo)
  - [SessionMessage](#api-sessionmessage)
  - [SessionMessagesQueryResponse](#api-sessionmessagesqueryresponse)
  - [SessionsQueryResponse](#api-sessionsqueryresponse)
  - [StartActivityResult](#api-startactivityresult)
  - [StateSnapshot](#api-statesnapshot)
  - [StepActivityEntry](#api-stepactivityentry)
  - [StepSuspensionEntry](#api-stepsuspensionentry)
  - [StreamChunk](#api-streamchunk)
  - [StreamingResult](#api-streamingresult)
  - [SuspendHandler](#api-suspendhandler)
  - [SuspendPayload](#api-suspendpayload)
  - [TenantScope](#api-tenantscope)
  - [ToolAPIError](#api-toolapierror)
  - [ToolAuthorization](#api-toolauthorization)
  - [ToolDefinition](#api-tooldefinition)
  - [ToolExecuteRequest](#api-toolexecuterequest)
  - [ToolExecuteResponse](#api-toolexecuteresponse)
  - [ToolFunctionRequest](#api-toolfunctionrequest)
  - [ToolPodExecuteRequest](#api-toolpodexecuterequest)
  - [ToolPodExecuteResponse](#api-toolpodexecuteresponse)
  - [ToolResponseFormat](#api-toolresponseformat)
  - [ToolResultRequest](#api-toolresultrequest)
  - [ToolsListResponse](#api-toolslistresponse)
  - [WorkflowDeclaration](#api-workflowdeclaration)
  - [WorkflowError](#api-workflowerror)
  - [WorkflowIdentity](#api-workflowidentity)
  - [WorkflowMessage](#api-workflowmessage)
- **Variables**
  - [ActivityCommandSchema](#api-activitycommandschema)
  - [ActivityContextSchema](#api-activitycontextschema)
  - [ActivityHeartbeatRequestSchema](#api-activityheartbeatrequestschema)
  - [ActivityKindSchema](#api-activitykindschema)
  - [ActivityMemoryCommandSchema](#api-activitymemorycommandschema)
  - [ActivityOutcomeKindSchema](#api-activityoutcomekindschema)
  - [ActivityOutcomeSchema](#api-activityoutcomeschema)
  - [ActivityPositionSchema](#api-activitypositionschema)
  - [ActivitySuspensionSchema](#api-activitysuspensionschema)
  - [AER\_BUILD\_AGENT](#api-aer_build_agent)
  - [AERExecuteResponseSchema](#api-aerexecuteresponseschema)
  - [AgentFeatureConfigSchema](#api-agentfeatureconfigschema)
  - [AgentResumeRequestSchema](#api-agentresumerequestschema)
  - [AgentResumeResponseSchema](#api-agentresumeresponseschema)
  - [AgentStartStreamRequestSchema](#api-agentstartstreamrequestschema)
  - [AttemptContextSchema](#api-attemptcontextschema)
  - [AttemptHeartbeatRequestSchema](#api-attemptheartbeatrequestschema)
  - [AttemptStartRequestSchema](#api-attemptstartrequestschema)
  - [AttemptStartResponseSchema](#api-attemptstartresponseschema)
  - [ATTR\_CACHE\_HIT](#api-attr_cache_hit)
  - [ATTR\_COLD\_START](#api-attr_cold_start)
  - [ATTR\_SKILLS\_LOADED\_COUNT](#api-attr_skills_loaded_count)
  - [ATTR\_SKILLS\_SOURCE\_COUNT](#api-attr_skills_source_count)
  - [BranchLineageSchema](#api-branchlineageschema)
  - [BUILTIN\_TOOL\_NAMES](#api-builtin_tool_names)
  - [CALL\_INTERRUPTED\_ARTIFACT\_KEY](#api-call_interrupted_artifact_key)
  - [CompleteExecutionCommandSchema](#api-completeexecutioncommandschema)
  - [CostByModelSchema](#api-costbymodelschema)
  - [CostByWorkspaceSchema](#api-costbyworkspaceschema)
  - [CostDashboardResponseSchema](#api-costdashboardresponseschema)
  - [CostSummarySchema](#api-costsummaryschema)
  - [DailyCostEntrySchema](#api-dailycostentryschema)
  - [DEFAULT\_MCP\_OAUTH\_CLIENT\_NAME](#api-default_mcp_oauth_client_name)
  - [DEFAULT\_MCP\_OAUTH\_REDIRECT\_URI](#api-default_mcp_oauth_redirect_uri)
  - [DEFAULT\_PORTS](#api-default_ports)
  - [ElicitationInfoSchema](#api-elicitationinfoschema)
  - [ExecuteRequestSchema](#api-executerequestschema)
  - [ExecutionDetailQueryResponseSchema](#api-executiondetailqueryresponseschema)
  - [ExecutionDocumentSchema](#api-executiondocumentschema)
  - [ExecutionLogsQueryResponseSchema](#api-executionlogsqueryresponseschema)
  - [ExecutionSchema](#api-executionschema)
  - [ExecutionsListQueryResponseSchema](#api-executionslistqueryresponseschema)
  - [ExecutionStatus](#api-executionstatus-1)
  - [ExecutionStatusResponseSchema](#api-executionstatusresponseschema)
  - [ExecutionStatusSchema](#api-executionstatusschema)
  - [ExecutionStepSchema](#api-executionstepschema)
  - [ExecutorCallbackRequestSchema](#api-executorcallbackrequestschema)
  - [EXIT\_IMPORT\_ERROR](#api-exit_import_error)
  - [EXIT\_NO\_ENTRYPOINT](#api-exit_no_entrypoint)
  - [EXIT\_STARTUP\_CRASH](#api-exit_startup_crash)
  - [file\_workflow\_v1\_activity](#api-file_workflow_v1_activity)
  - [file\_workflow\_v1\_common](#api-file_workflow_v1_common)
  - [file\_workflow\_v1\_runtime](#api-file_workflow_v1_runtime)
  - [file\_workflow\_v1\_state](#api-file_workflow_v1_state)
  - [FinalizeStepCommandSchema](#api-finalizestepcommandschema)
  - [FinalizeStepResponseSchema](#api-finalizestepresponseschema)
  - [GRAPH\_BUILD](#api-graph_build)
  - [GuardrailCheckContextSchema](#api-guardrailcheckcontextschema)
  - [GuardrailCheckDecision](#api-guardrailcheckdecision-1)
  - [GuardrailCheckDecisionSchema](#api-guardrailcheckdecisionschema)
  - [GuardrailCheckEvidenceSchema](#api-guardrailcheckevidenceschema)
  - [GuardrailCheckInputSchema](#api-guardrailcheckinputschema)
  - [GuardrailCheckRequestSchema](#api-guardrailcheckrequestschema)
  - [GuardrailCheckResponseSchema](#api-guardrailcheckresponseschema)
  - [GuardrailMetaSchema](#api-guardrailmetaschema)
  - [GuardrailRuntimePolicySchema](#api-guardrailruntimepolicyschema)
  - [GuardrailRuntimeStage](#api-guardrailruntimestage-1)
  - [GuardrailRuntimeStageSchema](#api-guardrailruntimestageschema)
  - [HealthResponseSchema](#api-healthresponseschema)
  - [HealthStatus](#api-healthstatus-1)
  - [HealthStatusSchema](#api-healthstatusschema)
  - [HumanReviewDataSchema](#api-humanreviewdataschema)
  - [InterruptResultSchema](#api-interruptresultschema)
  - [InvokeLLMRequestArgumentsSchema](#api-invokellmrequestargumentsschema)
  - [InvokeRequestSchema](#api-invokerequestschema)
  - [InvokeResponseSchema](#api-invokeresponseschema)
  - [LLM\_BACKOFF\_MULTIPLIER](#api-llm_backoff_multiplier)
  - [LLM\_INITIAL\_BACKOFF](#api-llm_initial_backoff)
  - [LLM\_MAX\_BACKOFF](#api-llm_max_backoff)
  - [LLM\_MAX\_RETRIES](#api-llm_max_retries)
  - [LLM\_READ\_TIMEOUT](#api-llm_read_timeout)
  - [LLMPodInvokeRequestSchema](#api-llmpodinvokerequestschema)
  - [LLMPodInvokeResponseSchema](#api-llmpodinvokeresponseschema)
  - [LLMPodStreamEventSchema](#api-llmpodstreameventschema)
  - [LLMResultSchema](#api-llmresultschema)
  - [MAX\_TOOL\_ARGUMENT\_BYTES](#api-max_tool_argument_bytes)
  - [MemoryWriteSchema](#api-memorywriteschema)
  - [MessageRoleSchema](#api-messageroleschema)
  - [MODEL\_REQUEST\_PREPARE](#api-model_request_prepare)
  - [MODEL\_RESPONSE\_PROCESS](#api-model_response_process)
  - [NodeExecutionRequestSchema](#api-nodeexecutionrequestschema)
  - [NodeExecutionsQueryResponseSchema](#api-nodeexecutionsqueryresponseschema)
  - [OE\_DISPATCH\_RETRY\_MAX\_WAIT\_MS](#api-oe_dispatch_retry_max_wait_ms)
  - [OE\_DISPATCH\_TAKEOVER\_RETRY\_DELAY\_MS](#api-oe_dispatch_takeover_retry_delay_ms)
  - [OE\_RETRYABLE\_MAX\_ATTEMPTS](#api-oe_retryable_max_attempts)
  - [oeStreamRetry](#api-oestreamretry)
  - [OPENINFERENCE\_SPAN\_KIND](#api-openinference_span_kind)
  - [OpenInferenceSpanKind](#api-openinferencespankind)
  - [OperationPathSchema](#api-operationpathschema)
  - [OperationPathSegmentSchema](#api-operationpathsegmentschema)
  - [PendingInterruptSchema](#api-pendinginterruptschema)
  - [RuntimeMCPAuthConfigSchema](#api-runtimemcpauthconfigschema)
  - [RuntimeMCPConfigSchema](#api-runtimemcpconfigschema)
  - [RuntimeMCPServerConfigSchema](#api-runtimemcpserverconfigschema)
  - [SecretsConfigSchema](#api-secretsconfigschema)
  - [SessionInfoSchema](#api-sessioninfoschema)
  - [SessionMessageSchema](#api-sessionmessageschema)
  - [SessionMessagesQueryResponseSchema](#api-sessionmessagesqueryresponseschema)
  - [SessionsQueryResponseSchema](#api-sessionsqueryresponseschema)
  - [SKILLS\_MIDDLEWARE\_BEFORE\_AGENT](#api-skills_middleware_before_agent)
  - [StateSnapshotSchema](#api-statesnapshotschema)
  - [StepActivityEntrySchema](#api-stepactivityentryschema)
  - [StepSuspensionEntrySchema](#api-stepsuspensionentryschema)
  - [StreamChunkSchema](#api-streamchunkschema)
  - [StreamingResultSchema](#api-streamingresultschema)
  - [SuspendPayloadSchema](#api-suspendpayloadschema)
  - [TenantScopeSchema](#api-tenantscopeschema)
  - [ToolAPIErrorSchema](#api-toolapierrorschema)
  - [ToolAuthorizationSchema](#api-toolauthorizationschema)
  - [ToolDefinitionSchema](#api-tooldefinitionschema)
  - [ToolExecuteRequestSchema](#api-toolexecuterequestschema)
  - [ToolExecuteResponseSchema](#api-toolexecuteresponseschema)
  - [ToolFunctionRequestSchema](#api-toolfunctionrequestschema)
  - [ToolPodExecuteRequestSchema](#api-toolpodexecuterequestschema)
  - [ToolPodExecuteResponseSchema](#api-toolpodexecuteresponseschema)
  - [ToolResultRequestSchema](#api-toolresultrequestschema)
  - [ToolsListResponseSchema](#api-toolslistresponseschema)
  - [WorkflowDeclarationSchema](#api-workflowdeclarationschema)
  - [WorkflowErrorCodeSchema](#api-workflowerrorcodeschema)
  - [WorkflowErrorSchema](#api-workflowerrorschema)
  - [WorkflowIdentitySchema](#api-workflowidentityschema)
  - [WorkflowMessageSchema](#api-workflowmessageschema)
  - [WORKSPACE\_DIR](#api-workspace_dir)
- **Functions**
  - [accumulateStreamUsage()](#api-accumulatestreamusage)
  - [activityRequiresReconstruction()](#api-activityrequiresreconstruction)
  - [addTokenUsage()](#api-addtokenusage)
  - [advanceStepOrdinal()](#api-advancestepordinal)
  - [allocateActivityOrdinal()](#api-allocateactivityordinal)
  - [appliesToStage()](#api-appliestostage)
  - [attachMongoTracing()](#api-attachmongotracing)
  - [attemptStartRequestFromExecute()](#api-attemptstartrequestfromexecute)
  - [boundText()](#api-boundtext)
  - [buildActivityCommand()](#api-buildactivitycommand)
  - [buildResultPayload()](#api-buildresultpayload)
  - [callMcpTool()](#api-callmcptool)
  - [captureException()](#api-captureexception)
  - [captureMessage()](#api-capturemessage)
  - [checkDecision()](#api-checkdecision)
  - [childOperationBoundaryScope()](#api-childoperationboundaryscope)
  - [clearWorkflowAdapter()](#api-clearworkflowadapter)
  - [closeSessionFinish()](#api-closesessionfinish)
  - [coerceTokenUsage()](#api-coercetokenusage)
  - [completedOutcome()](#api-completedoutcome)
  - [completeExecutionCommand()](#api-completeexecutioncommand-1)
  - [createSecureToolFunction()](#api-createsecuretoolfunction)
  - [currentAttemptContext()](#api-currentattemptcontext)
  - [currentOperationPath()](#api-currentoperationpath)
  - [currentPendingChildOperationBatch()](#api-currentpendingchildoperationbatch)
  - [currentStepOrdinal()](#api-currentstepordinal)
  - [discoverMcpTools()](#api-discovermcptools)
  - [emit()](#api-emit)
  - [emitStep()](#api-emitstep)
  - [encodeProtoJson()](#api-encodeprotojson)
  - [ensureUtc()](#api-ensureutc)
  - [entrypointScope()](#api-entrypointscope)
  - [evaluateGuardrailCheck()](#api-evaluateguardrailcheck)
  - [executionFromPersistenceDoc()](#api-executionfrompersistencedoc)
  - [executionStepFromLogDoc()](#api-executionstepfromlogdoc)
  - [executionToPersistenceDoc()](#api-executiontopersistencedoc)
  - [explicitFeatures()](#api-explicitfeatures)
  - [extractUsage()](#api-extractusage-1)
  - [failedOutcome()](#api-failedoutcome)
  - [fetchPlatform()](#api-fetchplatform)
  - [filesystemDownload()](#api-filesystemdownload)
  - [filesystemEdit()](#api-filesystemedit)
  - [filesystemGlob()](#api-filesystemglob)
  - [filesystemGrep()](#api-filesystemgrep)
  - [filesystemLs()](#api-filesystemls)
  - [filesystemRead()](#api-filesystemread)
  - [filesystemWrite()](#api-filesystemwrite)
  - [filterThinkingTokens()](#api-filterthinkingtokens)
  - [finalizeCurrentStepCommand()](#api-finalizecurrentstepcommand)
  - [finalizeCurrentStepSuspensionsCommand()](#api-finalizecurrentstepsuspensionscommand)
  - [flushErrorReporting()](#api-flusherrorreporting)
  - [formatLlmError()](#api-formatllmerror)
  - [getAllCustomHeaders()](#api-getallcustomheaders)
  - [getCallAbortSignal()](#api-getcallabortsignal)
  - [getCheckpointWorkspaceId()](#api-getcheckpointworkspaceid)
  - [getContentCaptureMode()](#api-getcontentcapturemode)
  - [getCurrentAuthorization()](#api-getcurrentauthorization)
  - [getCurrentCustomHeaders()](#api-getcurrentcustomheaders)
  - [getCurrentExecutionId()](#api-getcurrentexecutionid)
  - [getCurrentExecutionMetadata()](#api-getcurrentexecutionmetadata)
  - [getCurrentLogOrigin()](#api-getcurrentlogorigin)
  - [getCurrentOeOwnerUrl()](#api-getcurrentoeownerurl)
  - [getCurrentOeUrl()](#api-getcurrentoeurl)
  - [getCurrentPayload()](#api-getcurrentpayload)
  - [getCurrentRequestId()](#api-getcurrentrequestid)
  - [getCurrentSessionId()](#api-getcurrentsessionid-1)
  - [getCurrentTraceContext()](#api-getcurrenttracecontext)
  - [getCurrentTraceId()](#api-getcurrenttraceid)
  - [getCurrentUserId()](#api-getcurrentuserid-1)
  - [getCurrentWorkspaceId()](#api-getcurrentworkspaceid)
  - [getCurrentWrapper()](#api-getcurrentwrapper)
  - [getEnv()](#api-getenv)
  - [getEnvBool()](#api-getenvbool)
  - [getEnvFloat()](#api-getenvfloat)
  - [getEnvInt()](#api-getenvint)
  - [getFetchOptionsWithTLS()](#api-getfetchoptionswithtls)
  - [getInstrumentor()](#api-getinstrumentor)
  - [getLLMAdapterFactory()](#api-getllmadapterfactory)
  - [getLogger()](#api-getlogger)
  - [getNamedLlm()](#api-getnamedllm)
  - [getQueryPlugin()](#api-getqueryplugin)
  - [getRequestedSuspend()](#api-getrequestedsuspend)
  - [getRequestTimeout()](#api-getrequesttimeout)
  - [getRuntimeMode()](#api-getruntimemode)
  - [getStoreDbName()](#api-getstoredbname)
  - [getSuspendHandler()](#api-getsuspendhandler)
  - [getToolReadTimeout()](#api-gettoolreadtimeout)
  - [getTracePath()](#api-gettracepath)
  - [getTracer()](#api-gettracer)
  - [getWorkflowAdapter()](#api-getworkflowadapter)
  - [hasNamedLlms()](#api-hasnamedllms)
  - [initErrorReporting()](#api-initerrorreporting)
  - [installStructuredLogging()](#api-installstructuredlogging)
  - [interruptedActivities()](#api-interruptedactivities)
  - [isConfiguredMcpSdkToolName()](#api-isconfiguredmcpsdktoolname)
  - [isLlmCredentialRejection()](#api-isllmcredentialrejection)
  - [isPlatformEnvVar()](#api-isplatformenvvar)
  - [isRetryableError()](#api-isretryableerror)
  - [isSessionFinishRequested()](#api-issessionfinishrequested)
  - [jsonSafeMetadata()](#api-jsonsafemetadata)
  - [loadRuntimeAgentConfig()](#api-loadruntimeagentconfig)
  - [logCachedResult()](#api-logcachedresult)
  - [logExecutionCallback()](#api-logexecutioncallback)
  - [logExecutionStart()](#api-logexecutionstart)
  - [logLLMMessages()](#api-logllmmessages)
  - [logLLMResponse()](#api-logllmresponse)
  - [logPolicyBlocked()](#api-logpolicyblocked)
  - [logSection()](#api-logsection)
  - [logSeparator()](#api-logseparator)
  - [logToolRequest()](#api-logtoolrequest)
  - [logToolResult()](#api-logtoolresult)
  - [makeMcpClientCredentialsAuth()](#api-makemcpclientcredentialsauth)
  - [makeMcpOauthAuth()](#api-makemcpoauthauth)
  - [makeMcpSdkToolName()](#api-makemcpsdktoolname)
  - [makeMcpToolCallable()](#api-makemcptoolcallable)
  - [materializeMcpOauthSecretCache()](#api-materializemcpoauthsecretcache)
  - [mcpOauthCacheDir()](#api-mcpoauthcachedir)
  - [mcpOauthCacheName()](#api-mcpoauthcachename)
  - [mcpServerNetworkHosts()](#api-mcpservernetworkhosts)
  - [mergeTokenUsage()](#api-mergetokenusage)
  - [newDurabilityOwnerId()](#api-newdurabilityownerid)
  - [normalizeContent()](#api-normalizecontent)
  - [normalizeLLMPodInvokeResponse()](#api-normalizellmpodinvokeresponse)
  - [normalizeMcpToolResult()](#api-normalizemcptoolresult)
  - [normalizeOptionalStr()](#api-normalizeoptionalstr)
  - [normalizeToolCallArgs()](#api-normalizetoolcallargs)
  - [noteCheckpointWireWorkspaceId()](#api-notecheckpointwireworkspaceid)
  - [observedActivityPositions()](#api-observedactivitypositions)
  - [oeStreamRetryDelayMs()](#api-oestreamretrydelayms)
  - [preallocateActivityOrdinals()](#api-preallocateactivityordinals)
  - [preallocateChildOperationOrdinals()](#api-preallocatechildoperationordinals)
  - [projectScopingRequired()](#api-projectscopingrequired)
  - [quotePathSegment()](#api-quotepathsegment)
  - [recordCurrentMemoryMetadata()](#api-recordcurrentmemorymetadata)
  - [recordInterruptedActivity()](#api-recordinterruptedactivity)
  - [recordLatency()](#api-recordlatency-1)
  - [recordObservedActivity()](#api-recordobservedactivity)
  - [recordReconstructedActivityInterrupt()](#api-recordreconstructedactivityinterrupt)
  - [recordSuspendRequest()](#api-recordsuspendrequest)
  - [redactFields()](#api-redactfields)
  - [registerBuiltinTools()](#api-registerbuiltintools)
  - [registerGuardrailPolicyEngine()](#api-registerguardrailpolicyengine)
  - [registerInstrumentor()](#api-registerinstrumentor)
  - [registerLlm()](#api-registerllm)
  - [registerLLMAdapterFactory()](#api-registerllmadapterfactory)
  - [registerQueryPlugin()](#api-registerqueryplugin)
  - [registerRegexGuardrailPolicyEngine()](#api-registerregexguardrailpolicyengine)
  - [registerSuspendHandler()](#api-registersuspendhandler)
  - [registerWorkflowAdapter()](#api-registerworkflowadapter)
  - [reportOeOwnerUrlFailure()](#api-reportoeownerurlfailure)
  - [reportOeResult()](#api-reportoeresult)
  - [requestOeApproval()](#api-requestoeapproval)
  - [requestSessionFinish()](#api-requestsessionfinish)
  - [resetCheckpointWorkspaceState()](#api-resetcheckpointworkspacestate)
  - [resetHooks()](#api-resethooks)
  - [resetLlmRegistry()](#api-resetllmregistry)
  - [resetStoreDbCache()](#api-resetstoredbcache)
  - [resolveCheckpointWorkspaceId()](#api-resolvecheckpointworkspaceid)
  - [resolveConfiguredMcpToolBinding()](#api-resolveconfiguredmcptoolbinding)
  - [resolveEffectiveDb()](#api-resolveeffectivedb)
  - [resolveEntrypoint()](#api-resolveentrypoint)
  - [resolveImportTarget()](#api-resolveimporttarget)
  - [resolveListenHost()](#api-resolvelistenhost)
  - [resolveMcpAuth()](#api-resolvemcpauth)
  - [resolveMcpHeaders()](#api-resolvemcpheaders)
  - [resolveOeUrl()](#api-resolveoeurl)
  - [resolveProjectScopedDb()](#api-resolveprojectscopeddb)
  - [resolveStoreDbName()](#api-resolvestoredbname)
  - [runInstrumentor()](#api-runinstrumentor)
  - [runLauncher()](#api-runlauncher)
  - [runSerialActivity()](#api-runserialactivity)
  - [runStreamingActivity()](#api-runstreamingactivity)
  - [runWithAttemptContext()](#api-runwithattemptcontext)
  - [runWithCallAbortSignal()](#api-runwithcallabortsignal)
  - [runWithCustomerOrigin()](#api-runwithcustomerorigin)
  - [runWithExecutionContext()](#api-runwithexecutioncontext)
  - [runWithOperationPathResolver()](#api-runwithoperationpathresolver)
  - [runWithSuspendRequestContext()](#api-runwithsuspendrequestcontext)
  - [scrubCredentials()](#api-scrubcredentials)
  - [serializeInvokeLLMRequestArguments()](#api-serializeinvokellmrequestarguments)
  - [setActivityReconstructionIds()](#api-setactivityreconstructionids)
  - [setPendingChildOperationBatch()](#api-setpendingchildoperationbatch)
  - [setTerminationLogPathForTest()](#api-setterminationlogpathfortest)
  - [setupLogging()](#api-setuplogging)
  - [setupTracing()](#api-setuptracing)
  - [shellExecute()](#api-shellexecute)
  - [shutdownTracing()](#api-shutdowntracing)
  - [snapshotLlmRegistry()](#api-snapshotllmregistry)
  - [stripThinking()](#api-stripthinking)
  - [subprocessOutputTail()](#api-subprocessoutputtail)
  - [summarizeSubprocessFailure()](#api-summarizesubprocessfailure)
  - [suspendPayloadToJson()](#api-suspendpayloadtojson)
  - [tenantEnvVars()](#api-tenantenvvars)
  - [toolActivityKey()](#api-toolactivitykey)
  - [toolRedactFields()](#api-toolredactfields)
  - [tracingStatus()](#api-tracingstatus)
  - [unwrapActivityOutcome()](#api-unwrapactivityoutcome)
  - [validateDurableMemoryIdentity()](#api-validatedurablememoryidentity)
  - [valueToJson()](#api-valuetojson)
  - [withExecutionSignal()](#api-withexecutionsignal)
  - [withMetrics()](#api-withmetrics)
  - [writeTerminationMessage()](#api-writeterminationmessage)

## Namespaces

<a id="namespace-chunkTypes"></a>

### chunkTypes

#### Variables

<a id="namespace-chunkTypes-api-custom_event"></a>

##### CUSTOM\_EVENT

```ts
const CUSTOM_EVENT: "custom_event" = "custom_event";
```

***

<a id="namespace-chunkTypes-api-done"></a>

##### DONE

```ts
const DONE: "done" = "done";
```

***

<a id="namespace-chunkTypes-api-error"></a>

##### ERROR

```ts
const ERROR: "error" = "error";
```

***

<a id="namespace-chunkTypes-api-llm_credential_rejected_error_code"></a>

##### LLM\_CREDENTIAL\_REJECTED\_ERROR\_CODE

```ts
const LLM_CREDENTIAL_REJECTED_ERROR_CODE: "llm_credential_rejected" = "llm_credential_rejected";
```

Machine-readable discriminator set on `metadata.error_code` when the LLM
provider rejected the configured credential (HTTP 401/403). Unlike
`llm_invocation_failed` this is deliberately stamped WITHOUT
`metadata.source`: the fix is the customer's own project secret, and the
gateway's owner switch maps the bare code to client-owned, whereas an
`llm` source would attribute it to provider flakiness.

***

<a id="namespace-chunkTypes-api-llm_invocation_error_code"></a>

##### LLM\_INVOCATION\_ERROR\_CODE

```ts
const LLM_INVOCATION_ERROR_CODE: "llm_invocation_failed" = "llm_invocation_failed";
```

Machine-readable discriminator set on `metadata.error_code` of an ERROR
chunk/callback when the agent's LLM call failed after OE approval (provider
401/404, rate limit, etc.). `metadata.source` is `llm` so invoke
attribution does not have to string-match the error prose.

***

<a id="namespace-chunkTypes-api-llm_invocation_error_source"></a>

##### LLM\_INVOCATION\_ERROR\_SOURCE

```ts
const LLM_INVOCATION_ERROR_SOURCE: "llm" = "llm";
```

Owner attribution for LLM-provider failures. Must match the gateway invoke-owner source switch.

***

<a id="namespace-chunkTypes-api-policy_denied_error_code"></a>

##### POLICY\_DENIED\_ERROR\_CODE

```ts
const POLICY_DENIED_ERROR_CODE: "policy_denied" = "policy_denied";
```

***

<a id="namespace-chunkTypes-api-step"></a>

##### STEP

```ts
const STEP: "step" = "step";
```

***

<a id="namespace-chunkTypes-api-subagent_end"></a>

##### SUBAGENT\_END

```ts
const SUBAGENT_END: "subagent_end" = "subagent_end";
```

***

<a id="namespace-chunkTypes-api-subagent_start"></a>

##### SUBAGENT\_START

```ts
const SUBAGENT_START: "subagent_start" = "subagent_start";
```

***

<a id="namespace-chunkTypes-api-text"></a>

##### TEXT

```ts
const TEXT: "text" = "text";
```

Wire-protocol constants for AER → OE stream chunks.

Centralized so a typo in one site can't silently desync the consumer.
Values are stable strings forwarded verbatim through OE to the API gateway
and on to the UI; do not rename without coordinating across all three
layers (and the Go-side StreamChunk model).

***

<a id="namespace-chunkTypes-api-timeout_error_code"></a>

##### TIMEOUT\_ERROR\_CODE

```ts
const TIMEOUT_ERROR_CODE: "timeout" = "timeout";
```

Machine-readable discriminator set on `metadata.error_code` of an ERROR
chunk/callback when a deadline expired rather than the work failing. Covers
both flavours — one tool/LLM call overrunning, and the whole turn overrunning —
because a consumer's response to either is the same: offer more time or a
retry, not a bug report. Forwarded verbatim like the values above.

***

<a id="namespace-chunkTypes-api-tool_credential_rejected_error_code"></a>

##### TOOL\_CREDENTIAL\_REJECTED\_ERROR\_CODE

```ts
const TOOL_CREDENTIAL_REJECTED_ERROR_CODE: "tool_credential_rejected" = "tool_credential_rejected";
```

Same carrier when a run fails because a tool call's classified external-API
failure was an auth rejection (AUTH_FAILED): the rejected credential is one
of the project's secrets for that tool, not the LLM key. Also stamped
without `metadata.source` so the code drives client-owned attribution.

## Enumerations

<a id="api-activitykind"></a>

### ActivityKind

ActivityKind categorizes a replayable nondeterministic operation.

#### Generated

from enum mongodb.agentic.workflow.v1.ActivityKind

#### Enumeration Members

| Enumeration Member | Value | Description |
| :------ | :------ | :------ |
| <a id="api-enumeration-member-llm"></a> `LLM` | `1` | **Generated** from enum value: ACTIVITY_KIND_LLM = 1; |
| <a id="api-enumeration-member-memory"></a> `MEMORY` | `3` | **Generated** from enum value: ACTIVITY_KIND_MEMORY = 3; |
| <a id="api-enumeration-member-tool"></a> `TOOL` | `2` | **Generated** from enum value: ACTIVITY_KIND_TOOL = 2; |
| <a id="api-enumeration-member-unspecified"></a> `UNSPECIFIED` | `0` | **Generated** from enum value: ACTIVITY_KIND_UNSPECIFIED = 0; |

***

<a id="api-activityoutcomekind"></a>

### ActivityOutcomeKind

ActivityOutcomeKind describes one durable serial activity outcome.

#### Generated

from enum mongodb.agentic.workflow.v1.ActivityOutcomeKind

#### Enumeration Members

| Enumeration Member | Value | Description |
| :------ | :------ | :------ |
| <a id="api-enumeration-member-completed"></a> `COMPLETED` | `1` | **Generated** from enum value: ACTIVITY_OUTCOME_KIND_COMPLETED = 1; |
| <a id="api-enumeration-member-denied"></a> `DENIED` | `3` | **Generated** from enum value: ACTIVITY_OUTCOME_KIND_DENIED = 3; |
| <a id="api-enumeration-member-failed"></a> `FAILED` | `2` | **Generated** from enum value: ACTIVITY_OUTCOME_KIND_FAILED = 2; |
| <a id="api-enumeration-member-suspended"></a> `SUSPENDED` | `4` | **Generated** from enum value: ACTIVITY_OUTCOME_KIND_SUSPENDED = 4; |
| <a id="api-enumeration-member-unspecified-1"></a> `UNSPECIFIED` | `0` | **Generated** from enum value: ACTIVITY_OUTCOME_KIND_UNSPECIFIED = 0; |

***

<a id="api-messagerole"></a>

### MessageRole

MessageRole identifies the conversational role of a workflow message.

#### Generated

from enum mongodb.agentic.workflow.v1.MessageRole

#### Enumeration Members

| Enumeration Member | Value | Description |
| :------ | :------ | :------ |
| <a id="api-enumeration-member-assistant"></a> `ASSISTANT` | `2` | **Generated** from enum value: MESSAGE_ROLE_ASSISTANT = 2; |
| <a id="api-enumeration-member-system"></a> `SYSTEM` | `4` | **Generated** from enum value: MESSAGE_ROLE_SYSTEM = 4; |
| <a id="api-enumeration-member-tool-1"></a> `TOOL` | `3` | **Generated** from enum value: MESSAGE_ROLE_TOOL = 3; |
| <a id="api-enumeration-member-unspecified-2"></a> `UNSPECIFIED` | `0` | **Generated** from enum value: MESSAGE_ROLE_UNSPECIFIED = 0; |
| <a id="api-enumeration-member-user"></a> `USER` | `1` | **Generated** from enum value: MESSAGE_ROLE_USER = 1; |

***

<a id="api-runtimemode"></a>

### RuntimeMode

Runtime mode for the Runner SDK.

#### Enumeration Members

| Enumeration Member | Value |
| :------ | :------ |
| <a id="api-enumeration-member-aer"></a> `AER` | `"aer"` |
| <a id="api-enumeration-member-tool-2"></a> `TOOL` | `"tool"` |
| <a id="api-enumeration-member-tool_function"></a> `TOOL_FUNCTION` | `"tool_function"` |

***

<a id="api-workflowerrorcode"></a>

### WorkflowErrorCode

WorkflowErrorCode is the stable cross-language workflow failure vocabulary.

#### Generated

from enum mongodb.agentic.workflow.v1.WorkflowErrorCode

#### Enumeration Members

| Enumeration Member | Value | Description |
| :------ | :------ | :------ |
| <a id="api-enumeration-member-conflict"></a> `CONFLICT` | `3` | **Generated** from enum value: WORKFLOW_ERROR_CODE_CONFLICT = 3; |
| <a id="api-enumeration-member-invalid_argument"></a> `INVALID_ARGUMENT` | `1` | **Generated** from enum value: WORKFLOW_ERROR_CODE_INVALID_ARGUMENT = 1; |
| <a id="api-enumeration-member-nondeterministic"></a> `NONDETERMINISTIC` | `5` | **Generated** from enum value: WORKFLOW_ERROR_CODE_NONDETERMINISTIC = 5; |
| <a id="api-enumeration-member-not_found"></a> `NOT_FOUND` | `2` | **Generated** from enum value: WORKFLOW_ERROR_CODE_NOT_FOUND = 2; |
| <a id="api-enumeration-member-outcome_unknown"></a> `OUTCOME_UNKNOWN` | `7` | **Generated** from enum value: WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN = 7; |
| <a id="api-enumeration-member-stale_fence"></a> `STALE_FENCE` | `4` | **Generated** from enum value: WORKFLOW_ERROR_CODE_STALE_FENCE = 4; |
| <a id="api-enumeration-member-unauthorized"></a> `UNAUTHORIZED` | `8` | **Generated** from enum value: WORKFLOW_ERROR_CODE_UNAUTHORIZED = 8; |
| <a id="api-enumeration-member-unspecified-3"></a> `UNSPECIFIED` | `0` | **Generated** from enum value: WORKFLOW_ERROR_CODE_UNSPECIFIED = 0; |
| <a id="api-enumeration-member-version_unavailable"></a> `VERSION_UNAVAILABLE` | `6` | **Generated** from enum value: WORKFLOW_ERROR_CODE_VERSION_UNAVAILABLE = 6; |

## Classes

<a id="api-activitydispatch"></a>

### ActivityDispatch

<a id="api-constructor"></a>

#### Constructor

```ts
new ActivityDispatch(context): ActivityDispatch;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `context` | [`ActivityContext`](#api-activitycontext) |

**Returns**

[`ActivityDispatch`](#api-activitydispatch)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-context"></a> `context` | `readonly` | [`ActivityContext`](#api-activitycontext) |

***

<a id="api-activityreplay"></a>

### ActivityReplay

<a id="api-constructor-1"></a>

#### Constructor

```ts
new ActivityReplay(outcome): ActivityReplay;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `outcome` | [`ActivityOutcome`](#api-activityoutcome) |

**Returns**

[`ActivityReplay`](#api-activityreplay)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-outcome"></a> `outcome` | `readonly` | [`ActivityOutcome`](#api-activityoutcome) |

***

<a id="api-aerserver"></a>

### AERServer

#### Extends

- `BaseServer`

<a id="api-constructor-2"></a>

#### Constructor

```ts
new AERServer(runtime): AERServer;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `runtime` | [`ITenantRuntime`](#api-itenantruntime) |

**Returns**

[`AERServer`](#api-aerserver)

**Inherited from**

```ts
BaseServer.constructor
```

#### Properties

| Property | Modifier | Type | Default value | Inherited from |
| :------ | :------ | :------ | :------ | :------ |
| <a id="api-property-app"></a> `app` | `protected` | \| `FastifyInstance`\<`RawServerDefault`, `IncomingMessage`, `ServerResponse`\<`IncomingMessage`\>, `FastifyBaseLogger`, `FastifyTypeProviderDefault`\> \| `null` | `null` | `BaseServer.app` |
| <a id="api-property-ownercallbackurl"></a> `ownerCallbackUrl` | `public` | `Map`\<`string`, `string`\> | `undefined` | - |
| <a id="api-property-pendingsessionfinish"></a> `pendingSessionFinish` | `public` | `Set`\<`string`\> | `undefined` | - |
| <a id="api-property-runtime"></a> `runtime` | `readonly` | [`ITenantRuntime`](#api-itenantruntime) | `undefined` | `BaseServer.runtime` |

#### Accessors

<a id="api-defaultport"></a>

##### defaultPort

**Get Signature**

```ts
get defaultPort(): number;
```

**Returns**

`number`

**Inherited from**

```ts
BaseServer.defaultPort
```

<a id="api-drainregistry"></a>

##### drainRegistry

**Get Signature**

```ts
get drainRegistry(): DrainRegistry;
```

Execution-scoped drain state for POST /drain; work routes register
against it so a cancelled execution's in-flight work can be stopped.
Lazily created on the prototype so tests that skip the constructor
(`Object.create`) still get a working registry.

**Returns**

`DrainRegistry`

**Inherited from**

```ts
BaseServer.drainRegistry
```

<a id="api-modename"></a>

##### modeName

**Get Signature**

```ts
get modeName(): string;
```

**Returns**

`string`

**Overrides**

```ts
BaseServer.modeName
```

#### Methods

<a id="api-close"></a>

##### close()

```ts
close(): Promise<void>;
```

Gracefully close the underlying Fastify server, draining in-flight
requests and running `onClose` hooks. No-op if the app was never created.

**Returns**

`Promise`\<`void`\>

**Inherited from**

```ts
BaseServer.close
```

<a id="api-createapp"></a>

##### createApp()

```ts
createApp(): FastifyInstance;
```

**Returns**

`FastifyInstance`

**Inherited from**

```ts
BaseServer.createApp
```

<a id="api-gethealthdetails"></a>

##### getHealthDetails()

```ts
getHealthDetails(): Record<string, unknown>;
```

**Returns**

`Record`\<`string`, `unknown`\>

**Overrides**

```ts
BaseServer.getHealthDetails
```

<a id="api-onshutdown"></a>

##### onShutdown()

```ts
onShutdown(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

**Overrides**

```ts
BaseServer.onShutdown
```

<a id="api-onstartup"></a>

##### onStartup()

```ts
onStartup(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

**Overrides**

```ts
BaseServer.onStartup
```

<a id="api-registercommonroutes"></a>

##### registerCommonRoutes()

```ts
protected registerCommonRoutes(app): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `app` | `FastifyInstance` |

**Returns**

`void`

**Inherited from**

```ts
BaseServer.registerCommonRoutes
```

<a id="api-registerroutes"></a>

##### registerRoutes()

```ts
registerRoutes(app): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `app` | `FastifyInstance` |

**Returns**

`void`

**Overrides**

```ts
BaseServer.registerRoutes
```

<a id="api-run"></a>

##### run()

```ts
run(host?, port?): Promise<void>;
```

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `host` | `string` | `"0.0.0.0"` |
| `port?` | `number` | `undefined` |

**Returns**

`Promise`\<`void`\>

**Inherited from**

```ts
BaseServer.run
```

***

<a id="api-appboundcrudclient"></a>

### AppBoundCrudClient

`MemoryCrudClient` over the OE memory proxy.

#### Implements

- `MemoryCrudClient`

<a id="api-constructor-3"></a>

#### Constructor

```ts
new AppBoundCrudClient(): AppBoundCrudClient;
```

**Returns**

[`AppBoundCrudClient`](#api-appboundcrudclient)

#### Methods

<a id="api-createcustom"></a>

##### createCustom()

```ts
createCustom(args): Promise<{
  has_embedding: boolean;
  id: string;
  tags: z.ZodRecord<z.ZodString, z.ZodUnion<readonly [z.ZodString, z.ZodNumber, z.ZodBoolean]>>;
  type: string;
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `content`: `string`; `contextualMetadata?`: `Record`\<`string`, `unknown`\> \| `null`; `memoryType`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; \} |
| `args.content` | `string` |
| `args.contextualMetadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.memoryType` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |

**Returns**

`Promise`\<\{
  `has_embedding`: `boolean`;
  `id`: `string`;
  `tags`: `z.ZodRecord`\<`z.ZodString`, `z.ZodUnion`\<readonly \[`z.ZodString`, `z.ZodNumber`, `z.ZodBoolean`\]\>\>;
  `type`: `string`;
\}\>

**Implementation of**

```ts
MemoryCrudClient.createCustom
```

<a id="api-createepisodic"></a>

##### createEpisodic()

```ts
createEpisodic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  title: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `participants?`: `string`[] \| `null`; `sessionId`: `string`; `summaryText?`: `string` \| `null`; `tags?`: `string`[] \| `null`; `title`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.participants?` | `string`[] \| `null` |
| `args.sessionId` | `string` |
| `args.summaryText?` | `string` \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.title` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `title`: `string`;
\}\>

**Implementation of**

```ts
MemoryCrudClient.createEpisodic
```

<a id="api-createprocedural"></a>

##### createProcedural()

```ts
createProcedural(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  procedure: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `allowedTools?`: `string`[] \| `null`; `compatibility?`: `string` \| `null`; `content`: `string`; `description`: `string`; `extractionSource?`: `string` \| `null`; `license?`: `string` \| `null`; `procedure`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `sourceFormat?`: `string` \| `null`; `sourcePath?`: `string` \| `null`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `updateExisting?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.allowedTools?` | `string`[] \| `null` |
| `args.compatibility?` | `string` \| `null` |
| `args.content` | `string` |
| `args.description` | `string` |
| `args.extractionSource?` | `string` \| `null` |
| `args.license?` | `string` \| `null` |
| `args.procedure` | `string` |
| `args.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.sourceFormat?` | `string` \| `null` |
| `args.sourcePath?` | `string` \| `null` |
| `args.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.triggerConditions?` | `string`[] \| `null` |
| `args.updateExisting?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `procedure`: `string`;
\}\>

**Implementation of**

```ts
MemoryCrudClient.createProcedural
```

<a id="api-createsemantic"></a>

##### createSemantic()

```ts
createSemantic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  label: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `label`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `source?`: `string`; `text`: `string`; `upsert?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.label` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.source?` | `string` |
| `args.text` | `string` |
| `args.upsert?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `label`: `string`;
\}\>

**Implementation of**

```ts
MemoryCrudClient.createSemantic
```

<a id="api-createtaxonomic"></a>

##### createTaxonomic()

```ts
createTaxonomic(args): Promise<{
  acknowledged: boolean;
  domain: string;
  has_embedding: boolean;
  id: string;
  term: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `definition`: `string`; `domain`: `string`; `relatedTerms?`: `string`[] \| `null`; `term`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.definition` | `string` |
| `args.domain` | `string` |
| `args.relatedTerms?` | `string`[] \| `null` |
| `args.term` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `domain`: `string`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `term`: `string`;
\}\>

**Implementation of**

```ts
MemoryCrudClient.createTaxonomic
```

<a id="api-getdistinctdomains"></a>

##### getDistinctDomains()

```ts
getDistinctDomains(args): Promise<string[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `visibility?`: `string` \| `null`; \} |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`string`[]\>

**Implementation of**

```ts
MemoryCrudClient.getDistinctDomains
```

<a id="api-getprocedural"></a>

##### getProcedural()

```ts
getProcedural(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `includeDeleted?`: `boolean`; `procedure?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.includeDeleted?` | `boolean` |
| `args.procedure?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

```ts
MemoryCrudClient.getProcedural
```

<a id="api-getsemantic"></a>

##### getSemantic()

```ts
getSemantic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `label`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.label` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

```ts
MemoryCrudClient.getSemantic
```

<a id="api-gettaxonomic"></a>

##### getTaxonomic()

```ts
getTaxonomic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain`: `string`; `term?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain` | `string` |
| `args.term?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

```ts
MemoryCrudClient.getTaxonomic
```

<a id="api-listepisodic"></a>

##### listEpisodic()

```ts
listEpisodic(args): Promise<unknown[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `limit?`: `number`; `sessionId?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.limit?` | `number` |
| `args.sessionId?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`[]\>

**Implementation of**

```ts
MemoryCrudClient.listEpisodic
```

<a id="api-retrievecustom"></a>

##### retrieveCustom()

```ts
retrieveCustom(args): Promise<{
  count: number;
  results: object[];
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `memoryType`: `string`; `query`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; `topK?`: `number`; \} |
| `args.memoryType` | `string` |
| `args.query` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.topK?` | `number` |

**Returns**

`Promise`\<\{
  `count`: `number`;
  `results`: `object`[];
\}\>

**Implementation of**

```ts
MemoryCrudClient.retrieveCustom
```

***

<a id="api-appboundruntime"></a>

### AppBoundRuntime

`MemoryRuntime` over the OE memory proxy, exposing ambient identity.

#### Implements

- `MemoryRuntime`

<a id="api-constructor-4"></a>

#### Constructor

```ts
new AppBoundRuntime(): AppBoundRuntime;
```

**Returns**

[`AppBoundRuntime`](#api-appboundruntime)

#### Methods

<a id="api-buildcontext"></a>

##### buildContext()

```ts
buildContext(args): Promise<{
  formatted_context: string | Record<string, unknown>[];
  metadata: {
     memory_counts: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodNumber>>;
     timing: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodNumber>>;
     token_count: number;
  };
  selected_memories?: object[] | null;
}>;
```

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `enabledSources?`: `Set`\<`string`\> \| `null`; `maxTokens?`: `number`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} | - |
| `args.enabledSources?` | `Set`\<`string`\> \| `null` | - |
| `args.maxTokens?` | `number` | Optional gross context-construction budget. After retrieval and ranking, the server subtracts a 500-token formatting reserve, then greedily selects whole memory chunks that fit in the remainder. Positive values at or below 500 leave no budget for memories. Values above 500 can still yield empty context when no chunk fits. |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| `args.query` | `string` | - |
| `args.sessionId?` | `string` \| `null` | - |
| `args.topK?` | `number` | - |
| `args.userId?` | `string` \| `null` | - |
| `args.visibility?` | `string` \| `null` | - |

**Returns**

`Promise`\<\{
  `formatted_context`: `string` \| `Record`\<`string`, `unknown`\>[];
  `metadata`: \{
     `memory_counts`: `z.ZodDefault`\<`z.ZodRecord`\<`z.ZodString`, `z.ZodNumber`\>\>;
     `timing`: `z.ZodDefault`\<`z.ZodRecord`\<`z.ZodString`, `z.ZodNumber`\>\>;
     `token_count`: `number`;
  \};
  `selected_memories?`: `object`[] \| `null`;
\}\>

**Implementation of**

```ts
MemoryRuntime.buildContext
```

<a id="api-discoverprocedures"></a>

##### discoverProcedures()

```ts
discoverProcedures(args): Promise<Record<string, unknown>[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `similarityThreshold?`: `number`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.query` | `string` |
| `args.similarityThreshold?` | `number` |
| `args.tags?` | `string`[] \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`Record`\<`string`, `unknown`\>[]\>

**Implementation of**

```ts
MemoryRuntime.discoverProcedures
```

<a id="api-recordturn"></a>

##### recordTurn()

```ts
recordTurn(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  session_id: string;
  turn_seq: number;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content?`: `string` \| `null`; `idempotencyKey?`: `string` \| `null`; `isError?`: `boolean`; `modelName?`: `string` \| `null`; `role`: `string`; `sessionId?`: `string` \| `null`; `toolCallId?`: `string` \| `null`; `toolCalls?`: `Record`\<`string`, `unknown`\>[] \| `null`; `toolName?`: `string` \| `null`; `userId?`: `string` \| `null`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content?` | `string` \| `null` |
| `args.idempotencyKey?` | `string` \| `null` |
| `args.isError?` | `boolean` |
| `args.modelName?` | `string` \| `null` |
| `args.role` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.toolCallId?` | `string` \| `null` |
| `args.toolCalls?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.toolName?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `session_id`: `string`;
  `turn_seq`: `number`;
\}\>

**Implementation of**

```ts
MemoryRuntime.recordTurn
```

<a id="api-requestcontext"></a>

##### requestContext()

```ts
requestContext(): MemoryRequestContext | null;
```

Fresh identity from the runner context; blank fields are normalized by the facade.

**Returns**

`MemoryRequestContext` \| `null`

<a id="api-searchepisodes"></a>

##### searchEpisodes()

```ts
searchEpisodes(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

```ts
MemoryRuntime.searchEpisodes
```

<a id="api-searchsemantic"></a>

##### searchSemantic()

```ts
searchSemantic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

```ts
MemoryRuntime.searchSemantic
```

<a id="api-searchtaxonomic"></a>

##### searchTaxonomic()

```ts
searchTaxonomic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain?`: `string` \| `null`; `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain?` | `string` \| `null` |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

```ts
MemoryRuntime.searchTaxonomic
```

***

<a id="api-attemptheartbeat"></a>

### AttemptHeartbeat

<a id="api-constructor-5"></a>

#### Constructor

```ts
new AttemptHeartbeat(attempt, client): AttemptHeartbeat;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `attempt` | [`AttemptContext`](#api-attemptcontext) |
| `client` | [`WorkflowClient`](#api-workflowclient) |

**Returns**

[`AttemptHeartbeat`](#api-attemptheartbeat)

#### Methods

<a id="api-start"></a>

##### start()

```ts
start(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

<a id="api-stop"></a>

##### stop()

```ts
stop(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

***

<a id="api-contentpolicyotlpspanexporter"></a>

### ContentPolicyOTLPSpanExporter

Wraps an OTLP `SpanExporter`, enforcing `AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`.
Redaction failures fail closed: a batch that can't be safely redacted is
dropped rather than forwarded unredacted, matching the metadata-only
default's privacy-first intent. Transport failures on the wrapped exporter
never throw — they're surfaced only through `resultCallback`.

#### Implements

- `SpanExporter`

<a id="api-constructor-6"></a>

#### Constructor

```ts
new ContentPolicyOTLPSpanExporter(inner, contentCaptureMode?): ContentPolicyOTLPSpanExporter;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `inner` | `SpanExporter` |
| `contentCaptureMode?` | `string` |

**Returns**

[`ContentPolicyOTLPSpanExporter`](#api-contentpolicyotlpspanexporter)

#### Methods

<a id="api-export"></a>

##### export()

```ts
export(spans, resultCallback): void;
```

Called to export sampled ReadableSpans.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `spans` | `ReadableSpan`[] | the list of sampled Spans to be exported. |
| `resultCallback` | (`result`) => `void` | - |

**Returns**

`void`

**Implementation of**

```ts
SpanExporter.export
```

<a id="api-forceflush"></a>

##### forceFlush()

```ts
forceFlush(): Promise<void>;
```

Immediately export all spans

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.forceFlush
```

<a id="api-shutdown"></a>

##### shutdown()

```ts
shutdown(): Promise<void>;
```

Stops the exporter.

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.shutdown
```

***

<a id="api-durableactivitydeniederror"></a>

### DurableActivityDeniedError

#### Extends

- `Error`

<a id="api-constructor-7"></a>

#### Constructor

```ts
new DurableActivityDeniedError(message, options?): DurableActivityDeniedError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`DurableActivityDeniedError`](#api-durableactivitydeniederror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-durableactivityinterrupted"></a>

### DurableActivityInterrupted

#### Extends

- `Error`

<a id="api-constructor-8"></a>

#### Constructor

```ts
new DurableActivityInterrupted(controlFlow): DurableActivityInterrupted;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `controlFlow` | `unknown` |

**Returns**

[`DurableActivityInterrupted`](#api-durableactivityinterrupted)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-controlflow"></a> `controlFlow` | `readonly` | `unknown` |

***

<a id="api-durablememorystate"></a>

### DurableMemoryState

Own durable Memory projection, delivery, and pending user input.

<a id="api-constructor-9"></a>

#### Constructor

```ts
new DurableMemoryState(pendingUserMessage?): DurableMemoryState;
```

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `pendingUserMessage` | `string` \| `null` | `null` |

**Returns**

[`DurableMemoryState`](#api-durablememorystate)

#### Methods

<a id="api-synchronizellm"></a>

##### synchronizeLlm()

```ts
synchronizeLlm(
   client,
   context,
   result,
   userId
): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `client` | `WorkflowMemoryClient` |
| `context` | [`ActivityContext`](#api-activitycontext) |
| `result` | `unknown` |
| `userId` | `string` \| `null` \| `undefined` |

**Returns**

`Promise`\<`void`\>

<a id="api-synchronizetool"></a>

##### synchronizeTool()

```ts
synchronizeTool(
   client,
   context,
   result,
   userId,
   toolCallId,
   toolName
): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `client` | `WorkflowMemoryClient` |
| `context` | [`ActivityContext`](#api-activitycontext) |
| `result` | `unknown` |
| `userId` | `string` \| `null` \| `undefined` |
| `toolCallId` | `string` \| `undefined` |
| `toolName` | `string` |

**Returns**

`Promise`\<`void`\>

***

<a id="api-externalapicallerror"></a>

### ExternalAPICallError

Raised when tool execution fails.

#### Extends

- [`ToolExecutionError`](#api-toolexecutionerror)

<a id="api-constructor-10"></a>

#### Constructor

```ts
new ExternalAPICallError(
   error,
   toolApiError,
   options?
): ExternalAPICallError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `error` | `string` |
| `toolApiError` | \{ `classification`: `string`; `error_code?`: `string` \| `null`; `http_status?`: `number` \| `null`; `provider_type?`: `string` \| `null`; `reason?`: `string` \| `null`; `retryable`: `boolean`; \} |
| `toolApiError.classification` | `string` |
| `toolApiError.error_code?` | `string` \| `null` |
| `toolApiError.http_status?` | `number` \| `null` |
| `toolApiError.provider_type?` | `string` \| `null` |
| `toolApiError.reason?` | `string` \| `null` |
| `toolApiError.retryable?` | `boolean` |
| `options?` | `ErrorOptions` |

**Returns**

[`ExternalAPICallError`](#api-externalapicallerror)

**Overrides**

[`ToolExecutionError`](#api-toolexecutionerror).[`constructor`](#api-constructor-35)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-error"></a> `error` | `readonly` | `string` | [`ToolExecutionError`](#api-toolexecutionerror).[`error`](#api-property-error-4) |
| <a id="api-property-tool_api_error"></a> `tool_api_error` | `readonly` | `object` | - |
| `tool_api_error.classification` | `public` | `string` | - |
| `tool_api_error.error_code?` | `public` | `string` \| `null` | - |
| `tool_api_error.http_status?` | `public` | `number` \| `null` | - |
| `tool_api_error.provider_type?` | `public` | `string` \| `null` | - |
| `tool_api_error.reason?` | `public` | `string` \| `null` | - |
| `tool_api_error.retryable` | `public` | `boolean` | - |

***

<a id="api-jsonlspanexporter"></a>

### JSONLSpanExporter

#### Implements

- `SpanExporter`

<a id="api-constructor-11"></a>

#### Constructor

```ts
new JSONLSpanExporter(path?): JSONLSpanExporter;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `path?` | `string` |

**Returns**

[`JSONLSpanExporter`](#api-jsonlspanexporter)

#### Methods

<a id="api-export-1"></a>

##### export()

```ts
export(spans, resultCallback): void;
```

Called to export sampled ReadableSpans.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `spans` | `ReadableSpan`[] | the list of sampled Spans to be exported. |
| `resultCallback` | (`result`) => `void` | - |

**Returns**

`void`

**Implementation of**

```ts
SpanExporter.export
```

<a id="api-forceflush-1"></a>

##### forceFlush()

```ts
forceFlush(): Promise<void>;
```

Immediately export all spans

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.forceFlush
```

<a id="api-shutdown-1"></a>

##### shutdown()

```ts
shutdown(): Promise<void>;
```

Stops the exporter.

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.shutdown
```

***

<a id="api-llminvocationerror"></a>

### LLMInvocationError

#### Extends

- `Error`

<a id="api-constructor-12"></a>

#### Constructor

```ts
new LLMInvocationError(error, options?): LLMInvocationError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `error` | `string` |
| `options?` | `ErrorOptions` & `object` |

**Returns**

[`LLMInvocationError`](#api-llminvocationerror)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-error-1"></a> `error` | `readonly` | `string` | - |
| <a id="api-property-error_code"></a> `error_code?` | `readonly` | `string` | Machine-readable classification the tool pod stamped on the failure (e.g. a provider credential rejection), when it did. Travels to the OE/UI on the ERROR chunk metadata instead of the generic invocation code so consumers can classify without string-matching prose. |
| <a id="api-property-source"></a> `source?` | `readonly` | `string` | Invoke-owner attribution. Only `"llm"` means the provider failed. |

***

<a id="api-llminvocationoptions"></a>

### LLMInvocationOptions

Explicit provider/model options passed with an LLM invocation.

<a id="api-constructor-13"></a>

#### Constructor

```ts
new LLMInvocationOptions(data): LLMInvocationOptions;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ \[`key`: `string`\]: `unknown`; `frequency_penalty?`: `number`; `max_tokens?`: `number`; `parallel_tool_calls?`: `boolean`; `presence_penalty?`: `number`; `reasoning_effort?`: `string`; `response_format?`: [`JsonValue`](#api-jsonvalue); `seed?`: `number`; `timeout?`: `number`; `top_k?`: `number`; `top_p?`: `number`; \} |
| `data.frequency_penalty?` | `number` |
| `data.max_tokens?` | `number` |
| `data.parallel_tool_calls?` | `boolean` |
| `data.presence_penalty?` | `number` |
| `data.reasoning_effort?` | `string` |
| `data.response_format?` | [`JsonValue`](#api-jsonvalue) |
| `data.seed?` | `number` |
| `data.timeout?` | `number` |
| `data.top_k?` | `number` |
| `data.top_p?` | `number` |

**Returns**

[`LLMInvocationOptions`](#api-llminvocationoptions)

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-extras"></a> `extras` | `readonly` | `Record`\<`string`, `unknown`\> | Provider-specific kwargs not in the standard field set (e.g. temperature). Mirrors Python's extra="allow". |
| <a id="api-property-frequencypenalty"></a> `frequencyPenalty?` | `readonly` | `number` | - |
| <a id="api-property-maxtokens"></a> `maxTokens?` | `readonly` | `number` | - |
| <a id="api-property-paralleltoolcalls"></a> `parallelToolCalls?` | `readonly` | `boolean` | - |
| <a id="api-property-presencepenalty"></a> `presencePenalty?` | `readonly` | `number` | - |
| <a id="api-property-reasoningeffort"></a> `reasoningEffort?` | `readonly` | `string` | - |
| <a id="api-property-responseformat"></a> `responseFormat?` | `readonly` | [`JsonValue`](#api-jsonvalue) | - |
| <a id="api-property-seed"></a> `seed?` | `readonly` | `number` | - |
| <a id="api-property-timeout"></a> `timeout?` | `readonly` | `number` | - |
| <a id="api-property-topk"></a> `topK?` | `readonly` | `number` | - |
| <a id="api-property-topp"></a> `topP?` | `readonly` | `number` | - |

#### Methods

<a id="api-tojson"></a>

##### toJSON()

```ts
toJSON(): Record<string, unknown>;
```

Serializes to snake_case wire format so JSON.stringify round-trips through LLMInvocationOptionsSchema.

**Returns**

`Record`\<`string`, `unknown`\>

<a id="api-tomodelkwargs"></a>

##### toModelKwargs()

```ts
toModelKwargs(): Record<string, unknown>;
```

Converts to kwargs for underlying model invocation. Mirrors Python's model_dump(exclude_none=True) with extra="allow".

**Returns**

`Record`\<`string`, `unknown`\>

***

<a id="api-llmregistryloaderror"></a>

### LLMRegistryLoadError

Thrown when an LLM lookup fails because the entrypoint registry load failed.

The named-LLM registry is populated by running the user entrypoint. When that
run throws, the registry is left empty (or holding only import-time
registrations) and every lookup of an LLM the entrypoint would have
registered fails with a bare "not registered" error -- which points the agent
developer at their `app.llm()` calls instead of at the transient cause that
actually broke the load. The real cause is named in the message only; it is
deliberately not attached as `cause`, because the caller-facing response is
rendered by `formatLlmError`, which walks `cause` and would dump a raw
provider `.details`/`.body` from there, bypassing the message redaction.
Mirrors Python's LLMRegistryLoadError.

#### Extends

- `Error`

<a id="api-constructor-14"></a>

#### Constructor

```ts
new LLMRegistryLoadError(message, options?): LLMRegistryLoadError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`LLMRegistryLoadError`](#api-llmregistryloaderror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-llmresponse"></a>

### LLMResponse

Framework-neutral LLM response.

<a id="api-constructor-15"></a>

#### Constructor

```ts
new LLMResponse(data): LLMResponse;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `additionalKwargs?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `content`: `string`; `id?`: `string`; `metadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `name?`: `string`; `responseMetadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `toolCalls?`: [`LLMToolCall`](#api-llmtoolcall)[]; `usage?`: [`LLMTokenUsage`](#api-llmtokenusage); \} |
| `data.additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.content` | `string` |
| `data.id?` | `string` |
| `data.metadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.name?` | `string` |
| `data.responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.toolCalls?` | [`LLMToolCall`](#api-llmtoolcall)[] |
| `data.usage?` | [`LLMTokenUsage`](#api-llmtokenusage) |

**Returns**

[`LLMResponse`](#api-llmresponse)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-additionalkwargs"></a> `additionalKwargs?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content"></a> `content` | `readonly` | `string` |
| <a id="api-property-id"></a> `id?` | `readonly` | `string` |
| <a id="api-property-metadata"></a> `metadata` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-name"></a> `name?` | `readonly` | `string` |
| <a id="api-property-responsemetadata"></a> `responseMetadata?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-toolcalls"></a> `toolCalls?` | `readonly` | [`LLMToolCall`](#api-llmtoolcall)[] |
| <a id="api-property-usage"></a> `usage?` | `readonly` | [`LLMTokenUsage`](#api-llmtokenusage) |

#### Methods

<a id="api-fromraw"></a>

##### fromRaw()

```ts
static fromRaw(data): LLMResponse;
```

Constructs an LLMResponse from a raw provider dict, normalizing usage
from either a top-level 'usage' key or token keys embedded in 'metadata'.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | `Record`\<`string`, `unknown`\> |

**Returns**

[`LLMResponse`](#api-llmresponse)

***

<a id="api-llmresult"></a>

### LLMResult

Normalized result from an LLM invocation.

Provides a consistent shape for LLM responses regardless of the
underlying provider (OpenAI, Gemini, Anthropic, etc.). Implemented as
a class to mirror `LLMResponse` in agent-engine-sdk and to host the
`fromResponse` / `toResponse` / `extractUsage` conversion methods.

<a id="api-constructor-16"></a>

#### Constructor

```ts
new LLMResult(data): LLMResult;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `additionalKwargs?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `content?`: `string`; `id?`: `string`; `metadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `name?`: `string`; `responseMetadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `toolCalls?`: [`LLMToolCall`](#api-llmtoolcall)[]; `usage?`: [`LLMTokenUsage`](#api-llmtokenusage); \} |
| `data.additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.content?` | `string` |
| `data.id?` | `string` |
| `data.metadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.name?` | `string` |
| `data.responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.toolCalls?` | [`LLMToolCall`](#api-llmtoolcall)[] |
| `data.usage?` | [`LLMTokenUsage`](#api-llmtokenusage) |

**Returns**

[`LLMResult`](#api-llmresult)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-additionalkwargs-1"></a> `additionalKwargs?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content-1"></a> `content` | `readonly` | `string` |
| <a id="api-property-id-1"></a> `id?` | `readonly` | `string` |
| <a id="api-property-metadata-1"></a> `metadata` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-name-1"></a> `name?` | `readonly` | `string` |
| <a id="api-property-responsemetadata-1"></a> `responseMetadata?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-toolcalls-1"></a> `toolCalls` | `readonly` | [`LLMToolCall`](#api-llmtoolcall)[] |
| <a id="api-property-usage-1"></a> `usage?` | `readonly` | [`LLMTokenUsage`](#api-llmtokenusage) |

#### Methods

<a id="api-toresponse"></a>

##### toResponse()

```ts
toResponse(): LLMResponse;
```

Convert the normalized runner result to sdk-core's LLMResponse shape.

**Returns**

[`LLMResponse`](#api-llmresponse)

<a id="api-extractusage"></a>

##### extractUsage()

```ts
static extractUsage(response): LLMTokenUsage | undefined;
```

Extract token usage metadata from any response or chunk.

Checks (in order): sdk-core `LLMResponse.usage`, `usage_metadata`,
`response_metadata.usage`, `response_metadata.token_usage`,
`usage`, `metadata.usage`.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `response` | `unknown` |

**Returns**

[`LLMTokenUsage`](#api-llmtokenusage) \| `undefined`

<a id="api-fromresponse"></a>

##### fromResponse()

```ts
static fromResponse(response): LLMResult;
```

Extract a normalized result from any LLM response object.

Accepts an sdk-core `LLMResponse` directly, or a duck-typed object
(e.g. raw provider response) by falling back to string-coerced
content extraction.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `response` | `unknown` |

**Returns**

[`LLMResult`](#api-llmresult)

***

<a id="api-llmtokenusage"></a>

### LLMTokenUsage

Typed token-usage metadata for LLM calls.

<a id="api-constructor-17"></a>

#### Constructor

```ts
new LLMTokenUsage(data): LLMTokenUsage;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `completion_tokens?`: `number`; `input_tokens?`: `number`; `model?`: `string`; `output_tokens?`: `number`; `prompt_tokens?`: `number`; `total_tokens?`: `number`; \} |
| `data.completion_tokens?` | `number` |
| `data.input_tokens?` | `number` |
| `data.model?` | `string` |
| `data.output_tokens?` | `number` |
| `data.prompt_tokens?` | `number` |
| `data.total_tokens?` | `number` |

**Returns**

[`LLMTokenUsage`](#api-llmtokenusage)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-completiontokens"></a> `completionTokens?` | `readonly` | `number` |
| <a id="api-property-inputtokens"></a> `inputTokens?` | `readonly` | `number` |
| <a id="api-property-model"></a> `model?` | `readonly` | `string` |
| <a id="api-property-outputtokens"></a> `outputTokens?` | `readonly` | `number` |
| <a id="api-property-prompttokens"></a> `promptTokens?` | `readonly` | `number` |
| <a id="api-property-totaltokens"></a> `totalTokens?` | `public` | `number` |

#### Methods

<a id="api-tojson-1"></a>

##### toJSON()

```ts
toJSON(): Record<string, unknown>;
```

Serializes to snake_case wire format so JSON.stringify round-trips through LLMTokenUsageSchema.

**Returns**

`Record`\<`string`, `unknown`\>

<a id="api-tolangchainusagemetadata"></a>

##### toLangchainUsageMetadata()

```ts
toLangchainUsageMetadata(): object;
```

Returns LangChain's expected usage-metadata keys.

**Returns**

`object`

| Name | Type |
| :------ | :------ |
| `input_tokens` | `number` |
| `output_tokens` | `number` |
| `total_tokens` | `number` |

***

<a id="api-llmtoolcall"></a>

### LLMToolCall

Typed final tool call requested by an LLM.

<a id="api-constructor-18"></a>

#### Constructor

```ts
new LLMToolCall(data): LLMToolCall;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `args?`: [`JsonValue`](#api-jsonvalue); `arguments?`: [`JsonValue`](#api-jsonvalue); `id?`: `string`; `index?`: `number`; `name?`: `string`; `type?`: `string`; \} |
| `data.args?` | [`JsonValue`](#api-jsonvalue) |
| `data.arguments?` | [`JsonValue`](#api-jsonvalue) |
| `data.id?` | `string` |
| `data.index?` | `number` |
| `data.name?` | `string` |
| `data.type?` | `string` |

**Returns**

[`LLMToolCall`](#api-llmtoolcall)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-args"></a> `args?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-id-2"></a> `id?` | `readonly` | `string` |
| <a id="api-property-index"></a> `index?` | `readonly` | `number` |
| <a id="api-property-name-2"></a> `name?` | `readonly` | `string` |
| <a id="api-property-type"></a> `type?` | `readonly` | `string` |

#### Methods

<a id="api-tolangchaindict"></a>

##### toLangchainDict()

```ts
toLangchainDict(): Record<string, JsonValue>;
```

Returns a LangChain-compatible tool-call dict (excludes index).

**Returns**

`Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>

***

<a id="api-llmtoolschema"></a>

### LLMToolSchema

Serializable bound-tool schema forwarded with an LLM call.

<a id="api-constructor-19"></a>

#### Constructor

```ts
new LLMToolSchema(data): LLMToolSchema;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `description?`: `string`; `function?`: [`JsonValue`](#api-jsonvalue); `name?`: `string`; `parameters?`: [`JsonValue`](#api-jsonvalue); `strict?`: `boolean`; `type?`: `string`; \} |
| `data.description?` | `string` |
| `data.function?` | [`JsonValue`](#api-jsonvalue) |
| `data.name?` | `string` |
| `data.parameters?` | [`JsonValue`](#api-jsonvalue) |
| `data.strict?` | `boolean` |
| `data.type?` | `string` |

**Returns**

[`LLMToolSchema`](#api-llmtoolschema)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-description"></a> `description?` | `readonly` | `string` |
| <a id="api-property-function"></a> `function?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-name-3"></a> `name?` | `readonly` | `string` |
| <a id="api-property-parameters"></a> `parameters?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-strict"></a> `strict?` | `readonly` | `boolean` |
| <a id="api-property-type-1"></a> `type?` | `readonly` | `string` |

#### Methods

<a id="api-tolangchaindict-1"></a>

##### toLangchainDict()

```ts
toLangchainDict(): Record<string, JsonValue>;
```

Returns the schema in the shape LangChain bind_tools expects.

**Returns**

`Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>

***

<a id="api-mcpconfigerror"></a>

### MCPConfigError

Raised when remote MCP configuration cannot be resolved.

#### Extends

- `Error`

<a id="api-constructor-20"></a>

#### Constructor

```ts
new MCPConfigError(message): MCPConfigError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |

**Returns**

[`MCPConfigError`](#api-mcpconfigerror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-mcptoolerror"></a>

### MCPToolError

Raised when a remote MCP tool returns an MCP error result or a transport failure occurs.

#### Extends

- `Error`

<a id="api-constructor-21"></a>

#### Constructor

```ts
new MCPToolError(message): MCPToolError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |

**Returns**

[`MCPToolError`](#api-mcptoolerror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-metrics"></a>

### Metrics

<a id="api-constructor-22"></a>

#### Constructor

```ts
new Metrics(): Metrics;
```

**Returns**

[`Metrics`](#api-metrics)

#### Methods

<a id="api-getall"></a>

##### getAll()

```ts
static getAll(): object;
```

**Returns**

`object`

| Name | Type |
| :------ | :------ |
| `errors` | `Record`\<`string`, `number`\> |
| `latencies` | `Record`\<`string`, [`LatencyStatsJson`](#api-latencystatsjson)\> |
| `requests` | `Record`\<`string`, `number`\> |

<a id="api-recorderror"></a>

##### recordError()

```ts
static recordError(operation, labels?): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `operation` | `string` |
| `labels` | `Record`\<`string`, `string`\> |

**Returns**

`void`

<a id="api-recordlatency"></a>

##### recordLatency()

```ts
static recordLatency(
   operation,
   durationMs,
   labels?
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `operation` | `string` |
| `durationMs` | `number` |
| `labels` | `Record`\<`string`, `string`\> |

**Returns**

`void`

<a id="api-reset"></a>

##### reset()

```ts
static reset(): void;
```

**Returns**

`void`

***

<a id="api-mongodbspanexporter"></a>

### MongoDBSpanExporter

#### Implements

- `SpanExporter`

<a id="api-constructor-23"></a>

#### Constructor

```ts
new MongoDBSpanExporter(collection): MongoDBSpanExporter;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `collection` | `Collection` |

**Returns**

[`MongoDBSpanExporter`](#api-mongodbspanexporter)

#### Methods

<a id="api-export-2"></a>

##### export()

```ts
export(spans, resultCallback): void;
```

Called to export sampled ReadableSpans.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `spans` | `ReadableSpan`[] | the list of sampled Spans to be exported. |
| `resultCallback` | (`result`) => `void` | - |

**Returns**

`void`

**Implementation of**

```ts
SpanExporter.export
```

<a id="api-forceflush-2"></a>

##### forceFlush()

```ts
forceFlush(): Promise<void>;
```

Immediately export all spans

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.forceFlush
```

<a id="api-shutdown-2"></a>

##### shutdown()

```ts
shutdown(): Promise<void>;
```

Stops the exporter.

**Returns**

`Promise`\<`void`\>

**Implementation of**

```ts
SpanExporter.shutdown
```

***

<a id="api-nodeexecutionlogger"></a>

### NodeExecutionLogger

#### Extends

- `NullExecutionCallback`

<a id="api-constructor-24"></a>

#### Constructor

```ts
new NodeExecutionLogger(opts): NodeExecutionLogger;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts` | [`NodeExecutionLoggerOpts`](#api-nodeexecutionloggeropts) |

**Returns**

[`NodeExecutionLogger`](#api-nodeexecutionlogger)

**Overrides**

```ts
NullExecutionCallback.constructor
```

#### Methods

<a id="api-onnodeend"></a>

##### onNodeEnd()

```ts
onNodeEnd(
   nodeName,
   outputs,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `outputs` | `Record`\<`string`, `unknown`\> |
| `opts` | `NodeCallbackOpts` |

**Returns**

`void`

**Overrides**

```ts
NullExecutionCallback.onNodeEnd
```

<a id="api-onnodeerror"></a>

##### onNodeError()

```ts
onNodeError(
   nodeName,
   error,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `error` | `string` |
| `opts` | `NodeCallbackOpts` |

**Returns**

`void`

**Overrides**

```ts
NullExecutionCallback.onNodeError
```

<a id="api-onnodestart"></a>

##### onNodeStart()

```ts
onNodeStart(
   nodeName,
   inputs,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `inputs` | `Record`\<`string`, `unknown`\> |
| `opts` | `NodeCallbackOpts` |

**Returns**

`void`

**Overrides**

```ts
NullExecutionCallback.onNodeStart
```

<a id="api-onnodesuspend"></a>

##### onNodeSuspend()

```ts
onNodeSuspend(nodeName, opts): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `opts` | `NodeCallbackOpts` |

**Returns**

`void`

**Overrides**

```ts
NullExecutionCallback.onNodeSuspend
```

***

<a id="api-operationalstepallocator"></a>

### OperationalStepAllocator

Process-local operational step_number mint for one AER execution.

Shared by SecureToolWrapper and SecureLLMProxy so parallel tools and LLM
calls cannot collide under a single AER. Not a multi-OE authority.
Node's single-threaded event loop makes a plain counter
safe across concurrent async tasks as long as minting stays synchronous
before the first await.

#### Implements

- [`OperationalStepSource`](#api-operationalstepsource)

<a id="api-constructor-25"></a>

#### Constructor

```ts
new OperationalStepAllocator(): OperationalStepAllocator;
```

**Returns**

[`OperationalStepAllocator`](#api-operationalstepallocator)

#### Methods

<a id="api-current"></a>

##### current()

```ts
current(): number;
```

**Returns**

`number`

**Implementation of**

[`OperationalStepSource`](#api-operationalstepsource).[`current`](#api-current-1)

<a id="api-next"></a>

##### next()

```ts
next(): number;
```

**Returns**

`number`

**Implementation of**

[`OperationalStepSource`](#api-operationalstepsource).[`next`](#api-next-1)

<a id="api-observeatleast"></a>

##### observeAtLeast()

```ts
observeAtLeast(n): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `n` | `number` |

**Returns**

`void`

**Implementation of**

[`OperationalStepSource`](#api-operationalstepsource).[`observeAtLeast`](#api-observeatleast-1)

***

<a id="api-outputvalidationpolicyengine"></a>

### OutputValidationPolicyEngine

Contract implemented by native and future remote guardrail engines.

#### Implements

- [`GuardrailPolicyEngine`](#api-guardrailpolicyengine)

<a id="api-constructor-26"></a>

#### Constructor

```ts
new OutputValidationPolicyEngine(): OutputValidationPolicyEngine;
```

**Returns**

[`OutputValidationPolicyEngine`](#api-outputvalidationpolicyengine)

#### Properties

| Property | Modifier | Type | Default value |
| :------ | :------ | :------ | :------ |
| <a id="api-property-policytype"></a> `policyType` | `readonly` | `"output_validation"` | `OUTPUT_VALIDATION_POLICY_TYPE` |

#### Methods

<a id="api-evaluate"></a>

##### evaluate()

```ts
evaluate(policy, text):
  | GuardrailPolicyEngineResult
  | null;
```

Return engine-owned evidence and optional transformed text when triggered.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `policy` | \{ `action`: `string`; `config`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `id`: `string`; `stage_filter`: `string`[]; `status`: `string`; `type`: `string`; \} | - |
| `policy.action` | `string` | Action requested when the policy triggers. |
| `policy.config` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | Evaluator-specific policy configuration. |
| `policy.id` | `string` | Guardrail policy identifier. |
| `policy.stage_filter` | `string`[] | Runtime stages this policy applies to; empty means all stages. |
| `policy.status` | `string` | Guardrail policy status. |
| `policy.type` | `string` | Guardrail policy type. |
| `text` | `string` | - |

**Returns**

  \| [`GuardrailPolicyEngineResult`](#api-guardrailpolicyengineresult)
  \| `null`

**Implementation of**

[`GuardrailPolicyEngine`](#api-guardrailpolicyengine).[`evaluate`](#api-evaluate-1)

***

<a id="api-policydeniedexception"></a>

### PolicyDeniedException

#### Extends

- `Error`

<a id="api-constructor-27"></a>

#### Constructor

```ts
new PolicyDeniedException(
   reason,
   guardrailMeta?,
   options?
): PolicyDeniedException;
```

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `reason` | `string` | `undefined` |
| `guardrailMeta` | \| \{ `guardrail_category`: `string`; `guardrail_id`: `string`; \} \| `null` | `null` |
| `options?` | `ErrorOptions` | `undefined` |

**Returns**

[`PolicyDeniedException`](#api-policydeniedexception)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-guardrailmeta"></a> `guardrailMeta` | `readonly` | \| \{ `guardrail_category`: `string`; `guardrail_id`: `string`; \} \| `null` | Guardrail policy identity when the denial came from a guardrail policy. |
| <a id="api-property-reason"></a> `reason` | `readonly` | `string` | - |

***

<a id="api-replayedactivityfailederror"></a>

### ReplayedActivityFailedError

#### Extends

- [`WorkflowClientError`](#api-workflowclienterror)

<a id="api-constructor-28"></a>

#### Constructor

```ts
new ReplayedActivityFailedError(code, message): ReplayedActivityFailedError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `code` | [`WorkflowErrorCode`](#api-workflowerrorcode) |
| `message` | `string` |

**Returns**

[`ReplayedActivityFailedError`](#api-replayedactivityfailederror)

**Overrides**

[`WorkflowClientError`](#api-workflowclienterror).[`constructor`](#api-constructor-40)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code"></a> `code` | `readonly` | [`WorkflowErrorCode`](#api-workflowerrorcode) | [`WorkflowClientError`](#api-workflowclienterror).[`code`](#api-property-code-1) |

***

<a id="api-runtimeagentconfig"></a>

### RuntimeAgentConfig

Validated runtime view of `agent.yaml`.

Class-based because Python uses computed `@property` accessors and helper
methods (`feature_enabled`, `configured_feature`). Properties are exposed as
TS getters under camelCase names; Python field names on
`AgentFeatureConfig` are preserved verbatim so wire/log parity is
maintained.

<a id="api-constructor-29"></a>

#### Constructor

```ts
new RuntimeAgentConfig(init?): RuntimeAgentConfig;
```

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `init` | \{ `entrypoint?`: `string` \| `null`; `features?`: \{ `deep_agent`: `boolean` \| `null`; `durable_workflow`: `boolean` \| `null`; `guardrails`: `boolean` \| `null`; `memory`: `boolean` \| `null`; `playground`: `boolean` \| `null`; `use_custom_parser`: `boolean` \| `null`; \}; `framework?`: `string` \| `null`; `language?`: `string` \| `null`; `mcp?`: \{ `servers`: `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\>; \}; `path?`: `string` \| `null`; `requiredSecrets?`: \{ `aer?`: `string`[]; `disable_restriction?`: `boolean`; `tools?`: `Record`\<`string`, `string`[]\>; \}; \} | - |
| `init.entrypoint?` | `string` \| `null` | - |
| `init.features?` | \{ `deep_agent`: `boolean` \| `null`; `durable_workflow`: `boolean` \| `null`; `guardrails`: `boolean` \| `null`; `memory`: `boolean` \| `null`; `playground`: `boolean` \| `null`; `use_custom_parser`: `boolean` \| `null`; \} | - |
| `init.features.deep_agent` | `boolean` \| `null` | Whether the agent uses the deep-agent (deepagents) harness. Gates the Tool Pod's built-in filesystem + shell handler registration so tenants that don't run deep agents get no filesystem/shell surface on their Tool Pod. The key stays snake_case `deep_agent` because `agent.yaml` is a cross-language artifact shared with the Python runtime and platform. |
| `init.features.durable_workflow` | `boolean` \| `null` | Opt-in for OE-owned durable workflow. Omitted or false means the agent stays on native checkpoints; only an explicit true opts in. When set, OE uses this flag with the advertised language to sticky-assign the session's workflow authority. |
| `init.features.guardrails` | `boolean` \| `null` | Whether guardrails are enforced for this agent at runtime. `null` (the default) means omitted, letting the runtime fall back to legacy behavior. |
| `init.features.memory` | `boolean` \| `null` | - |
| `init.features.playground` | `boolean` \| `null` | Whether the platform provisions playground UI for the agent (null/true = provisioned, today's behavior). When false — e.g. for non-chat agents with no conversation to preview — no playground is built or served; callers use the invoke API directly. Read at deploy/provisioning time only — no runtime effect. |
| `init.features.use_custom_parser` | `boolean` \| `null` | Opt-in for author-defined streaming output shaping. When true the adapter runs the registered output parser and emits `custom_event` frames; off leaves the stream unchanged. |
| `init.framework?` | `string` \| `null` | - |
| `init.language?` | `string` \| `null` | - |
| `init.mcp?` | \{ `servers`: `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\>; \} | - |
| `init.mcp.servers` | `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\> | - |
| `init.path?` | `string` \| `null` | - |
| `init.requiredSecrets?` | \{ `aer?`: `string`[]; `disable_restriction?`: `boolean`; `tools?`: `Record`\<`string`, `string`[]\>; \} | - |
| `init.requiredSecrets.aer?` | `string`[] | - |
| `init.requiredSecrets.disable_restriction?` | `boolean` | - |
| `init.requiredSecrets.tools?` | `Record`\<`string`, `string`[]\> | - |

**Returns**

[`RuntimeAgentConfig`](#api-runtimeagentconfig)

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-entrypoint"></a> `entrypoint` | `readonly` | `string` \| `null` | - |
| <a id="api-property-features"></a> `features` | `readonly` | `object` | - |
| `features.deep_agent` | `public` | `boolean` \| `null` | Whether the agent uses the deep-agent (deepagents) harness. Gates the Tool Pod's built-in filesystem + shell handler registration so tenants that don't run deep agents get no filesystem/shell surface on their Tool Pod. The key stays snake_case `deep_agent` because `agent.yaml` is a cross-language artifact shared with the Python runtime and platform. |
| `features.durable_workflow` | `public` | `boolean` \| `null` | Opt-in for OE-owned durable workflow. Omitted or false means the agent stays on native checkpoints; only an explicit true opts in. When set, OE uses this flag with the advertised language to sticky-assign the session's workflow authority. |
| `features.guardrails` | `public` | `boolean` \| `null` | Whether guardrails are enforced for this agent at runtime. `null` (the default) means omitted, letting the runtime fall back to legacy behavior. |
| `features.memory` | `public` | `boolean` \| `null` | - |
| `features.playground` | `public` | `boolean` \| `null` | Whether the platform provisions playground UI for the agent (null/true = provisioned, today's behavior). When false — e.g. for non-chat agents with no conversation to preview — no playground is built or served; callers use the invoke API directly. Read at deploy/provisioning time only — no runtime effect. |
| `features.use_custom_parser` | `public` | `boolean` \| `null` | Opt-in for author-defined streaming output shaping. When true the adapter runs the registered output parser and emits `custom_event` frames; off leaves the stream unchanged. |
| <a id="api-property-framework"></a> `framework` | `readonly` | `string` \| `null` | Application framework from agent.yaml; omitted means unset at advertise time. |
| <a id="api-property-language"></a> `language` | `readonly` | `string` \| `null` | Runtime language from agent.yaml; omitted means unset at advertise time. |
| <a id="api-property-mcp"></a> `mcp` | `readonly` | `object` | - |
| `mcp.servers` | `public` | `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\> | - |
| <a id="api-property-path"></a> `path` | `readonly` | `string` \| `null` | - |
| <a id="api-property-requiredsecrets"></a> `requiredSecrets` | `readonly` | `object` | - |
| `requiredSecrets.aer` | `public` | `string`[] | - |
| `requiredSecrets.disable_restriction` | `public` | `boolean` | - |
| `requiredSecrets.tools` | `public` | `Record`\<`string`, `string`[]\> | - |

#### Methods

<a id="api-configuredfeature"></a>

##### configuredFeature()

```ts
configuredFeature(name): boolean | null;
```

Return the explicit feature value from `agent.yaml`, if it exists.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `name` | \| `"memory"` \| `"guardrails"` \| `"playground"` \| `"deep_agent"` \| `"use_custom_parser"` \| `"durable_workflow"` |

**Returns**

`boolean` \| `null`

<a id="api-featureenabled"></a>

##### featureEnabled()

```ts
featureEnabled(name, defaultValue?): boolean;
```

Return a feature flag value, falling back to `defaultValue` when omitted.

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `name` | \| `"memory"` \| `"guardrails"` \| `"playground"` \| `"deep_agent"` \| `"use_custom_parser"` \| `"durable_workflow"` | `undefined` |
| `defaultValue` | `boolean` | `false` |

**Returns**

`boolean`

***

<a id="api-securellmproxy"></a>

### SecureLLMProxy

<a id="api-constructor-30"></a>

#### Constructor

```ts
new SecureLLMProxy(args): SecureLLMProxy;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `boundToolChoice?`: `unknown`; `boundTools?`: `unknown`[] \| `null`; `durableMemory?`: [`DurableMemoryState`](#api-durablememorystate) \| `null`; `executionId`: `string`; `llmId?`: `string`; `modelName?`: `string`; `oeUrl`: `string`; `operationalSteps?`: [`OperationalStepSource`](#api-operationalstepsource); \} |
| `args.boundToolChoice?` | `unknown` |
| `args.boundTools?` | `unknown`[] \| `null` |
| `args.durableMemory?` | [`DurableMemoryState`](#api-durablememorystate) \| `null` |
| `args.executionId` | `string` |
| `args.llmId?` | `string` |
| `args.modelName?` | `string` |
| `args.oeUrl` | `string` |
| `args.operationalSteps?` | [`OperationalStepSource`](#api-operationalstepsource) |

**Returns**

[`SecureLLMProxy`](#api-securellmproxy)

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-boundtoolchoice"></a> `boundToolChoice` | `readonly` | `unknown` | - |
| <a id="api-property-boundtools"></a> `boundTools` | `readonly` | `unknown`[] \| `null` | - |
| <a id="api-property-durablememory"></a> `durableMemory` | `readonly` | [`DurableMemoryState`](#api-durablememorystate) \| `null` | - |
| <a id="api-property-executionid"></a> `executionId` | `readonly` | `string` | - |
| <a id="api-property-lastdurationms"></a> `lastDurationMs` | `public` | `number` | - |
| <a id="api-property-lastfromcache"></a> `lastFromCache` | `public` | `boolean` | - |
| <a id="api-property-lastlateststepnumber"></a> `lastLatestStepNumber` | `public` | `number` \| `null` | - |
| <a id="api-property-lastpodname"></a> `lastPodName` | `public` | `string` \| `null` | - |
| <a id="api-property-llmid"></a> `llmId` | `readonly` | `string` | - |
| <a id="api-property-modelname"></a> `modelName` | `readonly` | `string` | - |
| <a id="api-property-oeurl"></a> `oeUrl` | `readonly` | `string` | - |
| <a id="api-property-operationalsteps"></a> `operationalSteps` | `readonly` | [`OperationalStepSource`](#api-operationalstepsource) | Prefer the wrapper's allocator so tool + LLM share one sequence. |

#### Accessors

<a id="api-stepcounter"></a>

##### stepCounter

**Get Signature**

```ts
get stepCounter(): number;
```

Current operational-step watermark for compatibility readers.

**Returns**

`number`

#### Methods

<a id="api-invoke"></a>

##### invoke()

```ts
invoke(
   messages,
   step?,
   stop?,
   options?
): Promise<LLMResponse>;
```

Invoke LLM by collecting the stream-oriented execution path.
Equivalent to Python's `invoke()` which calls `list(self.stream(...))`.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `messages` | [`Message`](#api-message)[] |
| `step?` | `number` \| `null` |
| `stop?` | `string`[] \| `null` |
| `options?` | [`LLMInvocationOptions`](#api-llminvocationoptions) \| `null` |

**Returns**

`Promise`\<[`LLMResponse`](#api-llmresponse)\>

<a id="api-stream"></a>

##### stream()

```ts
stream(
   messages,
   step?,
   stop?,
   options?
): AsyncGenerator<LLMStreamChunk>;
```

Stream invoke_llm chunks through OE approval and OE-owned SSE relay.

Step auto-increments when not supplied. If OE returns a cached/sync
result, a synthetic LLMStreamChunk is yielded instead of opening an
SSE connection.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `messages` | [`Message`](#api-message)[] |
| `step?` | `number` \| `null` |
| `stop?` | `string`[] \| `null` |
| `options?` | [`LLMInvocationOptions`](#api-llminvocationoptions) \| `null` |

**Returns**

`AsyncGenerator`\<`LLMStreamChunk`\>

<a id="api-responsefromstreamchunks"></a>

##### responseFromStreamChunks()

```ts
static responseFromStreamChunks(chunks): LLMResponse;
```

Collect streamed sdk-core chunks into a final sdk-core LLMResponse.
Static — can be called without a proxy instance (e.g. after parallel streaming).

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `chunks` | `LLMStreamChunk`[] |

**Returns**

[`LLMResponse`](#api-llmresponse)

***

<a id="api-securetoolwrapper"></a>

### SecureToolWrapper

<a id="api-constructor-31"></a>

#### Constructor

```ts
new SecureToolWrapper(
   oeUrl,
   executionId,
   customHeaders?,
   oeOwnerUrl?,
   drainRegistry?
): SecureToolWrapper;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `oeUrl` | `string` |
| `executionId` | `string` |
| `customHeaders?` | `Record`\<`string`, `string`\> |
| `oeOwnerUrl?` | `string` \| `null` |
| `drainRegistry?` | `DrainRegistry` \| `null` |

**Returns**

[`SecureToolWrapper`](#api-securetoolwrapper)

#### Properties

| Property | Modifier | Type | Default value | Description |
| :------ | :------ | :------ | :------ | :------ |
| <a id="api-property-customheaders"></a> `customHeaders` | `readonly` | `Record`\<`string`, `string`\> | `undefined` | - |
| <a id="api-property-durablememory-1"></a> `durableMemory` | `public` | [`DurableMemoryState`](#api-durablememorystate) \| `null` | `null` | - |
| <a id="api-property-executionid-1"></a> `executionId` | `readonly` | `string` | `undefined` | - |
| <a id="api-property-oeownerurl"></a> `oeOwnerUrl` | `readonly` | `string` \| `null` | `undefined` | Validated replica-specific OE owner base URL, or null. Forwarded to `reportOeResult` for the in-process (local-callback) tool-result path so settlement prefers the owning OE replica. |
| <a id="api-property-oeurl-1"></a> `oeUrl` | `readonly` | `string` | `undefined` | - |
| <a id="api-property-operationalsteps-1"></a> `operationalSteps` | `readonly` | [`OperationalStepSource`](#api-operationalstepsource) | `undefined` | Shared with SecureLLMProxy for this execution (single-AER assumption). |

#### Accessors

<a id="api-stepcounter-1"></a>

##### stepCounter

**Get Signature**

```ts
get stepCounter(): number;
```

Current operational-step watermark for compatibility readers.

**Returns**

`number`

#### Methods

<a id="api-close-1"></a>

##### close()

```ts
close(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

<a id="api-executetool"></a>

##### executeTool()

```ts
executeTool(
   toolName,
   args,
   options?
): Promise<unknown>;
```

Execute a tool call through OE.

Flow:
  1. POST /tool/execute — OE approves or denies
  2. OE returns a final outcome or routes the call back in process
  3. Wrapper logs the outcome and converts suspend payloads back into framework interrupts

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `toolName` | `string` |
| `args` | `Record`\<`string`, `unknown`\> |
| `options` | `ExecuteToolOptions` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-nextoperationalstep"></a>

##### nextOperationalStep()

```ts
nextOperationalStep(): number;
```

Allocate the next operational step_number for this execution.

**Returns**

`number`

<a id="api-observeoperationalstep"></a>

##### observeOperationalStep()

```ts
observeOperationalStep(n): void;
```

Raise the allocator watermark (e.g. from OE latest_step_number).

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `n` | `number` |

**Returns**

`void`

***

<a id="api-tenantruntime"></a>

### TenantRuntime

Tenant Runtime SDK.

Constructed once per process. `registerAndRun` wires the framework SDK's
graph builder and starts the per-mode Fastify server.

#### Implements

- [`ITenantRuntime`](#api-itenantruntime)

<a id="api-constructor-32"></a>

#### Constructor

```ts
new TenantRuntime(opts?): TenantRuntime;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts` | [`TenantRuntimeOptions`](#api-tenantruntimeoptions) |

**Returns**

[`TenantRuntime`](#api-tenantruntime)

#### Properties

| Property | Modifier | Type | Default value | Description |
| :------ | :------ | :------ | :------ | :------ |
| <a id="api-property-appname"></a> `appName` | `readonly` | `string` | `undefined` | - |
| <a id="api-property-appversion"></a> `appVersion` | `readonly` | `string` | `undefined` | - |
| <a id="api-property-graphbuilder"></a> `graphBuilder` | `public` | [`GraphBuilderLike`](#api-graphbuilderlike) \| `null` | `null` | Graph builder instance — null if not set at startup. |
| <a id="api-property-memorywriter"></a> `memoryWriter` | `readonly` | `MemoryWriter` \| `null` | `undefined` | Native-checkpoint turn writer, present only when Memory is enabled. |
| <a id="api-property-mode"></a> `mode` | `readonly` | [`RuntimeMode`](#api-runtimemode) | `undefined` | - |
| <a id="api-property-orgid"></a> `orgId` | `readonly` | `string` \| `null` | `undefined` | - |
| <a id="api-property-projectid"></a> `projectId` | `readonly` | `string` \| `null` | `undefined` | Optional so pre-existing hand-written implementations (custom runtimes, test doubles) keep typechecking across the SDK bump that introduced it. The capability-advertise path treats absent and null alike by skipping. |
| <a id="api-property-tooldefinitions"></a> `toolDefinitions` | `public` | `Record`\<`string`, `Record`\<`string`, `unknown`\>\> | `{}` | Tool metadata (description, is_local, etc.), keyed by name. |
| <a id="api-property-tools"></a> `tools` | `public` | `Record`\<`string`, [`ServerToolFn`](#api-servertoolfn)\> | `{}` | Raw tool functions registered via `app.tool()`, keyed by name. |

#### Methods

<a id="api-getagent"></a>

##### getAgent()

```ts
getAgent(opts?): BaseAgent;
```

Get a `BaseAgent` instance, optionally wired with execution callbacks.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts?` | \{ `callbacks?`: `BaseExecutionCallback`[]; \} |
| `opts.callbacks?` | `BaseExecutionCallback`[] |

**Returns**

`BaseAgent`

**Implementation of**

[`ITenantRuntime`](#api-itenantruntime).[`getAgent`](#api-getagent-2)

<a id="api-getagentconfig"></a>

##### getAgentConfig()

```ts
getAgentConfig(): RuntimeAgentConfig;
```

Parsed `agent.yaml` configuration.

**Returns**

[`RuntimeAgentConfig`](#api-runtimeagentconfig)

**Implementation of**

[`ITenantRuntime`](#api-itenantruntime).[`getAgentConfig`](#api-getagentconfig-1)

<a id="api-getcurrentsessionid"></a>

##### getCurrentSessionId()

```ts
getCurrentSessionId(): string | null;
```

**Returns**

`string` \| `null`

<a id="api-getcurrentuserid"></a>

##### getCurrentUserId()

```ts
getCurrentUserId(): string | null;
```

Get the current user_id from execution context.

**Returns**

`string` \| `null`

User ID from the current execution context, or null if not available

<a id="api-getmongodburi"></a>

##### getMongodbUri()

```ts
getMongodbUri(): string | null;
```

Resolved MongoDB URI: constructor option > `MONGODB_URI` env var > null.

**Returns**

`string` \| `null`

**Implementation of**

[`ITenantRuntime`](#api-itenantruntime).[`getMongodbUri`](#api-getmongodburi-1)

<a id="api-gettoolmetadata"></a>

##### getToolMetadata()

```ts
getToolMetadata(name): Record<string, unknown>;
```

Get metadata for a registered tool.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `name` | `string` | Tool name |

**Returns**

`Record`\<`string`, `unknown`\>

Tool metadata object (is_local, network, timeout, etc.)
  or empty object if tool not found.

<a id="api-registerandrun"></a>

##### registerAndRun()

```ts
registerAndRun(graphBuilder?, options?): Promise<void>;
```

Register the graph builder and start the mode-specific server.

Resolves once Fastify is listening — the open socket keeps the Node
event loop alive, matching the practical effect of Python's
`asyncio.run(uvicorn.serve())` blocking until shutdown.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `graphBuilder?` | [`GraphBuilderLike`](#api-graphbuilderlike) \| `null` |
| `options?` | [`RegisterAndRunOptions`](#api-registerandrunoptions) |

**Returns**

`Promise`\<`void`\>

<a id="api-registertool"></a>

##### registerTool()

```ts
registerTool(
   name,
   func,
   metadata
): void;
```

Register a raw tool function and its metadata.

This stores the function for Tool Pod execution and the metadata
for routing decisions. It does NOT create any framework-specific
tool objects — that is the responsibility of the framework SDK.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `name` | `string` | Tool name |
| `func` | [`ServerToolFn`](#api-servertoolfn) | Raw tool function |
| `metadata` | `Record`\<`string`, `unknown`\> | Tool metadata (is_local, network, timeout, etc.) |

**Returns**

`void`

<a id="api-shutdown-3"></a>

##### shutdown()

```ts
shutdown(): Promise<void>;
```

Close the tracing MongoClient. Idempotent — safe to call from multiple
signal handlers. TS-only addition: Python's uvicorn handles SIGTERM
itself, so the Python `TenantRuntime` has no equivalent method.

**Returns**

`Promise`\<`void`\>

<a id="api-warmupagent"></a>

##### warmUpAgent()

```ts
warmUpAgent(): void;
```

Explicit pre-build via the graph builder's `warmUp()`, if it has one.

**Returns**

`void`

**Implementation of**

[`ITenantRuntime`](#api-itenantruntime).[`warmUpAgent`](#api-warmupagent-1)

***

<a id="api-terminalexecutionerror"></a>

### TerminalExecutionError

Raised when tool execution fails.

#### Extends

- [`ToolExecutionError`](#api-toolexecutionerror)

<a id="api-constructor-33"></a>

#### Constructor

```ts
new TerminalExecutionError(error, options?): TerminalExecutionError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `error` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`TerminalExecutionError`](#api-terminalexecutionerror)

**Overrides**

[`ToolExecutionError`](#api-toolexecutionerror).[`constructor`](#api-constructor-35)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-error-2"></a> `error` | `readonly` | `string` | [`ToolExecutionError`](#api-toolexecutionerror).[`error`](#api-property-error-4) |

***

<a id="api-toolcalltimeouterror"></a>

### ToolCallTimeoutError

Raised when a tool call outlives its deadline.

Distinct from PolicyDeniedException on purpose. A timeout means the call was
permitted and ran — it just ran too long — so reporting it as a denial sends
the developer to debug governance instead of their tool.

#### Extends

- [`ToolExecutionError`](#api-toolexecutionerror)

<a id="api-constructor-34"></a>

#### Constructor

```ts
new ToolCallTimeoutError(
   toolName,
   timeoutSeconds,
   elapsedSeconds,
   options?
): ToolCallTimeoutError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `toolName` | `string` |
| `timeoutSeconds` | `number` |
| `elapsedSeconds` | `number` |
| `options?` | `ErrorOptions` |

**Returns**

[`ToolCallTimeoutError`](#api-toolcalltimeouterror)

**Overrides**

[`ToolExecutionError`](#api-toolexecutionerror).[`constructor`](#api-constructor-35)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-elapsedseconds"></a> `elapsedSeconds` | `readonly` | `number` | - |
| <a id="api-property-error-3"></a> `error` | `readonly` | `string` | [`ToolExecutionError`](#api-toolexecutionerror).[`error`](#api-property-error-4) |
| <a id="api-property-timeoutseconds"></a> `timeoutSeconds` | `readonly` | `number` | - |
| <a id="api-property-toolname"></a> `toolName` | `readonly` | `string` | - |

***

<a id="api-toolexecutionerror"></a>

### ToolExecutionError

Raised when tool execution fails.

#### Extends

- `Error`

#### Extended by

- [`TerminalExecutionError`](#api-terminalexecutionerror)
- [`ToolCallTimeoutError`](#api-toolcalltimeouterror)
- [`ExternalAPICallError`](#api-externalapicallerror)

<a id="api-constructor-35"></a>

#### Constructor

```ts
new ToolExecutionError(error, options?): ToolExecutionError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `error` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`ToolExecutionError`](#api-toolexecutionerror)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-error-4"></a> `error` | `readonly` | `string` |

***

<a id="api-toolfunctionrunner"></a>

### ToolFunctionRunner

Runs one metadata-delivered tool call and exits. Not a server.

Reads the ToolFunctionRequest envelope from the guest metadata directory,
invokes the named tool from the registry, reports the ToolResultRequest
to {oe_url}/tool/result, and returns. Mirrors Python's ToolFunctionRunner.

`metadataDir` is injectable for tests.

<a id="api-constructor-36"></a>

#### Constructor

```ts
new ToolFunctionRunner(runtime, __namedParameters?): ToolFunctionRunner;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `runtime` | [`ITenantRuntime`](#api-itenantruntime) |
| `__namedParameters` | \{ `metadataDir?`: `string`; \} |
| `__namedParameters.metadataDir?` | `string` |

**Returns**

[`ToolFunctionRunner`](#api-toolfunctionrunner)

#### Methods

<a id="api-run-1"></a>

##### run()

```ts
run(): Promise<void>;
```

Run one tool from the metadata-delivered request and report the result.

Re-raises if the result POST cannot be delivered after retries so fctr
records the execution as failed.

**Returns**

`Promise`\<`void`\>

***

<a id="api-toolserver"></a>

### ToolServer

#### Extends

- `BaseServer`

<a id="api-constructor-37"></a>

#### Constructor

```ts
new ToolServer(runtime): ToolServer;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `runtime` | [`ITenantRuntime`](#api-itenantruntime) |

**Returns**

[`ToolServer`](#api-toolserver)

**Overrides**

```ts
BaseServer.constructor
```

#### Properties

| Property | Modifier | Type | Default value | Inherited from |
| :------ | :------ | :------ | :------ | :------ |
| <a id="api-property-app-1"></a> `app` | `protected` | \| `FastifyInstance`\<`RawServerDefault`, `IncomingMessage`, `ServerResponse`\<`IncomingMessage`\>, `FastifyBaseLogger`, `FastifyTypeProviderDefault`\> \| `null` | `null` | `BaseServer.app` |
| <a id="api-property-runtime-1"></a> `runtime` | `readonly` | [`ITenantRuntime`](#api-itenantruntime) | `undefined` | `BaseServer.runtime` |

#### Accessors

<a id="api-defaultport-1"></a>

##### defaultPort

**Get Signature**

```ts
get defaultPort(): number;
```

**Returns**

`number`

**Inherited from**

```ts
BaseServer.defaultPort
```

<a id="api-drainregistry-1"></a>

##### drainRegistry

**Get Signature**

```ts
get drainRegistry(): DrainRegistry;
```

Execution-scoped drain state for POST /drain; work routes register
against it so a cancelled execution's in-flight work can be stopped.
Lazily created on the prototype so tests that skip the constructor
(`Object.create`) still get a working registry.

**Returns**

`DrainRegistry`

**Inherited from**

```ts
BaseServer.drainRegistry
```

<a id="api-modename-1"></a>

##### modeName

**Get Signature**

```ts
get modeName(): string;
```

**Returns**

`string`

**Overrides**

```ts
BaseServer.modeName
```

#### Methods

<a id="api-close-2"></a>

##### close()

```ts
close(): Promise<void>;
```

Gracefully close the underlying Fastify server, draining in-flight
requests and running `onClose` hooks. No-op if the app was never created.

**Returns**

`Promise`\<`void`\>

**Inherited from**

```ts
BaseServer.close
```

<a id="api-createapp-1"></a>

##### createApp()

```ts
createApp(): FastifyInstance;
```

**Returns**

`FastifyInstance`

**Inherited from**

```ts
BaseServer.createApp
```

<a id="api-gethealthdetails-1"></a>

##### getHealthDetails()

```ts
getHealthDetails(): Record<string, unknown>;
```

**Returns**

`Record`\<`string`, `unknown`\>

**Overrides**

```ts
BaseServer.getHealthDetails
```

<a id="api-onshutdown-1"></a>

##### onShutdown()

```ts
onShutdown(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

**Overrides**

```ts
BaseServer.onShutdown
```

<a id="api-onstartup-1"></a>

##### onStartup()

```ts
onStartup(): Promise<void>;
```

**Returns**

`Promise`\<`void`\>

**Overrides**

```ts
BaseServer.onStartup
```

<a id="api-registercommonroutes-1"></a>

##### registerCommonRoutes()

```ts
protected registerCommonRoutes(app): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `app` | `FastifyInstance` |

**Returns**

`void`

**Inherited from**

```ts
BaseServer.registerCommonRoutes
```

<a id="api-registerroutes-1"></a>

##### registerRoutes()

```ts
registerRoutes(app): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `app` | `FastifyInstance` |

**Returns**

`void`

**Overrides**

```ts
BaseServer.registerRoutes
```

<a id="api-run-2"></a>

##### run()

```ts
run(host?, port?): Promise<void>;
```

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `host` | `string` | `"0.0.0.0"` |
| `port?` | `number` | `undefined` |

**Returns**

`Promise`\<`void`\>

**Inherited from**

```ts
BaseServer.run
```

***

<a id="api-unsupportedchildoperationfanouterror"></a>

### UnsupportedChildOperationFanOutError

#### Extends

- `Error`

<a id="api-constructor-38"></a>

#### Constructor

```ts
new UnsupportedChildOperationFanOutError(message): UnsupportedChildOperationFanOutError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |

**Returns**

[`UnsupportedChildOperationFanOutError`](#api-unsupportedchildoperationfanouterror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-workflowclient"></a>

### WorkflowClient

<a id="api-constructor-39"></a>

#### Constructor

```ts
new WorkflowClient(oeUrl, options?): WorkflowClient;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `oeUrl` | `string` |
| `options` | \{ `fetch?`: \{ (`input`, `init?`): `Promise`\<`Response`\>; (`input`, `init?`): `Promise`\<`Response`\>; \}; `timeoutMs?`: `number`; \} |
| `options.fetch?` | \{ (`input`, `init?`): `Promise`\<`Response`\>; (`input`, `init?`): `Promise`\<`Response`\>; \} |
| `options.timeoutMs?` | `number` |

**Returns**

[`WorkflowClient`](#api-workflowclient)

#### Methods

<a id="api-completeexecution"></a>

##### completeExecution()

```ts
completeExecution(command): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `command` | [`CompleteExecutionCommand`](#api-completeexecutioncommand) |

**Returns**

`Promise`\<`void`\>

<a id="api-ensurememorywritten"></a>

##### ensureMemoryWritten()

```ts
ensureMemoryWritten(command): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `command` | [`ActivityMemoryCommand`](#api-activitymemorycommand) |

**Returns**

`Promise`\<`void`\>

<a id="api-finalizestep"></a>

##### finalizeStep()

```ts
finalizeStep(command): Promise<StepActivityEntry[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `command` | [`FinalizeStepCommand`](#api-finalizestepcommand) |

**Returns**

`Promise`\<[`StepActivityEntry`](#api-stepactivityentry)[]\>

<a id="api-heartbeat"></a>

##### heartbeat()

```ts
heartbeat(request): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `request` | [`AttemptHeartbeatRequest`](#api-attemptheartbeatrequest) |

**Returns**

`Promise`\<`void`\>

<a id="api-reportoutcome"></a>

##### reportOutcome()

```ts
reportOutcome(outcome): Promise<void>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `outcome` | [`ActivityOutcome`](#api-activityoutcome) |

**Returns**

`Promise`\<`void`\>

<a id="api-startactivity"></a>

##### startActivity()

```ts
startActivity(command): Promise<StartActivityResult>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `command` | [`ActivityCommand`](#api-activitycommand) |

**Returns**

`Promise`\<[`StartActivityResult`](#api-startactivityresult)\>

<a id="api-startattempt"></a>

##### startAttempt()

```ts
startAttempt(request): Promise<AttemptContext | null>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `request` | [`AttemptStartRequest`](#api-attemptstartrequest) |

**Returns**

`Promise`\<[`AttemptContext`](#api-attemptcontext) \| `null`\>

***

<a id="api-workflowclienterror"></a>

### WorkflowClientError

#### Extends

- `Error`

#### Extended by

- [`ReplayedActivityFailedError`](#api-replayedactivityfailederror)

<a id="api-constructor-40"></a>

#### Constructor

```ts
new WorkflowClientError(
   code,
   message,
   options?
): WorkflowClientError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `code` | [`WorkflowErrorCode`](#api-workflowerrorcode) |
| `message` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`WorkflowClientError`](#api-workflowclienterror)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-code-1"></a> `code` | `readonly` | [`WorkflowErrorCode`](#api-workflowerrorcode) |

## Interfaces

<a id="api-aerqueryplugin"></a>

### AERQueryPlugin

Read-side plugin that surfaces framework-specific session state.

Implementations live alongside framework adapters and read from the
adapter's persistence (LangGraph checkpoint collections, ADK state
store, etc.).

#### Methods

<a id="api-getmessagesforsession"></a>

##### getMessagesForSession()

```ts
getMessagesForSession(sessionId): Promise<{
  messages: object[];
}>;
```

Given a session_id, return a SessionMessagesResponse.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionId` | `string` |

**Returns**

`Promise`\<\{
  `messages`: `object`[];
\}\>

<a id="api-getsummariesforsessions"></a>

##### getSummariesForSessions()

```ts
getSummariesForSessions(sessionIds): Promise<{
  sessions: object[];
}>;
```

Given a list of session_ids, return a SessionsSummaryResponse.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionIds` | `string`[] |

**Returns**

`Promise`\<\{
  `sessions`: `object`[];
\}\>

***

<a id="api-captureexceptionargs"></a>

### CaptureExceptionArgs

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-extra"></a> `extra?` | `Record`\<`string`, `unknown`\> |
| <a id="api-property-summary"></a> `summary?` | `string` \| `null` |

***

<a id="api-childoperationboundary"></a>

### ChildOperationBoundary

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-name-4"></a> `name` | `string` |
| <a id="api-property-occurrencekey"></a> `occurrenceKey` | `string` |

***

<a id="api-executionstore"></a>

### ExecutionStore

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-authorization"></a> `authorization` | \| \{ `expires_at?`: `number` \| `null`; `token`: `string`; \} \| `null` | Delegated credential injected by OE for tool execution. |
| <a id="api-property-customheaders-1"></a> `customHeaders` | `Record`\<`string`, `string`\> \| `null` | - |
| <a id="api-property-executionid-2"></a> `executionId` | `string` \| `null` | - |
| <a id="api-property-executionmetadata"></a> `executionMetadata` | `Record`\<`string`, `unknown`\> \| `null` | - |
| <a id="api-property-oeownerurl-1"></a> `oeOwnerUrl` | `string` \| `null` | Validated replica-specific OE owner callback URL, or null when the request carried none / it failed validation. Owner-preferring transports (e.g. module:progress) send here first and fall back to `oeUrl` on any owner failure. Already validated against `oeUrl` by the caller (see `server/owner_url.ts`), so consumers trust it as-is. |
| <a id="api-property-oeurl-2"></a> `oeUrl` | `string` \| `null` | - |
| <a id="api-property-ownerurlfailure"></a> `ownerUrlFailure` | `object` | One-way owner-failure latch: once an owner pre-attempt fails, later owner-preferring posts in this execution skip the owner URL entirely instead of re-paying the pre-attempt timeout on every emit. A nested object for the same reason as `sessionFinish`: the reference survives store spreads (e.g. [runWithSuspendRequestContext](#api-runwithsuspendrequestcontext)), a reassigned field would not. Mirrors the AER stream path's `onOwnerFailure` discard. |
| `ownerUrlFailure.failed` | `boolean` | - |
| <a id="api-property-payload"></a> `payload` | `Record`\<`string`, `unknown`\> \| `null` | Opaque caller-provided invocation payload (the request body beyond `message`). |
| <a id="api-property-requestid"></a> `requestId` | `string` \| `null` | - |
| <a id="api-property-sessionfinish"></a> `sessionFinish` | `object` | Holder for a pending session-finish request. A nested object rather than a bare boolean: the AER reads it from the frame that started the run, while agent code deep in the call chain mutates it in place — the object reference is shared across that chain, a reassigned boolean field on the store would not be. |
| `sessionFinish.closed` | `boolean` | - |
| `sessionFinish.requested` | `boolean` | - |
| <a id="api-property-sessionid"></a> `sessionId` | `string` \| `null` | - |
| <a id="api-property-signal"></a> `signal` | `AbortSignal` \| `null` | Execution-wide abort signal (fires on the AER execution timeout). Combined into in-flight OE/LLM fetches via [withExecutionSignal](#api-withexecutionsignal) so a timeout cancels active network I/O instead of leaving it to run. Null outside an AER execution (e.g. a single Tool Pod call). |
| <a id="api-property-suspendrequest"></a> `suspendRequest` | `object` | Out-of-band suspend signal. Only the author-facing `suspendPayloadToJson` writes it, so untrusted tool-result content — which cannot reach this process-local frame — can never forge a HITL suspend. A nested holder for the same reason as `sessionFinish`: a tool offloaded to a worker thread mutates the shared object, not a reassigned store field. |
| `suspendRequest.payload` | `Record`\<`string`, `unknown`\> \| `null` | - |
| <a id="api-property-traceid"></a> `traceId` | `string` \| `null` | Platform trace ID for log correlation. |
| <a id="api-property-userid"></a> `userId` | `string` \| `null` | - |
| <a id="api-property-workspaceid"></a> `workspaceId` | `string` \| `null` | - |
| <a id="api-property-wrapper"></a> `wrapper` | `unknown` | - |

***

<a id="api-extractedusage"></a>

### ExtractedUsage

Normalized usage shape returned by `extractUsage` / `extractPodUsage`.
Mirrors Python's dict-shaped return so downstream consumers can pass these
fields straight into `reportOeResult`.

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-completion_tokens"></a> `completion_tokens` | `number` \| `null` |
| <a id="api-property-model-1"></a> `model` | `string` \| `null` |
| <a id="api-property-prompt_tokens"></a> `prompt_tokens` | `number` \| `null` |
| <a id="api-property-total_tokens"></a> `total_tokens` | `number` \| `null` |

***

<a id="api-graphbuilderlike"></a>

### GraphBuilderLike

Minimal graph builder interface.

#### Methods

<a id="api-getagent-1"></a>

##### getAgent()?

```ts
optional getAgent(opts?): BaseAgent;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts?` | \{ `callbacks?`: `BaseExecutionCallback`[]; \} |
| `opts.callbacks?` | `BaseExecutionCallback`[] |

**Returns**

`BaseAgent`

<a id="api-ready"></a>

##### ready()?

```ts
optional ready(): Promise<void>;
```

Resolves once any async setup kicked off during construction (e.g. MCP
tool discovery) has finished registering tools. `TenantRuntime.runAsync()`
awaits this before binding the server, so no request can land before
setup completes. Optional — builders with no async setup can omit it.

**Returns**

`Promise`\<`void`\>

<a id="api-warmup"></a>

##### warmUp()?

```ts
optional warmUp(): void;
```

Build the agent graph before first use when explicitly requested. The
TypeScript AER does not call this synchronous hook during standby warming
because doing so would block the Node event loop and server health.

**Returns**

`void`

***

<a id="api-guardrailpolicyengine"></a>

### GuardrailPolicyEngine

Contract implemented by native and future remote guardrail engines.

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-policytype-1"></a> `policyType` | `readonly` | `string` |

#### Methods

<a id="api-evaluate-1"></a>

##### evaluate()

```ts
evaluate(policy, text):
  | GuardrailPolicyEngineResult
  | null;
```

Return engine-owned evidence and optional transformed text when triggered.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `policy` | \{ `action`: `string`; `config`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `id`: `string`; `stage_filter`: `string`[]; `status`: `string`; `type`: `string`; \} | - |
| `policy.action` | `string` | Action requested when the policy triggers. |
| `policy.config` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | Evaluator-specific policy configuration. |
| `policy.id` | `string` | Guardrail policy identifier. |
| `policy.stage_filter` | `string`[] | Runtime stages this policy applies to; empty means all stages. |
| `policy.status` | `string` | Guardrail policy status. |
| `policy.type` | `string` | Guardrail policy type. |
| `text` | `string` | - |

**Returns**

  \| [`GuardrailPolicyEngineResult`](#api-guardrailpolicyengineresult)
  \| `null`

***

<a id="api-guardrailpolicyengineresult"></a>

### GuardrailPolicyEngineResult

Engine-owned result for a policy evaluation.

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-evidence"></a> `evidence` | `object`[] |
| <a id="api-property-transformedtext"></a> `transformedText?` | `string` \| `null` |

***

<a id="api-initerrorreportingargs"></a>

### InitErrorReportingArgs

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-component"></a> `component?` | `string` \| `null` |
| <a id="api-property-mode-1"></a> `mode?` | `string` \| `null` |
| <a id="api-property-surface"></a> `surface` | `string` |

***

<a id="api-installstructuredloggingargs"></a>

### InstallStructuredLoggingArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-capturestdio"></a> `captureStdio?` | `boolean` | Set to `false` to skip `process.stdout` / `process.stderr` patching. Useful in tests that share a Node process across cases, or in callers that own their own stdio routing. Defaults to `true` (matches Python). |
| <a id="api-property-filelogpath"></a> `fileLogPath?` | `string` \| `null` | Absolute path for a dev-mode `dateFile` appender alongside the structured stdout appender. When set, a second log4js appender writes human-readable records to this path, rotating daily and keeping the last 5 files (Python `TimedRotatingFileHandler` parity). Only used by `setupLogging` when `AGENTIC_DEV_MODES` is set; left unset in production, where Fluent Bit is the only sink. |
| <a id="api-property-level"></a> `level?` | `string` \| `null` | Log level — string name (`"DEBUG"`, `"info"`). Defaults to `LOG_LEVEL` env var, then INFO. Must be honored on rollout because operators continue to flip `LOG_LEVEL=DEBUG` to chase issues. |
| <a id="api-property-mode-2"></a> `mode?` | `string` \| `null` | Runner mode (`"orchestrator"`, `"aer"`, `"tool"`, `"memory-server"`) used to derive `service` and `fields.component` on the wire. Takes precedence over `RUNNER_MODE` env var when set; falls back to env when `null`. Lets `setupLogging(mode=...)` callers stay authoritative even if pod env hasn't been stamped. |

***

<a id="api-interruptedactivity"></a>

### InterruptedActivity

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-command"></a> `command` | `readonly` | [`ActivityCommand`](#api-activitycommand) |
| <a id="api-property-controlflow-1"></a> `controlFlow` | `readonly` | `unknown` |

***

<a id="api-itenantruntime"></a>

### ITenantRuntime

Minimal runtime interface that server classes depend on.

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-appname-1"></a> `appName` | `readonly` | `string` | - |
| <a id="api-property-appversion-1"></a> `appVersion?` | `readonly` | `string` | - |
| <a id="api-property-graphbuilder-1"></a> `graphBuilder` | `readonly` | [`GraphBuilderLike`](#api-graphbuilderlike) \| `null` | Graph builder instance — null if not set at startup. |
| <a id="api-property-memorywriter-1"></a> `memoryWriter?` | `readonly` | `TurnMemoryWriter` \| `null` | Native-checkpoint turn writer, present only when Memory is enabled. |
| <a id="api-property-orgid-1"></a> `orgId` | `readonly` | `string` \| `null` | - |
| <a id="api-property-projectid-1"></a> `projectId?` | `readonly` | `string` \| `null` | Optional so pre-existing hand-written implementations (custom runtimes, test doubles) keep typechecking across the SDK bump that introduced it. The capability-advertise path treats absent and null alike by skipping. |
| <a id="api-property-tooldefinitions-1"></a> `toolDefinitions` | `readonly` | `Record`\<`string`, `Record`\<`string`, `unknown`\>\> | Tool metadata (description, is_local, etc.), keyed by name. |
| <a id="api-property-tools-1"></a> `tools` | `readonly` | `Record`\<`string`, [`ServerToolFn`](#api-servertoolfn)\> | Raw tool functions registered via `app.tool()`, keyed by name. |

#### Methods

<a id="api-getagent-2"></a>

##### getAgent()

```ts
getAgent(opts?): BaseAgent;
```

Get a `BaseAgent` instance, optionally wired with execution callbacks.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts?` | \{ `callbacks?`: `BaseExecutionCallback`[]; \} |
| `opts.callbacks?` | `BaseExecutionCallback`[] |

**Returns**

`BaseAgent`

<a id="api-getagentconfig-1"></a>

##### getAgentConfig()

```ts
getAgentConfig(): RuntimeAgentConfig;
```

Parsed `agent.yaml` configuration.

**Returns**

[`RuntimeAgentConfig`](#api-runtimeagentconfig)

<a id="api-getmongodburi-1"></a>

##### getMongodbUri()

```ts
getMongodbUri(): string | null;
```

Resolved MongoDB URI: constructor option > `MONGODB_URI` env var > null.

**Returns**

`string` \| `null`

<a id="api-warmupagent-1"></a>

##### warmUpAgent()?

```ts
optional warmUpAgent(): void;
```

Explicit pre-build via the graph builder's `warmUp()`, if it has one.

**Returns**

`void`

***

<a id="api-latencystatsjson"></a>

### LatencyStatsJson

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-avg_ms"></a> `avg_ms` | `number` |
| <a id="api-property-count"></a> `count` | `number` |
| <a id="api-property-max_ms"></a> `max_ms` | `number` |
| <a id="api-property-min_ms"></a> `min_ms` | `number` |
| <a id="api-property-total_ms"></a> `total_ms` | `number` |

***

<a id="api-mcptoolbinding"></a>

### MCPToolBinding

Discovered MCP tool bound to a configured server.

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-description-1"></a> `description` | `readonly` | `string` |
| <a id="api-property-inputschema"></a> `inputSchema` | `readonly` | `Record`\<`string`, `unknown`\> |
| <a id="api-property-sdktoolname"></a> `sdkToolName` | `readonly` | `string` |
| <a id="api-property-serverconfig"></a> `serverConfig` | `readonly` | `object` |
| `serverConfig.allowed_tools` | `public` | `string`[] \| `null` |
| `serverConfig.auth` | `public` | `object` |
| `serverConfig.auth.client_id_env` | `public` | `string` \| `null` |
| `serverConfig.auth.client_name` | `public` | `string` \| `null` |
| `serverConfig.auth.client_secret_env` | `public` | `string` \| `null` |
| `serverConfig.auth.redirect_uri` | `public` | `string` \| `null` |
| `serverConfig.auth.scope` | `public` | `string` \| `null` |
| `serverConfig.auth.token_env` | `public` | `string` \| `null` |
| `serverConfig.auth.token_url` | `public` | `string` \| `null` |
| `serverConfig.auth.type` | `public` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `serverConfig.headers` | `public` | `Record`\<`string`, `string`\> |
| `serverConfig.timeout_seconds` | `public` | `number` |
| `serverConfig.transport` | `public` | `"streamable_http"` |
| `serverConfig.url` | `public` | `string` |
| <a id="api-property-servername"></a> `serverName` | `readonly` | `string` |
| <a id="api-property-toolname-1"></a> `toolName` | `readonly` | `string` |

***

<a id="api-mcptoolresult"></a>

### MCPToolResult

JSON/BSON-safe result returned by an MCP `tools/call` invocation.

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-content-2"></a> `content` | `readonly` | `unknown`[] |
| <a id="api-property-iserror"></a> `isError` | `readonly` | `boolean` |
| <a id="api-property-structuredcontent"></a> `structuredContent` | `readonly` | `Record`\<`string`, `unknown`\> \| `null` |

***

<a id="api-message"></a>

### Message

Framework-neutral message.

Content can be a simple string for text-only messages, or a list of
content blocks for multimodal content (images, documents, etc.).

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-additionalkwargs-2"></a> `additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content-3"></a> `content` | \| `string` \| ( \| \{ `text`: `string`; `type`: `"text"`; \} \| \{ `mime_type?`: `string`; `type`: `"image"`; `url`: `string`; \} \| \{ `filename?`: `string`; `mime_type?`: `string`; `type`: `"document"`; `url`: `string`; \})[] |
| <a id="api-property-id-3"></a> `id?` | `string` |
| <a id="api-property-iserror-1"></a> `isError?` | `boolean` |
| <a id="api-property-name-5"></a> `name?` | `string` |
| <a id="api-property-responsemetadata-2"></a> `responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-role"></a> `role` | `Role` |
| <a id="api-property-toolcallid"></a> `toolCallId?` | `string` |
| <a id="api-property-toolcalls-2"></a> `toolCalls?` | [`LLMToolCall`](#api-llmtoolcall)[] |

***

<a id="api-nodeexecutionloggeropts"></a>

### NodeExecutionLoggerOpts

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-executionid-3"></a> `executionId` | `string` |
| <a id="api-property-oeurl-3"></a> `oeUrl` | `string` |
| <a id="api-property-orgid-2"></a> `orgId?` | `string` \| `null` |
| <a id="api-property-projectid-2"></a> `projectId?` | `string` \| `null` |
| <a id="api-property-sessionid-1"></a> `sessionId?` | `string` \| `null` |
| <a id="api-property-userid-1"></a> `userId?` | `string` \| `null` |

***

<a id="api-operationalstepsource"></a>

### OperationalStepSource

Process-local operational step_number mint for one AER execution.

Shared by SecureToolWrapper and SecureLLMProxy so parallel tools and LLM
calls cannot collide under a single AER. Not a multi-OE authority.
Node's single-threaded event loop makes a plain counter
safe across concurrent async tasks as long as minting stays synchronous
before the first await.

#### Methods

<a id="api-current-1"></a>

##### current()

```ts
current(): number;
```

**Returns**

`number`

<a id="api-next-1"></a>

##### next()

```ts
next(): number;
```

**Returns**

`number`

<a id="api-observeatleast-1"></a>

##### observeAtLeast()

```ts
observeAtLeast(n): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `n` | `number` |

**Returns**

`void`

***

<a id="api-ownerurlfailurestate"></a>

### OwnerUrlFailureState

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-failed"></a> `failed` | `boolean` |

***

<a id="api-recordmemorymetadataargs"></a>

### RecordMemoryMetadataArgs

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-action"></a> `action` | `string` |
| <a id="api-property-content-4"></a> `content?` | `string` |
| <a id="api-property-memorytype"></a> `memoryType` | `string` |
| <a id="api-property-query"></a> `query?` | `string` |
| <a id="api-property-relevancescore"></a> `relevanceScore?` | `number` |

***

<a id="api-registerandrunoptions"></a>

### RegisterAndRunOptions

Public options for `registerAndRun`.

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-loglevel"></a> `logLevel?` | `string` | Log level passed through to the underlying server. |

***

<a id="api-reportoeresultargs"></a>

### ReportOeResultArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-completiontokens-1"></a> `completionTokens?` | `number` \| `null` | - |
| <a id="api-property-durationms"></a> `durationMs` | `number` | - |
| <a id="api-property-error-5"></a> `error?` | `string` \| `null` | - |
| <a id="api-property-executionid-4"></a> `executionId` | `string` | - |
| <a id="api-property-kind"></a> `kind?` | `string` \| `null` | - |
| <a id="api-property-metadata-2"></a> `metadata?` | `Record`\<`string`, `unknown`\> | - |
| <a id="api-property-model-2"></a> `model?` | `string` \| `null` | - |
| <a id="api-property-oeurl-4"></a> `oeUrl` | `string` | - |
| <a id="api-property-onownerfailure"></a> `onOwnerFailure?` | (() => `void`) \| `null` | Called once when the owner pre-attempt fails, before service fallback. |
| <a id="api-property-ownerurl"></a> `ownerUrl?` | `string` \| `null` | Validated replica-specific OE owner base URL, already checked against `oeUrl` by the caller. When set, a single owner pre-attempt runs before the service loop and does not consume the service retry budget; the owner is best-effort, so any failure — a transport error OR any non-2xx response — marks the replica unusable and falls through to the trusted `oeUrl` loop. The owner is never retried and an owner response never throws. |
| <a id="api-property-podname"></a> `podName?` | `string` \| `null` | - |
| <a id="api-property-prompttokens-1"></a> `promptTokens?` | `number` \| `null` | - |
| <a id="api-property-result"></a> `result?` | `unknown` | - |
| <a id="api-property-status"></a> `status` | `string` | - |
| <a id="api-property-step"></a> `step` | `number` | - |
| <a id="api-property-toolapierror"></a> `toolApiError?` | \| \{ `classification`: `string`; `error_code?`: `string` \| `null`; `http_status?`: `number` \| `null`; `provider_type?`: `string` \| `null`; `reason?`: `string` \| `null`; `retryable`: `boolean`; \} \| `null` | - |
| <a id="api-property-toolcallid-1"></a> `toolCallId?` | `string` | - |
| <a id="api-property-toolname-2"></a> `toolName` | `string` | - |
| <a id="api-property-totaltokens-1"></a> `totalTokens?` | `number` \| `null` | - |

***

<a id="api-requestoeapprovalargs"></a>

### RequestOeApprovalArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-arguments"></a> `arguments` | `Record`\<`string`, `unknown`\> | - |
| <a id="api-property-customheaders-2"></a> `customHeaders?` | `Record`\<`string`, `string`\> | Caller-provided headers carried only for this active request. |
| <a id="api-property-executionid-5"></a> `executionId` | `string` | - |
| <a id="api-property-islocal"></a> `isLocal?` | `boolean` | - |
| <a id="api-property-kind-1"></a> `kind?` | `string` \| `null` | - |
| <a id="api-property-metadata-3"></a> `metadata?` | `Record`\<`string`, `unknown`\> | - |
| <a id="api-property-oeurl-5"></a> `oeUrl` | `string` | - |
| <a id="api-property-providertype"></a> `providerType?` | `string` \| `null` | - |
| <a id="api-property-redactfields"></a> `redactFields?` | readonly `string`[] | Top-level tool argument names to redact from execution logs. |
| <a id="api-property-scopes"></a> `scopes?` | readonly `string`[] | - |
| <a id="api-property-step-1"></a> `step` | `number` | - |
| <a id="api-property-timeoutms"></a> `timeoutMs?` | `number` | Overall request deadline in milliseconds. Defaults to `getToolReadTimeout()`. The LLM proxy passes a longer value (`LLM_READ_TIMEOUT`) for `invoke_llm`, because OE holds `/tool/execute` open while the LLM call completes — the analog of Python's `httpx.Timeout(get_request_timeout(), read=LLM_READ_TIMEOUT)`. |
| <a id="api-property-toolcallid-2"></a> `toolCallId?` | `string` | Stable LLM tool-call id, forwarded to OE for the execution-log join key. |
| <a id="api-property-toolname-3"></a> `toolName` | `string` | - |

***

<a id="api-resolvedentrypoint"></a>

### ResolvedEntrypoint

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-exportname"></a> `exportName` | `string` |
| <a id="api-property-modulepath"></a> `modulePath` | `string` |

***

<a id="api-setexecutioncontextargs"></a>

### SetExecutionContextArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-authorization-1"></a> `authorization?` | \| \{ `expires_at?`: `number` \| `null`; `token`: `string`; \} \| `null` | - |
| <a id="api-property-customheaders-3"></a> `customHeaders?` | `Record`\<`string`, `string`\> \| `null` | - |
| <a id="api-property-executionid-6"></a> `executionId` | `string` | - |
| <a id="api-property-oeownerurl-2"></a> `oeOwnerUrl?` | `string` \| `null` | Validated replica-specific OE owner callback URL. |
| <a id="api-property-oeurl-6"></a> `oeUrl` | `string` | - |
| <a id="api-property-ownerurlfailure-1"></a> `ownerUrlFailure?` | [`OwnerUrlFailureState`](#api-ownerurlfailurestate) | - |
| <a id="api-property-payload-1"></a> `payload?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| <a id="api-property-requestid-1"></a> `requestId?` | `string` \| `null` | - |
| <a id="api-property-sessionid-2"></a> `sessionId?` | `string` \| `null` | - |
| <a id="api-property-signal-1"></a> `signal?` | `AbortSignal` \| `null` | - |
| <a id="api-property-traceid-1"></a> `traceId?` | `string` \| `null` | - |
| <a id="api-property-userid-2"></a> `userId?` | `string` \| `null` | - |
| <a id="api-property-workspaceid-1"></a> `workspaceId?` | `string` \| `null` | - |
| <a id="api-property-wrapper-1"></a> `wrapper` | `unknown` | - |

***

<a id="api-setuploggingargs"></a>

### SetupLoggingArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-appname-2"></a> `appName?` | `string` | Application name — surfaced in the install log line and used to name the dev-mode log file (`<appName>-<mode>.log`) when AGENTIC_DEV_MODES is set. Matches Python `app_name`. |
| <a id="api-property-backupcount"></a> `backupCount?` | `number` | Unused — parity placeholder. |
| <a id="api-property-logdir"></a> `logDir?` | `string` \| `null` | Log directory — used only when `AGENTIC_DEV_MODES` is set to write a dev-mode file log alongside console output. |
| <a id="api-property-loglevel-1"></a> `logLevel?` | `string` \| `null` | Log level — string ("debug", "info", ...) or `LOG_LEVEL` env when omitted. |
| <a id="api-property-mode-3"></a> `mode?` | `string` | Runner mode (`aer` / `tool` / `orchestrator` / `memory-server`). |

***

<a id="api-setuptracingargs"></a>

### SetupTracingArgs

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-mongodbcollectionname"></a> `mongodbCollectionName?` | `string` | Collection name within `mongodbDatabaseName`. Defaults to `"traces"`. |
| <a id="api-property-mongodbdatabasename"></a> `mongodbDatabaseName?` | `string` \| `null` | Explicit database override. Omit to use the platform store default. |
| <a id="api-property-mongodburi"></a> `mongodbUri?` | `string` \| `null` | URI for the trace store. When provided, `setupTracing` builds the MongoClient, pings `admin`, and resolves the `(database, collection)` itself. Python's `TenantRuntime._setup_tracing` does this synchronously and passes the collection in; Node's driver is fundamentally async so the connect + ping live here instead. After a connection failure, a background retry attaches the MongoDB exporter once the store recovers — same recovery shape as Python. |
| <a id="api-property-servicename"></a> `serviceName?` | `string` | - |

***

<a id="api-subprocessfailure"></a>

### SubprocessFailure

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-command-1"></a> `command` | `string` \| readonly `string`[] | The command that failed — array (argv) or a single string. |
| <a id="api-property-exitcode"></a> `exitCode?` | `number` \| `null` | Process exit code, if known. |
| <a id="api-property-stderr"></a> `stderr?` | `string` \| `Uint8Array`\<`ArrayBufferLike`\> \| `null` | - |
| <a id="api-property-stdout"></a> `stdout?` | `string` \| `Uint8Array`\<`ArrayBufferLike`\> \| `null` | - |

***

<a id="api-tenantruntimeoptions"></a>

### TenantRuntimeOptions

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-appname-3"></a> `appName?` | `string` | Application name. Defaults to `"Agent"`. |
| <a id="api-property-appversion-2"></a> `appVersion?` | `string` | Application version. Defaults to `"1.0.0"`. |
| <a id="api-property-databasename"></a> `databaseName?` | `string` \| `null` | Explicit database name for trace storage. Defaults to the platform store. |
| <a id="api-property-mongodburi-1"></a> `mongodbUri?` | `string` \| `null` | MongoDB URI used for trace persistence. Falls back to `MONGODB_URI`. |
| <a id="api-property-orgid-3"></a> ~~`orgId?`~~ | `string` \| `null` | **Deprecated** Ignored. The org is taken from the `ORG_ID` env var, which the platform injects. |
| <a id="api-property-projectid-3"></a> ~~`projectId?`~~ | `string` \| `null` | **Deprecated** Ignored. The project is taken from the `PROJECT_ID` env var, which the platform injects. |
| <a id="api-property-tracescollectionname"></a> `tracesCollectionName?` | `string` | Collection name for trace storage. Defaults to `"traces"`. |

***

<a id="api-toolcallchunk"></a>

### ToolCallChunk

Partial tool call during streaming.

All fields are optional since chunks may contain partial data
(e.g., just the start of arguments, or just the tool name).

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-args-1"></a> `args?` | `string` |
| <a id="api-property-id-4"></a> `id?` | `string` |
| <a id="api-property-index-1"></a> `index?` | `number` |
| <a id="api-property-name-6"></a> `name?` | `string` |
| <a id="api-property-type-2"></a> `type?` | `string` |

***

<a id="api-workflowadapter"></a>

### WorkflowAdapter

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-name-7"></a> `name` | `readonly` | `string` |
| <a id="api-property-version"></a> `version` | `readonly` | `string` |

## Type Aliases

<a id="api-activitycommand"></a>

### ActivityCommand

```ts
type ActivityCommand = Message<"mongodb.agentic.workflow.v1.ActivityCommand"> & object;
```

ActivityCommand submits semantic input at a deterministic workflow position.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityKind` | [`ActivityKind`](#api-activitykind) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityKind activity_kind = 5; |
| `activityName` | `string` | **Generated** from field: string activity_name = 6; |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 2; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 3; |
| `position?` | [`ActivityPosition`](#api-activityposition) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityPosition position = 4; |
| `semanticInput?` | `Value` | **Generated** from field: google.protobuf.Value semantic_input = 7; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityCommand

***

<a id="api-activitycontext"></a>

### ActivityContext

```ts
type ActivityContext = Message<"mongodb.agentic.workflow.v1.ActivityContext"> & object;
```

ActivityContext is the minimal OE-issued context sent to an activity worker.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityId` | `string` | **Generated** from field: string activity_id = 2; |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 3; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 4; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityContext

***

<a id="api-activityheartbeatrequest"></a>

### ActivityHeartbeatRequest

```ts
type ActivityHeartbeatRequest = Message<"mongodb.agentic.workflow.v1.ActivityHeartbeatRequest"> & object;
```

ActivityHeartbeatRequest renews one activity lease under its current fence.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityId` | `string` | **Generated** from field: string activity_id = 2; |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 3; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 4; |
| `ownerId` | `string` | **Generated** from field: string owner_id = 5; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityHeartbeatRequest

***

<a id="api-activitymemorycommand"></a>

### ActivityMemoryCommand

```ts
type ActivityMemoryCommand = Message<"mongodb.agentic.workflow.v1.ActivityMemoryCommand"> & object;
```

ActivityMemoryCommand asks OE to ensure the wrapper-produced Memory batch
for one durable activity has been acknowledged. The same command follows a
freshly executed or replayed outcome.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityId` | `string` | **Generated** from field: string activity_id = 2; |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 3; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 4; |
| `memoryWrites` | [`MemoryWrite`](#api-memorywrite)[] | **Generated** from field: repeated mongodb.agentic.workflow.v1.MemoryWrite memory_writes = 5; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityMemoryCommand

***

<a id="api-activityoutcome"></a>

### ActivityOutcome

```ts
type ActivityOutcome = Message<"mongodb.agentic.workflow.v1.ActivityOutcome"> & object;
```

ActivityOutcome reports one result under its OE-issued attempt fence.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityId` | `string` | **Generated** from field: string activity_id = 2; |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 3; |
| `error?` | [`WorkflowError`](#api-workflowerror) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowError error = 7; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 4; |
| `outcomeKind` | [`ActivityOutcomeKind`](#api-activityoutcomekind) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityOutcomeKind outcome_kind = 5; |
| `result?` | `Value` | **Generated** from field: google.protobuf.Value result = 6; |
| `suspension?` | [`ActivitySuspension`](#api-activitysuspension) | **Generated** from field: mongodb.agentic.workflow.v1.ActivitySuspension suspension = 8; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityOutcome

***

<a id="api-activityposition"></a>

### ActivityPosition

```ts
type ActivityPosition = Message<"mongodb.agentic.workflow.v1.ActivityPosition"> & object;
```

ActivityPosition identifies one activity independently of retries. Step ordinal
scopes activities to a committed step; operation path and activity ordinal
locate the activity within that step.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityOrdinal` | `bigint` | **Generated** from field: int64 activity_ordinal = 2; |
| `operationPath?` | [`OperationPath`](#api-operationpath) | **Generated** from field: mongodb.agentic.workflow.v1.OperationPath operation_path = 1; |
| `stepOrdinal` | `bigint` | **Generated** from field: int64 step_ordinal = 3; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivityPosition

***

<a id="api-activityresolvedhook"></a>

### ActivityResolvedHook

```ts
type ActivityResolvedHook<T> = (client, context, result) => void | Promise<void>;
```

#### Type Parameters

| Type Parameter | Default type |
| :------ | :------ |
| `T` | `unknown` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `client` | `ActivityRuntimeClient` |
| `context` | [`ActivityContext`](#api-activitycontext) |
| `result` | `T` |

#### Returns

`void` \| `Promise`\<`void`\>

***

<a id="api-activitysuspension"></a>

### ActivitySuspension

```ts
type ActivitySuspension = Message<"mongodb.agentic.workflow.v1.ActivitySuspension"> & object;
```

ActivitySuspension describes why an activity is waiting for an external result.
The activity_id on ActivityOutcome is the stable wait identity.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `context?` | `JsonObject` | **Generated** from field: google.protobuf.Struct context = 2; |
| `reason` | `string` | **Generated** from field: string reason = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.ActivitySuspension

***

<a id="api-aerexecuteresponse"></a>

### AERExecuteResponse

```ts
type AERExecuteResponse = z.infer<typeof AERExecuteResponseSchema>;
```

***

<a id="api-agentfeatureconfig"></a>

### AgentFeatureConfig

```ts
type AgentFeatureConfig = z.infer<typeof AgentFeatureConfigSchema>;
```

***

<a id="api-agentresumerequest"></a>

### AgentResumeRequest

```ts
type AgentResumeRequest = z.infer<typeof AgentResumeRequestSchema>;
```

***

<a id="api-agentresumeresponse"></a>

### AgentResumeResponse

```ts
type AgentResumeResponse = z.infer<typeof AgentResumeResponseSchema>;
```

***

<a id="api-agentstartstreamrequest"></a>

### AgentStartStreamRequest

```ts
type AgentStartStreamRequest = z.infer<typeof AgentStartStreamRequestSchema>;
```

***

<a id="api-attemptcontext"></a>

### AttemptContext

```ts
type AttemptContext = Message<"mongodb.agentic.workflow.v1.AttemptContext"> & object;
```

AttemptContext is the OE-issued identity and fence for one runtime attempt.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 1; |
| `branchLineage?` | [`BranchLineage`](#api-branchlineage) | branch_lineage is present only for a branched execution and identifies the immutable source cutoff represented by previous_state. **Generated** from field: mongodb.agentic.workflow.v1.BranchLineage branch_lineage = 9; |
| `declaration?` | [`WorkflowDeclaration`](#api-workflowdeclaration) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowDeclaration declaration = 6; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 2; |
| `heartbeatIntervalMs` | `bigint` | **Generated** from field: int64 heartbeat_interval_ms = 7; |
| `ownerId` | `string` | **Generated** from field: string owner_id = 3; |
| `previousState?` | [`StateSnapshot`](#api-statesnapshot) | **Generated** from field: mongodb.agentic.workflow.v1.StateSnapshot previous_state = 8; |
| `replayMode` | `boolean` | **Generated** from field: bool replay_mode = 4; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 5; |

#### Generated

from message mongodb.agentic.workflow.v1.AttemptContext

***

<a id="api-attemptheartbeatrequest"></a>

### AttemptHeartbeatRequest

```ts
type AttemptHeartbeatRequest = Message<"mongodb.agentic.workflow.v1.AttemptHeartbeatRequest"> & object;
```

AttemptHeartbeatRequest renews one attempt lease under its current fence.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 2; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 3; |
| `ownerId` | `string` | **Generated** from field: string owner_id = 4; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.AttemptHeartbeatRequest

***

<a id="api-attemptstartrequest"></a>

### AttemptStartRequest

```ts
type AttemptStartRequest = Message<"mongodb.agentic.workflow.v1.AttemptStartRequest"> & object;
```

AttemptStartRequest asks OE to activate an attempt for an execution.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `declaration?` | [`WorkflowDeclaration`](#api-workflowdeclaration) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowDeclaration declaration = 3; |
| `ownerId` | `string` | **Generated** from field: string owner_id = 2; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.AttemptStartRequest

***

<a id="api-attemptstartresponse"></a>

### AttemptStartResponse

```ts
type AttemptStartResponse = Message<"mongodb.agentic.workflow.v1.AttemptStartResponse"> & object;
```

AttemptStartResponse returns the accepted attempt or a stable error.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `attemptContext?` | [`AttemptContext`](#api-attemptcontext) | **Generated** from field: mongodb.agentic.workflow.v1.AttemptContext attempt_context = 1; |
| `error?` | [`WorkflowError`](#api-workflowerror) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowError error = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.AttemptStartResponse

***

<a id="api-branchlineage"></a>

### BranchLineage

```ts
type BranchLineage = Message<"mongodb.agentic.workflow.v1.BranchLineage"> & object;
```

BranchLineage identifies the immutable source cutoff for a branch attempt.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `sourceStateHash` | `string` | source_state_hash is the canonical hash of the committed source snapshot at source_step_ordinal. **Generated** from field: string source_state_hash = 3; |
| `sourceStepOrdinal` | `bigint` | source_step_ordinal is the positive committed-step ordinal whose snapshot seeds the branch. Branch-local steps begin after this cutoff. **Generated** from field: int64 source_step_ordinal = 2; |
| `sourceWorkflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | source_workflow_identity identifies the source execution in the branch's organization, project, and workspace. **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity source_workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.BranchLineage

***

<a id="api-completeexecutioncommand"></a>

### CompleteExecutionCommand

```ts
type CompleteExecutionCommand = Message<"mongodb.agentic.workflow.v1.CompleteExecutionCommand"> & object;
```

CompleteExecutionCommand commits one successful execution's final state for
the next invocation in the session.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 2; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 3; |
| `state?` | [`StateSnapshot`](#api-statesnapshot) | **Generated** from field: mongodb.agentic.workflow.v1.StateSnapshot state = 4; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.CompleteExecutionCommand

***

<a id="api-costbymodel"></a>

### CostByModel

```ts
type CostByModel = z.infer<typeof CostByModelSchema>;
```

***

<a id="api-costbyworkspace"></a>

### CostByWorkspace

```ts
type CostByWorkspace = z.infer<typeof CostByWorkspaceSchema>;
```

***

<a id="api-costdashboardresponse"></a>

### CostDashboardResponse

```ts
type CostDashboardResponse = z.infer<typeof CostDashboardResponseSchema>;
```

***

<a id="api-costsummary"></a>

### CostSummary

```ts
type CostSummary = z.infer<typeof CostSummarySchema>;
```

***

<a id="api-dailycostentry"></a>

### DailyCostEntry

```ts
type DailyCostEntry = z.infer<typeof DailyCostEntrySchema>;
```

***

<a id="api-elicitationinfo"></a>

### ElicitationInfo

```ts
type ElicitationInfo = z.infer<typeof ElicitationInfoSchema>;
```

***

<a id="api-executerequest"></a>

### ExecuteRequest

```ts
type ExecuteRequest = z.infer<typeof ExecuteRequestSchema>;
```

***

<a id="api-execution"></a>

### Execution

```ts
type Execution = z.infer<typeof ExecutionSchema>;
```

***

<a id="api-executiondetailqueryresponse"></a>

### ExecutionDetailQueryResponse

```ts
type ExecutionDetailQueryResponse = z.infer<typeof ExecutionDetailQueryResponseSchema>;
```

***

<a id="api-executiondocument"></a>

### ExecutionDocument

```ts
type ExecutionDocument = z.infer<typeof ExecutionDocumentSchema>;
```

***

<a id="api-executionlogsqueryresponse"></a>

### ExecutionLogsQueryResponse

```ts
type ExecutionLogsQueryResponse = z.infer<typeof ExecutionLogsQueryResponseSchema>;
```

***

<a id="api-executionslistqueryresponse"></a>

### ExecutionsListQueryResponse

```ts
type ExecutionsListQueryResponse = z.infer<typeof ExecutionsListQueryResponseSchema>;
```

***

<a id="api-executionstatus"></a>

### ExecutionStatus

```ts
type ExecutionStatus = z.infer<typeof ExecutionStatusSchema>;
```

***

<a id="api-executionstatusresponse"></a>

### ExecutionStatusResponse

```ts
type ExecutionStatusResponse = z.infer<typeof ExecutionStatusResponseSchema>;
```

***

<a id="api-executionstep"></a>

### ExecutionStep

```ts
type ExecutionStep = z.infer<typeof ExecutionStepSchema>;
```

***

<a id="api-executorcallbackrequest"></a>

### ExecutorCallbackRequest

```ts
type ExecutorCallbackRequest = z.infer<typeof ExecutorCallbackRequestSchema>;
```

***

<a id="api-featurename"></a>

### FeatureName

```ts
type FeatureName = keyof AgentFeatureConfig;
```

Derived from `AgentFeatureConfigSchema` — do not maintain a parallel union.

***

<a id="api-finalizestepcommand"></a>

### FinalizeStepCommand

```ts
type FinalizeStepCommand = Message<"mongodb.agentic.workflow.v1.FinalizeStepCommand"> & object;
```

FinalizeStepCommand reaches one framework boundary. A settled root step
supplies state; an interrupted step supplies its complete suspension frontier.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `attemptId` | `string` | **Generated** from field: string attempt_id = 2; |
| `fencingToken` | `bigint` | **Generated** from field: int64 fencing_token = 3; |
| `observedActivityPositions` | [`ActivityPosition`](#api-activityposition)[] | observed_activity_positions reports the activity positions this attempt encountered before finalization. OE validates completeness against history. **Generated** from field: repeated mongodb.agentic.workflow.v1.ActivityPosition observed_activity_positions = 6; |
| `state?` | [`StateSnapshot`](#api-statesnapshot) | **Generated** from field: mongodb.agentic.workflow.v1.StateSnapshot state = 4; |
| `stepOrdinal` | `bigint` | **Generated** from field: int64 step_ordinal = 5; |
| `suspensions` | [`StepSuspensionEntry`](#api-stepsuspensionentry)[] | **Generated** from field: repeated mongodb.agentic.workflow.v1.StepSuspensionEntry suspensions = 7; |
| `workflowIdentity?` | [`WorkflowIdentity`](#api-workflowidentity) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowIdentity workflow_identity = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.FinalizeStepCommand

***

<a id="api-finalizestepresponse"></a>

### FinalizeStepResponse

```ts
type FinalizeStepResponse = Message<"mongodb.agentic.workflow.v1.FinalizeStepResponse"> & object;
```

FinalizeStepResponse returns every positioned outcome for an interrupted
frontier and no entries for a committed root step.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `entries` | [`StepActivityEntry`](#api-stepactivityentry)[] | **Generated** from field: repeated mongodb.agentic.workflow.v1.StepActivityEntry entries = 1; |
| `error?` | [`WorkflowError`](#api-workflowerror) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowError error = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.FinalizeStepResponse

***

<a id="api-guardrailcheckcontext"></a>

### GuardrailCheckContext

```ts
type GuardrailCheckContext = z.infer<typeof GuardrailCheckContextSchema>;
```

***

<a id="api-guardrailcheckdecision"></a>

### GuardrailCheckDecision

```ts
type GuardrailCheckDecision = z.infer<typeof GuardrailCheckDecisionSchema>;
```

***

<a id="api-guardrailcheckevidence"></a>

### GuardrailCheckEvidence

```ts
type GuardrailCheckEvidence = z.infer<typeof GuardrailCheckEvidenceSchema>;
```

***

<a id="api-guardrailcheckinput"></a>

### GuardrailCheckInput

```ts
type GuardrailCheckInput = z.infer<typeof GuardrailCheckInputSchema>;
```

***

<a id="api-guardrailcheckrequest"></a>

### GuardrailCheckRequest

```ts
type GuardrailCheckRequest = z.infer<typeof GuardrailCheckRequestSchema>;
```

***

<a id="api-guardrailcheckresponse"></a>

### GuardrailCheckResponse

```ts
type GuardrailCheckResponse = z.infer<typeof GuardrailCheckResponseSchema>;
```

***

<a id="api-guardrailmeta"></a>

### GuardrailMeta

```ts
type GuardrailMeta = z.infer<typeof GuardrailMetaSchema>;
```

***

<a id="api-guardrailruntimepolicy"></a>

### GuardrailRuntimePolicy

```ts
type GuardrailRuntimePolicy = z.infer<typeof GuardrailRuntimePolicySchema>;
```

***

<a id="api-guardrailruntimestage"></a>

### GuardrailRuntimeStage

```ts
type GuardrailRuntimeStage = z.infer<typeof GuardrailRuntimeStageSchema>;
```

***

<a id="api-healthresponse"></a>

### HealthResponse

```ts
type HealthResponse = z.infer<typeof HealthResponseSchema>;
```

***

<a id="api-healthstatus"></a>

### HealthStatus

```ts
type HealthStatus = z.infer<typeof HealthStatusSchema>;
```

***

<a id="api-humanreviewdata"></a>

### HumanReviewData

```ts
type HumanReviewData = z.infer<typeof HumanReviewDataSchema>;
```

***

<a id="api-instrumentor"></a>

### Instrumentor

```ts
type Instrumentor = () => void;
```

#### Returns

`void`

***

<a id="api-interruptresult"></a>

### InterruptResult

```ts
type InterruptResult = z.infer<typeof InterruptResultSchema>;
```

***

<a id="api-invokellmrequestarguments"></a>

### InvokeLLMRequestArguments

```ts
type InvokeLLMRequestArguments = z.infer<typeof InvokeLLMRequestArgumentsSchema>;
```

***

<a id="api-invokerequest"></a>

### InvokeRequest

```ts
type InvokeRequest = z.infer<typeof InvokeRequestSchema>;
```

***

<a id="api-invokeresponse"></a>

### InvokeResponse

```ts
type InvokeResponse = z.infer<typeof InvokeResponseSchema>;
```

***

<a id="api-jsonvalue"></a>

### JsonValue

```ts
type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | {
[key: string]: JsonValue;
};
```

***

<a id="api-llmadapterfactory"></a>

### LLMAdapterFactory

```ts
type LLMAdapterFactory = (rawLlm, options?) => BaseLLM;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `rawLlm` | `unknown` |
| `options?` | \{ `tool_choice?`: `unknown`; `tools?`: `unknown`[]; \} |
| `options.tool_choice?` | `unknown` |
| `options.tools?` | `unknown`[] |

#### Returns

`BaseLLM`

***

<a id="api-llmpodinvokerequest"></a>

### LLMPodInvokeRequest

```ts
type LLMPodInvokeRequest = z.infer<typeof LLMPodInvokeRequestSchema>;
```

***

<a id="api-llmpodinvokeresponse"></a>

### LLMPodInvokeResponse

```ts
type LLMPodInvokeResponse = z.infer<typeof LLMPodInvokeResponseSchema>;
```

***

<a id="api-llmpodstreamevent"></a>

### LLMPodStreamEvent

```ts
type LLMPodStreamEvent = z.infer<typeof LLMPodStreamEventSchema>;
```

***

<a id="api-memorywrite"></a>

### MemoryWrite

```ts
type MemoryWrite = Message<"mongodb.agentic.workflow.v1.MemoryWrite"> & object;
```

MemoryWrite is an exact short-term Memory request produced by a wrapper.
OE forwards payload_json without interpreting framework messages.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `id` | `string` | **Generated** from field: string id = 1; |
| `payloadJson` | `Uint8Array` | **Generated** from field: bytes payload_json = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.MemoryWrite

***

<a id="api-nodeexecutionrequest"></a>

### NodeExecutionRequest

```ts
type NodeExecutionRequest = z.infer<typeof NodeExecutionRequestSchema>;
```

***

<a id="api-nodeexecutionsqueryresponse"></a>

### NodeExecutionsQueryResponse

```ts
type NodeExecutionsQueryResponse = z.infer<typeof NodeExecutionsQueryResponseSchema>;
```

***

<a id="api-operationpath"></a>

### OperationPath

```ts
type OperationPath = Message<"mongodb.agentic.workflow.v1.OperationPath"> & object;
```

OperationPath is an absolute deterministic path within one execution.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `segments` | [`OperationPathSegment`](#api-operationpathsegment)[] | **Generated** from field: repeated mongodb.agentic.workflow.v1.OperationPathSegment segments = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.OperationPath

***

<a id="api-operationpathresolver"></a>

### OperationPathResolver

```ts
type OperationPathResolver = () => readonly ChildOperationBoundary[];
```

#### Returns

readonly [`ChildOperationBoundary`](#api-childoperationboundary)[]

***

<a id="api-operationpathsegment"></a>

### OperationPathSegment

```ts
type OperationPathSegment = Message<"mongodb.agentic.workflow.v1.OperationPathSegment"> & object;
```

OperationPathSegment is one deterministic position in an operation path.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `name` | `string` | **Generated** from field: string name = 1; |
| `ordinal` | `bigint` | **Generated** from field: int64 ordinal = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.OperationPathSegment

***

<a id="api-pendinginterrupt"></a>

### PendingInterrupt

```ts
type PendingInterrupt = z.infer<typeof PendingInterruptSchema>;
```

***

<a id="api-runtimemcpauthconfig"></a>

### RuntimeMCPAuthConfig

```ts
type RuntimeMCPAuthConfig = z.infer<typeof RuntimeMCPAuthConfigSchema>;
```

***

<a id="api-runtimemcpconfig"></a>

### RuntimeMCPConfig

```ts
type RuntimeMCPConfig = z.infer<typeof RuntimeMCPConfigSchema>;
```

***

<a id="api-runtimemcpserverconfig"></a>

### RuntimeMCPServerConfig

```ts
type RuntimeMCPServerConfig = z.infer<typeof RuntimeMCPServerConfigSchema>;
```

***

<a id="api-secretsconfig"></a>

### SecretsConfig

```ts
type SecretsConfig = z.infer<typeof SecretsConfigSchema>;
```

***

<a id="api-secretsconfiginput"></a>

### SecretsConfigInput

```ts
type SecretsConfigInput = z.input<typeof SecretsConfigSchema>;
```

Input shape callers may construct: every key is omittable, since Zod defaults
fill them in. The parsed [SecretsConfig](#api-secretsconfig) has them all present.

***

<a id="api-servertoolfn"></a>

### ServerToolFn

```ts
type ServerToolFn = (args) => unknown;
```

Tool function signature — receives the full arguments Record, may return a Promise.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`unknown`

***

<a id="api-sessionfinishstatus"></a>

### SessionFinishStatus

```ts
type SessionFinishStatus = "requested" | "already_requested" | "unavailable";
```

***

<a id="api-sessioninfo"></a>

### SessionInfo

```ts
type SessionInfo = z.infer<typeof SessionInfoSchema>;
```

***

<a id="api-sessionmessage"></a>

### SessionMessage

```ts
type SessionMessage = z.infer<typeof SessionMessageSchema>;
```

***

<a id="api-sessionmessagesqueryresponse"></a>

### SessionMessagesQueryResponse

```ts
type SessionMessagesQueryResponse = z.infer<typeof SessionMessagesQueryResponseSchema>;
```

***

<a id="api-sessionsqueryresponse"></a>

### SessionsQueryResponse

```ts
type SessionsQueryResponse = z.infer<typeof SessionsQueryResponseSchema>;
```

***

<a id="api-startactivityresult"></a>

### StartActivityResult

```ts
type StartActivityResult =
  | ActivityDispatch
  | ActivityReplay;
```

***

<a id="api-statesnapshot"></a>

### StateSnapshot

```ts
type StateSnapshot = Message<"mongodb.agentic.workflow.v1.StateSnapshot"> & object;
```

StateSnapshot is the complete application state needed by the next turn.
Properties excludes the conventional messages channel.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `messageEncodingVersion` | `number` | Version 0 is the legacy source_message encoding. Version 1 means every message carries the explicit fields required for framework reconstruction. **Generated** from field: uint32 message_encoding_version = 3; |
| `messages` | [`WorkflowMessage`](#api-workflowmessage)[] | **Generated** from field: repeated mongodb.agentic.workflow.v1.WorkflowMessage messages = 2; |
| `properties?` | `JsonObject` | **Generated** from field: google.protobuf.Struct properties = 1; |
| `replayProperties?` | `JsonObject` | Adapter-selected application state used only for replacement replay comparison. When absent, properties is the replay comparison input. **Generated** from field: google.protobuf.Struct replay_properties = 4; |

#### Generated

from message mongodb.agentic.workflow.v1.StateSnapshot

***

<a id="api-stepactivityentry"></a>

### StepActivityEntry

```ts
type StepActivityEntry = Message<"mongodb.agentic.workflow.v1.StepActivityEntry"> & object;
```

StepActivityEntry binds a durable activity outcome to its immutable
position. Response ordering has no meaning.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `outcome?` | [`ActivityOutcome`](#api-activityoutcome) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityOutcome outcome = 2; |
| `position?` | [`ActivityPosition`](#api-activityposition) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityPosition position = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.StepActivityEntry

***

<a id="api-stepsuspensionentry"></a>

### StepSuspensionEntry

```ts
type StepSuspensionEntry = Message<"mongodb.agentic.workflow.v1.StepSuspensionEntry"> & object;
```

StepSuspensionEntry describes one activity discovered suspended when its
framework reaches a quiescent step boundary. FinalizeStep admits the complete
set atomically, so no partial suspension frontier becomes durable.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `activityKind` | [`ActivityKind`](#api-activitykind) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityKind activity_kind = 2; |
| `activityName` | `string` | **Generated** from field: string activity_name = 3; |
| `position?` | [`ActivityPosition`](#api-activityposition) | **Generated** from field: mongodb.agentic.workflow.v1.ActivityPosition position = 1; |
| `semanticInput?` | `Value` | **Generated** from field: google.protobuf.Value semantic_input = 4; |
| `suspension?` | [`ActivitySuspension`](#api-activitysuspension) | **Generated** from field: mongodb.agentic.workflow.v1.ActivitySuspension suspension = 5; |

#### Generated

from message mongodb.agentic.workflow.v1.StepSuspensionEntry

***

<a id="api-streamchunk"></a>

### StreamChunk

```ts
type StreamChunk = z.infer<typeof StreamChunkSchema>;
```

***

<a id="api-streamingresult"></a>

### StreamingResult

```ts
type StreamingResult = z.infer<typeof StreamingResultSchema>;
```

***

<a id="api-suspendhandler"></a>

### SuspendHandler

```ts
type SuspendHandler = (payload) => Record<string, unknown>;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `payload` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-suspendpayload"></a>

### SuspendPayload

```ts
type SuspendPayload = z.infer<typeof SuspendPayloadSchema>;
```

***

<a id="api-tenantscope"></a>

### TenantScope

```ts
type TenantScope = Message<"mongodb.agentic.workflow.v1.TenantScope"> & object;
```

TenantScope identifies the data-plane tenant boundary for a workflow.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `orgId` | `string` | **Generated** from field: string org_id = 1; |
| `projectId` | `string` | **Generated** from field: string project_id = 2; |
| `workspaceId` | `string` | **Generated** from field: string workspace_id = 3; |

#### Generated

from message mongodb.agentic.workflow.v1.TenantScope

***

<a id="api-toolapierror"></a>

### ToolAPIError

```ts
type ToolAPIError = z.infer<typeof ToolAPIErrorSchema>;
```

***

<a id="api-toolauthorization"></a>

### ToolAuthorization

```ts
type ToolAuthorization = z.infer<typeof ToolAuthorizationSchema>;
```

***

<a id="api-tooldefinition"></a>

### ToolDefinition

```ts
type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;
```

***

<a id="api-toolexecuterequest"></a>

### ToolExecuteRequest

```ts
type ToolExecuteRequest = z.infer<typeof ToolExecuteRequestSchema>;
```

***

<a id="api-toolexecuteresponse"></a>

### ToolExecuteResponse

```ts
type ToolExecuteResponse = z.infer<typeof ToolExecuteResponseSchema>;
```

***

<a id="api-toolfunctionrequest"></a>

### ToolFunctionRequest

```ts
type ToolFunctionRequest = z.infer<typeof ToolFunctionRequestSchema>;
```

***

<a id="api-toolpodexecuterequest"></a>

### ToolPodExecuteRequest

```ts
type ToolPodExecuteRequest = z.infer<typeof ToolPodExecuteRequestSchema>;
```

***

<a id="api-toolpodexecuteresponse"></a>

### ToolPodExecuteResponse

```ts
type ToolPodExecuteResponse = z.infer<typeof ToolPodExecuteResponseSchema>;
```

***

<a id="api-toolresponseformat"></a>

### ToolResponseFormat

```ts
type ToolResponseFormat = "content" | "content_and_artifact";
```

***

<a id="api-toolresultrequest"></a>

### ToolResultRequest

```ts
type ToolResultRequest = z.infer<typeof ToolResultRequestSchema>;
```

***

<a id="api-toolslistresponse"></a>

### ToolsListResponse

```ts
type ToolsListResponse = z.infer<typeof ToolsListResponseSchema>;
```

***

<a id="api-workflowdeclaration"></a>

### WorkflowDeclaration

```ts
type WorkflowDeclaration = Message<"mongodb.agentic.workflow.v1.WorkflowDeclaration"> & object;
```

WorkflowDeclaration describes the immutable application and adapter contract.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `adapterName` | `string` | **Generated** from field: string adapter_name = 3; |
| `adapterVersion` | `string` | **Generated** from field: string adapter_version = 4; |
| `memoryEnabled` | `boolean` | memory_enabled pins agent.yaml features.memory for the execution so replacement attempts use the original synchronization contract. **Generated** from field: bool memory_enabled = 5; |
| `workflowName` | `string` | **Generated** from field: string workflow_name = 1; |
| `workflowVersion` | `string` | **Generated** from field: string workflow_version = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.WorkflowDeclaration

***

<a id="api-workflowerror"></a>

### WorkflowError

```ts
type WorkflowError = Message<"mongodb.agentic.workflow.v1.WorkflowError"> & object;
```

WorkflowError carries a stable code and a redacted diagnostic message.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `code` | [`WorkflowErrorCode`](#api-workflowerrorcode) | **Generated** from field: mongodb.agentic.workflow.v1.WorkflowErrorCode code = 1; |
| `message` | `string` | **Generated** from field: string message = 2; |

#### Generated

from message mongodb.agentic.workflow.v1.WorkflowError

***

<a id="api-workflowidentity"></a>

### WorkflowIdentity

```ts
type WorkflowIdentity = Message<"mongodb.agentic.workflow.v1.WorkflowIdentity"> & object;
```

WorkflowIdentity identifies one session and one durable execution.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `executionId` | `string` | **Generated** from field: string execution_id = 3; |
| `sessionId` | `string` | **Generated** from field: string session_id = 2; |
| `tenantScope?` | [`TenantScope`](#api-tenantscope) | **Generated** from field: mongodb.agentic.workflow.v1.TenantScope tenant_scope = 1; |

#### Generated

from message mongodb.agentic.workflow.v1.WorkflowIdentity

***

<a id="api-workflowmessage"></a>

### WorkflowMessage

```ts
type WorkflowMessage = Message<"mongodb.agentic.workflow.v1.WorkflowMessage"> & object;
```

WorkflowMessage is the durable, framework-neutral conversation value.
Framework adapters reconstruct their native message types from these
explicit fields; framework serializer output is not durable state.

#### Type Declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `additionalKwargs?` | `JsonObject` | **Generated** from field: google.protobuf.Struct additional_kwargs = 11; |
| `artifacts` | `JsonObject`[] | Deprecated platform attachment field paired with source_message. **Generated** from field: repeated google.protobuf.Struct artifacts = 10 [deprecated = true]; **Deprecated** |
| `content?` | `Value` | **Generated** from field: google.protobuf.Value content = 2; |
| `id?` | `string` | **Generated** from field: optional string id = 6; |
| `invalidToolCalls` | `JsonObject`[] | **Generated** from field: repeated google.protobuf.Struct invalid_tool_calls = 16; |
| `isError?` | `boolean` | **Generated** from field: optional bool is_error = 13; |
| `name?` | `string` | **Generated** from field: optional string name = 5; |
| `platformArtifacts` | `JsonObject`[] | Platform artifacts are platform-owned message attachments surfaced through message metadata. They are distinct from a tool result's tool_artifact. **Generated** from field: repeated google.protobuf.Struct platform_artifacts = 15; |
| `responseMetadata?` | `JsonObject` | **Generated** from field: google.protobuf.Struct response_metadata = 12; |
| `role` | [`MessageRole`](#api-messagerole) | **Generated** from field: mongodb.agentic.workflow.v1.MessageRole role = 1; |
| `sourceMessage?` | `Value` | Deprecated compatibility envelope for states written before explicit message fields were introduced. Adapters read it during replay but do not write it for new executions. **Generated** from field: google.protobuf.Value source_message = 9 [deprecated = true]; **Deprecated** |
| `toolArtifact?` | `Value` | Tool artifact is machine-readable result data kept off the model-facing content channel. It may contain any JSON value. **Generated** from field: google.protobuf.Value tool_artifact = 14; |
| `toolCallId?` | `string` | **Generated** from field: optional string tool_call_id = 4; |
| `toolCalls` | `JsonObject`[] | **Generated** from field: repeated google.protobuf.Struct tool_calls = 3; |

#### Generated

from message mongodb.agentic.workflow.v1.WorkflowMessage

## Variables

<a id="api-activitycommandschema"></a>

### ActivityCommandSchema

```ts
const ActivityCommandSchema: GenMessage<ActivityCommand>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityCommand.
Use `create(ActivityCommandSchema)` to create a new message.

***

<a id="api-activitycontextschema"></a>

### ActivityContextSchema

```ts
const ActivityContextSchema: GenMessage<ActivityContext>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityContext.
Use `create(ActivityContextSchema)` to create a new message.

***

<a id="api-activityheartbeatrequestschema"></a>

### ActivityHeartbeatRequestSchema

```ts
const ActivityHeartbeatRequestSchema: GenMessage<ActivityHeartbeatRequest>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityHeartbeatRequest.
Use `create(ActivityHeartbeatRequestSchema)` to create a new message.

***

<a id="api-activitykindschema"></a>

### ActivityKindSchema

```ts
const ActivityKindSchema: GenEnum<ActivityKind>;
```

Describes the enum mongodb.agentic.workflow.v1.ActivityKind.

***

<a id="api-activitymemorycommandschema"></a>

### ActivityMemoryCommandSchema

```ts
const ActivityMemoryCommandSchema: GenMessage<ActivityMemoryCommand>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityMemoryCommand.
Use `create(ActivityMemoryCommandSchema)` to create a new message.

***

<a id="api-activityoutcomekindschema"></a>

### ActivityOutcomeKindSchema

```ts
const ActivityOutcomeKindSchema: GenEnum<ActivityOutcomeKind>;
```

Describes the enum mongodb.agentic.workflow.v1.ActivityOutcomeKind.

***

<a id="api-activityoutcomeschema"></a>

### ActivityOutcomeSchema

```ts
const ActivityOutcomeSchema: GenMessage<ActivityOutcome>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityOutcome.
Use `create(ActivityOutcomeSchema)` to create a new message.

***

<a id="api-activitypositionschema"></a>

### ActivityPositionSchema

```ts
const ActivityPositionSchema: GenMessage<ActivityPosition>;
```

Describes the message mongodb.agentic.workflow.v1.ActivityPosition.
Use `create(ActivityPositionSchema)` to create a new message.

***

<a id="api-activitysuspensionschema"></a>

### ActivitySuspensionSchema

```ts
const ActivitySuspensionSchema: GenMessage<ActivitySuspension>;
```

Describes the message mongodb.agentic.workflow.v1.ActivitySuspension.
Use `create(ActivitySuspensionSchema)` to create a new message.

***

<a id="api-aer_build_agent"></a>

### AER\_BUILD\_AGENT

```ts
const AER_BUILD_AGENT: "aer.build_agent" = "aer.build_agent";
```

Stable span names for the first-invoke lifecycle. There is no manual
`llm.call` span; LangChain's auto-instrumentation already covers it.

***

<a id="api-aerexecuteresponseschema"></a>

### AERExecuteResponseSchema

```ts
const AERExecuteResponseSchema: ZodObject<{
  result: ZodOptional<ZodString>;
  status: ZodString;
  suspend_reason: ZodOptional<ZodString>;
}, $strip>;
```

Response from AER /execute endpoint.

Returned on both normal completion and HITL suspension.

***

<a id="api-agentfeatureconfigschema"></a>

### AgentFeatureConfigSchema

```ts
const AgentFeatureConfigSchema: ZodObject<{
  deep_agent: ZodDefault<ZodNullable<ZodBoolean>>;
  durable_workflow: ZodDefault<ZodNullable<ZodBoolean>>;
  guardrails: ZodDefault<ZodNullable<ZodBoolean>>;
  memory: ZodDefault<ZodNullable<ZodBoolean>>;
  playground: ZodDefault<ZodNullable<ZodBoolean>>;
  use_custom_parser: ZodDefault<ZodNullable<ZodBoolean>>;
}, $strip>;
```

Runtime feature flags from `agent.yaml`.

Keep field names in parity with Python `AgentFeatureConfig` and the CLI
allowlist. Add a flag by adding a field here; `FeatureName` and runtime
feature access derive from it. `null` means "omitted", letting the runtime
fall back to legacy environment variables.

***

<a id="api-agentresumerequestschema"></a>

### AgentResumeRequestSchema

```ts
const AgentResumeRequestSchema: ZodObject<{
  custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
  human_review: ZodObject<{
     decision: ZodString;
     reviewer_notes: ZodOptional<ZodString>;
  }, $strip>;
}, $strip>;
```

Request to resume a suspended execution.

***

<a id="api-agentresumeresponseschema"></a>

### AgentResumeResponseSchema

```ts
const AgentResumeResponseSchema: ZodObject<{
  execution_id: ZodString;
  status: ZodString;
}, $strip>;
```

Response from resuming an execution.

***

<a id="api-agentstartstreamrequestschema"></a>

### AgentStartStreamRequestSchema

```ts
const AgentStartStreamRequestSchema: ZodObject<{
  message: ZodString;
  org_id: ZodOptional<ZodString>;
  session_id: ZodOptional<ZodString>;
  user_id: ZodOptional<ZodString>;
}, $strip>;
```

Request to start an agent execution with streaming response.

***

<a id="api-attemptcontextschema"></a>

### AttemptContextSchema

```ts
const AttemptContextSchema: GenMessage<AttemptContext>;
```

Describes the message mongodb.agentic.workflow.v1.AttemptContext.
Use `create(AttemptContextSchema)` to create a new message.

***

<a id="api-attemptheartbeatrequestschema"></a>

### AttemptHeartbeatRequestSchema

```ts
const AttemptHeartbeatRequestSchema: GenMessage<AttemptHeartbeatRequest>;
```

Describes the message mongodb.agentic.workflow.v1.AttemptHeartbeatRequest.
Use `create(AttemptHeartbeatRequestSchema)` to create a new message.

***

<a id="api-attemptstartrequestschema"></a>

### AttemptStartRequestSchema

```ts
const AttemptStartRequestSchema: GenMessage<AttemptStartRequest>;
```

Describes the message mongodb.agentic.workflow.v1.AttemptStartRequest.
Use `create(AttemptStartRequestSchema)` to create a new message.

***

<a id="api-attemptstartresponseschema"></a>

### AttemptStartResponseSchema

```ts
const AttemptStartResponseSchema: GenMessage<AttemptStartResponse>;
```

Describes the message mongodb.agentic.workflow.v1.AttemptStartResponse.
Use `create(AttemptStartResponseSchema)` to create a new message.

***

<a id="api-attr_cache_hit"></a>

### ATTR\_CACHE\_HIT

```ts
const ATTR_CACHE_HIT: "cache_hit" = "cache_hit";
```

***

<a id="api-attr_cold_start"></a>

### ATTR\_COLD\_START

```ts
const ATTR_COLD_START: "cold_start" = "cold_start";
```

***

<a id="api-attr_skills_loaded_count"></a>

### ATTR\_SKILLS\_LOADED\_COUNT

```ts
const ATTR_SKILLS_LOADED_COUNT: "skills.loaded_count" = "skills.loaded_count";
```

***

<a id="api-attr_skills_source_count"></a>

### ATTR\_SKILLS\_SOURCE\_COUNT

```ts
const ATTR_SKILLS_SOURCE_COUNT: "skills.source_count" = "skills.source_count";
```

***

<a id="api-branchlineageschema"></a>

### BranchLineageSchema

```ts
const BranchLineageSchema: GenMessage<BranchLineage>;
```

Describes the message mongodb.agentic.workflow.v1.BranchLineage.
Use `create(BranchLineageSchema)` to create a new message.

***

<a id="api-builtin_tool_names"></a>

### BUILTIN\_TOOL\_NAMES

```ts
const BUILTIN_TOOL_NAMES: ReadonlySet<string>;
```

Canonical set of built-in tool names registered by `registerBuiltinTools`.
Consumed by `ToolServer.onStartup`'s completeness assertion and by tests so
the callsites can't drift silently.

***

<a id="api-call_interrupted_artifact_key"></a>

### CALL\_INTERRUPTED\_ARTIFACT\_KEY

```ts
const CALL_INTERRUPTED_ARTIFACT_KEY: "__agent_engine_oe_call_interrupted__" = "__agent_engine_oe_call_interrupted__";
```

Frozen interrupt-artifact wire key.

Dependency leaf (no imports): `workflow/memory.ts` and `secure_wrapper.ts`
both consume this without forming a module cycle through the workflow
barrel. The key is persisted into checkpoints and replay logs, and
duplicated in runner-shared/src/agent_engine_runner_shared/secure_wrapper.py. Changing
either value breaks interrupt detection on already-checkpointed sessions
and/or cross-language parity — keep the two in lockstep.

***

<a id="api-completeexecutioncommandschema"></a>

### CompleteExecutionCommandSchema

```ts
const CompleteExecutionCommandSchema: GenMessage<CompleteExecutionCommand>;
```

Describes the message mongodb.agentic.workflow.v1.CompleteExecutionCommand.
Use `create(CompleteExecutionCommandSchema)` to create a new message.

***

<a id="api-costbymodelschema"></a>

### CostByModelSchema

```ts
const CostByModelSchema: ZodObject<{
  call_count: ZodDefault<ZodNumber>;
  model: ZodString;
  percentage: ZodDefault<ZodNumber>;
  total_cost_usd: ZodDefault<ZodNumber>;
  total_tokens: ZodDefault<ZodNumber>;
}, $strip>;
```

Cost breakdown for a single model.

***

<a id="api-costbyworkspaceschema"></a>

### CostByWorkspaceSchema

```ts
const CostByWorkspaceSchema: ZodObject<{
  call_count: ZodDefault<ZodNumber>;
  percentage: ZodDefault<ZodNumber>;
  total_cost_usd: ZodDefault<ZodNumber>;
  total_tokens: ZodDefault<ZodNumber>;
  workspace_id: ZodString;
}, $strip>;
```

Cost breakdown for a single workspace.

***

<a id="api-costdashboardresponseschema"></a>

### CostDashboardResponseSchema

```ts
const CostDashboardResponseSchema: ZodObject<{
  by_model: ZodDefault<ZodArray<ZodObject<{
     call_count: ZodDefault<ZodNumber>;
     model: ZodString;
     percentage: ZodDefault<ZodNumber>;
     total_cost_usd: ZodDefault<ZodNumber>;
     total_tokens: ZodDefault<ZodNumber>;
  }, $strip>>>;
  by_workspace: ZodDefault<ZodArray<ZodObject<{
     call_count: ZodDefault<ZodNumber>;
     percentage: ZodDefault<ZodNumber>;
     total_cost_usd: ZodDefault<ZodNumber>;
     total_tokens: ZodDefault<ZodNumber>;
     workspace_id: ZodString;
  }, $strip>>>;
  daily_trend: ZodDefault<ZodArray<ZodObject<{
     call_count: ZodDefault<ZodNumber>;
     date: ZodString;
     total_cost_usd: ZodDefault<ZodNumber>;
     total_tokens: ZodDefault<ZodNumber>;
  }, $strip>>>;
  summary: ZodDefault<ZodObject<{
     total_completion_tokens: ZodDefault<ZodNumber>;
     total_cost_usd: ZodDefault<ZodNumber>;
     total_llm_calls: ZodDefault<ZodNumber>;
     total_prompt_tokens: ZodDefault<ZodNumber>;
     total_tokens: ZodDefault<ZodNumber>;
     unpriced_llm_calls: ZodDefault<ZodNumber>;
  }, $strip>>;
}, $strip>;
```

Response for cost dashboard aggregation (used by API Gateway proxy).

***

<a id="api-costsummaryschema"></a>

### CostSummarySchema

```ts
const CostSummarySchema: ZodObject<{
  total_completion_tokens: ZodDefault<ZodNumber>;
  total_cost_usd: ZodDefault<ZodNumber>;
  total_llm_calls: ZodDefault<ZodNumber>;
  total_prompt_tokens: ZodDefault<ZodNumber>;
  total_tokens: ZodDefault<ZodNumber>;
  unpriced_llm_calls: ZodDefault<ZodNumber>;
}, $strip>;
```

Aggregate cost metrics for the requested period.

***

<a id="api-dailycostentryschema"></a>

### DailyCostEntrySchema

```ts
const DailyCostEntrySchema: ZodObject<{
  call_count: ZodDefault<ZodNumber>;
  date: ZodString;
  total_cost_usd: ZodDefault<ZodNumber>;
  total_tokens: ZodDefault<ZodNumber>;
}, $strip>;
```

Cost data for a single day.

***

<a id="api-default_mcp_oauth_client_name"></a>

### DEFAULT\_MCP\_OAUTH\_CLIENT\_NAME

```ts
const DEFAULT_MCP_OAUTH_CLIENT_NAME: "Atlas Agent Engine Dev MCP Client" = "Atlas Agent Engine Dev MCP Client";
```

***

<a id="api-default_mcp_oauth_redirect_uri"></a>

### DEFAULT\_MCP\_OAUTH\_REDIRECT\_URI

```ts
const DEFAULT_MCP_OAUTH_REDIRECT_URI: "http://127.0.0.1:8765/callback" = "http://127.0.0.1:8765/callback";
```

***

<a id="api-default_ports"></a>

### DEFAULT\_PORTS

```ts
const DEFAULT_PORTS: Record<string, number>;
```

***

<a id="api-elicitationinfoschema"></a>

### ElicitationInfoSchema

```ts
const ElicitationInfoSchema: ZodObject<{
  authorization_url: ZodString;
  created: ZodDefault<ZodBoolean>;
  elicitation_id: ZodString;
  message: ZodDefault<ZodString>;
}, $strip>;
```

Authorization details returned when broker consent is required.

***

<a id="api-executerequestschema"></a>

### ExecuteRequestSchema

```ts
const ExecuteRequestSchema: ZodObject<{
  custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
  execution_id: ZodString;
  message: ZodDefault<ZodString>;
  metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  org_id: ZodOptional<ZodString>;
  payload: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  platform_api_owner_url: ZodOptional<ZodNullable<ZodString>>;
  platform_api_url: ZodString;
  platform_trace_id: ZodOptional<ZodNullable<ZodString>>;
  previous_execution_cancelled: ZodDefault<ZodBoolean>;
  project_id: ZodOptional<ZodString>;
  resume: ZodDefault<ZodBoolean>;
  resume_data: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  resume_from_step: ZodOptional<ZodNumber>;
  session_id: ZodOptional<ZodString>;
  suspend_generation: ZodOptional<ZodPipe<ZodNullable<ZodNumber>, ZodTransform<number | undefined, number | null>>>;
  user_id: ZodOptional<ZodString>;
  workspace_id: ZodOptional<ZodString>;
}, $strip>;
```

Request to execute an agent in AER.

***

<a id="api-executiondetailqueryresponseschema"></a>

### ExecutionDetailQueryResponseSchema

```ts
const ExecutionDetailQueryResponseSchema: ZodObject<{
  error: ZodOptional<ZodString>;
  execution: ZodOptional<ZodObject<{
     created_at: ZodOptional<ZodString>;
     error: ZodOptional<ZodString>;
     execution_id: ZodString;
     message: ZodDefault<ZodString>;
     org_id: ZodDefault<ZodString>;
     project_id: ZodDefault<ZodNullable<ZodString>>;
     result: ZodOptional<ZodUnknown>;
     session_id: ZodDefault<ZodString>;
     status: ZodDefault<ZodString>;
     suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     suspend_reason: ZodOptional<ZodString>;
     updated_at: ZodOptional<ZodString>;
     user_id: ZodDefault<ZodString>;
     workspace_id: ZodDefault<ZodNullable<ZodString>>;
  }, $strip>>;
  success: ZodDefault<ZodBoolean>;
}, $strip>;
```

Response for single execution detail query (used by API Gateway proxy).

***

<a id="api-executiondocumentschema"></a>

### ExecutionDocumentSchema

```ts
const ExecutionDocumentSchema: ZodObject<{
  created_at: ZodOptional<ZodString>;
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  message: ZodDefault<ZodString>;
  org_id: ZodDefault<ZodString>;
  project_id: ZodDefault<ZodNullable<ZodString>>;
  result: ZodOptional<ZodUnknown>;
  session_id: ZodDefault<ZodString>;
  status: ZodDefault<ZodString>;
  suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  suspend_reason: ZodOptional<ZodString>;
  updated_at: ZodOptional<ZodString>;
  user_id: ZodDefault<ZodString>;
  workspace_id: ZodDefault<ZodNullable<ZodString>>;
}, $strip>;
```

An execution document as stored in the platform database.

***

<a id="api-executionlogsqueryresponseschema"></a>

### ExecutionLogsQueryResponseSchema

```ts
const ExecutionLogsQueryResponseSchema: ZodObject<{
  count: ZodDefault<ZodNumber>;
  logs: ZodDefault<ZodArray<ZodRecord<ZodString, ZodUnknown>>>;
}, $strip>;
```

Response for execution logs query (used by API Gateway proxy).

***

<a id="api-executionschema"></a>

### ExecutionSchema

```ts
const ExecutionSchema: ZodObject<{
  aer_url: ZodOptional<ZodString>;
  created_at: ZodDefault<ZodCoercedDate<unknown>>;
  error: ZodOptional<ZodString>;
  id: ZodString;
  message: ZodString;
  org_id: ZodOptional<ZodString>;
  project_id: ZodOptional<ZodString>;
  result: ZodOptional<ZodUnknown>;
  session_id: ZodOptional<ZodString>;
  status: ZodEnum<{
     cancelled: "cancelled";
     completed: "completed";
     error: "error";
     pending: "pending";
     resuming: "resuming";
     running: "running";
     suspended: "suspended";
  }>;
  suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  suspend_reason: ZodOptional<ZodString>;
  tool_url: ZodOptional<ZodString>;
  updated_at: ZodDefault<ZodCoercedDate<unknown>>;
  user_id: ZodOptional<ZodString>;
  workspace_id: ZodOptional<ZodString>;
}, $strip>;
```

Execution record stored in TenantDB.

***

<a id="api-executionslistqueryresponseschema"></a>

### ExecutionsListQueryResponseSchema

```ts
const ExecutionsListQueryResponseSchema: ZodObject<{
  count: ZodDefault<ZodNumber>;
  executions: ZodDefault<ZodArray<ZodObject<{
     created_at: ZodOptional<ZodString>;
     error: ZodOptional<ZodString>;
     execution_id: ZodString;
     message: ZodDefault<ZodString>;
     org_id: ZodDefault<ZodString>;
     project_id: ZodDefault<ZodNullable<ZodString>>;
     result: ZodOptional<ZodUnknown>;
     session_id: ZodDefault<ZodString>;
     status: ZodDefault<ZodString>;
     suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     suspend_reason: ZodOptional<ZodString>;
     updated_at: ZodOptional<ZodString>;
     user_id: ZodDefault<ZodString>;
     workspace_id: ZodDefault<ZodNullable<ZodString>>;
  }, $strip>>>;
  success: ZodDefault<ZodBoolean>;
}, $strip>;
```

Response for executions list query (used by API Gateway proxy).

***

<a id="api-executionstatus-1"></a>

### ExecutionStatus

```ts
ExecutionStatus: object;
```

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-cancelled"></a> `CANCELLED` | `"cancelled"` | `"cancelled"` |
| <a id="api-property-completed"></a> `COMPLETED` | `"completed"` | `"completed"` |
| <a id="api-property-error-6"></a> `ERROR` | `"error"` | `"error"` |
| <a id="api-property-pending"></a> `PENDING` | `"pending"` | `"pending"` |
| <a id="api-property-resuming"></a> `RESUMING` | `"resuming"` | `"resuming"` |
| <a id="api-property-running"></a> `RUNNING` | `"running"` | `"running"` |
| <a id="api-property-suspended"></a> `SUSPENDED` | `"suspended"` | `"suspended"` |

***

<a id="api-executionstatusresponseschema"></a>

### ExecutionStatusResponseSchema

```ts
const ExecutionStatusResponseSchema: ZodObject<{
  created_at: ZodCoercedDate<unknown>;
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  result: ZodOptional<ZodUnknown>;
  status: ZodEnum<{
     cancelled: "cancelled";
     completed: "completed";
     error: "error";
     pending: "pending";
     resuming: "resuming";
     running: "running";
     suspended: "suspended";
  }>;
  suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  suspend_reason: ZodOptional<ZodString>;
  updated_at: ZodCoercedDate<unknown>;
}, $strip>;
```

Response for execution status query.

***

<a id="api-executionstatusschema"></a>

### ExecutionStatusSchema

```ts
const ExecutionStatusSchema: ZodEnum<{
  cancelled: "cancelled";
  completed: "completed";
  error: "error";
  pending: "pending";
  resuming: "resuming";
  running: "running";
  suspended: "suspended";
}>;
```

Status of an agent execution.

***

<a id="api-executionstepschema"></a>

### ExecutionStepSchema

```ts
const ExecutionStepSchema: ZodObject<{
  arguments: ZodRecord<ZodString, ZodUnknown>;
  duration_ms: ZodOptional<ZodNumber>;
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  id: ZodString;
  result: ZodOptional<ZodUnknown>;
  status: ZodString;
  step_number: ZodNumber;
  timestamp: ZodDefault<ZodCoercedDate<unknown>>;
  tool_name: ZodString;
}, $strip>;
```

Execution step record stored in TenantDB.

***

<a id="api-executorcallbackrequestschema"></a>

### ExecutorCallbackRequestSchema

```ts
const ExecutorCallbackRequestSchema: ZodObject<{
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  interrupts: ZodOptional<ZodArray<ZodObject<{
     id: ZodString;
     value: ZodUnknown;
  }, $strip>>>;
  metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  result: ZodOptional<ZodUnknown>;
  resume_schema: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  status: ZodString;
  suspend_context: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  suspend_generation: ZodOptional<ZodPipe<ZodNullable<ZodNumber>, ZodTransform<number | undefined, number | null>>>;
  suspend_reason: ZodOptional<ZodString>;
}, $strip>;
```

Callback from AER to OE when execution completes or suspends.

***

<a id="api-exit_import_error"></a>

### EXIT\_IMPORT\_ERROR

```ts
const EXIT_IMPORT_ERROR: 82 = 82;
```

Startup-failure exit codes. OE's readiness wait can observe a
workload's real exit code once it exits (via fctr's Wait RPC) but not the
exception that caused it, so the exit code itself is the only signal that
reliably survives a startup crash to reach OE. Chosen to avoid every range
fctr's own const.go already claims: 0/1 (generic), 64-78 (sysexits.h), and
128+signal (signal deaths, e.g. 137=SIGKILL, 143=SIGTERM).

Frozen wire contract: OE's classification of a startup failure depends on
these exact values, and they are duplicated in
runner-shared/src/agent_engine_runner_shared/launcher.py. Changing either file breaks
OE's ability to distinguish failure causes and/or cross-language parity —
keep the two in lockstep.

***

<a id="api-exit_no_entrypoint"></a>

### EXIT\_NO\_ENTRYPOINT

```ts
const EXIT_NO_ENTRYPOINT: 83 = 83;
```

***

<a id="api-exit_startup_crash"></a>

### EXIT\_STARTUP\_CRASH

```ts
const EXIT_STARTUP_CRASH: 84 = 84;
```

***

<a id="api-file_workflow_v1_activity"></a>

### file\_workflow\_v1\_activity

```ts
const file_workflow_v1_activity: GenFile;
```

Describes the file workflow/v1/activity.proto.

***

<a id="api-file_workflow_v1_common"></a>

### file\_workflow\_v1\_common

```ts
const file_workflow_v1_common: GenFile;
```

Describes the file workflow/v1/common.proto.

***

<a id="api-file_workflow_v1_runtime"></a>

### file\_workflow\_v1\_runtime

```ts
const file_workflow_v1_runtime: GenFile;
```

Describes the file workflow/v1/runtime.proto.

***

<a id="api-file_workflow_v1_state"></a>

### file\_workflow\_v1\_state

```ts
const file_workflow_v1_state: GenFile;
```

Describes the file workflow/v1/state.proto.

***

<a id="api-finalizestepcommandschema"></a>

### FinalizeStepCommandSchema

```ts
const FinalizeStepCommandSchema: GenMessage<FinalizeStepCommand>;
```

Describes the message mongodb.agentic.workflow.v1.FinalizeStepCommand.
Use `create(FinalizeStepCommandSchema)` to create a new message.

***

<a id="api-finalizestepresponseschema"></a>

### FinalizeStepResponseSchema

```ts
const FinalizeStepResponseSchema: GenMessage<FinalizeStepResponse>;
```

Describes the message mongodb.agentic.workflow.v1.FinalizeStepResponse.
Use `create(FinalizeStepResponseSchema)` to create a new message.

***

<a id="api-graph_build"></a>

### GRAPH\_BUILD

```ts
const GRAPH_BUILD: "graph.build" = "graph.build";
```

***

<a id="api-guardrailcheckcontextschema"></a>

### GuardrailCheckContextSchema

```ts
const GuardrailCheckContextSchema: ZodObject<{
  org_id: ZodString;
  project_id: ZodString;
  session_id: ZodOptional<ZodNullable<ZodString>>;
  user_id: ZodOptional<ZodNullable<ZodString>>;
  workspace_id: ZodOptional<ZodNullable<ZodString>>;
}, $strip>;
```

Execution context for a guardrail check.

***

<a id="api-guardrailcheckdecision-1"></a>

### GuardrailCheckDecision

```ts
GuardrailCheckDecision: object;
```

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-allow"></a> `ALLOW` | `"allow"` | `"allow"` |
| <a id="api-property-block"></a> `BLOCK` | `"block"` | `"block"` |
| <a id="api-property-log_only"></a> `LOG_ONLY` | `"log_only"` | `"log_only"` |
| <a id="api-property-modify"></a> `MODIFY` | `"modify"` | `"modify"` |
| <a id="api-property-require_review"></a> `REQUIRE_REVIEW` | `"require_review"` | `"require_review"` |

***

<a id="api-guardrailcheckdecisionschema"></a>

### GuardrailCheckDecisionSchema

```ts
const GuardrailCheckDecisionSchema: ZodEnum<{
  allow: "allow";
  block: "block";
  log_only: "log_only";
  modify: "modify";
  require_review: "require_review";
}>;
```

Decision returned by the Tool Pod guardrails evaluator.

***

<a id="api-guardrailcheckevidenceschema"></a>

### GuardrailCheckEvidenceSchema

```ts
const GuardrailCheckEvidenceSchema: ZodObject<{
  message: ZodDefault<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  policy_id: ZodString;
}, $strip>;
```

Evidence explaining why a guardrail policy triggered.

***

<a id="api-guardrailcheckinputschema"></a>

### GuardrailCheckInputSchema

```ts
const GuardrailCheckInputSchema: ZodObject<{
  metadata: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  text: ZodString;
}, $strip>;
```

Runtime content and metadata to evaluate.

***

<a id="api-guardrailcheckrequestschema"></a>

### GuardrailCheckRequestSchema

```ts
const GuardrailCheckRequestSchema: ZodObject<{
  context: ZodObject<{
     org_id: ZodString;
     project_id: ZodString;
     session_id: ZodOptional<ZodNullable<ZodString>>;
     user_id: ZodOptional<ZodNullable<ZodString>>;
     workspace_id: ZodOptional<ZodNullable<ZodString>>;
  }, $strip>;
  execution_id: ZodString;
  input: ZodObject<{
     metadata: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
     text: ZodString;
  }, $strip>;
  policies: ZodDefault<ZodArray<ZodObject<{
     action: ZodString;
     config: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
     id: ZodString;
     stage_filter: ZodDefault<ZodArray<ZodString>>;
     status: ZodDefault<ZodString>;
     type: ZodString;
  }, $strip>>>;
  stage: ZodEnum<{
     llm_input: "llm_input";
     llm_output: "llm_output";
     tool_input: "tool_input";
     tool_output: "tool_output";
  }>;
}, $strip>;
```

Request from OE to Tool Pod to evaluate selected guardrail policies.

***

<a id="api-guardrailcheckresponseschema"></a>

### GuardrailCheckResponseSchema

```ts
const GuardrailCheckResponseSchema: ZodObject<{
  allowed: ZodBoolean;
  decision: ZodEnum<{
     allow: "allow";
     block: "block";
     log_only: "log_only";
     modify: "modify";
     require_review: "require_review";
  }>;
  evidence: ZodDefault<ZodArray<ZodObject<{
     message: ZodDefault<ZodString>;
     metadata: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
     policy_id: ZodString;
  }, $strip>>>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  reason: ZodOptional<ZodNullable<ZodString>>;
  transformed_text: ZodOptional<ZodNullable<ZodString>>;
  triggered_policy_ids: ZodDefault<ZodArray<ZodString>>;
}, $strip>;
```

Decision returned by the Tool Pod guardrails evaluator.

***

<a id="api-guardrailmetaschema"></a>

### GuardrailMetaSchema

```ts
const GuardrailMetaSchema: ZodObject<{
  guardrail_category: ZodString;
  guardrail_id: ZodString;
}, $strip>;
```

Identity of the policy that caused a guardrail block or require_review halt.

***

<a id="api-guardrailruntimepolicyschema"></a>

### GuardrailRuntimePolicySchema

```ts
const GuardrailRuntimePolicySchema: ZodObject<{
  action: ZodString;
  config: ZodDefault<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  id: ZodString;
  stage_filter: ZodDefault<ZodArray<ZodString>>;
  status: ZodDefault<ZodString>;
  type: ZodString;
}, $strip>;
```

OE-selected guardrail policy sent to the Tool Pod for evaluation.

***

<a id="api-guardrailruntimestage-1"></a>

### GuardrailRuntimeStage

```ts
GuardrailRuntimeStage: object;
```

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-llm_input"></a> `LLM_INPUT` | `"llm_input"` | `"llm_input"` |
| <a id="api-property-llm_output"></a> `LLM_OUTPUT` | `"llm_output"` | `"llm_output"` |
| <a id="api-property-tool_input"></a> `TOOL_INPUT` | `"tool_input"` | `"tool_input"` |
| <a id="api-property-tool_output"></a> `TOOL_OUTPUT` | `"tool_output"` | `"tool_output"` |

***

<a id="api-guardrailruntimestageschema"></a>

### GuardrailRuntimeStageSchema

```ts
const GuardrailRuntimeStageSchema: ZodEnum<{
  llm_input: "llm_input";
  llm_output: "llm_output";
  tool_input: "tool_input";
  tool_output: "tool_output";
}>;
```

Runtime stage where OE is asking the Tool Pod to evaluate guardrails.

***

<a id="api-healthresponseschema"></a>

### HealthResponseSchema

```ts
const HealthResponseSchema: ZodObject<{
  component: ZodString;
  details: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  mode: ZodString;
  status: ZodEnum<{
     degraded: "degraded";
     healthy: "healthy";
     unhealthy: "unhealthy";
  }>;
  version: ZodDefault<ZodString>;
}, $strip>;
```

Health check response.

***

<a id="api-healthstatus-1"></a>

### HealthStatus

```ts
HealthStatus: object;
```

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-degraded"></a> `DEGRADED` | `"degraded"` | `"degraded"` |
| <a id="api-property-healthy"></a> `HEALTHY` | `"healthy"` | `"healthy"` |
| <a id="api-property-unhealthy"></a> `UNHEALTHY` | `"unhealthy"` | `"unhealthy"` |

***

<a id="api-healthstatusschema"></a>

### HealthStatusSchema

```ts
const HealthStatusSchema: ZodEnum<{
  degraded: "degraded";
  healthy: "healthy";
  unhealthy: "unhealthy";
}>;
```

Health status of a component.

***

<a id="api-humanreviewdataschema"></a>

### HumanReviewDataSchema

```ts
const HumanReviewDataSchema: ZodObject<{
  decision: ZodString;
  reviewer_notes: ZodOptional<ZodString>;
}, $strip>;
```

Data provided by human reviewer when resuming a suspended execution.

***

<a id="api-interruptresultschema"></a>

### InterruptResultSchema

```ts
const InterruptResultSchema: ZodObject<{
  interrupts: ZodOptional<ZodArray<ZodObject<{
     id: ZodString;
     value: ZodUnknown;
  }, $strip>>>;
  messages: ZodDefault<ZodArray<ZodPipe<ZodObject<{
     additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     content: ZodUnion<readonly [ZodString, ZodArray<ZodDiscriminatedUnion<..., ...>>]>;
     id: ZodOptional<ZodString>;
     is_error: ZodOptional<ZodBoolean>;
     name: ZodOptional<ZodString>;
     response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     role: ZodEnum<{
        assistant: "assistant";
        system: "system";
        tool: "tool";
        user: "user";
     }>;
     tool_call_id: ZodOptional<ZodString>;
     tool_calls: ZodOptional<ZodArray<ZodPipe<ZodObject<..., ...>, ZodTransform<..., ...>>>>;
   }, $loose>, ZodTransform<Message, {
   [x: string]: unknown;
     additional_kwargs?: Record<string, JsonValue>;
     content:   | string
        | (
        | {
        text: ...;
        type: ...;
      }
        | {
        mime_type?: ...;
        type: ...;
        url: ...;
      }
        | {
        filename?: ...;
        mime_type?: ...;
        type: ...;
        url: ...;
      })[];
     id?: string;
     is_error?: boolean;
     name?: string;
     response_metadata?: Record<string, JsonValue>;
     role: "user" | "assistant" | "tool" | "system";
     tool_call_id?: string;
     tool_calls?: LLMToolCall[];
  }>>>>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  resume_schema: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  suspend_payload: ZodUnknown;
}, $strip>;
```

***

<a id="api-invokellmrequestargumentsschema"></a>

### InvokeLLMRequestArgumentsSchema

```ts
const InvokeLLMRequestArgumentsSchema: ZodPreprocess<ZodObject<{
  llm_id: ZodDefault<ZodString>;
  messages: ZodArray<ZodPipe<ZodObject<{
     additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     content: ZodUnion<readonly [ZodString, ZodArray<ZodDiscriminatedUnion<..., ...>>]>;
     id: ZodOptional<ZodString>;
     is_error: ZodOptional<ZodBoolean>;
     name: ZodOptional<ZodString>;
     response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     role: ZodEnum<{
        assistant: "assistant";
        system: "system";
        tool: "tool";
        user: "user";
     }>;
     tool_call_id: ZodOptional<ZodString>;
     tool_calls: ZodOptional<ZodArray<ZodPipe<ZodObject<..., ...>, ZodTransform<..., ...>>>>;
   }, $loose>, ZodTransform<Message, {
   [x: string]: unknown;
     additional_kwargs?: Record<string, JsonValue>;
     content:   | string
        | (
        | {
        text: ...;
        type: ...;
      }
        | {
        mime_type?: ...;
        type: ...;
        url: ...;
      }
        | {
        filename?: ...;
        mime_type?: ...;
        type: ...;
        url: ...;
      })[];
     id?: string;
     is_error?: boolean;
     name?: string;
     response_metadata?: Record<string, JsonValue>;
     role: "user" | "assistant" | "tool" | "system";
     tool_call_id?: string;
     tool_calls?: LLMToolCall[];
  }>>>;
  model: ZodString;
  options: ZodOptional<ZodPreprocess<ZodPipe<ZodObject<{
     frequency_penalty: ZodOptional<ZodNumber>;
     max_tokens: ZodOptional<ZodNumber>;
     parallel_tool_calls: ZodOptional<ZodBoolean>;
     presence_penalty: ZodOptional<ZodNumber>;
     reasoning_effort: ZodOptional<ZodString>;
     response_format: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     seed: ZodOptional<ZodNumber>;
     timeout: ZodOptional<ZodNumber>;
     top_k: ZodOptional<ZodNumber>;
     top_p: ZodOptional<ZodNumber>;
   }, $loose>, ZodTransform<LLMInvocationOptions, {
   [x: string]: unknown;
     frequency_penalty?: number;
     max_tokens?: number;
     parallel_tool_calls?: boolean;
     presence_penalty?: number;
     reasoning_effort?: string;
     response_format?: JsonValue;
     seed?: number;
     timeout?: number;
     top_k?: number;
     top_p?: number;
  }>>>>;
  stop_sequences: ZodOptional<ZodArray<ZodString>>;
  stream: ZodDefault<ZodBoolean>;
  tool_choice: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  tools: ZodOptional<ZodArray<ZodPipe<ZodObject<{
     description: ZodOptional<ZodString>;
     function: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     name: ZodOptional<ZodString>;
     parameters: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     strict: ZodOptional<ZodBoolean>;
     type: ZodOptional<ZodString>;
   }, $strip>, ZodTransform<LLMToolSchema, {
     description?: string;
     function?: JsonValue;
     name?: string;
     parameters?: JsonValue;
     strict?: boolean;
     type?: string;
  }>>>>;
}, $strip>>;
```

Typed invoke_llm arguments forwarded through OE and tool pods.

Python uses Pydantic aliases:
  - validation alias `stop` ↔ `stop_sequences`, serialization alias `stop`
  - pre-validator renames `kwargs` → `options`

In Zod we replicate this with a `z.preprocess` that normalizes incoming
payloads into the canonical schema (using `stop_sequences` and
`options`). For outgoing serialization that needs the wire alias `stop`,
use `serializeInvokeLLMRequestArguments()`.

***

<a id="api-invokerequestschema"></a>

### InvokeRequestSchema

```ts
const InvokeRequestSchema: ZodObject<{
  aer_url: ZodOptional<ZodString>;
  custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
  message: ZodString;
  org_id: ZodOptional<ZodString>;
  project_id: ZodOptional<ZodString>;
  session_id: ZodOptional<ZodString>;
  tool_url: ZodOptional<ZodString>;
  user_id: ZodOptional<ZodString>;
  wait: ZodDefault<ZodBoolean>;
  workspace_id: ZodOptional<ZodString>;
}, $strip>;
```

Request to invoke the agent (compatible with agent-runtime).

***

<a id="api-invokeresponseschema"></a>

### InvokeResponseSchema

```ts
const InvokeResponseSchema: ZodObject<{
  error: ZodOptional<ZodString>;
  execution_id: ZodOptional<ZodString>;
  result: ZodOptional<ZodUnknown>;
  session_id: ZodOptional<ZodString>;
  status: ZodOptional<ZodString>;
  user_id: ZodOptional<ZodString>;
}, $strip>;
```

Response from invoking the agent (compatible with agent-runtime).

***

<a id="api-llm_backoff_multiplier"></a>

### LLM\_BACKOFF\_MULTIPLIER

```ts
const LLM_BACKOFF_MULTIPLIER: number;
```

***

<a id="api-llm_initial_backoff"></a>

### LLM\_INITIAL\_BACKOFF

```ts
const LLM_INITIAL_BACKOFF: number;
```

***

<a id="api-llm_max_backoff"></a>

### LLM\_MAX\_BACKOFF

```ts
const LLM_MAX_BACKOFF: number;
```

***

<a id="api-llm_max_retries"></a>

### LLM\_MAX\_RETRIES

```ts
const LLM_MAX_RETRIES: number;
```

***

<a id="api-llm_read_timeout"></a>

### LLM\_READ\_TIMEOUT

```ts
const LLM_READ_TIMEOUT: number;
```

***

<a id="api-llmpodinvokerequestschema"></a>

### LLMPodInvokeRequestSchema

```ts
const LLMPodInvokeRequestSchema: ZodPreprocess<ZodObject<{
  arguments: ZodPreprocess<ZodObject<{
     llm_id: ZodDefault<ZodString>;
     messages: ZodArray<ZodPipe<ZodObject<{
        additional_kwargs: ZodOptional<...>;
        content: ZodUnion<...>;
        id: ZodOptional<...>;
        is_error: ZodOptional<...>;
        name: ZodOptional<...>;
        response_metadata: ZodOptional<...>;
        role: ZodEnum<...>;
        tool_call_id: ZodOptional<...>;
        tool_calls: ZodOptional<...>;
      }, $loose>, ZodTransform<Message, {
      [x: string]: unknown;
        additional_kwargs?: ... | ...;
        content: ... | ...;
        id?: ... | ...;
        is_error?: ... | ... | ...;
        name?: ... | ...;
        response_metadata?: ... | ...;
        role: ... | ... | ... | ...;
        tool_call_id?: ... | ...;
        tool_calls?: ... | ...;
     }>>>;
     model: ZodString;
     options: ZodOptional<ZodPreprocess<ZodPipe<ZodObject<{
        frequency_penalty: ...;
        max_tokens: ...;
        parallel_tool_calls: ...;
        presence_penalty: ...;
        reasoning_effort: ...;
        response_format: ...;
        seed: ...;
        timeout: ...;
        top_k: ...;
        top_p: ...;
      }, $loose>, ZodTransform<LLMInvocationOptions, {
      [x: ...]: ...;
        frequency_penalty?: ...;
        max_tokens?: ...;
        parallel_tool_calls?: ...;
        presence_penalty?: ...;
        reasoning_effort?: ...;
        response_format?: ...;
        seed?: ...;
        timeout?: ...;
        top_k?: ...;
        top_p?: ...;
     }>>>>;
     stop_sequences: ZodOptional<ZodArray<ZodString>>;
     stream: ZodDefault<ZodBoolean>;
     tool_choice: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
     tools: ZodOptional<ZodArray<ZodPipe<ZodObject<{
        description: ...;
        function: ...;
        name: ...;
        parameters: ...;
        strict: ...;
        type: ...;
      }, $strip>, ZodTransform<LLMToolSchema, {
        description?: ...;
        function?: ...;
        name?: ...;
        parameters?: ...;
        strict?: ...;
        type?: ...;
     }>>>>;
  }, $strip>>;
  execution_id: ZodString;
  platform_trace_id: ZodOptional<ZodNullable<ZodString>>;
  step_number: ZodOptional<ZodNumber>;
}, $strip>>;
```

Request to invoke LLM on a tool executor pod.

Python pre-validator `_normalize_flat_payload`: if `arguments` is
missing but `execution_id` is present, treat the remaining keys as
the arguments payload. Same logic replicated via `z.preprocess`.

***

<a id="api-llmpodinvokeresponseschema"></a>

### LLMPodInvokeResponseSchema

```ts
const LLMPodInvokeResponseSchema: ZodObject<{
  duration_ms: ZodNumber;
  error: ZodOptional<ZodString>;
  error_code: ZodOptional<ZodString>;
  pod_name: ZodString;
  result: ZodOptional<ZodObject<{
     additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     content: ZodString;
     id: ZodOptional<ZodString>;
     metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     name: ZodOptional<ZodString>;
     response_metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     tool_calls: ZodOptional<ZodArray<ZodUnknown>>;
     usage: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  }, $loose>>;
  status: ZodString;
  usage: ZodOptional<ZodPreprocess<ZodPipe<ZodObject<{
     completion_tokens: ZodOptional<ZodNumber>;
     input_tokens: ZodOptional<ZodNumber>;
     model: ZodOptional<ZodString>;
     output_tokens: ZodOptional<ZodNumber>;
     prompt_tokens: ZodOptional<ZodNumber>;
     total_tokens: ZodOptional<ZodNumber>;
   }, $strip>, ZodTransform<LLMTokenUsage, {
     completion_tokens?: number;
     input_tokens?: number;
     model?: string;
     output_tokens?: number;
     prompt_tokens?: number;
     total_tokens?: number;
  }>>>>;
}, $strip>;
```

Response from LLM invocation on a tool executor pod.

Python `_sync_usage_with_result` is a model_validator that mutates
`result.usage` ↔ `usage`. TS `LLMResponse.usage` is readonly, so
mutation isn't possible; callers should use
`normalizeLLMPodInvokeResponse()` after parsing to sync the two fields
by constructing a fresh LLMResponse when needed.

***

<a id="api-llmpodstreameventschema"></a>

### LLMPodStreamEventSchema

```ts
const LLMPodStreamEventSchema: ZodObject<{
  additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  content: ZodOptional<ZodString>;
  done: ZodOptional<ZodBoolean>;
  duration_ms: ZodOptional<ZodNumber>;
  error: ZodOptional<ZodString>;
  error_code: ZodOptional<ZodString>;
  id: ZodOptional<ZodString>;
  interrupted: ZodOptional<ZodBoolean>;
  name: ZodOptional<ZodString>;
  pod_name: ZodOptional<ZodString>;
  response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  retry_after_ms: ZodOptional<ZodNumber>;
  retryable: ZodOptional<ZodBoolean>;
  tool_call_chunks: ZodOptional<ZodArray<ZodObject<{
     args: ZodOptional<ZodString>;
     id: ZodOptional<ZodString>;
     index: ZodOptional<ZodNumber>;
     name: ZodOptional<ZodString>;
     type: ZodOptional<ZodString>;
  }, $strip>>>;
  tool_calls: ZodOptional<ZodArray<ZodRecord<ZodString, ZodUnknown>>>;
  usage: ZodOptional<ZodPreprocess<ZodPipe<ZodObject<{
     completion_tokens: ZodOptional<ZodNumber>;
     input_tokens: ZodOptional<ZodNumber>;
     model: ZodOptional<ZodString>;
     output_tokens: ZodOptional<ZodNumber>;
     prompt_tokens: ZodOptional<ZodNumber>;
     total_tokens: ZodOptional<ZodNumber>;
   }, $strip>, ZodTransform<LLMTokenUsage, {
     completion_tokens?: number;
     input_tokens?: number;
     model?: string;
     output_tokens?: number;
     prompt_tokens?: number;
     total_tokens?: number;
  }>>>>;
}, $strip>;
```

SSE event emitted by a tool pod during invoke_llm streaming.

***

<a id="api-llmresultschema"></a>

### LLMResultSchema

```ts
const LLMResultSchema: ZodObject<{
  additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  content: ZodDefault<ZodString>;
  id: ZodOptional<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  name: ZodOptional<ZodString>;
  response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  tool_calls: ZodDefault<ZodArray<ZodUnknown>>;
  usage: ZodOptional<ZodPreprocess<ZodPipe<ZodObject<{
     completion_tokens: ZodOptional<ZodNumber>;
     input_tokens: ZodOptional<ZodNumber>;
     model: ZodOptional<ZodString>;
     output_tokens: ZodOptional<ZodNumber>;
     prompt_tokens: ZodOptional<ZodNumber>;
     total_tokens: ZodOptional<ZodNumber>;
   }, $strip>, ZodTransform<LLMTokenUsage, {
     completion_tokens?: number;
     input_tokens?: number;
     model?: string;
     output_tokens?: number;
     prompt_tokens?: number;
     total_tokens?: number;
  }>>>>;
}, $loose>;
```

Runtime validation schema for LLMResult wire data. `content` is the only required field.

***

<a id="api-max_tool_argument_bytes"></a>

### MAX\_TOOL\_ARGUMENT\_BYTES

```ts
const MAX_TOOL_ARGUMENT_BYTES: number;
```

***

<a id="api-memorywriteschema"></a>

### MemoryWriteSchema

```ts
const MemoryWriteSchema: GenMessage<MemoryWrite>;
```

Describes the message mongodb.agentic.workflow.v1.MemoryWrite.
Use `create(MemoryWriteSchema)` to create a new message.

***

<a id="api-messageroleschema"></a>

### MessageRoleSchema

```ts
const MessageRoleSchema: GenEnum<MessageRole>;
```

Describes the enum mongodb.agentic.workflow.v1.MessageRole.

***

<a id="api-model_request_prepare"></a>

### MODEL\_REQUEST\_PREPARE

```ts
const MODEL_REQUEST_PREPARE: "request.prepare" = "request.prepare";
```

***

<a id="api-model_response_process"></a>

### MODEL\_RESPONSE\_PROCESS

```ts
const MODEL_RESPONSE_PROCESS: "response.process" = "response.process";
```

***

<a id="api-nodeexecutionrequestschema"></a>

### NodeExecutionRequestSchema

```ts
const NodeExecutionRequestSchema: ZodObject<{
  duration_ms: ZodOptional<ZodNumber>;
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  inputs: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  node_name: ZodString;
  org_id: ZodOptional<ZodString>;
  outputs: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  parent_run_id: ZodOptional<ZodString>;
  project_id: ZodOptional<ZodString>;
  run_id: ZodString;
  session_id: ZodOptional<ZodString>;
  span_id: ZodOptional<ZodNullable<ZodString>>;
  status: ZodString;
  timestamp: ZodCoercedDate<unknown>;
  trace_id: ZodOptional<ZodNullable<ZodString>>;
  user_id: ZodOptional<ZodString>;
}, $strip>;
```

Report node execution event (AER → OE for logging).

***

<a id="api-nodeexecutionsqueryresponseschema"></a>

### NodeExecutionsQueryResponseSchema

```ts
const NodeExecutionsQueryResponseSchema: ZodObject<{
  count: ZodDefault<ZodNumber>;
  executions: ZodDefault<ZodArray<ZodRecord<ZodString, ZodUnknown>>>;
}, $strip>;
```

Response for node executions query (used by API Gateway proxy).

***

<a id="api-oe_dispatch_retry_max_wait_ms"></a>

### OE\_DISPATCH\_RETRY\_MAX\_WAIT\_MS

```ts
const OE_DISPATCH_RETRY_MAX_WAIT_MS: 60000 = 60_000;
```

***

<a id="api-oe_dispatch_takeover_retry_delay_ms"></a>

### OE\_DISPATCH\_TAKEOVER\_RETRY\_DELAY\_MS

```ts
const OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS: 30000 = 30_000;
```

Wait after a dropped SSE connection so the next attempt can land after
dispatch_heartbeat TTL (30s). Do not reuse HTTP Retry-After's 10s cap.
Lockstep with Python.

***

<a id="api-oe_retryable_max_attempts"></a>

### OE\_RETRYABLE\_MAX\_ATTEMPTS

```ts
const OE_RETRYABLE_MAX_ATTEMPTS: 3 = 3;
```

Same-step retries of /tool/execute (and the SSE relay) after OE advertises retryable=true, and after an SSE transport disconnect. 1 initial + 2 extras; lockstep with Python.

***

<a id="api-oestreamretry"></a>

### oeStreamRetry

```ts
const oeStreamRetry: object;
```

Sleep before an SSE same-URL retry. Tests spy this to avoid wall-clock waits.

#### Type Declaration

| Name | Type |
| :------ | :------ |
| `sleep()` | (`delayMs`) => `Promise`\<`void`\> |

***

<a id="api-openinference_span_kind"></a>

### OPENINFERENCE\_SPAN\_KIND

```ts
const OPENINFERENCE_SPAN_KIND: "openinference.span.kind" = openinferenceSpanKindAttr;
```

***

<a id="api-openinferencespankind"></a>

### OpenInferenceSpanKind

```ts
const OpenInferenceSpanKind: object;
```

OpenInference span kinds, matching the Python SDK's `OpenInferenceSpanKind`.

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-agent"></a> `AGENT` | `"AGENT"` | `"AGENT"` |
| <a id="api-property-chain"></a> `CHAIN` | `"CHAIN"` | `"CHAIN"` |
| <a id="api-property-tool"></a> `TOOL` | `"TOOL"` | `"TOOL"` |

***

<a id="api-operationpathschema"></a>

### OperationPathSchema

```ts
const OperationPathSchema: GenMessage<OperationPath>;
```

Describes the message mongodb.agentic.workflow.v1.OperationPath.
Use `create(OperationPathSchema)` to create a new message.

***

<a id="api-operationpathsegmentschema"></a>

### OperationPathSegmentSchema

```ts
const OperationPathSegmentSchema: GenMessage<OperationPathSegment>;
```

Describes the message mongodb.agentic.workflow.v1.OperationPathSegment.
Use `create(OperationPathSegmentSchema)` to create a new message.

***

<a id="api-pendinginterruptschema"></a>

### PendingInterruptSchema

```ts
const PendingInterruptSchema: ZodObject<{
  id: ZodString;
  value: ZodUnknown;
}, $strip>;
```

Result from agent execution when the agent is suspended.

Contains the suspend payload directly (framework-agnostic) rather than
wrapping framework-specific Interrupt objects. Any framework-specific
state needed to resume (LangGraph checkpoint id, ADK function-call
correlation, etc.) is carried opaquely in `metadata` — the AER never
inspects it; the framework adapter writes it on suspend and reads it back
from `RequestContext.metadata` on resume.

***

<a id="api-runtimemcpauthconfigschema"></a>

### RuntimeMCPAuthConfigSchema

```ts
const RuntimeMCPAuthConfigSchema: ZodObject<{
  client_id_env: ZodDefault<ZodNullable<ZodString>>;
  client_name: ZodDefault<ZodNullable<ZodString>>;
  client_secret_env: ZodDefault<ZodNullable<ZodString>>;
  redirect_uri: ZodDefault<ZodNullable<ZodString>>;
  scope: ZodDefault<ZodNullable<ZodString>>;
  token_env: ZodDefault<ZodNullable<ZodString>>;
  token_url: ZodDefault<ZodNullable<ZodString>>;
  type: ZodDefault<ZodEnum<{
     bearer_env: "bearer_env";
     client_credentials: "client_credentials";
     none: "none";
     oauth: "oauth";
  }>>;
}, $strip>;
```

***

<a id="api-runtimemcpconfigschema"></a>

### RuntimeMCPConfigSchema

```ts
const RuntimeMCPConfigSchema: ZodObject<{
  servers: ZodDefault<ZodRecord<ZodString, ZodObject<{
     allowed_tools: ZodDefault<ZodNullable<ZodArray<ZodString>>>;
     auth: ZodDefault<ZodObject<{
        client_id_env: ZodDefault<ZodNullable<...>>;
        client_name: ZodDefault<ZodNullable<...>>;
        client_secret_env: ZodDefault<ZodNullable<...>>;
        redirect_uri: ZodDefault<ZodNullable<...>>;
        scope: ZodDefault<ZodNullable<...>>;
        token_env: ZodDefault<ZodNullable<...>>;
        token_url: ZodDefault<ZodNullable<...>>;
        type: ZodDefault<ZodEnum<...>>;
     }, $strip>>;
     headers: ZodDefault<ZodRecord<ZodString, ZodString>>;
     timeout_seconds: ZodDefault<ZodNumber>;
     transport: ZodDefault<ZodLiteral<"streamable_http">>;
     url: ZodString;
  }, $strip>>>;
}, $strip>;
```

***

<a id="api-runtimemcpserverconfigschema"></a>

### RuntimeMCPServerConfigSchema

```ts
const RuntimeMCPServerConfigSchema: ZodObject<{
  allowed_tools: ZodDefault<ZodNullable<ZodArray<ZodString>>>;
  auth: ZodDefault<ZodObject<{
     client_id_env: ZodDefault<ZodNullable<ZodString>>;
     client_name: ZodDefault<ZodNullable<ZodString>>;
     client_secret_env: ZodDefault<ZodNullable<ZodString>>;
     redirect_uri: ZodDefault<ZodNullable<ZodString>>;
     scope: ZodDefault<ZodNullable<ZodString>>;
     token_env: ZodDefault<ZodNullable<ZodString>>;
     token_url: ZodDefault<ZodNullable<ZodString>>;
     type: ZodDefault<ZodEnum<{
        bearer_env: "bearer_env";
        client_credentials: "client_credentials";
        none: "none";
        oauth: "oauth";
     }>>;
  }, $strip>>;
  headers: ZodDefault<ZodRecord<ZodString, ZodString>>;
  timeout_seconds: ZodDefault<ZodNumber>;
  transport: ZodDefault<ZodLiteral<"streamable_http">>;
  url: ZodString;
}, $strip>;
```

***

<a id="api-secretsconfigschema"></a>

### SecretsConfigSchema

```ts
const SecretsConfigSchema: ZodObject<{
  aer: ZodDefault<ZodArray<ZodString>>;
  disable_restriction: ZodDefault<ZodBoolean>;
  tools: ZodDefault<ZodRecord<ZodString, ZodArray<ZodString>>>;
}, $strip>;
```

***

<a id="api-sessioninfoschema"></a>

### SessionInfoSchema

```ts
const SessionInfoSchema: ZodObject<{
  created_at: ZodString;
  last_activity: ZodString;
  last_message_preview: ZodDefault<ZodString>;
  message_count: ZodDefault<ZodNumber>;
  project_id: ZodDefault<ZodString>;
  session_id: ZodString;
  user_id: ZodDefault<ZodString>;
  visibility: ZodDefault<ZodString>;
  workspace_id: ZodDefault<ZodString>;
}, $strip>;
```

A single session entry returned by /query/sessions.

***

<a id="api-sessionmessageschema"></a>

### SessionMessageSchema

```ts
const SessionMessageSchema: ZodObject<{
  content: ZodString;
  id: ZodString;
  name: ZodOptional<ZodString>;
  role: ZodString;
  session_id: ZodString;
  timestamp: ZodString;
  tool_call_id: ZodOptional<ZodNullable<ZodString>>;
  tool_calls: ZodOptional<ZodNullable<ZodArray<ZodPipe<ZodObject<{
     args: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     arguments: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     id: ZodOptional<ZodString>;
     index: ZodOptional<ZodNumber>;
     name: ZodOptional<ZodString>;
     type: ZodOptional<ZodString>;
   }, $strip>, ZodTransform<LLMToolCall, {
     args?: JsonValue;
     arguments?: JsonValue;
     id?: string;
     index?: number;
     name?: string;
     type?: string;
  }>>>>>;
}, $strip>;
```

A single message within a session.

***

<a id="api-sessionmessagesqueryresponseschema"></a>

### SessionMessagesQueryResponseSchema

```ts
const SessionMessagesQueryResponseSchema: ZodObject<{
  messages: ZodDefault<ZodArray<ZodObject<{
     content: ZodString;
     id: ZodString;
     name: ZodOptional<ZodString>;
     role: ZodString;
     session_id: ZodString;
     timestamp: ZodString;
     tool_call_id: ZodOptional<ZodNullable<ZodString>>;
     tool_calls: ZodOptional<ZodNullable<ZodArray<ZodPipe<ZodObject<..., ...>, ZodTransform<..., ...>>>>>;
  }, $strip>>>;
}, $strip>;
```

Response for session messages query (used by API Gateway proxy).

***

<a id="api-sessionsqueryresponseschema"></a>

### SessionsQueryResponseSchema

```ts
const SessionsQueryResponseSchema: ZodObject<{
  limit: ZodDefault<ZodNumber>;
  offset: ZodDefault<ZodNumber>;
  sessions: ZodDefault<ZodArray<ZodObject<{
     created_at: ZodString;
     last_activity: ZodString;
     last_message_preview: ZodDefault<ZodString>;
     message_count: ZodDefault<ZodNumber>;
     project_id: ZodDefault<ZodString>;
     session_id: ZodString;
     user_id: ZodDefault<ZodString>;
     visibility: ZodDefault<ZodString>;
     workspace_id: ZodDefault<ZodString>;
  }, $strip>>>;
  total_count: ZodDefault<ZodNumber>;
}, $strip>;
```

Response for sessions list query (used by API Gateway proxy).

***

<a id="api-skills_middleware_before_agent"></a>

### SKILLS\_MIDDLEWARE\_BEFORE\_AGENT

```ts
const SKILLS_MIDDLEWARE_BEFORE_AGENT: "skills.middleware" = "skills.middleware";
```

***

<a id="api-statesnapshotschema"></a>

### StateSnapshotSchema

```ts
const StateSnapshotSchema: GenMessage<StateSnapshot>;
```

Describes the message mongodb.agentic.workflow.v1.StateSnapshot.
Use `create(StateSnapshotSchema)` to create a new message.

***

<a id="api-stepactivityentryschema"></a>

### StepActivityEntrySchema

```ts
const StepActivityEntrySchema: GenMessage<StepActivityEntry>;
```

Describes the message mongodb.agentic.workflow.v1.StepActivityEntry.
Use `create(StepActivityEntrySchema)` to create a new message.

***

<a id="api-stepsuspensionentryschema"></a>

### StepSuspensionEntrySchema

```ts
const StepSuspensionEntrySchema: GenMessage<StepSuspensionEntry>;
```

Describes the message mongodb.agentic.workflow.v1.StepSuspensionEntry.
Use `create(StepSuspensionEntrySchema)` to create a new message.

***

<a id="api-streamchunkschema"></a>

### StreamChunkSchema

```ts
const StreamChunkSchema: ZodObject<{
  chunk_type: ZodString;
  code: ZodOptional<ZodString>;
  content: ZodDefault<ZodString>;
  error: ZodOptional<ZodString>;
  execution_id: ZodOptional<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodString>>;
  step_number: ZodOptional<ZodNumber>;
  tool_call_id: ZodOptional<ZodString>;
  tool_name: ZodOptional<ZodString>;
}, $strip>;
```

A chunk of streaming response from the agent.

Used for real-time streaming of agent responses via SSE or gRPC.

***

<a id="api-streamingresultschema"></a>

### StreamingResultSchema

```ts
const StreamingResultSchema: ZodObject<{
  content: ZodDefault<ZodString>;
  messages: ZodDefault<ZodArray<ZodPipe<ZodObject<{
     additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     content: ZodUnion<readonly [ZodString, ZodArray<ZodDiscriminatedUnion<..., ...>>]>;
     id: ZodOptional<ZodString>;
     is_error: ZodOptional<ZodBoolean>;
     name: ZodOptional<ZodString>;
     response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>>;
     role: ZodEnum<{
        assistant: "assistant";
        system: "system";
        tool: "tool";
        user: "user";
     }>;
     tool_call_id: ZodOptional<ZodString>;
     tool_calls: ZodOptional<ZodArray<ZodPipe<ZodObject<..., ...>, ZodTransform<..., ...>>>>;
   }, $loose>, ZodTransform<Message, {
   [x: string]: unknown;
     additional_kwargs?: Record<string, JsonValue>;
     content:   | string
        | (
        | {
        text: ...;
        type: ...;
      }
        | {
        mime_type?: ...;
        type: ...;
        url: ...;
      }
        | {
        filename?: ...;
        mime_type?: ...;
        type: ...;
        url: ...;
      })[];
     id?: string;
     is_error?: boolean;
     name?: string;
     response_metadata?: Record<string, JsonValue>;
     role: "user" | "assistant" | "tool" | "system";
     tool_call_id?: string;
     tool_calls?: LLMToolCall[];
  }>>>>;
  metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
}, $strip>;
```

Result from agent execution for normal completion.

Returned by _execute_via_agent_stream when the agent finishes without
suspend. The caller uses content for the final response and messages
for memory writing.

***

<a id="api-suspendpayloadschema"></a>

### SuspendPayloadSchema

```ts
const SuspendPayloadSchema: ZodObject<{
  suspend_context: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  suspend_reason: ZodString;
}, $strip>;
```

Payload returned by an agent tool to trigger human-in-the-loop suspension.

Agent tools signal a suspend by returning a JSON string containing these
fields. The `__suspend__` flag is stripped before the payload is passed
to the framework's interrupt() handler.

Example usage in an agent tool:

    import { suspendPayloadToJson } from '@mongodb-js/agent-engine-runner-shared'
    return suspendPayloadToJson({
      suspend_reason: 'awaiting_human_review',
      suspend_context: { claim_id: 'C-123', task_id: 'T-456' },
    })

***

<a id="api-tenantscopeschema"></a>

### TenantScopeSchema

```ts
const TenantScopeSchema: GenMessage<TenantScope>;
```

Describes the message mongodb.agentic.workflow.v1.TenantScope.
Use `create(TenantScopeSchema)` to create a new message.

***

<a id="api-toolapierrorschema"></a>

### ToolAPIErrorSchema

```ts
const ToolAPIErrorSchema: ZodObject<{
  classification: ZodString;
  error_code: ZodOptional<ZodNullable<ZodString>>;
  http_status: ZodOptional<ZodNullable<ZodNumber>>;
  provider_type: ZodOptional<ZodNullable<ZodString>>;
  reason: ZodOptional<ZodNullable<ZodString>>;
  retryable: ZodBoolean;
}, $strip>;
```

***

<a id="api-toolauthorizationschema"></a>

### ToolAuthorizationSchema

```ts
const ToolAuthorizationSchema: ZodObject<{
  expires_at: ZodOptional<ZodNullable<ZodNumber>>;
  token: ZodString;
}, $strip>;
```

Delegated credential injected by OE for tool execution.

***

<a id="api-tooldefinitionschema"></a>

### ToolDefinitionSchema

```ts
const ToolDefinitionSchema: ZodObject<{
  description: ZodDefault<ZodString>;
  is_local: ZodDefault<ZodBoolean>;
  name: ZodString;
  parameters: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  provider_type: ZodOptional<ZodNullable<ZodString>>;
  scopes: ZodDefault<ZodArray<ZodString>>;
}, $strip>;
```

Definition of a registered tool.

***

<a id="api-toolexecuterequestschema"></a>

### ToolExecuteRequestSchema

```ts
const ToolExecuteRequestSchema: ZodObject<{
  arguments: ZodRecord<ZodString, ZodUnknown>;
  custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
  execution_id: ZodString;
  is_local: ZodDefault<ZodBoolean>;
  kind: ZodOptional<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  provider_type: ZodOptional<ZodNullable<ZodString>>;
  redact_fields: ZodDefault<ZodArray<ZodString>>;
  scopes: ZodDefault<ZodArray<ZodString>>;
  span_id: ZodOptional<ZodNullable<ZodString>>;
  step_number: ZodNumber;
  tool_call_id: ZodOptional<ZodString>;
  tool_name: ZodString;
  trace_id: ZodOptional<ZodNullable<ZodString>>;
}, $strip>;
```

Request to execute a tool (AER → OE for approval).

***

<a id="api-toolexecuteresponseschema"></a>

### ToolExecuteResponseSchema

```ts
const ToolExecuteResponseSchema: ZodPreprocess<ZodObject<{
  cached_result: ZodOptional<ZodNullable<ZodUnknown>>;
  duration_ms: ZodOptional<ZodNullable<ZodNumber>>;
  elicitation: ZodOptional<ZodNullable<ZodObject<{
     authorization_url: ZodString;
     created: ZodDefault<ZodBoolean>;
     elicitation_id: ZodString;
     message: ZodDefault<ZodString>;
  }, $strip>>>;
  error: ZodOptional<ZodNullable<ZodString>>;
  error_code: ZodOptional<ZodNullable<ZodString>>;
  from_cache: ZodDefault<ZodBoolean>;
  guardrail_meta: ZodOptional<ZodNullable<ZodObject<{
     guardrail_category: ZodString;
     guardrail_id: ZodString;
  }, $strip>>>;
  latest_step_number: ZodOptional<ZodNullable<ZodNumber>>;
  pod_name: ZodOptional<ZodNullable<ZodString>>;
  proceed: ZodBoolean;
  reason: ZodOptional<ZodNullable<ZodString>>;
  result: ZodOptional<ZodNullable<ZodUnknown>>;
  retryable: ZodDefault<ZodBoolean>;
  route_to: ZodOptional<ZodNullable<ZodString>>;
  status: ZodOptional<ZodNullable<ZodString>>;
  tool_api_error: ZodOptional<ZodNullable<ZodObject<{
     classification: ZodString;
     error_code: ZodOptional<ZodNullable<ZodString>>;
     http_status: ZodOptional<ZodNullable<ZodNumber>>;
     provider_type: ZodOptional<ZodNullable<ZodString>>;
     reason: ZodOptional<ZodNullable<ZodString>>;
     retryable: ZodBoolean;
  }, $strip>>>;
}, $strip>>;
```

Assemble `guardrail_meta` from the flat `guardrail_id`/`guardrail_category`
fields the OE wire format sends, so callers work with a single structured
object. Mirrors Python's `_assemble_guardrail_meta` model validator.

***

<a id="api-toolfunctionrequestschema"></a>

### ToolFunctionRequestSchema

```ts
const ToolFunctionRequestSchema: ZodObject<{
  request: ZodObject<{
     arguments: ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
     authorization: ZodOptional<ZodObject<{
        expires_at: ZodOptional<ZodNullable<ZodNumber>>;
        token: ZodString;
     }, $strip>>;
     custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
     execution_id: ZodString;
     metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
     oe_owner_url: ZodOptional<ZodNullable<ZodString>>;
     oe_url: ZodOptional<ZodString>;
     payload: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     platform_trace_id: ZodOptional<ZodNullable<ZodString>>;
     session_id: ZodString;
     step_number: ZodOptional<ZodNumber>;
     tool_call_id: ZodOptional<ZodString>;
     tool_name: ZodString;
     user_id: ZodOptional<ZodString>;
  }, $strip>;
  step: ZodNumber;
}, $strip>;
```

The function-mode invocation envelope delivered at /run/meta/request.

***

<a id="api-toolpodexecuterequestschema"></a>

### ToolPodExecuteRequestSchema

```ts
const ToolPodExecuteRequestSchema: ZodObject<{
  arguments: ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  authorization: ZodOptional<ZodObject<{
     expires_at: ZodOptional<ZodNullable<ZodNumber>>;
     token: ZodString;
  }, $strip>>;
  custom_headers: ZodOptional<ZodRecord<ZodString, ZodString>>;
  execution_id: ZodString;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  oe_owner_url: ZodOptional<ZodNullable<ZodString>>;
  oe_url: ZodOptional<ZodString>;
  payload: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  platform_trace_id: ZodOptional<ZodNullable<ZodString>>;
  session_id: ZodString;
  step_number: ZodOptional<ZodNumber>;
  tool_call_id: ZodOptional<ZodString>;
  tool_name: ZodString;
  user_id: ZodOptional<ZodString>;
}, $strip>;
```

Request to execute a tool in a Tool Pod.

***

<a id="api-toolpodexecuteresponseschema"></a>

### ToolPodExecuteResponseSchema

```ts
const ToolPodExecuteResponseSchema: ZodObject<{
  error: ZodOptional<ZodString>;
  kind: ZodOptional<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  oob_suspend_supported: ZodOptional<ZodBoolean>;
  pod_name: ZodOptional<ZodString>;
  result: ZodOptional<ZodUnknown>;
  status: ZodString;
  tool_api_error: ZodOptional<ZodNullable<ZodObject<{
     classification: ZodString;
     error_code: ZodOptional<ZodNullable<ZodString>>;
     http_status: ZodOptional<ZodNullable<ZodNumber>>;
     provider_type: ZodOptional<ZodNullable<ZodString>>;
     reason: ZodOptional<ZodNullable<ZodString>>;
     retryable: ZodBoolean;
  }, $strip>>>;
}, $strip>;
```

Response from Tool Pod execution.

***

<a id="api-toolresultrequestschema"></a>

### ToolResultRequestSchema

```ts
const ToolResultRequestSchema: ZodObject<{
  completion_tokens: ZodOptional<ZodNumber>;
  duration_ms: ZodNumber;
  error: ZodOptional<ZodString>;
  execution_id: ZodString;
  kind: ZodOptional<ZodString>;
  metadata: ZodDefault<ZodRecord<ZodString, ZodUnknown>>;
  model: ZodOptional<ZodString>;
  pod_name: ZodOptional<ZodString>;
  prompt_tokens: ZodOptional<ZodNumber>;
  result: ZodOptional<ZodUnknown>;
  span_id: ZodOptional<ZodNullable<ZodString>>;
  status: ZodString;
  step_number: ZodNumber;
  tool_api_error: ZodOptional<ZodNullable<ZodObject<{
     classification: ZodString;
     error_code: ZodOptional<ZodNullable<ZodString>>;
     http_status: ZodOptional<ZodNullable<ZodNumber>>;
     provider_type: ZodOptional<ZodNullable<ZodString>>;
     reason: ZodOptional<ZodNullable<ZodString>>;
     retryable: ZodBoolean;
  }, $strip>>>;
  tool_call_id: ZodOptional<ZodString>;
  tool_name: ZodString;
  total_tokens: ZodOptional<ZodNumber>;
  trace_id: ZodOptional<ZodNullable<ZodString>>;
  workspace_id: ZodOptional<ZodString>;
}, $strip>;
```

Report tool execution result (AER → OE).

***

<a id="api-toolslistresponseschema"></a>

### ToolsListResponseSchema

```ts
const ToolsListResponseSchema: ZodObject<{
  count: ZodOptional<ZodNumber>;
  tools: ZodArray<ZodRecord<ZodString, ZodUnknown>>;
}, $strip>;
```

Response from /tools endpoint listing registered tools.

***

<a id="api-workflowdeclarationschema"></a>

### WorkflowDeclarationSchema

```ts
const WorkflowDeclarationSchema: GenMessage<WorkflowDeclaration>;
```

Describes the message mongodb.agentic.workflow.v1.WorkflowDeclaration.
Use `create(WorkflowDeclarationSchema)` to create a new message.

***

<a id="api-workflowerrorcodeschema"></a>

### WorkflowErrorCodeSchema

```ts
const WorkflowErrorCodeSchema: GenEnum<WorkflowErrorCode>;
```

Describes the enum mongodb.agentic.workflow.v1.WorkflowErrorCode.

***

<a id="api-workflowerrorschema"></a>

### WorkflowErrorSchema

```ts
const WorkflowErrorSchema: GenMessage<WorkflowError>;
```

Describes the message mongodb.agentic.workflow.v1.WorkflowError.
Use `create(WorkflowErrorSchema)` to create a new message.

***

<a id="api-workflowidentityschema"></a>

### WorkflowIdentitySchema

```ts
const WorkflowIdentitySchema: GenMessage<WorkflowIdentity>;
```

Describes the message mongodb.agentic.workflow.v1.WorkflowIdentity.
Use `create(WorkflowIdentitySchema)` to create a new message.

***

<a id="api-workflowmessageschema"></a>

### WorkflowMessageSchema

```ts
const WorkflowMessageSchema: GenMessage<WorkflowMessage>;
```

Describes the message mongodb.agentic.workflow.v1.WorkflowMessage.
Use `create(WorkflowMessageSchema)` to create a new message.

***

<a id="api-workspace_dir"></a>

### WORKSPACE\_DIR

```ts
const WORKSPACE_DIR: string;
```

## Functions

<a id="api-accumulatestreamusage"></a>

### accumulateStreamUsage()

```ts
function accumulateStreamUsage(existing, incoming): LLMTokenUsage | undefined;
```

Fold a stream chunk's usage into a running snapshot.

Cumulative providers repeat growing totals (`10/1` then `10/2`);
last-wins merge keeps `10/2`. Additive providers zero-fill the
unchanged side (`18/1` then `0/4`); those deltas are summed.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `existing` | [`LLMTokenUsage`](#api-llmtokenusage) \| `undefined` |
| `incoming` | `unknown` |

#### Returns

[`LLMTokenUsage`](#api-llmtokenusage) \| `undefined`

***

<a id="api-activityrequiresreconstruction"></a>

### activityRequiresReconstruction()

```ts
function activityRequiresReconstruction(activityId): boolean;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `activityId` | `string` |

#### Returns

`boolean`

***

<a id="api-addtokenusage"></a>

### addTokenUsage()

```ts
function addTokenUsage(existing, incoming): LLMTokenUsage | undefined;
```

Sum LangChain-style additive usage deltas into a cumulative snapshot.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `existing` | [`LLMTokenUsage`](#api-llmtokenusage) \| `undefined` |
| `incoming` | `unknown` |

#### Returns

[`LLMTokenUsage`](#api-llmtokenusage) \| `undefined`

***

<a id="api-advancestepordinal"></a>

### advanceStepOrdinal()

```ts
function advanceStepOrdinal(committedStepOrdinal): number;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `committedStepOrdinal` | `number` |

#### Returns

`number`

***

<a id="api-allocateactivityordinal"></a>

### allocateActivityOrdinal()

```ts
function allocateActivityOrdinal(key?): number;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `key?` | `string` |

#### Returns

`number`

***

<a id="api-appliestostage"></a>

### appliesToStage()

```ts
function appliesToStage(policy, stage): boolean;
```

Whether `policy` applies at `stage`. Inactive policies never apply; an empty
stage filter applies to all stages; otherwise the stage must match (case- and
whitespace-insensitive). Mirrors `GuardrailRuntimePolicy.applies_to_stage`.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `policy` | \{ `action`: `string`; `config`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `id`: `string`; `stage_filter`: `string`[]; `status`: `string`; `type`: `string`; \} | - |
| `policy.action` | `string` | Action requested when the policy triggers. |
| `policy.config` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | Evaluator-specific policy configuration. |
| `policy.id` | `string` | Guardrail policy identifier. |
| `policy.stage_filter` | `string`[] | Runtime stages this policy applies to; empty means all stages. |
| `policy.status` | `string` | Guardrail policy status. |
| `policy.type` | `string` | Guardrail policy type. |
| `stage` | `"llm_input"` \| `"llm_output"` \| `"tool_input"` \| `"tool_output"` | - |

#### Returns

`boolean`

***

<a id="api-attachmongotracing"></a>

### attachMongoTracing()

```ts
function attachMongoTracing(collection): void;
```

Wire the recovered trace-store collection into the lazy Mongo exporter
registered at setup time. Called from the background retry loop
once a failed connection recovers. Idempotent — a no-op if already attached.
New spans created after this call route to the store; spans emitted while the
store was unreachable are not retroactively recovered.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `collection` | `Collection` |

#### Returns

`void`

***

<a id="api-attemptstartrequestfromexecute"></a>

### attemptStartRequestFromExecute()

```ts
function attemptStartRequestFromExecute(
   request,
   sessionId,
   ownerId,
   appName,
   workflowVersion?,
   memoryEnabled?
): AttemptStartRequest | null;
```

#### Parameters

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `request` | \{ `custom_headers?`: `Record`\<`string`, `string`\>; `execution_id`: `string`; `message`: `string`; `metadata?`: `Record`\<`string`, `unknown`\>; `org_id?`: `string`; `payload?`: `Record`\<`string`, `unknown`\>; `platform_api_owner_url?`: `string` \| `null`; `platform_api_url`: `string`; `platform_trace_id?`: `string` \| `null`; `previous_execution_cancelled`: `boolean`; `project_id?`: `string`; `resume`: `boolean`; `resume_data?`: `Record`\<`string`, `unknown`\>; `resume_from_step?`: `number`; `session_id?`: `string`; `suspend_generation?`: `number`; `user_id?`: `string`; `workspace_id?`: `string`; \} | `undefined` | - |
| `request.custom_headers?` | `Record`\<`string`, `string`\> | `...` | Caller-provided custom headers. |
| `request.execution_id` | `string` | `...` | Unique execution identifier. |
| `request.message` | `string` | `...` | User message. Defaults to empty. Promotion of `payload.message` into an empty top-level message happens in the AER (`resolveInvocationParams`), NOT at the model level — mirrors Python's `ExecuteRequest`. Keeping it out of the model lets the AER reject a message supplied in both places as a clean 400 instead of a 422 that would echo the payload into logs. |
| `request.metadata?` | `Record`\<`string`, `unknown`\> | `...` | Opaque framework-owned state passed through to the selected adapter. |
| `request.org_id?` | `string` | `...` | Organization ID for multi-tenant isolation. |
| `request.payload?` | `Record`\<`string`, `unknown`\> | `...` | Opaque caller-provided input forwarded unchanged on every dispatch; `payload.message` fills the top-level message when it is empty. |
| `request.platform_api_owner_url?` | `string` \| `null` | `...` | Replica-specific OE owner URL for callback fallback. |
| `request.platform_api_url?` | `string` | `...` | URL of the OE for callbacks. |
| `request.platform_trace_id?` | `string` \| `null` | `...` | - |
| `request.previous_execution_cancelled?` | `boolean` | `...` | True when this session's most recent prior execution was cancelled (pod torn down mid-run); the framework adapter fences off that run's partial checkpoint writes before running this turn. |
| `request.project_id?` | `string` | `...` | Project ID for project-level scoping. |
| `request.resume?` | `boolean` | `...` | Whether this is a resume after SUSPEND. |
| `request.resume_data?` | `Record`\<`string`, `unknown`\> | `...` | Structured data to inject on resume. |
| `request.resume_from_step?` | `number` | `...` | Step to resume from. |
| `request.session_id?` | `string` | `...` | Session ID; the AER adapter uses this as the LangGraph thread_id. |
| `request.suspend_generation?` | `number` | `...` | Suspension generation expected by this dispatch. |
| `request.user_id?` | `string` | `...` | User ID for personalization. |
| `request.workspace_id?` | `string` | `...` | Workspace identifier for cost tracking. |
| `sessionId?` | `string` | `undefined` | - |
| `ownerId?` | `string` | `undefined` | - |
| `appName?` | `string` | `undefined` | - |
| `workflowVersion?` | `string` | `undefined` | - |
| `memoryEnabled?` | `boolean` | `false` | - |

#### Returns

[`AttemptStartRequest`](#api-attemptstartrequest) \| `null`

***

<a id="api-boundtext"></a>

### boundText()

```ts
function boundText(
   text,
   maxBytes?,
   maxLines?
): string;
```

Truncate `text` to at most `maxLines` lines or `maxBytes` UTF-8 bytes,
whichever limit is hit first, appending a truncation marker when either
limit was hit.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `text` | `string` | `undefined` |
| `maxBytes` | `number` | `TERMINATION_MESSAGE_MAX_BYTES` |
| `maxLines` | `number` | `TERMINATION_MESSAGE_MAX_LINES` |

#### Returns

`string`

***

<a id="api-buildactivitycommand"></a>

### buildActivityCommand()

```ts
function buildActivityCommand(args): ActivityCommand;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `activityOrdinal`: `number`; `attempt`: [`AttemptContext`](#api-attemptcontext); `kind`: [`ActivityKind`](#api-activitykind); `name`: `string`; `operationPath?`: [`OperationPath`](#api-operationpath); `semanticInput`: `unknown`; `stepOrdinal?`: `number`; \} |
| `args.activityOrdinal` | `number` |
| `args.attempt` | [`AttemptContext`](#api-attemptcontext) |
| `args.kind` | [`ActivityKind`](#api-activitykind) |
| `args.name` | `string` |
| `args.operationPath?` | [`OperationPath`](#api-operationpath) |
| `args.semanticInput` | `unknown` |
| `args.stepOrdinal?` | `number` |

#### Returns

[`ActivityCommand`](#api-activitycommand)

***

<a id="api-buildresultpayload"></a>

### buildResultPayload()

```ts
function buildResultPayload(result, executionId): Record<string, unknown>;
```

Encode the result to a JSON-safe object for the /tool/result POST.

A tool may return a value the JSON encoder cannot handle (e.g. a circular
reference). Downgrade any encode failure to an error report so the OE still
receives a terminal signal. Mirrors Python's _result_payload.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `result` | \{ `completion_tokens?`: `number`; `duration_ms`: `number`; `error?`: `string`; `execution_id`: `string`; `kind?`: `string`; `metadata`: `Record`\<`string`, `unknown`\>; `model?`: `string`; `pod_name?`: `string`; `prompt_tokens?`: `number`; `result?`: `unknown`; `span_id?`: `string` \| `null`; `status`: `string`; `step_number`: `number`; `tool_api_error?`: \| \{ `classification`: `string`; `error_code?`: `string` \| `null`; `http_status?`: `number` \| `null`; `provider_type?`: `string` \| `null`; `reason?`: `string` \| `null`; `retryable`: `boolean`; \} \| `null`; `tool_call_id?`: `string`; `tool_name`: `string`; `total_tokens?`: `number`; `trace_id?`: `string` \| `null`; `workspace_id?`: `string`; \} | - |
| `result.completion_tokens?` | `number` | Completion/output tokens used. |
| `result.duration_ms` | `number` | Execution duration in milliseconds. |
| `result.error?` | `string` | Error message if failed. |
| `result.execution_id` | `string` | Execution identifier. |
| `result.kind?` | `string` | Explicit observability kind. |
| `result.metadata` | `Record`\<`string`, `unknown`\> | Observability metadata. |
| `result.model?` | `string` | LLM model name. |
| `result.pod_name?` | `string` | Hostname/pod name where tool was executed. |
| `result.prompt_tokens?` | `number` | Prompt/input tokens used. |
| `result.result?` | `unknown` | Tool result if successful. |
| `result.span_id?` | `string` \| `null` | Active OTel span ID, for OE execution-log correlation. |
| `result.status` | `string` | Execution status: success, error, suspend, interrupted. |
| `result.step_number` | `number` | Step number. |
| `result.tool_api_error?` | \| \{ `classification`: `string`; `error_code?`: `string` \| `null`; `http_status?`: `number` \| `null`; `provider_type?`: `string` \| `null`; `reason?`: `string` \| `null`; `retryable`: `boolean`; \} \| `null` | Structured external API failure classification. |
| `result.tool_call_id?` | `string` | Stable LLM tool-call id; joins this result's execution-log record to its call and the session message. |
| `result.tool_name` | `string` | Tool name. |
| `result.total_tokens?` | `number` | Total tokens used. |
| `result.trace_id?` | `string` \| `null` | Active OTel trace ID, for OE execution-log correlation. |
| `result.workspace_id?` | `string` | Workspace identifier. |
| `executionId` | `string` | - |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-callmcptool"></a>

### callMcpTool()

```ts
function callMcpTool(binding, args): Promise<MCPToolResult>;
```

Execute a remote MCP tool and normalize its result.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `binding` | [`MCPToolBinding`](#api-mcptoolbinding) |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Promise`\<[`MCPToolResult`](#api-mcptoolresult)\>

***

<a id="api-captureexception"></a>

### captureException()

```ts
function captureException(exc, args?): void;
```

Capture an exception. When `summary` is given, the reported error message is
the (redacted) summary with the original exception's type/message preserved
as scoped extras — matching Python's `_exception_for_capture`.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `exc` | `unknown` |
| `args` | [`CaptureExceptionArgs`](#api-captureexceptionargs) |

#### Returns

`void`

***

<a id="api-capturemessage"></a>

### captureMessage()

```ts
function captureMessage(
   message,
   level?,
   extra?
): void;
```

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `message` | `string` | `undefined` |
| `level` | `SeverityLevel` | `"error"` |
| `extra` | `Record`\<`string`, `unknown`\> | `{}` |

#### Returns

`void`

***

<a id="api-checkdecision"></a>

### checkDecision()

```ts
function checkDecision(policy): "allow" | "block" | "modify" | "require_review" | "log_only";
```

Resolve the decision a triggered policy requests, from its `action` (falling
back to `config.on_fail`). Mirrors `GuardrailRuntimePolicy.check_decision`.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `policy` | \{ `action`: `string`; `config`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `id`: `string`; `stage_filter`: `string`[]; `status`: `string`; `type`: `string`; \} | - |
| `policy.action` | `string` | Action requested when the policy triggers. |
| `policy.config` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | Evaluator-specific policy configuration. |
| `policy.id` | `string` | Guardrail policy identifier. |
| `policy.stage_filter` | `string`[] | Runtime stages this policy applies to; empty means all stages. |
| `policy.status` | `string` | Guardrail policy status. |
| `policy.type` | `string` | Guardrail policy type. |

#### Returns

`"allow"` \| `"block"` \| `"modify"` \| `"require_review"` \| `"log_only"`

***

<a id="api-childoperationboundaryscope"></a>

### childOperationBoundaryScope()

```ts
function childOperationBoundaryScope<T>(boundary, fn): T;
```

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `boundary` | [`ChildOperationBoundary`](#api-childoperationboundary) |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-clearworkflowadapter"></a>

### clearWorkflowAdapter()

```ts
function clearWorkflowAdapter(): void;
```

Drop durable eligibility for the latest materialized graph.

#### Returns

`void`

***

<a id="api-closesessionfinish"></a>

### closeSessionFinish()

```ts
function closeSessionFinish(): void;
```

Close the current execution's session-finish latch.

Called from the AER's `finally` once the execute frame ends (covers
success, error, policy-denied, and suspend paths alike). After this,
requestSessionFinish() reports "unavailable" instead of promising a
release nothing will act on — e.g. a setTimeout or floating promise
scheduled during the turn but resolving after it.

#### Returns

`void`

***

<a id="api-coercetokenusage"></a>

### coerceTokenUsage()

```ts
function coerceTokenUsage(value): LLMTokenUsage | undefined;
```

Coerce a provider usage blob into LLMTokenUsage, or undefined.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `unknown` |

#### Returns

[`LLMTokenUsage`](#api-llmtokenusage) \| `undefined`

***

<a id="api-completedoutcome"></a>

### completedOutcome()

```ts
function completedOutcome(context, result): ActivityOutcome;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `context` | [`ActivityContext`](#api-activitycontext) |
| `result` | `unknown` |

#### Returns

[`ActivityOutcome`](#api-activityoutcome)

***

<a id="api-completeexecutioncommand-1"></a>

### completeExecutionCommand()

```ts
function completeExecutionCommand(attempt, state): CompleteExecutionCommand;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `attempt` | [`AttemptContext`](#api-attemptcontext) |
| `state` | [`StateSnapshot`](#api-statesnapshot) |

#### Returns

[`CompleteExecutionCommand`](#api-completeexecutioncommand)

***

<a id="api-createsecuretoolfunction"></a>

### createSecureToolFunction()

```ts
function createSecureToolFunction(
   originalTool,
   toolName,
   allowDirect?,
   options?
): (kwargs?, config?) => Promise<unknown>;
```

Create a wrapped tool function that routes through SecureToolWrapper.

The wrapper is looked up from AsyncLocalStorage context at call time,
so the tool can be built before execution context exists.

The registered tool must expose an `invoke` method for both approved local
execution and the direct-execution debugging path. Direct execution is
UNSAFE and only for debugging.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `originalTool` | `unknown` | `undefined` |
| `toolName` | `string` | `undefined` |
| `allowDirect` | `boolean` | `false` |
| `options` | `CreateSecureToolFunctionOptions` | `{}` |

#### Returns

(`kwargs?`, `config?`) => `Promise`\<`unknown`\>

***

<a id="api-currentattemptcontext"></a>

### currentAttemptContext()

```ts
function currentAttemptContext(): AttemptContext | null;
```

#### Returns

[`AttemptContext`](#api-attemptcontext) \| `null`

***

<a id="api-currentoperationpath"></a>

### currentOperationPath()

```ts
function currentOperationPath(): OperationPath;
```

#### Returns

[`OperationPath`](#api-operationpath)

***

<a id="api-currentpendingchildoperationbatch"></a>

### currentPendingChildOperationBatch()

```ts
function currentPendingChildOperationBatch(): readonly ChildOperationBoundary[];
```

#### Returns

readonly [`ChildOperationBoundary`](#api-childoperationboundary)[]

***

<a id="api-currentstepordinal"></a>

### currentStepOrdinal()

```ts
function currentStepOrdinal(): number;
```

#### Returns

`number`

***

<a id="api-discovermcptools"></a>

### discoverMcpTools()

```ts
function discoverMcpTools(config): Promise<MCPToolBinding[]>;
```

Discover remote MCP tools for all configured servers.

Mirrors common MCP client adapters: initialize each server, call
`tools/list`, then expose the returned schemas as framework-native tools.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `config` | \{ `servers`: `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\>; \} |
| `config.servers` | `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\> |

#### Returns

`Promise`\<[`MCPToolBinding`](#api-mcptoolbinding)[]\>

***

<a id="api-emit"></a>

### emit()

```ts
function emit(event, data): Promise<void>;
```

Emit a chunk of the given event type to the client stream.

General-purpose API. For the common textual-step case, prefer
[emitStep](#api-emitstep).

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `event` | `string` | Chunk type identifier (e.g. `"step"`). Reserved infrastructure event types (`done`, `error`, `text`, `subagent_start`, `subagent_end`) reject with an `Error` to prevent tool code from prematurely closing or spoofing the stream. The reserved-event check fires even outside an execution context — a programmer error wins over the no-op. `"custom_event"` is not Atlas Agent Engine custom events; those require `emit_custom_event` and `features.use_custom_parser` in `agent.yaml`. Gateway drops `chunk_type: "custom_event"` unless that feature is on. |
| `data` | `string` | Payload string for the event. Transport failures (including a non-2xx response) are logged at debug and swallowed — best-effort delivery must never abort the tool function. No-op outside an execution context. |

#### Returns

`Promise`\<`void`\>

***

<a id="api-emitstep"></a>

### emitStep()

```ts
function emitStep(message): Promise<void>;
```

Emit a textual progress step from within a running tool function.

Convenience wrapper around `emit("step", message)`.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |

#### Returns

`Promise`\<`void`\>

#### Example

```ts
import { emitStep } from "@mongodb-js/agent-engine-runner-shared";

app.tool({ isLocal: false })(async function crawlWebsite(url: string) {
  await emitStep("Fetching page...");
  const html = await fetch(url);
  await emitStep(`Parsing links from ${url}...`);
  return parse(html);
});
```

***

<a id="api-encodeprotojson"></a>

### encodeProtoJson()

```ts
function encodeProtoJson<T>(schema, message): string;
```

Encode one protobuf message as a ProtoJSON string using wire field names.
Shared so framework adapters post platform-shaped JSON bodies without
hand-rolling serialization.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` *extends* `Message` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `schema` | `GenMessage`\<`T`\> |
| `message` | `T` |

#### Returns

`string`

***

<a id="api-ensureutc"></a>

### ensureUtc()

```ts
function ensureUtc(value): Date | undefined;
```

Normalize a value (Date, ISO string, or null/undefined) to a UTC Date.

MongoDB stores naive datetimes in UTC; this helper rehydrates them as
proper Date instances so callers don't mix string and Date types.

Mirrors Python `Execution._ensure_utc`. JS `Date` has no naive/aware
distinction (every Date is internally UTC ms-since-epoch), so the TS
version exercises the same "normalize input to a canonical Date" intent
by coercing ISO strings and passing Date instances through unchanged.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `unknown` |

#### Returns

`Date` \| `undefined`

***

<a id="api-entrypointscope"></a>

### entrypointScope()

```ts
function entrypointScope<T>(fn): T;
```

Mark the dynamic extent of the user's @app.entrypoint call.

**Framework-internal — user agent code must never call this.** Users only
declare an entrypoint (`app.entrypoint(fn)`); the framework SDK wraps its
own evaluation of that function (during AER graph construction and
during Tool preparation, retried after failure) in this helper so
that `registerLlm()` can reject `app.llm()` calls made outside the
entrypoint. Tests that call `app.llm()`/`registerLlm()` directly (bypassing
`app.entrypoint` + `getAgent()`) use it to simulate that framework
evaluation.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-evaluateguardrailcheck"></a>

### evaluateGuardrailCheck()

```ts
function evaluateGuardrailCheck(request): object;
```

Evaluate guardrail policies for one runtime boundary.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `request` | \{ `context`: \{ `org_id`: `string`; `project_id`: `string`; `session_id?`: `string` \| `null`; `user_id?`: `string` \| `null`; `workspace_id?`: `string` \| `null`; \}; `execution_id`: `string`; `input`: \{ `metadata`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `text`: `string`; \}; `policies`: `object`[]; `stage`: `"llm_input"` \| `"llm_output"` \| `"tool_input"` \| `"tool_output"`; \} | - |
| `request.context` | \{ `org_id`: `string`; `project_id`: `string`; `session_id?`: `string` \| `null`; `user_id?`: `string` \| `null`; `workspace_id?`: `string` \| `null`; \} | Execution context. |
| `request.context.org_id` | `string` | Organization ID. |
| `request.context.project_id` | `string` | Project ID. |
| `request.context.session_id?` | `string` \| `null` | Session ID. |
| `request.context.user_id?` | `string` \| `null` | User ID. |
| `request.context.workspace_id?` | `string` \| `null` | Workspace ID. |
| `request.execution_id` | `string` | Execution identifier. |
| `request.input` | \{ `metadata`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `text`: `string`; \} | Content to evaluate. |
| `request.input.metadata` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | Runtime metadata such as model, tool name, or source. |
| `request.input.text` | `string` | Text content to evaluate. |
| `request.policies` | `object`[] | OE-selected policies to evaluate. |
| `request.stage` | `"llm_input"` \| `"llm_output"` \| `"tool_input"` \| `"tool_output"` | Runtime stage being evaluated. |

#### Returns

| Name | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `allowed` | `boolean` | - | Whether execution may continue. |
| `decision` | `"allow"` \| `"block"` \| `"modify"` \| `"require_review"` \| `"log_only"` | `GuardrailCheckDecisionSchema` | Guardrails decision. |
| `evidence` | `object`[] | - | Policy trigger evidence. |
| `metadata` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> | - | Evaluator-specific response metadata. |
| `reason?` | `string` \| `null` | - | Decision reason. |
| `transformed_text?` | `string` \| `null` | - | Modified text when decision is modify, or original text for allow/no-op. |
| `triggered_policy_ids` | `string`[] | - | Policy IDs that triggered. |

***

<a id="api-executionfrompersistencedoc"></a>

### executionFromPersistenceDoc()

```ts
function executionFromPersistenceDoc(doc): object;
```

Rehydrate an Execution from a MongoDB document.

Uses default values for Optional fields so documents created before
new fields were added still deserialize safely.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `doc` | `Record`\<`string`, `unknown`\> |

#### Returns

| Name | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `aer_url?` | `string` | - | AER HTTP endpoint for this agent; overrides default AER_URL. |
| `created_at` | `Date` | - | - |
| `error?` | `string` | - | Error message. |
| `id` | `string` | - | Unique execution identifier. |
| `message` | `string` | - | Original user message. |
| `org_id?` | `string` | - | Organization ID for multi-tenant isolation. |
| `project_id?` | `string` | - | Project ID for project-level scoping. |
| `result?` | `unknown` | - | Final result. |
| `session_id?` | `string` | - | Session ID for trace correlation. |
| `status` | \| `"error"` \| `"pending"` \| `"running"` \| `"suspended"` \| `"resuming"` \| `"completed"` \| `"cancelled"` | `ExecutionStatusSchema` | Current status. |
| `suspend_context?` | `Record`\<`string`, `unknown`\> | - | Context for resume. |
| `suspend_reason?` | `string` | - | Reason for suspension. |
| `tool_url?` | `string` | - | Tool HTTP endpoint for this agent; overrides default TOOL_URL. |
| `updated_at` | `Date` | - | - |
| `user_id?` | `string` | - | User ID for personalization. |
| `workspace_id?` | `string` | - | Workspace identifier for cost tracking. |

***

<a id="api-executionstepfromlogdoc"></a>

### executionStepFromLogDoc()

```ts
function executionStepFromLogDoc(doc): object;
```

Reconstruct an ExecutionStep from an execution_logs MongoDB document.

The execution_logs collection stores tool start/result events with
fields: execution_id, step_number, tool, inputs, status, output,
error, duration_ms, timestamp. This maps those fields back to the
ExecutionStep model for step-cache rehydration after OE restart.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `doc` | `Record`\<`string`, `unknown`\> |

#### Returns

| Name | Type | Description |
| :------ | :------ | :------ |
| `arguments` | `Record`\<`string`, `unknown`\> | Arguments passed. |
| `duration_ms?` | `number` | Execution duration. |
| `error?` | `string` | Error message. |
| `execution_id` | `string` | Parent execution ID. |
| `id` | `string` | Unique step identifier. |
| `result?` | `unknown` | Step result. |
| `status` | `string` | Step status: pending, success, error. |
| `step_number` | `number` | Step sequence number. |
| `timestamp` | `Date` | ISO strings are coerced to Date, matching Pydantic. |
| `tool_name` | `string` | Tool or operation name. |

***

<a id="api-executiontopersistencedoc"></a>

### executionToPersistenceDoc()

```ts
function executionToPersistenceDoc(exec, updatedAt?): Record<string, unknown>;
```

Convert an Execution to a MongoDB persistence document.

`updatedAt` defaults to now if omitted.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `exec` | \{ `aer_url?`: `string`; `created_at`: `Date`; `error?`: `string`; `id`: `string`; `message`: `string`; `org_id?`: `string`; `project_id?`: `string`; `result?`: `unknown`; `session_id?`: `string`; `status`: \| `"error"` \| `"pending"` \| `"running"` \| `"suspended"` \| `"resuming"` \| `"completed"` \| `"cancelled"`; `suspend_context?`: `Record`\<`string`, `unknown`\>; `suspend_reason?`: `string`; `tool_url?`: `string`; `updated_at`: `Date`; `user_id?`: `string`; `workspace_id?`: `string`; \} | - |
| `exec.aer_url?` | `string` | AER HTTP endpoint for this agent; overrides default AER_URL. |
| `exec.created_at?` | `Date` | - |
| `exec.error?` | `string` | Error message. |
| `exec.id?` | `string` | Unique execution identifier. |
| `exec.message?` | `string` | Original user message. |
| `exec.org_id?` | `string` | Organization ID for multi-tenant isolation. |
| `exec.project_id?` | `string` | Project ID for project-level scoping. |
| `exec.result?` | `unknown` | Final result. |
| `exec.session_id?` | `string` | Session ID for trace correlation. |
| `exec.status?` | \| `"error"` \| `"pending"` \| `"running"` \| `"suspended"` \| `"resuming"` \| `"completed"` \| `"cancelled"` | Current status. |
| `exec.suspend_context?` | `Record`\<`string`, `unknown`\> | Context for resume. |
| `exec.suspend_reason?` | `string` | Reason for suspension. |
| `exec.tool_url?` | `string` | Tool HTTP endpoint for this agent; overrides default TOOL_URL. |
| `exec.updated_at?` | `Date` | - |
| `exec.user_id?` | `string` | User ID for personalization. |
| `exec.workspace_id?` | `string` | Workspace identifier for cost tracking. |
| `updatedAt?` | `Date` | - |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-explicitfeatures"></a>

### explicitFeatures()

```ts
function explicitFeatures(features): Record<string, boolean>;
```

Return only the flags explicitly present in `agent.yaml` (omit unset /
`null` fields). Mirrors Python's `AgentFeatureConfig.explicit()` — used by
the AER's capability advertise to send OE only the flags the author
actually set, before SDK-injected defaults (e.g. `owner_callback_fallback`)
are layered on top.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `features` | \{ `deep_agent`: `boolean` \| `null`; `durable_workflow`: `boolean` \| `null`; `guardrails`: `boolean` \| `null`; `memory`: `boolean` \| `null`; `playground`: `boolean` \| `null`; `use_custom_parser`: `boolean` \| `null`; \} | - |
| `features.deep_agent` | `boolean` \| `null` | Whether the agent uses the deep-agent (deepagents) harness. Gates the Tool Pod's built-in filesystem + shell handler registration so tenants that don't run deep agents get no filesystem/shell surface on their Tool Pod. The key stays snake_case `deep_agent` because `agent.yaml` is a cross-language artifact shared with the Python runtime and platform. |
| `features.durable_workflow` | `boolean` \| `null` | Opt-in for OE-owned durable workflow. Omitted or false means the agent stays on native checkpoints; only an explicit true opts in. When set, OE uses this flag with the advertised language to sticky-assign the session's workflow authority. |
| `features.guardrails` | `boolean` \| `null` | Whether guardrails are enforced for this agent at runtime. `null` (the default) means omitted, letting the runtime fall back to legacy behavior. |
| `features.memory` | `boolean` \| `null` | - |
| `features.playground` | `boolean` \| `null` | Whether the platform provisions playground UI for the agent (null/true = provisioned, today's behavior). When false — e.g. for non-chat agents with no conversation to preview — no playground is built or served; callers use the invoke API directly. Read at deploy/provisioning time only — no runtime effect. |
| `features.use_custom_parser` | `boolean` \| `null` | Opt-in for author-defined streaming output shaping. When true the adapter runs the registered output parser and emits `custom_event` frames; off leaves the stream unchanged. |

#### Returns

`Record`\<`string`, `boolean`\>

***

<a id="api-extractusage-1"></a>

### extractUsage()

```ts
function extractUsage(response, fallbackModel?): ExtractedUsage;
```

Extract token usage from an LLM response's `response_metadata`.

Handles provider-variant key names:
  - OpenAI:   `response_metadata.usage.{prompt_tokens, completion_tokens, total_tokens}`
  - Anthropic: `response_metadata.usage.{input_tokens, output_tokens}`
  - Some providers: `response_metadata.token_usage.{...}`

Returns an object with keys `prompt_tokens`, `completion_tokens`,
`total_tokens`, `model`. All values may be `null` if unavailable.

Mirrors Python's `agent_engine_runner_shared.secure_wrapper.extract_usage` 1:1.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `response` | `unknown` | `undefined` |
| `fallbackModel` | `string` \| `null` | `null` |

#### Returns

[`ExtractedUsage`](#api-extractedusage)

***

<a id="api-failedoutcome"></a>

### failedOutcome()

```ts
function failedOutcome(
   context,
   message,
   code?
): ActivityOutcome;
```

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `context` | [`ActivityContext`](#api-activitycontext) | `undefined` |
| `message` | `string` | `undefined` |
| `code` | [`WorkflowErrorCode`](#api-workflowerrorcode) | `WorkflowErrorCode.OUTCOME_UNKNOWN` |

#### Returns

[`ActivityOutcome`](#api-activityoutcome)

***

<a id="api-fetchplatform"></a>

### fetchPlatform()

```ts
function fetchPlatform(
   url,
   init,
   fetchImpl?
): Promise<Response>;
```

Carry the active trace across a platform RPC without recording transport.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `url` | `string` | `undefined` |
| `init` | `RequestInit` | `undefined` |
| `fetchImpl` | \{ (`input`, `init?`): `Promise`\<`Response`\>; (`input`, `init?`): `Promise`\<`Response`\>; \} | `fetch` |

#### Returns

`Promise`\<`Response`\>

***

<a id="api-filesystemdownload"></a>

### filesystemDownload()

```ts
function filesystemDownload(args): Record<string, unknown>;
```

Download a file's raw bytes base64-encoded so the JSON transport can carry
arbitrary binary content. `AgentEngineToolPodBackend.downloadFiles` base64-decodes
this to produce the response bytes.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filesystemedit"></a>

### filesystemEdit()

```ts
function filesystemEdit(args): Record<string, unknown>;
```

Find and replace text in a file, returning the number of occurrences
replaced. When *replace_all* is false the match must be unique — a non-unique
`old_string` is rejected so a caller cannot silently corrupt the wrong
region. Files larger than FILESYSTEM_READ_MAX_BYTES (before or after the
edit) are rejected to mirror filesystem_read's memory guard.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filesystemglob"></a>

### filesystemGlob()

```ts
function filesystemGlob(args): Promise<Record<string, unknown>>;
```

Match files using a glob pattern under *path* (workspace-relative, default
`"."`). Supports recursive `**` patterns. Bounded by MAX_GLOB_MATCHES.
Absolute patterns and `..` traversal segments are rejected so they cannot
bypass the sandbox; returned matches are also filtered through
`isWithinReadableRoot` for defense-in-depth against symlinks.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Promise`\<`Record`\<`string`, `unknown`\>\>

***

<a id="api-filesystemgrep"></a>

### filesystemGrep()

```ts
function filesystemGrep(args): Record<string, unknown>;
```

Search file contents for a literal substring (per `BackendProtocol.grep`,
not a regex). Walks the tree under *path* (default `"."`); when *glob* is
given only files whose names match are searched. Binary files are skipped.
Bounded by MAX_GREP_MATCHES and MAX_GREP_SECONDS so a deep-tree walk cannot
peg a worker indefinitely.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filesystemls"></a>

### filesystemLs()

```ts
function filesystemLs(args): Record<string, unknown>;
```

List directory contents, sorted by name. When the directory has more than
MAX_LS_ENTRIES entries, `truncated=true` and the returned subset is the
readdir-order first MAX_LS_ENTRIES entries (filesystem-defined), then sorted
by path. The early break is intentional — sorting all entries first would
defeat the memory cap on directories with hundreds of thousands of entries.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filesystemread"></a>

### filesystemRead()

```ts
function filesystemRead(args): Record<string, unknown>;
```

Read file content as text with line-based slicing (lines
`[offset, offset+limit)`). Rejects files larger than
FILESYSTEM_READ_MAX_BYTES upfront so pathologically large files cannot
exhaust pod memory.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filesystemwrite"></a>

### filesystemWrite()

```ts
function filesystemWrite(args): Record<string, unknown>;
```

Write *content* to a new file, creating parent directories. Per
`BackendProtocol.write` this is create-only: if the file exists the call
fails. Agents modify existing files via filesystem_edit (which has a
unique-match guard).

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-filterthinkingtokens"></a>

### filterThinkingTokens()

```ts
function filterThinkingTokens(
   token,
   buffer,
   inside
): [string, string, boolean];
```

Filter `<think>` blocks from a stream of token chunks.

Accumulates text in *buffer* until we can determine whether content
is inside a thinking block. Returns `[streamable, newBuffer, inside]`
where *streamable* is the text safe to send to the client.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `token` | `string` |
| `buffer` | `string` |
| `inside` | `boolean` |

#### Returns

\[`string`, `string`, `boolean`\]

***

<a id="api-finalizecurrentstepcommand"></a>

### finalizeCurrentStepCommand()

```ts
function finalizeCurrentStepCommand(attempt, state): FinalizeStepCommand;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `attempt` | [`AttemptContext`](#api-attemptcontext) |
| `state` | [`StateSnapshot`](#api-statesnapshot) |

#### Returns

[`FinalizeStepCommand`](#api-finalizestepcommand)

***

<a id="api-finalizecurrentstepsuspensionscommand"></a>

### finalizeCurrentStepSuspensionsCommand()

```ts
function finalizeCurrentStepSuspensionsCommand(attempt, suspensions): FinalizeStepCommand;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `attempt` | [`AttemptContext`](#api-attemptcontext) |
| `suspensions` | readonly [`StepSuspensionEntry`](#api-stepsuspensionentry)[] |

#### Returns

[`FinalizeStepCommand`](#api-finalizestepcommand)

***

<a id="api-flusherrorreporting"></a>

### flushErrorReporting()

```ts
function flushErrorReporting(timeoutMs?): Promise<void>;
```

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `timeoutMs` | `number` | `2000` |

#### Returns

`Promise`\<`void`\>

***

<a id="api-formatllmerror"></a>

### formatLlmError()

```ts
function formatLlmError(error): string;
```

Render an LLM provider error for display, without escaped-JSON text.

Some provider SDKs attach the raw API error body as an object on the
error or its `cause` (`.details`, `.body`). That object's values are
often themselves JSON-encoded strings containing real newlines (e.g. a
pretty-printed nested error payload), so `String()`/template-literal
stringification ends up producing a repr that turns the newlines into
literal `\n` sequences. Decode any JSON-encoded string values
first and re-serialize with `JSON.stringify(..., null, 2)` so the result
renders as readable, indented JSON instead.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `error` | `unknown` |

#### Returns

`string`

***

<a id="api-getallcustomheaders"></a>

### getAllCustomHeaders()

```ts
function getAllCustomHeaders(): Record<string, string>;
```

All custom headers including platform-internal `a2a-` entries.

#### Returns

`Record`\<`string`, `string`\>

***

<a id="api-getcallabortsignal"></a>

### getCallAbortSignal()

```ts
function getCallAbortSignal(): AbortSignal | undefined;
```

The per-call stop signal for the in-flight callback-routed tool call.
Defined only inside a tool body that declared call-interrupt support; a
cooperative body checks it (or forwards it to `fetch` etc.) to stop at its
next checkpoint. Undefined everywhere else.

#### Returns

`AbortSignal` \| `undefined`

***

<a id="api-getcheckpointworkspaceid"></a>

### getCheckpointWorkspaceId()

```ts
function getCheckpointWorkspaceId(): string;
```

Resolved workspace scope for checkpoint reads (matches write path).

Returns "" only for intentionally unscoped local runtimes. Managed AERs
carry REQUIRE_PROJECT_SCOPED_DB and require APP_ID; they deliberately reject
the wire fallback if APP_ID is missing.

#### Returns

`string`

***

<a id="api-getcontentcapturemode"></a>

### getContentCaptureMode()

```ts
function getContentCaptureMode(): string;
```

Read the content-capture policy from `AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`.
Defaults to `"metadata-only"`; any value other than `"full"` is treated as
`"metadata-only"` so a typo'd env var fails closed, not open.

#### Returns

`string`

***

<a id="api-getcurrentauthorization"></a>

### getCurrentAuthorization()

```ts
function getCurrentAuthorization():
  | {
  expires_at?: number | null;
  token: string;
}
  | null;
```

Delegated authorization for the current execution, if OE injected one.

#### Returns

##### Type Literal

```ts
{
  expires_at?: number | null;
  token: string;
}
```

| Name | Type | Description |
| :------ | :------ | :------ |
| `expires_at?` | `number` \| `null` | Token expiry as Unix seconds. |
| `token` | `string` | Bearer token for third-party API access. |

***

`null`

***

<a id="api-getcurrentcustomheaders"></a>

### getCurrentCustomHeaders()

```ts
function getCurrentCustomHeaders(): Record<string, string>;
```

Caller-provided custom headers, with platform-internal `a2a-` entries
stripped — agent code should never see A2A tokens or routing metadata.
Internal platform code that needs the full set (e.g. the A2A client) should
call [getAllCustomHeaders](#api-getallcustomheaders) instead.

#### Returns

`Record`\<`string`, `string`\>

***

<a id="api-getcurrentexecutionid"></a>

### getCurrentExecutionId()

```ts
function getCurrentExecutionId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentexecutionmetadata"></a>

### getCurrentExecutionMetadata()

```ts
function getCurrentExecutionMetadata(): Record<string, unknown>;
```

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-getcurrentlogorigin"></a>

### getCurrentLogOrigin()

```ts
function getCurrentLogOrigin(): string | null;
```

`"customer"` inside a customer-code boundary, else null.

#### Returns

`string` \| `null`

***

<a id="api-getcurrentoeownerurl"></a>

### getCurrentOeOwnerUrl()

```ts
function getCurrentOeOwnerUrl(): string | null;
```

Validated replica-specific OE owner callback URL for the current execution,
or null when none was forwarded or a previous owner pre-attempt already
failed (see [reportOeOwnerUrlFailure](#api-reportoeownerurlfailure)). Owner-preferring transports
send here first and fall back to [getCurrentOeUrl](#api-getcurrentoeurl) on any owner
failure.

#### Returns

`string` \| `null`

***

<a id="api-getcurrentoeurl"></a>

### getCurrentOeUrl()

```ts
function getCurrentOeUrl(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentpayload"></a>

### getCurrentPayload()

```ts
function getCurrentPayload(): Record<string, unknown> | null;
```

The opaque caller-provided invocation payload for the current execution
(the request body beyond `message`), or null if none was forwarded.

#### Returns

`Record`\<`string`, `unknown`\> \| `null`

***

<a id="api-getcurrentrequestid"></a>

### getCurrentRequestId()

```ts
function getCurrentRequestId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentsessionid-1"></a>

### getCurrentSessionId()

```ts
function getCurrentSessionId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrenttracecontext"></a>

### getCurrentTraceContext()

```ts
function getCurrentTraceContext(): object;
```

#### Returns

`object`

| Name | Type |
| :------ | :------ |
| `spanId` | `string` \| `null` |
| `traceId` | `string` \| `null` |

***

<a id="api-getcurrenttraceid"></a>

### getCurrentTraceId()

```ts
function getCurrentTraceId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentuserid-1"></a>

### getCurrentUserId()

```ts
function getCurrentUserId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentworkspaceid"></a>

### getCurrentWorkspaceId()

```ts
function getCurrentWorkspaceId(): string | null;
```

#### Returns

`string` \| `null`

***

<a id="api-getcurrentwrapper"></a>

### getCurrentWrapper()

```ts
function getCurrentWrapper(): unknown;
```

#### Returns

`unknown`

***

<a id="api-getenv"></a>

### getEnv()

```ts
function getEnv(name, defaultValue?): string;
```

Get environment variable with default.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `name` | `string` | `undefined` |
| `defaultValue` | `string` | `""` |

#### Returns

`string`

***

<a id="api-getenvbool"></a>

### getEnvBool()

```ts
function getEnvBool(name, defaultValue?): boolean;
```

Get boolean environment variable with default.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `name` | `string` | `undefined` |
| `defaultValue` | `boolean` | `false` |

#### Returns

`boolean`

***

<a id="api-getenvfloat"></a>

### getEnvFloat()

```ts
function getEnvFloat(name, defaultValue): number;
```

Get float environment variable with default.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `name` | `string` |
| `defaultValue` | `number` |

#### Returns

`number`

***

<a id="api-getenvint"></a>

### getEnvInt()

```ts
function getEnvInt(name, defaultValue): number;
```

Get integer environment variable with default.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `name` | `string` |
| `defaultValue` | `number` |

#### Returns

`number`

***

<a id="api-getfetchoptionswithtls"></a>

### getFetchOptionsWithTLS()

```ts
function getFetchOptionsWithTLS(url): RequestInit | undefined;
```

Get fetch options configured for the given URL with mTLS support.

If url uses HTTPS and TLS certificate environment variables are set
(TLS_CERT_PATH, TLS_KEY_PATH, TLS_CA_CERT_PATH), returns options with
an HTTPS agent configured for mTLS. For HTTP URLs or when certificates
are not available, returns undefined (use default fetch behavior).

The HTTPS agent is cached per base URL for connection pooling.

Performance: checks the agent cache BEFORE reading certificate files to avoid
blocking the event loop on filesystem I/O for every request. Certificate files
are only read once per base URL when creating the agent, not on every fetch.

Certificate rotation: In container mode (TLS_*_PATH), automatically detects
cert-manager rotation by comparing file mtimes and invalidates the cached agent
so the next request picks up renewed certificates. In VM mode (TLS_*_PEM), the
env vars are static for the pod's lifetime, so rotation requires a pod restart.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `url` | `string` | The target service URL (http:// or https://) |

#### Returns

`RequestInit` \| `undefined`

Fetch options with agent, or undefined for HTTP/no-TLS

#### Throws

Error if HTTPS is used but TLS configuration is incomplete

***

<a id="api-getinstrumentor"></a>

### getInstrumentor()

```ts
function getInstrumentor(): Instrumentor | null;
```

#### Returns

[`Instrumentor`](#api-instrumentor) \| `null`

***

<a id="api-getllmadapterfactory"></a>

### getLLMAdapterFactory()

```ts
function getLLMAdapterFactory(): LLMAdapterFactory;
```

#### Returns

[`LLMAdapterFactory`](#api-llmadapterfactory)

***

<a id="api-getlogger"></a>

### getLogger()

```ts
function getLogger(name?): Logger;
```

Return a logger named after the caller's module.

Mirrors Python `logging.getLogger(__name__)`. log4js maintains a global
registry, so a later `setupLogging` reconfiguration is automatically
visible to every previously-issued reference. First call performs a
default configuration so module-scope `const logger = getLogger(__name__)`
patterns work without an explicit `setupLogging` at startup.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `name?` | `string` |

#### Returns

`Logger`

***

<a id="api-getnamedllm"></a>

### getNamedLlm()

```ts
function getNamedLlm(llmId): unknown;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `llmId` | `string` |

#### Returns

`unknown`

***

<a id="api-getqueryplugin"></a>

### getQueryPlugin()

```ts
function getQueryPlugin(): AERQueryPlugin | null;
```

Return the registered `AERQueryPlugin`, or `null` if none has been registered.

#### Returns

[`AERQueryPlugin`](#api-aerqueryplugin) \| `null`

***

<a id="api-getrequestedsuspend"></a>

### getRequestedSuspend()

```ts
function getRequestedSuspend(): Record<string, unknown> | null;
```

The suspend payload the current tool call requested via
`suspendPayloadToJson`, or null if it did not. The Tool Pod reads this after
the tool returns to decide whether to report `status: "suspend"`.

#### Returns

`Record`\<`string`, `unknown`\> \| `null`

***

<a id="api-getrequesttimeout"></a>

### getRequestTimeout()

```ts
function getRequestTimeout(): number;
```

Get the HTTP request timeout from environment.

Uses RUNNER_REQUEST_TIMEOUT env var, defaults to 60.0 seconds.

#### Returns

`number`

***

<a id="api-getruntimemode"></a>

### getRuntimeMode()

```ts
function getRuntimeMode(): RuntimeMode;
```

Get the current runtime mode from environment variable.

#### Returns

[`RuntimeMode`](#api-runtimemode)

RuntimeMode based on RUNNER_MODE environment variable.

#### Throws

Error If RUNNER_MODE is not set or is not a valid mode.

***

<a id="api-getstoredbname"></a>

### getStoreDbName()

```ts
function getStoreDbName(): string;
```

Return the base (unscoped) consolidated data-plane database name.

Reads `MDB_AGENTIC_STORE_DB` from the environment on every call so that
late configuration (e.g. loading a .env file after import) is respected. For
the per-project-scoped name use [resolveStoreDbName](#api-resolvestoredbname).

#### Returns

`string`

***

<a id="api-getsuspendhandler"></a>

### getSuspendHandler()

```ts
function getSuspendHandler(): SuspendHandler | null;
```

#### Returns

[`SuspendHandler`](#api-suspendhandler) \| `null`

***

<a id="api-gettoolreadtimeout"></a>

### getToolReadTimeout()

```ts
function getToolReadTimeout(): number;
```

Get the tool-call read timeout from environment.

The OE holds /tool/execute open until the tool result comes back, so this
bounds the tool's own runtime rather than the handshake. It matches the OE's
own tool deadline: a smaller value abandons a tool the platform is still
happily running, leaving the SDK with no result to report.

Uses RUNNER_TOOL_READ_TIMEOUT env var, defaults to 600.0 seconds.

#### Returns

`number`

***

<a id="api-gettracepath"></a>

### getTracePath()

```ts
function getTracePath(): string;
```

Resolve the on-disk trace file path from env vars (mirrors Python).

#### Returns

`string`

***

<a id="api-gettracer"></a>

### getTracer()

```ts
function getTracer(name?): Tracer;
```

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `name` | `string` | `"runner-sdk"` |

#### Returns

`Tracer`

***

<a id="api-getworkflowadapter"></a>

### getWorkflowAdapter()

```ts
function getWorkflowAdapter(): WorkflowAdapter | null;
```

#### Returns

[`WorkflowAdapter`](#api-workflowadapter) \| `null`

***

<a id="api-hasnamedllms"></a>

### hasNamedLlms()

```ts
function hasNamedLlms(): boolean;
```

#### Returns

`boolean`

***

<a id="api-initerrorreporting"></a>

### initErrorReporting()

```ts
function initErrorReporting(args): Promise<boolean>;
```

Initialize Sentry error reporting. No-op (returns `false`) unless
`AGENTIC_SENTRY_ENABLED=1` and `AGENTIC_SENTRY_DSN` are both set, and
`@sentry/node` is installed. Tags every event with `surface` and, when
provided, `component` / `mode`.

Never throws: a missing module or a faulting `Sentry.init` leaves reporting
disabled rather than propagating — enabling observability must not turn into
a boot blocker.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`InitErrorReportingArgs`](#api-initerrorreportingargs) |

#### Returns

`Promise`\<`boolean`\>

***

<a id="api-installstructuredlogging"></a>

### installStructuredLogging()

```ts
function installStructuredLogging(args?): Logger;
```

Install structured logging on the log4js root.

Behaves like the Python SDK's `install_structured_logging`:

- Idempotent: re-installing routes the appender through the snapshotted
  original `stdout.write` (stashed under `ORIGINAL_WRITE_SLOT`), so it
  keeps writing to the real stdout instead of recursing through a
  patched write.
- Silences noisy HTTP libraries (see `NOISY_LOGGER_NAMES`) — equivalent
  to Python's `logging.getLogger("httpx").setLevel(WARNING)` block.
- Patches `process.stdout.write` / `process.stderr.write` (unless
  `captureStdio: false`) so stray writes emerge as structured-logging records
  tagged with `source: "stdout"` (INFO) or `source: "stderr"` (WARNING).
  stderr was deliberately *not* mapped to ERROR — Python had to back that
  out because every deprecation warning would have paged on-call.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`InstallStructuredLoggingArgs`](#api-installstructuredloggingargs) |

#### Returns

`Logger`

***

<a id="api-interruptedactivities"></a>

### interruptedActivities()

```ts
function interruptedActivities(stepOrdinal): InterruptedActivity[];
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `stepOrdinal` | `number` |

#### Returns

[`InterruptedActivity`](#api-interruptedactivity)[]

***

<a id="api-isconfiguredmcpsdktoolname"></a>

### isConfiguredMcpSdkToolName()

```ts
function isConfiguredMcpSdkToolName(config, sdkToolName): boolean;
```

Return whether `sdkToolName` belongs to a configured MCP server.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `config` | \{ `servers`: `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\>; \} |
| `config.servers` | `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\> |
| `sdkToolName` | `string` |

#### Returns

`boolean`

***

<a id="api-isllmcredentialrejection"></a>

### isLlmCredentialRejection()

```ts
function isLlmCredentialRejection(error): boolean;
```

True when an LLM provider error is an auth rejection (HTTP 401/403).

Reads only the provider SDK's own status fields — `status` (openai-node /
anthropic-node style), `status_code`, `response.status` /
`response.status_code` (fetch/axios style), and an integer `code` —
walking the `cause` chain because adapters occasionally re-throw. Text is
never matched: the caller maps a positive result onto the wire-level
credential error code, and anything unrecognized returns false so the
failure keeps its existing generic classification.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `error` | `unknown` |

#### Returns

`boolean`

***

<a id="api-isplatformenvvar"></a>

### isPlatformEnvVar()

```ts
function isPlatformEnvVar(name): boolean;
```

Return `true` if `name` matches a platform-owned env var.

Public-API view of the same membership check used by `tenantEnvVars`.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `name` | `string` |

#### Returns

`boolean`

***

<a id="api-isretryableerror"></a>

### isRetryableError()

```ts
function isRetryableError(error): boolean;
```

Classify provider failures before any LLM output has been exposed.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `error` | `unknown` |

#### Returns

`boolean`

***

<a id="api-issessionfinishrequested"></a>

### isSessionFinishRequested()

```ts
function isSessionFinishRequested(): boolean;
```

#### Returns

`boolean`

***

<a id="api-jsonsafemetadata"></a>

### jsonSafeMetadata()

```ts
function jsonSafeMetadata(value, depth?): Record<string, JsonValue> | undefined;
```

Copy provider metadata into JSON-safe values, or undefined.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `value` | `unknown` | `undefined` |
| `depth` | `number` | `0` |

#### Returns

`Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> \| `undefined`

***

<a id="api-loadruntimeagentconfig"></a>

### loadRuntimeAgentConfig()

```ts
function loadRuntimeAgentConfig(configPath?, envVars?): RuntimeAgentConfig;
```

Load and validate `agent.yaml` for runtime use.

Missing files are treated as an empty config so unit tests and ad-hoc
SDK usage can still construct `App`/`TenantRuntime` outside generated
runtime environments. When a file exists, the runtime validates the
fields it owns directly (entrypoint/features) while passing the
application-owned `config` block through as raw data.

`envVars` is the substitution mapping used to resolve `${VAR}` references
at allowlisted YAML paths (currently `mcp.servers.*.url`). Pass the
tenant-owned subset of the process environment — via `tenantEnvVars()` —
never raw `process.env`, so tenant `agent.yaml` cannot dereference platform
secrets. When `envVars` is `undefined` and a `${...}` reference is present
at an allowlisted path, the loader throws so the misconfiguration is
visible instead of falling through to `new URL()` with the literal string.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `configPath?` | `string` |
| `envVars?` | `Record`\<`string`, `string`\> |

#### Returns

[`RuntimeAgentConfig`](#api-runtimeagentconfig)

***

<a id="api-logcachedresult"></a>

### logCachedResult()

```ts
function logCachedResult(
   toolName,
   step,
   prefix?
): void;
```

Log that a cached result is being used (replay scenario).

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `toolName` | `string` | `undefined` |
| `step` | `number` | `undefined` |
| `prefix` | `string` | `"TOOL"` |

#### Returns

`void`

***

<a id="api-logexecutioncallback"></a>

### logExecutionCallback()

```ts
function logExecutionCallback(
   executionId,
   status,
   result?,
   error?,
   suspendReason?,
   prefix?
): void;
```

Log an execution callback (completion/suspension/error).

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `executionId` | `string` | `undefined` |
| `status` | `string` | `undefined` |
| `result` | `unknown` | `null` |
| `error` | `string` \| `null` | `null` |
| `suspendReason` | `string` \| `null` | `null` |
| `prefix` | `string` | `"OE"` |

#### Returns

`void`

***

<a id="api-logexecutionstart"></a>

### logExecutionStart()

```ts
function logExecutionStart(
   executionId,
   inputKeys,
   prefix?
): void;
```

Log the start of an execution.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `executionId` | `string` | `undefined` |
| `inputKeys` | `string`[] | `undefined` |
| `prefix` | `string` | `"OE"` |

#### Returns

`void`

***

<a id="api-logllmmessages"></a>

### logLLMMessages()

```ts
function logLLMMessages(
   messages,
   prefix?,
   countOnly?
): void;
```

Log LLM conversation messages in a consistent format.

#### Parameters

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `messages` | `unknown`[] | `undefined` | List of message dicts or LangChain message objects |
| `prefix` | `string` | `"LLM"` | Log line prefix (e.g., "OE", "LLM", "AER") |
| `countOnly` | `boolean` | `false` | If true, only log message count (for info level) |

#### Returns

`void`

***

<a id="api-logllmresponse"></a>

### logLLMResponse()

```ts
function logLLMResponse(
   result,
   step,
   prefix?
): void;
```

Log an LLM response in a consistent format.

#### Parameters

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `result` | `unknown` | `undefined` | LLM response (dict or AIMessage) |
| `step` | `number` | `undefined` | Step number |
| `prefix` | `string` | `"LLM"` | Log line prefix |

#### Returns

`void`

***

<a id="api-logpolicyblocked"></a>

### logPolicyBlocked()

```ts
function logPolicyBlocked(
   toolName,
   step,
   reason,
   prefix?
): void;
```

Log that a tool call was blocked by policy.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `toolName` | `string` | `undefined` |
| `step` | `number` | `undefined` |
| `reason` | `string` | `undefined` |
| `prefix` | `string` | `"TOOL"` |

#### Returns

`void`

***

<a id="api-logsection"></a>

### logSection()

```ts
function logSection(): void;
```

Log a major section separator (for execution boundaries).

#### Returns

`void`

***

<a id="api-logseparator"></a>

### logSeparator()

```ts
function logSeparator(): void;
```

Log a minor section separator (for individual operations).

#### Returns

`void`

***

<a id="api-logtoolrequest"></a>

### logToolRequest()

```ts
function logToolRequest(
   toolName,
   args,
   step,
   prefix?,
   fieldsToRedact?
): void;
```

Log a tool execution request.

#### Parameters

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `toolName` | `string` | `undefined` | Name of the tool |
| `args` | `Record`\<`string`, `unknown`\> | `undefined` | Tool arguments |
| `step` | `number` | `undefined` | Step number |
| `prefix` | `string` | `"TOOL"` | Log line prefix |
| `fieldsToRedact` | readonly `string`[] | `[]` | Catalog sensitive-field policy. Matching fields omit even their type/length metadata; no values are logged. |

#### Returns

`void`

***

<a id="api-logtoolresult"></a>

### logToolResult()

```ts
function logToolResult(
   toolName,
   step,
   status,
   result?,
   error?,
   durationMs?,
   prefix?
): void;
```

Log a tool execution result.

#### Parameters

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `toolName` | `string` | `undefined` | Name of the tool |
| `step` | `number` | `undefined` | Step number |
| `status` | `string` | `undefined` | Execution status (success, error, suspend, interrupted) |
| `result` | `unknown` | `null` | Tool result |
| `error` | `string` \| `null` | `null` | Error message if failed |
| `durationMs` | `number` | `0` | Execution duration in milliseconds |
| `prefix` | `string` | `"TOOL"` | Log line prefix |

#### Returns

`void`

***

<a id="api-makemcpclientcredentialsauth"></a>

### makeMcpClientCredentialsAuth()

```ts
function makeMcpClientCredentialsAuth(
   serverName,
   config,
   cacheDir?
): OAuthClientProvider;
```

Return a client-credentials OAuth provider for a configured MCP server.

`cacheDir` overrides `mcpOauthCacheDir()` — used by tests and any caller
that needs an isolated cache location instead of the shared `agentengine dev`
cache directory.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `config` | \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \} |
| `config.allowed_tools` | `string`[] \| `null` |
| `config.auth?` | \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \} |
| `config.auth.client_id_env?` | `string` \| `null` |
| `config.auth.client_name?` | `string` \| `null` |
| `config.auth.client_secret_env?` | `string` \| `null` |
| `config.auth.redirect_uri?` | `string` \| `null` |
| `config.auth.scope?` | `string` \| `null` |
| `config.auth.token_env?` | `string` \| `null` |
| `config.auth.token_url?` | `string` \| `null` |
| `config.auth.type?` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `config.headers?` | `Record`\<`string`, `string`\> |
| `config.timeout_seconds?` | `number` |
| `config.transport?` | `"streamable_http"` |
| `config.url?` | `string` |
| `cacheDir?` | `string` |

#### Returns

`OAuthClientProvider`

***

<a id="api-makemcpoauthauth"></a>

### makeMcpOauthAuth()

```ts
function makeMcpOauthAuth(
   serverName,
   config,
   cacheDir?
): OAuthClientProvider;
```

Return a non-interactive OAuth provider for a configured MCP server.

`cacheDir` overrides `mcpOauthCacheDir()` — used by tests and any caller
that needs an isolated cache location instead of the shared `agentengine dev`
cache directory.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `config` | \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \} |
| `config.allowed_tools` | `string`[] \| `null` |
| `config.auth?` | \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \} |
| `config.auth.client_id_env?` | `string` \| `null` |
| `config.auth.client_name?` | `string` \| `null` |
| `config.auth.client_secret_env?` | `string` \| `null` |
| `config.auth.redirect_uri?` | `string` \| `null` |
| `config.auth.scope?` | `string` \| `null` |
| `config.auth.token_env?` | `string` \| `null` |
| `config.auth.token_url?` | `string` \| `null` |
| `config.auth.type?` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `config.headers?` | `Record`\<`string`, `string`\> |
| `config.timeout_seconds?` | `number` |
| `config.transport?` | `"streamable_http"` |
| `config.url?` | `string` |
| `cacheDir?` | `string` |

#### Returns

`OAuthClientProvider`

***

<a id="api-makemcpsdktoolname"></a>

### makeMcpSdkToolName()

```ts
function makeMcpSdkToolName(serverName, toolName): string;
```

Return the server-prefixed SDK name for a remote MCP tool.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `toolName` | `string` |

#### Returns

`string`

***

<a id="api-makemcptoolcallable"></a>

### makeMcpToolCallable()

```ts
function makeMcpToolCallable(binding): (args) => Promise<MCPToolResult>;
```

Create the async callable used to invoke a discovered MCP tool.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `binding` | [`MCPToolBinding`](#api-mcptoolbinding) |

#### Returns

(`args`) => `Promise`\<[`MCPToolResult`](#api-mcptoolresult)\>

***

<a id="api-materializemcpoauthsecretcache"></a>

### materializeMcpOauthSecretCache()

```ts
function materializeMcpOauthSecretCache(__namedParameters?): Promise<number>;
```

Establish the writable runtime cache and decode MCP OAuth secrets into it.

Sets `AGENTIC_MCP_OAUTH_DIR` (if unset) so the reader in `mcp_oauth.ts`
resolves the same directory. Returns the number of secrets materialized.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `__namedParameters` | `MaterializeOptions` |

#### Returns

`Promise`\<`number`\>

***

<a id="api-mcpoauthcachedir"></a>

### mcpOauthCacheDir()

```ts
function mcpOauthCacheDir(): string;
```

Cache directory, resolved at call time — not a module-level constant.

`materializeMcpOauthSecretCache()` (in `mcp_oauth_secret.ts`) sets
`AGENTIC_MCP_OAUTH_DIR` at startup. A frozen import-time const would capture
the value from before that runs whenever this module is imported first (e.g.
via the package barrel), so the reader and writer could resolve to different
directories. Reading the env on each call keeps them in lockstep regardless
of import order.

#### Returns

`string`

***

<a id="api-mcpoauthcachename"></a>

### mcpOauthCacheName()

```ts
function mcpOauthCacheName(serverName, serverUrl): string;
```

Return a collision-free cache basename scoped to alias and endpoint.
agent.yaml imposes no charset on aliases, so uniqueness comes from hashing
the raw alias whenever the readable form would lose information, and from
the endpoint hash that stops the same alias sharing credentials across
different MCP servers.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `serverUrl` | `string` |

#### Returns

`string`

***

<a id="api-mcpservernetworkhosts"></a>

### mcpServerNetworkHosts()

```ts
function mcpServerNetworkHosts(config): string[];
```

Return the outbound hosts needed to reach a configured MCP server.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `config` | \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \} |
| `config.allowed_tools` | `string`[] \| `null` |
| `config.auth` | \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \} |
| `config.auth.client_id_env` | `string` \| `null` |
| `config.auth.client_name` | `string` \| `null` |
| `config.auth.client_secret_env` | `string` \| `null` |
| `config.auth.redirect_uri` | `string` \| `null` |
| `config.auth.scope` | `string` \| `null` |
| `config.auth.token_env` | `string` \| `null` |
| `config.auth.token_url` | `string` \| `null` |
| `config.auth.type` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `config.headers` | `Record`\<`string`, `string`\> |
| `config.timeout_seconds` | `number` |
| `config.transport` | `"streamable_http"` |
| `config.url` | `string` |

#### Returns

`string`[]

***

<a id="api-mergetokenusage"></a>

### mergeTokenUsage()

```ts
function mergeTokenUsage(existing, incoming): LLMTokenUsage | undefined;
```

Merge stream usage field-by-field; incoming non-null fields win.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `existing` | [`LLMTokenUsage`](#api-llmtokenusage) \| `undefined` |
| `incoming` | `unknown` |

#### Returns

[`LLMTokenUsage`](#api-llmtokenusage) \| `undefined`

***

<a id="api-newdurabilityownerid"></a>

### newDurabilityOwnerId()

```ts
function newDurabilityOwnerId(): string;
```

#### Returns

`string`

***

<a id="api-normalizecontent"></a>

### normalizeContent()

```ts
function normalizeContent(content): string;
```

Normalize LLM message content to a string.

LLM content can be:
- A string (normal text)
- A list (multimodal content with text and other parts)
- null/undefined or empty

This handles cases where LangChain's AIMessageChunk.content is a list
of content blocks (e.g., from Gemini multimodal responses) rather than
a simple string.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `content` | `unknown` | The content to normalize (string, list, or null/undefined) |

#### Returns

`string`

A string representation of the content

***

<a id="api-normalizellmpodinvokeresponse"></a>

### normalizeLLMPodInvokeResponse()

```ts
function normalizeLLMPodInvokeResponse(resp): object;
```

Sync `usage` and `result.usage` on a parsed LLMPodInvokeResponse,
mirroring Python's `_sync_usage_with_result` model_validator.

- If `result.usage` is unset and `usage` is present, returns a new
  response whose result carries the top-level usage.
- If `usage` is unset and `result.usage` is present, lifts that
  `result.usage` to the top level.

Returns the response (possibly with a freshly-built `result`/`usage`).

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `resp` | \{ `duration_ms`: `number`; `error?`: `string`; `error_code?`: `string`; `pod_name`: `string`; `result?`: \{ \[`key`: `string`\]: `unknown`; `additional_kwargs?`: `Record`\<`string`, `unknown`\>; `content`: `string`; `id?`: `string`; `metadata?`: `Record`\<`string`, `unknown`\>; `name?`: `string`; `response_metadata?`: `Record`\<`string`, `unknown`\>; `tool_calls?`: `unknown`[]; `usage?`: `Record`\<`string`, `unknown`\>; \}; `status`: `string`; `usage?`: [`LLMTokenUsage`](#api-llmtokenusage); \} | - |
| `resp.duration_ms` | `number` | Execution duration in milliseconds. |
| `resp.error?` | `string` | Error message if failed. |
| `resp.error_code?` | `string` | Machine-readable failure classification (e.g. llm_credential_rejected). |
| `resp.pod_name` | `string` | Hostname/pod name where LLM was executed. |
| `resp.result?` | \{ \[`key`: `string`\]: `unknown`; `additional_kwargs?`: `Record`\<`string`, `unknown`\>; `content`: `string`; `id?`: `string`; `metadata?`: `Record`\<`string`, `unknown`\>; `name?`: `string`; `response_metadata?`: `Record`\<`string`, `unknown`\>; `tool_calls?`: `unknown`[]; `usage?`: `Record`\<`string`, `unknown`\>; \} | LLM result with content, tool_calls, and usage. |
| `resp.result.additional_kwargs?` | `Record`\<`string`, `unknown`\> | - |
| `resp.result.content` | `string` | - |
| `resp.result.id?` | `string` | - |
| `resp.result.metadata?` | `Record`\<`string`, `unknown`\> | - |
| `resp.result.name?` | `string` | - |
| `resp.result.response_metadata?` | `Record`\<`string`, `unknown`\> | - |
| `resp.result.tool_calls?` | `unknown`[] | - |
| `resp.result.usage?` | `Record`\<`string`, `unknown`\> | - |
| `resp.status` | `string` | Execution status: success, error. |
| `resp.usage?` | [`LLMTokenUsage`](#api-llmtokenusage) | Token usage (input_tokens, output_tokens, total_tokens). |

#### Returns

| Name | Type | Description |
| :------ | :------ | :------ |
| `duration_ms` | `number` | Execution duration in milliseconds. |
| `error?` | `string` | Error message if failed. |
| `error_code?` | `string` | Machine-readable failure classification (e.g. llm_credential_rejected). |
| `pod_name` | `string` | Hostname/pod name where LLM was executed. |
| `result?` | `object` | LLM result with content, tool_calls, and usage. |
| `result.additional_kwargs?` | `Record`\<`string`, `unknown`\> | - |
| `result.content` | `string` | - |
| `result.id?` | `string` | - |
| `result.metadata?` | `Record`\<`string`, `unknown`\> | - |
| `result.name?` | `string` | - |
| `result.response_metadata?` | `Record`\<`string`, `unknown`\> | - |
| `result.tool_calls?` | `unknown`[] | - |
| `result.usage?` | `Record`\<`string`, `unknown`\> | - |
| `status` | `string` | Execution status: success, error. |
| `usage?` | [`LLMTokenUsage`](#api-llmtokenusage) | Token usage (input_tokens, output_tokens, total_tokens). |

***

<a id="api-normalizemcptoolresult"></a>

### normalizeMcpToolResult()

```ts
function normalizeMcpToolResult(result): MCPToolResult;
```

Convert an MCP `CallToolResult` into a JSON-safe result, raising on an error result.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `result` | \{ `content?`: `unknown`[]; `isError?`: `boolean`; `structuredContent?`: `unknown`; \} |
| `result.content?` | `unknown`[] |
| `result.isError?` | `boolean` |
| `result.structuredContent?` | `unknown` |

#### Returns

[`MCPToolResult`](#api-mcptoolresult)

***

<a id="api-normalizeoptionalstr"></a>

### normalizeOptionalStr()

```ts
function normalizeOptionalStr(value): string | null;
```

Trim a string tool-metadata value, collapsing blank/non-string to null.

Shared by the framework SDKs' tool-wrapping code so metadata values written
as "" or "  " are treated the same as absent rather
than sent to the OE as a non-empty-looking but meaningless string.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `unknown` |

#### Returns

`string` \| `null`

***

<a id="api-normalizetoolcallargs"></a>

### normalizeToolCallArgs()

```ts
function normalizeToolCallArgs(args): string | null;
```

Normalize streamed tool-call args into the wire-format string payload.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `unknown` |

#### Returns

`string` \| `null`

***

<a id="api-notecheckpointwireworkspaceid"></a>

### noteCheckpointWireWorkspaceId()

```ts
function noteCheckpointWireWorkspaceId(wireWorkspaceId?): void;
```

Remember the wire `workspace_id` from `/execute` for query reads.

Query routes carry no workspace identifier; when `APP_ID` is unset the read
path falls back to the most recently observed wire value.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `wireWorkspaceId?` | `string` \| `null` |

#### Returns

`void`

***

<a id="api-observedactivitypositions"></a>

### observedActivityPositions()

```ts
function observedActivityPositions(stepOrdinal): ActivityPosition[];
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `stepOrdinal` | `number` |

#### Returns

[`ActivityPosition`](#api-activityposition)[]

***

<a id="api-oestreamretrydelayms"></a>

### oeStreamRetryDelayMs()

```ts
function oeStreamRetryDelayMs(retryAfterMs?): number;
```

Honor a server-provided SSE retry_after_ms, capped. Absent means retry immediately.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `retryAfterMs?` | `number` \| `null` |

#### Returns

`number`

***

<a id="api-preallocateactivityordinals"></a>

### preallocateActivityOrdinals()

```ts
function preallocateActivityOrdinals(keys): number[];
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `keys` | readonly `string`[] |

#### Returns

`number`[]

***

<a id="api-preallocatechildoperationordinals"></a>

### preallocateChildOperationOrdinals()

```ts
function preallocateChildOperationOrdinals(boundaries): number[];
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `boundaries` | readonly [`ChildOperationBoundary`](#api-childoperationboundary)[] |

#### Returns

`number`[]

***

<a id="api-projectscopingrequired"></a>

### projectScopingRequired()

```ts
function projectScopingRequired(): boolean;
```

Whether an empty PROJECT_ID must fail closed (REQUIRE_PROJECT_SCOPED_DB).

ECP stamps this flag on managed AER pods (where PROJECT_ID is always injected),
so an empty PROJECT_ID there fails closed rather than silently writing to the
unscoped store. Local CLI dev leaves the flag unset and uses the unscoped name.

#### Returns

`boolean`

***

<a id="api-quotepathsegment"></a>

### quotePathSegment()

```ts
function quotePathSegment(value): string;
```

Percent-encode one path segment; reject separators and `.` / `..`.

Decode repeatedly before quoting so encoded, double-encoded, and mixed
traversal forms cannot survive a later routing decode.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `string` |

#### Returns

`string`

***

<a id="api-recordcurrentmemorymetadata"></a>

### recordCurrentMemoryMetadata()

```ts
function recordCurrentMemoryMetadata(args): void;
```

Attach explicit memory metadata to the current execution step, if one exists.
Mutates the metadata object in-place (matching the Python behaviour).

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`RecordMemoryMetadataArgs`](#api-recordmemorymetadataargs) |

#### Returns

`void`

***

<a id="api-recordinterruptedactivity"></a>

### recordInterruptedActivity()

```ts
function recordInterruptedActivity(command, controlFlow): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `command` | [`ActivityCommand`](#api-activitycommand) |
| `controlFlow` | `unknown` |

#### Returns

`void`

***

<a id="api-recordlatency-1"></a>

### recordLatency()

```ts
function recordLatency<T>(
   operation,
   fn,
   labels?
): Promise<T>;
```

Run `fn` and record its wall-clock duration as a latency metric.
Errors propagate; their duration is still recorded.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `operation` | `string` |
| `fn` | () => `T` \| `Promise`\<`T`\> |
| `labels` | `Record`\<`string`, `string`\> |

#### Returns

`Promise`\<`T`\>

***

<a id="api-recordobservedactivity"></a>

### recordObservedActivity()

```ts
function recordObservedActivity(position): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `position` | [`ActivityPosition`](#api-activityposition) |

#### Returns

`void`

***

<a id="api-recordreconstructedactivityinterrupt"></a>

### recordReconstructedActivityInterrupt()

```ts
function recordReconstructedActivityInterrupt(activityId): number;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `activityId` | `string` |

#### Returns

`number`

***

<a id="api-recordsuspendrequest"></a>

### recordSuspendRequest()

```ts
function recordSuspendRequest(payload): void;
```

Record an author-intended HITL suspend for the current tool call. Called
only by `suspendPayloadToJson`, so the signal's provenance is the tool
author's code, not tool-result data. Reads `storage.getStore()` directly so
a call outside a run is a no-op instead of mutating the frozen fallback.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `payload` | `Record`\<`string`, `unknown`\> |

#### Returns

`void`

***

<a id="api-redactfields"></a>

### redactFields()

```ts
function redactFields(data, fieldsToRedact): Record<string, unknown>;
```

Redact sensitive fields from an arguments record.

Returns a copy with each listed field replaced by "[REDACTED]"; unlisted
fields pass through unchanged, and an empty list returns the input as-is.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `data` | `Record`\<`string`, `unknown`\> |
| `fieldsToRedact` | readonly `string`[] |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-registerbuiltintools"></a>

### registerBuiltinTools()

```ts
function registerBuiltinTools(runtime): void;
```

Register all built-in tool handlers on *runtime* (callable lookup +
metadata for all 8 handlers). Called by `ToolServer.onStartup` when
`features.deep_agent` is on.

If a user `@app.tool()` registered a tool with a reserved built-in name,
this logs a WARNING and overrides it with the built-in — without the warning
the collision was silent and surfaced only when the built-in was invoked.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `runtime` | [`ITenantRuntime`](#api-itenantruntime) |

#### Returns

`void`

***

<a id="api-registerguardrailpolicyengine"></a>

### registerGuardrailPolicyEngine()

```ts
function registerGuardrailPolicyEngine(engine): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `engine` | [`GuardrailPolicyEngine`](#api-guardrailpolicyengine) |

#### Returns

`void`

***

<a id="api-registerinstrumentor"></a>

### registerInstrumentor()

```ts
function registerInstrumentor(fn): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `fn` | [`Instrumentor`](#api-instrumentor) |

#### Returns

`void`

***

<a id="api-registerllm"></a>

### registerLlm()

```ts
function registerLlm(llmId, llm): void;
```

Register an LLM by id. Throws on duplicate id.

Framework SDKs call this from `app.llm()` for every LLM the agent uses.
The unnamed-LLM convenience case is represented by registering under the
sentinel id `"__default__"`; a second unnamed call therefore raises the
same duplicate-id error as a second named call with the same id.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `llmId` | `string` |
| `llm` | `unknown` |

#### Returns

`void`

***

<a id="api-registerllmadapterfactory"></a>

### registerLLMAdapterFactory()

```ts
function registerLLMAdapterFactory(factory): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `factory` | [`LLMAdapterFactory`](#api-llmadapterfactory) |

#### Returns

`void`

***

<a id="api-registerqueryplugin"></a>

### registerQueryPlugin()

```ts
function registerQueryPlugin(plugin): void;
```

Register the framework adapter's `AERQueryPlugin`.

Mirrors Python's `TenantRuntime.register_query_plugin`. The AER's
`/query/sessions` routes return 501 until a plugin is registered.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `plugin` | [`AERQueryPlugin`](#api-aerqueryplugin) |

#### Returns

`void`

***

<a id="api-registerregexguardrailpolicyengine"></a>

### registerRegexGuardrailPolicyEngine()

```ts
function registerRegexGuardrailPolicyEngine(): void;
```

#### Returns

`void`

***

<a id="api-registersuspendhandler"></a>

### registerSuspendHandler()

```ts
function registerSuspendHandler(handler): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `handler` | [`SuspendHandler`](#api-suspendhandler) |

#### Returns

`void`

***

<a id="api-registerworkflowadapter"></a>

### registerWorkflowAdapter()

```ts
function registerWorkflowAdapter(name, version): void;
```

Declare that the materialized graph supports OE durable workflow routing.

Call this or [clearWorkflowAdapter](#api-clearworkflowadapter) on every `getAgent()` materialization
so a later ineligible graph cannot reuse the previous request's identity.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `name` | `string` |
| `version` | `string` |

#### Returns

`void`

***

<a id="api-reportoeownerurlfailure"></a>

### reportOeOwnerUrlFailure()

```ts
function reportOeOwnerUrlFailure(): void;
```

Mark the current execution's owner URL unusable. One-way: after this,
[getCurrentOeOwnerUrl](#api-getcurrentoeownerurl) returns null for the rest of the execution so
repeated emits stop re-paying the pre-attempt timeout against a dead owner.
Reads `storage.getStore()` directly (like [recordSuspendRequest](#api-recordsuspendrequest)) so a
call outside a run is a no-op instead of mutating the frozen fallback.

#### Returns

`void`

***

<a id="api-reportoeresult"></a>

### reportOeResult()

```ts
function reportOeResult(args): Promise<void>;
```

Report an execution result and require OE to acknowledge settlement.

Started as the active span before getCurrentTraceContext() is read below,
so the trace_id/span_id put on the wire reflect this span. Without a span
here, a slow-to-ack OE (or a retried settlement) is invisible: the
surrounding tool-node span just looks slower, with no record of how many
attempts happened.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`ReportOeResultArgs`](#api-reportoeresultargs) |

#### Returns

`Promise`\<`void`\>

***

<a id="api-requestoeapproval"></a>

### requestOeApproval()

```ts
function requestOeApproval(args): Promise<{
  cached_result?: unknown;
  duration_ms?: number | null;
  elicitation?:   | {
     authorization_url: string;
     created: boolean;
     elicitation_id: string;
     message: string;
   }
     | null;
  error?: string | null;
  error_code?: string | null;
  from_cache: boolean;
  guardrail_meta?:   | {
     guardrail_category: string;
     guardrail_id: string;
   }
     | null;
  latest_step_number?: number | null;
  pod_name?: string | null;
  proceed: boolean;
  reason?: string | null;
  result?: unknown;
  retryable: boolean;
  route_to?: string | null;
  status?: string | null;
  tool_api_error?:   | {
     classification: string;
     error_code?: string | null;
     http_status?: number | null;
     provider_type?: string | null;
     reason?: string | null;
     retryable: boolean;
   }
     | null;
}>;
```

Request approval from OE before executing a tool/LLM call.
Any HTTP or network error becomes PolicyDeniedException — fail-safe.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`RequestOeApprovalArgs`](#api-requestoeapprovalargs) |

#### Returns

`Promise`\<\{
  `cached_result?`: `unknown`;
  `duration_ms?`: `number` \| `null`;
  `elicitation?`:   \| \{
     `authorization_url`: `string`;
     `created`: `boolean`;
     `elicitation_id`: `string`;
     `message`: `string`;
   \}
     \| `null`;
  `error?`: `string` \| `null`;
  `error_code?`: `string` \| `null`;
  `from_cache`: `boolean`;
  `guardrail_meta?`:   \| \{
     `guardrail_category`: `string`;
     `guardrail_id`: `string`;
   \}
     \| `null`;
  `latest_step_number?`: `number` \| `null`;
  `pod_name?`: `string` \| `null`;
  `proceed`: `boolean`;
  `reason?`: `string` \| `null`;
  `result?`: `unknown`;
  `retryable`: `boolean`;
  `route_to?`: `string` \| `null`;
  `status?`: `string` \| `null`;
  `tool_api_error?`:   \| \{
     `classification`: `string`;
     `error_code?`: `string` \| `null`;
     `http_status?`: `number` \| `null`;
     `provider_type?`: `string` \| `null`;
     `reason?`: `string` \| `null`;
     `retryable`: `boolean`;
   \}
     \| `null`;
\}\>

***

<a id="api-requestsessionfinish"></a>

### requestSessionFinish()

```ts
function requestSessionFinish(): SessionFinishStatus;
```

Record that the agent considers this session finished. Reads directly off
`storage.getStore()` (not `current()`) so a call outside a run reports
"unavailable" without ever touching the frozen `EMPTY_STORE` fallback.
`wrapper` is null for Tool Pod / Function contexts (see server/tool.ts,
server/function.ts) — only the AER holds the finish latch, so those
contexts must also report "unavailable" rather than a misleading success.

#### Returns

[`SessionFinishStatus`](#api-sessionfinishstatus)

***

<a id="api-resetcheckpointworkspacestate"></a>

### resetCheckpointWorkspaceState()

```ts
function resetCheckpointWorkspaceState(): void;
```

Test helper — reset module state between cases.

#### Returns

`void`

***

<a id="api-resethooks"></a>

### resetHooks()

```ts
function resetHooks(): void;
```

#### Returns

`void`

***

<a id="api-resetllmregistry"></a>

### resetLlmRegistry()

```ts
function resetLlmRegistry(): void;
```

#### Returns

`void`

***

<a id="api-resetstoredbcache"></a>

### resetStoreDbCache()

```ts
function resetStoreDbCache(): void;
```

Clear the memoized resolved store DB names (test seam).

#### Returns

`void`

***

<a id="api-resolvecheckpointworkspaceid"></a>

### resolveCheckpointWorkspaceId()

```ts
function resolveCheckpointWorkspaceId(wireWorkspaceId?): string | null;
```

Resolve the workspace scope used for LangGraph checkpoint thread_ids.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `wireWorkspaceId?` | `string` \| `null` |

#### Returns

`string` \| `null`

***

<a id="api-resolveconfiguredmcptoolbinding"></a>

### resolveConfiguredMcpToolBinding()

```ts
function resolveConfiguredMcpToolBinding(
   config,
   sdkToolName,
   mcpServerName,
   mcpToolName
): MCPToolBinding | null;
```

Map an SDK-visible tool name back to configured MCP call metadata.

AER discovers remote MCP schemas at startup and exposes server-prefixed
names such as `github__search_issues` to the LLM. Tool Pods skip
`tools/list` startup discovery, so call-time execution resolves that SDK
name against `agent.yaml` and calls the original MCP tool name.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `config` | \{ `servers`: `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\>; \} |
| `config.servers` | `Record`\<`string`, \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \}\> |
| `sdkToolName` | `string` |
| `mcpServerName` | `string` |
| `mcpToolName` | `string` |

#### Returns

[`MCPToolBinding`](#api-mcptoolbinding) \| `null`

***

<a id="api-resolveeffectivedb"></a>

### resolveEffectiveDb()

```ts
function resolveEffectiveDb(
   client,
   base,
   projectId,
   opts?
): Promise<string>;
```

Resolve `base` against the live cluster via `listDatabaseNames`.

Empty `projectId` throws when scoping is required (`required`, defaulting to
[projectScopingRequired](#api-projectscopingrequired)), else warns and returns `base`. A listing
failure throws so a transient error cannot create a competing current-name
database beside an existing legacy one. `legacyBases` are previous defaults
whose project-scoped forms, then bare forms, are adopted before a fresh
scoped database is created.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `client` | `ListsDatabaseNames` |
| `base` | `string` |
| `projectId` | `string` |
| `opts` | \{ `label?`: `string`; `legacyBases?`: `string`[]; `required?`: `boolean`; \} |
| `opts.label?` | `string` |
| `opts.legacyBases?` | `string`[] |
| `opts.required?` | `boolean` |

#### Returns

`Promise`\<`string`\>

***

<a id="api-resolveentrypoint"></a>

### resolveEntrypoint()

```ts
function resolveEntrypoint(): ResolvedEntrypoint;
```

Parse `AGENT_ENTRYPOINT` into `{ modulePath, exportName }`.

Exits the process with `EXIT_NO_ENTRYPOINT` if the env var is unset or
malformed — matches Python `_resolve_entrypoint`'s equivalent exit code.
Exported so tests can spawn this in a subprocess and assert on exit
code / stderr.

#### Returns

[`ResolvedEntrypoint`](#api-resolvedentrypoint)

***

<a id="api-resolveimporttarget"></a>

### resolveImportTarget()

```ts
function resolveImportTarget(modulePath, agentRoot?): string;
```

Resolve an AGENT_ENTRYPOINT module path to a value Node's dynamic `import()`
can load.

The platform bakes the `agent.yaml` `entrypoint` verbatim into
AGENT_ENTRYPOINT (same contract as the Python launcher), so the common form
is a dotted, Python-style module path — e.g. `agent_pkg.main` — that maps to
the agent's *compiled* output `<agentRoot>/dist/agent_pkg/main.js`. Node ESM
`import()` treats a dotted string as a bare package specifier and cannot
resolve it, so we translate it here: dots → path separators, under the
compiled `dist/` directory, with a `.js` suffix, returned as a `file://` URL
(the portable form for importing an absolute path across platforms).

`agentRoot` defaults to `process.cwd()`, which at runtime is the agent
package root: the generated Dockerfile sets `WORKDIR` to the install target
and the `CMD` runs the launcher from there. It is injectable for tests.

A value that is already directly importable — a relative path, an absolute
path, or any path-bearing specifier (one that contains a `/`) — is returned
unchanged, so a pre-resolved entrypoint (or a test passing an absolute file
path) still works and is never double-translated. A path-shape signal (not a
file extension) is used deliberately: a dotted module path whose final
segment happens to be `js`/`mjs`/`cjs` (e.g. `agent_pkg.cjs`) must still be
translated, not mistaken for an already-importable file.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `modulePath` | `string` |
| `agentRoot` | `string` |

#### Returns

`string`

***

<a id="api-resolvelistenhost"></a>

### resolveListenHost()

```ts
function resolveListenHost(): string;
```

Resolve the listen host from `APP_HOST`, defaulting to `"0.0.0.0"`.

Matches Python's `_resolve_listen_host`: ECP stamps `APP_HOST` per
executor type at deploy time (`"::"` for vm-mode, `"0.0.0.0"` for
container-mode). Empty string normalises to the default — a shell that
exports `APP_HOST=""` must not bind to `""`.

#### Returns

`string`

***

<a id="api-resolvemcpauth"></a>

### resolveMcpAuth()

```ts
function resolveMcpAuth(serverName, config): Promise<OAuthClientProvider | undefined>;
```

Resolve an MCP SDK OAuth provider for the configured auth mode, if any.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `config` | \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \} |
| `config.allowed_tools` | `string`[] \| `null` |
| `config.auth` | \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \} |
| `config.auth.client_id_env` | `string` \| `null` |
| `config.auth.client_name` | `string` \| `null` |
| `config.auth.client_secret_env` | `string` \| `null` |
| `config.auth.redirect_uri` | `string` \| `null` |
| `config.auth.scope` | `string` \| `null` |
| `config.auth.token_env` | `string` \| `null` |
| `config.auth.token_url` | `string` \| `null` |
| `config.auth.type` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `config.headers` | `Record`\<`string`, `string`\> |
| `config.timeout_seconds` | `number` |
| `config.transport` | `"streamable_http"` |
| `config.url` | `string` |

#### Returns

`Promise`\<`OAuthClientProvider` \| `undefined`\>

***

<a id="api-resolvemcpheaders"></a>

### resolveMcpHeaders()

```ts
function resolveMcpHeaders(serverName, config): Record<string, string>;
```

Resolve headers for a remote MCP server without exposing secret values in config.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `serverName` | `string` |
| `config` | \{ `allowed_tools`: `string`[] \| `null`; `auth`: \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \}; `headers`: `Record`\<`string`, `string`\>; `timeout_seconds`: `number`; `transport`: `"streamable_http"`; `url`: `string`; \} |
| `config.allowed_tools` | `string`[] \| `null` |
| `config.auth` | \{ `client_id_env`: `string` \| `null`; `client_name`: `string` \| `null`; `client_secret_env`: `string` \| `null`; `redirect_uri`: `string` \| `null`; `scope`: `string` \| `null`; `token_env`: `string` \| `null`; `token_url`: `string` \| `null`; `type`: `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"`; \} |
| `config.auth.client_id_env` | `string` \| `null` |
| `config.auth.client_name` | `string` \| `null` |
| `config.auth.client_secret_env` | `string` \| `null` |
| `config.auth.redirect_uri` | `string` \| `null` |
| `config.auth.scope` | `string` \| `null` |
| `config.auth.token_env` | `string` \| `null` |
| `config.auth.token_url` | `string` \| `null` |
| `config.auth.type` | `"none"` \| `"bearer_env"` \| `"oauth"` \| `"client_credentials"` |
| `config.headers` | `Record`\<`string`, `string`\> |
| `config.timeout_seconds` | `number` |
| `config.transport` | `"streamable_http"` |
| `config.url` | `string` |

#### Returns

`Record`\<`string`, `string`\>

***

<a id="api-resolveoeurl"></a>

### resolveOeUrl()

```ts
function resolveOeUrl(requested, env?): string;
```

Returns the OE base URL to use for callbacks during an execution.

The runner's own `OE_URL` wins whenever it is configured. A differing
request value is discarded and logged rather than honoured — quietly
ignoring it would hide an attempted hijack.

When `OE_URL` is unset the requested value is used. That is the local
`agentengine dev` and unit-test path, where no deploy-time environment exists
and the request originates from the developer's own stack; a warning is
emitted so the weaker configuration is visible.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `requested` | `string` | `undefined` |
| `env` | `ProcessEnv` | `process.env` |

#### Returns

`string`

***

<a id="api-resolveprojectscopeddb"></a>

### resolveProjectScopedDb()

```ts
function resolveProjectScopedDb(
   base,
   projectId,
   existing,
   legacyBases?
): string;
```

Derive the effective database name for `base` scoped to `projectId`.

1. empty `projectId` -> return `base` unchanged.
2. `base` already ends with `_{projectId}` -> return as-is (idempotent).
3. `scoped = base + "_" + projectId`:
   - `scoped` exists on the cluster -> use it.
   - else an existing scoped `legacyBases` candidate -> use it.
   - else an existing unscoped `legacyBases` candidate -> use it.
   - else -> use `scoped` (fresh deployment).

Unscoped fallback is limited to known platform defaults. The current base and arbitrary names are never auto-adopted.

#### Parameters

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `base` | `string` | `undefined` |
| `projectId` | `string` | `undefined` |
| `existing` | `string`[] | `undefined` |
| `legacyBases` | `string`[] | `[]` |

#### Returns

`string`

***

<a id="api-resolvestoredbname"></a>

### resolveStoreDbName()

```ts
function resolveStoreDbName(client, base?): Promise<string>;
```

Return the per-project-scoped store DB name, resolved once and cached.

Applies the same resolution the OE uses ([resolveEffectiveDb](#api-resolveeffectivedb)) against
the live cluster, so the AER/SDK writers converge on the same database the OE
reads. `base` defaults to [getStoreDbName](#api-getstoredbname). Explicit values are
returned exactly and skip discovery.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `client` | `MongoClient` |
| `base?` | `string` |

#### Returns

`Promise`\<`string`\>

***

<a id="api-runinstrumentor"></a>

### runInstrumentor()

```ts
function runInstrumentor(): void;
```

#### Returns

`void`

***

<a id="api-runlauncher"></a>

### runLauncher()

```ts
function runLauncher(): Promise<void>;
```

Entry point invoked when this file is run directly:

  `node /app/node_modules/@mongodb-js/agent-engine-runner-shared/dist/launcher.js`

Exits the process with the exit code matching the failure category
(see `EXIT_IMPORT_ERROR`/`EXIT_NO_ENTRYPOINT`/`EXIT_STARTUP_CRASH` above)
on any error path; resolves normally after the user's target function or
`.run()` returns.

#### Returns

`Promise`\<`void`\>

***

<a id="api-runserialactivity"></a>

### runSerialActivity()

```ts
function runSerialActivity<T>(args): Promise<T>;
```

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `activityOrdinal`: `number`; `attempt?`: [`AttemptContext`](#api-attemptcontext); `client`: `ActivityRuntimeClient`; `exclusive?`: `boolean`; `execute`: (`context`) => `T` \| `Promise`\<`T`\>; `kind`: [`ActivityKind`](#api-activitykind); `name`: `string`; `onActivityResolved?`: [`ActivityResolvedHook`](#api-activityresolvedhook)\<`T`\>; `semanticInput`: `unknown`; `stepOrdinal?`: `number`; \} | - |
| `args.activityOrdinal` | `number` | - |
| `args.attempt?` | [`AttemptContext`](#api-attemptcontext) | - |
| `args.client` | `ActivityRuntimeClient` | - |
| `args.exclusive?` | `boolean` | False only when the framework has assigned a stable activity identity. |
| `args.execute` | (`context`) => `T` \| `Promise`\<`T`\> | - |
| `args.kind` | [`ActivityKind`](#api-activitykind) | - |
| `args.name` | `string` | - |
| `args.onActivityResolved?` | [`ActivityResolvedHook`](#api-activityresolvedhook)\<`T`\> | - |
| `args.semanticInput` | `unknown` | - |
| `args.stepOrdinal?` | `number` | - |

#### Returns

`Promise`\<`T`\>

***

<a id="api-runstreamingactivity"></a>

### runStreamingActivity()

```ts
function runStreamingActivity<T>(args): AsyncGenerator<T>;
```

Run one generator-shaped durable activity.

Replay expands a recorded result back into stream items without executing
the effect. A fresh dispatch yields live items immediately, then folds them
into the single terminal result recorded by OE. Closing the stream early
terminal-fails the dispatch instead of leaving it in flight.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `activityOrdinal`: `number`; `attempt?`: [`AttemptContext`](#api-attemptcontext); `client`: `ActivityRuntimeClient`; `exclusive?`: `boolean`; `execute`: () => `StreamingItems`\<`T`\>; `fold`: (`items`) => `unknown`; `kind`: [`ActivityKind`](#api-activitykind); `name`: `string`; `onActivityResolved?`: [`ActivityResolvedHook`](#api-activityresolvedhook)\<`unknown`\>; `replay`: (`result`) => `StreamingItems`\<`T`\>; `semanticInput`: `unknown`; `stepOrdinal?`: `number`; \} | - |
| `args.activityOrdinal` | `number` | - |
| `args.attempt?` | [`AttemptContext`](#api-attemptcontext) | - |
| `args.client` | `ActivityRuntimeClient` | - |
| `args.exclusive?` | `boolean` | False when the caller has assigned a stable activity identity. |
| `args.execute` | () => `StreamingItems`\<`T`\> | - |
| `args.fold` | (`items`) => `unknown` | - |
| `args.kind` | [`ActivityKind`](#api-activitykind) | - |
| `args.name` | `string` | - |
| `args.onActivityResolved?` | [`ActivityResolvedHook`](#api-activityresolvedhook)\<`unknown`\> | - |
| `args.replay` | (`result`) => `StreamingItems`\<`T`\> | - |
| `args.semanticInput` | `unknown` | - |
| `args.stepOrdinal?` | `number` | - |

#### Returns

`AsyncGenerator`\<`T`\>

***

<a id="api-runwithattemptcontext"></a>

### runWithAttemptContext()

```ts
function runWithAttemptContext<T>(attempt, fn): T;
```

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `attempt` | [`AttemptContext`](#api-attemptcontext) |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-runwithcallabortsignal"></a>

### runWithCallAbortSignal()

```ts
function runWithCallAbortSignal<T>(signal, fn): T;
```

Run one callback-routed tool body with its per-call abort signal attached.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `signal` | `AbortSignal` |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-runwithcustomerorigin"></a>

### runWithCustomerOrigin()

```ts
function runWithCustomerOrigin<T>(fn): T;
```

Mark the dynamic extent of customer agent/tool code for log attribution.
Nested scopes are a no-op. Missing origin is unclassified, not proven
platform-authored.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-runwithexecutioncontext"></a>

### runWithExecutionContext()

```ts
function runWithExecutionContext<T>(args, fn): T;
```

Run `fn` with an isolated execution context frame.

`storage.run` creates a child frame that never mutates the parent, so cleanup
is automatic even if `fn` throws or spawns work via setTimeout/Promise. This
is the equivalent of Python's `set_execution_context(...)` followed by a
try/finally `clear_execution_context(...)`.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`SetExecutionContextArgs`](#api-setexecutioncontextargs) |
| `fn` | () => `T` |

#### Returns

`T`

#### Example

```ts
await runWithExecutionContext({ executionId, wrapper, oeUrl, userId }, async () => {
    await runAgentLogic()  // getCurrentUserId() works anywhere in this call chain
  })
```

***

<a id="api-runwithoperationpathresolver"></a>

### runWithOperationPathResolver()

```ts
function runWithOperationPathResolver<T>(resolver, fn): T;
```

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `resolver` | [`OperationPathResolver`](#api-operationpathresolver) |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-runwithsuspendrequestcontext"></a>

### runWithSuspendRequestContext()

```ts
function runWithSuspendRequestContext<T>(fn): T;
```

Run one in-process tool call with an isolated suspend marker.

#### Type Parameters

| Type Parameter |
| :------ |
| `T` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `fn` | () => `T` |

#### Returns

`T`

***

<a id="api-scrubcredentials"></a>

### scrubCredentials()

```ts
function scrubCredentials(message): string;
```

Mask the userinfo of any `mongodb://` / `mongodb+srv://` URI embedded in a
credential-bearing error message (driver parse/connect errors echo the raw
connection string).

The userinfo run is GREEDY up to the LAST '@' that a host-shaped token
follows, so passwords containing an unescaped '@', space, or '/' are still
fully masked — the old `://[^@\s]+@` pattern stopped at the first '@' and
could not cross whitespace, leaking the password tail (or the whole
userinfo) to the centralized log sink. The run is tempered so it never
crosses into a second URI's scheme; when message text after the URI
contains its own '@' the mask may extend to it — over-redaction is the
fail-closed direction, a leak is not recoverable. Keep in sync with
`scrub_credentials` in Python's `agent_engine_runner_shared/tracing/setup.py`.

Accepts `unknown` and coerces: error paths hand this whatever a rejection
carried (string throws, objects without a string `message`, undefined) —
it must never throw itself, or a degraded trace store turns into a
startup failure.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `message` | `unknown` |

#### Returns

`string`

***

<a id="api-serializeinvokellmrequestarguments"></a>

### serializeInvokeLLMRequestArguments()

```ts
function serializeInvokeLLMRequestArguments(args): Record<string, unknown>;
```

Serialize InvokeLLMRequestArguments to the snake_case wire shape, the
equivalent of Python's `model_dump(by_alias=True)`:
 - `stop_sequences` → `stop` (Pydantic `serialization_alias="stop"`), and
 - each Message is dumped camel→snake via `serializeMessage`.

Use this when sending to a Python consumer (OE / Tool Pod) that expects the
snake_case wire fields.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `llm_id`: `string`; `messages`: [`Message`](#api-message)[]; `model`: `string`; `options?`: [`LLMInvocationOptions`](#api-llminvocationoptions); `stop_sequences?`: `string`[]; `stream`: `boolean`; `tool_choice?`: [`JsonValue`](#api-jsonvalue); `tools?`: [`LLMToolSchema`](#api-llmtoolschema)[]; \} | - |
| `args.llm_id` | `string` | Stable identifier for the LLM instance registered via `app.llm({ llmId })`; the tool pod resolves this id against its named-LLM registry. Unnamed `app.llm()` calls register under the sentinel id `"__default__"`. Defaults to `"__default__"` for backward compatibility with callers that omit this field. |
| `args.messages` | [`Message`](#api-message)[] | Conversation in sdk-core Message wire format. |
| `args.model` | `string` | Model name (e.g. 'gpt-5.4-mini', 'gemini-3-flash-preview'). |
| `args.options?` | [`LLMInvocationOptions`](#api-llminvocationoptions) | Explicit provider/model invocation options (e.g. max_tokens). |
| `args.stop_sequences?` | `string`[] | Stop sequences forwarded to the underlying LLM provider. |
| `args.stream` | `boolean` | Whether OE should approve and route a real streaming invoke_llm call. |
| `args.tool_choice?` | [`JsonValue`](#api-jsonvalue) | Forced tool selection forwarded to the tool pod's bind_tools call (e.g. a function name from withStructuredOutput). LangChain translates the value against the bound tools. |
| `args.tools?` | [`LLMToolSchema`](#api-llmtoolschema)[] | Serialized tool schemas for bind_tools. |

#### Returns

`Record`\<`string`, `unknown`\>

***

<a id="api-setactivityreconstructionids"></a>

### setActivityReconstructionIds()

```ts
function setActivityReconstructionIds(activityIds): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `activityIds` | readonly `string`[] |

#### Returns

`void`

***

<a id="api-setpendingchildoperationbatch"></a>

### setPendingChildOperationBatch()

```ts
function setPendingChildOperationBatch(boundaries): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `boundaries` | readonly [`ChildOperationBoundary`](#api-childoperationboundary)[] |

#### Returns

`void`

***

<a id="api-setterminationlogpathfortest"></a>

### setTerminationLogPathForTest()

```ts
function setTerminationLogPathForTest(path): void;
```

Test-only override, mirroring `setHomeDirForTest` in error_reporting.ts.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `path` | `string` |

#### Returns

`void`

***

<a id="api-setuplogging"></a>

### setupLogging()

```ts
function setupLogging(args?): Logger;
```

Configure the root logger. Mirrors `agent_engine_runner_shared.utils.setup_logging`.

When `STRUCTURED_LOGGING=true` is set, delegates to
`installStructuredLogging` so all output emerges as single-line JSON
matching the agent-log contract. On-disk logging is not part of the
production model — Fluent Bit ships container stdout to S3, so a
duplicate copy adds no value. The exception is local dev: when
`AGENTIC_DEV_MODES` is set (dev-up compose stacks only), a rotating file
sink is attached alongside the console/structured output.

Unlike Python's `setup_logging` we don't need to manually drop existing
handlers before re-adding the console handler: `log4js.configure` fully
replaces the prior `appenders`/`categories` config on each call, so a
repeat call never accumulates duplicate writers. There's also no
stdout/stderr handler-close hazard to guard against — log4js's `stdout`
appender owns its own write path.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`SetupLoggingArgs`](#api-setuploggingargs) |

#### Returns

`Logger`

***

<a id="api-setuptracing"></a>

### setupTracing()

```ts
function setupTracing(args?): Promise<void>;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | [`SetupTracingArgs`](#api-setuptracingargs) |

#### Returns

`Promise`\<`void`\>

***

<a id="api-shellexecute"></a>

### shellExecute()

```ts
function shellExecute(args): Promise<Record<string, unknown>>;
```

Run a shell command and capture its output. Per
`SandboxBackendProtocol.execute`, *timeout* is `null` = "use the backend
default" (SHELL_DEFAULT_TIMEOUT_SECONDS), not "no timeout".

Threat model: `shell: true` is intentional — agents need pipes, redirects,
globs. The sandbox is the Tool Pod itself.

Per-session isolation gap: unlike the filesystem handlers (which enforce the
workspace boundary via `resolvePath()` + `isWithinWorkspace()` realpath
checks), `cwd` here is only the spawned shell's default working directory —
the process is not OS-confined, so `..`/absolute paths can reach sibling
sessions or arbitrary pod paths. Treat as session-shared until the
bubblewrap sandbox lands. There is no upstream approval gate on the command
string; treat the content as agent-authored. This means shell_execute does
NOT match the per-session filesystem isolation the fs handlers enforce —
an accepted, tracked risk, not local path validation.

Output handling: stdout/stderr are drained as they stream. Each stream stops
*appending* once SHELL_OUTPUT_MAX_BYTES is captured but the stream is never
paused, so the child never blocks on a full pipe — a runaway `yes` runs to
its timeout without OOMing the pod. Partial output is preserved on timeout.
Framework errors (e.g. missing `/bin/sh`) return
`exit_code = EXIT_CODE_FRAMEWORK_ERROR`, distinct from the timeout sentinel.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | `Record`\<`string`, `unknown`\> |

#### Returns

`Promise`\<`Record`\<`string`, `unknown`\>\>

***

<a id="api-shutdowntracing"></a>

### shutdownTracing()

```ts
function shutdownTracing(): Promise<void>;
```

#### Returns

`Promise`\<`void`\>

***

<a id="api-snapshotllmregistry"></a>

### snapshotLlmRegistry()

```ts
function snapshotLlmRegistry(): Map<string, unknown>;
```

Return a shallow copy of the current registry.

#### Returns

`Map`\<`string`, `unknown`\>

***

<a id="api-stripthinking"></a>

### stripThinking()

```ts
function stripThinking(text): string;
```

Remove `<think>...</think>` blocks and unclosed `<think>` tails.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `text` | `string` |

#### Returns

`string`

***

<a id="api-subprocessoutputtail"></a>

### subprocessOutputTail()

```ts
function subprocessOutputTail(exc): string;
```

The trailing `MAX_OUTPUT_TAIL_CHARS` of a subprocess's stdout+stderr.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `exc` | [`SubprocessFailure`](#api-subprocessfailure) |

#### Returns

`string`

***

<a id="api-summarizesubprocessfailure"></a>

### summarizeSubprocessFailure()

```ts
function summarizeSubprocessFailure(exc): string;
```

A one-line summary of a failed subprocess: the command label plus the most
meaningful line of its output, or the exit code when no output stands out.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `exc` | [`SubprocessFailure`](#api-subprocessfailure) |

#### Returns

`string`

***

<a id="api-suspendpayloadtojson"></a>

### suspendPayloadToJson()

```ts
function suspendPayloadToJson(payload): string;
```

Serialize a SuspendPayload to the suspend wire marker, and record an
out-of-band suspend request on the current execution frame.

The Tool Pod honors suspend from that recorded signal — set only here, in
the tool author's own code — not by sniffing tool-result content, so
untrusted data a tool relays can no longer forge a HITL suspend. The marker
string is still returned unchanged for wire/replay compatibility.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `payload` | \{ `suspend_context`: `Record`\<`string`, `unknown`\>; `suspend_reason`: `string`; \} | - |
| `payload.suspend_context` | `Record`\<`string`, `unknown`\> | Arbitrary context the human reviewer needs to make a decision. |
| `payload.suspend_reason` | `string` | Why the agent is suspending (e.g. 'awaiting_human_review'). |

#### Returns

`string`

***

<a id="api-tenantenvvars"></a>

### tenantEnvVars()

```ts
function tenantEnvVars(source?): Record<string, string>;
```

Return the tenant-owned subset of environment variables.

`source` defaults to `process.env`. Names in `PLATFORM_ENV_VARS` or
matching any prefix in `PLATFORM_ENV_VAR_PREFIXES` are excluded, so the
result is safe to pass as the substitution mapping to
`loadRuntimeAgentConfig({ envVars: ... })`.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `source?` | `Record`\<`string`, `string` \| `undefined`\> |

#### Returns

`Record`\<`string`, `string`\>

***

<a id="api-toolactivitykey"></a>

### toolActivityKey()

```ts
function toolActivityKey(toolCallId): string;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `toolCallId` | `string` |

#### Returns

`string`

***

<a id="api-toolredactfields"></a>

### toolRedactFields()

```ts
function toolRedactFields(toolDefinitions, toolName): readonly string[];
```

Extract a tool's `redact_fields` policy from its registered definition.
The definition is an opaque metadata record (framework-SDK populated), so
non-string-array values degrade to no redaction rather than throwing.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `toolDefinitions` | `Record`\<`string`, `Record`\<`string`, `unknown`\>\> |
| `toolName` | `string` \| `undefined` |

#### Returns

readonly `string`[]

***

<a id="api-tracingstatus"></a>

### tracingStatus()

```ts
function tracingStatus(): object;
```

Snapshot of the database trace-store exporter for /health. Mirrors Python's
`tracing_status`: `"attached"` (MongoDB span processor live), `"degraded"`
(store URI configured but unreachable at startup, retrying), or `"disabled"`
(no store URI configured, not a degradation).

#### Returns

`object`

| Name | Type |
| :------ | :------ |
| `database_exporter` | `string` |

***

<a id="api-unwrapactivityoutcome"></a>

### unwrapActivityOutcome()

```ts
function unwrapActivityOutcome(outcome): unknown;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `outcome` | [`ActivityOutcome`](#api-activityoutcome) |

#### Returns

`unknown`

***

<a id="api-validatedurablememoryidentity"></a>

### validateDurableMemoryIdentity()

```ts
function validateDurableMemoryIdentity(identity, userId): void;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `identity` | [`WorkflowIdentity`](#api-workflowidentity) \| `undefined` |
| `userId` | `string` \| `null` \| `undefined` |

#### Returns

`void`

***

<a id="api-valuetojson"></a>

### valueToJson()

```ts
function valueToJson(value): unknown;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `Value` \| `undefined` |

#### Returns

`unknown`

***

<a id="api-withexecutionsignal"></a>

### withExecutionSignal()

```ts
function withExecutionSignal(callSignal): AbortSignal;
```

Combine a per-call `AbortSignal` (e.g. a request/read timeout) with the
current execution's abort signal, if one is set. The returned signal aborts
when *either* fires, so an execution-wide timeout cancels the in-flight OE/LLM
fetch instead of leaving it to run until its own deadline. Returns
`callSignal` unchanged when there is no execution signal (e.g. a Tool Pod
call outside an AER execution). Mirrors the effect of Python's
`asyncio.wait_for` cancelling in-flight I/O on timeout.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `callSignal` | `AbortSignal` |

#### Returns

`AbortSignal`

***

<a id="api-withmetrics"></a>

### withMetrics()

```ts
function withMetrics<A, R>(
   operation,
   fn,
   opts?
): (...args) => Promise<R>;
```

Wrap a function to record latency (and optionally errors).

Replacement for the Python `@with_metrics(...)` decorator. The returned
function preserves the original signature and awaits any returned promise
so async/sync code can use the same wrapper.

#### Type Parameters

| Type Parameter |
| :------ |
| `A` *extends* `unknown`[] |
| `R` |

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `operation` | `string` |
| `fn` | (...`args`) => `R` \| `Promise`\<`R`\> |
| `opts` | \{ `recordErrors?`: `boolean`; \} |
| `opts.recordErrors?` | `boolean` |

#### Returns

(...`args`) => `Promise`\<`R`\>

***

<a id="api-writeterminationmessage"></a>

### writeTerminationMessage()

```ts
function writeTerminationMessage(summary, err?): void;
```

Record a bounded, redacted summary of a fatal startup error to the
container's termination message before the process exits, so the real
cause of the crash survives past this process's own stdout into
`ContainerStatus.LastTerminationState.Terminated.Message` — the field the
platform's crash diagnostics read, and from there into the
customer-facing deploy timeline.

This is customer code (container mode), so unlike the platform's own
components the design intentionally keeps the full exception name,
message, and stack — redacted, not summarized away — since that detail is
what the customer needs to fix their own agent. Best-effort: if the write
fails, nothing is lost beyond what
`TerminationMessagePolicy: FallbackToLogsOnError` already provides.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `summary` | `string` |
| `err?` | `Error` |

#### Returns

`void`
