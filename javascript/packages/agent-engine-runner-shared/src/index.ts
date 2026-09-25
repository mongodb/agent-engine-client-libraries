// Copyright 2026 MongoDB, Inc.
// SPDX-License-Identifier: Apache-2.0

// agent-engine-runner-shared public API barrel

export * from "./utils.js";
export {
  getStoreDbName,
  resolveStoreDbName,
  resetStoreDbCache,
} from "./db_config.js";
export {
  resolveProjectScopedDb,
  resolveEffectiveDb,
  projectScopingRequired,
} from "./db_naming.js";
export { getLogger, setupLogging, type SetupLoggingArgs } from "./logger.js";
export type { Logger } from "log4js";
export {
  AERServer,
  ToolServer,
  LLMRegistryLoadError,
  ToolFunctionRunner,
  ToolFunctionRequestSchema,
  buildResultPayload,
  DEFAULT_PORTS,
  type ToolFunctionRequest,
  type ITenantRuntime,
  type GraphBuilderLike,
  type ServerToolFn,
  type AERQueryPlugin,
} from "./server/index.js";
export { chunkTypes } from "./server/index.js";

export * from "./models.js";

export * from "./guardrails_evaluator/index.js";

export * from "./agent_config.js";

export * from "./mcp_tools.js";

export {
  DEFAULT_MCP_OAUTH_CLIENT_NAME,
  DEFAULT_MCP_OAUTH_REDIRECT_URI,
  mcpOauthCacheDir,
  mcpOauthCacheName,
  makeMcpOauthAuth,
  makeMcpClientCredentialsAuth,
} from "./mcp_oauth.js";

export { materializeMcpOauthSecretCache } from "./mcp_oauth_secret.js";

export * from "./context.js";

export {
  BUILTIN_TOOL_NAMES,
  registerBuiltinTools,
  WORKSPACE_DIR,
  filesystemLs,
  filesystemRead,
  filesystemWrite,
  filesystemEdit,
  filesystemGlob,
  filesystemGrep,
  filesystemDownload,
  shellExecute,
} from "./toolpod_handlers.js";

export * from "./hooks.js";
export * from "./workflow/index.js";
export { fetchPlatform, getFetchOptionsWithTLS } from "./tls_client.js";
export { quotePathSegment } from "./http_path.js";
export { resolveOeUrl } from "./server/oe_url.js";

export {
  getCheckpointWorkspaceId,
  noteCheckpointWireWorkspaceId,
  resolveCheckpointWorkspaceId,
  resetCheckpointWorkspaceState,
} from "./checkpoint_workspace.js";

export {
  TenantRuntime,
  resolveListenHost,
  type TenantRuntimeOptions,
  type RegisterAndRunOptions,
} from "./runtime.js";

export {
  resolveEntrypoint,
  resolveImportTarget,
  runLauncher,
  boundText,
  writeTerminationMessage,
  setTerminationLogPathForTest,
  EXIT_IMPORT_ERROR,
  EXIT_NO_ENTRYPOINT,
  EXIT_STARTUP_CRASH,
  type ResolvedEntrypoint,
} from "./launcher.js";

export {
  PolicyDeniedException,
  TerminalExecutionError,
  ToolCallTimeoutError,
  ToolExecutionError,
  ExternalAPICallError,
  LLMInvocationError,
  OperationalStepAllocator,
  SecureToolWrapper,
  createSecureToolFunction,
  requestOeApproval,
  reportOeResult,
  extractUsage,
  extractPodUsage,
  type RequestOeApprovalArgs,
  type ReportOeResultArgs,
  type ExtractedUsage,
  type ToolResponseFormat,
  type OperationalStepSource,
  CALL_INTERRUPTED_ARTIFACT_KEY,
} from "./secure_wrapper.js";
export { SecureLLMProxy } from "./secure_llm_proxy.js";
export {
  initErrorReporting,
  captureException,
  captureMessage,
  flush as flushErrorReporting,
  summarizeSubprocessFailure,
  subprocessOutputTail,
  type InitErrorReportingArgs,
  type CaptureExceptionArgs,
  type SubprocessFailure,
} from "./error_reporting.js";
export { emit, emitStep } from "./progress.js";

// `metrics.ts` defines `logToolResult`/`logToolCall`/etc which collide with
// the human-readable versions in `utils.ts`. Export the metrics surface
// explicitly to keep the API surface clear: `Metrics`, `recordLatency`,
// `withMetrics` are the canonical names; the structured `log*` helpers stay
// internal (imported directly from `agent-engine-runner-shared/metrics`).
export {
  Metrics,
  recordLatency,
  withMetrics,
  type LatencyStatsJson,
} from "./metrics.js";
export * from "./node_logger.js";
export * from "./structured_logging.js";
export * from "./tracing/index.js";
export * from "./span_names.js";
export { AppBoundRuntime, AppBoundCrudClient } from "./memory_appbound.js";
export { ToolAPIErrorSchema, type ToolAPIError } from "./tool_api_error.js";
