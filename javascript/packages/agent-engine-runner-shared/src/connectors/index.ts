/**
 * Compiled OpenAPI connector contract and authoring materialization.
 *
 * This module is the TypeScript counterpart of
 * `agent_engine_runner_shared/connectors/`.
 */

export {
  AuthSchema,
  BasicAuthSchema,
  ConnectorDefinitionError,
  CredentialSchema,
  OperationSchema,
  ParametersSchema,
  SourceSchema,
  ToolDefinitionsSchema,
  authHeaderNames,
  isRecord,
  loadToolDefs,
  toolSchemas,
  validateConnectorName,
  validateSegment,
  validateUrl,
  type Auth,
  type BasicAuth,
  type Credential,
  type Operation,
  type Parameters,
  type Source,
  type ToolDefinitions,
} from "./definitions.js";
export {
  BundledTool,
  ConnectorBundle,
  ConnectorBundleError,
  DEFAULT_CONNECTOR_TIMEOUT_SECONDS,
  type BundledConnector,
  type ConnectorRegistrationMetadata,
  type ConnectorToolSchema,
} from "./bundle.js";
export {
  MaterializeError,
  materializeConnectors,
  resolveRuntimeBundle,
} from "./materialize.js";
export {
  ConnectorExecutor,
  ConnectorHTTPError,
  DEFAULT_EXECUTION_TIMEOUT_SECONDS,
  MAX_ERROR_ENVELOPE_BYTES,
  MAX_REDIRECTS,
  MAX_RESPONSE_BYTES,
  MAX_RETRIES,
  USER_AGENT,
  type ConnectorExecutorOptions,
  type ConnectorTransport,
  type ConnectorTransportRequest,
  type ConnectorTransportResponse,
} from "./executor.js";
export {
  ConnectorToolRegistration,
  ConnectorToolRuntime,
  registerConnectorTools,
  withConnectorEnvironment,
  type ConnectorRuntimeTarget,
} from "./runtime.js";
