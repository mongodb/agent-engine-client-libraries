/**
 * Compiled OpenAPI connector contract and authoring materialization.
 *
 * This module is the TypeScript counterpart of
 * `agent_engine_runner_shared/connectors/`.
 */

export {
  AuthSchema,
  ConnectorDefinitionError,
  CredentialSchema,
  OperationSchema,
  ParametersSchema,
  SourceSchema,
  ToolDefinitionsSchema,
  authCredentialBindings,
  authHeaderName,
  authHeaderNames,
  isRecord,
  loadToolDefs,
  toolSchemas,
  validateConnectorName,
  validateSegment,
  validateUrl,
  type Auth,
  type AuthType,
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
