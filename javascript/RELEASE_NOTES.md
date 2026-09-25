# Release Notes — JavaScript Client Libraries

Release notes for the JavaScript SDK packages. Per-package historical changelogs:

- `packages/agent-engine-runner-shared`: [CHANGELOG.md](packages/agent-engine-runner-shared/CHANGELOG.md)
- `packages/agent-engine-sdk`: [CHANGELOG.md](packages/agent-engine-sdk/CHANGELOG.md)
- `packages/agent-engine-sdk-langgraph`: [CHANGELOG.md](packages/agent-engine-sdk-langgraph/CHANGELOG.md)

<!-- New entries go at the top. -->

## v0.1.111

- **Updated default gateway domain**: The JavaScript SDK now connects to `agentengine.mongodb.com` by default, replacing the previous domain, so no configuration change is needed for users on the latest version.
- **Improved TypeScript API reference**: API documentation now renders parameters and properties as tables with code-block signatures, includes a table of contents, and is co-located with each package at `<package>/docs/api.md` for easier navigation.
- **Durable workflow recovery across framework and model upgrades**: Replacement workflow attempts can now recover successfully after a model or framework upgrade by comparing only application-controlled inputs and state, rather than framework-generated IDs and metadata, so workflows are no longer incorrectly rejected as nondeterministic when internal identifiers change.


## v0.1.110

- Rename Agentic-era HTTP header contracts to Agent Engine names


## v0.1.109-alpha

- **Breaking: Environment variable names updated for observability configuration**: The four runner-shared observability environment variables are now `AGENT_ENGINE_ENVIRONMENT`, `AGENT_ENGINE_MAX_ACTIVE_SERIES`, `AGENT_ENGINE_PROJECT_ID`, and `AGENT_ENGINE_TENANT_ID`. Update any configuration that sets the previous names before upgrading, as the old names are no longer read.
- **Rebranding to Atlas Agent Engine**: User-facing labels, docstrings, and the MCP client name now use the `agent-engine-runner` name to reflect the Atlas Agent Engine product name. Update any tooling that matches the previous MCP client name.


## v0.1.108-alpha

- **Breaking: `LLMRegistryLoadError` replaces `KeyError` on entrypoint failures**: If your agent code catches `KeyError` around LLM calls to handle a failed entrypoint load, update those catch blocks to handle `LLMRegistryLoadError` instead, which is now exported from the package entry point.
- **Clearer LLM registry errors when agent startup fails**: When the agent entrypoint fails to load (for example, due to a missing credential or transient database error), LLM calls now report the actual cause of the failure with sensitive values redacted, instead of a misleading message about missing `app.llm()` calls.
- **Accurate error classification for timed-out tool calls**: Tool calls that are rejected because a prior invocation already timed out no longer appear as policy denials; they now correctly surface as execution errors, so you can distinguish a real governance block from a timeout.
- **Credential rejection errors surfaced at invoke time**: When an LLM provider or external tool rejects a configured API key or secret with a 401/403, the runner now reports a machine-readable credential error code instead of a generic invocation failure, enabling clearer guidance to fix the affected secret.
- **Memory recall traces now include the effective query**: The query actually sent to the memory engine during semantic, taxonomic, and episodic recall is now recorded in memory-event metadata (`query` field), so traces show the real search term instead of an empty value.
- **On-disk log files written in TypeScript dev mode**: When running with `agentic dev`, the TypeScript runner now writes `<app>-aer.log` and `<app>-tool.log` files to `LOG_DIR`, filling a gap where TypeScript agents and structured-logging mode previously produced no on-disk log copy.
- **README and license included in published packages**: All JavaScript SDK packages now bundle their `README.md` and `LICENSE.md` in the npm distribution, making each published package self-contained.


## v0.1.107-alpha

No user-facing changes.


## v0.1.106-alpha

No user-facing changes.


## v0.1.105-alpha

- **Breaking: CLI renamed from `agentic` to `agentengine`**: Replace all invocations of the `agentic` command with `agentengine` before upgrading; no compatibility alias is provided.
- **Breaking: Environment variables renamed from `AGENTIC_*` to `AGENTENGINE_*`**: Any CLI-owned environment variables you previously set under `AGENTIC_*` names (for example `AGENTIC_SENTRY_*`, `AGENTIC_MCP_OAUTH_DIR`, `AGENTIC_DEV_WATCH`) must be renamed to their `AGENTENGINE_*` equivalents; the old names are silently ignored and the CLI will warn you at startup if any legacy names are detected in your environment.
- **Breaking: Ignore file renamed to `.agentengineignore`**: Rename your project's `.agenticignore` file to `.agentengineignore` before upgrading; existing customizations are migrated automatically on first run, but the old filename is no longer read.
- **MCP OAuth login command updated**: Error messages and documentation that previously instructed you to run `agentic dev mcp auth login <server>` now correctly reference `agentengine dev mcp auth login <server>`.
- **Service account token minting command updated**: The `agent-engine-memory` README now correctly shows `agentengine service-account create` as the command to mint a service-account access token.


## v0.1.104-alpha

- **Bounded tool-call logging**: Debug logs for tool arguments and results now emit only shallow metadata (up to 20 top-level field names, type/length descriptions, no values) instead of serializing full payloads, preventing large tool calls such as `filesystem_write` from stalling execution while logging.
- **Corrected tool result status values**: The `cached` status was removed from the SDK's documented set of valid tool result statuses — it was never a real value; the supported statuses are `success`, `error`, `suspend`, `blocked`, `require_review`, and `interrupted`.
- **Corrected `shell_execute` environment isolation documentation**: The `buildShellEnv` and `shell_execute` API reference now accurately describes the scope of environment variable scrubbing, removing misleading claims about tenant secret isolation and execution approval gates that did not reflect actual behavior.


## v0.1.103-alpha

- **Breaking: All JavaScript SDK packages renamed to Atlas Agent Engine identities**: Every SDK package has moved to a new name with no compatibility aliases. Update your `package.json` dependencies and all import statements to the new identities: `@mongodb-js/agent-engine-sdk` (core SDK), `@mongodb-js/agent-engine-runner-shared` (runner shared), `@mongodb-js/agent-engine-sdk-memory` (memory), and `@mongodb-js/agent-engine-sdk-langgraph` (LangGraph). See each package's CHANGELOG for the name it replaces.
- **Breaking: Checkpoint and artifact keys renamed**: Internal wire keys have changed to reflect the new Atlas Agent Engine identity. Plain session resume continues to work after upgrading, but LangGraph session forks of checkpoints created before the rename and restores of ADK durable sessions created before the rename will fail.
- **Breaking: `CALL_INTERRUPTED_ARTIFACT_KEY` wire value changed**: The frozen interrupt-artifact key is now `__agent_engine_oe_call_interrupted__`, so any sessions checkpointed under the previous key value will not be detected correctly after upgrading.
- **Breaking: `DEFAULT_MCP_OAUTH_CLIENT_NAME` value changed**: The constant now reads `"Atlas Agent Engine Dev MCP Client"`, which affects any code that matched against the previous string literal.
- **Durable history inspection and branch continuation for LangGraph agents**: TypeScript LangGraph agents can now inspect past execution history and continue from a prior branch using the new `session_fork` API.
- **Memory SDK authentication uses service-account tokens**: The hosted Memory SDK (`@mongodb-js/agent-engine-sdk-memory`) now authenticates with service-account tokens instead of the previous auth mechanism.
- **`collective_metadata` removed from memory records**: Memory records no longer include the `collective_metadata` field; update any code that reads or writes this property.


## v0.1.102-alpha

- **Consistent tool failure errors across live and replay execution**: Durable workflows now surface the same error type whether a registered tool fails during initial execution or during replay, preventing false nondeterminism rejections that could crash a restarting application.
- **Remote MCP OAuth login now works with public-client servers**: Connecting to hosted MCP servers (such as Sentry's) that issue no client secret during dynamic registration no longer causes the loopback login flow to stall.
- **Stable durable replay when tool message artifacts are absent**: A `ToolMessage` with a missing or `null` artifact is now treated identically by the durable message encoder, so an application restart no longer produces a different state hash and fails replay.
- **LangGraph TypeScript adapter updated to 1.4.15**: The minimum supported LangGraph version for the TypeScript durable adapter has been raised to 1.4.15; upgrade your LangGraph dependency if you use durable workflows.


## v0.1.101-alpha

- **Richer connector tool error messages**: When a connector tool call fails, the error message now includes the external API's own explanation (for example, `HTTP 400 UNKNOWN — The value 'OpenJira' does not exist for the field 'project'`), so you can immediately see why a request was rejected instead of receiving only a generic status code.
- **Reduced durable workflow storage with full replay fidelity**: Durable workflow sessions now write a more compact message snapshot that eliminates redundant framework-specific envelopes, while existing sessions created before this release continue to replay, resume, and branch correctly without any migration step.


## v0.1.100-alpha

No user-facing changes.


## v0.1.99-alpha

- **LangGraph TypeScript durable session forking**: Durable LangGraph TypeScript agents now support history inspection and branch continuation — calling `graph.updateState(config, null)` on a committed checkpoint forks into a new session backed by an immutable branch, matching the capability already available in the Python adapter.
- **Improved resilience for tool dispatch during replica failover**: Established tool executions and LLM steps now recover automatically when a session owner replica goes down — settled steps replay from durable storage, and the runner retries HTTP 503 responses with `Retry-After` instead of failing immediately.
- **LLM dispatch heartbeats and abandoned relay takeover**: Active LLM dispatches now maintain a heartbeat so that if the owning replica dies, a surviving replica can take over the relay or direct call rather than leaving the session silently stuck.
- **Structured LLM failure attribution**: LLM invocation failures (such as provider 401 or 404 errors) are now classified with a structured error code and source instead of being reported as unknown, making it easier to distinguish provider errors from platform or client errors.
- **Clearer termination messages on fatal startup failures**: When a tool pod or agent runner encounters a fatal error at startup (such as a bad entry point or an import failure), it now writes a concise, redacted description of the failure reason that is visible in the pod's termination status, rather than leaving only a bare exit code.
- **Platform trace identity preserved through agent execution and resume**: The platform request trace ID is now propagated through agent execution, resume, and tool runtimes and included in structured log output, making it possible to correlate agent logs with the originating request.
- **TypeScript SDK API reference**: A generated Markdown API reference is now available for all four TypeScript SDK packages (`agent-engine-sdk-core`, `agent-engine-memory`, `agent-engine-runner-shared`, and `agent-engine-sdk-langgraph`) under `docs/api/`.
