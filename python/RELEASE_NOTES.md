# Release Notes — Python Client Libraries

Release notes for the Python SDK packages. Per-package historical changelogs:

- `packages/agent-engine-runner-shared`: [CHANGELOG.md](packages/agent-engine-runner-shared/CHANGELOG.md)
- `packages/agent-engine-sdk`: [CHANGELOG.md](packages/agent-engine-sdk/CHANGELOG.md)
- `packages/agent-engine-sdk-memory`: [release-notes/](packages/agent-engine-sdk-memory/release-notes/)

<!-- New entries go at the top. -->

## v0.1.115

- Add generated API-reference docs pipeline for agent-engine-runner-shared


## v0.1.114

- **Per-call interrupt for in-runtime tool calls**: Clicking Stop on an individual tool call now correctly interrupts tools running inside the agent runtime — previously, the interrupt was silently ignored and the tool ran to completion.
- **Continued execution after a per-call stop in deep agents**: Stopping a single tool call in a deep agent no longer ends the entire turn; the remaining tool calls and the model's response continue as normal.
- **Honest stop outcomes for non-cancellable tools**: Synchronous tool bodies that cannot be preempted now report themselves as non-cancellable instead of being incorrectly marked as stopped, so their real success or error result is preserved.


## v0.1.113

- **Improved transient LLM failure recovery**: Workflows that encounter timeouts, connection resets, or provider-side errors (such as rate limits, server overload, or gateway timeouts) before any response output is produced now automatically retry the LLM call instead of failing immediately, so transient provider disruptions are less likely to surface as workflow errors.
- **Fixed replay hash mismatch for empty application state**: Workflows that completed with an explicitly empty application-state selection no longer fail with a `replay state hash does not match` error after persistence, ensuring ADK-produced sessions with empty state are accepted and replayed correctly.


## v0.1.112

No user-facing changes.


## v0.1.111

- **Updated default gateway domain**: The Memory SDK now connects to `agentengine.mongodb.com` by default, replacing the previous domain, so existing code that relied on the old default URL should be updated to point to the new domain if a custom URL was not already configured.
- **Improved durable workflow recovery across upgrades**: Workflows can now recover correctly after a model or framework upgrade by distinguishing framework-managed metadata (such as message IDs and provider fields) from application-controlled inputs and state, so replacement attempts no longer fail due to ID or metadata drift introduced by the framework.
- **Stricter durable message history contract**: Durable message history is now append-only; operations such as ID-based message replacement, `RemoveMessage`, and `Overwrite` on committed history are unsupported, so workflows that need stable identity should store a business key in application state and append new messages instead.
- **Fixed ADK parallel branch recovery**: Parallel wait branches now each append their own scoped function-response event on suspension and resume the original invocation correctly, resolving an issue where sibling branch responses were lost when ADK rebuilt model history.


## v0.1.110

- Rename Agentic-era HTTP header contracts to Agent Engine names


## v0.1.109-alpha

- **Fixed: Taxonomic memory search now correctly filters by user**: `MemoryClient.fetch_taxonomic_memories` now accepts and forwards a `user_id` parameter to the memory server, so results are properly scoped to the requesting user instead of returning entries from all users in the project.
- **Breaking: Observability environment variables renamed**: The four runner-shared observability environment variables are now `AGENT_ENGINE_ENVIRONMENT`, `AGENT_ENGINE_MAX_ACTIVE_SERIES`, `AGENT_ENGINE_PROJECT_ID`, and `AGENT_ENGINE_TENANT_ID`. Update any deployment configuration that sets the previous names before upgrading.
- **Improved memory extraction reliability**: Transient provider failures (5xx errors, rate limits, timeouts) now cause memory extraction tasks to retry automatically instead of silently completing with no memories written; content-level failures isolate the offending item and mark it rather than failing the entire batch, so one unembeddable entry no longer blocks the rest.


## v0.1.108-alpha

- **Clearer LLM credential errors at invoke time**: When an LLM provider rejects your API key with a 401 or 403 during a run, the platform now surfaces a specific credential error with a "Configure secrets" prompt instead of a generic failure message, so you can identify and fix the misconfigured secret immediately.
- **Clearer tool credential errors at invoke time**: When a tool call fails because an external API rejects a project secret, the error is now attributed to the credential rather than the platform, with a targeted prompt to update your secrets configuration.
- **Accurate error type for timed-out tool calls**: Tool calls that are rejected because a prior sync invoke already timed out now raise a `ToolExecutionError` instead of a `PolicyDeniedException`, so your error handling correctly distinguishes a timeout consequence from an actual policy denial.
- **Improved LLM registry load error reporting**: When your agent's entrypoint fails to load (for example, due to a missing credential or a transient database error), LLM calls now report the actual entrypoint failure — redacted of any secrets — instead of a misleading message about missing `app.llm()` calls.
- **Memory search query now recorded in trace metadata**: The effective query sent to the memory engine during semantic, taxonomic, and episodic recall is now captured in memory-event metadata (`metadata.memory.query`), so memory recall traces show the real query used rather than an empty placeholder.
- **Log files written in dev mode for structured-logging agents**: When running locally with `agentic dev`, agents using structured logging now write human-readable log files to disk alongside JSON stdout, filling a gap where the Logs page and `agentic dev logs` previously showed no output for those agents.
- **SDK packages now include README and LICENSE**: All Python SDK package distributions (`agent-engine-sdk`, `agent-engine-sdk-adk`, `agent-engine-sdk-memory`, `agent-engine-sdk-langgraph`, `agent-engine-runner-shared`) now bundle their `README.md` and `LICENSE` files, making each published package self-contained.


## v0.1.107-alpha

No user-facing changes.


## v0.1.106-alpha

No user-facing changes.


## v0.1.105-alpha

- Rename the agentic CLI to agentengine (consolidated stack: –) (#4861)


## v0.1.104-alpha

- **Faster tool invocations with large payloads**: Debug logging for tool arguments and results now emits only bounded, shallow metadata (at most 20 top-level fields, no recursive traversal, no values) instead of serializing full payloads, eliminating multi-second stalls that large `filesystem_write` calls previously caused on the invoke path.
- **Removed undocumented `cached` tool status**: The `cached` status value was never sent by any runner and has been removed from the SDK documentation; the supported statuses are `success`, `error`, `suspend`, `blocked`, `require_review`, and `interrupted`.
- **Corrected `shell_execute` environment isolation documentation**: Docstrings for `shell_execute` and `_build_shell_env` now accurately describe the scope of the environment allowlist, removing misleading claims about tenant secret isolation and a non-existent OE approval gate.


## v0.1.103-alpha

- **Breaking: Python SDK packages renamed to Atlas Agent Engine identities**: All Python packages have been renamed with no compatibility aliases — update your dependencies and imports to the new identities before upgrading: `agent-engine-sdk` (import `agent_engine_sdk`), `agent-engine-sdk-memory` (import `agent_engine_sdk_memory`), `agent-engine-sdk-langgraph` (import `agent_engine_sdk_langgraph`), `agent-engine-sdk-adk` (import `agent_engine_sdk_adk`), and `agent-engine-runner-shared` (import `agent_engine_runner_shared`). See each package's CHANGELOG for the name it replaces.
- **Breaking: Some pre-rename durable state is not portable across the rename**: Plain session resume continues to work after upgrading, but LangGraph session forks of checkpoints created before the rename and restores of ADK durable sessions created before the rename will fail.
- **Breaking: `collective_metadata` removed from memory records**: The `collective_metadata` field no longer exists on memory records, so any code that reads or writes it must be updated before upgrading.
- **Memory SDK authentication now uses service-account tokens**: The hosted Memory SDK (`agent-engine-sdk-memory`) now authenticates with service-account tokens instead of API keys, so update your credential configuration when upgrading.
- **Session history inspection and branch continuation for LangGraph agents**: LangGraph agents can now inspect durable session history and continue execution from a prior branch point, enabling richer replay and branching workflows.


## v0.1.102-alpha

- **Reliable durable replay after application restart**: LangGraph agents no longer fail replay with a nondeterminism error when a `ToolMessage` artifact was absent during recording but serialized as `null` after a checkpoint round-trip — both shapes are now treated identically at the adapter boundary.
- **Consistent tool failure semantics across durable replay**: Registered tool failures now replay with the same error type and fields that live execution produced (`ToolExecutionError`, `ToolCallTimeoutError`, or `ExternalAPICallError`), so LangGraph and ADK agents can branch on error types without triggering a nondeterminism rejection after application replacement.
- **Stable ADK structured tool results on replay**: ADK agents that use structured tool responses no longer fail replay after application replacement due to protobuf-induced differences in object key order or integer-vs-float representation — results are now canonicalized before being hashed into the durable activity input.
- **Parallel ADK workers are replay-safe**: Child run identities for parallel ADK workers are now allocated from deterministic input order before execution begins, so concurrent task scheduling and completion order no longer affect durable identity after application replacement.
- **Unsupported suspension inside ADK parallel workers rejected at startup**: Configuring a `LongRunningFunctionTool` inside a parallel worker subtree now raises an error before any external work begins, rather than failing unpredictably mid-execution.
- **Remote MCP OAuth login fixed for servers that default to confidential clients**: The OAuth provider now explicitly declares itself a public client during dynamic registration, so loopback login against hosted MCP servers (such as Sentry's) completes successfully instead of stalling.
- **Updated minimum framework versions for durable adapters**: The Python LangGraph adapter now requires LangGraph 1.2.11 or later, and the Google ADK adapter now requires ADK 2.9.0 or later.


## v0.1.101-alpha

- **Improved resilience for agent-to-agent calls**: `AgentToAgent.invoke_agent` now automatically retries on 5xx errors and transient transport failures (honoring `Retry-After` headers, capped at two total attempts), so a brief infrastructure disruption no longer surfaces as an immediate error to your agent; 4xx errors are never retried.
- **Richer connector tool error messages**: When an external API call fails, the error message now includes the provider's own explanation (e.g., `HTTP 400 UNKNOWN — The value 'OpenJira' does not exist for the field 'project'`), so you can diagnose rejected requests without inspecting raw API responses.
- **Connector policy requirements surfaced in generated `tool.yaml`**: The generated `tool.yaml` now includes a comment listing the egress destinations and credential secrets your connector requires, with paste-ready `agentic agent egress add` commands, so missing policy is visible at authoring time rather than at runtime.
- **Reduced durable workflow storage with full replay compatibility**: Durable sessions now write a compact version 1 message snapshot that omits the redundant framework-serializer envelope, lowering storage overhead while preserving history, resume, replay, and branch fidelity for both new and existing sessions.


## v0.1.100-alpha

No user-facing changes.


## v0.1.99-alpha

- **Durable session history and branching for LangGraph Python agents**: The `agent-engine-sdk-langgraph` package now validates the destination patch node before creating a branch, so unsupported fork targets fail immediately rather than after the branch is created.
- **Improved resilience for agent runners on replica failover**: Python runners now retry HTTP 503 responses with `Retry-After` backoff, and established tool-execute and tool-stream calls can continue on a surviving replica without requiring a live session owner.
- **Dispatch heartbeats and abandoned-relay takeover**: When an agent runner dies mid-dispatch, a surviving instance can now detect the expired heartbeat and take over the relay or direct LLM call, preventing silent loss of in-flight work.
- **Clearer termination messages on fatal startup failures**: When an agent runner pod exits due to a bad entrypoint, import error, or unhandled exception, a bounded, redacted summary is now written to the container termination log so the cause is visible in `kubectl describe pod`.
- **Platform trace identity preserved through agent execution**: Request trace IDs are now propagated from the gateway through agent execution and resume, and emitted in Python runtime logs, making it easier to correlate agent output with platform-level traces.
- **Breaking: `publishing.*` metadata filter fields removed from memory retrieval**: Passing a `publishing.*` field as a metadata filter to the memory retrieval API now returns a 422 error; remove any such filters from your retrieval calls before upgrading.
- **Memory SDK documents extraction failure behavior**: The memory SDK README now explains how background extraction failures (LLM, embedding, or database errors) are classified, retried, and surfaced, so you can reason about when extracted memories may be delayed or absent.
- **Improved LLM invocation error attribution**: LLM provider failures (such as 401 or 404 responses) during agent invocation are now correctly attributed to the LLM layer rather than reported as unknown errors, giving more accurate failure diagnostics.
- **`agentic create-tool` CLI command (experimental)**: A new `agentic create-tool` command generates tool catalogs and scaffolded `tool.yaml` files from an OpenAPI spec URL or file, with three modes: compile (from spec), scaffold (consumer `tool.yaml` only), and regenerate (recompile from a local config).
- **Generated API reference docs for Python SDK packages**: All public Python SDK packages (`agent-engine-sdk`, `agent-engine-sdk-adk`, `agent-engine-sdk-memory`, `agent-engine-sdk-langgraph`) now ship committed `docs/api.md` files generated from docstrings, giving you a browsable API reference alongside the packages.
