# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `encode_tool_failure` and `decode_tool_failure`: a framework adapter can keep
  a tool call's failure in its own checkpoint and raise it again later, in this
  process or another, with the same error type and fields.

### Changed
- Spans go to the platform trace endpoint (`POST /internal/traces`) by default
  whenever `OE_URL` is set. `0.11.14` is the first version that does this.
  Earlier releases write spans straight to MongoDB unless
  `AGENTIC_TRACE_EXPORTER=1`, and the platform cannot detect this.
- **Breaking:** the connector `tool.yaml` `auth` block no longer takes `type`.
  It names exactly one credential: `bearer: <secret>`, `basic: {username,
  password}`, `api_keys: [{env, location, name, prefix}]`, or
  `provider: <OAuth app>` with optional `scopes`, plus optional `headers`; omit
  the block for no credential. A block that still uses `type` or its fields
  fails validation and names the key to use instead. `compile_catalog` and
  `scaffold_tool_yaml` propose the block from the document's security schemes
  and derive secret names from the connector name; `auth` now takes `none`,
  `bearer`, `basic` or `api_keys`, and an OAuth document needs `provider`.
  Generated catalogs record the credential kind without `type`.
  `Auth.header_name()` and `Auth.credential_bindings()` are removed; read
  `Auth.api_keys`. Regenerating a connector with no `auth` block compiles every
  operation anonymously instead of from the OpenAPI document's security, so
  operations the first compile skipped can appear as tools; they send no
  credential.

### Deprecated
- The LangGraph and ADK `App`s accept `database_name` and
  `traces_collection_name`, and the OpenAI Agents `App` accepts `database_name`,
  as ignored arguments. Each raises a `DeprecationWarning` and will be removed
  in a future release.

### Removed
- **Breaking:** the runner sends spans only to the Orchestration Engine
  (`POST $OE_URL/internal/traces`) and, when `OTEL_EXPORTER_OTLP_*` is set, to
  that OTLP endpoint. It no longer writes spans to MongoDB or to a local JSONL
  file. A runner without `OE_URL` stores no traces unless an OTLP endpoint is
  configured, and logs one warning when neither is.
- **Breaking:** `AGENTIC_TRACE_EXPORTER` no longer has any effect. Setting it
  to `0` or `false` does not turn off the platform trace exporter.
- **Breaking:** `MongoDBSpanExporter`, `JSONLSpanExporter`, `get_trace_path`,
  `attach_mongodb_tracing`, `start_mongodb_tracing_retry`, and
  `platform_trace_exporter_enabled` are removed, along with the
  `AGENTIC_TRACES_FILE` and `AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS`
  settings.
- **Breaking:** `setup_tracing()` no longer accepts `mongodb_collection` or
  `take_client_ownership`. `TenantRuntime` no longer accepts `database_name` or
  `traces_collection_name`. `mongodb_uri` remains for the checkpointer and
  sessions.
- **Breaking:** `tracing_status()` and the `/health` response's
  `details.tracing` no longer include `database_exporter`; they report only
  `platform_exporter`.
- Removed the unused pre-materialized connector bundle override
  (`AGENTIC_CONNECTOR_BUNDLE_PATH`) and the `load_connector_bundle` /
  `CONNECTOR_BUNDLE_PATH_ENV` exports. Connectors resolve only from
  `agent.yaml` and its local connector authoring files.

### Changed
- Remote MCP servers now use MCP Python SDK 2.x (`mcp==2.3.0`). Each call
  negotiates the protocol automatically: it probes `server/discover` and falls
  back to the legacy `initialize` handshake, so servers on `2025-11-25` and
  older keep working while `2026-07-28` servers are reached directly. Tool
  result content keeps its wire field names (`mimeType`, `readOnlyHint`) and
  failed HTTP calls keep reporting their status in error messages. OAuth caches
  written by the previous SDK still load; new entries record the authorization
  server's `issuer` when it provides one.
- Reuse a process-wide keep-alive mTLS client for OE callbacks (tool
  execute/result, node events, progress and custom events) and drain callback
  response bodies so connections return to the pool; removes a full TLS
  handshake per callback.

### Fixed
- A native OAuth consent pause carries `kind: oauth_authorization`, so the
  Orchestration Engine publishes its interrupt ID and the application answers
  it in `resume_map`; the consent answer can no longer satisfy a later
  application wait. After consent the token gate is asked again at a fresh
  step instead of the challenged one.
- AER and Tool Pod log records written outside agent or tool code now carry
  `executionId` when they belong to a known execution, so filtering logs by
  execution finds setup failures, cleanup, callback retries, guardrail
  warnings and stream handling. Partial stdout/stderr lines from concurrent
  executions are no longer merged or tagged with another execution's ID.
- Uncaught exceptions in agent code now emit one filterable `level=ERROR`
  record instead of only `WARNING`. Ordinary stderr writes with no
  associated exception (deprecation warnings, retry banners) stay `WARNING`,
  and the existing line-by-line traceback capture is unchanged — this adds a
  summary record rather than replacing anything.
- The Launcher now validates `RUNNER_MODE`, falling back to `aer` when it is
  unset or unrecognized. An unrecognized value previously reached
  the `service` field verbatim, and `/agent-logs` drops any record outside
  `{agent-execution-runtime, tool-executor}` — so the crash log vanished.
- Launcher installs logging before importing `AGENT_ENTRYPOINT` and emits
  `ERROR` (with traceback metadata) for import and entrypoint failures, so
  workspace logs no longer show those crashes as `INFO`.
- `build_uvicorn_log_config()` now routes `uvicorn`/`uvicorn.error`/
  `uvicorn.access` through the structured JSON pipeline when
  `STRUCTURED_LOGGING=true`. An unhandled exception in a route
  handler previously arrived as `level=WARNING` with `"ERROR:    "` baked
  into the message text and no `fields.exc_traceback`.

  For uvicorn/FastAPI records only (agent-emitted records are unaffected),
  four fields change: `level` now carries the real severity, `source`
  becomes `python-logging` instead of `stderr`, `logger` becomes the real
  origin instead of `root`, and the `"ERROR:    "` prefix is gone from
  `message`. A multi-line traceback now arrives as one `ERROR` record with
  the trace in `fields.exc_traceback` rather than several `WARNING` lines.

  **Migration:** saved queries or dashboards filtering on `source=stderr`
  or `logger=root` to find these lines will stop matching — re-target
  `logger` (`uvicorn`/`uvicorn.error`/`uvicorn.access`) or `level=ERROR`.

### Changed
- **Behavioral change (fail closed):** Managed runtimes that require project-
  scoped storage now require `APP_ID` when resolving checkpoint workspace
  scope. A missing `APP_ID` raises instead of trusting the wire workspace or
  using a bare session key. Intentionally unscoped local runtimes retain the
  wire fallback and bare-key behavior.
- **Behavioral change (fail closed):** `agent_engine_runner_shared.server.metadata.applied_metadata_env`
  now raises `RuntimeError` — before touching `os.environ` — when a region
  overlaps a concurrent secret-bearing region: either another region already
  has secrets applied, or this region carries secrets while any region is
  active. Previously overlap was silent and could swap one request's secrets
  into another's. In-repo call sites (`/execute`,
  `/invoke_llm`, `/invoke_llm/stream`) are serialized by the ToolServer's
  per-instance execute gate and are unaffected; a downstream consumer calling
  `applied_metadata_env` directly, or a second in-process call site, must be
  prepared to handle the exception per request.
- A region entered while another region is active now performs a *delete-only*
  restore: keys it added are removed, but overwritten values are no longer
  reverted, since its snapshot is tainted with the other region's
  request-scoped state. Regions entered on a quiet environment (the serialized
  production flow) keep the full snapshot restore.
