# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Workflow child operation paths gained the Python parity surface for framework
  adapters that dispatch dynamic children outside the graph topology:
  `preallocateChildOperationOrdinals` assigns ordinals to a complete same-step
  sibling batch up front, and `childOperationBoundaryScope` runs a callback with
  one appended child boundary (composing with any active operation-path
  resolver). A pending-batch holder on the attempt context backs the
  adapter's two-phase model-step/tool-step stamping.

### Fixed
- Uncaught exceptions and unhandled rejections in agent code now emit a
  filterable `level=ERROR` record. Ordinary stderr writes with no
  associated exception stay `WARNING`. The record is written even when the agent
  installed its own crash handler first, and recovering from a crash no longer
  makes a later graceful shutdown report failure.
- **Log shape change:** for uncaught exceptions the stack now arrives on the
  ERROR record's `fields.exc_traceback` plus Node's raw output, instead of
  per-line `WARNING` records. Unhandled rejections are unchanged.
- An unset or unrecognized `RUNNER_MODE` no longer drops the crash record from
  `/agent-logs`; it falls back to `aer`.
- Launcher installs logging before importing `AGENT_ENTRYPOINT` and emits
  `ERROR` (with traceback metadata) for import and entrypoint failures, so
  workspace logs no longer show those crashes as `INFO`.

### Changed
- **Behavioral change (fail closed):** Managed runtimes that require project-
  scoped storage now require `APP_ID` when resolving checkpoint workspace
  scope. A missing `APP_ID` throws instead of trusting the wire workspace or
  using a bare session key. Intentionally unscoped local runtimes retain the
  wire fallback and bare-key behavior.
- **Behavioral change (fail closed):** `withMetadataEnv` and `withMetadataEnvGen`
  (`src/server/metadata.ts`) now throw — before touching `process.env` — when a
  region overlaps a concurrent secret-bearing region: either another region
  already has secrets applied, or this region carries secrets while any region
  is active. Previously overlap was silent and could swap one request's secrets
  into another's. In-repo call sites (`/execute`,
  `/invoke_llm`, `/invoke_llm/stream`) are serialized by the ToolServer's
  per-instance execute gate and are unaffected; a downstream consumer calling
  these directly, or a second in-process call site, must be prepared to handle
  the error per request. Note `withMetadataEnvGen` holds its region open for
  the whole iteration.
- A region entered while another region is active now performs a *delete-only*
  restore: keys it added are removed, but overwritten values are no longer
  reverted, since its snapshot is tainted with the other region's
  request-scoped state. Regions entered on a quiet environment (the serialized
  production flow) keep the full snapshot restore.

### Removed
- Unreferenced query/stream response models — execution logs, node executions,
  cost dashboard, executions, and sessions — and the `logExecutionStart` and
  `extractPodUsage` helpers were removed from the public surface. The
  orchestration-engine query shapes were consumed only by the retired Python
  orchestration engine, which the Go engine replaced; the session-query shapes
  and both helpers had no maintained caller.
