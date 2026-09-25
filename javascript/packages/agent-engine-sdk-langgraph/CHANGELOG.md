# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Durable LLM activities now replay deterministically: LangGraph runtime keys
  injected into the model call options (`executionInfo` — per-attempt
  checkpoint/task UUIDs and `nodeFirstAttemptTime` — plus `durability` and
  `control`) no longer reach the OE activity semantic input. Replaying a
  durable attempt whose suspension turn included LLM calls previously failed
  with `workflow is nondeterministic: activity ... changed` (deep Agent HITL
  recovery was the first scenario to hit this).
- Session summaries' `message_count` now reports the actual message count, matching the session messages endpoint. It previously reported the checkpoint count, which LangGraph writes once per graph super-step.

### Changed
- Session-history reads with a workspace scope now query only the scoped
  checkpoint key, so legacy bare-key histories are no longer returned. A
  workspace resolver returning `null` or `undefined` now fails closed;
  explicitly unscoped local runtimes continue to read bare-key checkpoints.
- Session-history messages for human-authored turns now report role `"user"` instead of `"human"`, matching the live invoke/stream `Message.role` vocabulary.

### Added
- Deep Agent graphs now run on `durable_workflow` sessions. Deep Agent task
  delegation is adapter-owned rather than a compiled child graph, so `createAgentEngineDeepAgent`
  installs a `DurableDeepAgentMiddleware` that attributes subagent work to a deterministic
  child operation path (name + ToolCall id, preallocated per same-step batch), stamps the
  subagent's input and result messages with replay-stable ids, and rejects compiled
  subagents whose runnable carries its own checkpointer. The graph's checkpointer is
  wrapped to tolerate LangChain's internal Send routing while sharing the platform
  scratch state.
- Initial package scaffold (Step 1).
- `src/_stubs.ts` — temporary stand-ins for agent-engine-runner-shared Phase 9 + Phase 12 symbols.
- `App.resolveThreadId()` — agent-owned LangGraph checkpoint keys. The return value is used verbatim on fresh and resume invocations instead of the default `session_id:workspace_id` derivation.
- `CHECKPOINT_DB_NAME` — exact `MongoDBSaver` database override with no project scoping, for dual-runtime shared checkpoint stores.
