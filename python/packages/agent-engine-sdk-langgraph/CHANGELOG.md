# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deprecated
- `App(database_name=...)` and `App(traces_collection_name=...)` are ignored.
  They only chose where the runner wrote traces to MongoDB, and the runner now
  sends traces only to the Orchestration Engine and an optional OTLP endpoint.
  The checkpointer database is unaffected. Both arguments still parse so an
  agent that sets them keeps starting, and passing either raises a
  `DeprecationWarning`; they will be removed in a future release.

### Fixed
- Delegated OAuth tools on native checkpoint executions (durable workflow off,
  MongoDB checkpointer) continue after consent and return the real API result,
  for invoke and stream. Each completed call and its result are kept in a child
  checkpoint bound to the call's tool-call ID (its position when it has none),
  so a run continued in a new process returns the stored result, or raises the
  call's own recorded failure with its error type, instead of calling the
  provider again. A recorded call whose tool or arguments changed
  fails before any provider call. Stateless graphs, flat tools and durable
  executions are unchanged.
- Cancelling a turn on a non-durable Deep Agent keeps the session's earlier
  conversation. The cancelled-turn checkpoint copied the thread's stored channel
  values, which omit a `DeltaChannel` between snapshots, so it carried only the
  stopped turn's message and the next turn started without history.
- Durable Deep Agents on the pinned deepagents version remember the conversation
  between turns. deepagents 0.7 keeps `messages` in a `DeltaChannel`, which
  checkpoints omit between snapshots, so every durable turn committed its state
  without messages. Sessions already committed that way stay empty. A turn in
  flight across this upgrade fails on replay, because its recorded steps hold no
  messages, and must be run again.
- Durable Deep Agent tool calls from one model turn can run Tool Pod backend
  operations at the same time. A backend operation left in flight by a lost
  runner now reaches the model as non-retryable, with the Orchestration Engine's
  explanation.
- Tool results from parallel calls of a durable Deep Agent turn all reach the
  model. Their message IDs collided, so one result was replaced by a failure
  placeholder.
- Messages from durable Deep Agents on deepagents versions that keep `messages`
  in a `DeltaChannel` keep the same IDs when a replacement attempt rebuilds them.
- A model-supplied `timeout` of 0 on the Deep Agent `execute` tool runs the
  command under the Tool Pod's default timeout. It used to kill the command
  immediately.

### Changed
- `app.checkpointer()` loads checkpoints with an explicit type allowlist instead
  of LangGraph's permissive default, which imports and calls any callable named
  in stored checkpoint bytes. LangChain messages, `Interrupt`, `Command`,
  standard library values, plain dicts and `TypedDict` state round-trip as
  before. Any other pydantic model, dataclass or Enum in graph state, including
  a `response_format` model in `structured_response`, now comes back as raw
  field data, and LangGraph logs a `Blocked deserialization of <module>.<name>`
  warning. `testing.build_test_agent` uses the same serializer.
- Requires `langgraph-checkpoint>=4.1.1`.

### Added
- `App(checkpoint_allowed_types=[...])` lists the custom state types a
  checkpoint load may reconstruct, as classes or `(module, name)` pairs. `True`
  is rejected, so the permissive mode cannot be turned back on.

  ```python
  from my_agent.state import Ticket

  app = App(
      app_name="My Agent",
      checkpoint_allowed_types=[Ticket, ("my_agent.state", "Priority")],
  )
  ```

  **List your custom state types before upgrading.** A session that resumes
  without its type allowlisted writes the raw data into its next checkpoint,
  and adding the type later does not convert it back, so the session's current
  state keeps the raw data.
