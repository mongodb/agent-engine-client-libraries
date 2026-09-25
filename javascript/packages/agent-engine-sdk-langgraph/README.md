# @mongodb-js/agent-engine-sdk-langgraph

TypeScript port of [`agent-engine-sdk-langgraph`](../python/packages/agent-engine-sdk-langgraph/) — the LangGraph-based agent SDK that wraps customer agents with platform security, audit, and observability.

> **Status:** 🚧 Under active development. Not ready for production use.

---

## Architecture

This package mirrors the Python `agent-engine-sdk-langgraph` module-for-module. Each TypeScript file ports a single Python file of the same name.

| TypeScript file              | Python original          | Purpose                                                                             |
| ---------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| `src/index.ts`               | `__init__.py`            | Public re-exports (`App`, `LangGraphBaseAgent`, etc.)                               |
| `src/runtime.ts`             | `runtime.py`             | `App` class — the SDK entry point                                                   |
| `src/agent.ts`               | `agent.py`               | `LangGraphBaseAgent` wrapping `CompiledStateGraph`                                  |
| `src/query.ts`               | `query.py`               | `LangGraphQueryPlugin` — session summaries/messages from `MongoDBSaver` collections |
| `src/secure_llm.ts`          | `secure_llm.py`          | `SecureWrappedLLM` — routes LLM calls through proxy                                 |
| `src/messages.ts`            | `messages.py`            | LangChain ↔ platform message translator                                             |
| `src/llm_adapter.ts`         | `llm_adapter.py`         | Adapter from `BaseChatModel` to platform `BaseLLM`                                  |
| `src/subagents.ts`           | `subagents.py`           | Subagent dispatch + `lc_agent_name` tracking                                        |
| `src/node_logger_adapter.ts` | `node_logger_adapter.py` | LangGraph callbacks → `NodeExecutionLogger`                                         |
| `src/deep_agent.ts`          | `deep_agent.py`          | `deepagents` library wrapper                                                        |
| `src/deep_agent_task.ts`     | `deep_agent_task.py`     | Deep Agent `task` tool-call parsing                                                 |
| `src/deep_agent_checkpointer.ts` | `deep_agent_checkpointer.py` | Deep Agent checkpointer policy (tolerates adapter-owned Send routing)     |
| `src/durable_deep_agent.ts`  | `durable_deep_agent.py`  | Durable `task` dispatch: child operation paths + message-id stamping                |
| `src/session_fork.ts`        | `session_fork.py`        | Session fork: native copy + durable OE branch, wrapped into `updateState`           |
| `src/backends/toolpod.ts`    | `backends/toolpod.py`    | `AgentEngineToolPodBackend` (sandbox backend)                                       |
| `src/_stubs.ts`              | (none)                   | Temporary stubs for Phase 9/12 agent-engine-runner-shared symbols                             |

## Dependency direction

Each line below is one dependency rank, top to bottom (computed from the
actual local import graph):

```
index.ts
runtime.ts
agent.ts
session_factory.ts
durable_session.ts
deep_agent.ts · execution_session.ts · secure_llm.ts · backends/toolpod.ts
deep_agent_checkpointer.ts · durable_deep_agent.ts · durable_subgraphs.ts · session_fork.ts
platform_checkpointer.ts
workflow_state.ts
durable_tools.ts · llm_adapter.ts · query.ts · suspend.ts · workflow_message.ts
messages.ts · checkpoint_branch.ts · checkpointer.ts · deep_agent_task.ts ·
durable_message_identity.ts · node_logger_adapter.ts ·
stopped_tool_call_middleware.ts · subagents.ts · thread_id.ts ·
backends/tool_sandbox.ts · workflow_json.ts
```

A file **may only import from files on lower lines** in this tree. Imports going the other way are bugs.

## Configuration

| Environment variable | Default | Description |
| -------------------- | ------- | ----------- |
| `MONGODB_URI` | _(unset)_ | MongoDB connection source for the checkpointer and query plugin in AER mode. |
| `MDB_AGENTIC_STORE_DB` | `mdb_store` | Base name for the per-project MongoDB store used for LangGraph checkpoints in AER mode. Project scoping and discovery still apply unless overridden below. |
| `CHECKPOINT_DB_NAME` | _(unset)_ | Exact `MongoDBSaver` database name when set. Skips project scoping and discovery. Set it on the agent AER pod environment or a SecretRef to opt into a checkpoint database shared by dual-runtime agents. |

By default, the LangGraph checkpoint `thread_id` is `session_id:workspace_id`.
Agents can register `app.resolveThreadId((ctx) => ...)`; its return value is
used verbatim on fresh and resume invocations, with no workspace suffix
appended. Custom keys are invisible to Atlas Agent Engine `/query/sessions*` history,
which still looks up only the default session/workspace-derived keys. Agents
that bypass workspace scoping own collision isolation within the checkpoint
database. The key must be reconstructible from `RequestContext` (including the
session and authenticated identity) on every turn.

**Reads are scoped-only.** Session history expands each Atlas Agent Engine `session_id`
to only its workspace-scoped composite key; the bare unscoped key is never
queried once a workspace scope is known, because bare keys are readable and
writable by every workspace on the shared store. Legacy checkpoints written
before scoping existed are therefore not served by the history endpoints. An
empty scope is legitimate only on explicitly unscoped runtimes (local
development and tests, with no `APP_ID`). Managed AERs carry
`REQUIRE_PROJECT_SCOPED_DB`; if `APP_ID` is missing there, reads and writes
fail closed instead of trusting the wire workspace or using bare keys.
Production adopters of custom keys should still treat checkpoint-key
uniqueness inside a shared database as agent-owned.

```ts
import { App } from "@mongodb-js/agent-engine-sdk-langgraph";

const app = new App({ appName: "support-agent" });

app.resolveThreadId((ctx) => `${ctx.sessionId}__${ctx.userId}`);

app.entrypoint(() => {
  // Build the LangGraph graph here and pass this saver to graph.compile().
  const checkpointer = app.checkpointer();
  return buildGraph().compile({ checkpointer });
});

// On the agent AER pod, set CHECKPOINT_DB_NAME to the exact shared database.
```

## Status — what is ready vs. blocked

| Module                 | Status  | Blocking dependency                 |
| ---------------------- | ------- | ----------------------------------- |
| messages.ts            | pending | none — depends only on LangChain    |
| llm_adapter.ts         | pending | none — runner-shared symbols ready  |
| node_logger_adapter.ts | pending | none — `Metrics` ready              |
| deep_agent.ts          | pending | none — `deepagents` npm available   |
| deep_agent_task.ts     | done    | none — implemented + tested         |
| deep_agent_checkpointer.ts | done | none — implemented + tested        |
| durable_deep_agent.ts  | done    | none — implemented + tested         |
| platform_checkpointer.ts | done  | none — implemented + tested         |
| secure_llm.ts          | pending | agent-engine-runner-shared Phase 9            |
| backends/toolpod.ts    | done    | none — implemented + tested         |
| subagents.ts           | pending | agent-engine-runner-shared Phase 9            |
| agent.ts               | pending | agent-engine-runner-shared Phase 9            |
| runtime.ts             | pending | agent-engine-runner-shared Phase 9 + Phase 12 |
| index.ts               | pending | all of the above                    |

## Known gaps from Python parity

| Feature                     | Python | TypeScript | Notes                                                                                     |
| --------------------------- | ------ | ---------- | ----------------------------------------------------------------------------------------- |
| MCP tool servers            | ✅     | ✅         | `agent-engine-runner-shared`'s `mcp_tools.ts`/`mcp_oauth.ts`, discovery wired into `App`. |
| `langgraph.types.Overwrite` | ✅     | ⚠️ shim    | Not in LangGraph.js yet — handled via defensive unwrap.                                   |

## Skills

Pass parent source directories through `App.deepAgent(..., { skills: [...] })`.
At runtime,
deepagents lists each source through the configured backend and discovers only
its immediate child directories containing `SKILL.md`; discovery is not
recursive. deepagents skips unreadable or unparsable frontmatter and skills
missing `name` or `description`; it warns but may still load Agent Skills naming
or directory-name violations. This SDK forwards declared paths without
inspecting or filtering them. The skills root is resolved at Tool Pod
startup, not at SDK import time, so a normal static SDK import works — no
import-order workaround is needed.

## Development

```sh
# install (requires workspace setup once teammate's branch lands)
npm install

# type-check only (no emit)
npm run typecheck

# build distributable
npm run build

# unit tests
npm run test

# lint
npm run lint
```

## Coding standards

- Strict TypeScript (`strict: true` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`).
- Snake_case filenames (matching the Python original 1:1 for review).
- camelCase function/variable names.
- PascalCase class/type names.
- Each file = single responsibility (one class or one focused concept).
- Public surface depends on interfaces (Dependency Inversion).
- No hardcoded secrets, ever.

See the root `docs/coding-standards/typescript.md` for repo-wide rules.
