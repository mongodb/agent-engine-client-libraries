# Atlas Agent Engine Python Client Libraries

uv workspace for the Python SDK packages of MongoDB's Atlas Agent Engine.

## Packages

Directory names match distribution names.

| Package | Description |
|---|---|
| [`packages/agent-engine-sdk`](packages/agent-engine-sdk) | Framework-neutral core SDK primitives |
| [`packages/agent-engine-sdk-memory`](packages/agent-engine-sdk-memory) | Memory SDK (semantic/episodic project memory) |
| [`packages/agent-engine-runner-shared`](packages/agent-engine-runner-shared) | Runner SDK shared runtime (AER/tool server, drain, telemetry) |
| [`packages/agent-engine-sdk-langgraph`](packages/agent-engine-sdk-langgraph) | LangGraph framework adapter |
| [`packages/agent-engine-sdk-adk`](packages/agent-engine-sdk-adk) | Google ADK framework adapter |

## Example agents

`examples/` contains deployable agents used by the platform's integration and
E2E suites (`test-agent`, `insurance-agent`, `mta-alerts-agent`). They are
workspace members but are **not** published libraries.

## Development

```bash
uv sync          # from this directory
uv run ruff check .
```

Docs live in [`docs/`](docs/); release notes in [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
