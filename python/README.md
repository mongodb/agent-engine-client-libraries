# Atlas Agent Engine Python SDKs

Python SDK packages for MongoDB's Atlas Agent Engine.

## Framework SDKs

Install the SDK for your agent framework, plus `agent-engine-sdk-memory` if
your agent uses project memory. The Google ADK SDK is Python-only.

| Package | Install | Description |
|---|---|---|
| [`agent-engine-sdk-langgraph`](packages/agent-engine-sdk-langgraph) | `pip install agent-engine-sdk-langgraph` | LangGraph framework SDK |
| [`agent-engine-sdk-adk`](packages/agent-engine-sdk-adk) | `pip install agent-engine-sdk-adk` | Google ADK framework SDK |
| [`agent-engine-sdk-memory`](packages/agent-engine-sdk-memory) | `pip install agent-engine-sdk-memory` | Project memory (semantic and episodic) |

## Internal core packages

The framework SDKs depend on these packages. Agents don't install or import
them directly.

| Package | Description |
|---|---|
| [`agent-engine-sdk`](packages/agent-engine-sdk) | Framework-neutral core protocols and models |
| [`agent-engine-runner-shared`](packages/agent-engine-runner-shared) | Shared runtime that serves agents and tools on the platform |

## Development

```bash
uv sync
uv run ruff check .
```

Release notes are in [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
