# Atlas Agent Engine TypeScript SDKs

TypeScript SDK packages for MongoDB's Atlas Agent Engine.

## Framework SDKs

Install the SDK for your agent framework, plus
`@mongodb-js/agent-engine-sdk-memory` if your agent uses project memory.

| Package | Install | Description |
|---|---|---|
| [`@mongodb-js/agent-engine-sdk-langgraph`](packages/agent-engine-sdk-langgraph) | `npm install @mongodb-js/agent-engine-sdk-langgraph` | LangGraph framework SDK |
| [`@mongodb-js/agent-engine-sdk-memory`](packages/agent-engine-memory) | `npm install @mongodb-js/agent-engine-sdk-memory` | Project memory (semantic and episodic) |

## Internal core packages

The framework SDKs depend on these packages. Agents don't install or import
them directly.

| Package | Description |
|---|---|
| [`@mongodb-js/agent-engine-sdk`](packages/agent-engine-sdk) | Framework-neutral core protocols, models, and schemas |
| [`@mongodb-js/agent-engine-runner-shared`](packages/agent-engine-runner-shared) | Shared runtime that serves agents and tools on the platform |

## Development

```bash
npm ci
npm run build --workspaces
npm test --workspaces
```

Release notes are in [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
