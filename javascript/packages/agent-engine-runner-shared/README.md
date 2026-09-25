# @mongodb-js/agent-engine-runner-shared

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

The TypeScript runner runtime for Atlas Agent Engine. It provides the platform
protocol, secure tool and LLM wrappers, request-scoped runtime context, and
server adapters used by the TypeScript LangGraph SDK.

This package is an open-source runtime integration library. Running an agent
through the managed service requires Atlas Agent Engine services such as the
Orchestration Engine and Agent Execution Runtime; those services are not
distributed with this package.

## Development

Run commands from the TypeScript workspace root:

```bash
npm ci
npm run build --workspace=@mongodb-js/agent-engine-runner-shared
npm run test --workspace=@mongodb-js/agent-engine-runner-shared
```

See the workspace [open-source distribution notes](../../OPEN_SOURCE.md) for
the package boundary and dependency policy.

Copyright 2026 MongoDB, Inc. Licensed under the Apache License, Version 2.0.
