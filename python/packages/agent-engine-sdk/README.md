# Agent Engine SDK

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Framework-neutral protocols, models, and interfaces shared by Atlas Agent
Engine SDK integrations. The package has no dependency on the platform runtime
and can be used to define agent, execution, message, tool, and LLM contracts in
an independent application.
> This is an internal core package; agents don't install it directly. Install a
> framework SDK (`agent-engine-sdk-langgraph` or `agent-engine-sdk-adk`), which depends on it.

## Install

```bash
pip install agent-engine-sdk
```

## Development

```bash
uv sync --extra dev
uv run pytest
```

Copyright 2026 MongoDB, Inc. Licensed under the Apache License, Version 2.0.
