# test-agent

A minimal deployable agent with controllable hooks. It is **not a mock** — it runs inside the local stack (orchestration engine, agent execution runtime, and tool pod) and is exercised via real HTTP calls.

## Purpose

Integration tests need a way to deterministically trigger specific SDK code paths:

| Input | Behavior | SDK path exercised |
|-------|----------|--------------------|
| `__fail__` | `raise RuntimeError("Deliberate test failure")` | Error reporting, execution → `error` status |
| `__slow__` | `time.sleep(30)` | Timeout handling (`RUNNER_EXECUTION_TIMEOUT`) |
| `__slow__:N` | `time.sleep(N)` for N seconds | Configurable timeout testing |
| `__review__` | `app.suspend(reason="awaiting_human_review")` | HITL suspend/resume flow |
| Any city name | Returns `"{city}: 22°F"` + writes to semantic memory | Basic invoke, tool logging, memory persistence |

These are branches in a single tool (`get_weather`) rather than separate tools, keeping the agent minimal per the [test fixture pattern](https://xunitpatterns.com/Test%20Fixture.html).

## Quick start

```bash
cd packages/python/examples/test-agent
cp .env.example .env
# Edit .env: set at least one LLM key
agentengine dev up
```

Services start at:

| Service | URL |
|---------|-----|
| OE (Orchestration Engine) | http://localhost:8000 |
| AER (Agent Execution Runtime) | http://localhost:8001 |
| Tool Pod | http://localhost:8002 |
| MongoDB | mongodb://localhost:27017 |

## Example requests

**Normal invoke:**
```bash
curl -s -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","thread_id":"t1","message":"What is the weather in London?"}'
```

**Trigger error path:**
```bash
curl -s -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","thread_id":"t2","message":"What is the weather in __fail__?"}'
```

**Trigger slow path (5s delay):**
```bash
curl -s -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","thread_id":"t3","message":"What is the weather in __slow__:5?"}'
```

**Trigger HITL suspend:**
```bash
# Invoke — execution will suspend
curl -s -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","thread_id":"t4","message":"What is the weather in __review__?"}'

# Resume with approval — calls the OE directly (local dev only).
# For the API Gateway use POST /api/v1/projects/{project_id}/executions/{execution_id}/resume with flat {"decision": "approved"}.
# org_id and project_id are required and must match the execution. PROJECT_ID is
# generated per stack, so read both from the running OE rather than hardcoding.
ORG_ID=$(docker compose exec -T oe printenv ORG_ID)
PROJECT_ID=$(docker compose exec -T oe printenv PROJECT_ID)
curl -s -X POST "http://localhost:8000/resume/<execution_id>?org_id=$ORG_ID&project_id=$PROJECT_ID" \
  -H "Content-Type: application/json" \
  -d '{"human_review":{"decision":"approved","reviewer_notes":"test approval"}}'
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | One of these | OpenAI API key |
| `ANTHROPIC_API_KEY` | One of these | Anthropic API key |
| `GEMINI_API_KEY` | One of these | Google Gemini API key |
| `CEREBRAS_API_KEY` | One of these | Cerebras API key |
| `MONGODB_URI` | Optional | External MongoDB URI. If unset, a local Atlas container is started automatically. |
| `VOYAGE_API_KEY` | Yes | Required because test-agent enables `features.memory: true` |

`features.memory` comes from `agent.yaml`. The sample app keeps model defaults
in source, infers the provider from the available API key, and reads an optional
`<PROVIDER>_BASE_URL` from the environment.

## Directory structure

```
test-agent/
├── src/test_agent/
│   └── agent.py        # Agent with get_weather tool and test hooks
├── agent.yaml          # agentengine CLI config
├── pyproject.toml      # uv workspace member
├── .env.example        # Environment template
└── README.md
```

## Known limitations

- `save_semantic` (memory write) is called correctly but will fail silently in the Tool Pod because the generated `docker-compose.yml` does not set `MONGODB_URI` for that container. Memory persistence is visible from the agent execution runtime container.
- `user_id` is hardcoded to `"test-user"` — this agent is only intended for isolated local test environments.
