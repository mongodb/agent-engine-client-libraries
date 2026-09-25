# Insurance Agent Fixture

This package is a realistic insurance-agent fixture. It stays close to a
customer-support agent so the Runner stack is exercised with meaningful
business behavior instead of a synthetic toy flow.

It is not the user-facing standalone example. For a maintained runnable demo,
use the insurance agent in
[mongodb/agent-engine-examples](https://github.com/mongodb/agent-engine-examples).

## Purpose

The fixture covers:

- multi-turn thread continuity from customer profile capture to policy creation
- cross-thread customer recall through persisted memory
- HITL suspend/resume behavior for high-risk claims
- SSE streaming through the local stack
- memory bootstrap when Voyage credentials are available

## What Lives Here

- [`main.py`](src/insurance_agent/main.py)
  The insurance workflow, tools, prompts, and LangGraph wiring used by the test
  fixture.
- [`policy_store.py`](src/insurance_agent/policy_store.py)
  Mongo-backed policy and claim storage used by the fixture.
- [`agent.yaml`](agent.yaml)
  Minimal agent metadata for packaging.
- [`env.example`](env.example)
  Reference environment file for the fixture's runtime configuration.

## Why There Are No Local Scripts Or Package Tests

The package-local `run-local.sh`, `test_flow.sh`, and `tests/` directory were
removed once this package became a fixture instead of a standalone example
application. The maintained runnable demo is the insurance agent in
[mongodb/agent-engine-examples](https://github.com/mongodb/agent-engine-examples).

## Runtime Notes

The fixture still supports the same core insurance behaviors used by the tests:

- quote generation and policy creation
- policy lookup and listing
- claim filing and risk analysis
- human-review suspend/resume
- optional memory-backed customer recall

A local `agentengine dev up` generates the runtime `.env` and controls whether
memory and search-index bootstrap are enabled for a given run.
