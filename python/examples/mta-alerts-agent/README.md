# mta-alerts-agent

A LangGraph agent that answers questions about NYC subway service using the
MTA's public GTFS-realtime feeds (`api-endpoint.mta.info`).

It is a deployable fixture for the CLI's live-dev E2E suite
(`TestDevDualWorkspaceLifecycle`), which needs a second, distinct agent next to
`insurance-agent` — one whose manifest declares `network.egress` to a non-LLM
endpoint. The suite copies the package out of the uv workspace before running
`agentengine build`, so the archive contains only this agent.

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` to select a
provider. `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` are optional overrides;
when unset, the provider SDK uses its public default endpoint.

