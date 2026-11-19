# a2a-relay-agent

A deliberately minimal LangGraph agent whose only behaviour is to forward the
incoming message to a peer agent in the same project over A2A and reply with
the peer's result verbatim. It has no LLM, so a run's outcome depends only on
the platform's A2A path (OE discovery, JWT, child execution, callback) and on
the peer.

It is a deployable fixture for the CLI's live-environment E2E suite
(`TestDevAgentMemoryAndA2A`), which pairs it with `insurance-agent` to prove
that a memory written in one agent is readable by that agent when it is
invoked over A2A. The suite copies the package out of the uv workspace before
`agentengine build`, so the archive contains only this agent.

`A2A_RELAY_TARGET` names the peer by the `app_name` it registers for discovery
(default `Insurance Agent`); matching ignores case and separators. Prefix a message
with `relay:` to have the prefix stripped before forwarding.
