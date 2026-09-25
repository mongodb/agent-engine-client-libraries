"""LangChain tool wrappers for Agent-to-Agent (A2A) operations.

These tools execute in the AER process (not via the tool pod) because
A2A requires the execution context's OE URL and A2A JWT, which are
only available in the AER during an active /execute request.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool

from agent_engine_runner_shared import TenantRuntime

logger = logging.getLogger(__name__)

# Named so empty discovery is not mistaken for "no peers configured".
_A2A_JWT_SECRET_HINT = (
    "If you expected peers, confirm each agent has a2a.enabled and registered "
    "successfully. Locally, A2A requires A2A_JWT_SECRET in the project .env "
    "(same value for every agent); the deployed platform seeds this automatically."
)


def build_a2a_tools(runtime: TenantRuntime) -> list[Any]:
    """Build LangChain tools for A2A agent discovery and invocation.

    Args:
        runtime: The TenantRuntime instance (must be in AER mode for
            the tools to function at call time).

    Returns:
        List of StructuredTool instances for discover_agents and invoke_agent.
    """

    def discover_available_agents() -> str:
        """Discover all agents available for A2A communication.

        Returns a JSON list of agents with their names, descriptions,
        skills, and capabilities. Use this to determine which agent can
        best handle a user's request before invoking one.
        """
        a2a = runtime.a2a
        if a2a is None:
            return json.dumps(
                {
                    "error": (
                        "A2A is not available — no OE connection or token. "
                        + _A2A_JWT_SECRET_HINT
                    )
                }
            )

        try:
            agents = a2a.discover_agents()
        except Exception as e:
            logger.error("A2A discovery failed: %s", e)
            return json.dumps({"error": f"Failed to discover agents: {e}"})

        my_workspace = os.environ.get("APP_ID", "")
        agents = [a for a in agents if a.agent_id != my_workspace]

        if not agents:
            return json.dumps(
                {
                    "agents": [],
                    "message": (
                        "No other agents are available. " + _A2A_JWT_SECRET_HINT
                    ),
                }
            )

        return json.dumps(
            {
                "agents": [
                    {
                        "agent_id": a.agent_id,
                        "name": a.name,
                        "description": a.description,
                        "skills": [
                            {"name": s.name, "description": s.description}
                            for s in a.skills
                        ],
                        "capabilities": a.capabilities,
                    }
                    for a in agents
                ]
            },
            indent=2,
        )

    def invoke_a2a_agent(agent_id: str, message: str, custom_headers: str = "") -> str:
        """Invoke another agent via A2A protocol.

        Use this after discovering agents to send a request to a specific
        agent. The agent_id must come from the discover_available_agents
        results.

        Args:
            agent_id: The target agent's ID from discovery results.
            message: The message to send to the agent.
            custom_headers: Optional JSON object of headers to forward to
                the target agent (e.g. '{"oauth-token": "..."}').
        """
        a2a = runtime.a2a
        if a2a is None:
            return json.dumps({"error": "A2A is not available."})

        headers_dict: dict[str, str] | None = None
        if custom_headers:
            try:
                headers_dict = json.loads(custom_headers)
            except (json.JSONDecodeError, TypeError):
                return json.dumps(
                    {"error": "custom_headers must be a valid JSON object"}
                )

        logger.info("Invoking agent via A2A: %s", agent_id)
        try:
            response = a2a.invoke_agent(
                agent_id=agent_id,
                message=message,
                custom_headers=headers_dict,
            )
        except Exception as e:
            logger.error("A2A invoke failed: %s", e)
            return json.dumps({"error": f"Failed to reach agent {agent_id}: {e}"})

        return json.dumps(
            {
                "status": response.status,
                "result": response.result or None,
                "error": response.error or None,
                "execution_id": response.execution_id,
            }
        )

    return [
        StructuredTool.from_function(
            func=discover_available_agents,
            name="discover_available_agents",
            description=(discover_available_agents.__doc__ or "").strip(),
        ),
        StructuredTool.from_function(
            func=invoke_a2a_agent,
            name="invoke_a2a_agent",
            description=(invoke_a2a_agent.__doc__ or "").strip(),
        ),
    ]
