"""
Agent-to-Agent (A2A) client for the Atlas Agent Engine platform.

Provides methods for agent discovery and invocation through the
Orchestration Engine's REST API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from agent_engine_runner_shared.server.http_retry import (
    RETRY_AFTER_MAX_WAIT_S,
    parse_retry_after_seconds,
)
from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls

logger = logging.getLogger(__name__)

# Bounded retry so a churn-window 503 does not surface to the relay. Only a
# 503 carrying Retry-After is retried: the OE uses that marker for pre-dispatch
# busy, and retrying a post-dispatch failure would mint a second child.
_A2A_INVOKE_MAX_ATTEMPTS = 2
_A2A_RETRY_BASE_DELAY_S = 0.5


@dataclass(frozen=True)
class AgentSkill:
    """A skill advertised by an agent."""

    name: str
    description: str
    example_input: Optional[str] = None
    example_output: Optional[str] = None


@dataclass(frozen=True)
class DiscoveredAgent:
    """An agent visible to the caller after policy filtering."""

    agent_id: str
    name: str
    description: str
    project_id: str
    skills: List[AgentSkill] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=list)
    output_modes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResponse:
    """Response from an A2A invoke_agent call."""

    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    execution_id: str = ""


class AgentToAgent:
    """Client for agent-to-agent communication via the Orchestration Engine.

    All calls route through the OE's REST API, ensuring policy enforcement,
    logging, and HITL controls.

    Args:
        oe_url: The OE's HTTP base URL (e.g. "http://localhost:8000").
        auth_token: Bearer token for A2A authentication (the A2A JWT).
    """

    def __init__(self, oe_url: str, auth_token: str = "") -> None:
        self._oe_url = oe_url.rstrip("/")
        self._auth_token = auth_token
        self._client: Optional[httpx.Client] = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            headers: Dict[str, str] = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            # Use create_httpx_client_with_tls to enable mTLS when OE_URL is https://
            self._client = create_httpx_client_with_tls(
                self._oe_url,
                timeout=310.0,
                verify_hostname=True,
                set_base_url=True,
            )
            # Update headers after client creation
            self._client.headers.update(headers)
        return self._client

    def _invoke_with_retry(
        self, client: httpx.Client, body: Dict[str, Any]
    ) -> tuple[httpx.Response, Dict[str, Any]]:
        delay = _A2A_RETRY_BASE_DELAY_S
        for attempt in range(1, _A2A_INVOKE_MAX_ATTEMPTS + 1):
            try:
                resp = client.post("/a2a/invoke", json=body)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Connect-phase only. Read errors are not retried: the request
                # may already have minted a child execution.
                if attempt >= _A2A_INVOKE_MAX_ATTEMPTS:
                    raise
                logger.warning("A2A invoke_agent connect error; retrying")
                time.sleep(delay)
                continue
            retryable = (
                resp.status_code == 503
                and resp.headers.get("Retry-After") is not None
                and attempt < _A2A_INVOKE_MAX_ATTEMPTS
            )
            if not retryable:
                if resp.status_code >= 400:
                    return resp, {}
                return resp, resp.json()

            retry_after = parse_retry_after_seconds(resp.headers.get("Retry-After"))
            wait = min(retry_after or delay, RETRY_AFTER_MAX_WAIT_S)
            logger.warning(
                "A2A invoke_agent retrying after %s on status %s",
                wait,
                resp.status_code,
            )
            time.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover

    def discover_agents(
        self,
        project_id: str = "",
        skills: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        input_modes: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[DiscoveredAgent]:
        """Discover agents available for A2A calls.

        All filters are ANDed. Results are restricted by OE policy.

        Args:
            project_id: Scope to a specific project. Empty = caller's project.
            skills: Filter to agents advertising any of these skill names.
            capabilities: Filter to agents with any of these capability tags.
            input_modes: Filter to agents accepting any of these MIME types.
            limit: Max results to return.

        Returns:
            List of discovered agents.
        """
        client = self._ensure_client()
        params: Dict[str, Any] = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        for skill in skills or []:
            params.setdefault("skills", [])
            params["skills"].append(skill)
        for cap in capabilities or []:
            params.setdefault("capabilities", [])
            params["capabilities"].append(cap)
        for mode in input_modes or []:
            params.setdefault("input_modes", [])
            params["input_modes"].append(mode)

        try:
            resp = client.get("/a2a/discover", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("A2A discover_agents failed: %s", e)
            raise

        data = resp.json()
        agents = []
        for a in data.get("agents", []):
            agent_skills = [
                AgentSkill(
                    name=sk.get("name", ""),
                    description=sk.get("description", ""),
                    example_input=sk.get("example_input") or None,
                    example_output=sk.get("example_output") or None,
                )
                for sk in a.get("skills", [])
            ]
            agents.append(
                DiscoveredAgent(
                    agent_id=a.get("agent_id", ""),
                    name=a.get("name", ""),
                    description=a.get("description", ""),
                    project_id=a.get("project_id", ""),
                    skills=agent_skills,
                    capabilities=a.get("capabilities", []),
                    input_modes=a.get("input_modes", []),
                    output_modes=a.get("output_modes", []),
                )
            )
        return agents

    def invoke_agent(
        self,
        agent_id: str,
        message: str,
        skill: str = "",
        timeout: Optional[float] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> AgentResponse:
        """Invoke another agent via the Orchestration Engine.

        Args:
            agent_id: Target agent identifier (from discover_agents).
            message: The message to send to the agent.
            skill: Optional skill name hint for multi-skill agents.
            timeout: Optional timeout in seconds. None uses the OE default.
            custom_headers: Optional headers forwarded to the target agent's
                execution context. Keys must not use the reserved ``a2a-`` prefix.

        Returns:
            AgentResponse with status, result, and execution_id.
        """
        client = self._ensure_client()
        body: Dict[str, Any] = {
            "target_agent_id": agent_id,
            "message": message,
        }
        if skill:
            body["skill"] = skill
        if timeout is not None:
            body["timeout_seconds"] = timeout
        if custom_headers:
            body["custom_headers"] = custom_headers

        try:
            try:
                from agent_engine_runner_shared.tracing import get_tracer
            except Exception:
                resp, data = self._invoke_with_retry(client, body)
                if resp.status_code >= 400:
                    resp.raise_for_status()
            else:
                tracer = get_tracer("agent-engine-runner-shared.a2a")
                with tracer.start_as_current_span("a2a.invoke_agent") as span:
                    span.set_attribute("agentic.span.kind", "a2a")
                    span.set_attribute("a2a.target_agent_id", agent_id)
                    if skill:
                        span.set_attribute("a2a.skill", skill)
                    resp, data = self._invoke_with_retry(client, body)
                    if resp.status_code >= 400:
                        # Inside the span so failures record ERROR.
                        resp.raise_for_status()
                    child_execution_id = data.get("execution_id", "")
                    if child_execution_id:
                        span.set_attribute("a2a.child_execution_id", child_execution_id)
        except httpx.HTTPError as e:
            logger.error("A2A invoke_agent failed: %s", e)
            raise

        return AgentResponse(
            status=data.get("status", "failed"),
            result=data.get("result") or None,
            error=data.get("error") or None,
            execution_id=data.get("execution_id", ""),
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
