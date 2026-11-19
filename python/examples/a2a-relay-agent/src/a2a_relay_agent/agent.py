# Copyright 2026 MongoDB, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic A2A relay.

One graph node, no LLM: the latest human message is forwarded to a peer agent
through the SDK's A2A tools and the peer's result becomes this agent's reply.
The agent exists so an E2E suite can exercise the platform's A2A path
(discovery, JWT, child execution, callback) against a deployed environment
without a model's tool-choice in the loop.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from agent_engine_sdk_langgraph import App

app = App(app_name="A2A Relay Agent")

logger = logging.getLogger(__name__)

# Peer agent addressed by the name it registers for A2A discovery, which is
# the App(app_name=...) value, not the agent.yaml name. Matching is
# normalised (case and separators) so either spelling finds the peer.
_TARGET_NAME = os.environ.get("A2A_RELAY_TARGET", "Insurance Agent")
_RELAY_PREFIX = "relay:"


def _norm(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _latest_human_text(state: MessagesState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _reply(payload: dict[str, object]) -> dict[str, list[AIMessage]]:
    return {"messages": [AIMessage(content=json.dumps(payload))]}


@app.entrypoint
def build_agent():
    # Built once per process; both tools hold the runtime and resolve the OE URL
    # and A2A token from the current execution context at call time.
    tools = {tool.name: tool for tool in app.a2a_tools()}
    if not tools:
        # a2a_tools() is empty when agent.yaml lacks a2a.enabled; a relay
        # without A2A has nothing to do, so refuse to start rather than fail
        # every invoke.
        raise RuntimeError("a2a-relay-agent requires a2a.enabled: true in agent.yaml")
    discover = tools["discover_available_agents"]
    invoke = tools["invoke_a2a_agent"]

    def relay(state: MessagesState):
        text = _latest_human_text(state)
        if text.startswith(_RELAY_PREFIX):
            text = text[len(_RELAY_PREFIX) :].lstrip()
        if not text:
            return _reply({"status": "error", "error": "empty message"})

        discovered = json.loads(discover.invoke({}))
        if "error" in discovered:
            return _reply({"status": "error", "stage": "discover", **discovered})
        targets = [
            a
            for a in discovered.get("agents", [])
            if _norm(a.get("name")) == _norm(_TARGET_NAME)
        ]
        if not targets:
            return _reply(
                {
                    "status": "error",
                    "stage": "discover",
                    "error": f"peer {_TARGET_NAME!r} not discoverable",
                    "visible": [a.get("name") for a in discovered.get("agents", [])],
                }
            )
        target_id = targets[0]["agent_id"]

        logger.info("relaying to %s (%s)", _TARGET_NAME, target_id)
        outcome = json.loads(invoke.invoke({"agent_id": target_id, "message": text}))
        return _reply(
            {
                "status": outcome.get("status"),
                "stage": "invoke",
                "target": _TARGET_NAME,
                "target_agent_id": target_id,
                "child_execution_id": outcome.get("execution_id"),
                "error": outcome.get("error"),
                "result": outcome.get("result"),
            }
        )

    graph = StateGraph(MessagesState)
    graph.add_node("relay", relay)
    graph.set_entry_point("relay")
    graph.add_edge("relay", END)
    return graph.compile(checkpointer=app.checkpointer())
