"""The agents an app composes, understood before a turn does any work.

The entrypoint returns one native ``Agent``. Handoffs and agents used as tools
make other agents reachable from it, so the durable runtime validates each of
them and indexes them by name: a turn starts from the agent that ended the
previous one.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from typing import Any, cast

from agents import Agent, FunctionTool, Handoff
from agents import handoffs as sdk_handoffs

from agent_engine_sdk_openai_agents.agent_tools import is_agent_tool, scoped_agent_tool
from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError
from agent_engine_sdk_openai_agents.llm_adapter import RegisteredModel
from agent_engine_sdk_openai_agents.secure_model import SecureModel
from agent_engine_sdk_openai_agents.tools import SecureToolInvoker
from agent_engine_sdk_openai_agents.wire import invocation_options

__all__ = ["agent_graph"]


def agent_graph(
    root: Agent[Any], *, registered_llms: Mapping[str, object], tool_issuer: object
) -> dict[str, Agent[Any]]:
    """Validate every agent reachable from ``root`` and index them by name.

    Rejects, before any model or tool work, what OE cannot own or replay.
    """
    agents: dict[str, Agent[Any]] = {}
    reachable: dict[str, set[str]] = {}
    agent_tool_edges: list[tuple[str, str]] = []
    pending = [root]
    while pending:
        agent = pending.pop()
        known = agents.get(agent.name)
        if known is agent:
            continue
        if known is not None or not agent.name:
            # Durable state records the active agent by name.
            raise UnsupportedDurableOpenAIAgentsError(
                f"agent name {agent.name!r} must be unique and not empty"
            )
        _validate(agent, registered_llms, tool_issuer)
        agents[agent.name] = agent
        handoff_targets = [_handoff_target(entry) for entry in agent.handoffs]
        # The model routes by name. The Runner only compares the names of the
        # agent it is running, so a clash on an agent reached by a handoff
        # would otherwise surface after earlier agents' model work.
        names = [tool.name for tool in agent.tools]
        for entry, target in zip(agent.handoffs, handoff_targets, strict=True):
            if not isinstance(entry, Handoff):
                names.append(Handoff.default_tool_name(target))
            elif entry.is_enabled:
                names.append(entry.tool_name)
        shared = sorted({name for name in names if names.count(name) > 1})
        if shared:
            raise UnsupportedDurableOpenAIAgentsError(
                f"agent {agent.name!r} has more than one tool or handoff named "
                f"{', '.join(map(repr, shared))}; give each a unique name"
            )
        nested = [
            scoped_agent_tool(cast(FunctionTool, tool))
            for tool in agent.tools
            if is_agent_tool(tool)
        ]
        reachable[agent.name] = {a.name for a in (*handoff_targets, *nested)}
        agent_tool_edges.extend((agent.name, target.name) for target in nested)
        pending.extend((*handoff_targets, *nested))
    for parent, child in agent_tool_edges:
        nested = _within(reachable, child)
        # An agent that can reach itself through an agent tool could nest
        # runs without bound.
        if parent in nested:
            raise UnsupportedDurableOpenAIAgentsError(
                f"agent {parent!r} is reachable from its own agent tool {child!r}"
            )
        # Refused here, before any tool runs, rather than when the nested run
        # first asks for an approval.
        for name in sorted(nested):
            for tool in agents[name].tools:
                if getattr(tool, "needs_approval", False) is not False:
                    raise UnsupportedDurableOpenAIAgentsError(
                        f"tool {tool.name!r} needs approval inside the agent tool "
                        f"{child!r}, which durable OpenAI Agents runs do not "
                        "support yet"
                    )
    return agents


def _within(reachable: Mapping[str, set[str]], start: str) -> set[str]:
    """``start`` and every agent it can reach."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(reachable.get(name, ()))
    return seen


def _validate(
    agent: Agent[Any], registered_llms: Mapping[str, object], tool_issuer: object
) -> None:
    model = agent.model
    registered = (
        registered_llms.get(model.llm_id) if isinstance(model, SecureModel) else None
    )
    if not isinstance(registered, RegisteredModel):
        raise UnsupportedDurableOpenAIAgentsError(
            f"agent {agent.name!r} model must be the model returned by app.llm(...)"
        )
    registered.validate()
    for tool in agent.tools:
        if is_agent_tool(tool):
            # Read and scoped with the rest of the graph, in agent_graph.
            continue
        # Issued by this app, not only securely wrapped: another App's tool
        # would run that app's callback and policy, which this app's runtime
        # never registered.
        if not (
            isinstance(tool, FunctionTool)
            and isinstance(tool.on_invoke_tool, SecureToolInvoker)
            and tool.on_invoke_tool.issuer is tool_issuer
        ):
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {getattr(tool, 'name', type(tool).__name__)!r} must come "
                "from this app's app.tools() or Agent.as_tool(...)"
            )
        # The builder can still change an issued tool before returning it. A
        # callable could decide differently when a replacement attempt replays.
        if not isinstance(tool.needs_approval, bool):
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {tool.name!r} needs_approval must be True or False"
            )
        # The SDK would cancel the call and tell the model it timed out, while
        # the OE tool activity keeps running: the turn could commit as a
        # success. The platform owns the timeout.
        if tool.timeout_seconds is not None:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {tool.name!r} sets a native timeout; use "
                "@app.tool(timeout=...) instead"
            )
        if tool.tool_input_guardrails or tool.tool_output_guardrails:
            raise UnsupportedDurableOpenAIAgentsError(
                f"tool {tool.name!r} guardrails are not supported yet"
            )
    unsupported = {
        "MCP servers": bool(agent.mcp_servers),
        "guardrails": bool(agent.input_guardrails or agent.output_guardrails),
        "agent hooks": agent.hooks is not None,
        "hosted prompts": agent.prompt is not None,
        "dynamic instructions": callable(agent.instructions),
        "custom tool_use_behavior": agent.tool_use_behavior != "run_llm_again",
    }
    found = [name for name, present in unsupported.items() if present]
    if found:
        raise UnsupportedDurableOpenAIAgentsError(
            f"durable OpenAI Agents runs do not support {', '.join(found)} yet"
        )
    invocation_options(agent.model_settings, None)


def _handoff_target(entry: object) -> Agent[Any]:
    """The agent a handoff transfers to, for a handoff replay can reproduce."""
    if isinstance(entry, Agent):
        return cast(Agent[Any], entry)
    if not isinstance(entry, Handoff):
        raise UnsupportedDurableOpenAIAgentsError(
            f"handoff {type(entry).__name__} is not an Agent or agents.handoff(...)"
        )
    handoff = cast(Handoff[Any, Any], entry)
    captured = _captured_by_sdk_handoff(handoff.on_invoke_handoff)
    target = captured.get("agent")
    if not isinstance(target, Agent) or not {"on_handoff", "input_type"} <= set(
        captured
    ):
        # A hand-built or wrapped Handoff, or an SDK whose handoff() differs
        # from the one this adapter was built against: what the Runner would
        # invoke cannot be read.
        raise UnsupportedDurableOpenAIAgentsError(
            f"handoff {handoff.tool_name!r} must be built with agents.handoff(...) "
            "from the openai-agents version this adapter pins"
        )
    # A filter rewrites history the durable state owns, and a callback or a
    # callable switch would run or decide again on every replay.
    unsupported = {
        "input_filter": handoff.input_filter is not None,
        "nest_handoff_history": bool(handoff.nest_handoff_history),
        "a callable is_enabled": not isinstance(handoff.is_enabled, bool),
        "on_handoff": captured["on_handoff"] is not None,
        "input_type": captured["input_type"] is not None,
    }
    found = [name for name, present in unsupported.items() if present]
    if found:
        raise UnsupportedDurableOpenAIAgentsError(
            f"handoff {handoff.tool_name!r} uses {', '.join(found)}, which durable "
            "OpenAI Agents runs do not support yet"
        )
    return cast(Agent[Any], target)


def _captured_by_sdk_handoff(invoke: object) -> dict[str, Any]:
    """What agents.handoff() captured in the invoker it built, else nothing.

    The Runner calls ``on_invoke_handoff`` itself, so every layer must be the
    SDK's own: its wrapper, applied to exactly the implementation ``handoff()``
    defines. Anything else could run code this validation never saw.
    """
    if not (
        isinstance(invoke, functools.partial)
        and invoke.func is getattr(sdk_handoffs, "_invoke_handoff_with_redaction", None)
        and len(invoke.args) == 1
        and not invoke.keywords
    ):
        return {}
    implementation = invoke.args[0]
    if not (
        inspect.isfunction(implementation)
        and implementation.__module__ == sdk_handoffs.__name__
        and implementation.__qualname__ == "handoff.<locals>._invoke_handoff_impl"
    ):
        return {}
    return dict(inspect.getclosurevars(implementation).nonlocals)
