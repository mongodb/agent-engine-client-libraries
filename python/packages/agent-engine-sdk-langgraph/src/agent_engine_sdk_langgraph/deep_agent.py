"""Deep agent factory for Atlas Agent Engine AER integration.

:func:`create_agent_engine_deep_agent` resolves relative skill paths and validates
subagent specifications before delegating to deepagents:

* :func:`~agent_engine_sdk_langgraph.subagents.validate_subagent_tree` rejects
  subagent specs with string models (which would bypass OE routing
  because the upstream factory instantiates a raw LLM without
  ``SecureWrappedLLM``).

The subagent validator lives in its own module so tests can drive it directly
instead of monkeypatching ``create_deep_agent``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, cast

from deepagents import (
    AsyncSubAgent,
    CompiledSubAgent,
    SubAgent,
    create_deep_agent,
)
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from agent_engine_sdk_langgraph import skills_tracing
from agent_engine_sdk_langgraph.deep_agent_checkpointer import (
    checkpointer_for_deep_agent,
)
from agent_engine_sdk_langgraph.durable_deep_agent import DurableDeepAgentMiddleware
from agent_engine_sdk_langgraph.subagents import validate_subagent_tree

# Patches SkillsMiddleware.before_agent/abefore_agent to emit a wrapper span.
# Applied at import time (rather than swapping in a subclass below) so it
# covers every SkillsMiddleware create_deep_agent constructs -- the main
# agent, the auto-injected general-purpose subagent, and any subagent with
# its own skills= -- without disturbing deepagents' own skills= wiring for
# those other two paths. See skills_tracing.py for why.
skills_tracing.apply()

SkillsBaseDir = str | os.PathLike[str]
SubAgentSpec = SubAgent | CompiledSubAgent | AsyncSubAgent


def _load_skill_paths(
    paths: Sequence[str],
    *,
    base_dir: SkillsBaseDir | None,
) -> list[str]:
    if base_dir is None:
        return list(paths)
    return [
        path if os.path.isabs(path) else os.path.join(base_dir, path) for path in paths
    ]


def _load_subagent_skill_paths(
    subagents: Sequence[SubAgentSpec],
    *,
    base_dir: SkillsBaseDir | None,
) -> list[SubAgentSpec]:
    resolved: list[SubAgentSpec] = []
    for spec in subagents:
        if "runnable" in spec or "graph_id" in spec or "skills" not in spec:
            resolved.append(spec)
            continue

        updated = dict(spec)
        updated["skills"] = _load_skill_paths(
            cast(Sequence[str], spec["skills"]),
            base_dir=base_dir,
        )
        resolved.append(cast(SubAgentSpec, updated))

    return resolved


def _compiled_subagent_checkpointer_names(
    subagents: Sequence[SubAgentSpec] | None,
) -> frozenset[str]:
    return frozenset(
        spec["name"]
        for spec in subagents or ()
        if "runnable" in spec
        and getattr(spec["runnable"], "checkpointer", None) is not None
    )


def create_agent_engine_deep_agent(
    secure_llm: BaseChatModel,
    backend: BackendProtocol,
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: None | bool | BaseCheckpointSaver = None,
    store: BaseStore | None = None,
    skills: list[str] | None = None,
    skills_base_dir: SkillsBaseDir | None = None,
) -> CompiledStateGraph:
    """Create a deep agent graph pre-configured for Atlas Agent Engine AER.

    Resolves relative skill paths, validates the subagent tree, then delegates
    to ``deepagents.create_deep_agent``.

    Args:
        secure_llm: ``SecureWrappedLLM`` instance — every LLM call is
            audited by the OE.
        backend: Backend for filesystem/shell ops; resolved by
            ``App.deep_agent()`` before calling this function.
        tools: Additional tools for the deep agent.
        subagents: SubAgent specs. Plain ``SubAgent`` entries with a
            ``model`` field must use a ``BaseChatModel`` instance, not a
            string — string models bypass OE routing.
        system_prompt: Custom system instructions.
        middleware: Additional middleware, run after the SDK's interrupt
            recovery and durable local-subagent nesting middleware.
        checkpointer: LangGraph checkpointer for state persistence,
            HITL suspend/resume, and multi-turn conversations.
        store: LangGraph store for skills and shared data.
        skills: Parent directories (as ``str``) for deepagents skill
            discovery. Each immediate child directory containing ``SKILL.md``
            is one skill. Discovery runs through the backend at runtime;
            deepagents validates frontmatter and warns when it skips malformed
            skills. Relative paths resolve against ``skills_base_dir``;
            absolute paths are forwarded unchanged. ``Path`` objects are NOT
            accepted — deepagents calls ``.rstrip("/")`` on the string,
            which crashes on ``pathlib.Path``. Convert with ``str(path)``
            at the call site.
        skills_base_dir: Base directory for relative skill paths. ``App``
            sets this from the directory containing ``agent.yaml``, optionally
            plus the relative ``AGENTIC_SKILLS_DIR`` override.

    Returns:
        A ``CompiledStateGraph`` ready for ``LangGraphBaseAgent`` wrapping.

    Raises:
        RuntimeError: A subagent spec uses a string model, or nesting
            exceeds the maximum recursion depth.
    """
    validate_subagent_tree(subagents)
    unsupported_subagent_names = _compiled_subagent_checkpointer_names(subagents)

    forwarded_skills = (
        _load_skill_paths(skills, base_dir=skills_base_dir)
        if skills is not None
        else None
    )
    forwarded_subagents = (
        _load_subagent_skill_paths(subagents, base_dir=skills_base_dir)
        if subagents is not None
        else None
    )

    return create_deep_agent(
        model=secure_llm,
        backend=backend,
        tools=tools,
        subagents=forwarded_subagents,
        system_prompt=system_prompt,
        # Prepended so every agent gets durable local-subagent attribution by
        # default. A per-call-stopped tool batch goes to the model as ordinary
        # interrupted ToolMessages — continuation is the model's call,
        # matching the TypeScript twin.
        middleware=[
            DurableDeepAgentMiddleware(
                unsupported_subagent_names=unsupported_subagent_names
            ),
            *middleware,
        ],
        checkpointer=checkpointer_for_deep_agent(checkpointer),
        store=store,
        skills=forwarded_skills,
    )
