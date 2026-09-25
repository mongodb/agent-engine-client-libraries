"""Map configured ADK sub-agent transfers onto durable child operations.

ADK assigns the selected sub-agent its canonical ``Context.node_path`` before
the child runs. The route adapter already binds that path for model and tool
activities; this module only reserves the new child boundary under its
canonical parent when ADK enters a configured ``sub_agents`` target.

``AgentTool`` follows a different private-Runner lifecycle and is intentionally
unsupported. Keeping that bridge out of this feature leaves agent transfer and
Workflow routing on the same public ADK node-path primitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar

from google.adk.agents import BaseAgent
from google.adk.workflow import BaseNode

from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    preallocate_child_operation_ordinals,
)

_TRANSFER_AGENT_PROVENANCE_ATTRIBUTE = "_agent_engine_durable_transfer_provenance"


def configured_sub_agents(root: BaseNode) -> tuple[BaseAgent, ...]:
    """Return configured transfer targets anywhere in the static topology."""
    nodes: list[BaseAgent] = []
    seen: set[int] = set()

    def visit(node: BaseNode, *, transfer_target: bool = False) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))

        if transfer_target:
            if not isinstance(node, BaseAgent):
                raise TypeError("Google ADK sub_agents must contain agents")
            nodes.append(node)

        if isinstance(node, BaseAgent):
            for child in node.sub_agents:
                visit(child, transfer_target=True)

        graph = getattr(node, "graph", None)
        if graph is not None:
            for child in graph.nodes:
                if isinstance(child, BaseNode):
                    visit(child)

        for attribute in ("_inner_node", "_node"):
            child = getattr(node, attribute, None)
            if isinstance(child, BaseNode):
                visit(child)

    visit(root)
    return tuple(nodes)


def _configured_agent_suffix(agent: BaseAgent) -> tuple[str, ...]:
    names: list[str] = []
    current: BaseAgent | None = agent
    while current is not None:
        names.append(current.name)
        current = current.parent_agent
    return tuple(reversed(names))


class AgentRouteAdapter:
    """Discover configured sub-agents and reserve their durable child runs."""

    def __init__(
        self,
        root: BaseNode,
        operation_path: ContextVar[tuple[ChildOperationBoundary, ...] | None],
    ) -> None:
        self._operation_path = operation_path
        self.sub_agents = configured_sub_agents(root)
        transfer_agents: dict[int, BaseAgent] = {}
        for node in self.sub_agents:
            current: BaseAgent | None = node
            while current is not None:
                transfer_agents[id(current)] = current
                current = current.parent_agent
        transfer_provenance: set[object] = set()
        for node in transfer_agents.values():
            provenance = getattr(node, _TRANSFER_AGENT_PROVENANCE_ATTRIBUTE, None)
            if provenance is None:
                provenance = object()
                object.__setattr__(
                    node, _TRANSFER_AGENT_PROVENANCE_ATTRIBUTE, provenance
                )
            transfer_provenance.add(provenance)
        self._transfer_agent_provenance = frozenset(transfer_provenance)
        self._transfer_agent_suffixes = frozenset(
            _configured_agent_suffix(node) for node in transfer_agents.values()
        )

    def is_configured_transfer_agent(self, node: BaseNode) -> bool:
        """Whether this node descends from a configured transfer template."""
        return (
            getattr(node, _TRANSFER_AGENT_PROVENANCE_ATTRIBUTE, None)
            in self._transfer_agent_provenance
        )

    def is_transfer_agent_path(self, node_path: str) -> bool:
        """Whether a canonical path ends in a configured transfer tree."""
        segments = node_path.split("/")
        names: list[str] = []
        for segment in segments:
            name, separator, run_id = segment.rpartition("@")
            if not separator or not name or not run_id:
                return False
            names.append(name)
        return any(
            len(names) >= len(suffix) and tuple(names[-len(suffix) :]) == suffix
            for suffix in self._transfer_agent_suffixes
        )

    def reserve_child(
        self,
        node_boundaries: Sequence[ChildOperationBoundary],
    ) -> None:
        """Reserve one entered child under the parent in its canonical path.

        For example, ADK path ``router@1/reviewer@1/specialist@1`` becomes the
        OE boundaries ``reviewer -> specialist`` because ``router`` is OE's
        existing root. When ``specialist`` starts, this method temporarily
        binds ``reviewer`` and reserves ``specialist`` beneath it. ``route.py``
        then binds the complete ``reviewer -> specialist`` path while the child
        runs, so its model, tool, and suspension activities use that path.
        """
        boundaries = tuple(node_boundaries)
        if not boundaries:
            return

        # ADK exits the parent callback before entering the selected child.
        # Rebind the parent encoded in the canonical path for allocation.
        parent_token = self._operation_path.set(boundaries[:-1])
        try:
            preallocate_child_operation_ordinals((boundaries[-1],))
        finally:
            self._operation_path.reset(parent_token)
