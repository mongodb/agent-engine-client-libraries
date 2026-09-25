"""Subagent spec validation for ``create_agent_engine_deep_agent``.

Hard-fails at agent construction time when a subagent spec uses a
string model — string models bypass the ``SecureWrappedLLM`` seam and
therefore the OE audit path. Pulled out of ``deep_agent.py`` so the
validator has its own importable surface for tests instead of being
exercised only through the wrapper factory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent

logger = logging.getLogger(__name__)

# Recursion is bounded so an adversarial spec (or an accidental cycle)
# can't blow the Python stack at agent-construction time.
MAX_SUBAGENT_NESTING_DEPTH = 10


def validate_subagent_tree(
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None,
) -> None:
    """Walk the subagent tree, raising on any string-model spec.

    String model specs would bypass OE routing because the upstream
    factory's ``resolve_model`` instantiates a raw LLM unwrapped in
    ``SecureWrappedLLM``. Only ``BaseChatModel`` instances (e.g.
    ``SecureWrappedLLM``) are accepted.

    Spec types handled:

    * ``CompiledSubAgent`` (``runnable`` set) — pre-built graph; the
      model has already been bound and we can't see it. Skipped.
    * ``AsyncSubAgent`` (``graph_id`` set) — remote graph that we can't
      validate in-process. Logged at WARNING and skipped — the receiving
      end is responsible for routing through the OE.
    * Plain ``SubAgent`` — validated; if ``model`` is a string we raise.

    Recursion depth is capped at :data:`MAX_SUBAGENT_NESTING_DEPTH` so
    an adversarial deeply-nested ``subagents`` chain cannot blow the
    Python stack at agent-construction time. The depth counter is held
    in a closure rather than a public kwarg so callers cannot start
    recursion mid-tree and bypass the cap.

    Raises:
        RuntimeError: A subagent spec uses a string ``model`` field, or
            nesting exceeds the maximum recursion depth.
    """

    def _walk(
        nodes: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None,
        depth: int,
    ) -> None:
        if nodes is None:
            return
        if depth > MAX_SUBAGENT_NESTING_DEPTH:
            raise RuntimeError(
                f"SubAgent nesting exceeds maximum recursion depth "
                f"({MAX_SUBAGENT_NESTING_DEPTH})"
            )
        for spec in nodes:
            # Pre-built graph — model already bound, nothing to validate.
            if "runnable" in spec:
                continue
            # Remote graph — defer to the receiving end.
            if "graph_id" in spec:
                logger.warning(
                    "AsyncSubAgent '%s' targets remote graph_id='%s' — "
                    "verify the remote graph routes through the OE for "
                    "audit compliance",
                    spec.get("name", "<unnamed>"),
                    spec["graph_id"],
                )
                continue
            model = spec.get("model")
            if model is not None and isinstance(model, str):
                name = spec.get("name", "<unnamed>")
                raise RuntimeError(
                    f"SubAgent '{name}' uses a string model spec "
                    f"('{model}') which bypasses OE routing. "
                    f"Pass a BaseChatModel instance (e.g. SecureWrappedLLM) instead."
                )
            _walk(spec.get("subagents"), depth + 1)

    _walk(subagents, 0)
