"""Tests for ``validate_subagent_tree``.

Pulled out of ``test_factory_sdk.py`` so the validator has its own test
file matching the new module layout. ``test_factory_sdk.py`` keeps the
end-to-end build tests; this file targets the validator function in
isolation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from agent_engine_sdk_langgraph.subagents import (
    MAX_SUBAGENT_NESTING_DEPTH,
    validate_subagent_tree,
)

if TYPE_CHECKING:
    from deepagents import SubAgent


def _spec(**fields: Any) -> SubAgent:
    """Build a SubAgent-shaped dict — typed loosely so test inputs read naturally.

    Returned as ``SubAgent`` (a TypedDict) so call sites match
    ``validate_subagent_tree``'s ``Sequence[SubAgent | ...]`` parameter
    without per-call casts. The runtime value is just a plain dict.
    """
    return cast("SubAgent", dict(fields))


# ---------------------------------------------------------------------------
# String models bypass OE routing → must raise
# ---------------------------------------------------------------------------


class TestStringModelRejection:
    def test_raises_on_top_level_string_model(self) -> None:
        with pytest.raises(RuntimeError, match="string model spec"):
            validate_subagent_tree([_spec(name="bad", description="d", model="gpt-4o")])

    def test_raises_on_nested_string_model(self) -> None:
        """A string model deep in a nested ``subagents`` chain is still caught."""
        nested = _spec(name="leaf", description="d", model="gpt-4o")
        with pytest.raises(RuntimeError, match="string model spec"):
            validate_subagent_tree(
                [
                    _spec(
                        name="parent",
                        description="d",
                        model=MagicMock(spec=BaseChatModel),
                        subagents=[nested],
                    )
                ]
            )

    def test_error_names_the_offending_subagent(self) -> None:
        with pytest.raises(RuntimeError, match="'culprit'"):
            validate_subagent_tree(
                [_spec(name="culprit", description="d", model="gpt-4o")]
            )

    def test_validate_subagent_tree_string_model_without_name_raises_runtime_error(
        self,
    ) -> None:
        """A malformed spec with a string model but no 'name' field should
        raise the documented RuntimeError (not a KeyError from the f-string)."""
        malformed = [{"model": "gpt-4"}]  # no 'name'
        with pytest.raises(RuntimeError, match="<unnamed>"):
            validate_subagent_tree(malformed)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Accepted shapes
# ---------------------------------------------------------------------------


class TestAcceptedSpecs:
    def test_none_is_accepted(self) -> None:
        # No subagents declared — nothing to validate, no raise.
        validate_subagent_tree(None)

    def test_empty_list_is_accepted(self) -> None:
        validate_subagent_tree([])

    def test_basechatmodel_instance_is_accepted(self) -> None:
        validate_subagent_tree(
            [_spec(name="ok", description="d", model=MagicMock(spec=BaseChatModel))]
        )

    def test_compiled_subagent_is_skipped(self) -> None:
        """``runnable``-keyed specs are pre-built graphs; the model has
        already been bound and there is nothing for us to inspect."""
        validate_subagent_tree(
            [_spec(name="compiled", description="d", runnable=MagicMock())]
        )

    def test_no_model_field_is_accepted(self) -> None:
        """A SubAgent that doesn't override the parent's model is fine —
        it inherits ``secure_llm`` from the parent agent."""
        validate_subagent_tree([_spec(name="default-model", description="d")])


# ---------------------------------------------------------------------------
# AsyncSubAgent — log + skip
# ---------------------------------------------------------------------------


class TestAsyncSubAgentWarning:
    def test_async_subagent_logs_warning_and_continues(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.WARNING, logger="agent_engine_sdk_langgraph.subagents"
        ):
            validate_subagent_tree(
                [_spec(name="remote", description="d", graph_id="some-graph")]
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "AsyncSubAgent 'remote'" in message
        assert "graph_id='some-graph'" in message

    def test_async_subagent_warning_is_per_spec(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two AsyncSubAgents → two warnings, not deduplicated."""
        with caplog.at_level(
            logging.WARNING, logger="agent_engine_sdk_langgraph.subagents"
        ):
            validate_subagent_tree(
                [
                    _spec(name="r1", description="d", graph_id="g1"),
                    _spec(name="r2", description="d", graph_id="g2"),
                ]
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2


# ---------------------------------------------------------------------------
# Recursion depth cap
# ---------------------------------------------------------------------------


class TestRecursionDepthCap:
    def _nested(self, depth: int) -> SubAgent:
        """Build a chain of depth *depth* — every level has a nested subagents key."""
        spec: dict[str, Any] = dict(
            _spec(
                name=f"d{depth}",
                description="d",
                model=MagicMock(spec=BaseChatModel),
            )
        )
        if depth > 0:
            spec["subagents"] = [self._nested(depth - 1)]
        return cast("SubAgent", spec)

    def test_depth_at_max_is_accepted(self) -> None:
        validate_subagent_tree([self._nested(MAX_SUBAGENT_NESTING_DEPTH)])

    def test_depth_one_past_max_raises(self) -> None:
        with pytest.raises(RuntimeError, match="exceeds maximum recursion depth"):
            validate_subagent_tree([self._nested(MAX_SUBAGENT_NESTING_DEPTH + 1)])
