"""Tests for the memory seam primitives: MemoryRequestContext and MemoryRuntime."""

from typing import Any

from agent_engine_sdk_memory.models import ContextResponse, MemoryChunk, WriteTurnResult
from agent_engine_sdk_memory.protocol import (
    AmbientIdentityRuntime,
    MemoryRequestContext,
    MemoryRuntime,
)


def test_request_context_fields_default_to_none_and_round_trip():
    empty = MemoryRequestContext()
    assert (empty.user_id, empty.agent_id, empty.session_id) == (None, None, None)
    filled = MemoryRequestContext(user_id="u", agent_id="a", session_id="s")
    assert (filled.user_id, filled.agent_id, filled.session_id) == ("u", "a", "s")


_RUNTIME_METHODS = (
    "record_turn",
    "build_context",
    "build_context_from_sources",
    "search_semantic",
    "search_episodes",
    "search_taxonomic",
    "discover_procedures",
)


class _FullRuntime:
    def record_turn(self, *, role: str, **kwargs: Any) -> WriteTurnResult:
        return WriteTurnResult(id="i", session_id="s", turn_seq=0)

    def build_context(self, *, query: str, **kwargs: Any) -> ContextResponse:
        return ContextResponse(formatted_context="", metadata={})  # type: ignore[arg-type]

    def build_context_from_sources(
        self, *, query: str, **kwargs: Any
    ) -> ContextResponse:
        return ContextResponse(formatted_context="", metadata={})  # type: ignore[arg-type]

    def search_semantic(self, *, query: str, **kwargs: Any) -> list[MemoryChunk]:
        return []

    def search_episodes(self, *, query: str, **kwargs: Any) -> list[MemoryChunk]:
        return []

    def search_taxonomic(self, *, query: str, **kwargs: Any) -> list[MemoryChunk]:
        return []

    def discover_procedures(self, *, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class _PartialRuntime:
    def record_turn(self, *, role: str, **kwargs: Any) -> WriteTurnResult:
        return WriteTurnResult(id="i", session_id="s", turn_seq=0)


def test_runtime_is_runtime_checkable_positive():
    assert isinstance(_FullRuntime(), MemoryRuntime)


def test_runtime_is_runtime_checkable_negative():
    assert not isinstance(_PartialRuntime(), MemoryRuntime)


def test_runtime_declares_exactly_the_expected_methods():
    for name in _RUNTIME_METHODS:
        assert hasattr(MemoryRuntime, name)
    declared = {
        name
        for name in dir(MemoryRuntime)
        if not name.startswith("_") and callable(getattr(MemoryRuntime, name))
    }
    assert declared == set(_RUNTIME_METHODS)


# ---------------------------------------------------------------------------
# AmbientIdentityRuntime Protocol
# ---------------------------------------------------------------------------


class _WithRequestContext:
    """Minimal structural match for AmbientIdentityRuntime."""

    def request_context(self) -> MemoryRequestContext | None:
        return MemoryRequestContext(user_id="u")


class _WithoutRequestContext:
    """Does not expose request_context(); should not satisfy the Protocol."""

    pass


def test_ambient_identity_runtime_positive():
    # A runtime that exposes request_context() satisfies the Protocol and the
    # facade will call it to get per-request ambient identity.
    assert isinstance(_WithRequestContext(), AmbientIdentityRuntime)


def test_ambient_identity_runtime_negative():
    # A runtime without request_context() does not satisfy the Protocol; the
    # facade falls back to call-arg / bind identity for those runtimes.
    assert not isinstance(_WithoutRequestContext(), AmbientIdentityRuntime)
