"""Memory.search: cross-source fan-out, merge/ordering, adaptation, fail-loud."""

from datetime import datetime, timezone
from typing import Any

import pytest
from agent_engine_sdk_memory import Memory
from agent_engine_sdk_memory.errors import MemoryNotSupportedError
from agent_engine_sdk_memory.models import MemoryChunk, MemorySource, SearchSource


def _chunk(source: MemorySource, score: float | None, id_: str = "i") -> MemoryChunk:
    return MemoryChunk(
        id=id_,
        content="c",
        source=source,
        timestamp=datetime.now(timezone.utc),
        similarity_score=score,
    )


class _FakeRuntime:
    """A MemoryRuntime stand-in returning preset results per source."""

    def __init__(
        self,
        *,
        semantic: list[MemoryChunk] = [],
        episodic: list[MemoryChunk] = [],
        taxonomic: list[MemoryChunk] = [],
        procedures: list[dict[str, Any]] = [],
        taxonomic_error: Exception | None = None,
    ) -> None:
        self._semantic = list(semantic)
        self._episodic = list(episodic)
        self._taxonomic = list(taxonomic)
        self._procedures = list(procedures)
        self._taxonomic_error = taxonomic_error
        self.calls: list[str] = []

    def search_semantic(self, *, query, user_id=None, visibility=None, top_k=50):
        self.calls.append("semantic")
        return list(self._semantic)

    def search_episodes(
        self, *, query, user_id=None, visibility=None, session_id=None, top_k=50
    ):
        self.calls.append("episodic")
        return list(self._episodic)

    def search_taxonomic(
        self, *, query, user_id=None, domain=None, visibility=None, top_k=50
    ):
        self.calls.append("taxonomic")
        if self._taxonomic_error is not None:
            raise self._taxonomic_error
        return list(self._taxonomic)

    def discover_procedures(
        self,
        *,
        query,
        user_id=None,
        visibility=None,
        tags=None,
        top_k=10,
        similarity_threshold=0.0,
        metadata_filter=None,
    ):
        self.calls.append("procedural")
        return list(self._procedures)


def test_search_defaults_to_semantic_and_episodic():
    rt = _FakeRuntime(
        semantic=[_chunk(MemorySource.SEMANTIC, 0.5)],
        episodic=[_chunk(MemorySource.EPISODIC, 0.9)],
        taxonomic=[_chunk(MemorySource.TAXONOMIC, 0.99)],
    )
    results = Memory(runtime=rt).search("q", user_id="u")
    # taxonomic is NOT queried by default; results sorted by score descending
    assert rt.calls == ["semantic", "episodic"]
    assert [c.source for c in results] == ["episodic", "semantic"]


def test_search_merges_and_sorts_by_score_unscored_last():
    rt = _FakeRuntime(
        semantic=[
            _chunk(MemorySource.SEMANTIC, 0.2, id_="a"),
            _chunk(MemorySource.SEMANTIC, None, id_="b"),
        ],
        episodic=[_chunk(MemorySource.EPISODIC, 0.8, id_="c")],
    )
    results = Memory(runtime=rt).search("q")
    assert [c.id for c in results] == ["c", "a", "b"]


def test_search_truncates_to_top_k():
    rt = _FakeRuntime(
        semantic=[
            _chunk(MemorySource.SEMANTIC, 0.9, id_="a"),
            _chunk(MemorySource.SEMANTIC, 0.8, id_="b"),
        ],
        episodic=[_chunk(MemorySource.EPISODIC, 0.7, id_="c")],
    )
    results = Memory(runtime=rt).search("q", top_k=2)
    assert [c.id for c in results] == ["a", "b"]


def test_search_adapts_procedures_to_chunks():
    rt = _FakeRuntime(
        procedures=[{"procedure": "start-quote", "description": "d", "score": 0.7}]
    )
    results = Memory(runtime=rt).search("q", sources=["procedural"])
    assert len(results) == 1
    chunk = results[0]
    assert chunk.source == "procedural"
    assert chunk.content == "d"
    assert chunk.similarity_score == 0.7
    assert chunk.metadata == {
        "procedure": "start-quote",
        "description": "d",
        "score": 0.7,
    }


def test_search_accepts_string_and_enum_sources():
    rt = _FakeRuntime(taxonomic=[_chunk(MemorySource.TAXONOMIC, 0.5)])
    assert len(Memory(runtime=rt).search("q", sources=["taxonomic"])) == 1
    rt2 = _FakeRuntime(taxonomic=[_chunk(MemorySource.TAXONOMIC, 0.5)])
    assert len(Memory(runtime=rt2).search("q", sources=[SearchSource.TAXONOMIC])) == 1


def test_search_accepts_a_bare_string_source():
    # A bare string is treated as one source, not iterated character by character.
    rt = _FakeRuntime(taxonomic=[_chunk(MemorySource.TAXONOMIC, 0.5)])
    assert len(Memory(runtime=rt).search("q", sources="taxonomic")) == 1


def test_search_raises_when_a_requested_source_is_unsupported():
    rt = _FakeRuntime(
        semantic=[_chunk(MemorySource.SEMANTIC, 0.5)],
        taxonomic_error=MemoryNotSupportedError("no taxonomic in api-key mode"),
    )
    with pytest.raises(MemoryNotSupportedError):
        Memory(runtime=rt).search("q", sources=["semantic", "taxonomic"])


def test_search_rejects_unknown_source():
    with pytest.raises(ValueError):
        Memory(runtime=_FakeRuntime()).search("q", sources=["stm"])
