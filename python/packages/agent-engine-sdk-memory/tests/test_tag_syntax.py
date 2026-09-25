"""Client-side syntax checks must match the server's messages verbatim.

The authoritative strings live in the memory-server's tag_paths module;
these tests pin the subset the SDK mirrors so fail-fast client errors read
identically to server rejections.
"""

import json
from pathlib import Path

import pytest
from agent_engine_sdk_memory._tag_syntax import (
    _NAME_RE,
    RESERVED_TYPE_NAMES,
    validate_memory_type,
    validate_tag_syntax,
)
from agent_engine_sdk_memory.errors import MemoryClientError


class TestValidateMemoryType:
    def test_accepts_custom_name(self) -> None:
        validate_memory_type("support_tickets")

    @pytest.mark.parametrize(
        "name", ["semantic", "episodic", "taxonomic", "procedural"]
    )
    def test_rejects_builtin(self, name: str) -> None:
        with pytest.raises(MemoryClientError) as exc:
            validate_memory_type(name)
        assert str(exc.value) == (
            f"'{name}' is a built-in memory type and is not accepted by this operation"
        )

    def test_rejects_empty(self) -> None:
        with pytest.raises(
            MemoryClientError, match="memory_type must be a non-empty string"
        ):
            validate_memory_type("")

    def test_reserved_set_matches_repo_fixture(self) -> None:
        # In-monorepo parity check against the canonical fixture; skipped in
        # downstream checkouts where pkg/ is absent.
        fixture = (
            Path(__file__).resolve().parents[6]
            / "pkg"
            / "memoryconfig"
            / "reserved_names.json"
        )
        if not fixture.is_file():
            pytest.skip("reserved_names.json fixture not present in this checkout")
        data = json.loads(fixture.read_text())
        assert set(data["reserved_type_names"]) == set(RESERVED_TYPE_NAMES)
        assert data["name_pattern"] == _NAME_RE.pattern

    @pytest.mark.parametrize(
        "name", ["tickets", "support_tickets", "a", "t1", "a" * 64]
    )
    def test_accepts_names_matching_the_platform_pattern(self, name: str) -> None:
        validate_memory_type(name)

    @pytest.mark.parametrize(
        "name",
        [
            "a b",  # space
            "a/b",  # separator: would otherwise route to a bare 404
            "Tickets",  # uppercase
            "1tickets",  # leading digit
            "a-b",  # hyphen
            "a" * 65,  # too long
            "tickets.sub",  # dot
        ],
    )
    def test_rejects_names_the_platform_could_never_declare(self, name: str) -> None:
        with pytest.raises(MemoryClientError) as exc:
            validate_memory_type(name)
        assert str(exc.value) == (
            f"custom type name {name!r} must match ^[a-z][a-z0-9_]{{0,63}}$"
        )


class TestValidateTagSyntax:
    def test_accepts_scalars_and_one_level_nesting(self) -> None:
        validate_tag_syntax(
            {
                "queue": "billing",
                "priority": 3,
                "score": 1.5,
                "open": True,
                "profile": {"location": "nyc"},
                "profile.tier": "gold",
            }
        )

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(
            MemoryClientError, match=r"tag '' must have a non-empty path"
        ):
            validate_tag_syntax({"": "x"})

    def test_rejects_empty_segment(self) -> None:
        with pytest.raises(
            MemoryClientError, match=r"tag 'a\.' must have a non-empty path"
        ):
            validate_tag_syntax({"a.": "x"})

    def test_rejects_two_level_dotted(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag 'a\.b\.c': nesting is at most one level deep"
        ):
            validate_tag_syntax({"a.b.c": "x"})

    def test_rejects_dotted_key_with_mapping_value(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag 'a\.b': nesting is at most one level deep"
        ):
            validate_tag_syntax({"a.b": {"c": "x"}})

    def test_rejects_two_level_nested_mapping(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag 'a': nesting is at most one level deep"
        ):
            validate_tag_syntax({"a": {"b": {"c": "x"}}})

    def test_rejects_empty_group(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag group 'a' must contain at least one key"
        ):
            validate_tag_syntax({"a": {}})

    def test_rejects_non_scalar_value(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag 'a' must be a non-empty string, number, or boolean"
        ):
            validate_tag_syntax({"a": ["x"]})

    def test_rejects_empty_string_value(self) -> None:
        with pytest.raises(
            ValueError, match=r"tag 'a' must have a non-empty string value"
        ):
            validate_tag_syntax({"a": "  "})

    def test_none_tags_is_error_free_at_call_sites(self) -> None:
        # Call sites skip validation for None; the validator itself requires a mapping.
        validate_tag_syntax({})


class TestNonFiniteFloatTags:
    """JSON cannot carry NaN or Infinity, so the client must reject them.

    Left to the wire, Python's encoder raises an opaque error naming no tag and
    JavaScript rewrites the value to null; neither lets the server's own
    finite-number rule reach the caller.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_floats(self, value: float) -> None:
        with pytest.raises(MemoryClientError) as exc:
            validate_tag_syntax({"score": value})
        assert str(exc.value) == "tag 'score' must be a finite number"

    @pytest.mark.parametrize("value", [1.5, 0.0, -3.25, 1e308])
    def test_accepts_finite_floats(self, value: float) -> None:
        validate_tag_syntax({"score": value})

    def test_rejects_non_finite_inside_a_nested_group(self) -> None:
        with pytest.raises(MemoryClientError) as exc:
            validate_tag_syntax({"profile": {"score": float("nan")}})
        assert str(exc.value) == "tag 'profile.score' must be a finite number"

    def test_forwards_oversized_ints_for_the_server_to_judge(self) -> None:
        # The int64 bound is server-owned: the value survives JSON intact, so
        # the platform's own rejection is the one the caller should see.
        validate_tag_syntax({"count": 2**63})
