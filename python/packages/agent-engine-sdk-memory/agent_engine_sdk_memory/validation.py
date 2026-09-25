"""Public validation helpers for Memory client callers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_engine_sdk_memory.errors import MemoryIdentityError
from agent_engine_sdk_memory.models import FormatStyle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent_engine_sdk_memory.models import SourceSpec

__all__ = [
    "require_positive_max_tokens",
    "require_session_for_stm",
    "require_valid_format_style",
    "require_valid_sources",
]

_MSG = "max_tokens must be a positive integer"


def require_positive_max_tokens(max_tokens: int | None) -> None:
    """Raise ``ValueError`` unless ``max_tokens`` is ``None`` or a positive ``int``.

    Rejects ``bool`` (a subclass of ``int``), floats (including NaN/inf), and
    other non-integers so invalid budgets never reach the wire. Type hints alone
    do not stop these at runtime.
    """
    if max_tokens is None:
        return
    # ``type(...) is int`` rejects bool; ``isinstance(..., int)`` would not.
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError(_MSG)


def require_valid_format_style(format_style: FormatStyle | str | None) -> None:
    """Raise ``ValueError`` unless ``format_style`` is ``None`` or a known style.

    The server coerces valid values case-insensitively but raises ``ValueError``
    on an unknown one, which its routes surface as a 500. Failing fast here
    turns a typo into a clear local error before any network I/O.
    """
    if format_style is None:
        return
    # FormatStyle members are str instances whose string content is the value,
    # so .lower() normalizes plain strings and enum members alike.
    try:
        FormatStyle(format_style.lower())
    except ValueError:
        valid = ", ".join(sorted(style.value for style in FormatStyle))
        raise ValueError(
            f"format_style must be one of: {valid}; got {format_style!r}"
        ) from None


def require_valid_sources(sources: Sequence[SourceSpec]) -> list[SourceSpec]:
    """Validate the per-source spec list for ``build_context_from_sources``.

    Returns the specs as a list. Raises ``ValueError`` if it is empty or lists a
    source more than once, mirroring the server's one-spec-per-source rule so a
    caller gets a clear local error instead of a 422. Shared by the public facade
    and the direct ``MemoryClient`` so both per-source paths reject the same
    inputs before any network I/O.
    """
    specs = list(sources)
    if not specs:
        raise ValueError("sources must contain at least one SourceSpec")
    # ``source`` is a string here because SourceSpec uses use_enum_values.
    seen = {spec.source for spec in specs}
    if len(seen) != len(specs):
        raise ValueError("each source may be listed at most once in 'sources'")
    return specs


def require_session_for_stm(
    sources: Sequence[SourceSpec], session_id: str | None
) -> None:
    """Reject a ``stm`` source spec unless a non-blank ``session_id`` is present.

    Short-term memory is scoped by conversation, so retrieving it without a
    session would silently return nothing. Raising ``MemoryIdentityError`` here
    (at the SDK boundary, before any network I/O) turns that into an explicit
    error instead of an empty result. A blank/whitespace ``session_id`` counts
    as unset, matching identity resolution elsewhere. Sources are strings here
    because SourceSpec uses use_enum_values.
    """
    if session_id and session_id.strip():
        return
    if any(spec.source == "stm" for spec in sources):
        raise MemoryIdentityError(
            "the 'stm' source requires a session_id; pass session_id= or bind "
            "one via the execution context"
        )
