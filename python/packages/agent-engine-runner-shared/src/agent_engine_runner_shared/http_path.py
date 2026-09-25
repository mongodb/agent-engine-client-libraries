"""URL path helpers for outbound platform HTTP calls.

Shared choke point for percent-encoding one path segment. Framework adapters
must route every interpolated identifier through :func:`quote_path_segment`
instead of hand-rolling quoting: bare ``quote``/``encodeURIComponent`` leaves
``.``/``..`` untouched and string-concatenated identifiers forge paths.
"""

from __future__ import annotations

from urllib.parse import quote, unquote

__all__ = ["quote_path_segment"]

_MAX_DECODE_ROUNDS = 8


def quote_path_segment(value: str) -> str:
    """Percent-encode one path segment; reject separators and ``.`` / ``..``.

    Decode repeatedly before quoting so encoded, double-encoded, and mixed
    traversal forms cannot survive a later routing decode.
    """
    decoded = _fully_decode(value)
    if not decoded or any(sep in decoded for sep in "/\\?#") or decoded in {".", ".."}:
        raise RuntimeError("URL path segment is invalid")
    return quote(decoded, safe="")


def _fully_decode(value: str) -> str:
    decoded = value
    for _ in range(_MAX_DECODE_ROUNDS):
        _reject_malformed_percent(decoded)
        try:
            next_decoded = unquote(decoded, errors="strict")
        except UnicodeDecodeError as error:
            raise RuntimeError("URL path segment is invalid") from error
        if next_decoded == decoded:
            return decoded
        decoded = next_decoded
    raise RuntimeError("URL path segment is invalid")


def _reject_malformed_percent(value: str) -> None:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        hex_pair = value[index + 1 : index + 3]
        if len(hex_pair) != 2 or not all(
            character in "0123456789abcdefABCDEF" for character in hex_pair
        ):
            raise RuntimeError("URL path segment is invalid")
        index += 3
