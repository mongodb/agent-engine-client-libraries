"""Wire-body builders for the custom-type routes.

Shared by the two HTTP CRUD clients (``MemoryClient`` and
``_EmptyTenancyCrudClient``) so both send identical request bodies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def custom_save_body(
    *,
    content: str,
    tags: Mapping[str, Any] | None,
    contextual_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"content": content}
    if tags is not None:
        body["tags"] = dict(tags)
    if contextual_metadata is not None:
        body["contextual_metadata"] = contextual_metadata
    return body


def custom_retrieve_body(
    *,
    query: str,
    top_k: int,
    tags: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": query, "top_k": top_k}
    if tags is not None:
        body["tags"] = dict(tags)
    return body
