"""Request-local ownership for Memory reads performed by registered Tools."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class _ToolMemoryReadOwnership:
    active: bool = True


_CURRENT_OWNERSHIP: ContextVar[_ToolMemoryReadOwnership | None] = ContextVar(
    "current_tool_memory_read_ownership", default=None
)


def is_memory_read_owned_by_tool() -> bool:
    """Return whether a live registered Tool callback owns this Memory read."""
    ownership = _CURRENT_OWNERSHIP.get()
    return ownership is not None and ownership.active


@contextmanager
def tool_memory_read_ownership() -> Iterator[None]:
    """Scope Memory-read ownership to one registered Tool callback."""
    ownership = _ToolMemoryReadOwnership()
    token = _CURRENT_OWNERSHIP.set(ownership)
    try:
        yield
    finally:
        # Copied contexts retain this holder, so descendants lose ownership
        # when the registered Tool callback returns.
        ownership.active = False
        _CURRENT_OWNERSHIP.reset(token)
