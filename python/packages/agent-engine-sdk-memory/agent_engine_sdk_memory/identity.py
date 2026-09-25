"""Identity-resolution algorithm: call arg > bind context > runtime context."""

from __future__ import annotations

from agent_engine_sdk_memory.errors import MemoryIdentityError
from agent_engine_sdk_memory.protocol import MemoryRequestContext

_FIELDS = ("user_id", "agent_id", "session_id")


def _normalize(value: str | None) -> str | None:
    # A blank or whitespace-only value carries no identity; treating it as
    # unset keeps it from satisfying `required` or shadowing a bound value,
    # matching the blank-rejection convention of the internal HTTP client.
    if value is None or not value.strip():
        return None
    return value


def resolve_identity(
    *,
    call_args: dict[str, str | None],
    bind_ctx: MemoryRequestContext | None,
    runtime_ctx: MemoryRequestContext | None,
    required: tuple[str, ...] = (),
    suppress_runtime_user_id: bool = False,
    suppress_inherited_session_id: bool = False,
) -> dict[str, str | None]:
    """Resolve identity fields by precedence: call arg > bind ctx > runtime ctx.

    Blank or whitespace-only values are treated as unset at every tier.

    Two optional guards drop tiers for one field each. They drop different
    tiers because they protect different things.

    ``suppress_runtime_user_id`` drops the runtime tier for ``user_id``. A read
    that asks for a broad visibility must not borrow the ambient principal as
    its filter. A ``user_id`` set by the caller or by bind is deliberate and
    still applies.

    ``suppress_inherited_session_id`` drops the bind and runtime tiers for
    ``session_id``, so only an explicit call arg applies. A ``session_id``
    scopes conversation I/O: the turns and short-term context of one thread. It
    is not a long-term search filter. Episodic search matches ``session_id``
    exactly, and consolidated episodic memory is stored without one, so an
    inherited session matches nothing and returns empty. Pass ``session_id`` on
    the search to scope it to one conversation.
    """
    resolved: dict[str, str | None] = {}
    for field in _FIELDS:
        value = _normalize(call_args.get(field))
        # Two per-field guards, both off by default. skip_inherited drops the
        # bind and runtime tiers for session_id. suppress_runtime_user_id drops
        # the runtime tier for user_id.
        skip_inherited = suppress_inherited_session_id and field == "session_id"
        if value is None and bind_ctx is not None and not skip_inherited:
            value = _normalize(getattr(bind_ctx, field, None))
        skip_runtime = skip_inherited or (
            suppress_runtime_user_id and field == "user_id"
        )
        if value is None and runtime_ctx is not None and not skip_runtime:
            value = _normalize(getattr(runtime_ctx, field, None))
        resolved[field] = value

    missing = [field for field in required if resolved.get(field) is None]
    if missing:
        raise MemoryIdentityError(
            f"could not resolve required identity field(s): {', '.join(missing)}"
        )
    return resolved


__all__ = ["resolve_identity"]
