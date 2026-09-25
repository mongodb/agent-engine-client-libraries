"""Tests for the identity-resolution precedence algorithm."""

import pytest
from agent_engine_sdk_memory.errors import MemoryIdentityError
from agent_engine_sdk_memory.identity import resolve_identity
from agent_engine_sdk_memory.protocol import MemoryRequestContext


def test_call_arg_wins_over_bind_and_runtime():
    resolved = resolve_identity(
        call_args={"user_id": "u_call"},
        bind_ctx=MemoryRequestContext(user_id="u_bind"),
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
    )
    assert resolved["user_id"] == "u_call"


def test_bind_wins_over_runtime_when_call_absent():
    resolved = resolve_identity(
        call_args={"user_id": None},
        bind_ctx=MemoryRequestContext(user_id="u_bind"),
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
    )
    assert resolved["user_id"] == "u_bind"


def test_runtime_used_when_call_and_bind_absent():
    resolved = resolve_identity(
        call_args={"user_id": None},
        bind_ctx=None,
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
    )
    assert resolved["user_id"] == "u_rt"


def test_suppress_runtime_user_id_skips_only_user_id_runtime_fallback():
    # A read with explicit visibility must not borrow the ambient principal for
    # user_id, but other fields still fall through to the runtime context.
    resolved = resolve_identity(
        call_args={"user_id": None, "session_id": None},
        bind_ctx=None,
        runtime_ctx=MemoryRequestContext(user_id="u_rt", session_id="s_rt"),
        suppress_runtime_user_id=True,
    )
    assert resolved["user_id"] is None
    assert resolved["session_id"] == "s_rt"


def test_suppress_runtime_user_id_does_not_block_call_or_bind_user_id():
    resolved = resolve_identity(
        call_args={"user_id": "u_call"},
        bind_ctx=None,
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
        suppress_runtime_user_id=True,
    )
    assert resolved["user_id"] == "u_call"

    resolved_bind = resolve_identity(
        call_args={"user_id": None},
        bind_ctx=MemoryRequestContext(user_id="u_bind"),
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
        suppress_runtime_user_id=True,
    )
    assert resolved_bind["user_id"] == "u_bind"


def test_suppress_inherited_session_id_skips_both_bind_and_runtime():
    # Episodic reads must resolve session_id from the call arg only: a bound or
    # ambient session would silently filter out session-unscoped episodic memory.
    resolved = resolve_identity(
        call_args={"user_id": None, "session_id": None},
        bind_ctx=MemoryRequestContext(user_id="u_bind", session_id="s_bind"),
        runtime_ctx=MemoryRequestContext(session_id="s_rt"),
        suppress_inherited_session_id=True,
    )
    assert resolved["session_id"] is None
    # Other fields still inherit normally.
    assert resolved["user_id"] == "u_bind"


def test_suppress_inherited_session_id_keeps_explicit_call_session():
    resolved = resolve_identity(
        call_args={"session_id": "s_call"},
        bind_ctx=MemoryRequestContext(session_id="s_bind"),
        runtime_ctx=MemoryRequestContext(session_id="s_rt"),
        suppress_inherited_session_id=True,
    )
    assert resolved["session_id"] == "s_call"


def test_suppress_inherited_session_id_does_not_affect_user_or_agent_inheritance():
    resolved = resolve_identity(
        call_args={"user_id": None, "agent_id": None, "session_id": None},
        bind_ctx=MemoryRequestContext(user_id="u_bind", agent_id="a_bind"),
        runtime_ctx=MemoryRequestContext(user_id="u_rt", agent_id="a_rt"),
        suppress_inherited_session_id=True,
    )
    assert resolved["user_id"] == "u_bind"
    assert resolved["agent_id"] == "a_bind"
    assert resolved["session_id"] is None


def test_resolves_all_three_fields():
    resolved = resolve_identity(
        call_args={"session_id": "s_call"},
        bind_ctx=MemoryRequestContext(agent_id="a_bind"),
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
    )
    assert resolved == {
        "user_id": "u_rt",
        "agent_id": "a_bind",
        "session_id": "s_call",
    }


def test_unresolved_required_field_raises():
    with pytest.raises(MemoryIdentityError) as exc_info:
        resolve_identity(
            call_args={},
            bind_ctx=None,
            runtime_ctx=None,
            required=("user_id",),
        )
    assert "user_id" in str(exc_info.value)


def test_error_names_all_missing_required_fields():
    with pytest.raises(MemoryIdentityError) as exc_info:
        resolve_identity(
            call_args={"user_id": "u"},
            bind_ctx=None,
            runtime_ctx=None,
            required=("user_id", "session_id"),
        )
    message = str(exc_info.value)
    assert "session_id" in message
    assert "user_id" not in message


def test_satisfied_required_field_does_not_raise():
    resolved = resolve_identity(
        call_args={"user_id": "u"},
        bind_ctx=None,
        runtime_ctx=None,
        required=("user_id",),
    )
    assert resolved["user_id"] == "u"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_call_arg_falls_through_to_bind(blank):
    # Both empty and whitespace-only normalize to unset (`not value.strip()`).
    resolved = resolve_identity(
        call_args={"user_id": blank},
        bind_ctx=MemoryRequestContext(user_id="u_bind"),
        runtime_ctx=None,
    )
    assert resolved["user_id"] == "u_bind"


def test_empty_string_does_not_satisfy_required():
    with pytest.raises(MemoryIdentityError, match="user_id"):
        resolve_identity(
            call_args={"user_id": ""},
            bind_ctx=None,
            runtime_ctx=None,
            required=("user_id",),
        )


def test_empty_string_in_bind_ctx_falls_through_to_runtime():
    resolved = resolve_identity(
        call_args={},
        bind_ctx=MemoryRequestContext(user_id=""),
        runtime_ctx=MemoryRequestContext(user_id="u_rt"),
    )
    assert resolved["user_id"] == "u_rt"
