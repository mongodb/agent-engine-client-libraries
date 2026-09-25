"""Per-session isolation for Tool-Pod filesystem handlers.

Covers every filesystem handler, concurrent session interleavings,
attack paths (traversal, absolute paths, symlinks), the resume case,
and the no-session fallback.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import os
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def handlers(monkeypatch, tmp_path):
    """Reload toolpod_handlers with WORKSPACE_DIR pointing at tmp_path.

    Reload is required because ``WORKSPACE_DIR`` is captured at import time
    from the env. After the test, reload again so other tests are not
    polluted by our env override.
    """
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    import agent_engine_runner_shared.toolpod_handlers as mod

    mod = importlib.reload(mod)
    yield mod
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    importlib.reload(mod)


def _enter_session(session_id: str, *, workspace_id: str = "ws-shared-agent") -> Any:
    """Push the same execution context that AER pushes on every request."""
    from agent_engine_runner_shared.context import set_execution_context

    return set_execution_context(
        execution_id=f"exec-{session_id}",
        wrapper=Mock(),
        oe_url="http://oe.local",
        user_id=f"user-for-{session_id}",
        session_id=session_id,
        workspace_id=workspace_id,
    )


def _leave_session(tokens: Any) -> None:
    from agent_engine_runner_shared.context import clear_execution_context

    clear_execution_context(tokens)


def _session_dir_for(handlers, session_id: str) -> Path:
    """Return the on-disk per-session workspace directory for ``session_id``.

    The directory name is derived from the session id by the handler module
    (currently a hash), so tests should not hardcode the layout — ask the
    module via its public effective-workspace-dir helper instead.
    """
    tokens = _enter_session(session_id)
    try:
        return Path(handlers._effective_workspace_dir())
    finally:
        _leave_session(tokens)


def _ls_names(handlers, where: str = ".") -> list[str]:
    """Return basename names visible to the current session at *where*."""
    result = handlers.filesystem_ls(where)
    if "error" in result:
        return []
    return [Path(e["path"]).name for e in result["entries"]]


# ---------------------------------------------------------------------------
# 1. Every filesystem handler isolates by session
# ---------------------------------------------------------------------------


def test_resolve_produces_distinct_paths_per_session(handlers):
    """Resolver-level structural assertion: two sessions cannot collide on
    the same on-disk path for the same input."""
    tokens = _enter_session("alice")
    try:
        path_a = handlers._resolve("notes.txt")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        path_b = handlers._resolve("notes.txt")
    finally:
        _leave_session(tokens)

    assert path_a != path_b, (
        f"_resolve('notes.txt') returned the same path {path_a!r} for two sessions"
    )


def test_write_then_other_session_ls_does_not_see_file(handlers):
    tokens = _enter_session("alice")
    try:
        assert "error" not in handlers.filesystem_write("note.txt", "alice")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        names = _ls_names(handlers)
    finally:
        _leave_session(tokens)

    assert "note.txt" not in names, names


def test_write_then_other_session_read_returns_error(handlers):
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("note.txt", "alice secret")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        result = handlers.filesystem_read("note.txt")
    finally:
        _leave_session(tokens)

    # Either a clean error, or empty content — but never alice's bytes.
    if "error" not in result:
        assert "alice secret" not in result.get("content", "")


def test_edit_in_one_session_does_not_affect_other_session_file(handlers):
    """Each session has its own copy of ``note.txt``; an edit in alice's
    session must not modify bob's file (which can have the same name)."""
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("note.txt", "hello alice")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        handlers.filesystem_write("note.txt", "hello bob")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("alice")
    try:
        result = handlers.filesystem_edit("note.txt", "alice", "ALICE")
        assert result.get("occurrences", 0) == 1
        contents_after = handlers.filesystem_read("note.txt")["content"]
    finally:
        _leave_session(tokens)
    assert contents_after == "hello ALICE"

    tokens = _enter_session("bob")
    try:
        bob_after = handlers.filesystem_read("note.txt")["content"]
    finally:
        _leave_session(tokens)
    assert bob_after == "hello bob", "alice's edit leaked into bob's session"


def test_glob_only_returns_caller_session_files(handlers):
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("alice-only.md", "x")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        result = handlers.filesystem_glob("**/*.md")
    finally:
        _leave_session(tokens)

    matches = [Path(m["path"]).name for m in result.get("matches", [])]
    assert "alice-only.md" not in matches, matches


def test_grep_only_searches_caller_session_files(handlers):
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("notes.txt", "needle-alice")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        result = handlers.filesystem_grep("needle-alice")
    finally:
        _leave_session(tokens)

    assert result.get("matches", []) == [], result


def test_download_does_not_expose_other_session_files(handlers):
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("attachments/data.bin", "ALICE")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        result = handlers.filesystem_download("attachments/data.bin")
    finally:
        _leave_session(tokens)

    if "error" not in result:
        decoded = base64.b64decode(result["content_base64"]).decode()
        assert "ALICE" not in decoded


def test_shell_execute_runs_under_caller_session_cwd(handlers):
    """Sanity that shell_execute actually cd's into the per-session dir."""
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("hello-from-alice.txt", "x")
        result = handlers.shell_execute("ls -1")
    finally:
        _leave_session(tokens)
    assert "hello-from-alice.txt" in result["output"]

    tokens = _enter_session("bob")
    try:
        result = handlers.shell_execute("ls -1")
    finally:
        _leave_session(tokens)
    assert "hello-from-alice.txt" not in result["output"], "bob's shell saw alice's files via cwd"


# ---------------------------------------------------------------------------
# 2. Concurrent sessions in the same process (the AER common case)
# ---------------------------------------------------------------------------


def test_concurrent_async_sessions_do_not_leak(handlers):
    """ContextVar-based isolation must hold across concurrent asyncio tasks."""
    barrier_started = asyncio.Event()
    barrier_writes_done = asyncio.Event()
    write_count = {"done": 0}

    async def session_task(session_id: str, payload: str) -> list[str]:
        tokens = _enter_session(session_id)
        try:
            handlers.filesystem_write("scratch.txt", payload)
            write_count["done"] += 1
            if write_count["done"] == 2:
                barrier_writes_done.set()
            barrier_started.set()
            await barrier_writes_done.wait()
            # Both tasks have written by now. Each one's ls should still
            # show only its own file.
            return _ls_names(handlers)
        finally:
            _leave_session(tokens)

    async def main():
        return await asyncio.gather(
            session_task("alice-async", "ALICE"),
            session_task("bob-async", "BOB"),
        )

    alice_listing, bob_listing = asyncio.run(main())

    # Both saw exactly one file (their own scratch.txt) and no others.
    assert alice_listing == ["scratch.txt"], alice_listing
    assert bob_listing == ["scratch.txt"], bob_listing

    # And the on-disk subtrees actually contain different content.
    tokens = _enter_session("alice-async")
    try:
        alice_content = handlers.filesystem_read("scratch.txt")["content"]
    finally:
        _leave_session(tokens)
    tokens = _enter_session("bob-async")
    try:
        bob_content = handlers.filesystem_read("scratch.txt")["content"]
    finally:
        _leave_session(tokens)
    assert alice_content == "ALICE"
    assert bob_content == "BOB"


# ---------------------------------------------------------------------------
# 3. Attack attempts: cannot escape into other sessions
# ---------------------------------------------------------------------------


def test_relative_traversal_to_other_session_rejected(handlers):
    """``../<other-session>/...`` from inside one session must be rejected."""
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("secret.txt", "alice secret")
    finally:
        _leave_session(tokens)

    tokens = _enter_session("bob")
    try:
        with pytest.raises(ValueError, match="escapes workspace sandbox"):
            handlers._resolve("../alice/secret.txt")
    finally:
        _leave_session(tokens)


def test_absolute_path_to_other_session_rejected(handlers, tmp_path):
    """An absolute path naming another session's subtree is inside the
    reserved sessions namespace but not the caller's subtree, so it must
    raise rather than silently expose another session's bytes."""
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("secret.txt", "alice secret")
    finally:
        _leave_session(tokens)

    alice_abs = str(_session_dir_for(handlers, "alice") / "secret.txt")

    tokens = _enter_session("bob")
    try:
        with pytest.raises(ValueError, match="escapes workspace sandbox"):
            handlers._resolve(alice_abs)
        # The handler-level read returns the structured error envelope.
        result = handlers.filesystem_read(alice_abs)
        assert "error" in result, result
        assert "alice secret" not in result.get("content", "")
    finally:
        _leave_session(tokens)


def test_symlink_into_other_session_rejected(handlers, tmp_path):
    """A symlink inside one session's subtree pointing at another session's
    subtree must be rejected by the realpath-based boundary check."""
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("flag.txt", "ALICE-FLAG")
    finally:
        _leave_session(tokens)

    alice_dir = _session_dir_for(handlers, "alice")

    tokens = _enter_session("bob")
    try:
        handlers.filesystem_write("placeholder.txt", "x")
        bob_dir = _session_dir_for(handlers, "bob")
        os.symlink(str(alice_dir), str(bob_dir / "alice-link"))

        with pytest.raises(ValueError, match="escapes workspace sandbox"):
            handlers._resolve("alice-link/flag.txt")
    finally:
        _leave_session(tokens)


def test_shell_execute_traversal_to_other_session_currently_permitted(handlers):
    """KNOWN GAP: pins the current *permissive* behavior.

    Unlike the filesystem handlers above, ``shell_execute`` takes an opaque
    command string, so it cannot route through ``_resolve`` — ``cwd`` is only
    the child's default directory and the process is not OS-confined. A
    ``..`` traversal (or absolute path) from one session's shell reaches a
    sibling session on the same Tool Pod today.

    This test is the tripwire for the planned bubblewrap-based sandbox: the
    day OS-level confinement lands, this test must fail, and the assertion
    must then be inverted to require rejection — matching the ``_resolve``
    traversal tests above.
    """
    tokens = _enter_session("alice")
    try:
        handlers.filesystem_write("secret.txt", "alice secret")
    finally:
        _leave_session(tokens)
    alice_file = _session_dir_for(handlers, "alice") / "secret.txt"

    tokens = _enter_session("bob")
    try:
        bob_dir = _session_dir_for(handlers, "bob")
        rel = os.path.relpath(alice_file, bob_dir)
        assert rel.startswith(".."), rel  # the traversal form under test
        result = handlers.shell_execute(f"cat {shlex.quote(rel)}")
    finally:
        _leave_session(tokens)

    # Permissive today: bob's shell reads alice's file. Invert on sandbox.
    assert result.get("exit_code") == 0, result
    assert "alice secret" in result["output"], result


@pytest.mark.parametrize(
    "session_id",
    [
        ".",
        "..",
        "alice/../bob",
        "team/thread-1",
        "hello world",
        "café",
        "x" * 1024,
    ],
)
def test_arbitrary_session_id_admitted_and_isolated(handlers, session_id):
    """Session ids the upstream contract admits — slashes, unicode, dot
    traversal forms, long strings — must produce a usable per-session
    workspace, not raise. The on-disk slot is filesystem-safe by
    construction even though the raw input isn't."""
    tokens = _enter_session(session_id)
    try:
        resolved = handlers._resolve("x.txt")
    finally:
        _leave_session(tokens)

    sessions_root = Path(handlers.WORKSPACE_DIR) / ".sessions"
    assert str(resolved).startswith(str(sessions_root) + os.sep), resolved
    slot = Path(resolved).relative_to(sessions_root).parts[0]
    assert slot not in {".", ".."}, slot
    assert os.sep not in slot, slot


def test_session_ids_normalize_to_nfc(handlers):
    """Session ids that differ only by Unicode normalization form (NFC vs
    NFD of the same logical string) resolve to the same on-disk slot, so
    a client that drifts between normalizations across requests sees
    continuity instead of a silently fresh workspace."""
    nfc = "café"
    nfd = "café"
    assert nfc != nfd

    tokens = _enter_session(nfc)
    try:
        path_nfc = handlers._resolve("x.txt")
    finally:
        _leave_session(tokens)

    tokens = _enter_session(nfd)
    try:
        path_nfd = handlers._resolve("x.txt")
    finally:
        _leave_session(tokens)

    assert path_nfc == path_nfd


def test_session_write_outside_own_subtree_rejected(handlers, tmp_path):
    """A session must not be able to create files at the WORKSPACE_DIR
    root (or anywhere outside its own ``.sessions/<slot>/`` subtree),
    otherwise another session would observe them."""
    outside_target = str(tmp_path / "shared.txt")

    tokens = _enter_session("alice")
    try:
        result = handlers.filesystem_write(outside_target, "alice writes outside")
    finally:
        _leave_session(tokens)
    assert "error" in result, result
    assert "escapes workspace sandbox" in result["error"], result
    assert not (tmp_path / "shared.txt").exists(), "outside file was created"


def test_session_edit_outside_own_subtree_rejected(handlers, tmp_path):
    """A session must not be able to ``filesystem_edit`` content outside
    its own ``.sessions/<slot>/`` subtree, even if a sibling file pre-exists
    at the workspace root."""
    sibling_dir = tmp_path / "security"
    sibling_dir.mkdir()
    sibling_file = sibling_dir / "notes.md"
    sibling_file.write_text("# checklist v1")

    tokens = _enter_session("alice")
    try:
        result = handlers.filesystem_edit(str(sibling_file), "v1", "v2")
    finally:
        _leave_session(tokens)
    assert "error" in result, result
    assert "escapes workspace sandbox" in result["error"], result
    assert sibling_file.read_text() == "# checklist v1", "outside file was modified"


@pytest.mark.parametrize("op", ["filesystem_write", "filesystem_edit"])
def test_mutating_handlers_reject_other_session_absolute_path(handlers, tmp_path, op):
    """Mutating handlers must reject an absolute path inside another
    session's subtree — the boundary check rejects any path outside the
    caller's own ``.sessions/<slot>/`` subtree regardless of operation."""
    tokens = _enter_session("bob")
    try:
        handlers.filesystem_write("notes.txt", "bob v1")
    finally:
        _leave_session(tokens)
    bob_notes = _session_dir_for(handlers, "bob") / "notes.txt"
    bob_path = str(bob_notes)

    tokens = _enter_session("alice")
    try:
        if op == "filesystem_write":
            result = handlers.filesystem_write(bob_path, "alice clobber")
        else:
            result = handlers.filesystem_edit(bob_path, "bob v1", "alice clobber")
    finally:
        _leave_session(tokens)

    assert "error" in result, result
    assert "escapes workspace sandbox" in result["error"], result
    assert bob_notes.read_text() == "bob v1"


# ---------------------------------------------------------------------------
# 4. Same-session resume preserves files (not over-isolating)
# ---------------------------------------------------------------------------


def test_same_session_id_two_invocations_share_files(handlers):
    """The flip side of isolation: a *resumed* session (same session_id)
    must see its own files. Otherwise we've over-isolated and broken
    multi-turn conversation continuity."""
    tokens = _enter_session("carol")
    try:
        handlers.filesystem_write("draft.md", "v1")
    finally:
        _leave_session(tokens)

    # Same session_id → same effective workspace.
    tokens = _enter_session("carol")
    try:
        result = handlers.filesystem_read("draft.md")
    finally:
        _leave_session(tokens)
    assert result.get("content") == "v1"


# ---------------------------------------------------------------------------
# 5. No-session fallback — direct calls still work outside AER
# ---------------------------------------------------------------------------


def test_no_session_falls_back_to_base_workspace_dir(handlers, tmp_path):
    """Direct invocations outside an AER request (dev scripts, low-level
    tests) have no session in context. They must still resolve under the
    base ``WORKSPACE_DIR``, not crash."""
    # Don't enter a session.
    resolved = handlers._resolve("scratch.txt")
    assert resolved == str(tmp_path / "scratch.txt"), resolved


def test_session_and_no_session_subtrees_are_disjoint(handlers, tmp_path):
    """No-session callers see the reserved ``.sessions`` namespace as a
    directory at the root but cannot see files written inside any session."""
    tokens = _enter_session("dave")
    try:
        handlers.filesystem_write("inside.txt", "scoped")
    finally:
        _leave_session(tokens)

    listing = handlers.filesystem_ls(".")
    names_at_root = [Path(e["path"]).name for e in listing.get("entries", [])]
    assert "inside.txt" not in names_at_root, names_at_root
    assert ".sessions" in names_at_root, names_at_root


# ---------------------------------------------------------------------------
# 6. End-to-end through the AER entry point shape
# ---------------------------------------------------------------------------


def test_aer_style_request_pair_does_not_leak(handlers):
    """Drive two simulated AER requests through the actual context entry
    point AER uses, mirroring two API calls hitting the same Tool Pod."""

    def aer_request(thread_id: str, workspace_id: str, op: str, *args: Any) -> Any:
        from agent_engine_runner_shared.context import (
            clear_execution_context,
            set_execution_context,
        )

        tokens = set_execution_context(
            execution_id=f"exec-{thread_id}",
            wrapper=Mock(),
            oe_url="http://oe.local",
            user_id="auth-user",
            session_id=thread_id,
            workspace_id=workspace_id,
        )
        try:
            return getattr(handlers, op)(*args)
        finally:
            clear_execution_context(tokens)

    shared_workspace = "ws-helpdesk-bot"

    # Request 1: thread A writes a private note.
    aer_request("thread-A", shared_workspace, "filesystem_write", "private.md", "A only")
    # Request 2: thread B (same workspace, different thread) lists.
    listing = aer_request("thread-B", shared_workspace, "filesystem_ls", ".")
    names = [Path(e["path"]).name for e in listing.get("entries", [])]
    assert "private.md" not in names, f"cross-thread leak through AER entry point. Listing: {names}"
    # Request 3: thread A resumes — must see its own file.
    read_back = aer_request("thread-A", shared_workspace, "filesystem_read", "private.md")
    assert read_back.get("content") == "A only"


# ---------------------------------------------------------------------------
# 7. Session boundary × read-only resource roots
# ---------------------------------------------------------------------------


def test_session_can_read_bundled_skill_via_readonly_root(monkeypatch, tmp_path):
    """A session caller can still read a bundled skill that lives outside
    WORKSPACE_DIR via the read-only resource roots. Per-session narrowing
    must not block reads admitted by ``allow_readonly_roots=True``.
    """
    agent_dir = tmp_path / "local-agent"
    skill_dir = agent_dir / "skills" / "local-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("# Local skill content")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
    import agent_engine_runner_shared.toolpod_handlers as mod

    mod = importlib.reload(mod)
    try:
        tokens = _enter_session("alice")
        try:
            result = mod.filesystem_read(str(skill_md))
        finally:
            _leave_session(tokens)
    finally:
        monkeypatch.delenv("AGENTIC_AGENT_WORKDIR", raising=False)
        monkeypatch.delenv("WORKSPACE_DIR", raising=False)
        importlib.reload(mod)

    assert result.get("content") == "# Local skill content", result


def test_session_cannot_write_to_bundled_skill_via_readonly_root(monkeypatch, tmp_path):
    """The flip side: a session caller must NOT be able to write or edit
    a bundled skill file. ``filesystem_write`` does not pass
    ``allow_readonly_roots=True`` and the absolute path falls outside the
    session's subtree, so the boundary check rejects it.
    """
    agent_dir = tmp_path / "local-agent"
    skill_dir = agent_dir / "skills" / "local-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("# Original content")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
    import agent_engine_runner_shared.toolpod_handlers as mod

    mod = importlib.reload(mod)
    try:
        tokens = _enter_session("alice")
        try:
            result = mod.filesystem_write(str(skill_md), "# Overwritten by alice")
        finally:
            _leave_session(tokens)
    finally:
        monkeypatch.delenv("AGENTIC_AGENT_WORKDIR", raising=False)
        monkeypatch.delenv("WORKSPACE_DIR", raising=False)
        importlib.reload(mod)

    assert "error" in result, result
    assert skill_md.read_text() == "# Original content", "bundled skill was modified"
