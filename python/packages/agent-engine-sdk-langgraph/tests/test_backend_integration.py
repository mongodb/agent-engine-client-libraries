"""Integration tests for ``AgentEngineToolPodBackend``.

The mock-heavy tests in ``test_backend_protocol.py`` stub out
``SecureToolWrapper.execute_tool`` with ``MagicMock`` and feed the backend
hand-rolled JSON dicts. Those tests catch field-shape regressions in the
parser but never exercise the **wiring**: backend → wrapper → OE → real
``agent_engine_runner_shared.toolpod_handlers`` → tmp filesystem.

This module wires the real pieces together. Each test:

* Spins up a small ``http.server`` thread playing the OE.
* Builds a real ``SecureToolWrapper`` pointed at the fake OE URL.
* Sets the ``current_wrapper`` ContextVar so the backend resolves it.
* Drives the backend through one protocol method.
* Asserts on real artifacts on disk (and on what the OE saw on the wire).

A failure here means the backend cannot actually drive a Tool Pod — the
exact bug class the mock-heavy tests can't catch.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from agent_engine_sdk_langgraph.backends import toolpod
from agent_engine_sdk_langgraph.backends.toolpod import AgentEngineToolPodBackend
from agent_engine_runner_shared import toolpod_handlers
from agent_engine_runner_shared.context import current_wrapper
from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper


@dataclass
class _LegacyGrepResult:
    """deepagents 0.5.x GrepResult: no ``truncated`` field."""

    error: str | None = None
    matches: list[Any] | None = None


@dataclass
class _LegacyGlobResult:
    """deepagents 0.5.x GlobResult: no ``truncated`` field."""

    error: str | None = None
    matches: list[Any] | None = None


# ---------------------------------------------------------------------------
# Fake OE — dispatches POST /tool/execute to real toolpod_handlers
# ---------------------------------------------------------------------------


class _FakeOE:
    """A minimal OE that auto-approves and runs the named handler.

    Tests can override ``deny_reason`` or ``error_message`` to exercise the
    PolicyDenied / ToolExecutionError branches of ``SecureToolWrapper`` with
    a real HTTP round-trip rather than a stubbed exception.
    """

    def __init__(
        self,
        handlers: dict[str, Callable[..., dict[str, Any]]],
    ) -> None:
        self.handlers = handlers
        self.deny_reason: str | None = None
        self.error_message: str | None = None
        self.requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None, "FakeOE.start() not called"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                response = outer._respond(body)
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _respond(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.deny_reason is not None:
            return {"proceed": False, "reason": self.deny_reason}
        if self.error_message is not None:
            return {"proceed": True, "status": "error", "error": self.error_message}
        tool_name = body.get("tool_name", "")
        handler = self.handlers.get(tool_name)
        if handler is None:
            return {
                "proceed": True,
                "status": "error",
                "error": f"unknown tool {tool_name!r}",
            }
        try:
            result = handler(**body.get("arguments", {}))
            return {"proceed": True, "status": "success", "result": result}
        except Exception as exc:
            return {"proceed": True, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``toolpod_handlers.WORKSPACE_DIR`` at a per-test tmp directory.

    The handler module captures ``WORKSPACE_DIR`` as a module-level constant
    at import time; both ``_resolve`` and ``_is_within_workspace`` read it
    at call time, so a single ``monkeypatch.setattr`` redirects every
    filesystem op for this test.
    """
    monkeypatch.setattr(toolpod_handlers, "WORKSPACE_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture()
def fake_oe(workspace: Path) -> Iterator[_FakeOE]:
    """A running fake OE wired to the real built-in handlers."""
    oe = _FakeOE(handlers=dict(toolpod_handlers._HANDLERS))
    oe.start()
    try:
        yield oe
    finally:
        oe.stop()


@contextmanager
def _bound_wrapper(oe_url: str) -> Iterator[SecureToolWrapper]:
    """Install a real ``SecureToolWrapper`` on the ``current_wrapper``
    ContextVar for the duration of the block.
    """
    wrapper = SecureToolWrapper(oe_url=oe_url, execution_id="exec-test")
    token = current_wrapper.set(wrapper)
    try:
        yield wrapper
    finally:
        current_wrapper.reset(token)


@pytest.fixture()
def backend(fake_oe: _FakeOE) -> Iterator[AgentEngineToolPodBackend]:
    """A backend wired to a real wrapper that talks to the fake OE."""
    with _bound_wrapper(fake_oe.url):
        yield AgentEngineToolPodBackend()


# ---------------------------------------------------------------------------
# Read ops — ls, read, grep, glob
# ---------------------------------------------------------------------------


class TestReadOpsAgainstRealFilesystem:
    def test_ls_returns_real_directory_entries(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "a.txt").write_text("hello")
        (workspace / "sub").mkdir()

        result = backend.ls(".")

        assert result.error is None
        assert result.entries is not None
        paths = {e["path"] for e in result.entries}
        assert paths == {str(workspace / "a.txt"), str(workspace / "sub")}
        # Directory flag must round-trip from the handler's ``is_dir`` field.
        sub = next(e for e in result.entries if e["path"].endswith("sub"))
        assert sub.get("is_dir") is True

    def test_read_returns_file_content(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "f.txt").write_text("line1\nline2\nline3\n")

        result = backend.read("f.txt")

        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "line1\nline2\nline3\n"

    def test_read_offset_and_limit_slice_lines(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "f.txt").write_text(
            "\n".join(f"line{i}" for i in range(10)) + "\n"
        )

        result = backend.read("f.txt", offset=2, limit=3)

        assert result.file_data is not None
        assert result.file_data["content"] == "line2\nline3\nline4\n"

    def test_read_missing_file_returns_sanitized_error(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        result = backend.read("nope.txt")

        assert result.error is not None
        # The handler-supplied OSError text carries the resolved workspace
        # path; ``_sanitize_handler_error`` rewrites paths to ``<path>`` /
        # ``<workspace>`` before the agent sees it. Either replacement is
        # acceptable; what's not acceptable is the raw workspace prefix.
        assert "<workspace>" in result.error or "<path>" in result.error
        assert str(workspace) not in result.error

    def test_grep_finds_matches_across_files(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "a.py").write_text("# TODO fix\nok\n")
        (workspace / "b.py").write_text("nothing here\n")

        result = backend.grep("TODO")

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["path"].endswith("a.py")
        assert result.matches[0]["line"] == 1
        assert "TODO fix" in result.matches[0]["text"]

    def test_grep_glob_filter_uses_handler_kwarg(
        self, workspace: Path, backend: AgentEngineToolPodBackend, fake_oe: _FakeOE
    ) -> None:
        """Pin the ``glob`` wire-name. A rename on either side would surface
        as a TypeError when the handler is invoked through the real OE.
        """
        (workspace / "a.py").write_text("TODO here\n")
        (workspace / "a.txt").write_text("TODO here\n")

        result = backend.grep("TODO", glob="*.py")

        assert result.error is None
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {str(workspace / "a.py")}
        # And the wire actually carried the kwarg under its expected name.
        recorded = fake_oe.requests[0]
        assert "glob" in recorded["arguments"]
        assert recorded["arguments"]["glob"] == "*.py"

    def test_glob_returns_real_matches(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "src").mkdir()
        (workspace / "src" / "main.py").write_text("x")
        (workspace / "src" / "main.txt").write_text("y")

        result = backend.glob("**/*.py")

        assert result.error is None
        assert result.matches is not None
        names = {m["path"] for m in result.matches}
        assert names == {str(workspace / "src" / "main.py")}

    def test_grep_max_count_caps_matches_and_flags_truncated(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "a.py").write_text("TODO one\nTODO two\nTODO three\n")

        uncapped = backend.grep("TODO")
        assert uncapped.error is None
        assert uncapped.matches is not None
        assert len(uncapped.matches) == 3
        assert uncapped.truncated is False

        capped = backend.grep("TODO", max_count=1)
        assert capped.error is None
        assert capped.matches is not None
        assert len(capped.matches) == 1
        assert capped.truncated is True

    def test_glob_without_path_omits_the_wire_path(
        self, workspace: Path, backend: AgentEngineToolPodBackend, fake_oe: _FakeOE
    ) -> None:
        (workspace / "a.py").write_text("x")

        result = backend.glob("*.py")

        assert result.error is None
        assert result.matches is not None
        # The protocol treats a missing path as the working directory; the
        # handler's own default resolves the workspace root.
        assert "path" not in fake_oe.requests[0]["arguments"]

    def test_grep_tolerates_legacy_result_type(
        self,
        workspace: Path,
        backend: AgentEngineToolPodBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An agent on deepagents 0.5.x must not crash on a successful grep."""
        (workspace / "a.py").write_text("TODO one\nTODO two\n")
        monkeypatch.setattr(toolpod, "GrepResult", _LegacyGrepResult)

        result = backend.grep("TODO", max_count=1)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1

    def test_glob_tolerates_legacy_result_type(
        self,
        workspace: Path,
        backend: AgentEngineToolPodBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An agent on deepagents 0.5.x must not crash on a successful glob."""
        (workspace / "a.py").write_text("x")
        monkeypatch.setattr(toolpod, "GlobResult", _LegacyGlobResult)

        result = backend.glob("*.py")

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1


# ---------------------------------------------------------------------------
# Write ops — write, edit
# ---------------------------------------------------------------------------


class TestWriteOpsAgainstRealFilesystem:
    def test_write_creates_file(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        result = backend.write("new.txt", "hello world")

        assert result.error is None
        assert result.path == str(workspace / "new.txt")
        assert (workspace / "new.txt").read_text() == "hello world"

    def test_write_rejects_existing_file_via_handler(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "exists.txt").write_text("old")

        result = backend.write("exists.txt", "new")

        # Per the create-only contract the handler enforces — existing files
        # must use ``edit`` so wrong-region rewrites are caught by the
        # unique-match guard.
        assert result.error is not None
        assert "edit" in result.error.lower()
        assert (workspace / "exists.txt").read_text() == "old"

    def test_edit_replaces_unique_match(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "f.txt").write_text("alpha beta gamma\n")

        result = backend.edit("f.txt", "beta", "BETA")

        assert result.error is None
        assert result.occurrences == 1
        assert (workspace / "f.txt").read_text() == "alpha BETA gamma\n"

    def test_edit_rejects_non_unique_match_without_replace_all(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "f.txt").write_text("foo foo\n")

        result = backend.edit("f.txt", "foo", "bar")

        assert result.error is not None
        assert "not unique" in result.error or "longer anchor" in result.error
        # File untouched — partial rewrite would be a silent corruption bug.
        assert (workspace / "f.txt").read_text() == "foo foo\n"

    def test_edit_replace_all_substitutes_every_match(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "f.txt").write_text("foo foo foo\n")

        result = backend.edit("f.txt", "foo", "bar", replace_all=True)

        assert result.error is None
        assert result.occurrences == 3
        assert (workspace / "f.txt").read_text() == "bar bar bar\n"


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------


class TestExecuteAgainstRealShell:
    def test_execute_runs_command_and_captures_stdout(
        self, backend: AgentEngineToolPodBackend
    ) -> None:
        result = backend.execute("printf hello")

        assert result.exit_code == 0
        assert result.output == "hello"

    def test_execute_captures_nonzero_exit_code(
        self, backend: AgentEngineToolPodBackend
    ) -> None:
        result = backend.execute("false")
        assert result.exit_code == 1

    def test_execute_strips_tenant_secrets_from_child_env(
        self,
        backend: AgentEngineToolPodBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The shell handler scrubs env beyond a small allowlist. A
        prompt-injected ``env`` call must not surface tenant secrets.
        """
        monkeypatch.setenv("TENANT_SECRET", "shhhh")

        result = backend.execute("env")

        assert result.exit_code == 0
        assert "TENANT_SECRET" not in result.output


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownloadAgainstRealFilesystem:
    def test_download_returns_raw_bytes(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "blob.bin").write_bytes(b"\x00\x01binary\xffstuff")

        results = backend.download_files(["blob.bin"])

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].content == b"\x00\x01binary\xffstuff"

    def test_download_preserves_input_order(
        self, workspace: Path, backend: AgentEngineToolPodBackend
    ) -> None:
        (workspace / "a").write_bytes(b"A")
        (workspace / "b").write_bytes(b"B")
        (workspace / "c").write_bytes(b"C")

        results = backend.download_files(["c", "a", "b"])

        assert [r.path for r in results] == ["c", "a", "b"]
        assert [r.content for r in results] == [b"C", b"A", b"B"]

    def test_download_files_one_approval_per_file(
        self,
        workspace: Path,
        backend: AgentEngineToolPodBackend,
        fake_oe: _FakeOE,
    ) -> None:
        files = {"a.bin": b"A", "b.bin": b"BB", "c.bin": b"CCC"}
        for name, data in files.items():
            (workspace / name).write_bytes(data)
        paths = ["c.bin", "a.bin", "b.bin"]

        results = backend.download_files(paths)

        assert [r.path for r in results] == paths
        assert [r.content for r in results] == [files[p] for p in paths]
        assert all(r.error is None for r in results)
        assert len(fake_oe.requests) == len(paths)
        assert all(r["tool_name"] == "filesystem_download" for r in fake_oe.requests)
        requested = [r["arguments"]["file_path"] for r in fake_oe.requests]
        assert sorted(requested) == sorted(paths)

    @pytest.mark.anyio
    async def test_adownload_files_one_approval_per_file(
        self,
        workspace: Path,
        backend: AgentEngineToolPodBackend,
        fake_oe: _FakeOE,
    ) -> None:
        files = {"a.bin": b"A", "b.bin": b"BB", "c.bin": b"CCC"}
        for name, data in files.items():
            (workspace / name).write_bytes(data)
        paths = ["c.bin", "a.bin", "b.bin"]

        results = await backend.adownload_files(paths)

        assert [r.path for r in results] == paths
        assert [r.content for r in results] == [files[p] for p in paths]
        assert all(r.error is None for r in results)
        assert len(fake_oe.requests) == len(paths)
        assert all(r["tool_name"] == "filesystem_download" for r in fake_oe.requests)
        requested = [r["arguments"]["file_path"] for r in fake_oe.requests]
        assert sorted(requested) == sorted(paths)


# ---------------------------------------------------------------------------
# Wrapper-level error paths surfaced through the real OE
# ---------------------------------------------------------------------------


class TestErrorPathsViaRealWrapper:
    def test_policy_denied_short_circuits_before_handler_runs(
        self,
        workspace: Path,
        backend: AgentEngineToolPodBackend,
        fake_oe: _FakeOE,
    ) -> None:
        fake_oe.deny_reason = "tool blocked by admin"
        (workspace / "f.txt").write_text("untouched")

        result = backend.write("f.txt", "would-be-overwrite")

        assert result.error is not None
        assert "[NON-RETRYABLE]" in result.error
        assert "Policy denied" in result.error
        # The raw deny reason must not leak.
        assert "blocked by admin" not in result.error
        # The OE saw the request, but the handler never ran (file untouched).
        assert len(fake_oe.requests) == 1
        assert (workspace / "f.txt").read_text() == "untouched"

    def test_oe_status_error_surfaces_as_classified_failure(
        self,
        backend: AgentEngineToolPodBackend,
        fake_oe: _FakeOE,
    ) -> None:
        fake_oe.error_message = "downstream-host:27017 unreachable"

        result = backend.ls(".")

        assert result.error is not None
        # ``ToolExecutionError`` falls through the default branch of
        # ``_classify_error`` to retryable.
        assert "[RETRYABLE]" in result.error
        # Internal host:port from the OE error must not reach the agent.
        assert "27017" not in result.error
        assert "downstream-host" not in result.error
