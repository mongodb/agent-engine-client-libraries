"""Behavioral tests for ``agent_engine_runner_shared.toolpod_handlers``."""

from __future__ import annotations

import base64
import importlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def handlers(monkeypatch, tmp_path):
    """Reload ``toolpod_handlers`` with WORKSPACE_DIR pointing at tmp_path."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    import agent_engine_runner_shared.toolpod_handlers as mod

    mod = importlib.reload(mod)
    yield mod
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    importlib.reload(mod)


class TestSandbox:
    def test_relative_path_joins_to_workspace(self, handlers, tmp_path):
        resolved = handlers._resolve("foo/bar.txt")
        assert resolved.startswith(str(tmp_path))

    def test_absolute_path_outside_workspace_rebases(self, handlers, tmp_path):
        resolved = handlers._resolve("/etc/passwd")
        # Rebased under workspace, leading / stripped.
        assert resolved.startswith(str(tmp_path))
        assert "passwd" in resolved

    def test_parent_traversal_rejected(self, handlers):
        with pytest.raises(ValueError, match="escapes workspace sandbox"):
            handlers._resolve("../../../etc/passwd")

    def test_symlink_escape_rejected(self, handlers, tmp_path):
        escape_target = tmp_path.parent / "outside.txt"
        escape_target.write_text("secret")
        os.symlink(str(escape_target), str(tmp_path / "evil"))
        with pytest.raises(ValueError, match="escapes workspace sandbox"):
            handlers._resolve("evil")


class TestFilesystemLs:
    def test_lists_files_and_dirs_sorted(self, handlers, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()
        result = handlers.filesystem_ls(".")
        names = [Path(e["path"]).name for e in result["entries"]]
        assert names == ["a.txt", "b.txt", "subdir"]
        is_dir_by_name = {Path(e["path"]).name: e["is_dir"] for e in result["entries"]}
        assert is_dir_by_name == {"a.txt": False, "b.txt": False, "subdir": True}

    def test_error_on_missing_directory(self, handlers):
        result = handlers.filesystem_ls("does-not-exist")
        assert "error" in result

    def test_entry_cap_truncates(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "MAX_LS_ENTRIES", 3)
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text("")
        result = handlers.filesystem_ls(".")
        assert len(result["entries"]) == 3
        assert result["truncated"] is True

    def test_entry_cap_stops_consuming_iterator(self, handlers, tmp_path, monkeypatch):
        """Prove the early-break: a regression that scanned everything and
        then sliced would still pass ``test_entry_cap_truncates`` (same
        truncated shape), so we count iterator reads with a counting fake
        scandir and assert we stopped near the cap."""
        monkeypatch.setattr(handlers, "MAX_LS_ENTRIES", 3)
        consumed = [0]

        class _FakeEntry:
            def __init__(self, name: str) -> None:
                self.path = str(tmp_path / name)

            def is_dir(self) -> bool:
                return False

        class _FakeScandir:
            def __enter__(self):
                def _gen():
                    for i in range(100):
                        consumed[0] += 1
                        yield _FakeEntry(f"f{i:03d}.txt")

                return _gen()

            def __exit__(self, *_a) -> None:
                pass

        monkeypatch.setattr(handlers.os, "scandir", lambda _path: _FakeScandir())

        result = handlers.filesystem_ls(".")
        assert result["truncated"] is True
        assert len(result["entries"]) == 3
        # Loop consumes one extra entry to trip the break (4 reads for cap=3).
        # A regression that materialized the full iterator would consume 100.
        assert consumed[0] <= 4


class TestFilesystemRead:
    def test_reads_full_file(self, handlers, tmp_path):
        (tmp_path / "f.txt").write_text("line1\nline2\nline3\n")
        result = handlers.filesystem_read("f.txt")
        assert result["content"] == "line1\nline2\nline3\n"

    def test_reads_default_agent_skills_dir(self, monkeypatch, tmp_path):
        agent_dir = tmp_path / "local-agent"
        skill_dir = agent_dir / "skills" / "local-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: local-skill\ndescription: local\n---\n\n# Local\n")

        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        import agent_engine_runner_shared.toolpod_handlers as mod

        mod = importlib.reload(mod)
        try:
            result = mod.filesystem_read(str(skill_md))
        finally:
            monkeypatch.delenv("AGENTIC_AGENT_WORKDIR", raising=False)
            importlib.reload(mod)

        assert result["content"].endswith("# Local\n")

    def test_reads_agent_skill_from_custom_skills_dir(self, monkeypatch, tmp_path):
        agent_dir = tmp_path / "app"
        skill_dir = agent_dir / "src" / "code_reviewer_agent" / "skills" / "nested-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: nested-skill\ndescription: nested\n---\n\n# Nested\n")
        agent_yaml = agent_dir / "agent.yaml"
        agent_yaml.write_text("name: reviewer\nentrypoint: reviewer.main:app\n")

        monkeypatch.setenv("AGENTIC_AGENT_CONFIG_PATH", str(agent_yaml))
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", "src/code_reviewer_agent/skills")
        import agent_engine_runner_shared.toolpod_handlers as mod

        mod = importlib.reload(mod)
        try:
            result = mod.filesystem_read(str(skill_md))
        finally:
            monkeypatch.delenv("AGENTIC_AGENT_CONFIG_PATH", raising=False)
            monkeypatch.delenv("AGENTIC_SKILLS_DIR", raising=False)
            importlib.reload(mod)

        assert result["content"].endswith("# Nested\n")

    @pytest.mark.parametrize(
        ("skills_dir", "has_agent_root", "warning"),
        [
            ("/etc", True, "must be relative to the agent source root"),
            ("../shared-skills", True, "resolves outside the agent source root"),
            ("skills", False, "agent source root is unknown"),
        ],
    )
    def test_ignores_invalid_custom_skills_dir(
        self, monkeypatch, tmp_path, caplog, skills_dir, has_agent_root, warning
    ):
        if has_agent_root:
            monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(tmp_path / "app"))
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", skills_dir)
        import agent_engine_runner_shared.toolpod_handlers as mod

        with caplog.at_level(logging.WARNING, logger="agent_engine_runner_shared.toolpod_handlers"):
            mod = importlib.reload(mod)
        try:
            readonly_roots = mod.get_readonly_resource_roots()
        finally:
            monkeypatch.delenv("AGENTIC_SKILLS_DIR", raising=False)
            monkeypatch.delenv("AGENTIC_AGENT_WORKDIR", raising=False)
            importlib.reload(mod)

        assert readonly_roots == ()
        assert warning in caplog.text

    def test_line_slicing_with_offset_and_limit(self, handlers, tmp_path):
        (tmp_path / "f.txt").write_text("\n".join(f"line{i}" for i in range(10)))
        result = handlers.filesystem_read("f.txt", offset=2, limit=3)
        content = result["content"]
        assert "line2" in content
        assert "line3" in content
        assert "line4" in content
        assert "line0" not in content
        assert "line5" not in content

    def test_size_cap_rejects_large_file(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "FILESYSTEM_READ_MAX_BYTES", 10)
        (tmp_path / "big.txt").write_text("x" * 100)
        result = handlers.filesystem_read("big.txt")
        assert "error" in result
        assert "FILESYSTEM_READ_MAX_BYTES" in result["error"]


class TestReadonlySkillRoots:
    """skill roots resolve lazily, so import order no longer matters."""

    @staticmethod
    def _make_agent_with_skill(agent_dir):
        skill_dir = agent_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# demo skill\n")
        return skill_md

    @staticmethod
    def _runtime():
        return SimpleNamespace(_tools={}, _tool_definitions={})

    def test_reads_bundled_skill_when_agent_root_set_after_import(
        self, handlers, monkeypatch, tmp_path_factory
    ):
        # No reload: `handlers` imported the module before the env was set. The
        # agent dir lives outside WORKSPACE_DIR so the read must go through the
        # readonly root, not the workspace.
        agent_dir = tmp_path_factory.mktemp("agent")
        skill_md = self._make_agent_with_skill(agent_dir)
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        result = handlers.filesystem_read(str(skill_md))
        assert "error" not in result
        assert result["content"] == "# demo skill\n"

    def test_same_root_for_import_before_and_after_env(
        self, handlers, monkeypatch, tmp_path_factory
    ):
        agent_dir = tmp_path_factory.mktemp("agent")
        skill_md = self._make_agent_with_skill(agent_dir)
        # `handlers` was imported before the env is set; a fresh reload is the
        # import-after-env form. Both must resolve the same root.
        early = handlers
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        import agent_engine_runner_shared.toolpod_handlers as mod

        late = importlib.reload(mod)
        try:
            assert early.filesystem_read(str(skill_md))["content"] == "# demo skill\n"
            assert late.filesystem_read(str(skill_md))["content"] == "# demo skill\n"
        finally:
            monkeypatch.delenv("AGENTIC_AGENT_WORKDIR", raising=False)
            importlib.reload(mod)

    def test_startup_raises_when_explicit_skills_dir_unresolvable(
        self, handlers, monkeypatch, tmp_path_factory
    ):
        agent_dir = tmp_path_factory.mktemp("agent")
        self._make_agent_with_skill(agent_dir)
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", "no-such-dir")
        with pytest.raises(RuntimeError, match="AGENTIC_SKILLS_DIR=.*no readable skills root"):
            handlers.register_builtin_tools(self._runtime())

    def test_startup_warns_when_default_skills_path_is_a_file(
        self, handlers, monkeypatch, tmp_path_factory, caplog
    ):
        # Previously a silent no-op, so warn (not raise) for one release to
        # avoid crash-looping already-deployed agents.
        agent_dir = tmp_path_factory.mktemp("agent")
        (agent_dir / "skills").write_text("not a dir")
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        with caplog.at_level(logging.WARNING):
            handlers.register_builtin_tools(self._runtime())
        assert "not a readable directory" in caplog.text

    def test_startup_ok_and_skill_readable_with_default_skills_dir(
        self, handlers, monkeypatch, tmp_path_factory
    ):
        agent_dir = tmp_path_factory.mktemp("agent")
        skill_md = self._make_agent_with_skill(agent_dir)
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        handlers.register_builtin_tools(self._runtime())
        assert handlers.filesystem_read(str(skill_md))["content"] == "# demo skill\n"

    def test_startup_ok_with_no_skills_bundled(self, handlers, monkeypatch, tmp_path_factory):
        agent_dir = tmp_path_factory.mktemp("agent")
        (agent_dir / "notes.txt").write_text("x")
        monkeypatch.setenv("AGENTIC_AGENT_WORKDIR", str(agent_dir))
        handlers.register_builtin_tools(self._runtime())
        # No skills dir → agent source root is not a readable root; an absolute
        # path there is rebased into the workspace and misses.
        result = handlers.filesystem_read(str(agent_dir / "notes.txt"))
        assert "error" in result


class TestFilesystemWrite:
    def test_creates_parent_directories(self, handlers, tmp_path):
        result = handlers.filesystem_write("deeply/nested/f.txt", "hello")
        assert "error" not in result
        assert (tmp_path / "deeply" / "nested" / "f.txt").read_text() == "hello"

    def test_rejects_existing_file(self, handlers, tmp_path):
        """Per deepagents.BackendProtocol.write, write is create-only."""
        (tmp_path / "f.txt").write_text("old")
        result = handlers.filesystem_write("f.txt", "new")
        assert "already exists" in result["error"]
        assert (tmp_path / "f.txt").read_text() == "old"

    def test_large_write_logs_warning_and_still_writes(
        self, handlers, tmp_path, monkeypatch, caplog
    ):
        """A write over FILESYSTEM_WRITE_WARN_BYTES logs a warning but succeeds."""
        import logging

        monkeypatch.setattr(handlers, "FILESYSTEM_WRITE_WARN_BYTES", 10)
        with caplog.at_level(logging.WARNING, logger=handlers.__name__):
            result = handlers.filesystem_write("big.txt", "x" * 20)
        assert "error" not in result
        assert (tmp_path / "big.txt").read_text() == "x" * 20
        assert any("large" in rec.message or "RAM" in rec.message for rec in caplog.records)

    def test_small_write_does_not_warn(self, handlers, tmp_path, monkeypatch, caplog):
        """A write under FILESYSTEM_WRITE_WARN_BYTES produces no warning."""
        import logging

        monkeypatch.setattr(handlers, "FILESYSTEM_WRITE_WARN_BYTES", 1000)
        with caplog.at_level(logging.WARNING, logger=handlers.__name__):
            result = handlers.filesystem_write("small.txt", "hello")
        assert "error" not in result
        assert not any("RAM" in rec.message or "large" in rec.message for rec in caplog.records)


class TestFilesystemEdit:
    def test_unique_match_replaces(self, handlers, tmp_path):
        (tmp_path / "f.txt").write_text("alpha beta gamma")
        result = handlers.filesystem_edit("f.txt", "beta", "BETA")
        assert result == {"occurrences": 1}
        assert (tmp_path / "f.txt").read_text() == "alpha BETA gamma"

    def test_non_unique_default_rejected(self, handlers, tmp_path):
        """Non-unique old_string without replace_all must reject, not silently pick first."""
        (tmp_path / "f.txt").write_text("abc abc abc")
        result = handlers.filesystem_edit("f.txt", "abc", "XYZ")
        assert "not unique" in result["error"]
        assert (tmp_path / "f.txt").read_text() == "abc abc abc"

    def test_replace_all_replaces_every_occurrence(self, handlers, tmp_path):
        (tmp_path / "f.txt").write_text("abc abc abc")
        result = handlers.filesystem_edit("f.txt", "abc", "X", replace_all=True)
        assert result == {"occurrences": 3}
        assert (tmp_path / "f.txt").read_text() == "X X X"

    def test_missing_old_string_reports_not_found(self, handlers, tmp_path):
        (tmp_path / "f.txt").write_text("hello")
        result = handlers.filesystem_edit("f.txt", "goodbye", "X")
        assert "not found" in result["error"]

    def test_empty_old_string_rejected(self, handlers, tmp_path):
        """str.replace('', x, all) interleaves between every char — reject."""
        (tmp_path / "f.txt").write_text("abc")
        result = handlers.filesystem_edit("f.txt", "", "X", replace_all=True)
        assert "non-empty" in result["error"]
        assert (tmp_path / "f.txt").read_text() == "abc"

    def test_size_cap_rejects_large_file(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "FILESYSTEM_READ_MAX_BYTES", 10)
        (tmp_path / "big.txt").write_text("x" * 100)
        result = handlers.filesystem_edit("big.txt", "x", "y")
        assert "FILESYSTEM_READ_MAX_BYTES" in result["error"]

    def test_post_replacement_size_cap_rejected(self, handlers, tmp_path, monkeypatch):
        """Distinct from the up-front cap: file fits, but replacement would
        push it past FILESYSTEM_READ_MAX_BYTES. A regression that drops the
        post-replacement re-check would not be caught by the up-front test."""
        monkeypatch.setattr(handlers, "FILESYSTEM_READ_MAX_BYTES", 20)
        (tmp_path / "f.txt").write_text("abc abc")
        result = handlers.filesystem_edit("f.txt", "abc", "0123456789", replace_all=True)
        assert "Edit would exceed FILESYSTEM_READ_MAX_BYTES" in result["error"]
        # File untouched — the guard runs before the write.
        assert (tmp_path / "f.txt").read_text() == "abc abc"


class TestFilesystemGlob:
    def test_matches_pattern(self, handlers, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = handlers.filesystem_glob("*.py")
        names = sorted(Path(m["path"]).name for m in result["matches"])
        assert names == ["a.py", "b.py"]
        assert result["truncated"] is False

    def test_recursive_pattern(self, handlers, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.py").write_text("")
        result = handlers.filesystem_glob("**/*.py")
        names = sorted(Path(m["path"]).name for m in result["matches"])
        assert "nested.py" in names

    def test_absolute_pattern_rejected(self, handlers):
        """Catches the os.path.join left-discard sandbox bypass."""
        result = handlers.filesystem_glob("/etc/*")
        assert "absolute" in result["error"].lower()

    @pytest.mark.parametrize(
        "pattern",
        ["../../etc/*", "../*", "foo/../../../etc/*", "..\\..\\etc\\*"],
    )
    def test_traversal_pattern_rejected(self, handlers, pattern):
        result = handlers.filesystem_glob(pattern)
        assert "traversal" in result["error"]

    def test_match_cap_truncates(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "MAX_GLOB_MATCHES", 3)
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text("")
        result = handlers.filesystem_glob("*.txt")
        assert len(result["matches"]) == 3
        assert result["truncated"] is True


class TestFilesystemGrep:
    def test_finds_matching_lines(self, handlers, tmp_path):
        (tmp_path / "a.txt").write_text("hello\nworld\nhello again\n")
        result = handlers.filesystem_grep("hello")
        matches = result["matches"]
        assert len(matches) == 2
        lines = [(m["line"], m["text"]) for m in matches]
        assert (1, "hello") in lines
        assert (3, "hello again") in lines

    def test_glob_filter(self, handlers, tmp_path):
        (tmp_path / "a.py").write_text("match\n")
        (tmp_path / "b.txt").write_text("match\n")
        result = handlers.filesystem_grep("match", glob="*.py")
        names = {Path(m["path"]).name for m in result["matches"]}
        assert names == {"a.py"}

    def test_literal_substring_not_regex(self, handlers, tmp_path):
        """Per deepagents.BackendProtocol.grep, pattern is literal, not regex."""
        (tmp_path / "f.txt").write_text("foo.bar\nfooXbar\n")
        result = handlers.filesystem_grep("foo.bar")
        texts = [m["text"] for m in result["matches"]]
        assert "foo.bar" in texts
        assert "fooXbar" not in texts

    def test_symlinked_file_outside_workspace_skipped(self, handlers, tmp_path):
        """A symlink inside workspace pointing outside must not be grepped."""
        secret = tmp_path.parent / "outside-secret.txt"
        secret.write_text("match-me-if-you-can\n")
        os.symlink(str(secret), str(tmp_path / "leaky"))
        (tmp_path / "inside.txt").write_text("match-me-if-you-can\n")

        result = handlers.filesystem_grep("match-me-if-you-can")
        names = {Path(m["path"]).name for m in result["matches"]}
        assert names == {"inside.txt"}

    def test_binary_files_skipped(self, handlers, tmp_path):
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\xff match \xc0\x80")
        (tmp_path / "text.txt").write_text("match\n")
        result = handlers.filesystem_grep("match")
        names = {Path(m["path"]).name for m in result["matches"]}
        assert names == {"text.txt"}

    def test_match_cap_truncates(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "MAX_GREP_MATCHES", 3)
        (tmp_path / "f.txt").write_text("match\n" * 10)
        result = handlers.filesystem_grep("match")
        assert len(result["matches"]) == 3
        assert result["truncated"] is True

    def test_wall_clock_budget_trips(self, handlers, tmp_path, monkeypatch):
        """Budget must fire deterministically — patch time.monotonic so the
        second call is already past the deadline regardless of real timing."""
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("line\n")

        calls = iter([0.0, 100.0, 100.0, 100.0, 100.0])
        monkeypatch.setattr(handlers, "MAX_GREP_SECONDS", 1.0)
        monkeypatch.setattr(handlers.time, "monotonic", lambda: next(calls, 100.0))

        result = handlers.filesystem_grep("line")
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# filesystem_download
# ---------------------------------------------------------------------------


class TestFilesystemDownload:
    def test_base64_roundtrip(self, handlers, tmp_path):
        raw = b"\x00binary\xffpayload"
        (tmp_path / "f.bin").write_bytes(raw)
        result = handlers.filesystem_download("f.bin")
        assert base64.b64decode(result["content_base64"]) == raw

    def test_size_cap_rejects_large_file(self, handlers, tmp_path, monkeypatch):
        monkeypatch.setattr(handlers, "DOWNLOAD_MAX_BYTES", 10)
        (tmp_path / "big.bin").write_bytes(b"x" * 100)
        result = handlers.filesystem_download("big.bin")
        assert "DOWNLOAD_MAX_BYTES" in result["error"]


class TestShellExecute:
    def test_runs_command_and_returns_stdout(self, handlers):
        result = handlers.shell_execute("echo hi")
        assert result["exit_code"] == 0
        assert "hi" in result["output"]
        assert result["truncated"] is False

    def test_captures_stderr_combined_with_stdout(self, handlers):
        result = handlers.shell_execute("echo out; echo err >&2")
        assert "out" in result["output"]
        assert "err" in result["output"]

    def test_nonzero_exit_code_surfaced(self, handlers):
        result = handlers.shell_execute("exit 7")
        assert result["exit_code"] == 7

    def test_timeout_returns_sentinel_with_partial_output(self, handlers):
        result = handlers.shell_execute("echo EARLY; sleep 5; echo LATE", timeout=1)
        assert result["exit_code"] == handlers.EXIT_CODE_TIMEOUT
        assert "EARLY" in result["output"]
        assert "LATE" not in result["output"]

    def test_unbounded_output_does_not_oom_pod(self, handlers, monkeypatch):
        """Drain threads must keep the child unblocked while discarding bytes past the cap."""
        monkeypatch.setattr(handlers, "SHELL_OUTPUT_MAX_BYTES", 1024)
        cmd = "python3 -c \"import sys; sys.stdout.write('x' * 200000)\""
        result = handlers.shell_execute(cmd, timeout=10)
        assert result["exit_code"] == 0
        assert len(result["output"].encode("utf-8")) <= 1024 + 16
        assert result["truncated"] is True

    def test_tenant_env_scrubbed_from_child(self, handlers, monkeypatch):
        """OWASP LLM01: tenant secrets must not leak into shell children."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-LEAK")
        monkeypatch.setenv("MONGODB_URI", "mongodb://user:pw@host/db")

        result = handlers.shell_execute("env")
        assert "sk-LEAK" not in result["output"]
        assert "mongodb://user:pw@host/db" not in result["output"]
        assert "OPENAI_API_KEY" not in result["output"]
        assert "MONGODB_URI" not in result["output"]

    def test_allowlisted_env_reaches_child(self, handlers, monkeypatch):
        """PATH must still reach the child so /bin/sh can locate utilities."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin:/custom/sentinel-path")
        result = handlers.shell_execute("echo $PATH")
        assert "/custom/sentinel-path" in result["output"]

    def test_framework_error_returns_distinct_sentinel(self, handlers, monkeypatch):
        """Popen failures must not masquerade as timeouts."""

        def fail_popen(*_a, **_kw):
            raise OSError("boom")

        monkeypatch.setattr(handlers.subprocess, "Popen", fail_popen)
        result = handlers.shell_execute("echo hi")
        assert result["exit_code"] == handlers.EXIT_CODE_FRAMEWORK_ERROR
        assert "boom" in result.get("error", "")

    def test_framework_error_after_thread_start_runs_cleanup(self, handlers, monkeypatch):
        """Levi: the existing framework-error test fails inside ``Popen``,
        so ``proc`` is never bound and the cleanup ``finally`` is skipped.
        Inject a failure *after* Popen + thread.start succeed by monkey-
        patching ``_truncate_output`` (the last call before return), and
        assert the function returns the sentinel without hanging — i.e.
        the cleanup arm joined the drain threads instead of leaving them
        live."""

        def boom(*_a, **_kw):
            raise RuntimeError("synthetic post-start failure")

        monkeypatch.setattr(handlers, "_truncate_output", boom)

        result = handlers.shell_execute("echo hi", timeout=5)

        assert result["exit_code"] == handlers.EXIT_CODE_FRAMEWORK_ERROR
        assert "synthetic post-start failure" in result.get("error", "")


class TestDrainBounded:
    def test_caps_at_limit_while_still_consuming(self, handlers):
        r, w = os.pipe()
        os.write(w, b"a" * 500)
        os.close(w)

        bucket: list[bytes] = []
        overflow: list[bool] = [False]
        with os.fdopen(r, "rb") as reader:
            handlers._drain_bounded(reader, bucket, cap=100, overflow=overflow)

        assert sum(len(c) for c in bucket) == 100
        assert overflow[0] is True

    def test_under_cap_does_not_flag_overflow(self, handlers):
        r, w = os.pipe()
        os.write(w, b"a" * 50)
        os.close(w)

        bucket: list[bytes] = []
        overflow: list[bool] = [False]
        with os.fdopen(r, "rb") as reader:
            handlers._drain_bounded(reader, bucket, cap=100, overflow=overflow)

        assert sum(len(c) for c in bucket) == 50
        assert overflow[0] is False

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("I/O operation on closed file"),
            OSError("Bad file descriptor"),
        ],
    )
    def test_returns_cleanly_when_pipe_read_raises(self, handlers, exc):
        """When the main thread closes the pipe mid-drain, ``pipe.read``
        raises ``ValueError`` (CPython) or ``OSError`` (some platforms).
        The drain thread must catch and exit cleanly — without this
        guard, the daemon thread would die with an unhandled exception."""

        class _RaisingPipe:
            def __init__(self, error: BaseException) -> None:
                self._error = error

            def read(self, _n: int) -> bytes:
                raise self._error

        bucket: list[bytes] = []
        overflow: list[bool] = [False]
        # Must return without re-raising.
        handlers._drain_bounded(_RaisingPipe(exc), bucket, cap=100, overflow=overflow)
        assert bucket == []
        assert overflow[0] is False
