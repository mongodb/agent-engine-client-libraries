"""Tests for agent_engine_runner_shared.launcher module."""

from __future__ import annotations

import json
import logging
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared.launcher import (
    EXIT_IMPORT_ERROR,
    EXIT_NO_ENTRYPOINT,
    EXIT_STARTUP_CRASH,
    _bound_text,
    _resolve_entrypoint,
    _write_termination_message,
    main,
)


@pytest.fixture
def structured_logging_env(monkeypatch: pytest.MonkeyPatch):
    """Enable STRUCTURED_LOGGING for a test and restore the global logging
    state (handlers, level, sys.stdout/stderr, excepthook) afterward, so a
    test exercising main()'s early install doesn't leak a patched
    excepthook or handler into later tests in the same process.
    """
    monkeypatch.setenv("STRUCTURED_LOGGING", "true")

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    saved_excepthook = sys.excepthook
    saved_thread_excepthook = threading.excepthook
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        sys.excepthook = saved_excepthook
        threading.excepthook = saved_thread_excepthook
        root.handlers = saved_handlers
        root.setLevel(saved_level)


class TestResolveEntrypoint:
    """Tests for _resolve_entrypoint()."""

    def test_module_only_defaults_to_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent.app")
        assert _resolve_entrypoint() == ("my_agent.app", "main")

    def test_colon_form_splits_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent.main:run_agent")
        assert _resolve_entrypoint() == ("my_agent.main", "run_agent")

    def test_colon_in_module_path_uses_rightmost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent.main:run")
        assert _resolve_entrypoint() == ("my_agent.main", "run")

    def test_trailing_carriage_return_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # an agent.yaml authored on Windows leaves a CR on the value the
        # build pipeline greps out of it and bakes into AGENT_ENTRYPOINT.
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent.main:app\r")
        assert _resolve_entrypoint() == ("my_agent.main", "app")

    def test_surrounding_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "  my_agent.main : app \r\n")
        assert _resolve_entrypoint() == ("my_agent.main", "app")

    def test_module_only_with_carriage_return_defaults_to_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent.app\r")
        assert _resolve_entrypoint() == ("my_agent.app", "main")

    def test_whitespace_only_env_var_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", " \r\n")
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _resolve_entrypoint()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "AGENT_ENTRYPOINT is not set" in caplog.text

    def test_missing_env_var_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _resolve_entrypoint()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "AGENT_ENTRYPOINT is not set" in caplog.text

    def test_empty_env_var_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "")
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _resolve_entrypoint()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "AGENT_ENTRYPOINT is not set" in caplog.text

    def test_trailing_colon_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "my_agent:")
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _resolve_entrypoint()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "both module path and function name must be non-empty" in caplog.text

    def test_leading_colon_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", ":run")
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _resolve_entrypoint()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "both module path and function name must be non-empty" in caplog.text


class TestMain:
    """Tests for main() dispatch logic."""

    @pytest.fixture(autouse=True)
    def _stub_setup_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )

    def test_callable_target_invoked_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        mock_fn = MagicMock()
        fake_module = types.ModuleType("fake_module")
        fake_module.start = mock_fn  # type: ignore[attr-defined]

        with patch(
            "agent_engine_runner_shared.launcher.importlib.import_module", return_value=fake_module
        ):
            main()

        mock_fn.assert_called_once()

    def test_materializes_mcp_oauth_secrets_before_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        events: list[str] = []

        def start() -> None:
            events.append("start")

        fake_module = types.ModuleType("fake_module")
        fake_module.start = start  # type: ignore[attr-defined]

        def import_module(_module_path: str) -> types.ModuleType:
            events.append("import")
            return fake_module

        with (
            patch(
                "agent_engine_runner_shared.launcher.materialize_mcp_oauth_secret_cache",
                side_effect=lambda: events.append("materialize"),
            ),
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=import_module,
            ),
        ):
            main()

        assert events == ["materialize", "import", "start"]

    def test_materialize_mcp_oauth_secret_error_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        with (
            patch(
                "agent_engine_runner_shared.launcher.materialize_mcp_oauth_secret_cache",
                side_effect=RuntimeError("invalid MCP OAuth secret AGENTIC_MCP_OAUTH_B64_GITHUB"),
            ),
            patch("agent_engine_runner_shared.launcher.importlib.import_module") as import_module,
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        import_module.assert_not_called()
        # caplog rather than capsys.err: logger.error() only lands on the
        # real stderr via logging.lastResort, which applies only when the
        # root logger has zero handlers — order-dependent on any handler
        # leaked by another test in the same process.
        assert "Cannot materialize MCP OAuth credentials" in caplog.text
        assert "Launcher failed to start agent" not in caplog.text

    def test_materialize_mcp_oauth_secret_os_error_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        with (
            patch(
                "agent_engine_runner_shared.launcher.materialize_mcp_oauth_secret_cache",
                side_effect=OSError("permission denied"),
            ),
            patch("agent_engine_runner_shared.launcher.importlib.import_module") as import_module,
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        import_module.assert_not_called()
        assert "Cannot materialize MCP OAuth credentials" in caplog.text
        assert "Launcher failed to start agent" not in caplog.text

    def test_non_callable_with_run_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:app")
        mock_run = MagicMock()

        class AppObj:
            run = mock_run

        fake_module = types.ModuleType("fake_module")
        fake_module.app = AppObj()  # type: ignore[attr-defined]

        with patch(
            "agent_engine_runner_shared.launcher.importlib.import_module", return_value=fake_module
        ):
            main()

        mock_run.assert_called_once()

    def test_module_not_found_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "nonexistent.module:main")
        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'nonexistent'"),
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_IMPORT_ERROR
        assert "Cannot import module" in caplog.text
        assert "No module named 'nonexistent'" in caplog.text

    def test_module_not_found_from_nested_import_keeps_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing transitive dependency raises ModuleNotFoundError too, and
        that is the common case. The traceback is the only thing naming the
        import that actually failed inside the agent's own code, so it has to
        be attached rather than logged as a bare message."""
        monkeypatch.setenv("AGENT_ENTRYPOINT", "agent.main:start")
        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'langchain_core'"),
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_IMPORT_ERROR
        assert "No module named 'langchain_core'" in caplog.text
        assert any(rec.exc_info for rec in caplog.records), caplog.records

    def test_import_error_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "broken.module:main")
        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=ImportError("circular import"),
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_IMPORT_ERROR
        assert "Failed to import module" in caplog.text
        assert "circular import" in caplog.text

    def test_syntax_error_on_import_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "broken.module:main")
        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=SyntaxError("invalid syntax"),
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_IMPORT_ERROR
        assert "Failed to import module" in caplog.text

    def test_missing_attribute_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:missing")
        fake_module = types.ModuleType("fake_module")

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "has no attribute 'missing'" in caplog.text
        assert "Launcher failed to start agent" not in caplog.text

    def test_non_callable_without_run_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:config")
        fake_module = types.ModuleType("fake_module")
        fake_module.config = {"key": "value"}  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "is not callable and has no callable 'run()' method" in caplog.text
        assert "Launcher failed to start agent" not in caplog.text

    def test_entrypoint_exception_logs_and_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        def start() -> None:
            raise RuntimeError("crash before logging is configured")

        fake_module = types.ModuleType("fake_module")
        fake_module.start = start  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_STARTUP_CRASH
        assert "Unhandled exception from agent entrypoint" in caplog.text
        assert "crash before logging is configured" in caplog.text

    def test_run_method_exception_logs_and_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:app")

        class AppObj:
            def run(self) -> None:
                raise ValueError("app run failed")

        fake_module = types.ModuleType("fake_module")
        fake_module.app = AppObj()  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_STARTUP_CRASH
        assert "Unhandled exception from agent entrypoint" in caplog.text
        assert "app run failed" in caplog.text

    def test_system_exit_from_target_is_not_wrapped(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        def start() -> None:
            raise SystemExit(2)

        fake_module = types.ModuleType("fake_module")
        fake_module.start = start  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2
        assert "Launcher failed to start agent" not in caplog.text
        assert "Unhandled exception from agent entrypoint" not in caplog.text


class TestMainStructuredLogging:
    """Tests for main()'s STRUCTURED_LOGGING=true early-install path."""

    def test_bootstrap_failure_emits_error_with_allowlisted_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        structured_logging_env: None,
    ) -> None:
        """A launcher bootstrap failure (here: AGENT_ENTRYPOINT unset) must
        round-trip through the JSON pipeline as level=ERROR with the
        message intact, instead of showing up as an unstructured line the
        API Gateway's raw-text heuristic doesn't recognize as an error."""
        monkeypatch.setenv("RUNNER_MODE", "aer")
        monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        rec = next(r for r in lines if r["message"] == "AGENT_ENTRYPOINT is not set")
        assert rec["level"] == "ERROR"
        assert rec["service"] == "agent-execution-runtime"

    def test_unrecognized_runner_mode_falls_back_to_allowlisted_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        structured_logging_env: None,
    ) -> None:
        """The actual regression case for the silent-drop risk: a non-empty,
        unrecognized RUNNER_MODE must not pass through verbatim as
        `service` — /agent-logs' hard allowlist silently drops any record
        outside {agent-execution-runtime, tool-executor}, which would be
        worse than today's mislabeled-but-visible line."""
        monkeypatch.setenv("RUNNER_MODE", "not-a-real-mode")
        monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)

        with pytest.raises(SystemExit):
            main()

        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        rec = next(r for r in lines if r["message"] == "AGENT_ENTRYPOINT is not set")
        assert rec["service"] == "agent-execution-runtime", rec

    def test_entrypoint_exception_emits_single_error_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        structured_logging_env: None,
    ) -> None:
        """An uncaught exception raised by the agent's own entrypoint
        function must be caught locally by main()'s safety net and emit
        exactly one ERROR record with full exception fields — not a
        WARNING trace, and not left to escape to the process-level
        excepthook."""
        monkeypatch.setenv("RUNNER_MODE", "aer")
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        def start() -> None:
            raise RuntimeError("agent-entrypoint-boom")

        fake_module = types.ModuleType("fake_module")
        fake_module.start = start  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == EXIT_STARTUP_CRASH
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        error_recs = [
            r for r in lines if r["level"] == "ERROR" and "agent-entrypoint-boom" in r["message"]
        ]
        assert len(error_recs) == 1, lines
        assert error_recs[0]["fields"]["exc_type"] == "RuntimeError"
        assert "agent-entrypoint-boom" in error_recs[0]["fields"]["exc_traceback"]


class TestMainLoggingInstall:
    """Tests that require the real setup_logging path."""

    def test_setup_logging_defaults_mode_aer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RUNNER_MODE", raising=False)
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        fake_module = types.ModuleType("fake_module")
        fake_module.start = lambda: None  # type: ignore[attr-defined]

        with (
            patch("agent_engine_runner_shared.launcher.setup_logging") as setup,
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
        ):
            main()

        setup.assert_called_once_with(app_name="launcher", mode="aer")

    def test_setup_logging_uses_runner_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNNER_MODE", "tool")
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        fake_module = types.ModuleType("fake_module")
        fake_module.start = lambda: None  # type: ignore[attr-defined]

        with (
            patch("agent_engine_runner_shared.launcher.setup_logging") as setup,
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
        ):
            main()

        setup.assert_called_once_with(app_name="launcher", mode="tool")

    def test_setup_logging_falls_back_to_aer_for_unrecognized_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression case: an unrecognized RUNNER_MODE must not be
        passed through to setup_logging verbatim — /agent-logs' hard service
        allowlist would silently drop the record."""
        monkeypatch.setenv("RUNNER_MODE", "not-a-real-mode")
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        fake_module = types.ModuleType("fake_module")
        fake_module.start = lambda: None  # type: ignore[attr-defined]

        with (
            patch("agent_engine_runner_shared.launcher.setup_logging") as setup,
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
        ):
            main()

        setup.assert_called_once_with(app_name="launcher", mode="aer")

    def test_setup_logging_runs_before_materialize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")
        events: list[str] = []
        fake_module = types.ModuleType("fake_module")
        fake_module.start = lambda: events.append("start")  # type: ignore[attr-defined]

        with (
            patch(
                "agent_engine_runner_shared.launcher.setup_logging",
                side_effect=lambda **_kwargs: events.append("setup"),
            ),
            patch(
                "agent_engine_runner_shared.launcher.materialize_mcp_oauth_secret_cache",
                side_effect=lambda: events.append("materialize"),
            ),
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=lambda _path: events.append("import") or fake_module,
            ),
        ):
            main()

        assert events == ["setup", "materialize", "import", "start"]

    def test_import_failure_emits_structured_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("STRUCTURED_LOGGING", "true")
        monkeypatch.setenv("RUNNER_MODE", "aer")
        monkeypatch.setenv("ORG_ID", "tenant-acme")
        monkeypatch.setenv("PROJECT_ID", "project-zeta")
        monkeypatch.setenv("AGENT_ENTRYPOINT", "nonexistent.module:main")
        monkeypatch.delenv("WORKSPACE_ID", raising=False)

        root = logging.getLogger()
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        saved_handlers = list(root.handlers)
        saved_level = root.level
        try:
            with (
                patch(
                    "agent_engine_runner_shared.launcher.importlib.import_module",
                    side_effect=ModuleNotFoundError("No module named 'nonexistent'"),
                ),
                pytest.raises(SystemExit) as exc_info,
            ):
                main()
            assert exc_info.value.code == EXIT_IMPORT_ERROR
            records = [
                json.loads(line)
                for line in capsys.readouterr().out.splitlines()
                if line.strip().startswith("{")
            ]
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
            root.handlers = saved_handlers
            root.setLevel(saved_level)

        error_records = [rec for rec in records if rec.get("level") == "ERROR"]
        assert error_records, records
        assert any("Cannot import module" in rec.get("message", "") for rec in error_records), (
            error_records
        )
        assert any("No module named 'nonexistent'" in json.dumps(rec) for rec in error_records), (
            error_records
        )


class TestBoundText:
    """Tests for _bound_text()."""

    def test_under_limits_unchanged(self) -> None:
        text = "short message\nwith two lines"
        assert _bound_text(text) == text

    def test_truncates_by_line_count(self) -> None:
        text = "\n".join(["line"] * 50)
        result = _bound_text(text, max_lines=10)
        assert result.count("\n") <= 10
        assert result.endswith("[truncated]")

    def test_truncates_by_byte_size(self) -> None:
        text = "a" * 10_000
        result = _bound_text(text, max_bytes=100)
        assert len(result.encode("utf-8")) <= 100 + len("\n[truncated]")
        assert result.endswith("[truncated]")

    def test_never_splits_a_multibyte_rune(self) -> None:
        # Each "é" is 2 bytes in UTF-8; an odd byte limit forces the naive cut
        # point to land mid-character.
        text = "é" * 60
        result = _bound_text(text, max_bytes=101)
        body = result.removesuffix("\n[truncated]")
        # A mangled cut would raise on encode or contain U+FFFD; neither
        # should happen since decode(errors="ignore") drops the partial tail.
        body.encode("utf-8")

    def test_marker_never_pushes_result_past_max_bytes(self) -> None:
        # The marker itself must fit within max_bytes — appending it after an
        # unreserved full-budget truncation would silently exceed kubelet's
        # own read cap on /dev/termination-log.
        text = "a" * 10_000
        result = _bound_text(text, max_bytes=100)
        assert len(result.encode("utf-8")) <= 100


class TestWriteTerminationMessage:
    """Tests for _write_termination_message()."""

    def test_redacts_credentials_and_preserves_type_and_host(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))

        try:
            raise ValueError("failed with mongodb+srv://dbuser:hunter2@cluster0.mongodb.net")
        except ValueError as exc:
            _write_termination_message("Unhandled exception from agent entrypoint 'x:y'", exc)

        text = path.read_text()
        assert "hunter2" not in text
        assert "cluster0.mongodb.net" in text
        assert "ValueError" in text
        assert "Unhandled exception from agent entrypoint 'x:y'" in text

    def test_no_exception_writes_bare_summary(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))

        _write_termination_message("AGENT_ENTRYPOINT is not set")

        assert path.read_text() == "AGENT_ENTRYPOINT is not set"

    def test_best_effort_on_write_failure(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A path under a non-existent directory: open() must fail, and the
        # function must not raise.
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH",
            str(tmp_path / "no-such-dir" / "termination-log"),
        )
        _write_termination_message("does not matter", ValueError("boom"))

    def test_redacts_credential_past_truncation_boundary(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Filler pushes the credential's end near the 4KiB cutoff. With the
        # old bound-before-redact order this would truncate mid-credential
        # before _redact_text ever saw the full userinfo shape, leaking a
        # partial secret.
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))

        filler = "x" * 4090
        try:
            raise ValueError(f"{filler} mongodb+srv://dbuser:hunter2@cluster0.mongodb.net")
        except ValueError as exc:
            _write_termination_message("startup failed", exc)

        assert "hunter2" not in path.read_text()


class TestMainWritesTerminationMessage:
    """Integration: main()'s crash paths call _write_termination_message."""

    def test_module_not_found_writes_termination_message(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )
        monkeypatch.setenv("AGENT_ENTRYPOINT", "nonexistent.module:main")

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'nonexistent'"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == EXIT_IMPORT_ERROR
        assert "No module named 'nonexistent'" in path.read_text()

    def test_unhandled_entrypoint_exception_writes_termination_message(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        fake_module = types.ModuleType("fake_module")
        fake_module.start = MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("boom mongodb+srv://u:p@host/db")
        )

        with (
            patch(
                "agent_engine_runner_shared.launcher.importlib.import_module",
                return_value=fake_module,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == EXIT_STARTUP_CRASH
        text = path.read_text()
        assert "RuntimeError" in text
        assert "boom" in text
        assert ":p@" not in text

    def test_missing_agent_entrypoint_writes_termination_message(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )
        monkeypatch.delenv("AGENT_ENTRYPOINT", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "AGENT_ENTRYPOINT is not set" in path.read_text()

    def test_malformed_agent_entrypoint_writes_termination_message(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )
        monkeypatch.setenv("AGENT_ENTRYPOINT", ":run")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == EXIT_NO_ENTRYPOINT
        assert "must be non-empty" in path.read_text()

    def test_mcp_oauth_failure_writes_termination_message(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "termination-log"
        monkeypatch.setattr("agent_engine_runner_shared.launcher._TERMINATION_LOG_PATH", str(path))
        monkeypatch.setattr(
            "agent_engine_runner_shared.launcher.setup_logging", lambda **_kwargs: None
        )
        monkeypatch.setenv("AGENT_ENTRYPOINT", "fake_module:start")

        with (
            patch(
                "agent_engine_runner_shared.launcher.materialize_mcp_oauth_secret_cache",
                side_effect=RuntimeError("invalid MCP OAuth secret AGENTIC_MCP_OAUTH_B64_GITHUB"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        text = path.read_text()
        assert "Cannot materialize MCP OAuth credentials" in text
        assert "AGENTIC_MCP_OAUTH_B64_GITHUB" in text
