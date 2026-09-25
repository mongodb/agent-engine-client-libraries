"""Unit tests for ``AgentEngineToolPodBackend`` parser + classifier branches.

End-to-end behavior is covered by ``test_backend_integration.py`` which
drives the backend through a real ``SecureToolWrapper``, a fake OE HTTP
server, and the real ``agent_engine_runner_shared.toolpod_handlers``. Keep tests
**here** narrowly focused on things the integration tests cannot
cleanly exercise:

1. ``_classify_error`` — exception → ``[RETRYABLE]`` / ``[NON-RETRYABLE]``
   taxonomy (no I/O involved).
2. ``_sanitize_handler_error`` — regex coverage for path / host:port / DNS
   redaction (no I/O involved).
3. **Defensive parser branches** that production handlers never produce
   (non-dict response, missing ``occurrences`` field, malformed base64,
   ContextVar absent). Real handlers always emit well-shaped payloads;
   these tests guard the backend's behavior when the contract is broken
   from the outside.
4. **Download fan-out** — concurrency, order, ContextVar propagation, and
   the durable sequential fallback. These need a mock wrapper with a
   thread-safe in-flight counter; the real OE path cannot pin max
   in-flight without becoming a timing test of HTTPServer.

A test that "calls the wrapper with the right tool name" or "returns the
right ``LsResult`` for a happy-path payload" belongs in the integration
file — running it against a ``MagicMock`` here only proves the mock
agreed with itself.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_sdk_langgraph.backends.toolpod import (
    AgentEngineToolPodBackend,
    _classify_error,
    _get_wrapper,
    _sanitize_handler_error,
)
from agent_engine_runner_shared.context import current_wrapper
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.secure_wrapper import (
    TerminalExecutionError,
    ToolCallTimeoutError,
)
from agent_engine_runner_shared.workflow import attempt_context_scope


@pytest.fixture()
def backend_with_mock_wrapper():
    """Backend bound to a ``MagicMock`` wrapper via the ContextVar.

    Used only by the protocol-violation tests below — those need a wrapper
    that returns hand-rolled malformed payloads to exercise the backend's
    defensive parsing branches. Happy-path coverage uses real handlers in
    ``test_backend_integration.py``.
    """
    backend = AgentEngineToolPodBackend()
    mock_wrapper = MagicMock()
    with patch(
        "agent_engine_sdk_langgraph.backends.toolpod.current_wrapper"
    ) as mock_ctx:
        mock_ctx.get.return_value = mock_wrapper
        yield backend, mock_wrapper


# ---------------------------------------------------------------------------
# Error classification taxonomy
# ---------------------------------------------------------------------------


class _UnexpectedError(Exception):
    pass


class TestClassifyError:
    """Programming errors are non-retryable; transient I/O is retryable.

    The default branch must stay retryable so agents don't give up on
    transients they can't recognize. ``RuntimeError`` is the load-bearing
    one — raised by ``_get_wrapper`` on a missing ContextVar — and must
    be non-retryable to avoid an infinite retry loop on misconfiguration.
    """

    @pytest.mark.parametrize(
        "exc, retryable",
        [
            (ValueError("bad input"), False),
            (KeyError("missing"), False),
            (TypeError("wrong type"), False),
            (RuntimeError("anything"), False),
            (TimeoutError("timed out"), True),
            (ConnectionError("refused"), True),
            (_UnexpectedError("mystery"), True),
            # A tool that burned its whole deadline: retrying spends another
            # full one to most likely fail identically.
            (ToolCallTimeoutError("slow_tool", 600.0, 601.0), False),
            (
                TerminalExecutionError(
                    "This execution already ended in error; later tool calls are rejected."
                ),
                False,
            ),
        ],
    )
    def test_taxonomy(self, exc: Exception, retryable: bool) -> None:
        msg, got_retryable = _classify_error(exc)
        assert got_retryable is retryable
        prefix = "[RETRYABLE]" if retryable else "[NON-RETRYABLE]"
        assert prefix in msg


# ---------------------------------------------------------------------------
# Handler-error sanitizer
# ---------------------------------------------------------------------------


class TestSanitizeHandlerError:
    """The four redaction patterns must not leak internal topology.

    ``_classify_error`` enforces the same invariant for exception-based
    failures; the sanitizer covers handler-supplied error strings, which
    take a different path through the backend.
    """

    def test_redacts_workspace_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORKSPACE_DIR", "/tmp/agent-workspace")
        out = _sanitize_handler_error(
            "permission denied at /tmp/agent-workspace/foo/bar"
        )
        assert "/tmp/agent-workspace" not in out
        assert "<workspace>" in out

    def test_redacts_host_port_pairs(self) -> None:
        out = _sanitize_handler_error("connection refused to db.internal:27017")
        assert "27017" not in out
        assert "db.internal" not in out
        assert "<host>" in out

    @pytest.mark.parametrize(
        "ipv6_host_port",
        [
            "[::1]:8080",
            "[fd00::1]:27017",
            "[fe80::abcd]:6443",
        ],
    )
    def test_redacts_ipv6_host_port_pairs(self, ipv6_host_port: str) -> None:
        """Bracketed IPv6 ``host:port`` pairs must be sanitized too — the
        alphanumeric ``_HOST_PORT_RE`` can't match ``[`` (its ``\\b`` anchor
        and ``[a-z0-9]`` leading class both reject the bracket), so without
        the dedicated IPv6 regex these would leak unsanitized.
        """
        out = _sanitize_handler_error(f"connection refused to {ipv6_host_port}")
        assert ipv6_host_port not in out
        assert "<host>" in out

    def test_redacts_cluster_internal_dns(self) -> None:
        out = _sanitize_handler_error("upstream unreachable: db.foo.svc.cluster.local")
        assert "svc.cluster.local" not in out
        assert "<host>" in out

    def test_redacts_remaining_absolute_paths(self) -> None:
        out = _sanitize_handler_error("disk full at /mnt/data/scratch")
        assert "/mnt/data" not in out
        assert "<path>" in out

    def test_preserves_failure_category(self) -> None:
        """Sanitization must not eat the human-readable category prefix —
        the agent still needs to know *what* went wrong.
        """
        out = _sanitize_handler_error(
            "permission denied at /tmp/agent-workspace/foo; "
            "upstream unreachable: db.internal.svc.cluster.local:27017"
        )
        assert "permission denied" in out
        assert "upstream unreachable" in out


# ---------------------------------------------------------------------------
# ContextVar enforcement
# ---------------------------------------------------------------------------


class TestContextVarEnforcement:
    """A missing ``SecureToolWrapper`` is a programming bug, not a
    transient. It must classify non-retryable so the agent doesn't loop
    on misconfiguration, and the raw ``RuntimeError`` text must not leak.
    """

    def test_get_wrapper_raises_when_no_context(self) -> None:
        with patch(
            "agent_engine_sdk_langgraph.backends.toolpod.current_wrapper"
        ) as mock_ctx:
            mock_ctx.get.return_value = None
            with pytest.raises(RuntimeError, match="No SecureToolWrapper in context"):
                _get_wrapper()

    def test_op_returns_classified_error_when_no_context(self) -> None:
        backend = AgentEngineToolPodBackend()
        with patch(
            "agent_engine_sdk_langgraph.backends.toolpod.current_wrapper"
        ) as mock_ctx:
            mock_ctx.get.return_value = None
            result = backend.ls("/app")
            assert result.error is not None
            assert "[NON-RETRYABLE]" in result.error
            assert "No SecureToolWrapper in context" not in result.error


# ---------------------------------------------------------------------------
# Defensive parser branches — payload shapes real handlers never emit
# ---------------------------------------------------------------------------


class TestProtocolViolationGuards:
    """Production ``agent_engine_runner_shared.toolpod_handlers`` always emit well-shaped
    payloads, but a malicious or buggy alternate handler implementation
    could return malformed dicts. The backend MUST NOT silently accept
    them — every method validates shape and reports
    ``[NON-RETRYABLE] ... protocol violation`` on mismatch.
    """

    def test_write_rejects_non_dict_response(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps("unexpected")
        result = backend.write("/f.txt", "x")
        assert result.error is not None
        assert "protocol violation" in result.error

    def test_write_rejects_missing_path_in_success(
        self, backend_with_mock_wrapper
    ) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"status": "ok"})
        result = backend.write("/f.txt", "x")
        assert result.error is not None
        assert "missing 'path'" in result.error

    def test_edit_rejects_missing_occurrences(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"path": "/f.txt"})
        result = backend.edit("/f.txt", "a", "b")
        assert result.error is not None
        assert "missing 'occurrences'" in result.error

    def test_edit_rejects_non_int_occurrences(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"occurrences": "three"})
        result = backend.edit("/f.txt", "a", "b")
        assert result.error is not None

    def test_execute_rejects_missing_output(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"exit_code": 0})
        result = backend.execute("echo")
        assert result.exit_code == 1
        assert "protocol violation" in result.output

    def test_execute_rejects_string_response(self, backend_with_mock_wrapper) -> None:
        """A raw string response (no dict shape) must NOT silently become
        the agent-visible output — that would let a buggy handler skip
        the exit-code contract."""
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = "raw output text"
        result = backend.execute("echo hello")
        assert result.exit_code == 1
        assert "protocol violation" in result.output


class TestDownloadFilesParserGuards:
    """Defensive branches in ``download_files`` that real handlers never
    trip but the backend must handle without crashing the agent turn.
    """

    def test_invalid_base64_returns_error(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps(
            {"path": "/ws/a.txt", "content_base64": "!!!not-base64!!!"}
        )
        results = backend.download_files(["a.txt"])
        assert len(results) == 1
        assert results[0].content is None
        assert results[0].error is not None

    def test_missing_content_field_returns_error(
        self, backend_with_mock_wrapper
    ) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"unexpected": "shape"})
        results = backend.download_files(["a.txt"])
        assert results[0].content is None
        assert results[0].error is not None

    def test_handler_error_dict_is_surfaced(self, backend_with_mock_wrapper) -> None:
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps({"error": "file_not_found"})
        results = backend.download_files(["missing.txt"])
        assert results[0].error == "file_not_found"

    def test_empty_paths_returns_empty_list(self, backend_with_mock_wrapper) -> None:
        backend, _ = backend_with_mock_wrapper
        assert backend.download_files([]) == []

    def test_wire_name_is_filesystem_download(self, backend_with_mock_wrapper) -> None:
        """Pin the wire tool name. A rename would silently decouple the
        backend from the agent-engine-runner-shared filesystem_download handler.
        """
        backend, mock_wrapper = backend_with_mock_wrapper
        mock_wrapper.execute_tool.return_value = json.dumps(
            {
                "path": "/ws/a.txt",
                "content_base64": base64.b64encode(b"x").decode("ascii"),
            }
        )
        backend.download_files(["a.txt"])
        tool_name, args = mock_wrapper.execute_tool.call_args.args
        assert tool_name == "filesystem_download"
        assert args == {"file_path": "a.txt"}
        assert mock_wrapper.execute_tool.call_args.kwargs == {"is_local": False}


# ---------------------------------------------------------------------------
# Download fan-out (native parallel, durable sequential)
# ---------------------------------------------------------------------------

_FANOUT_DELAY_S = 0.05
_FANOUT_PATHS = ["a.txt", "b.txt", "c.txt", "d.txt"]


class _InFlight:
    """Thread-safe peak-concurrency counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.max = 0

    def __enter__(self) -> _InFlight:
        with self._lock:
            self.current += 1
            if self.current > self.max:
                self.max = self.current
        return self

    def __exit__(self, *args: object) -> None:
        with self._lock:
            self.current -= 1


def _payload(path: str, content: bytes) -> str:
    return json.dumps(
        {
            "path": path,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
    )


def _wrapper_with_delay(
    inflight: _InFlight,
    delay_s: float | dict[str, float],
    payloads: dict[str, str] | None = None,
) -> MagicMock:
    """Mock wrapper that sleeps per call so overlap is observable."""

    def execute_tool(
        _tool_name: str, arguments: dict[str, Any], *, is_local: bool
    ) -> str:
        assert is_local is False
        path = arguments["file_path"]
        sleep_for = delay_s[path] if isinstance(delay_s, dict) else delay_s
        with inflight:
            time.sleep(sleep_for)
        if payloads is not None:
            return payloads[path]
        return _payload(path, path.encode())

    wrapper = MagicMock()
    wrapper.execute_tool.side_effect = execute_tool
    return wrapper


@contextmanager
def _bound_backend(wrapper: MagicMock) -> Iterator[AgentEngineToolPodBackend]:
    """Bind *wrapper* on the real ContextVar.

    Fan-out workers only see the wrapper if ``copy_context`` is used; a
    regression that drops the copy makes ``_get_wrapper`` raise and every
    response come back ``[NON-RETRYABLE]``.
    """
    token = current_wrapper.set(wrapper)
    try:
        yield AgentEngineToolPodBackend()
    finally:
        current_wrapper.reset(token)


class TestDownloadFilesFanOut:
    def test_download_files_fans_out(self) -> None:
        inflight = _InFlight()
        wrapper = _wrapper_with_delay(inflight, _FANOUT_DELAY_S)
        with _bound_backend(wrapper) as backend:
            results = backend.download_files(_FANOUT_PATHS)

        assert inflight.max > 1
        assert [r.path for r in results] == _FANOUT_PATHS
        assert [r.content for r in results] == [p.encode() for p in _FANOUT_PATHS]
        assert all(r.error is None for r in results)

    @pytest.mark.anyio
    async def test_adownload_files_fans_out(self) -> None:
        inflight = _InFlight()
        wrapper = _wrapper_with_delay(inflight, _FANOUT_DELAY_S)
        with _bound_backend(wrapper) as backend:
            results = await backend.adownload_files(_FANOUT_PATHS)

        assert inflight.max > 1
        assert [r.path for r in results] == _FANOUT_PATHS
        assert all(r.error is None for r in results)

    def test_download_files_preserves_order_when_slow_path_finishes_last(self) -> None:
        inflight = _InFlight()
        delays = {"slow.txt": 0.08, "fast.txt": 0.01}
        wrapper = _wrapper_with_delay(inflight, delays)
        with _bound_backend(wrapper) as backend:
            results = backend.download_files(["slow.txt", "fast.txt"])

        assert [r.path for r in results] == ["slow.txt", "fast.txt"]
        assert [r.content for r in results] == [b"slow.txt", b"fast.txt"]

    @pytest.mark.anyio
    async def test_adownload_files_preserves_order_when_slow_path_finishes_last(
        self,
    ) -> None:
        inflight = _InFlight()
        delays = {"slow.txt": 0.08, "fast.txt": 0.01}
        wrapper = _wrapper_with_delay(inflight, delays)
        with _bound_backend(wrapper) as backend:
            results = await backend.adownload_files(["slow.txt", "fast.txt"])

        assert [r.path for r in results] == ["slow.txt", "fast.txt"]
        assert [r.content for r in results] == [b"slow.txt", b"fast.txt"]

    def test_download_files_isolates_per_file_errors(self) -> None:
        inflight = _InFlight()
        payloads = {
            "ok.txt": _payload("ok.txt", b"ok"),
            "missing.txt": json.dumps({"error": "file_not_found"}),
            "bad.txt": json.dumps({"path": "bad.txt", "content_base64": "!!!"}),
        }
        wrapper = _wrapper_with_delay(inflight, 0.01, payloads=payloads)
        with _bound_backend(wrapper) as backend:
            results = backend.download_files(["ok.txt", "missing.txt", "bad.txt"])

        assert results[0].content == b"ok"
        assert results[0].error is None
        assert results[1].content is None
        assert results[1].error == "file_not_found"
        assert results[2].content is None
        assert results[2].error is not None
        assert "invalid base64" in results[2].error

    @pytest.mark.anyio
    async def test_adownload_files_isolates_per_file_errors(self) -> None:
        inflight = _InFlight()
        payloads = {
            "ok.txt": _payload("ok.txt", b"ok"),
            "missing.txt": json.dumps({"error": "file_not_found"}),
            "bad.txt": json.dumps({"path": "bad.txt", "content_base64": "!!!"}),
        }
        wrapper = _wrapper_with_delay(inflight, 0.01, payloads=payloads)
        with _bound_backend(wrapper) as backend:
            results = await backend.adownload_files(
                ["ok.txt", "missing.txt", "bad.txt"]
            )

        assert results[0].content == b"ok"
        assert results[0].error is None
        assert results[1].error == "file_not_found"
        assert results[2].content is None
        assert results[2].error is not None

    def test_download_files_stays_sequential_under_durable_attempt(self) -> None:
        inflight = _InFlight()
        wrapper = _wrapper_with_delay(inflight, 0.03)
        with _bound_backend(wrapper) as backend:
            with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
                results = backend.download_files(_FANOUT_PATHS)

        assert inflight.max == 1
        assert [r.path for r in results] == _FANOUT_PATHS
        assert all(r.error is None for r in results)

    @pytest.mark.anyio
    async def test_adownload_files_stays_sequential_under_durable_attempt(self) -> None:
        inflight = _InFlight()
        wrapper = _wrapper_with_delay(inflight, 0.03)
        with _bound_backend(wrapper) as backend:
            with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
                results = await backend.adownload_files(_FANOUT_PATHS)

        assert inflight.max == 1
        assert [r.path for r in results] == _FANOUT_PATHS
        assert all(r.error is None for r in results)

    def test_empty_paths_returns_empty_list_without_wrapper_calls(self) -> None:
        wrapper = MagicMock()
        with _bound_backend(wrapper) as backend:
            assert backend.download_files([]) == []
        wrapper.execute_tool.assert_not_called()

    def test_single_path_stays_sequential(self) -> None:
        inflight = _InFlight()
        wrapper = _wrapper_with_delay(inflight, 0.02)
        with _bound_backend(wrapper) as backend:
            results = backend.download_files(["only.txt"])

        assert inflight.max == 1
        assert len(results) == 1
        assert results[0].content == b"only.txt"
        assert results[0].error is None


# ---------------------------------------------------------------------------
# Identity / not-implemented contract
# ---------------------------------------------------------------------------


class TestStaticContract:
    def test_id_is_agent_engine_toolpod(self) -> None:
        assert AgentEngineToolPodBackend().id == "agent-engine-toolpod"

    def test_upload_files_returns_classified_per_file_error(self) -> None:
        """Returning per-file classified errors keeps the [NON-RETRYABLE]
        contract every other protocol method honours; raising raw
        NotImplementedError would bypass ``_safe_call`` and reach the agent
        graph un-prefixed."""
        responses = AgentEngineToolPodBackend().upload_files(
            [("a.txt", b"a"), ("b.txt", b"b")]
        )
        assert len(responses) == 2
        for response, expected_path in zip(responses, ["a.txt", "b.txt"]):
            assert response.path == expected_path
            assert response.error is not None
            assert response.error.startswith("[NON-RETRYABLE]")
            assert "upload_files" in response.error
