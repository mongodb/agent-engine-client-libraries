"""
Integration tests proving TLS client fail-closed behavior at call sites.

These tests verify that HTTPS misconfiguration is caught at client creation
time (when the wrapper or proxy is instantiated), not at request time.
"""

import os
from unittest import mock

import pytest

from agent_engine_runner_shared.secure_wrapper import (
    PolicyDeniedException,
    ToolExecutionError,
    request_oe_approval,
)


@pytest.fixture
def clear_tls_env():
    """Clear TLS environment variables before each test."""
    env_vars = ["TLS_CERT_PATH", "TLS_KEY_PATH", "TLS_CA_CERT_PATH"]
    original = {k: os.environ.get(k) for k in env_vars}
    for k in env_vars:
        os.environ.pop(k, None)
    yield
    # Restore original values
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class TestSecureWrapperHTTPSFailClosed:
    """Test that secure_wrapper fails closed on HTTPS misconfiguration."""

    def test_request_oe_approval_fails_on_https_without_certs(self, clear_tls_env):
        """
        request_oe_approval should fail immediately when OE URL is HTTPS
        but certificates are not configured.

        This proves fail-closed: misconfiguration is caught at client creation
        time, not when the request is made.
        """
        https_oe_url = "https://oe-service:8443"

        # Should raise ValueError from create_httpx_client_with_tls,
        # not PolicyDeniedException
        with pytest.raises(ValueError) as exc_info:
            request_oe_approval(
                oe_url=https_oe_url,
                execution_id="test-exec-123",
                tool_name="test_tool",
                arguments={"arg": "value"},
                step=1,
            )

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg
        assert "fail-closed policy" in error_msg
        # Should mention the missing env vars
        assert "TLS_CERT_PATH" in error_msg

    def test_http_oe_url_does_not_require_certs(self, clear_tls_env):
        """
        HTTP OE URLs should work without certificates (backward compatibility).

        The request will fail because there's no actual OE server, but the
        client should be created successfully without TLS config.
        """
        import httpx

        http_oe_url = "http://localhost:8000"

        # Mock httpx.Client.post to avoid actual network call.
        # Must raise httpx.HTTPError subclass (not generic Exception) because
        # request_oe_approval only catches httpx.HTTPError.
        with mock.patch("httpx.Client.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            # Should raise PolicyDeniedException (OE unreachable), not ValueError
            with pytest.raises(PolicyDeniedException) as exc_info:
                request_oe_approval(
                    oe_url=http_oe_url,
                    execution_id="test-exec-123",
                    tool_name="test_tool",
                    arguments={"arg": "value"},
                    step=1,
                )

            # This proves the client was created (HTTP doesn't need certs)
            assert "OE unreachable" in str(exc_info.value)

    def test_https_with_partial_cert_config_fails_immediately(self, clear_tls_env, tmp_path):
        """
        Partial certificate configuration should fail immediately, not at request time.
        """
        # Set only cert path, missing key and CA
        cert_path = tmp_path / "cert.pem"
        cert_pem = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"
        cert_path.write_text(cert_pem)
        os.environ["TLS_CERT_PATH"] = str(cert_path)

        https_oe_url = "https://oe-service:8443"

        with pytest.raises(ValueError) as exc_info:
            request_oe_approval(
                oe_url=https_oe_url,
                execution_id="test-exec-123",
                tool_name="test_tool",
                arguments={"arg": "value"},
                step=1,
            )

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg
        assert "fail-closed policy" in error_msg


class TestReportOeResultHTTPSFailClosed:
    """Test that result settlement fails closed on HTTPS misconfiguration."""

    def test_report_oe_result_rejects_https_without_certs(self, clear_tls_env):
        """
        An in-process tool result is incomplete until OE acknowledges it, so a
        missing mTLS configuration must fail the call instead of being swallowed.
        """
        from agent_engine_runner_shared.secure_wrapper import report_oe_result

        https_oe_url = "https://oe-service:8443"

        with pytest.raises(ToolExecutionError, match="Failed to prepare OE result settlement"):
            report_oe_result(
                oe_url=https_oe_url,
                execution_id="test-exec-123",
                tool_name="test_tool",
                step=1,
                status="success",
                result={"output": "test"},
                error=None,
                duration_ms=100.0,
            )


class TestNodeLoggerHTTPSFailClosed:
    """Test that NodeExecutionLogger swallows HTTPS misconfiguration (fire-and-forget)."""

    def test_node_logger_send_swallows_https_without_certs(self, clear_tls_env, caplog):
        """
        NodeExecutionLogger should swallow HTTPS misconfiguration errors (fire-and-forget).

        Tests through the public on_node_start callback which calls _send_node_event
        internally. The fail-closed ValueError is caught and logged to avoid aborting
        node execution over logging failures.
        """
        import logging

        from agent_engine_runner_shared.node_logger import NodeExecutionLogger

        https_oe_url = "https://oe-service:8443"

        logger = NodeExecutionLogger(
            oe_url=https_oe_url,
            execution_id="test-exec-123",
        )

        # Capture debug logs
        with caplog.at_level(logging.DEBUG, logger="agent_engine_runner_shared.node_logger"):
            # Should NOT raise (fire-and-forget swallows the error)
            logger.on_node_start(
                node_name="test_node",
                inputs={"test": "input"},
                run_id="test-run-123",
            )

        # Verify the error was logged at DEBUG level
        assert any(
            "Failed to send node event to OE" in record.message
            and "HTTPS URL requires mTLS" in record.message
            for record in caplog.records
        ), "Expected debug log about HTTPS/mTLS configuration failure"


class TestAERServerHTTPSFailClosed:
    """Test that AER server client creation fails closed on HTTPS misconfiguration."""

    @pytest.mark.asyncio
    async def test_aer_get_client_fails_on_https_without_certs(self, clear_tls_env):
        """
        AER._get_client() should fail when OE_URL is HTTPS without certs.

        This is the hottest path this PR touches (AER per-chunk delivery to OE).
        Fail-closed ensures misconfiguration is caught at client creation, not
        when chunks are posted during agent execution.
        """
        from agent_engine_runner_shared.server.aer import AERServer

        https_oe_url = "https://oe-service:8443"

        # Create minimal AERServer instance (doesn't need runtime for this test)
        # AERServer.__init__ doesn't require runtime to be fully initialized
        aer = AERServer(runtime=mock.MagicMock())

        # Calling _get_client with HTTPS URL should raise ValueError immediately
        with pytest.raises(ValueError) as exc_info:
            await aer._get_client(https_oe_url)

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg
        assert "fail-closed policy" in error_msg
        # Should mention the missing env vars
        assert "TLS_CERT_PATH" in error_msg or "TLS_CA_CERT_PATH" in error_msg


class TestProgressEmitHTTPSFailClosed:
    """Test that progress.emit fails closed on HTTPS misconfiguration.

    Note: progress.emit is best-effort and catches all exceptions, logging them
    at DEBUG level. The fail-closed behavior occurs inside the function but is
    swallowed to prevent aborting the tool function.
    """

    def test_progress_emit_swallows_https_misconfiguration(self, clear_tls_env, caplog):
        """
        progress.emit should swallow HTTPS misconfiguration errors (best-effort).

        The fail-closed ValueError is raised when creating the client, but
        progress.emit catches it and logs at DEBUG to avoid aborting the tool.
        This test verifies the intended best-effort behavior.
        """
        import logging

        from agent_engine_runner_shared import progress
        from agent_engine_runner_shared.context import current_execution_id, current_oe_url

        https_oe_url = "https://oe-service:8443"

        # Set up context
        current_execution_id.set("test-exec-123")
        current_oe_url.set(https_oe_url)

        # Capture debug logs
        with caplog.at_level(logging.DEBUG, logger="agent_engine_runner_shared.progress"):
            # Should NOT raise (best-effort swallows the error)
            progress.emit("test_event", "test data")

        # Verify the error was logged at DEBUG level
        assert any(
            "failed to send chunk" in record.message and "HTTPS URL requires mTLS" in record.message
            for record in caplog.records
        ), "Expected debug log about HTTPS/mTLS configuration failure"


class TestVMModePEMTempFileWriteFailure:
    """Test that VM mode temp file write failures produce clear error messages."""

    def test_pem_temp_file_write_failure_raises_clear_error(self, clear_tls_env, tmp_path):
        """
        When TLS_*_PEM env vars are set but temp file write fails, the error
        should clearly state the write failure, not the generic "env vars must be set".

        In VM mode the env vars ARE set (SecretKeyRef injects PEM content), so
        the generic error is misleading. The real cause is filesystem issues
        (read-only /tmp, disk full, permission denied).
        """
        from agent_engine_runner_shared.tls_client import _get_cert_path_from_env

        # Set PEM env var (simulating VM mode with SecretKeyRef)
        os.environ["TLS_CERT_PEM"] = (
            "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"
        )

        # Mock mkstemp to raise OSError (simulating write failure)
        def failing_mkstemp(*args, **kwargs):
            raise OSError("[Errno 30] Read-only file system")

        with mock.patch(
            "agent_engine_runner_shared.tls_client.tempfile.mkstemp", side_effect=failing_mkstemp
        ):
            # Should raise RuntimeError with clear message about write failure
            with pytest.raises(RuntimeError) as exc_info:
                _get_cert_path_from_env("CERT")

            error_msg = str(exc_info.value)
            assert "Failed to write TLS_CERT_PEM to temporary file" in error_msg
            assert "VM mode requires writable /tmp" in error_msg
            # Should NOT be the generic "env vars must be set" error
            assert "all three env vars must be set" not in error_msg
