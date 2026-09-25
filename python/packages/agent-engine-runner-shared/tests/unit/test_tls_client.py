"""
Unit tests for tls_client module (HTTPS mTLS configuration).

Tests both the happy path and fail-closed error paths to ensure HTTPS
misconfiguration is detected at client creation time, not at request time.
"""

import os
import ssl
from pathlib import Path
from unittest import mock

import httpx
import pytest

from agent_engine_runner_shared.tls_client import (
    create_async_httpx_client_with_tls,
    create_httpx_client_with_tls,
)


@pytest.fixture
def mock_certs(tmp_path: Path) -> dict[str, Path]:
    """Create temporary certificate files for testing."""
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    ca_path = tmp_path / "ca.pem"

    # Write minimal PEM content (not valid certs, but parseable as PEM)
    cert_pem = "-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----\n"
    key_pem = "-----BEGIN PRIVATE KEY-----\nMOCK\n-----END PRIVATE KEY-----\n"
    cert_path.write_text(cert_pem)
    key_path.write_text(key_pem)
    ca_path.write_text(cert_pem)

    return {
        "cert": cert_path,
        "key": key_path,
        "ca": ca_path,
    }


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


class TestHTTPURLs:
    """Test behavior with plain HTTP URLs (no TLS required)."""

    def test_http_url_returns_plain_client(self, clear_tls_env):
        """HTTP URLs should return a plain client without TLS configuration."""
        client = create_httpx_client_with_tls("http://localhost:8000", 30.0)
        assert isinstance(client, httpx.Client)
        # Verify it's not configured with custom TLS (using default transport)
        # The client should work normally
        client.close()

    @pytest.mark.asyncio
    async def test_http_url_returns_plain_async_client(self, clear_tls_env):
        """HTTP URLs should return a plain async client without TLS configuration."""
        client = await create_async_httpx_client_with_tls("http://localhost:8000", 30.0)
        assert isinstance(client, httpx.AsyncClient)
        await client.aclose()


class TestHTTPSWithoutCerts:
    """Test fail-closed behavior when HTTPS is used without certificates."""

    def test_https_without_certs_raises_error(self, clear_tls_env):
        """HTTPS URL without any certs should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg
        assert "fail-closed policy" in error_msg
        assert "TLS_CERT_PATH" in error_msg
        assert "TLS_KEY_PATH" in error_msg
        assert "TLS_CA_CERT_PATH" in error_msg

    @pytest.mark.asyncio
    async def test_async_https_without_certs_raises_error(self, clear_tls_env):
        """HTTPS URL without any certs should raise ValueError (async)."""
        with pytest.raises(ValueError) as exc_info:
            await create_async_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg


class TestPartialCertConfig:
    """Test fail-closed behavior with incomplete certificate configuration."""

    def test_cert_only_raises_error(self, clear_tls_env, mock_certs):
        """HTTPS with only cert path should raise ValueError."""
        os.environ["TLS_CERT_PATH"] = str(mock_certs["cert"])
        # Missing TLS_KEY_PATH and TLS_CA_CERT_PATH

        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg
        assert "fail-closed policy" in error_msg

    def test_cert_and_key_only_raises_error(self, clear_tls_env, mock_certs):
        """HTTPS with cert and key but no CA should raise ValueError."""
        os.environ["TLS_CERT_PATH"] = str(mock_certs["cert"])
        os.environ["TLS_KEY_PATH"] = str(mock_certs["key"])
        # Missing TLS_CA_CERT_PATH

        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "HTTPS URL requires mTLS configuration" in error_msg

    @pytest.mark.asyncio
    async def test_async_partial_config_raises_error(self, clear_tls_env, mock_certs):
        """Async version should also fail with partial config."""
        os.environ["TLS_CERT_PATH"] = str(mock_certs["cert"])
        # Missing key and CA

        with pytest.raises(ValueError) as exc_info:
            await create_async_httpx_client_with_tls("https://oe-service:8443", 30.0)

        assert "HTTPS URL requires mTLS configuration" in str(exc_info.value)


class TestValidCertConfig:
    """Test successful client creation with valid certificate configuration."""

    def test_https_with_valid_certs_creates_client(self, clear_tls_env, mock_certs):
        """HTTPS with all three cert paths should create a client."""
        os.environ["TLS_CERT_PATH"] = str(mock_certs["cert"])
        os.environ["TLS_KEY_PATH"] = str(mock_certs["key"])
        os.environ["TLS_CA_CERT_PATH"] = str(mock_certs["ca"])

        # Mock the SSL context creation since our mock certs aren't valid
        with mock.patch("ssl.create_default_context") as mock_ssl_ctx:
            mock_ctx = mock.Mock(spec=ssl.SSLContext)
            mock_ssl_ctx.return_value = mock_ctx

            client = create_httpx_client_with_tls("https://oe-service:8443", 30.0)

            assert isinstance(client, httpx.Client)
            # Verify SSL context was created with the CA file
            mock_ssl_ctx.assert_called_once_with(
                purpose=ssl.Purpose.SERVER_AUTH,
                cafile=str(mock_certs["ca"]),
            )
            # Verify client cert was loaded
            mock_ctx.load_cert_chain.assert_called_once_with(
                certfile=str(mock_certs["cert"]),
                keyfile=str(mock_certs["key"]),
            )

            client.close()

    @pytest.mark.asyncio
    async def test_async_https_with_valid_certs_creates_client(self, clear_tls_env, mock_certs):
        """Async HTTPS with all three cert paths should create a client."""
        os.environ["TLS_CERT_PATH"] = str(mock_certs["cert"])
        os.environ["TLS_KEY_PATH"] = str(mock_certs["key"])
        os.environ["TLS_CA_CERT_PATH"] = str(mock_certs["ca"])

        with mock.patch("ssl.create_default_context") as mock_ssl_ctx:
            mock_ctx = mock.Mock(spec=ssl.SSLContext)
            mock_ssl_ctx.return_value = mock_ctx

            client = await create_async_httpx_client_with_tls("https://oe-service:8443", 30.0)

            assert isinstance(client, httpx.AsyncClient)
            mock_ssl_ctx.assert_called_once()
            mock_ctx.load_cert_chain.assert_called_once()

            await client.aclose()


class TestInvalidCertFiles:
    """Test error handling when certificate files are invalid or missing."""

    def test_nonexistent_cert_file_raises_error(self, clear_tls_env):
        """HTTPS with non-existent cert file should raise ValueError."""
        os.environ["TLS_CERT_PATH"] = "/nonexistent/cert.pem"
        os.environ["TLS_KEY_PATH"] = "/nonexistent/key.pem"
        os.environ["TLS_CA_CERT_PATH"] = "/nonexistent/ca.pem"

        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "Failed to configure mTLS" in error_msg
        assert "TLS_CERT_PATH" in error_msg

    def test_invalid_cert_content_raises_error(self, clear_tls_env, tmp_path):
        """HTTPS with invalid cert content should raise ValueError."""
        cert_path = tmp_path / "invalid_cert.pem"
        key_path = tmp_path / "invalid_key.pem"
        ca_path = tmp_path / "invalid_ca.pem"

        # Write invalid content (not valid PEM format)
        cert_path.write_text("NOT A VALID CERTIFICATE")
        key_path.write_text("NOT A VALID KEY")
        ca_path.write_text("NOT A VALID CA")

        os.environ["TLS_CERT_PATH"] = str(cert_path)
        os.environ["TLS_KEY_PATH"] = str(key_path)
        os.environ["TLS_CA_CERT_PATH"] = str(ca_path)

        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        error_msg = str(exc_info.value)
        assert "Failed to configure mTLS" in error_msg


class TestTimeoutConfiguration:
    """Test that timeout configuration is properly passed through."""

    def test_timeout_is_applied_to_client(self, clear_tls_env):
        """HTTP client should respect the provided timeout."""
        import httpx

        timeout = 42.0
        client = create_httpx_client_with_tls("http://localhost:8000", timeout)

        # httpx.Client.timeout is an httpx.Timeout object, not the raw float
        assert isinstance(client.timeout, httpx.Timeout)
        # When a float is passed, httpx uses it for all timeout phases
        assert client.timeout.connect == timeout
        assert client.timeout.read == timeout
        assert client.timeout.write == timeout
        assert client.timeout.pool == timeout

        client.close()

    @pytest.mark.asyncio
    async def test_async_timeout_is_applied_to_client(self, clear_tls_env):
        """Async HTTP client should respect the provided timeout."""
        import httpx

        timeout = 42.0
        client = await create_async_httpx_client_with_tls("http://localhost:8000", timeout)

        # httpx.AsyncClient.timeout is an httpx.Timeout object, not the raw float
        assert isinstance(client.timeout, httpx.Timeout)
        # When a float is passed, httpx uses it for all timeout phases
        assert client.timeout.connect == timeout
        assert client.timeout.read == timeout
        assert client.timeout.write == timeout
        assert client.timeout.pool == timeout

        await client.aclose()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_https_url_with_trailing_slash(self, clear_tls_env):
        """HTTPS URL with trailing slash should still require certs."""
        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443/", 30.0)

        assert "HTTPS URL requires mTLS configuration" in str(exc_info.value)

    def test_https_url_with_path(self, clear_tls_env):
        """HTTPS URL with path component should still require certs."""
        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443/api/v1", 30.0)

        assert "HTTPS URL requires mTLS configuration" in str(exc_info.value)

    def test_empty_cert_path_treated_as_missing(self, clear_tls_env):
        """Empty string cert path should be treated as missing."""
        os.environ["TLS_CERT_PATH"] = ""
        os.environ["TLS_KEY_PATH"] = ""
        os.environ["TLS_CA_CERT_PATH"] = ""

        with pytest.raises(ValueError) as exc_info:
            create_httpx_client_with_tls("https://oe-service:8443", 30.0)

        assert "HTTPS URL requires mTLS configuration" in str(exc_info.value)
