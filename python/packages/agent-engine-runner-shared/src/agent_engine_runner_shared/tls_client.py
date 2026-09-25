"""
TLS client configuration for HTTPS connections with mTLS support.

This module provides utilities for configuring httpx.Client instances with
mTLS (mutual TLS) when connecting to platform services over HTTPS. When
TLS certificate paths are available via environment variables, the client
will present its certificate to the server for authentication.
"""

from __future__ import annotations

import atexit
import logging
import os
import ssl
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

# Cache for written cert paths (VM mode only).
# Write-once, never invalidated: VM mode rotation intentionally requires pod restart.
# SecretKeyRef injects TLS_*_PEM env vars once at pod startup; Kubernetes does not
# update env vars on Secret changes. Container mode (TLS_*_PATH) uses Secret volume
# mounts which Kubelet propagates on cert-manager rotation, enabling mtime-based
# detection. VM mode cannot detect rotation via mtime, so _get_cert_mtimes() in
# progress.py/aer.py returns None for PEM mode, skipping rotation checks.
_decoded_cert_paths: dict[str, str] = {}


@contextmanager
def _propagate_without_http_span(request: httpx.Request) -> Iterator[None]:
    """Carry the active trace across a platform RPC without recording transport."""
    try:
        from opentelemetry.instrumentation.utils import suppress_http_instrumentation
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        yield
        return

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    request.headers.update(carrier)
    with suppress_http_instrumentation():
        yield


class _PlatformHTTPClient(httpx.Client):
    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        with _propagate_without_http_span(request):
            return super().send(request, **kwargs)


class _PlatformAsyncHTTPClient(httpx.AsyncClient):
    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        with _propagate_without_http_span(request):
            return await super().send(request, **kwargs)


def _get_cert_path_from_env(cert_type: str) -> str:
    """
    Get certificate path from environment, writing PEM content to temp file if needed.

    For container mode, reads TLS_*_PATH directly.
    For VM mode, reads TLS_*_PEM (raw PEM from SecretKeyRef), writes to /tmp.

    Args:
        cert_type: One of "CERT", "KEY", or "CA_CERT"

    Returns:
        Path to the certificate file, or empty string if not available
    """
    # Check for direct path first (container mode)
    path_env = f"TLS_{cert_type}_PATH"
    if path := os.environ.get(path_env, ""):
        return path

    # Check for raw PEM content (VM mode - SecretKeyRef injects decoded value)
    pem_env = f"TLS_{cert_type}_PEM"
    pem_content = os.environ.get(pem_env, "")
    if not pem_content:
        return ""

    # Return cached path if already written
    if pem_env in _decoded_cert_paths:
        return _decoded_cert_paths[pem_env]

    # Write PEM content to temp file
    try:
        # Use named temp file with appropriate suffix
        suffix_map = {"CERT": ".crt", "KEY": ".key", "CA_CERT": ".crt"}
        suffix = suffix_map.get(cert_type, ".pem")

        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f"tls-{cert_type.lower()}-")
        try:
            os.write(fd, pem_content.encode("utf-8"))
        finally:
            os.close(fd)

        # Make it read-only (especially important for private keys)
        os.chmod(temp_path, 0o400)

        # Register cleanup on interpreter shutdown to unlink the temp file.
        # This keeps private keys off disk deterministically rather than relying
        # on /tmp cleanup. Fine to leave for pod lifetime in ephemeral single-tenant
        # pods, but atexit mirrors TypeScript's closeAllTLSAgents() pattern.
        def cleanup_temp_cert():
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary TLS file: {temp_path}")
            except OSError:
                pass  # File already deleted or pod terminating

        atexit.register(cleanup_temp_cert)

        # Cache the path
        _decoded_cert_paths[pem_env] = temp_path

        logger.debug(
            f"Wrote TLS_{cert_type}_PEM to temporary file",
            extra={"temp_path": temp_path},
        )

        return temp_path
    except Exception as e:
        # Re-raise with a clear message so fail-closed error reflects the actual cause.
        # Don't return "" - that would trigger the generic "all three env vars must be
        # set" error, which is misleading in VM mode where the vars ARE set but the
        # write failed (e.g., read-only filesystem, disk full, permission denied).
        raise RuntimeError(
            f"Failed to write TLS_{cert_type}_PEM to temporary file for mTLS: {e}. "
            f"VM mode requires writable /tmp for cert materialization. Check filesystem "
            f"permissions and available disk space."
        ) from e


def create_httpx_client_with_tls(
    base_url: str,
    timeout: float | httpx.Timeout,
    *,
    verify_hostname: bool = True,
    set_base_url: bool = False,
) -> httpx.Client:
    """
    Create an httpx.Client configured for the given base URL.

    If base_url uses HTTPS and TLS certificate environment variables are set
    (TLS_CERT_PATH, TLS_KEY_PATH, TLS_CA_CERT_PATH), the client will be
    configured with mTLS. For HTTP URLs or when certificates are not available,
    returns a standard httpx.Client with no TLS configuration.

    Args:
        base_url: The target service URL (http:// or https://)
        timeout: httpx timeout configuration
        verify_hostname: Whether to verify the server's hostname against its
            certificate's Subject Alternative Names (SANs). Default is True
            (secure). Only set to False for testing with self-signed certs.
        set_base_url: If True, set the base_url on the returned client so
            relative paths in requests are resolved against it. Default False.

    Returns:
        httpx.Client configured with mTLS if applicable

    Raises:
        ValueError: If HTTPS is used but TLS configuration is incomplete
            (some but not all certificate paths are set)
    """
    # HTTP URLs don't need TLS configuration
    if not base_url.startswith("https://"):
        return _PlatformHTTPClient(timeout=timeout, base_url=base_url if set_base_url else "")

    # HTTPS URLs require mTLS configuration for AER→OE communication.
    # Fail-closed: if the URL is HTTPS, we must have client certificates.
    # Support both direct paths (container mode) and PEM env content (VM mode).
    cert_path = _get_cert_path_from_env("CERT")
    key_path = _get_cert_path_from_env("KEY")
    ca_path = _get_cert_path_from_env("CA_CERT")

    # All three certificate paths are required for HTTPS
    if not cert_path or not key_path or not ca_path:
        raise ValueError(
            f"HTTPS URL requires mTLS configuration (fail-closed policy): "
            f"all three env vars must be set when base_url uses https:// "
            f"(TLS_CERT_PATH={cert_path!r}, TLS_KEY_PATH={key_path!r}, "
            f"TLS_CA_CERT_PATH={ca_path!r}). "
            f"For AER→OE communication, the OE HTTPS server (port 8443) requires "
            f"client certificates. If TLS certificates are not available, use "
            f"http:// in the URL."
        )

    # Load and configure mTLS
    try:
        # Create SSL context with client certificate
        # Using PROTOCOL_TLS_CLIENT ensures modern TLS (1.2+) and secure defaults
        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=ca_path,
        )

        # Load client certificate and key
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

        # Hostname verification: enabled by default (secure), disable for testing
        if not verify_hostname:
            ssl_context.check_hostname = False
            logger.warning("TLS hostname verification disabled; only for testing")

        logger.info(
            "Configured HTTPS client with mTLS",
            extra={
                "cert_path": cert_path,
                "ca_path": ca_path,
                "verify_hostname": verify_hostname,
            },
        )

        return _PlatformHTTPClient(
            timeout=timeout,
            verify=ssl_context,
            base_url=base_url if set_base_url else "",
        )

    except Exception as e:
        raise ValueError(
            f"Failed to configure mTLS for HTTPS connection: {e}. "
            f"Check that TLS_CERT_PATH ({cert_path}), TLS_KEY_PATH ({key_path}), "
            f"and TLS_CA_CERT_PATH ({ca_path}) point to valid certificate files."
        ) from e


async def create_async_httpx_client_with_tls(
    base_url: str,
    timeout: float | httpx.Timeout,
    *,
    verify_hostname: bool = True,
) -> httpx.AsyncClient:
    """
    Create an httpx.AsyncClient configured for the given base URL.

    Async version of create_httpx_client_with_tls. See that function for
    full documentation.

    Args:
        base_url: The target service URL (http:// or https://)
        timeout: httpx timeout configuration
        verify_hostname: Whether to verify the server's hostname. Default True.

    Returns:
        httpx.AsyncClient configured with mTLS if applicable

    Raises:
        ValueError: If HTTPS is used but TLS configuration is incomplete
    """
    # HTTP URLs don't need TLS configuration
    if not base_url.startswith("https://"):
        return _PlatformAsyncHTTPClient(timeout=timeout)

    # HTTPS URLs require mTLS configuration for AER→OE communication.
    # Fail-closed: if the URL is HTTPS, we must have client certificates.
    # Support both direct paths (container mode) and PEM env content (VM mode).
    cert_path = _get_cert_path_from_env("CERT")
    key_path = _get_cert_path_from_env("KEY")
    ca_path = _get_cert_path_from_env("CA_CERT")

    # All three certificate paths are required for HTTPS
    if not cert_path or not key_path or not ca_path:
        raise ValueError(
            f"HTTPS URL requires mTLS configuration (fail-closed policy): "
            f"all three env vars must be set when base_url uses https:// "
            f"(TLS_CERT_PATH={cert_path!r}, TLS_KEY_PATH={key_path!r}, "
            f"TLS_CA_CERT_PATH={ca_path!r}). "
            f"For AER→OE communication, the OE HTTPS server (port 8443) requires "
            f"client certificates. If TLS certificates are not available, use "
            f"http:// in the URL."
        )

    # Load and configure mTLS
    try:
        # Create SSL context with client certificate
        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=ca_path,
        )

        # Load client certificate and key
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

        # Hostname verification
        if not verify_hostname:
            ssl_context.check_hostname = False
            logger.warning("TLS hostname verification disabled; only for testing")

        logger.info(
            "Configured async HTTPS client with mTLS",
            extra={
                "cert_path": cert_path,
                "ca_path": ca_path,
                "verify_hostname": verify_hostname,
            },
        )

        return _PlatformAsyncHTTPClient(timeout=timeout, verify=ssl_context)

    except Exception as e:
        raise ValueError(
            f"Failed to configure mTLS for HTTPS connection: {e}. "
            f"Check that TLS_CERT_PATH ({cert_path}), TLS_KEY_PATH ({key_path}), "
            f"and TLS_CA_CERT_PATH ({ca_path}) point to valid certificate files."
        ) from e
