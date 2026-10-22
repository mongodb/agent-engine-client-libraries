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
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import httpx

from agent_engine_runner_shared.caller_trace import (
    merge_runner_root_trace_state,
    runner_rooted_trace_active,
)

logger = logging.getLogger(__name__)

# Cache for written cert paths (VM mode only).
# Write-once, never invalidated: VM mode rotation intentionally requires pod restart.
# SecretKeyRef injects TLS_*_PEM env vars once at pod startup; Kubernetes does not
# update env vars on Secret changes. Container mode (TLS_*_PATH) uses Secret volume
# mounts which Kubelet propagates on cert-manager rotation, enabling mtime-based
# detection. VM mode cannot detect rotation via mtime, so _get_cert_mtimes() in
# progress.py/aer.py returns None for PEM mode, skipping rotation checks.
_decoded_cert_paths: dict[str, str] = {}


def _inject_platform_trace_headers(carrier: dict[str, str]) -> None:
    """Inject the active trace, marking roots this runner created.

    A locally created root is a remote parent on the next hop. The tracestate
    marker lets the tool pod keep guardrail content off that trace while still
    joining it. An inbound caller trace is injected unchanged.
    """
    try:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        return

    TraceContextTextMapPropagator().inject(carrier)
    if carrier.get("traceparent") and runner_rooted_trace_active():
        carrier["tracestate"] = merge_runner_root_trace_state(carrier.get("tracestate"))


@contextmanager
def _propagate_without_http_span(request: httpx.Request) -> Iterator[None]:
    """Carry the active trace across a platform RPC without recording transport."""
    try:
        from opentelemetry.instrumentation.utils import suppress_http_instrumentation
    except ImportError:
        yield
        return

    carrier: dict[str, str] = {}
    _inject_platform_trace_headers(carrier)
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


def _uses_https(base_url: str) -> bool:
    """Whether ``base_url`` is HTTPS, decided on the normalized scheme.

    ``httpx.URL`` lowercases the scheme, so ``HTTPS://…`` is HTTPS. Deciding on
    the raw string (``base_url.startswith("https://")``) would build a non-TLS
    client for a non-canonical scheme while ``_shared_client_key`` still keys it
    as https — letting a certless client be cached and served to a later
    canonical https call (fail-open). Both the key and the TLS decision must use
    the same normalization.
    """
    return httpx.URL(base_url).scheme == "https"


def create_httpx_client_with_tls(
    base_url: str,
    timeout: float | httpx.Timeout,
    *,
    verify_hostname: bool = True,
    set_base_url: bool = False,
    limits: httpx.Limits | None = None,
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
        limits: Optional connection-pool limits. ``None`` keeps httpx defaults.

    Returns:
        httpx.Client configured with mTLS if applicable

    Raises:
        ValueError: If HTTPS is used but TLS configuration is incomplete
            (some but not all certificate paths are set)
    """
    # HTTP URLs don't need TLS configuration
    pool_kwargs: dict[str, Any] = {} if limits is None else {"limits": limits}
    if not _uses_https(base_url):
        return _PlatformHTTPClient(
            timeout=timeout, base_url=base_url if set_base_url else "", **pool_kwargs
        )

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

        logger.debug(
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
            **pool_kwargs,
        )

    except Exception as e:
        raise ValueError(
            f"Failed to configure mTLS for HTTPS connection: {e}. "
            f"Check that TLS_CERT_PATH ({cert_path}), TLS_KEY_PATH ({key_path}), "
            f"and TLS_CA_CERT_PATH ({ca_path}) point to valid certificate files."
        ) from e


# ---------------------------------------------------------------------------
# Process-wide shared clients for runner -> OE callbacks
# ---------------------------------------------------------------------------
#
# Every OE callback used to build a fresh client (and therefore a fresh mTLS
# handshake) per request. On the OE side each handshake is an RSA private-key
# signature, which was the single largest fixed CPU cost under load. One
# long-lived, keep-alive client per origin lets callbacks reuse connections.
#
# keepalive_expiry: the OE HTTPS listener sets no idle timeout, so the client
# has to be the side that retires idle sockets; 60s matches the TypeScript
# runner's undici keepAliveTimeout. max_connections must exceed the number of
# concurrently *held* calls: /tool/execute keeps its connection open for the
# whole tool run, so a low cap would surface as PoolTimeout during fan-out.
_SHARED_CLIENT_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=60.0,
)

# Bound on the number of distinct origins retained at once. In a deployed
# runner the origin is the single deploy-time OE URL, but the callback URL is
# caller-supplied in local-dev and unit-test mode, so without a cap a long-lived
# process would retain a client (and its TLS context and pool) for every origin
# it ever resolved. Eviction is least-recently-used and closes the evicted
# client; mirrors TLS_AGENT_CACHE_MAX_ENTRIES in the TypeScript runner.
_MAX_SHARED_CLIENTS = 32

# Sentinel mtime tuple for "file-based certs are configured but a file could not
# be stat'd" (a mount removed or replaced mid-run). Real certificate mtimes are
# non-negative epoch seconds, so this never equals a readable mtime: the cached
# client is dropped and rebuilt — which fails closed for HTTPS — instead of
# being reused with stale TLS material. Distinct from ``None``, which means no
# file-based certs are configured and rotation is undetectable by design.
_CERTS_UNREADABLE: tuple[float, float, float] = (-1.0, -1.0, -1.0)

# Stat passes before declaring the mounted certs unreadable. A transient stat
# failure (a kubelet symlink swap, a momentary mount flip) must not drop a
# working client on the first miss, so re-stat once before failing closed.
_CERT_STAT_ATTEMPTS = 2


@dataclass
class _SharedClientEntry:
    client: httpx.Client
    cert_mtimes: Optional[tuple[float, float, float]]
    pid: int


# Insertion-ordered so the cache can evict the least-recently-used origin.
_shared_clients: "OrderedDict[str, _SharedClientEntry]" = OrderedDict()
# Guards lazy creation and rotation only; httpx.Client is thread-safe for
# requests. Callbacks run on threads spawned by asyncio.to_thread.
_shared_clients_lock = threading.Lock()


def _reset_shared_clients_after_fork() -> None:
    """Re-arm the shared-client cache in a forked child.

    A ``threading.Lock`` held by another thread at ``fork()`` time stays locked
    in the child forever — the owning thread does not exist there — so
    ``get_shared_client`` would block before it ever reached the pid check. Give
    the child a fresh lock and drop the inherited entries without closing them:
    the parent still owns those sockets.
    """
    global _shared_clients_lock
    _shared_clients_lock = threading.Lock()
    _shared_clients.clear()


if hasattr(os, "register_at_fork"):  # POSIX only; Windows has no fork
    os.register_at_fork(after_in_child=_reset_shared_clients_after_fork)


def _get_cert_mtimes() -> Optional[tuple[float, float, float]]:
    """Return mtimes of the mounted TLS files, or None when rotation is undetectable.

    Container mode (TLS_*_PATH) mounts a Secret volume that kubelet rewrites
    on cert-manager rotation, so mtimes are a complete change signal. VM mode
    (TLS_*_PEM) materialises env vars once and needs a pod restart, so it
    returns None and callers skip rotation checks.

    When file-based certs are configured but a file cannot be stat'd (the mount
    was removed or replaced mid-run), returns ``_CERTS_UNREADABLE`` rather than
    None so callers drop and rebuild the cached client instead of reusing stale
    TLS material. The stat is retried once first, so a *transient* failure does
    not drop a working client.
    """
    cert_path = os.environ.get("TLS_CERT_PATH", "")
    key_path = os.environ.get("TLS_KEY_PATH", "")
    ca_path = os.environ.get("TLS_CA_CERT_PATH", "")
    if not (cert_path and key_path and ca_path):
        return None
    for attempt in range(_CERT_STAT_ATTEMPTS):
        try:
            return (
                os.path.getmtime(cert_path),
                os.path.getmtime(key_path),
                os.path.getmtime(ca_path),
            )
        except (OSError, ValueError) as exc:
            logger.debug(
                "TLS cert stat failed (attempt %d/%d): %s",
                attempt + 1,
                _CERT_STAT_ATTEMPTS,
                exc,
            )
    return _CERTS_UNREADABLE


def _shared_client_key(base_url: str, verify_hostname: bool) -> str:
    """Cache key: the origin only, so path and trailing-slash variants share one pool."""
    url = httpx.URL(base_url)
    if not url.scheme or not url.host:
        raise ValueError(f"shared client requires an absolute URL, got {base_url!r}")
    if url.scheme not in ("http", "https"):
        # The default-port mapping below assumes http(s); another scheme would
        # otherwise be keyed under port 80 and could collide with a distinct
        # origin. Only http(s) is a supported callback transport.
        raise ValueError(f"shared client requires an http(s) URL, got {base_url!r}")
    port = url.port or (443 if url.scheme == "https" else 80)
    key = f"{url.scheme}://{url.host}:{port}"
    return key if verify_hostname else key + "#noverify"


def get_shared_client(base_url: str, *, verify_hostname: bool = True) -> httpx.Client:
    """Return the process-wide keep-alive client for ``base_url``'s origin.

    The client carries only a default timeout; call sites pass their own
    ``timeout=`` per request so one pool serves every callback profile.
    Never close the returned client; :func:`close_shared_clients` owns that.

    The cached client is rebuilt when the mounted certificates change (so a
    cert-manager rotation is picked up without a restart) or when the pid
    changes (a pooled socket must not be shared across a fork). A request in
    flight on the old client during rotation may fail; every call site either
    retries or is best-effort, and rotation is rare, so this matches the
    trade-off the previous per-module caches already accepted.

    At most ``_MAX_SHARED_CLIENTS`` origins are retained. Resolving a new origin
    at capacity evicts and closes the least-recently-used client, so a
    long-lived process that sees its origin change cannot grow the cache without
    bound. As with rotation, a request in flight on an evicted client may fail;
    call sites retry or are best-effort.

    Fail-closed is preserved: an HTTPS URL without certificates — including one
    whose configured certificate file has since disappeared — raises
    ``ValueError`` and nothing is cached, rather than reusing stale TLS material.
    """
    from agent_engine_runner_shared.utils import get_request_timeout

    key = _shared_client_key(base_url, verify_hostname)
    pid = os.getpid()

    with _shared_clients_lock:
        entry = _shared_clients.get(key)
        if entry is not None and entry.pid != pid:
            # Inherited across fork: drop without closing, the parent still
            # owns the sockets.
            logger.debug("shared HTTP client inherited across fork; rebuilding")
            _shared_clients.pop(key, None)
            entry = None
        # Read the mounted-certificate mtimes *inside* the lock. A snapshot
        # taken before acquiring it can be stale relative to an entry another
        # thread has already rebuilt; comparing that stale snapshot against the
        # fresh entry would close a client the other thread is actively using.
        current_mtimes = _get_cert_mtimes()
        if entry is not None and current_mtimes is not None and entry.cert_mtimes != current_mtimes:
            logger.info(
                "TLS certificates rotated, invalidating cached HTTP client",
                extra={
                    "base_url": key,
                    "old_mtimes": entry.cert_mtimes,
                    "new_mtimes": current_mtimes,
                },
            )
            _shared_clients.pop(key, None)
            _close_quietly(entry.client)
            entry = None
        if entry is None:
            if len(_shared_clients) >= _MAX_SHARED_CLIENTS:
                # At capacity: evict the least-recently-used origin and close
                # its client so the cache stays bounded under origin churn.
                _, evicted = _shared_clients.popitem(last=False)
                logger.debug(
                    "evicting least-recently-used shared HTTP client",
                    extra={"base_url": key},
                )
                _close_quietly(evicted.client)
            client = create_httpx_client_with_tls(
                base_url,
                get_request_timeout(),
                verify_hostname=verify_hostname,
                limits=_SHARED_CLIENT_LIMITS,
            )
            entry = _SharedClientEntry(client=client, cert_mtimes=current_mtimes, pid=pid)
            _shared_clients[key] = entry
        else:
            # Mark most-recently-used so eviction spares active origins.
            _shared_clients.move_to_end(key)
        return entry.client


def close_shared_clients() -> None:
    """Close and forget every shared client (interpreter exit and test isolation)."""
    with _shared_clients_lock:
        entries = list(_shared_clients.values())
        _shared_clients.clear()
    for entry in entries:
        _close_quietly(entry.client)


def _close_quietly(client: httpx.Client) -> None:
    try:
        client.close()
    except Exception as exc:  # noqa: BLE001 - closing is best-effort
        logger.debug("shared HTTP client close failed: %s", exc)


atexit.register(close_shared_clients)


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
    if not _uses_https(base_url):
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

        logger.debug(
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
