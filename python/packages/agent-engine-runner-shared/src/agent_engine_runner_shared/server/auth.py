"""
Bearer-token authentication for Runner SDK servers.

Mirrors TypeScript's ``agent-engine-runner-shared/src/server/auth.ts``. Both runtimes
are parallel implementations of the same server architecture, so this file
and its TypeScript twin must be changed together.

Runner servers registered every route — ``/execute``, ``/invoke_llm``, the
AER session-message reads — with no authentication at all, relying on network
reachability. The only legitimate caller is the Orchestration Engine, so this
is service-to-service auth using the same bearer-token shape as the
operator's gRPC control plane.

Enforcement is conditional on ``RUNNER_AUTH_TOKEN`` being configured. The
runner image also runs under ``agentengine dev up`` on a developer's laptop,
where no deploy-time secret exists and hard-failing would break local
development outright; an unconfigured server logs a warning at startup so the
weaker posture is visible rather than silent.

Deployments that provision the token can additionally set
``RUNNER_AUTH_REQUIRED=true`` to make a missing token fatal at startup: the
server raises :class:`MissingRunnerAuthTokenError` instead of serving
unauthenticated, so a provisioning gap crash-loops loudly rather than
silently weakening auth. The signal is opt-in so shipping this code changes
no running behavior until token provisioning is in place.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

__all__ = [
    "UNAUTHENTICATED_PATHS",
    "MissingRunnerAuthTokenError",
    "is_unauthenticated_path",
    "tokens_equal",
    "bearer_token",
    "register_auth_middleware",
]


class MissingRunnerAuthTokenError(RuntimeError):
    """Raised at startup when ``RUNNER_AUTH_REQUIRED=true`` is set but no
    auth token is configured."""

    def __init__(self, mode_name: str) -> None:
        super().__init__(
            f"{mode_name} server refusing to start without request "
            "authentication: RUNNER_AUTH_REQUIRED is set but "
            "RUNNER_AUTH_TOKEN is not, so any caller that could reach this "
            "port would be able to execute tools and read session data. "
            "Provision RUNNER_AUTH_TOKEN, or unset RUNNER_AUTH_REQUIRED to "
            "accept running unauthenticated."
        )


# Routes reachable without a token.
#
# Liveness and readiness probes originate from the kubelet and cannot present
# one, and these responses carry no tenant data. Everything else — including
# every credential-bearing and data-returning route — is gated.
#
# ``/metrics`` is deliberately NOT here: it returns ``Metrics.get_all()``,
# whose counter names carry tenant-derived labels (``tool_name``,
# ``model_name``), so an unauthenticated reader could enumerate which tools
# and models an agent uses. It is not a Prometheus scrape target — nothing in
# the platform reads it, so gating it costs nothing.
UNAUTHENTICATED_PATHS = frozenset({"/", "/health", "/healthz", "/ready", "/readyz"})


def is_unauthenticated_path(path: str) -> bool:
    return (path or "").split("?")[0] in UNAUTHENTICATED_PATHS


def tokens_equal(presented: str, expected: str) -> bool:
    """Constant-time comparison, so a caller cannot recover the configured
    token byte by byte from response timing.

    Both values are hashed first so the comparison is always over two 32-byte
    digests; comparing the raw strings would leak the configured token's
    length through the early length check.
    """
    a = hashlib.sha256(presented.encode("utf-8")).digest()
    b = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an Authorization header value."""
    if not header_value:
        return None
    prefix = "bearer "
    if len(header_value) <= len(prefix):
        return None
    if header_value[: len(prefix)].lower() != prefix:
        return None
    return header_value[len(prefix) :]


def register_auth_middleware(
    app: FastAPI,
    mode_name: str,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Register the authentication middleware on ``app`` if a token is
    configured. Returns whether enforcement is active.

    Raises :class:`MissingRunnerAuthTokenError` when ``RUNNER_AUTH_REQUIRED``
    is set to ``true`` (or ``1``) but no token is configured, so a deployment
    that mandates auth fails loudly instead of serving unauthenticated.
    """
    source = os.environ if env is None else env
    expected = (source.get("RUNNER_AUTH_TOKEN") or "").strip()

    if not expected:
        required = (source.get("RUNNER_AUTH_REQUIRED") or "").strip().lower()
        if required in ("1", "true"):
            raise MissingRunnerAuthTokenError(mode_name)
        logger.warning(
            "%s server is running without request authentication: "
            "RUNNER_AUTH_TOKEN is not set, so any caller that can reach this "
            "port can execute tools and read session data.",
            mode_name,
        )
        return False

    @app.middleware("http")
    async def _require_bearer_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Match the router's dispatch value, not request.url.path. The router
        # dispatches on scope["path"]; .url.path is Host-header-derived and, on
        # an unpatched starlette, a crafted Host can make it disagree with the
        # routed path. Read the same value by construction.
        if is_unauthenticated_path(request.scope["path"]):
            return await call_next(request)
        presented = bearer_token(request.headers.get("authorization"))
        if presented is None or not tokens_equal(presented, expected):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    return True
