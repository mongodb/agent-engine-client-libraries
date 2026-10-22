"""Return a machine-readable code when an agent cannot resolve its Atlas host.

A vm-mode agent that cannot resolve the platform-injected Atlas host fails at
graph build with a pymongo ``ConfigurationError``. Without this, FastAPI returns
a bare 500 and the OE can only report a generic dispatch failure. The handler
answers 503 with an ``error_code`` the OE classifies on, and never returns the
exception text (it can embed the customer's hostname).
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)

# Must match the OE's atlasDNSErrorCode (orchestration-engine/handlers.go).
ATLAS_DNS_ERROR_CODE = "ATLAS_DNS_UNAVAILABLE"
ATLAS_DNS_DETAIL = "The agent could not resolve its MongoDB Atlas host."

# pymongo wraps the underlying dns.resolver.NoNameservers in a plain
# ConfigurationError, so the check is on the message.
_DNS_SIGNATURES = ("_mongodb._tcp", "NoNameservers", "failed to answer the query")

# A resolution timeout raises dns.resolver.LifetimeTimeout, whose message names
# the nameservers but not the query host.
_DNS_TIMEOUT_SIGNATURES = (
    "The resolution lifetime expired",
    "The DNS operation timed out",
)


def _platform_atlas_host() -> str:
    """Host of the platform-injected ``MONGODB_URI``, or "" when unset."""
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        return ""
    after = uri.split("://", 1)[-1]
    if "@" in after:
        after = after.rsplit("@", 1)[-1]
    return after.split("/", 1)[0].split(",", 1)[0].split(":", 1)[0].strip()


def is_atlas_dns_failure(exc: BaseException) -> bool:
    """True only for an SRV DNS failure on the platform-injected Atlas host.

    The host check keeps a customer graph that contacts its own SRV host from
    being mislabeled as a platform egress-activation failure.
    """
    host = _platform_atlas_host()
    if not host:
        return False
    message = str(exc)
    if any(signature in message for signature in _DNS_SIGNATURES):
        return host in message
    # A resolution timeout names no query host, so it cannot be matched against
    # MONGODB_URI; treat it as the platform Atlas host when the URI is set.
    return any(signature in message for signature in _DNS_TIMEOUT_SIGNATURES)


def register_atlas_dns_error_handler(app: FastAPI) -> None:
    try:
        from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
    except Exception:  # pragma: no cover - pymongo is a declared dependency
        return

    def _handler(request: Request, exc: Exception) -> Response:
        if is_atlas_dns_failure(exc):
            logger.error(
                "Atlas DNS resolution failed during agent setup: %s",
                exc,
                exc_info=exc,
            )
            return JSONResponse(
                status_code=503,
                content={"detail": ATLAS_DNS_DETAIL, "error_code": ATLAS_DNS_ERROR_CODE},
            )
        # Non-Atlas pymongo failures keep Starlette's default 500 shape. Log the
        # traceback here: handling the exception suppresses Uvicorn's own
        # logging, and the caller only sees an opaque 500.
        logger.error("Unclassified pymongo failure during agent setup", exc_info=exc)
        return PlainTextResponse("Internal Server Error", status_code=500)

    app.add_exception_handler(ConfigurationError, _handler)
    app.add_exception_handler(ServerSelectionTimeoutError, _handler)
