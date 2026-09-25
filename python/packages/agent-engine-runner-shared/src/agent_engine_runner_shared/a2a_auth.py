"""Minimal HS256 JWT signing for A2A registration authentication.

Uses stdlib only (no PyJWT dependency). The OE validates these tokens
using the same A2A_JWT_SECRET that both the AER and OE receive via ASM.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_registration_jwt(
    secret: str,
    workspace_id: str,
    issuer: str,
    org_id: str = "",
    project_id: str = "",
    ttl: int = 30,
) -> str:
    """Create a short-lived HS256 JWT for authenticating A2A registration.

    Args:
        secret: The shared A2A_JWT_SECRET (project-level HMAC key).
        workspace_id: The registering agent's workspace ID (becomes source_agent claim).
        issuer: The token issuer (must match the OE's issuer, e.g. "oe.mdb_store").
        org_id: The agent's organization ID.
        project_id: The agent's project ID.
        ttl: Token lifetime in seconds (default 30).

    Returns:
        Signed JWT string suitable for an Authorization: Bearer header.
    """
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    claims: dict = {
        "source_agent": workspace_id,
        "iss": issuer,
        "iat": now,
        "exp": now + ttl,
        # Stamp the registration purpose so /a2a/invoke and /a2a/discover refuse
        # this token on those routes (they require "invoke"), closing the
        # cross-route replay. /a2a/register accepts "register" and, for backward
        # compatibility, an empty purpose — so already-deployed AERs that predate
        # this claim keep registering without a rebuild (SECBUG-5264).
        "purpose": "register",
    }
    if org_id:
        claims["org_id"] = org_id
    if project_id:
        claims["project_id"] = project_id
    payload = _b64url_encode(json.dumps(claims).encode())
    signing_input = f"{header}.{payload}"
    signature = _b64url_encode(
        hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    return f"{signing_input}.{signature}"
