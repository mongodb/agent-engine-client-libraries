"""Tests for A2A registration JWT signing."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from agent_engine_runner_shared.a2a_auth import sign_registration_jwt


def _decode_jwt_payload(token: str) -> dict:
    parts = token.split(".")
    payload = parts[1]
    payload += "=" * (4 - len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _decode_jwt_header(token: str) -> dict:
    parts = token.split(".")
    header = parts[0]
    header += "=" * (4 - len(header) % 4)
    return json.loads(base64.urlsafe_b64decode(header))


def test_sign_registration_jwt_produces_three_part_token():
    token = sign_registration_jwt("secret", "ws-123", "oe.test")
    parts = token.split(".")
    assert len(parts) == 3


def test_sign_registration_jwt_header_is_hs256():
    token = sign_registration_jwt("secret", "ws-123", "oe.test")
    header = _decode_jwt_header(token)
    assert header["alg"] == "HS256"
    assert header["typ"] == "JWT"


def test_sign_registration_jwt_claims():
    before = int(time.time())
    token = sign_registration_jwt("secret", "ws-abc", "oe.test", ttl=60)
    after = int(time.time())

    payload = _decode_jwt_payload(token)
    assert payload["source_agent"] == "ws-abc"
    assert payload["iss"] == "oe.test"
    assert before <= payload["iat"] <= after
    assert payload["exp"] == payload["iat"] + 60


def test_sign_registration_jwt_carries_register_purpose():
    # SECBUG-5264: the registration token must carry purpose="register" so the
    # OE rejects it on /a2a/invoke and /a2a/discover, and so a purpose-less
    # (forged) token is distinguishable from a legitimate one.
    token = sign_registration_jwt("secret", "ws-abc", "oe.test")
    payload = _decode_jwt_payload(token)
    assert payload["purpose"] == "register"


def test_sign_registration_jwt_default_ttl():
    token = sign_registration_jwt("secret", "ws-123", "oe.test")
    payload = _decode_jwt_payload(token)
    assert payload["exp"] - payload["iat"] == 30


def test_sign_registration_jwt_signature_verifies():
    secret = "my-secret-key"
    token = sign_registration_jwt(secret, "ws-123", "oe.test")
    parts = token.split(".")
    signing_input = f"{parts[0]}.{parts[1]}"
    expected_sig = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    assert parts[2] == expected_sig


def test_sign_registration_jwt_different_secrets_produce_different_signatures():
    token_a = sign_registration_jwt("secret-a", "ws-123", "oe.test")
    token_b = sign_registration_jwt("secret-b", "ws-123", "oe.test")
    sig_a = token_a.split(".")[2]
    sig_b = token_b.split(".")[2]
    assert sig_a != sig_b


def test_sign_registration_jwt_different_workspaces_produce_different_payloads():
    token_a = sign_registration_jwt("secret", "ws-aaa", "oe.test")
    token_b = sign_registration_jwt("secret", "ws-bbb", "oe.test")
    payload_a = _decode_jwt_payload(token_a)
    payload_b = _decode_jwt_payload(token_b)
    assert payload_a["source_agent"] != payload_b["source_agent"]


def test_sign_registration_jwt_includes_org_and_project():
    token = sign_registration_jwt(
        "secret", "ws-123", "oe.test", org_id="org-abc", project_id="proj-xyz"
    )
    payload = _decode_jwt_payload(token)
    assert payload["org_id"] == "org-abc"
    assert payload["project_id"] == "proj-xyz"


def test_sign_registration_jwt_omits_empty_org_and_project():
    token = sign_registration_jwt("secret", "ws-123", "oe.test")
    payload = _decode_jwt_payload(token)
    assert "org_id" not in payload
    assert "project_id" not in payload


def test_sign_registration_jwt_org_only():
    token = sign_registration_jwt("secret", "ws-123", "oe.test", org_id="org-abc")
    payload = _decode_jwt_payload(token)
    assert payload["org_id"] == "org-abc"
    assert "project_id" not in payload
