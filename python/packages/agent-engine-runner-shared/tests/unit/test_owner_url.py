"""Unit tests for resolve_owner_url.

Mirrors TypeScript's tests/unit/owner_url.test.ts — the two runner runtimes are
parallel implementations, so the twins must stay behaviourally identical.

resolve_owner_url guards a trust boundary: it decides whether a request-supplied
owner URL may become a callback target. A forged value must be discarded (return
None) so the caller falls back to the already-trusted service URL — never an
error, never an attacker-chosen host.
"""

from __future__ import annotations

import logging

import pytest

from agent_engine_runner_shared.server.owner_url import resolve_owner_url

SERVICE = "https://oe.ns.svc.cluster.local:8443"
VALID_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"


def test_accepts_valid_headless_replica() -> None:
    assert resolve_owner_url(VALID_OWNER, SERVICE) == VALID_OWNER


def test_accepts_with_scheme_default_ports() -> None:
    # https default 443 on both sides, expressed implicitly.
    assert (
        resolve_owner_url(
            "https://10-1-2-3.oe-headless.ns.svc.cluster.local",
            "https://oe.ns.svc.cluster.local",
        )
        == "https://10-1-2-3.oe-headless.ns.svc.cluster.local"
    )


def test_canonicalizes_explicit_default_port_away() -> None:
    assert (
        resolve_owner_url(
            "https://10-1-2-3.oe-headless.ns.svc.cluster.local:443",
            "https://oe.ns.svc.cluster.local:443",
        )
        == "https://10-1-2-3.oe-headless.ns.svc.cluster.local"
    )


def test_preserves_explicit_non_default_port() -> None:
    assert (
        resolve_owner_url(
            "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443",
            "https://oe.ns.svc.cluster.local:8443",
        )
        == "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"
    )


def test_strips_trailing_slash() -> None:
    assert resolve_owner_url(VALID_OWNER + "/", SERVICE) == VALID_OWNER


@pytest.mark.parametrize("owner", [None, "", "   "])
def test_absent_owner_returns_none(owner: str | None) -> None:
    assert resolve_owner_url(owner, SERVICE) is None


# --- security-negatives: every forged shape degrades to None --------------


def test_rejects_external_host() -> None:
    assert resolve_owner_url("https://attacker.example:8443", SERVICE) is None


def test_rejects_external_host_mimicking_suffix() -> None:
    # A convincing-looking but foreign host must not be accepted.
    assert (
        resolve_owner_url(
            "https://10-1-2-3.oe-headless.ns.svc.cluster.local.evil.com:8443", SERVICE
        )
        is None
    )


def test_rejects_wrong_port() -> None:
    assert (
        resolve_owner_url("https://10-1-2-3.oe-headless.ns.svc.cluster.local:9999", SERVICE) is None
    )


def test_rejects_wrong_scheme() -> None:
    assert (
        resolve_owner_url("http://10-1-2-3.oe-headless.ns.svc.cluster.local:8443", SERVICE) is None
    )


def test_rejects_host_without_headless_service_segment() -> None:
    # <label>.<service-label>.<rest> — missing the "-headless" marker.
    assert resolve_owner_url("https://10-1-2-3.oe.ns.svc.cluster.local:8443", SERVICE) is None


def test_rejects_headless_service_without_replica_label() -> None:
    # The headless service itself is not a single replica.
    assert resolve_owner_url("https://oe-headless.ns.svc.cluster.local:8443", SERVICE) is None


def test_rejects_extra_leading_labels() -> None:
    # Exactly one replica label is allowed before the headless service host.
    assert (
        resolve_owner_url("https://a.10-1-2-3.oe-headless.ns.svc.cluster.local:8443", SERVICE)
        is None
    )


def test_rejects_userinfo() -> None:
    assert (
        resolve_owner_url("https://evil@10-1-2-3.oe-headless.ns.svc.cluster.local:8443", SERVICE)
        is None
    )


def test_rejects_path() -> None:
    assert resolve_owner_url(VALID_OWNER + "/evil", SERVICE) is None


def test_rejects_query() -> None:
    assert resolve_owner_url(VALID_OWNER + "?x=1", SERVICE) is None


def test_rejects_fragment() -> None:
    assert resolve_owner_url(VALID_OWNER + "#frag", SERVICE) is None


def test_rejects_when_service_host_is_single_label() -> None:
    # Cannot derive <service-label>.<rest> from a bare host.
    assert resolve_owner_url("https://10-1-2-3.oe-headless:8443", "https://oe:8443") is None


def test_rejects_wrong_namespace() -> None:
    assert (
        resolve_owner_url("https://10-1-2-3.oe-headless.other-ns.svc.cluster.local:8443", SERVICE)
        is None
    )


def test_rejects_raw_ascii_control_characters_before_parsing() -> None:
    assert (
        resolve_owner_url("https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443\n", SERVICE)
        is None
    )


def test_warning_logs_do_not_expose_credentials_from_rejected_owner_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner = "https://user:super-secret@10-1-2-3.oe-headless.ns.svc.cluster.local:8443"

    with caplog.at_level(logging.WARNING, logger="agent_engine_runner_shared.server.owner_url"):
        assert resolve_owner_url(owner, SERVICE) is None

    assert "not a bare origin" in caplog.text
    assert "super-secret" not in caplog.text
    assert "user:" not in caplog.text
    assert owner not in caplog.text
