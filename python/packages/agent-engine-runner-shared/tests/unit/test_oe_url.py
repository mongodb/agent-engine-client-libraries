"""
Unit tests for resolve_oe_url.

Mirrors TypeScript's tests/unit/oe_url.test.ts — the two runner runtimes are
parallel implementations, so the twins must stay behaviourally identical.

The behaviour under test is a trust boundary inside the agent execution loop:
whatever this returns becomes the base for every outbound approval, stream and
result call, and the runner treats those responses as authoritative results
and policy decisions.
"""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.server.oe_url import resolve_oe_url


def test_prefers_configured_oe_url_over_request_value() -> None:
    assert (
        resolve_oe_url("http://attacker.example", {"OE_URL": "http://oe:8000"}) == "http://oe:8000"
    )


def test_discards_request_value_even_when_plausible() -> None:
    assert (
        resolve_oe_url("http://oe.other-tenant:8000", {"OE_URL": "http://oe:8000"})
        == "http://oe:8000"
    )


def test_returns_configured_value_when_they_agree() -> None:
    assert resolve_oe_url("http://oe:8000", {"OE_URL": "http://oe:8000"}) == "http://oe:8000"


def test_trims_whitespace_on_configured_value() -> None:
    assert (
        resolve_oe_url("http://attacker.example", {"OE_URL": "  http://oe:8000 "})
        == "http://oe:8000"
    )


@pytest.mark.parametrize("env", [{}, {"OE_URL": ""}, {"OE_URL": "   "}])
def test_falls_back_to_request_value_when_unset(env: dict) -> None:
    # Local `agentengine dev` and unit tests have no deploy-time environment.
    assert resolve_oe_url("http://oe:8000", env) == "http://oe:8000"


def test_ignores_empty_request_value_when_configured() -> None:
    assert resolve_oe_url("", {"OE_URL": "http://oe:8000"}) == "http://oe:8000"


# ---------------------------------------------------------------------------
# Owner-callback fallback regression: the owner-URL fallback work must leave resolve_oe_url's
# behaviour byte-identical. Owner URLs are validated by the separate
# resolve_owner_url helper; the base-URL hardening here is not loosened.
# ---------------------------------------------------------------------------


def test_configured_oe_url_still_wins_over_replica_shaped_request_value() -> None:
    # Even a request value shaped exactly like a valid headless replica of the
    # configured service must not replace the deploy-time OE_URL.
    service = "https://oe.ns.svc.cluster.local:8443"
    replica = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443"
    assert resolve_oe_url(replica, {"OE_URL": service}) == service


def test_unset_env_still_returns_request_value_verbatim() -> None:
    # The local-dev fallback is unchanged: no OE_URL means the request value is
    # used as-is, with no owner-style validation or normalization applied.
    assert resolve_oe_url("http://oe.ns.svc:8000/", {}) == "http://oe.ns.svc:8000/"
