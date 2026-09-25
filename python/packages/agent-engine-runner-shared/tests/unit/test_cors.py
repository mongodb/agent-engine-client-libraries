"""
Unit tests for resolve_cors_policy.

Mirrors TypeScript's tests/unit/cors.test.ts — the two runner runtimes are
parallel implementations, so the twins must stay behaviourally identical.

The behaviour under test is a security default: runner servers register
credential-bearing routes with no authentication, so the previous ``*``
default meant any page a developer visited could drive tool execution once
``agentengine dev`` published the ports.
"""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.server.cors import resolve_cors_policy


def test_defaults_to_deny_all_when_unset() -> None:
    policy = resolve_cors_policy(None)
    assert policy.allow_origins == []
    assert policy.allow_credentials is False


@pytest.mark.parametrize("raw", ["", "   ", ",,"])
def test_denies_all_for_empty_values(raw: str) -> None:
    policy = resolve_cors_policy(raw)
    assert policy.allow_origins == []
    assert policy.allow_credentials is False


def test_allows_explicit_origin_with_credentials() -> None:
    policy = resolve_cors_policy("https://a.com")
    assert policy.allow_origins == ["https://a.com"]
    assert policy.allow_credentials is True


def test_trims_whitespace_so_spaced_entry_still_matches() -> None:
    # Regression: the previous implementation compared the untrimmed
    # " https://b.com", so this entry silently never matched and an operator
    # could believe they had restricted origins when they had not.
    policy = resolve_cors_policy("https://a.com, https://b.com")
    assert policy.allow_origins == ["https://a.com", "https://b.com"]
    assert policy.allow_credentials is True


def test_never_pairs_wildcard_with_credentials() -> None:
    policy = resolve_cors_policy("*")
    assert policy.allow_origins == ["*"]
    assert policy.allow_credentials is False


def test_wildcard_anywhere_wins_and_drops_credentials() -> None:
    policy = resolve_cors_policy("https://a.com,*")
    assert policy.allow_origins == ["*"]
    assert policy.allow_credentials is False


def test_drops_malformed_entries() -> None:
    policy = resolve_cors_policy("https://a.com,not-a-url")
    assert policy.allow_origins == ["https://a.com"]
    assert policy.allow_credentials is True


def test_rejects_origin_carrying_a_path() -> None:
    policy = resolve_cors_policy("https://a.com/app")
    assert policy.allow_origins == []
    assert policy.allow_credentials is False


def test_falls_back_to_deny_all_when_every_entry_malformed() -> None:
    policy = resolve_cors_policy("nonsense,also-nonsense")
    assert policy.allow_origins == []
    assert policy.allow_credentials is False


def test_preserves_explicit_port() -> None:
    policy = resolve_cors_policy("http://localhost:3005")
    assert policy.allow_origins == ["http://localhost:3005"]
    assert policy.allow_credentials is True


def test_accepts_literal_null_origin() -> None:
    policy = resolve_cors_policy("null")
    assert policy.allow_origins == ["null"]
    assert policy.allow_credentials is True


def test_normalizes_default_port_away() -> None:
    policy = resolve_cors_policy("https://a.com:443,http://b.com:80")
    assert policy.allow_origins == ["https://a.com", "http://b.com"]
    assert policy.allow_credentials is True


def test_lowercases_scheme_and_host() -> None:
    policy = resolve_cors_policy("HTTPS://A.COM")
    assert policy.allow_origins == ["https://a.com"]
    assert policy.allow_credentials is True


def test_collapses_entries_normalizing_to_same_origin() -> None:
    policy = resolve_cors_policy("https://a.com,https://A.com:443")
    assert policy.allow_origins == ["https://a.com"]
    assert policy.allow_credentials is True


def test_accepts_bracketed_ipv6_host_with_port() -> None:
    policy = resolve_cors_policy("http://[::1]:3005")
    assert policy.allow_origins == ["http://[::1]:3005"]
    assert policy.allow_credentials is True


@pytest.mark.parametrize(
    "raw",
    [
        "https://*.a.com",  # a wildcard host never matches a real Origin header
        "https://u:p@a.com",  # userinfo
        "https://a.com?q=1",  # query
        "ftp://a.com",  # non-http(s) scheme
        "file://",
    ],
)
def test_rejects_non_origin_shaped_values(raw: str) -> None:
    policy = resolve_cors_policy(raw)
    assert policy.allow_origins == []
    assert policy.allow_credentials is False
