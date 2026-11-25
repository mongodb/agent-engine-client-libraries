"""Unit tests for scrub_credentials (Mongo URI userinfo masking)."""

import pytest

from agent_engine_runner_shared.tracing import scrub_credentials

SCRUB_CASES = [
    (
        "connect to mongodb://user:password@host:27017/db failed: TLS",
        "connect to mongodb://***@host:27017/db failed: TLS",
    ),
    (
        "parse error: mongodb://user@ssword@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pass word@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pa/ss@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        "parse error: mongodb://user:pass\nword@host/db",
        "parse error: mongodb://***@host/db",
    ),
    (
        '{"error":"mongodb://u:p@h1/db","email":"a@b.com"}',
        '{"error":"mongodb://***@h1/db","email":"a@b.com"}',
    ),
    (
        "parse error: mongodb://user:p@ ss word@host/db tail",
        "parse error: mongodb://***@host/db tail",
    ),
    (
        "mongodb+srv://user@ss@host/db failed",
        "mongodb+srv://***@host/db failed",
    ),
    (
        "mongodb://host/db unreachable",
        "mongodb://host/db unreachable",
    ),
    (
        "a mongodb://u:p@h1/db b mongodb://u2:p2@h2/db2 c",
        "a mongodb://***@h1/db b mongodb://***@h2/db2 c",
    ),
    (
        "mongodb://u:p@h1:27017,h2:27017/db?replicaSet=rs",
        "mongodb://***@h1:27017,h2:27017/db?replicaSet=rs",
    ),
    (
        'Invalid connection string "mongodb://user:pass@host/db" foo',
        'Invalid connection string "mongodb://***@host/db" foo',
    ),
    (
        "MongoDB://User:P@ss@Host/db",
        "MongoDB://***@Host/db",
    ),
    (
        "something else entirely",
        "something else entirely",
    ),
]


@pytest.mark.parametrize("raw,expected", SCRUB_CASES)
def test_scrub_credentials_masks_uri_userinfo(raw, expected):
    assert scrub_credentials(raw) == expected


@pytest.mark.parametrize(
    "password", ["p@ss", "pass word", "pa/ss", "p@ ss word", "p:a@s/s w", "pass\nword"]
)
def test_scrub_credentials_no_fragment_survives(password):
    out = scrub_credentials(f"err mongodb://user:{password}@host/db")
    assert out == "err mongodb://***@host/db"
    assert password not in out
    assert "user" not in out
