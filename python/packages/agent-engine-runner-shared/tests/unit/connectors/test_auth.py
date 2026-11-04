"""The connector auth block: one credential key, no ``type``, shared refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_engine_runner_shared.connectors.definitions import Auth

FIXTURES = Path(__file__).parent / "fixtures"
CASES = yaml.safe_load((FIXTURES / "auth_cases.yaml").read_text(encoding="utf-8"))
TYPESCRIPT_CASES = (
    Path(__file__).parents[6]
    / "javascript/packages/agent-engine-runner-shared/tests/unit/connectors/fixtures/auth_cases.yaml"
)


def test_typescript_suite_asserts_the_same_cases():
    """Both runtimes read one contract, so their case lists cannot drift apart."""
    assert TYPESCRIPT_CASES.read_bytes() == (FIXTURES / "auth_cases.yaml").read_bytes()


@pytest.mark.parametrize("case", CASES["valid"], ids=lambda case: case["name"])
def test_accepts_one_credential_key(case):
    auth = Auth.model_validate(case["auth"])

    assert auth.model_dump(exclude_none=True, exclude_defaults=True) == case["auth"]


@pytest.mark.parametrize("case", CASES["refused"], ids=lambda case: case["name"])
def test_refuses_with_the_shared_message(case):
    with pytest.raises(ValueError) as caught:
        Auth.model_validate(case["auth"])

    assert case["error"] in str(caught.value)


@pytest.mark.parametrize(
    "block",
    [
        {"type": "bearer", "env": "ghp_SENTINELVALUE"},
        {"bearer": "ghp_SENTINELVALUE", "provider": "ghp_SENTINELVALUE"},
        {"bearer": "ghp_SENTINELVALUE", "scopes": ["ghp_SENTINELVALUE"]},
        {"basic": {"username": "ghp_SENTINELVALUE", "password": "ghp_SENTINELVALUE"}},
        {"api_keys": [{"env": "ghp SENTINELVALUE", "location": "header", "name": "X-Key"}]},
        {"provider": "ghp SENTINELVALUE"},
        {"bearer": "ghp SENTINELVALUE"},
        {"bearer": "SERVICE_TOKEN", "headers": {"X-Key": "ghp_SENTINELVALUE\n"}},
    ],
)
def test_refusals_never_echo_a_value(block):
    """A secret pasted where a secret name belongs must not reach logs."""
    with pytest.raises(ValueError) as caught:
        Auth.model_validate(block)

    assert "SENTINELVALUE" not in str(caught.value)


@pytest.mark.parametrize(
    "block",
    [
        {"bearer": 1},
        {"bearer": ["SERVICE_TOKEN"]},
        {"basic": "SERVICE_USERNAME"},
        {"basic": {"username": 1, "password": "SERVICE_PASSWORD"}},
        {"api_keys": {"env": "SERVICE_KEY", "location": "header", "name": "X-Key"}},
        {"api_keys": [{"env": "SERVICE_KEY", "location": "body", "name": "key"}]},
        {"provider": ["github"]},
        {"provider": "github", "scopes": "repo"},
    ],
)
def test_refuses_malformed_values(block):
    with pytest.raises(ValueError):
        Auth.model_validate(block)


def test_null_keys_are_absent():
    """YAML ``key:`` with no value means the key is not set, in both runtimes."""
    auth = Auth.model_validate({"bearer": None, "provider": "github", "scopes": None})

    assert auth.provider == "github"
    assert auth.bearer is None


@pytest.mark.parametrize(
    ("block", "headers"),
    [
        ({"bearer": "SERVICE_TOKEN"}, {"authorization"}),
        (
            {"basic": {"username": "SERVICE_USERNAME", "password": "SERVICE_PASSWORD"}},
            {"authorization"},
        ),
        (
            {"provider": "github", "headers": {"X-Api-Version": "2"}},
            {"authorization", "x-api-version"},
        ),
        (
            {
                "api_keys": [
                    {"env": "KEY", "location": "header", "name": "X-Key"},
                    {"env": "QUERY_KEY", "location": "query", "name": "key"},
                ]
            },
            {"x-key"},
        ),
    ],
)
def test_reserves_every_credential_header(block, headers):
    """Model arguments can never set a header the credential owns."""
    assert Auth.model_validate(block).header_names() == headers
