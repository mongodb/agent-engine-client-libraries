"""Tests for the OE-to-Tool-Pod guardrails runtime protocol."""

from __future__ import annotations

import logging
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import LogCaptureFixture

from agent_engine_runner_shared.models import (
    GuardrailCheckDecision,
    GuardrailCheckResponse,
)
from agent_engine_runner_shared.server.tool import ToolServer


def _guardrails_request(
    *,
    text: str = "We should compare Acme and MongoDB.",
    policy_id: str = "000000000000000000000003",
    action: str = "modify",
    on_fail: str | None = "fix",
    regex_patterns: object | None = None,
    replacement_text: object | None = None,
    stage_filter: list[str] | None = None,
    status: str = "active",
    policies: list[dict] | None = None,
) -> dict:
    policy = {
        "id": policy_id,
        "type": "output_validation",
        "status": status,
        "action": action,
        "stage_filter": ["llm_output"] if stage_filter is None else stage_filter,
        "config": {
            "regex_patterns": ["Acme"] if regex_patterns is None else regex_patterns,
        },
    }
    if on_fail is not None:
        policy["config"]["on_fail"] = on_fail
    if replacement_text is not None:
        policy["config"]["replacement_text"] = replacement_text
    return {
        "execution_id": "exec-1",
        "stage": "llm_output",
        "input": {"text": text, "metadata": {"source": "invoke_llm"}},
        "context": {
            "org_id": "000000000000000000000001",
            "project_id": "000000000000000000000002",
            "workspace_id": "workspace-1",
            "session_id": "session-1",
            "user_id": "user-1",
        },
        "policies": [policy] if policies is None else policies,
    }


def _policy(
    policy_id: str,
    *,
    action: str = "modify",
    regex_patterns: object | None = None,
    on_fail: str | None = None,
    replacement_text: object | None = None,
    stage_filter: list[str] | None = None,
    status: str = "active",
    policy_type: str = "output_validation",
) -> dict:
    config: dict[str, object] = {
        "regex_patterns": ["Acme"] if regex_patterns is None else regex_patterns
    }
    if on_fail is not None:
        config["on_fail"] = on_fail
    if replacement_text is not None:
        config["replacement_text"] = replacement_text
    return {
        "id": policy_id,
        "type": policy_type,
        "status": status,
        "action": action,
        "stage_filter": ["llm_output"] if stage_filter is None else stage_filter,
        "config": config,
    }


def _guardrails_client() -> TestClient:
    runtime = Mock()
    runtime._tools = {}
    runtime._tool_definitions = {}
    app = FastAPI()
    ToolServer(runtime).register_routes(app)
    return TestClient(app)


def test_tool_pod_guardrails_check_allows_clean_text() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="MongoDB is the only company mentioned here."),
    )

    assert response.status_code == 200
    assert GuardrailCheckResponse.model_validate(response.json()) == GuardrailCheckResponse(
        decision=GuardrailCheckDecision.ALLOW,
        allowed=True,
        transformed_text="MongoDB is the only company mentioned here.",
    )


def test_tool_pod_guardrails_check_modifies_regex_matches_when_fixing() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="geico and acme should both be hidden."),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "geico and [BLOCKED] should both be hidden."
    assert check.triggered_policy_ids == ["000000000000000000000003"]
    assert check.evidence[0].policy_id == "000000000000000000000003"
    assert check.evidence[0].metadata["matched_patterns"] == ["Acme"]
    assert check.evidence[0].message == "Regex match modified content: Acme"


def test_tool_pod_guardrails_check_blocks_when_policy_action_blocks() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="Acme appears here.", action="block", on_fail="fix"),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.transformed_text is None
    assert check.triggered_policy_ids == ["000000000000000000000003"]


def test_tool_pod_guardrails_check_fails_closed_on_unresolved_action() -> None:
    # A triggered policy whose action is unrecognized (operator typo) and has no
    # usable on_fail must fail closed to BLOCK, not fall through to ALLOW.
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="Acme appears here.", action="blcok", on_fail=None),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.transformed_text is None
    assert check.triggered_policy_ids == ["000000000000000000000003"]


def test_tool_pod_guardrails_check_warns_without_blocking() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="Acme appears here.", action="noop", on_fail=None),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.LOG_ONLY
    assert check.allowed is True
    assert check.transformed_text == "Acme appears here."
    assert check.triggered_policy_ids == ["000000000000000000000003"]
    assert check.evidence[0].message == "Regex match logged: Acme"


def test_tool_pod_guardrails_check_ignores_malformed_regex_patterns_config() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="Acme appears here.", regex_patterns="Acme"),
    )

    assert response.status_code == 200
    assert GuardrailCheckResponse.model_validate(response.json()) == GuardrailCheckResponse(
        decision=GuardrailCheckDecision.ALLOW,
        allowed=True,
        transformed_text="Acme appears here.",
    )


def test_tool_pod_guardrails_check_fails_closed_when_regex_pattern_is_invalid(
    caplog: LogCaptureFixture,
) -> None:
    client = _guardrails_client()
    caplog.set_level(
        logging.WARNING, logger="agent_engine_runner_shared.guardrails_evaluator.regex"
    )

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme appears here.",
            regex_patterns=["Acme("],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    # An un-compilable pattern must not look like a clean allow: the policy
    # fails closed and the failure is surfaced in the evidence so an operator
    # can tell "evaluated and allowed" from "errored and skipped".
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED]"
    assert check.triggered_policy_ids == ["000000000000000000000003"]
    assert check.evidence[0].metadata["invalid_patterns"] == ["Acme("]
    assert "Invalid guardrail regex pattern" in caplog.text
    assert "000000000000000000000003" in caplog.text
    assert "Acme(" not in caplog.text


def test_tool_pod_guardrails_check_fails_closed_when_one_of_many_patterns_invalid() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme appears here.",
            regex_patterns=["Acme", "Bad("],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    # One invalid pattern fails the whole policy closed rather than silently
    # enforcing only the valid subset.
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED]"
    assert check.evidence[0].metadata["invalid_patterns"] == ["Bad("]
    assert check.evidence[0].metadata["matched_patterns"] == ["Acme"]


def test_tool_pod_guardrails_check_blocks_when_invalid_pattern_and_action_blocks() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="nothing matches here.",
            regex_patterns=["Bad("],
            action="block",
            on_fail=None,
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.transformed_text is None
    assert check.evidence[0].metadata["invalid_patterns"] == ["Bad("]


def test_tool_pod_guardrails_check_ignores_dropped_blocked_entities_config() -> None:
    client = _guardrails_client()
    request = _guardrails_request(text="Acme appears here.", regex_patterns=[])
    request["policies"][0]["config"]["blocked_entities"] = ["Acme"]

    response = client.post("/guardrails/check", json=request)

    assert response.status_code == 200
    assert GuardrailCheckResponse.model_validate(response.json()) == GuardrailCheckResponse(
        decision=GuardrailCheckDecision.ALLOW,
        allowed=True,
        transformed_text="Acme appears here.",
    )


def test_tool_pod_guardrails_check_blocks_unsupported_policy_type() -> None:
    client = _guardrails_client()
    request = _guardrails_request(text="Acme appears here.")
    request["policies"][0]["type"] = "future_policy"

    response = client.post("/guardrails/check", json=request)

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.triggered_policy_ids == ["000000000000000000000003"]
    assert check.reason == "Unsupported guardrail policy type: future_policy"


def test_tool_pod_guardrails_check_redacts_regex_matches() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="SSN 123-45-6789 should not leave the pod.",
            regex_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "SSN [BLOCKED] should not leave the pod."
    assert check.evidence[0].metadata["matched_patterns"] == [r"\b\d{3}-\d{2}-\d{4}\b"]
    matches = check.evidence[0].metadata["matches"]
    assert isinstance(matches, list)
    match = matches[0]
    assert isinstance(match, dict)
    assert "text" not in match
    assert match["length"] == 11


def test_tool_pod_guardrails_check_redacts_overlapping_matches_once() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text="abc should be hidden.", regex_patterns=["bc", "abc"]),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.transformed_text == "[BLOCKED] should be hidden."


def test_tool_pod_guardrails_check_redacts_entire_text_when_regex_times_out() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(text=f"{'a' * 1000}!", regex_patterns=[r"(a+)+$"]),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED]"
    assert check.evidence[0].metadata["timed_out_patterns"] == [r"(a+)+$"]


def test_tool_pod_guardrails_check_supports_literal_regex_patterns() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme SSN 123-45-6789 should not leave the pod.",
            regex_patterns=["Acme", r"\b\d{3}-\d{2}-\d{4}\b"],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED] SSN [BLOCKED] should not leave the pod."
    assert check.evidence[0].metadata["matched_patterns"] == [
        "Acme",
        r"\b\d{3}-\d{2}-\d{4}\b",
    ]


def test_tool_pod_guardrails_check_uses_configured_replacement_text() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme should be hidden.",
            regex_patterns=["Acme"],
            replacement_text=r"[\REDACTED]",
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == r"[\REDACTED] should be hidden."


def test_tool_pod_guardrails_check_uses_running_text_after_modify() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme should be hidden.",
            policies=[
                _policy("modify-policy", action="modify", regex_patterns=["Acme"]),
                _policy("block-policy", action="block", regex_patterns=["Acme"]),
            ],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED] should be hidden."
    assert check.triggered_policy_ids == ["modify-policy"]


def test_tool_pod_guardrails_check_escalates_multi_policy_decisions() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme and Secret should be hidden.",
            policies=[
                _policy("modify-policy", action="modify", regex_patterns=["Acme"]),
                _policy("block-policy", action="block", regex_patterns=["Secret"]),
            ],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.transformed_text is None
    assert check.triggered_policy_ids == ["modify-policy", "block-policy"]
    assert [item.policy_id for item in check.evidence] == ["modify-policy", "block-policy"]


def test_tool_pod_guardrails_check_chains_modify_policies() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme and MongoDB should be hidden.",
            policies=[
                _policy(
                    "acme-policy",
                    action="modify",
                    regex_patterns=["Acme"],
                    replacement_text="[A]",
                ),
                _policy(
                    "mongodb-policy",
                    action="modify",
                    regex_patterns=["MongoDB"],
                    replacement_text="[M]",
                ),
            ],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[A] and [M] should be hidden."
    assert check.triggered_policy_ids == ["acme-policy", "mongodb-policy"]


def test_tool_pod_guardrails_check_skips_non_matching_stage_filter() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme should not trigger.",
            policies=[
                _policy(
                    "input-policy",
                    action="block",
                    regex_patterns=["Acme"],
                    stage_filter=["llm_input"],
                )
            ],
        ),
    )

    assert response.status_code == 200
    assert GuardrailCheckResponse.model_validate(response.json()) == GuardrailCheckResponse(
        decision=GuardrailCheckDecision.ALLOW,
        allowed=True,
        transformed_text="Acme should not trigger.",
    )


def test_tool_pod_guardrails_check_skips_inactive_policies() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme should not trigger.",
            policies=[
                _policy(
                    "inactive-policy",
                    action="block",
                    regex_patterns=["Acme"],
                    status="disabled",
                )
            ],
        ),
    )

    assert response.status_code == 200
    assert GuardrailCheckResponse.model_validate(response.json()) == GuardrailCheckResponse(
        decision=GuardrailCheckDecision.ALLOW,
        allowed=True,
        transformed_text="Acme should not trigger.",
    )


def test_tool_pod_guardrails_check_keeps_prior_evidence_when_policy_type_unsupported() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Acme should be hidden.",
            policies=[
                _policy("modify-policy", action="modify", regex_patterns=["Acme"]),
                _policy(
                    "future-policy",
                    action="block",
                    regex_patterns=["Acme"],
                    policy_type="future_policy",
                ),
            ],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.BLOCK
    assert check.allowed is False
    assert check.triggered_policy_ids == ["modify-policy", "future-policy"]
    assert [item.policy_id for item in check.evidence] == ["modify-policy", "future-policy"]
    assert check.metadata["triggered_count"] == 2


def test_tool_pod_guardrails_check_fails_closed_when_too_many_patterns_are_supplied(
    caplog: LogCaptureFixture,
) -> None:
    client = _guardrails_client()
    caplog.set_level(
        logging.WARNING, logger="agent_engine_runner_shared.guardrails_evaluator.regex"
    )

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="Nothing matches, but the policy exceeds the pattern limit.",
            regex_patterns=[f"pattern-{index}" for index in range(101)],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED]"
    assert check.evidence[0].metadata["skipped_patterns"] == ["pattern-100"]
    assert "exceeding max 100; skipping 1 patterns" in caplog.text
    assert "pattern-100" not in caplog.text


def test_tool_pod_guardrails_check_fails_closed_when_text_is_too_long() -> None:
    client = _guardrails_client()

    response = client.post(
        "/guardrails/check",
        json=_guardrails_request(
            text="a" * 20_001,
            regex_patterns=["a"],
        ),
    )

    assert response.status_code == 200
    check = GuardrailCheckResponse.model_validate(response.json())
    assert check.decision == GuardrailCheckDecision.MODIFY
    assert check.allowed is True
    assert check.transformed_text == "[BLOCKED]"
    assert check.evidence[0].metadata["text_too_long"] is True
