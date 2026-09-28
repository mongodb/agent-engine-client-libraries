"""Core guardrail policy evaluation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from agent_engine_runner_shared.models import (
    GuardrailCheckDecision,
    GuardrailCheckEvidence,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailRuntimePolicy,
)

logger = logging.getLogger(__name__)

_DECISION_PRIORITY: dict[GuardrailCheckDecision, int] = {
    GuardrailCheckDecision.ALLOW: 0,
    GuardrailCheckDecision.LOG_ONLY: 1,
    GuardrailCheckDecision.MODIFY: 2,
    GuardrailCheckDecision.REQUIRE_REVIEW: 3,
    GuardrailCheckDecision.BLOCK: 4,
}


@dataclass(frozen=True)
class GuardrailPolicyEngineResult:
    """Engine-owned result for a policy evaluation."""

    evidence: list[GuardrailCheckEvidence] = field(default_factory=list)
    transformed_text: str | None = None


class GuardrailPolicyEngine(Protocol):
    """Contract implemented by native and future remote guardrail engines."""

    policy_type: str

    def evaluate(
        self, policy: GuardrailRuntimePolicy, text: str
    ) -> GuardrailPolicyEngineResult | None:
        """Return engine-owned evidence and optional transformed text when triggered."""
        ...


_POLICY_ENGINES: dict[str, GuardrailPolicyEngine] = {}


def register_guardrail_policy_engine(engine: GuardrailPolicyEngine) -> None:
    """Register a policy engine so ``evaluate_guardrail_check`` can dispatch to it by ``policy_type``."""
    _POLICY_ENGINES[engine.policy_type] = engine


def evaluate_guardrail_check(request: GuardrailCheckRequest) -> GuardrailCheckResponse:
    """Evaluate guardrail policies for one runtime boundary."""

    triggered_policy_ids: list[str] = []
    evidence: list[GuardrailCheckEvidence] = []
    transformed_text = request.input.text
    decision = GuardrailCheckDecision.ALLOW

    for policy in request.policies:
        if not policy.applies_to_stage(request.stage):
            continue

        engine = _POLICY_ENGINES.get(policy.type)
        if engine is None:
            reason = f"Unsupported guardrail policy type: {policy.type}"
            unsupported_evidence = GuardrailCheckEvidence(
                policy_id=policy.id,
                message=reason,
                metadata={"policy_type": policy.type},
            )
            policy_ids = [*triggered_policy_ids, policy.id]
            return GuardrailCheckResponse(
                decision=GuardrailCheckDecision.BLOCK,
                allowed=False,
                triggered_policy_ids=policy_ids,
                evidence=[*evidence, unsupported_evidence],
                reason=reason,
                metadata={"triggered_count": len(policy_ids)},
            )

        policy_decision = policy.check_decision()
        should_transform = policy_decision == GuardrailCheckDecision.MODIFY
        result = engine.evaluate(policy, transformed_text)

        if result is None:
            continue

        # A policy that actually triggered but whose action (and config.on_fail)
        # does not resolve to a known decision would otherwise fall through to
        # ALLOW and let the matching content pass. That is fail-open on a
        # misconfiguration (e.g. an operator typo), and inconsistent with the
        # fail-closed handling of an unknown policy type above. Fail closed to
        # BLOCK and log so the misconfig is observable instead.
        if policy_decision == GuardrailCheckDecision.ALLOW:
            logger.warning(
                "Guardrail policy %s triggered but its action %r did not resolve "
                "to a known decision; failing closed to BLOCK",
                policy.id,
                policy.action,
            )
            policy_decision = GuardrailCheckDecision.BLOCK

        evidence.extend(result.evidence)
        if should_transform and result.transformed_text is not None:
            transformed_text = result.transformed_text

        triggered_policy_ids.append(policy.id)
        if _DECISION_PRIORITY[policy_decision] > _DECISION_PRIORITY[decision]:
            decision = policy_decision

    if decision == GuardrailCheckDecision.ALLOW:
        return GuardrailCheckResponse(
            decision=GuardrailCheckDecision.ALLOW,
            allowed=True,
            transformed_text=request.input.text,
        )

    policy_count = len(triggered_policy_ids)
    if decision == GuardrailCheckDecision.BLOCK:
        reason = f"{policy_count} guardrail policy triggered a block decision"
    elif decision == GuardrailCheckDecision.REQUIRE_REVIEW:
        reason = f"{policy_count} guardrail policy requires review"
    elif decision == GuardrailCheckDecision.MODIFY:
        reason = f"{policy_count} guardrail policy modified the content"
    else:
        reason = f"{policy_count} guardrail policy triggered"

    allowed = decision in {GuardrailCheckDecision.LOG_ONLY, GuardrailCheckDecision.MODIFY}
    return GuardrailCheckResponse(
        decision=decision,
        allowed=allowed,
        transformed_text=transformed_text if allowed else None,
        triggered_policy_ids=triggered_policy_ids,
        evidence=evidence,
        reason=reason,
        metadata={"triggered_count": policy_count},
    )
