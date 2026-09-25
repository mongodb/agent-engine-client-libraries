/**
 * Core guardrail policy evaluation orchestration.
 *
 * Mirrors Python's `agent_engine_runner_shared/guardrails_evaluator/core.py`: a registry of
 * per-`policy_type` engines plus `evaluateGuardrailCheck`, which aggregates the
 * strictest decision across the OE-selected policies for one runtime boundary.
 */

import { getLogger } from "../logger.js";
import { Metrics } from "../metrics.js";
import {
  appliesToStage,
  checkDecision,
  GuardrailCheckDecision,
  type GuardrailCheckEvidence,
  type GuardrailCheckRequest,
  type GuardrailCheckResponse,
  type GuardrailRuntimePolicy,
} from "../models.js";

const logger = getLogger(
  "agent_engine_runner_shared.guardrails_evaluator.core",
);

const DECISION_PRIORITY: Record<GuardrailCheckDecision, number> = {
  [GuardrailCheckDecision.ALLOW]: 0,
  [GuardrailCheckDecision.LOG_ONLY]: 1,
  [GuardrailCheckDecision.MODIFY]: 2,
  [GuardrailCheckDecision.REQUIRE_REVIEW]: 3,
  [GuardrailCheckDecision.BLOCK]: 4,
};

/** Engine-owned result for a policy evaluation. */
export interface GuardrailPolicyEngineResult {
  evidence: GuardrailCheckEvidence[];
  transformedText?: string | null;
}

/** Contract implemented by native and future remote guardrail engines. */
export interface GuardrailPolicyEngine {
  readonly policyType: string;
  /** Return engine-owned evidence and optional transformed text when triggered. */
  evaluate(
    policy: GuardrailRuntimePolicy,
    text: string,
  ): GuardrailPolicyEngineResult | null;
}

const POLICY_ENGINES = new Map<string, GuardrailPolicyEngine>();

export function registerGuardrailPolicyEngine(
  engine: GuardrailPolicyEngine,
): void {
  POLICY_ENGINES.set(engine.policyType, engine);
}

/** Evaluate guardrail policies for one runtime boundary. */
export function evaluateGuardrailCheck(
  request: GuardrailCheckRequest,
): GuardrailCheckResponse {
  const triggeredPolicyIds: string[] = [];
  const evidence: GuardrailCheckEvidence[] = [];
  let transformedText = request.input.text;
  let decision: GuardrailCheckDecision = GuardrailCheckDecision.ALLOW;

  for (const policy of request.policies) {
    if (!appliesToStage(policy, request.stage)) {
      continue;
    }

    const engine = POLICY_ENGINES.get(policy.type);
    if (engine === undefined) {
      // An unsupported policy type cannot be enforced, so fail closed to BLOCK
      // rather than silently allowing content through.
      const reason = `Unsupported guardrail policy type: ${policy.type}`;
      const policyIds = [...triggeredPolicyIds, policy.id];
      return {
        decision: GuardrailCheckDecision.BLOCK,
        allowed: false,
        transformed_text: null,
        triggered_policy_ids: policyIds,
        evidence: [
          ...evidence,
          {
            policy_id: policy.id,
            message: reason,
            metadata: { policy_type: policy.type },
          },
        ],
        reason,
        metadata: { triggered_count: policyIds.length },
      };
    }

    let policyDecision = checkDecision(policy);
    const shouldTransform = policyDecision === GuardrailCheckDecision.MODIFY;
    const result = engine.evaluate(policy, transformedText);

    if (result === null) {
      continue;
    }

    // A policy that actually triggered but whose `action` (and `on_fail`) does
    // not resolve to a known verb would otherwise fall through to ALLOW and let
    // the matching content pass. That is fail-open on a misconfiguration (e.g.
    // an operator typo), and inconsistent with the fail-closed handling of an
    // unknown policy type above. Fail closed to BLOCK and make the misconfig
    // observable instead.
    if (policyDecision === GuardrailCheckDecision.ALLOW) {
      logger.warn(
        `Guardrail policy ${policy.id} triggered but its action ` +
          `"${policy.action}" did not resolve to a known decision; ` +
          `failing closed to BLOCK`,
      );
      Metrics.recordError("guardrails_unresolved_action", {
        policy_id: policy.id,
      });
      policyDecision = GuardrailCheckDecision.BLOCK;
    }

    evidence.push(...result.evidence);
    if (shouldTransform && result.transformedText != null) {
      transformedText = result.transformedText;
    }

    triggeredPolicyIds.push(policy.id);
    if (DECISION_PRIORITY[policyDecision] > DECISION_PRIORITY[decision]) {
      decision = policyDecision;
    }
  }

  if (decision === GuardrailCheckDecision.ALLOW) {
    return {
      decision: GuardrailCheckDecision.ALLOW,
      allowed: true,
      transformed_text: request.input.text,
      triggered_policy_ids: [],
      evidence: [],
      reason: null,
      metadata: {},
    };
  }

  const policyCount = triggeredPolicyIds.length;
  let reason: string;
  if (decision === GuardrailCheckDecision.BLOCK) {
    reason = `${policyCount} guardrail policy triggered a block decision`;
  } else if (decision === GuardrailCheckDecision.REQUIRE_REVIEW) {
    reason = `${policyCount} guardrail policy requires review`;
  } else if (decision === GuardrailCheckDecision.MODIFY) {
    reason = `${policyCount} guardrail policy modified the content`;
  } else {
    reason = `${policyCount} guardrail policy triggered`;
  }

  const allowed =
    decision === GuardrailCheckDecision.LOG_ONLY ||
    decision === GuardrailCheckDecision.MODIFY;
  return {
    decision,
    allowed,
    transformed_text: allowed ? transformedText : null,
    triggered_policy_ids: triggeredPolicyIds,
    evidence,
    reason,
    metadata: { triggered_count: policyCount },
  };
}
