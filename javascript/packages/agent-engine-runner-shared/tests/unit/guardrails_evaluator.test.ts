/**
 * Tests for the guardrails evaluator (registry + strictest-wins aggregation +
 * output_validation regex engine).
 *
 * Mirrors Python tests/unit/test_guardrails_protocol.py, testing
 * `evaluateGuardrailCheck` directly rather than over HTTP.
 *
 * Divergence: the Python `...redacts_entire_text_when_regex_times_out` case
 * drives a catastrophically-backtracking pattern `(a+)+$`. Python's `regex`
 * module interrupts it via a per-match timeout; stock JS `RegExp` cannot, so
 * running that pattern would hang the event loop. It is therefore
 * intentionally not ported — the timeout path is a between-patterns
 * wall-clock guard, not a per-match interrupt.
 */

import { describe, test, expect } from "vitest";
import {
  evaluateGuardrailCheck,
  GuardrailCheckDecision,
  GuardrailCheckRequestSchema,
  type GuardrailCheckRequest,
} from "../../src/index.js";

interface PolicyOverrides {
  id?: string;
  action?: string;
  regex_patterns?: unknown;
  on_fail?: string | null;
  replacement_text?: unknown;
  stage_filter?: string[];
  status?: string;
  type?: string;
}

function policy(overrides: PolicyOverrides = {}): Record<string, unknown> {
  const config: Record<string, unknown> = {
    regex_patterns:
      overrides.regex_patterns === undefined
        ? ["Acme"]
        : overrides.regex_patterns,
  };
  if (overrides.on_fail != null) config.on_fail = overrides.on_fail;
  if (overrides.replacement_text != null) {
    config.replacement_text = overrides.replacement_text;
  }
  return {
    id: overrides.id ?? "000000000000000000000003",
    type: overrides.type ?? "output_validation",
    status: overrides.status ?? "active",
    action: overrides.action ?? "modify",
    stage_filter: overrides.stage_filter ?? ["llm_output"],
    config,
  };
}

function request(opts: {
  text: string;
  policies?: Array<Record<string, unknown>>;
  action?: string;
  on_fail?: string | null;
  regex_patterns?: unknown;
  replacement_text?: unknown;
  stage_filter?: string[];
  status?: string;
}): GuardrailCheckRequest {
  const singlePolicy = policy({
    action: opts.action,
    on_fail: opts.on_fail === undefined ? "fix" : opts.on_fail,
    regex_patterns: opts.regex_patterns,
    replacement_text: opts.replacement_text,
    stage_filter: opts.stage_filter,
    status: opts.status,
  });
  return GuardrailCheckRequestSchema.parse({
    execution_id: "exec-1",
    stage: "llm_output",
    input: { text: opts.text, metadata: { source: "invoke_llm" } },
    context: {
      org_id: "000000000000000000000001",
      project_id: "000000000000000000000002",
      workspace_id: "workspace-1",
      session_id: "session-1",
      user_id: "user-1",
    },
    policies: opts.policies ?? [singlePolicy],
  });
}

const meta = (check: ReturnType<typeof evaluateGuardrailCheck>) =>
  check.evidence[0].metadata as Record<string, unknown>;

describe("evaluateGuardrailCheck", () => {
  test("allows clean text", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "MongoDB is the only company mentioned here." }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.ALLOW);
    expect(check.allowed).toBe(true);
    expect(check.transformed_text).toBe(
      "MongoDB is the only company mentioned here.",
    );
    expect(check.triggered_policy_ids).toEqual([]);
  });

  test("modifies regex matches when fixing (case-insensitive)", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "geico and acme should both be hidden." }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.allowed).toBe(true);
    expect(check.transformed_text).toBe(
      "geico and [BLOCKED] should both be hidden.",
    );
    expect(check.triggered_policy_ids).toEqual(["000000000000000000000003"]);
    expect(check.evidence[0].policy_id).toBe("000000000000000000000003");
    expect(meta(check).matched_patterns).toEqual(["Acme"]);
    expect(check.evidence[0].message).toBe(
      "Regex match modified content: Acme",
    );
  });

  test("blocks when policy action blocks", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", action: "block" }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(check.triggered_policy_ids).toEqual(["000000000000000000000003"]);
  });

  test("warns without blocking (log_only)", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", action: "noop", on_fail: null }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.LOG_ONLY);
    expect(check.allowed).toBe(true);
    expect(check.transformed_text).toBe("Acme appears here.");
    expect(check.evidence[0].message).toBe("Regex match logged: Acme");
  });

  test("ignores malformed regex_patterns config (not a list)", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", regex_patterns: "Acme" }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.ALLOW);
    expect(check.allowed).toBe(true);
    expect(check.transformed_text).toBe("Acme appears here.");
  });

  test("fails closed when a regex pattern is invalid", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", regex_patterns: ["Acme("] }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.allowed).toBe(true);
    expect(check.transformed_text).toBe("[BLOCKED]");
    expect(meta(check).invalid_patterns).toEqual(["Acme("]);
  });

  test("fails closed when one of many patterns is invalid", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", regex_patterns: ["Acme", "Bad("] }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe("[BLOCKED]");
    expect(meta(check).invalid_patterns).toEqual(["Bad("]);
    expect(meta(check).matched_patterns).toEqual(["Acme"]);
  });

  test("blocks when invalid pattern and action blocks", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "nothing matches here.",
        regex_patterns: ["Bad("],
        action: "block",
        on_fail: null,
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(meta(check).invalid_patterns).toEqual(["Bad("]);
  });

  test("blocks unsupported policy type", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme appears here.",
        policies: [policy({ type: "future_policy" })],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.triggered_policy_ids).toEqual(["000000000000000000000003"]);
    expect(check.reason).toBe(
      "Unsupported guardrail policy type: future_policy",
    );
  });

  test("redacts regex matches and omits matched text from metadata", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "SSN 123-45-6789 should not leave the pod.",
        regex_patterns: ["\\b\\d{3}-\\d{2}-\\d{4}\\b"],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe(
      "SSN [BLOCKED] should not leave the pod.",
    );
    const matches = meta(check).matches as Array<Record<string, unknown>>;
    expect(matches[0].text).toBeUndefined();
    expect(matches[0].length).toBe(11);
  });

  test("redacts overlapping matches once", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "abc should be hidden.", regex_patterns: ["bc", "abc"] }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe("[BLOCKED] should be hidden.");
  });

  test("supports multiple literal regex patterns", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme SSN 123-45-6789 should not leave the pod.",
        regex_patterns: ["Acme", "\\b\\d{3}-\\d{2}-\\d{4}\\b"],
      }),
    );
    expect(check.transformed_text).toBe(
      "[BLOCKED] SSN [BLOCKED] should not leave the pod.",
    );
    expect(meta(check).matched_patterns).toEqual([
      "Acme",
      "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    ]);
  });

  test("uses configured replacement text", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme should be hidden.",
        regex_patterns: ["Acme"],
        replacement_text: "[\\REDACTED]",
      }),
    );
    expect(check.transformed_text).toBe("[\\REDACTED] should be hidden.");
  });

  test("uses running text after a modify", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme should be hidden.",
        policies: [
          policy({
            id: "modify-policy",
            action: "modify",
            regex_patterns: ["Acme"],
          }),
          policy({
            id: "block-policy",
            action: "block",
            regex_patterns: ["Acme"],
          }),
        ],
      }),
    );
    // The block policy sees the already-redacted text, so it no longer matches.
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe("[BLOCKED] should be hidden.");
    expect(check.triggered_policy_ids).toEqual(["modify-policy"]);
  });

  test("escalates to the strictest decision across policies", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme and Secret should be hidden.",
        policies: [
          policy({
            id: "modify-policy",
            action: "modify",
            regex_patterns: ["Acme"],
          }),
          policy({
            id: "block-policy",
            action: "block",
            regex_patterns: ["Secret"],
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(check.triggered_policy_ids).toEqual([
      "modify-policy",
      "block-policy",
    ]);
    expect(check.evidence.map((e) => e.policy_id)).toEqual([
      "modify-policy",
      "block-policy",
    ]);
  });

  test("requires review when policy action is require_review", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "Acme appears here.", action: "require_review" }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.REQUIRE_REVIEW);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(check.triggered_policy_ids).toEqual(["000000000000000000000003"]);
    expect(check.evidence[0].message).toBe("Regex match requires review: Acme");
  });

  test("require_review wins over modify (strictest-wins ordering)", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme and Secret should be handled.",
        policies: [
          policy({
            id: "modify-policy",
            action: "modify",
            regex_patterns: ["Acme"],
          }),
          policy({
            id: "review-policy",
            action: "require_review",
            regex_patterns: ["Secret"],
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.REQUIRE_REVIEW);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(check.triggered_policy_ids).toEqual([
      "modify-policy",
      "review-policy",
    ]);
  });

  test("block wins over require_review (strictest-wins ordering)", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme and Secret should be handled.",
        policies: [
          policy({
            id: "review-policy",
            action: "require_review",
            regex_patterns: ["Acme"],
          }),
          policy({
            id: "block-policy",
            action: "block",
            regex_patterns: ["Secret"],
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
  });

  test("fails closed to BLOCK when a triggered policy's action is unrecognized", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme appears here.",
        action: "blcok", // operator typo — must not fail open
        on_fail: null,
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.transformed_text).toBeNull();
    expect(check.triggered_policy_ids).toEqual(["000000000000000000000003"]);
  });

  test("chains modify policies", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme and MongoDB should be hidden.",
        policies: [
          policy({
            id: "acme-policy",
            action: "modify",
            regex_patterns: ["Acme"],
            replacement_text: "[A]",
          }),
          policy({
            id: "mongodb-policy",
            action: "modify",
            regex_patterns: ["MongoDB"],
            replacement_text: "[M]",
          }),
        ],
      }),
    );
    expect(check.transformed_text).toBe("[A] and [M] should be hidden.");
    expect(check.triggered_policy_ids).toEqual([
      "acme-policy",
      "mongodb-policy",
    ]);
  });

  test("skips non-matching stage filter", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme should not trigger.",
        policies: [
          policy({
            id: "input-policy",
            action: "block",
            regex_patterns: ["Acme"],
            stage_filter: ["llm_input"],
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.ALLOW);
    expect(check.transformed_text).toBe("Acme should not trigger.");
  });

  test("skips inactive policies", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme should not trigger.",
        policies: [
          policy({
            id: "inactive-policy",
            action: "block",
            regex_patterns: ["Acme"],
            status: "disabled",
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.ALLOW);
    expect(check.transformed_text).toBe("Acme should not trigger.");
  });

  test("keeps prior evidence when a later policy type is unsupported", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme should be hidden.",
        policies: [
          policy({
            id: "modify-policy",
            action: "modify",
            regex_patterns: ["Acme"],
          }),
          policy({
            id: "future-policy",
            action: "block",
            regex_patterns: ["Acme"],
            type: "future_policy",
          }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
    expect(check.allowed).toBe(false);
    expect(check.triggered_policy_ids).toEqual([
      "modify-policy",
      "future-policy",
    ]);
    expect(check.evidence.map((e) => e.policy_id)).toEqual([
      "modify-policy",
      "future-policy",
    ]);
    expect(check.metadata.triggered_count).toBe(2);
  });

  test("fails closed when too many patterns are supplied", () => {
    const patterns = Array.from({ length: 101 }, (_, i) => `pattern-${i}`);
    const check = evaluateGuardrailCheck(
      request({
        text: "Nothing matches, but the policy exceeds the pattern limit.",
        regex_patterns: patterns,
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe("[BLOCKED]");
    expect(meta(check).skipped_patterns).toEqual(["pattern-100"]);
  });

  test("fails closed when text is too long", () => {
    const check = evaluateGuardrailCheck(
      request({ text: "a".repeat(20_001), regex_patterns: ["a"] }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.MODIFY);
    expect(check.transformed_text).toBe("[BLOCKED]");
    expect(meta(check).text_too_long).toBe(true);
  });
});

describe("guardrail policy helpers (via evaluator behavior)", () => {
  test("empty stage_filter applies to all stages", () => {
    const check = evaluateGuardrailCheck(
      request({
        text: "Acme here",
        policies: [
          policy({ id: "all-stages", action: "block", stage_filter: [] }),
        ],
      }),
    );
    expect(check.decision).toBe(GuardrailCheckDecision.BLOCK);
  });
});
