/**
 * Regex guardrail policy engine (`output_validation`).
 *
 * Mirrors Python's `agent_engine_runner_shared/guardrails_evaluator/regex.py`, with one
 * deliberate divergence: Python uses the `regex` module's per-call `timeout=`
 * to bound each pattern. Stock JS `RegExp` has no per-match timeout, so a single
 * catastrophically-backtracking pattern cannot be interrupted mid-match. Per the
 * approved scope we bound risk with a
 * total wall-clock deadline checked *between* patterns plus hard pattern-count
 * and input-length caps, and emit a `guardrails_regex_overrun` metric so the
 * alarm signal exists if we later need RE2/worker isolation (option (b)).
 */

import { getLogger } from "../logger.js";
import { Metrics } from "../metrics.js";
import {
  checkDecision,
  GuardrailCheckDecision,
  type GuardrailCheckEvidence,
  type GuardrailRuntimePolicy,
} from "../models.js";
import {
  registerGuardrailPolicyEngine,
  type GuardrailPolicyEngine,
  type GuardrailPolicyEngineResult,
} from "./core.js";

const DEFAULT_REPLACEMENT_TEXT = "[BLOCKED]";
const OUTPUT_VALIDATION_POLICY_TYPE = "output_validation";
const MAX_REGEX_PATTERNS = 100;
const MAX_REGEX_TEXT_LENGTH = 20_000;
const TOTAL_REGEX_TIMEOUT_MS = 200;

const logger = getLogger(
  "agent_engine_runner_shared.guardrails_evaluator.regex",
);

interface RegexPattern {
  pattern: string;
  compiled: RegExp;
}

interface RegexMatch {
  pattern: string;
  start: number;
  end: number;
}

export class OutputValidationPolicyEngine implements GuardrailPolicyEngine {
  readonly policyType = OUTPUT_VALIDATION_POLICY_TYPE;

  evaluate(
    policy: GuardrailRuntimePolicy,
    text: string,
  ): GuardrailPolicyEngineResult | null {
    const { patterns, invalidPatterns } = compileRegexPatterns(policy);
    const checkedPatterns = patterns.slice(0, MAX_REGEX_PATTERNS);
    let skippedPatterns = patterns
      .slice(MAX_REGEX_PATTERNS)
      .map((p) => p.pattern);
    const textTooLong = text.length > MAX_REGEX_TEXT_LENGTH;

    if (skippedPatterns.length > 0) {
      logger.warn(
        `Guardrail policy ${policy.id} has ${patterns.length} regex patterns, ` +
          `exceeding max ${MAX_REGEX_PATTERNS}; skipping ${skippedPatterns.length} patterns`,
      );
    }

    let matches: RegexMatch[];
    let timedOutPatterns: string[];
    if (textTooLong) {
      logger.warn(
        `Guardrail policy ${policy.id} regex input length ${text.length} ` +
          `exceeds max ${MAX_REGEX_TEXT_LENGTH}; failing closed`,
      );
      matches = [];
      timedOutPatterns = [];
      skippedPatterns = patterns.map((p) => p.pattern);
    } else {
      ({ matches, timedOutPatterns } = findRegexMatches(
        text,
        checkedPatterns,
        policy.id,
      ));
    }

    if (
      matches.length === 0 &&
      timedOutPatterns.length === 0 &&
      skippedPatterns.length === 0 &&
      invalidPatterns.length === 0
    ) {
      return null;
    }

    const replacementText = replacementTextFor(policy);
    // An invalid pattern means the policy cannot be enforced as written, so it
    // fails closed alongside timed-out and skipped patterns rather than silently
    // passing content through as a clean allow.
    const failClosed =
      timedOutPatterns.length > 0 ||
      skippedPatterns.length > 0 ||
      invalidPatterns.length > 0;
    return {
      evidence: [
        regexEvidence(
          policy,
          matches,
          timedOutPatterns,
          skippedPatterns,
          invalidPatterns,
          textTooLong,
        ),
      ],
      transformedText: failClosed
        ? replacementText
        : redactPatterns(text, matches, replacementText),
    };
  }
}

export function registerRegexGuardrailPolicyEngine(): void {
  registerGuardrailPolicyEngine(new OutputValidationPolicyEngine());
}

function findRegexMatches(
  text: string,
  patterns: RegexPattern[],
  policyId: string,
): { matches: RegexMatch[]; timedOutPatterns: string[] } {
  const matches: RegexMatch[] = [];
  const timedOutPatterns: string[] = [];
  const deadline = performance.now() + TOTAL_REGEX_TIMEOUT_MS;

  for (let index = 0; index < patterns.length; index += 1) {
    if (performance.now() >= deadline) {
      // Deadline exceeded: JS cannot interrupt a running match, so the best we
      // can do is stop before the remaining patterns and fail closed on them.
      timedOutPatterns.push(...patterns.slice(index).map((p) => p.pattern));
      Metrics.recordError("guardrails_regex_overrun", { policy_id: policyId });
      logger.warn(
        `Guardrail policy ${policyId} regex evaluation exceeded ` +
          `${TOTAL_REGEX_TIMEOUT_MS}ms; failing closed on ` +
          `${patterns.length - index} remaining patterns`,
      );
      break;
    }

    const pattern = patterns[index];
    for (const match of text.matchAll(pattern.compiled)) {
      // `match.index` is typed as optional; a `NaN`/undefined start would
      // corrupt the sort and span-merge redaction below, so skip such matches
      // rather than emit a match with a `NaN` offset.
      const start = match.index;
      if (start === undefined || Number.isNaN(start)) {
        continue;
      }
      matches.push({
        pattern: pattern.pattern,
        start,
        end: start + match[0].length,
      });
    }
  }

  matches.sort((a, b) => a.start - b.start || a.end - b.end);
  return { matches, timedOutPatterns };
}

function regexEvidence(
  policy: GuardrailRuntimePolicy,
  matches: RegexMatch[],
  timedOutPatterns: string[],
  skippedPatterns: string[],
  invalidPatterns: string[],
  textTooLong: boolean,
): GuardrailCheckEvidence {
  const matchedPatterns = [...new Set(matches.map((m) => m.pattern))];
  const triggeredPatterns = [
    ...new Set([
      ...matchedPatterns,
      ...timedOutPatterns,
      ...skippedPatterns,
      ...invalidPatterns,
    ]),
  ];
  const regexMatches = matches.map(regexMatchMetadata);
  return {
    policy_id: policy.id,
    message: regexEvidenceMessage(
      policy,
      triggeredPatterns,
      invalidPatterns,
      textTooLong || timedOutPatterns.length > 0 || skippedPatterns.length > 0,
    ),
    metadata: {
      engine: OUTPUT_VALIDATION_POLICY_TYPE,
      validator: "regex_match",
      matched_patterns: matchedPatterns,
      timed_out_patterns: timedOutPatterns,
      skipped_patterns: skippedPatterns,
      invalid_patterns: invalidPatterns,
      text_too_long: textTooLong,
      matches: regexMatches,
      regex_matches: regexMatches,
    },
  };
}

function regexEvidenceMessage(
  policy: GuardrailRuntimePolicy,
  triggeredPatterns: string[],
  invalidPatterns: string[],
  safetyLimited: boolean,
): string {
  const patternSummary = triggeredPatterns.join(", ");
  if (invalidPatterns.length > 0) {
    return (
      "Regex guardrail failed closed; invalid patterns could not be " +
      `evaluated: ${invalidPatterns.join(", ")}`
    );
  }
  if (safetyLimited) {
    return `Regex evaluation exceeded safety limits: ${patternSummary}`;
  }

  const decision = checkDecision(policy);
  if (
    triggeredPatterns.length > 0 &&
    decision === GuardrailCheckDecision.BLOCK
  ) {
    return `Regex match blocked content: ${patternSummary}`;
  }
  if (
    triggeredPatterns.length > 0 &&
    decision === GuardrailCheckDecision.MODIFY
  ) {
    return `Regex match modified content: ${patternSummary}`;
  }
  if (
    triggeredPatterns.length > 0 &&
    decision === GuardrailCheckDecision.LOG_ONLY
  ) {
    return `Regex match logged: ${patternSummary}`;
  }
  if (
    triggeredPatterns.length > 0 &&
    decision === GuardrailCheckDecision.REQUIRE_REVIEW
  ) {
    return `Regex match requires review: ${patternSummary}`;
  }
  return `Regex match triggered: ${patternSummary}`;
}

function regexMatchMetadata(
  match: RegexMatch,
): Record<string, number | string> {
  return {
    pattern: match.pattern,
    start: match.start,
    end: match.end,
    length: match.end - match.start,
  };
}

function redactPatterns(
  text: string,
  matches: RegexMatch[],
  replacementText: string,
): string {
  let redacted = text;
  const spans = mergedSpans(matches);
  for (let i = spans.length - 1; i >= 0; i -= 1) {
    const [start, end] = spans[i];
    redacted = `${redacted.slice(0, start)}${replacementText}${redacted.slice(end)}`;
  }
  return redacted;
}

function mergedSpans(matches: RegexMatch[]): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  const sorted = [...matches].sort(
    (a, b) => a.start - b.start || a.end - b.end,
  );
  for (const match of sorted) {
    if (match.start === match.end) {
      continue;
    }
    const last = spans[spans.length - 1];
    if (last === undefined || match.start > last[1]) {
      spans.push([match.start, match.end]);
      continue;
    }
    last[1] = Math.max(last[1], match.end);
  }
  return spans;
}

/**
 * Compile configured patterns, returning valid patterns and the raw strings of
 * any that failed to compile. Un-compilable patterns are surfaced (not dropped)
 * so the caller can fail closed — a pattern that cannot compile means the
 * operator's policy is not being enforced as written. The warning omits the
 * pattern text to avoid leaking policy content into logs; the pattern is still
 * returned so it can appear in operator-facing evidence.
 */
function compileRegexPatterns(policy: GuardrailRuntimePolicy): {
  patterns: RegexPattern[];
  invalidPatterns: string[];
} {
  const raw = policy.config.regex_patterns;
  if (!Array.isArray(raw)) {
    return { patterns: [], invalidPatterns: [] };
  }

  const patterns: RegexPattern[] = [];
  const invalidPatterns: string[] = [];
  for (const value of raw) {
    if (typeof value !== "string") {
      continue;
    }
    const pattern = value.trim();
    if (pattern === "") {
      continue;
    }
    try {
      // `g` for finditer-style global scan, `i` to mirror Python's IGNORECASE.
      patterns.push({ pattern, compiled: new RegExp(pattern, "gi") });
    } catch {
      logger.warn(
        `Invalid guardrail regex pattern for policy ${policy.id}; failing closed`,
      );
      invalidPatterns.push(pattern);
    }
  }
  return { patterns, invalidPatterns };
}

function replacementTextFor(policy: GuardrailRuntimePolicy): string {
  const value = policy.config.replacement_text;
  return typeof value === "string" ? value : DEFAULT_REPLACEMENT_TEXT;
}
