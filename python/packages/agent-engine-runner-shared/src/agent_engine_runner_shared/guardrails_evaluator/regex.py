"""Regex guardrail policy engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import regex
from pydantic import JsonValue

from agent_engine_runner_shared.guardrails_evaluator.core import (
    GuardrailPolicyEngineResult,
    register_guardrail_policy_engine,
)
from agent_engine_runner_shared.models import GuardrailCheckEvidence, GuardrailRuntimePolicy

_DEFAULT_REPLACEMENT_TEXT = "[BLOCKED]"
_OUTPUT_VALIDATION_POLICY_TYPE = "output_validation"
_MAX_REGEX_PATTERNS = 100
_MAX_REGEX_TEXT_LENGTH = 20_000
_TOTAL_REGEX_TIMEOUT_SECONDS = 0.2
_REGEX_TIMEOUT_SECONDS = 0.05

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RegexPattern:
    pattern: str
    compiled: Any


@dataclass(frozen=True)
class _RegexMatch:
    pattern: str
    start: int
    end: int


class OutputValidationPolicyEngine:
    policy_type = _OUTPUT_VALIDATION_POLICY_TYPE

    def evaluate(
        self, policy: GuardrailRuntimePolicy, text: str
    ) -> GuardrailPolicyEngineResult | None:
        patterns, invalid_patterns = _compile_regex_patterns(policy)
        checked_patterns = patterns[:_MAX_REGEX_PATTERNS]
        skipped_patterns = [pattern.pattern for pattern in patterns[_MAX_REGEX_PATTERNS:]]
        text_too_long = len(text) > _MAX_REGEX_TEXT_LENGTH

        if skipped_patterns:
            logger.warning(
                "Guardrail policy %s has %d regex patterns, exceeding max %d; skipping %d patterns",
                policy.id,
                len(patterns),
                _MAX_REGEX_PATTERNS,
                len(skipped_patterns),
            )

        if text_too_long:
            logger.warning(
                "Guardrail policy %s regex input length %d exceeds max %d; failing closed",
                policy.id,
                len(text),
                _MAX_REGEX_TEXT_LENGTH,
            )
            matches: list[_RegexMatch] = []
            timed_out_patterns: list[str] = []
            skipped_patterns = [pattern.pattern for pattern in patterns]
        else:
            matches, timed_out_patterns = _find_regex_matches(text, checked_patterns)

        if not matches and not timed_out_patterns and not skipped_patterns and not invalid_patterns:
            return None

        replacement_text = _replacement_text(policy)
        # An invalid pattern means the policy cannot be enforced as written, so
        # it fails closed alongside timed-out and skipped patterns rather than
        # silently passing content through as a clean allow.
        fail_closed = bool(timed_out_patterns or skipped_patterns or invalid_patterns)
        return GuardrailPolicyEngineResult(
            evidence=[
                _regex_evidence(
                    policy,
                    matches,
                    timed_out_patterns,
                    skipped_patterns,
                    invalid_patterns,
                    text_too_long,
                )
            ],
            transformed_text=(
                replacement_text
                if fail_closed
                else _redact_patterns(text, matches, replacement_text)
            ),
        )


def register_regex_guardrail_policy_engine() -> None:
    register_guardrail_policy_engine(OutputValidationPolicyEngine())


def _find_regex_matches(
    text: str, patterns: list[_RegexPattern]
) -> tuple[list[_RegexMatch], list[str]]:
    matches: list[_RegexMatch] = []
    timed_out_patterns: list[str] = []
    deadline = time.monotonic() + _TOTAL_REGEX_TIMEOUT_SECONDS
    for index, pattern in enumerate(patterns):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            timed_out_patterns.extend(pattern.pattern for pattern in patterns[index:])
            break

        try:
            for match in pattern.compiled.finditer(
                text,
                timeout=min(_REGEX_TIMEOUT_SECONDS, remaining_seconds),
            ):
                matches.append(
                    _RegexMatch(
                        pattern=pattern.pattern,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        except TimeoutError:
            timed_out_patterns.append(pattern.pattern)
    matches.sort(key=lambda match: (match.start, match.end))
    return matches, timed_out_patterns


def _regex_evidence(
    policy: GuardrailRuntimePolicy,
    matches: list[_RegexMatch],
    timed_out_patterns: list[str],
    skipped_patterns: list[str],
    invalid_patterns: list[str],
    text_too_long: bool,
) -> GuardrailCheckEvidence:
    matched_patterns = list(dict.fromkeys(match.pattern for match in matches))
    triggered_patterns = list(
        dict.fromkeys(
            [*matched_patterns, *timed_out_patterns, *skipped_patterns, *invalid_patterns]
        )
    )
    regex_matches = [_regex_match_metadata(match) for match in matches]
    return GuardrailCheckEvidence(
        policy_id=policy.id,
        message=_regex_evidence_message(
            policy,
            triggered_patterns,
            invalid_patterns,
            text_too_long or bool(timed_out_patterns) or bool(skipped_patterns),
        ),
        metadata=cast(
            dict[str, JsonValue],
            {
                "engine": OutputValidationPolicyEngine.policy_type,
                "validator": "regex_match",
                "matched_patterns": matched_patterns,
                "timed_out_patterns": timed_out_patterns,
                "skipped_patterns": skipped_patterns,
                "invalid_patterns": invalid_patterns,
                "text_too_long": text_too_long,
                "matches": regex_matches,
                "regex_matches": regex_matches,
            },
        ),
    )


def _regex_evidence_message(
    policy: GuardrailRuntimePolicy,
    triggered_patterns: list[str],
    invalid_patterns: list[str],
    safety_limited: bool,
) -> str:
    pattern_summary = ", ".join(triggered_patterns)
    if invalid_patterns:
        invalid_summary = ", ".join(invalid_patterns)
        return (
            "Regex guardrail failed closed; invalid patterns could not be "
            f"evaluated: {invalid_summary}"
        )
    if safety_limited:
        return f"Regex evaluation exceeded safety limits: {pattern_summary}"

    decision = policy.check_decision()
    if triggered_patterns and decision.value == "block":
        return f"Regex match blocked content: {pattern_summary}"
    if triggered_patterns and decision.value == "modify":
        return f"Regex match modified content: {pattern_summary}"
    if triggered_patterns and decision.value == "log_only":
        return f"Regex match logged: {pattern_summary}"
    if triggered_patterns and decision.value == "require_review":
        return f"Regex match requires review: {pattern_summary}"
    return f"Regex match triggered: {pattern_summary}"


def _regex_match_metadata(match: _RegexMatch) -> dict[str, JsonValue]:
    return {
        "pattern": match.pattern,
        "start": match.start,
        "end": match.end,
        "length": match.end - match.start,
    }


def _redact_patterns(text: str, matches: list[_RegexMatch], replacement_text: str) -> str:
    redacted = text
    for start, end in reversed(_merged_spans(matches)):
        redacted = f"{redacted[:start]}{replacement_text}{redacted[end:]}"
    return redacted


def _merged_spans(matches: list[_RegexMatch]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in sorted(matches, key=lambda m: (m.start, m.end)):
        if match.start == match.end:
            continue
        if not spans or match.start > spans[-1][1]:
            spans.append((match.start, match.end))
            continue
        spans[-1] = (spans[-1][0], max(spans[-1][1], match.end))
    return spans


def _compile_regex_patterns(
    policy: GuardrailRuntimePolicy,
) -> tuple[list[_RegexPattern], list[str]]:
    """Compile configured patterns, returning (valid_patterns, invalid_pattern_strings).

    Un-compilable patterns are returned separately rather than dropped so the
    caller can fail closed and surface them — a pattern that cannot be compiled
    means the operator's policy is not being enforced as written.

    The warning log line deliberately omits the pattern text to avoid leaking
    policy content into logs. The pattern is still returned to the caller, which
    includes it in the operator-facing check evidence (as with matched,
    skipped, and timed-out patterns) so the failing pattern is identifiable.
    """
    raw = policy.config.get("regex_patterns")
    if not isinstance(raw, list):
        return [], []

    patterns: list[_RegexPattern] = []
    invalid_patterns: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        pattern = value.strip()
        if pattern == "":
            continue
        try:
            compiled = regex.compile(pattern, regex.IGNORECASE)
        except regex.error:
            logger.warning(
                "Invalid guardrail regex pattern for policy %s; failing closed",
                policy.id,
            )
            invalid_patterns.append(pattern)
            continue
        patterns.append(_RegexPattern(pattern=pattern, compiled=compiled))
    return patterns, invalid_patterns


def _replacement_text(policy: GuardrailRuntimePolicy) -> str:
    value = policy.config.get("replacement_text")
    if isinstance(value, str):
        return value
    return _DEFAULT_REPLACEMENT_TEXT


__all__ = [
    "OutputValidationPolicyEngine",
    "register_regex_guardrail_policy_engine",
]
