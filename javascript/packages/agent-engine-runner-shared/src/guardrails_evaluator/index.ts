/**
 * Guardrails evaluator public surface.
 *
 * Importing this module registers the native `output_validation` regex engine
 * as a side effect (mirroring Python's `guardrails_evaluator/__init__.py`), so
 * `evaluateGuardrailCheck` can resolve it without an explicit registration call.
 */

import { registerRegexGuardrailPolicyEngine } from "./regex.js";

registerRegexGuardrailPolicyEngine();

export {
  evaluateGuardrailCheck,
  registerGuardrailPolicyEngine,
  type GuardrailPolicyEngine,
  type GuardrailPolicyEngineResult,
} from "./core.js";
export {
  OutputValidationPolicyEngine,
  registerRegexGuardrailPolicyEngine,
} from "./regex.js";
