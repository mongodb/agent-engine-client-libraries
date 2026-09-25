/**
 * Frozen interrupt-artifact wire key.
 *
 * Dependency leaf (no imports): `workflow/memory.ts` and `secure_wrapper.ts`
 * both consume this without forming a module cycle through the workflow
 * barrel. The key is persisted into checkpoints and replay logs, and
 * duplicated in runner-shared/src/agent_engine_runner_shared/secure_wrapper.py. Changing
 * either value breaks interrupt detection on already-checkpointed sessions
 * and/or cross-language parity — keep the two in lockstep.
 */
export const CALL_INTERRUPTED_ARTIFACT_KEY =
  "__agent_engine_oe_call_interrupted__";
