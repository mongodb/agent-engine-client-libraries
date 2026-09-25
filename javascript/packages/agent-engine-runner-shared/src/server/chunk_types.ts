/**
 * Wire-protocol constants for AER → OE stream chunks.
 *
 * Centralized so a typo in one site can't silently desync the consumer.
 * Values are stable strings forwarded verbatim through OE to the API gateway
 * and on to the UI; do not rename without coordinating across all three
 * layers (and the Go-side StreamChunk model).
 */

// Token of agent output (post-<think> filter, per-source state).
export const TEXT = "text";

// Subagent lifecycle markers — emitted only when guardrails are disabled.
export const SUBAGENT_START = "subagent_start";
export const SUBAGENT_END = "subagent_end";

// Mid-execution progress update emitted by tool functions via emitStep().
export const STEP = "step";

// Output-parser frame: carries only the custom_event payload
// field, no platform data. Consumers gate on both this chunk_type and the
// field's presence — chunk_type is the explicit discriminator; presence guards
// against a parser payload of JSON null.
export const CUSTOM_EVENT = "custom_event";

// Stream terminators.
export const DONE = "done";
export const ERROR = "error";

// Machine-readable discriminator set on metadata.error_code of an ERROR
// chunk/callback when the OE policy engine denied the call (as opposed to a
// crash or timeout). Lets the OE/UI render a "blocked by policy" outcome
// distinctly instead of string-matching the error message. Forwarded verbatim
// like the chunk_type values above.
export const POLICY_DENIED_ERROR_CODE = "policy_denied";

/**
 * Machine-readable discriminator set on `metadata.error_code` of an ERROR
 * chunk/callback when a deadline expired rather than the work failing. Covers
 * both flavours — one tool/LLM call overrunning, and the whole turn overrunning —
 * because a consumer's response to either is the same: offer more time or a
 * retry, not a bug report. Forwarded verbatim like the values above.
 */
export const TIMEOUT_ERROR_CODE = "timeout";

/**
 * Machine-readable discriminator set on `metadata.error_code` of an ERROR
 * chunk/callback when the agent's LLM call failed after OE approval (provider
 * 401/404, rate limit, etc.). `metadata.source` is `llm` so invoke
 * attribution does not have to string-match the error prose.
 */
export const LLM_INVOCATION_ERROR_CODE = "llm_invocation_failed";

/** Owner attribution for LLM-provider failures. Must match the gateway invoke-owner source switch. */
export const LLM_INVOCATION_ERROR_SOURCE = "llm";

/**
 * Machine-readable discriminator set on `metadata.error_code` when the LLM
 * provider rejected the configured credential (HTTP 401/403). Unlike
 * `llm_invocation_failed` this is deliberately stamped WITHOUT
 * `metadata.source`: the fix is the customer's own project secret, and the
 * gateway's owner switch maps the bare code to client-owned, whereas an
 * `llm` source would attribute it to provider flakiness.
 */
export const LLM_CREDENTIAL_REJECTED_ERROR_CODE = "llm_credential_rejected";

/**
 * Same carrier when a run fails because a tool call's classified external-API
 * failure was an auth rejection (AUTH_FAILED): the rejected credential is one
 * of the project's secrets for that tool, not the LLM key. Also stamped
 * without `metadata.source` so the code drives client-owned attribution.
 */
export const TOOL_CREDENTIAL_REJECTED_ERROR_CODE = "tool_credential_rejected";
