/**
 * Shared wire-protocol models for Runner SDK components.
 *
 * Defines the API contracts between:
 * - Orchestration Engine (OE)
 * - Agent Execution Runtime (AER)
 * - Tool Executor Pod
 *
 * Each Pydantic model in `agent_engine_runner_shared/models.py` maps to:
 *  - a Zod schema (`FooSchema`) for runtime validation, and
 *  - the inferred TS type (`Foo = z.infer<typeof FooSchema>`).
 *
 * Models with non-trivial methods (`LLMResult`) follow the agent-engine-sdk
 * pattern of class + separate schema.
 */

import { z } from "zod";
import { recordSuspendRequest } from "./context.js";
import { ToolAPIErrorSchema } from "./tool_api_error.js";
import type { JsonValue } from "@mongodb-js/agent-engine-sdk";
import {
  JsonValueSchema,
  LLMInvocationOptions,
  LLMInvocationOptionsSchema,
  LLMResponse,
  LLMResponseSchema,
  LLMTokenUsage,
  LLMTokenUsageSchema,
  LLMToolCall,
  LLMToolSchema,
  LLMToolSchemaValidator,
  MessageSchema,
  serializeMessage,
  ToolCallChunkSchema,
  type Message,
  type ToolCallChunk,
} from "@mongodb-js/agent-engine-sdk";

// =============================================================================
// Execution Status
// =============================================================================

/** Status of an agent execution. */
export const ExecutionStatusSchema = z.enum([
  "pending",
  "running",
  "suspended",
  "resuming",
  "completed",
  "error",
  "cancelled",
]);
export type ExecutionStatus = z.infer<typeof ExecutionStatusSchema>;
export const ExecutionStatus = {
  PENDING: "pending",
  RUNNING: "running",
  SUSPENDED: "suspended",
  RESUMING: "resuming",
  COMPLETED: "completed",
  ERROR: "error",
  CANCELLED: "cancelled",
} as const;

// =============================================================================
// Suspend Payload (Agent → Platform)
// =============================================================================

/**
 * Payload returned by an agent tool to trigger human-in-the-loop suspension.
 *
 * Agent tools signal a suspend by returning a JSON string containing these
 * fields. The `__suspend__` flag is stripped before the payload is passed
 * to the framework's interrupt() handler.
 *
 * Example usage in an agent tool:
 *
 *     import { suspendPayloadToJson } from '@mongodb-js/agent-engine-runner-shared'
 *     return suspendPayloadToJson({
 *       suspend_reason: 'awaiting_human_review',
 *       suspend_context: { claim_id: 'C-123', task_id: 'T-456' },
 *     })
 */
export const SuspendPayloadSchema = z.object({
  /** Why the agent is suspending (e.g. 'awaiting_human_review'). */
  suspend_reason: z.string(),
  /** Arbitrary context the human reviewer needs to make a decision. */
  suspend_context: z.record(z.string(), z.unknown()).default({}),
});
export type SuspendPayload = z.infer<typeof SuspendPayloadSchema>;

/**
 * Serialize a SuspendPayload to the suspend wire marker, and record an
 * out-of-band suspend request on the current execution frame.
 *
 * The Tool Pod honors suspend from that recorded signal — set only here, in
 * the tool author's own code — not by sniffing tool-result content, so
 * untrusted data a tool relays can no longer forge a HITL suspend. The marker
 * string is still returned unchanged for wire/replay compatibility.
 */
export function suspendPayloadToJson(payload: SuspendPayload): string {
  const marker = { ...payload, __suspend__: true };
  recordSuspendRequest(marker);
  return JSON.stringify(marker);
}

// =============================================================================
// Streaming / Interrupt Results (AER internal)
// =============================================================================

/**
 * Result from agent execution when the agent is suspended.
 *
 * Contains the suspend payload directly (framework-agnostic) rather than
 * wrapping framework-specific Interrupt objects. Any framework-specific
 * state needed to resume (LangGraph checkpoint id, ADK function-call
 * correlation, etc.) is carried opaquely in `metadata` — the AER never
 * inspects it; the framework adapter writes it on suspend and reads it back
 * from `RequestContext.metadata` on resume.
 */
export const PendingInterruptSchema = z.object({
  id: z.string().min(1),
  value: z.unknown(),
});
export type PendingInterrupt = z.infer<typeof PendingInterruptSchema>;

export const InterruptResultSchema = z.object({
  suspend_payload: z.unknown(),
  interrupts: z.array(PendingInterruptSchema).optional(),
  resume_schema: z.record(z.string(), z.unknown()).optional(),
  /** Opaque framework-owned resume state, round-tripped via the OE. */
  metadata: z.record(z.string(), z.unknown()).default({}),
  /** Assistant/tool messages produced before suspension, for Memory ingestion. */
  messages: z.array(MessageSchema).default([]),
});
export type InterruptResult = z.infer<typeof InterruptResultSchema>;

/**
 * Result from agent execution for normal completion.
 *
 * Returned by _execute_via_agent_stream when the agent finishes without
 * suspend. The caller uses content for the final response and messages
 * for memory writing.
 */
export const StreamingResultSchema = z.object({
  /** Sanitized final AI response (thinking stripped, trailing empties skipped). */
  content: z.string().default(""),
  /** All messages from agent execution (sdk-core Message objects), for memory writer. */
  messages: z.array(MessageSchema).default([]),
  /** Opaque framework-owned state produced by the completed execution. */
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type StreamingResult = z.infer<typeof StreamingResultSchema>;

// =============================================================================
// Agent Start (Client → OE)
// =============================================================================

/** Request to invoke the agent (compatible with agent-runtime). */
export const InvokeRequestSchema = z.object({
  /** User message. */
  message: z.string(),
  /** Session ID for trace correlation. */
  session_id: z.string().optional(),
  // Multi-tenant context
  /** Organization ID for multi-tenant isolation. */
  org_id: z.string().optional(),
  /** User ID for personalization. */
  user_id: z.string().optional(),
  /** Workspace identifier for cost tracking. */
  workspace_id: z.string().optional(),
  /** Project ID for project-level scoping. */
  project_id: z.string().optional(),
  // Per-agent AER routing
  /** AER HTTP endpoint for this agent; overrides default AER_URL. */
  aer_url: z.string().optional(),
  /** Tool HTTP endpoint for this agent; overrides default TOOL_URL. */
  tool_url: z.string().optional(),
  // Sync mode (default: wait for completion)
  /** If true, wait for completion and return result. */
  wait: z.boolean().default(true),
  /** Caller-provided custom headers (X-Mdb-Agent-Engine-Custom-* HTTP headers, prefix-stripped and lowercased). */
  custom_headers: z.record(z.string(), z.string()).optional(),
});
export type InvokeRequest = z.infer<typeof InvokeRequestSchema>;

/** Response from invoking the agent (compatible with agent-runtime). */
export const InvokeResponseSchema = z.object({
  /** The agent's response. */
  result: z.unknown().optional(),
  /** Session ID. */
  session_id: z.string().optional(),
  /** User ID used. */
  user_id: z.string().optional(),
  // Runner-specific fields (for async mode and tracking)
  /** Unique execution identifier. */
  execution_id: z.string().optional(),
  /** Execution status. */
  status: z.string().optional(),
  /** Error message if failed. */
  error: z.string().optional(),
});
export type InvokeResponse = z.infer<typeof InvokeResponseSchema>;

// =============================================================================
// Execute Request (OE → AER)
// =============================================================================

/** Request to execute an agent in AER. */
export const ExecuteRequestSchema = z.object({
  platform_trace_id: z.string().nullable().optional(),
  /** Unique execution identifier. */
  execution_id: z.string(),
  /**
   * User message. Defaults to empty. Promotion of `payload.message` into an
   * empty top-level message happens in the AER (`resolveInvocationParams`),
   * NOT at the model level — mirrors Python's `ExecuteRequest`. Keeping it
   * out of the model lets the AER reject a message supplied in both places
   * as a clean 400 instead of a 422 that would echo the payload into logs.
   */
  message: z.string().default(""),
  /** URL of the OE for callbacks. */
  platform_api_url: z.string(),
  /** Suspension generation expected by this dispatch. */
  suspend_generation: z
    .number()
    .int()
    .nonnegative()
    .nullable()
    .transform((value) => value ?? undefined)
    .optional(),
  /** Replica-specific OE owner URL for callback fallback. */
  platform_api_owner_url: z.string().nullable().optional(),
  /** Whether this is a resume after SUSPEND. */
  resume: z.boolean().default(false),
  /** Step to resume from. */
  resume_from_step: z.number().int().optional(),
  /** Structured data to inject on resume. */
  resume_data: z.record(z.string(), z.unknown()).optional(),
  // Multi-tenant context
  /** Organization ID for multi-tenant isolation. */
  org_id: z.string().optional(),
  /** User ID for personalization. */
  user_id: z.string().optional(),
  /** Session ID; the AER adapter uses this as the LangGraph thread_id. */
  session_id: z.string().optional(),
  /** Workspace identifier for cost tracking. */
  workspace_id: z.string().optional(),
  /** Project ID for project-level scoping. */
  project_id: z.string().optional(),
  /** Caller-provided custom headers. */
  custom_headers: z.record(z.string(), z.string()).optional(),
  /**
   * Opaque caller-provided input forwarded unchanged on every dispatch;
   * `payload.message` fills the top-level message when it is empty.
   */
  payload: z.record(z.string(), z.unknown()).optional(),
  /**
   * Opaque framework-owned state passed through to the selected adapter.
   */
  metadata: z.record(z.string(), z.unknown()).optional(),
  /**
   * True when this session's most recent prior execution was cancelled (pod
   * torn down mid-run); the framework adapter fences off that run's partial
   * checkpoint writes before running this turn.
   */
  previous_execution_cancelled: z.boolean().default(false),
});
export type ExecuteRequest = z.infer<typeof ExecuteRequestSchema>;

// =============================================================================
// Tool Execution (AER → OE → AER/ToolPod)
// =============================================================================

/**
 * Request to execute a tool (AER → OE for approval).
 */
export const ToolExecuteRequestSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Name of the tool to execute. */
  tool_name: z.string(),
  /** Tool arguments. */
  arguments: z.record(z.string(), z.unknown()),
  /** Step number in execution sequence. */
  step_number: z.number().int(),
  /** Stable LLM tool-call id; joins this call's execution-log records to the session message. */
  tool_call_id: z.string().optional(),
  /** Explicit observability kind. */
  kind: z.string().optional(),
  /** Whether OE should return the approved call to the current AER stack. */
  is_local: z.boolean().default(true),
  /** Top-level tool argument names to redact from execution logs. */
  redact_fields: z.array(z.string()).default([]),
  /** Credential provider type for delegated auth. */
  provider_type: z.string().nullish(),
  /** Requested OAuth scopes for delegated auth. */
  scopes: z.array(z.string()).default([]),
  /** Observability metadata. */
  metadata: z.record(z.string(), z.unknown()).default({}),
  /** Caller-provided headers carried only for this active request. */
  custom_headers: z.record(z.string(), z.string()).optional(),
  /** Active OTel trace ID, for OE execution-log correlation. */
  trace_id: z.string().nullish(),
  /** Active OTel span ID, for OE execution-log correlation. */
  span_id: z.string().nullish(),
  /**
   * Set by callers that pause for the guardrail review a halt names and then
   * resolve the call with `review_id`.
   */
  review_protocol: z.number().int().nullish(),
  /** Asks OE to answer this invoke_llm call from a decided guardrail review. */
  review_id: z.string().nullish(),
});
export type ToolExecuteRequest = z.infer<typeof ToolExecuteRequestSchema>;

/** Authorization details returned when broker consent is required. */
export const ElicitationInfoSchema = z.object({
  elicitation_id: z.string(),
  authorization_url: z.string(),
  message: z.string().default(""),
  created: z.boolean().default(false),
});
export type ElicitationInfo = z.infer<typeof ElicitationInfoSchema>;

/** Identity of the policy that caused a guardrail block or require_review halt. */
export const GuardrailMetaSchema = z.object({
  /** ID of the policy that caused the halt. */
  guardrail_id: z.string(),
  /** Category of the policy that caused the halt. */
  guardrail_category: z.string(),
});
export type GuardrailMeta = z.infer<typeof GuardrailMetaSchema>;

/** One policy that required a guardrail review. */
export const GuardrailReviewPolicySchema = z.object({
  /** Policy ID. */
  id: z.string(),
  /** Policy name. */
  name: z.string().nullish(),
  /** Policy category. */
  category: z.string().nullish(),
});
export type GuardrailReviewPolicy = z.infer<typeof GuardrailReviewPolicySchema>;

/** The review OE opened for a require_review halt of a review_protocol call. */
export const GuardrailReviewHaltSchema = z.object({
  /** ID of the review to pause for. */
  review_id: z.string().min(1),
  /** Decisions a reviewer may give. */
  allowed_decisions: z.array(z.string()).default([]),
  /** Why the output needs review. */
  reason: z.string().nullish(),
  /** Every policy that required review. */
  guardrails: z.array(GuardrailReviewPolicySchema).default([]),
});
export type GuardrailReviewHalt = z.infer<typeof GuardrailReviewHaltSchema>;

/** Response from OE for tool execution request. */
const ToolExecuteResponseObjectSchema = z.object({
  /** Whether to proceed with execution. */
  proceed: z.boolean(),
  /** Cached result if replaying. */
  cached_result: z.unknown().nullish(),
  /** URL to route to (callback or OE-owned stream relay). Null when OE returns result directly. */
  route_to: z.string().nullish(),
  /** Reason if blocked. */
  reason: z.string().nullish(),
  /** Authorization elicitation details when broker consent is required. */
  elicitation: ElicitationInfoSchema.nullish(),
  /** Final status when OE directly executes the intercepted tool. */
  status: z.string().nullish(),
  /** Final result from OE-owned execution. */
  result: z.unknown().nullish(),
  /** Execution error message. */
  error: z.string().nullish(),
  /** Machine-readable failure classification for an invoke_llm error. */
  error_code: z.string().nullish(),
  /** Whether retry is safe. */
  retryable: z.boolean().default(false),
  /** Highest step number observed during nested execution. */
  latest_step_number: z.number().int().nullish(),
  /** Whether replay cache answered. */
  from_cache: z.boolean().default(false),
  /** Execution duration. */
  duration_ms: z.number().nullish(),
  /** Execution pod name. */
  pod_name: z.string().nullish(),
  /** Policy identity when a guardrail halt fires. */
  guardrail_meta: GuardrailMetaSchema.nullish(),
  /** The review to pause for, on a require_review halt of a review_protocol call. */
  guardrail_review: GuardrailReviewHaltSchema.nullish(),
  /** Structured external API failure classification. */
  tool_api_error: ToolAPIErrorSchema.nullish(),
});

/**
 * Assemble `guardrail_meta` from the flat `guardrail_id`/`guardrail_category`
 * fields the OE wire format sends, so callers work with a single structured
 * object. Mirrors Python's `_assemble_guardrail_meta` model validator.
 */
export const ToolExecuteResponseSchema = z.preprocess((data) => {
  if (data != null && typeof data === "object" && !Array.isArray(data)) {
    const d = data as Record<string, unknown>;
    if (d.guardrail_id && d.guardrail_category && d.guardrail_meta == null) {
      return {
        ...d,
        guardrail_meta: {
          guardrail_id: d.guardrail_id,
          guardrail_category: d.guardrail_category,
        },
      };
    }
  }
  return data;
}, ToolExecuteResponseObjectSchema);
export type ToolExecuteResponse = z.infer<typeof ToolExecuteResponseSchema>;

/**
 * Report tool execution result (AER → OE).
 */
export const ToolResultRequestSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Step number. */
  step_number: z.number().int(),
  /** Tool name. */
  tool_name: z.string(),
  /** Stable LLM tool-call id; joins this result's execution-log record to its call and the session message. */
  tool_call_id: z.string().optional(),
  /** Execution status: success, error, suspend, interrupted. */
  status: z.string(),
  /** Tool result if successful. */
  result: z.unknown().optional(),
  /** Error message if failed. */
  error: z.string().optional(),
  /** Execution duration in milliseconds. */
  duration_ms: z.number(),
  /** Hostname/pod name where tool was executed. */
  pod_name: z.string().optional(),
  // Token usage fields (populated for invoke_llm calls)
  /** Prompt/input tokens used. */
  prompt_tokens: z.number().int().optional(),
  /** Completion/output tokens used. */
  completion_tokens: z.number().int().optional(),
  /** Total tokens used. */
  total_tokens: z.number().int().optional(),
  /** LLM model name. */
  model: z.string().optional(),
  /** Workspace identifier. */
  workspace_id: z.string().optional(),
  /** Explicit observability kind. */
  kind: z.string().optional(),
  /** Observability metadata. */
  metadata: z.record(z.string(), z.unknown()).default({}),
  /** Active OTel trace ID, for OE execution-log correlation. */
  trace_id: z.string().nullish(),
  /** Active OTel span ID, for OE execution-log correlation. */
  span_id: z.string().nullish(),
  /** Structured external API failure classification. */
  tool_api_error: ToolAPIErrorSchema.nullish(),
});
export type ToolResultRequest = z.infer<typeof ToolResultRequestSchema>;

// =============================================================================
// Tool Pod Execution (OE → ToolPod)
// =============================================================================

export const MAX_TOOL_ARGUMENT_BYTES = 16 * 1024 * 1024;

const ToolArgumentsSchema = z
  .record(z.string(), JsonValueSchema)
  .superRefine((toolArguments, context) => {
    let serialized: string;
    try {
      serialized = JSON.stringify(toolArguments);
    } catch {
      context.addIssue({
        code: "custom",
        message: "tool arguments must contain JSON values",
      });
      return;
    }
    if (Buffer.byteLength(serialized, "utf-8") > MAX_TOOL_ARGUMENT_BYTES) {
      context.addIssue({
        code: "custom",
        message: `tool arguments exceed ${MAX_TOOL_ARGUMENT_BYTES} bytes`,
      });
    }
  });

/** Delegated credential injected by OE for tool execution. */
export const ToolAuthorizationSchema = z.object({
  /** Bearer token for third-party API access. */
  token: z.string(),
  /** Token expiry as Unix seconds. */
  expires_at: z.number().int().nullish(),
});
export type ToolAuthorization = z.infer<typeof ToolAuthorizationSchema>;

/** Request to execute a tool in a Tool Pod. */
export const ToolPodExecuteRequestSchema = z.object({
  platform_trace_id: z.string().nullable().optional(),
  /** Execution identifier. */
  execution_id: z.string(),
  /** Name of the tool. */
  tool_name: z.string(),
  /** Tool arguments. */
  arguments: ToolArgumentsSchema,
  /** Tool call ID from LLM. */
  tool_call_id: z.string().optional(),
  /**
   * Step number of this call in the execution sequence, so the per-call
   * interrupt (POST /interrupt/call) can address exactly this work.
   * Optional for backward compatibility with older OE dispatchers: absent,
   * the call is only drain-addressable.
   */
  step_number: z.number().int().nonnegative().optional(),
  /**
   * Session ID for memory context. For framework integrations, this is
   * the framework's thread ID. Must be non-empty.
   */
  session_id: z.string().min(1),
  /** User ID for context. */
  user_id: z.string().optional(),
  /** OE callback URL for memory and other operations. */
  oe_url: z.string().optional(),
  /** Replica-specific OE owner callback URL. */
  oe_owner_url: z.string().nullable().optional(),
  /** Delegated credential injected by OE via the credential broker. */
  authorization: ToolAuthorizationSchema.optional(),
  /** Caller-provided custom headers. */
  custom_headers: z.record(z.string(), z.string()).optional(),
  /**
   * Opaque caller-provided invocation payload, forwarded so tools can read it
   * via `getCurrentPayload()`.
   */
  payload: z.record(z.string(), z.unknown()).optional(),
  /**
   * Tool execution metadata forwarded by OE (recognized keys: mcp_server,
   * mcp_tool). Accepted for wire-compatibility; MCP tool resolution is not
   * ported to TS, so it is currently unused here.
   */
  metadata: z.record(z.string(), z.unknown()).default({}),
});
export type ToolPodExecuteRequest = z.infer<typeof ToolPodExecuteRequestSchema>;

/** Response from Tool Pod execution. */
export const ToolPodExecuteResponseSchema = z.object({
  /** Execution status: success, error. */
  status: z.string(),
  /** Tool result. */
  result: z.unknown().optional(),
  /** Error message if failed. */
  error: z.string().optional(),
  /** Hostname/pod name where tool was executed. */
  pod_name: z.string().optional(),
  /** Explicit observability kind. */
  kind: z.string().optional(),
  /** Observability metadata. */
  metadata: z.record(z.string(), z.unknown()).default({}),
  /**
   * True when this runner reports HITL suspend out of band via `status`.
   * Absent on older runners; the OE treats that absence as a
   * legacy pod that still signals suspend through in-band result content.
   */
  oob_suspend_supported: z.boolean().optional(),
  /** Structured external API failure classification. */
  tool_api_error: ToolAPIErrorSchema.nullish(),
});
export type ToolPodExecuteResponse = z.infer<
  typeof ToolPodExecuteResponseSchema
>;

// =============================================================================
// Guardrails Runtime Check (OE → ToolPod)
// =============================================================================

/** Runtime stage where OE is asking the Tool Pod to evaluate guardrails. */
export const GuardrailRuntimeStageSchema = z.enum([
  "llm_input",
  "llm_output",
  "tool_input",
  "tool_output",
]);
export type GuardrailRuntimeStage = z.infer<typeof GuardrailRuntimeStageSchema>;
export const GuardrailRuntimeStage = {
  LLM_INPUT: "llm_input",
  LLM_OUTPUT: "llm_output",
  TOOL_INPUT: "tool_input",
  TOOL_OUTPUT: "tool_output",
} as const;

/** Decision returned by the Tool Pod guardrails evaluator. */
export const GuardrailCheckDecisionSchema = z.enum([
  "allow",
  "block",
  "modify",
  "require_review",
  "log_only",
]);
export type GuardrailCheckDecision = z.infer<
  typeof GuardrailCheckDecisionSchema
>;
export const GuardrailCheckDecision = {
  ALLOW: "allow",
  BLOCK: "block",
  MODIFY: "modify",
  REQUIRE_REVIEW: "require_review",
  LOG_ONLY: "log_only",
} as const;

/** OE-selected guardrail policy sent to the Tool Pod for evaluation. */
export const GuardrailRuntimePolicySchema = z.object({
  /** Guardrail policy identifier. */
  id: z.string(),
  /** Guardrail policy type. */
  type: z.string(),
  /** Guardrail policy status. */
  status: z.string().default("active"),
  /** Action requested when the policy triggers. */
  action: z.string(),
  /** Runtime stages this policy applies to; empty means all stages. */
  stage_filter: z.array(z.string()).default([]),
  /** Evaluator-specific policy configuration. */
  config: z.record(z.string(), JsonValueSchema).default({}),
});
export type GuardrailRuntimePolicy = z.infer<
  typeof GuardrailRuntimePolicySchema
>;

/**
 * Whether `policy` applies at `stage`. Inactive policies never apply; an empty
 * stage filter applies to all stages; otherwise the stage must match (case- and
 * whitespace-insensitive). Mirrors `GuardrailRuntimePolicy.applies_to_stage`.
 */
export function appliesToStage(
  policy: GuardrailRuntimePolicy,
  stage: GuardrailRuntimeStage,
): boolean {
  if (policy.status.trim().toLowerCase() !== "active") {
    return false;
  }
  if (policy.stage_filter.length === 0) {
    return true;
  }
  return policy.stage_filter.some(
    (value) => value.trim().toLowerCase() === stage,
  );
}

/**
 * Resolve the decision a triggered policy requests, from its `action` (falling
 * back to `config.on_fail`). Mirrors `GuardrailRuntimePolicy.check_decision`.
 */
export function checkDecision(
  policy: GuardrailRuntimePolicy,
): GuardrailCheckDecision {
  const action = policy.action.trim().toLowerCase();
  if (["modify", "transform", "fix"].includes(action)) {
    return GuardrailCheckDecision.MODIFY;
  }
  if (["noop", "no_op", "warn", "log_only"].includes(action)) {
    return GuardrailCheckDecision.LOG_ONLY;
  }
  if (action === "require_review") {
    return GuardrailCheckDecision.REQUIRE_REVIEW;
  }
  if (["block", "exception"].includes(action)) {
    return GuardrailCheckDecision.BLOCK;
  }

  const onFail = policy.config.on_fail;
  if (typeof onFail === "string") {
    const normalized = onFail.trim().toLowerCase();
    if (normalized === "fix") {
      return GuardrailCheckDecision.MODIFY;
    }
    if (["noop", "warn", "log_only"].includes(normalized)) {
      return GuardrailCheckDecision.LOG_ONLY;
    }
    if (["block", "exception"].includes(normalized)) {
      return GuardrailCheckDecision.BLOCK;
    }
  }

  return GuardrailCheckDecision.ALLOW;
}

/** Runtime content and metadata to evaluate. */
export const GuardrailCheckInputSchema = z.object({
  /** Text content to evaluate. */
  text: z.string(),
  /** Runtime metadata such as model, tool name, or source. */
  metadata: z.record(z.string(), JsonValueSchema).default({}),
});
export type GuardrailCheckInput = z.infer<typeof GuardrailCheckInputSchema>;

/** Execution context for a guardrail check. */
export const GuardrailCheckContextSchema = z.object({
  /** Organization ID. */
  org_id: z.string(),
  /** Project ID. */
  project_id: z.string(),
  /** Workspace ID. */
  workspace_id: z.string().nullish(),
  /** Session ID. */
  session_id: z.string().nullish(),
  /** User ID. */
  user_id: z.string().nullish(),
});
export type GuardrailCheckContext = z.infer<typeof GuardrailCheckContextSchema>;

/** Request from OE to Tool Pod to evaluate selected guardrail policies. */
export const GuardrailCheckRequestSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Runtime stage being evaluated. */
  stage: GuardrailRuntimeStageSchema,
  /** Content to evaluate. */
  input: GuardrailCheckInputSchema,
  /** Execution context. */
  context: GuardrailCheckContextSchema,
  /** OE-selected policies to evaluate. */
  policies: z.array(GuardrailRuntimePolicySchema).default([]),
});
export type GuardrailCheckRequest = z.infer<typeof GuardrailCheckRequestSchema>;

/** Evidence explaining why a guardrail policy triggered. */
export const GuardrailCheckEvidenceSchema = z.object({
  /** Triggered policy identifier. */
  policy_id: z.string(),
  /** Human-readable evidence summary. */
  message: z.string().default(""),
  /** Evaluator-specific evidence metadata. */
  metadata: z.record(z.string(), JsonValueSchema).default({}),
});
export type GuardrailCheckEvidence = z.infer<
  typeof GuardrailCheckEvidenceSchema
>;

/** Decision returned by the Tool Pod guardrails evaluator. */
export const GuardrailCheckResponseSchema = z.object({
  /** Guardrails decision. */
  decision: GuardrailCheckDecisionSchema,
  /** Whether execution may continue. */
  allowed: z.boolean(),
  /** Modified text when decision is modify, or original text for allow/no-op. */
  transformed_text: z.string().nullish(),
  /** Policy IDs that triggered. */
  triggered_policy_ids: z.array(z.string()).default([]),
  /** Policy trigger evidence. */
  evidence: z.array(GuardrailCheckEvidenceSchema).default([]),
  /** Decision reason. */
  reason: z.string().nullish(),
  /** Evaluator-specific response metadata. */
  metadata: z.record(z.string(), JsonValueSchema).default({}),
});
export type GuardrailCheckResponse = z.infer<
  typeof GuardrailCheckResponseSchema
>;

// =============================================================================
// LLM Pod Execution (AER → ToolPod)
// =============================================================================

/**
 * Typed invoke_llm arguments forwarded through OE and tool pods.
 *
 * Python uses Pydantic aliases:
 *   - validation alias `stop` ↔ `stop_sequences`, serialization alias `stop`
 *   - pre-validator renames `kwargs` → `options`
 *
 * In Zod we replicate this with a `z.preprocess` that normalizes incoming
 * payloads into the canonical schema (using `stop_sequences` and
 * `options`). For outgoing serialization that needs the wire alias `stop`,
 * use `serializeInvokeLLMRequestArguments()`.
 */
export const InvokeLLMRequestArgumentsSchema = z.preprocess(
  (value) => {
    if (value === null || typeof value !== "object" || Array.isArray(value))
      return value;
    const obj = value as Record<string, unknown>;
    const normalized: Record<string, unknown> = { ...obj };
    // `stop` ↔ `stop_sequences` validation alias.
    if (!("stop_sequences" in normalized) && "stop" in normalized) {
      normalized["stop_sequences"] = normalized["stop"];
      delete normalized["stop"];
    }
    // `kwargs` → `options` pre-validator.
    if (!("options" in normalized) && "kwargs" in normalized) {
      normalized["options"] = normalized["kwargs"];
      delete normalized["kwargs"];
    }
    return normalized;
  },
  z.object({
    /** Model name (e.g. 'gpt-5.4-mini', 'gemini-3-flash-preview'). */
    model: z.string(),
    /** Conversation in sdk-core Message wire format. */
    messages: z.array(MessageSchema),
    /**
     * Stable identifier for the LLM instance registered via
     * `app.llm({ llmId })`; the tool pod resolves this id against its
     * named-LLM registry. Unnamed `app.llm()` calls register under the
     * sentinel id `"__default__"`. Defaults to `"__default__"` for backward
     * compatibility with callers that omit this field.
     */
    llm_id: z.string().default("__default__"),
    /** Stop sequences forwarded to the underlying LLM provider. */
    stop_sequences: z.array(z.string()).optional(),
    /** Serialized tool schemas for bind_tools. */
    tools: z.array(LLMToolSchemaValidator).optional(),
    /**
     * Forced tool selection forwarded to the tool pod's bind_tools call
     * (e.g. a function name from withStructuredOutput). LangChain translates
     * the value against the bound tools.
     */
    tool_choice: JsonValueSchema.optional(),
    /** Explicit provider/model invocation options (e.g. max_tokens). */
    options: LLMInvocationOptionsSchema.optional(),
    /** Whether OE should approve and route a real streaming invoke_llm call. */
    stream: z.boolean().default(false),
  }),
);
export type InvokeLLMRequestArguments = z.infer<
  typeof InvokeLLMRequestArgumentsSchema
>;

/**
 * Serialize InvokeLLMRequestArguments to the snake_case wire shape, the
 * equivalent of Python's `model_dump(by_alias=True)`:
 *  - `stop_sequences` → `stop` (Pydantic `serialization_alias="stop"`), and
 *  - each Message is dumped camel→snake via `serializeMessage`.
 *
 * Use this when sending to a Python consumer (OE / Tool Pod) that expects the
 * snake_case wire fields.
 */
export function serializeInvokeLLMRequestArguments(
  args: InvokeLLMRequestArguments,
): Record<string, unknown> {
  const { stop_sequences, messages, ...rest } = args;
  const out: Record<string, unknown> = { ...rest };
  out["messages"] = messages.map(serializeMessage);
  if (stop_sequences !== undefined) out["stop"] = stop_sequences;
  return out;
}

/**
 * Request to invoke LLM on a tool executor pod.
 *
 * Python pre-validator `_normalize_flat_payload`: if `arguments` is
 * missing but `execution_id` is present, treat the remaining keys as
 * the arguments payload. Same logic replicated via `z.preprocess`.
 */
export const LLMPodInvokeRequestSchema = z.preprocess(
  (value) => {
    if (value === null || typeof value !== "object" || Array.isArray(value))
      return value;
    const obj = value as Record<string, unknown>;
    if ("arguments" in obj || !("execution_id" in obj)) return value;
    // Routing metadata stays top-level rather than being folded into the
    // legacy flat payload's LLM arguments.
    const { execution_id, platform_trace_id, step_number, ...args } = obj;
    const out: Record<string, unknown> = {
      execution_id,
      platform_trace_id,
      arguments: args,
    };
    if (step_number !== undefined && step_number !== null) {
      out["step_number"] = step_number;
    }
    return out;
  },
  z.object({
    platform_trace_id: z.string().nullable().optional(),
    /** Execution identifier. */
    execution_id: z.string(),
    /** Typed invoke_llm arguments. */
    arguments: InvokeLLMRequestArgumentsSchema,
    /**
     * Step number of this LLM call, so the per-call interrupt
     * (POST /interrupt/call) can address exactly this invocation. Optional
     * for backward compatibility with older OE dispatchers.
     */
    step_number: z.number().int().nonnegative().optional(),
  }),
);
export type LLMPodInvokeRequest = z.infer<typeof LLMPodInvokeRequestSchema>;

/**
 * Normalized result from an LLM invocation.
 *
 * Provides a consistent shape for LLM responses regardless of the
 * underlying provider (OpenAI, Gemini, Anthropic, etc.). Implemented as
 * a class to mirror `LLMResponse` in agent-engine-sdk and to host the
 * `fromResponse` / `toResponse` / `extractUsage` conversion methods.
 */
export class LLMResult {
  readonly content: string;
  readonly toolCalls: LLMToolCall[];
  readonly usage?: LLMTokenUsage;
  readonly metadata: Record<string, JsonValue>;
  readonly id?: string;
  readonly name?: string;
  readonly additionalKwargs?: Record<string, JsonValue>;
  readonly responseMetadata?: Record<string, JsonValue>;

  constructor(data: {
    content?: string;
    toolCalls?: LLMToolCall[];
    usage?: LLMTokenUsage;
    metadata?: Record<string, JsonValue>;
    id?: string;
    name?: string;
    additionalKwargs?: Record<string, JsonValue>;
    responseMetadata?: Record<string, JsonValue>;
  }) {
    this.content = data.content ?? "";
    this.toolCalls = data.toolCalls ?? [];
    this.usage = data.usage;
    this.metadata = data.metadata ?? {};
    this.id = data.id;
    this.name = data.name;
    this.additionalKwargs = data.additionalKwargs;
    this.responseMetadata = data.responseMetadata;
  }

  /**
   * Extract a normalized result from any LLM response object.
   *
   * Accepts an sdk-core `LLMResponse` directly, or a duck-typed object
   * (e.g. raw provider response) by falling back to string-coerced
   * content extraction.
   */
  static fromResponse(response: unknown): LLMResult {
    if (response instanceof LLMResponse) {
      return new LLMResult({
        content: response.content,
        toolCalls: response.toolCalls ?? [],
        usage: response.usage,
        metadata: jsonSafeMetadata(response.metadata) ?? {},
        id: response.id,
        name: response.name,
        additionalKwargs: jsonSafeMetadata(response.additionalKwargs),
        responseMetadata: jsonSafeMetadata(response.responseMetadata),
      });
    }

    const isObj =
      response !== null &&
      response !== undefined &&
      typeof response === "object";
    const obj = (isObj ? response : {}) as Record<string, unknown>;
    // Mirror Python `response.content if hasattr(response, "content") else str(response)`:
    // if the response carries a `content` attribute we use it (coerced to string),
    // otherwise we fall back to String(response).
    const content =
      isObj && "content" in obj
        ? typeof obj["content"] === "string"
          ? (obj["content"] as string)
          : obj["content"] === undefined || obj["content"] === null
            ? ""
            : String(obj["content"])
        : String(response);
    const toolCallsRaw = obj["tool_calls"];
    const toolCalls = Array.isArray(toolCallsRaw)
      ? (toolCallsRaw as LLMToolCall[])
      : [];
    const usage = LLMResult.extractUsage(response);
    const metadata = jsonSafeMetadata(obj["metadata"]) ?? {};
    const responseMetadata = jsonSafeMetadata(obj["response_metadata"]);
    const additionalKwargs = jsonSafeMetadata(obj["additional_kwargs"]);
    const id =
      typeof obj["id"] === "string" ? (obj["id"] as string) : undefined;
    const name =
      typeof obj["name"] === "string" ? (obj["name"] as string) : undefined;

    return new LLMResult({
      content,
      toolCalls,
      usage,
      metadata,
      id,
      name,
      additionalKwargs,
      responseMetadata,
    });
  }

  /**
   * Extract token usage metadata from any response or chunk.
   *
   * Checks (in order): sdk-core `LLMResponse.usage`, `usage_metadata`,
   * `response_metadata.usage`, `response_metadata.token_usage`,
   * `usage`, `metadata.usage`.
   */
  static extractUsage(response: unknown): LLMTokenUsage | undefined {
    try {
      if (response instanceof LLMResponse && response.usage !== undefined) {
        return response.usage;
      }
      const obj = (response ?? {}) as Record<string, unknown>;

      const fromMeta = coerceTokenUsage(obj["usage_metadata"]);
      if (fromMeta !== undefined) return fromMeta;

      const respMeta = obj["response_metadata"];
      if (isRecord(respMeta)) {
        for (const key of ["usage", "token_usage"] as const) {
          const coerced = coerceTokenUsage(respMeta[key]);
          if (coerced !== undefined) return coerced;
        }
      }

      const fromUsageAttr = coerceTokenUsage(obj["usage"]);
      if (fromUsageAttr !== undefined) return fromUsageAttr;

      const metadata = obj["metadata"];
      if (isRecord(metadata)) {
        return coerceTokenUsage(metadata["usage"]);
      }
      return undefined;
    } catch {
      return undefined;
    }
  }

  /** Convert the normalized runner result to sdk-core's LLMResponse shape. */
  toResponse(): LLMResponse {
    let metadata = { ...this.metadata };
    if (this.usage !== undefined && Object.keys(metadata).length === 0) {
      // Mirror Python: dump usage (excluding undefined) into metadata when
      // metadata was otherwise empty.
      const dump: Record<string, JsonValue> = {};
      if (this.usage.inputTokens !== undefined)
        dump["input_tokens"] = this.usage.inputTokens;
      if (this.usage.outputTokens !== undefined)
        dump["output_tokens"] = this.usage.outputTokens;
      if (this.usage.promptTokens !== undefined)
        dump["prompt_tokens"] = this.usage.promptTokens;
      if (this.usage.completionTokens !== undefined)
        dump["completion_tokens"] = this.usage.completionTokens;
      if (this.usage.totalTokens !== undefined)
        dump["total_tokens"] = this.usage.totalTokens;
      if (this.usage.model !== undefined) dump["model"] = this.usage.model;
      metadata = dump;
    }
    return new LLMResponse({
      content: this.content,
      toolCalls: this.toolCalls.length > 0 ? this.toolCalls : undefined,
      metadata,
      usage: this.usage,
      id: this.id,
      name: this.name,
      additionalKwargs: this.additionalKwargs,
      responseMetadata: this.responseMetadata,
    });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function usageInt(value: unknown): number | undefined {
  if (typeof value === "boolean" || value == null) return undefined;
  if (typeof value === "number" && Number.isFinite(value) && value >= 0)
    return Math.trunc(value);
  return undefined;
}

function hasTokenCounts(usage: LLMTokenUsage): boolean {
  return (
    usage.inputTokens !== undefined ||
    usage.outputTokens !== undefined ||
    usage.promptTokens !== undefined ||
    usage.completionTokens !== undefined ||
    usage.totalTokens !== undefined
  );
}

function promptOf(usage: LLMTokenUsage): number | undefined {
  return usage.inputTokens ?? usage.promptTokens;
}

function completionOf(usage: LLMTokenUsage): number | undefined {
  return usage.outputTokens ?? usage.completionTokens;
}

/** Coerce a provider usage blob into LLMTokenUsage, or undefined. */
export function coerceTokenUsage(value: unknown): LLMTokenUsage | undefined {
  try {
    return coerceTokenUsageInner(value);
  } catch {
    return undefined;
  }
}

function readUsageInt(
  rec: Record<string, unknown>,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    try {
      const parsed = usageInt(rec[key]);
      if (parsed !== undefined) return parsed;
    } catch {
      continue;
    }
  }
  return undefined;
}

function coerceTokenUsageInner(value: unknown): LLMTokenUsage | undefined {
  if (value == null) return undefined;
  if (value instanceof LLMTokenUsage) {
    return hasTokenCounts(value) ? value : undefined;
  }
  if (typeof value !== "object" || Array.isArray(value)) return undefined;
  const rec = value as Record<string, unknown>;
  const data: ConstructorParameters<typeof LLMTokenUsage>[0] = {};
  const input = readUsageInt(rec, "input_tokens", "inputTokens");
  const output = readUsageInt(rec, "output_tokens", "outputTokens");
  const prompt = readUsageInt(rec, "prompt_tokens", "promptTokens");
  const completion = readUsageInt(rec, "completion_tokens", "completionTokens");
  const total = readUsageInt(rec, "total_tokens", "totalTokens");
  if (input !== undefined) data.input_tokens = input;
  if (output !== undefined) data.output_tokens = output;
  if (prompt !== undefined) data.prompt_tokens = prompt;
  if (completion !== undefined) data.completion_tokens = completion;
  if (total !== undefined) data.total_tokens = total;
  try {
    const model = rec["model"];
    if (typeof model === "string" && model) data.model = model;
  } catch {
    // ignore
  }
  if (
    data.input_tokens === undefined &&
    data.output_tokens === undefined &&
    data.prompt_tokens === undefined &&
    data.completion_tokens === undefined &&
    data.total_tokens === undefined
  ) {
    return undefined;
  }
  return new LLMTokenUsage(data);
}

/** Merge stream usage field-by-field; incoming non-null fields win. */
export function mergeTokenUsage(
  existing: LLMTokenUsage | undefined,
  incoming: unknown,
): LLMTokenUsage | undefined {
  const next =
    incoming instanceof LLMTokenUsage && hasTokenCounts(incoming)
      ? incoming
      : coerceTokenUsage(incoming);
  if (next === undefined) return existing;
  if (existing === undefined) return next;
  const incomingPrompt = promptOf(next);
  const incomingCompletion = completionOf(next);
  const prompt = incomingPrompt ?? promptOf(existing);
  const completion = incomingCompletion ?? completionOf(existing);
  const model = next.model ?? existing.model;
  let total = next.totalTokens ?? existing.totalTokens;
  if (total === undefined && prompt !== undefined && completion !== undefined) {
    total = prompt + completion;
  }
  return new LLMTokenUsage({
    input_tokens: prompt,
    output_tokens: completion,
    total_tokens: total,
    reasoning_tokens: next.reasoningTokens ?? existing.reasoningTokens,
    model,
  });
}

const JSON_SAFE_DEPTH = 8;
const USAGE_META_KEYS = new Set(["usage", "token_usage"]);

function usageJson(usage: LLMTokenUsage): Record<string, JsonValue> {
  return usage.toJSON() as Record<string, JsonValue>;
}

function jsonSafeValue(value: unknown, depth: number): JsonValue | undefined {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return value;
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (value instanceof LLMTokenUsage) return usageJson(value);
  if (depth >= JSON_SAFE_DEPTH) return undefined;
  if (Array.isArray(value)) {
    const items: JsonValue[] = [];
    for (const item of value) {
      if (item === null) {
        items.push(null);
        continue;
      }
      const safe = jsonSafeValue(item, depth + 1);
      if (safe !== undefined) items.push(safe);
    }
    return items;
  }
  if (isRecord(value)) return jsonSafeMetadata(value, depth);
  const coerced = coerceTokenUsage(value);
  return coerced !== undefined ? usageJson(coerced) : undefined;
}

/** Copy provider metadata into JSON-safe values, or undefined. */
export function jsonSafeMetadata(
  value: unknown,
  depth = 0,
): Record<string, JsonValue> | undefined {
  if (!isRecord(value) || Object.keys(value).length === 0) return undefined;
  if (depth >= JSON_SAFE_DEPTH) return undefined;
  const out: Record<string, JsonValue> = {};
  for (const [key, raw] of Object.entries(value)) {
    if (USAGE_META_KEYS.has(key)) {
      const coerced = coerceTokenUsage(raw);
      if (coerced !== undefined) out[key] = usageJson(coerced);
      continue;
    }
    if (raw === null) {
      out[key] = null;
      continue;
    }
    const safe = jsonSafeValue(raw, depth + 1);
    if (safe !== undefined) out[key] = safe;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/** Sum LangChain-style additive usage deltas into a cumulative snapshot. */
export function addTokenUsage(
  existing: LLMTokenUsage | undefined,
  incoming: unknown,
): LLMTokenUsage | undefined {
  const next =
    incoming instanceof LLMTokenUsage && hasTokenCounts(incoming)
      ? incoming
      : coerceTokenUsage(incoming);
  if (next === undefined) return existing;
  if (existing === undefined) return next;
  const prompt = (promptOf(existing) ?? 0) + (promptOf(next) ?? 0);
  const completion = (completionOf(existing) ?? 0) + (completionOf(next) ?? 0);
  const model = next.model ?? existing.model;
  let total: number | undefined;
  if (existing.totalTokens !== undefined || next.totalTokens !== undefined) {
    total = (existing.totalTokens ?? 0) + (next.totalTokens ?? 0);
  } else {
    total = prompt + completion;
  }
  const reasoning =
    existing.reasoningTokens !== undefined || next.reasoningTokens !== undefined
      ? (existing.reasoningTokens ?? 0) + (next.reasoningTokens ?? 0)
      : undefined;
  return new LLMTokenUsage({
    input_tokens: prompt,
    output_tokens: completion,
    total_tokens: total,
    reasoning_tokens: reasoning,
    model,
  });
}

function looksLikeUsageDelta(
  prior: LLMTokenUsage,
  incoming: LLMTokenUsage,
): boolean {
  const pairs: Array<[number | undefined, number | undefined]> = [
    [promptOf(prior), promptOf(incoming)],
    [completionOf(prior), completionOf(incoming)],
    [prior.totalTokens, incoming.totalTokens],
  ];
  return pairs.some(
    ([prev, nxt]) => prev !== undefined && nxt !== undefined && nxt < prev,
  );
}

/** Fold a stream chunk's usage into a running snapshot.
 *
 * Cumulative providers repeat growing totals (`10/1` then `10/2`);
 * last-wins merge keeps `10/2`. Additive providers zero-fill the
 * unchanged side (`18/1` then `0/4`); those deltas are summed.
 */
export function accumulateStreamUsage(
  existing: LLMTokenUsage | undefined,
  incoming: unknown,
): LLMTokenUsage | undefined {
  const next =
    incoming instanceof LLMTokenUsage && hasTokenCounts(incoming)
      ? incoming
      : coerceTokenUsage(incoming);
  if (next === undefined) return existing;
  if (existing === undefined) return next;
  if (looksLikeUsageDelta(existing, next)) return addTokenUsage(existing, next);
  return mergeTokenUsage(existing, next);
}

/** Runtime validation schema for LLMResult wire data. `content` is the only required field. */
export const LLMResultSchema = z.looseObject({
  content: z.string().default(""),
  tool_calls: z.array(z.unknown()).default([]),
  usage: LLMTokenUsageSchema.optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
  id: z.string().optional(),
  name: z.string().optional(),
  additional_kwargs: z.record(z.string(), JsonValueSchema).optional(),
  response_metadata: z.record(z.string(), JsonValueSchema).optional(),
});

/**
 * Response from LLM invocation on a tool executor pod.
 *
 * Python `_sync_usage_with_result` is a model_validator that mutates
 * `result.usage` ↔ `usage`. TS `LLMResponse.usage` is readonly, so
 * mutation isn't possible; callers should use
 * `normalizeLLMPodInvokeResponse()` after parsing to sync the two fields
 * by constructing a fresh LLMResponse when needed.
 */
export const LLMPodInvokeResponseSchema = z.object({
  /** Execution status: success, error. */
  status: z.string(),
  /** LLM result with content, tool_calls, and usage. */
  result: LLMResponseSchema.optional(),
  /** Error message if failed. */
  error: z.string().optional(),
  /** Machine-readable failure classification (e.g. llm_credential_rejected). */
  error_code: z.string().optional(),
  /** Hostname/pod name where LLM was executed. */
  pod_name: z.string(),
  /** Execution duration in milliseconds. */
  duration_ms: z.number(),
  /** Token usage (input_tokens, output_tokens, total_tokens). */
  usage: LLMTokenUsageSchema.optional(),
});
export type LLMPodInvokeResponse = z.infer<typeof LLMPodInvokeResponseSchema>;

/**
 * Sync `usage` and `result.usage` on a parsed LLMPodInvokeResponse,
 * mirroring Python's `_sync_usage_with_result` model_validator.
 *
 * - If `result.usage` is unset and `usage` is present, returns a new
 *   response whose result carries the top-level usage.
 * - If `usage` is unset and `result.usage` is present, lifts that
 *   `result.usage` to the top level.
 *
 * Returns the response (possibly with a freshly-built `result`/`usage`).
 */
export function normalizeLLMPodInvokeResponse(
  resp: LLMPodInvokeResponse,
): LLMPodInvokeResponse {
  const resultObj = resp.result as Record<string, unknown> | undefined;
  const resultHasUsage =
    resultObj !== undefined &&
    isRecord(resultObj["usage"]) &&
    resultObj["usage"] !== null;

  let nextResult = resp.result;
  let nextUsage = resp.usage;

  if (resultObj !== undefined && !resultHasUsage && resp.usage !== undefined) {
    nextResult = {
      ...resultObj,
      usage: resp.usage,
    } as unknown as typeof resp.result;
  }
  if (resp.usage === undefined && resultObj !== undefined && resultHasUsage) {
    nextUsage = resultObj["usage"] as typeof resp.usage;
  }

  return { ...resp, result: nextResult, usage: nextUsage };
}

/** SSE event emitted by a tool pod during invoke_llm streaming. */
export const LLMPodStreamEventSchema = z.object({
  /** Generated text delta. */
  content: z.string().optional(),
  /** Partial tool-call deltas. */
  tool_call_chunks: z.array(ToolCallChunkSchema).optional(),
  /** Complete tool calls for compatibility with older stream senders. */
  tool_calls: z.array(z.record(z.string(), z.unknown())).optional(),
  /** Provider message identifier. */
  id: z.string().optional(),
  /** Provider message name. */
  name: z.string().optional(),
  /** LangChain additional message kwargs. */
  additional_kwargs: z.record(z.string(), JsonValueSchema).optional(),
  /** LangChain response metadata. */
  response_metadata: z.record(z.string(), JsonValueSchema).optional(),
  /** Whether the stream is complete. */
  done: z.boolean().optional(),
  /** OE stopped this call on interrupt request; terminal, distinct from done/error. */
  interrupted: z.boolean().optional(),
  /** Error message if streaming failed. */
  error: z.string().optional(),
  /** Machine-readable failure classification (e.g. llm_credential_rejected). */
  error_code: z.string().optional(),
  /** When true with error, the same stream URL may be retried. */
  retryable: z.boolean().optional(),
  /** When set with retryable, wait this many milliseconds before retrying. */
  retry_after_ms: z.number().optional(),
  /** Hostname/pod name. */
  pod_name: z.string().optional(),
  /** Stream duration. */
  duration_ms: z.number().optional(),
  /** Final token usage. */
  usage: LLMTokenUsageSchema.optional(),
});
export type LLMPodStreamEvent = z.infer<typeof LLMPodStreamEventSchema>;

// =============================================================================
// Executor Callback (AER → OE)
// =============================================================================

/** Callback from AER to OE when execution completes or suspends. */
export const ExecutorCallbackRequestSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Status: COMPLETED, SUSPENDED, ERROR. */
  status: z.string(),
  /** Suspension generation expected by the originating dispatch. */
  suspend_generation: z
    .number()
    .int()
    .nonnegative()
    .nullable()
    .transform((value) => value ?? undefined)
    .optional(),
  /** Final result if completed. */
  result: z.unknown().optional(),
  /** Error message if failed. */
  error: z.string().optional(),
  /** Reason for suspension. */
  suspend_reason: z.string().optional(),
  suspend_context: z.record(z.string(), z.unknown()).optional(),
  interrupts: z.array(PendingInterruptSchema).optional(),
  resume_schema: z.record(z.string(), z.unknown()).optional(),
  /** Opaque framework-owned state returned by the selected adapter. */
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type ExecutorCallbackRequest = z.infer<
  typeof ExecutorCallbackRequestSchema
>;

// =============================================================================
// Resume (Client → OE)
// =============================================================================

/** Data provided by human reviewer when resuming a suspended execution. */
export const HumanReviewDataSchema = z.object({
  /** Review decision: 'approved', 'rejected', or custom value. */
  decision: z.string(),
  /** Optional notes from the reviewer. */
  reviewer_notes: z.string().optional(),
});
export type HumanReviewData = z.infer<typeof HumanReviewDataSchema>;

/** Request to resume a suspended execution. */
export const AgentResumeRequestSchema = z.object({
  /** Human review decision data (for human-in-the-loop resume). */
  human_review: HumanReviewDataSchema,
  /** Caller-provided custom headers. */
  custom_headers: z.record(z.string(), z.string()).optional(),
});
export type AgentResumeRequest = z.infer<typeof AgentResumeRequestSchema>;

/** Response from resuming an execution. */
export const AgentResumeResponseSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Execution status. */
  status: z.string(),
});
export type AgentResumeResponse = z.infer<typeof AgentResumeResponseSchema>;

// =============================================================================
// Execution Status Query
// =============================================================================

/** Response for execution status query. */
export const ExecutionStatusResponseSchema = z.object({
  /** Execution identifier. */
  execution_id: z.string(),
  /** Current status. */
  status: ExecutionStatusSchema,
  /** Result if completed. */
  result: z.unknown().optional(),
  /** Error if failed. */
  error: z.string().optional(),
  /** Reason if suspended. */
  suspend_reason: z.string().optional(),
  /** Context if suspended. */
  suspend_context: z.record(z.string(), z.unknown()).optional(),
  /** Creation timestamp. ISO strings are coerced to Date, matching Pydantic. */
  created_at: z.coerce.date(),
  /** Last update timestamp. ISO strings are coerced to Date, matching Pydantic. */
  updated_at: z.coerce.date(),
});
export type ExecutionStatusResponse = z.infer<
  typeof ExecutionStatusResponseSchema
>;

// =============================================================================
// Database Models
// =============================================================================

/** Execution record stored in TenantDB. */
export const ExecutionSchema = z.object({
  /** Unique execution identifier. */
  id: z.string(),
  /** Current status. */
  status: ExecutionStatusSchema,
  /** Original user message. */
  message: z.string(),
  /** Final result. */
  result: z.unknown().optional(),
  /** Error message. */
  error: z.string().optional(),
  /** Reason for suspension. */
  suspend_reason: z.string().optional(),
  /** Context for resume. */
  suspend_context: z.record(z.string(), z.unknown()).optional(),
  // Multi-tenant context
  /** Session ID for trace correlation. */
  session_id: z.string().optional(),
  /** Organization ID for multi-tenant isolation. */
  org_id: z.string().optional(),
  /** User ID for personalization. */
  user_id: z.string().optional(),
  /** Workspace identifier for cost tracking. */
  workspace_id: z.string().optional(),
  /** Project ID for project-level scoping. */
  project_id: z.string().optional(),
  // Per-agent AER routing
  /** AER HTTP endpoint for this agent; overrides default AER_URL. */
  aer_url: z.string().optional(),
  /** Tool HTTP endpoint for this agent; overrides default TOOL_URL. */
  tool_url: z.string().optional(),
  // Timestamps. ISO strings are coerced to Date, matching Pydantic and the
  // other datetime fields in this file (ExecutionStatusResponse).
  created_at: z.coerce.date().default(() => new Date()),
  updated_at: z.coerce.date().default(() => new Date()),
});
export type Execution = z.infer<typeof ExecutionSchema>;

/**
 * Normalize a value (Date, ISO string, or null/undefined) to a UTC Date.
 *
 * MongoDB stores naive datetimes in UTC; this helper rehydrates them as
 * proper Date instances so callers don't mix string and Date types.
 *
 * Mirrors Python `Execution._ensure_utc`. JS `Date` has no naive/aware
 * distinction (every Date is internally UTC ms-since-epoch), so the TS
 * version exercises the same "normalize input to a canonical Date" intent
 * by coercing ISO strings and passing Date instances through unchanged.
 */
export function ensureUtc(value: unknown): Date | undefined {
  if (value === null || value === undefined) return undefined;
  if (value instanceof Date) return value;
  if (typeof value === "string") {
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? undefined : d;
  }
  return undefined;
}

/**
 * Convert an Execution to a MongoDB persistence document.
 *
 * `updatedAt` defaults to now if omitted.
 */
export function executionToPersistenceDoc(
  exec: Execution,
  updatedAt?: Date,
): Record<string, unknown> {
  return {
    execution_id: exec.id,
    status: exec.status,
    message: exec.message,
    session_id: exec.session_id,
    user_id: exec.user_id,
    org_id: exec.org_id,
    workspace_id: exec.workspace_id,
    project_id: exec.project_id,
    result: exec.result,
    error: exec.error,
    suspend_reason: exec.suspend_reason,
    suspend_context: exec.suspend_context,
    updated_at: updatedAt ?? new Date(),
  };
}

/**
 * Rehydrate an Execution from a MongoDB document.
 *
 * Uses default values for Optional fields so documents created before
 * new fields were added still deserialize safely.
 */
export function executionFromPersistenceDoc(
  doc: Record<string, unknown>,
): Execution {
  const now = new Date();
  return ExecutionSchema.parse({
    id: doc["execution_id"],
    status: doc["status"],
    message: (doc["message"] as string | undefined) ?? "",
    result: doc["result"],
    error: doc["error"],
    suspend_reason: doc["suspend_reason"],
    suspend_context: doc["suspend_context"],
    session_id: doc["session_id"],
    org_id: doc["org_id"],
    user_id: doc["user_id"],
    workspace_id: doc["workspace_id"],
    project_id: doc["project_id"],
    created_at:
      ensureUtc(doc["created_at"]) ?? ensureUtc(doc["updated_at"]) ?? now,
    updated_at: ensureUtc(doc["updated_at"]) ?? now,
  });
}

/** Execution step record stored in TenantDB. */
export const ExecutionStepSchema = z.object({
  /** Unique step identifier. */
  id: z.string(),
  /** Parent execution ID. */
  execution_id: z.string(),
  /** Step sequence number. */
  step_number: z.number().int(),
  /** Tool or operation name. */
  tool_name: z.string(),
  /** Arguments passed. */
  arguments: z.record(z.string(), z.unknown()),
  /** Step status: pending, success, error. */
  status: z.string(),
  /** Step result. */
  result: z.unknown().optional(),
  /** Error message. */
  error: z.string().optional(),
  /** Execution duration. */
  duration_ms: z.number().optional(),
  /** ISO strings are coerced to Date, matching Pydantic. */
  timestamp: z.coerce.date().default(() => new Date()),
});
export type ExecutionStep = z.infer<typeof ExecutionStepSchema>;

/**
 * Reconstruct an ExecutionStep from an execution_logs MongoDB document.
 *
 * The execution_logs collection stores tool start/result events with
 * fields: execution_id, step_number, tool, inputs, status, output,
 * error, duration_ms, timestamp. This maps those fields back to the
 * ExecutionStep model for step-cache rehydration after OE restart.
 */
export function executionStepFromLogDoc(
  doc: Record<string, unknown>,
): ExecutionStep {
  return ExecutionStepSchema.parse({
    id: (doc["id"] as string | undefined) ?? crypto.randomUUID(),
    execution_id: doc["execution_id"],
    step_number: (doc["step_number"] as number | undefined) ?? 0,
    tool_name: (doc["tool"] as string | undefined) ?? "",
    arguments: (doc["inputs"] as Record<string, unknown> | undefined) ?? {},
    status: (doc["status"] as string | undefined) ?? "error",
    result: doc["output"],
    error: doc["error"],
    duration_ms: doc["duration_ms"],
    timestamp: ensureUtc(doc["timestamp"]) ?? new Date(),
  });
}

// =============================================================================
// Health Check
// =============================================================================

/** Health status of a component. */
export const HealthStatusSchema = z.enum(["healthy", "unhealthy", "degraded"]);
export type HealthStatus = z.infer<typeof HealthStatusSchema>;
export const HealthStatus = {
  HEALTHY: "healthy",
  UNHEALTHY: "unhealthy",
  DEGRADED: "degraded",
} as const;

/** Health check response. */
export const HealthResponseSchema = z.object({
  /** Health status. */
  status: HealthStatusSchema,
  /** Component name. */
  component: z.string(),
  /** Runtime mode. */
  mode: z.string(),
  /** Component version. */
  version: z.string().default("1.0.0"),
  /** Additional details. */
  details: z.record(z.string(), z.unknown()).default({}),
});
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

// =============================================================================
// Tool Definition
// =============================================================================

/** Definition of a registered tool. */
export const ToolDefinitionSchema = z.object({
  /** Tool name. */
  name: z.string(),
  /** Tool description. */
  description: z.string().default(""),
  /** Whether tool runs locally. */
  is_local: z.boolean().default(true),
  /** Credential provider type for delegated auth. */
  provider_type: z.string().nullish(),
  /** Requested OAuth scopes for delegated auth. */
  scopes: z.array(z.string()).default([]),
  /** JSON schema for parameters. */
  parameters: z.record(z.string(), z.unknown()).optional(),
});
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;

// =============================================================================
// AER Route Response Models
// =============================================================================

/**
 * Response from AER /execute endpoint.
 *
 * Returned on normal completion, HITL suspension, or cancellation of the
 * handler by a drain/teardown.
 */
export const AERExecuteResponseSchema = z.object({
  /** Execution outcome: 'completed', 'suspended', or 'cancelled'. */
  status: z.string(),
  /** Final agent response text (present when status is 'completed'). */
  result: z.string().optional(),
  /** Why the agent suspended (present when status is 'suspended'). */
  suspend_reason: z.string().optional(),
});
export type AERExecuteResponse = z.infer<typeof AERExecuteResponseSchema>;

/** Response from /tools endpoint listing registered tools. */
export const ToolsListResponseSchema = z.object({
  /** Registered tool definitions with name and metadata. */
  tools: z.array(z.record(z.string(), z.unknown())),
  /** Number of registered tools (included by Tool Pod). */
  count: z.number().int().optional(),
});
export type ToolsListResponse = z.infer<typeof ToolsListResponseSchema>;

// =============================================================================
// Streaming
// =============================================================================

/**
 * A chunk of streaming response from the agent.
 *
 * Used for real-time streaming of agent responses via SSE or gRPC.
 */
export const StreamChunkSchema = z.object({
  /**
   * Chunk type string. AER-emitted values are defined in
   * `server/chunk_types` ('text', 'done', 'error'). OE may inject
   * additional infrastructure types (e.g. 'metadata', 'tool_call',
   * 'tool_result') before forwarding to clients.
   */
  chunk_type: z.string(),
  /** Content of the chunk. */
  content: z.string().default(""),
  /** Optional metadata. */
  metadata: z.record(z.string(), z.string()).default({}),
  /** Error message if chunk_type is 'error'. */
  error: z.string().optional(),
  /** Machine-readable error code if chunk_type is 'error' (e.g. AGENT_UNAVAILABLE). */
  code: z.string().optional(),
  // Tool-specific fields
  /** Tool name for tool_call/tool_result chunks. */
  tool_name: z.string().optional(),
  /** Tool call ID. */
  tool_call_id: z.string().optional(),
  // Execution context
  /** Execution ID. */
  execution_id: z.string().optional(),
  /** Step number in execution. */
  step_number: z.number().int().optional(),
});
export type StreamChunk = z.infer<typeof StreamChunkSchema>;

// Re-export commonly-needed sdk-core types so consumers can import them
// from agent-engine-runner-shared without dropping down to agent-engine-sdk directly.
export type { Message, ToolCallChunk, JsonValue };
export {
  LLMResponse,
  LLMTokenUsage,
  LLMToolCall,
  LLMToolSchema,
  LLMInvocationOptions,
};
