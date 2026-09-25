/**
 * Tests for agent-engine-runner-shared wire models — token/cost/workspace_id field
 * propagation and execution persistence.
 *
 * Mirrors Python tests/unit/test_models.py. Of the 38 Python tests:
 *   - 21 ported here (this file).
 *   - 17 deliberately dropped because they target `.to_log()` /
 *     `ExecutionLog` / `NodeExecutionLog`, all dropped post-Go-OE
 *     migration. See AGENTS.md "Phase 14 parked unit-test files".
 *
 * Parity notes:
 *   - Python `Execution(...)` (Pydantic class) → TS `ExecutionSchema.parse({...})`
 *     (Zod-inferred type, not a constructor).
 *   - Python `execution.to_persistence_doc()` / `Execution.from_persistence_doc(doc)` /
 *     `ExecutionStep.from_log_doc(doc)` are instance/class methods.
 *     TS exposes free functions: `executionToPersistenceDoc`,
 *     `executionFromPersistenceDoc`, `executionStepFromLogDoc`.
 *   - Python `Optional[X]` (None | X) ↔ TS `.optional()` (undefined | X).
 *     Where Python tests pass `None`, TS uses `undefined` / omits the key.
 *   - JS `Date` has no naive/aware distinction (every Date is intrinsically
 *     UTC ms-since-epoch). `TestEnsureUtc::test_naive_datetime_gets_utc`
 *     and `::test_aware_datetime_unchanged` are adapted to TS-native
 *     normalization paths (ISO string → Date, Date → Date) rather than
 *     the Python naive→aware path which has no JS analog.
 */

import { describe, test, expect } from "vitest";
import {
  ensureUtc,
  ExecuteRequestSchema,
  ExecutionSchema,
  ExecutionStatus,
  ExecutionStatusSchema,
  executionFromPersistenceDoc,
  executionStepFromLogDoc,
  executionToPersistenceDoc,
  InvokeLLMRequestArgumentsSchema,
  InvokeRequestSchema,
  serializeInvokeLLMRequestArguments,
  ToolExecuteRequestSchema,
  ToolPodExecuteRequestSchema,
  ToolResultRequestSchema,
} from "../../src/index.js";

// ---------------------------------------------------------------------------
// TestExecutionPersistence — Execution.to_persistence_doc() with workspace_id
// ---------------------------------------------------------------------------

describe("executionToPersistenceDoc — workspace_id", () => {
  test("workspace_id appears in the persistence document", () => {
    // test_workspace_id_in_persistence_doc
    const execution = ExecutionSchema.parse({
      id: "exec-1",
      status: ExecutionStatus.RUNNING,
      message: "test",
      workspace_id: "my-agent",
    });
    const doc = executionToPersistenceDoc(execution);
    expect(doc["workspace_id"]).toBe("my-agent");
  });

  test("absent workspace_id appears as undefined in persistence doc", () => {
    // test_workspace_id_none_in_persistence_doc
    // Python: doc["workspace_id"] is None. TS uses undefined for parity.
    const execution = ExecutionSchema.parse({
      id: "exec-1",
      status: ExecutionStatus.RUNNING,
      message: "test",
    });
    const doc = executionToPersistenceDoc(execution);
    expect(doc["workspace_id"]).toBeUndefined();
  });

  test("persistence doc contains all expected keys", () => {
    // test_persistence_doc_has_expected_keys
    const execution = ExecutionSchema.parse({
      id: "exec-1",
      status: ExecutionStatus.RUNNING,
      message: "test",
      workspace_id: "agent-1",
      org_id: "org-1",
      session_id: "sess-1",
    });
    const doc = executionToPersistenceDoc(execution);
    const expectedKeys = new Set([
      "execution_id",
      "status",
      "message",
      "session_id",
      "user_id",
      "org_id",
      "workspace_id",
      "project_id",
      "result",
      "error",
      "suspend_reason",
      "suspend_context",
      "updated_at",
    ]);
    expect(new Set(Object.keys(doc))).toEqual(expectedKeys);
  });
});

// ---------------------------------------------------------------------------
// TestPydanticSerialization — round-trips for new fields
// (ExecutionLog tests dropped: target removed type — see AGENTS.md scope)
// ---------------------------------------------------------------------------

describe("Schema serialization round-trips", () => {
  test("ToolResultRequest survives a parse → serialize → parse round-trip", () => {
    // test_tool_result_request_round_trip
    const original = ToolResultRequestSchema.parse({
      execution_id: "exec-1",
      step_number: 1,
      tool_name: "invoke_llm",
      status: "success",
      result: "Hello",
      duration_ms: 100.0,
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      model: "gpt-4o",
      workspace_id: "my-agent",
    });
    const dumped = JSON.parse(JSON.stringify(original));
    const restored = ToolResultRequestSchema.parse(dumped);
    expect(restored.prompt_tokens).toBe(100);
    expect(restored.completion_tokens).toBe(50);
    expect(restored.total_tokens).toBe(150);
    expect(restored.model).toBe("gpt-4o");
    expect(restored.workspace_id).toBe("my-agent");
  });

  test("ToolResultRequest payloads without new fields default to undefined", () => {
    // test_backward_compat_missing_new_fields
    const oldPayload = {
      execution_id: "exec-1",
      step_number: 1,
      tool_name: "invoke_llm",
      status: "success",
      result: "Hello",
      duration_ms: 100.0,
      // No token/cost/workspace_id fields
    };
    const req = ToolResultRequestSchema.parse(oldPayload);
    expect(req.prompt_tokens).toBeUndefined();
    expect(req.completion_tokens).toBeUndefined();
    expect(req.total_tokens).toBeUndefined();
    expect(req.model).toBeUndefined();
    expect(req.workspace_id).toBeUndefined();
  });

  test("ToolExecuteRequest redaction fields survive a round-trip", () => {
    const original = ToolExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "charge_customer",
      arguments: { amount: 100 },
      step_number: 1,
      redact_fields: ["card_number"],
    });
    const dumped = JSON.parse(JSON.stringify(original));
    expect(dumped.redact_fields).toEqual(["card_number"]);
    const restored = ToolExecuteRequestSchema.parse(dumped);
    expect(restored.redact_fields).toEqual(["card_number"]);
  });

  test("ToolExecuteRequest defaults to local execution", () => {
    const req = ToolExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "get_weather",
      arguments: { city: "Tokyo" },
      step_number: 1,
    });
    expect(req.is_local).toBe(true);
  });

  test("InvokeLLMRequestArguments accepts and serializes tool_choice", () => {
    const args = InvokeLLMRequestArgumentsSchema.parse({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      tool_choice: "Brief",
    });
    expect(args.tool_choice).toBe("Brief");

    const wire = serializeInvokeLLMRequestArguments(args);
    expect(wire["tool_choice"]).toBe("Brief");
  });

  test("InvokeLLMRequestArguments omits tool_choice when absent", () => {
    const args = InvokeLLMRequestArgumentsSchema.parse({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(args.tool_choice).toBeUndefined();

    const wire = serializeInvokeLLMRequestArguments(args);
    expect(wire).not.toHaveProperty("tool_choice");
  });

  test("InvokeRequest accepts and serializes workspace_id", () => {
    // test_invoke_request_workspace_id
    const req = InvokeRequestSchema.parse({
      message: "hello",
      workspace_id: "test-agent",
    });
    const data = JSON.parse(JSON.stringify(req));
    expect(data["workspace_id"]).toBe("test-agent");
  });

  test("ExecuteRequest accepts and serializes workspace_id", () => {
    // test_execute_request_workspace_id
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://localhost:8080",
      workspace_id: "test-agent",
    });
    const data = JSON.parse(JSON.stringify(req));
    expect(data["workspace_id"]).toBe("test-agent");
  });

  test("ExecuteRequest defaults message to empty string when omitted", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      platform_api_url: "http://localhost:8080",
    });
    expect(req.message).toBe("");
  });

  test("ExecuteRequest does not promote payload.message at the model level", () => {
    // Python parity: test_payload_only_message_not_promoted_at_model_level.
    // The model is a plain carrier; promoting payload["message"] to the
    // top-level message happens in the AER's resolveInvocationParams, not at
    // parse time. This keeps both values visible so the AER can reject an
    // ambiguous request (message in both places) with a clean 400.
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      platform_api_url: "http://localhost:8080",
      payload: { message: "from payload" },
    });
    expect(req.message).toBe("");
    expect(req.payload).toEqual({ message: "from payload" });
  });

  test("ExecuteRequest keeps both top-level and payload message verbatim", () => {
    // Both survive parse so the AER boundary can detect the ambiguity.
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "top level",
      platform_api_url: "http://localhost:8080",
      payload: { message: "from payload" },
    });
    expect(req.message).toBe("top level");
    expect(req.payload?.["message"]).toBe("from payload");
  });

  test("ExecuteRequest strips legacy scalar resume messages", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      platform_api_url: "http://localhost:8080",
      resume: true,
      resume_message: "approve",
    });
    expect(req).not.toHaveProperty("resume_message");
  });

  test("ExecuteRequest retains payload and metadata", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      platform_api_url: "http://localhost:8080",
      payload: { message: "hi", extra: 1 },
      metadata: { thread_id: "t-1", checkpoint_id: "c-1" },
    });
    expect(req.payload).toEqual({ message: "hi", extra: 1 });
    expect(req.metadata).toEqual({ thread_id: "t-1", checkpoint_id: "c-1" });
  });
});

// ---------------------------------------------------------------------------
// TestOwnerCallbackUrlFields — optional owner-URL fallback fields
//
// These request fields drive owner-first callback routing. These tests prove
// the wire-compatible shape. Mirrors Python
// TestOwnerCallbackUrlFields in test_models.py.
// ---------------------------------------------------------------------------

describe("owner callback URL fields", () => {
  test("ExecuteRequest without platform_api_owner_url still parses", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://localhost:8080",
    });
    expect(req.platform_api_owner_url).toBeUndefined();
  });

  test("ExecuteRequest with platform_api_owner_url round-trips", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://localhost:8080",
      platform_api_owner_url: "http://oe-replica-1.internal:8080",
    });
    expect(req.platform_api_owner_url).toBe(
      "http://oe-replica-1.internal:8080",
    );
    const dumped = JSON.parse(JSON.stringify(req));
    expect(dumped.platform_api_owner_url).toBe(
      "http://oe-replica-1.internal:8080",
    );
    const restored = ExecuteRequestSchema.parse(dumped);
    expect(restored.platform_api_owner_url).toBe(
      "http://oe-replica-1.internal:8080",
    );
  });

  test("ExecuteRequest accepts explicit null platform_api_owner_url", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://localhost:8080",
      platform_api_owner_url: null,
    });
    expect(req.platform_api_owner_url).toBeNull();
    const dumped = JSON.parse(JSON.stringify(req));
    expect(dumped.platform_api_owner_url).toBeNull();
  });

  test("ExecuteRequest ignores unknown extra fields", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://localhost:8080",
      some_future_field: "unexpected",
    });
    expect(req).not.toHaveProperty("some_future_field");
  });

  test("ToolPodExecuteRequest without oe_owner_url still parses", () => {
    const req = ToolPodExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "get_weather",
      arguments: { city: "Tokyo" },
      session_id: "sess-1",
    });
    expect(req.oe_owner_url).toBeUndefined();
  });

  test("ToolPodExecuteRequest with oe_owner_url round-trips", () => {
    const req = ToolPodExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "get_weather",
      arguments: { city: "Tokyo" },
      session_id: "sess-1",
      oe_owner_url: "http://oe-replica-1.internal:8080",
    });
    expect(req.oe_owner_url).toBe("http://oe-replica-1.internal:8080");
    const dumped = JSON.parse(JSON.stringify(req));
    expect(dumped.oe_owner_url).toBe("http://oe-replica-1.internal:8080");
    const restored = ToolPodExecuteRequestSchema.parse(dumped);
    expect(restored.oe_owner_url).toBe("http://oe-replica-1.internal:8080");
  });

  test("ToolPodExecuteRequest accepts explicit null oe_owner_url", () => {
    const req = ToolPodExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "get_weather",
      arguments: { city: "Tokyo" },
      session_id: "sess-1",
      oe_owner_url: null,
    });
    expect(req.oe_owner_url).toBeNull();
    const dumped = JSON.parse(JSON.stringify(req));
    expect(dumped.oe_owner_url).toBeNull();
  });

  test("ToolPodExecuteRequest ignores unknown extra fields", () => {
    const req = ToolPodExecuteRequestSchema.parse({
      execution_id: "exec-1",
      tool_name: "get_weather",
      arguments: { city: "Tokyo" },
      session_id: "sess-1",
      some_future_field: "unexpected",
    });
    expect(req).not.toHaveProperty("some_future_field");
  });

  test("ToolPodExecuteRequest rejects oversized arguments", () => {
    expect(() =>
      ToolPodExecuteRequestSchema.parse({
        execution_id: "exec-1",
        tool_name: "write",
        arguments: { content: "x".repeat(16 * 1024 * 1024) },
        session_id: "sess-1",
      }),
    ).toThrow(/tool arguments exceed 16777216 bytes/);
  });

  test("ToolPodExecuteRequest rejects non-JSON arguments", () => {
    expect(() =>
      ToolPodExecuteRequestSchema.parse({
        execution_id: "exec-1",
        tool_name: "write",
        arguments: { content: undefined },
        session_id: "sess-1",
      }),
    ).toThrow();
  });
});

// ---------------------------------------------------------------------------
// TestExecutionCheckpointPersistence — resume state is opaque to the platform
// ---------------------------------------------------------------------------

describe("executionToPersistenceDoc — checkpoint fields", () => {
  test("persistence document has no checkpoint fields", () => {
    // test_to_persistence_doc_has_no_checkpoint_fields
    const execution = ExecutionSchema.parse({
      id: "exec-1",
      status: ExecutionStatus.SUSPENDED,
      message: "test",
    });
    const doc = executionToPersistenceDoc(execution);
    expect(doc).not.toHaveProperty("checkpoint_id");
    expect(doc).not.toHaveProperty("state_snapshot");
  });
});

// ---------------------------------------------------------------------------
// TestExecutionFromPersistenceDoc — rehydration
// ---------------------------------------------------------------------------

function makeDoc(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  // Mirrors Python TestExecutionFromPersistenceDoc._make_doc. Python's
  // `result: None` / `error: None` fields are omitted (TS-native Optional
  // semantics — see top-of-file parity notes). checkpoint_id/state_snapshot
  // are legacy fields kept here to prove old documents still deserialize.
  return {
    execution_id: "exec-1",
    status: "suspended",
    message: "I was in an accident",
    session_id: "sess-1",
    user_id: "user-1",
    thread_id: "thread-1",
    org_id: "org-1",
    workspace_id: "insurance-agent",
    suspend_reason: "awaiting_human_review",
    suspend_context: { claim_id: "C-123", claim_amount: 15000 },
    checkpoint_id: "ckpt-abc123",
    state_snapshot: { key: "value" },
    created_at: new Date("2025-05-20T12:00:00Z"),
    updated_at: new Date("2025-05-20T12:01:00Z"),
    ...overrides,
  };
}

describe("executionFromPersistenceDoc", () => {
  test("serialize then deserialize preserves all fields", () => {
    // test_round_trip
    const original = ExecutionSchema.parse({
      id: "exec-1",
      status: ExecutionStatus.SUSPENDED,
      message: "test",
      session_id: "sess-1",
      user_id: "user-1",
      thread_id: "thread-1",
      org_id: "org-1",
      workspace_id: "my-agent",
      suspend_reason: "awaiting_human_review",
      suspend_context: { claim_id: "C-123" },
    });
    const doc = executionToPersistenceDoc(original);
    // to_persistence_doc omits created_at (matches Python). Inject it so
    // from_persistence_doc has both timestamps.
    doc["created_at"] = original.created_at;
    const restored = executionFromPersistenceDoc(doc);

    expect(restored.id).toBe(original.id);
    expect(restored.status).toBe(original.status);
    expect(restored.message).toBe(original.message);
    expect(restored.session_id).toBe(original.session_id);
    expect(restored.org_id).toBe(original.org_id);
    expect(restored.suspend_reason).toBe(original.suspend_reason);
    expect(restored.suspend_context).toEqual(original.suspend_context);
    expect(restored.workspace_id).toBe(original.workspace_id);
  });

  test("old documents missing optional fields deserialize safely", () => {
    // test_missing_optional_fields
    const doc = {
      execution_id: "exec-old",
      status: "suspended",
      message: "old message",
    };
    const execution = executionFromPersistenceDoc(doc);
    expect(execution.id).toBe("exec-old");
    expect(execution.status).toBe(ExecutionStatus.SUSPENDED);
    expect(execution.workspace_id).toBeUndefined();
    expect(execution.session_id).toBeUndefined();
  });

  test("legacy checkpoint_id / state_snapshot fields are ignored", () => {
    // test_legacy_checkpoint_fields_ignored — documents persisted before
    // these fields were removed from the Execution model still deserialize.
    const execution = executionFromPersistenceDoc(makeDoc());
    expect(execution.id).toBe("exec-1");
    expect(execution).not.toHaveProperty("checkpoint_id");
    expect(execution).not.toHaveProperty("state_snapshot");
  });

  test("status string converts to ExecutionStatus enum value", () => {
    // test_enum_conversion
    const statuses = [
      "pending",
      "running",
      "suspended",
      "completed",
      "error",
      "resuming",
      "cancelled",
    ] as const;
    for (const statusStr of statuses) {
      const doc = makeDoc({ status: statusStr });
      const execution = executionFromPersistenceDoc(doc);
      expect(execution.status).toBe(statusStr);
    }
  });

  test("parses cancelled execution status", () => {
    expect(ExecutionStatusSchema.parse("cancelled")).toBe("cancelled");
    expect(ExecutionStatus.CANCELLED).toBe("cancelled");
  });

  test("document with all fields deserializes completely", () => {
    // test_all_fields_populated
    const doc = makeDoc();
    const execution = executionFromPersistenceDoc(doc);
    expect(execution.id).toBe("exec-1");
    expect(execution.status).toBe(ExecutionStatus.SUSPENDED);
    expect(execution.message).toBe("I was in an accident");
    expect(execution.session_id).toBe("sess-1");
    expect(execution.user_id).toBe("user-1");
    expect(execution.org_id).toBe("org-1");
    expect(execution.workspace_id).toBe("insurance-agent");
    expect(execution.suspend_reason).toBe("awaiting_human_review");
    expect(execution.suspend_context).toEqual({
      claim_id: "C-123",
      claim_amount: 15000,
    });
    expect(execution.created_at).toEqual(new Date("2025-05-20T12:00:00Z"));
    expect(execution.updated_at).toEqual(new Date("2025-05-20T12:01:00Z"));
  });
});

// ---------------------------------------------------------------------------
// TestExecutionStepFromLogDoc
// ---------------------------------------------------------------------------

describe("executionStepFromLogDoc", () => {
  test("reconstructs ExecutionStep from an execution_logs document", () => {
    // test_from_log_doc
    const ts = new Date("2025-05-20T12:00:00Z");
    const doc = {
      id: "log-1",
      execution_id: "exec-1",
      step_number: 3,
      tool: "file_claim",
      inputs: { policy_number: "P-001", amount: 15000 },
      status: "success",
      output: { claim_id: "C-123" },
      duration_ms: 250.5,
      timestamp: ts,
    };
    const step = executionStepFromLogDoc(doc);
    expect(step.id).toBe("log-1");
    expect(step.execution_id).toBe("exec-1");
    expect(step.step_number).toBe(3);
    expect(step.tool_name).toBe("file_claim");
    expect(step.arguments).toEqual({
      policy_number: "P-001",
      amount: 15000,
    });
    expect(step.status).toBe("success");
    expect(step.result).toEqual({ claim_id: "C-123" });
    expect(step.duration_ms).toBe(250.5);
    expect(step.timestamp).toEqual(ts);
  });

  test("handles documents with minimal fields", () => {
    // test_from_log_doc_missing_optional_fields
    const doc = {
      execution_id: "exec-1",
      step_number: 1,
      tool: "invoke_llm",
      status: "success",
    };
    const step = executionStepFromLogDoc(doc);
    expect(step.execution_id).toBe("exec-1");
    expect(step.tool_name).toBe("invoke_llm");
    expect(step.arguments).toEqual({});
    expect(step.result).toBeUndefined();
    expect(step.error).toBeUndefined();
    expect(step.duration_ms).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// TestEnsureUtc — timezone/normalization
// ---------------------------------------------------------------------------

describe("ensureUtc", () => {
  test("coerces an ISO string input to a Date", () => {
    // Adapted from Python test_naive_datetime_gets_utc.
    // JS Date has no naive/aware distinction; the TS analog of "input lacks
    // UTC tzinfo" is an un-parsed ISO string. ensureUtc parses it to a Date.
    const iso = "2025-05-20T12:00:00Z";
    const result = ensureUtc(iso);
    expect(result).toBeInstanceOf(Date);
    expect(result?.getUTCFullYear()).toBe(2025);
    expect(result?.getUTCHours()).toBe(12);
  });

  test("returns a Date instance unchanged", () => {
    // Adapted from Python test_aware_datetime_unchanged.
    // JS Date is intrinsically UTC; "already-aware" maps to "already a Date".
    const date = new Date("2025-05-20T12:00:00Z");
    const result = ensureUtc(date);
    expect(result).toBe(date);
  });

  test("executionFromPersistenceDoc normalizes ISO-string timestamps to Date", () => {
    // Adapted from test_from_persistence_doc_normalizes_naive_timestamps.
    // Python passes naive datetimes; TS exercises the same coercion path
    // via ISO strings, which is what a JSON-round-tripped doc carries.
    const doc = {
      execution_id: "exec-1",
      status: "suspended",
      created_at: "2025-05-20T12:00:00Z",
      updated_at: "2025-05-20T12:01:00Z",
    };
    const execution = executionFromPersistenceDoc(doc);
    expect(execution.created_at).toBeInstanceOf(Date);
    expect(execution.updated_at).toBeInstanceOf(Date);
  });

  test("created_at falls back to updated_at when missing", () => {
    // test_from_persistence_doc_created_at_falls_back_to_updated_at
    const updated = new Date("2025-05-20T12:01:00Z");
    const doc = {
      execution_id: "exec-1",
      status: "running",
      updated_at: updated,
    };
    const execution = executionFromPersistenceDoc(doc);
    expect(execution.created_at).toEqual(updated);
  });

  test("executionStepFromLogDoc normalizes ISO-string timestamp to Date", () => {
    // Adapted from test_from_log_doc_normalizes_naive_timestamp.
    const doc = {
      execution_id: "exec-1",
      step_number: 1,
      tool: "test",
      status: "success",
      timestamp: "2025-05-20T12:00:00Z",
    };
    const step = executionStepFromLogDoc(doc);
    expect(step.timestamp).toBeInstanceOf(Date);
  });
});
