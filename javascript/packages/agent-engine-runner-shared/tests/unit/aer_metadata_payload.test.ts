/**
 * Tests for metadata/payload fields on ExecuteRequest + ExecutorCallbackRequest
 * (models.ts) — MODEL LAYER ONLY.
 *
 * Mirrors runner-shared/tests/unit/test_aer_metadata_payload.py, the
 * TestExecuteRequestMetadataPayload + TestExecutorCallbackRequestMetadata
 * classes only.
 *
 * Ported cases:
 *   - ExecuteRequest accepts metadata + payload ← test_metadata_and_payload_accepted.
 *   - ExecuteRequest metadata/payload default when absent
 *     ← test_metadata_and_payload_default_to_none.
 *   - ExecuteRequest round-trips through parse(parse(...)) ← test_round_trip...
 *   - ExecuteRequest drops removed legacy fields (thread_id/checkpoint_id)
 *     ← test_old_payload_with_removed_fields_still_parses.
 *   - payload.message NOT promoted to top-level message at model level
 *     ← test_payload_only_message_not_promoted_at_model_level.
 *   - ExecutorCallbackRequest accepts metadata ← test_metadata_accepted.
 *   - ExecutorCallbackRequest metadata default ← test_metadata_defaults_to_none.
 *   - ExecutorCallbackRequest round-trips ← test_round_trip...
 *   - ExecutorCallbackRequest drops removed legacy checkpoint_id
 *     ← test_old_payload_with_removed_checkpoint_id_still_parses.
 *
 * Parked (handled by other TS tests, per task scope):
 *   - TestHandleExecuteMetadataResolution — session/message/checkpoint resolution
 *     is AER-layer; covered by the aer_chunk_ and hitl_interrupt tests.
 *   - TestReportCallbackMetadata / TestCustomEventForwarding /
 *     TestEndToEndBackwardCompat — AER integration, out of scope here.
 *
 * Python→TS adaptation notes:
 *   - Pydantic ExecuteRequest(...) / model_validate → Zod
 *     ExecuteRequestSchema.parse(...). model_dump round-trip → parse(parse(x)).
 *   - Pydantic's extra="ignore" (drops unknown keys) → Zod z.object() strips
 *     unknown keys by default; asserted by round-tripping data with legacy keys.
 *   - DIVERGENCE from the Python "defaults to None": absent optional fields are
 *     `undefined` in TS (z.optional), not `null`. Semantically the same "unset".
 *     Note: the AER-layer prompt described ExecutorCallbackRequest.metadata as
 *     "defaults {}", but the model uses z.record(...).optional(), so an omitted
 *     metadata parses to `undefined` — asserted below as the actual behavior.
 */

import { describe, test, expect } from "vitest";
import {
  ExecuteRequestSchema,
  ExecutorCallbackRequestSchema,
  type ExecuteRequest,
  type ExecutorCallbackRequest,
} from "../../src/index.js";

describe("ExecuteRequest metadata/payload", () => {
  test("accepts metadata and payload", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-1",
      message: "hello",
      platform_api_url: "http://oe:8000",
      metadata: {
        langgraph_branch_point: {
          thread_id: "t-1",
          checkpoint_id: "ckpt-1",
        },
      },
      payload: { context: "from-payload" },
    });
    expect(req.metadata).toEqual({
      langgraph_branch_point: {
        thread_id: "t-1",
        checkpoint_id: "ckpt-1",
      },
    });
    expect(req.payload).toEqual({ context: "from-payload" });
  });

  test("metadata and payload are unset (undefined) when omitted", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-2",
      message: "hello",
      platform_api_url: "http://oe:8000",
    });
    expect(req.metadata).toBeUndefined();
    expect(req.payload).toBeUndefined();
  });

  test("round-trips through a second parse unchanged", () => {
    const req: ExecuteRequest = ExecuteRequestSchema.parse({
      execution_id: "exec-3",
      message: "hello",
      platform_api_url: "http://oe:8000",
      metadata: { thread_id: "t-1" },
      payload: { context: "test" },
    });
    const restored = ExecuteRequestSchema.parse(req);
    expect(restored.metadata).toEqual(req.metadata);
    expect(restored.payload).toEqual(req.payload);
    expect(restored).toEqual(req);
  });

  test("drops removed legacy fields (thread_id/checkpoint_id) rather than failing", () => {
    // thread_id and checkpoint_id were removed from the model; an old OE that
    // still sends them must not break parsing. Zod strips unknown keys.
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-4",
      message: "hello",
      platform_api_url: "http://oe:8000",
      thread_id: "t-old",
      checkpoint_id: "ckpt-old",
    });
    expect(req.suspend_generation).toBeUndefined();
    expect(req.metadata).toBeUndefined();
    expect(req.payload).toBeUndefined();
    expect("thread_id" in req).toBe(false);
    expect("checkpoint_id" in req).toBe(false);
  });

  test("normalizes an explicit null suspension generation to unknown", () => {
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-null-generation",
      message: "hello",
      platform_api_url: "http://oe:8000",
      suspend_generation: null,
    });
    expect(req.suspend_generation).toBeUndefined();
  });

  test("payload.message is NOT promoted to top-level message at the model level", () => {
    // The model is a plain carrier; promoting payload.message to the top-level
    // message happens in the AER (resolveInvocationParams), not at parse time.
    const req = ExecuteRequestSchema.parse({
      execution_id: "exec-5",
      platform_api_url: "http://oe:8000",
      payload: { message: "from-payload-only" },
    });
    expect(req.message).toBe("");
    expect(req.payload).toEqual({ message: "from-payload-only" });
  });
});

describe("ExecutorCallbackRequest metadata", () => {
  test("accepts metadata", () => {
    const req = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-1",
      status: "SUSPENDED",
      suspend_generation: 3,
      metadata: { checkpoint_id: "ckpt-1", function_call_id: "fc-1" },
    });
    expect(req.suspend_generation).toBe(3);
    expect(req.metadata).toEqual({
      checkpoint_id: "ckpt-1",
      function_call_id: "fc-1",
    });
  });

  test("metadata is unset (undefined) when omitted", () => {
    const req = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-2",
      status: "COMPLETED",
    });
    expect(req.suspend_generation).toBeUndefined();
    expect(req.metadata).toBeUndefined();
  });

  test("normalizes an explicit null suspension generation to unknown", () => {
    const req = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-null-generation",
      status: "SUSPENDED",
      suspend_generation: null,
    });
    expect(req.suspend_generation).toBeUndefined();
  });

  test("accepts framework checkpoint data inside opaque metadata", () => {
    const req = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-checkpoint",
      status: "COMPLETED",
      metadata: {
        langgraph_checkpoint: {
          thread_id: "destination-thread",
          checkpoint_id: "destination-checkpoint",
        },
      },
    });

    expect(req.metadata).toEqual({
      langgraph_checkpoint: {
        thread_id: "destination-thread",
        checkpoint_id: "destination-checkpoint",
      },
    });
  });

  test("round-trips through a second parse unchanged", () => {
    const req: ExecutorCallbackRequest = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-3",
      status: "SUSPENDED",
      metadata: { checkpoint_id: "ckpt-1", function_call_id: "fc-1" },
    });
    const restored = ExecutorCallbackRequestSchema.parse(req);
    expect(restored.metadata).toEqual(req.metadata);
    expect(restored).toEqual(req);
  });

  test("drops removed legacy checkpoint_id rather than failing", () => {
    const req = ExecutorCallbackRequestSchema.parse({
      execution_id: "exec-4",
      status: "COMPLETED",
      result: "done",
      checkpoint_id: "ckpt-old",
    });
    expect(req.metadata).toBeUndefined();
    expect("checkpoint_id" in req).toBe(false);
  });
});
