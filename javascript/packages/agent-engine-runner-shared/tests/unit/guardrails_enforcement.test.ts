/**
 * Tests for guardrail enforcement surfacing:
 *   - ToolExecuteResponse assembling guardrail_meta from flat wire fields
 *   - SecureLLMProxy block-substitute + require_review (approve/deny/no-handler)
 *   - PolicyDeniedException carrying guardrail_meta on a genuine block
 *
 * Mirrors the guardrail portions of Python's test_secure_llm_proxy.py /
 * test_models.py. fetch is stubbed to return the OE /tool/execute response.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { type Message } from "@mongodb-js/agent-engine-sdk";
import {
  SecureLLMProxy,
  PolicyDeniedException,
  ToolExecuteResponseSchema,
  registerSuspendHandler,
  resetHooks,
} from "../../src/index.js";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const baseMessage: Message = { role: "user", content: "Hello" };

function makeProxy(): SecureLLMProxy {
  return new SecureLLMProxy({
    oeUrl: "http://localhost:8080",
    executionId: "exec-123",
    llmId: "primary",
  });
}

beforeEach(() => {
  vi.unstubAllGlobals();
  resetHooks();
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetHooks();
});

describe("ToolExecuteResponse guardrail_meta assembly", () => {
  test("assembles guardrail_meta from flat guardrail_id/category", () => {
    const parsed = ToolExecuteResponseSchema.parse({
      proceed: false,
      reason: "blocked",
      guardrail_id: "gr-1",
      guardrail_category: "pii",
    });
    expect(parsed.guardrail_meta).toEqual({
      guardrail_id: "gr-1",
      guardrail_category: "pii",
    });
  });

  test("preserves an explicit guardrail_meta object", () => {
    const parsed = ToolExecuteResponseSchema.parse({
      proceed: false,
      guardrail_meta: { guardrail_id: "explicit", guardrail_category: "tox" },
    });
    expect(parsed.guardrail_meta).toEqual({
      guardrail_id: "explicit",
      guardrail_category: "tox",
    });
  });

  test("leaves guardrail_meta unset when flat fields are absent", () => {
    const parsed = ToolExecuteResponseSchema.parse({ proceed: true });
    expect(parsed.guardrail_meta ?? null).toBeNull();
  });
});

describe("SecureLLMProxy guardrail block-substitute", () => {
  test("yields the substitute string as content instead of throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: false,
          reason: "blocked by policy",
          result: "[BLOCKED]",
        }),
      ),
    );

    const response = await makeProxy().invoke([baseMessage]);
    expect(response.content).toBe("[BLOCKED]");
  });
});

describe("SecureLLMProxy genuine block", () => {
  test("throws PolicyDeniedException carrying guardrail_meta", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: false,
          reason: "blocked",
          guardrail_id: "gr-9",
          guardrail_category: "secrets",
        }),
      ),
    );

    await expect(makeProxy().invoke([baseMessage])).rejects.toMatchObject({
      name: "PolicyDeniedException",
      guardrailMeta: { guardrail_id: "gr-9", guardrail_category: "secrets" },
    });
  });
});

describe("SecureLLMProxy require_review", () => {
  const requireReviewResponse = {
    proceed: false,
    status: "require_review",
    reason: "needs human review",
    guardrail_id: "gr-5",
    guardrail_category: "policy",
  };

  test("suspends with a payload carrying guardrail_meta, then yields approved content", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(requireReviewResponse)),
    );

    let captured: Record<string, unknown> | null = null;
    registerSuspendHandler((payload) => {
      captured = payload;
      return {
        guardrail_review: {
          decision: "approve",
          pending_llm_content: { content: "the original answer" },
        },
      };
    });

    const response = await makeProxy().invoke([baseMessage]);
    expect(response.content).toBe("the original answer");
    expect(captured).toMatchObject({
      suspend_reason: "guardrail_require_review",
      allowed_decisions: ["approve", "deny"],
      guardrail_meta: { guardrail_id: "gr-5", guardrail_category: "policy" },
    });
  });

  test("denies by throwing PolicyDeniedException", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(requireReviewResponse)),
    );
    registerSuspendHandler(() => ({
      guardrail_review: { decision: "deny" },
    }));

    await expect(makeProxy().invoke([baseMessage])).rejects.toBeInstanceOf(
      PolicyDeniedException,
    );
  });

  test("throws PolicyDeniedException when no suspend handler is registered", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(requireReviewResponse)),
    );

    await expect(makeProxy().invoke([baseMessage])).rejects.toBeInstanceOf(
      PolicyDeniedException,
    );
  });
});
