/**
 * Tests for the tool API error classifier.
 *
 * Mirrors Python's tests/unit/test_tool_api_error.py.
 *
 * Native fetch (Node ≥ 24, backed by Undici) does NOT throw on non-2xx
 * responses — it only rejects for transport/abort failures. So HTTP-status
 * classification relies on a thrown error that retains a numeric `status`,
 * `statusCode`, or `response.status` property (e.g. an Axios-shaped error
 * or a custom wrapper around a fetch Response).
 */

import { describe, test, expect } from "vitest";
import { classifyToolAPIError } from "../../src/tool_api_error.js";

function statusError(status: number, message = `HTTP ${status}`): Error {
  const e = new Error(message) as Error & { status: number };
  e.status = status;
  return e;
}

function axiosError(status: number, data?: Record<string, unknown>): Error {
  const e = new Error(`Request failed with status code ${status}`) as Error & {
    response: { status: number; data?: Record<string, unknown> };
  };
  e.response = { status, data };
  return e;
}

function nativeResponseError(
  status: number,
  body: Record<string, unknown>,
): Error {
  const e = new Error(`HTTP ${status}`) as Error & { response: Response };
  e.response = new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
  return e;
}

function fetchFailedError(cause: { code: string }): TypeError {
  const e = new TypeError("fetch failed") as TypeError & { cause: unknown };
  e.cause = cause;
  return e;
}

function timeoutError(): Error {
  const e = new Error("timed out") as Error & { name: string };
  e.name = "TimeoutError";
  return e;
}

describe("classifyToolAPIError", () => {
  test.each([
    [statusError(401), "AUTH_FAILED", false, 401],
    [statusError(403), "AUTH_FAILED", false, 403],
    [statusError(429), "RATE_LIMITED", true, 429],
    [statusError(503), "PROVIDER_UNAVAILABLE", true, 503],
    [statusError(418), "UNKNOWN", false, 418],
    [statusError(500), "UNKNOWN", false, 500],
    [axiosError(429), "RATE_LIMITED", true, 429],
    [
      nativeResponseError(503, { message: "down" }),
      "PROVIDER_UNAVAILABLE",
      true,
      503,
    ],
    [timeoutError(), "TIMEOUT", true, null],
    [fetchFailedError({ code: "ETIMEDOUT" }), "TIMEOUT", true, null],
    [
      fetchFailedError({ code: "UND_ERR_HEADERS_TIMEOUT" }),
      "TIMEOUT",
      true,
      null,
    ],
    [fetchFailedError({ code: "UND_ERR_BODY_TIMEOUT" }), "TIMEOUT", true, null],
    [
      fetchFailedError({ code: "UND_ERR_CONNECT_TIMEOUT" }),
      "TIMEOUT",
      true,
      null,
    ],
    [
      fetchFailedError({ code: "ECONNREFUSED" }),
      "CONNECTION_ERROR",
      true,
      null,
    ],
    [fetchFailedError({ code: "ECONNRESET" }), "CONNECTION_ERROR", true, null],
    [fetchFailedError({ code: "ENOTFOUND" }), "CONNECTION_ERROR", true, null],
    [fetchFailedError({ code: "EAI_AGAIN" }), "CONNECTION_ERROR", true, null],
    [
      fetchFailedError({ code: "UND_ERR_SOCKET" }),
      "CONNECTION_ERROR",
      true,
      null,
    ],
    [
      fetchFailedError({ code: "EHOSTUNREACH" }),
      "CONNECTION_ERROR",
      true,
      null,
    ],
    [fetchFailedError({ code: "ENETUNREACH" }), "CONNECTION_ERROR", true, null],
  ] as const)(
    "classifies %#",
    async (exc, classification, retryable, status) => {
      const result = await classifyToolAPIError(exc, "atlas");
      expect(result?.toolApiError.classification).toBe(classification);
      expect(result?.toolApiError.retryable).toBe(retryable);
      expect(result?.toolApiError.http_status).toBe(status);
      expect(result?.toolApiError.provider_type).toBe("atlas");
    },
  );

  test("walks cause chain to find nested status", async () => {
    const result = await classifyToolAPIError(
      new Error("wrapper", { cause: statusError(429) }),
    );
    expect(result?.toolApiError.classification).toBe("RATE_LIMITED");
  });

  test("classifies top-level statusCode", async () => {
    const e = new Error("HTTP 401") as Error & { statusCode: number };
    e.statusCode = 401;
    const result = await classifyToolAPIError(e);
    expect(result?.toolApiError.classification).toBe("AUTH_FAILED");
  });

  test.each([
    (() => {
      const e = new Error("aborted") as Error & { name: string };
      e.name = "AbortError";
      return e;
    })(),
    new TypeError("fetch failed"),
    new Error("HTTP 429"),
    new Error("something broke"),
  ])("does not classify %#", async (exc) => {
    expect(await classifyToolAPIError(exc)).toBeUndefined();
  });
});

describe("classifyToolAPIError — envelope", () => {
  test("extracts errorCode and reason from Axios data and native Response JSON", async () => {
    const axiosResult = await classifyToolAPIError(
      axiosError(429, { errorCode: "rate_exceeded", reason: "Too many calls" }),
    );
    expect(axiosResult?.toolApiError.error_code).toBe("rate_exceeded");
    expect(axiosResult?.toolApiError.reason).toBe("Too many calls");

    const nativeResult = await classifyToolAPIError(
      nativeResponseError(429, {
        errorCode: "quota_exceeded",
        reason: "Daily limit reached",
      }),
    );
    expect(nativeResult?.toolApiError.error_code).toBe("quota_exceeded");
    expect(nativeResult?.toolApiError.reason).toBe("Daily limit reached");
  });

  test("bounds error_code to 128 and reason to 256", async () => {
    const result = await classifyToolAPIError(
      axiosError(429, { errorCode: "x".repeat(200), reason: "y".repeat(300) }),
    );
    expect(result?.toolApiError.error_code?.length).toBe(128);
    expect(result?.toolApiError.reason?.length).toBe(256);
  });

  test("redacts home directory paths in error_code and reason", async () => {
    const home = process.env["HOME"] ?? "";
    if (!home) return;
    const result = await classifyToolAPIError(
      axiosError(429, {
        errorCode: `err_at_${home}/secrets`,
        reason: `config in ${home}/.env`,
      }),
    );
    expect(result?.toolApiError.error_code).not.toContain(home);
    expect(result?.toolApiError.reason).not.toContain(home);
  });

  test("drops native Response envelope when body exceeds 4096 bytes", async () => {
    const result = await classifyToolAPIError(
      nativeResponseError(429, {
        errorCode: "rate_exceeded",
        pad: "x".repeat(5000),
      }),
    );
    expect(result?.toolApiError.error_code).toBeNull();
    expect(result?.toolApiError.reason).toBeNull();
  });

  test("drops error_code and reason that contain URLs", async () => {
    const result = await classifyToolAPIError(
      axiosError(429, {
        errorCode: "see https://evil.example/x",
        reason: "docs at http://provider.example/help",
      }),
    );
    expect(result?.toolApiError.error_code).toBeNull();
    expect(result?.toolApiError.reason).toBeNull();
  });

  test("returns null error_code/reason when the body has no recognized field", async () => {
    const result = await classifyToolAPIError(axiosError(429, { foo: "bar" }));
    expect(result?.toolApiError.error_code).toBeNull();
    expect(result?.toolApiError.reason).toBeNull();
    expect(result?.message).toBe(
      "External API call failed: HTTP 429 RATE_LIMITED",
    );
  });

  test("surfaces Jira errorMessages and the errors map", async () => {
    const fromMessages = await classifyToolAPIError(
      nativeResponseError(400, {
        errorMessages: [
          "The value 'OpenJira' does not exist for the field 'project'.",
        ],
        errors: {},
      }),
      "jiradc",
    );
    expect(fromMessages?.toolApiError.reason).toBe(
      "The value 'OpenJira' does not exist for the field 'project'.",
    );
    expect(fromMessages?.message).toContain(
      "The value 'OpenJira' does not exist for the field 'project'.",
    );

    const fromErrors = await classifyToolAPIError(
      nativeResponseError(400, {
        errorMessages: [],
        errors: { project: "Project 'X' is unknown" },
      }),
    );
    expect(fromErrors?.toolApiError.reason).toBe("Project 'X' is unknown");
  });

  test("recognizes generic and nested envelopes", async () => {
    const generic = await classifyToolAPIError(
      nativeResponseError(422, { message: "Validation failed" }),
    );
    expect(generic?.toolApiError.reason).toBe("Validation failed");

    const nested = await classifyToolAPIError(
      nativeResponseError(400, {
        error: { message: "Invalid model", code: "invalid_request" },
      }),
      "openai",
    );
    expect(nested?.toolApiError.reason).toBe("Invalid model");
    expect(nested?.toolApiError.error_code).toBe("invalid_request");
    expect(nested?.message).toContain("Invalid model");
  });

  test("drops control characters from surfaced text", async () => {
    const result = await classifyToolAPIError(
      nativeResponseError(400, { message: "bad\u009bmessage\u0007" }),
    );
    expect(result?.toolApiError.reason).toBe("badmessage");
    expect(result?.message).not.toContain("\u009b");
    expect(result?.message).not.toContain("\u0007");
  });

  test("rejects a URL reconstructed by control-character removal", async () => {
    const result = await classifyToolAPIError(
      nativeResponseError(400, { message: "https:/\u0000/evil.example" }),
    );
    expect(result?.toolApiError.reason).toBeNull();
    expect(result?.message).toBe("External API call failed: HTTP 400 UNKNOWN");
  });

  test("ignores an oversized Axios-shaped body", async () => {
    const result = await classifyToolAPIError(
      axiosError(422, { message: "Validation failed", pad: "x".repeat(5000) }),
    );
    expect(result?.toolApiError.reason).toBeNull();
    expect(result?.message).toBe("External API call failed: HTTP 422 UNKNOWN");
  });

  test("redacts an unlabeled credential passed as request-local material", async () => {
    const result = await classifyToolAPIError(
      nativeResponseError(401, { message: "Rejected credential sk-live-123" }),
      "svc",
      ["sk-live-123"],
    );
    expect(result?.toolApiError.reason).toBe("Rejected credential <redacted>");
    expect(result?.message).not.toContain("sk-live-123");
  });

  test("drops URL-shaped text while keeping double-slash prose", async () => {
    for (const message of [
      "see https://internal.example/private",
      "target //api.internal.example/private",
    ]) {
      const url = await classifyToolAPIError(
        nativeResponseError(400, { message }),
      );
      expect(url?.toolApiError.reason).toBeNull();
      expect(url?.message).toBe("External API call failed: HTTP 400 UNKNOWN");
    }

    const prose = await classifyToolAPIError(
      nativeResponseError(400, { message: "retry // later" }),
    );
    expect(prose?.toolApiError.reason).toBe("retry // later");
  });
});

describe("classifyToolAPIError — security", () => {
  test("raw response body does not appear in the classified output", async () => {
    const result = await classifyToolAPIError(
      nativeResponseError(429, {
        errorCode: "err_code",
        reason: "rate limited",
        detail: "secret_internal_details",
        parameters: { api_key: "sk-1234567890" },
      }),
    );
    const json = JSON.stringify(result);
    expect(json).not.toContain("secret_internal_details");
    expect(json).not.toContain("sk-1234567890");
    expect(json).not.toContain("api_key");
    expect(json).not.toContain('"detail"');
    expect(json).not.toContain('"parameters"');
  });

  test("URLs in error messages and response headers are not extracted", async () => {
    const urlErr = new Error(
      "Request to https://api.example.com/v1/data failed",
    ) as Error & { status: number };
    urlErr.status = 429;
    expect(JSON.stringify(await classifyToolAPIError(urlErr))).not.toContain(
      "api.example.com",
    );

    const e = axiosError(429, { errorCode: "x", reason: "y" });
    (e.response as Record<string, unknown>).headers = {
      authorization: "Bearer secret-token",
      "x-api-key": "key123",
    };
    const json = JSON.stringify(await classifyToolAPIError(e));
    expect(json).not.toContain("secret-token");
    expect(json).not.toContain("key123");
  });

  test("cause chain stops at max depth 5 (cycle-safe)", async () => {
    const a = new Error("a") as Error & { cause: unknown };
    const b = new Error("b") as Error & { cause: unknown };
    const c = new Error("c") as Error & { cause: unknown };
    a.cause = b;
    b.cause = c;
    c.cause = a;
    expect(await classifyToolAPIError(a)).toBeUndefined();
  });

  test("safe message format", async () => {
    expect(
      (await classifyToolAPIError(statusError(429), "atlas"))?.message,
    ).toBe("atlas API call failed: HTTP 429 RATE_LIMITED");
    expect((await classifyToolAPIError(statusError(503)))?.message).toBe(
      "External API call failed: HTTP 503 PROVIDER_UNAVAILABLE",
    );
  });
});
