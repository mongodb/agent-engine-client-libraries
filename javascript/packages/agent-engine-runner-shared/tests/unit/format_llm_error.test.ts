import { describe, expect, it } from "vitest";
import { formatLlmError } from "../../src/utils.js";

/** Mimics an OpenAI-compatible SDK's `APIError` carrying a structured body. */
class FakeAPIError extends Error {
  constructor(public readonly body: Record<string, unknown>) {
    // Template-literal stringification (like Python's `repr()`) never throws
    // on circular structures, unlike `JSON.stringify` — matches real SDK
    // error constructors that build their `.message` this way.
    super(`${body}`);
    this.name = "FakeAPIError";
  }
}

/** Mimics a provider-wrapper error chaining the raw API error via `cause`. */
class FakeProviderWrapperError extends Error {
  constructor(message: string, options: { cause: Error }) {
    super(message, options);
    this.name = "FakeProviderWrapperError";
  }
}

describe("formatLlmError", () => {
  it("falls back to the message for plain errors", () => {
    expect(formatLlmError(new Error("boom"))).toBe("boom");
  });

  it("decodes a nested JSON string in the cause's body", () => {
    const innerJson = JSON.stringify(
      {
        error: {
          code: 400,
          message: "API key not valid. Please pass a valid API key.",
          status: "INVALID_ARGUMENT",
        },
      },
      null,
      2,
    );
    const cause = new FakeAPIError({
      message: innerJson,
      status: "INVALID_ARGUMENT",
    });
    const exc = new FakeProviderWrapperError(
      "Error calling model 'gemini-2.5-flash' (Bad Request)",
      { cause },
    );

    const result = formatLlmError(exc);

    expect(result).not.toContain("\\n");
    expect(result).toContain('"code": 400');
    expect(result).toContain("API key not valid");
  });

  it("reads the body from the error itself, not only from cause", () => {
    const exc = new FakeAPIError({
      message: "plain message",
      status: "INVALID_ARGUMENT",
    });

    const result = formatLlmError(exc);

    expect(result).toBe(
      '{\n  "message": "plain message",\n  "status": "INVALID_ARGUMENT"\n}',
    );
  });

  it("leaves non-JSON-looking string values untouched", () => {
    const exc = new FakeAPIError({
      message: "not json",
      status: "INVALID_ARGUMENT",
    });

    expect(formatLlmError(exc)).toContain("not json");
  });

  it("falls back to the message without throwing on a circular body", () => {
    const body: Record<string, unknown> = { status: "INVALID_ARGUMENT" };
    body["self"] = body;
    const exc = new FakeAPIError(body);

    expect(() => formatLlmError(exc)).not.toThrow();
  });
});
