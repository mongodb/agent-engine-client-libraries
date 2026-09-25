import { describe, expect, test } from "vitest";

import { quotePathSegment } from "../../src/http_path.js";

describe("quotePathSegment", () => {
  test("encodes special characters", () => {
    expect(quotePathSegment("session 1")).toBe("session%201");
  });

  test("keeps Python-parity unreserved characters readable", () => {
    expect(quotePathSegment("org-project_1.ws")).toBe("org-project_1.ws");
  });

  test("rejects traversal and separators", () => {
    for (const bad of [
      "",
      ".",
      "..",
      "a/b",
      "a\\b",
      "%2e%2e",
      "%252e%252e",
      "a?b",
      "a#b",
    ]) {
      expect(() => quotePathSegment(bad)).toThrow(/path segment is invalid/);
    }
  });

  test("rejects malformed percent escapes", () => {
    expect(() => quotePathSegment("100%zz")).toThrow(/path segment is invalid/);
  });
});
