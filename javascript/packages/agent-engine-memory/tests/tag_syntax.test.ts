import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { MemoryClientError } from "../src/errors.js";
import {
  RESERVED_TYPE_NAMES,
  validateMemoryType,
  validateTagSyntax,
} from "../src/tag_syntax.js";

describe("validateMemoryType", () => {
  it("accepts a custom name", () => {
    expect(() => validateMemoryType("support_tickets")).not.toThrow();
  });

  it.each(["semantic", "episodic", "taxonomic", "procedural"])(
    "rejects built-in %s with the server's message",
    (name) => {
      expect(() => validateMemoryType(name)).toThrow(
        `'${name}' is a built-in memory type and is not accepted by this operation`,
      );
    },
  );

  it("rejects empty", () => {
    expect(() => validateMemoryType("")).toThrow(
      "memory_type must be a non-empty string",
    );
  });

  it("mirrors the reserved set", () => {
    expect(RESERVED_TYPE_NAMES.has("stm")).toBe(true);
    expect(RESERVED_TYPE_NAMES.size).toBe(10);
  });

  it("pins the reserved set to the repo fixture", (ctx) => {
    // The hardcoded list would otherwise drift silently from the platform's
    // canonical names. Matches the Python SDK's equivalent guard. A missing
    // fixture reports as skipped rather than passing, so a moved path or a
    // downstream checkout without pkg/ cannot mask real drift.
    const fixturePath = resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../../../..",
      "pkg/memoryconfig/reserved_names.json",
    );
    if (!existsSync(fixturePath)) {
      ctx.skip();
      return;
    }
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      reserved_type_names: string[];
      name_pattern: string;
    };
    expect(new Set(fixture.reserved_type_names)).toEqual(
      new Set(RESERVED_TYPE_NAMES),
    );
    expect(fixture.name_pattern).toBe("^[a-z][a-z0-9_]{0,63}$");
  });
});

describe("validateTagSyntax", () => {
  it("accepts scalars and one-level nesting", () => {
    expect(() =>
      validateTagSyntax({
        queue: "billing",
        priority: 3,
        score: 1.5,
        open: true,
        profile: { location: "nyc" },
        "profile.tier": "gold",
      }),
    ).not.toThrow();
  });

  it("rejects an empty path", () => {
    expect(() => validateTagSyntax({ "": "x" })).toThrow(
      "tag '' must have a non-empty path",
    );
  });

  it("rejects an empty segment", () => {
    expect(() => validateTagSyntax({ "a.": "x" })).toThrow(
      "tag 'a.' must have a non-empty path",
    );
  });

  it("rejects two-level dotted keys", () => {
    expect(() => validateTagSyntax({ "a.b.c": "x" })).toThrow(
      "tag 'a.b.c': nesting is at most one level deep",
    );
  });

  it("rejects two-level nested objects", () => {
    expect(() => validateTagSyntax({ a: { b: { c: "x" } } })).toThrow(
      "tag 'a': nesting is at most one level deep",
    );
  });

  it("rejects an empty group", () => {
    expect(() => validateTagSyntax({ a: {} })).toThrow(
      "tag group 'a' must contain at least one key",
    );
  });

  it("rejects non-scalar values", () => {
    expect(() => validateTagSyntax({ a: ["x"] })).toThrow(
      "tag 'a' must be a non-empty string, number, or boolean",
    );
    expect(() => validateTagSyntax({ a: null })).toThrow(
      "tag 'a' must be a non-empty string, number, or boolean",
    );
  });

  it("rejects empty string values", () => {
    expect(() => validateTagSyntax({ a: "  " })).toThrow(
      "tag 'a' must have a non-empty string value",
    );
  });

  it("throws MemoryClientError instances", () => {
    expect(() => validateTagSyntax({ "": "x" })).toThrow(MemoryClientError);
  });
});

describe("validateMemoryType name pattern", () => {
  it.each(["tickets", "support_tickets", "a", "t1", "a".repeat(64)])(
    "accepts %s",
    (name) => {
      expect(() => validateMemoryType(name)).not.toThrow();
    },
  );

  it.each([
    "a b",
    "a/b",
    "Tickets",
    "1tickets",
    "a-b",
    "a".repeat(65),
    "tickets.sub",
  ])("rejects %s, which the platform could never declare", (name) => {
    expect(() => validateMemoryType(name)).toThrow(
      `custom type name '${name}' must match ^[a-z][a-z0-9_]{0,63}$`,
    );
  });
});

describe("non-finite number tags", () => {
  // JSON.stringify rewrites these to null, which would store a value the
  // server never agreed to, so the client rejects them instead.
  it.each([NaN, Infinity, -Infinity])("rejects %s", (value) => {
    expect(() => validateTagSyntax({ score: value })).toThrow(
      "tag 'score' must be a finite number",
    );
  });

  it.each([1.5, 0, -3.25, 1e308])("accepts finite %s", (value) => {
    expect(() => validateTagSyntax({ score: value })).not.toThrow();
  });

  it("rejects a non-finite value inside a nested group", () => {
    expect(() => validateTagSyntax({ profile: { score: NaN } })).toThrow(
      "tag 'profile.score' must be a finite number",
    );
  });

  it("forwards large integers for the server to judge", () => {
    expect(() =>
      validateTagSyntax({ count: Number.MAX_SAFE_INTEGER }),
    ).not.toThrow();
  });
});
