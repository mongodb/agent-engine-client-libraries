/**
 * Unit tests for resolveCorsPolicy.
 *
 * Mirrors Python's tests/unit/test_cors.py — the two runner runtimes are
 * parallel implementations, so the twins must stay behaviourally identical.
 *
 * The behaviour under test is a security default: runner servers register
 * credential-bearing routes with no authentication, so the previous `*`
 * default meant any page a developer visited could drive tool execution
 * once `agentengine dev` published the ports.
 */

import { describe, test, expect } from "vitest";
import { resolveCorsPolicy } from "../../src/server/cors.js";

describe("resolveCorsPolicy", () => {
  test("defaults to deny-all when unset", () => {
    expect(resolveCorsPolicy(undefined)).toEqual({
      origin: false,
      credentials: false,
    });
  });

  test("denies all for an empty or whitespace-only value", () => {
    expect(resolveCorsPolicy("")).toEqual({
      origin: false,
      credentials: false,
    });
    expect(resolveCorsPolicy("   ")).toEqual({
      origin: false,
      credentials: false,
    });
    expect(resolveCorsPolicy(",,")).toEqual({
      origin: false,
      credentials: false,
    });
  });

  test("allows an explicit origin with credentials", () => {
    expect(resolveCorsPolicy("https://a.com")).toEqual({
      origin: ["https://a.com"],
      credentials: true,
    });
  });

  test("trims surrounding whitespace so a spaced entry still matches", () => {
    // Regression: the previous implementation compared the untrimmed
    // " https://b.com", so this entry silently never matched and an
    // operator could believe they had restricted origins when they had not.
    expect(resolveCorsPolicy("https://a.com, https://b.com")).toEqual({
      origin: ["https://a.com", "https://b.com"],
      credentials: true,
    });
  });

  test("never pairs a wildcard with credentials", () => {
    const policy = resolveCorsPolicy("*");
    expect(policy.origin).toBe("*");
    expect(policy.credentials).toBe(false);
  });

  test("wildcard anywhere in the list wins and still drops credentials", () => {
    const policy = resolveCorsPolicy("https://a.com,*");
    expect(policy.origin).toBe("*");
    expect(policy.credentials).toBe(false);
  });

  test("drops malformed entries rather than passing them through", () => {
    expect(resolveCorsPolicy("https://a.com,not-a-url")).toEqual({
      origin: ["https://a.com"],
      credentials: true,
    });
  });

  test("rejects an origin carrying a path", () => {
    expect(resolveCorsPolicy("https://a.com/app")).toEqual({
      origin: false,
      credentials: false,
    });
  });

  test("falls back to deny-all when every entry is malformed", () => {
    expect(resolveCorsPolicy("nonsense,also-nonsense")).toEqual({
      origin: false,
      credentials: false,
    });
  });

  test("preserves an explicit port", () => {
    expect(resolveCorsPolicy("http://localhost:3005")).toEqual({
      origin: ["http://localhost:3005"],
      credentials: true,
    });
  });

  test("accepts the literal null origin", () => {
    expect(resolveCorsPolicy("null")).toEqual({
      origin: ["null"],
      credentials: true,
    });
  });

  test("normalizes a scheme's default port away", () => {
    expect(resolveCorsPolicy("https://a.com:443,http://b.com:80")).toEqual({
      origin: ["https://a.com", "http://b.com"],
      credentials: true,
    });
  });

  test("lowercases scheme and host", () => {
    expect(resolveCorsPolicy("HTTPS://A.COM")).toEqual({
      origin: ["https://a.com"],
      credentials: true,
    });
  });

  test("collapses entries that normalize to the same origin", () => {
    expect(resolveCorsPolicy("https://a.com,https://A.com:443")).toEqual({
      origin: ["https://a.com"],
      credentials: true,
    });
  });

  test("accepts a bracketed IPv6 host with a port", () => {
    expect(resolveCorsPolicy("http://[::1]:3005")).toEqual({
      origin: ["http://[::1]:3005"],
      credentials: true,
    });
  });

  test.each([
    "https://*.a.com", // a wildcard host never matches a real Origin header
    "https://u:p@a.com", // userinfo
    "https://a.com?q=1", // query
    "ftp://a.com", // non-http(s) scheme
    "file://",
  ])("rejects %s", (raw) => {
    expect(resolveCorsPolicy(raw)).toEqual({
      origin: false,
      credentials: false,
    });
  });
});
