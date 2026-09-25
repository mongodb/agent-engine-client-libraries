/**
 * Unit tests for resolveOeUrl.
 *
 * Mirrors Python's tests/unit/test_oe_url.py — the two runner runtimes are
 * parallel implementations, so the twins must stay behaviourally identical.
 *
 * The behaviour under test is a trust boundary inside the agent execution
 * loop: whatever this returns becomes the base for every outbound approval,
 * stream and result call, and the runner treats those responses as
 * authoritative results and policy decisions.
 */

import { describe, test, expect } from "vitest";
import { resolveOeUrl } from "../../src/server/oe_url.js";

describe("resolveOeUrl", () => {
  test("prefers the runner's configured OE_URL over the request value", () => {
    expect(
      resolveOeUrl("http://attacker.example", {
        OE_URL: "http://oe:8000",
      } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  test("discards a request value even when it looks plausible", () => {
    expect(
      resolveOeUrl("http://oe.other-tenant:8000", {
        OE_URL: "http://oe:8000",
      } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  test("returns the configured value unchanged when they agree", () => {
    expect(
      resolveOeUrl("http://oe:8000", {
        OE_URL: "http://oe:8000",
      } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  test("trims surrounding whitespace on the configured value", () => {
    expect(
      resolveOeUrl("http://attacker.example", {
        OE_URL: "  http://oe:8000 ",
      } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  test("falls back to the request value when OE_URL is unset", () => {
    // Local `agentengine dev` and unit tests have no deploy-time environment.
    expect(resolveOeUrl("http://oe:8000", {} as NodeJS.ProcessEnv)).toBe(
      "http://oe:8000",
    );
    expect(
      resolveOeUrl("http://oe:8000", { OE_URL: "" } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
    expect(
      resolveOeUrl("http://oe:8000", { OE_URL: "   " } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  test("ignores an empty request value when OE_URL is configured", () => {
    expect(
      resolveOeUrl("", { OE_URL: "http://oe:8000" } as NodeJS.ProcessEnv),
    ).toBe("http://oe:8000");
  });

  // Owner-callback fallback regression: the owner-callback fallback added owner-URL validation
  // in a sibling module (owner_url.ts). resolveOeUrl's own behaviour must stay
  // byte-identical — the deploy-time OE_URL still wins over any request value,
  // and no owner-shaped normalization leaks into this path.
  describe("Owner-callback fallback regression", () => {
    test("configured OE_URL still wins over a replica-shaped request value", () => {
      // A request value shaped exactly like a valid headless replica of the
      // configured service must not be preferred over the trusted OE_URL.
      expect(
        resolveOeUrl("https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443", {
          OE_URL: "https://oe.ns.svc.cluster.local:8443",
        } as NodeJS.ProcessEnv),
      ).toBe("https://oe.ns.svc.cluster.local:8443");
    });

    test("unset OE_URL still returns the request value verbatim (no owner-style validation)", () => {
      const requested =
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443";
      expect(resolveOeUrl(requested, {} as NodeJS.ProcessEnv)).toBe(requested);
    });
  });
});
