/**
 * Unit tests for resolveOwnerUrl.
 *
 * Mirrors Python's tests/unit/test_owner_url.py — the two runner runtimes are
 * parallel implementations, so the twins must stay behaviourally identical.
 *
 * resolveOwnerUrl guards a trust boundary: it decides whether a request-supplied
 * owner URL may become a callback target. A forged value must be discarded
 * (return null) so the caller falls back to the already-trusted service URL —
 * never an error, never an attacker-chosen host.
 */

import { beforeEach, describe, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

import { resolveOwnerUrl } from "../../src/server/owner_url.js";

const SERVICE = "https://oe.ns.svc.cluster.local:8443";
const VALID_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443";

beforeEach(() => {
  for (const method of Object.values(mocks.logger)) {
    method.mockClear();
  }
});

describe("resolveOwnerUrl accepts valid replicas", () => {
  test("accepts a valid headless replica", () => {
    expect(resolveOwnerUrl(VALID_OWNER, SERVICE)).toBe(VALID_OWNER);
  });

  test("accepts scheme-default ports expressed implicitly on both sides", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local",
        "https://oe.ns.svc.cluster.local",
      ),
    ).toBe("https://10-1-2-3.oe-headless.ns.svc.cluster.local");
  });

  test("canonicalizes an explicit default port away", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local:443",
        "https://oe.ns.svc.cluster.local:443",
      ),
    ).toBe("https://10-1-2-3.oe-headless.ns.svc.cluster.local");
  });

  test("preserves an explicit non-default port in canonical form", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443",
        "https://oe.ns.svc.cluster.local:8443",
      ),
    ).toBe("https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443");
  });

  test("strips a trailing slash from the accepted owner URL", () => {
    expect(resolveOwnerUrl(VALID_OWNER + "/", SERVICE)).toBe(VALID_OWNER);
  });

  test.each([null, undefined, "", "   "])(
    "returns null for absent owner (%p)",
    (owner) => {
      expect(resolveOwnerUrl(owner, SERVICE)).toBeNull();
    },
  );
});

describe("resolveOwnerUrl rejects forged shapes (degrade to null)", () => {
  test("rejects an external host", () => {
    expect(
      resolveOwnerUrl("https://attacker.example:8443", SERVICE),
    ).toBeNull();
  });

  test("rejects an external host mimicking the service suffix", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local.evil.com:8443",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects a wrong port", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local:9999",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects a wrong scheme", () => {
    expect(
      resolveOwnerUrl(
        "http://10-1-2-3.oe-headless.ns.svc.cluster.local:8443",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects a host without the -headless service segment", () => {
    expect(
      resolveOwnerUrl("https://10-1-2-3.oe.ns.svc.cluster.local:8443", SERVICE),
    ).toBeNull();
  });

  test("rejects the headless service host without a replica label", () => {
    expect(
      resolveOwnerUrl("https://oe-headless.ns.svc.cluster.local:8443", SERVICE),
    ).toBeNull();
  });

  test("rejects extra leading labels", () => {
    expect(
      resolveOwnerUrl(
        "https://a.10-1-2-3.oe-headless.ns.svc.cluster.local:8443",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects embedded userinfo", () => {
    expect(
      resolveOwnerUrl(
        "https://evil@10-1-2-3.oe-headless.ns.svc.cluster.local:8443",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects a path", () => {
    expect(resolveOwnerUrl(VALID_OWNER + "/evil", SERVICE)).toBeNull();
  });

  test("rejects a query", () => {
    expect(resolveOwnerUrl(VALID_OWNER + "?x=1", SERVICE)).toBeNull();
  });

  test("rejects a fragment", () => {
    expect(resolveOwnerUrl(VALID_OWNER + "#frag", SERVICE)).toBeNull();
  });

  test("rejects when the service host is a single label", () => {
    expect(
      resolveOwnerUrl("https://10-1-2-3.oe-headless:8443", "https://oe:8443"),
    ).toBeNull();
  });

  test("rejects a wrong namespace", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.other-ns.svc.cluster.local:8443",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("rejects an unparseable owner URL", () => {
    expect(resolveOwnerUrl("http://[::bad", SERVICE)).toBeNull();
  });

  test("rejects raw ASCII control characters before parsing", () => {
    expect(
      resolveOwnerUrl(
        "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443\n",
        SERVICE,
      ),
    ).toBeNull();
  });

  test("warning logs do not expose credentials from a rejected owner URL", () => {
    const owner =
      "https://user:super-secret@10-1-2-3.oe-headless.ns.svc.cluster.local:8443";

    expect(resolveOwnerUrl(owner, SERVICE)).toBeNull();

    const warnings = mocks.logger.warn.mock.calls.map((c) => String(c[0]));
    expect(warnings.join("\n")).toContain("not a bare origin");
    expect(warnings.join("\n")).not.toContain("super-secret");
    expect(warnings.join("\n")).not.toContain("user:");
    expect(warnings.join("\n")).not.toContain(owner);
  });
});
