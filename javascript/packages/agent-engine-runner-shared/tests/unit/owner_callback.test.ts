/**
 * Tests for the shared owner-callback pre-attempt policy (`ownerDelivered`).
 *
 * This is the single place the fallback status matrix lives; call-site tests
 * (progress, aer, function mode, secure wrapper) verify wiring, not policy.
 * Mirrors Python's tests/unit/test_owner_callback.py.
 */

import { describe, test, expect, vi, afterEach } from "vitest";

import { ownerDelivered } from "../../src/owner_callback.js";

const OWNER = "http://10-1-2-3.oe-headless.ns:8000/stream/chunk";

const INIT: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: '{"k":"v"}',
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ownerDelivered", () => {
  test("returns true on a 2xx response and cancels the body", async () => {
    const cancel = vi.fn(async () => {});
    const mock = vi.fn(
      async () =>
        ({
          ok: true,
          status: 200,
          body: { cancel },
        }) as unknown as Response,
    );
    vi.stubGlobal("fetch", mock);
    const log = vi.fn();

    await expect(ownerDelivered(OWNER, INIT, log)).resolves.toBe(true);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(log).not.toHaveBeenCalled();
  });

  test("keeps a successful delivery when body cancellation fails", async () => {
    const mock = vi.fn(
      async () =>
        ({
          ok: true,
          status: 200,
          body: {
            cancel: vi.fn(async () => Promise.reject(new Error("closed"))),
          },
        }) as unknown as Response,
    );
    vi.stubGlobal("fetch", mock);
    const log = vi.fn();

    await expect(ownerDelivered(OWNER, INIT, log)).resolves.toBe(true);
    expect(log).not.toHaveBeenCalled();
  });

  test.each([307, 404, 500, 503])(
    "returns false on an owner HTTP %d response without retrying",
    async (status) => {
      const cancel = vi.fn(async () => {});
      const mock = vi.fn(
        async () =>
          ({
            ok: false,
            status,
            body: { cancel },
          }) as unknown as Response,
      );
      vi.stubGlobal("fetch", mock);
      const log = vi.fn();

      await expect(ownerDelivered(OWNER, INIT, log)).resolves.toBe(false);
      expect(mock).toHaveBeenCalledTimes(1);
      expect(cancel).toHaveBeenCalledTimes(1);
      expect(log).toHaveBeenCalledWith(
        `owner URL ${OWNER} unusable (HTTP ${status}); falling back to service`,
      );
    },
  );

  test("returns false on a transport error without throwing", async () => {
    const mock = vi.fn(async () => {
      throw new Error("connection refused");
    });
    vi.stubGlobal("fetch", mock);
    const log = vi.fn();

    await expect(ownerDelivered(OWNER, INIT, log)).resolves.toBe(false);
    expect(log).toHaveBeenCalledWith(
      `owner URL ${OWNER} unusable (connection refused); falling back to service`,
    );
  });

  test("returns false when request setup fails before fetch", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const log = vi.fn();

    await expect(
      ownerDelivered(
        OWNER,
        () => {
          throw new Error("owner TLS unavailable");
        },
        log,
      ),
    ).resolves.toBe(false);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(log).toHaveBeenCalledWith(
      `owner URL ${OWNER} unusable (owner TLS unavailable); falling back to service`,
    );
  });

  test("forces redirect: manual, overriding any caller value", async () => {
    let seenRedirect: RequestRedirect | undefined;
    const mock = vi.fn(async (_url: string, init?: RequestInit) => {
      seenRedirect = init?.redirect;
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await ownerDelivered(OWNER, { ...INIT, redirect: "follow" }, vi.fn());
    expect(seenRedirect).toBe("manual");
  });

  test("never logs response bodies", async () => {
    const text = vi.fn(async () => "internal diagnostics");
    const mock = vi.fn(
      async () =>
        ({
          ok: false,
          status: 500,
          body: { cancel: async () => {} },
          text,
        }) as unknown as Response,
    );
    vi.stubGlobal("fetch", mock);
    const log = vi.fn();

    await ownerDelivered(OWNER, INIT, log);
    expect(text).not.toHaveBeenCalled();
    expect(log.mock.calls[0]?.[0]).not.toContain("internal diagnostics");
  });
});
