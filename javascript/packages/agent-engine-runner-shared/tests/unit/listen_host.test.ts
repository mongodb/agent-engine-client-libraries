/**
 * Unit tests for resolveListenHost.
 *
 * Mirrors Python's tests/unit/test_listen_host.py.
 *
 * The resolver is a thin wrapper around APP_HOST with a safe default —
 * ECP stamps APP_HOST on every runner component (AER, Tool, memory-server)
 * at deploy time, and the runner-base image bakes in APP_HOST=0.0.0.0
 * as a baseline. The fallback only matters for ad-hoc invocations that
 * bypass both.
 *
 * Source change to enable parity: resolveListenHost was exported from
 * runtime.ts. Python imports _resolve_listen_host directly (leading-underscore
 * = convention-private but still importable). TS strict export semantics
 * required adding `export`. 2-line pure read-from-env helper; no behavioural
 * risk in exposing it.
 *
 * Env isolation — monkeypatch parallel:
 *   - Python's autouse _clear_app_host uses `monkeypatch.delenv` (deletes
 *     + auto-restores at test end).
 *   - Vitest's `vi.stubEnv` only SETS values, not deletes. So beforeEach
 *     does the delete (one-line direct mutation) and `vi.stubEnv` handles
 *     the set cases (auto-restored by afterEach).
 *   - Vitest workers isolate process.env per file, so no file-level
 *     restore is needed — mutations don't leak out.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { resolveListenHost } from "../../src/index.js";

beforeEach(() => {
  delete process.env["APP_HOST"];
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("resolveListenHost", () => {
  test("returns APP_HOST when set", () => {
    // test_returns_app_host_when_set
    vi.stubEnv("APP_HOST", "::");
    expect(resolveListenHost()).toBe("::");
  });

  test("passes through arbitrary APP_HOST value verbatim", () => {
    // test_returns_arbitrary_app_host_value
    // ECP can stamp any bindable host (e.g. a specific interface IP for tests).
    vi.stubEnv("APP_HOST", "192.0.2.42");
    expect(resolveListenHost()).toBe("192.0.2.42");
  });

  test("defaults to 0.0.0.0 when APP_HOST unset", () => {
    // test_defaults_to_v4_when_app_host_unset
    expect(resolveListenHost()).toBe("0.0.0.0");
  });

  test("empty APP_HOST treated as unset (falls through to default)", () => {
    // test_empty_app_host_treated_as_unset
    // A shell that exports APP_HOST="" must not produce a bind to ""
    // (which Fastify/uvicorn would reject).
    vi.stubEnv("APP_HOST", "");
    expect(resolveListenHost()).toBe("0.0.0.0");
  });

  test("default value is bind-compatible (not URI-bracketed)", () => {
    // test_default_value_is_bind_compatible
    // The default-path return (APP_HOST unset) must be directly usable as
    // the host arg to socket.bind / fastify.listen — bare IP literal,
    // never the bracketed URI form ([::]).
    const host = resolveListenHost();
    expect(host.startsWith("[")).toBe(false);
  });
});
