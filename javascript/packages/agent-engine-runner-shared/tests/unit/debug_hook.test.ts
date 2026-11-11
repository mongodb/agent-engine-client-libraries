/**
 * Unit tests for maybeStartDebugger.
 *
 * Mirrors Python's tests/unit/test_debug_hook.py.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const inspectorMock = vi.hoisted(() => ({
  open: vi.fn(),
  url: vi.fn<() => string | undefined>(() => undefined),
}));

vi.mock("node:inspector", () => ({ default: inspectorMock }));

import { maybeStartDebugger } from "../../src/runtime.js";

beforeEach(() => {
  delete process.env["MDBAE_LOCAL_MODE"];
  delete process.env["MDBAE_LOCAL_PORT"];
  inspectorMock.open.mockReset();
  inspectorMock.url.mockReset();
  inspectorMock.url.mockReturnValue(undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("maybeStartDebugger", () => {
  test("is a no-op when MDBAE_LOCAL_MODE is unset", () => {
    maybeStartDebugger();
    expect(inspectorMock.open).not.toHaveBeenCalled();
  });

  test("is a no-op when MDBAE_LOCAL_MODE is false", () => {
    vi.stubEnv("MDBAE_LOCAL_MODE", "false");
    maybeStartDebugger();
    expect(inspectorMock.open).not.toHaveBeenCalled();
  });

  test("opens the inspector on loopback without waiting for a client", () => {
    vi.stubEnv("MDBAE_LOCAL_MODE", "true");
    maybeStartDebugger();
    expect(inspectorMock.open).toHaveBeenCalledExactlyOnceWith(
      5678,
      "127.0.0.1",
      false,
    );
  });

  test("honors MDBAE_LOCAL_PORT", () => {
    vi.stubEnv("MDBAE_LOCAL_MODE", "true");
    vi.stubEnv("MDBAE_LOCAL_PORT", "9229");
    maybeStartDebugger();
    expect(inspectorMock.open).toHaveBeenCalledExactlyOnceWith(
      9229,
      "127.0.0.1",
      false,
    );
  });

  test("does not reopen an inspector that is already active", () => {
    // inspector.open() throws ERR_INSPECTOR_ALREADY_ACTIVATED when the
    // process was started with --inspect or the hook already ran.
    vi.stubEnv("MDBAE_LOCAL_MODE", "true");
    inspectorMock.url.mockReturnValue("ws://127.0.0.1:9229/abc");
    maybeStartDebugger();
    expect(inspectorMock.open).not.toHaveBeenCalled();
  });
});
