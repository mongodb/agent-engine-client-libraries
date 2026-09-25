/**
 * Tests for error_reporting redaction + subprocess summarization.
 *
 * Mirrors Python's tests/unit/test_error_reporting.py. Only the pure helpers
 * are exercised (no live Sentry): `redactText`, `beforeSend`,
 * `summarizeSubprocessFailure`, `subprocessOutputTail` — matching the Python
 * file, whose tests likewise never init the SDK.
 */

import { afterEach, expect, test } from "vitest";

import {
  beforeSend,
  redactText,
  setHomeDirForTest,
  subprocessOutputTail,
  summarizeSubprocessFailure,
} from "../../src/error_reporting.js";

afterEach(() => {
  // Restore the real home dir after tests that override it.
  setHomeDirForTest(process.env["HOME"] ?? "");
});

test("redactText scrubs common secret patterns", () => {
  const home = "/tmp/example-home";
  setHomeDirForTest(home);
  const redacted = redactText(
    `token=abc123 password:shhh mongodb+srv://user:pass@example.mongodb.net/app ${home}/project`,
  );

  expect(redacted).not.toContain("abc123");
  expect(redacted).not.toContain("shhh");
  expect(redacted).toContain("token=<redacted>");
  expect(redacted).toContain("password:<redacted>");
  expect(redacted).toContain(
    "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app",
  );
  expect(redacted).not.toContain(home);
});

test("redactText scrubs a bare-token URL credential (no user:pass pair)", () => {
  const redacted = redactText(
    "clone failed: https://ghp_abcdef123456@github.com/org/repo",
  );

  expect(redacted).not.toContain("ghp_abcdef123456");
  expect(redacted).toContain("github.com/org/repo");
});

test("summarizeSubprocessFailure prefers the meaningful output line", () => {
  const exc = {
    command: ["uv", "sync", "--no-dev"],
    exitCode: 1,
    stdout:
      "Resolved 12 packages\n" +
      "  × No solution found when resolving dependencies:\n" +
      "  ╰─▶ Because missing-package was not found in the package registry\n",
  };

  expect(summarizeSubprocessFailure(exc)).toBe(
    "uv sync failed: Because missing-package was not found in the package registry",
  );
});

test("subprocessOutputTail keeps recent output", () => {
  const exc = {
    command: ["uv", "pip", "install"],
    exitCode: 1,
    stdout: "first line\nsecond line\nthird line",
  };

  expect(subprocessOutputTail(exc)).toBe("first line\nsecond line\nthird line");
});

test("beforeSend redacts exception values recursively", () => {
  const event = {
    request: {
      url: "https://user:pass@example.com/path?token=abc123",
      headers: { Authorization: "Bearer abc.def" },
    },
    user: { email: "mongodb+srv://user:pass@example.mongodb.net/app" },
    tags: { dsn: "mongodb+srv://user:pass@example.mongodb.net/app" },
    modules: { pkg: "/tmp/home/project" },
    exception: {
      values: [
        {
          value: "token=abc123 at /tmp/home/project",
          stacktrace: {
            frames: [
              {
                filename: "/tmp/home/project/app.py",
                vars: { Authorization: "Bearer abc.def" },
              },
            ],
          },
        },
      ],
    },
  };

  setHomeDirForTest("/tmp/home");
  // `any`: the redacted event is an arbitrary, deeply-nested Sentry payload
  // (exception.values[0].stacktrace.frames[0].vars...) that the assertions
  // index positionally. Typing every intermediate shape here would add noise
  // without buying safety in a test that deliberately reaches into the guts.
  const redacted = beforeSend(
    event as unknown as Record<string, unknown>,
  ) as unknown as Record<
    string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    any
  >;

  const excValue = redacted["exception"].values[0].value as string;
  const frame = redacted["exception"].values[0].stacktrace.frames[0];
  expect(redacted["request"].url).toBe(
    "https://<redacted>:<redacted>@example.com/path?token=<redacted>",
  );
  expect(redacted["request"].headers.Authorization).toBe("Bearer <redacted>");
  expect(redacted["user"].email).toBe(
    "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app",
  );
  expect(redacted["tags"].dsn).toBe(
    "mongodb+srv://<redacted>:<redacted>@example.mongodb.net/app",
  );
  expect(redacted["modules"].pkg).toBe("<home>/project");
  expect(excValue).not.toContain("abc123");
  expect(excValue).toContain("<home>");
  expect(frame.filename).toBe("<home>/project/app.py");
  expect(frame.vars.Authorization).toBe("Bearer <redacted>");
});

test("redactText tolerates a missing home dir", () => {
  setHomeDirForTest("");
  expect(redactText("token=abc123 /tmp/project")).toBe(
    "token=<redacted> /tmp/project",
  );
});

test("beforeSend tolerates a self-referential extra without overflowing", () => {
  setHomeDirForTest("/tmp/home");
  const cyclic: Record<string, unknown> = { secret: "token=abc123" };
  cyclic["self"] = cyclic; // self-reference

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let redacted: any;
  expect(() => {
    redacted = beforeSend({ extra: cyclic } as unknown as Record<
      string,
      unknown
    >);
  }).not.toThrow();

  // The string is still redacted, and the cycle is replaced with a sentinel
  // rather than driving infinite recursion.
  expect(redacted.extra.secret).toBe("token=<redacted>");
  expect(redacted.extra.self).toBe("[Circular]");
});

test("beforeSend caps pathologically deep extra structures", () => {
  setHomeDirForTest("");
  // Build a chain deeper than MAX_REDACT_DEPTH (12).
  let deep: Record<string, unknown> = { leaf: "token=abc123" };
  for (let i = 0; i < 30; i++) deep = { nested: deep };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let redacted: any;
  expect(() => {
    redacted = beforeSend({ extra: deep } as unknown as Record<
      string,
      unknown
    >);
  }).not.toThrow();

  // Walk down to the depth cutoff and confirm it terminates in the sentinel.
  let node = redacted.extra;
  let hops = 0;
  while (node && typeof node === "object" && "nested" in node && hops < 40) {
    node = node.nested;
    hops++;
  }
  expect(node).toBe("[MaxDepth]");
});
