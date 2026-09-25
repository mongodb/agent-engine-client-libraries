/**
 * Per-session filesystem isolation tests — TS port of Python's
 * `test_session_filesystem_isolation.py`.
 *
 * When a session is active every writable/readable op narrows to that
 * session's own `.sessions/<slot>/` subtree, so concurrent sessions on the
 * same Tool Pod cannot read, write, glob, grep, or shell into each other's
 * files. `runInSession` runs a handler under a given session id via the
 * execution-context ALS (the same mechanism AER sets per request).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type * as ToolpodHandlers from "../../src/toolpod_handlers.js";
import type * as ContextModule from "../../src/context.js";

type Handlers = typeof ToolpodHandlers;
type Context = typeof ContextModule;

let tmpRoot: string;
let mod: Handlers;
let ctx: Context;

/** Run `fn` under the execution context for `sessionId`. */
function runInSession<T>(sessionId: string | null, fn: () => T): T {
  return ctx.runWithExecutionContext(
    { executionId: "e", wrapper: null, oeUrl: "", sessionId },
    fn,
  );
}

beforeEach(async () => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "sess-iso-"));
  vi.resetModules();
  process.env["WORKSPACE_DIR"] = fs.mkdtempSync(path.join(tmpRoot, "ws-"));
  // Import both modules in the same reset generation so they share one
  // AsyncLocalStorage instance (getCurrentSessionId reads the same ALS the
  // test writes to).
  mod = await import("../../src/toolpod_handlers.js");
  ctx = await import("../../src/context.js");
});

afterEach(() => {
  delete process.env["WORKSPACE_DIR"];
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe("session filesystem isolation", () => {
  it("a file written in one session is invisible to another session's ls", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "secret.txt", content: "hi" }),
    );
    const res = runInSession("bob", () => mod.filesystemLs({ path: "." }));
    const names = (res["entries"] as Array<{ path: string }>).map((e) =>
      path.basename(e.path),
    );
    expect(names).not.toContain("secret.txt");
  });

  it("another session cannot read the file", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "secret.txt", content: "hi" }),
    );
    const res = runInSession("bob", () =>
      mod.filesystemRead({ file_path: "secret.txt" }),
    );
    expect(res["error"]).toBeDefined();
  });

  it("an edit in one session does not affect another session's same-named file", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "f.txt", content: "alice" }),
    );
    runInSession("bob", () =>
      mod.filesystemWrite({ file_path: "f.txt", content: "bob" }),
    );
    runInSession("alice", () =>
      mod.filesystemEdit({
        file_path: "f.txt",
        old_string: "alice",
        new_string: "ALICE",
      }),
    );
    const bobRead = runInSession("bob", () =>
      mod.filesystemRead({ file_path: "f.txt" }),
    );
    expect(bobRead["content"]).toBe("bob");
  });

  it("glob only returns the caller session's files", async () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "a.txt", content: "" }),
    );
    runInSession("bob", () =>
      mod.filesystemWrite({ file_path: "b.txt", content: "" }),
    );
    const res = await runInSession("alice", () =>
      mod.filesystemGlob({ pattern: "*.txt" }),
    );
    const names = (res["matches"] as Array<{ path: string }>).map((m) =>
      path.basename(m.path),
    );
    expect(names).toEqual(["a.txt"]);
  });

  it("grep only searches the caller session's files", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "a.txt", content: "needle\n" }),
    );
    runInSession("bob", () =>
      mod.filesystemWrite({ file_path: "b.txt", content: "needle\n" }),
    );
    const res = runInSession("alice", () =>
      mod.filesystemGrep({ pattern: "needle" }),
    );
    const matches = res["matches"] as Array<{ path: string }>;
    expect(matches).toHaveLength(1);
    expect(path.basename(matches[0]?.path ?? "")).toBe("a.txt");
  });

  it("download does not expose another session's file", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "secret.bin", content: "hi" }),
    );
    const res = runInSession("bob", () =>
      mod.filesystemDownload({ file_path: "secret.bin" }),
    );
    expect(res["error"]).toBeDefined();
  });

  it("two invocations with the same session id share files", () => {
    runInSession("alice", () =>
      mod.filesystemWrite({ file_path: "f.txt", content: "shared" }),
    );
    const res = runInSession("alice", () =>
      mod.filesystemRead({ file_path: "f.txt" }),
    );
    expect(res["content"]).toBe("shared");
  });

  it("no session falls back to the base workspace, disjoint from any session subtree", () => {
    runInSession(null, () =>
      mod.filesystemWrite({ file_path: "root.txt", content: "root" }),
    );
    // A session cannot see the base-workspace file (its view is .sessions/<slot>).
    const res = runInSession("alice", () => mod.filesystemLs({ path: "." }));
    const names = (res["entries"] as Array<{ path: string }>).map((e) =>
      path.basename(e.path),
    );
    expect(names).not.toContain("root.txt");
  });

  it("rejects a relative traversal into another session's subtree", () => {
    const res = runInSession("alice", () =>
      mod.filesystemRead({ file_path: "../../secret.txt" }),
    );
    expect(res["error"]).toMatch(/escapes workspace sandbox/);
  });

  it("admits an arbitrary session id and isolates it", () => {
    runInSession("team/thread-1", () =>
      mod.filesystemWrite({ file_path: "f.txt", content: "x" }),
    );
    const res = runInSession("team/thread-2", () =>
      mod.filesystemRead({ file_path: "f.txt" }),
    );
    expect(res["error"]).toBeDefined();
  });

  it("does not leak across concurrent async sessions", async () => {
    // Suspend inside the session on a real microtask boundary (not a timer):
    // both writes await, interleave, then resume. If the execution-context ALS
    // didn't propagate across the await each would write under the wrong slot.
    const write = (sid: string) =>
      runInSession(sid, async () => {
        await Promise.resolve();
        mod.filesystemWrite({ file_path: "f.txt", content: sid });
      });
    await Promise.all([write("s1"), write("s2")]);
    const r1 = runInSession("s1", () =>
      mod.filesystemRead({ file_path: "f.txt" }),
    );
    const r2 = runInSession("s2", () =>
      mod.filesystemRead({ file_path: "f.txt" }),
    );
    expect(r1["content"]).toBe("s1");
    expect(r2["content"]).toBe("s2");
  });
});
