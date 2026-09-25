/**
 * Behavioral tests for `toolpod_handlers` — the TS port of Python's
 * `test_toolpod_handlers.py`.
 *
 * Each test reloads the module with a fresh `WORKSPACE_DIR` (and any cap
 * overrides) via `loadHandlers`, mirroring Python's `importlib.reload`
 * fixture. Module-level caps are env-driven, so a cap test just sets the env
 * var before import rather than monkeypatching a constant.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type * as ToolpodHandlers from "../../src/toolpod_handlers.js";

type Handlers = typeof ToolpodHandlers;

let tmpRoot: string;

/**
 * Reload the handlers module with `WORKSPACE_DIR` pointed at a fresh temp dir.
 * `env` supplies additional module-level env (cap overrides). Returns the
 * reloaded module and the realpath'd workspace root (the module realpath's
 * WORKSPACE_DIR, so assertions must compare against the resolved form).
 */
async function loadHandlers(
  env: Record<string, string> = {},
): Promise<{ mod: Handlers; workspace: string }> {
  vi.resetModules();
  const dir = fs.mkdtempSync(path.join(tmpRoot, "ws-"));
  process.env["WORKSPACE_DIR"] = dir;
  for (const [k, v] of Object.entries(env)) process.env[k] = v;
  const mod = await import("../../src/toolpod_handlers.js");
  return { mod, workspace: fs.realpathSync(dir) };
}

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "toolpod-"));
});

afterEach(() => {
  delete process.env["WORKSPACE_DIR"];
  for (const key of [
    "MAX_LS_ENTRIES",
    "MAX_GLOB_MATCHES",
    "MAX_GREP_MATCHES",
    "MAX_GREP_SECONDS",
    "FILESYSTEM_READ_MAX_BYTES",
    "FILESYSTEM_WRITE_WARN_BYTES",
    "DOWNLOAD_MAX_BYTES",
    "SHELL_OUTPUT_MAX_BYTES",
    "AGENTIC_AGENT_WORKDIR",
    "AGENTIC_AGENT_CONFIG_PATH",
    "AGENTIC_SKILLS_DIR",
  ]) {
    delete process.env[key];
  }
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe("readonly skill roots", () => {
  beforeEach(() => {
    // The skill root resolves from these env vars; tests must start clean so
    // import-time vs post-import ordering is the only variable.
    delete process.env["AGENTIC_AGENT_WORKDIR"];
    delete process.env["AGENTIC_AGENT_CONFIG_PATH"];
    delete process.env["AGENTIC_SKILLS_DIR"];
  });

  function makeAgentWithSkill(): { agentDir: string; skillPath: string } {
    const agentDir = fs.mkdtempSync(path.join(tmpRoot, "agent-"));
    const skillDir = path.join(agentDir, "skills", "demo");
    fs.mkdirSync(skillDir, { recursive: true });
    const skillPath = path.join(skillDir, "SKILL.md");
    fs.writeFileSync(skillPath, "# demo skill\n");
    return { agentDir, skillPath };
  }

  function fakeRuntime(): Parameters<Handlers["registerBuiltinTools"]>[0] {
    return { tools: {}, toolDefinitions: {} } as unknown as Parameters<
      Handlers["registerBuiltinTools"]
    >[0];
  }

  it("reads a bundled skill when the agent root is set after a static import", async () => {
    // Regression: module imported with no agent root env, env assigned after
    // import, then the first read resolves the root lazily — no reload.
    const { mod } = await loadHandlers();
    const { agentDir, skillPath } = makeAgentWithSkill();
    process.env["AGENTIC_AGENT_WORKDIR"] = agentDir;
    const res = mod.filesystemRead({ file_path: skillPath });
    expect(res["error"]).toBeUndefined();
    expect(res["content"]).toBe("# demo skill\n");
  });

  it("resolves the same root for import-before-env and import-after-env", async () => {
    const { agentDir, skillPath } = makeAgentWithSkill();
    const early = await loadHandlers();
    process.env["AGENTIC_AGENT_WORKDIR"] = agentDir;
    // Fresh module imported only now that the env is set (dynamic-import form).
    vi.resetModules();
    const late = await import("../../src/toolpod_handlers.js");
    const a = early.mod.filesystemRead({ file_path: skillPath });
    const b = late.filesystemRead({ file_path: skillPath });
    expect(a["content"]).toBe("# demo skill\n");
    expect(b["content"]).toBe(a["content"]);
  });

  it("throws an actionable startup error when AGENTIC_SKILLS_DIR is set but unresolvable", async () => {
    const { agentDir } = makeAgentWithSkill();
    const { mod } = await loadHandlers({
      AGENTIC_AGENT_WORKDIR: agentDir,
      AGENTIC_SKILLS_DIR: "no-such-dir",
    });
    expect(() => mod.registerBuiltinTools(fakeRuntime())).toThrow(
      /AGENTIC_SKILLS_DIR=.*no readable skills root/,
    );
  });

  it("warns at startup when the default skills path exists but is not a directory", async () => {
    // Previously a silent no-op, so warn (not throw) for one release to avoid
    // crash-looping already-deployed agents.
    const agentDir = fs.mkdtempSync(path.join(tmpRoot, "agent-"));
    fs.writeFileSync(path.join(agentDir, "skills"), "not a dir");
    const { mod } = await loadHandlers({ AGENTIC_AGENT_WORKDIR: agentDir });
    // log4js hands out a fresh wrapper per getLogger() call; the level methods
    // live on the shared prototype, so spy there (see tool_server_startup).
    const { getLogger } = await import("../../src/logger.js");
    const warn = vi.spyOn(
      Object.getPrototypeOf(getLogger("x")) as { warn: (msg: string) => void },
      "warn",
    );
    expect(() => mod.registerBuiltinTools(fakeRuntime())).not.toThrow();
    expect(warn).toHaveBeenCalledWith(
      expect.stringMatching(/not a readable directory/),
    );
    warn.mockRestore();
  });

  it("starts cleanly and reads a bundled skill with a valid default skills dir", async () => {
    const { agentDir, skillPath } = makeAgentWithSkill();
    const { mod } = await loadHandlers({ AGENTIC_AGENT_WORKDIR: agentDir });
    expect(() => mod.registerBuiltinTools(fakeRuntime())).not.toThrow();
    expect(mod.filesystemRead({ file_path: skillPath })["content"]).toBe(
      "# demo skill\n",
    );
  });

  it("starts cleanly with no skills bundled and exposes no readonly root", async () => {
    const agentDir = fs.mkdtempSync(path.join(tmpRoot, "agent-"));
    fs.writeFileSync(path.join(agentDir, "notes.txt"), "x");
    const { mod } = await loadHandlers({ AGENTIC_AGENT_WORKDIR: agentDir });
    expect(() => mod.registerBuiltinTools(fakeRuntime())).not.toThrow();
    // No skills dir → the agent source root is not a readable root, so an
    // absolute path there is rebased into the workspace and misses.
    const res = mod.filesystemRead({
      file_path: path.join(agentDir, "notes.txt"),
    });
    expect(res["error"]).toMatch(/ENOENT|no such file/i);
  });
});

describe("sandbox path resolution", () => {
  it("rebases an absolute path outside the workspace and keeps it inside", async () => {
    const { mod, workspace } = await loadHandlers();
    // /etc/passwd is rebased under the workspace (leading / stripped), so the
    // read fails with ENOENT rather than reading the host's file.
    const res = mod.filesystemRead({ file_path: "/etc/passwd" });
    expect(res["error"]).toMatch(/ENOENT|no such file/i);
    // Confirm a write of the same path lands inside the workspace.
    const w = mod.filesystemWrite({ file_path: "/etc/x.txt", content: "hi" });
    expect(String(w["path"])).toContain(workspace);
  });

  it("rejects parent traversal", async () => {
    const { mod } = await loadHandlers();
    const res = mod.filesystemRead({ file_path: "../../../etc/passwd" });
    expect(res["error"]).toMatch(/escapes workspace sandbox/);
  });

  it("rejects a symlink escape", async () => {
    const { mod, workspace } = await loadHandlers();
    const outside = path.join(tmpRoot, "outside.txt");
    fs.writeFileSync(outside, "secret");
    fs.symlinkSync(outside, path.join(workspace, "evil"));
    const res = mod.filesystemRead({ file_path: "evil" });
    expect(res["error"]).toMatch(/escapes workspace sandbox/);
  });
});

describe("filesystem_ls", () => {
  it("lists files and dirs sorted, with is_dir flags", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "a.txt"), "a");
    fs.writeFileSync(path.join(workspace, "b.txt"), "b");
    fs.mkdirSync(path.join(workspace, "subdir"));
    const res = mod.filesystemLs({ path: "." });
    const entries = res["entries"] as Array<{ path: string; is_dir: boolean }>;
    expect(entries.map((e) => path.basename(e.path))).toEqual([
      "a.txt",
      "b.txt",
      "subdir",
    ]);
    const byName = Object.fromEntries(
      entries.map((e) => [path.basename(e.path), e.is_dir]),
    );
    expect(byName).toEqual({ "a.txt": false, "b.txt": false, subdir: true });
  });

  it("errors on a missing directory", async () => {
    const { mod } = await loadHandlers();
    expect(mod.filesystemLs({ path: "does-not-exist" })["error"]).toBeDefined();
  });

  it("truncates at the entry cap", async () => {
    const { mod, workspace } = await loadHandlers({ MAX_LS_ENTRIES: "3" });
    for (let i = 0; i < 10; i++) {
      fs.writeFileSync(path.join(workspace, `f${i}.txt`), "");
    }
    const res = mod.filesystemLs({ path: "." });
    expect((res["entries"] as unknown[]).length).toBe(3);
    expect(res["truncated"]).toBe(true);
  });
});

describe("filesystem_read", () => {
  it("reads a full file", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "line1\nline2\n");
    const res = mod.filesystemRead({ file_path: "f.txt" });
    expect(res["content"]).toBe("line1\nline2\n");
    expect(res["encoding"]).toBe("utf-8");
  });

  it("slices by offset and limit", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "l0\nl1\nl2\nl3\nl4\n");
    const res = mod.filesystemRead({ file_path: "f.txt", offset: 1, limit: 2 });
    expect(res["content"]).toBe("l1\nl2\n");
  });

  it("rejects a file over the size cap", async () => {
    const { mod, workspace } = await loadHandlers({
      FILESYSTEM_READ_MAX_BYTES: "5",
    });
    fs.writeFileSync(path.join(workspace, "big.txt"), "0123456789");
    expect(mod.filesystemRead({ file_path: "big.txt" })["error"]).toMatch(
      /FILESYSTEM_READ_MAX_BYTES/,
    );
  });
});

describe("filesystem_write", () => {
  it("creates parent directories", async () => {
    const { mod, workspace } = await loadHandlers();
    const res = mod.filesystemWrite({ file_path: "a/b/c.txt", content: "x" });
    expect(res["path"]).toBeDefined();
    expect(fs.readFileSync(path.join(workspace, "a/b/c.txt"), "utf-8")).toBe(
      "x",
    );
  });

  it("rejects an existing file (create-only)", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "old");
    expect(
      mod.filesystemWrite({ file_path: "f.txt", content: "new" })["error"],
    ).toMatch(/already exists/);
  });
});

describe("filesystem_edit", () => {
  it("replaces a unique match", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "hello world");
    const res = mod.filesystemEdit({
      file_path: "f.txt",
      old_string: "world",
      new_string: "there",
    });
    expect(res["occurrences"]).toBe(1);
    expect(fs.readFileSync(path.join(workspace, "f.txt"), "utf-8")).toBe(
      "hello there",
    );
  });

  it("rejects a non-unique match by default", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "a a a");
    expect(
      mod.filesystemEdit({
        file_path: "f.txt",
        old_string: "a",
        new_string: "b",
      })["error"],
    ).toMatch(/not unique/);
  });

  it("replace_all replaces every occurrence", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "a a a");
    const res = mod.filesystemEdit({
      file_path: "f.txt",
      old_string: "a",
      new_string: "b",
      replace_all: true,
    });
    expect(res["occurrences"]).toBe(3);
    expect(fs.readFileSync(path.join(workspace, "f.txt"), "utf-8")).toBe(
      "b b b",
    );
  });

  it("rejects an empty old_string", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "x");
    expect(
      mod.filesystemEdit({
        file_path: "f.txt",
        old_string: "",
        new_string: "y",
      })["error"],
    ).toMatch(/non-empty/);
  });

  it("reports not-found for a missing old_string", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "x");
    expect(
      mod.filesystemEdit({
        file_path: "f.txt",
        old_string: "z",
        new_string: "y",
      })["error"],
    ).toMatch(/not found/);
  });
});

describe("filesystem_glob", () => {
  it("matches a recursive pattern", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.mkdirSync(path.join(workspace, "sub"));
    fs.writeFileSync(path.join(workspace, "a.py"), "");
    fs.writeFileSync(path.join(workspace, "sub/b.py"), "");
    fs.writeFileSync(path.join(workspace, "sub/c.txt"), "");
    const res = await mod.filesystemGlob({ pattern: "**/*.py" });
    const names = (res["matches"] as Array<{ path: string }>)
      .map((m) => path.basename(m.path))
      .sort();
    expect(names).toEqual(["a.py", "b.py"]);
  });

  it("rejects an absolute pattern", async () => {
    const { mod } = await loadHandlers();
    expect((await mod.filesystemGlob({ pattern: "/etc/*" }))["error"]).toMatch(
      /Absolute pattern/,
    );
  });

  it("rejects a traversal pattern", async () => {
    const { mod } = await loadHandlers();
    expect((await mod.filesystemGlob({ pattern: "../*" }))["error"]).toMatch(
      /traversal/,
    );
  });

  it("truncates at the match cap", async () => {
    const { mod, workspace } = await loadHandlers({ MAX_GLOB_MATCHES: "2" });
    for (let i = 0; i < 5; i++)
      fs.writeFileSync(path.join(workspace, `f${i}.txt`), "");
    const res = await mod.filesystemGlob({ pattern: "*.txt" });
    expect((res["matches"] as unknown[]).length).toBe(2);
    expect(res["truncated"]).toBe(true);
  });
});

describe("filesystem_grep", () => {
  it("finds matching lines", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "alpha\nbeta\ngamma\n");
    const res = mod.filesystemGrep({ pattern: "beta" });
    const matches = res["matches"] as Array<{ line: number; text: string }>;
    expect(matches).toHaveLength(1);
    expect(matches[0]?.line).toBe(2);
    expect(matches[0]?.text).toBe("beta");
  });

  it("filters by glob", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "a.py"), "needle\n");
    fs.writeFileSync(path.join(workspace, "b.txt"), "needle\n");
    const res = mod.filesystemGrep({ pattern: "needle", glob: "*.py" });
    const matches = res["matches"] as Array<{ path: string }>;
    expect(matches).toHaveLength(1);
    expect(path.basename(matches[0]?.path ?? "")).toBe("a.py");
  });

  it("matches a literal substring, not a regex", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(path.join(workspace, "f.txt"), "a.b\naxb\n");
    // "a.b" as a regex would also match "axb"; literal matching must not.
    const res = mod.filesystemGrep({ pattern: "a.b" });
    expect((res["matches"] as unknown[]).length).toBe(1);
  });

  it("skips binary files", async () => {
    const { mod, workspace } = await loadHandlers();
    fs.writeFileSync(
      path.join(workspace, "bin"),
      Buffer.from([0x00, 0x01, 0x02]),
    );
    fs.writeFileSync(path.join(workspace, "t.txt"), "hit\n");
    const res = mod.filesystemGrep({ pattern: "hit" });
    expect((res["matches"] as unknown[]).length).toBe(1);
  });

  it("truncates at the match cap", async () => {
    const { mod, workspace } = await loadHandlers({ MAX_GREP_MATCHES: "2" });
    fs.writeFileSync(path.join(workspace, "f.txt"), "x\nx\nx\nx\n");
    const res = mod.filesystemGrep({ pattern: "x" });
    expect((res["matches"] as unknown[]).length).toBe(2);
    expect(res["truncated"]).toBe(true);
  });

  it("finds a match on a line spanning multiple read chunks", async () => {
    // The file is streamed in 64 KiB chunks; a hit whose line straddles a chunk
    // boundary must still be found with the correct line number.
    const { mod, workspace } = await loadHandlers();
    const filler = "a".repeat(200_000);
    fs.writeFileSync(
      path.join(workspace, "big.txt"),
      `${filler}NEEDLE\nsecond\n`,
    );
    const res = mod.filesystemGrep({ pattern: "NEEDLE" });
    const matches = res["matches"] as Array<{ line: number }>;
    expect(matches).toHaveLength(1);
    expect(matches[0]?.line).toBe(1);
  });

  it("reassembles a multi-byte char split across a chunk boundary", async () => {
    // A 3-byte UTF-8 char landing exactly on the 64 KiB read boundary must not
    // be mangled — the StringDecoder holds the partial bytes across chunks.
    const { mod, workspace } = await loadHandlers();
    const pad = "a".repeat(65_535); // pushes the € (3 bytes) across 64 KiB
    fs.writeFileSync(path.join(workspace, "u.txt"), `${pad}€needle\n`);
    const res = mod.filesystemGrep({ pattern: "€needle" });
    expect((res["matches"] as unknown[]).length).toBe(1);
  });
});

describe("filesystem_download", () => {
  it("base64-roundtrips file bytes", async () => {
    const { mod, workspace } = await loadHandlers();
    const raw = Buffer.from([0x00, 0xff, 0x10, 0x20]);
    fs.writeFileSync(path.join(workspace, "b.bin"), raw);
    const res = mod.filesystemDownload({ file_path: "b.bin" });
    expect(res["encoding"]).toBe("base64");
    expect(Buffer.from(String(res["content_base64"]), "base64")).toEqual(raw);
  });

  it("rejects a file over the size cap", async () => {
    const { mod, workspace } = await loadHandlers({ DOWNLOAD_MAX_BYTES: "3" });
    fs.writeFileSync(path.join(workspace, "big.bin"), Buffer.alloc(10));
    expect(mod.filesystemDownload({ file_path: "big.bin" })["error"]).toMatch(
      /DOWNLOAD_MAX_BYTES/,
    );
  });
});

describe("shell_execute", () => {
  it("runs a command and returns stdout", async () => {
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({ command: "echo hello" });
    expect(res["exit_code"]).toBe(0);
    expect(String(res["output"])).toContain("hello");
  });

  it("combines stderr with stdout", async () => {
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({ command: "echo out; echo err 1>&2" });
    expect(String(res["output"])).toContain("out");
    expect(String(res["output"])).toContain("err");
  });

  it("surfaces a non-zero exit code", async () => {
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({ command: "exit 3" });
    expect(res["exit_code"]).toBe(3);
  });

  it("returns the timeout sentinel with partial output", async () => {
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({
      command: "echo EARLY; sleep 5",
      timeout: 1,
    });
    expect(res["exit_code"]).toBe(-1);
    expect(String(res["output"])).toContain("EARLY");
    expect(String(res["output"])).toContain("timed out");
  });

  it("does not report a timeout for a signal-killed command", async () => {
    // Regression: only our timeout SIGKILL counts as a timeout. A command that
    // dies from a different signal (here SIGTERM, self-sent) well inside the
    // timeout window must surface as a framework error, not a spurious timeout.
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({
      command: "kill -TERM $$",
      timeout: 30,
    });
    expect(String(res["output"])).not.toContain("timed out");
    expect(res["exit_code"]).not.toBe(-1);
  });

  it("reports the real exit code for a command that finishes before the deadline", async () => {
    // Regression for the timer-vs-close race: a command completing on its own
    // shortly before the timeout must report its genuine exit code, never the
    // timeout sentinel.
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({
      command: "sleep 0.2; exit 7",
      timeout: 5,
    });
    expect(res["exit_code"]).toBe(7);
    expect(String(res["output"])).not.toContain("timed out");
  });

  it("bounds unbounded output without hanging", async () => {
    const { mod } = await loadHandlers({ SHELL_OUTPUT_MAX_BYTES: "1024" });
    const res = await mod.shellExecute({
      command: "yes | head -c 100000",
      timeout: 10,
    });
    expect(
      Buffer.byteLength(String(res["output"]), "utf-8"),
    ).toBeLessThanOrEqual(1024);
    expect(res["truncated"]).toBe(true);
  });

  it("scrubs tenant env from the child", async () => {
    process.env["TENANT_SECRET"] = "s3cr3t";
    try {
      const { mod } = await loadHandlers();
      const res = await mod.shellExecute({
        command: "echo v=${TENANT_SECRET:-MISSING}",
      });
      expect(String(res["output"])).toContain("MISSING");
    } finally {
      delete process.env["TENANT_SECRET"];
    }
  });

  it("passes allowlisted env through to the child", async () => {
    const { mod } = await loadHandlers();
    const res = await mod.shellExecute({ command: "echo p=${PATH:+set}" });
    expect(String(res["output"])).toContain("p=set");
  });
});
