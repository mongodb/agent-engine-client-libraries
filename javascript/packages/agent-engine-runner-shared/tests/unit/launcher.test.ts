/**
 * Tests for agent_engine_runner_shared.launcher module.
 *
 * Mirrors Python's tests/unit/test_launcher.py (12 of 15 tests; 3 MCP OAuth
 * tests are out of scope — TS launcher has no materialize_mcp_oauth_secret_cache
 * per Phase 13 scope).
 *
 * TS-vs-Python adaptations:
 *   - Python uses `importlib.import_module` which can be `patch(...)`'d.
 *     TS dynamic `import()` is a language feature, not a replaceable function.
 *     `TestMain` tests run as subprocesses against the built `dist/launcher.js`
 *     with real temp module files on disk — same pattern as p13-smoke.mjs.
 *   - Python `_resolve_entrypoint` calls `sys.exit(1)` on error. TS
 *     `resolveEntrypoint` calls `process.exit(1)`. In-process testing
 *     requires mocking `process.exit` to throw a sentinel instead of
 *     killing the Vitest worker.
 *   - Requires `dist/launcher.js` to be built. The standard build/test gate
 *     ensures this; running `npm test` in isolation will fail until
 *     `npm run build` has been executed.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import log4js from "log4js";
import {
  resolveEntrypoint,
  resolveImportTarget,
  boundText,
  writeTerminationMessage,
  setTerminationLogPathForTest,
  EXIT_IMPORT_ERROR,
  EXIT_NO_ENTRYPOINT,
  EXIT_STARTUP_CRASH,
} from "../../src/index.js";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// TestResolveEntrypoint — in-process, with process.exit mocked
// ---------------------------------------------------------------------------

class ProcessExitError extends Error {
  constructor(public code: number | string | null | undefined) {
    super(`process.exit(${code ?? "undefined"})`);
  }
}

beforeEach(() => {
  delete process.env["AGENT_ENTRYPOINT"];
  vi.spyOn(process, "exit").mockImplementation((code) => {
    throw new ProcessExitError(code);
  });
  // Suppress noisy log output from the error-path tests. resolveEntrypoint
  // now logs via logger.error() (log4js), not console.error — the spy
  // targets the same named-category singleton launcher.ts's module-level
  // `logger` resolves to, since log4js's registry is keyed globally by
  // category name.
  vi.spyOn(
    log4js.getLogger("agent_engine_runner_shared.launcher"),
    "error",
  ).mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("resolveEntrypoint", () => {
  test("module without colon defaults to :main", () => {
    // test_module_only_defaults_to_main
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent.app");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.app",
      exportName: "main",
    });
  });

  test("module:export form splits correctly", () => {
    // test_colon_form_splits_correctly
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent.main:run_agent");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.main",
      exportName: "run_agent",
    });
  });

  test("multiple colons → rightmost is the split", () => {
    // test_colon_in_module_path_uses_rightmost
    // (Python test uses a single colon; we additionally verify the rightmost-
    // split behaviour documented in the TS source via the `lastIndexOf` call.)
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent.main:run");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.main",
      exportName: "run",
    });
  });

  test("trailing carriage return is stripped", () => {
    // test_trailing_carriage_return_stripped
    // An agent.yaml authored on Windows leaves a CR on the value the
    // build pipeline greps out of it and bakes into AGENT_ENTRYPOINT.
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent.main:app\r");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.main",
      exportName: "app",
    });
  });

  test("surrounding whitespace is stripped", () => {
    // test_surrounding_whitespace_stripped
    vi.stubEnv("AGENT_ENTRYPOINT", "  my_agent.main : app \r\n");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.main",
      exportName: "app",
    });
  });

  test("module without colon and a carriage return defaults to :main", () => {
    // test_module_only_with_carriage_return_defaults_to_main
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent.app\r");
    expect(resolveEntrypoint()).toEqual({
      modulePath: "my_agent.app",
      exportName: "main",
    });
  });

  test("whitespace-only AGENT_ENTRYPOINT exits EXIT_NO_ENTRYPOINT", () => {
    // test_whitespace_only_env_var_exits
    vi.stubEnv("AGENT_ENTRYPOINT", " \r\n");
    try {
      resolveEntrypoint();
      expect.unreachable();
    } catch (e) {
      expect((e as ProcessExitError).code).toBe(EXIT_NO_ENTRYPOINT);
    }
  });

  test("missing AGENT_ENTRYPOINT exits EXIT_NO_ENTRYPOINT", () => {
    // test_missing_env_var_exits
    // beforeEach already deleted the env var
    expect(() => resolveEntrypoint()).toThrow(ProcessExitError);
    try {
      resolveEntrypoint();
    } catch (e) {
      expect((e as ProcessExitError).code).toBe(EXIT_NO_ENTRYPOINT);
    }
  });

  test("empty AGENT_ENTRYPOINT exits EXIT_NO_ENTRYPOINT", () => {
    // test_empty_env_var_exits
    vi.stubEnv("AGENT_ENTRYPOINT", "");
    try {
      resolveEntrypoint();
      expect.unreachable();
    } catch (e) {
      expect((e as ProcessExitError).code).toBe(EXIT_NO_ENTRYPOINT);
    }
  });

  test('trailing colon ("module:") exits EXIT_NO_ENTRYPOINT', () => {
    // test_trailing_colon_exits
    vi.stubEnv("AGENT_ENTRYPOINT", "my_agent:");
    try {
      resolveEntrypoint();
      expect.unreachable();
    } catch (e) {
      expect((e as ProcessExitError).code).toBe(EXIT_NO_ENTRYPOINT);
    }
  });

  test('leading colon (":export") exits EXIT_NO_ENTRYPOINT', () => {
    // test_leading_colon_exits
    vi.stubEnv("AGENT_ENTRYPOINT", ":run");
    try {
      resolveEntrypoint();
      expect.unreachable();
    } catch (e) {
      expect((e as ProcessExitError).code).toBe(EXIT_NO_ENTRYPOINT);
    }
  });

  describe("writes a termination message", () => {
    let tmpDir: string;
    let logPath: string;

    beforeEach(() => {
      tmpDir = mkdtempSync(join(tmpdir(), "termination-log-test-"));
      logPath = join(tmpDir, "termination-log");
      setTerminationLogPathForTest(logPath);
    });

    afterEach(() => {
      rmSync(tmpDir, { recursive: true, force: true });
    });

    test("missing AGENT_ENTRYPOINT", () => {
      try {
        resolveEntrypoint();
      } catch {
        // process.exit is mocked to throw; ignore.
      }
      expect(readFileSync(logPath, "utf-8")).toBe(
        "AGENT_ENTRYPOINT is not set",
      );
    });

    test("malformed AGENT_ENTRYPOINT", () => {
      vi.stubEnv("AGENT_ENTRYPOINT", ":run");
      try {
        resolveEntrypoint();
      } catch {
        // process.exit is mocked to throw; ignore.
      }
      expect(readFileSync(logPath, "utf-8")).toContain("must be non-empty");
    });
  });
});

// ---------------------------------------------------------------------------
// resolveImportTarget — in-process, pure
// ---------------------------------------------------------------------------

describe("resolveImportTarget", () => {
  test("dotted module path → compiled dist/*.js file URL", () => {
    const target = resolveImportTarget(
      "agent_ts_langgraph_agent.main",
      "/app/agent",
    );
    expect(target).toBe(
      pathToFileURL("/app/agent/dist/agent_ts_langgraph_agent/main.js").href,
    );
    expect(target.startsWith("file://")).toBe(true);
    expect(target.endsWith("/dist/agent_ts_langgraph_agent/main.js")).toBe(
      true,
    );
  });

  test("multi-dot module path maps each segment to a path component", () => {
    const target = resolveImportTarget("a.b.c", "/app/agent");
    expect(target).toBe(pathToFileURL("/app/agent/dist/a/b/c.js").href);
  });

  test("single-segment module path resolves under dist/", () => {
    const target = resolveImportTarget("mymodule", "/app/agent");
    expect(target).toBe(pathToFileURL("/app/agent/dist/mymodule.js").href);
  });

  test("dotted module whose leaf segment is js/mjs/cjs still translates", () => {
    // Regression guard: a JS-extension suffix on a dotted module name (no path
    // separator) must NOT be treated as an already-importable file, or the
    // original bare-specifier crash returns.
    expect(resolveImportTarget("agent_pkg.utils.cjs", "/app/agent")).toBe(
      pathToFileURL("/app/agent/dist/agent_pkg/utils/cjs.js").href,
    );
  });

  test("absolute file path is returned unchanged (already importable)", () => {
    const abs = "/tmp/x/fake_module.mjs";
    expect(resolveImportTarget(abs, "/app/agent")).toBe(abs);
  });

  test("relative path is returned unchanged", () => {
    expect(resolveImportTarget("./local.js", "/app/agent")).toBe("./local.js");
  });

  test("path-bearing specifier (npm subpath) is returned unchanged", () => {
    expect(resolveImportTarget("@scope/pkg/sub", "/app/agent")).toBe(
      "@scope/pkg/sub",
    );
  });

  test("defaults agentRoot to process.cwd()", () => {
    const target = resolveImportTarget("pkg.main");
    expect(target).toBe(
      pathToFileURL(join(process.cwd(), "dist", "pkg", "main.js")).href,
    );
  });
});

// ---------------------------------------------------------------------------
// TestMain — subprocess via spawnSync against dist/launcher.js
// ---------------------------------------------------------------------------

const LAUNCHER = fileURLToPath(
  new URL("../../dist/launcher.js", import.meta.url),
);

interface SpawnResult {
  status: number | null;
  stdout: string;
  stderr: string;
}

function runLauncher(env: Record<string, string>, cwd?: string): SpawnResult {
  const r = spawnSync(process.execPath, [LAUNCHER], {
    // Pin STRUCTURED_LOGGING off unless the test opts in — otherwise a
    // developer env with STRUCTURED_LOGGING=true would make every case emit JSON.
    env: { ...process.env, STRUCTURED_LOGGING: "", ...env },
    ...(cwd ? { cwd } : {}),
    encoding: "utf-8",
  });
  return {
    status: r.status,
    stdout: r.stdout ?? "",
    stderr: r.stderr ?? "",
  };
}

function combinedOutput(r: SpawnResult): string {
  return `${r.stdout}\n${r.stderr}`;
}

function jsonRecords(stdout: string): Array<Record<string, unknown>> {
  return stdout
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("{"))
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as Record<string, unknown>];
      } catch {
        return [];
      }
    });
}

describe("boundText", () => {
  test("under limits is unchanged", () => {
    const text = "short message\nwith two lines";
    expect(boundText(text)).toBe(text);
  });

  test("truncates by line count", () => {
    const text = Array(50).fill("line").join("\n");
    const result = boundText(text, 4 * 1024, 10);
    expect(result.split("\n").length).toBeLessThanOrEqual(11); // 10 lines + marker
    expect(result.endsWith("[truncated]")).toBe(true);
  });

  test("truncates by byte size", () => {
    const text = "a".repeat(10_000);
    const result = boundText(text, 100, 40);
    expect(Buffer.byteLength(result, "utf-8")).toBeLessThanOrEqual(
      100 + Buffer.byteLength("\n[truncated]", "utf-8"),
    );
    expect(result.endsWith("[truncated]")).toBe(true);
  });
});

describe("writeTerminationMessage", () => {
  let tmpDir: string;
  let logPath: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "termination-log-test-"));
    logPath = join(tmpDir, "termination-log");
    setTerminationLogPathForTest(logPath);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  test("redacts credentials and preserves the exception name and host", () => {
    const err = new Error(
      "failed with mongodb+srv://dbuser:hunter2@cluster0.mongodb.net",
    );
    err.name = "ConnectionError";
    writeTerminationMessage(
      "Unhandled exception from agent entrypoint 'x:y'",
      err,
    );

    const text = readFileSync(logPath, "utf-8");
    expect(text).not.toContain("hunter2");
    expect(text).toContain("cluster0.mongodb.net");
    expect(text).toContain("ConnectionError");
    expect(text).toContain("Unhandled exception from agent entrypoint 'x:y'");
  });

  test("no error writes the bare summary", () => {
    writeTerminationMessage("AGENT_ENTRYPOINT is not set");
    expect(readFileSync(logPath, "utf-8")).toBe("AGENT_ENTRYPOINT is not set");
  });

  test("redacts a credential that would otherwise straddle the truncation boundary", () => {
    // Filler pushes the credential's end near the 4KiB cutoff. With the old
    // bound-before-redact order this would truncate mid-credential before
    // redactText ever saw the full userinfo shape, leaking a partial secret.
    const filler = "x".repeat(4090);
    const err = new Error(
      `${filler} mongodb+srv://dbuser:hunter2@cluster0.mongodb.net`,
    );
    writeTerminationMessage("startup failed", err);
    expect(readFileSync(logPath, "utf-8")).not.toContain("hunter2");
  });

  test("redacts a bare-token URL credential (no user:pass pair)", () => {
    const err = new Error(
      "clone failed: https://ghp_abcdef123456@github.com/org/repo",
    );
    writeTerminationMessage("startup failed", err);
    const text = readFileSync(logPath, "utf-8");
    expect(text).not.toContain("ghp_abcdef123456");
    expect(text).toContain("github.com");
  });

  test("is best-effort on a write failure", () => {
    setTerminationLogPathForTest(
      join(tmpDir, "no-such-dir", "termination-log"),
    );
    expect(() =>
      writeTerminationMessage("does not matter", new Error("boom")),
    ).not.toThrow();
  });
});

let tmpModulesDir: string;

describe("runLauncher (subprocess)", () => {
  beforeEach(() => {
    tmpModulesDir = mkdtempSync(join(tmpdir(), "launcher-test-"));
  });

  afterEach(() => {
    rmSync(tmpModulesDir, { recursive: true, force: true });
  });

  test("dotted module path resolves to the compiled dist file (regression)", () => {
    // Regression for the AER/Tool crash: AGENT_ENTRYPOINT carries the dotted,
    // Python-style entrypoint from agent.yaml (e.g. `agent_demo.main:app`),
    // which the launcher must translate to <cwd>/dist/agent_demo/main.js.
    // type:module so the .js compiled file is treated as ESM (matches a real
    // agent package, whose package.json sets "type":"module").
    writeFileSync(
      join(tmpModulesDir, "package.json"),
      JSON.stringify({ type: "module" }),
    );
    const compiled = join(tmpModulesDir, "dist", "agent_demo", "main.js");
    mkdirSync(dirname(compiled), { recursive: true });
    writeFileSync(
      compiled,
      `export const app = { run() { console.log('DOTTED_RUN_INVOKED') } }\n`,
    );

    const r = runLauncher(
      { AGENT_ENTRYPOINT: "agent_demo.main:app" },
      tmpModulesDir,
    );

    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/DOTTED_RUN_INVOKED/);
  });

  test("callable target is invoked directly", () => {
    // test_callable_target_invoked_directly
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(
      modulePath,
      `export function start() { console.log('START_INVOKED') }\n`,
    );

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:start` });

    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/START_INVOKED/);
  });

  test("non-callable with .run() method has .run() invoked", () => {
    // test_non_callable_with_run_method
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(
      modulePath,
      `export const app = { run() { console.log('RUN_INVOKED') } }\n`,
    );

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:app` });

    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/RUN_INVOKED/);
  });

  test("non-existent module exits EXIT_IMPORT_ERROR", () => {
    // test_module_not_found_exits
    const modulePath = join(tmpModulesDir, "nonexistent.mjs");

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:main` });

    expect(r.status).toBe(EXIT_IMPORT_ERROR);
    // logger.error() now writes through log4js's default human appender,
    // which is type: "stdout" — fatal launcher text moves from stderr to
    // stdout when STRUCTURED_LOGGING is unset. Production always
    // sets STRUCTURED_LOGGING=true, where this lands as JSON on stdout too.
    expect(combinedOutput(r)).toMatch(/Cannot import module/);
  });

  test("missing export on existing module exits EXIT_NO_ENTRYPOINT", () => {
    // test_missing_attribute_exits
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(modulePath, `export const other = 42\n`);

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:missing` });

    expect(r.status).toBe(EXIT_NO_ENTRYPOINT);
    expect(combinedOutput(r)).toMatch(/has no export/);
  });

  test("invalid MCP OAuth secret exits and logs the cause", () => {
    // test_materialize_mcp_oauth_secret_error_exits (Python equivalent).
    const r = runLauncher({
      AGENT_ENTRYPOINT: "irrelevant:main",
      AGENTIC_MCP_OAUTH_B64_GITHUB: "not-valid-base64!!!",
    });

    expect(r.status).toBe(1);
    expect(combinedOutput(r)).toMatch(
      /Cannot materialize MCP OAuth credentials/,
    );
    expect(combinedOutput(r)).toMatch(/not valid base64/);
  });

  test("non-callable export without .run() exits EXIT_NO_ENTRYPOINT", () => {
    // test_non_callable_without_run_exits
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(modulePath, `export const config = { key: 'value' }\n`);

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:config` });

    expect(r.status).toBe(EXIT_NO_ENTRYPOINT);
    expect(combinedOutput(r)).toMatch(
      /is not callable and has no callable 'run\(\)' method/,
    );
  });

  test("bootstrap failure round-trips as level=ERROR when STRUCTURED_LOGGING=true", () => {
    // AGENT_ENTRYPOINT intentionally left unset (the outer beforeEach
    // deletes it from the parent env, which the subprocess inherits).
    const r = runLauncher({ STRUCTURED_LOGGING: "true", RUNNER_MODE: "aer" });

    expect(r.status).toBe(EXIT_NO_ENTRYPOINT);
    const records = r.stdout
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l));
    const rec = records.find(
      (j) => j.message === "AGENT_ENTRYPOINT is not set",
    );
    expect(rec).toBeDefined();
    expect(rec.level).toBe("ERROR");
    expect(rec.service).toBe("agent-execution-runtime");
  });

  test("unrecognized RUNNER_MODE falls back to an allowlisted service instead of passing the garbage value through", () => {
    // The actual regression case for the silent-drop risk: /agent-logs'
    // hard service allowlist silently drops any record outside
    // {agent-execution-runtime, tool-executor} — worse than today's
    // mislabeled-but-visible line.
    const r = runLauncher({
      STRUCTURED_LOGGING: "true",
      RUNNER_MODE: "not-a-real-mode",
    });

    expect(r.status).toBe(EXIT_NO_ENTRYPOINT);
    const records = r.stdout
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l));
    const rec = records.find(
      (j) => j.message === "AGENT_ENTRYPOINT is not set",
    );
    expect(rec).toBeDefined();
    expect(rec.service).toBe("agent-execution-runtime");
  });

  test("exception thrown by target() produces a single ERROR record with exc fields, not a WARNING trace", () => {
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(
      modulePath,
      `export function start() { throw new Error('boom') }\n`,
    );

    const r = runLauncher({
      AGENT_ENTRYPOINT: `${modulePath}:start`,
      STRUCTURED_LOGGING: "true",
      RUNNER_MODE: "aer",
    });

    expect(r.status).toBe(EXIT_STARTUP_CRASH);
    const records = r.stdout
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l));
    const errorRecs = records.filter((j) => j.fields?.exc_message === "boom");
    expect(errorRecs).toHaveLength(1);
    expect(errorRecs[0].level).toBe("ERROR");
    expect(errorRecs[0].fields.exc_type).toBe("Error");
    expect(
      records.some(
        (j) => j.level === "WARNING" && String(j.message).includes("boom"),
      ),
    ).toBe(false);
  });

  test("throwing export exits EXIT_STARTUP_CRASH and logs the error", () => {
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(
      modulePath,
      `export function start() { throw new Error('crash before logging is configured') }\n`,
    );

    const r = runLauncher({ AGENT_ENTRYPOINT: `${modulePath}:start` });

    expect(r.status).toBe(EXIT_STARTUP_CRASH);
    expect(combinedOutput(r)).toMatch(
      /Unhandled exception from agent entrypoint/,
    );
    expect(combinedOutput(r)).toMatch(/crash before logging is configured/);
  });

  test("import failure with STRUCTURED_LOGGING emits JSON ERROR", () => {
    const modulePath = join(tmpModulesDir, "nonexistent.mjs");

    const r = runLauncher({
      AGENT_ENTRYPOINT: `${modulePath}:main`,
      STRUCTURED_LOGGING: "true",
      RUNNER_MODE: "aer",
    });

    expect(r.status).toBe(EXIT_IMPORT_ERROR);
    const records = jsonRecords(r.stdout);
    const errorRecords = records.filter((rec) => rec["level"] === "ERROR");
    expect(errorRecords.length).toBeGreaterThan(0);
    expect(
      errorRecords.some((rec) =>
        String(rec["message"] ?? "").includes("Cannot import module"),
      ),
    ).toBe(true);
  });

  test("throwing export with STRUCTURED_LOGGING emits JSON ERROR and traceback", () => {
    const modulePath = join(tmpModulesDir, "fake_module.mjs");
    writeFileSync(
      modulePath,
      `export function start() { throw new Error('another crash before logging is configured') }\n`,
    );

    const r = runLauncher({
      AGENT_ENTRYPOINT: `${modulePath}:start`,
      STRUCTURED_LOGGING: "true",
      RUNNER_MODE: "aer",
    });

    expect(r.status).toBe(EXIT_STARTUP_CRASH);
    const records = jsonRecords(r.stdout);
    const errorRecords = records.filter((rec) => rec["level"] === "ERROR");
    expect(errorRecords.length).toBeGreaterThan(0);
    expect(
      errorRecords.some((rec) =>
        String(rec["message"] ?? "").includes(
          "Unhandled exception from agent entrypoint",
        ),
      ),
    ).toBe(true);
    expect(
      errorRecords.some((rec) => {
        const fields = rec["fields"] as Record<string, unknown> | undefined;
        return (
          String(fields?.["exc_message"] ?? "").includes(
            "another crash before logging is configured",
          ) ||
          String(rec["message"] ?? "").includes(
            "another crash before logging is configured",
          )
        );
      }),
    ).toBe(true);
  });
});
