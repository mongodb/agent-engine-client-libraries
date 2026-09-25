/**
 * Port-parity tests for AgentEngineToolPodBackend.
 *
 * Covers the backend-method → Tool Pod wire-handler mapping: each method sends
 * the right handler name + arguments through SecureToolWrapper.executeTool, and
 * maps the wire result (or its `{error}`) into the correct deepagents V2 Result
 * type. Topology sanitization and the no-wrapper failure path are also checked.
 *
 * Download fan-out tests pin concurrency, order, per-file error isolation, and
 * the durable sequential fallback. They use a delayed fake wrapper and an
 * in-flight counter rather than wall-clock elapsed time.
 */

import { describe, expect, it } from "vitest";
import {
  runWithAttemptContext,
  runWithExecutionContext,
  TerminalExecutionError,
  ToolCallTimeoutError,
  type AttemptContext,
} from "@mongodb-js/agent-engine-runner-shared";

import { AgentEngineToolPodBackend } from "../src/backends/toolpod.js";

type Call = { tool: string; args: Record<string, unknown> };

/** A fake SecureToolWrapper: records calls, returns a scripted wire result. */
function fakeWrapper(reply: unknown | ((c: Call) => unknown)) {
  const calls: Call[] = [];
  const wrapper = {
    executeTool(tool: string, args: Record<string, unknown>): Promise<unknown> {
      const call = { tool, args };
      calls.push(call);
      const r =
        typeof reply === "function"
          ? (reply as (c: Call) => unknown)(call)
          : reply;
      return Promise.resolve(r);
    },
  };
  return { wrapper, calls };
}

/** Run `fn` with a fake wrapper installed in the execution context. */
async function withWrapper<T>(
  reply: unknown | ((c: Call) => unknown),
  fn: (backend: AgentEngineToolPodBackend, calls: Call[]) => Promise<T>,
): Promise<T> {
  const { wrapper, calls } = fakeWrapper(reply);
  return runWithExecutionContext(
    { executionId: "exec-1", wrapper, oeUrl: "http://oe.test" },
    () => fn(new AgentEngineToolPodBackend(), calls),
  );
}

describe("AgentEngineToolPodBackend wire mapping", () => {
  it("has the sandbox identity", () => {
    expect(new AgentEngineToolPodBackend().id).toBe("agent-engine-toolpod");
  });

  it("ls → filesystem_ls, maps entries to files", async () => {
    await withWrapper(
      {
        entries: [
          { path: "/w/a", is_dir: false },
          { path: "/w/d", is_dir: true },
        ],
      },
      async (backend, calls) => {
        const res = await backend.ls("/w");
        expect(calls[0]).toEqual({
          tool: "filesystem_ls",
          args: { path: "/w" },
        });
        expect(res.files).toEqual([
          { path: "/w/a", is_dir: false },
          { path: "/w/d", is_dir: true },
        ]);
      },
    );
  });

  it("read → filesystem_read with offset/limit, maps content", async () => {
    await withWrapper(
      { content: "hello", encoding: "utf-8" },
      async (backend, calls) => {
        const res = await backend.read("/w/a", 5, 10);
        expect(calls[0]).toEqual({
          tool: "filesystem_read",
          args: { file_path: "/w/a", offset: 5, limit: 10 },
        });
        expect(res.content).toBe("hello");
        expect(res.mimeType).toBe("text/plain");
      },
    );
  });

  it("readRaw wraps read content in FileData", async () => {
    await withWrapper({ content: "raw" }, async (backend, calls) => {
      const res = await backend.readRaw("/w/a");
      expect(calls[0]?.tool).toBe("filesystem_read");
      expect(res.data?.content).toBe("raw");
    });
  });

  it("grep → filesystem_grep, omits null path/glob, maps matches", async () => {
    await withWrapper(
      { matches: [{ path: "/w/a", line: 3, text: "hit" }] },
      async (backend, calls) => {
        const res = await backend.grep("hit");
        expect(calls[0]).toEqual({
          tool: "filesystem_grep",
          args: { pattern: "hit" },
        });
        expect(res.matches).toEqual([{ path: "/w/a", line: 3, text: "hit" }]);
      },
    );
  });

  it("grep forwards path and glob when provided", async () => {
    await withWrapper({ matches: [] }, async (backend, calls) => {
      await backend.grep("x", "/w", "*.ts");
      expect(calls[0]?.args).toEqual({
        pattern: "x",
        path: "/w",
        glob: "*.ts",
      });
    });
  });

  it("glob → filesystem_glob, maps matches to files", async () => {
    await withWrapper(
      { matches: [{ path: "/w/a.ts", is_dir: false }] },
      async (backend, calls) => {
        const res = await backend.glob("*.ts", "/w");
        expect(calls[0]).toEqual({
          tool: "filesystem_glob",
          args: { pattern: "*.ts", path: "/w" },
        });
        expect(res.files).toEqual([{ path: "/w/a.ts", is_dir: false }]);
      },
    );
  });

  it("write → filesystem_write, returns path", async () => {
    await withWrapper({ path: "/w/a" }, async (backend, calls) => {
      const res = await backend.write("/w/a", "body");
      expect(calls[0]).toEqual({
        tool: "filesystem_write",
        args: { file_path: "/w/a", content: "body" },
      });
      expect(res.path).toBe("/w/a");
    });
  });

  it("edit → filesystem_edit, returns occurrences", async () => {
    await withWrapper({ occurrences: 2 }, async (backend, calls) => {
      const res = await backend.edit("/w/a", "old", "new", true);
      expect(calls[0]).toEqual({
        tool: "filesystem_edit",
        args: {
          file_path: "/w/a",
          old_string: "old",
          new_string: "new",
          replace_all: true,
        },
      });
      expect(res.path).toBe("/w/a");
      expect(res.occurrences).toBe(2);
    });
  });

  it("execute → shell_execute, maps output/exitCode/truncated", async () => {
    await withWrapper(
      { output: "ok", exit_code: 0, truncated: true },
      async (backend, calls) => {
        const res = await backend.execute("ls");
        expect(calls[0]).toEqual({
          tool: "shell_execute",
          args: { command: "ls" },
        });
        expect(res).toEqual({ output: "ok", exitCode: 0, truncated: true });
      },
    );
  });

  it("execute surfaces and sanitizes a handler framework error", async () => {
    // shell_execute returns an `error` field alongside a non-zero exit_code and
    // empty output on a framework failure; execute() must surface the
    // (topology-sanitized) message rather than dropping it, and keep the code.
    await withWrapper(
      {
        output: "",
        exit_code: 2,
        truncated: false,
        error: "spawn failed at /opt/bin/sh",
      },
      async (backend) => {
        const res = await backend.execute("sh -c true");
        expect(res.exitCode).toBe(2);
        expect(res.output).not.toContain("/opt/bin/sh");
        expect(res.output).toContain("<path>");
      },
    );
  });

  it("downloadFiles → one filesystem_download per path, base64-decodes", async () => {
    const b64 = Buffer.from("bytes").toString("base64");
    await withWrapper({ content_base64: b64 }, async (backend, calls) => {
      const res = await backend.downloadFiles(["/w/a", "/w/b"]);
      expect(calls.map((c) => c.tool)).toEqual([
        "filesystem_download",
        "filesystem_download",
      ]);
      expect(calls.map((c) => c.args["file_path"]).sort()).toEqual([
        "/w/a",
        "/w/b",
      ]);
      expect(res.map((r) => r.path)).toEqual(["/w/a", "/w/b"]);
      expect(Buffer.from(res[0]?.content ?? new Uint8Array()).toString()).toBe(
        "bytes",
      );
      expect(res[0]?.error).toBeNull();
    });
  });

  it("uploadFiles is not implemented — returns per-file error", async () => {
    await withWrapper({}, async (backend) => {
      const res = await backend.uploadFiles([["/w/a", new Uint8Array()]]);
      expect(res[0]?.path).toBe("/w/a");
      expect(String(res[0]?.error)).toContain("[NON-RETRYABLE]");
    });
  });
});

describe("AgentEngineToolPodBackend error handling", () => {
  it("surfaces a handler {error} on ls", async () => {
    await withWrapper({ error: "bad path" }, async (backend) => {
      const res = await backend.ls("/w");
      expect(res.error).toBe("bad path");
      expect(res.files).toBeUndefined();
    });
  });

  it("sanitizes internal topology from handler errors", async () => {
    await withWrapper(
      { error: "connect db.svc.cluster.local:27017 at /var/run/secret failed" },
      async (backend) => {
        const res = await backend.ls("/w");
        expect(res.error).not.toContain("cluster.local");
        expect(res.error).not.toContain("27017");
        expect(res.error).not.toContain("/var/run/secret");
      },
    );
  });

  it("flags a protocol violation when the shape is wrong", async () => {
    await withWrapper({ notEntries: [] }, async (backend) => {
      const res = await backend.ls("/w");
      expect(res.error).toContain("[NON-RETRYABLE]");
      expect(res.error).toContain("protocol violation");
    });
  });

  it('flags a malformed ls entry rather than coercing to "undefined"', async () => {
    await withWrapper({ entries: [{ is_dir: false }] }, async (backend) => {
      const res = await backend.ls("/w");
      expect(res.error).toContain("[NON-RETRYABLE]");
      expect(res.error).toContain("malformed entry");
      expect(res.files).toBeUndefined();
    });
  });

  it("flags a malformed grep match rather than coercing to NaN", async () => {
    await withWrapper(
      { matches: [{ path: "/w/a", line: "3", text: "hit" }] },
      async (backend) => {
        const res = await backend.grep("hit");
        expect(res.error).toContain("[NON-RETRYABLE]");
        expect(res.error).toContain("malformed match");
        expect(res.matches).toBeUndefined();
      },
    );
  });

  it("flags a malformed glob match", async () => {
    await withWrapper({ matches: [42] }, async (backend) => {
      const res = await backend.glob("*.ts");
      expect(res.error).toContain("[NON-RETRYABLE]");
      expect(res.error).toContain("malformed match");
    });
  });

  it("classifies a missing wrapper as a non-retryable error", async () => {
    // No execution context → getCurrentWrapper() returns null.
    const res = await new AgentEngineToolPodBackend().ls("/w");
    expect(res.error).toContain("[NON-RETRYABLE]");
  });

  it("execute maps a failure into a non-zero exit code", async () => {
    const res = await new AgentEngineToolPodBackend().execute("ls");
    expect(res.exitCode).toBe(1);
    expect(res.output).toContain("[NON-RETRYABLE]");
  });
});

describe("tool-call timeout classification", () => {
  it("reports a timeout as non-retryable rather than a generic failure", async () => {
    // The call already spent its whole deadline, so an automatic retry spends
    // another one to most likely fail the same way.
    await withWrapper(
      () => {
        throw new ToolCallTimeoutError("slow_tool", 600, 601);
      },
      async (backend) => {
        const res = await backend.ls("/w");
        expect(res.error).toContain("[NON-RETRYABLE]");
        expect(res.error).toContain("timed out");
      },
    );
  });
});

describe("terminal execution classification", () => {
  it("reports a terminal execution rejection as non-retryable", async () => {
    await withWrapper(
      () => {
        throw new TerminalExecutionError(
          "This execution already ended in error; later tool calls are rejected.",
        );
      },
      async (backend) => {
        const res = await backend.ls("/w");
        expect(res.error).toContain("[NON-RETRYABLE]");
        expect(res.error).toContain("already ended");
      },
    );
  });
});

// ---------------------------------------------------------------------------
// Download fan-out (native parallel, durable sequential)
// ---------------------------------------------------------------------------

const FANOUT_PATHS = ["a.txt", "b.txt", "c.txt", "d.txt"];
const FANOUT_DELAY_MS = 50;

type InFlight = { current: number; max: number };

function b64(content: string): string {
  return Buffer.from(content).toString("base64");
}

function textOf(content: Uint8Array | null | undefined): string {
  return Buffer.from(content ?? new Uint8Array()).toString();
}

function fakeAttempt(): AttemptContext {
  return { attemptId: "attempt-1" } as AttemptContext;
}

async function withDelayedWrapper<T>(
  delayMs: number | Record<string, number>,
  reply: (path: string) => unknown,
  fn: (
    backend: AgentEngineToolPodBackend,
    inflight: InFlight,
    calls: Call[],
  ) => Promise<T>,
): Promise<T> {
  const inflight: InFlight = { current: 0, max: 0 };
  const calls: Call[] = [];
  const wrapper = {
    async executeTool(
      tool: string,
      args: Record<string, unknown>,
    ): Promise<unknown> {
      const path = String(args["file_path"]);
      calls.push({ tool, args });
      inflight.current += 1;
      inflight.max = Math.max(inflight.max, inflight.current);
      const ms = typeof delayMs === "number" ? delayMs : delayMs[path];
      try {
        await new Promise((resolve) => setTimeout(resolve, ms ?? 0));
        return reply(path);
      } finally {
        inflight.current -= 1;
      }
    },
  };
  return runWithExecutionContext(
    { executionId: "exec-1", wrapper, oeUrl: "http://oe.test" },
    () => fn(new AgentEngineToolPodBackend(), inflight, calls),
  );
}

describe("downloadFiles fan-out", () => {
  it("fans out concurrent filesystem_download calls", async () => {
    await withDelayedWrapper(
      FANOUT_DELAY_MS,
      (path) => ({ content_base64: b64(path) }),
      async (backend, inflight) => {
        const results = await backend.downloadFiles(FANOUT_PATHS);
        expect(inflight.max).toBeGreaterThan(1);
        expect(results.map((r) => r.path)).toEqual(FANOUT_PATHS);
        expect(results.map((r) => textOf(r.content))).toEqual(FANOUT_PATHS);
        expect(results.every((r) => r.error === null)).toBe(true);
      },
    );
  });

  it("preserves input order when a slow path finishes last", async () => {
    await withDelayedWrapper(
      { "slow.txt": 80, "fast.txt": 10 },
      (path) => ({ content_base64: b64(path) }),
      async (backend) => {
        const results = await backend.downloadFiles(["slow.txt", "fast.txt"]);
        expect(results.map((r) => r.path)).toEqual(["slow.txt", "fast.txt"]);
        expect(results.map((r) => textOf(r.content))).toEqual([
          "slow.txt",
          "fast.txt",
        ]);
      },
    );
  });

  it("isolates per-file errors", async () => {
    const replies: Record<string, unknown> = {
      "ok.txt": { content_base64: b64("ok") },
      "missing.txt": { error: "file_not_found" },
      "bad.txt": { unexpected: "shape" },
    };
    await withDelayedWrapper(
      10,
      (path) => replies[path],
      async (backend) => {
        const results = await backend.downloadFiles([
          "ok.txt",
          "missing.txt",
          "bad.txt",
        ]);
        expect(textOf(results[0]?.content)).toBe("ok");
        expect(results[0]?.error).toBeNull();
        expect(results[1]?.content).toBeNull();
        expect(results[1]?.error).toBe("file_not_found");
        expect(results[2]?.content).toBeNull();
        expect(String(results[2]?.error)).toContain("content_base64");
      },
    );
  });

  it("stays sequential under a durable attempt", async () => {
    await withDelayedWrapper(
      30,
      (path) => ({ content_base64: b64(path) }),
      async (backend, inflight) => {
        const results = await runWithAttemptContext(fakeAttempt(), () =>
          backend.downloadFiles(FANOUT_PATHS),
        );
        expect(inflight.max).toBe(1);
        expect(results.map((r) => r.path)).toEqual(FANOUT_PATHS);
        expect(results.every((r) => r.error === null)).toBe(true);
      },
    );
  });

  it("returns [] for an empty path list without calling the wrapper", async () => {
    await withWrapper({ content_base64: b64("x") }, async (backend, calls) => {
      expect(await backend.downloadFiles([])).toEqual([]);
      expect(calls).toEqual([]);
    });
  });

  it("stays sequential for a single path", async () => {
    await withDelayedWrapper(
      20,
      (path) => ({ content_base64: b64(path) }),
      async (backend, inflight) => {
        const results = await backend.downloadFiles(["only.txt"]);
        expect(inflight.max).toBe(1);
        expect(results).toHaveLength(1);
        expect(textOf(results[0]?.content)).toBe("only.txt");
        expect(results[0]?.error).toBeNull();
      },
    );
  });

  it("records exactly one approval request per file", async () => {
    await withDelayedWrapper(
      10,
      (path) => ({ content_base64: b64(path) }),
      async (backend, _inflight, calls) => {
        const paths = ["c.bin", "a.bin", "b.bin"];
        const results = await backend.downloadFiles(paths);
        expect(results.map((r) => r.path)).toEqual(paths);
        expect(calls).toHaveLength(paths.length);
        expect(calls.every((c) => c.tool === "filesystem_download")).toBe(true);
        expect(calls.map((c) => c.args["file_path"]).sort()).toEqual(
          [...paths].sort(),
        );
      },
    );
  });
});
