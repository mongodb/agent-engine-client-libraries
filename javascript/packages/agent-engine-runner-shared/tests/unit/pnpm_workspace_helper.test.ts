import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  test,
} from "vitest";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Dockerfile.runner-base-typescript-langgraph copies scripts/ to
// /opt/agent-engine-runner-shared/bin/, so these tests run the files the
// image ships.
const packageRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const scriptsDir = join(packageRoot, "scripts");
const requireScript = createRequire(import.meta.url);

interface PackageManifest {
  name?: string;
  version?: string;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
}

function present(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

const helper = join(scriptsDir, "agentengine-pnpm-workspace.sh");

function writeJSON(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value));
}

describe("agentengine-sdk.pnpmfile.cjs", () => {
  const pnpmfile = requireScript(
    join(scriptsDir, "agentengine-sdk.pnpmfile.cjs"),
  ) as {
    hooks: { readPackage(pkg: PackageManifest): PackageManifest };
    loadSdkSpecs(distDir: string | undefined): Map<string, string>;
    createReadPackage(
      sdkSpecs: Map<string, string>,
      agentPackage: string | undefined,
    ): (pkg: PackageManifest) => PackageManifest;
  };
  const sdkSpecs = new Map([
    [
      "@mongodb-js/agent-engine-sdk",
      "file:/opt/dist/mongodb-js-agent-engine-sdk-9.9.9.tgz",
    ],
    [
      "@mongodb-js/agent-engine-runner-shared",
      "file:/opt/dist/mongodb-js-agent-engine-runner-shared-9.9.9.tgz",
    ],
  ]);

  test("pins the SDK through the hook pnpm loads, configured by the helper's environment", () => {
    const dist = mkdtempSync(join(tmpdir(), "pnpmfile-dist-"));
    try {
      writeFileSync(
        join(dist, "sdk-packages.json"),
        JSON.stringify([
          {
            name: "@mongodb-js/agent-engine-sdk",
            version: "9.9.9",
            filename: "mongodb-js-agent-engine-sdk-9.9.9.tgz",
          },
          {
            name: "@mongodb-js/agent-engine-runner-shared",
            version: "9.9.9",
            filename: "mongodb-js-agent-engine-runner-shared-9.9.9.tgz",
          },
        ]),
      );
      // A fresh process loads the pnpmfile as pnpm does, with the variables
      // agentengine-pnpm-workspace.sh exports for every pnpm call.
      const result = spawnSync(
        "node",
        [
          "-e",
          "const { hooks } = require(process.argv[1]);" +
            "process.stdout.write(JSON.stringify(hooks.readPackage(JSON.parse(process.argv[2]))));",
          join(scriptsDir, "agentengine-sdk.pnpmfile.cjs"),
          JSON.stringify({
            name: "my-agent",
            dependencies: { "@repo/lib": "workspace:*" },
            devDependencies: { "@mongodb-js/agent-engine-sdk": "0.11.6" },
          }),
        ],
        {
          encoding: "utf8",
          env: {
            ...process.env,
            AGENTIC_SDK_DIST_DIR: dist,
            AGENTIC_AGENT_PACKAGE: "my-agent",
          },
        },
      );
      expect(result.status, result.stderr).toBe(0);
      const pkg = JSON.parse(result.stdout) as PackageManifest;
      expect(pkg.dependencies).toEqual({
        "@repo/lib": "workspace:*",
        "@mongodb-js/agent-engine-sdk": `file:${join(dist, "mongodb-js-agent-engine-sdk-9.9.9.tgz")}`,
        "@mongodb-js/agent-engine-runner-shared": `file:${join(dist, "mongodb-js-agent-engine-runner-shared-9.9.9.tgz")}`,
      });
      expect(pkg.devDependencies).toEqual({});
    } finally {
      rmSync(dist, { recursive: true, force: true });
    }
  });

  test("maps each packed SDK name to its tarball in the dist directory", () => {
    const dist = mkdtempSync(join(tmpdir(), "pnpmfile-dist-"));
    try {
      writeFileSync(
        join(dist, "sdk-packages.json"),
        JSON.stringify([
          {
            name: "@mongodb-js/agent-engine-sdk",
            version: "9.9.9",
            filename: "mongodb-js-agent-engine-sdk-9.9.9.tgz",
          },
        ]),
      );
      expect(Object.fromEntries(pnpmfile.loadSdkSpecs(dist))).toEqual({
        "@mongodb-js/agent-engine-sdk": `file:${join(dist, "mongodb-js-agent-engine-sdk-9.9.9.tgz")}`,
      });
    } finally {
      rmSync(dist, { recursive: true, force: true });
    }
  });

  test("pins every SDK dependency of a workspace package to the image tarball", () => {
    const readPackage = pnpmfile.createReadPackage(sdkSpecs, "my-agent");
    const pkg = readPackage({
      name: "@repo/core",
      dependencies: { "@mongodb-js/agent-engine-sdk": "0.11.6", zod: "^4.4.3" },
      devDependencies: { "@mongodb-js/agent-engine-runner-shared": "0.11.6" },
      optionalDependencies: { "@mongodb-js/agent-engine-sdk": "0.11.6" },
      peerDependencies: { "@mongodb-js/agent-engine-sdk": ">=0.11" },
    });
    expect(pkg.dependencies).toEqual({
      "@mongodb-js/agent-engine-sdk": sdkSpecs.get(
        "@mongodb-js/agent-engine-sdk",
      ),
      zod: "^4.4.3",
    });
    expect(pkg.devDependencies).toEqual({
      "@mongodb-js/agent-engine-runner-shared": sdkSpecs.get(
        "@mongodb-js/agent-engine-runner-shared",
      ),
    });
    expect(pkg.optionalDependencies).toEqual({
      "@mongodb-js/agent-engine-sdk": sdkSpecs.get(
        "@mongodb-js/agent-engine-sdk",
      ),
    });
    // The dependency entry already pins it, so the peer entry goes.
    expect(pkg.peerDependencies).toEqual({});
  });

  test("turns a peer-only SDK declaration into a dependency on the image tarball", () => {
    const readPackage = pnpmfile.createReadPackage(sdkSpecs, "my-agent");
    const pkg = readPackage({
      name: "@repo/lib",
      devDependencies: { "@mongodb-js/agent-engine-sdk": "0.10.2" },
      peerDependencies: {
        "@mongodb-js/agent-engine-sdk": "^0.10.0",
        zod: "^4",
      },
      peerDependenciesMeta: {
        "@mongodb-js/agent-engine-sdk": { optional: true },
      },
    } as PackageManifest);
    // pnpm would install a registry copy for an unmet peer range, and the
    // production install leaves devDependencies out.
    expect(pkg.dependencies).toEqual({
      "@mongodb-js/agent-engine-sdk": sdkSpecs.get(
        "@mongodb-js/agent-engine-sdk",
      ),
    });
    expect(pkg.devDependencies).toEqual({});
    expect(pkg.peerDependencies).toEqual({ zod: "^4" });
    expect(
      (pkg as { peerDependenciesMeta?: Record<string, unknown> })
        .peerDependenciesMeta,
    ).toEqual({});
  });

  test("makes every SDK package a runtime dependency of the agent", () => {
    const readPackage = pnpmfile.createReadPackage(sdkSpecs, "my-agent");
    const pkg = readPackage({
      name: "my-agent",
      dependencies: { "@repo/lib": "workspace:*" },
      devDependencies: {
        "@mongodb-js/agent-engine-sdk": "0.11.6",
        vitest: "^4",
      },
    });
    expect(pkg.dependencies).toEqual({
      "@repo/lib": "workspace:*",
      "@mongodb-js/agent-engine-sdk": sdkSpecs.get(
        "@mongodb-js/agent-engine-sdk",
      ),
      "@mongodb-js/agent-engine-runner-shared": sdkSpecs.get(
        "@mongodb-js/agent-engine-runner-shared",
      ),
    });
    // Left in devDependencies, the production prune would drop it.
    expect(pkg.devDependencies).toEqual({ vitest: "^4" });
  });
});

describe("agentengine-workspace-mirror.cjs", () => {
  const mirrorScript = join(scriptsDir, "agentengine-workspace-mirror.cjs");
  const { mirrorTree } = requireScript(mirrorScript) as {
    mirrorTree(src: string, dest: string): void;
  };
  let root: string;
  let src: string;
  let dest: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "workspace-mirror-"));
    src = join(root, "src");
    dest = join(root, "dest");
    mkdirSync(join(src, "agents", "a", "node_modules", "host-only"), {
      recursive: true,
    });
    mkdirSync(join(src, "agents", "a", "dist"), { recursive: true });
    mkdirSync(join(src, ".git"), { recursive: true });
    writeFileSync(join(src, "pnpm-lock.yaml"), "lockfileVersion: '9.0'\n");
    writeFileSync(join(src, "agents", "a", "main.ts"), "one\n");
    writeFileSync(join(src, "agents", "a", ".env"), "KEY=value\n");
    writeFileSync(join(src, "agents", "a", "dist", "stale.js"), "host build\n");
    writeFileSync(join(src, ".git", "HEAD"), "ref\n");
    symlinkSync("agents/a/main.ts", join(src, "link.ts"));
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  test("copies files, links and dist output, and never copies dependencies, VCS or Python environments", () => {
    mkdirSync(join(src, ".venv-aer-tool", "lib"), { recursive: true });
    writeFileSync(join(src, ".venv-aer-tool", "lib", "site.py"), "x\n");
    mkdirSync(join(src, "packages", "py", "__pycache__"), { recursive: true });
    writeFileSync(join(src, "packages", "py", "__pycache__", "m.pyc"), "x");
    writeFileSync(join(src, "packages", "py", "m.pyc"), "x");
    mirrorTree(src, dest);
    expect(readFileSync(join(dest, "agents", "a", "main.ts"), "utf8")).toBe(
      "one\n",
    );
    expect(readFileSync(join(dest, "agents", "a", ".env"), "utf8")).toBe(
      "KEY=value\n",
    );
    expect(readlinkSync(join(dest, "link.ts"))).toBe("agents/a/main.ts");
    // dist can be committed output that a package runs from.
    expect(
      readFileSync(join(dest, "agents", "a", "dist", "stale.js"), "utf8"),
    ).toBe("host build\n");
    expect(present(join(dest, "agents", "a", "node_modules"))).toBe(false);
    expect(present(join(dest, ".git"))).toBe(false);
    expect(present(join(dest, ".venv-aer-tool"))).toBe(false);
    expect(present(join(dest, "packages", "py", "__pycache__"))).toBe(false);
    expect(present(join(dest, "packages", "py", "m.pyc"))).toBe(false);
  });

  test("follows edits, adds and deletes, dist included, and keeps the copy's own node_modules", () => {
    mirrorTree(src, dest);
    mkdirSync(join(dest, "agents", "a", "node_modules", "container-only"), {
      recursive: true,
    });
    mkdirSync(join(dest, "agents", "a", "dist"), { recursive: true });
    writeFileSync(
      join(dest, "agents", "a", "dist", "main.js"),
      "container build\n",
    );
    writeFileSync(join(dest, "pnpm-lock.yaml"), "rewritten by pnpm\n");

    writeFileSync(join(src, "agents", "a", "main.ts"), "two\n");
    // Same size as before, so only the mtime tells the edit apart.
    utimesSync(
      join(src, "agents", "a", "main.ts"),
      new Date(),
      new Date(Date.now() + 5000),
    );
    writeFileSync(join(src, "agents", "a", "added.ts"), "new\n");
    rmSync(join(src, "link.ts"));
    mirrorTree(src, dest);

    expect(readFileSync(join(dest, "agents", "a", "main.ts"), "utf8")).toBe(
      "two\n",
    );
    expect(readFileSync(join(dest, "agents", "a", "added.ts"), "utf8")).toBe(
      "new\n",
    );
    expect(present(join(dest, "link.ts"))).toBe(false);
    expect(readFileSync(join(dest, "pnpm-lock.yaml"), "utf8")).toBe(
      "lockfileVersion: '9.0'\n",
    );
    expect(
      existsSync(join(dest, "agents", "a", "node_modules", "container-only")),
    ).toBe(true);
    // The copy's dist follows the source; the next build writes it again.
    expect(present(join(dest, "agents", "a", "dist", "main.js"))).toBe(false);
    expect(
      readFileSync(join(dest, "agents", "a", "dist", "stale.js"), "utf8"),
    ).toBe("host build\n");
  });

  test("copies an edit that keeps the file's size and mtime", () => {
    mirrorTree(src, dest);
    const main = join(src, "agents", "a", "main.ts");
    const { atime, mtime } = lstatSync(main);
    writeFileSync(main, "two\n");
    utimesSync(main, atime, mtime);
    mirrorTree(src, dest);
    expect(readFileSync(join(dest, "agents", "a", "main.ts"), "utf8")).toBe(
      "two\n",
    );
  });

  test("refuses a destination inside the source whose name starts with ..", () => {
    const result = spawnSync("node", [mirrorScript, src, join(src, "..work")], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("must not contain each other");
    expect(present(join(src, "..work"))).toBe(false);
  });

  test("refuses a source inside the destination whose name starts with .., and leaves it intact", () => {
    const run = join(root, "run");
    const inner = join(run, "..src");
    mkdirSync(inner, { recursive: true });
    writeFileSync(join(inner, "main.ts"), "one\n");
    const result = spawnSync("node", [mirrorScript, inner, run], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(readFileSync(join(inner, "main.ts"), "utf8")).toBe("one\n");
  });

  test("refuses a destination inside the source", () => {
    const result = spawnSync("node", [mirrorScript, src, join(src, "copy")], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("must not contain each other");
    expect(present(join(src, "copy"))).toBe(false);
  });

  test("refuses a destination that contains the source, and leaves the source intact", () => {
    const result = spawnSync("node", [mirrorScript, src, root], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(readFileSync(join(src, "agents", "a", "main.ts"), "utf8")).toBe(
      "one\n",
    );
  });

  test("refuses a destination that is a symlink to a parent of the source, and leaves the source intact", () => {
    const link = join(root, "run");
    symlinkSync(root, link);
    const result = spawnSync("node", [mirrorScript, src, link], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("must not contain each other");
    expect(readFileSync(join(src, "agents", "a", "main.ts"), "utf8")).toBe(
      "one\n",
    );
  });

  test("refuses a destination not yet created below a symlink into the source", () => {
    const link = join(root, "run");
    symlinkSync(src, link);
    const result = spawnSync("node", [mirrorScript, src, join(link, "copy")], {
      encoding: "utf8",
    });
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("must not contain each other");
    expect(present(join(src, "copy"))).toBe(false);
  });
});

describe("agentengine-pnpm-workspace.sh", () => {
  // A stand-in pnpm that records each call and answers `ls` with
  // PNPM_STUB_LS, so the helper's decisions are checked without a registry.
  // Each install copies PNPM_STUB_INSTALLS into the workspace, standing in
  // for what pnpm installs. The production install then writes
  // PNPM_STUB_MODULES, when set, as the node_modules/.modules.yaml that
  // records pnpm's pending builds.
  const stubPnpm = `#!/usr/bin/env node
const fs = require("node:fs");
const args = process.argv.slice(2);
const env = {};
for (const key of [
  "AGENTIC_AGENT_PACKAGE",
  "AGENTIC_SDK_DIST_DIR",
  "pnpm_config_global_pnpmfile",
  "pnpm_config_store_dir",
  "pnpm_config_pm_on_fail",
  "pnpm_config_verify_deps_before_run",
  "pnpm_config_strict_dep_builds",
  "pnpm_config_update_notifier",
  "pnpm_config_virtual_store_type",
  "pnpm_config_virtual_store_dir",
  "pnpm_config_node_linker",
]) {
  env[key] = process.env[key];
}
fs.appendFileSync(process.env.PNPM_STUB_LOG, JSON.stringify({ args, cwd: process.cwd(), env }) + "\\n");
if (args[0] === "install" && process.env.PNPM_STUB_INSTALLS) {
  fs.cpSync(process.env.PNPM_STUB_INSTALLS, process.cwd(), { recursive: true });
}
if (args[0] === "install" && args.includes("--prod") && process.env.PNPM_STUB_MODULES !== undefined) {
  fs.mkdirSync("node_modules", { recursive: true });
  fs.writeFileSync("node_modules/.modules.yaml", process.env.PNPM_STUB_MODULES);
}
if (args[0] === "--version") process.stdout.write("11.25.0\\n");
if (args.includes("ls")) process.stdout.write(process.env.PNPM_STUB_LS || "[]");
if (process.env.PNPM_STUB_FAIL && args.includes(process.env.PNPM_STUB_FAIL)) process.exit(1);
`;
  let root: string;
  let app: string;
  let bin: string;
  let dist: string;
  let log: string;
  let installs: string;

  beforeEach(() => {
    // The helper resolves real paths (pwd -P), so the fixture does too.
    root = realpathSync(mkdtempSync(join(tmpdir(), "pnpm-helper-")));
    app = join(root, "app");
    bin = join(root, "bin");
    dist = join(root, "dist");
    log = join(root, "pnpm.log");
    installs = join(root, "installs");
    mkdirSync(installs);
    mkdirSync(bin);
    mkdirSync(dist);
    writeFileSync(join(bin, "pnpm"), stubPnpm);
    chmodSync(join(bin, "pnpm"), 0o755);
    writeFileSync(join(dist, "sdk-packages.json"), "[]");
    writeFileSync(log, "");
    writeJSON(join(app, "package.json"), { name: "@repo/root", private: true });
    writeFileSync(
      join(app, "pnpm-workspace.yaml"),
      'packages:\n  - "agents/*"\n',
    );
    writeJSON(join(app, "agents", "a", "package.json"), {
      name: "my-agent",
      dependencies: { "@repo/lib": "workspace:*" },
    });
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  function runHelper(
    args: string[],
    ls: unknown[] = [],
    extraEnv: Record<string, string> = {},
  ) {
    const result = spawnSync("bash", [helper, ...args], {
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: `${bin}:${process.env["PATH"] ?? ""}`,
        PNPM_STUB_LOG: log,
        PNPM_STUB_INSTALLS: installs,
        PNPM_STUB_LS: JSON.stringify(ls),
        AGENTIC_SDK_DIST_DIR: dist,
        AGENTIC_PNPM_STORE_DIR: join(root, "store"),
        ...extraEnv,
      },
    });
    const calls = readFileSync(log, "utf8")
      .split("\n")
      .filter(Boolean)
      .map(
        (line) =>
          JSON.parse(line) as {
            args: string[];
            cwd: string;
            env: Record<string, string>;
          },
      );
    return {
      status: result.status,
      stdout: result.stdout,
      stderr: result.stderr,
      calls,
    };
  }

  const listing = (dir: string) => [{ name: "my-agent", path: dir }];

  // Where to put a file so that the stand-in pnpm installs it at path, a
  // path in the workspace (app, or the copy the dev command builds in).
  function installed(path: string): string {
    for (const base of [app, join(root, "work")]) {
      if (path.startsWith(`${base}/`)) {
        return join(installs, relative(base, path));
      }
    }
    throw new Error(`${path} is not in a workspace`);
  }

  function addSdkPackage(agentDir: string, version: string): void {
    writeJSON(
      installed(
        join(
          agentDir,
          "node_modules",
          "@mongodb-js",
          "agent-engine-sdk",
          "package.json",
        ),
      ),
      { name: "@mongodb-js/agent-engine-sdk", version },
    );
  }

  const imageTarball = "the image's SDK tarball";
  const integrityOf = (bytes: string) =>
    `sha512-${createHash("sha512").update(bytes).digest("base64")}`;

  // The packages pnpm records that it installed, in the
  // node_modules/.pnpm/lock.yaml that the stand-in pnpm installs.
  function writeInstalledPackages(
    packages: Array<{ key: string; integrity: string }>,
  ): void {
    const lock = installed(join(app, "node_modules", ".pnpm", "lock.yaml"));
    mkdirSync(dirname(lock), { recursive: true });
    writeFileSync(
      lock,
      [
        "lockfileVersion: '9.0'",
        "",
        "packages:",
        "",
        ...packages.flatMap((pkg) => [
          `  '${pkg.key}':`,
          `    resolution: {integrity: ${pkg.integrity}, tarball: file:x.tgz}`,
          "",
        ]),
      ].join("\n"),
    );
  }

  // The manifest and tarball the image stages, and pnpm's record of
  // installing that tarball.
  function writeSdkManifest(version: string): void {
    writeFileSync(join(dist, "x.tgz"), imageTarball);
    writeJSON(join(dist, "sdk-packages.json"), [
      { name: "@mongodb-js/agent-engine-sdk", version, filename: "x.tgz" },
    ]);
    writeInstalledPackages([
      {
        key: "@mongodb-js/agent-engine-sdk@file:x.tgz",
        integrity: integrityOf(imageTarball),
      },
    ]);
  }

  function addLauncher(agentDir: string): void {
    const launcher = installed(
      join(
        agentDir,
        "node_modules",
        "@mongodb-js",
        "agent-engine-runner-shared",
        "dist",
        "launcher.js",
      ),
    );
    mkdirSync(dirname(launcher), { recursive: true });
    writeFileSync(launcher, "");
  }

  test("detect prints the workspace root for a package of the workspace", () => {
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(app);
    expect(result.calls.map((c) => c.args)).toEqual([
      ["--filter", "my-agent", "ls", "--depth", "-1", "--json"],
    ]);
    expect(result.calls[0]?.cwd).toBe(app);
  });

  test("detect runs the membership check on the image's pnpm, warning on a packageManager pin and skipping the update check", () => {
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.calls[0]?.env["pnpm_config_pm_on_fail"]).toBe("warn");
    expect(result.calls[0]?.env["pnpm_config_update_notifier"]).toBe("false");
  });

  test("detect uses the nearest pnpm-workspace.yaml above the agent", () => {
    writeFileSync(
      join(app, "agents", "pnpm-workspace.yaml"),
      'packages:\n  - "*"\n',
    );
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(join(app, "agents"));
  });

  test("detect exits 1 when pnpm does not list the agent", () => {
    expect(runHelper(["detect", app, "./agents/a"], []).status).toBe(1);
  });

  // Each case gets a fresh fixture from beforeEach, so one case's file
  // cannot be what makes another case pass.
  test.each([
    ["the agent commits a package-lock.json", "package-lock.json", "{}"],
    [
      "the agent is its own workspace root",
      "pnpm-workspace.yaml",
      "packages: []\n",
    ],
  ])("detect exits 1, without asking pnpm, when %s", (_name, file, content) => {
    writeFileSync(join(app, "agents", "a", file), content);
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(1);
    expect(result.calls).toEqual([]);
  });

  // Only a dependency npm cannot install, one using the workspace: or
  // catalog: protocol, needs pnpm. Everything else keeps the npm path.
  test.each([
    [
      "only registry dependencies",
      '{"name":"my-agent","dependencies":{"is-number":"7.0.0"}}',
    ],
    ["no dependencies at all", '{"name":"my-agent"}'],
    ["an unparseable package.json", "{ not json"],
  ])(
    "detect exits 1, without asking pnpm, when the agent has %s",
    (_name, contents) => {
      writeFileSync(join(app, "agents", "a", "package.json"), contents);
      const result = runHelper(
        ["detect", app, "./agents/a"],
        listing(join(app, "agents", "a")),
      );
      expect(result.status).toBe(1);
      expect(result.calls).toEqual([]);
    },
  );

  test("detect takes the pnpm path when a devDependency uses the catalog: protocol", () => {
    writeJSON(join(app, "agents", "a", "package.json"), {
      name: "my-agent",
      devDependencies: { "@repo/lib": "catalog:" },
    });
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(app);
  });

  test("detect takes the pnpm path when an optionalDependency uses the workspace: protocol", () => {
    writeJSON(join(app, "agents", "a", "package.json"), {
      name: "my-agent",
      optionalDependencies: { "@repo/lib": "workspace:*" },
    });
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(app);
  });

  test("detect exits 1, without asking pnpm, when no pnpm-workspace.yaml sits above the agent", () => {
    rmSync(join(app, "pnpm-workspace.yaml"));
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(1);
    expect(result.calls).toEqual([]);
  });

  test("detect exits 1 for an agent at the source root", () => {
    expect(runHelper(["detect", app, "."]).status).toBe(1);
  });

  // The workspace search stops at the source root, so a pnpm-workspace.yaml
  // above it never decides the install path. Each row returns the install
  // target, and pnpm would list that target as a member if it were asked.
  test.each([
    [
      "the only pnpm-workspace.yaml is above the source root",
      () => {
        renameSync(
          join(app, "pnpm-workspace.yaml"),
          join(root, "pnpm-workspace.yaml"),
        );
        return "./agents/a";
      },
    ],
    [
      "the agent is the source root and a pnpm-workspace.yaml sits above it",
      () => {
        writeJSON(join(app, "package.json"), {
          name: "@repo/root",
          private: true,
          dependencies: { "@repo/lib": "workspace:*" },
        });
        renameSync(
          join(app, "pnpm-workspace.yaml"),
          join(root, "pnpm-workspace.yaml"),
        );
        return ".";
      },
    ],
  ])("detect exits 1, without asking pnpm, when %s", (_name, arrange) => {
    const target = arrange();
    const result = runHelper(
      ["detect", app, target],
      listing(resolve(app, target)),
    );
    expect(result.status).toBe(1);
    expect(result.calls).toEqual([]);
  });

  test("detect fails, rather than choosing npm, for an install target outside the source", () => {
    const result = runHelper(["detect", app, "../"]);
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("is outside");
  });

  test("detect fails when the agent package has no name", () => {
    writeJSON(join(app, "agents", "a", "package.json"), {
      version: "1.0.0",
      dependencies: { "@repo/lib": "workspace:*" },
    });
    const result = runHelper(["detect", app, "./agents/a"]);
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('needs a "name"');
  });

  test("detect fails, rather than choosing npm, when pnpm cannot list the workspace", () => {
    const result = runHelper(
      ["detect", app, "./agents/a"],
      listing(join(app, "agents", "a")),
      { PNPM_STUB_FAIL: "ls" },
    );
    expect(result.status).toBe(2);
  });

  test("detect fails when pnpm returns an unreadable project list", () => {
    const result = runHelper(["detect", app, "./agents/a"], [], {
      PNPM_STUB_LS: "not json",
    });
    expect(result.status).toBe(2);
  });

  test("detect fails when pnpm returns a project list that is not an array", () => {
    const result = runHelper(["detect", app, "./agents/a"], [], {
      PNPM_STUB_LS: "{}",
    });
    expect(result.status).toBe(2);
  });

  test("install builds the workspace dependencies, then the agent, then reinstalls its production closure", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.calls.map((c) => c.args)).toEqual([
      ["--filter", "my-agent", "ls", "--depth", "-1", "--json"],
      ["--version"],
      [
        "install",
        "--no-frozen-lockfile",
        "--filter",
        "my-agent...",
        "--config.confirmModulesPurge=false",
      ],
      ["--filter", "my-agent^...", "run", "--if-present", "build"],
      ["--filter", "my-agent", "run", "build"],
      [
        "install",
        "--prod",
        "--frozen-lockfile",
        "--ignore-scripts",
        "--filter-prod",
        "my-agent...",
        "--config.confirmModulesPurge=false",
      ],
    ]);
    for (const call of result.calls.slice(1)) {
      expect(call.cwd).toBe(app);
      expect(call.env).toEqual({
        AGENTIC_AGENT_PACKAGE: "my-agent",
        AGENTIC_SDK_DIST_DIR: dist,
        pnpm_config_global_pnpmfile: join(
          scriptsDir,
          "agentengine-sdk.pnpmfile.cjs",
        ),
        pnpm_config_store_dir: join(root, "store"),
        pnpm_config_pm_on_fail: "warn",
        pnpm_config_verify_deps_before_run: "false",
        pnpm_config_strict_dep_builds: "false",
        pnpm_config_update_notifier: "false",
        pnpm_config_virtual_store_type: "project",
        pnpm_config_virtual_store_dir: "node_modules/.pnpm",
        pnpm_config_node_linker: "isolated",
      });
    }
  });

  test("install rebuilds the dependencies the production install left unbuilt, and no workspace package", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
      {
        PNPM_STUB_MODULES: JSON.stringify({
          pendingBuilds: [
            "agents/a",
            "native@file:../ext/native",
            "@scope/addon@1.2.3",
            "packages/@scope/lib",
          ],
        }),
      },
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.calls.at(-1)?.args).toEqual([
      "--filter-prod",
      "my-agent...",
      "rebuild",
      "native",
      "@scope/addon",
    ]);
  });

  test("install rebuilds nothing when no dependency build is pending", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
      { PNPM_STUB_MODULES: JSON.stringify({ pendingBuilds: ["agents/a"] }) },
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.calls.some((c) => c.args.includes("rebuild"))).toBe(false);
  });

  test("install fails when pnpm's pending builds cannot be read", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
      { PNPM_STUB_MODULES: "pendingBuilds: [not json" },
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("pending builds");
  });

  test("install --prod removes what earlier installs left in node_modules before the production install", () => {
    addLauncher(join(app, "agents", "a"));
    // A devDependency's tree from an earlier install, at the root and in a
    // workspace package.
    writeJSON(
      join(
        app,
        "node_modules",
        ".pnpm",
        "dev-tool@1.0.0",
        "node_modules",
        "dev-tool",
        "package.json",
      ),
      {
        name: "dev-tool",
      },
    );
    writeJSON(
      join(app, "packages", "tool", "node_modules", "tool-dep", "package.json"),
      { name: "tool-dep" },
    );
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(present(join(app, "node_modules", ".pnpm", "dev-tool@1.0.0"))).toBe(
      false,
    );
    expect(present(join(app, "packages", "tool", "node_modules"))).toBe(false);
  });

  test("install fails before installing when the workspace pins another pnpm major", () => {
    writeJSON(join(app, "package.json"), {
      name: "@repo/root",
      private: true,
      packageManager: "pnpm@10.4.1",
    });
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("pins pnpm@10.4.1 in packageManager");
    expect(result.calls.some((c) => c.args[0] === "install")).toBe(false);
  });

  test("install fails before installing when the workspace pins another package manager", () => {
    writeJSON(join(app, "package.json"), {
      name: "@repo/root",
      private: true,
      packageManager: "npm@10.9.0",
    });
    const result = runHelper(
      ["build", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("pins npm@10.9.0 in packageManager");
    expect(result.calls.some((c) => c.args[0] === "install")).toBe(false);
  });

  test("install accepts a pin of another pnpm 11 version", () => {
    addLauncher(join(app, "agents", "a"));
    writeJSON(join(app, "package.json"), {
      name: "@repo/root",
      private: true,
      packageManager: "pnpm@11.2.2",
    });
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
  });

  test("install fails when the runner launcher is missing afterwards", () => {
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("runner launcher is missing");
  });

  test("install accepts an agent whose SDK packages resolve at the image's version", () => {
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    writeSdkManifest("9.9.9");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
  });

  test("install fails when an overrides entry leaves the agent on a registry SDK copy", () => {
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "0.11.6");
    writeSdkManifest("9.9.9");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("@mongodb-js/agent-engine-sdk");
    expect(result.stderr).toContain("overrides");
  });

  test("install fails when a scoped override gives a workspace dependency a second SDK copy", () => {
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    writeSdkManifest("9.9.9");
    const pnpmCopy = join(
      "node_modules",
      ".pnpm",
      "@mongodb-js+agent-engine-sdk@0.11.6",
      "node_modules",
      "@mongodb-js",
      "agent-engine-sdk",
    );
    writeJSON(installed(join(app, pnpmCopy, "package.json")), {
      name: "@mongodb-js/agent-engine-sdk",
      version: "0.11.6",
    });
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(`${pnpmCopy} is another copy`);
    expect(result.stderr).toContain("overrides");
  });

  test("install fails when a workspace package's own node_modules holds another SDK copy", () => {
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    addSdkPackage(join(app, "packages", "lib"), "0.11.6");
    writeSdkManifest("9.9.9");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(
      `${join("packages", "lib", "node_modules", "@mongodb-js", "agent-engine-sdk")} is another copy`,
    );
  });

  test("install fails on a second SDK copy even at the image's version", () => {
    // Two copies split state such as AsyncLocalStorage, whatever their
    // version.
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    addSdkPackage(join(app, "packages", "lib"), "9.9.9");
    writeSdkManifest("9.9.9");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(
      `${join("packages", "lib", "node_modules", "@mongodb-js", "agent-engine-sdk")} is another copy`,
    );
  });

  test("install fails when a link: override points a workspace dependency's SDK outside node_modules", () => {
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    writeSdkManifest("9.9.9");
    const linked = join(root, "linked-sdk");
    writeJSON(join(linked, "package.json"), {
      name: "@mongodb-js/agent-engine-sdk",
      version: "0.0.1",
    });
    const link = installed(
      join(
        app,
        "packages",
        "lib",
        "node_modules",
        "@mongodb-js",
        "agent-engine-sdk",
      ),
    );
    mkdirSync(dirname(link), { recursive: true });
    symlinkSync(linked, link);
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(`is another copy, at ${linked}`);
  });

  test("install fails when pnpm installed an SDK package from a tarball other than the image's", () => {
    // An unscoped override at the image's version passes the version and
    // single-copy checks, so the tarball's integrity decides.
    addLauncher(join(app, "agents", "a"));
    addSdkPackage(join(app, "agents", "a"), "9.9.9");
    writeSdkManifest("9.9.9");
    writeInstalledPackages([
      {
        key: "@mongodb-js/agent-engine-sdk@9.9.9",
        integrity: integrityOf("a registry tarball"),
      },
    ]);
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(
      "pnpm installed @mongodb-js/agent-engine-sdk@9.9.9, which is not the runner image's tarball",
    );
  });

  test("build fails when the launcher was a link into the pnpm store it removes", () => {
    const stored = join(root, "store", "runner-shared");
    mkdirSync(join(stored, "dist"), { recursive: true });
    writeFileSync(join(stored, "dist", "launcher.js"), "");
    const link = installed(
      join(
        app,
        "agents",
        "a",
        "node_modules",
        "@mongodb-js",
        "agent-engine-runner-shared",
      ),
    );
    mkdirSync(dirname(link), { recursive: true });
    symlinkSync(stored, link);
    const result = runHelper(
      ["build", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("runner launcher is missing");
  });

  test("install fails when an SDK package is missing from the agent's node_modules", () => {
    addLauncher(join(app, "agents", "a"));
    writeSdkManifest("9.9.9");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("@mongodb-js/agent-engine-sdk");
  });

  test("install fails before installing when the image has no SDK manifest", () => {
    rmSync(join(dist, "sdk-packages.json"));
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("rebuild runner-base-typescript-langgraph");
    expect(result.calls.some((c) => c.args[0] === "install")).toBe(false);
  });

  test("install refuses a directory that is not a workspace package", () => {
    const result = runHelper(["install", app, "./agents/a"], []);
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("is not a package of a pnpm workspace");
  });

  test("build installs a workspace package for production and exits 0", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["build", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(result.calls.map((c) => c.args.join(" "))).toContain(
      "install --prod --frozen-lockfile --ignore-scripts --filter-prod my-agent... --config.confirmModulesPurge=false",
    );
  });

  test("build exits 1 and installs nothing when the npm path applies", () => {
    const result = runHelper(["build", app, "./agents/a"], []);
    expect(result.status).toBe(1);
    expect(result.calls.some((c) => c.args[0] === "install")).toBe(false);
  });

  test("build reports a failing pnpm step as 2, never as 1", () => {
    addLauncher(join(app, "agents", "a"));
    const result = runHelper(
      ["build", app, "./agents/a"],
      listing(join(app, "agents", "a")),
      {
        PNPM_STUB_FAIL: "install",
      },
    );
    expect(result.status).toBe(2);
    expect(result.calls.some((c) => c.args.includes("run"))).toBe(false);
  });

  test("build removes the pnpm store once the image build's install is done", () => {
    addLauncher(join(app, "agents", "a"));
    mkdirSync(join(root, "store"), { recursive: true });
    writeFileSync(join(root, "store", "packed-package"), "");
    const result = runHelper(
      ["build", app, "./agents/a"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(present(join(root, "store"))).toBe(false);
  });

  // Skipped as root, which ignores the directory mode and removes the store.
  test.skipIf(process.getuid?.() === 0)(
    "build exits 0 when pnpm built the agent but the store cannot be removed",
    () => {
      addLauncher(join(app, "agents", "a"));
      const locked = join(root, "store", "locked");
      mkdirSync(locked, { recursive: true });
      writeFileSync(join(locked, "packed-package"), "");
      chmodSync(locked, 0o555);
      try {
        const result = runHelper(
          ["build", app, "./agents/a"],
          listing(join(app, "agents", "a")),
        );
        expect(result.status, result.stderr).toBe(0);
        // The removal really failed, or the case would not test anything.
        expect(present(join(locked, "packed-package"))).toBe(true);
      } finally {
        chmodSync(locked, 0o755);
      }
    },
  );

  test("install leaves the pnpm store in place", () => {
    addLauncher(join(app, "agents", "a"));
    mkdirSync(join(root, "store"), { recursive: true });
    writeFileSync(join(root, "store", "packed-package"), "");
    const result = runHelper(
      ["install", app, "./agents/a", "--prod"],
      listing(join(app, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(present(join(root, "store", "packed-package"))).toBe(true);
  });

  test("dev installs from a copy, keeping devDependencies, and leaves the source untouched", () => {
    const run = join(root, "work");
    addLauncher(join(run, "agents", "a"));
    const result = runHelper(
      ["dev", app, run, "./agents/a"],
      listing(join(run, "agents", "a")),
    );
    expect(result.status, result.stderr).toBe(0);
    expect(readFileSync(join(run, "pnpm-workspace.yaml"), "utf8")).toContain(
      "agents/*",
    );
    expect(result.calls.slice(1).every((c) => c.cwd === run)).toBe(true);
    expect(result.calls.some((c) => c.args.includes("--prod"))).toBe(false);
    expect(readdirSync(app).sort()).toEqual([
      "agents",
      "package.json",
      "pnpm-workspace.yaml",
    ]);
    expect(readdirSync(join(app, "agents", "a"))).toEqual(["package.json"]);
  });

  test("dev refuses a run directory inside the source", () => {
    const result = runHelper(["dev", app, join(app, "work"), "./agents/a"]);
    expect(result.status).toBe(2);
    expect(result.calls).toEqual([]);
  });
});

describe("agentengine-pnpm-workspace.cjs packageManagerProblem", () => {
  const { packageManagerProblem } = requireScript(
    join(scriptsDir, "agentengine-pnpm-workspace.cjs"),
  ) as {
    packageManagerProblem(
      pkg: Record<string, unknown>,
      pnpmVersion: string,
      packageJSONPath: string,
    ): string;
  };
  const check = (pkg: Record<string, unknown>) =>
    packageManagerProblem(pkg, "11.25.0", "/repo/package.json");

  test("accepts a workspace with no pin, or a pin the image's pnpm major satisfies", () => {
    expect(check({})).toBe("");
    expect(check({ packageManager: "pnpm@11.2.2" })).toBe("");
    expect(check({ packageManager: "pnpm@11.25.0+sha512.abc" })).toBe("");
    expect(
      check({
        devEngines: { packageManager: { name: "pnpm", version: "^11.0.0" } },
      }),
    ).toBe("");
    expect(
      check({
        devEngines: { packageManager: { name: "pnpm", version: ">=10" } },
      }),
    ).toBe("");
    expect(check({ devEngines: { packageManager: { name: "pnpm" } } })).toBe(
      "",
    );
  });

  test("rejects another pnpm major, naming the file, the pin and the field", () => {
    expect(check({ packageManager: "pnpm@10.4.1" })).toBe(
      "/repo/package.json pins pnpm@10.4.1 in packageManager, but the runner image builds this agent with pnpm 11.25.0. " +
        "Pin pnpm 11 there (for example pnpm@11.25.0), or remove the field.",
    );
    expect(
      check({
        devEngines: { packageManager: { name: "pnpm", version: "^10.2.0" } },
      }),
    ).toContain("pins pnpm@^10.2.0 in devEngines.packageManager");
  });

  test("reads compound and upper-bounded devEngines ranges", () => {
    const dev = (version: string) =>
      check({ devEngines: { packageManager: { name: "pnpm", version } } });
    expect(dev("^10.0.0 || ^11.0.0")).toBe("");
    expect(dev("11.20 - 12")).toBe("");
    expect(dev("~11.2")).toBe("");
    expect(dev(">=10.0.0 <11.0.0")).toContain("pins pnpm@>=10.0.0 <11.0.0");
    expect(dev("<11.0.0")).toContain("pins pnpm@<11.0.0");
    expect(dev("^10.0.0 || ^12.0.0")).toContain("pins pnpm@^10.0.0 || ^12.0.0");
  });

  test("rejects another package manager", () => {
    expect(check({ packageManager: "yarn@4.5.0" })).toContain(
      "pins yarn@4.5.0 in packageManager",
    );
    expect(
      check({
        devEngines: { packageManager: { name: "npm", version: "10.9.0" } },
      }),
    ).toContain("pins npm@10.9.0 in devEngines.packageManager");
  });

  test("accepts a devEngines list when one entry is the image's pnpm major", () => {
    expect(
      check({
        devEngines: {
          packageManager: [
            { name: "yarn", version: "4.5.0" },
            { name: "pnpm", version: "11.x" },
          ],
        },
      }),
    ).toBe("");
    expect(
      check({
        devEngines: {
          packageManager: [
            { name: "yarn", version: "4.5.0" },
            { name: "pnpm", version: "9.15.0" },
          ],
        },
      }),
    ).toContain("pins yarn@4.5.0 or pnpm@9.15.0 in devEngines.packageManager");
  });
});

// The Dockerfile is outside client-libraries/packages/, so the public
// mirror of the packages has no copy of it.
const runnerBaseDockerfile = join(
  packageRoot,
  "..",
  "..",
  "..",
  "..",
  "Dockerfile.runner-base-typescript-langgraph",
);

describe.runIf(existsSync(runnerBaseDockerfile))(
  "runner-base-typescript-langgraph image",
  () => {
    let dockerfile = "";
    beforeAll(() => {
      dockerfile = readFileSync(runnerBaseDockerfile, "utf8");
    });

    test("installs a checksum-pinned pnpm and ships the helper files together", () => {
      expect(dockerfile).toMatch(/^ARG PNPM_VERSION=\d+\.\d+\.\d+$/m);
      expect(dockerfile).toContain("ARG PNPM_SHA512=");
      expect(dockerfile).toContain("sha512sum -c -");
      expect(dockerfile).toContain("npm install -g /tmp/pnpm.tgz");
      for (const name of [
        "agentengine-pnpm-workspace.sh",
        "agentengine-pnpm-workspace.cjs",
        "agentengine-sdk.pnpmfile.cjs",
        "agentengine-workspace-mirror.cjs",
      ]) {
        expect(dockerfile).toContain(
          `packages/javascript/packages/agent-engine-runner-shared/scripts/${name} /opt/agent-engine-runner-shared/bin/${name}`,
        );
      }
      expect(dockerfile).toContain(
        "COPY --chmod=0755 packages/javascript/packages/agent-engine-runner-shared/scripts/agentengine-pnpm-workspace.sh",
      );
    });

    test("pins the pnpm that the real-pnpm tests run, the npm workspace's devDependency", () => {
      const workspace = JSON.parse(
        readFileSync(join(packageRoot, "..", "..", "package.json"), "utf8"),
      ) as { devDependencies?: Record<string, string> };
      expect(dockerfile).toContain(
        `ARG PNPM_VERSION=${workspace.devDependencies?.["pnpm"]}\n`,
      );
    });

    test("writes the SDK manifest the pnpmfile reads next to the tarballs", () => {
      expect(dockerfile).toContain(
        "--json > /opt/agent-engine-runner-shared/dist/sdk-packages.json",
      );
    });
  },
);
