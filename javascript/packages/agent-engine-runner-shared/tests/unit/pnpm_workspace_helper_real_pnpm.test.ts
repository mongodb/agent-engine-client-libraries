import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { execFileSync, spawnSync } from "node:child_process";
import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// The helper's `build` against the pnpm that the runner image installs, the
// devDependency at the npm workspace root. The stub pnpm in
// pnpm_workspace_helper.test.ts cannot show virtual store links, overrides,
// or what is left once the store is gone. Every package here is local, so
// pnpm never asks a registry.
const packageRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const helper = join(packageRoot, "scripts", "agentengine-pnpm-workspace.sh");
const pnpmBin = join(
  packageRoot,
  "..",
  "..",
  "node_modules",
  "pnpm",
  "bin",
  "pnpm.mjs",
);

function writeJSON(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value));
}

function pack(dir: string, destination: string): void {
  execFileSync("npm", ["pack", "--silent", "--pack-destination", destination], {
    cwd: dir,
    stdio: "ignore",
  });
}

describe("agentengine-pnpm-workspace.sh build with real pnpm", () => {
  let root: string;
  let bin: string;
  let dist: string;
  let fixture: string;

  beforeAll(() => {
    root = realpathSync(mkdtempSync(join(tmpdir(), "pnpm-helper-real-")));
    bin = join(root, "bin");
    dist = join(root, "dist");
    mkdirSync(bin);
    mkdirSync(dist);
    writeFileSync(
      join(bin, "pnpm"),
      `#!/bin/sh\nexec "${process.execPath}" "${pnpmBin}" "$@"\n`,
    );
    chmodSync(join(bin, "pnpm"), 0o755);

    // The image's SDK tarballs, and a second SDK tarball at the same version
    // that is not the image's.
    const sdk = join(root, "src", "sdk");
    writeJSON(join(sdk, "package.json"), {
      name: "@mongodb-js/agent-engine-sdk",
      version: "9.9.9",
      main: "index.js",
    });
    writeFileSync(join(sdk, "index.js"), 'module.exports = "image sdk";\n');
    pack(sdk, dist);
    const runner = join(root, "src", "runner");
    writeJSON(join(runner, "package.json"), {
      name: "@mongodb-js/agent-engine-runner-shared",
      version: "9.9.9",
    });
    mkdirSync(join(runner, "dist"));
    writeFileSync(join(runner, "dist", "launcher.js"), "");
    pack(runner, dist);
    writeJSON(join(dist, "sdk-packages.json"), [
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
    ]);
    const other = join(root, "src", "other-sdk");
    writeJSON(join(other, "package.json"), {
      name: "@mongodb-js/agent-engine-sdk",
      version: "9.9.9",
      main: "index.js",
    });
    writeFileSync(join(other, "index.js"), 'module.exports = "other sdk";\n');
    mkdirSync(join(root, "other"));
    pack(other, join(root, "other"));
    writeJSON(join(root, "linked-sdk", "package.json"), {
      name: "@mongodb-js/agent-engine-sdk",
      version: "0.0.1",
    });
    writeJSON(join(root, "tool-dep", "package.json"), {
      name: "tool-dep",
      version: "1.0.0",
    });

    // The agent depends on a workspace library that declares the SDK only
    // as a peer, and on a workspace tool only as a devDependency. The
    // workspace also asks for a layout the helper must override.
    fixture = join(root, "fixture");
    writeJSON(join(fixture, "package.json"), {
      name: "root",
      private: true,
      packageManager: "pnpm@11.2.2",
    });
    writeJSON(join(fixture, "agents", "a", "package.json"), {
      name: "agent-a",
      version: "1.0.0",
      scripts: { build: "node -e 0" },
      dependencies: {
        "@t/lib": "workspace:*",
        "@mongodb-js/agent-engine-sdk": "*",
        "@mongodb-js/agent-engine-runner-shared": "*",
      },
      devDependencies: { "@t/tool": "workspace:*" },
    });
    writeJSON(join(fixture, "packages", "lib", "package.json"), {
      name: "@t/lib",
      version: "1.0.0",
      scripts: { build: "node -e 0" },
      peerDependencies: { "@mongodb-js/agent-engine-sdk": "^0.10.0" },
    });
    writeJSON(join(fixture, "packages", "tool", "package.json"), {
      name: "@t/tool",
      version: "1.0.0",
      dependencies: { "tool-dep": `file:${join(root, "tool-dep")}` },
    });
  });

  afterAll(() => rmSync(root, { recursive: true, force: true }));

  function build(name: string, workspaceYAML: string) {
    const app = join(root, name);
    cpSync(fixture, app, { recursive: true });
    writeFileSync(
      join(app, "pnpm-workspace.yaml"),
      `packages:\n  - agents/*\n  - packages/*\n${workspaceYAML}`,
    );
    const store = join(root, `${name}-store`);
    const result = spawnSync("bash", [helper, "build", app, "./agents/a"], {
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: `${bin}:${process.env["PATH"] ?? ""}`,
        AGENTIC_SDK_DIST_DIR: dist,
        AGENTIC_PNPM_STORE_DIR: store,
      },
    });
    return { app, store, status: result.status, stderr: result.stderr };
  }

  test("uses the pnpm version the image pins", () => {
    expect(existsSync(pnpmBin)).toBe(true);
  });

  test("installs the agent's production closure in its own layout, and the SDK loads once the store is gone", () => {
    const result = build(
      "closure",
      "nodeLinker: hoisted\nvirtualStoreDir: ../outside-store\n",
    );
    expect(result.status, result.stderr).toBe(0);
    expect(existsSync(result.store)).toBe(false);
    expect(existsSync(join(root, "outside-store"))).toBe(false);
    const loaded = execFileSync(
      process.execPath,
      ["-p", 'require("@mongodb-js/agent-engine-sdk")'],
      { cwd: join(result.app, "agents", "a"), encoding: "utf8" },
    );
    expect(loaded.trim()).toBe("image sdk");
    // The library's peer resolves the agent's copy, the only one.
    expect(
      realpathSync(
        join(
          result.app,
          "packages",
          "lib",
          "node_modules",
          "@mongodb-js",
          "agent-engine-sdk",
        ),
      ),
    ).toBe(
      realpathSync(
        join(
          result.app,
          "agents",
          "a",
          "node_modules",
          "@mongodb-js",
          "agent-engine-sdk",
        ),
      ),
    );
    // The tool is only a devDependency, so its own dependency stays out.
    expect(
      readdirSync(join(result.app, "node_modules", ".pnpm")).filter((id) =>
        id.startsWith("tool-dep@"),
      ),
    ).toEqual([]);
  }, 120_000);

  test("fails when a scoped override gives a workspace dependency a second SDK copy at the image's version", () => {
    const result = build(
      "scoped",
      `overrides:\n  "@t/lib>@mongodb-js/agent-engine-sdk": "file:${join(root, "other", "mongodb-js-agent-engine-sdk-9.9.9.tgz")}"\n`,
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("is another copy");
  }, 120_000);

  test("fails when an unscoped override replaces the image's SDK tarball at the same version", () => {
    const result = build(
      "unscoped",
      `overrides:\n  "@mongodb-js/agent-engine-sdk": "file:${join(root, "other", "mongodb-js-agent-engine-sdk-9.9.9.tgz")}"\n`,
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain("which is not the runner image's tarball");
  }, 120_000);

  test("fails when a link: override points a workspace dependency's SDK outside node_modules", () => {
    const result = build(
      "linked",
      `overrides:\n  "@t/lib>@mongodb-js/agent-engine-sdk": "link:${join(root, "linked-sdk")}"\n`,
    );
    expect(result.status).toBe(2);
    expect(result.stderr).toContain(
      `is another copy, at ${join(root, "linked-sdk")}`,
    );
  }, 120_000);
});
