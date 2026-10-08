import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

// Generated references overlay packages/ as <package-dir>/docs/api.md.
const docsDir = process.env.SDK_DOCS_DIR ?? fileURLToPath(new URL("../packages/", import.meta.url));
const pages = new Map(
  readdirSync(docsDir)
    .map((name) => resolve(docsDir, name, "docs/api.md"))
    .filter((file) => existsSync(file))
    .map((file) => {
      return [file, readFileSync(file, "utf8")];
    }),
);

// Internal publication policy (no internal ticket IDs, internal-only URLs, or
// legacy branding) is enforced separately over the freshly generated output by
// the monorepo-only shared checker (scripts/lint-sdk-open-source-refs.sh,
// suite: sdk-docs-internal-refs), so its rules and allowlist never ship in this
// exported tree. The tests below cover public-API selection, package identity,
// and link resolution — which are language-specific and stay with the SDK.
test("reference titles and index use package identities", () => {
  const workspace = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  const index = readFileSync(new URL("../docs/README.md", import.meta.url), "utf8");
  for (const path of workspace.workspaces) {
    const pkg = JSON.parse(readFileSync(new URL(`../${path}/package.json`, import.meta.url), "utf8"));
    const directory = basename(path);
    const content = pages.get(resolve(docsDir, directory, "docs/api.md"));
    assert.ok(content.startsWith(`# ${pkg.name}\n`));
    assert.ok(content.includes("\n## Table of Contents\n"));
    assert.ok(index.includes(`\`${pkg.name}\``), `Missing ${pkg.name} in docs/README.md`);
    assert.ok(index.includes(`(../packages/${directory}/docs/api.md)`), `Missing ${directory} API link in docs/README.md`);
  }
  for (const [file, content] of pages) {
    assert.ok(!content.includes(workspace.name), `Private workspace title in ${file}`);
    assert.doesNotMatch(content, /agent-engine-[\w-]+\/src/, file);
  }
});

test("public reference excludes declarations marked internal", () => {
  const content = pages.get(resolve(docsDir, "agent-engine-runner-shared/docs/api.md")).replaceAll("\\_", "_");
  for (const name of ["LoggingStream", "MAX_BUFFER_BYTES", "MAX_TRACEBACK_BYTES", "FALLBACK_EXIT_MS", "populateLlmRegistryFromEntrypoint"]) {
    assert.doesNotMatch(content, new RegExp(`^#{1,6} ${name}(?:\\(\\))?$`, "m"));
  }
  assert.match(content, /^### SecureToolWrapper$/m);
});

test("generated reference links resolve to unique explicit anchors", () => {
  const anchorsByFile = new Map();
  for (const [file, content] of pages) {
    const anchors = [...content.matchAll(/<a id="([^"]+)"><\/a>/g)].map((match) => match[1]);
    assert.equal(new Set(anchors).size, anchors.length, `Duplicate anchors in ${file}`);
    anchorsByFile.set(file, new Set(anchors));
  }
  for (const [file, content] of pages) {
    for (const [, target] of content.matchAll(/\]\(([^)]+)\)/g)) {
      if (/^[a-z][a-z\d+.-]*:/i.test(target)) continue;
      const [path, anchor] = target.split("#");
      if (path !== "" && !path.endsWith(".md")) continue;
      const targetFile = path === "" ? file : resolve(dirname(file), decodeURIComponent(path));
      assert.ok(pages.has(targetFile), `Missing page ${target} from ${file}`);
      if (anchor) {
        assert.ok(anchorsByFile.get(targetFile).has(decodeURIComponent(anchor)), `Missing anchor ${target} from ${file}`);
      }
    }
  }
});

test("ResolveThreadId references target the type declaration", () => {
  const content = pages.get(resolve(docsDir, "agent-engine-sdk-langgraph/docs/api.md"));
  const declaration = content.match(/<a id="([^"]+)"><\/a>\s+### ResolveThreadId\n/);
  assert.ok(declaration, "The type declaration must have an explicit anchor");
  const references = [...content.matchAll(/\[`ResolveThreadId`\]\(#([^)]+)\)/g)];
  assert.ok(references.length > 0);
  for (const [, anchor] of references) assert.equal(anchor, declaration[1]);
});

test("namespace inlining preserves distinct anchors and local and cross-package links", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sdk-docs-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const write = (file, content) => {
    mkdirSync(dirname(join(root, file)), { recursive: true });
    writeFileSync(join(root, file), content);
  };
  write("runner/src/README.md", `# runner

## Namespaces

- [chunkTypes](namespaces/chunkTypes.md)

<a id="api-error"></a>

## ERROR

[own](#api-error)
[namespace](namespaces/chunkTypes.md#api-error)
[namespace page](namespaces/chunkTypes.md)
`);
  write("runner/src/namespaces/chunkTypes.md", `# chunkTypes

<a id="api-error"></a>

## ERROR

[own](#api-error)
[parent](../README.md#api-error)
[other](../../../other/src.md#api-value)
`);
  write("other/src.md", `# other

<a id="api-value"></a>

## value

[namespace](../runner/src/namespaces/chunkTypes.md#api-error)
`);
  execFileSync(process.execPath, [fileURLToPath(new URL("./restructure-docs.mjs", import.meta.url)), root]);
  const runner = readFileSync(join(root, "runner/docs/api.md"), "utf8");
  const other = readFileSync(join(root, "other/docs/api.md"), "utf8");
  assert.ok(runner.includes('<a id="api-error"></a>'));
  assert.ok(runner.includes('<a id="namespace-chunkTypes-api-error"></a>'));
  assert.ok(runner.includes('<a id="namespace-chunkTypes"></a>'));
  assert.ok(runner.includes("[own](#namespace-chunkTypes-api-error)"));
  assert.ok(runner.includes("[parent](#api-error)"));
  assert.ok(runner.includes("[other](../../other/docs/api.md#api-value)"));
  assert.ok(runner.includes("[namespace](#namespace-chunkTypes-api-error)"));
  assert.ok(runner.includes("[namespace page](#namespace-chunkTypes)"));
  assert.ok(other.includes("[namespace](../../runner/docs/api.md#namespace-chunkTypes-api-error)"));
  assert.equal(existsSync(join(root, "runner/namespaces")), false);
});

test("missing namespace entries fail without deleting the namespace content", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sdk-docs-missing-entry-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, "runner/src/namespaces"), { recursive: true });
  writeFileSync(join(root, "runner/src/README.md"), "# runner\n\n## Namespaces\n");
  writeFileSync(join(root, "runner/src/namespaces/chunkTypes.md"), "# chunkTypes\n\nPreserve this namespace content.\n");
  assert.throws(
    () => execFileSync(process.execPath, [fileURLToPath(new URL("./restructure-docs.mjs", import.meta.url)), root], { stdio: "pipe" }),
    /Missing namespace entry: runner\/chunkTypes\.md/,
  );
  assert.match(readFileSync(join(root, "runner/namespaces/chunkTypes.md"), "utf8"), /Preserve this namespace content\./);
});

test("tidy pass collapses constructors, bolds h6 labels, and adds a table of contents", (t) => {
  const root = mkdtempSync(join(tmpdir(), "sdk-docs-tidy-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, "pkg"), { recursive: true });
  writeFileSync(join(root, "pkg/src.md"), `# pkg

## Classes

<a id="api-thing"></a>

### Thing

#### Constructors

<a id="api-constructor"></a>

##### Constructor

###### Parameters

#### Methods

<a id="api-run"></a>

##### run()
`);
  execFileSync(process.execPath, [fileURLToPath(new URL("./restructure-docs.mjs", import.meta.url)), root]);
  const page = readFileSync(join(root, "pkg/docs/api.md"), "utf8");
  assert.match(page, /^# pkg\n\n## Table of Contents\n\n- \*\*Classes\*\*\n {2}- \[Thing\]\(#api-thing\)\n {4}- \[run\(\)\]\(#api-run\)\n/);
  assert.doesNotMatch(page, /Constructors/);
  assert.match(page, /^#### Constructor$/m);
  assert.match(page, /^\*\*Parameters\*\*$/m);
});
