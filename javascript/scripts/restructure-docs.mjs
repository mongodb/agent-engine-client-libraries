// Restructure the raw TypeDoc output into a per-package layout that mirrors the
// Python SDK's convention: each package's reference lives at
// `<package-dir>/docs/api.md`, so the output tree overlays `packages/` directly.
//
// TypeDoc names each entry module after its path (`agent-engine-<pkg>/src`), so a
// package emits `agent-engine-<pkg>/src.md`, or — when it exports a namespace —
// `agent-engine-<pkg>/src/README.md` plus `src/namespaces/*.md`. This pass moves
// each module entry file to `agent-engine-<pkg>/docs/api.md`, inlines namespace
// pages, drops TypeDoc's workspace index (the hand-written docs/README.md indexes
// the packages), and recomputes every relative Markdown link.
//
// Usage: node restructure-docs.mjs <docs-dir>
import { readdirSync, statSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join, posix } from "node:path";
import { fileURLToPath } from "node:url";

const root = process.argv[2];
if (!root) {
  console.error("usage: restructure-docs.mjs <docs-dir>");
  process.exit(1);
}

const packagesRoot = fileURLToPath(new URL("../packages/", import.meta.url));
const packageNames = new Map(
  readdirSync(packagesRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => [
      `${entry.name}/src`,
      JSON.parse(readFileSync(join(packagesRoot, entry.name, "package.json"), "utf8")).name,
    ]),
);

// Display package identities without changing page paths or symbol anchors.
function rewritePackageNames(content) {
  for (const [module, name] of packageNames) {
    content = content
      .replaceAll(`# ${module}\n`, () => `# ${name}\n`)
      .replaceAll(` / ${module}\n`, () => ` / ${name}\n`)
      .replaceAll(`[${module}](`, () => `[${name}](`);
  }
  return content;
}

// Collect every Markdown file as a POSIX path relative to the docs root.
function listMarkdown(dir, base = "") {
  const out = [];
  for (const name of readdirSync(dir)) {
    const abs = join(dir, name);
    const rel = base ? posix.join(base, name) : name;
    if (statSync(abs).isDirectory()) out.push(...listMarkdown(abs, rel));
    else if (name.endsWith(".md")) out.push(rel);
  }
  return out;
}

// Map an original relative path to its restructured location.
//   agent-engine-<pkg>/src.md           -> agent-engine-<pkg>/docs/api.md
//   agent-engine-<pkg>/src/README.md    -> agent-engine-<pkg>/docs/api.md
//   agent-engine-<pkg>/src/<rest>       -> agent-engine-<pkg>/<rest>
function targetOf(rel) {
  let m = rel.match(/^(.+)\/src\.md$/);
  if (m) return `${m[1]}/docs/api.md`;
  m = rel.match(/^(.+)\/src\/README\.md$/);
  if (m) return `${m[1]}/docs/api.md`;
  m = rel.match(/^(.+)\/src\/(.+)$/);
  if (m) return `${m[1]}/${m[2]}`;
  return rel;
}

const WORKSPACE_INDEX = "README.md";
const files = listMarkdown(root).sort();
const moves = new Map(files.map((f) => [f, targetOf(f)]));

function namespaceName(rel) {
  return rel.match(/\/namespaces\/([^/]+)\.md$/)?.[1];
}

function finalTarget(rel) {
  const moved = moves.get(rel) ?? rel;
  return namespaceName(rel) ? `${rel.split("/")[0]}/docs/api.md` : moved;
}

// Namespace pages have independent anchor counters before they share api.md.
function finalAnchor(rel, anchor) {
  const name = namespaceName(rel);
  return name ? `#namespace-${name}${anchor ? `-${anchor.slice(1)}` : ""}` : anchor;
}

// Rewrite a single file's relative Markdown links for its new location.
function rewrite(content, oldRel) {
  const oldDir = posix.dirname(oldRel);
  const newRel = finalTarget(oldRel);
  const newDir = posix.dirname(newRel);
  content = content.replace(/<a id="([^"]+)"><\/a>/g, (_, anchor) =>
    `<a id="${finalAnchor(oldRel, `#${anchor}`).slice(1)}"></a>`,
  );
  content = rewritePackageNames(content);
  return content.replace(/\]\(([^)]+)\)/g, (whole, target) => {
    const hash = target.indexOf("#");
    const path = hash === -1 ? target : target.slice(0, hash);
    const anchor = hash === -1 ? "" : target.slice(hash);
    if (/^[a-z]+:/i.test(path) || (path !== "" && !path.endsWith(".md"))) return whole;
    const oldTarget = path === "" ? oldRel : posix.normalize(posix.join(oldDir, path));
    const newTarget = finalTarget(oldTarget);
    const newPath = newTarget === newRel ? "" : posix.relative(newDir, newTarget);
    return `](${newPath}${finalAnchor(oldTarget, anchor)})`;
  });
}

// Read + rewrite everything before touching the tree, then rewrite the layout.
const rewritten = new Map();
for (const oldRel of files) {
  if (oldRel === WORKSPACE_INDEX) continue;
  rewritten.set(moves.get(oldRel), rewrite(readFileSync(join(root, oldRel), "utf8"), oldRel));
}
for (const oldRel of files) rmSync(join(root, oldRel));
// Prune now-empty `src` directories left behind by the collapse.
for (const name of readdirSync(root)) {
  const src = join(root, name, "src");
  try {
    if (statSync(src).isDirectory() && listMarkdown(src).length === 0) rmSync(src, { recursive: true });
  } catch {
    // no src/ under this entry — nothing to prune.
  }
}
for (const [newRel, content] of rewritten) {
  mkdirSync(join(root, posix.dirname(newRel)), { recursive: true });
  writeFileSync(join(root, newRel), content);
}

// Shift every Markdown heading down by `by` levels (capped at 6), leaving fenced
// code blocks untouched.
function demoteHeadings(md, by) {
  let inFence = false;
  return md
    .split("\n")
    .map((line) => {
      if (line.startsWith("```")) inFence = !inFence;
      const m = inFence ? null : line.match(/^(#{1,6})(\s.*)$/);
      if (!m) return line;
      return "#".repeat(Math.min(m[1].length + by, 6)) + m[2];
    })
    .join("\n");
}

// Fold each package's namespace pages into its api.md. typedoc-plugin-markdown
// always emits a namespace (`export * as ns`) as its own page regardless of
// outputFileStrategy, which would leave a stray file beside the otherwise
// one-file-per-package tree. Inlining keeps every package a single self-contained
// api.md. Handles the flat namespace case (the only shape this SDK produces).
function inlineNamespaces(dir) {
  for (const pkg of readdirSync(dir)) {
    const nsDir = join(dir, pkg, "namespaces");
    let nsFiles;
    try {
      nsFiles = readdirSync(nsDir).filter((n) => n.endsWith(".md"));
    } catch {
      continue; // no namespaces/ under this package
    }
    const apiPath = join(dir, pkg, "docs", "api.md");
    let api = readFileSync(apiPath, "utf8");
    for (const file of nsFiles) {
      const name = file.replace(/\.md$/, "");
      const raw = readFileSync(join(nsDir, file), "utf8");
      // Drop the breadcrumb header (keep from the first H1) and demote so the
      // namespace nests as a `### <name>` member under the module's `## Namespaces`.
      const anchor = finalAnchor(`${pkg}/namespaces/${file}`, "");
      const entry = `- [${name}](${anchor})`;
      if (!api.includes(entry)) throw new Error(`Missing namespace entry: ${pkg}/${file}`);
      const block = `<a id="${anchor.slice(1)}"></a>\n\n${demoteHeadings(raw.slice(raw.search(/^# /m)), 2).trimEnd()}`;
      api = api.replace(entry, () => block);
      rmSync(join(nsDir, file));
    }
    writeFileSync(apiPath, api);
    if (readdirSync(nsDir).length === 0) rmSync(nsDir, { recursive: true });
  }
}

// TypeDoc nests the single constructor under a redundant "Constructors" group,
// and renders Parameters/Returns/etc. as h6 headings, which GitHub shows as tiny
// grey text. Collapse the former and render the latter as bold labels.
function tidy(md) {
  let inFence = false;
  return md
    .replace(/^(#{2,6}) Constructors\n\n(<a id="[^"]+"><\/a>)\n\n#{3,6} Constructor$/gm, "$2\n\n$1 Constructor")
    .split("\n")
    .map((line) => {
      if (line.startsWith("```")) inFence = !inFence;
      const m = inFence ? null : line.match(/^###### (.+)$/);
      return m ? `**${m[1]}**` : line;
    })
    .join("\n");
}

// Build a table of contents from the page's groups (h2), anchored symbols (h3),
// and anchored class/interface methods (h5 under an h4 "Methods" group).
function tableOfContents(md) {
  const entries = [];
  let inFence = false;
  let pendingAnchor;
  let subgroup;
  for (const line of md.split("\n")) {
    if (line.startsWith("```")) inFence = !inFence;
    if (inFence) continue;
    const anchor = line.match(/^<a id="([^"]+)"><\/a>$/);
    if (anchor) {
      pendingAnchor = anchor[1];
      continue;
    }
    const heading = line.match(/^(#{2,5}) (.+)$/);
    if (!heading) {
      if (line.trim() !== "") pendingAnchor = undefined;
      continue;
    }
    const level = heading[1].length;
    const title = heading[2];
    if (level === 2) entries.push(`- **${title}**`);
    else if (level === 3 && pendingAnchor) entries.push(`  - [${title}](#${pendingAnchor})`);
    else if (level === 4) subgroup = title;
    else if (level === 5 && pendingAnchor && subgroup === "Methods") entries.push(`    - [${title}](#${pendingAnchor})`);
    if (level <= 3) subgroup = undefined;
    pendingAnchor = undefined;
  }
  return entries.join("\n");
}

function withTableOfContents(md) {
  const toc = tableOfContents(md);
  if (!toc) return md;
  return md.replace(/^(# .+\n)/m, (title) => `${title}\n## Table of Contents\n\n${toc}\n`);
}

inlineNamespaces(root);
for (const file of listMarkdown(root)) {
  const path = join(root, file);
  writeFileSync(path, withTableOfContents(tidy(readFileSync(path, "utf8"))));
}
