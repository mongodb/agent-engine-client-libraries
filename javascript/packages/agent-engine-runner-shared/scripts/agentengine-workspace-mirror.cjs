'use strict';
// agentengine-workspace-mirror.cjs: Make one directory tree an exact copy of
// another, minus dependencies, version control, and local tool state.
//
// `agentengine dev up` mounts the developer's checkout at /app and builds a
// pnpm workspace agent from a copy in the container
// (agentengine-pnpm-workspace.sh dev): pnpm rewrites pnpm-lock.yaml to pin
// the runner image's SDK, and that rewrite must never reach the checkout.
// Each restart re-runs the copy, so it follows every edit, add and delete.
//
// Usage: node agentengine-workspace-mirror.cjs <source_dir> <dest_dir>
//
// CommonJS for the same reason as the pnpmfile: no customer package.json
// "type" can change how it loads.

const fs = require('node:fs');
const path = require('node:path');

// node_modules holds host-platform binaries, and the rest is large or local
// state: version control, the pnpm store, CLI state, and the Python
// environments and caches that the CLI's archive also leaves out
// (internal/archiveignore). The copy keeps its own node_modules across
// restarts. A dist directory is copied: it can be committed output that a
// package runs from, and each package's build overwrites its own.
const SKIP_NAMES = new Set([
  'node_modules',
  '.git',
  '.pnpm-store',
  '.agentengine',
  '.agentic',
  '.devcontainer',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
]);

function isSkipped(name) {
  return SKIP_NAMES.has(name) || name.startsWith('.venv') || name.endsWith('.pyc');
}

// A copied file takes its source's mtime, so a file pnpm rewrote in the copy
// differs and is restored. utimes keeps the source's mtime only to about a
// millisecond, hence the tolerance. An edit can keep both the size and the
// mtime, so the bytes decide when the metadata matches.
function sameFile(from, source, to, target) {
  if (!target.isFile() || target.size !== source.size || Math.abs(target.mtimeMs - source.mtimeMs) >= 1) {
    return false;
  }
  return fs.readFileSync(from).equals(fs.readFileSync(to));
}

function mirrorTree(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  const kept = new Set();
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (isSkipped(entry.name)) continue;
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    let target = null;
    try {
      target = fs.lstatSync(to);
    } catch {
      // Not in the copy yet.
    }
    if (entry.isSymbolicLink()) {
      // Copied as a link, never followed: the copy must not reach outside
      // the checkout any further than the checkout itself does.
      const link = fs.readlinkSync(from);
      if (!target || !target.isSymbolicLink() || fs.readlinkSync(to) !== link) {
        fs.rmSync(to, { recursive: true, force: true });
        fs.symlinkSync(link, to);
      }
    } else if (entry.isDirectory()) {
      if (target && !target.isDirectory()) fs.rmSync(to, { recursive: true, force: true });
      mirrorTree(from, to);
    } else if (entry.isFile()) {
      const source = fs.statSync(from);
      if (!target || !sameFile(from, source, to, target)) {
        fs.rmSync(to, { recursive: true, force: true });
        fs.copyFileSync(from, to);
        fs.utimesSync(to, source.atime, source.mtime);
      }
    } else {
      continue; // Sockets, FIFOs and devices are not source.
    }
    kept.add(entry.name);
  }
  for (const name of fs.readdirSync(dest)) {
    if (kept.has(name) || isSkipped(name)) continue;
    fs.rmSync(path.join(dest, name), { recursive: true, force: true });
  }
}

// A name that only starts with ".." (such as "..work") is still inside.
function isInside(parent, child) {
  const rel = path.relative(parent, child);
  return rel === '' || (rel !== '..' && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel));
}

// p with every symlink resolved. A path that does not exist yet (the copy,
// on its first run) resolves through its nearest existing parent.
function realPath(p) {
  const missing = [];
  let current = p;
  for (;;) {
    try {
      return path.join(fs.realpathSync(current), ...missing);
    } catch (err) {
      const parent = path.dirname(current);
      if (err.code !== 'ENOENT' || parent === current) throw err;
      missing.unshift(path.basename(current));
      current = parent;
    }
  }
}

module.exports = { mirrorTree };

if (require.main === module) {
  const [source, dest] = process.argv.slice(2).map((p) => p && path.resolve(p));
  if (!source || !dest) {
    process.stderr.write('usage: agentengine-workspace-mirror.cjs <source_dir> <dest_dir>\n');
    process.exit(2);
  }
  // Real paths: a destination behind a symlink into the source, or into a
  // parent of it, contains the source, and the copy's deletes would then
  // reach the checkout.
  const realSource = realPath(source);
  const realDest = realPath(dest);
  if (isInside(realSource, realDest) || isInside(realDest, realSource)) {
    process.stderr.write(`ERROR: ${dest} and ${source} must not contain each other\n`);
    process.exit(2);
  }
  mirrorTree(source, dest);
}
