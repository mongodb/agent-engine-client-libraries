'use strict';
// agentengine-pnpm-workspace.cjs: The package.json and node_modules reads
// that agentengine-pnpm-workspace.sh makes, in one file it runs with node,
// so each read has one parse, one error contract, and a test.
//
// Usage: node agentengine-pnpm-workspace.cjs <command> <args...>
//   agent <dir>
//       Print "pnpm" or "npm", a tab, and the package name (empty when it
//       has none). "pnpm" means <dir>/package.json declares a workspace: or
//       catalog: dependency, which npm cannot install. Exit 1 when the
//       package.json is missing or unreadable, which keeps the npm path.
//   member <pnpm ls --json output> <agent_dir>
//       Exit 0 when pnpm listed the project at <agent_dir> (a real path),
//       and 1 when it did not.
//   package-manager <workspace_root> <pnpm_version>
//       Exit 1, with the reason on stderr, when the workspace root's
//       package.json pins a package manager other than pnpm, or a pnpm major
//       other than that of <pnpm_version>, the pnpm in the runner image.
//   verify-sdk <sdk-packages.json> <agent_dir> <workspace_root>
//       Exit 1, listing each problem on stderr, unless the runner image's
//       copy of each SDK package in the manifest, from the tarball beside
//       it, is the only copy in the workspace (see sdkMismatches).
//   pending-builds <node_modules/.modules.yaml>
//       Print, one per line, the dependencies pnpm left unbuilt.
//
// Exit 2 means an input this script cannot read. CommonJS for the same
// reason as the pnpmfile: no customer package.json "type" can change how it
// loads.

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const PROTOCOL_FIELDS = ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies'];

// <dir>/package.json as an object, or null when it is missing, unparsable,
// or not an object.
function readManifest(dir) {
  let pkg;
  try {
    pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  } catch {
    return null;
  }
  return pkg !== null && typeof pkg === 'object' && !Array.isArray(pkg) ? pkg : null;
}

function usesWorkspaceProtocol(pkg) {
  return PROTOCOL_FIELDS.some((field) => {
    const deps = pkg[field] !== null && typeof pkg[field] === 'object' ? pkg[field] : {};
    return Object.values(deps).some(
      (spec) => typeof spec === 'string' && (spec.startsWith('workspace:') || spec.startsWith('catalog:')),
    );
  });
}

function packageName(pkg) {
  return typeof pkg.name === 'string' ? pkg.name : '';
}

function isListedMember(listing, agentDir) {
  const projects = JSON.parse(listing);
  if (!Array.isArray(projects)) throw new Error('pnpm ls did not return a project list');
  return projects.some((project) => {
    try {
      return fs.realpathSync(project.path) === agentDir;
    } catch {
      return false;
    }
  });
}

// The package managers a workspace pins: packageManager ("pnpm@11.2.2"),
// and devEngines.packageManager, one entry or a list of accepted ones.
function packageManagerPins(pkg) {
  const pins = [];
  if (typeof pkg.packageManager === 'string') {
    const at = pkg.packageManager.indexOf('@', 1);
    pins.push({
      field: 'packageManager',
      alternatives: [
        at === -1
          ? { name: pkg.packageManager, version: '' }
          : { name: pkg.packageManager.slice(0, at), version: pkg.packageManager.slice(at + 1) },
      ],
    });
  }
  const devEngine = pkg.devEngines !== null && typeof pkg.devEngines === 'object' ? pkg.devEngines.packageManager : undefined;
  const entries = (Array.isArray(devEngine) ? devEngine : devEngine ? [devEngine] : []).filter(
    (entry) => entry !== null && typeof entry === 'object' && typeof entry.name === 'string',
  );
  if (entries.length > 0) {
    pins.push({
      field: 'devEngines.packageManager',
      alternatives: entries.map((entry) => ({
        name: entry.name,
        version: typeof entry.version === 'string' ? entry.version : '',
      })),
    });
  }
  return pins;
}

function compareVersions(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

// The versions one semver comparator admits, as { min, minIn, max, maxIn }
// (max null for no upper bound), or null when the comparator is not one this
// reads. A partial version ("11", "11.2", "11.x") stands for every version it
// leaves open.
function comparatorBounds(comparator) {
  const match = /^(<=|>=|<|>|=|\^|~)?v?(\d+|[xX*])(?:\.(\d+|[xX*]))?(?:\.(\d+|[xX*]))?(?:[-+].*)?$/.exec(comparator);
  if (!match) return null;
  const op = match[1] || '=';
  const parts = [match[2], match[3], match[4]];
  const given = parts.findIndex((part) => part === undefined || /^[xX*]$/.test(part));
  const known = given === -1 ? 3 : given;
  const v = parts.map((part, i) => (i < known ? Number(part) : 0));
  // The first version after every version the partial one stands for.
  const next = known === 0 ? null : known === 3 ? null : v.map((n, i) => (i < known - 1 ? n : i === known - 1 ? n + 1 : 0));
  const all = { min: [0, 0, 0], minIn: true, max: null, maxIn: false };
  switch (op) {
    case '=':
      if (known === 0) return all;
      return known === 3 ? { min: v, minIn: true, max: v, maxIn: true } : { min: v, minIn: true, max: next, maxIn: false };
    case '>=':
      return { min: v, minIn: true, max: null, maxIn: false };
    case '>':
      return known === 3 ? { min: v, minIn: false, max: null, maxIn: false } : { min: next || [Infinity, 0, 0], minIn: true, max: null, maxIn: false };
    case '<':
      return { min: [0, 0, 0], minIn: true, max: v, maxIn: false };
    case '<=':
      return known === 3 ? { min: [0, 0, 0], minIn: true, max: v, maxIn: true } : { min: [0, 0, 0], minIn: true, max: next, maxIn: false };
    case '^': {
      const lead = v.findIndex((n, i) => n !== 0 || i === known - 1);
      const top = lead === -1 || known === 0 ? null : v.map((n, i) => (i < lead ? n : i === lead ? n + 1 : 0));
      return { min: v, minIn: true, max: top, maxIn: false };
    }
    case '~':
      return { min: v, minIn: true, max: known <= 1 ? [v[0] + 1, 0, 0] : [v[0], v[1] + 1, 0], maxIn: false };
    default:
      return null;
  }
}

// Whether a pinned version or semver range admits some release of the given
// major: "11.2.2", "^11", "11.x", ">=10" and "^10.0.0 || ^11.0.0" do for 11,
// and "<11.0.0" and ">=10.0.0 <11.0.0" do not. An empty pin, and one this
// cannot read ("latest"), admit it.
function rangeAdmitsMajor(range, major) {
  return range.split('||').some((alternative) => {
    const set = alternative.trim();
    if (set === '') return true;
    const hyphen = /^(\S+)\s+-\s+(\S+)$/.exec(set);
    const comparators = hyphen ? [`>=${hyphen[1]}`, `<=${hyphen[2]}`] : set.split(/\s+/);
    let min = [major, 0, 0];
    let minIn = true;
    let max = [major + 1, 0, 0];
    let maxIn = false;
    for (const comparator of comparators) {
      const bounds = comparatorBounds(comparator);
      if (bounds === null) return true;
      const low = compareVersions(bounds.min, min);
      if (low > 0 || (low === 0 && !bounds.minIn)) {
        min = bounds.min;
        minIn = bounds.minIn;
      }
      if (bounds.max !== null) {
        const high = compareVersions(bounds.max, max);
        if (high < 0 || (high === 0 && !bounds.maxIn)) {
          max = bounds.max;
          maxIn = bounds.maxIn;
        }
      }
    }
    const span = compareVersions(min, max);
    return span < 0 || (span === 0 && minIn && maxIn);
  });
}

// Why the runner image's pnpm cannot build this workspace, or "" when it
// can. Another pnpm major reads the workspace's settings and lockfile
// differently, and the image runs only its own pnpm.
function packageManagerProblem(pkg, pnpmVersion, packageJSONPath) {
  const major = Number(String(pnpmVersion).split('.')[0]);
  if (!Number.isInteger(major)) throw new Error(`cannot read the pnpm version "${pnpmVersion}"`);
  for (const pin of packageManagerPins(pkg)) {
    if (pin.alternatives.some((alt) => alt.name === 'pnpm' && rangeAdmitsMajor(alt.version, major))) continue;
    const pinned = pin.alternatives.map((alt) => (alt.version ? `${alt.name}@${alt.version}` : alt.name)).join(' or ');
    return (
      `${packageJSONPath} pins ${pinned} in ${pin.field}, but the runner image builds this agent with pnpm ${pnpmVersion}. ` +
      `Pin pnpm ${major} there (for example pnpm@${pnpmVersion}), or remove the field.`
    );
  }
  return '';
}

// The packages pnpm installed, from its record of the install
// (node_modules/.pnpm/lock.yaml): each entry's key, with the integrity of
// the tarball it came from. A link: dependency is not a package there.
function installedPackages(lockYAML) {
  const lines = fs.readFileSync(lockYAML, 'utf8').split('\n');
  const packages = [];
  let inPackages = false;
  for (const line of lines) {
    if (/^\S/.test(line)) {
      inPackages = line === 'packages:';
      continue;
    }
    if (!inPackages) continue;
    const key = /^ {2}(['"]?)(.+?)\1:\s*$/.exec(line);
    if (key) {
      packages.push({ key: key[2], integrity: '' });
      continue;
    }
    const integrity = /^ {4}resolution: .*\bintegrity: (sha512-[A-Za-z0-9+/=]+)/.exec(line);
    if (integrity && packages.length > 0) packages[packages.length - 1].integrity = integrity[1];
  }
  return packages;
}

function packageNameOf(key) {
  const at = key.indexOf('@', 1);
  return at === -1 ? key : key.slice(0, at);
}

// What makes the workspace's SDK packages something other than the runner
// image's copies, one line each. The image's copy must be the only one:
//   - the agent resolves each SDK package at the image's version;
//   - every copy that any node_modules in the workspace reaches, through a
//     symlink or not, is the agent's copy. A second copy, even at the same
//     version, splits state such as AsyncLocalStorage, and a scoped or link:
//     override makes one for a single workspace dependency;
//   - every SDK package pnpm installed came from the image's tarball, by its
//     integrity. An unscoped same-version override replaces the tarball
//     with a copy that passes both checks above.
function sdkMismatches(manifest, distDir, agentDir, workspaceRoot) {
  const wrong = [];
  const realOrNull = (p) => {
    try {
      return fs.realpathSync(p);
    } catch {
      return null;
    }
  };
  const versionAt = (dir) => {
    try {
      return JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8')).version;
    } catch {
      return 'nothing';
    }
  };
  const shown = (p) => (p.startsWith(`${workspaceRoot}${path.sep}`) ? path.relative(workspaceRoot, p) : p);

  const images = new Map();
  for (const entry of manifest) {
    const own = path.join(agentDir, 'node_modules', entry.name);
    const found = versionAt(own);
    if (found !== entry.version) {
      wrong.push(`${entry.name}: ${found} instead of the runner image version ${entry.version}`);
    }
    const tarball = fs.readFileSync(path.join(distDir, entry.filename));
    images.set(entry.name, {
      real: realOrNull(own),
      integrity: `sha512-${crypto.createHash('sha512').update(tarball).digest('base64')}`,
    });
  }

  // Every entry of a node_modules directory, following symlinks, and the
  // real directories below it: packages, @scope directories, and the .pnpm
  // virtual store, whose entries each hold a node_modules.
  const entries = (dir) => {
    try {
      return fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return [];
    }
  };
  const isDir = (p) => {
    try {
      return fs.statSync(p).isDirectory();
    } catch {
      return false;
    }
  };
  const checkCopy = (name, at) => {
    const image = images.get(name);
    if (image !== undefined && image.real !== null) {
      const real = realOrNull(at);
      if (real !== image.real) {
        wrong.push(
          `${name}: ${shown(at)} is ${real === null ? 'a link that resolves nowhere' : `another copy, at ${shown(real)}`}, not the agent's copy at ${shown(image.real)}`,
        );
      }
    }
  };
  const scanModules = (modules) => {
    for (const entry of entries(modules)) {
      const at = path.join(modules, entry.name);
      if (entry.name === '.pnpm') {
        for (const id of entries(at)) {
          if (id.isDirectory()) scanModules(path.join(at, id.name, 'node_modules'));
        }
      } else if (entry.name.startsWith('@') && isDir(at)) {
        for (const pkg of entries(at)) visit(`${entry.name}/${pkg.name}`, path.join(at, pkg.name), pkg);
      } else {
        visit(entry.name, at, entry);
      }
    }
  };
  const visit = (name, at, entry) => {
    checkCopy(name, at);
    // A real package directory can hold nested dependencies. A symlink's
    // target is scanned where it lives, inside .pnpm, or is a link: target
    // that checkCopy has already compared.
    if (entry.isDirectory()) scanModules(path.join(at, 'node_modules'));
  };
  // Every node_modules in the workspace: the root one, and any that a
  // workspace package has of its own.
  const scanSource = (dir) => {
    for (const entry of entries(dir)) {
      if (!entry.isDirectory() || entry.name === '.git' || entry.name === '.pnpm-store') continue;
      const child = path.join(dir, entry.name);
      if (entry.name === 'node_modules') scanModules(child);
      else scanSource(child);
    }
  };
  scanSource(workspaceRoot);

  if (images.size > 0) {
    const lockYAML = path.join(workspaceRoot, 'node_modules', '.pnpm', 'lock.yaml');
    for (const pkg of installedPackages(lockYAML)) {
      const image = images.get(packageNameOf(pkg.key));
      if (image !== undefined && pkg.integrity !== image.integrity) {
        wrong.push(`${packageNameOf(pkg.key)}: pnpm installed ${pkg.key}, which is not the runner image's tarball`);
      }
    }
  }
  return wrong;
}

// The dependencies pnpm records as unbuilt in node_modules/.modules.yaml
// (JSON in pnpm 11). A dependency entry is "<name>@<version>"; a workspace
// package is listed by its directory and is never named here, so its own
// lifecycle scripts do not run again.
function pendingBuildNames(modulesYAML) {
  let text;
  try {
    text = fs.readFileSync(modulesYAML, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
  const names = new Set();
  for (const id of JSON.parse(text).pendingBuilds ?? []) {
    const match = /^(@[^/@]+\/[^/@]+|[^/@]+)@/.exec(id);
    if (match) names.add(match[1]);
  }
  return [...names];
}

function main([command, ...args]) {
  switch (command) {
    case 'agent': {
      const pkg = readManifest(args[0]);
      if (pkg === null) return 1;
      process.stdout.write(`${usesWorkspaceProtocol(pkg) ? 'pnpm' : 'npm'}\t${packageName(pkg)}\n`);
      return 0;
    }
    case 'member':
      return isListedMember(args[0], args[1]) ? 0 : 1;
    case 'package-manager': {
      const packageJSONPath = path.join(args[0], 'package.json');
      const problem = packageManagerProblem(readManifest(args[0]) ?? {}, args[1], packageJSONPath);
      if (problem === '') return 0;
      process.stderr.write(`${problem}\n`);
      return 1;
    }
    case 'verify-sdk': {
      const wrong = sdkMismatches(JSON.parse(fs.readFileSync(args[0], 'utf8')), path.dirname(args[0]), args[1], args[2]);
      if (wrong.length === 0) return 0;
      process.stderr.write(wrong.map((line) => `  ${line}\n`).join(''));
      return 1;
    }
    case 'pending-builds': {
      const names = pendingBuildNames(args[0]);
      process.stdout.write(names.map((name) => `${name}\n`).join(''));
      return 0;
    }
    default:
      throw new Error(`unknown command "${command}"`);
  }
}

module.exports = { usesWorkspaceProtocol, packageName, isListedMember, packageManagerProblem, sdkMismatches, pendingBuildNames };

if (require.main === module) {
  let code;
  try {
    code = main(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`agentengine-pnpm-workspace.cjs: ${err.message}\n`);
    code = 2;
  }
  process.exit(code);
}
