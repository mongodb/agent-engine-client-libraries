'use strict';
// agentengine-sdk.pnpmfile.cjs: pnpm hooks that pin the Agent Engine SDK
// packages to the tarballs the runner base image carries.
//
// agentengine-pnpm-workspace.sh loads this file through
// pnpm_config_global_pnpmfile, so every pnpm process in the build sees it,
// including pnpm commands that a customer build script runs. The npm install
// path gets the same result by installing the tarballs into the agent: the
// runner image's SDK always wins over the version the project declares, and
// the agent always depends on the runner launcher.
//
// CommonJS on purpose: pnpm requires a pnpmfile, and a customer
// package.json with "type": "module" must not change how it loads.

const fs = require('node:fs');
const path = require('node:path');

const DEPENDENCY_FIELDS = ['dependencies', 'devDependencies', 'optionalDependencies'];

// Maps each SDK package name to a file: specifier for its tarball. The
// runner base image writes sdk-packages.json next to the tarballs with
// `npm pack --json`, so the names come from the packages themselves.
function loadSdkSpecs(distDir) {
  if (!distDir) return new Map();
  const manifest = JSON.parse(fs.readFileSync(path.join(distDir, 'sdk-packages.json'), 'utf8'));
  return new Map(manifest.map((entry) => [entry.name, `file:${path.join(distDir, entry.filename)}`]));
}

function createReadPackage(sdkSpecs, agentPackage) {
  return function readPackage(pkg) {
    // pnpm installs a missing peer from the registry, at the declared range,
    // so a package that declares an SDK package only as a peer would get a
    // registry copy. It becomes a dependency on the image's tarball instead,
    // which the production install keeps too.
    for (const name of Object.keys(pkg.peerDependencies || {})) {
      const spec = sdkSpecs.get(name);
      if (!spec) continue;
      delete pkg.peerDependencies[name];
      if (pkg.peerDependenciesMeta) delete pkg.peerDependenciesMeta[name];
      if (pkg.devDependencies) delete pkg.devDependencies[name];
      if (!(pkg.optionalDependencies && name in pkg.optionalDependencies)) {
        pkg.dependencies = pkg.dependencies || {};
        pkg.dependencies[name] = spec;
      }
    }
    for (const field of DEPENDENCY_FIELDS) {
      const deps = pkg[field];
      if (!deps) continue;
      for (const name of Object.keys(deps)) {
        const spec = sdkSpecs.get(name);
        if (spec) deps[name] = spec;
      }
    }
    if (agentPackage && pkg.name === agentPackage) {
      // The launcher runs from the agent's own node_modules, and the
      // production install leaves out devDependencies, so every SDK package
      // becomes a runtime dependency of the agent.
      pkg.dependencies = pkg.dependencies || {};
      for (const [name, spec] of sdkSpecs) {
        if (pkg.devDependencies) delete pkg.devDependencies[name];
        if (pkg.optionalDependencies) delete pkg.optionalDependencies[name];
        pkg.dependencies[name] = spec;
      }
    }
    return pkg;
  };
}

module.exports = {
  hooks: {
    readPackage: createReadPackage(
      loadSdkSpecs(process.env.AGENTIC_SDK_DIST_DIR),
      process.env.AGENTIC_AGENT_PACKAGE,
    ),
  },
  loadSdkSpecs,
  createReadPackage,
};
