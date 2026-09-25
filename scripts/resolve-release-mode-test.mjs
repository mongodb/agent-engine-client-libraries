#!/usr/bin/env node

import assert from "node:assert/strict";
import test from "node:test";

import { resolveMode } from "./resolve-release-mode.mjs";

const registryEnv = {
  NPM_REGISTRY_URL: "https://registry.npmjs.org",
  PYPI_UPLOAD_URL: "https://upload.pypi.org/legacy/",
  PYPI_INDEX_URL: "https://pypi.org",
};

test("defaults to a loopback rehearsal", () => {
  assert.deepEqual(resolveMode(registryEnv), {
    startMock: true,
    dryRun: false,
    description: "rehearsal against loopback registries (nothing leaves the runner)",
  });
});

test("defaults to dry run when publishing is enabled", () => {
  assert.deepEqual(resolveMode({ ...registryEnv, SDK_PUBLISH_ENABLED: "true" }), {
    startMock: false,
    dryRun: true,
    npmRegistry: registryEnv.NPM_REGISTRY_URL,
    pypiUploadUrl: registryEnv.PYPI_UPLOAD_URL,
    pypiIndexUrl: registryEnv.PYPI_INDEX_URL,
    description: "dry run against the real registries (validates artifacts, uploads nothing)",
  });
});

test("publishes only when explicitly enabled with dry run disabled", () => {
  const mode = resolveMode({
    ...registryEnv,
    SDK_PUBLISH_ENABLED: "true",
    SDK_PUBLISH_DRY_RUN: "false",
  });

  assert.equal(mode.startMock, false);
  assert.equal(mode.dryRun, false);
  assert.equal(mode.description, "REAL PUBLISH to the public registries");
});

test("rejects malformed repository variables", () => {
  assert.throws(
    () => resolveMode({ ...registryEnv, SDK_PUBLISH_ENABLED: "yes" }),
    /SDK_PUBLISH_ENABLED variable must be true or false/,
  );
  assert.throws(
    () => resolveMode({ ...registryEnv, SDK_PUBLISH_DRY_RUN: "yes" }),
    /SDK_PUBLISH_DRY_RUN variable must be true or false/,
  );
});
