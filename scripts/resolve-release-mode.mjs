#!/usr/bin/env node
/**
 * Decide how an SDK release run publishes, and emit the result as step outputs.
 *
 * This is the only place the decision is made. Repository variables default to
 * publishing off and dry run on, so a run publishes for real only when someone
 * has deliberately enabled publishing and deliberately turned off the dry run.
 *
 * Reads: SDK_PUBLISH_ENABLED, SDK_PUBLISH_DRY_RUN, NPM_REGISTRY_URL,
 * PYPI_UPLOAD_URL, PYPI_INDEX_URL.
 * Writes: GITHUB_OUTPUT, GITHUB_STEP_SUMMARY.
 */

import { appendFileSync } from "node:fs";

function parseBoolean(label, value, fallback) {
  const raw = value === undefined || value === "" ? fallback : value;
  if (raw !== "true" && raw !== "false") {
    throw new Error(`${label} must be true or false, got '${raw}'`);
  }
  return raw === "true";
}

export function resolveMode(env) {
  const publish = parseBoolean(
    "SDK_PUBLISH_ENABLED variable",
    env.SDK_PUBLISH_ENABLED,
    "false",
  );
  const requestedDryRun = parseBoolean(
    "SDK_PUBLISH_DRY_RUN variable",
    env.SDK_PUBLISH_DRY_RUN,
    "true",
  );

  // A rehearsal uploads to a throwaway loopback registry, so a dry run there
  // would skip the only thing the rehearsal exists to exercise.
  if (!publish) {
    return {
      startMock: true,
      dryRun: false,
      description: "rehearsal against loopback registries (nothing leaves the runner)",
    };
  }

  return {
    startMock: false,
    dryRun: requestedDryRun,
    npmRegistry: env.NPM_REGISTRY_URL,
    pypiUploadUrl: env.PYPI_UPLOAD_URL,
    pypiIndexUrl: env.PYPI_INDEX_URL,
    description: requestedDryRun
      ? "dry run against the real registries (validates artifacts, uploads nothing)"
      : "REAL PUBLISH to the public registries",
  };
}

function toOutputs(mode) {
  const outputs = {
    "start-mock": String(mode.startMock),
    "dry-run": String(mode.dryRun),
  };
  if (!mode.startMock) {
    outputs["npm-registry"] = mode.npmRegistry;
    outputs["pypi-upload-url"] = mode.pypiUploadUrl;
    outputs["pypi-index-url"] = mode.pypiIndexUrl;
  }
  return outputs;
}

function main() {
  let mode;
  try {
    mode = resolveMode(process.env);
  } catch (error) {
    console.log(`::error::${error.message}`);
    process.exit(1);
  }

  const lines = Object.entries(toOutputs(mode)).map(([key, value]) => `${key}=${value}`);
  appendFileSync(process.env.GITHUB_OUTPUT, `${lines.join("\n")}\n`);

  console.log(`Mode: ${mode.description}`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### SDK release mode: ${mode.description}\n`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
