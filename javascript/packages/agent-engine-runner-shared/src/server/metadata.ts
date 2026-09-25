/**
 * Read request-scoped secrets from the fctr metadata directory.
 *
 * In RPC mode the OE delivers per-tool secrets via `SetMetadata` before each
 * tool call and clears them after (when restriction is enabled). fctr
 * materializes the keyset as files at `/run/meta/<KEY>`. This module reads
 * those files so the tool server can inject them into `process.env` for the
 * duration of the call — or permanently when restriction is disabled.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.metadata`.
 */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.metadata");

const METADATA_DIR = "/run/meta";

// Overlap accounting for applied-env regions. process.env is process-global:
// if a region overlaps one that has secrets applied, its snapshot captures
// those secrets as baseline and its restore re-installs them permanently
// while deleting the other call's live ones. The ToolServer's
// per-instance executeGate serializes its own routes, but that
// invariant lives in the caller — these counters make the module itself fail
// closed (throw before any env mutation) if any future call site overlaps a
// secret-bearing region. Secret-less overlap stays allowed: the snapshots
// are then identical to the live env and restore is idempotent, which is
// what the cross-instance ToolServer overlap contract relies on.
let activeRegions = 0;
let activeSecretRegions = 0;

/**
 * Read all key-value pairs from the metadata directory.
 *
 * Returns a record mapping filename (the secret name) to file content (the
 * secret value). Skips files that cannot be read and logs a warning. Returns
 * an empty record when the directory does not exist.
 */
export function readMetadataSecrets(
  metadataDir: string = METADATA_DIR,
): Record<string, string> {
  let entries;
  try {
    entries = readdirSync(metadataDir, { withFileTypes: true });
  } catch {
    return {};
  }

  const secrets: Record<string, string> = {};
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    try {
      secrets[entry.name] = readFileSync(
        join(metadataDir, entry.name),
        "utf-8",
      );
    } catch (e) {
      logger.warn(`failed to read metadata file ${entry.name}`, e);
    }
  }
  return secrets;
}

/**
 * Overwrite `process.env` from `/run/meta`; no restore or prune.
 */
export function mergeMetadataEnv(metadataDir: string = METADATA_DIR): void {
  const secrets = readMetadataSecrets(metadataDir);
  const count = Object.keys(secrets).length;
  if (count === 0) return;
  Object.assign(process.env, secrets);
  logger.debug(
    `merged ${count} metadata secret(s) into process.env (no restore)`,
  );
}

/**
 * Snapshot `process.env`, merge in the metadata secrets, and return a
 * `restore()` that reverts `process.env`.
 *
 * What `restore()` does depends on how the region was entered:
 *
 * - *clean* (no other region was active — the serialized production flow):
 *   full snapshot restore. Keys the region added are removed, overwritten
 *   keys are reverted, and keys the callback deleted are reinstated.
 * - *tainted* (entered while another, necessarily secret-less, region was
 *   active): delete-only restore. Keys added since the snapshot are removed,
 *   but no values are written back, since the snapshot holds the other
 *   region's request-scoped state and rewriting it could outlive that
 *   region's own restore. A tainted callback's overwrite of a pre-existing
 *   key therefore survives the region.
 *
 * @throws if the region overlaps a concurrent secret-bearing region — either
 * another region already has secrets applied, or this region carries secrets
 * while any region is active. Thrown before `process.env` is touched (fail
 * closed), so callers see a per-request error instead of a silent
 * cross-request credential swap. Production call sites (`/execute`,
 * `/invoke_llm`, `/invoke_llm/stream`) are serialized by the ToolServer's
 * per-instance execute gate and do not hit this.
 */
function applyMetadataEnv(metadataDir: string): { restore: () => void } {
  const secrets = readMetadataSecrets(metadataDir);
  const count = Object.keys(secrets).length;
  if (activeSecretRegions > 0 || (count > 0 && activeRegions > 0)) {
    // Fail closed before mutating anything: proceeding would snapshot or
    // clobber another request's live secrets.
    throw new Error(
      "metadata env region overlaps a concurrent secret-bearing region; " +
        "refusing to apply request secrets to the shared process.env",
    );
  }

  const snapshot: NodeJS.ProcessEnv = { ...process.env };
  // A region that entered on a quiet environment owns the true baseline:
  // its restore is a full snapshot restore — deleting added keys, reverting
  // overwritten ones, and reinstating keys the callback deleted — the
  // documented contract for the serialized (production) flow. A region that
  // entered while another was active holds a snapshot tainted with that
  // region's request-scoped state: writing any value from it could
  // permanently reassert the other request's data after its restore, so a
  // tainted restore only deletes the keys added since its snapshot and
  // never writes values. (Tainted regions are secret-less by the guard
  // above, so at worst a tainted callback's own overwrite of a pre-existing
  // key outlives it.)
  const enteredClean = activeRegions === 0;
  const restoreEnv = () => {
    for (const key of Object.keys(process.env)) {
      if (!(key in snapshot)) delete process.env[key];
    }
    if (enteredClean) {
      Object.assign(process.env, snapshot);
    }
  };

  try {
    Object.assign(process.env, secrets);
  } catch (e) {
    // Node rejects some values (e.g. containing NUL) and Object.assign may
    // have applied a prefix of the keys before throwing: roll the partial
    // application back, and only count the region once the apply succeeded —
    // otherwise the counters stay unbalanced and every later region rejects.
    restoreEnv();
    throw e;
  }
  activeRegions += 1;
  if (count > 0) {
    activeSecretRegions += 1;
    logger.debug(`applied ${count} metadata secret(s) to process.env`);
  }

  let restored = false;
  return {
    restore() {
      if (restored) return;
      restored = true;
      activeRegions -= 1;
      if (count > 0) activeSecretRegions -= 1;
      restoreEnv();
    },
  };
}

/**
 * Run `fn` with metadata secrets merged into `process.env`, restoring the
 * environment afterwards (even on throw).
 *
 * Mirrors Python's `applied_metadata_env` context manager for non-streaming
 * call paths.
 *
 * @throws if this region overlaps a concurrent secret-bearing region; see
 * {@link applyMetadataEnv} for the fail-closed rule and the clean-vs-tainted
 * restore semantics.
 */
export async function withMetadataEnv<T>(
  fn: () => Promise<T>,
  metadataDir: string = METADATA_DIR,
): Promise<T> {
  const applied = applyMetadataEnv(metadataDir);
  try {
    return await fn();
  } finally {
    applied.restore();
  }
}

/**
 * Wrap an async generator so metadata secrets are present in `process.env`
 * for the lifetime of the iteration, restoring the environment when iteration
 * completes, throws, or is closed early.
 *
 * Mirrors Python's `applied_metadata_env` context manager for streaming call
 * paths.
 *
 * @throws if this region overlaps a concurrent secret-bearing region — note a
 * stream holds its region open for the whole iteration; see
 * {@link applyMetadataEnv} for the fail-closed rule and the clean-vs-tainted
 * restore semantics.
 */
export async function* withMetadataEnvGen<T>(
  genFactory: () => AsyncGenerator<T>,
  metadataDir: string = METADATA_DIR,
): AsyncGenerator<T> {
  const applied = applyMetadataEnv(metadataDir);
  try {
    yield* genFactory();
  } finally {
    applied.restore();
  }
}
