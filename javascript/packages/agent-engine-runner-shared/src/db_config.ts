/**
 * Single source of truth for the consolidated data-plane database name.
 *
 * All platform-owned stores (execution logs, checkpoints, memory, traces)
 * live in one database.  Override via the `MDB_AGENTIC_STORE_DB` environment
 * variable; the default is `mdb_store`.
 */

import { resolveEffectiveDb, type ListsDatabaseNames } from "./db_naming.js";
// `import type` is erased at compile time, so this adds no runtime dependency on
// mongodb (which is loaded dynamically as an optional dep elsewhere).
import type { MongoClient } from "mongodb";

const DEFAULT_DB_NAME = "mdb_store";
const LEGACY_DEFAULT_DB_NAMES = ["mdb_agentic_store"];
const ENV_VAR = "MDB_AGENTIC_STORE_DB";

/**
 * Return the base (unscoped) consolidated data-plane database name.
 *
 * Reads `MDB_AGENTIC_STORE_DB` from the environment on every call so that
 * late configuration (e.g. loading a .env file after import) is respected. For
 * the per-project-scoped name use {@link resolveStoreDbName}.
 */
export function getStoreDbName(): string {
  const value = process.env[ENV_VAR]?.trim();
  return value || DEFAULT_DB_NAME;
}

// Memoized per-project-resolved store DB names, keyed by base. Resolution needs
// a connected MongoClient (to list databases), which is not available when
// getStoreDbName runs during construction, so it is deferred to the first
// consumer that has a client and cached process-wide. The cached value is a
// Promise so concurrent callers share one listDatabases round trip per base.
const resolvedStoreDb: Record<string, Promise<string>> = {};

/**
 * Return the per-project-scoped store DB name, resolved once and cached.
 *
 * Applies the same resolution the OE uses ({@link resolveEffectiveDb}) against
 * the live cluster, so the AER/SDK writers converge on the same database the OE
 * reads. `base` defaults to {@link getStoreDbName}. Explicit values are
 * returned exactly and skip discovery.
 */
export function resolveStoreDbName(
  client: MongoClient,
  base?: string,
): Promise<string> {
  if (base !== undefined && base !== "") return Promise.resolve(base);
  const configured = (process.env[ENV_VAR] ?? "").trim();
  if (configured) return Promise.resolve(configured);
  const b = DEFAULT_DB_NAME;
  if (!(b in resolvedStoreDb)) {
    const lister: ListsDatabaseNames = {
      listDatabaseNames: async () => {
        const res = await client.db().admin().listDatabases({ nameOnly: true });
        return res.databases.map((d) => d.name);
      },
    };
    resolvedStoreDb[b] = resolveEffectiveDb(
      lister,
      b,
      (process.env["PROJECT_ID"] ?? "").trim(),
      {
        label: "agent store database",
        legacyBases: LEGACY_DEFAULT_DB_NAMES,
      },
    ).catch((err) => {
      // Don't cache a rejection — allow a later retry.
      delete resolvedStoreDb[b];
      throw err;
    });
  }
  return resolvedStoreDb[b];
}

/** Clear the memoized resolved store DB names (test seam). */
export function resetStoreDbCache(): void {
  for (const key of Object.keys(resolvedStoreDb)) delete resolvedStoreDb[key];
}
