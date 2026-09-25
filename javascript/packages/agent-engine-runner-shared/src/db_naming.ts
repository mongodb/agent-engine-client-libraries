/**
 * Per-project database name resolution with historical-name compatibility.
 *
 * Mirrors the orchestration-engine Go implementation
 * (`internal/domains/orchestration-engine/dbresolve.go`) and the Python copies
 * (`mongomem_core/db_naming.py`, `agent_engine_runner_shared/db_naming.py`). The codebases
 * are independent, so the algorithm is intentionally duplicated rather than
 * shared. Keep them in sync. See
 * `docs/decisions/008-per-project-database-isolation.md`.
 *
 * The runner/AER applies this to the agent *store* database only; memory
 * writes go through the memory-server proxy, which scopes its own database.
 */

import { getLogger } from "./logger.js";

const logger = getLogger("agent_engine_runner_shared.db_naming");

const TRUTHY = new Set(["1", "true", "yes", "on"]);

/**
 * Derive the effective database name for `base` scoped to `projectId`.
 *
 * 1. empty `projectId` -> return `base` unchanged.
 * 2. `base` already ends with `_{projectId}` -> return as-is (idempotent).
 * 3. `scoped = base + "_" + projectId`:
 *    - `scoped` exists on the cluster -> use it.
 *    - else an existing scoped `legacyBases` candidate -> use it.
 *    - else an existing unscoped `legacyBases` candidate -> use it.
 *    - else -> use `scoped` (fresh deployment).
 *
 * Unscoped fallback is limited to known platform defaults during private
 * preview. The current base and arbitrary names are never auto-adopted.
 */
export function resolveProjectScopedDb(
  base: string,
  projectId: string,
  existing: string[],
  legacyBases: string[] = [],
): string {
  if (!projectId) return base;
  const suffix = "_" + projectId;
  if (base.endsWith(suffix)) return base;
  const scoped = base + suffix;
  const names = new Set(existing);
  if (names.has(scoped)) return scoped;
  for (const legacy of legacyBases) {
    const legacyScoped = legacy + suffix;
    if (names.has(legacyScoped)) return legacyScoped;
  }
  for (const legacy of legacyBases) {
    if (names.has(legacy)) return legacy;
  }
  return scoped;
}

/**
 * Whether an empty PROJECT_ID must fail closed (REQUIRE_PROJECT_SCOPED_DB).
 *
 * ECP stamps this flag on managed AER pods (where PROJECT_ID is always injected),
 * so an empty PROJECT_ID there fails closed rather than silently writing to the
 * unscoped store. Local CLI dev leaves the flag unset and uses the unscoped name.
 */
export function projectScopingRequired(): boolean {
  return TRUTHY.has(
    (process.env["REQUIRE_PROJECT_SCOPED_DB"] ?? "").trim().toLowerCase(),
  );
}

export interface ListsDatabaseNames {
  listDatabaseNames(): Promise<string[]>;
}

/**
 * Resolve `base` against the live cluster via `listDatabaseNames`.
 *
 * Empty `projectId` throws when scoping is required (`required`, defaulting to
 * {@link projectScopingRequired}), else warns and returns `base`. A listing
 * failure throws so a transient error cannot create a competing current-name
 * database beside an existing legacy one. `legacyBases` are previous defaults
 * whose project-scoped forms, then bare forms, are adopted before a fresh
 * scoped database is created during private preview.
 */
export async function resolveEffectiveDb(
  client: ListsDatabaseNames,
  base: string,
  projectId: string,
  opts: { required?: boolean; label?: string; legacyBases?: string[] } = {},
): Promise<string> {
  const required = opts.required ?? projectScopingRequired();
  const label = opts.label ?? "database";

  if (!projectId) {
    if (required) {
      throw new Error(
        `PROJECT_ID is empty but per-project DB isolation is required ` +
          `(REQUIRE_PROJECT_SCOPED_DB); refusing to use the unscoped ${label} '${base}'`,
      );
    }
    logger.warn(
      `PROJECT_ID is empty; per-project DB isolation disabled, using unscoped ${label} '${base}'`,
    );
    return base;
  }

  let existing: string[];
  try {
    existing = await client.listDatabaseNames();
  } catch (err) {
    throw new Error(
      `database discovery failed for project '${projectId}'; refusing to select a database`,
      { cause: err },
    );
  }

  const resolved = resolveProjectScopedDb(
    base,
    projectId,
    existing,
    opts.legacyBases,
  );
  if (resolved !== base + "_" + projectId && !base.endsWith("_" + projectId)) {
    logger.info(
      `using historical ${label} '${resolved}' for project '${projectId}'`,
    );
  }
  return resolved;
}
