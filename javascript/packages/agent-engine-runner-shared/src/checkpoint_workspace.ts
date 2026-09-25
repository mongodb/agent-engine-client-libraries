/**
 * Checkpoint `thread_id` workspace scope — shared by write and read paths.
 *
 * Mirrors Python's `agent_engine_runner_shared/checkpoint_workspace.py`.
 */

import { projectScopingRequired } from "./db_naming.js";

let checkpointWireWorkspaceId: string | null = null;

/** Resolve the workspace scope used for LangGraph checkpoint thread_ids. */
export function resolveCheckpointWorkspaceId(
  wireWorkspaceId?: string | null,
): string | null {
  const appId = process.env["APP_ID"];
  if (appId) return appId;
  if (projectScopingRequired()) {
    throw new Error(
      "checkpoint workspace scope is required but APP_ID is not set",
    );
  }
  if (wireWorkspaceId) return wireWorkspaceId;
  return null;
}

/**
 * Remember the wire `workspace_id` from `/execute` for query reads.
 *
 * Query routes carry no workspace identifier; when `APP_ID` is unset the read
 * path falls back to the most recently observed wire value.
 */
export function noteCheckpointWireWorkspaceId(
  wireWorkspaceId?: string | null,
): void {
  if (wireWorkspaceId) {
    checkpointWireWorkspaceId = wireWorkspaceId;
  }
}

/**
 * Resolved workspace scope for checkpoint reads (matches write path).
 *
 * Returns "" only for intentionally unscoped local runtimes. Managed AERs
 * carry REQUIRE_PROJECT_SCOPED_DB and require APP_ID; they deliberately reject
 * the wire fallback if APP_ID is missing.
 */
export function getCheckpointWorkspaceId(): string {
  return resolveCheckpointWorkspaceId(checkpointWireWorkspaceId) ?? "";
}

/** Test helper — reset module state between cases. */
export function resetCheckpointWorkspaceState(): void {
  checkpointWireWorkspaceId = null;
}
