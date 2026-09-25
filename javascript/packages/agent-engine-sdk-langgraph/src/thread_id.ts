/**
 * Workspace-scoped LangGraph `thread_id` composition.
 *
 * Port of Python's `agent_engine_sdk_langgraph/thread_id.py`.
 */

import type { RequestContext } from "@mongodb-js/agent-engine-sdk";

/** Tenant hook mapping a `RequestContext` to a checkpoint `thread_id`. */
export type ResolveThreadIdHook = (ctx: RequestContext) => string;

const DEFAULT_SESSION_ID = "default";

/** Compose the workspace-scoped `thread_id` for a checkpoint key. */
export function scopedThreadId(
  sessionId?: string | null,
  workspaceId?: string | null,
): string {
  const sid = sessionId || DEFAULT_SESSION_ID;
  if (!workspaceId) return sid;
  return `${sid}:${workspaceId}`;
}

/**
 * Return the `thread_id` keys that may hold checkpoints for a session.
 *
 * When a workspace scope is known, only the composite key is queried —
 * never the bare `session_id`. Bare keys are shared by every workspace on
 * the same store, so including them would let one tenant read or poison
 * another's conversation history. Legacy unscoped checkpoints are
 * intentionally not queried.
 */
export function threadIdsForQuery(
  sessionId: string,
  workspaceId?: string | null,
): string[] {
  if (!workspaceId) return [sessionId];
  return [scopedThreadId(sessionId, workspaceId)];
}

/** Flatten `threadIdsForQuery` across many sessions, deduped. */
export function threadIdsForSessionsQuery(
  sessionIds: string[],
  workspaceId?: string | null,
): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const sessionId of sessionIds) {
    for (const threadId of threadIdsForQuery(sessionId, workspaceId)) {
      if (!seen.has(threadId)) {
        seen.add(threadId);
        result.push(threadId);
      }
    }
  }
  return result;
}

/** Recover the plain `session_id` from a workspace-scoped `thread_id`. */
export function sessionIdFromThreadId(
  threadId: string,
  workspaceId?: string | null,
): string {
  if (!workspaceId) return threadId;
  const suffix = `:${workspaceId}`;
  if (threadId.endsWith(suffix)) {
    return threadId.slice(0, -suffix.length);
  }
  return threadId;
}

/**
 * LangGraph `thread_id` for a platform request — the same key invoke uses.
 * Port of Python's `checkpoint_thread_id`.
 */
export function checkpointThreadId(
  ctx: RequestContext,
  resolveThreadId?: ResolveThreadIdHook | null,
): string {
  if (resolveThreadId != null) {
    const threadId = resolveThreadId(ctx);
    if (typeof threadId !== "string" || threadId.trim() === "") {
      const rendered =
        typeof threadId === "string"
          ? JSON.stringify(threadId)
          : typeof threadId;
      throw new Error(
        `App.resolveThreadId must return a non-empty string; got ${rendered}`,
      );
    }
    return threadId;
  }
  return scopedThreadId(ctx.sessionId, ctx.workspaceId);
}
