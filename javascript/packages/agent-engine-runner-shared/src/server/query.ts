/**
 * Framework-agnostic query plugin protocol for the AER.
 *
 * The AER hosts two read endpoints that surface framework-specific persisted
 * state — per-session summary and conversation messages. The implementations
 * live in framework adapter packages (e.g. `agent-engine-sdk-langgraph-ts`); the
 * AER itself only hosts the routes and delegates to whichever plugin the
 * framework SDK registered via `registerQueryPlugin()`.
 *
 * Mirrors Python's `agent_engine_runner_shared/server/query.py`.
 *
 * Trust model
 * -----------
 *
 * These endpoints take no workspace identifier on the wire. The AER trusts its
 * caller (the platform's Orchestration Engine proxy) to pass only session_ids
 * that belong to the caller's workspace; the OE establishes that scope by
 * consulting the `executions` collection — which is tagged with
 * `workspace_id` — before issuing the request. The framework plugin then
 * narrows reads by its own session key. For the LangGraph plugin that key is
 * the workspace-scoped composite `session_id:workspace_id`. The workspace is
 * workspace is resolved the same way on read and write — `APP_ID` when set,
 * otherwise the wire `workspace_id` from the most recent `/execute` — so a
 * plugin keyed to one workspace cannot read another's checkpoints even if a
 * stray session_id slipped past the proxy — defense in depth, not a substitute
 * for the upstream ownership check.
 */

import type {
  SessionMessagesResponse,
  SessionsSummaryResponse,
} from "@mongodb-js/agent-engine-sdk";

/**
 * Read-side plugin that surfaces framework-specific session state.
 *
 * Implementations live alongside framework adapters and read from the
 * adapter's persistence (LangGraph checkpoint collections, ADK state
 * store, etc.).
 */
export interface AERQueryPlugin {
  /** Given a list of session_ids, return a SessionsSummaryResponse. */
  getSummariesForSessions(
    sessionIds: string[],
  ): Promise<SessionsSummaryResponse>;

  /** Given a session_id, return a SessionMessagesResponse. */
  getMessagesForSession(sessionId: string): Promise<SessionMessagesResponse>;
}
