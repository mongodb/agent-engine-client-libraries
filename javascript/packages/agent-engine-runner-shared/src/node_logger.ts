/**
 * Node execution logger — framework-neutral.
 *
 * Logs graph node traversals to the OE for observability. The OE writes them
 * to MongoDB so persistence stays centralized.
 *
 * Mirrors Python `agent_engine_runner_shared/node_logger.py`. Extends `NullExecutionCallback`
 * (from agent-engine-sdk) so future protocol additions don't break implementations
 * that didn't override the new method.
 */

import { NullExecutionCallback } from "@mongodb-js/agent-engine-sdk";

import { getLogger } from "./logger.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "./tls_client.js";

const logger = getLogger("agent_engine_runner_shared.node_logger");

interface NodeCallbackOpts {
  runId: string;
  parentRunId?: string;
  metadata?: Record<string, unknown>;
  durationMs?: number;
}

export interface NodeExecutionLoggerOpts {
  oeUrl: string;
  executionId: string;
  sessionId?: string | null;
  userId?: string | null;
  orgId?: string | null;
  projectId?: string | null;
}

export class NodeExecutionLogger extends NullExecutionCallback {
  private readonly oeUrl: string;
  private readonly executionId: string;
  private readonly sessionId: string | null;
  private readonly userId: string | null;
  private readonly orgId: string | null;
  private readonly projectId: string | null;
  private readonly nodeStartTimes: Map<string, Date> = new Map();

  constructor(opts: NodeExecutionLoggerOpts) {
    super();
    this.oeUrl = opts.oeUrl;
    this.executionId = opts.executionId;
    this.sessionId = opts.sessionId ?? null;
    this.userId = opts.userId ?? null;
    this.orgId = opts.orgId ?? null;
    this.projectId = opts.projectId ?? null;
  }

  override onNodeStart(
    nodeName: string,
    inputs: Record<string, unknown>,
    opts: NodeCallbackOpts,
  ): void {
    const start = new Date();
    this.nodeStartTimes.set(opts.runId, start);
    void this.sendNodeEvent({
      nodeName,
      status: "started",
      timestamp: start,
      runId: opts.runId,
      parentRunId: opts.parentRunId,
      inputs,
    });
  }

  override onNodeEnd(
    nodeName: string,
    outputs: Record<string, unknown>,
    opts: NodeCallbackOpts,
  ): void {
    const start = this.nodeStartTimes.get(opts.runId);
    this.nodeStartTimes.delete(opts.runId);
    const end = new Date();
    const durationMs =
      opts.durationMs ??
      (start !== undefined ? end.getTime() - start.getTime() : undefined);
    void this.sendNodeEvent({
      nodeName,
      status: "success",
      timestamp: end,
      runId: opts.runId,
      parentRunId: opts.parentRunId,
      outputs,
      durationMs,
    });
  }

  override onNodeError(
    nodeName: string,
    error: string,
    opts: NodeCallbackOpts,
  ): void {
    this.nodeStartTimes.delete(opts.runId);
    void this.sendNodeEvent({
      nodeName,
      status: "error",
      timestamp: new Date(),
      runId: opts.runId,
      parentRunId: opts.parentRunId,
      error,
    });
  }

  override onNodeSuspend(nodeName: string, opts: NodeCallbackOpts): void {
    const start = this.nodeStartTimes.get(opts.runId);
    this.nodeStartTimes.delete(opts.runId);
    const end = new Date();
    const durationMs =
      start !== undefined ? end.getTime() - start.getTime() : undefined;
    void this.sendNodeEvent({
      nodeName,
      status: "suspend",
      timestamp: end,
      runId: opts.runId,
      parentRunId: opts.parentRunId,
      durationMs,
    });
  }

  override onNodeInterrupted(nodeName: string, opts: NodeCallbackOpts): void {
    // Cooperative cancellation closed the node (per-call stop, run drain,
    // pod teardown): terminal but not a failure, with the partial duration.
    const start = this.nodeStartTimes.get(opts.runId);
    this.nodeStartTimes.delete(opts.runId);
    const end = new Date();
    const durationMs =
      start !== undefined ? end.getTime() - start.getTime() : undefined;
    void this.sendNodeEvent({
      nodeName,
      status: "interrupted",
      timestamp: end,
      runId: opts.runId,
      parentRunId: opts.parentRunId,
      durationMs,
    });
  }

  private serializeForJson(obj: unknown): unknown {
    if (obj === null || obj === undefined) return obj;
    const t = typeof obj;
    if (t === "string" || t === "number" || t === "boolean") return obj;
    if (Array.isArray(obj)) return obj.map((v) => this.serializeForJson(v));
    if (t === "object") {
      // Pydantic .dict() equivalent: if a custom toJSON exists, use it.
      const maybeToJson = (obj as { toJSON?: () => unknown }).toJSON;
      if (typeof maybeToJson === "function") {
        try {
          return maybeToJson.call(obj);
        } catch {
          /* fall through */
        }
      }
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
        out[k] = this.serializeForJson(v);
      }
      return out;
    }
    return String(obj);
  }

  private async sendNodeEvent(event: {
    nodeName: string;
    status: "started" | "success" | "error" | "suspend" | "interrupted";
    timestamp: Date;
    runId: string;
    parentRunId?: string;
    inputs?: Record<string, unknown>;
    outputs?: Record<string, unknown>;
    error?: string;
    durationMs?: number;
  }): Promise<void> {
    const payload: Record<string, unknown> = {
      execution_id: this.executionId,
      node_name: event.nodeName,
      status: event.status,
      timestamp: event.timestamp.toISOString(),
      run_id: event.runId,
      parent_run_id: event.parentRunId ?? null,
      session_id: this.sessionId,
      user_id: this.userId,
      org_id: this.orgId,
      project_id: this.projectId,
    };
    if (event.inputs !== undefined)
      payload["inputs"] = this.serializeForJson(event.inputs);
    if (event.outputs !== undefined)
      payload["outputs"] = this.serializeForJson(event.outputs);
    if (event.error !== undefined) payload["error"] = event.error;
    if (event.durationMs !== undefined)
      payload["duration_ms"] = event.durationMs;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2000);
    try {
      const tlsOptions = getFetchOptionsWithTLS(this.oeUrl);
      const response = await fetchPlatform(`${this.oeUrl}/node/execution`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
        ...tlsOptions,
      });
      // `fetch` only rejects on network/abort errors, not on HTTP error
      // statuses — surface a non-2xx so an OE-side failure isn't silently
      // swallowed. Stays non-fatal: node events are best-effort observability.
      if (!response.ok) {
        logger.debug(
          { status: response.status, node: event.nodeName },
          "OE rejected node event",
        );
      }
    } catch (err) {
      logger.debug(
        { err: (err as Error).message },
        "Failed to send node event to OE",
      );
    } finally {
      clearTimeout(timer);
    }
  }
}
