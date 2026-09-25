/**
 * Metrics and observability for the Runner SDK.
 *
 * Ported from `agent_engine_runner_shared/metrics.py`. Provides:
 *   - `Metrics` class — process-singleton latency/error/request counters.
 *   - `recordLatency(operation, fn, labels?)` — async wrapper that measures duration.
 *   - `withMetrics(operation, fn, opts?)` — HOF replacement for the Python decorator.
 *   - `logToolCall`/`logToolResult`/`logLlmCall`/`logLlmResult`/`logExecutionEvent`
 *     structured-log helpers.
 *
 * Node is single-threaded per event loop, so no explicit lock is needed for
 * the counter maps (the GIL did the same job in Python).
 */

import { getLogger } from "./logger.js";

const logger = getLogger("agent_engine_runner_shared.metrics");

export interface LatencyStatsJson {
  count: number;
  total_ms: number;
  min_ms: number;
  max_ms: number;
  avg_ms: number;
}

class LatencyStats {
  count = 0;
  totalMs = 0;
  minMs = Number.POSITIVE_INFINITY;
  maxMs = 0;

  record(durationMs: number): void {
    this.count += 1;
    this.totalMs += durationMs;
    if (durationMs < this.minMs) this.minMs = durationMs;
    if (durationMs > this.maxMs) this.maxMs = durationMs;
  }

  get avgMs(): number {
    return this.count > 0 ? this.totalMs / this.count : 0;
  }

  toJSON(): LatencyStatsJson {
    return {
      count: this.count,
      total_ms: round2(this.totalMs),
      min_ms: this.minMs === Number.POSITIVE_INFINITY ? 0 : round2(this.minMs),
      max_ms: round2(this.maxMs),
      avg_ms: round2(this.avgMs),
    };
  }
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function labelKey(operation: string, labels: Record<string, string>): string {
  const entries = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return operation;
  const labelStr = entries.map(([k, v]) => `${k}=${v}`).join(",");
  return `${operation}:${labelStr}`;
}

export class Metrics {
  private static latencies: Map<string, LatencyStats> = new Map();
  private static errors: Map<string, number> = new Map();
  private static requests: Map<string, number> = new Map();

  static recordLatency(
    operation: string,
    durationMs: number,
    labels: Record<string, string> = {},
  ): void {
    const key = labelKey(operation, labels);
    let stats = this.latencies.get(key);
    if (!stats) {
      stats = new LatencyStats();
      this.latencies.set(key, stats);
    }
    stats.record(durationMs);
    this.requests.set(operation, (this.requests.get(operation) ?? 0) + 1);

    logger.debug(
      {
        metric_type: "latency",
        operation,
        duration_ms: round2(durationMs),
        ...labels,
      },
      "METRIC latency",
    );
  }

  static recordError(
    operation: string,
    labels: Record<string, string> = {},
  ): void {
    const key = labelKey(operation, labels);
    this.errors.set(key, (this.errors.get(key) ?? 0) + 1);

    logger.warn(
      {
        metric_type: "error",
        operation,
        ...labels,
      },
      "METRIC error",
    );
  }

  static getAll(): {
    latencies: Record<string, LatencyStatsJson>;
    errors: Record<string, number>;
    requests: Record<string, number>;
  } {
    return {
      latencies: Object.fromEntries(
        Array.from(this.latencies.entries()).map(([k, v]) => [k, v.toJSON()]),
      ),
      errors: Object.fromEntries(this.errors),
      requests: Object.fromEntries(this.requests),
    };
  }

  static reset(): void {
    this.latencies.clear();
    this.errors.clear();
    this.requests.clear();
  }
}

/**
 * Run `fn` and record its wall-clock duration as a latency metric.
 * Errors propagate; their duration is still recorded.
 */
export async function recordLatency<T>(
  operation: string,
  fn: () => Promise<T> | T,
  labels: Record<string, string> = {},
): Promise<T> {
  const start = performance.now();
  try {
    return await fn();
  } finally {
    Metrics.recordLatency(operation, performance.now() - start, labels);
  }
}

/**
 * Wrap a function to record latency (and optionally errors).
 *
 * Replacement for the Python `@with_metrics(...)` decorator. The returned
 * function preserves the original signature and awaits any returned promise
 * so async/sync code can use the same wrapper.
 */
export function withMetrics<A extends unknown[], R>(
  operation: string,
  fn: (...args: A) => R | Promise<R>,
  opts: { recordErrors?: boolean } = {},
): (...args: A) => Promise<R> {
  const recordErrors = opts.recordErrors !== false;
  return async (...args: A): Promise<R> => {
    const start = performance.now();
    try {
      return await fn(...args);
    } catch (err) {
      if (recordErrors) Metrics.recordError(operation);
      throw err;
    } finally {
      Metrics.recordLatency(operation, performance.now() - start);
    }
  };
}

// =============================================================================
// Structured Logging Helpers
// =============================================================================

export function logToolCall(args: {
  executionId: string;
  stepNumber: number;
  toolName: string;
  argumentKeys: string[];
  isLocal: boolean;
}): void {
  logger.info(
    {
      event_type: "tool_call",
      execution_id: args.executionId,
      step_number: args.stepNumber,
      tool_name: args.toolName,
      is_local: args.isLocal,
      argument_keys: args.argumentKeys,
    },
    `TOOL_CALL step=${args.stepNumber} tool=${args.toolName}`,
  );
}

export function logToolResult(args: {
  executionId: string;
  stepNumber: number;
  toolName: string;
  status: string;
  durationMs: number;
  error?: string | null;
}): void {
  const ok =
    args.status === "success" ||
    args.status === "suspend" ||
    args.status === "cached";
  const fields = {
    event_type: "tool_result",
    execution_id: args.executionId,
    step_number: args.stepNumber,
    tool_name: args.toolName,
    status: args.status,
    duration_ms: round2(args.durationMs),
    error: args.error ?? null,
  };
  const msg = `TOOL_RESULT step=${args.stepNumber} tool=${args.toolName} status=${args.status}`;
  if (ok) logger.info(fields, msg);
  else logger.error(fields, msg);

  Metrics.recordLatency("tool_execution", args.durationMs, {
    tool_name: args.toolName,
  });
  if (args.status === "error")
    Metrics.recordError("tool_execution", { tool_name: args.toolName });
}

export function logLlmCall(args: {
  executionId: string;
  stepNumber: number;
  modelName: string;
  messageCount: number;
}): void {
  logger.info(
    {
      event_type: "llm_call",
      execution_id: args.executionId,
      step_number: args.stepNumber,
      model_name: args.modelName,
      message_count: args.messageCount,
    },
    `LLM_CALL step=${args.stepNumber} model=${args.modelName}`,
  );
}

export function logLlmResult(args: {
  executionId: string;
  stepNumber: number;
  modelName: string;
  status: string;
  durationMs: number;
  toolCalls?: string[] | null;
  error?: string | null;
}): void {
  const fields = {
    event_type: "llm_result",
    execution_id: args.executionId,
    step_number: args.stepNumber,
    model_name: args.modelName,
    status: args.status,
    duration_ms: round2(args.durationMs),
    tool_calls: args.toolCalls ?? null,
    error: args.error ?? null,
  };
  const msg = `LLM_RESULT step=${args.stepNumber} model=${args.modelName} status=${args.status}`;
  if (args.status === "success") logger.info(fields, msg);
  else logger.error(fields, msg);

  Metrics.recordLatency("llm_call", args.durationMs, {
    model_name: args.modelName,
  });
  if (args.status === "error")
    Metrics.recordError("llm_call", { model_name: args.modelName });
}

export function logExecutionEvent(args: {
  executionId: string;
  event: string;
  details?: Record<string, unknown>;
}): void {
  logger.info(
    {
      event_type: `execution_${args.event.toLowerCase()}`,
      execution_id: args.executionId,
      ...(args.details ?? {}),
    },
    `EXECUTION_${args.event} execution_id=${args.executionId.slice(0, 8)}...`,
  );
}
