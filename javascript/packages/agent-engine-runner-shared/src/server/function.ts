/**
 * Function mode for the Runner SDK.
 *
 * Function mode runs a tool CLI-style: the process boots, runs a single named
 * tool from the registry, reports the result to the OE, and exits. It is the
 * per-call counterpart to the long-running tool server (mode `tool`).
 *
 * Mirrors Python's `agent_engine_runner_shared/server/function.py`.
 *
 * How the invocation arrives: fctr seeds `RunRequest.metadata` into the guest
 * metadata directory before the workload starts. The OE sets one key,
 * `request`, to a JSON `ToolFunctionRequest` envelope carrying both the
 * `ToolPodExecuteRequest` and the tool-call `step` number (required by
 * `ToolResultRequest`, owned by OE dispatch). The result is POSTed to
 * `{oe_url}/tool/result`, the same endpoint the AER already uses.
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { z } from "zod";
import { getLogger } from "../logger.js";
import { ownerDelivered } from "../owner_callback.js";
import {
  getCurrentExecutionMetadata,
  runWithCustomerOrigin,
  runWithExecutionContext,
  type OwnerUrlFailureState,
} from "../context.js";
import {
  ToolPodExecuteRequestSchema,
  type ToolPodExecuteRequest,
  type ToolPodExecuteResponse,
  type ToolResultRequest,
} from "../models.js";
import { populateLlmRegistryFromEntrypoint } from "../hooks.js";
import {
  getRequestTimeout,
  logToolRequest,
  toolRedactFields,
} from "../utils.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "../tls_client.js";
import {
  executeErrorResponse,
  requestCredentialValues,
} from "../tool_api_error.js";
import { getCurrentTraceContext } from "../tracing/setup.js";
import { discardResponseBody, retryAfterWaitMs } from "./http_retry.js";
import { resolveOeUrl } from "./oe_url.js";
import { resolveOwnerUrl } from "./owner_url.js";
import type { ITenantRuntime } from "./base.js";

const logger = getLogger("agent_engine_runner_shared.server.function");

// Guest metadata directory, mirroring fctr's util.MetadataDir. fctr
// materializes RunRequest.metadata into <dir>/<key> files before the
// workload starts. A fixed convention: the platform does not announce it.
const METADATA_DIR = "/run/meta";
const REQUEST_KEY = "request";
const TOOL_RESULT_PATH = "/tool/result";

const MAX_POST_ATTEMPTS = 3;
const POST_BASE_DELAY_MS = 100;

/** The function-mode invocation envelope delivered at /run/meta/request. */
export const ToolFunctionRequestSchema = z.object({
  request: ToolPodExecuteRequestSchema,
  step: z.number().int(),
});
export type ToolFunctionRequest = z.infer<typeof ToolFunctionRequestSchema>;

function readMetadata(metadataDir: string, key: string): string {
  const filePath = path.join(metadataDir, key);
  let value: string;
  try {
    value = fs.readFileSync(filePath, "utf-8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error(
        `function mode requires metadata key '${key}' at ${filePath}`,
        { cause: e },
      );
    }
    throw e;
  }
  if (!value.trim()) {
    throw new Error(
      `function mode metadata key '${key}' at ${filePath} is empty`,
    );
  }
  return value;
}

function readFunctionRequest(metadataDir: string): ToolFunctionRequest {
  const raw = readMetadata(metadataDir, REQUEST_KEY);
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new Error(
      `metadata key '${REQUEST_KEY}' is not a valid ToolFunctionRequest: ${(e as Error).message}`,
      { cause: e },
    );
  }
  const result = ToolFunctionRequestSchema.safeParse(parsed);
  if (!result.success) {
    throw new Error(
      `metadata key '${REQUEST_KEY}' is not a valid ToolFunctionRequest: ${result.error.message}`,
    );
  }
  return result.data;
}

function buildResultRequest(
  request: ToolPodExecuteRequest,
  response: ToolPodExecuteResponse,
  step: number,
  durationMs: number,
): ToolResultRequest {
  const { traceId, spanId } = getCurrentTraceContext();
  return {
    execution_id: request.execution_id,
    step_number: step,
    tool_name: request.tool_name,
    status: response.status,
    result: response.result,
    error: response.error,
    duration_ms: durationMs,
    pod_name: response.pod_name,
    kind: response.kind,
    metadata: response.metadata ?? {},
    trace_id: traceId ?? undefined,
    span_id: spanId ?? undefined,
    tool_api_error: response.tool_api_error,
  };
}

/**
 * Encode the result to a JSON-safe object for the /tool/result POST.
 *
 * A tool may return a value the JSON encoder cannot handle (e.g. a circular
 * reference). Downgrade any encode failure to an error report so the OE still
 * receives a terminal signal. Mirrors Python's _result_payload.
 */
export function buildResultPayload(
  result: ToolResultRequest,
  executionId: string,
): Record<string, unknown> {
  try {
    JSON.stringify(result);
    return result as unknown as Record<string, unknown>;
  } catch (e) {
    logger.error(
      `function mode: result for execution_id=${executionId} is not JSON-serializable: ${(e as Error).message}`,
    );
    return {
      execution_id: result.execution_id,
      step_number: result.step_number,
      tool_name: result.tool_name,
      status: "error",
      result: undefined,
      duration_ms: result.duration_ms,
      pod_name: result.pod_name,
      kind: result.kind,
      metadata: {},
      error: `tool result is not JSON-serializable: ${(e as Error).message}`,
    };
  }
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * POST body to url with bounded exponential backoff.
 *
 * Retries on transient transport errors and 5xx responses. Does not retry on
 * 4xx. Re-throws the final error on exhaustion. Mirrors Python's
 * post_json_with_retries and the shared postWithRetries server helper.
 *
 * `ownerUrl` — when given (already validated against `url` and path-appended by
 * the caller), a single owner pre-attempt runs *before* the service loop and
 * does not consume the service retry budget (so the trusted `url` always gets
 * its full `maxAttempts`; total tries are `maxAttempts + 1`). The owner is
 * best-effort: any failure — a transport error OR any non-2xx response (4xx or
 * 5xx) — marks the replica unusable, is logged, and falls through to the
 * service loop. The owner is never retried and an owner response never throws.
 * When the trusted `url` returns 503 with a delta-seconds `Retry-After`, that
 * attempt waits the capped hint instead of the exponential backoff.
 */
async function postWithRetries(
  url: string,
  body: unknown,
  ownerUrl?: string | null,
  onOwnerFailure?: (() => void) | null,
  maxAttempts = MAX_POST_ATTEMPTS,
  baseDelayMs = POST_BASE_DELAY_MS,
): Promise<void> {
  const timeoutMs = getRequestTimeout() * 1000;
  const serialized = JSON.stringify(body);

  // Owner pre-attempt per the shared owner-callback policy (owner_callback.ts):
  // a single best-effort POST that does not consume the service retry budget.
  if (ownerUrl != null) {
    if (
      await ownerDelivered(
        ownerUrl,
        () => ({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: serialized,
          signal: AbortSignal.timeout(timeoutMs),
          ...getFetchOptionsWithTLS(ownerUrl),
        }),
        (msg) => logger.warn(`function mode: /tool/result ${msg} URL ${url}`),
      )
    ) {
      return;
    }
    onOwnerFailure?.();
  }

  const tlsOptions = getFetchOptionsWithTLS(url);
  let lastError: unknown = null;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let retryWaitMs: number | null = null;
    let response: Response | undefined;
    try {
      response = await fetchPlatform(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serialized,
        signal: AbortSignal.timeout(timeoutMs),
        ...tlsOptions,
      });
    } catch (e) {
      lastError = e;
    }
    if (response) {
      const ok = response.ok;
      const status = response.status;
      const retryAfter = response.headers?.get?.("Retry-After") ?? null;
      await discardResponseBody(response);
      if (ok) return;
      if (status < 500) {
        throw new Error(`HTTP ${status} from ${url}`);
      }
      lastError = new Error(`HTTP ${status} from ${url}`);
      if (status === 503) {
        retryWaitMs = retryAfterWaitMs(retryAfter);
      }
    }
    if (attempt < maxAttempts - 1) {
      await sleep(retryWaitMs ?? baseDelayMs * 2 ** attempt);
    }
  }
  throw (
    lastError ??
    new Error(
      `postWithRetries exhausted ${maxAttempts} attempts but recorded no error`,
    )
  );
}

/**
 * Runs one metadata-delivered tool call and exits. Not a server.
 *
 * Reads the ToolFunctionRequest envelope from the guest metadata directory,
 * invokes the named tool from the registry, reports the ToolResultRequest
 * to {oe_url}/tool/result, and returns. Mirrors Python's ToolFunctionRunner.
 *
 * `metadataDir` is injectable for tests.
 */
export class ToolFunctionRunner {
  private readonly metadataDir: string;

  constructor(
    private readonly runtime: ITenantRuntime,
    { metadataDir = METADATA_DIR }: { metadataDir?: string } = {},
  ) {
    this.metadataDir = metadataDir;
  }

  /**
   * Run one tool from the metadata-delivered request and report the result.
   *
   * Re-raises if the result POST cannot be delivered after retries so fctr
   * records the execution as failed.
   */
  async run(): Promise<void> {
    const functionRequest = readFunctionRequest(this.metadataDir);
    const { request, step } = functionRequest;
    // Pin the callback base to the runner's own configured OE (deploy-time
    // OE_URL) before it is used as the callback base or the owner-URL trust
    // anchor. The request field is only honoured when no OE_URL is stamped
    // (local dev / tests). Resolving once here means the owner URL is validated
    // against a trusted anchor, not the raw request value.
    const oeUrl = resolveOeUrl(request.oe_url ?? "").replace(/\/+$/, "");
    if (!oeUrl) {
      throw new Error(
        "function mode requires oe_url on the request to report the result",
      );
    }
    // Prefer the replica-specific owner OE, validated against the trusted
    // resolved oeUrl above. A forged owner URL validates to null and is ignored.
    const ownerUrl = resolveOwnerUrl(request.oe_owner_url, oeUrl);
    const ownerUrlFailure: OwnerUrlFailureState = { failed: false };

    // Install before ready()/registry prep and keep through result delivery so
    // startup and callback failures still emit platform_trace_id.
    await runWithExecutionContext(
      {
        executionId: request.execution_id,
        traceId: request.platform_trace_id,
        wrapper: null,
        oeUrl,
        oeOwnerUrl: ownerUrl,
        ownerUrlFailure,
        userId: request.user_id,
        sessionId: request.session_id,
        authorization: request.authorization,
        customHeaders: request.custom_headers,
        payload: request.payload,
      },
      async () => {
        await this.runtime.graphBuilder?.ready?.();
        // Populate the named-LLM registry so a tool calling an LLM works,
        // same as ToolServer.onStartup(). Not required if no LLMs are called,
        // but matches the server-mode startup contract.
        this._prepare();

        logger.info(
          `function mode: invoking tool=${JSON.stringify(request.tool_name)} execution_id=${request.execution_id} step=${step}`,
        );

        const started = performance.now();
        const response = await this._invokeToolFn(request, ownerUrlFailure);
        const durationMs = performance.now() - started;

        const payload = buildResultPayload(
          buildResultRequest(request, response, step, durationMs),
          request.execution_id,
        );

        try {
          await postWithRetries(
            `${oeUrl}${TOOL_RESULT_PATH}`,
            payload,
            ownerUrl && !ownerUrlFailure.failed
              ? `${ownerUrl}${TOOL_RESULT_PATH}`
              : null,
            () => {
              ownerUrlFailure.failed = true;
            },
          );
        } catch (e) {
          logger.error(
            `function mode: failed to deliver result for execution_id=${request.execution_id}: ${(e as Error).message}`,
          );
          throw e;
        }
        logger.info(
          `function mode: reported ${payload["status"]} result for execution_id=${request.execution_id} step=${step}`,
        );
      },
    );
  }

  private _prepare(): void {
    populateLlmRegistryFromEntrypoint(this.runtime.graphBuilder, logger, {
      entrypointFailed:
        "Failed to run entrypoint for LLM registration; LLM calls from this tool will fail",
      noLlmRegistered:
        "Entrypoint ran but did not call app.llm(); LLM calls from this tool will fail",
    });
  }

  private async _invokeToolFn(
    request: ToolPodExecuteRequest,
    ownerUrlFailure: OwnerUrlFailureState,
  ): Promise<ToolPodExecuteResponse> {
    const podName = os.hostname();

    logToolRequest(
      request.tool_name ?? "(empty)",
      request.arguments,
      0,
      "TOOL_FUNCTION",
      toolRedactFields(this.runtime.toolDefinitions, request.tool_name),
    );

    if (!request.tool_name) {
      return {
        status: "error",
        error: "Tool name cannot be empty",
        pod_name: podName,
        metadata: {},
      };
    }

    const func = this.runtime.tools[request.tool_name];
    if (!func) {
      return {
        status: "error",
        error: `Unknown tool: ${request.tool_name}`,
        pod_name: podName,
        metadata: {},
      };
    }

    // Pin the callback base to the trusted resolved OE before it anchors the
    // execution context and the owner-URL validation (see run()).
    const oeUrl = resolveOeUrl(request.oe_url ?? "");
    return runWithExecutionContext(
      {
        executionId: request.execution_id,
        traceId: request.platform_trace_id,
        wrapper: null,
        oeUrl,
        // Emit() from the tool prefers this owner replica, validated against
        // the trusted resolved oeUrl above.
        oeOwnerUrl: resolveOwnerUrl(request.oe_owner_url, oeUrl),
        ownerUrlFailure,
        userId: request.user_id,
        sessionId: request.session_id,
        authorization: request.authorization,
        customHeaders: request.custom_headers,
        payload: request.payload,
      },
      async () => {
        try {
          const maybePromise = runWithCustomerOrigin(() =>
            func(request.arguments),
          );
          const result =
            maybePromise instanceof Promise ? await maybePromise : maybePromise;
          const metadata = getCurrentExecutionMetadata();
          return {
            status: "success",
            result,
            pod_name: podName,
            kind: metadata["memory"] ? "memory" : undefined,
            metadata,
          };
        } catch (e) {
          return executeErrorResponse(
            e,
            this.runtime.toolDefinitions[request.tool_name],
            podName,
            getCurrentExecutionMetadata(),
            requestCredentialValues(),
          );
        }
      },
    );
  }
}
