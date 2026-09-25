/** Thin ProtoJSON HTTP client for OE durable attempt, activity, and step APIs. */

import {
  fromJson,
  toJsonString,
  type JsonValue,
  type Message,
} from "@bufbuild/protobuf";
import type { GenMessage } from "@bufbuild/protobuf/codegenv2";

import {
  ActivityCommandSchema,
  ActivityContextSchema,
  ActivityMemoryCommandSchema,
  ActivityOutcomeSchema,
  type ActivityCommand,
  type ActivityContext,
  type ActivityMemoryCommand,
  type ActivityOutcome,
  type StepActivityEntry,
} from "../generated/workflow/v1/activity_pb.js";
import {
  WorkflowErrorCode,
  WorkflowErrorSchema,
  type WorkflowIdentity,
} from "../generated/workflow/v1/common_pb.js";
import {
  AttemptHeartbeatRequestSchema,
  AttemptStartRequestSchema,
  AttemptStartResponseSchema,
  type AttemptContext,
  type AttemptHeartbeatRequest,
  type AttemptStartRequest,
} from "../generated/workflow/v1/runtime_pb.js";
import {
  CompleteExecutionCommandSchema,
  FinalizeStepCommandSchema,
  FinalizeStepResponseSchema,
  type CompleteExecutionCommand,
  type FinalizeStepCommand,
} from "../generated/workflow/v1/state_pb.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "../tls_client.js";

const ATTEMPT_START_PATH = "/executor/attempt/start";
const ATTEMPT_HEARTBEAT_PATH = "/executor/attempt/heartbeat";
const ACTIVITY_START_PATH = "/executor/activity/start";
const ACTIVITY_OUTCOME_PATH = "/executor/activity/outcome";
const ACTIVITY_MEMORY_PATH = "/executor/activity/memory";
const STEP_FINALIZE_PATH = "/executor/step/finalize";
const EXECUTION_COMPLETE_PATH = "/executor/complete";

const DEFAULT_TIMEOUT_MS = 10_000;

const STATUS_TO_CODE: Readonly<Record<number, WorkflowErrorCode>> = {
  400: WorkflowErrorCode.INVALID_ARGUMENT,
  401: WorkflowErrorCode.UNAUTHORIZED,
  403: WorkflowErrorCode.UNAUTHORIZED,
  404: WorkflowErrorCode.NOT_FOUND,
  409: WorkflowErrorCode.CONFLICT,
  412: WorkflowErrorCode.STALE_FENCE,
};

export class ActivityDispatch {
  constructor(readonly context: ActivityContext) {}
}

export class ActivityReplay {
  constructor(readonly outcome: ActivityOutcome) {}
}

export type StartActivityResult = ActivityDispatch | ActivityReplay;

export class WorkflowClientError extends Error {
  readonly code: WorkflowErrorCode;

  constructor(
    code: WorkflowErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "WorkflowClientError";
    this.code = code;
  }
}

type FetchFn = typeof fetch;

/**
 * Encode one protobuf message as a ProtoJSON string using wire field names.
 * Shared so framework adapters post platform-shaped JSON bodies without
 * hand-rolling serialization.
 */
export function encodeProtoJson<T extends Message>(
  schema: GenMessage<T>,
  message: T,
): string {
  return toJsonString(schema, message, { useProtoFieldName: true });
}

function codeForStatus(status: number): WorkflowErrorCode {
  return STATUS_TO_CODE[status] ?? WorkflowErrorCode.OUTCOME_UNKNOWN;
}

function parseJsonObject(body: string): Record<string, JsonValue> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body || "{}");
  } catch (error) {
    throw new Error("response is not valid JSON", { cause: error });
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("response must be a ProtoJSON object");
  }
  return parsed as Record<string, JsonValue>;
}

function errorFromObject(value: JsonValue | undefined) {
  if (value === undefined || value === null) return undefined;
  try {
    return fromJson(WorkflowErrorSchema, value, {
      ignoreUnknownFields: true,
    });
  } catch {
    return undefined;
  }
}

function raiseOnError(status: number, body: string, operation: string): void {
  let object: Record<string, JsonValue>;
  try {
    object = parseJsonObject(body);
  } catch (error) {
    if (status >= 200 && status < 300) return;
    throw new WorkflowClientError(
      codeForStatus(status),
      `${operation} failed with HTTP ${status}`,
      { cause: error },
    );
  }
  // A present `error` member is always a failure, even on 2xx and even when
  // it is not a coded WorkflowError. Matches Python `_raise_error_member`.
  const errorMember = object["error"];
  if (errorMember !== undefined && errorMember !== null) {
    const wrapped = errorFromObject(errorMember);
    if (wrapped?.code) {
      throw new WorkflowClientError(
        wrapped.code,
        wrapped.message || `${operation} failed`,
      );
    }
    throw new WorkflowClientError(
      codeForStatus(status),
      wrapped?.message || `${operation} failed with HTTP ${status}`,
    );
  }
  const bare = errorFromObject(object);
  if (bare?.code) {
    throw new WorkflowClientError(
      bare.code,
      bare.message || `${operation} failed`,
    );
  }
  if (status < 200 || status >= 300) {
    throw new WorkflowClientError(
      codeForStatus(status),
      `${operation} failed with HTTP ${status}`,
    );
  }
}

function isCompleteWorkflowIdentity(
  identity: WorkflowIdentity | undefined,
): boolean {
  const scope = identity?.tenantScope;
  if (identity === undefined || scope === undefined) return false;
  return Boolean(
    scope.orgId.trim() &&
    scope.projectId.trim() &&
    scope.workspaceId.trim() &&
    identity.sessionId.trim() &&
    identity.executionId.trim(),
  );
}

function workflowIdentitiesEqual(
  left: WorkflowIdentity | undefined,
  right: WorkflowIdentity | undefined,
): boolean {
  if (left === undefined || right === undefined) return false;
  return (
    (left.tenantScope?.orgId ?? "") === (right.tenantScope?.orgId ?? "") &&
    (left.tenantScope?.projectId ?? "") ===
      (right.tenantScope?.projectId ?? "") &&
    (left.tenantScope?.workspaceId ?? "") ===
      (right.tenantScope?.workspaceId ?? "") &&
    left.sessionId === right.sessionId &&
    left.executionId === right.executionId
  );
}

function requireDispatchContext(
  command: ActivityCommand,
  context: ActivityContext,
): ActivityContext {
  // Reject tickets that cannot later record an outcome under this command.
  if (
    !context.activityId ||
    !context.attemptId ||
    context.fencingToken <= 0n ||
    !isCompleteWorkflowIdentity(context.workflowIdentity)
  ) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      "activity start response activity_context is missing identity fields",
    );
  }
  if (
    context.attemptId !== command.attemptId ||
    context.fencingToken !== command.fencingToken ||
    !workflowIdentitiesEqual(context.workflowIdentity, command.workflowIdentity)
  ) {
    throw new WorkflowClientError(
      WorkflowErrorCode.INVALID_ARGUMENT,
      "activity start response activity_context does not match the submitted command",
    );
  }
  return context;
}

async function readResponseText(
  response: Response,
  operation: string,
): Promise<string> {
  try {
    return await response.text();
  } catch (error) {
    throw new WorkflowClientError(
      WorkflowErrorCode.OUTCOME_UNKNOWN,
      `${operation} transport failure: ${error instanceof Error ? error.name : typeof error}`,
      { cause: error },
    );
  }
}

function interpretActivityStartResponse(
  command: ActivityCommand,
  status: number,
  body: string,
): StartActivityResult {
  let object: Record<string, JsonValue>;
  try {
    object = parseJsonObject(body);
  } catch (error) {
    throw new WorkflowClientError(
      status >= 200 && status < 300
        ? WorkflowErrorCode.INVALID_ARGUMENT
        : codeForStatus(status),
      "activity start response is not valid JSON",
      { cause: error },
    );
  }
  raiseOnError(status, body, "activity start");
  if (status < 200 || status >= 300) {
    throw new WorkflowClientError(
      codeForStatus(status),
      `activity start failed with HTTP ${status}`,
    );
  }
  const outcomeMember = object["outcome"];
  if (outcomeMember !== undefined && outcomeMember !== null) {
    try {
      return new ActivityReplay(
        fromJson(ActivityOutcomeSchema, outcomeMember, {
          ignoreUnknownFields: true,
        }),
      );
    } catch (error) {
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "activity start response is not valid ProtoJSON",
        { cause: error },
      );
    }
  }
  const contextMember = object["activity_context"];
  if (contextMember !== undefined && contextMember !== null) {
    try {
      return new ActivityDispatch(
        requireDispatchContext(
          command,
          fromJson(ActivityContextSchema, contextMember, {
            ignoreUnknownFields: true,
          }),
        ),
      );
    } catch (error) {
      if (error instanceof WorkflowClientError) throw error;
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "activity start response is not valid ProtoJSON",
        { cause: error },
      );
    }
  }
  throw new WorkflowClientError(
    WorkflowErrorCode.INVALID_ARGUMENT,
    "activity start response carries neither activity_context nor outcome",
  );
}

export class WorkflowClient {
  private readonly baseUrl: string;
  private readonly fetchFn: FetchFn;
  private readonly timeoutMs: number;

  constructor(
    oeUrl: string,
    options: { fetch?: FetchFn; timeoutMs?: number } = {},
  ) {
    this.baseUrl = oeUrl.replace(/\/$/, "");
    this.fetchFn = options.fetch ?? fetch;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  async startAttempt(
    request: AttemptStartRequest,
  ): Promise<AttemptContext | null> {
    const response = await this.post(
      ATTEMPT_START_PATH,
      encodeProtoJson(AttemptStartRequestSchema, request),
      "attempt start",
    );
    const body = await readResponseText(response, "attempt start");
    let object: Record<string, JsonValue>;
    try {
      object = parseJsonObject(body);
    } catch (error) {
      if (response.status === 404) return null;
      throw new WorkflowClientError(
        response.ok
          ? WorkflowErrorCode.INVALID_ARGUMENT
          : codeForStatus(response.status),
        response.ok
          ? "attempt start response is not valid ProtoJSON"
          : `attempt start failed with HTTP ${response.status}`,
        { cause: error },
      );
    }
    let parsed;
    try {
      parsed = fromJson(AttemptStartResponseSchema, object, {
        ignoreUnknownFields: true,
      });
    } catch (error) {
      if (response.status === 404) return null;
      throw new WorkflowClientError(
        response.ok
          ? WorkflowErrorCode.INVALID_ARGUMENT
          : codeForStatus(response.status),
        response.ok
          ? "attempt start response is not valid ProtoJSON"
          : `attempt start failed with HTTP ${response.status}`,
        { cause: error },
      );
    }
    if (parsed.error !== undefined) {
      throw new WorkflowClientError(parsed.error.code, parsed.error.message);
    }
    const bareError = errorFromObject(object);
    if (bareError?.code) {
      throw new WorkflowClientError(
        bareError.code,
        bareError.message || "attempt start failed",
      );
    }
    if (!response.ok) {
      if (response.status === 404) return null;
      throw new WorkflowClientError(
        codeForStatus(response.status),
        `attempt start failed with HTTP ${response.status}`,
      );
    }
    if (parsed.attemptContext === undefined) {
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "attempt start response carries neither attempt_context nor error",
      );
    }
    return parsed.attemptContext;
  }

  async heartbeat(request: AttemptHeartbeatRequest): Promise<void> {
    const response = await this.post(
      ATTEMPT_HEARTBEAT_PATH,
      encodeProtoJson(AttemptHeartbeatRequestSchema, request),
      "attempt heartbeat",
    );
    raiseOnError(
      response.status,
      await readResponseText(response, "attempt heartbeat"),
      "attempt heartbeat",
    );
  }

  async startActivity(command: ActivityCommand): Promise<StartActivityResult> {
    const response = await this.post(
      ACTIVITY_START_PATH,
      encodeProtoJson(ActivityCommandSchema, command),
      "activity start",
    );
    return interpretActivityStartResponse(
      command,
      response.status,
      await readResponseText(response, "activity start"),
    );
  }

  async reportOutcome(outcome: ActivityOutcome): Promise<void> {
    const response = await this.post(
      ACTIVITY_OUTCOME_PATH,
      encodeProtoJson(ActivityOutcomeSchema, outcome),
      "activity outcome",
    );
    raiseOnError(
      response.status,
      await readResponseText(response, "activity outcome"),
      "activity outcome",
    );
  }

  async ensureMemoryWritten(command: ActivityMemoryCommand): Promise<void> {
    const response = await this.post(
      ACTIVITY_MEMORY_PATH,
      encodeProtoJson(ActivityMemoryCommandSchema, command),
      "activity memory",
    );
    raiseOnError(
      response.status,
      await readResponseText(response, "activity memory"),
      "activity memory",
    );
  }

  async finalizeStep(
    command: FinalizeStepCommand,
  ): Promise<StepActivityEntry[]> {
    const response = await this.post(
      STEP_FINALIZE_PATH,
      encodeProtoJson(FinalizeStepCommandSchema, command),
      "step finalization",
    );
    const body = await readResponseText(response, "step finalization");
    raiseOnError(response.status, body, "step finalization");
    let parsed;
    try {
      parsed = fromJson(FinalizeStepResponseSchema, parseJsonObject(body), {
        ignoreUnknownFields: true,
      });
    } catch (error) {
      throw new WorkflowClientError(
        WorkflowErrorCode.INVALID_ARGUMENT,
        "step finalization response is not valid ProtoJSON",
        { cause: error },
      );
    }
    return parsed.entries;
  }

  async completeExecution(command: CompleteExecutionCommand): Promise<void> {
    const response = await this.post(
      EXECUTION_COMPLETE_PATH,
      encodeProtoJson(CompleteExecutionCommandSchema, command),
      "execution completion",
    );
    raiseOnError(
      response.status,
      await readResponseText(response, "execution completion"),
      "execution completion",
    );
  }

  private async post(
    path: string,
    payload: string,
    operation: string,
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    try {
      return await fetchPlatform(
        url,
        {
          ...getFetchOptionsWithTLS(url),
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          signal: AbortSignal.timeout(this.timeoutMs),
        },
        this.fetchFn,
      );
    } catch (error) {
      throw new WorkflowClientError(
        WorkflowErrorCode.OUTCOME_UNKNOWN,
        `${operation} transport failure: ${error instanceof Error ? error.name : typeof error}`,
        { cause: error },
      );
    }
  }
}
