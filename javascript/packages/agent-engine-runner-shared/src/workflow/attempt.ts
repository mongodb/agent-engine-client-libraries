/** Build durable attempt inputs from an AER execute request. */

import { hostname } from "node:os";
import { randomUUID } from "node:crypto";
import { create } from "@bufbuild/protobuf";

import type { ExecuteRequest } from "../models.js";
import type { StepSuspensionEntry } from "../generated/workflow/v1/activity_pb.js";
import {
  TenantScopeSchema,
  WorkflowIdentitySchema,
  type WorkflowIdentity,
} from "../generated/workflow/v1/common_pb.js";
import {
  AttemptStartRequestSchema,
  WorkflowDeclarationSchema,
  type AttemptContext,
  type AttemptStartRequest,
} from "../generated/workflow/v1/runtime_pb.js";
import {
  CompleteExecutionCommandSchema,
  FinalizeStepCommandSchema,
  type CompleteExecutionCommand,
  type FinalizeStepCommand,
  type StateSnapshot,
} from "../generated/workflow/v1/state_pb.js";
import { getWorkflowAdapter } from "../hooks.js";
import { currentStepOrdinal, observedActivityPositions } from "./context.js";

/** Default workflow version when the runtime does not declare one. */
const DEFAULT_WORKFLOW_VERSION = "1";

export function completeExecutionCommand(
  attempt: AttemptContext,
  state: StateSnapshot,
): CompleteExecutionCommand {
  return create(CompleteExecutionCommandSchema, {
    workflowIdentity: attempt.workflowIdentity,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
    state,
  });
}

export function newDurabilityOwnerId(): string {
  return `${hostname()}:${process.pid}:${randomUUID()}`;
}

export function finalizeCurrentStepCommand(
  attempt: AttemptContext,
  state: StateSnapshot,
): FinalizeStepCommand {
  const stepOrdinal = currentStepOrdinal();
  return create(FinalizeStepCommandSchema, {
    workflowIdentity: attempt.workflowIdentity,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
    state,
    stepOrdinal: BigInt(stepOrdinal),
    observedActivityPositions: observedActivityPositions(stepOrdinal),
  });
}

export function finalizeCurrentStepSuspensionsCommand(
  attempt: AttemptContext,
  suspensions: readonly StepSuspensionEntry[],
): FinalizeStepCommand {
  if (suspensions.length === 0) {
    throw new Error("step suspension finalization requires at least one entry");
  }
  const stepOrdinal = currentStepOrdinal();
  if (
    suspensions.some(
      (entry) => entry.position?.stepOrdinal !== BigInt(stepOrdinal),
    )
  ) {
    throw new Error("step suspension entry belongs to another workflow step");
  }
  return create(FinalizeStepCommandSchema, {
    workflowIdentity: attempt.workflowIdentity,
    attemptId: attempt.attemptId,
    fencingToken: attempt.fencingToken,
    stepOrdinal: BigInt(stepOrdinal),
    observedActivityPositions: observedActivityPositions(stepOrdinal),
    suspensions: [...suspensions],
  });
}

function workflowIdentityFromExecuteRequest(
  request: ExecuteRequest,
  sessionId: string,
): WorkflowIdentity | null {
  const orgId = (request.org_id ?? "").trim();
  const projectId = (request.project_id ?? "").trim();
  const workspaceId = (request.workspace_id ?? "").trim();
  const executionId = (request.execution_id ?? "").trim();
  const resolvedSession = sessionId.trim();
  if (
    !orgId ||
    !projectId ||
    !workspaceId ||
    !executionId ||
    !resolvedSession
  ) {
    return null;
  }
  return create(WorkflowIdentitySchema, {
    tenantScope: create(TenantScopeSchema, {
      orgId,
      projectId,
      workspaceId,
    }),
    sessionId: resolvedSession,
    executionId,
  });
}

export function attemptStartRequestFromExecute(
  request: ExecuteRequest,
  sessionId: string,
  ownerId: string,
  appName: string,
  workflowVersion?: string,
  memoryEnabled = false,
): AttemptStartRequest | null {
  const workflowIdentity = workflowIdentityFromExecuteRequest(
    request,
    sessionId,
  );
  const adapter = getWorkflowAdapter();
  if (workflowIdentity === null || adapter === null) return null;
  return create(AttemptStartRequestSchema, {
    workflowIdentity,
    ownerId,
    declaration: create(WorkflowDeclarationSchema, {
      workflowName: appName,
      workflowVersion: workflowVersion || DEFAULT_WORKFLOW_VERSION,
      adapterName: adapter.name,
      adapterVersion: adapter.version,
      memoryEnabled,
    }),
  });
}

export function heartbeatIntervalMs(value: bigint): number {
  if (value <= 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(
      "AttemptContext.heartbeat_interval_ms must be a positive safe duration",
    );
  }
  return Number(value);
}
