/** Fail-closed attempt lease heartbeat. */

import { create } from "@bufbuild/protobuf";

import { AttemptHeartbeatRequestSchema } from "../generated/workflow/v1/runtime_pb.js";
import type { AttemptContext } from "../generated/workflow/v1/runtime_pb.js";
import { WorkflowErrorCode } from "../generated/workflow/v1/common_pb.js";
import { getLogger } from "../logger.js";
import { heartbeatIntervalMs } from "./attempt.js";
import { WorkflowClientError, type WorkflowClient } from "./client.js";

const logger = getLogger("agent_engine_runner_shared.workflow.heartbeat");

const LEASE_LOST_CODES: ReadonlySet<WorkflowErrorCode> = new Set([
  WorkflowErrorCode.STALE_FENCE,
  WorkflowErrorCode.UNAUTHORIZED,
  WorkflowErrorCode.NOT_FOUND,
]);

export class AttemptHeartbeat {
  private timer: ReturnType<typeof setInterval> | null = null;
  private stopped = false;
  private inFlight: Promise<void> = Promise.resolve();

  constructor(
    private readonly attempt: AttemptContext,
    private readonly client: WorkflowClient,
  ) {}

  async start(): Promise<void> {
    if (this.timer !== null) {
      throw new Error("attempt heartbeat already started");
    }
    const request = this.request();
    try {
      await this.client.heartbeat(request);
    } catch (error) {
      throw new Error("initial durable attempt heartbeat failed", {
        cause: error,
      });
    }
    const interval = heartbeatIntervalMs(this.attempt.heartbeatIntervalMs);
    this.timer = setInterval(() => {
      this.inFlight = this.inFlight.then(async () => {
        if (this.stopped) return;
        try {
          await this.client.heartbeat(request);
        } catch (error) {
          if (
            error instanceof WorkflowClientError &&
            LEASE_LOST_CODES.has(error.code)
          ) {
            this.stopped = true;
            if (this.timer !== null) clearInterval(this.timer);
            logger.error(
              { err: error, attempt_id: this.attempt.attemptId },
              "Durable attempt lease permanently lost",
            );
            return;
          }
          logger.warn(
            { err: error, attempt_id: this.attempt.attemptId },
            "Failed to renew durable attempt lease; retrying",
          );
        }
      });
    }, interval);
    this.timer.unref();
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    await this.inFlight;
  }

  private request() {
    if (this.attempt.workflowIdentity === undefined) {
      throw new Error("AttemptContext.workflow_identity is required");
    }
    return create(AttemptHeartbeatRequestSchema, {
      workflowIdentity: this.attempt.workflowIdentity,
      attemptId: this.attempt.attemptId,
      fencingToken: this.attempt.fencingToken,
      ownerId: this.attempt.ownerId,
    });
  }
}
