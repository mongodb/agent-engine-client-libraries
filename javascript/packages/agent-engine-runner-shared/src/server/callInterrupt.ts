/**
 * Per-call interrupt receiver for Runner SDK servers.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.call_interrupt`. Both runtimes are
 * parallel implementations of the same server architecture, so this file and
 * its Python twin must be changed together.
 *
 * When the OE's per-call interrupt fires, it POSTs `/interrupt/call` with
 * the execution and the step to abandon. The runtime aborts exactly that
 * call's controller and nothing else — unlike `POST /drain`, which is
 * terminal for the execution, an interrupt is surgical: no admission latch,
 * no record, and later steps of the same execution proceed.
 *
 * Contract (pinned by the shared fixture in
 * `client-libraries/test-fixtures/interrupt-call/contract.json`):
 *
 * - `POST /interrupt/call {execution_id, step_number, workspace_id}` —
 *   `workspace_id` is mandatory on workspace-scoped runtimes (`APP_ID` set)
 *   and must match it exactly.
 * - Always 200 with an outcome: `interrupted` (the call's controller was
 *   aborted), `not_found` (no such in-flight call), `already_settled` (the
 *   call ended first), `not_cancellable` (tracked work with no signal
 *   channel — a plain tool function — reported honestly, never claimed
 *   abandoned).
 * - Idempotent by (execution_id, step_number): a repeat re-reads the
 *   outcome.
 */

import { z } from "zod";
import type { FastifyInstance } from "fastify";

import { getLogger } from "../logger.js";
import type { DrainRegistry } from "./drain.js";

const logger = getLogger("agent_engine_runner_shared.server.callInterrupt");

export const CallInterruptRequestSchema = z.object({
  execution_id: z.string().min(1).max(256),
  step_number: z.number().int().nonnegative(),
  // Null is an accepted spelling of omission, matching the Python twin and
  // the drain contract (string|null). The route's truthiness checks treat it
  // as a missing scope claim either way.
  workspace_id: z.string().max(256).nullish(),
});

export type CallInterruptRequest = z.infer<typeof CallInterruptRequestSchema>;

/**
 * Register `POST /interrupt/call` on a runner server.
 *
 * The bearer-auth middleware already gates the path; this adds the
 * execution/workspace-scoped validation on top, identical to the drain
 * route's: when the platform scopes this process to a workspace (`APP_ID`
 * set — every managed runtime), the request must name that workspace
 * exactly; a missing `workspace_id` is a 400, a mismatch a 403. When
 * `APP_ID` is unset (local development) no workspace check is possible and
 * none is enforced.
 */
export function registerCallInterruptRoute(
  app: FastifyInstance,
  registry: DrainRegistry,
  env: NodeJS.ProcessEnv = process.env,
): void {
  const workspaceId = (env["APP_ID"] ?? "").trim();

  app.post("/interrupt/call", async (request, reply) => {
    const body = CallInterruptRequestSchema.parse(request.body);
    if (workspaceId) {
      if (!body.workspace_id) {
        return reply.code(400).send({ detail: "workspace_id required" });
      }
      if (body.workspace_id !== workspaceId) {
        return reply.code(403).send({ detail: "workspace mismatch" });
      }
    }
    const outcome = registry.abortCall(body.execution_id, body.step_number);
    // The runtime-side evidence of a per-call abort, mirroring the drain
    // receiver's accepted/finalized markers.
    logger.info(
      `Call interrupt ${outcome} for execution ${body.execution_id} (step ${body.step_number})`,
    );
    return reply.send({ outcome });
  });
}
